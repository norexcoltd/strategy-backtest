#!/usr/bin/env python3
"""
Kaggle GPU Backtest Runner
- DB 의존성 제거 (Parquet + symbol_map.csv 단독)
- DEVICE = "cuda" (T4 GPU 14.6GB VRAM)
- RAM 최적화: .reshape() + 중간변수 해제
- LP 수정: 양쪽 unrealized 합산 (Cross Margin)
- v42 설정: 롱 1h BB 진입 + 4h 모멘텀/BB_TP/손절 + 롱 20x
"""

import sys
import os
import gc
import time
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set
from collections import Counter

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import torch
import psutil

# === 환경 확인 ===
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print(f"RAM: {psutil.virtual_memory().total / 1024**3:.1f} GB")

# === 설정 ===
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = Path("/kaggle/input/parquet")
INITIAL_WALLET = 13000.0
BACKTEST_START = datetime(2024, 2, 1)
BACKTEST_END = datetime(2026, 3, 1)
LONG_UTILIZATION = 0.10
SHORT_UTILIZATION = 0.20
LONG_LEVERAGE = 20
SHORT_LEVERAGE = 20

print(f"Device: {DEVICE}")
print(f"Long: {LONG_UTILIZATION*100}% util, {LONG_LEVERAGE}x lever")
print(f"Short: {SHORT_UTILIZATION*100}% util, {SHORT_LEVERAGE}x lever")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# === symbol_map (DB 대체) ===
_sym_df = pd.read_csv(DATA_DIR / "symbol_map.csv")
SYM_TO_ID = dict(zip(_sym_df["symbol"], _sym_df["id"]))
logger.info(f"symbol_map: {len(SYM_TO_ID)} symbols")


def get_symbol_id(symbol):
    return SYM_TO_ID.get(symbol)


# === hourly_top150 (DB 대체) ===
def load_hourly_top150(start_dt, end_dt):
    pq_path = DATA_DIR / "hourly_top150.parquet"
    table = pq.read_table(pq_path)
    df = table.to_pandas()
    time_col = "bucket" if "bucket" in df.columns else "hour"
    df[time_col] = pd.to_datetime(df[time_col], utc=True).dt.tz_localize(None)
    df = df[(df[time_col] >= start_dt) & (df[time_col] < end_dt)]
    result = {}
    for _, row in df.iterrows():
        dt = row[time_col].to_pydatetime().replace(minute=0, second=0, microsecond=0)
        syms = row["symbols"] if isinstance(row["symbols"], list) else []
        result[dt] = set(syms)
    logger.info(f"hourly_top150: {len(result)} hours")
    return result


# === Parquet 데이터 로더 ===
def load_batch(symbols, start_date, end_date):
    symbol_ids = [get_symbol_id(s) for s in symbols]
    valid_sid = [sid for sid in symbol_ids if sid is not None]
    feat_cols = [f"f{i:02d}" for i in range(79)]
    ind_dir = DATA_DIR / "indicator"

    months = []
    cur = datetime(start_date.year, start_date.month, 1)
    while cur < end_date:
        months.append(cur)
        cur = datetime(cur.year + (1 if cur.month == 12 else 0), (cur.month % 12) + 1, 1)

    start_ts = pd.Timestamp(start_date, tz="UTC")
    end_ts = pd.Timestamp(end_date, tz="UTC")

    tables = []
    for m in months:
        pf = ind_dir / f"{m.strftime('%Y-%m')}.parquet"
        if not pf.exists():
            continue
        t0 = time.time()
        tbl = pq.read_table(pf, columns=["symbol_id", "ts"] + feat_cols,
                            filters=[("symbol_id", "in", valid_sid), ("ts", ">=", start_ts), ("ts", "<", end_ts)])
        logger.info(f"  {m.strftime('%Y-%m')}: {tbl.num_rows} rows in {time.time()-t0:.2f}s")
        if tbl.num_rows > 0:
            tables.append(tbl)

    if not tables:
        raise ValueError(f"No data for {start_date.date()} ~ {end_date.date()}")

    table = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
    sid_col = table["symbol_id"].to_numpy(zero_copy_only=False)
    feat_mat = table.select(feat_cols).to_pandas(self_destruct=True).values.astype(np.float32, copy=False)
    del table, tables; gc.collect()

    sort_idx = np.argsort(sid_col, kind="stable")
    sid_sorted = sid_col[sort_idx]
    feat_sorted = feat_mat[sort_idx]
    del sid_col, feat_mat; gc.collect()

    unique_sids, counts = np.unique(sid_sorted, return_counts=True)
    offsets = np.concatenate([[0], np.cumsum(counts)])
    n_bars = int(counts.max())
    n_sym = len(symbols)
    feature_np = np.zeros((n_sym, n_bars, 79), dtype=np.float32)

    sid_to_idx = {sid: i for i, sid in enumerate(symbol_ids) if sid is not None}
    for j, (sid, n) in enumerate(zip(unique_sids, counts)):
        idx = sid_to_idx.get(int(sid))
        if idx is None:
            continue
        s = int(offsets[j])
        feature_np[idx, :n, :] = feat_sorted[s:s+n]

    del sort_idx, sid_sorted, feat_sorted; gc.collect()

    # funding overlay
    funding_path = DATA_DIR / "funding_rate.parquet"
    if funding_path.exists():
        _overlay_funding(feature_np, symbols, symbol_ids, start_date, end_date, n_bars, funding_path)

    # GPU 전송
    tensor = torch.from_numpy(feature_np).to(dtype=torch.float32)
    if DEVICE == "cuda":
        pinned = tensor.pin_memory()
        del tensor, feature_np
        tensor_gpu = pinned.to(DEVICE, non_blocking=True)
        torch.cuda.synchronize()
        del pinned; gc.collect()
        logger.info(f"GPU: {tensor_gpu.shape}, VRAM={torch.cuda.memory_allocated()/1024**3:.2f}GB")
        return tensor_gpu
    del feature_np; gc.collect()
    return tensor


def _overlay_funding(feature_np, symbols, symbol_ids, start_date, end_date, n_bars, funding_path):
    start_utc = start_date.replace(tzinfo=timezone.utc)
    end_utc = end_date.replace(tzinfo=timezone.utc)
    fr_table = pq.read_table(funding_path, filters=[("funding_time", ">=", start_utc), ("funding_time", "<", end_utc)])
    if fr_table.num_rows == 0:
        return
    fr_df = fr_table.to_pandas()
    sym_to_idx = {s: i for i, s in enumerate(symbols)}
    filled = 0
    for sym, grp in fr_df.groupby("symbol"):
        idx = sym_to_idx.get(sym)
        if idx is None:
            continue
        for _, row in grp.iterrows():
            ft = row["funding_time"].to_pydatetime().replace(tzinfo=None)
            bar_idx = int((ft - start_date.replace(tzinfo=None)).total_seconds() / 60)
            if 0 <= bar_idx < n_bars:
                end_bar = min(bar_idx + 480, n_bars)
                feature_np[idx, bar_idx:end_bar, 78] = row["funding_rate"]
                filled += 1
    logger.info(f"Funding overlay: {filled} entries")


def get_month_symbols(hourly, batch_start, batch_end, max_symbols=200, min_appearances=100):
    counter = Counter()
    cur = batch_start.replace(tzinfo=None)
    end = batch_end.replace(tzinfo=None)
    while cur < end:
        hour_dt = cur.replace(minute=0, second=0, microsecond=0)
        counter.update(hourly.get(hour_dt, set()))
        cur += timedelta(hours=1)
    candidates = sorted(((s, c) for s, c in counter.items() if c >= min_appearances), key=lambda x: x[1], reverse=True)
    return sorted(s for s, _ in candidates[:max_symbols])


def generate_monthly_batches(start, end):
    batches = []
    cur = start
    while cur < end:
        nxt = datetime(cur.year + (1 if cur.month == 12 else 0), (cur.month % 12) + 1, 1)
        if nxt > end:
            nxt = end
        batches.append((cur, nxt))
        cur = nxt
    return batches


# === 메인: 전체 백테스트 실행 ===
if __name__ == "__main__":
    from simulator import GPUPortfolioSimulator

    logger.info("=" * 60)
    logger.info("Kaggle GPU Backtest")
    logger.info(f"Period: {BACKTEST_START.date()} ~ {BACKTEST_END.date()}")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Long: {LONG_UTILIZATION*100}%/{LONG_LEVERAGE}x, Short: {SHORT_UTILIZATION*100}%/{SHORT_LEVERAGE}x")
    logger.info("=" * 60)

    # 시뮬레이터 생성
    simulator = GPUPortfolioSimulator(
        initial_wallet=INITIAL_WALLET,
        long_utilization=LONG_UTILIZATION,
        short_utilization=SHORT_UTILIZATION,
        long_leverage=LONG_LEVERAGE,
        short_leverage=SHORT_LEVERAGE,
        device=DEVICE,
    )

    # 데이터 로드
    hourly = load_hourly_top150(BACKTEST_START, BACKTEST_END)
    batches = generate_monthly_batches(BACKTEST_START, BACKTEST_END)
    logger.info(f"Batches: {len(batches)}")

    total_start = time.time()

    for i, (batch_start, batch_end) in enumerate(batches):
        symbols = get_month_symbols(hourly, batch_start, batch_end)

        logger.info(f"\n{'='*60}")
        logger.info(f"[Batch {i+1}/{len(batches)}] {batch_start.strftime('%Y-%m')} ({batch_start.date()} ~ {batch_end.date()})")
        logger.info(f"  Symbols: {len(symbols)}, Top 5: {symbols[:5]}")

        # 데이터 로드
        t0 = time.time()
        data_tensor = load_batch(symbols, batch_start, batch_end)
        load_time = time.time() - t0

        # 시뮬레이션 실행
        t1 = time.time()
        equity_before = simulator.equity
        trades_before = len(simulator.trades)

        with torch.no_grad():
            result = simulator.run_simulation(
                symbols=symbols,
                start_date=batch_start,
                end_date=batch_end,
                data_tensor=data_tensor,
                hourly_top150=hourly,
            )

        sim_time = time.time() - t1
        equity_after = simulator.equity
        withdrawn_after = simulator.total_withdrawn
        month_trades = len(simulator.trades) - trades_before
        month_pnl = (equity_after - equity_before) + (withdrawn_after - (simulator.total_withdrawn - (withdrawn_after - withdrawn_after)))

        logger.info(f"  Load: {load_time:.1f}s, Sim: {sim_time:.1f}s")
        logger.info(f"  Equity: ${equity_after:,.0f}, Trades: {month_trades}, Total: {len(simulator.trades)}")
        logger.info(f"  RAM: {psutil.Process().memory_info().rss/1024**3:.2f}GB, "
                     f"VRAM: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

        # 쿨다운 이월
        batch_bars = int((batch_end - batch_start).total_seconds() / 60)
        simulator.cooldown_until = max(0, simulator.cooldown_until - batch_bars)

        # 메모리 해제
        del data_tensor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_time = time.time() - total_start

    # === 최종 결과 ===
    logger.info("\n" + "=" * 60)
    logger.info("FINAL RESULTS")
    logger.info("=" * 60)

    trades = simulator.get_trades()
    final_equity = simulator.equity
    total_withdrawn = simulator.total_withdrawn
    total_asset = final_equity + total_withdrawn
    total_return = (total_asset - INITIAL_WALLET) / INITIAL_WALLET * 100

    logger.info(f"Final Equity: ${final_equity:,.0f}")
    logger.info(f"Total Withdrawn: ${total_withdrawn:,.0f}")
    logger.info(f"Total Asset: ${total_asset:,.0f} (+{total_return:,.1f}%)")
    logger.info(f"Total Trades: {len(trades)}")
    logger.info(f"Total Time: {total_time:.0f}s ({total_time/len(batches):.1f}s/batch)")

    # 방향별 요약
    long_trades = [t for t in trades if t["direction"] == "long"]
    short_trades = [t for t in trades if t["direction"] == "short"]
    long_pnl = sum(t["realized_pnl"] for t in long_trades)
    short_pnl = sum(t["realized_pnl"] for t in short_trades)
    long_wr = sum(1 for t in long_trades if t["realized_pnl"] > 0) / max(len(long_trades), 1) * 100
    short_wr = sum(1 for t in short_trades if t["realized_pnl"] > 0) / max(len(short_trades), 1) * 100

    logger.info(f"\nLONG:  {len(long_trades)} trades, WR {long_wr:.1f}%, PnL ${long_pnl:+,.0f}")
    logger.info(f"SHORT: {len(short_trades)} trades, WR {short_wr:.1f}%, PnL ${short_pnl:+,.0f}")

    # 결과 CSV 저장 (Kaggle output으로 다운로드 가능)
    import json
    output_dir = Path("/kaggle/working")
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / "result_summary.json", "w") as f:
        json.dump({
            "total_asset": total_asset,
            "total_return_pct": total_return,
            "final_equity": final_equity,
            "total_withdrawn": total_withdrawn,
            "total_trades": len(trades),
            "long_trades": len(long_trades),
            "short_trades": len(short_trades),
            "long_pnl": long_pnl,
            "short_pnl": short_pnl,
            "long_wr": long_wr,
            "short_wr": short_wr,
            "total_time_sec": total_time,
            "device": DEVICE,
            "long_leverage": LONG_LEVERAGE,
            "short_leverage": SHORT_LEVERAGE,
        }, f, indent=2)

    pd.DataFrame(trades).to_csv(output_dir / "trades.csv", index=False)
    logger.info(f"\nResults saved to {output_dir}")
    logger.info("=" * 60)
