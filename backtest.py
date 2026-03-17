"""
GPU Portfolio Simulator - Scalping Strategy with Progressive DCA

전략:
- Entry: 7 TF CMO 정렬 + 1m BB 터치
- Exit: BB 반대편 OR 모멘텀 반전
- DCA: 64 units (1+1+2+4+8+16+32)
- Position: 멀티 포지션 (Long + Short 동시 보유 가능)
  - Long: 20x 레버리지, 10% 자금 활용 (v42+롱20x)
  - Short: 20x 레버리지, 20% 자금 활용
  - Short 존재 시 Long 신규진입 금지, Long 존재 시 Short 신규진입 허용
- 복리: equity 기반 유닛 재계산
"""

import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch

# DB/데이터로더 의존성 제거 - Kaggle에서 Parquet 직접 로딩

logger = logging.getLogger(__name__)


# ============================================================================
# Position Management
# ============================================================================


@dataclass
class Position:
    """포지션 상태 관리"""

    symbol_idx: int  # 배치 내 인덱스
    symbol_name: str
    direction: int  # 1=long, -1=short
    entry_price: float
    entry_time: int  # 시간 인덱스
    entry_tf: int  # 진입근거 TF 인덱스 (DCA/Exit 모멘텀 체크용)
    strategy: str  # "S2", "S3", or "S4h" (Exit BB 결정용)
    dca_level: int  # 0~6
    total_margin: float
    total_notional: float
    entry_fee_paid: float = 0.0  # 진입 시 즉시 차감된 누적 수수료 (LP 계산 정확도)
    entry_equity: float = 0.0   # 진입 시점 잔액 (ENTRY_STOP -50% 기준)
    def to_dict(self) -> Dict:
        """거래 내역 출력용"""
        return {
            "symbol": self.symbol_name,
            "direction": "long" if self.direction == 1 else "short",
            "entry_time_idx": self.entry_time,
            "entry_price": self.entry_price,
            "entry_tf": self.entry_tf,
            "strategy": self.strategy,
            "dca_level": self.dca_level,
            "total_margin": self.total_margin,
            "total_notional": self.total_notional,
            "entry_equity": self.entry_equity,
        }


# ============================================================================
# GPU Portfolio Simulator
# ============================================================================


class GPUPortfolioSimulator:
    """
    GPU 벡터 연산 기반 포트폴리오 시뮬레이터

    특징:
    - 시간축 순차 처리 (t=0 → T)
    - 배치 심볼 병렬 처리 (GPU)
    - 멀티 포지션: Long + Short 동시 보유 가능
      - Short 존재 시 Long 신규진입 금지
      - Long 존재 시 Short 신규진입 허용
    - 복리 적용
    """

    def __init__(
        self,
        initial_wallet: float = 13000.0,
        long_utilization: float = 0.10,
        short_utilization: float = 0.20,
        long_leverage: int = 20,
        short_leverage: int = 20,
        taker_fee: float = 0.0005,
        slippage: float = 0.0002,
        device: str = "cuda",
    ):
        """
        Args:
            initial_wallet: 초기 자본 (USD)
            long_utilization: 롱 자금 활용률 (0.10 = 10%)
            short_utilization: 숏 자금 활용률 (0.20 = 20%)
            long_leverage: 롱 레버리지 (20x)
            short_leverage: 숏 레버리지 (20x)
            taker_fee: Taker 수수료 (0.05%)
            slippage: 슬리피지 (0.02%)
            device: 'cuda' or 'cpu'
        """
        self.initial_wallet = initial_wallet
        self.long_utilization = long_utilization
        self.short_utilization = short_utilization
        self.long_leverage = long_leverage
        self.short_leverage = short_leverage
        self.taker_fee = taker_fee          # 수수료만 (슬리피지는 가격에 반영)
        self.base_slippage = slippage       # 기준 슬리피지 (0.02%)
        self.device = device

        # DCA 구조 (64 units)
        self.dca_units = [1, 1, 2, 4, 8, 16, 32]
        self.dca_tf_triggers = [0, 0, 1, 2, 3, 4, 5]  # TF 인덱스

        # 동적 포지션 축소: 손실 누적 시 진입 규모 줄이기 (update_unit_sizing 전 초기화 필수)
        self.position_scale: float = 1.0   # unit_margin 배율 (0.10 ~ 1.0)
        self.loss_count_scale: int = 0     # 현재 축소 누적 횟수

        # 유닛 사이징 (복리 적용)
        self.equity = initial_wallet
        self.update_unit_sizing()

        # 결과 저장
        self.trades: List[Dict] = []
        self.equity_curve: List[float] = [initial_wallet]
        self.long_position: Optional[Position] = None
        self.short_position: Optional[Position] = None
        self.funding_cost_total: float = 0.0

        # 23:59 일일 수익 인출 추적
        self.day_start_equity: float = initial_wallet
        self.total_withdrawn: float = 0.0
        self.daily_withdrawn_log: List[Dict] = []

        # 연속 손실 쿨다운 (2회 → 6시간 거래 중단)
        self.consecutive_losses: int = 0
        self.cooldown_until: int = 0

        # same-bar 재진입 방지
        self.last_close_bar: int = -1

        # HWM (성과 통계용만 — 긴급탈출 트리거 아님, entry_equity 기반으로 변경됨)
        self.hwm: float = initial_wallet
        # blocked_until_dt: 심볼별 재진입 차단 만료 시각 (7,200분=5일)
        self.blocked_until_dt: Dict[str, datetime] = {}
        # blocked_until_bar: 배치 내 bar 인덱스 기반 차단 만료 bar (GPU 텐서, 배치 시작마다 재빌드)
        self.blocked_until_bar: Optional[torch.Tensor] = None

        # 레짐 필터: BTC 4h StochRSI 기반 방향 어드밴티지
        self.market_regime: int = 0              # 0=중립, 1=하락경고(숏우선), -1=상승경고(롱우선)
        self.regime_advantage: float = 3.0       # 순방향 어드밴티지 (3배 가산)

        logger.info(
            f"GPUPortfolioSimulator initialized: wallet=${initial_wallet:.2f}, "
            f"long={long_utilization*100:.0f}%/{long_leverage}x, short={short_utilization*100:.0f}%/{short_leverage}x, "
            f"taker_fee={taker_fee*100:.3f}%, base_slippage={slippage*100:.3f}%, "
            f"long_unit_margin=${self.long_unit_margin:.2f}, short_unit_margin=${self.short_unit_margin:.2f}"
        )

    def update_unit_sizing(self):
        """
        복리: equity 기준으로 유닛 재계산 (방향별 분리)
        롱: equity * 10% / 64 * 5x
        숏: equity * 20% / 64 * 20x
        """
        self.long_unit_margin = self.equity * self.long_utilization / 64 * self.position_scale
        self.long_unit_notional = self.long_unit_margin * self.long_leverage
        self.short_unit_margin = self.equity * self.short_utilization / 64 * self.position_scale
        self.short_unit_notional = self.short_unit_margin * self.short_leverage

    def _calc_slippage(self, total_notional: float, is_emergency: bool = False) -> float:
        """
        포지션 규모(notional) 비례 동적 슬리피지 계산

        DCA가 진행될수록 포지션이 커지면 호가창을 더 깊게 소진 → 슬리피지 증가.
        긴급 청산(ENTRY_STOP)은 패닉셀 환경 → 2.5배 가산.

        Args:
            total_notional: 진입/청산 시점의 포지션 명목가치 ($)
            is_emergency: ENTRY_STOP 긴급탈출 여부

        Returns:
            slip_rate: 적용할 슬리피지 비율
        """
        notional_factor = total_notional / max(self.equity, 1.0) * 0.0001
        rate = self.base_slippage + notional_factor
        rate = min(rate, 0.010)  # 최대 1.0% 캡
        if is_emergency:
            rate *= 2.5
        return rate

    def _apply_slippage_to_price(
        self, price: float, direction: int, total_notional: float, is_exit: bool = False,
        is_emergency: bool = False
    ) -> float:
        """
        슬리피지 반영 실효 체결가 계산

        진입(buy): 실제보다 비싸게 체결 (불리한 방향)
        청산(sell): 실제보다 싸게 체결 (불리한 방향)

        Args:
            direction: 1=long, -1=short
            is_exit: True=청산, False=진입
        """
        slip = self._calc_slippage(total_notional, is_emergency=is_emergency)
        if not is_exit:
            # 진입: Long은 더 비싸게, Short은 더 싸게 체결
            return price * (1.0 + slip) if direction == 1 else price * (1.0 - slip)
        else:
            # 청산: Long은 더 싸게, Short은 더 비싸게 체결
            return price * (1.0 - slip) if direction == 1 else price * (1.0 + slip)

    def _build_active_mask(
        self,
        symbols: List[str],
        start_date: datetime,
        time_steps: int,
        hourly_top150: Dict[datetime, Set[str]],
    ) -> torch.Tensor:
        """
        hourly_top150 기반 활성 심볼 마스크 사전 계산

        Args:
            symbols: 현재 배치 심볼 리스트
            start_date: 시뮬레이션 시작 시각 (1m 기준)
            time_steps: 총 바 수
            hourly_top150: {hour_datetime: set of symbol strings}

        Returns:
            active_mask: (n_symbols, n_bars) bool tensor on self.device
                         True = 해당 hour에 top150 포함 → 신규 진입 허용
        """
        n_symbols = len(symbols)
        sym_to_idx = {sym: i for i, sym in enumerate(symbols)}

        mask_np = np.zeros((n_symbols, time_steps), dtype=bool)

        base = start_date.replace(tzinfo=None)

        hour_count = (time_steps + 59) // 60
        for h in range(hour_count):
            t_start = h * 60
            if t_start >= time_steps:
                break
            t_end = min(t_start + 60, time_steps)

            hour_dt = (base + timedelta(hours=h)).replace(minute=0, second=0, microsecond=0)
            active_set = hourly_top150.get(hour_dt, set())

            active_indices = [sym_to_idx[s] for s in active_set if s in sym_to_idx]
            if active_indices:
                mask_np[active_indices, t_start:t_end] = True

        active_mask = torch.from_numpy(mask_np).to(self.device)
        active_ratio = active_mask.float().mean().item()
        logger.info(f"Active mask built: {active_ratio*100:.1f}% of (symbol, bar) pairs active")
        return active_mask

    def _build_top50_mask(
        self,
        symbols: List[str],
        start_date: datetime,
        time_steps: int,
        hourly_top150: Dict[datetime, Set[str]],
    ) -> torch.Tensor:
        """
        hourly top50 기반 마스크 사전 계산 (DCA5+ 필터용)

        hourly_top150 dict에서 상위 50개만 사용.
        (데이터가 Set이므로 정렬 기준 없이 임의 50개가 될 수 있음 — 유동성 필터 목적상 충분)

        Returns:
            top50_mask: (n_symbols, n_bars) bool tensor on self.device
        """
        n_symbols = len(symbols)
        sym_to_idx = {sym: i for i, sym in enumerate(symbols)}

        mask_np = np.zeros((n_symbols, time_steps), dtype=bool)

        base = start_date.replace(tzinfo=None)

        hour_count = (time_steps + 59) // 60
        for h in range(hour_count):
            t_start = h * 60
            if t_start >= time_steps:
                break
            t_end = min(t_start + 60, time_steps)

            hour_dt = (base + timedelta(hours=h)).replace(minute=0, second=0, microsecond=0)
            active_set = hourly_top150.get(hour_dt, set())

            # top50: set에서 임의 50개 (유동성 필터 목적)
            top50_set = set(list(active_set)[:50])
            active_indices = [sym_to_idx[s] for s in top50_set if s in sym_to_idx]
            if active_indices:
                mask_np[active_indices, t_start:t_end] = True

        top50_mask = torch.from_numpy(mask_np).to(self.device)
        top50_ratio = top50_mask.float().mean().item()
        logger.info(f"Top50 mask built: {top50_ratio*100:.1f}% of (symbol, bar) pairs in top50")
        return top50_mask

    def run_simulation(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        data_tensor: torch.Tensor = None,
        hourly_top150: Optional[Dict[datetime, Set[str]]] = None,
    ) -> Dict:
        """
        시뮬레이션 실행

        Args:
            symbols: 심볼 리스트
            start_date: 시작 날짜
            end_date: 종료 날짜
            hourly_top150: {hour_datetime: set_of_symbols} — None이면 전체 허용

        Returns:
            성과 지표 dict
        """
        logger.info(
            f"Starting simulation: {len(symbols)} symbols, "
            f"{start_date.date()} ~ {end_date.date()}, "
            f"hourly_top150={'enabled' if hourly_top150 else 'disabled'}"
        )

        # 시뮬레이션 기간 저장 (성과 계산용)
        self.start_date = start_date
        self.end_date = end_date

        if self.equity > self.hwm:
            self.hwm = self.equity
        batch_start_dt = start_date.replace(tzinfo=None)
        self.blocked_until_dt = {
            s: v for s, v in self.blocked_until_dt.items() if v > batch_start_dt
        }
        self.sim_base = batch_start_dt
        self.consecutive_losses = 0
        self.last_close_bar = -1

        n_sym_batch = len(symbols)
        self.blocked_until_bar = torch.zeros(n_sym_batch, dtype=torch.long, device=self.device)
        if self.blocked_until_dt:
            sym_to_idx_batch = {s: i for i, s in enumerate(symbols)}
            for sym, expire_dt in self.blocked_until_dt.items():
                if sym in sym_to_idx_batch:
                    remaining_min = (expire_dt - batch_start_dt).total_seconds() / 60
                    if remaining_min > 0:
                        self.blocked_until_bar[sym_to_idx_batch[sym]] = int(remaining_min)
        logger.info(
            f"blocked_until_bar built: {(self.blocked_until_bar > 0).sum().item()} symbols blocked"
        )

        # 0. 이전 배치 data_tensor 해제 (OOM 방지)
        if hasattr(self, '_sim_data_tensor'):
            delattr(self, '_sim_data_tensor')
        import gc; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 1. 데이터 (외부에서 전달받음)
        if data_tensor is None:
            raise ValueError("data_tensor must be provided (DB-free mode)")
        logger.info(f"Data received: {data_tensor.shape} (symbols, bars, features)")

        # 2. 피처 분리
        prices, rsi, cmo, prev_cmo, stochrsi, bb_upper, bb_lower, bb_pctb, adx, funding_rates = self._extract_features(data_tensor)
        prices_high = data_tensor[:, :, 1]
        prices_low  = data_tensor[:, :, 2]

        self._sim_data_tensor = data_tensor
        self._sim_base_dt = start_date.replace(tzinfo=None)

        # 3. 시간축 루프
        time_steps = data_tensor.shape[1]
        logger.info(f"Simulating {time_steps} time steps...")

        # hourly_top150 동적 필터
        if hourly_top150 is not None:
            active_mask = self._build_active_mask(symbols, start_date, time_steps, hourly_top150)
            top50_mask = self._build_top50_mask(symbols, start_date, time_steps, hourly_top150)
        else:
            active_mask = None
            top50_mask = None

        sim_base = start_date.replace(tzinfo=None)

        # 23:59 인출 bar 사전 계산
        base_total_min = sim_base.hour * 60 + sim_base.minute
        withdraw_bars = set(
            t for t in range(time_steps)
            if (base_total_min + t) % 1440 == 1439
        )

        # 펀딩비 발생 bar 사전 계산
        _funding_hours = {0, 8, 16}
        funding_bars = set(
            t for t in range(time_steps)
            if (sim_base + timedelta(minutes=t)).minute == 0
            and (sim_base + timedelta(minutes=t)).hour in _funding_hours
        )
        logger.info(
            f"Funding bars precomputed: {len(funding_bars)} bars "
            f"(UTC 00:00/08:00/16:00, first few: {sorted(list(funding_bars))[:5]})"
        )

        # BTC 인덱스
        btc_idx = symbols.index("BTCUSDT") if "BTCUSDT" in symbols else -1
        if btc_idx >= 0:
            logger.info(f"BTC idx={btc_idx} for regime filter (4h StRSI, current={self.market_regime})")
        else:
            logger.warning("BTCUSDT not in batch — regime filter disabled this batch")

        for t in range(time_steps):
            # Step 1: 펀딩비 (UTC 00:00/08:00/16:00 정각) — 포지션 보유 시 즉시 차감
            if t in funding_bars:
                if self.long_position is not None or self.short_position is not None:
                    self._apply_funding_cost(funding_rates, t)
                    # 펀딩비로 equity 감소 → LP 도달 여부 즉시 확인
                    self._check_liquidation(prices, prices_low, prices_high, t)

            # Step 2: 23:59 일일 수익 인출
            if t in withdraw_bars:
                bar_dt = sim_base + timedelta(minutes=t)
                daily_profit = self.equity - self.day_start_equity
                if daily_profit > 0:
                    withdraw = daily_profit * 0.5
                    self.equity -= withdraw
                    self.total_withdrawn += withdraw
                    if self.equity > self.hwm:
                        self.hwm = self.equity
                    self.update_unit_sizing()
                    self.daily_withdrawn_log.append({
                        "date": bar_dt.strftime("%Y-%m-%d"),
                        "equity_before": self.day_start_equity,
                        "equity_at_withdraw": self.equity + withdraw,
                        "daily_profit": daily_profit,
                        "withdrawn": withdraw,
                        "equity_after": self.equity,
                        "total_withdrawn": self.total_withdrawn,
                    })
                    logger.info(
                        f"[WITHDRAW] {bar_dt.date()} Daily profit=${daily_profit:.2f}, "
                        f"Withdrawn=${withdraw:.2f}, Equity=${self.equity:.2f}, "
                        f"Cumulative withdrawn=${self.total_withdrawn:.2f}"
                    )
                self.day_start_equity = self.equity

            # BTC 4h StochRSI 기반 레짐 업데이트
            if t % 240 == 0 and btc_idx >= 0:
                btc_4h_stochrsi = stochrsi[btc_idx, t, 5].item()
                prev_regime = self.market_regime

                if btc_4h_stochrsi >= 80:
                    self.market_regime = 1
                elif btc_4h_stochrsi <= 20:
                    self.market_regime = -1
                else:
                    self.market_regime = 0

                if self.market_regime != prev_regime:
                    regime_str = {0: "NEUTRAL", 1: "BEAR(숏우선)", -1: "SURGE(롱우선)"}
                    bar_dt = sim_base + timedelta(minutes=t)
                    logger.info(
                        f"[REGIME] {bar_dt.strftime('%Y-%m-%d %H:%M')} BTC 4h StRSI={btc_4h_stochrsi:.1f} "
                        f"-> {regime_str[self.market_regime]}"
                    )

            # 진행 상황 로깅 (1000 bars마다)
            if t % 1000 == 0:
                long_sym = self.long_position.symbol_name if self.long_position else 'None'
                short_sym = self.short_position.symbol_name if self.short_position else 'None'
                logger.info(
                    f"Progress: {t}/{time_steps} bars ({t/time_steps*100:.1f}%), "
                    f"Equity: ${self.equity:.2f}, Trades: {len(self.trades)}, "
                    f"Long: {long_sym}, Short: {short_sym}"
                )

            # 진입 분석 로깅 (10000 bars마다)
            if t % 10000 == 0 and self.long_position is None and self.short_position is None and logger.isEnabledFor(logging.DEBUG):
                short_tf_oversold = (cmo[:, t, :4] < -50).sum(dim=1)
                long_tf_oversold = (cmo[:, t, 4:] < -40).sum(dim=1)
                total_oversold = short_tf_oversold + long_tf_oversold
                bb_s2_count = (prices[:, t] <= bb_lower[:, t, 1]).sum().item()
                bb_s3_count = (prices[:, t] <= bb_lower[:, t, 2]).sum().item()
                active_count = active_mask[:, t].sum().item() if active_mask is not None else len(symbols)
                logger.debug(
                    f"[{t}] No entry: Active={active_count}/{len(symbols)}, "
                    f"Oversold5+={( total_oversold >= 5).sum().item()}, "
                    f"BB_S2={bb_s2_count}, BB_S3={bb_s3_count}"
                )

            # 파산 체크
            if self.equity <= 0:
                logger.error(f"[BANKRUPTCY] t={t} equity={self.equity:.2f}. Stopping simulation.")
                break

            # price=0 안전장치: 포지션 보유 중 해당 심볼 가격이 0이면 강제 청산
            for pos in [self.long_position, self.short_position]:
                if pos is not None:
                    pos_price = prices[pos.symbol_idx, t].item()
                    if pos_price <= 0:
                        logger.warning(
                            f"[DATA_GAP] t={t} price=0 for {pos.symbol_name} "
                            f"({'LONG' if pos.direction == 1 else 'SHORT'}), force closing at prev bar"
                        )
                        self._close_position(prices, max(0, t - 1), "DATA_GAP", pos.direction)

            # HWM 갱신
            if self.equity > self.hwm:
                self.hwm = self.equity

            # Step 5: Entry
            # 쿨다운 + same-bar 재진입 차단 조건
            can_enter = t >= self.cooldown_until and t != self.last_close_bar
            if can_enter:
                bar_active_mask = active_mask[:, t] if active_mask is not None else None
                self._process_entry(symbols, prices, bb_upper, bb_lower, cmo, prev_cmo, stochrsi, t, bar_active_mask)

            # Step 6: ENTRY_STOP — 포지션별 독립 체크
            # Long ENTRY_STOP
            if self.long_position is not None:
                pos = self.long_position
                entry_eq = pos.entry_equity
                worst_price = prices_low[pos.symbol_idx, t].item()
                unrealized = (worst_price - pos.entry_price) / pos.entry_price * pos.total_notional
                mtm_equity = self.equity + unrealized
                entry_drawdown = (entry_eq - mtm_equity) / entry_eq if entry_eq > 0 else 0.0
                if entry_drawdown >= 0.50:
                    sym = pos.symbol_name
                    sym_idx = pos.symbol_idx
                    logger.info(
                        f"[ENTRY_STOP/LONG] t={t} entry_drawdown={entry_drawdown:.1%} "
                        f"entry_equity=${entry_eq:.2f} MTM=${mtm_equity:.2f} blocked={sym}"
                    )
                    self._close_position(prices, t, "ENTRY_STOP", direction=1)
                    block_expire_dt = (self.sim_base + timedelta(minutes=t)) + timedelta(minutes=7200)
                    self.blocked_until_dt[sym] = block_expire_dt
                    if self.blocked_until_bar is not None:
                        self.blocked_until_bar[sym_idx] = t + 7200
                    logger.info(f"[BLOCK] {sym} blocked until {block_expire_dt} (7200min=5days)")

            # Short ENTRY_STOP
            if self.short_position is not None:
                pos = self.short_position
                entry_eq = pos.entry_equity
                worst_price = prices_high[pos.symbol_idx, t].item()
                unrealized = -1 * (worst_price - pos.entry_price) / pos.entry_price * pos.total_notional
                mtm_equity = self.equity + unrealized
                entry_drawdown = (entry_eq - mtm_equity) / entry_eq if entry_eq > 0 else 0.0
                if entry_drawdown >= 0.50:
                    sym = pos.symbol_name
                    sym_idx = pos.symbol_idx
                    logger.info(
                        f"[ENTRY_STOP/SHORT] t={t} entry_drawdown={entry_drawdown:.1%} "
                        f"entry_equity=${entry_eq:.2f} MTM=${mtm_equity:.2f} blocked={sym}"
                    )
                    self._close_position(prices, t, "ENTRY_STOP", direction=-1)
                    block_expire_dt = (self.sim_base + timedelta(minutes=t)) + timedelta(minutes=7200)
                    self.blocked_until_dt[sym] = block_expire_dt
                    if self.blocked_until_bar is not None:
                        self.blocked_until_bar[sym_idx] = t + 7200
                    logger.info(f"[BLOCK] {sym} blocked until {block_expire_dt} (7200min=5days)")

            # Step 7: DCA — 포지션별 독립 적용 (청산 선행 체크)
            if self.long_position is not None:
                self._check_liquidation(prices, prices_low, prices_high, t)
            if self.long_position is not None:
                self._process_dca(prices, bb_upper, bb_lower, cmo, prev_cmo, stochrsi, t,
                                   self.long_position, top50_mask)

            if self.short_position is not None:
                self._check_liquidation(prices, prices_low, prices_high, t)
            if self.short_position is not None:
                self._process_dca(prices, bb_upper, bb_lower, cmo, prev_cmo, stochrsi, t,
                                   self.short_position, top50_mask)

            # Step 8: LP 체크 — 포지션별 독립
            if self.long_position is not None:
                self._check_liquidation(prices, prices_low, prices_high, t)
            if self.short_position is not None:
                self._check_liquidation(prices, prices_low, prices_high, t)

            # Step 9: Exit — 포지션별 독립
            if self.long_position is not None:
                self._process_exit(prices, prices_high, prices_low, bb_upper, bb_lower, cmo, prev_cmo, stochrsi, t,
                                   self.long_position)
            if self.short_position is not None:
                self._process_exit(prices, prices_high, prices_low, bb_upper, bb_lower, cmo, prev_cmo, stochrsi, t,
                                   self.short_position)

            # Equity curve 기록 (매 100 bars)
            if t % 100 == 0:
                self.equity_curve.append(self.equity)

        # 4. 미청산 포지션 강제 청산
        if self.long_position is not None:
            logger.warning(f"Force closing LONG position at end: {self.long_position.symbol_name}")
            self._close_position(prices, time_steps - 1, "FORCE_CLOSE", direction=1)
        if self.short_position is not None:
            logger.warning(f"Force closing SHORT position at end: {self.short_position.symbol_name}")
            self._close_position(prices, time_steps - 1, "FORCE_CLOSE", direction=-1)

        # 5. 대형 GPU 텐서 즉시 해제
        del prices, prices_high, prices_low
        del rsi, cmo, prev_cmo, stochrsi, bb_upper, bb_lower, bb_pctb, adx, funding_rates
        del data_tensor

        # 6. 성과 계산
        result = self._calculate_performance()

        logger.info(
            f"Simulation completed: {len(self.trades)} trades, "
            f"Final equity: ${self.equity:.2f}, "
            f"Return: {result['return_pct']:.2f}%"
        )

        return result

    def _extract_features(
        self, data_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        데이터 텐서에서 피처 분리

        Args:
            data_tensor: (n_symbols, n_bars, 79)

        Returns:
            prices: (n_symbols, n_bars) - close price
            rsi: (n_symbols, n_bars, 9) - RSI (9 TF)
            cmo: (n_symbols, n_bars, 9) - CMO (9 TF)
            prev_cmo: (n_symbols, n_bars, 9) - prev_CMO (9 TF)
            stochrsi: (n_symbols, n_bars, 9) - StochRSI_K (9 TF)
            bb_upper: (n_symbols, n_bars, 9) - BB upper (9 TF)
            bb_lower: (n_symbols, n_bars, 9) - BB lower (9 TF)
            bb_pctb: (n_symbols, n_bars, 9) - BB %B (9 TF)
            adx: (n_symbols, n_bars, 9) - ADX (9 TF)
            funding_rates: (n_symbols, n_bars) - funding rate

        피처 순서 (총 79개):
            [0-5]: open, high, low, close, volume, quote_volume
            [6-13]: 1m_rsi, 1m_cmo, 1m_prev_cmo, 1m_stochrsi_k, 1m_bb_upper, 1m_bb_lower, 1m_bb_pctb, 1m_adx
            [14-21]: 5m_rsi, 5m_cmo, 5m_prev_cmo, 5m_stochrsi_k, 5m_bb_upper, 5m_bb_lower, 5m_bb_pctb, 5m_adx
            [22-29]: 15m (same pattern)
            [30-37]: 30m (same pattern)
            [38-45]: 1h (same pattern)
            [46-53]: 4h (same pattern)
            [54-61]: 1d (same pattern)
            [62-69]: 1w (same pattern)
            [70-77]: 1M (same pattern)
            [78]: funding_rate
        """
        # Close price + funding rate
        prices        = data_tensor[:, :, 3]
        funding_rates = data_tensor[:, :, 78]

        n_sym, n_bars, _ = data_tensor.shape
        feat_block = data_tensor[:, :, 6:78].reshape(n_sym, n_bars, 9, 8)

        rsi      = feat_block[:, :, :, 0]
        cmo      = feat_block[:, :, :, 1]
        prev_cmo = feat_block[:, :, :, 2]
        stochrsi = feat_block[:, :, :, 3]
        bb_upper = feat_block[:, :, :, 4]
        bb_lower = feat_block[:, :, :, 5]
        bb_pctb  = feat_block[:, :, :, 6]
        adx      = feat_block[:, :, :, 7]

        logger.info(
            f"Extracted features: prices={prices.shape}, "
            f"rsi={rsi.shape}, cmo={cmo.shape}, prev_cmo={prev_cmo.shape}, "
            f"stochrsi={stochrsi.shape}, bb_upper={bb_upper.shape}, "
            f"bb_lower={bb_lower.shape}, bb_pctb={bb_pctb.shape}, adx={adx.shape}, "
            f"funding_rates={funding_rates.shape}"
        )

        return prices, rsi, cmo, prev_cmo, stochrsi, bb_upper, bb_lower, bb_pctb, adx, funding_rates

    def _compute_long_entry_mask(
        self,
        prices: torch.Tensor,
        bb_lower: torch.Tensor,
        cmo: torch.Tensor,
        prev_cmo: torch.Tensor,
        stochrsi: torch.Tensor,
        t: int,
    ) -> torch.Tensor:
        """
        롱 진입 마스크 계산 (S4h — 4h BB 트리거 기반)

        조건:
        1. 4h BB lower 터치 (bb_idx=5)
        2. 4h 모멘텀 요건: CMO[4h] < -50 AND CMO[4h] >= prev_CMO[4h]
                           AND StochRSI[4h] < 20 AND StochRSI[4h] >= prev_StochRSI[4h]
        3. TF 합의: 4h(idx5)는 필수 1개 + 나머지 8개 TF 중 4개 이상 → 총 5개 이상
           - 단기 TF (0-3: 1m,5m,15m,30m): CMO < -50 AND StochRSI < 25
           - 장기 TF (4-8: 1h,4h,1d,1w,1M): CMO < -40 AND cmo >= prev_cmo AND StochRSI < 30 AND stochrsi >= prev_stochrsi
        4. 포지션 없음 체크는 호출자에서 처리

        Returns:
            mask: (n_symbols,) bool tensor
        """
        if self.short_position is not None:
            # short 존재 시 long 진입 금지
            return torch.zeros(cmo.shape[0], dtype=torch.bool, device=cmo.device)

        t_prev = max(0, t - 1)

        # 4h (tf_idx=5) 필수 조건
        cmo_4h = cmo[:, t, 5]
        prev_cmo_4h = prev_cmo[:, t, 5]
        stochrsi_4h = stochrsi[:, t, 5]
        prev_stochrsi_4h = stochrsi[:, t_prev, 5]

        bb_touch_4h = prices[:, t] <= bb_lower[:, t, 4]  # 1h BB lower (idx=4)
        cond_4h = (
            (cmo_4h < -50)
            & (cmo_4h >= prev_cmo_4h)
            & (stochrsi_4h < 20)
            & (stochrsi_4h >= prev_stochrsi_4h)
        )
        # 4h 조건: BB 터치 + 모멘텀 요건 → 필수
        mandatory_4h = bb_touch_4h & cond_4h  # (n_symbols,)

        # 나머지 8개 TF (0-4: 1m,5m,15m,30m,1h + 6-8: 1d,1w,1M) 중 4개 이상
        short_cmo = cmo[:, t, :4] < -50
        short_stoch = stochrsi[:, t, :4] < 25
        short_ok = short_cmo & short_stoch  # (n_symbols, 4)

        # 장기 TF (4-8) 단 4h(idx5) 제외 → 1h(4), 1d(6), 1w(7), 1M(8)
        # 인덱스 4(1h) + 6,7,8(1d,1w,1M) → 별도 처리
        tf_1h_cmo = (cmo[:, t, 4] < -40) & (cmo[:, t, 4] >= prev_cmo[:, t, 4])
        tf_1h_stoch = (stochrsi[:, t, 4] < 30) & (stochrsi[:, t, 4] >= stochrsi[:, t_prev, 4])
        tf_1h_ok = tf_1h_cmo & tf_1h_stoch  # (n_symbols,)

        long_upper_cmo = (cmo[:, t, 6:] < -40) & (cmo[:, t, 6:] >= prev_cmo[:, t, 6:])
        long_upper_stoch = (stochrsi[:, t, 6:] < 30) & (stochrsi[:, t, 6:] >= stochrsi[:, t_prev, 6:])
        long_upper_ok = long_upper_cmo & long_upper_stoch  # (n_symbols, 3)

        # 8개 TF 합산 (4h 제외): short_ok(4) + tf_1h_ok(1) + long_upper_ok(3)
        remaining_ok = torch.cat([
            short_ok,                         # (n_symbols, 4)
            tf_1h_ok.unsqueeze(1),            # (n_symbols, 1)
            long_upper_ok,                    # (n_symbols, 3)
        ], dim=1)  # (n_symbols, 8)
        four_plus_remaining = remaining_ok.sum(dim=1) >= 4  # (n_symbols,)

        if t % 10000 == 0:
            logger.info(
                f"[{t}] Long entry (S4h): mandatory_4h={mandatory_4h.any().item()}, "
                f"4plus_remaining={four_plus_remaining.any().item()}"
            )

        return mandatory_4h & four_plus_remaining

    def _compute_short_entry_mask(
        self,
        prices: torch.Tensor,
        bb_upper: torch.Tensor,
        cmo: torch.Tensor,
        prev_cmo: torch.Tensor,
        stochrsi: torch.Tensor,
        t: int,
        strategy: str = "S2",
    ) -> torch.Tensor:
        """
        숏 진입 마스크 계산 (Plan Step 2 - 역추세 스캘핑)

        조건:
        1. 9 TF 중 5+개 과매수 조건 충족
           - 단기 TF (0-3: 1m,5m,15m,30m): CMO > 50 AND StochRSI > 75
           - 장기 TF (4-8: 1h,4h,1d,1w,1M): CMO > 40 AND cmo <= prev_cmo AND StochRSI > 70 AND stochrsi <= prev_stochrsi
        2. BB 트리거 TF에서 BB upper 터치 + 트리거 TF 조건 충족
        3. Long 존재 여부와 무관하게 진입 가능

        Args:
            strategy: "S2" (5m BB 트리거) or "S3" (15m BB 트리거)
        """
        if self.short_position is not None:
            return torch.zeros(cmo.shape[0], dtype=torch.bool, device=cmo.device)

        t_prev = max(0, t - 1)

        # 단기 TF (0-3: 1m, 5m, 15m, 30m): CMO > 50 AND StochRSI > 75
        short_cmo = cmo[:, t, :4] > 50
        short_stoch = stochrsi[:, t, :4] > 75
        short_ok = short_cmo & short_stoch  # (n_symbols, 4)

        # 장기 TF (4-8: 1h, 4h, 1d, 1w, 1M): CMO > 40 AND 하락 중 AND StochRSI > 70 AND 하락 중
        long_cmo = (cmo[:, t, 4:] > 40) & (cmo[:, t, 4:] <= prev_cmo[:, t, 4:])
        long_stoch = (stochrsi[:, t, 4:] > 70) & (stochrsi[:, t, 4:] <= stochrsi[:, t_prev, 4:])
        long_ok = long_cmo & long_stoch  # (n_symbols, 5)

        # 9개 TF 중 5개 이상 조건 충족
        total_ok = torch.cat([short_ok, long_ok], dim=1)  # (n_symbols, 9)
        five_plus = total_ok.sum(dim=1) >= 5  # (n_symbols,)

        # BB 트리거: S2=5m(idx1), S3=15m(idx2)
        bb_idx = 1 if strategy == "S2" else 2
        bb_cmo = cmo[:, t, bb_idx] > 50
        bb_prev_cmo = cmo[:, t, bb_idx] >= prev_cmo[:, t, bb_idx]
        bb_stoch = stochrsi[:, t, bb_idx] > 75
        bb_prev_stoch = stochrsi[:, t, bb_idx] >= stochrsi[:, t_prev, bb_idx]
        bb_touch = prices[:, t] >= bb_upper[:, t, bb_idx]
        bb_ok = bb_cmo & bb_prev_cmo & bb_stoch & bb_prev_stoch & bb_touch

        return five_plus & bb_ok

    def _select_best_entry(
        self,
        mask: torch.Tensor,
        cmo: torch.Tensor,
        t: int,
    ) -> int:
        """
        동시 진입 후보 중 1개 선택

        기준: CMO 절댓값 합이 가장 큰 심볼 (롱/숏 모두 가장 극단적 = 가장 강한 신호)

        Returns:
            symbol_idx: 배치 내 인덱스
        """
        cmo_strength = cmo[:, t, :].abs().sum(dim=1)
        cmo_strength[~mask] = -1e9
        best_idx = cmo_strength.argmax().item()
        return best_idx

    def _select_entry_tf(
        self,
        symbol_idx: int,
        direction: int,
        cmo: torch.Tensor,
        prev_cmo: torch.Tensor,
        stochrsi: torch.Tensor,
        t: int,
    ) -> int:
        """
        진입근거 TF 선택 (4h > 1h > 30m > 15m > 5m > 1m 우선순위)

        Args:
            direction: 1=long, -1=short

        Returns:
            entry_tf: TF 인덱스 (0=1m, 1=5m, 2=15m, 3=30m, 4=1h, 5=4h, ...)
        """
        t_prev = max(0, t - 1)

        if direction == 1:  # LONG: 4h(5) > 1h(4) > 30m(3) > 15m(2) > 5m(1) > 1m(0)
            for tf_idx in [5, 4]:
                if (
                    cmo[symbol_idx, t, tf_idx].item() < -40
                    and cmo[symbol_idx, t, tf_idx].item() >= prev_cmo[symbol_idx, t, tf_idx].item()
                    and stochrsi[symbol_idx, t, tf_idx].item() < 30
                    and stochrsi[symbol_idx, t, tf_idx].item() >= stochrsi[symbol_idx, t_prev, tf_idx].item()
                ):
                    return tf_idx
            for tf_idx in [3, 2, 1, 0]:
                if (
                    cmo[symbol_idx, t, tf_idx].item() < -50
                    and stochrsi[symbol_idx, t, tf_idx].item() < 25
                ):
                    return tf_idx
            return 3

        else:  # SHORT: 4h(5) > 1h(4) > 30m(3) > 15m(2) > 5m(1) > 1m(0)
            for tf_idx in [5, 4]:
                if (
                    cmo[symbol_idx, t, tf_idx].item() > 40
                    and cmo[symbol_idx, t, tf_idx].item() <= prev_cmo[symbol_idx, t, tf_idx].item()
                    and stochrsi[symbol_idx, t, tf_idx].item() > 70
                    and stochrsi[symbol_idx, t, tf_idx].item() <= stochrsi[symbol_idx, t_prev, tf_idx].item()
                ):
                    return tf_idx
            for tf_idx in [3, 2, 1, 0]:
                if (
                    cmo[symbol_idx, t, tf_idx].item() > 50
                    and stochrsi[symbol_idx, t, tf_idx].item() > 75
                ):
                    return tf_idx
            return 3

    def _process_entry(
        self,
        symbols: List[str],
        prices: torch.Tensor,
        bb_upper: torch.Tensor,
        bb_lower: torch.Tensor,
        cmo: torch.Tensor,
        prev_cmo: torch.Tensor,
        stochrsi: torch.Tensor,
        t: int,
        active_mask: Optional[torch.Tensor] = None,
    ):
        """
        Entry 처리 (멀티 포지션)

        멀티 포지션 규칙:
        - Short 존재 시 Long 신규진입 금지
        - Long 존재 시 Short 신규진입 허용
        - 둘 다 없으면 신호 강도 기반 우선순위로 1개 진입

        Long 전략: S4h (4h BB 트리거)
        Short 전략: S2/S3 (5m/15m BB 트리거, 기존 동일)
        """
        # PERF: blocked_until_bar 텐서 비교
        if self.blocked_until_bar is not None and self.blocked_until_bar.any():
            blocked_mask = self.blocked_until_bar <= t  # True = allowed
        else:
            blocked_mask = None

        # Long: S4h (4h BB 기반) — short 존재 시 금지
        if self.short_position is None:
            long_s4h = self._compute_long_entry_mask(prices, bb_lower, cmo, prev_cmo, stochrsi, t)
        else:
            long_s4h = torch.zeros(cmo.shape[0], dtype=torch.bool, device=cmo.device)

        # Short: S2/S3 (기존과 동일) — long 존재 여부 무관
        if self.short_position is None:
            short_s2 = self._compute_short_entry_mask(prices, bb_upper, cmo, prev_cmo, stochrsi, t, "S2")
            short_s3 = self._compute_short_entry_mask(prices, bb_upper, cmo, prev_cmo, stochrsi, t, "S3")
        else:
            short_s2 = torch.zeros(cmo.shape[0], dtype=torch.bool, device=cmo.device)
            short_s3 = torch.zeros(cmo.shape[0], dtype=torch.bool, device=cmo.device)

        # hourly_top150 필터 + blocked 필터 적용
        if active_mask is not None:
            long_s4h = long_s4h & active_mask
            short_s2 = short_s2 & active_mask
            short_s3 = short_s3 & active_mask
        if blocked_mask is not None:
            long_s4h = long_s4h & blocked_mask
            short_s2 = short_s2 & blocked_mask
            short_s3 = short_s3 & blocked_mask

        has_long_s4h = long_s4h.any()
        has_short_s2 = short_s2.any()
        has_short_s3 = short_s3.any()

        if not (has_long_s4h or has_short_s2 or has_short_s3):
            return

        cmo_strength = cmo[:, t, :].abs().sum(dim=1)

        if has_long_s4h:
            tmp = cmo_strength.clone()
            tmp[~long_s4h] = -1e9
            best_long_score = tmp.max().item()
        else:
            best_long_score = -1.0

        if has_short_s2 or has_short_s3:
            short_all = short_s2 | short_s3
            tmp = cmo_strength.clone()
            tmp[~short_all] = -1e9
            best_short_score = tmp.max().item()
        else:
            best_short_score = -1.0

        # Long이 이미 존재하면 Short만 볼 수 있음
        if self.long_position is not None:
            # Long 존재: Short 신규진입만 처리
            if has_short_s2:
                best_idx = self._select_best_entry(short_s2, cmo, t)
                self._open_position(symbols[best_idx], best_idx, -1, prices, bb_upper, bb_lower, cmo, prev_cmo, stochrsi, t, "S2")
            elif has_short_s3:
                best_idx = self._select_best_entry(short_s3, cmo, t)
                self._open_position(symbols[best_idx], best_idx, -1, prices, bb_upper, bb_lower, cmo, prev_cmo, stochrsi, t, "S3")
            return

        # 둘 다 없는 경우: 레짐 기반 어드밴티지 적용
        if self.market_regime == 1:  # 하락 경고: 숏에 어드밴티지
            adj_long_score = best_long_score
            adj_short_score = best_short_score * self.regime_advantage
        elif self.market_regime == -1:  # 상승 경고: 롱에 어드밴티지
            adj_long_score = best_long_score * self.regime_advantage
            adj_short_score = best_short_score
        else:
            adj_long_score = best_long_score
            adj_short_score = best_short_score

        short_priority = adj_short_score > adj_long_score

        if short_priority:
            if has_short_s2:
                best_idx = self._select_best_entry(short_s2, cmo, t)
                self._open_position(symbols[best_idx], best_idx, -1, prices, bb_upper, bb_lower, cmo, prev_cmo, stochrsi, t, "S2")
            elif has_short_s3:
                best_idx = self._select_best_entry(short_s3, cmo, t)
                self._open_position(symbols[best_idx], best_idx, -1, prices, bb_upper, bb_lower, cmo, prev_cmo, stochrsi, t, "S3")
            elif has_long_s4h:
                best_idx = self._select_best_entry(long_s4h, cmo, t)
                self._open_position(symbols[best_idx], best_idx, 1, prices, bb_upper, bb_lower, cmo, prev_cmo, stochrsi, t, "S4h")
        else:
            if has_long_s4h:
                best_idx = self._select_best_entry(long_s4h, cmo, t)
                self._open_position(symbols[best_idx], best_idx, 1, prices, bb_upper, bb_lower, cmo, prev_cmo, stochrsi, t, "S4h")
            elif has_short_s2:
                best_idx = self._select_best_entry(short_s2, cmo, t)
                self._open_position(symbols[best_idx], best_idx, -1, prices, bb_upper, bb_lower, cmo, prev_cmo, stochrsi, t, "S2")
            elif has_short_s3:
                best_idx = self._select_best_entry(short_s3, cmo, t)
                self._open_position(symbols[best_idx], best_idx, -1, prices, bb_upper, bb_lower, cmo, prev_cmo, stochrsi, t, "S3")

    def _open_position(
        self,
        symbol_name: str,
        symbol_idx: int,
        direction: int,
        prices: torch.Tensor,
        bb_upper: torch.Tensor,
        bb_lower: torch.Tensor,
        cmo: torch.Tensor,
        prev_cmo: torch.Tensor,
        stochrsi: torch.Tensor,
        t: int,
        strategy: str = "S2",
    ):
        """
        포지션 열기 (초기 1 unit)

        Args:
            direction: 1=long, -1=short
            strategy: "S2", "S3" (short), or "S4h" (long)
        """
        market_price = prices[symbol_idx, t].item()

        # 방향별 unit_margin / unit_notional / leverage 선택
        if direction == 1:
            unit_margin = self.long_unit_margin
            leverage = self.long_leverage
        else:
            unit_margin = self.short_unit_margin
            leverage = self.short_leverage

        # BB 95% 도달 가능 여부 체크 (30m BB 고정)
        bb_up_30m = bb_upper[symbol_idx, t, 3].item()
        bb_lo_30m = bb_lower[symbol_idx, t, 3].item()
        bb_width_30m = bb_up_30m - bb_lo_30m if bb_up_30m > bb_lo_30m else 0
        if bb_width_30m > 0:
            if direction == 1:
                bb_target = bb_lo_30m + bb_width_30m * 0.95
                profit_room = (bb_target - market_price) / market_price
            else:
                bb_target = bb_lo_30m + bb_width_30m * 0.05
                profit_room = (market_price - bb_target) / market_price
            if profit_room < 0.002:
                logger.info(f"[ENTRY_FILTER] {symbol_name} dir={direction} profit_room={profit_room:.4f} REJECTED")
                return

        # 복리: equity 기반 유닛 재계산
        self.update_unit_sizing()
        if direction == 1:
            unit_margin = self.long_unit_margin
            unit_notional = self.long_unit_notional
        else:
            unit_margin = self.short_unit_margin
            unit_notional = self.short_unit_notional

        initial_margin = unit_margin * 1  # 1 unit
        initial_notional = unit_notional * 1

        # 슬리피지 반영 실효 진입가
        entry_price = self._apply_slippage_to_price(
            market_price, direction, initial_notional, is_exit=False
        )

        # 진입근거 TF 선택
        entry_tf = self._select_entry_tf(symbol_idx, direction, cmo, prev_cmo, stochrsi, t)

        # 진입 수수료 즉시 차감
        open_entry_fee = initial_notional * self.taker_fee
        self.equity -= open_entry_fee

        new_pos = Position(
            symbol_idx=symbol_idx,
            symbol_name=symbol_name,
            direction=direction,
            entry_price=entry_price,
            entry_time=t,
            entry_tf=entry_tf,
            strategy=strategy,
            dca_level=0,
            total_margin=initial_margin,
            total_notional=initial_notional,
            entry_fee_paid=open_entry_fee,
            entry_equity=self.equity,
        )

        if direction == 1:
            self.long_position = new_pos
        else:
            self.short_position = new_pos

        regime_str = {0: "NEUTRAL", 1: "BEAR(S)", -1: "SURGE(L)"}.get(self.market_regime, "")
        entry_dt = (self.sim_base + timedelta(minutes=t)).strftime("%Y-%m-%d %H:%M") if hasattr(self, 'sim_base') else str(t)
        logger.info(
            f"[TRADE] {entry_dt} OPEN {symbol_name} {'LONG' if direction == 1 else 'SHORT'} {strategy} "
            f"TF={entry_tf} regime={regime_str} scale={self.position_scale:.2f} "
            f"price=${market_price:.4f}->${entry_price:.4f} "
            f"margin=${initial_margin:.2f} notional=${initial_notional:.2f} equity=${self.equity:.2f}"
        )

    def _process_dca(
        self,
        prices: torch.Tensor,
        bb_upper: torch.Tensor,
        bb_lower: torch.Tensor,
        cmo: torch.Tensor,
        prev_cmo: torch.Tensor,
        stochrsi: torch.Tensor,
        t: int,
        pos: "Position",
        top50_mask: Optional[torch.Tensor] = None,
    ):
        """DCA 처리 (지정된 포지션에 적용)

        Args:
            pos: 대상 포지션 (long_position or short_position)
            top50_mask: (n_symbols, n_bars) bool — DCA5+ 시 top50 심볼만 허용
        """
        if pos is None:
            return

        # 진입 캔들에서는 DCA 금지
        if t == pos.entry_time:
            return

        # 1. 진입근거 TF 기준 모멘텀 유지 확인
        t_prev = max(0, t - 1)
        curr_cmo = cmo[pos.symbol_idx, t, pos.entry_tf].item()
        curr_prev_cmo = prev_cmo[pos.symbol_idx, t, pos.entry_tf].item()
        curr_stochrsi = stochrsi[pos.symbol_idx, t, pos.entry_tf].item()
        prev_stochrsi = stochrsi[pos.symbol_idx, t_prev, pos.entry_tf].item()

        if pos.direction == 1:  # LONG: 과매도 유지 확인
            if pos.entry_tf < 4:  # 단기 TF
                momentum_ok = curr_cmo < -50 and curr_stochrsi < 25
            else:  # 장기 TF: StochRSI 반등 중이어야 DCA 허용
                momentum_ok = (
                    curr_cmo < -40
                    and curr_cmo >= curr_prev_cmo
                    and curr_stochrsi < 30
                    and curr_stochrsi >= prev_stochrsi
                )
        else:  # SHORT: 과매수 유지 확인
            if pos.entry_tf < 4:  # 단기 TF
                momentum_ok = curr_cmo > 50 and curr_stochrsi > 75
            else:  # 장기 TF
                momentum_ok = (
                    curr_cmo > 40
                    and curr_cmo <= curr_prev_cmo
                    and curr_stochrsi > 70
                    and curr_stochrsi <= prev_stochrsi
                )

        # 2. DCA 레벨 체크
        can_dca = pos.dca_level < 6

        # 3. DCA5+ 유동성 필터: top50 심볼만 허용
        if can_dca and pos.dca_level + 1 == 6 and top50_mask is not None:
            in_top50 = top50_mask[pos.symbol_idx, t].item()
            if not in_top50:
                logger.debug(
                    f"[DCA_TOP50_FILTER] t={t} {pos.symbol_name} DCA{pos.dca_level+1} 거부: "
                    f"not in top50"
                )
                return

        # 4. BB 트리거
        if can_dca and momentum_ok:
            next_dca_tf = self.dca_tf_triggers[pos.dca_level + 1]
            bb_trigger = False

            if pos.direction == 1:
                bb_trigger = prices[pos.symbol_idx, t] <= bb_lower[pos.symbol_idx, t, next_dca_tf]
            else:
                bb_trigger = prices[pos.symbol_idx, t] >= bb_upper[pos.symbol_idx, t, next_dca_tf]

            if bb_trigger:
                dca_price = prices[pos.symbol_idx, t].item()

                # DCA 직전 LP 시뮬레이션
                next_level = pos.dca_level + 1
                sim_dca_size = self.dca_units[next_level]

                # 방향별 unit_margin/leverage
                if pos.direction == 1:
                    sim_unit_margin = self.equity * self.long_utilization / 64
                    sim_leverage = self.long_leverage
                else:
                    sim_unit_margin = self.equity * self.short_utilization / 64
                    sim_leverage = self.short_leverage

                sim_new_notional = sim_unit_margin * sim_dca_size * sim_leverage
                sim_total_notional = pos.total_notional + sim_new_notional
                current_unrealized = pos.direction * (dca_price - pos.entry_price) / pos.entry_price * pos.total_notional
                sim_equity = self.equity + current_unrealized - sim_new_notional * self.taker_fee

                n = sim_total_notional
                if n <= 50_000:
                    sim_mmr = 0.015
                elif n <= 250_000:
                    sim_mmr = 0.020
                elif n <= 1_000_000:
                    sim_mmr = 0.025
                else:
                    sim_mmr = 0.050

                sim_entry = (pos.entry_price * pos.total_notional + dca_price * sim_new_notional) / sim_total_notional

                dca_refused = False
                if pos.direction == 1:  # Long LP 시뮬
                    if sim_equity < sim_total_notional:
                        sim_lp = sim_entry * (sim_total_notional - sim_equity) / (sim_total_notional * (1.0 - sim_mmr))
                        if dca_price <= sim_lp:
                            logger.info(
                                f"[DCA_REFUSED] t={t} {pos.symbol_name} DCA{next_level} 거부: "
                                f"post-DCA LP=${sim_lp:.4f} >= current=${dca_price:.4f}"
                            )
                            dca_refused = True
                else:  # Short LP 시뮬
                    sim_lp = sim_entry * (sim_total_notional + sim_equity) / (sim_total_notional * (1.0 + sim_mmr))
                    if dca_price >= sim_lp:
                        logger.info(
                            f"[DCA_REFUSED] t={t} {pos.symbol_name} DCA{next_level} 거부: "
                            f"post-DCA LP=${sim_lp:.4f} <= current=${dca_price:.4f}"
                        )
                        dca_refused = True

                if not dca_refused:
                    self._add_dca(t, dca_price, pos)

    def _add_dca(self, t: int, dca_price: float, pos: "Position"):
        """
        DCA 레벨 증가

        Args:
            t: 현재 시간 인덱스
            dca_price: DCA 진입 가격
            pos: 대상 포지션
        """
        if pos is None:
            return

        pos.dca_level += 1
        dca_size = self.dca_units[pos.dca_level]

        # DCA 직전 최신 equity 반영
        self.update_unit_sizing()

        # 방향별 unit_margin/notional
        if pos.direction == 1:
            new_margin = self.long_unit_margin * dca_size
            new_notional = self.long_unit_notional * dca_size
        else:
            new_margin = self.short_unit_margin * dca_size
            new_notional = self.short_unit_notional * dca_size

        # 슬리피지 반영
        effective_dca_price = self._apply_slippage_to_price(
            dca_price, pos.direction, pos.total_notional + new_notional, is_exit=False
        )

        # 가중평균 진입가 업데이트
        old_notional = pos.total_notional
        pos.entry_price = (
            pos.entry_price * old_notional + effective_dca_price * new_notional
        ) / (old_notional + new_notional)

        pos.total_margin += new_margin
        pos.total_notional += new_notional

        # DCA 진입 수수료 즉시 차감
        dca_entry_fee = new_notional * self.taker_fee
        self.equity -= dca_entry_fee
        pos.entry_fee_paid += dca_entry_fee

        logger.debug(
            f"[{t}] DCA{pos.dca_level} {pos.symbol_name} ({'L' if pos.direction==1 else 'S'}): "
            f"+{dca_size} units @ ${dca_price:.4f}(slip->{effective_dca_price:.4f}) "
            f"avg_entry=${pos.entry_price:.4f} "
            f"total_margin=${pos.total_margin:.2f}, total_notional=${pos.total_notional:.2f}, "
            f"dca_fee=${dca_entry_fee:.2f}"
        )

    def _process_exit(
        self,
        prices: torch.Tensor,
        prices_high: torch.Tensor,
        prices_low: torch.Tensor,
        bb_upper: torch.Tensor,
        bb_lower: torch.Tensor,
        cmo: torch.Tensor,
        prev_cmo: torch.Tensor,
        stochrsi: torch.Tensor,
        t: int,
        pos: "Position",
    ):
        """
        Exit 처리 (지정된 포지션)

        Long 전용 변경사항:
        - BB_TP: 4h BB upper (bb_idx=5) 사용
        - Momentum SL: 4h TF 고정 (tf_idx=5)

        Short은 기존 동일 (30m BB_TP, entry_tf 기반 모멘텀 SL)

        5가지 청산 조건 (우선순위 순):
        1. BB_TP: Long=4h BB upper, Short=30m BB 95%
        2. EMERGENCY_SL: 진입근거 TF 급속 모멘텀 악화
        3. 1D_ABS_SL: 1D CMO/StochRSI 급락
        4. ENTRY_TF_SL: 진입근거 TF 모멘텀 조건 소멸
        """
        if pos is None:
            return

        # 진입 캔들에서는 Exit 금지
        if t == pos.entry_time:
            return

        t_prev = max(0, t - 1)
        sym = pos.symbol_idx

        # Long: 모멘텀 SL에 4h TF 고정 사용
        # Short: entry_tf 기반 (기존과 동일)
        if pos.direction == 1:
            sl_tf_idx = 5  # 4h 고정
        else:
            sl_tf_idx = pos.entry_tf

        curr_cmo_sl = cmo[sym, t, sl_tf_idx].item()
        curr_prev_cmo_sl = prev_cmo[sym, t, sl_tf_idx].item()
        curr_stochrsi_sl = stochrsi[sym, t, sl_tf_idx].item()
        prev_stochrsi_sl = stochrsi[sym, t_prev, sl_tf_idx].item()

        # BB_TP: Long=4h BB upper, Short=30m BB 95%
        if pos.direction == 1:
            # Long: 4h BB upper (bb_idx=5)
            bb_tp_tf = 5
            bb_up = bb_upper[sym, t, bb_tp_tf].item()
            bb_lo = bb_lower[sym, t, bb_tp_tf].item()
            bb_exit = prices_high[sym, t].item() >= bb_up
        else:
            # Short: 30m BB 95% (bb_idx=3, 기존 동일)
            bb_tp_tf = 3
            bb_up = bb_upper[sym, t, bb_tp_tf].item()
            bb_lo = bb_lower[sym, t, bb_tp_tf].item()
            bb_width = bb_up - bb_lo if bb_up > bb_lo else 1e-10
            bb_05_level = bb_lo + bb_width * 0.05
            bb_exit = prices_low[sym, t].item() <= bb_05_level

        # Exit: Entry TF SL — 모멘텀 조건 소멸
        if pos.direction == 1:
            # Long: 4h 고정 (sl_tf_idx=5)
            if sl_tf_idx < 4:
                sl_tf = curr_cmo_sl >= -50 or curr_stochrsi_sl >= 25
            else:
                sl_tf = (
                    curr_cmo_sl >= -40
                    or curr_cmo_sl < curr_prev_cmo_sl
                    or curr_stochrsi_sl >= 30
                    or curr_stochrsi_sl < prev_stochrsi_sl
                )
        else:
            # Short: entry_tf 기반 (기존 동일)
            if sl_tf_idx < 4:
                sl_tf = curr_cmo_sl <= 50 or curr_stochrsi_sl <= 75
            else:
                sl_tf = (
                    curr_cmo_sl <= 40
                    or curr_cmo_sl > curr_prev_cmo_sl
                    or curr_stochrsi_sl <= 70
                    or curr_stochrsi_sl > prev_stochrsi_sl
                )

        # Exit: 1D 절대 손절
        curr_1d_cmo = cmo[sym, t, 6].item()
        prev_1d_cmo = prev_cmo[sym, t, 6].item()
        curr_1d_stochrsi = stochrsi[sym, t, 6].item()
        prev_1d_stochrsi = stochrsi[sym, t_prev, 6].item()

        if pos.direction == 1:
            sl_1d = (prev_1d_cmo - curr_1d_cmo) > 15 or (prev_1d_stochrsi - curr_1d_stochrsi) > 30
        else:
            sl_1d = (curr_1d_cmo - prev_1d_cmo) > 15 or (curr_1d_stochrsi - prev_1d_stochrsi) > 30

        # Exit: 긴급 손절
        if pos.direction == 1:
            cmo_slope = curr_prev_cmo_sl - curr_cmo_sl
            stochrsi_slope = prev_stochrsi_sl - curr_stochrsi_sl
        else:
            cmo_slope = curr_cmo_sl - curr_prev_cmo_sl
            stochrsi_slope = curr_stochrsi_sl - prev_stochrsi_sl
        sl_emergency = cmo_slope > 35 or stochrsi_slope > 50

        # 우선순위 순서로 청산
        direction = pos.direction
        if bb_exit:
            self._close_position(prices, t, "BB_TP", direction=direction)
        elif sl_emergency:
            self._close_position(prices, t, "EMERGENCY_SL", direction=direction)
        elif sl_1d:
            self._close_position(prices, t, "1D_ABS_SL", direction=direction)
        elif sl_tf:
            unrealized = pos.direction * (prices[sym, t].item() - pos.entry_price) / pos.entry_price * pos.total_notional
            if unrealized > 0:
                self._close_position(prices, t, "MOMENTUM_TP", direction=direction)
            else:
                self._close_position(prices, t, "MOMENTUM_SL", direction=direction)

    def _close_position(
        self,
        prices: torch.Tensor,
        t: int,
        exit_reason: str = "FORCE_CLOSE",
        direction: int = 0,
    ):
        """
        포지션 청산 및 실현손익 계산

        Args:
            prices: (n_symbols, n_bars)
            t: 청산 시간 인덱스
            exit_reason: 청산 사유
            direction: 1=long_position 청산, -1=short_position 청산, 0=둘 다 None 체크
        """
        if direction == 1:
            pos = self.long_position
        elif direction == -1:
            pos = self.short_position
        else:
            # 레거시 호환: direction 미지정 시 존재하는 포지션 청산
            pos = self.long_position or self.short_position

        if pos is None:
            return

        market_price = prices[pos.symbol_idx, t].item()

        # 슬리피지 반영
        is_emergency = exit_reason == "ENTRY_STOP"
        exit_price = self._apply_slippage_to_price(
            market_price, pos.direction, pos.total_notional,
            is_exit=True, is_emergency=is_emergency
        )

        # PnL 계산
        price_change_pct = (exit_price - pos.entry_price) / pos.entry_price
        unrealized_pnl = price_change_pct * pos.total_notional * pos.direction

        # 수수료
        exit_fee = pos.total_notional * self.taker_fee
        total_fee = pos.entry_fee_paid + exit_fee

        realized_pnl = unrealized_pnl - exit_fee

        # Equity 업데이트
        self.equity += realized_pnl

        # MAX_WALLET $1M 캡
        if self.equity > 1_000_000:
            excess = self.equity - 1_000_000
            self.total_withdrawn += excess
            self.equity = 1_000_000
            logger.info(
                f"[MAX_WALLET] t={t} Capped: excess=${excess:.2f} withdrawn, equity=${self.equity:.2f}"
            )

        # 파산 체크
        if self.equity <= 0:
            logger.error(
                f"[BANKRUPTCY] t={t} equity=${self.equity:.2f} <= 0. "
                f"Simulation halted. total_notional=${pos.total_notional:.2f}"
            )
            self.equity = 0.0
            self.update_unit_sizing()
            slip_rate = self._calc_slippage(pos.total_notional, is_emergency=is_emergency)
            duration_min = t - pos.entry_time
            entry_dt = self._sim_base_dt + timedelta(minutes=pos.entry_time)
            exit_dt = self._sim_base_dt + timedelta(minutes=t)
            bankrupt_trade = {
                **pos.to_dict(),
                "exit_time_idx": t,
                "exit_price": exit_price,
                "exit_reason": "BANKRUPTCY",
                "realized_pnl": realized_pnl,
                "fee": total_fee,
                "slippage_cost": slip_rate * pos.total_notional * 2,
                "equity_after": 0.0,
                "duration_min": duration_min,
                "entry_datetime": entry_dt.strftime("%Y-%m-%d %H:%M"),
                "exit_datetime": exit_dt.strftime("%Y-%m-%d %H:%M"),
                "position_scale": self.position_scale,
                "market_regime": self.market_regime,
                "entry_cmo": 0.0, "entry_stochrsi": 0.0,
                "exit_cmo": 0.0, "exit_stochrsi": 0.0,
                "exit_bb_upper": 0.0, "exit_bb_lower": 0.0,
            }
            self.trades.append(bankrupt_trade)
            self.last_close_bar = t
            if pos.direction == 1:
                self.long_position = None
            else:
                self.short_position = None
            return

        # 복리: 유닛 재계산
        self.update_unit_sizing()

        # 슬리피지 비용 기록
        entry_slip_rate = self._calc_slippage(pos.total_notional)
        exit_slip_rate = self._calc_slippage(pos.total_notional, is_emergency=is_emergency)
        slippage_cost = (entry_slip_rate + exit_slip_rate) * pos.total_notional

        # 거래 내역 저장
        duration_min = t - pos.entry_time
        entry_dt = self._sim_base_dt + timedelta(minutes=pos.entry_time)
        exit_dt = self._sim_base_dt + timedelta(minutes=t)
        sym_i = pos.symbol_idx
        etf = pos.entry_tf
        dt = self._sim_data_tensor
        n_bars = dt.shape[1]
        cmo_col = 6 + etf * 8 + 1
        str_col = 6 + etf * 8 + 3
        bbu_col = 6 + etf * 8 + 4
        bbl_col = 6 + etf * 8 + 5
        entry_cmo = dt[sym_i, pos.entry_time, cmo_col].item() if pos.entry_time < n_bars else 0.0
        entry_stochrsi = dt[sym_i, pos.entry_time, str_col].item() if pos.entry_time < n_bars else 0.0
        exit_cmo = dt[sym_i, t, cmo_col].item() if t < n_bars else 0.0
        exit_stochrsi = dt[sym_i, t, str_col].item() if t < n_bars else 0.0
        exit_bb_upper = dt[sym_i, t, bbu_col].item() if t < n_bars else 0.0
        exit_bb_lower = dt[sym_i, t, bbl_col].item() if t < n_bars else 0.0
        trade = {
            **pos.to_dict(),
            "exit_time_idx": t,
            "exit_price": exit_price,
            "exit_reason": exit_reason,
            "realized_pnl": realized_pnl,
            "fee": total_fee,
            "slippage_cost": slippage_cost,
            "equity_after": self.equity,
            "duration_min": duration_min,
            "entry_datetime": entry_dt.strftime("%Y-%m-%d %H:%M"),
            "exit_datetime": exit_dt.strftime("%Y-%m-%d %H:%M"),
            "position_scale": self.position_scale,
            "market_regime": self.market_regime,
            "entry_cmo": round(entry_cmo, 2),
            "entry_stochrsi": round(entry_stochrsi, 2),
            "exit_cmo": round(exit_cmo, 2),
            "exit_stochrsi": round(exit_stochrsi, 2),
            "exit_bb_upper": round(exit_bb_upper, 6),
            "exit_bb_lower": round(exit_bb_lower, 6),
        }
        self.trades.append(trade)

        exit_dt_str = (self._sim_base_dt + timedelta(minutes=t)).strftime("%Y-%m-%d %H:%M")
        logger.info(
            f"[TRADE] {exit_dt_str} CLOSE {pos.symbol_name} {'LONG' if pos.direction == 1 else 'SHORT'} "
            f"{exit_reason} DCA{pos.dca_level} dur={duration_min}min "
            f"PnL=${realized_pnl:+,.0f} fee=${total_fee:.0f} equity=${self.equity:,.0f} "
            f"entry=${pos.entry_price:.4f} exit=${exit_price:.4f} regime={self.market_regime} scale={self.position_scale:.2f}"
        )

        # 연속 손실 추적
        if realized_pnl < 0:
            self.consecutive_losses += 1
            if self.consecutive_losses >= 2:
                self.cooldown_until = t + 360
                logger.info(
                    f"[COOLDOWN] {self.consecutive_losses} consecutive losses. "
                    f"Trading halted until bar {self.cooldown_until} (+6h)"
                )
                self.consecutive_losses = 0
        else:
            self.consecutive_losses = 0

        # 동적 포지션 축소
        if realized_pnl < 0 and self.equity > 0:
            loss_ratio = min(abs(realized_pnl) / self.equity, 0.30)
            base_reduction = min(max(0.20 + (loss_ratio / 0.30) * 0.50, 0.20), 0.70)
            self.loss_count_scale += 1
            effective_reduction = min(0.90, base_reduction * self.loss_count_scale)
            self.position_scale = max(0.10, self.position_scale * (1.0 - effective_reduction))
            logger.debug(
                f"[SCALE_DOWN] loss_ratio={loss_ratio:.4f}, base_red={base_reduction:.2f}, "
                f"eff_red={effective_reduction:.2f}, scale={self.position_scale:.3f}"
            )
        elif realized_pnl > 0:
            self.position_scale = 1.0
            self.loss_count_scale = 0

        # same-bar 재진입 방지
        self.last_close_bar = t

        # 포지션 초기화 (방향별)
        if pos.direction == 1:
            self.long_position = None
        else:
            self.short_position = None

    def _check_liquidation(
        self,
        prices: torch.Tensor,
        prices_low: torch.Tensor,
        prices_high: torch.Tensor,
        t: int,
    ):
        """
        강제청산 체크 (P1) — 포지션별 독립 처리

        Cross Margin USDT-M Perpetual:
            Long  LP = entry * (N - E) / (N * (1 - MMR))
            Short LP = entry * (N + E) / (N * (1 + MMR))
            where E = MTM equity, N = total_notional

        방향별 leverage로 개별 계산.
        """
        for pos in [self.long_position, self.short_position]:
            if pos is None:
                continue

            n = pos.total_notional
            if n <= 50_000:
                mmr = 0.015
            elif n <= 250_000:
                mmr = 0.020
            elif n <= 1_000_000:
                mmr = 0.025
            else:
                mmr = 0.050

            # Cross Margin: 양쪽 포지션의 미실현 손익 합산이 담보
            total_unrealized = 0.0
            for p in [self.long_position, self.short_position]:
                if p is not None:
                    p_price = prices[p.symbol_idx, t].item()
                    total_unrealized += p.direction * (p_price - p.entry_price) / p.entry_price * p.total_notional
            e = self.equity + total_unrealized

            if pos.direction == 1:  # LONG
                if e >= n:
                    continue
                lp = pos.entry_price * (n - e) / (n * (1.0 - mmr))
                if lp <= 0:
                    continue
                liquidated = prices_low[pos.symbol_idx, t].item() <= lp
            else:  # SHORT
                lp = pos.entry_price * (n + e) / (n * (1.0 + mmr))
                liquidated = prices_high[pos.symbol_idx, t].item() >= lp

            if not liquidated:
                continue

            # 청산 처리
            price_change_pct = (lp - pos.entry_price) / pos.entry_price
            realized_pnl = price_change_pct * pos.total_notional * pos.direction
            liq_fee = pos.total_notional * self.taker_fee
            total_liq_fee = pos.entry_fee_paid + liq_fee
            net_pnl = realized_pnl - liq_fee

            logger.info(
                f"[LIQUIDATED] t={t} {pos.symbol_name} "
                f"{'LONG' if pos.direction == 1 else 'SHORT'} "
                f"entry=${pos.entry_price:.4f} LP=${lp:.4f} "
                f"pnl=${net_pnl:.2f} equity_before=${self.equity:.2f}"
            )

            self.equity += net_pnl

            if self.equity <= 0:
                logger.error(
                    f"[BANKRUPTCY] t={t} equity=${self.equity:.2f} <= 0 after liquidation. "
                    f"total_notional=${pos.total_notional:.2f}"
                )
                self.equity = 0.0

            self.update_unit_sizing()

            # 거래 내역 저장
            duration_min = t - pos.entry_time
            entry_dt = self._sim_base_dt + timedelta(minutes=pos.entry_time)
            exit_dt = self._sim_base_dt + timedelta(minutes=t)
            trade = {
                **pos.to_dict(),
                "exit_time_idx": t,
                "exit_price": lp,
                "exit_reason": "LIQUIDATION",
                "realized_pnl": net_pnl,
                "fee": total_liq_fee,
                "slippage_cost": 0.0,
                "equity_after": self.equity,
                "duration_min": duration_min,
                "entry_datetime": entry_dt.strftime("%Y-%m-%d %H:%M"),
                "exit_datetime": exit_dt.strftime("%Y-%m-%d %H:%M"),
                "position_scale": self.position_scale,
                "market_regime": self.market_regime,
                "entry_cmo": 0.0, "entry_stochrsi": 0.0,
                "exit_cmo": 0.0, "exit_stochrsi": 0.0,
                "exit_bb_upper": 0.0, "exit_bb_lower": 0.0,
            }
            self.trades.append(trade)

            # 연속 손실 쿨다운
            self.consecutive_losses += 1
            if self.consecutive_losses >= 2:
                self.cooldown_until = t + 360
                logger.info(
                    f"[COOLDOWN] {self.consecutive_losses} consecutive losses. "
                    f"Trading halted until bar {self.cooldown_until} (+6h)"
                )
                self.consecutive_losses = 0

            # 동적 포지션 축소
            if self.equity > 0:
                loss_ratio = min(abs(net_pnl) / self.equity, 0.30)
                base_reduction = min(max(0.20 + (loss_ratio / 0.30) * 0.50, 0.20), 0.70)
                self.loss_count_scale += 1
                effective_reduction = min(0.90, base_reduction * self.loss_count_scale)
                self.position_scale = max(0.10, self.position_scale * (1.0 - effective_reduction))
                logger.debug(
                    f"[SCALE_DOWN/LIQ] loss_ratio={loss_ratio:.4f}, eff_red={effective_reduction:.2f}, "
                    f"scale={self.position_scale:.3f}"
                )

            # same-bar 재진입 방지
            self.last_close_bar = t

            # 포지션 해제
            if pos.direction == 1:
                self.long_position = None
            else:
                self.short_position = None

    def _apply_funding_cost(
        self,
        funding_rates: torch.Tensor,
        t: int,
    ):
        """
        펀딩비 차감 (8시간마다) — 보유 중인 모든 포지션에 적용

        Args:
            funding_rates: (n_symbols, n_bars)
            t: 현재 시간 인덱스
        """
        for pos in [self.long_position, self.short_position]:
            if pos is None:
                continue

            rate = funding_rates[pos.symbol_idx, t].item()

            # 펀딩비 = 노셔널 × 펀딩비율
            # 롱: rate > 0이면 비용 (-), 숏: rate > 0이면 수익 (+)
            funding_cost = pos.total_notional * rate * pos.direction

            self.equity -= funding_cost
            self.funding_cost_total += abs(funding_cost)

            logger.debug(
                f"[{t}] FUNDING {pos.symbol_name} ({'L' if pos.direction==1 else 'S'}): "
                f"rate={rate:.6f}, cost=${funding_cost:.2f}, equity=${self.equity:.2f}"
            )

        # 펀딩비 후 equity 변동 → 유닛 재계산
        self.update_unit_sizing()

    def _calculate_performance(self) -> Dict:
        """
        성과 지표 계산

        Returns:
            dict: 성과 지표
        """
        if len(self.trades) == 0:
            logger.warning(
                "No trades executed during simulation!\n"
                "Possible reasons:\n"
                "1. Entry conditions too strict (try relaxing CMO threshold)\n"
                "2. No BB touches during period (check market regime)\n"
                "3. Insufficient data or symbols\n"
                "Recommendation: Review entry conditions and market data"
            )
            return {
                "total_profit_usd": 0.0,
                "daily_profit_usd": 0.0,
                "return_pct": 0.0,
                "total_trades": 0,
                "win_rate": 0.0,
                "max_drawdown_usd": 0.0,
                "sharpe_ratio": 0.0,
                "profit_factor": 0.0,
                "fee_cost_total": 0.0,
                "funding_cost_total": 0.0,
                "by_direction": {
                    "long": {"profit": 0.0, "trades": 0, "win_rate": 0.0},
                    "short": {"profit": 0.0, "trades": 0, "win_rate": 0.0},
                },
            }

        # 총 손익
        total_profit = (self.equity - self.initial_wallet) + self.total_withdrawn
        return_pct = (total_profit / self.initial_wallet) * 100

        # 승률
        wins = sum(1 for t in self.trades if t["realized_pnl"] > 0)
        win_rate = wins / len(self.trades) if len(self.trades) > 0 else 0.0

        # 최대 드로다운
        equity_array = np.array(self.equity_curve)
        running_max = np.maximum.accumulate(equity_array)
        drawdown = running_max - equity_array
        max_drawdown = drawdown.max()

        # 방향별 성과
        long_trades = [t for t in self.trades if t["direction"] == "long"]
        short_trades = [t for t in self.trades if t["direction"] == "short"]

        long_profit = sum(t["realized_pnl"] for t in long_trades)
        short_profit = sum(t["realized_pnl"] for t in short_trades)

        long_wins = sum(1 for t in long_trades if t["realized_pnl"] > 0)
        short_wins = sum(1 for t in short_trades if t["realized_pnl"] > 0)

        # 수수료 + 슬리피지 합계
        total_fee = sum(t["fee"] for t in self.trades)
        total_slippage = sum(t.get("slippage_cost", 0.0) for t in self.trades)

        # daily_profit_usd
        total_days = (self.end_date - self.start_date).days
        daily_profit_usd = total_profit / total_days if total_days > 0 else 0.0

        # sharpe_ratio
        if len(self.equity_curve) > 1:
            equity_array = np.array(self.equity_curve)
            daily_step = max(1, int(round(1440 / 100)))
            daily_equity = equity_array[::daily_step]
            if len(daily_equity) > 1:
                daily_returns = np.diff(daily_equity) / daily_equity[:-1]
                if len(daily_returns) > 1 and np.std(daily_returns) > 0:
                    sharpe_ratio = (
                        np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(365)
                    )
                else:
                    sharpe_ratio = 0.0
            else:
                sharpe_ratio = 0.0
        else:
            sharpe_ratio = 0.0

        # profit_factor
        wins_list = [t["realized_pnl"] for t in self.trades if t["realized_pnl"] > 0]
        losses_list = [abs(t["realized_pnl"]) for t in self.trades if t["realized_pnl"] < 0]
        total_loss = sum(losses_list)
        if total_loss > 0:
            profit_factor = sum(wins_list) / total_loss
        elif len(wins_list) > 0:
            profit_factor = float("inf")
        else:
            profit_factor = 0.0

        result = {
            "total_profit_usd": total_profit,
            "daily_profit_usd": daily_profit_usd,
            "return_pct": return_pct,
            "win_rate": win_rate,
            "max_drawdown_usd": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "profit_factor": profit_factor,
            "total_trades": len(self.trades),
            "by_direction": {
                "long": {
                    "profit": long_profit,
                    "trades": len(long_trades),
                    "win_rate": long_wins / len(long_trades) if len(long_trades) > 0 else 0.0,
                },
                "short": {
                    "profit": short_profit,
                    "trades": len(short_trades),
                    "win_rate": short_wins / len(short_trades) if len(short_trades) > 0 else 0.0,
                },
            },
            "fee_cost_total": total_fee,
            "slippage_cost_total": total_slippage,
            "funding_cost_total": self.funding_cost_total,
            "total_withdrawn": self.total_withdrawn,
            "withdrawal_days": len(self.daily_withdrawn_log),
        }

        return result

    def get_trades(self) -> List[Dict]:
        """거래 내역 반환"""
        return self.trades


# ======================================================================
# BACKTEST RUNNER
# ======================================================================

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
# 데이터 경로 자동 감지
_candidates = [
    Path("/kaggle/input/parquet"),
    Path("/kaggle/input/datasets/norexinc/parquet"),
    Path("/kaggle/input/norexinc/parquet"),
]
DATA_DIR = None
for p in _candidates:
    if p.exists():
        DATA_DIR = p
        break
if DATA_DIR is None:
    # 실제 경로 탐색
    import glob
    found = glob.glob("/kaggle/input/**/symbol_map.csv", recursive=True)
    if found:
        DATA_DIR = Path(found[0]).parent
    else:
        raise FileNotFoundError(f"Data not found in: {_candidates}. /kaggle/input contents: {os.listdir('/kaggle/input') if os.path.exists('/kaggle/input') else 'N/A'}")
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
        syms = list(row["symbols"]) if hasattr(row["symbols"], '__iter__') and not isinstance(row["symbols"], str) else []
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
    # GPUPortfolioSimulator is defined above

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
