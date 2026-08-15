#!/usr/bin/env python3
"""
Crypto Rotation V3 Research Harness for AdaptiveConfluenceLab.

V3 corrections and research changes
-----------------------------------
- No trading before the common evaluation start. Initial evaluation equity is
  exactly $100,000; indicator history before that date is used only as warmup.
- Relative liquidity ranking (top 15 by trailing 30-day median dollar volume).
- Stablecoin/cash-like bases excluded from the risk-asset ranking.
- BTC regime-controlled exposure and BTC-volatility-scaled exposure.
- Compact 288-configuration grid focused on diversified top-3/top-4 portfolios.
- Chronological train/validation selection with robustness scoring.
- The previously viewed V2 holdout is retained only as a historical diagnostic;
  it is NOT described as unseen validation for V3.
- 1x/2x/3x transaction-cost stress.
- Long/cash only. RESEARCH ONLY. Never places orders.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import time
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests

PAPER_BASE = "https://paper-api.alpaca.markets"
DATA_BASE = "https://data.alpaca.markets"
DEFAULT_START = "2021-01-01T00:00:00Z"
DEFAULT_END = None

EXCLUDED_BASES = {
    "USD", "USDC", "USDT", "DAI", "PYUSD", "TUSD", "FDUSD", "USDP",
    "USDG", "GUSD", "USDS", "USDE", "EURC", "USD0", "USDY",
}

INITIAL_EQUITY = 100_000.0
BASE_COST_BPS = 35.0
LIQUIDITY_LOOKBACK = 30
DEFAULT_MAX_LIQUID_ASSETS = 15
DEFAULT_MIN_MEDIAN_DOLLAR_VOLUME = 0.0

MOMENTUM_GRID = (42, 63, 90)
SMA_GRID = (100, 150, 200)
REBALANCE_GRID = (5, 7)
TOP_K_GRID = (3, 4)
WEIGHTING_GRID = ("equal", "rank")
REGIME_PROFILE_GRID = ("defensive", "balanced")
VOL_TARGET_GRID = (45.0, 65.0)

REGIME_PROFILES = {
    # Sideways was V2's major failure mode, so risk is deliberately reduced.
    "defensive": {"up": 1.00, "sideways": 0.20, "down": 0.00},
    "balanced": {"up": 1.00, "sideways": 0.40, "down": 0.10},
}

BTC_REGIME_SMA_DAYS = 100
BTC_REGIME_MOM_DAYS = 30
BTC_REGIME_MOM_THRESHOLD = 0.05
BTC_VOL_LOOKBACK = 20
MAX_SMA = max(max(SMA_GRID), BTC_REGIME_SMA_DAYS)
MAX_MOM = max(max(MOMENTUM_GRID), BTC_REGIME_MOM_DAYS)
WARMUP_DAYS = max(MAX_SMA, MAX_MOM + 1, LIQUIDITY_LOOKBACK + 1, BTC_VOL_LOOKBACK + 2)
PACK_DIRNAME = "CryptoRotationV3_research_pack"


@dataclass(frozen=True)
class CryptoConfig:
    momentum_days: int
    sma_days: int
    rebalance_days: int
    top_k: int
    weighting: str
    regime_profile: str
    vol_target_pct: float
    cost_bps: float = BASE_COST_BPS
    max_liquid_assets: int = DEFAULT_MAX_LIQUID_ASSETS
    min_median_dollar_volume: float = DEFAULT_MIN_MEDIAN_DOLLAR_VOLUME


@dataclass
class MarketMatrices:
    dates: pd.DatetimeIndex
    symbols: List[str]
    opens: np.ndarray
    closes: np.ndarray
    valuation_opens: np.ndarray
    valuation_closes: np.ndarray
    dollar_volume_median: np.ndarray
    momentum: Dict[int, np.ndarray]
    sma: Dict[int, np.ndarray]
    btc_regime: np.ndarray
    btc_realized_vol_pct: np.ndarray


@dataclass
class BacktestResult:
    equity: pd.DataFrame
    cycles: pd.DataFrame
    turnover: float
    rebalance_count: int
    initial_equity: float
    evaluation_start: pd.Timestamp


def credentials() -> Tuple[str, str]:
    key = os.getenv("ALPACA_PAPER_API_KEY") or os.getenv("ALPACA_API_KEY")
    secret = os.getenv("ALPACA_PAPER_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY")
    if not key or not secret:
        raise SystemExit(
            "Alpaca credentials are missing in this PowerShell session.\n"
            "Set ALPACA_PAPER_API_KEY and ALPACA_PAPER_SECRET_KEY, then rerun.\n"
            "Do not paste the keys into chat."
        )
    return key, secret


class AlpacaCryptoData:
    def __init__(self) -> None:
        key, secret = credentials()
        self.session = requests.Session()
        self.session.headers.update({
            "APCA-API-KEY-ID": key,
            "APCA-API-SECRET-KEY": secret,
        })

    def _get(self, url: str, *, params: Optional[dict] = None, retries: int = 5):
        delay = 1.0
        last_error = None
        for _ in range(retries):
            try:
                response = self.session.get(url, params=params, timeout=60)
                if response.status_code == 429:
                    time.sleep(delay)
                    delay = min(delay * 2.0, 16.0)
                    continue
                if response.status_code >= 500:
                    last_error = RuntimeError(
                        f"Alpaca server error {response.status_code}: {response.text[:500]}"
                    )
                    time.sleep(delay)
                    delay = min(delay * 2.0, 16.0)
                    continue
                if not response.ok:
                    raise RuntimeError(
                        f"Alpaca request failed {response.status_code}: {response.text[:1000]}"
                    )
                return response.json()
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_error = exc
                time.sleep(delay)
                delay = min(delay * 2.0, 16.0)
        raise RuntimeError(f"Alpaca request failed after retries: {last_error}")

    def active_usd_pairs(self) -> List[str]:
        payload = self._get(
            f"{PAPER_BASE}/v2/assets",
            params={"status": "active", "asset_class": "crypto"},
        )
        pairs = []
        for asset in payload if isinstance(payload, list) else []:
            symbol = str(asset.get("symbol") or "").upper()
            base = symbol.split("/")[0] if "/" in symbol else symbol
            if (
                asset.get("tradable") is True
                and str(asset.get("status") or "").lower() == "active"
                and symbol.endswith("/USD")
                and base not in EXCLUDED_BASES
            ):
                pairs.append(symbol)
        return sorted(set(pairs))

    def daily_bars(
        self,
        symbols: Sequence[str],
        start: str,
        end: Optional[str] = None,
        batch_size: int = 15,
    ) -> Dict[str, pd.DataFrame]:
        output: Dict[str, List[dict]] = {symbol: [] for symbol in symbols}
        for batch_start in range(0, len(symbols), batch_size):
            batch = list(symbols[batch_start:batch_start + batch_size])
            token = None
            page = 0
            while True:
                params = {
                    "symbols": ",".join(batch),
                    "timeframe": "1Day",
                    "start": start,
                    "sort": "asc",
                    "limit": 10000,
                }
                if end:
                    params["end"] = end
                if token:
                    params["page_token"] = token
                payload = self._get(
                    f"{DATA_BASE}/v1beta3/crypto/us/bars",
                    params=params,
                )
                bars = payload.get("bars") or {}
                for symbol, rows in bars.items():
                    if isinstance(rows, list):
                        output.setdefault(symbol, []).extend(rows)
                token = payload.get("next_page_token")
                page += 1
                if not token:
                    break
                if page > 100:
                    raise RuntimeError("Unexpected crypto bars pagination depth.")
            print(
                f"Fetched daily bars for {batch_start + 1}-"
                f"{min(batch_start + len(batch), len(symbols))} of {len(symbols)} pairs."
            )

        frames: Dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            rows = output.get(symbol) or []
            if not rows:
                continue
            frame = pd.DataFrame(rows)
            if "t" not in frame or "c" not in frame:
                continue
            frame["date"] = pd.to_datetime(frame["t"], utc=True, errors="coerce").dt.floor("D")
            frame["open"] = pd.to_numeric(frame.get("o"), errors="coerce")
            frame["close"] = pd.to_numeric(frame.get("c"), errors="coerce")
            frame["volume"] = pd.to_numeric(frame.get("v"), errors="coerce").fillna(0.0)
            frame = (
                frame[["date", "open", "close", "volume"]]
                .dropna(subset=["date", "open", "close"])
                .sort_values("date")
                .drop_duplicates("date", keep="last")
                .reset_index(drop=True)
            )
            frame = frame[(frame["open"] > 0) & (frame["close"] > 0)]
            if not frame.empty:
                frames[symbol] = frame
        return frames


def _rolling_median(arr: np.ndarray, window: int, min_periods: int) -> np.ndarray:
    return pd.DataFrame(arr).rolling(window, min_periods=min_periods).median().to_numpy(dtype=float)


def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    return pd.DataFrame(arr).rolling(window, min_periods=window).mean().to_numpy(dtype=float)


def build_matrices(frames: Mapping[str, pd.DataFrame]) -> MarketMatrices:
    if not frames:
        raise ValueError("No crypto data frames were supplied.")

    symbols = sorted(frames)
    all_dates = sorted(set().union(*[set(pd.DatetimeIndex(f["date"])) for f in frames.values()]))
    dates = pd.DatetimeIndex(all_dates)
    n, m = len(dates), len(symbols)
    opens = np.full((n, m), np.nan, dtype=float)
    closes = np.full((n, m), np.nan, dtype=float)
    volumes = np.zeros((n, m), dtype=float)
    date_to_i = {d: i for i, d in enumerate(dates)}

    for j, symbol in enumerate(symbols):
        frame = frames[symbol]
        for row in frame.itertuples(index=False):
            i = date_to_i.get(pd.Timestamp(row.date))
            if i is None:
                continue
            opens[i, j] = float(row.open)
            closes[i, j] = float(row.close)
            volumes[i, j] = float(row.volume)

    close_df = pd.DataFrame(closes, index=dates, columns=symbols)
    valuation_closes = close_df.ffill().to_numpy(dtype=float)
    open_df = pd.DataFrame(opens, index=dates, columns=symbols)
    valuation_opens = open_df.where(open_df.notna(), close_df.ffill().shift(1)).to_numpy(dtype=float)

    dollar_volume = closes * volumes
    liq = _rolling_median(
        dollar_volume,
        LIQUIDITY_LOOKBACK,
        max(10, LIQUIDITY_LOOKBACK // 2),
    )

    momentum = {}
    for lookback in MOMENTUM_GRID:
        shifted = np.vstack([np.full((lookback, m), np.nan), closes[:-lookback]])
        momentum[lookback] = closes / shifted - 1.0

    sma = {lookback: _rolling_mean(closes, lookback) for lookback in SMA_GRID}

    if "BTC/USD" in symbols:
        b = symbols.index("BTC/USD")
        btc = closes[:, b]
        btc_s = pd.Series(btc, index=dates)
        btc_sma = btc_s.rolling(BTC_REGIME_SMA_DAYS, min_periods=BTC_REGIME_SMA_DAYS).mean()
        btc_mom = btc_s / btc_s.shift(BTC_REGIME_MOM_DAYS) - 1.0
        btc_logret = np.log(btc_s / btc_s.shift(1))
        btc_vol = (
            btc_logret.rolling(BTC_VOL_LOOKBACK, min_periods=BTC_VOL_LOOKBACK)
            .std(ddof=1) * math.sqrt(365.0) * 100.0
        )
        regime = np.full(n, "sideways", dtype=object)
        valid = btc_s.notna() & btc_sma.notna() & btc_mom.notna()
        up = valid & (btc_s > btc_sma) & (btc_mom > BTC_REGIME_MOM_THRESHOLD)
        down = valid & (btc_s < btc_sma) & (btc_mom < -BTC_REGIME_MOM_THRESHOLD)
        regime[up.to_numpy()] = "up"
        regime[down.to_numpy()] = "down"
        btc_realized_vol_pct = btc_vol.to_numpy(dtype=float)
    else:
        regime = np.full(n, "sideways", dtype=object)
        btc_realized_vol_pct = np.full(n, np.nan, dtype=float)

    return MarketMatrices(
        dates=dates,
        symbols=symbols,
        opens=opens,
        closes=closes,
        valuation_opens=valuation_opens,
        valuation_closes=valuation_closes,
        dollar_volume_median=liq,
        momentum=momentum,
        sma=sma,
        btc_regime=regime,
        btc_realized_vol_pct=btc_realized_vol_pct,
    )


def exposure_for_signal(matrices: MarketMatrices, signal_index: int, config: CryptoConfig) -> Tuple[float, str, float]:
    profile = REGIME_PROFILES[config.regime_profile]
    regime = str(matrices.btc_regime[signal_index])
    regime_exposure = float(profile.get(regime, profile["sideways"]))
    realized_vol = float(matrices.btc_realized_vol_pct[signal_index])
    if math.isfinite(realized_vol) and realized_vol > 0:
        vol_scale = min(1.0, max(0.0, config.vol_target_pct / realized_vol))
    else:
        vol_scale = 0.50
    exposure = min(1.0, max(0.0, regime_exposure * vol_scale))
    return exposure, regime, realized_vol


def target_weights(matrices: MarketMatrices, signal_index: int, config: CryptoConfig) -> Tuple[np.ndarray, str, float, float]:
    m = len(matrices.symbols)
    weights = np.zeros(m, dtype=float)
    if signal_index < 0:
        return weights, "sideways", float("nan"), 0.0

    close = matrices.closes[signal_index]
    mom = matrices.momentum[config.momentum_days][signal_index]
    avg = matrices.sma[config.sma_days][signal_index]
    liq = matrices.dollar_volume_median[signal_index]

    finite = np.isfinite(close) & np.isfinite(mom) & np.isfinite(avg) & np.isfinite(liq)
    finite &= liq > max(0.0, config.min_median_dollar_volume)
    idx = np.flatnonzero(finite)
    if idx.size == 0:
        exposure, regime, vol = exposure_for_signal(matrices, signal_index, config)
        return weights, regime, vol, exposure

    # Top relative-liquidity assets only, using signal-date data.
    idx = idx[np.argsort(liq[idx])[::-1][:config.max_liquid_assets]]
    eligible = idx[(mom[idx] > 0.0) & (close[idx] > avg[idx])]
    if eligible.size:
        eligible = eligible[np.argsort(mom[eligible])[::-1][:config.top_k]]

    exposure, regime, vol = exposure_for_signal(matrices, signal_index, config)
    if eligible.size == 0 or exposure <= 0:
        return weights, regime, vol, exposure

    if config.weighting == "equal":
        weights[eligible] = exposure / eligible.size
    elif config.weighting == "rank":
        scores = np.arange(eligible.size, 0, -1, dtype=float)
        weights[eligible] = exposure * scores / scores.sum()
    else:
        raise ValueError(f"Unknown weighting: {config.weighting}")
    return weights, regime, vol, exposure


def run_backtest(
    matrices: MarketMatrices,
    config: CryptoConfig,
    *,
    trade_start_index: int,
    initial_equity: float = INITIAL_EQUITY,
) -> BacktestResult:
    dates = matrices.dates
    n, m = matrices.closes.shape
    if trade_start_index <= 0 or trade_start_index >= n - 2:
        raise ValueError("Invalid trade_start_index.")

    cash = float(initial_equity)
    qty = np.zeros(m, dtype=float)
    rows = []
    cycle_rows = []
    last_rebalance_i: Optional[int] = None
    cycle_start_equity: Optional[float] = None
    cycle_start_date: Optional[pd.Timestamp] = None
    turnover = 0.0
    rebalance_count = 0
    cost_rate = float(config.cost_bps) / 10000.0

    for i in range(trade_start_index, n):
        date = dates[i]
        op = matrices.valuation_opens[i].copy()
        cp = matrices.valuation_closes[i].copy()
        op = np.where(np.isfinite(op) & (op > 0), op, 0.0)
        cp = np.where(np.isfinite(cp) & (cp > 0), cp, op)
        pretrade_equity = cash + float(np.dot(qty, op))

        due = last_rebalance_i is None or (i - last_rebalance_i) >= config.rebalance_days
        traded_notional = 0.0
        regime = str(matrices.btc_regime[max(i - 1, 0)])
        realized_vol = float(matrices.btc_realized_vol_pct[max(i - 1, 0)])
        target_exposure = 0.0

        if due:
            weights, regime, realized_vol, target_exposure = target_weights(matrices, i - 1, config)

            if cycle_start_equity is not None and cycle_start_equity > 0:
                cycle_rows.append({
                    "start_date": cycle_start_date,
                    "end_date": date,
                    "return": pretrade_equity / cycle_start_equity - 1.0,
                    "pnl": pretrade_equity - cycle_start_equity,
                })

            current_values = qty * op
            desired_values = pretrade_equity * weights
            delta = desired_values - current_values

            # Cannot transact a symbol without a real current open.
            tradable = np.isfinite(matrices.opens[i]) & (matrices.opens[i] > 0)

            # Sell reductions first.
            sell_idx = np.flatnonzero((delta < -1e-9) & tradable & (qty > 0))
            for j in sell_idx:
                price = float(matrices.opens[i, j])
                sell_value = min(float(current_values[j]), float(-delta[j]))
                sell_qty = min(float(qty[j]), sell_value / price)
                actual_value = sell_qty * price
                fee = actual_value * cost_rate
                qty[j] = max(0.0, qty[j] - sell_qty)
                cash += actual_value - fee
                traded_notional += actual_value

            # Recompute current values after sells and buy additions.
            current_values = qty * op
            delta = desired_values - current_values
            buy_idx = np.flatnonzero((delta > 1e-9) & tradable)
            if buy_idx.size:
                buy_idx = buy_idx[np.argsort(desired_values[buy_idx])[::-1]]
            for j in buy_idx:
                price = float(matrices.opens[i, j])
                max_buy = cash / (1.0 + cost_rate) if cost_rate >= 0 else cash
                buy_value = max(0.0, min(float(delta[j]), max_buy))
                if buy_value <= 0:
                    continue
                fee = buy_value * cost_rate
                qty[j] += buy_value / price
                cash -= buy_value + fee
                traded_notional += buy_value

            posttrade_equity = cash + float(np.dot(qty, op))
            cycle_start_equity = posttrade_equity
            cycle_start_date = date
            last_rebalance_i = i
            rebalance_count += 1

        close_equity = cash + float(np.dot(qty, cp))
        rows.append({
            "date": date,
            "equity": close_equity,
            "cash": cash,
            "positions": int(np.sum(qty > 1e-12)),
            "rebalanced": bool(due),
            "turnover_notional": traded_notional,
            "btc_regime": regime,
            "btc_realized_vol_pct": realized_vol if math.isfinite(realized_vol) else np.nan,
            "target_exposure": target_exposure if due else np.nan,
        })
        turnover += traded_notional

    if cycle_start_equity is not None and cycle_start_equity > 0 and rows:
        final_equity = float(rows[-1]["equity"])
        cycle_rows.append({
            "start_date": cycle_start_date,
            "end_date": pd.Timestamp(rows[-1]["date"]),
            "return": final_equity / cycle_start_equity - 1.0,
            "pnl": final_equity - cycle_start_equity,
        })

    return BacktestResult(
        equity=pd.DataFrame(rows),
        cycles=pd.DataFrame(cycle_rows),
        turnover=float(turnover),
        rebalance_count=int(rebalance_count),
        initial_equity=float(initial_equity),
        evaluation_start=pd.Timestamp(dates[trade_start_index]),
    )


def segment_metrics(result: BacktestResult, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    eq = result.equity.copy()
    eq["date"] = pd.to_datetime(eq["date"], utc=True)
    prior = eq[eq["date"] < start]
    start_equity = float(prior["equity"].iloc[-1]) if not prior.empty else float(result.initial_equity)
    seg = eq[(eq["date"] >= start) & (eq["date"] <= end)].copy()
    if seg.empty:
        return {k: 0.0 for k in (
            "return_pct", "cagr_pct", "max_drawdown_pct", "sharpe",
            "cycles", "win_rate_pct", "profit_factor",
        )}

    last = float(seg["equity"].iloc[-1])
    total_return = last / start_equity - 1.0 if start_equity > 0 else 0.0
    days = max(1.0, (seg["date"].iloc[-1] - start).total_seconds() / 86400.0 + 1.0)
    cagr = (last / start_equity) ** (365.0 / days) - 1.0 if start_equity > 0 and last > 0 else -1.0

    curve = pd.Series([start_equity] + seg["equity"].astype(float).tolist())
    dd = curve / curve.cummax() - 1.0
    max_dd = float(dd.min())
    daily = curve.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    sharpe = (
        float(daily.mean() / daily.std(ddof=1) * math.sqrt(365.0))
        if len(daily) >= 3 and float(daily.std(ddof=1)) > 0
        else 0.0
    )

    cycles = result.cycles.copy()
    if not cycles.empty:
        cycles["end_date"] = pd.to_datetime(cycles["end_date"], utc=True)
        cseg = cycles[(cycles["end_date"] >= start) & (cycles["end_date"] <= end)]
        returns = pd.to_numeric(cseg["return"], errors="coerce").dropna()
    else:
        returns = pd.Series(dtype=float)
    cycle_count = int(len(returns))
    win_rate = float((returns > 0).mean()) if cycle_count else 0.0
    positive = float(returns[returns > 0].sum())
    negative = float(-returns[returns < 0].sum())
    pf = positive / negative if negative > 0 else (float("inf") if positive > 0 else 0.0)

    return {
        "return_pct": total_return * 100.0,
        "cagr_pct": cagr * 100.0,
        "max_drawdown_pct": max_dd * 100.0,
        "sharpe": sharpe,
        "cycles": cycle_count,
        "win_rate_pct": win_rate * 100.0,
        "profit_factor": pf,
    }


def metric_score(metrics: Mapping[str, float]) -> float:
    cagr = float(metrics.get("cagr_pct", 0.0)) / 100.0
    dd = abs(float(metrics.get("max_drawdown_pct", 0.0))) / 100.0
    win = float(metrics.get("win_rate_pct", 0.0)) / 100.0
    pf = float(metrics.get("profit_factor", 0.0))
    cycles = int(metrics.get("cycles", 0))
    if cycles < 8:
        return -999.0
    if math.isfinite(pf) and pf > 0:
        pf_term = max(-1.0, min(1.25, math.log(max(pf, 1e-9))))
    elif pf == float("inf"):
        pf_term = 1.25
    else:
        pf_term = -1.0
    return 1.8 * cagr + 0.40 * (win - 0.50) + 0.55 * pf_term - 1.50 * dd


def chronological_splits(eval_dates: pd.DatetimeIndex):
    if len(eval_dates) < 300:
        raise ValueError("Need at least 300 evaluation dates.")
    n = len(eval_dates)
    train_end = max(1, int(n * 0.60) - 1)
    val_end = max(train_end + 1, int(n * 0.80) - 1)
    return (
        (eval_dates[0], eval_dates[train_end]),
        (eval_dates[train_end + 1], eval_dates[val_end]),
        (eval_dates[val_end + 1], eval_dates[-1]),
    )


def config_from_row(row: Mapping) -> CryptoConfig:
    return CryptoConfig(
        momentum_days=int(row["momentum_days"]),
        sma_days=int(row["sma_days"]),
        rebalance_days=int(row["rebalance_days"]),
        top_k=int(row["top_k"]),
        weighting=str(row["weighting"]),
        regime_profile=str(row["regime_profile"]),
        vol_target_pct=float(row["vol_target_pct"]),
        cost_bps=float(row.get("cost_bps", BASE_COST_BPS)),
        max_liquid_assets=int(row.get("max_liquid_assets", DEFAULT_MAX_LIQUID_ASSETS)),
        min_median_dollar_volume=float(row.get("min_median_dollar_volume", 0.0)),
    )


def evaluate_grid(matrices: MarketMatrices, trade_start_index: int, eval_dates: pd.DatetimeIndex):
    train, validation, diagnostic = chronological_splits(eval_dates)
    configs = [
        CryptoConfig(mom, sma, reb, top_k, weighting, profile, vol_target)
        for mom, sma, reb, top_k, weighting, profile, vol_target in itertools.product(
            MOMENTUM_GRID, SMA_GRID, REBALANCE_GRID, TOP_K_GRID,
            WEIGHTING_GRID, REGIME_PROFILE_GRID, VOL_TARGET_GRID,
        )
    ]
    print(f"Testing {len(configs)} Crypto V3 configurations...")
    cache: Dict[CryptoConfig, BacktestResult] = {}
    train_rows = []
    for index, config in enumerate(configs, start=1):
        result = run_backtest(matrices, config, trade_start_index=trade_start_index)
        cache[config] = result
        met = segment_metrics(result, *train)
        train_rows.append({
            **asdict(config),
            **{f"train_{k}": v for k, v in met.items()},
            "train_score": metric_score(met),
        })
        if index % 12 == 0 or index == len(configs):
            print(f"  grid progress: {index}/{len(configs)}")

    train_df = pd.DataFrame(train_rows).sort_values(
        ["train_score", "train_return_pct"], ascending=False
    ).reset_index(drop=True)

    shortlist = train_df.head(24).copy()
    rows = []
    for _, row in shortlist.iterrows():
        config = config_from_row(row)
        result = cache[config]
        val = segment_metrics(result, *validation)
        tscore = float(row["train_score"])
        vscore = metric_score(val)
        train_cagr = float(row["train_cagr_pct"])
        val_cagr = float(val["cagr_pct"])
        divergence_penalty = min(2.0, abs(train_cagr - val_cagr) / 100.0)
        robust_score = min(tscore, vscore) + 0.25 * ((tscore + vscore) / 2.0) - 0.35 * divergence_penalty
        rows.append({
            **asdict(config),
            **{f"train_{k}": row[f"train_{k}"] for k in (
                "return_pct", "cagr_pct", "max_drawdown_pct", "sharpe",
                "cycles", "win_rate_pct", "profit_factor",
            )},
            "train_score": tscore,
            **{f"validation_{k}": v for k, v in val.items()},
            "validation_score": vscore,
            "robust_score": robust_score,
        })

    val_df = pd.DataFrame(rows)
    eligible = val_df[
        (val_df["train_return_pct"] > 0)
        & (val_df["validation_return_pct"] > 0)
        & (val_df["train_profit_factor"] > 1.0)
        & (val_df["validation_profit_factor"] > 1.0)
        & (val_df["train_max_drawdown_pct"] > -45.0)
        & (val_df["validation_max_drawdown_pct"] > -45.0)
        & (val_df["train_cycles"] >= 12)
        & (val_df["validation_cycles"] >= 8)
    ].copy()
    pool = eligible if not eligible.empty else val_df
    pool = pool.sort_values(["robust_score", "validation_score"], ascending=False).reset_index(drop=True)
    chosen_row = pool.iloc[0].to_dict()
    chosen = config_from_row(chosen_row)
    result = cache[chosen]
    diagnostic_metrics = segment_metrics(result, *diagnostic)
    full_metrics = segment_metrics(result, eval_dates[0], eval_dates[-1])

    report = {
        "config": asdict(chosen),
        "train": {k.replace("train_", ""): chosen_row[k] for k in chosen_row if k.startswith("train_") and k != "train_score"},
        "validation": {k.replace("validation_", ""): chosen_row[k] for k in chosen_row if k.startswith("validation_") and k != "validation_score"},
        "historical_diagnostic": diagnostic_metrics,
        "full": full_metrics,
        "train_score": float(chosen_row["train_score"]),
        "validation_score": float(chosen_row["validation_score"]),
        "robust_score": float(chosen_row["robust_score"]),
        "eligible_robust_shortlist_count": int(len(eligible)),
        "shortlist_size": int(len(val_df)),
        "splits": {
            "train": [train[0].isoformat(), train[1].isoformat()],
            "validation": [validation[0].isoformat(), validation[1].isoformat()],
            "historical_diagnostic_not_unseen": [diagnostic[0].isoformat(), diagnostic[1].isoformat()],
        },
    }
    return train_df, val_df, report, result


def benchmark_btc(matrices: MarketMatrices, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    if "BTC/USD" not in matrices.symbols:
        return {"available": False}
    j = matrices.symbols.index("BTC/USD")
    s = pd.Series(matrices.closes[:, j], index=matrices.dates).loc[start:end].dropna()
    if len(s) < 2:
        return {"available": False}
    first, last = float(s.iloc[0]), float(s.iloc[-1])
    ret = last / first - 1.0
    dd = s / s.cummax() - 1.0
    days = max(1.0, (s.index[-1] - s.index[0]).total_seconds() / 86400.0)
    cagr = (last / first) ** (365.0 / days) - 1.0
    return {
        "available": True,
        "return_pct": ret * 100.0,
        "cagr_pct": cagr * 100.0,
        "max_drawdown_pct": float(dd.min()) * 100.0,
    }


def regime_analysis(result: BacktestResult, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    eq = result.equity.copy()
    eq["date"] = pd.to_datetime(eq["date"], utc=True)
    eq = eq[(eq["date"] >= start) & (eq["date"] <= end)].copy()
    eq["ret"] = eq["equity"].pct_change().fillna(0.0)
    out = {}
    for name in ("up", "down", "sideways"):
        r = eq.loc[eq["btc_regime"] == name, "ret"].astype(float)
        out[name] = {
            "days": int(len(r)),
            "return_pct": float(np.prod(1.0 + r.to_numpy()) - 1.0) * 100.0 if len(r) else 0.0,
            "positive_day_pct": float((r > 0).mean()) * 100.0 if len(r) else 0.0,
        }
    return out


def stress_costs(matrices: MarketMatrices, config: CryptoConfig, trade_start_index: int, start: pd.Timestamp, end: pd.Timestamp):
    rows = []
    for mult in (1.0, 2.0, 3.0):
        stressed = CryptoConfig(
            momentum_days=config.momentum_days,
            sma_days=config.sma_days,
            rebalance_days=config.rebalance_days,
            top_k=config.top_k,
            weighting=config.weighting,
            regime_profile=config.regime_profile,
            vol_target_pct=config.vol_target_pct,
            cost_bps=config.cost_bps * mult,
            max_liquid_assets=config.max_liquid_assets,
            min_median_dollar_volume=config.min_median_dollar_volume,
        )
        result = run_backtest(matrices, stressed, trade_start_index=trade_start_index)
        rows.append({
            "cost_multiplier": mult,
            "one_way_cost_bps": stressed.cost_bps,
            **segment_metrics(result, start, end),
        })
    return rows


def candidate_status(report: Mapping, stress: Sequence[Mapping]) -> str:
    train = report["train"]
    val = report["validation"]
    diag = report["historical_diagnostic"]
    full = report["full"]
    def positive(seg: Mapping, min_cycles: int) -> bool:
        return (
            float(seg.get("return_pct", 0)) > 0
            and float(seg.get("profit_factor", 0)) > 1.0
            and int(seg.get("cycles", 0)) >= min_cycles
        )
    double = next((x for x in stress if float(x["cost_multiplier"]) == 2.0), None)
    robust = (
        positive(train, 12)
        and positive(val, 8)
        and positive(diag, 8)
        and float(train.get("max_drawdown_pct", -100)) > -40
        and float(val.get("max_drawdown_pct", -100)) > -40
        and float(diag.get("max_drawdown_pct", -100)) > -40
        and float(full.get("max_drawdown_pct", -100)) > -35
        and double is not None
        and float(double.get("return_pct", 0)) > 0
        and float(double.get("max_drawdown_pct", -100)) > -40
    )
    return "HISTORICAL_PASS_REQUIRES_PAPER_FORWARD" if robust else "REJECT_OR_RESEARCH_FURTHER"


def save_pack(out_dir: Path, *, summary: dict, universe_df: pd.DataFrame, train_df: pd.DataFrame, validation_df: pd.DataFrame, result: BacktestResult) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "crypto_rotation_v3_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    universe_df.to_csv(out_dir / "universe.csv", index=False)
    train_df.to_csv(out_dir / "candidate_grid_train.csv", index=False)
    validation_df.to_csv(out_dir / "shortlist_validation.csv", index=False)
    result.equity.to_csv(out_dir / "selected_equity.csv", index=False)
    result.cycles.to_csv(out_dir / "selected_cycles.csv", index=False)
    readme = f"""# Crypto Rotation V3 Research Pack

Status: {summary['status']}

Research only. No Alpaca orders are placed.

Selected config:
{json.dumps(summary['selected']['config'], indent=2)}

V3 methodology notes:
- Evaluation starts at exactly $100,000 with zero pre-evaluation positions.
- Earlier bars are warmup data only.
- Stablecoin/cash-like bases are excluded from risk-asset ranking; USD cash remains available.
- Liquidity is relative: top 15 by trailing 30-day median Alpaca-feed dollar volume.
- BTC regime controls maximum exposure; BTC realized volatility scales exposure further.
- Signals use the prior completed UTC daily bar; execution is next UTC daily open.
- Train and validation choose the candidate using a robustness score.
- The final historical segment is diagnostic only because V2 results were already viewed.
- True unseen validation must be paper-forward after any historical pass.
- Long/cash only; no simulated shorting.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    zip_path = out_dir.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(out_dir.rglob("*")):
            if path.is_file():
                zf.write(path, arcname=f"{out_dir.name}/{path.relative_to(out_dir)}")
    return zip_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Research Crypto Rotation V3 using Alpaca historical data.")
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--output-dir", type=Path, default=Path(PACK_DIRNAME))
    args = parser.parse_args()

    print("CRYPTO ROTATION V3 RESEARCH — NO ORDERS WILL BE PLACED")
    print("V3 accounting fix: no positions/trades before the common evaluation start.")
    print("Discovering active Alpaca USD crypto pairs...")
    api = AlpacaCryptoData()
    symbols = api.active_usd_pairs()
    if len(symbols) < 3:
        raise RuntimeError(f"Only {len(symbols)} eligible crypto pairs discovered.")
    print(f"Discovered {len(symbols)} active non-stablecoin USD crypto pairs.")
    frames = api.daily_bars(symbols, args.start, args.end)

    universe_rows = []
    for symbol in symbols:
        frame = frames.get(symbol)
        if frame is None or frame.empty:
            universe_rows.append({"symbol": symbol, "bars": 0, "first_date": None, "last_date": None, "recent_median_dollar_volume": 0.0, "usable": False})
            continue
        dv = frame["close"] * frame["volume"]
        universe_rows.append({
            "symbol": symbol,
            "bars": int(len(frame)),
            "first_date": frame["date"].min(),
            "last_date": frame["date"].max(),
            "recent_median_dollar_volume": float(dv.tail(90).median()),
            "usable": bool(len(frame) >= WARMUP_DAYS + 60),
        })
    universe_df = pd.DataFrame(universe_rows).sort_values("recent_median_dollar_volume", ascending=False)
    usable = set(universe_df.loc[universe_df["usable"], "symbol"].astype(str))
    usable_frames = {s: f for s, f in frames.items() if s in usable}
    if len(usable_frames) < 3:
        raise RuntimeError("Fewer than 3 pairs have sufficient history.")
    if "BTC/USD" not in usable_frames:
        raise RuntimeError("BTC/USD is required for V3 regime and volatility controls.")

    matrices = build_matrices(usable_frames)
    if len(matrices.dates) <= WARMUP_DAYS + 300:
        raise RuntimeError("Not enough daily history for V3 chronological evaluation.")
    trade_start_index = WARMUP_DAYS
    eval_dates = matrices.dates[trade_start_index:]

    train_df, validation_df, selected_report, selected_result = evaluate_grid(
        matrices, trade_start_index, eval_dates
    )
    selected_config = CryptoConfig(**selected_report["config"])
    stress = stress_costs(matrices, selected_config, trade_start_index, eval_dates[0], eval_dates[-1])
    regimes = regime_analysis(selected_result, eval_dates[0], eval_dates[-1])
    selected_report["cost_stress"] = stress
    selected_report["regimes"] = regimes
    status = candidate_status(selected_report, stress)

    first_recorded = float(selected_result.equity["equity"].iloc[0])
    ending = float(selected_result.equity["equity"].iloc[-1])
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "strategy": "CRYPTO_ROTATION_V3_RESEARCH",
        "research_only": True,
        "orders_placed": False,
        "status": status,
        "accounting_check": {
            "initial_evaluation_equity": INITIAL_EQUITY,
            "pre_evaluation_trades": 0,
            "evaluation_start": eval_dates[0].isoformat(),
            "first_recorded_close_equity": first_recorded,
            "ending_equity": ending,
            "return_from_initial_capital_pct": (ending / INITIAL_EQUITY - 1.0) * 100.0,
        },
        "data": {
            "requested_start": args.start,
            "requested_end": args.end,
            "active_usd_pairs_discovered": len(symbols),
            "usable_pairs": len(usable_frames),
            "evaluation_start": eval_dates[0].isoformat(),
            "evaluation_end": eval_dates[-1].isoformat(),
            "survivorship_note": "Universe discovery begins from currently active/tradable Alpaca pairs. Delisted historical pairs are not recovered by this harness.",
        },
        "selected": selected_report,
        "btc_buy_and_hold_benchmark": benchmark_btc(matrices, eval_dates[0], eval_dates[-1]),
        "methodology": {
            "signal_timing": "prior completed UTC daily bar",
            "execution": "next UTC daily bar open",
            "trade_direction": "long/cash only",
            "grid_size": len(MOMENTUM_GRID) * len(SMA_GRID) * len(REBALANCE_GRID) * len(TOP_K_GRID) * len(WEIGHTING_GRID) * len(REGIME_PROFILE_GRID) * len(VOL_TARGET_GRID),
            "base_one_way_cost_bps": BASE_COST_BPS,
            "liquidity_lookback_days": LIQUIDITY_LOOKBACK,
            "max_liquid_assets_per_rebalance": DEFAULT_MAX_LIQUID_ASSETS,
            "minimum_trailing_median_dollar_volume": DEFAULT_MIN_MEDIAN_DOLLAR_VOLUME,
            "liquidity_policy": "Rank by trailing 30-day median Alpaca-feed dollar volume; no fixed dollar floor.",
            "regime_profiles": REGIME_PROFILES,
            "btc_regime_rule": "UP: prior BTC close > SMA100 and 30d momentum > +5%; DOWN: close < SMA100 and momentum < -5%; else SIDEWAYS.",
            "volatility_scaling": "total exposure *= min(1, target_vol / prior BTC 20d annualized realized vol)",
            "selection_policy": "train shortlist -> validation robustness score; historical final segment is diagnostic, not unseen",
            "true_unseen_policy": "paper-forward required before any live consideration",
        },
    }

    zip_path = save_pack(
        args.output_dir.resolve(),
        summary=summary,
        universe_df=universe_df,
        train_df=train_df,
        validation_df=validation_df,
        result=selected_result,
    )

    s = selected_report
    print("\n=== CRYPTO ROTATION V3 RESULT ===")
    print(f"Status: {status}")
    print(f"Usable pairs: {len(usable_frames)}")
    print(f"Selected config: {s['config']}")
    print(f"TRAIN: return={s['train']['return_pct']:.2f}% win={s['train']['win_rate_pct']:.1f}% PF={s['train']['profit_factor']:.2f} DD={s['train']['max_drawdown_pct']:.2f}%")
    print(f"VALIDATION: return={s['validation']['return_pct']:.2f}% win={s['validation']['win_rate_pct']:.1f}% PF={s['validation']['profit_factor']:.2f} DD={s['validation']['max_drawdown_pct']:.2f}%")
    d = s['historical_diagnostic']
    print(f"HISTORICAL DIAGNOSTIC: return={d['return_pct']:.2f}% win={d['win_rate_pct']:.1f}% PF={d['profit_factor']:.2f} DD={d['max_drawdown_pct']:.2f}%")
    print(f"FULL FROM $100K: return={s['full']['return_pct']:.2f}% win={s['full']['win_rate_pct']:.1f}% PF={s['full']['profit_factor']:.2f} DD={s['full']['max_drawdown_pct']:.2f}%")
    print(f"Ending equity from $100,000: ${ending:,.2f}")
    print(f"\nResearch pack: {zip_path}")
    print("NO ORDERS WERE PLACED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
