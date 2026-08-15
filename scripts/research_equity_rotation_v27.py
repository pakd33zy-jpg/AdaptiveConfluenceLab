#!/usr/bin/env python3
"""
Expanded ETF/equities rotation research harness (V27).

Research goals
--------------
- Preserve frozen V26 as the paper-forward benchmark; V27 is separate research.
- Use a broad, diversified liquid ETF universe spanning US equity, sectors,
  industries, real estate, international, bonds, metals and oil.
- Use only completed daily bars for signals and simulate fills at the NEXT daily open.
- Apply market-regime control from SPY, volatility-scaled exposure, relative liquidity,
  category diversification, immediate next-open risk reduction, transaction-cost stress,
  chronological development folds, and an untouched-by-selection historical diagnostic.
- Research only. NEVER places orders.

Known limitations
-----------------
- Universe is a hand-selected set of currently liquid ETFs, so selection/survivorship bias
  remains possible.
- Alpaca IEX daily bars are used for broad accessibility; they are not consolidated SIP data.
- Historical diagnostic is not truly unseen once reviewed. A historical pass must still go
  through paper-forward validation before any live-trading discussion.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import re
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

DEFAULT_START = "2018-01-01T00:00:00Z"
INITIAL_EQUITY = 100_000.0
BASE_COST_BPS = 10.0
LIQUIDITY_LOOKBACK = 20
DEFAULT_MAX_LIQUID_ASSETS = 24
RISK_ON_CONFIRM_DAYS = 2
MIN_RISK_ON_BREADTH = 4
PACK_DIRNAME = "EquityRotationV27_research_pack"

UNIVERSE = (
    "SPY", "QQQ", "IWM", "DIA", "VTI", "RSP", "MDY",
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLC",
    "SMH", "SOXX", "IBB", "XBI", "KRE", "XHB", "ITB",
    "IYR", "VNQ", "EFA", "EEM",
    "TLT", "IEF", "HYG", "LQD",
    "GLD", "SLV", "USO",
)

GROUPS = {
    # Duplicates/near-substitutes are grouped to limit concentration.
    "SPY": "core_us", "QQQ": "core_us", "IWM": "core_us", "DIA": "core_us",
    "VTI": "core_us", "RSP": "core_us", "MDY": "core_us",
    "XLK": "sector", "XLF": "sector", "XLE": "sector", "XLV": "sector",
    "XLI": "sector", "XLY": "sector", "XLP": "sector", "XLU": "sector",
    "XLB": "sector", "XLC": "sector",
    "SMH": "semis", "SOXX": "semis",
    "IBB": "biotech", "XBI": "biotech",
    "KRE": "regional_banks",
    "XHB": "housing", "ITB": "housing",
    "IYR": "real_estate", "VNQ": "real_estate",
    "EFA": "international", "EEM": "international",
    "TLT": "treasury", "IEF": "treasury",
    "HYG": "credit", "LQD": "credit",
    "GLD": "metals", "SLV": "metals",
    "USO": "oil",
}

DEFENSIVE_SYMBOLS = frozenset(("TLT", "IEF", "LQD", "GLD"))

MOMENTUM_GRID = (63, 126)
SMA_GRID = (150, 200)
REBALANCE_GRID = (5, 10, 20)
TOP_K_GRID = (3, 4, 5)
WEIGHTING_GRID = ("equal", "rank")
RISK_PROFILE_GRID = ("strict", "defensive")
VOL_TARGET_GRID = (12.0, 16.0)

SPY_REGIME_SMA_DAYS = 200
SPY_REGIME_MOM_DAYS = 63
SPY_VOL_LOOKBACK = 20
MAX_SMA = max(max(SMA_GRID), SPY_REGIME_SMA_DAYS)
MAX_MOM = max(max(MOMENTUM_GRID), SPY_REGIME_MOM_DAYS)
WARMUP_DAYS = max(MAX_SMA, MAX_MOM + 1, LIQUIDITY_LOOKBACK + 1, SPY_VOL_LOOKBACK + 2)


@dataclass(frozen=True)
class EquityConfig:
    momentum_days: int
    sma_days: int
    rebalance_days: int
    top_k: int
    weighting: str
    risk_profile: str
    vol_target_pct: float
    cost_bps: float = BASE_COST_BPS
    max_liquid_assets: int = DEFAULT_MAX_LIQUID_ASSETS


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
    spy_regime: np.ndarray
    spy_realized_vol_pct: np.ndarray


@dataclass
class BacktestResult:
    equity: pd.DataFrame
    cycles: pd.DataFrame
    turnover: float
    rebalance_count: int
    initial_equity: float
    evaluation_start: pd.Timestamp


def credentials() -> Tuple[str, str]:
    key = os.getenv("ALPACA_PAPER_API_KEY") or os.getenv("ALPACA_API_KEY") or ""
    secret = os.getenv("ALPACA_PAPER_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY") or ""
    key = re.sub(r"\s+", "", key)
    secret = re.sub(r"\s+", "", secret)
    if not key or not secret:
        raise SystemExit(
            "Alpaca PAPER credentials are missing. Add repository secrets "
            "ALPACA_PAPER_API_KEY and ALPACA_PAPER_SECRET_KEY."
        )
    return key, secret


class AlpacaEquityData:
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

    def validate_assets(self, symbols: Sequence[str]) -> List[str]:
        payload = self._get(
            f"{PAPER_BASE}/v2/assets",
            params={"status": "active", "asset_class": "us_equity"},
        )
        active = {
            str(x.get("symbol") or "").upper()
            for x in payload if isinstance(payload, list)
            if x.get("tradable") is True and str(x.get("status") or "").lower() == "active"
        }
        return [s for s in symbols if s in active]

    def daily_bars(
        self,
        symbols: Sequence[str],
        start: str = DEFAULT_START,
        end: Optional[str] = None,
        batch_size: int = 10,
    ) -> Dict[str, pd.DataFrame]:
        out: Dict[str, List[dict]] = {s: [] for s in symbols}
        for batch_start in range(0, len(symbols), batch_size):
            batch = list(symbols[batch_start:batch_start + batch_size])
            token = None
            pages = 0
            while True:
                params = {
                    "symbols": ",".join(batch),
                    "timeframe": "1Day",
                    "start": start,
                    "feed": "iex",
                    "adjustment": "all",
                    "sort": "asc",
                    "limit": 10000,
                }
                if end:
                    params["end"] = end
                if token:
                    params["page_token"] = token
                payload = self._get(f"{DATA_BASE}/v2/stocks/bars", params=params)
                bars = payload.get("bars") or {}
                for symbol, rows in bars.items():
                    if isinstance(rows, list):
                        out.setdefault(symbol, []).extend(rows)
                token = payload.get("next_page_token")
                pages += 1
                if not token:
                    break
                if pages > 100:
                    raise RuntimeError("Unexpected stock-bars pagination depth.")
            print(
                f"Fetched daily bars for {batch_start + 1}-"
                f"{min(batch_start + len(batch), len(symbols))} of {len(symbols)} ETFs."
            )

        frames: Dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            rows = out.get(symbol) or []
            if not rows:
                continue
            f = pd.DataFrame(rows)
            if "t" not in f or "c" not in f:
                continue
            f["date"] = pd.to_datetime(f["t"], utc=True, errors="coerce").dt.floor("D")
            f["open"] = pd.to_numeric(f.get("o"), errors="coerce")
            f["close"] = pd.to_numeric(f.get("c"), errors="coerce")
            f["volume"] = pd.to_numeric(f.get("v"), errors="coerce").fillna(0.0)
            f = (
                f[["date", "open", "close", "volume"]]
                .dropna(subset=["date", "open", "close"])
                .sort_values("date")
                .drop_duplicates("date", keep="last")
                .reset_index(drop=True)
            )
            f = f[(f["open"] > 0) & (f["close"] > 0)]
            if len(f) >= 300:
                frames[symbol] = f
        return frames


def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    return pd.DataFrame(arr).rolling(window, min_periods=window).mean().to_numpy(dtype=float)


def _rolling_median(arr: np.ndarray, window: int, min_periods: int) -> np.ndarray:
    return pd.DataFrame(arr).rolling(window, min_periods=min_periods).median().to_numpy(dtype=float)


def build_matrices(frames: Mapping[str, pd.DataFrame]) -> MarketMatrices:
    if "SPY" not in frames:
        raise ValueError("SPY data is required for V27 regime control.")
    symbols = sorted(frames)
    all_dates = sorted(set().union(*[set(pd.DatetimeIndex(f["date"])) for f in frames.values()]))
    dates = pd.DatetimeIndex(all_dates)
    n, m = len(dates), len(symbols)

    opens = np.full((n, m), np.nan, dtype=float)
    closes = np.full((n, m), np.nan, dtype=float)
    volumes = np.zeros((n, m), dtype=float)
    date_to_i = {d: i for i, d in enumerate(dates)}
    for j, symbol in enumerate(symbols):
        for row in frames[symbol].itertuples(index=False):
            i = date_to_i.get(pd.Timestamp(row.date))
            if i is not None:
                opens[i, j] = float(row.open)
                closes[i, j] = float(row.close)
                volumes[i, j] = float(row.volume)

    close_df = pd.DataFrame(closes, index=dates, columns=symbols)
    open_df = pd.DataFrame(opens, index=dates, columns=symbols)
    valuation_closes = close_df.ffill().to_numpy(dtype=float)
    valuation_opens = open_df.where(open_df.notna(), close_df.ffill().shift(1)).to_numpy(dtype=float)

    dollar_volume = closes * volumes
    liq = _rolling_median(
        dollar_volume, LIQUIDITY_LOOKBACK, max(10, LIQUIDITY_LOOKBACK // 2)
    )

    momentum: Dict[int, np.ndarray] = {}
    for lookback in sorted(set(MOMENTUM_GRID + (SPY_REGIME_MOM_DAYS,))):
        shifted = np.vstack([np.full((lookback, m), np.nan), closes[:-lookback]])
        momentum[lookback] = closes / shifted - 1.0

    sma = {
        lookback: _rolling_mean(closes, lookback)
        for lookback in sorted(set(SMA_GRID + (SPY_REGIME_SMA_DAYS,)))
    }

    spy_j = symbols.index("SPY")
    spy = pd.Series(closes[:, spy_j], index=dates)
    spy_sma = spy.rolling(SPY_REGIME_SMA_DAYS, min_periods=SPY_REGIME_SMA_DAYS).mean()
    spy_mom = spy / spy.shift(SPY_REGIME_MOM_DAYS) - 1.0
    logret = np.log(spy / spy.shift(1))
    spy_vol = (
        logret.rolling(SPY_VOL_LOOKBACK, min_periods=SPY_VOL_LOOKBACK)
        .std(ddof=1) * math.sqrt(252.0) * 100.0
    )

    regime = np.full(n, "neutral", dtype=object)
    valid = spy.notna() & spy_sma.notna() & spy_mom.notna()
    up = valid & (spy > spy_sma) & (spy_mom > 0.0)
    down = valid & (spy < spy_sma) & (spy_mom < 0.0)
    regime[up.to_numpy()] = "up"
    regime[down.to_numpy()] = "down"

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
        spy_regime=regime,
        spy_realized_vol_pct=spy_vol.to_numpy(dtype=float),
    )


def confirmed_risk_on(matrices: MarketMatrices, signal_i: int) -> bool:
    start = signal_i - RISK_ON_CONFIRM_DAYS + 1
    if start < 0:
        return False
    return bool(np.all(matrices.spy_regime[start:signal_i + 1] == "up"))


def exposure_and_mode(
    matrices: MarketMatrices,
    signal_i: int,
    config: EquityConfig,
) -> Tuple[float, str, str, float]:
    regime = str(matrices.spy_regime[signal_i]) if signal_i >= 0 else "neutral"
    vol = float(matrices.spy_realized_vol_pct[signal_i]) if signal_i >= 0 else float("nan")

    if regime == "up" and confirmed_risk_on(matrices, signal_i):
        base = 1.0
        mode = "risk"
    elif config.risk_profile == "defensive" and regime == "neutral":
        base = 0.60
        mode = "defensive"
    elif config.risk_profile == "defensive" and regime == "down":
        base = 0.50
        mode = "defensive"
    else:
        return 0.0, regime, "cash", vol

    vol_scale = min(1.0, config.vol_target_pct / vol) if math.isfinite(vol) and vol > 0 else 0.50
    exposure = max(0.0, min(1.0, base * vol_scale))
    return exposure, regime, mode, vol


def _group_diversified_selection(
    symbols: Sequence[str],
    candidate_indices: Sequence[int],
    momentum_row: np.ndarray,
    top_k: int,
) -> List[int]:
    chosen: List[int] = []
    used_groups = set()
    ranked = sorted(candidate_indices, key=lambda j: float(momentum_row[j]), reverse=True)
    for j in ranked:
        group = GROUPS.get(symbols[j], symbols[j])
        if group in used_groups:
            continue
        chosen.append(int(j))
        used_groups.add(group)
        if len(chosen) >= top_k:
            break
    return chosen


def target_weights(
    matrices: MarketMatrices,
    signal_i: int,
    config: EquityConfig,
) -> Tuple[np.ndarray, str, str, float, float]:
    m = len(matrices.symbols)
    weights = np.zeros(m, dtype=float)
    if signal_i < 0:
        return weights, "neutral", "cash", float("nan"), 0.0

    exposure, regime, mode, vol = exposure_and_mode(matrices, signal_i, config)
    if exposure <= 0 or mode == "cash":
        return weights, regime, mode, vol, 0.0

    close = matrices.closes[signal_i]
    mom = matrices.momentum[config.momentum_days][signal_i]
    avg = matrices.sma[config.sma_days][signal_i]
    liq = matrices.dollar_volume_median[signal_i]

    finite = np.isfinite(close) & np.isfinite(mom) & np.isfinite(avg) & np.isfinite(liq)
    idx = np.flatnonzero(finite)
    if idx.size == 0:
        return weights, regime, mode, vol, 0.0

    idx = idx[np.argsort(liq[idx])[::-1][:config.max_liquid_assets]]

    if mode == "defensive":
        idx = np.array(
            [j for j in idx if matrices.symbols[j] in DEFENSIVE_SYMBOLS],
            dtype=int,
        )

    eligible = idx[(mom[idx] > 0.0) & (close[idx] > avg[idx])]
    if mode == "risk" and len(eligible) < MIN_RISK_ON_BREADTH:
        return weights, regime, mode, vol, 0.0
    if mode == "defensive" and len(eligible) < 1:
        return weights, regime, mode, vol, 0.0

    chosen = _group_diversified_selection(
        matrices.symbols, eligible.tolist(), mom, config.top_k
    )
    if not chosen:
        return weights, regime, mode, vol, 0.0

    if config.weighting == "equal":
        raw = np.ones(len(chosen), dtype=float)
    elif config.weighting == "rank":
        raw = np.arange(len(chosen), 0, -1, dtype=float)
    else:
        raise ValueError(f"Unknown weighting: {config.weighting}")

    raw /= raw.sum()
    for j, w in zip(chosen, raw):
        weights[j] = exposure * float(w)
    return weights, regime, mode, vol, exposure


def run_backtest(
    matrices: MarketMatrices,
    config: EquityConfig,
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
    cycles = []
    last_rebalance_i: Optional[int] = None
    prior_mode: Optional[str] = None
    cycle_start_equity: Optional[float] = None
    cycle_start_date: Optional[pd.Timestamp] = None
    cycle_active = False
    turnover = 0.0
    rebalances = 0
    cost_rate = config.cost_bps / 10000.0

    for i in range(trade_start_index, n):
        date = dates[i]
        op = np.where(
            np.isfinite(matrices.valuation_opens[i]) & (matrices.valuation_opens[i] > 0),
            matrices.valuation_opens[i], 0.0
        )
        cp = np.where(
            np.isfinite(matrices.valuation_closes[i]) & (matrices.valuation_closes[i] > 0),
            matrices.valuation_closes[i], op
        )
        pre_equity = cash + float(np.dot(qty, op))

        signal_i = i - 1
        _, _, current_mode, _ = exposure_and_mode(matrices, signal_i, config)
        scheduled = last_rebalance_i is None or (i - last_rebalance_i) >= config.rebalance_days
        regime_flip = prior_mode is not None and current_mode != prior_mode
        due = bool(scheduled or regime_flip)
        reason = "regime_flip" if regime_flip else ("scheduled" if scheduled else "")
        traded_notional = 0.0
        regime = str(matrices.spy_regime[signal_i])
        vol = float(matrices.spy_realized_vol_pct[signal_i])
        target_exposure = np.nan

        if due:
            weights, regime, current_mode, vol, target_exposure = target_weights(
                matrices, signal_i, config
            )

            if cycle_start_equity is not None and cycle_start_equity > 0:
                cycles.append({
                    "start_date": cycle_start_date,
                    "end_date": date,
                    "return": pre_equity / cycle_start_equity - 1.0,
                    "pnl": pre_equity - cycle_start_equity,
                    "active": bool(cycle_active),
                })

            desired = pre_equity * weights
            current = qty * op
            delta = desired - current
            tradable = np.isfinite(matrices.opens[i]) & (matrices.opens[i] > 0)

            # Sell first so risk reductions happen at this next open.
            for j in np.flatnonzero((delta < -1e-9) & tradable & (qty > 0)):
                price = float(matrices.opens[i, j])
                value = min(float(current[j]), float(-delta[j]))
                sell_qty = min(float(qty[j]), value / price)
                actual = sell_qty * price
                fee = actual * cost_rate
                qty[j] = max(0.0, qty[j] - sell_qty)
                cash += actual - fee
                traded_notional += actual

            current = qty * op
            delta = desired - current
            buys = np.flatnonzero((delta > 1e-9) & tradable)
            if buys.size:
                buys = buys[np.argsort(desired[buys])[::-1]]
            for j in buys:
                price = float(matrices.opens[i, j])
                max_buy = cash / (1.0 + cost_rate)
                value = max(0.0, min(float(delta[j]), max_buy))
                if value <= 0:
                    continue
                fee = value * cost_rate
                qty[j] += value / price
                cash -= value + fee
                traded_notional += value

            post_equity = cash + float(np.dot(qty, op))
            cycle_start_equity = post_equity
            cycle_start_date = date
            cycle_active = bool(np.any(qty > 1e-12))
            last_rebalance_i = i
            rebalances += 1

        close_equity = cash + float(np.dot(qty, cp))
        rows.append({
            "date": date,
            "equity": close_equity,
            "cash": cash,
            "positions": int(np.sum(qty > 1e-12)),
            "rebalanced": due,
            "rebalance_reason": reason,
            "spy_regime": regime,
            "mode": current_mode,
            "spy_realized_vol_pct": vol if math.isfinite(vol) else np.nan,
            "target_exposure": target_exposure,
            "turnover_notional": traded_notional,
        })
        turnover += traded_notional
        prior_mode = current_mode

    if cycle_start_equity is not None and cycle_start_equity > 0 and rows:
        final_equity = float(rows[-1]["equity"])
        cycles.append({
            "start_date": cycle_start_date,
            "end_date": pd.Timestamp(rows[-1]["date"]),
            "return": final_equity / cycle_start_equity - 1.0,
            "pnl": final_equity - cycle_start_equity,
            "active": bool(cycle_active),
        })

    return BacktestResult(
        equity=pd.DataFrame(rows),
        cycles=pd.DataFrame(cycles),
        turnover=float(turnover),
        rebalance_count=int(rebalances),
        initial_equity=float(initial_equity),
        evaluation_start=pd.Timestamp(dates[trade_start_index]),
    )


def segment_metrics(result: BacktestResult, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    eq = result.equity.copy()
    eq["date"] = pd.to_datetime(eq["date"], utc=True)
    seg = eq[(eq["date"] >= start) & (eq["date"] <= end)].copy()
    if seg.empty:
        return {
            "return_pct": 0.0, "cagr_pct": 0.0, "max_drawdown_pct": 0.0,
            "sharpe": 0.0, "cycles": 0, "win_rate_pct": 0.0,
            "profit_factor": 0.0,
        }

    start_equity = result.initial_equity if start <= result.evaluation_start else float(seg["equity"].iloc[0])
    ending = float(seg["equity"].iloc[-1])
    total_return = ending / start_equity - 1.0

    days = max(1, (pd.Timestamp(seg["date"].iloc[-1]) - pd.Timestamp(seg["date"].iloc[0])).days)
    years = days / 365.25
    cagr = (ending / start_equity) ** (1.0 / years) - 1.0 if years > 0 and ending > 0 else 0.0

    curve = pd.to_numeric(seg["equity"], errors="coerce")
    dd = curve / curve.cummax() - 1.0
    max_dd = float(dd.min()) if len(dd) else 0.0
    daily = curve.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    sharpe = float(daily.mean() / daily.std(ddof=1) * math.sqrt(252.0)) if len(daily) > 2 and daily.std(ddof=1) > 0 else 0.0

    c = result.cycles.copy()
    if not c.empty:
        c["end_date"] = pd.to_datetime(c["end_date"], utc=True)
        c = c[(c["end_date"] >= start) & (c["end_date"] <= end)]
        if "active" in c:
            c = c[c["active"].astype(bool)]
        returns = pd.to_numeric(c["return"], errors="coerce").dropna()
        returns = returns[returns.abs() > 1e-12]
    else:
        returns = pd.Series(dtype=float)

    wins = returns[returns > 0]
    losses = returns[returns < 0]
    pf = float(wins.sum() / abs(losses.sum())) if len(losses) and abs(losses.sum()) > 0 else (float("inf") if len(wins) else 0.0)

    return {
        "return_pct": float(total_return * 100.0),
        "cagr_pct": float(cagr * 100.0),
        "max_drawdown_pct": float(max_dd * 100.0),
        "sharpe": sharpe,
        "cycles": int(len(returns)),
        "win_rate_pct": float((returns > 0).mean() * 100.0) if len(returns) else 0.0,
        "profit_factor": pf,
    }


def development_splits(eval_dates: pd.DatetimeIndex):
    if len(eval_dates) < 1000:
        raise ValueError("Need at least 1000 evaluation sessions for V27.")
    n = len(eval_dates)
    dev_n = int(n * 0.80)
    bounds = np.linspace(0, dev_n, 5, dtype=int)
    folds = []
    for k in range(4):
        a = int(bounds[k])
        b = int(bounds[k + 1]) - 1
        folds.append((eval_dates[a], eval_dates[b]))
    diagnostic = (eval_dates[dev_n], eval_dates[-1])
    return folds, diagnostic


def metric_score(m: Mapping[str, float]) -> float:
    cycles = int(m.get("cycles", 0))
    if cycles < 6:
        return -999.0
    cagr = float(m.get("cagr_pct", 0.0)) / 100.0
    dd = abs(float(m.get("max_drawdown_pct", 0.0))) / 100.0
    sharpe = max(-2.0, min(3.0, float(m.get("sharpe", 0.0))))
    pf = float(m.get("profit_factor", 0.0))
    if pf == float("inf"):
        pf_term = 1.0
    elif pf > 0:
        pf_term = max(-1.0, min(1.0, math.log(pf)))
    else:
        pf_term = -1.0
    return 1.5 * cagr + 0.30 * sharpe + 0.45 * pf_term - 2.25 * dd


def config_from_row(row: Mapping) -> EquityConfig:
    return EquityConfig(
        momentum_days=int(row["momentum_days"]),
        sma_days=int(row["sma_days"]),
        rebalance_days=int(row["rebalance_days"]),
        top_k=int(row["top_k"]),
        weighting=str(row["weighting"]),
        risk_profile=str(row["risk_profile"]),
        vol_target_pct=float(row["vol_target_pct"]),
        cost_bps=float(row.get("cost_bps", BASE_COST_BPS)),
        max_liquid_assets=int(row.get("max_liquid_assets", DEFAULT_MAX_LIQUID_ASSETS)),
    )


def evaluate_grid(matrices: MarketMatrices, trade_start_index: int):
    eval_dates = matrices.dates[trade_start_index:]
    folds, diagnostic = development_splits(eval_dates)
    configs = [
        EquityConfig(mom, sma, reb, top_k, weighting, risk_profile, vol_target)
        for mom, sma, reb, top_k, weighting, risk_profile, vol_target in itertools.product(
            MOMENTUM_GRID, SMA_GRID, REBALANCE_GRID, TOP_K_GRID,
            WEIGHTING_GRID, RISK_PROFILE_GRID, VOL_TARGET_GRID,
        )
    ]
    print(f"Testing {len(configs)} V27 ETF configurations...")
    cache: Dict[EquityConfig, BacktestResult] = {}
    rows = []

    for idx, config in enumerate(configs, start=1):
        result = run_backtest(matrices, config, trade_start_index=trade_start_index)
        cache[config] = result
        mets = [segment_metrics(result, *fold) for fold in folds]
        scores = [metric_score(x) for x in mets]
        cagrs = np.array([x["cagr_pct"] for x in mets], dtype=float)
        row = {**asdict(config)}
        for k, met in enumerate(mets, start=1):
            for name, value in met.items():
                row[f"fold{k}_{name}"] = value
            row[f"fold{k}_score"] = scores[k - 1]
        row["worst_fold_score"] = float(min(scores))
        row["mean_fold_score"] = float(np.mean(scores))
        row["cagr_dispersion"] = float(np.std(cagrs) / 100.0)
        row["robust_score"] = float(min(scores) + 0.35 * np.mean(scores) - 0.25 * np.std(cagrs) / 100.0)
        rows.append(row)
        if idx % 24 == 0 or idx == len(configs):
            print(f"  grid progress: {idx}/{len(configs)}")

    grid = pd.DataFrame(rows).sort_values(
        ["robust_score", "worst_fold_score"], ascending=False
    ).reset_index(drop=True)

    eligible = grid.copy()
    for k in range(1, 5):
        eligible = eligible[
            (eligible[f"fold{k}_return_pct"] > 0)
            & (eligible[f"fold{k}_profit_factor"] > 1.0)
            & (eligible[f"fold{k}_max_drawdown_pct"] > -22.0)
            & (eligible[f"fold{k}_cycles"] >= 6)
        ]

    pool = eligible if not eligible.empty else grid
    chosen_row = pool.iloc[0].to_dict()
    chosen = config_from_row(chosen_row)
    result = cache[chosen]

    development = []
    for k, fold in enumerate(folds, start=1):
        development.append({
            "fold": k,
            "start": fold[0].isoformat(),
            "end": fold[1].isoformat(),
            **{name: chosen_row[f"fold{k}_{name}"] for name in (
                "return_pct", "cagr_pct", "max_drawdown_pct",
                "sharpe", "cycles", "win_rate_pct", "profit_factor",
            )},
            "score": chosen_row[f"fold{k}_score"],
        })

    report = {
        "config": asdict(chosen),
        "development_folds": development,
        "historical_diagnostic": segment_metrics(result, *diagnostic),
        "full": segment_metrics(result, eval_dates[0], eval_dates[-1]),
        "grid_size": int(len(grid)),
        "eligible_development_count": int(len(eligible)),
        "robust_score": float(chosen_row["robust_score"]),
        "worst_fold_score": float(chosen_row["worst_fold_score"]),
        "diagnostic_range": [diagnostic[0].isoformat(), diagnostic[1].isoformat()],
    }
    return grid, grid.head(30).copy(), report, result


def benchmark_spy(matrices: MarketMatrices, trade_start_index: int) -> dict:
    j = matrices.symbols.index("SPY")
    dates = matrices.dates[trade_start_index:]
    op = matrices.opens[trade_start_index:, j]
    cp = matrices.valuation_closes[trade_start_index:, j]
    valid = np.flatnonzero(np.isfinite(op) & (op > 0) & np.isfinite(cp) & (cp > 0))
    if len(valid) < 2:
        return {}
    a, b = int(valid[0]), int(valid[-1])
    start_price = float(op[a])
    values = INITIAL_EQUITY * cp[a:b + 1] / start_price
    curve = pd.Series(values)
    total = float(values[-1] / INITIAL_EQUITY - 1.0)
    dd = float((curve / curve.cummax() - 1.0).min())
    days = max(1, (dates[b] - dates[a]).days)
    years = days / 365.25
    cagr = (values[-1] / INITIAL_EQUITY) ** (1.0 / years) - 1.0
    ret = curve.pct_change().dropna()
    sharpe = float(ret.mean() / ret.std(ddof=1) * math.sqrt(252.0)) if len(ret) > 2 and ret.std(ddof=1) > 0 else 0.0
    return {
        "return_pct": total * 100.0,
        "cagr_pct": cagr * 100.0,
        "max_drawdown_pct": dd * 100.0,
        "sharpe": sharpe,
    }


def cost_stress(matrices: MarketMatrices, config: EquityConfig, trade_start_index: int) -> List[dict]:
    out = []
    start = matrices.dates[trade_start_index]
    end = matrices.dates[-1]
    for mult in (1.0, 2.0, 4.0):
        c = EquityConfig(
            momentum_days=config.momentum_days,
            sma_days=config.sma_days,
            rebalance_days=config.rebalance_days,
            top_k=config.top_k,
            weighting=config.weighting,
            risk_profile=config.risk_profile,
            vol_target_pct=config.vol_target_pct,
            cost_bps=config.cost_bps * mult,
            max_liquid_assets=config.max_liquid_assets,
        )
        r = run_backtest(matrices, c, trade_start_index=trade_start_index)
        out.append({"cost_multiplier": mult, "cost_bps": c.cost_bps, **segment_metrics(r, start, end)})
    return out


def regime_breakdown(result: BacktestResult) -> List[dict]:
    eq = result.equity.copy()
    if eq.empty:
        return []
    eq["daily_return"] = pd.to_numeric(eq["equity"]).pct_change().fillna(0.0)
    out = []
    for regime in ("up", "neutral", "down"):
        r = eq.loc[eq["spy_regime"] == regime, "daily_return"]
        compounded = float((1.0 + r).prod() - 1.0) if len(r) else 0.0
        out.append({"regime": regime, "sessions": int(len(r)), "compounded_return_pct": compounded * 100.0})
    return out


def candidate_status(report: Mapping, stress: Sequence[Mapping], benchmark: Mapping) -> str:
    folds = report["development_folds"]
    diag = report["historical_diagnostic"]
    full = report["full"]
    dev_ok = all(
        float(f["return_pct"]) > 0
        and float(f["profit_factor"]) > 1.0
        and float(f["max_drawdown_pct"]) > -22.0
        and int(f["cycles"]) >= 6
        for f in folds
    )
    diag_ok = (
        float(diag["return_pct"]) > 0
        and float(diag["profit_factor"]) > 1.0
        and float(diag["max_drawdown_pct"]) > -22.0
    )
    full_ok = (
        float(full["cagr_pct"]) > 7.0
        and float(full["profit_factor"]) > 1.10
        and float(full["max_drawdown_pct"]) > -22.0
    )
    double = next((x for x in stress if float(x["cost_multiplier"]) == 2.0), None)
    quadruple = next((x for x in stress if float(x["cost_multiplier"]) == 4.0), None)
    stress_ok = bool(
        double and quadruple
        and float(double["return_pct"]) > 0
        and float(quadruple["return_pct"]) > 0
        and float(double["max_drawdown_pct"]) > -25.0
    )
    risk_improvement = True
    if benchmark:
        risk_improvement = float(full["max_drawdown_pct"]) > float(benchmark["max_drawdown_pct"])
    return (
        "HISTORICAL_PASS_REQUIRES_PAPER_FORWARD"
        if dev_ok and diag_ok and full_ok and stress_ok and risk_improvement
        else "REJECT_OR_RESEARCH_FURTHER"
    )


def save_pack(
    output_dir: Path,
    grid: pd.DataFrame,
    shortlist: pd.DataFrame,
    report: dict,
    result: BacktestResult,
    benchmark: dict,
    stress: List[dict],
    regimes: List[dict],
    symbols: Sequence[str],
    status: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    result.equity.to_csv(output_dir / "selected_equity_curve.csv", index=False)
    result.cycles.to_csv(output_dir / "selected_active_cycles.csv", index=False)
    grid.to_csv(output_dir / "grid_all_288.csv", index=False)
    shortlist.to_csv(output_dir / "shortlist_top30.csv", index=False)
    pd.DataFrame(stress).to_csv(output_dir / "cost_stress.csv", index=False)
    pd.DataFrame(regimes).to_csv(output_dir / "regime_breakdown.csv", index=False)

    summary = {
        "strategy": "EQUITY_ROTATION_V27_RESEARCH",
        "status": status,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "orders_placed": False,
        "initial_equity": INITIAL_EQUITY,
        "universe_requested": list(UNIVERSE),
        "universe_used": list(symbols),
        "group_map": {s: GROUPS.get(s, s) for s in symbols},
        "selected": report,
        "benchmark_spy": benchmark,
        "cost_stress": stress,
        "regime_breakdown": regimes,
        "methodology": {
            "signal_execution": "prior completed daily bar -> next daily open",
            "market_data": "Alpaca IEX 1Day bars, adjustment=all",
            "risk_on_confirmation_days": RISK_ON_CONFIRM_DAYS,
            "minimum_risk_on_breadth": MIN_RISK_ON_BREADTH,
            "category_cap": "maximum one selected ETF per correlation/category group",
            "defensive_set": sorted(DEFENSIVE_SYMBOLS),
            "base_one_way_cost_bps": BASE_COST_BPS,
            "selection": "4 chronological development folds; worst-fold robustness",
            "historical_diagnostic": "final 20% is not used in config selection",
            "limitations": [
                "hand-selected current ETF universe can create selection/survivorship bias",
                "IEX is not consolidated SIP market data",
                "historical diagnostic is not true future data after review",
            ],
        },
    }
    (output_dir / "equity_rotation_v27_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )

    readme = f"""# Equity Rotation V27 Research Pack

Status: **{status}**

This is research only. No orders were placed.

V27 is separate from frozen V26. It tests an expanded ETF universe using
next-open execution, diversification caps, SPY regime control, volatility
scaling, four chronological development folds, a final historical diagnostic,
SPY benchmark comparison, and 1x/2x/4x cost stress.

A historical PASS is not approval for live trading. It only means the candidate
may advance to paper-forward validation.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")

    zip_path = output_dir.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(output_dir.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(output_dir.parent))
    return zip_path


def parse_args():
    p = argparse.ArgumentParser(description="Research expanded ETF/equities rotation V27.")
    p.add_argument("--start", default=DEFAULT_START)
    p.add_argument("--end", default=None)
    p.add_argument("--output", default=PACK_DIRNAME)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    print("EQUITY ROTATION V27 RESEARCH — NO ORDERS WILL BE PLACED")
    print(f"Requested ETF universe: {len(UNIVERSE)} symbols")
    api = AlpacaEquityData()
    active = api.validate_assets(UNIVERSE)
    print(f"Active/tradable ETFs found: {len(active)}/{len(UNIVERSE)}")
    if "SPY" not in active:
        raise SystemExit("SPY is unavailable; cannot run V27.")

    frames = api.daily_bars(active, start=args.start, end=args.end)
    usable = {s: f for s, f in frames.items() if len(f) >= 300}
    print(f"Usable ETF histories: {len(usable)}")
    if len(usable) < 20:
        raise SystemExit("Fewer than 20 ETFs have usable history; refusing to optimize a narrow universe.")

    matrices = build_matrices(usable)
    spy_j = matrices.symbols.index("SPY")
    valid_spy = np.flatnonzero(np.isfinite(matrices.sma[SPY_REGIME_SMA_DAYS][:, spy_j]))
    if not len(valid_spy):
        raise SystemExit("Not enough SPY history for V27 warmup.")
    trade_start = max(WARMUP_DAYS, int(valid_spy[0]) + 1)
    if len(matrices.dates) - trade_start < 1000:
        raise SystemExit("Not enough post-warmup sessions for four-fold V27 evaluation.")

    grid, shortlist, report, result = evaluate_grid(matrices, trade_start)
    selected = config_from_row(report["config"])
    benchmark = benchmark_spy(matrices, trade_start)
    stress = cost_stress(matrices, selected, trade_start)
    regimes = regime_breakdown(result)
    status = candidate_status(report, stress, benchmark)

    output_dir = Path(args.output)
    zip_path = save_pack(
        output_dir, grid, shortlist, report, result, benchmark, stress,
        regimes, matrices.symbols, status,
    )

    print("\n=== EQUITY ROTATION V27 RESULT ===")
    print(f"Status: {status}")
    print(f"Usable ETFs: {len(matrices.symbols)}")
    print(f"Selected config: {report['config']}")
    for f in report["development_folds"]:
        print(
            f"DEV FOLD {f['fold']}: return={f['return_pct']:.2f}% "
            f"PF={f['profit_factor']:.2f} DD={f['max_drawdown_pct']:.2f}%"
        )
    d = report["historical_diagnostic"]
    full = report["full"]
    print(
        f"HISTORICAL DIAGNOSTIC: return={d['return_pct']:.2f}% "
        f"PF={d['profit_factor']:.2f} DD={d['max_drawdown_pct']:.2f}%"
    )
    print(
        f"FULL: return={full['return_pct']:.2f}% CAGR={full['cagr_pct']:.2f}% "
        f"PF={full['profit_factor']:.2f} DD={full['max_drawdown_pct']:.2f}%"
    )
    if benchmark:
        print(
            f"SPY B&H: return={benchmark['return_pct']:.2f}% "
            f"CAGR={benchmark['cagr_pct']:.2f}% DD={benchmark['max_drawdown_pct']:.2f}%"
        )
    print(f"Research pack: {zip_path}")
    print("NO ORDERS WERE PLACED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
