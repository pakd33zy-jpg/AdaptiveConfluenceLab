#!/usr/bin/env python3
"""
Crypto Rotation Research Harness for AdaptiveConfluenceLab.

Purpose
-------
Research a crypto-specific, LONG/CASH momentum-rotation strategy using Alpaca
historical crypto bars. This is separate from the frozen ETF V26 forward test.

Design goals
------------
- Discover currently active/tradable Alpaca USD crypto pairs dynamically.
- Exclude stablecoin-like bases.
- Use only information available before each simulated execution.
- Rebalance at the next UTC daily bar open.
- Test a compact but meaningful parameter grid.
- Use chronological train / validation / final holdout splits.
- Stress-test transaction costs.
- Report behavior in BTC up/down/sideways regimes.
- Never place orders. This script is RESEARCH ONLY.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
import time
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests

PAPER_BASE = "https://paper-api.alpaca.markets"
DATA_BASE = "https://data.alpaca.markets"

DEFAULT_START = "2021-01-01T00:00:00Z"
DEFAULT_END = None

# Avoid stablecoin / cash-like bases that can distort a momentum ranking.
EXCLUDED_BASES = {
    "USD", "USDC", "USDT", "DAI", "PYUSD", "TUSD", "FDUSD", "USDP",
}

# The active universe can be larger; this is only the maximum number of liquid
# assets admitted to the momentum ranking on each rebalance date.
DEFAULT_MAX_LIQUID_ASSETS = 15
DEFAULT_MIN_MEDIAN_DOLLAR_VOLUME = 250_000.0
LIQUIDITY_LOOKBACK = 30

INITIAL_EQUITY = 100_000.0
BASE_COST_BPS = 35.0

MOMENTUM_GRID = (21, 42, 63, 90)
SMA_GRID = (100, 150, 200)
REBALANCE_GRID = (3, 5, 7)
TOP_K_GRID = (2, 3, 4)
WEIGHTING_GRID = ("equal", "rank")

MAX_SMA = max(SMA_GRID)
MAX_MOM = max(MOMENTUM_GRID)
WARMUP_DAYS = max(MAX_SMA, MAX_MOM + 1, LIQUIDITY_LOOKBACK + 1)

PACK_DIRNAME = "CryptoRotation_research_pack"


@dataclass(frozen=True)
class CryptoConfig:
    momentum_days: int
    sma_days: int
    rebalance_days: int
    top_k: int
    weighting: str
    cost_bps: float = BASE_COST_BPS
    max_liquid_assets: int = DEFAULT_MAX_LIQUID_ASSETS
    min_median_dollar_volume: float = DEFAULT_MIN_MEDIAN_DOLLAR_VOLUME


def credentials() -> Tuple[str, str]:
    key = (
        os.getenv("ALPACA_PAPER_API_KEY")
        or os.getenv("ALPACA_API_KEY")
    )
    secret = (
        os.getenv("ALPACA_PAPER_SECRET_KEY")
        or os.getenv("ALPACA_SECRET_KEY")
    )
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
        self.headers = {
            "APCA-API-KEY-ID": key,
            "APCA-API-SECRET-KEY": secret,
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def _get(
        self,
        url: str,
        *,
        params: Optional[dict] = None,
        retries: int = 5,
    ):
        delay = 1.0
        last_error = None
        for attempt in range(retries):
            try:
                response = self.session.get(
                    url,
                    params=params,
                    timeout=60,
                )
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
                        f"Alpaca request failed {response.status_code}: "
                        f"{response.text[:1000]}"
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
        pairs: List[str] = []
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
            batch = list(symbols[batch_start : batch_start + batch_size])
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
                    if symbol not in output:
                        output[symbol] = []
                    if isinstance(rows, list):
                        output[symbol].extend(rows)

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
            frame["date"] = pd.to_datetime(
                frame["t"], utc=True, errors="coerce"
            ).dt.floor("D")
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


@dataclass
class MarketMatrices:
    dates: pd.DatetimeIndex
    opens: pd.DataFrame
    closes: pd.DataFrame
    volumes: pd.DataFrame
    dollar_volume_median: pd.DataFrame
    momentum: Dict[int, pd.DataFrame]
    sma: Dict[int, pd.DataFrame]


def build_matrices(frames: Mapping[str, pd.DataFrame]) -> MarketMatrices:
    if not frames:
        raise ValueError("No crypto data frames were supplied.")

    def pivot(column: str) -> pd.DataFrame:
        series = []
        for symbol, frame in frames.items():
            s = frame.set_index("date")[column].rename(symbol)
            series.append(s)
        out = pd.concat(series, axis=1).sort_index()
        out.index = pd.DatetimeIndex(out.index)
        return out

    opens = pivot("open")
    closes = pivot("close")
    volumes = pivot("volume").fillna(0.0)
    dollar_volume = closes * volumes

    liq = (
        dollar_volume
        .rolling(LIQUIDITY_LOOKBACK, min_periods=max(10, LIQUIDITY_LOOKBACK // 2))
        .median()
    )

    momentum = {
        lookback: closes / closes.shift(lookback) - 1.0
        for lookback in MOMENTUM_GRID
    }
    sma = {
        lookback: closes.rolling(lookback, min_periods=lookback).mean()
        for lookback in SMA_GRID
    }

    return MarketMatrices(
        dates=pd.DatetimeIndex(closes.index),
        opens=opens,
        closes=closes,
        volumes=volumes,
        dollar_volume_median=liq,
        momentum=momentum,
        sma=sma,
    )


def target_weights(
    matrices: MarketMatrices,
    signal_index: int,
    config: CryptoConfig,
) -> Dict[str, float]:
    if signal_index < 0:
        return {}

    date = matrices.dates[signal_index]
    close_row = matrices.closes.loc[date]
    mom_row = matrices.momentum[config.momentum_days].loc[date]
    sma_row = matrices.sma[config.sma_days].loc[date]
    liq_row = matrices.dollar_volume_median.loc[date]

    candidates = []
    for symbol in matrices.closes.columns:
        close = float(close_row.get(symbol, np.nan))
        mom = float(mom_row.get(symbol, np.nan))
        avg = float(sma_row.get(symbol, np.nan))
        liq = float(liq_row.get(symbol, np.nan))

        if not all(math.isfinite(x) for x in (close, mom, avg, liq)):
            continue
        if liq < config.min_median_dollar_volume:
            continue
        candidates.append((symbol, liq, mom, close, avg))

    if not candidates:
        return {}

    # Liquidity gate is determined from information available on the signal date.
    candidates.sort(key=lambda row: row[1], reverse=True)
    candidates = candidates[: config.max_liquid_assets]

    eligible = [
        row
        for row in candidates
        if row[2] > 0.0 and row[3] > row[4]
    ]
    eligible.sort(key=lambda row: row[2], reverse=True)
    selected = eligible[: config.top_k]

    if not selected:
        return {}

    if config.weighting == "equal":
        weight = 1.0 / len(selected)
        return {symbol: weight for symbol, *_ in selected}

    if config.weighting == "rank":
        scores = list(range(len(selected), 0, -1))
        total = float(sum(scores))
        return {
            selected[i][0]: scores[i] / total
            for i in range(len(selected))
        }

    raise ValueError(f"Unknown weighting: {config.weighting}")


@dataclass
class BacktestResult:
    equity: pd.DataFrame
    cycles: pd.DataFrame
    turnover: float
    rebalance_count: int


def _price(
    frame: pd.DataFrame,
    date: pd.Timestamp,
    symbol: str,
    fallback: float,
) -> float:
    try:
        value = float(frame.at[date, symbol])
    except Exception:
        value = float("nan")
    return value if math.isfinite(value) and value > 0 else fallback


def run_backtest(
    matrices: MarketMatrices,
    config: CryptoConfig,
    *,
    initial_equity: float = INITIAL_EQUITY,
) -> BacktestResult:
    dates = matrices.dates
    if len(dates) < 3:
        raise ValueError("Not enough dates to backtest.")

    cash = float(initial_equity)
    qty: Dict[str, float] = {}
    last_close: Dict[str, float] = {}
    rows: List[dict] = []
    cycle_rows: List[dict] = []
    last_rebalance_i: Optional[int] = None
    cycle_start_equity: Optional[float] = None
    cycle_start_date: Optional[pd.Timestamp] = None
    turnover = 0.0
    rebalance_count = 0

    cost_rate = float(config.cost_bps) / 10000.0

    for i, date in enumerate(dates):
        # Use a valid current open for valuation. If a symbol has a missing daily bar,
        # carry the most recent close only for mark-to-market; it cannot be newly bought
        # without a valid open.
        open_prices: Dict[str, float] = {}
        close_prices: Dict[str, float] = {}

        for symbol in matrices.closes.columns:
            fallback = last_close.get(symbol, 0.0)
            op = _price(matrices.opens, date, symbol, fallback)
            cp = _price(matrices.closes, date, symbol, op)
            if op > 0:
                open_prices[symbol] = op
            if cp > 0:
                close_prices[symbol] = cp

        pretrade_equity = cash + sum(
            amount * open_prices.get(symbol, last_close.get(symbol, 0.0))
            for symbol, amount in qty.items()
        )

        due = (
            i > 0
            and (
                last_rebalance_i is None
                or (i - last_rebalance_i) >= config.rebalance_days
            )
        )

        traded_notional = 0.0
        targets: Dict[str, float] = {}

        if due:
            targets = target_weights(matrices, i - 1, config)

            if cycle_start_equity is not None and cycle_start_equity > 0:
                cycle_return = pretrade_equity / cycle_start_equity - 1.0
                cycle_rows.append(
                    {
                        "start_date": cycle_start_date,
                        "end_date": date,
                        "return": cycle_return,
                        "pnl": pretrade_equity - cycle_start_equity,
                    }
                )

            current_values = {
                symbol: amount
                * open_prices.get(symbol, last_close.get(symbol, 0.0))
                for symbol, amount in qty.items()
            }

            desired_values = {
                symbol: pretrade_equity * weight
                for symbol, weight in targets.items()
            }

            # Sell reductions first.
            for symbol in sorted(set(current_values) | set(desired_values)):
                current = current_values.get(symbol, 0.0)
                desired = desired_values.get(symbol, 0.0)
                delta = desired - current
                if delta >= -1e-9:
                    continue
                price = open_prices.get(symbol, last_close.get(symbol, 0.0))
                if price <= 0:
                    continue
                sell_value = min(current, -delta)
                sell_qty = min(qty.get(symbol, 0.0), sell_value / price)
                actual_value = sell_qty * price
                fee = actual_value * cost_rate
                qty[symbol] = max(0.0, qty.get(symbol, 0.0) - sell_qty)
                cash += actual_value - fee
                traded_notional += actual_value
                if qty[symbol] <= 1e-12:
                    qty.pop(symbol, None)

            # Buy additions after sells. Cap each buy by available cash including fees.
            current_values = {
                symbol: qty.get(symbol, 0.0)
                * open_prices.get(symbol, last_close.get(symbol, 0.0))
                for symbol in set(qty) | set(desired_values)
            }

            for symbol, desired in sorted(
                desired_values.items(),
                key=lambda item: item[1],
                reverse=True,
            ):
                price = open_prices.get(symbol, 0.0)
                if price <= 0:
                    continue
                current = current_values.get(symbol, 0.0)
                delta = desired - current
                if delta <= 1e-9:
                    continue
                max_buy = cash / (1.0 + cost_rate) if cost_rate >= 0 else cash
                buy_value = max(0.0, min(delta, max_buy))
                if buy_value <= 0:
                    continue
                fee = buy_value * cost_rate
                buy_qty = buy_value / price
                qty[symbol] = qty.get(symbol, 0.0) + buy_qty
                cash -= buy_value + fee
                traded_notional += buy_value

            posttrade_equity = cash + sum(
                amount * open_prices.get(symbol, last_close.get(symbol, 0.0))
                for symbol, amount in qty.items()
            )
            cycle_start_equity = posttrade_equity
            cycle_start_date = date
            last_rebalance_i = i
            rebalance_count += 1

        close_equity = cash + sum(
            amount * close_prices.get(symbol, open_prices.get(symbol, last_close.get(symbol, 0.0)))
            for symbol, amount in qty.items()
        )

        rows.append(
            {
                "date": date,
                "equity": close_equity,
                "cash": cash,
                "positions": len(qty),
                "rebalanced": bool(due),
                "turnover_notional": traded_notional,
            }
        )
        turnover += traded_notional

        for symbol, price in close_prices.items():
            if price > 0:
                last_close[symbol] = price

    if cycle_start_equity is not None and cycle_start_equity > 0 and rows:
        final_equity = float(rows[-1]["equity"])
        final_date = pd.Timestamp(rows[-1]["date"])
        cycle_rows.append(
            {
                "start_date": cycle_start_date,
                "end_date": final_date,
                "return": final_equity / cycle_start_equity - 1.0,
                "pnl": final_equity - cycle_start_equity,
            }
        )

    return BacktestResult(
        equity=pd.DataFrame(rows),
        cycles=pd.DataFrame(cycle_rows),
        turnover=float(turnover),
        rebalance_count=int(rebalance_count),
    )


def segment_metrics(
    result: BacktestResult,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict:
    eq = result.equity.copy()
    eq["date"] = pd.to_datetime(eq["date"], utc=True)
    mask = (eq["date"] >= start) & (eq["date"] <= end)
    seg = eq.loc[mask].copy()

    if len(seg) < 2:
        return {
            "return_pct": 0.0,
            "cagr_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "sharpe": 0.0,
            "cycles": 0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
        }

    first = float(seg["equity"].iloc[0])
    last = float(seg["equity"].iloc[-1])
    total_return = last / first - 1.0 if first > 0 else 0.0

    days = max(
        1.0,
        (seg["date"].iloc[-1] - seg["date"].iloc[0]).total_seconds() / 86400.0,
    )
    if first > 0 and last > 0:
        cagr = (last / first) ** (365.0 / days) - 1.0
    else:
        cagr = -1.0

    curve = seg["equity"].astype(float)
    running_max = curve.cummax()
    drawdown = curve / running_max - 1.0
    max_dd = float(drawdown.min())

    daily = curve.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if len(daily) >= 3 and float(daily.std(ddof=1)) > 0:
        sharpe = float(daily.mean() / daily.std(ddof=1) * math.sqrt(365.0))
    else:
        sharpe = 0.0

    cycles = result.cycles.copy()
    if not cycles.empty:
        cycles["end_date"] = pd.to_datetime(cycles["end_date"], utc=True)
        cseg = cycles[
            (cycles["end_date"] >= start)
            & (cycles["end_date"] <= end)
        ].copy()
    else:
        cseg = cycles

    if cseg.empty:
        win_rate = 0.0
        pf = 0.0
        cycle_count = 0
    else:
        returns = pd.to_numeric(cseg["return"], errors="coerce").dropna()
        cycle_count = int(len(returns))
        win_rate = float((returns > 0).mean()) if cycle_count else 0.0
        positive = float(returns[returns > 0].sum())
        negative = float(-returns[returns < 0].sum())
        if negative > 0:
            pf = positive / negative
        elif positive > 0:
            pf = float("inf")
        else:
            pf = 0.0

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
    ret = float(metrics.get("cagr_pct", 0.0)) / 100.0
    dd = abs(float(metrics.get("max_drawdown_pct", 0.0))) / 100.0
    win = float(metrics.get("win_rate_pct", 0.0)) / 100.0
    pf = float(metrics.get("profit_factor", 0.0))
    cycles = int(metrics.get("cycles", 0))

    if cycles < 8:
        return -999.0

    pf_term = 0.0
    if math.isfinite(pf) and pf > 0:
        pf_term = max(-1.0, min(1.25, math.log(max(pf, 1e-9))))
    elif pf == float("inf"):
        pf_term = 1.25

    # Return is the primary objective, but the score penalizes deep drawdowns and
    # rewards robust cycle wins/PF so one lucky moonshot does not dominate.
    return (
        2.0 * ret
        + 0.45 * (win - 0.50)
        + 0.50 * pf_term
        - 1.25 * dd
    )


def chronological_splits(dates: pd.DatetimeIndex) -> Tuple[Tuple[pd.Timestamp, pd.Timestamp], ...]:
    if len(dates) < 300:
        raise ValueError("Need at least 300 evaluation dates for chronological validation.")

    n = len(dates)
    train_end_i = max(1, int(n * 0.60) - 1)
    val_end_i = max(train_end_i + 1, int(n * 0.80) - 1)

    return (
        (dates[0], dates[train_end_i]),
        (dates[train_end_i + 1], dates[val_end_i]),
        (dates[val_end_i + 1], dates[-1]),
    )


def evaluate_grid(
    matrices: MarketMatrices,
    eval_dates: pd.DatetimeIndex,
) -> Tuple[pd.DataFrame, List[dict]]:
    train, validation, holdout = chronological_splits(eval_dates)

    configs = [
        CryptoConfig(
            momentum_days=mom,
            sma_days=sma,
            rebalance_days=reb,
            top_k=top_k,
            weighting=weighting,
        )
        for mom, sma, reb, top_k, weighting in itertools.product(
            MOMENTUM_GRID,
            SMA_GRID,
            REBALANCE_GRID,
            TOP_K_GRID,
            WEIGHTING_GRID,
        )
    ]

    train_rows: List[dict] = []
    cache: Dict[CryptoConfig, BacktestResult] = {}

    print(f"Testing {len(configs)} crypto rotation configurations...")

    for index, config in enumerate(configs, start=1):
        result = run_backtest(matrices, config)
        cache[config] = result
        metrics = segment_metrics(result, *train)
        row = {
            **asdict(config),
            **{f"train_{k}": v for k, v in metrics.items()},
            "train_score": metric_score(metrics),
        }
        train_rows.append(row)
        if index % 24 == 0 or index == len(configs):
            print(f"  grid progress: {index}/{len(configs)}")

    train_df = pd.DataFrame(train_rows).sort_values(
        ["train_score", "train_return_pct"],
        ascending=False,
    ).reset_index(drop=True)

    shortlist = train_df.head(18).copy()
    validation_rows: List[dict] = []

    for _, row in shortlist.iterrows():
        config = CryptoConfig(
            momentum_days=int(row["momentum_days"]),
            sma_days=int(row["sma_days"]),
            rebalance_days=int(row["rebalance_days"]),
            top_k=int(row["top_k"]),
            weighting=str(row["weighting"]),
            cost_bps=float(row["cost_bps"]),
            max_liquid_assets=int(row["max_liquid_assets"]),
            min_median_dollar_volume=float(row["min_median_dollar_volume"]),
        )
        result = cache[config]
        val_metrics = segment_metrics(result, *validation)
        validation_rows.append(
            {
                **asdict(config),
                **{
                    f"train_{k}": row[f"train_{k}"]
                    for k in (
                        "return_pct",
                        "cagr_pct",
                        "max_drawdown_pct",
                        "sharpe",
                        "cycles",
                        "win_rate_pct",
                        "profit_factor",
                    )
                },
                "train_score": float(row["train_score"]),
                **{f"validation_{k}": v for k, v in val_metrics.items()},
                "validation_score": metric_score(val_metrics),
            }
        )

    val_df = pd.DataFrame(validation_rows)

    # Prefer positive validation expectancy proxies before maximizing validation score.
    valid = val_df[
        (val_df["validation_return_pct"] > 0)
        & (val_df["validation_profit_factor"] > 1.0)
        & (val_df["validation_cycles"] >= 8)
    ].copy()

    selection_pool = valid if not valid.empty else val_df
    selection_pool = selection_pool.sort_values(
        ["validation_score", "train_score"],
        ascending=False,
    ).reset_index(drop=True)

    chosen_row = selection_pool.iloc[0].to_dict()
    chosen = CryptoConfig(
        momentum_days=int(chosen_row["momentum_days"]),
        sma_days=int(chosen_row["sma_days"]),
        rebalance_days=int(chosen_row["rebalance_days"]),
        top_k=int(chosen_row["top_k"]),
        weighting=str(chosen_row["weighting"]),
        cost_bps=float(chosen_row["cost_bps"]),
        max_liquid_assets=int(chosen_row["max_liquid_assets"]),
        min_median_dollar_volume=float(chosen_row["min_median_dollar_volume"]),
    )

    # This is the first and only holdout inspection for the selected candidate.
    chosen_result = cache[chosen]
    hold_metrics = segment_metrics(chosen_result, *holdout)
    full_metrics = segment_metrics(chosen_result, eval_dates[0], eval_dates[-1])

    chosen_report = {
        "config": asdict(chosen),
        "train": {
            k.replace("train_", ""): chosen_row[k]
            for k in chosen_row
            if k.startswith("train_") and k != "train_score"
        },
        "validation": {
            k.replace("validation_", ""): chosen_row[k]
            for k in chosen_row
            if k.startswith("validation_") and k != "validation_score"
        },
        "holdout": hold_metrics,
        "full": full_metrics,
        "train_score": float(chosen_row["train_score"]),
        "validation_score": float(chosen_row["validation_score"]),
        "validation_positive_shortlist_count": int(
            (
                (val_df["validation_return_pct"] > 0)
                & (val_df["validation_profit_factor"] > 1.0)
            ).sum()
        ),
        "shortlist_size": int(len(val_df)),
        "splits": {
            "train": [train[0].isoformat(), train[1].isoformat()],
            "validation": [validation[0].isoformat(), validation[1].isoformat()],
            "holdout": [holdout[0].isoformat(), holdout[1].isoformat()],
        },
    }

    return train_df, [chosen_report, val_df, chosen_result]


def benchmark_btc(
    matrices: MarketMatrices,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict:
    symbol = "BTC/USD"
    if symbol not in matrices.closes.columns:
        return {"available": False}

    s = matrices.closes[symbol].loc[start:end].dropna()
    if len(s) < 2:
        return {"available": False}

    first = float(s.iloc[0])
    last = float(s.iloc[-1])
    ret = last / first - 1.0
    running = s.cummax()
    dd = s / running - 1.0
    days = max(1.0, (s.index[-1] - s.index[0]).total_seconds() / 86400.0)
    cagr = (last / first) ** (365.0 / days) - 1.0

    return {
        "available": True,
        "return_pct": ret * 100.0,
        "cagr_pct": cagr * 100.0,
        "max_drawdown_pct": float(dd.min()) * 100.0,
    }


def regime_analysis(
    matrices: MarketMatrices,
    result: BacktestResult,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict:
    if "BTC/USD" not in matrices.closes.columns:
        return {}

    btc = matrices.closes["BTC/USD"].copy()
    btc_sma = btc.rolling(100, min_periods=100).mean()
    btc_mom = btc / btc.shift(30) - 1.0

    eq = result.equity.set_index("date")["equity"].astype(float)
    eq.index = pd.DatetimeIndex(eq.index)
    strategy_ret = eq.pct_change().fillna(0.0)

    out = {}
    for regime_name in ("up", "down", "sideways"):
        returns = []

        for date in strategy_ret.loc[start:end].index:
            loc = btc.index.get_indexer([date])[0]
            if loc <= 0:
                continue
            signal_date = btc.index[loc - 1]
            close = float(btc.loc[signal_date]) if pd.notna(btc.loc[signal_date]) else np.nan
            sma = float(btc_sma.loc[signal_date]) if pd.notna(btc_sma.loc[signal_date]) else np.nan
            mom = float(btc_mom.loc[signal_date]) if pd.notna(btc_mom.loc[signal_date]) else np.nan

            if not all(math.isfinite(x) for x in (close, sma, mom)):
                regime = "sideways"
            elif close > sma and mom > 0.05:
                regime = "up"
            elif close < sma and mom < -0.05:
                regime = "down"
            else:
                regime = "sideways"

            if regime == regime_name:
                returns.append(float(strategy_ret.loc[date]))

        if returns:
            compounded = float(np.prod([1.0 + r for r in returns]) - 1.0)
            out[regime_name] = {
                "days": len(returns),
                "return_pct": compounded * 100.0,
                "positive_day_pct": float(np.mean(np.array(returns) > 0)) * 100.0,
            }
        else:
            out[regime_name] = {
                "days": 0,
                "return_pct": 0.0,
                "positive_day_pct": 0.0,
            }

    return out


def stress_costs(
    matrices: MarketMatrices,
    config: CryptoConfig,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> List[dict]:
    rows = []
    for multiplier in (1.0, 2.0, 3.0):
        stressed = CryptoConfig(
            momentum_days=config.momentum_days,
            sma_days=config.sma_days,
            rebalance_days=config.rebalance_days,
            top_k=config.top_k,
            weighting=config.weighting,
            cost_bps=config.cost_bps * multiplier,
            max_liquid_assets=config.max_liquid_assets,
            min_median_dollar_volume=config.min_median_dollar_volume,
        )
        result = run_backtest(matrices, stressed)
        metrics = segment_metrics(result, start, end)
        rows.append(
            {
                "cost_multiplier": multiplier,
                "one_way_cost_bps": stressed.cost_bps,
                **metrics,
            }
        )
    return rows


def candidate_status(report: Mapping, stress: Sequence[Mapping]) -> str:
    train = report["train"]
    validation = report["validation"]
    holdout = report["holdout"]
    full = report["full"]

    def pass_segment(seg: Mapping, min_cycles: int) -> bool:
        return (
            float(seg.get("return_pct", 0.0)) > 0
            and float(seg.get("profit_factor", 0.0)) > 1.0
            and int(seg.get("cycles", 0)) >= min_cycles
        )

    base_ok = (
        pass_segment(train, 12)
        and pass_segment(validation, 8)
        and pass_segment(holdout, 8)
        and float(full.get("max_drawdown_pct", -100.0)) > -35.0
    )

    double = next(
        (row for row in stress if float(row["cost_multiplier"]) == 2.0),
        None,
    )
    stress_ok = bool(
        double
        and float(double.get("return_pct", 0.0)) > 0
        and float(double.get("max_drawdown_pct", -100.0)) > -40.0
    )

    if base_ok and stress_ok:
        return "PROVISIONAL_PASS_PAPER_FORWARD_CANDIDATE"
    return "REJECT_OR_RESEARCH_FURTHER"


def save_pack(
    out_dir: Path,
    *,
    summary: dict,
    universe_df: pd.DataFrame,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    result: BacktestResult,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "crypto_rotation_summary.json").write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )
    universe_df.to_csv(out_dir / "universe.csv", index=False)
    train_df.to_csv(out_dir / "candidate_grid_train.csv", index=False)
    validation_df.to_csv(out_dir / "shortlist_validation.csv", index=False)
    result.equity.to_csv(out_dir / "selected_equity.csv", index=False)
    result.cycles.to_csv(out_dir / "selected_cycles.csv", index=False)

    readme = f"""# Crypto Rotation Research Pack

Status: {summary['status']}

This pack is research-only. It does not place Alpaca orders.

Selected config:
{json.dumps(summary['selected']['config'], indent=2)}

Important methodology:
- Current active Alpaca USD crypto pairs are discovered dynamically.
- Stablecoin-like bases are excluded.
- Liquidity is measured from trailing data only.
- Signals use the prior completed UTC daily bar.
- Simulated execution occurs at the next UTC daily bar open.
- Selection uses train + validation only.
- The final holdout is inspected once for the selected configuration.
- Crypto is long/cash only; no simulated shorting.
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
    parser = argparse.ArgumentParser(
        description="Research a crypto momentum-rotation strategy using Alpaca data."
    )
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(PACK_DIRNAME),
    )
    args = parser.parse_args()

    print("CRYPTO ROTATION RESEARCH — NO ORDERS WILL BE PLACED")
    print("Discovering active Alpaca USD crypto pairs...")

    api = AlpacaCryptoData()
    symbols = api.active_usd_pairs()

    if "BTC/USD" not in symbols:
        # BTC is useful as the regime/benchmark anchor. If Alpaca does not report it,
        # continue without it but make that explicit.
        print("WARNING: BTC/USD was not returned as an active tradable pair.")

    if len(symbols) < 3:
        raise RuntimeError(
            f"Only {len(symbols)} eligible USD crypto pairs were discovered; "
            "not enough for rotation research."
        )

    print(f"Discovered {len(symbols)} active non-stablecoin USD crypto pairs.")
    frames = api.daily_bars(symbols, args.start, args.end)

    if len(frames) < 3:
        raise RuntimeError("Too few crypto pairs returned usable historical bars.")

    universe_rows = []
    for symbol in symbols:
        frame = frames.get(symbol)
        if frame is None or frame.empty:
            universe_rows.append(
                {
                    "symbol": symbol,
                    "bars": 0,
                    "first_date": None,
                    "last_date": None,
                    "recent_median_dollar_volume": 0.0,
                    "usable": False,
                }
            )
            continue

        dv = frame["close"] * frame["volume"]
        universe_rows.append(
            {
                "symbol": symbol,
                "bars": int(len(frame)),
                "first_date": frame["date"].min(),
                "last_date": frame["date"].max(),
                "recent_median_dollar_volume": float(dv.tail(90).median()),
                "usable": bool(len(frame) >= WARMUP_DAYS + 60),
            }
        )

    universe_df = pd.DataFrame(universe_rows).sort_values(
        "recent_median_dollar_volume",
        ascending=False,
    )

    usable_symbols = set(
        universe_df.loc[universe_df["usable"], "symbol"].astype(str)
    )
    usable_frames = {
        symbol: frame
        for symbol, frame in frames.items()
        if symbol in usable_symbols
    }

    if len(usable_frames) < 3:
        raise RuntimeError(
            "Fewer than 3 pairs have enough historical bars after the warmup filter."
        )

    matrices = build_matrices(usable_frames)

    # Evaluation starts only after the maximum warmup. Signals may still admit newer
    # assets later when each asset individually has sufficient trailing history.
    if len(matrices.dates) <= WARMUP_DAYS + 300:
        raise RuntimeError(
            f"Only {len(matrices.dates)} daily dates are available; "
            "need more history for train/validation/holdout research."
        )

    eval_dates = matrices.dates[WARMUP_DAYS:]

    train_df, objects = evaluate_grid(matrices, eval_dates)
    selected_report, validation_df, selected_result = objects
    selected_config = CryptoConfig(**selected_report["config"])

    stress = stress_costs(
        matrices,
        selected_config,
        eval_dates[0],
        eval_dates[-1],
    )
    regimes = regime_analysis(
        matrices,
        selected_result,
        eval_dates[0],
        eval_dates[-1],
    )
    btc = benchmark_btc(
        matrices,
        eval_dates[0],
        eval_dates[-1],
    )

    selected_report["cost_stress"] = stress
    selected_report["regimes"] = regimes

    status = candidate_status(selected_report, stress)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "strategy": "CRYPTO_ROTATION_V1_RESEARCH",
        "research_only": True,
        "orders_placed": False,
        "status": status,
        "data": {
            "requested_start": args.start,
            "requested_end": args.end,
            "active_usd_pairs_discovered": len(symbols),
            "usable_pairs": len(usable_frames),
            "evaluation_start": eval_dates[0].isoformat(),
            "evaluation_end": eval_dates[-1].isoformat(),
            "survivorship_note": (
                "Universe discovery begins from currently active/tradable Alpaca pairs. "
                "Delisted historical pairs are not recovered by this harness."
            ),
        },
        "objective": (
            "Strong total return + frequent winning rebalance cycles + positive profit "
            "factor + controlled drawdown across changing crypto regimes."
        ),
        "selected": selected_report,
        "btc_buy_and_hold_benchmark": btc,
        "methodology": {
            "signal_timing": "prior completed UTC daily bar",
            "execution": "next UTC daily bar open",
            "trade_direction": "long/cash only",
            "grid_size": int(
                len(MOMENTUM_GRID)
                * len(SMA_GRID)
                * len(REBALANCE_GRID)
                * len(TOP_K_GRID)
                * len(WEIGHTING_GRID)
            ),
            "base_one_way_cost_bps": BASE_COST_BPS,
            "liquidity_lookback_days": LIQUIDITY_LOOKBACK,
            "max_liquid_assets_per_rebalance": DEFAULT_MAX_LIQUID_ASSETS,
            "minimum_trailing_median_dollar_volume": DEFAULT_MIN_MEDIAN_DOLLAR_VOLUME,
            "holdout_policy": (
                "Train grid -> validation shortlist -> one final holdout inspection "
                "for the selected configuration."
            ),
        },
    }

    out_dir = args.output_dir.resolve()
    zip_path = save_pack(
        out_dir,
        summary=summary,
        universe_df=universe_df,
        train_df=train_df,
        validation_df=validation_df,
        result=selected_result,
    )

    selected = selected_report
    print("\n=== CRYPTO ROTATION RESULT ===")
    print(f"Status: {status}")
    print(f"Usable pairs: {len(usable_frames)}")
    print(f"Selected config: {selected['config']}")
    print(
        "TRAIN: "
        f"return={float(selected['train']['return_pct']):.2f}% "
        f"win={float(selected['train']['win_rate_pct']):.1f}% "
        f"PF={float(selected['train']['profit_factor']):.2f} "
        f"DD={float(selected['train']['max_drawdown_pct']):.2f}%"
    )
    print(
        "VALIDATION: "
        f"return={float(selected['validation']['return_pct']):.2f}% "
        f"win={float(selected['validation']['win_rate_pct']):.1f}% "
        f"PF={float(selected['validation']['profit_factor']):.2f} "
        f"DD={float(selected['validation']['max_drawdown_pct']):.2f}%"
    )
    print(
        "HOLDOUT: "
        f"return={float(selected['holdout']['return_pct']):.2f}% "
        f"win={float(selected['holdout']['win_rate_pct']):.1f}% "
        f"PF={float(selected['holdout']['profit_factor']):.2f} "
        f"DD={float(selected['holdout']['max_drawdown_pct']):.2f}%"
    )
    print(
        "FULL: "
        f"return={float(selected['full']['return_pct']):.2f}% "
        f"win={float(selected['full']['win_rate_pct']):.1f}% "
        f"PF={float(selected['full']['profit_factor']):.2f} "
        f"DD={float(selected['full']['max_drawdown_pct']):.2f}%"
    )
    print(f"\nResearch pack: {zip_path}")
    print("NO ORDERS WERE PLACED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
