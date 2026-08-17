#!/usr/bin/env python3
from __future__ import annotations

import itertools
import json
import math
import os
import shutil
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

VERSION = "INTRADAY_V2"
OUT_DIR = Path("EquityIntradayV2_research_pack")
OUT_ZIP = Path("EquityIntradayV2_research_pack.zip")
SYMBOLS = ("SPY", "QQQ", "IWM", "AAPL", "NVDA", "MSFT", "AMD", "AMZN", "META", "TSLA", "SOFI", "F", "BAC")
STARTING_CAPITAL = 100.0

# Research only. This file contains no trading/order endpoint and never submits orders.
ALPACA_DATA_BASE_URL = os.getenv("ALPACA_DATA_BASE_URL", "https://data.alpaca.markets")
ALPACA_PAPER_API_KEY = os.getenv("ALPACA_PAPER_API_KEY", "")
ALPACA_PAPER_SECRET_KEY = os.getenv("ALPACA_PAPER_SECRET_KEY", "")

@dataclass(frozen=True)
class Config:
    fast_ema: int
    slow_ema: int
    momentum_bars: int
    min_momentum_pct: float
    min_volume_ratio: float
    max_vwap_extension_atr: float
    stop_atr: float
    target_r: float
    max_hold_bars: int
    risk_fraction: float = 0.0075
    max_position_fraction: float = 0.50
    min_entry_price: float = 5.0
    max_entry_price: float = 100.0
    fee_bps_per_side: float = 1.0
    slippage_bps_per_side: float = 2.0

def config_grid() -> list[Config]:
    # V2 is intentionally much pickier than V1.
    return [
        Config(*x)
        for x in itertools.product(
            (8, 12),        # fast EMA
            (36,),          # slower trend confirmation
            (3, 6),         # momentum bars
            (0.15, 0.25),   # stronger momentum threshold
            (1.25, 1.50),   # stronger relative volume
            (0.75, 1.00),   # don't chase too far above VWAP
            (1.0,),         # ATR stop
            (1.50, 2.00),   # reward/risk
            (6,),           # max hold: 30 minutes
        )
    ]

def _headers() -> dict[str, str]:
    if not ALPACA_PAPER_API_KEY or not ALPACA_PAPER_SECRET_KEY:
        raise RuntimeError("Alpaca paper data credentials are required for research.")
    return {
        "APCA-API-KEY-ID": ALPACA_PAPER_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_PAPER_SECRET_KEY,
    }

def _fetch_chunk(symbols: Iterable[str], start: pd.Timestamp, end: pd.Timestamp) -> dict[str, list[dict]]:
    symbols = list(symbols)
    out = {s: [] for s in symbols}
    page_token = None
    while True:
        params = {
            "symbols": ",".join(symbols),
            "timeframe": "5Min",
            "start": start.isoformat().replace("+00:00", "Z"),
            "end": end.isoformat().replace("+00:00", "Z"),
            "limit": "10000",
            "adjustment": "all",
            "feed": "iex",
            "sort": "asc",
        }
        if page_token:
            params["page_token"] = page_token
        r = requests.get(
            f"{ALPACA_DATA_BASE_URL}/v2/stocks/bars",
            headers=_headers(),
            params=params,
            timeout=45,
        )
        r.raise_for_status()
        data = r.json()
        for symbol, rows in (data.get("bars") or {}).items():
            out.setdefault(symbol, []).extend(rows or [])
        page_token = data.get("next_page_token")
        if not page_token:
            break
    return out

def fetch_history(years: float = 2.0) -> dict[str, pd.DataFrame]:
    end = pd.Timestamp.now(tz="UTC").floor("D")
    start = end - pd.Timedelta(days=int(365.25 * years))
    combined = {s: [] for s in SYMBOLS}

    # Smaller chunks make pagination predictable and keep memory bounded.
    cursor = start
    while cursor < end:
        chunk_end = min(end, cursor + pd.Timedelta(days=30))
        data = _fetch_chunk(SYMBOLS, cursor, chunk_end)
        for s in SYMBOLS:
            combined[s].extend(data.get(s, []))
        cursor = chunk_end

    result: dict[str, pd.DataFrame] = {}
    for symbol, rows in combined.items():
        if not rows:
            continue
        df = pd.DataFrame(rows).rename(
            columns={"t": "time", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
        )
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
        for c in ("open", "high", "low", "close", "volume"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = (
            df.dropna(subset=["time", "open", "high", "low", "close", "volume"])
            .drop_duplicates(subset=["time"], keep="last")
            .sort_values("time")
            .reset_index(drop=True)
        )
        # Regular US equity session only. Convert to NY to handle DST correctly.
        ny = df["time"].dt.tz_convert("America/New_York")
        mins = ny.dt.hour * 60 + ny.dt.minute
        df = df[(mins >= 9 * 60 + 30) & (mins <= 15 * 60 + 55)].copy()
        df["session"] = ny.loc[df.index].dt.date.astype(str).values
        result[symbol] = df.reset_index(drop=True)
    return result

def add_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    x = df.copy()
    x["ema_fast"] = x["close"].ewm(span=cfg.fast_ema, adjust=False).mean()
    x["ema_slow"] = x["close"].ewm(span=cfg.slow_ema, adjust=False).mean()

    prev_close = x["close"].shift(1)
    tr = pd.concat(
        [
            x["high"] - x["low"],
            (x["high"] - prev_close).abs(),
            (x["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    x["atr"] = tr.rolling(14, min_periods=14).mean()
    x["momentum_pct"] = (x["close"] / x["close"].shift(cfg.momentum_bars) - 1.0) * 100.0
    x["volume_ratio"] = x["volume"] / x["volume"].rolling(20, min_periods=20).mean().shift(1)

    typical = (x["high"] + x["low"] + x["close"]) / 3.0
    pv = typical * x["volume"]
    x["vwap"] = pv.groupby(x["session"]).cumsum() / x["volume"].groupby(x["session"]).cumsum()

    ny = x["time"].dt.tz_convert("America/New_York")
    x["minute_of_day"] = ny.dt.hour * 60 + ny.dt.minute

    # First 30 minutes define the opening range.
    opening = x["minute_of_day"].between(570, 595)
    x["opening_range_high"] = x["high"].where(opening).groupby(x["session"]).transform("max")

    x["vwap_extension_atr"] = (x["close"] - x["vwap"]) / x["atr"]
    trade_window = x["minute_of_day"].between(600, 870)  # 10:00–14:30 ET

    # V2: confirmed breakout + trend + VWAP + momentum + relative volume,
    # while rejecting entries that are already too extended.
    x["signal_long"] = (
        trade_window
        & (x["close"] > x["opening_range_high"])
        & (x["ema_fast"] > x["ema_slow"])
        & (x["close"] > x["vwap"])
        & (x["momentum_pct"] >= cfg.min_momentum_pct)
        & (x["volume_ratio"] >= cfg.min_volume_ratio)
        & (x["vwap_extension_atr"] >= 0.0)
        & (x["vwap_extension_atr"] <= cfg.max_vwap_extension_atr)
        & (x["atr"] > 0)
    )
    return x

def prepare_events(history: dict[str, pd.DataFrame], cfg: Config) -> dict[pd.Timestamp, list[dict]]:
    featured = {symbol: add_features(raw, cfg) for symbol, raw in history.items()}

    # Market confirmation: only take stock longs while SPY is also above VWAP
    # and in a positive EMA trend on the completed signal bar.
    spy_ok = {}
    spy = featured.get("SPY")
    if spy is not None:
        for _, row in spy.iterrows():
            spy_ok[row["time"]] = bool(
                np.isfinite(row["vwap"])
                and row["close"] > row["vwap"]
                and row["ema_fast"] > row["ema_slow"]
            )

    events: dict[pd.Timestamp, list[dict]] = {}
    for symbol, x in featured.items():
        for i in range(1, len(x)):
            row = x.iloc[i]
            prev = x.iloc[i - 1]
            if row["session"] != prev["session"]:
                continue
            event = {
                "symbol": symbol,
                "time": row["time"],
                "session": row["session"],
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "entry_signal": bool(prev["signal_long"]) and bool(spy_ok.get(prev["time"], False)),
                "signal_atr": float(prev["atr"]) if np.isfinite(prev["atr"]) else np.nan,
                "signal_strength": (
                    float(prev["momentum_pct"]) + 0.15 * float(prev["volume_ratio"])
                    if np.isfinite(prev["momentum_pct"]) and np.isfinite(prev["volume_ratio"])
                    else -np.inf
                ),
            }
            events.setdefault(row["time"], []).append(event)
    return events

def backtest(history: dict[str, pd.DataFrame], cfg: Config, compound: bool = True, start_capital: float = STARTING_CAPITAL, cost_mult: float = 1.0):
    events = prepare_events(history, cfg)
    cash = float(start_capital)
    position = None
    trades = []
    equity_rows = []
    fee_rate = cfg.fee_bps_per_side * cost_mult / 10000.0
    slip_rate = cfg.slippage_bps_per_side * cost_mult / 10000.0

    for ts in sorted(events):
        rows = events[ts]

        if position is not None:
            bar = next((r for r in rows if r["symbol"] == position["symbol"]), None)
            if bar is not None:
                position["age"] += 1
                stop = position["entry_fill"] - position["stop_distance"]
                target = position["entry_fill"] + cfg.target_r * position["stop_distance"]
                hit_stop = bar["low"] <= stop
                hit_target = bar["high"] >= target

                exit_reason = None
                raw_exit = None
                # Conservative OHLC assumption: if stop and target both hit, stop wins.
                if hit_stop:
                    exit_reason, raw_exit = "STOP", stop
                elif hit_target:
                    exit_reason, raw_exit = "TARGET", target
                elif position["age"] >= cfg.max_hold_bars:
                    exit_reason, raw_exit = "TIME", bar["close"]
                elif bar["session"] != position["session"]:
                    exit_reason, raw_exit = "SESSION", bar["open"]

                # Never carry after the 15:55 ET bar.
                ny = pd.Timestamp(ts).tz_convert("America/New_York")
                if exit_reason is None and ny.hour == 15 and ny.minute >= 55:
                    exit_reason, raw_exit = "EOD", bar["close"]

                if exit_reason is not None:
                    exit_fill = float(raw_exit) * (1.0 - slip_rate)
                    proceeds = position["qty"] * exit_fill
                    exit_fee = proceeds * fee_rate
                    cash += proceeds - exit_fee
                    pnl = cash - position["cash_after_entry"] - position["reserved_cost_basis"]
                    trades.append({
                        "symbol": position["symbol"],
                        "entry_time": position["entry_time"],
                        "exit_time": ts,
                        "entry_fill": position["entry_fill"],
                        "exit_fill": exit_fill,
                        "qty": position["qty"],
                        "pnl": pnl,
                        "return_on_equity_pct": 100.0 * pnl / max(position["equity_at_entry"], 1e-12),
                        "reason": exit_reason,
                    })
                    position = None

        if position is None:
            candidates = [
                r for r in rows
                if r["entry_signal"] and np.isfinite(r["signal_atr"]) and r["signal_atr"] > 0 and cfg.min_entry_price <= r["open"] <= cfg.max_entry_price
            ]
            if candidates:
                best = max(candidates, key=lambda r: r["signal_strength"])
                equity_now = cash
                sizing_equity = equity_now if compound else start_capital
                stop_distance = cfg.stop_atr * best["signal_atr"]
                risk_budget = sizing_equity * cfg.risk_fraction
                qty_risk = risk_budget / stop_distance
                max_notional = sizing_equity * cfg.max_position_fraction
                entry_fill = best["open"] * (1.0 + slip_rate)
                qty_cap = max_notional / entry_fill
                qty_cash = cash / (entry_fill * (1.0 + fee_rate))
                qty = max(0.0, min(qty_risk, qty_cap, qty_cash))

                if qty * entry_fill >= 1.0:
                    notional = qty * entry_fill
                    entry_fee = notional * fee_rate
                    reserved = notional + entry_fee
                    cash_before = cash
                    cash -= reserved
                    position = {
                        "symbol": best["symbol"],
                        "session": best["session"],
                        "entry_time": ts,
                        "entry_fill": entry_fill,
                        "qty": qty,
                        "stop_distance": stop_distance,
                        "age": 0,
                        "equity_at_entry": equity_now,
                        "cash_after_entry": cash,
                        "reserved_cost_basis": reserved,
                    }

        mark = cash
        if position is not None:
            bar = next((r for r in rows if r["symbol"] == position["symbol"]), None)
            px = bar["close"] if bar else position["entry_fill"]
            mark += position["qty"] * px
        equity_rows.append({"time": ts, "equity": mark})

    if position is not None:
        # Mark final open position to last available close and charge exit costs.
        last_ts = max(events)
        last_rows = events[last_ts]
        bar = next((r for r in last_rows if r["symbol"] == position["symbol"]), None)
        raw_exit = bar["close"] if bar else position["entry_fill"]
        exit_fill = raw_exit * (1.0 - slip_rate)
        proceeds = position["qty"] * exit_fill
        exit_fee = proceeds * fee_rate
        cash += proceeds - exit_fee
        pnl = cash - position["cash_after_entry"] - position["reserved_cost_basis"]
        trades.append({
            "symbol": position["symbol"],
            "entry_time": position["entry_time"],
            "exit_time": last_ts,
            "entry_fill": position["entry_fill"],
            "exit_fill": exit_fill,
            "qty": position["qty"],
            "pnl": pnl,
            "return_on_equity_pct": 100.0 * pnl / max(position["equity_at_entry"], 1e-12),
            "reason": "FINAL",
        })
        position = None

    eq = pd.DataFrame(equity_rows)
    tr = pd.DataFrame(trades)
    ending = float(cash)
    return ending, eq, tr

def summarize(start_capital: float, ending: float, eq: pd.DataFrame, trades: pd.DataFrame) -> dict:
    if eq.empty:
        return {
            "start_capital": start_capital, "ending_capital": ending, "return_pct": 0.0,
            "max_drawdown_pct": 0.0, "trades": 0, "win_rate_pct": 0.0, "profit_factor": 0.0,
            "avg_pnl_per_trade": 0.0, "avg_pnl_per_calendar_day": 0.0,
        }
    curve = pd.to_numeric(eq["equity"], errors="coerce").ffill().dropna()
    dd = curve / curve.cummax() - 1.0
    pnl = pd.to_numeric(trades.get("pnl", pd.Series(dtype=float)), errors="coerce").dropna()
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    pf = float(wins.sum() / abs(losses.sum())) if len(losses) and abs(losses.sum()) > 0 else (float("inf") if len(wins) else 0.0)
    t0 = pd.Timestamp(eq["time"].iloc[0])
    t1 = pd.Timestamp(eq["time"].iloc[-1])
    days = max(1.0, (t1 - t0).total_seconds() / 86400.0)
    return {
        "start_capital": round(float(start_capital), 4),
        "ending_capital": round(float(ending), 4),
        "return_pct": round((ending / start_capital - 1.0) * 100.0, 4),
        "max_drawdown_pct": round(float(dd.min()) * 100.0, 4),
        "trades": int(len(pnl)),
        "win_rate_pct": round(float((pnl > 0).mean() * 100.0), 4) if len(pnl) else 0.0,
        "profit_factor": round(pf, 4) if np.isfinite(pf) else "Infinity",
        "avg_pnl_per_trade": round(float(pnl.mean()), 6) if len(pnl) else 0.0,
        "avg_pnl_per_calendar_day": round(float((ending - start_capital) / days), 6),
        "days_ge_1_dollar": int((
            trades.assign(
                exit_day=pd.to_datetime(trades["exit_time"], utc=True).dt.tz_convert("America/New_York").dt.date
            ).groupby("exit_day")["pnl"].sum() >= 1.0
        ).sum()) if not trades.empty else 0,
        "positive_days_pct": round(
            100.0 * float((
                trades.assign(
                    exit_day=pd.to_datetime(trades["exit_time"], utc=True).dt.tz_convert("America/New_York").dt.date
                ).groupby("exit_day")["pnl"].sum() > 0
            ).mean()), 4
        ) if not trades.empty else 0.0,
    }

def chronological_folds(history: dict[str, pd.DataFrame], folds: int = 4):
    all_times = sorted(set().union(*[set(df["time"]) for df in history.values() if not df.empty]))
    if len(all_times) < 1000:
        raise RuntimeError("Not enough intraday history for chronological validation.")
    bounds = np.linspace(0, len(all_times), folds + 1, dtype=int)
    out = []
    for i in range(folds):
        lo, hi = all_times[bounds[i]], all_times[max(bounds[i + 1] - 1, bounds[i])]
        subset = {}
        for s, df in history.items():
            x = df[(df["time"] >= lo) & (df["time"] <= hi)].copy()
            if len(x) >= 100:
                subset[s] = x
        out.append((lo, hi, subset))
    return out

def select_config(history: dict[str, pd.DataFrame]):
    folds = chronological_folds(history, 4)
    rows = []
    for cfg in config_grid():
        fold_returns, fold_dds, fold_pfs, fold_trades = [], [], [], []
        for _, _, h in folds[:3]:  # first 3 folds develop; 4th is final diagnostic
            end, eq, tr = backtest(h, cfg, compound=True)
            m = summarize(STARTING_CAPITAL, end, eq, tr)
            fold_returns.append(float(m["return_pct"]))
            fold_dds.append(float(m["max_drawdown_pct"]))
            pf = m["profit_factor"]
            fold_pfs.append(float(pf) if pf != "Infinity" else 99.0)
            fold_trades.append(int(m["trades"]))
        robust = (
            min(fold_returns) > 0.0
            and max(abs(x) for x in fold_dds) <= 25.0
            and min(fold_pfs) >= 1.05
            and min(fold_trades) >= 20
        )
        score = min(fold_returns) + 0.25 * np.mean(fold_returns) + 2.0 * min(fold_pfs) - 0.20 * max(abs(x) for x in fold_dds)
        rows.append({
            **asdict(cfg),
            "robust": bool(robust),
            "selection_score": float(score),
            "worst_fold_return_pct": float(min(fold_returns)),
            "mean_fold_return_pct": float(np.mean(fold_returns)),
            "worst_fold_profit_factor": float(min(fold_pfs)),
            "worst_fold_trades": int(min(fold_trades)),
            "max_fold_drawdown_abs_pct": float(max(abs(x) for x in fold_dds)),
        })
    table = pd.DataFrame(rows).sort_values(["robust", "selection_score"], ascending=[False, False]).reset_index(drop=True)
    best = Config(**{k: table.iloc[0][k] for k in asdict(config_grid()[0]).keys()})
    return best, table, folds

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    history = fetch_history(years=2.0)
    if len(history) < 5:
        raise RuntimeError(f"Only received usable data for {len(history)} symbols.")

    selected, grid, folds = select_config(history)
    grid.to_csv(OUT_DIR / "grid_development.csv", index=False)

    # Final chronological fold: untouched by selection.
    _, _, final_history = folds[3]
    final_end, final_eq, final_tr = backtest(final_history, selected, compound=True)
    final_flat_end, final_flat_eq, final_flat_tr = backtest(final_history, selected, compound=False)

    # Full-period diagnostics after selection.
    full_end, full_eq, full_tr = backtest(history, selected, compound=True)
    flat_end, flat_eq, flat_tr = backtest(history, selected, compound=False)

    cost_stress = {}
    for mult in (1.0, 2.0, 4.0):
        stress_end, stress_eq, stress_tr = backtest(history, selected, compound=True, cost_mult=mult)
        cost_stress[f"{int(mult)}x"] = summarize(STARTING_CAPITAL, stress_end, stress_eq, stress_tr)

    summary = {
        "version": VERSION,
        "research_only": True,
        "places_orders": False,
        "starting_capital": STARTING_CAPITAL,
        "symbols": list(history),
        "selected_config": asdict(selected),
        "development_configs": len(grid),
        "development_robust_configs": int(grid["robust"].sum()),
        "final_chronological_fold_compounding": summarize(STARTING_CAPITAL, final_end, final_eq, final_tr),
        "final_chronological_fold_flat_capital": summarize(STARTING_CAPITAL, final_flat_end, final_flat_eq, final_flat_tr),
        "full_period_compounding": summarize(STARTING_CAPITAL, full_end, full_eq, full_tr),
        "full_period_flat_capital": summarize(STARTING_CAPITAL, flat_end, flat_eq, flat_tr),
        "cost_stress": cost_stress,
        "methodology": {
            "bars": "5-minute Alpaca IEX adjusted bars",
            "history_years": 2.0,
            "signal_execution": "signal on completed bar, entry at next bar open",
            "same_bar_stop_target": "stop-first conservative",
            "overnight": "disabled",
            "positioning": "one position at a time; no leverage",
            "compounding": "risk and notional sizing use current realized account equity",
            "flat_comparison": "risk/notional sizing use original $100 base; actual cash still constrains trades",
            "costs": "1 bp fee + 2 bp slippage per side, plus 2x and 4x stress",
            "v2_entry": "under-$100, opening-range breakout, SPY regime, stronger RVOL/momentum, anti-chase VWAP extension",
        },
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    full_eq.to_csv(OUT_DIR / "full_compounding_equity.csv", index=False)
    full_tr.to_csv(OUT_DIR / "full_compounding_trades.csv", index=False)
    final_eq.to_csv(OUT_DIR / "final_fold_compounding_equity.csv", index=False)
    final_tr.to_csv(OUT_DIR / "final_fold_compounding_trades.csv", index=False)

    readme = f"""# Equity Intraday Research V2

Research only. No orders are placed.

Starting capital: ${STARTING_CAPITAL:.2f}

Selected config:
```json
{json.dumps(asdict(selected), indent=2)}
```

Final chronological fold, compounded:
```json
{json.dumps(summary["final_chronological_fold_compounding"], indent=2)}
```

Full-period compounded diagnostic:
```json
{json.dumps(summary["full_period_compounding"], indent=2)}
```

Flat-capital comparison:
```json
{json.dumps(summary["full_period_flat_capital"], indent=2)}
```

The compounded path reinvests realized gains into later position sizing. It does not assume leverage or reuse unrealized profits.
"""
    (OUT_DIR / "README.md").write_text(readme, encoding="utf-8")

    if OUT_ZIP.exists():
        OUT_ZIP.unlink()
    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in OUT_DIR.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(OUT_DIR.parent))

    print(json.dumps(summary, indent=2, default=str))

if __name__ == "__main__":
    main()
