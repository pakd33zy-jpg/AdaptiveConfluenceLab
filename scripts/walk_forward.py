#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from adaptive_confluence import BacktestConfig, StrategyConfig, run_backtest


MIN_TRAIN_TRADES = 30
MIN_TEST_TRADES = 10


def score(stats):
    """Select robust activity first, then risk-adjusted performance.

    Configurations with fewer than MIN_TRAIN_TRADES are intentionally ranked
    below every sufficiently active configuration. If all are too quiet, the
    most active one wins so the JSON still exposes the failure mode clearly.
    """
    trades = int(stats["trades"])
    if trades < MIN_TRAIN_TRADES:
        return -10_000.0 + trades
    pf = min(float(stats["profit_factor"]), 3.0)
    return (
        float(stats["return_pct"])
        + 2.0 * pf
        + 1.25 * float(stats["max_drawdown_pct"])
        + (0.75 if float(stats["expectancy"]) > 0 else -0.75)
    )


def fmt_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def candidate_status(train_stats: dict, test_stats: dict) -> str:
    if int(train_stats["trades"]) < MIN_TRAIN_TRADES:
        return "REJECT_INSUFFICIENT_TRAIN_ACTIVITY"
    if int(test_stats["trades"]) < MIN_TEST_TRADES:
        return "REJECT_INSUFFICIENT_TEST_ACTIVITY"
    if float(test_stats["expectancy"]) <= 0 or float(test_stats["profit_factor"]) <= 1.0 or float(test_stats["return_pct"]) <= 0:
        return "REJECT_NEGATIVE_OUT_OF_SAMPLE_EDGE"
    return "CANDIDATE_FOR_MULTI_SYMBOL_PAPER_VALIDATION"


def main():
    p = argparse.ArgumentParser(description="V2.1 compact chronological walk-forward sweep")
    p.add_argument("csv")
    p.add_argument("--out", default="walk_forward.json")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    time_col = next((c for c in ["timestamp", "time", "datetime", "date"] if c in df.columns), None)
    if time_col:
        df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
        df = df.dropna(subset=[time_col]).set_index(time_col).sort_index()

    split = int(len(df) * 0.70)
    train, test = df.iloc[:split], df.iloc[split:]

    # Stage 1 tunes entry selectivity only. Exit parameters stay fixed so we do
    # not confuse a lucky stop/target combination with a real entry edge.
    grid = list(itertools.product(
        [14.0, 18.0, 22.0],    # ADX
        [0.50, 0.70],          # pullback relative volume
        [0.90, 1.10],          # breakout relative volume
        [10, 16],              # prior channel length
    ))

    rows = []
    started = time.perf_counter()
    bt_cfg = BacktestConfig()
    total = len(grid)
    print(f"V2.1 walk-forward: {len(train):,} train / {len(test):,} test bars; {total} configs", flush=True)

    for n, (adx, min_rv, breakout_rv, donchian) in enumerate(grid, 1):
        cfg = StrategyConfig(
            trend_adx_min=adx,
            min_relative_volume=min_rv,
            breakout_relative_volume=breakout_rv,
            donchian_length=donchian,
        )
        stats = run_backtest(train, cfg, bt_cfg)["stats"]
        rows.append((score(stats), cfg, stats))
        elapsed = time.perf_counter() - started
        eta = elapsed / n * (total - n)
        print(
            f"[{n:02d}/{total}] adx={adx:.0f} rv={min_rv:.2f}/{breakout_rv:.2f} ch={donchian:02d} | "
            f"ret={stats['return_pct']:.3f}% PF={stats['profit_factor']:.2f} "
            f"trades={stats['trades']} | ETA {fmt_duration(eta)}",
            flush=True,
        )

    rows.sort(key=lambda x: x[0], reverse=True)
    best = rows[0][1]
    train_stats = rows[0][2]
    test_stats = run_backtest(test, best, bt_cfg)["stats"]
    status = candidate_status(train_stats, test_stats)

    output = {
        "strategy_version": "V2.1 trend+breakout",
        "method": "70% chronological train / 30% untouched test",
        "tested_configs": total,
        "minimum_train_trades": MIN_TRAIN_TRADES,
        "minimum_test_trades": MIN_TEST_TRADES,
        "best_train_config": best.to_dict(),
        "best_train_stats": train_stats,
        "out_of_sample_test_stats": test_stats,
        "candidate_status": status,
        "warning": "A positive backtest is not proof of future profitability. Pass status only means continue to multi-symbol and paper-forward validation.",
    }
    Path(args.out).write_text(json.dumps(output, indent=2, default=str))
    print(f"Finished in {fmt_duration(time.perf_counter() - started)}", flush=True)
    print(f"STATUS: {status}", flush=True)
    print(json.dumps(output, indent=2, default=str))


if __name__ == "__main__":
    main()
