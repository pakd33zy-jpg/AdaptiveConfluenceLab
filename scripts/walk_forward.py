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


def score(stats):
    # Do not reward a configuration with too few trades. Profit factor and
    # expectancy matter more than raw return during selection.
    if stats["trades"] < 20:
        return -1e9
    pf = min(float(stats["profit_factor"]), 3.0)
    return (
        float(stats["return_pct"]) * 1.0
        + pf * 2.0
        + float(stats["max_drawdown_pct"]) * 1.25
        + (0.5 if float(stats["expectancy"]) > 0 else -0.5)
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


def main():
    p = argparse.ArgumentParser(description="V2 compact chronological walk-forward sweep")
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

    # V2 deliberately uses a compact grid. The goal is robustness, not finding
    # one lucky point in an 81-cell search.
    grid = list(itertools.product(
        [20.0, 24.0],          # ADX
        [0.90, 1.05],          # pullback relative volume
        [1.20, 1.40],          # breakout relative volume
        [1.20, 1.40],          # breakout stop ATR
        [2.0, 2.4],            # breakout target R
    ))

    rows = []
    started = time.perf_counter()
    bt_cfg = BacktestConfig()
    total = len(grid)
    print(f"V2 walk-forward: {len(train):,} train / {len(test):,} test bars; {total} configs", flush=True)

    for n, (adx, min_rv, breakout_rv, stop_atr, target_r) in enumerate(grid, 1):
        cfg = StrategyConfig(
            trend_adx_min=adx,
            min_relative_volume=min_rv,
            breakout_relative_volume=breakout_rv,
            breakout_stop_atr=stop_atr,
            breakout_target_r=target_r,
        )
        stats = run_backtest(train, cfg, bt_cfg)["stats"]
        rows.append((score(stats), cfg, stats))
        elapsed = time.perf_counter() - started
        eta = elapsed / n * (total - n)
        print(
            f"[{n:02d}/{total}] adx={adx:.0f} rv={min_rv:.2f}/{breakout_rv:.2f} "
            f"stop={stop_atr:.2f} target={target_r:.1f} | "
            f"ret={stats['return_pct']:.3f}% PF={stats['profit_factor']:.2f} "
            f"trades={stats['trades']} | ETA {fmt_duration(eta)}",
            flush=True,
        )

    rows.sort(key=lambda x: x[0], reverse=True)
    best = rows[0][1]
    test_stats = run_backtest(test, best, bt_cfg)["stats"]
    output = {
        "strategy_version": "V2 trend+breakout",
        "method": "70% chronological train / 30% untouched test",
        "tested_configs": total,
        "best_train_config": best.to_dict(),
        "best_train_stats": rows[0][2],
        "out_of_sample_test_stats": test_stats,
        "warning": "A positive backtest is not proof of future profitability. Repeat across symbols/regimes and paper-forward-test.",
    }
    Path(args.out).write_text(json.dumps(output, indent=2, default=str))
    print(f"Finished in {fmt_duration(time.perf_counter() - started)}", flush=True)
    print(json.dumps(output, indent=2, default=str))


if __name__ == "__main__":
    main()
