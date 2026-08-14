#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from adaptive_confluence import BacktestConfig, StrategyConfig, run_backtest


def score(stats):
    # Reward return and profit factor, penalize drawdown. Require some activity.
    if stats["trades"] < 8:
        return -1e9
    pf = min(stats["profit_factor"], 5.0) if stats["profit_factor"] != float("inf") else 5.0
    return stats["return_pct"] + 3.0 * pf + stats["max_drawdown_pct"] * 1.5


def main():
    p = argparse.ArgumentParser(description="Simple chronological walk-forward robustness sweep")
    p.add_argument("csv")
    p.add_argument("--out", default="walk_forward.json")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    time_col = next((c for c in ["timestamp", "time", "datetime", "date"] if c in df.columns), None)
    if time_col:
        df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
        df = df.dropna(subset=[time_col]).set_index(time_col).sort_index()

    n = len(df)
    split = int(n * 0.70)
    train, test = df.iloc[:split], df.iloc[split:]
    grid = itertools.product([0.24, 0.28, 0.32], [20.0, 22.0, 24.0], [1.1, 1.2, 1.35], [2.0, 2.4, 2.8])
    rows = []
    for threshold, adx_min, stop_atr, target_r in grid:
        cfg = StrategyConfig(
            score_threshold=threshold,
            trend_adx_min=adx_min,
            breakout_stop_atr=stop_atr,
            breakout_target_r=target_r,
        )
        train_stats = run_backtest(train, cfg, BacktestConfig())["stats"]
        rows.append((score(train_stats), cfg, train_stats))
    rows.sort(key=lambda x: x[0], reverse=True)

    best = rows[0][1]
    test_stats = run_backtest(test, best, BacktestConfig())["stats"]
    output = {
        "method": "70% chronological train / 30% untouched test",
        "best_train_config": best.to_dict(),
        "best_train_stats": rows[0][2],
        "out_of_sample_test_stats": test_stats,
        "warning": "Do not deploy from one split. Repeat across symbols, regimes, and dates; then forward-test in paper.",
    }
    Path(args.out).write_text(json.dumps(output, indent=2, default=str))
    print(json.dumps(output, indent=2, default=str))


if __name__ == "__main__":
    main()
