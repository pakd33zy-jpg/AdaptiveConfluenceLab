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
from adaptive_confluence import BacktestConfig, StrategyConfig, compute_features, run_backtest


def score(stats):
    # Reward return and profit factor, penalize drawdown. Require some activity.
    if stats["trades"] < 8:
        return -1e9
    pf = min(stats["profit_factor"], 5.0) if stats["profit_factor"] != float("inf") else 5.0
    return stats["return_pct"] + 3.0 * pf + stats["max_drawdown_pct"] * 1.5


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

    thresholds = [0.24, 0.28, 0.32]
    adx_values = [20.0, 22.0, 24.0]
    stop_values = [1.1, 1.2, 1.35]
    target_values = [2.0, 2.4, 2.8]
    total = len(thresholds) * len(adx_values) * len(stop_values) * len(target_values)

    rows = []
    completed = 0
    started = time.perf_counter()
    bt_cfg = BacktestConfig()

    print(
        f"Walk-forward: {len(train):,} train bars / {len(test):,} untouched test bars; "
        f"{total} parameter combinations.",
        flush=True,
    )

    # Only threshold and ADX change the signal frame in this grid. The stop and
    # target values change trade management only, so each expensive feature
    # build is reused for all nine stop/target combinations.
    for threshold, adx_min in itertools.product(thresholds, adx_values):
        feature_started = time.perf_counter()
        signal_cfg = StrategyConfig(
            score_threshold=threshold,
            trend_adx_min=adx_min,
        )
        print(
            f"Building features for threshold={threshold:.2f}, ADX={adx_min:.0f} ...",
            flush=True,
        )
        features = compute_features(train, signal_cfg)
        print(
            f"  features ready in {fmt_duration(time.perf_counter() - feature_started)}",
            flush=True,
        )

        for stop_atr, target_r in itertools.product(stop_values, target_values):
            cfg = StrategyConfig(
                score_threshold=threshold,
                trend_adx_min=adx_min,
                breakout_stop_atr=stop_atr,
                breakout_target_r=target_r,
            )
            train_stats = run_backtest(
                train,
                cfg,
                bt_cfg,
                precomputed_features=features,
            )["stats"]
            rows.append((score(train_stats), cfg, train_stats))
            completed += 1

            elapsed = time.perf_counter() - started
            eta = (elapsed / completed) * (total - completed) if completed else 0
            print(
                f"[{completed:02d}/{total}] "
                f"thr={threshold:.2f} adx={adx_min:.0f} stop={stop_atr:.2f} target={target_r:.1f} | "
                f"return={train_stats['return_pct']:.3f}% "
                f"PF={train_stats['profit_factor']:.2f} "
                f"trades={train_stats['trades']} | "
                f"elapsed {fmt_duration(elapsed)} ETA {fmt_duration(eta)}",
                flush=True,
            )

        del features

    rows.sort(key=lambda x: x[0], reverse=True)
    best = rows[0][1]

    print("Running best configuration on untouched 30% test set ...", flush=True)
    test_features = compute_features(test, best)
    test_stats = run_backtest(
        test,
        best,
        bt_cfg,
        precomputed_features=test_features,
    )["stats"]

    output = {
        "method": "70% chronological train / 30% untouched test",
        "best_train_config": best.to_dict(),
        "best_train_stats": rows[0][2],
        "out_of_sample_test_stats": test_stats,
        "warning": "Do not deploy from one split. Repeat across symbols, regimes, and dates; then forward-test in paper.",
    }
    Path(args.out).write_text(json.dumps(output, indent=2, default=str))
    print(f"Finished in {fmt_duration(time.perf_counter() - started)}", flush=True)
    print(json.dumps(output, indent=2, default=str))


if __name__ == "__main__":
    main()
