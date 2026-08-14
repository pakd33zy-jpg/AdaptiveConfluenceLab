#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adaptive_confluence import BacktestConfig, StrategyConfig, run_backtest


def main():
    p = argparse.ArgumentParser(description="Backtest Adaptive Confluence on an OHLCV CSV")
    p.add_argument("csv", help="CSV with timestamp/open/high/low/close/volume")
    p.add_argument("--capital", type=float, default=100000)
    p.add_argument("--commission-pct", type=float, default=0.01)
    p.add_argument("--slippage-bps", type=float, default=1.0)
    p.add_argument("--out", default="backtest_output")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    time_col = next((c for c in ["timestamp", "time", "datetime", "date"] if c in df.columns), None)
    if time_col:
        df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
        df = df.dropna(subset=[time_col]).set_index(time_col).sort_index()

    result = run_backtest(
        df,
        StrategyConfig(),
        BacktestConfig(args.capital, args.commission_pct, args.slippage_bps),
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "stats.json").write_text(json.dumps(result["stats"], indent=2, default=str))
    result["trades"].to_csv(out / "trades.csv", index=False)
    result["equity_curve"].rename("equity").to_csv(out / "equity.csv")
    print(json.dumps(result["stats"], indent=2))


if __name__ == "__main__":
    main()
