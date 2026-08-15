#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from adaptive_confluence import StrategyConfig, compute_features


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    time_col = next((c for c in ["timestamp", "time", "datetime", "date"] if c in df.columns), None)
    if time_col:
        df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
        df = df.dropna(subset=[time_col]).set_index(time_col).sort_index()
    return df


def funnel(feat: pd.DataFrame, title: str, columns: list[str]) -> None:
    print(f"\n{title}")
    mask = pd.Series(True, index=feat.index)
    for col in columns:
        cond = feat[col].fillna(False).astype(bool)
        mask &= cond
        print(f"  {col:<28} {int(mask.sum()):>8,}")


def main():
    p = argparse.ArgumentParser(description="Show how V2.1 signal filters reduce candidate bars")
    p.add_argument("csv")
    args = p.parse_args()

    raw = load_csv(args.csv)
    cfg = StrategyConfig()
    feat = compute_features(raw, cfg)

    print(f"V2.1 diagnostics for {args.csv}")
    print(f"bars: {len(feat):,}")
    print(f"final signals: {int((feat.signal != 0).sum()):,}")
    print(f"  long:       {int((feat.signal > 0).sum()):,}")
    print(f"  short:      {int((feat.signal < 0).sum()):,}")
    print(f"  TREND:      {int((feat.setup == 'TREND').sum()):,}")
    print(f"  BREAKOUT:   {int((feat.setup == 'BREAKOUT').sum()):,}")

    common_long = ["session_ok", "volatility_ok", "long_trend_ok", "adx_ok"]
    common_short = ["session_ok", "volatility_ok", "short_trend_ok", "adx_ok"]

    funnel(feat, "TREND LONG funnel", common_long + ["trend_volume_ok", "long_di_ok", "pull_touch_long", "pull_confirm_long"])
    funnel(feat, "TREND SHORT funnel", common_short + ["trend_volume_ok", "short_di_ok", "pull_touch_short", "pull_confirm_short"])
    funnel(feat, "BREAKOUT LONG funnel", common_long + ["breakout_volume_ok", "long_vwap_ok", "long_di_ok", "breakout_structure_long"])
    funnel(feat, "BREAKOUT SHORT funnel", common_short + ["breakout_volume_ok", "short_vwap_ok", "short_di_ok", "breakout_structure_short"])


if __name__ == "__main__":
    main()
