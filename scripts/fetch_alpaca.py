#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
import requests


def main():
    p = argparse.ArgumentParser(description="Fetch Alpaca stock bars for research/backtests")
    p.add_argument("symbol")
    p.add_argument("--start", required=True, help="ISO date/time, e.g. 2026-07-01T13:30:00Z")
    p.add_argument("--end", required=True)
    p.add_argument("--timeframe", default="1Min")
    p.add_argument("--feed", default="iex")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    key = os.getenv("ALPACA_API_KEY") or os.getenv("ALPACA_PAPER_API_KEY")
    secret = os.getenv("ALPACA_SECRET_KEY") or os.getenv("ALPACA_PAPER_SECRET_KEY")
    if not key or not secret:
        raise SystemExit("Set ALPACA_API_KEY/ALPACA_SECRET_KEY or paper equivalents first.")

    url = f"https://data.alpaca.markets/v2/stocks/{args.symbol}/bars"
    params = {
        "start": args.start,
        "end": args.end,
        "timeframe": args.timeframe,
        "feed": args.feed,
        "limit": 10000,
        "adjustment": "raw",
        "sort": "asc",
    }
    headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
    rows = []
    token = None
    while True:
        if token:
            params["page_token"] = token
        r = requests.get(url, params=params, headers=headers, timeout=30)
        r.raise_for_status()
        payload = r.json()
        rows.extend(payload.get("bars", []))
        token = payload.get("next_page_token")
        if not token:
            break

    df = pd.DataFrame(rows).rename(columns={
        "t": "timestamp", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"
    })
    out = Path(args.out or f"{args.symbol}_{args.timeframe}.csv")
    df.to_csv(out, index=False)
    print(f"Wrote {len(df):,} bars to {out}")


if __name__ == "__main__":
    main()
