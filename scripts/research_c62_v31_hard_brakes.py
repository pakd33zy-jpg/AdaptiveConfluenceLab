#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

DATA_BASE = "https://data.alpaca.markets"
BTC = "BTC/USD"
TARGETS = ("ETH/USD", "SOL/USD", "LINK/USD")
SYMBOLS = (BTC, *TARGETS)
ROUND_TRIP_COST = 0.01
OUT = Path("C62V31HardBrakes_research_pack")

WINDOWS = {
    "older": (
        datetime(2024, 10, 7, tzinfo=timezone.utc),
        datetime(2025, 10, 7, tzinfo=timezone.utc),
    ),
    "recent": (
        datetime(2025, 10, 8, tzinfo=timezone.utc),
        datetime(2026, 10, 8, tzinfo=timezone.utc),
    ),
}

@dataclass
class Trade:
    symbol: str
    signal_time: datetime
    entry_time: datetime
    exit_time: datetime
    net_return: float
    reason: str
    added: bool


def credentials():
    key = (os.getenv("ALPACA_PAPER_API_KEY") or os.getenv("ALPACA_API_KEY") or "").strip()
    secret = (os.getenv("ALPACA_PAPER_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY") or "").strip()
    if not key or not secret:
        raise SystemExit("Missing Alpaca paper-data credentials")
    return key, secret


class AlpacaCrypto:
    def __init__(self):
        key, secret = credentials()
        self.s = requests.Session()
        self.s.headers.update({
            "APCA-API-KEY-ID": key,
            "APCA-API-SECRET-KEY": secret,
        })

    def get(self, path, params, retries=8):
        delay = 1.0
        last = None
        for _ in range(retries):
            try:
                r = self.s.get(f"{DATA_BASE}{path}", params=params, timeout=60)
                if r.status_code == 429:
                    time.sleep(delay)
                    delay = min(delay * 2, 16)
                    continue
                if r.status_code >= 500:
                    last = RuntimeError(f"Alpaca {r.status_code}: {r.text[:300]}")
                    time.sleep(delay)
                    delay = min(delay * 2, 16)
                    continue
                if not r.ok:
                    raise RuntimeError(f"Alpaca {r.status_code}: {r.text[:1000]}")
                return r.json()
            except (requests.Timeout, requests.ConnectionError) as exc:
                last = exc
                time.sleep(delay)
                delay = min(delay * 2, 16)
        raise RuntimeError(f"Alpaca request failed after retries: {last}")

    def bars(self, symbols, timeframe, start, end):
        out = {s: [] for s in symbols}
        token = None
        pages = 0
        while True:
            p = {
                "symbols": ",".join(symbols),
                "timeframe": timeframe,
                "start": start.isoformat().replace("+00:00", "Z"),
                "end": end.isoformat().replace("+00:00", "Z"),
                "sort": "asc",
                "limit": 10000,
            }
            if token:
                p["page_token"] = token
            payload = self.get("/v1beta3/crypto/us/bars", p)
            for symbol, rows in (payload.get("bars") or {}).items():
                out.setdefault(symbol, []).extend(rows or [])
            token = payload.get("next_page_token")
            pages += 1
            if not token:
                break
            if pages > 50:
                raise RuntimeError("Unexpected crypto pagination depth")
        for s in out:
            out[s].sort(key=lambda b: b.get("t") or "")
        return out


def dt(bar):
    return datetime.fromisoformat(str(bar["t"]).replace("Z", "+00:00"))


def p(bar, key):
    return float(bar[key])


def atr(rows, end, length=14):
    total = 0.0
    for i in range(end - length + 1, end + 1):
        hi, lo = p(rows[i], "h"), p(rows[i], "l")
        prev = p(rows[i - 1], "c")
        total += max(hi - lo, abs(hi - prev), abs(lo - prev))
    return total / length


def structure(rows, end, length=64):
    recent = rows[end - length + 1:end + 1]
    prior = rows[end - 2 * length + 1:end - length + 1]
    rh = max(p(x, "h") for x in recent)
    rl = min(p(x, "l") for x in recent)
    ph = max(p(x, "h") for x in prior)
    pl = min(p(x, "l") for x in prior)
    if rh > ph and rl > pl:
        return 1
    if rh < ph and rl < pl:
        return -1
    return 0


def c62_signal(asset, ai, btc, bi):
    if ai < 129 or bi < 129:
        return None
    close = p(asset[ai], "c")
    btc_close = p(btc[bi], "c")
    a_atr = atr(asset, ai)
    b_atr = atr(btc, bi)
    btc_move = btc_close / p(btc[bi - 8], "c") - 1
    asset_move = close / p(asset[ai - 8], "c") - 1
    checks = (
        structure(btc, bi) == 1,
        structure(asset, ai) >= 0,
        b_atr / btc_close >= 0.004,
        btc_move >= 0.006,
        asset_move >= 0,
        asset_move <= btc_move * 0.5,
        close > p(asset[ai], "o"),
        close > p(asset[ai - 1], "h"),
    )
    if not all(checks):
        return None
    return {"risk_pct": 2.5 * a_atr / close * 100.0}


def prior_daily_state(signal_time, daily):
    usable = [b for b in daily if dt(b) + timedelta(days=1) <= signal_time]
    if len(usable) < 201:
        return None
    closes = [p(b, "c") for b in usable]
    close = closes[-1]
    sma200 = sum(closes[-200:]) / 200
    mom63 = close / closes[-64] - 1
    mom126 = close / closes[-127] - 1

    lr = [math.log(closes[i] / closes[i - 1]) for i in range(len(closes) - 20, len(closes))]
    if len(lr) < 20:
        return None
    mean = sum(lr) / len(lr)
    var = sum((x - mean) ** 2 for x in lr) / (len(lr) - 1)
    vol20 = math.sqrt(var) * math.sqrt(252.0) * 100.0

    broken = close < sma200 and mom126 < 0
    fast_broken = close < sma200 and mom63 < 0
    shock = mom63 < -0.06 or (vol20 > 30 and mom63 < 0)

    return {
        "close": close,
        "sma200": sma200,
        "mom63": mom63,
        "mom126": mom126,
        "vol20_pct": vol20,
        "broken": broken,
        "fast_broken": fast_broken,
        "shock": shock,
    }


def allowed(variant, state):
    if variant == "baseline":
        return True
    if state is None:
        return False
    if variant == "trend_cash_hard":
        return not state["broken"]
    if variant == "fast_brake_hard":
        return not (state["fast_broken"] or state["shock"])
    if variant == "two_stage_hard":
        return not (state["broken"] and state["shock"])
    raise ValueError(variant)


def finish(legs, exit_price):
    # Match shadow accounting: return is normalized to actually deployed notional.
    gross_weight = sum(w for _, w in legs)
    total = 0.0
    for entry, weight in legs:
        total += weight * ((exit_price / entry - 1.0) - ROUND_TRIP_COST)
    return total / gross_weight


def simulate_trade(rows, signal_i, risk_pct):
    entry_i = signal_i + 1
    if entry_i >= len(rows):
        return None
    if dt(rows[entry_i]) - dt(rows[signal_i]) > timedelta(minutes=20):
        return None

    entry = p(rows[entry_i], "o")
    r = risk_pct / 100.0
    stop = entry * (1 - r)
    target = entry * (1 + 3.25 * r)
    add_price = entry * (1 + r)
    trigger_pct = risk_pct * 2.5
    trail_distance_pct = risk_pct * 1.5

    peak = entry
    added = False
    legs = [(entry, 0.5)]
    max_i = min(len(rows) - 1, entry_i + 96)

    for i in range(entry_i, max_i + 1):
        bar = rows[i]
        op, hi, lo, cl = p(bar, "o"), p(bar, "h"), p(bar, "l"), p(bar, "c")

        effective_stop = stop
        gain_pct = (peak / entry - 1) * 100.0
        if gain_pct >= trigger_pct:
            effective_stop = max(effective_stop, peak * (1 - trail_distance_pct / 100.0), entry)

        if op <= effective_stop:
            return i, finish(legs, op), "STOP_GAP", added
        stop_hit = lo <= effective_stop
        target_hit = hi >= target
        if stop_hit:
            return i, finish(legs, effective_stop), "STOP_SAME_BAR" if target_hit else "STOP_OR_TRAIL", added
        if target_hit:
            return i, finish(legs, target), "TARGET", added
        if i == max_i:
            return i, finish(legs, cl), "MAX_HOLD", added

        if not added and hi >= add_price:
            legs.append((add_price, 0.5))
            added = True
        peak = max(peak, hi, cl)

    return None


def summarize(trades):
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    gross_win = 0.0
    gross_loss = 0.0
    wins = 0
    by_symbol = {}

    for tr in sorted(trades, key=lambda x: x.signal_time):
        r = tr.net_return
        equity *= 1 + r
        peak = max(peak, equity)
        max_dd = max(max_dd, 1 - equity / peak)
        if r > 0:
            wins += 1
            gross_win += r
        elif r < 0:
            gross_loss += abs(r)

        b = by_symbol.setdefault(tr.symbol, {"equity": 1.0, "gw": 0.0, "gl": 0.0, "trades": 0})
        b["equity"] *= 1 + r
        b["trades"] += 1
        if r > 0:
            b["gw"] += r
        elif r < 0:
            b["gl"] += abs(r)

    return {
        "trades": len(trades),
        "return_pct": round((equity - 1) * 100, 3),
        "profit_factor": round(gross_win / gross_loss, 3) if gross_loss else (999.0 if gross_win else 0.0),
        "win_rate_pct": round(wins / len(trades) * 100, 2) if trades else 0.0,
        "max_drawdown_pct": round(max_dd * 100, 3),
        "by_symbol": {
            s: {
                "trades": x["trades"],
                "return_pct": round((x["equity"] - 1) * 100, 3),
                "profit_factor": round(x["gw"] / x["gl"], 3) if x["gl"] else (999.0 if x["gw"] else 0.0),
            }
            for s, x in by_symbol.items()
        },
    }


def run_variant(window_name, start, end, bars, daily, variant):
    btc = bars[BTC]
    btc_index = {dt(b): i for i, b in enumerate(btc)}
    trades = []
    raw_signals = 0
    state_ready = 0
    allowed_raw = 0

    for symbol in TARGETS:
        rows = bars[symbol]
        i = 129
        while i < len(rows) - 1:
            signal_time = dt(rows[i])
            if signal_time < start:
                i += 1
                continue
            if signal_time >= end:
                break

            bi = btc_index.get(signal_time)
            if bi is None:
                i += 1
                continue
            sig = c62_signal(rows, i, btc, bi)
            if sig is None:
                i += 1
                continue

            raw_signals += 1
            state = prior_daily_state(signal_time, daily)
            if state is not None:
                state_ready += 1
            if allowed(variant, state):
                allowed_raw += 1
            else:
                i += 1
                continue

            sim = simulate_trade(rows, i, sig["risk_pct"])
            if sim is None:
                i += 1
                continue
            exit_i, net_return, reason, added = sim
            trades.append(Trade(
                symbol=symbol,
                signal_time=signal_time,
                entry_time=dt(rows[i + 1]),
                exit_time=dt(rows[exit_i]),
                net_return=net_return,
                reason=reason,
                added=added,
            ))
            # Position blocked through exit; enforce one full 15m cooldown after exit bar.
            i = exit_i + 2

    out = summarize(trades)
    out.update({
        "window": window_name,
        "variant": variant,
        "raw_signals": raw_signals,
        "state_ready_signals": state_ready,
        "raw_signals_allowed": allowed_raw,
        "raw_signal_allow_rate_pct": round(allowed_raw / state_ready * 100, 2) if state_ready else 0.0,
    })
    return out


def fetch_window(api, start, end):
    warm15 = start - timedelta(days=3)
    warmdaily = start - timedelta(days=240)
    bars = api.bars(SYMBOLS, "15Min", warm15, end)
    daily = api.bars((BTC,), "1Day", warmdaily, end)[BTC]
    return bars, daily


def main():
    print("C62 x V31 HARD-BRAKE RESEARCH — NO ORDERS")
    api = AlpacaCrypto()
    variants = ("baseline", "trend_cash_hard", "fast_brake_hard", "two_stage_hard")
    results = {}

    for window_name, (start, end) in WINDOWS.items():
        print(f"Fetching {window_name}: {start.date()} -> {end.date()}")
        bars, daily = fetch_window(api, start, end)
        print("15m counts:", {s: len(bars[s]) for s in SYMBOLS}, "daily BTC:", len(daily))
        results[window_name] = {}
        for variant in variants:
            row = run_variant(window_name, start, end, bars, daily, variant)
            results[window_name][variant] = row
            print(json.dumps(row, sort_keys=True))

    baseline_old = results["older"]["baseline"]
    baseline_recent = results["recent"]["baseline"]
    comparisons = {}
    for variant in variants[1:]:
        old = results["older"][variant]
        recent = results["recent"][variant]
        comparisons[variant] = {
            "older_return_improvement_pct_points": round(old["return_pct"] - baseline_old["return_pct"], 3),
            "older_trade_capture_pct": round(old["trades"] / baseline_old["trades"] * 100, 2) if baseline_old["trades"] else 0,
            "recent_return_change_pct_points": round(recent["return_pct"] - baseline_recent["return_pct"], 3),
            "recent_trade_capture_pct": round(recent["trades"] / baseline_recent["trades"] * 100, 2) if baseline_recent["trades"] else 0,
        }

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "orders_placed": False,
        "frozen_strategy": "C62 source-recovered lead-lag + current staged/target/trailing exit",
        "source_rule": {
            "v31_fast_days": 63,
            "v31_slow_days": 126,
            "v31_trend_sma_days": 200,
            "v31_vol_lookback_days": 20,
            "v31_vol_annualization": "std(log daily returns, ddof=1) * sqrt(252) * 100",
            "trend_cash_hard": "block when BTC close<SMA200 and momentum126<0",
            "fast_brake_hard": "block when (close<SMA200 and momentum63<0) OR momentum63<-6% OR (vol20>30% and momentum63<0)",
            "two_stage_hard": "block when broken AND shock",
            "note": "Only V31 source-defined hard cash predicates are ported to BTC. No crypto breadth proxy is invented.",
        },
        "results": results,
        "comparisons": comparisons,
        "baseline_provenance_targets": {
            "older_saved": {"trades": 60, "return_pct": -44.37, "profit_factor": 0.384},
            "recent_consistent_replay_saved": {"trades": 33, "return_pct": 0.70, "profit_factor": 1.053},
        },
    }

    OUT.mkdir(exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print("FINAL_SUMMARY=" + json.dumps(summary, separators=(",", ":")))
    print("NO ORDERS WERE PLACED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
