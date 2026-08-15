#!/usr/bin/env python3
"""
Frozen V26 Alpaca PAPER runner.

Strategy:
- Universe: SPY, QQQ, IWM, DIA, XLK, XLF, XLE, GLD, TLT
- 63-trading-day momentum
- Eligible only when momentum > 0 and close > SMA150
- Rank eligible ETFs by momentum
- Every 5 completed trading sessions:
    #1 = 70%
    #2 = 30%
  If only one ETF is eligible, #1 gets 70% and 30% stays cash.
  If none are eligible, remain in cash.
- Orders are sent ONLY to Alpaca's paper endpoint.
- Regular-session market DAY orders submitted outside market hours are queued
  for the next regular trading session by Alpaca.

First run is a preview unless --execute is supplied. Existing non-V26 paper positions are preserved.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import pandas as pd
import requests

UNIVERSE = ("SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "GLD", "TLT")
MOMENTUM_DAYS = 63
SMA_DAYS = 150
REBALANCE_EVERY = 5
TOP_WEIGHTS = (0.70, 0.30)

PAPER_BASE = "https://paper-api.alpaca.markets"
DATA_BASE = "https://data.alpaca.markets"

STATE_DIR = Path("paper")
STATE_PATH = STATE_DIR / "v26_state.json"
EVENT_LOG = STATE_DIR / "v26_events.jsonl"


@dataclass(frozen=True)
class SignalRow:
    symbol: str
    close: float
    momentum: float
    sma150: float
    eligible: bool


def _fmt_num(x: float, places: int = 8) -> str:
    s = f"{float(x):.{places}f}".rstrip("0").rstrip(".")
    return s if s else "0"


def load_state(path: Path = STATE_PATH) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Could not read state file {path}: {exc}") from exc


def save_state(state: dict, path: Path = STATE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def log_event(event: dict, path: Path = EVENT_LOG) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"logged_at_utc": datetime.now(timezone.utc).isoformat(), **event}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True, default=str) + "\n")


def common_sessions(bars_by_symbol: Mapping[str, pd.DataFrame]) -> List[str]:
    sets = []
    for symbol in UNIVERSE:
        df = bars_by_symbol[symbol]
        sets.append(set(df["session"].astype(str)))
    if not sets:
        return []
    return sorted(set.intersection(*sets))


def compute_targets(
    bars_by_symbol: Mapping[str, pd.DataFrame],
) -> Tuple[str, Dict[str, float], List[SignalRow]]:
    sessions = common_sessions(bars_by_symbol)
    if not sessions:
        raise RuntimeError("No common trading sessions across the V26 universe.")
    session = sessions[-1]

    rows: List[SignalRow] = []
    for symbol in UNIVERSE:
        df = (
            bars_by_symbol[symbol]
            .loc[lambda x: x["session"].astype(str) <= session]
            .sort_values("session")
            .drop_duplicates("session", keep="last")
            .copy()
        )
        if len(df) < SMA_DAYS:
            raise RuntimeError(
                f"{symbol} has only {len(df)} completed sessions; need at least {SMA_DAYS}."
            )

        closes = pd.to_numeric(df["close"], errors="coerce").dropna().reset_index(drop=True)
        if len(closes) < max(SMA_DAYS, MOMENTUM_DAYS + 1):
            raise RuntimeError(f"{symbol} does not have enough valid closes.")

        close = float(closes.iloc[-1])
        past = float(closes.iloc[-1 - MOMENTUM_DAYS])
        momentum = close / past - 1.0
        sma150 = float(closes.iloc[-SMA_DAYS:].mean())
        eligible = bool(momentum > 0.0 and close > sma150)

        rows.append(
            SignalRow(
                symbol=symbol,
                close=close,
                momentum=momentum,
                sma150=sma150,
                eligible=eligible,
            )
        )

    rows.sort(key=lambda r: r.momentum, reverse=True)
    eligible = [r for r in rows if r.eligible]

    targets: Dict[str, float] = {}
    if eligible:
        targets[eligible[0].symbol] = TOP_WEIGHTS[0]
    if len(eligible) >= 2:
        targets[eligible[1].symbol] = TOP_WEIGHTS[1]

    return session, targets, rows


def rebalance_due(
    all_sessions: Iterable[str],
    last_rebalance_session: Optional[str],
    every: int = REBALANCE_EVERY,
) -> Tuple[bool, int]:
    sessions = sorted(set(str(x) for x in all_sessions))
    if not sessions:
        return False, 0
    if not last_rebalance_session:
        return True, every
    elapsed = sum(s > str(last_rebalance_session) for s in sessions)
    return elapsed >= every, elapsed


class AlpacaHTTP:
    def __init__(self) -> None:
        self.key = os.getenv("ALPACA_PAPER_API_KEY") or os.getenv("ALPACA_API_KEY")
        self.secret = os.getenv("ALPACA_PAPER_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY")
        if not self.key or not self.secret:
            raise SystemExit(
                "Paper credentials missing. Set ALPACA_PAPER_API_KEY and "
                "ALPACA_PAPER_SECRET_KEY in this PowerShell window."
            )
        self.headers = {
            "APCA-API-KEY-ID": self.key,
            "APCA-API-SECRET-KEY": self.secret,
        }

    def _request(self, method: str, base: str, path: str, **kwargs):
        headers = dict(self.headers)
        headers.update(kwargs.pop("headers", {}))
        r = requests.request(
            method,
            base + path,
            headers=headers,
            timeout=45,
            **kwargs,
        )
        if not r.ok:
            body = r.text[:1000]
            raise RuntimeError(f"Alpaca {method} {path} failed {r.status_code}: {body}")
        if not r.content:
            return None
        return r.json()

    def account(self) -> dict:
        return self._request("GET", PAPER_BASE, "/v2/account")

    def clock(self) -> dict:
        return self._request("GET", PAPER_BASE, "/v2/clock")

    def positions(self) -> List[dict]:
        return self._request("GET", PAPER_BASE, "/v2/positions")

    def open_orders(self) -> List[dict]:
        return self._request(
            "GET", PAPER_BASE, "/v2/orders",
            params={"status": "open", "limit": 500, "direction": "desc"},
        )

    def submit_order(self, payload: dict) -> dict:
        return self._request("POST", PAPER_BASE, "/v2/orders", json=payload)

    def daily_bars(self, symbol: str, lookback_calendar_days: int = 520) -> pd.DataFrame:
        now = datetime.now(timezone.utc)
        start = now - timedelta(days=lookback_calendar_days)
        params = {
            "start": start.isoformat().replace("+00:00", "Z"),
            "end": now.isoformat().replace("+00:00", "Z"),
            "timeframe": "1Day",
            "feed": "iex",
            "adjustment": "all",
            "sort": "asc",
            "limit": 10000,
        }
        rows: List[dict] = []
        token = None
        while True:
            if token:
                params["page_token"] = token
            else:
                params.pop("page_token", None)
            payload = self._request(
                "GET", DATA_BASE, f"/v2/stocks/{symbol}/bars", params=params
            )
            rows.extend(payload.get("bars", []))
            token = payload.get("next_page_token")
            if not token:
                break

        if not rows:
            raise RuntimeError(f"No daily bars returned for {symbol}.")

        df = pd.DataFrame(rows).rename(
            columns={
                "t": "timestamp",
                "o": "open",
                "h": "high",
                "l": "low",
                "c": "close",
                "v": "volume",
            }
        )
        ts = pd.to_datetime(df["timestamp"], utc=True)
        df["session"] = ts.dt.tz_convert("America/New_York").dt.date.astype(str)
        return df[["session", "open", "high", "low", "close", "volume"]].copy()


def _exclude_incomplete_session(
    bars_by_symbol: Dict[str, pd.DataFrame],
    clock: Mapping[str, object],
) -> Dict[str, pd.DataFrame]:
    if not bool(clock.get("is_open")):
        return bars_by_symbol

    market_now = pd.Timestamp(str(clock["timestamp"]))
    market_date = market_now.tz_convert("America/New_York").date().isoformat()

    cleaned = {}
    for symbol, df in bars_by_symbol.items():
        cleaned[symbol] = df.loc[df["session"].astype(str) < market_date].copy()
    return cleaned


def print_signal(session: str, targets: Mapping[str, float], rows: List[SignalRow]) -> None:
    print(f"\nV26 signal from completed session: {session}")
    print("Frozen rule: momentum63 + SMA150, top two, weights 70% / 30%\n")
    print(f"{'SYMBOL':<7}{'MOM63':>10}{'CLOSE':>12}{'SMA150':>12}{'ELIGIBLE':>11}")
    for r in rows:
        print(
            f"{r.symbol:<7}{r.momentum*100:>9.2f}%"
            f"{r.close:>12.2f}{r.sma150:>12.2f}{str(r.eligible):>11}"
        )

    if targets:
        print("\nTARGETS:")
        for symbol, weight in targets.items():
            print(f"  {symbol}: {weight*100:.0f}%")
        cash = max(0.0, 1.0 - sum(targets.values()))
        if cash > 1e-9:
            print(f"  CASH: {cash*100:.0f}%")
    else:
        print("\nTARGETS: 100% CASH")


def _position_map(positions: Iterable[dict]) -> Dict[str, dict]:
    return {p["symbol"]: p for p in positions if p.get("symbol") in UNIVERSE}


def build_orders(
    session: str,
    targets: Mapping[str, float],
    positions: Iterable[dict],
    account_equity: float,
    min_notional: float = 5.0,
) -> List[dict]:
    pos = _position_map(positions)
    orders: List[dict] = []

    for symbol in UNIVERSE:
        target_value = account_equity * float(targets.get(symbol, 0.0))
        p = pos.get(symbol)
        current_value = float(p.get("market_value", 0.0)) if p else 0.0
        delta = target_value - current_value

        if abs(delta) < min_notional:
            continue

        side = "buy" if delta > 0 else "sell"
        client_order_id = f"v26-{session.replace('-', '')}-{symbol}-{side}"

        if side == "buy":
            orders.append(
                {
                    "symbol": symbol,
                    "notional": _fmt_num(delta, 2),
                    "side": "buy",
                    "type": "market",
                    "time_in_force": "day",
                    "extended_hours": False,
                    "client_order_id": client_order_id,
                }
            )
        else:
            if not p:
                continue
            qty = float(p.get("qty", 0.0))
            current_price = abs(float(p.get("current_price") or 0.0))
            if qty <= 0 or current_price <= 0:
                continue

            if target_value <= min_notional:
                sell_qty = qty
            else:
                sell_qty = min(qty, abs(delta) / current_price)

            if sell_qty * current_price < min_notional:
                continue

            orders.append(
                {
                    "symbol": symbol,
                    "qty": _fmt_num(sell_qty, 9),
                    "side": "sell",
                    "type": "market",
                    "time_in_force": "day",
                    "extended_hours": False,
                    "client_order_id": client_order_id,
                }
            )

    orders.sort(key=lambda o: 0 if o["side"] == "sell" else 1)
    return orders


def _check_account_safety(
    account: Mapping[str, object],
    positions: List[dict],
    open_orders: List[dict],
    state: Mapping[str, object],
) -> None:
    if str(account.get("status", "")).upper() != "ACTIVE":
        raise RuntimeError(f"Paper account status is {account.get('status')!r}, not ACTIVE.")
    if bool(account.get("trading_blocked")):
        raise RuntimeError("Paper account says trading_blocked=true.")

    # Sharing the paper account is allowed, but V26 will never touch symbols outside
    # its own ETF universe. To avoid conflicting automation, block any non-V26 open
    # order that is already working in a V26-universe symbol.
    conflicting_orders = [
        o for o in open_orders
        if o.get("symbol") in UNIVERSE
        and not str(o.get("client_order_id", "")).startswith("v26-")
    ]
    if conflicting_orders:
        symbols = ", ".join(sorted({o.get("symbol", "?") for o in conflicting_orders}))
        raise RuntimeError(
            "Found non-V26 open order(s) in V26 universe symbol(s): "
            f"{symbols}. Cancel those paper orders first so two strategies do not "
            "fight over the same ETF."
        )

    # Existing positions outside UNIVERSE are deliberately ignored and preserved.
    # Existing positions inside UNIVERSE are treated as V26 inventory and may be
    # rebalanced, because there is no reliable way to distinguish ownership after fill.


def main() -> int:
    p = argparse.ArgumentParser(description="Frozen V26 Alpaca PAPER rotation runner")
    p.add_argument("--execute", action="store_true")
    p.add_argument(
        "--budget",
        type=float,
        default=100000.0,
        help=(
            "Maximum paper dollars assigned to V26. Existing non-V26 positions are "
            "preserved. Actual allocation is capped by cash plus current V26 positions."
        ),
    )
    p.add_argument("--force-rebalance", action="store_true")
    p.add_argument("--min-order-notional", type=float, default=5.0)
    args = p.parse_args()

    api = AlpacaHTTP()
    account = api.account()
    clock = api.clock()
    positions = api.positions()
    open_orders = api.open_orders()
    state = load_state()

    _check_account_safety(account, positions, open_orders, state)

    print("CONNECTED TO ALPACA PAPER ENDPOINT ONLY")
    print(f"Account status: {account.get('status')}")
    print(f"Paper equity: ${float(account.get('equity', 0.0)):,.2f}")
    print(f"Market open now: {clock.get('is_open')}")
    print(f"Next market open: {clock.get('next_open')}")

    bars = {symbol: api.daily_bars(symbol) for symbol in UNIVERSE}
    bars = _exclude_incomplete_session(bars, clock)

    session, targets, rows = compute_targets(bars)
    sessions = common_sessions(bars)
    due, elapsed = rebalance_due(sessions, state.get("last_rebalance_session"))
    if args.force_rebalance:
        due = True

    print_signal(session, targets, rows)
    print(
        f"\nLast V26 rebalance session: "
        f"{state.get('last_rebalance_session', 'NONE (first run)')}"
    )
    print(f"Completed sessions since last rebalance: {elapsed}")
    print(f"Rebalance due: {due}")

    if not due:
        print("\nNo V26 order is due yet.")
        return 0

    requested_budget = float(args.budget)
    if requested_budget <= 0:
        raise RuntimeError("--budget must be positive.")

    cash = float(account.get("cash") or 0.0)
    v26_positions = _position_map(positions)
    v26_market_value = sum(
        max(0.0, float(p.get("market_value", 0.0)))
        for p in v26_positions.values()
    )
    allocatable = max(0.0, cash + v26_market_value)
    strategy_budget = min(requested_budget, allocatable)
    if strategy_budget <= 0:
        raise RuntimeError(
            "No allocatable paper capital is available for V26. "
            "Cash plus current V26-universe market value is zero."
        )

    print(f"Requested V26 budget: ${requested_budget:,.2f}")
    print(f"Available to V26 now: ${allocatable:,.2f}")
    print(f"V26 strategy budget used: ${strategy_budget:,.2f}")

    orders = build_orders(
        session=session,
        targets=targets,
        positions=positions,
        account_equity=strategy_budget,
        min_notional=args.min_order_notional,
    )

    print("\nPROPOSED PAPER ORDERS:")
    if not orders:
        print("  No order deltas required.")
    for order in orders:
        amount = (
            f"${order['notional']} notional"
            if "notional" in order
            else f"{order['qty']} shares"
        )
        print(f"  {order['side'].upper():4} {order['symbol']:<4} {amount}")

    if not args.execute:
        print("\nPREVIEW ONLY. Nothing was submitted.")
        print("Run again with --execute to submit these orders to ALPACA PAPER.")
        return 0

    submitted = []
    failures = []
    for order in orders:
        try:
            result = api.submit_order(order)
            submitted.append(
                {
                    "symbol": order["symbol"],
                    "side": order["side"],
                    "client_order_id": order["client_order_id"],
                    "alpaca_order_id": result.get("id"),
                    "status": result.get("status"),
                }
            )
            print(
                f"SUBMITTED PAPER: {order['side'].upper()} {order['symbol']} "
                f"status={result.get('status')}"
            )
        except Exception as exc:
            failures.append(
                {
                    "symbol": order["symbol"],
                    "side": order["side"],
                    "client_order_id": order["client_order_id"],
                    "error": str(exc),
                }
            )
            print(f"FAILED: {order['side'].upper()} {order['symbol']}: {exc}")

    new_state = {
        **state,
        "strategy": "V26_FROZEN_ROTATION",
        "universe": list(UNIVERSE),
        "momentum_days": MOMENTUM_DAYS,
        "sma_days": SMA_DAYS,
        "rebalance_every_completed_sessions": REBALANCE_EVERY,
        "weights": list(TOP_WEIGHTS),
        "last_signal_session": session,
        "last_targets": targets,
        "last_run_utc": datetime.now(timezone.utc).isoformat(),
        "requested_budget": requested_budget,
        "strategy_budget": strategy_budget,
        "last_submitted_orders": submitted,
        "last_failures": failures,
    }
    if not failures:
        new_state["last_rebalance_session"] = session

    save_state(new_state)
    log_event(
        {
            "event": "rebalance",
            "session": session,
            "targets": targets,
            "submitted": submitted,
            "failures": failures,
            "paper_account_equity": float(account.get("equity") or 0.0),
            "v26_strategy_budget": strategy_budget,
            "market_open": clock.get("is_open"),
            "next_open": clock.get("next_open"),
        }
    )

    if failures:
        print(
            "\nSome PAPER orders failed. State was saved, but the rebalance cadence "
            "was NOT advanced. Review the errors before rerunning."
        )
        return 2

    if not orders:
        print("\nRebalance complete: portfolio already matched the V26 target.")
    elif bool(clock.get("is_open")):
        print("\nPaper orders were submitted during the regular session.")
    else:
        print(
            "\nPaper orders were submitted while the market is closed. "
            "Alpaca will queue regular-session DAY orders for the next trading session."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
