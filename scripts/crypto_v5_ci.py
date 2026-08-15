#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import math
import sys
import zipfile
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "scripts" / "research_crypto_rotation.py"
SPEC = importlib.util.spec_from_file_location("crypto_base", BASE)
mod = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = mod
SPEC.loader.exec_module(mod)

# V5 keeps V4's broad search, but makes risk control asymmetric:
# exit risk on the first completed non-UP BTC signal, while re-entry needs
# either 2 or 4 consecutive UP signals. The two confirmation lengths are
# encoded as regime-profile names so the existing V4 grid/evaluator can be reused.
mod.REGIME_PROFILE_GRID = ("strict2", "strict4")
mod.REGIME_PROFILES = {
    "strict2": {"up": 1.0, "sideways": 0.0, "down": 0.0},
    "strict4": {"up": 1.0, "sideways": 0.0, "down": 0.0},
}
mod.PACK_DIRNAME = "CryptoRotationV5_research_pack"


def confirm_days(config) -> int:
    return 4 if str(config.regime_profile).endswith("4") else 2


def confirmed_risk_on(matrices, signal_index: int, config) -> bool:
    n = confirm_days(config)
    start = signal_index - n + 1
    if start < 0:
        return False
    window = matrices.btc_regime[start : signal_index + 1]
    return bool(len(window) == n and np.all(window == "up"))


def exposure_for_signal_v5(matrices, signal_index: int, config):
    if signal_index < 0:
        return 0.0, "sideways", float("nan")
    regime = str(matrices.btc_regime[signal_index])
    vol = float(matrices.btc_realized_vol_pct[signal_index])
    if regime != "up" or not confirmed_risk_on(matrices, signal_index, config):
        return 0.0, regime, vol
    if math.isfinite(vol) and vol > 0:
        vol_scale = min(1.0, max(0.0, float(config.vol_target_pct) / vol))
    else:
        vol_scale = 0.50
    return vol_scale, regime, vol


mod.exposure_for_signal = exposure_for_signal_v5


def run_backtest_v5(
    matrices,
    config,
    *,
    trade_start_index: int,
    initial_equity: float = mod.INITIAL_EQUITY,
):
    dates = matrices.dates
    n, m = matrices.closes.shape
    if trade_start_index <= 0 or trade_start_index >= n - 2:
        raise ValueError("Invalid trade_start_index.")

    cash = float(initial_equity)
    qty = np.zeros(m, dtype=float)
    rows = []
    cycle_rows = []
    last_rebalance_i: Optional[int] = None
    cycle_start_equity: Optional[float] = None
    cycle_start_date: Optional[pd.Timestamp] = None
    cycle_active = False
    turnover = 0.0
    rebalance_count = 0
    cost_rate = float(config.cost_bps) / 10000.0
    previous_signal_risk_on: Optional[bool] = None

    for i in range(trade_start_index, n):
        date = dates[i]
        op = matrices.valuation_opens[i].copy()
        cp = matrices.valuation_closes[i].copy()
        op = np.where(np.isfinite(op) & (op > 0), op, 0.0)
        cp = np.where(np.isfinite(cp) & (cp > 0), cp, op)
        pretrade_equity = cash + float(np.dot(qty, op))

        signal_i = i - 1
        signal_exposure, regime, realized_vol = exposure_for_signal_v5(
            matrices, signal_i, config
        )
        signal_risk_on = signal_exposure > 1e-12

        scheduled_due = (
            last_rebalance_i is None
            or (i - last_rebalance_i) >= int(config.rebalance_days)
        )
        risk_flip_due = (
            previous_signal_risk_on is not None
            and signal_risk_on != previous_signal_risk_on
        )
        due = bool(scheduled_due or risk_flip_due)
        traded_notional = 0.0
        target_exposure = signal_exposure if due else np.nan
        reason = "risk_flip" if risk_flip_due else ("scheduled" if scheduled_due else "")

        if due:
            weights, regime, realized_vol, target_exposure = mod.target_weights(
                matrices, signal_i, config
            )

            # Only invested periods count as strategy cycles. Cash waiting time is not
            # treated as a losing trade/cycle.
            if cycle_active and cycle_start_equity is not None and cycle_start_equity > 0:
                cycle_rows.append({
                    "start_date": cycle_start_date,
                    "end_date": date,
                    "return": pretrade_equity / cycle_start_equity - 1.0,
                    "pnl": pretrade_equity - cycle_start_equity,
                })

            current_values = qty * op
            desired_values = pretrade_equity * weights
            delta = desired_values - current_values
            tradable = np.isfinite(matrices.opens[i]) & (matrices.opens[i] > 0)

            # Reductions happen first. A risk-off flip therefore liquidates at the
            # next daily open rather than waiting for the next scheduled rotation.
            sell_idx = np.flatnonzero((delta < -1e-9) & tradable & (qty > 0))
            for j in sell_idx:
                price = float(matrices.opens[i, j])
                sell_value = min(float(current_values[j]), float(-delta[j]))
                sell_qty = min(float(qty[j]), sell_value / price)
                actual_value = sell_qty * price
                fee = actual_value * cost_rate
                qty[j] = max(0.0, qty[j] - sell_qty)
                cash += actual_value - fee
                traded_notional += actual_value

            current_values = qty * op
            delta = desired_values - current_values
            buy_idx = np.flatnonzero((delta > 1e-9) & tradable)
            if buy_idx.size:
                buy_idx = buy_idx[np.argsort(desired_values[buy_idx])[::-1]]
            for j in buy_idx:
                price = float(matrices.opens[i, j])
                max_buy = cash / (1.0 + cost_rate) if cost_rate >= 0 else cash
                buy_value = max(0.0, min(float(delta[j]), max_buy))
                if buy_value <= 0:
                    continue
                fee = buy_value * cost_rate
                qty[j] += buy_value / price
                cash -= buy_value + fee
                traded_notional += buy_value

            posttrade_equity = cash + float(np.dot(qty, op))
            cycle_active = bool(np.any(qty > 1e-12))
            cycle_start_equity = posttrade_equity if cycle_active else None
            cycle_start_date = date if cycle_active else None
            last_rebalance_i = i
            rebalance_count += 1

        close_equity = cash + float(np.dot(qty, cp))
        rows.append({
            "date": date,
            "equity": close_equity,
            "cash": cash,
            "positions": int(np.sum(qty > 1e-12)),
            "rebalanced": bool(due),
            "rebalance_reason": reason,
            "turnover_notional": traded_notional,
            "btc_regime": regime,
            "btc_realized_vol_pct": realized_vol if math.isfinite(realized_vol) else np.nan,
            "target_exposure": target_exposure,
        })
        turnover += traded_notional
        previous_signal_risk_on = signal_risk_on

    if cycle_active and cycle_start_equity is not None and cycle_start_equity > 0 and rows:
        final_equity = float(rows[-1]["equity"])
        cycle_rows.append({
            "start_date": cycle_start_date,
            "end_date": pd.Timestamp(rows[-1]["date"]),
            "return": final_equity / cycle_start_equity - 1.0,
            "pnl": final_equity - cycle_start_equity,
        })

    return mod.BacktestResult(
        equity=pd.DataFrame(rows),
        cycles=pd.DataFrame(cycle_rows),
        turnover=float(turnover),
        rebalance_count=int(rebalance_count),
        initial_equity=float(initial_equity),
        evaluation_start=pd.Timestamp(dates[trade_start_index]),
    )


mod.run_backtest = run_backtest_v5


def synthetic_frames(periods=800):
    dates = pd.date_range("2022-01-01", periods=periods, freq="D", tz="UTC")
    frames: Dict[str, pd.DataFrame] = {}
    params = {
        "BTC/USD": 0.0010,
        "ETH/USD": 0.0014,
        "SOL/USD": 0.0018,
        "AVAX/USD": 0.0012,
        "LINK/USD": 0.0011,
    }
    for k, (symbol, drift) in enumerate(params.items()):
        wiggle = 0.003 * np.sin(np.arange(periods) / (11.0 + k))
        rets = drift + wiggle
        close = 100.0 * np.cumprod(1.0 + rets)
        open_ = np.r_[close[0], close[:-1]]
        volume = np.full(periods, 500000.0 + k * 10000)
        frames[symbol] = pd.DataFrame(
            {"date": dates, "open": open_, "close": close, "volume": volume}
        )
    return frames


def self_test() -> None:
    matrices = mod.build_matrices(synthetic_frames())
    matrices.btc_regime = np.full(len(matrices.dates), "up", dtype=object)
    matrices.btc_realized_vol_pct = np.full(len(matrices.dates), 30.0, dtype=float)
    cfg = mod.CryptoConfig(
        momentum_days=63,
        sma_days=150,
        rebalance_days=10,
        top_k=4,
        weighting="equal",
        regime_profile="strict2",
        vol_target_pct=45.0,
    )
    start = mod.WARMUP_DAYS
    matrices.btc_regime[start + 15] = "sideways"
    result = run_backtest_v5(matrices, cfg, trade_start_index=start)
    row = result.equity[result.equity["date"] == matrices.dates[start + 16]].iloc[0]
    assert bool(row["rebalanced"])
    assert row["rebalance_reason"] == "risk_flip"
    assert int(row["positions"]) == 0

    # Cash-only periods should not manufacture zero-return losing cycles.
    matrices2 = mod.build_matrices(synthetic_frames())
    matrices2.btc_regime = np.full(len(matrices2.dates), "sideways", dtype=object)
    matrices2.btc_realized_vol_pct = np.full(len(matrices2.dates), 30.0, dtype=float)
    result2 = run_backtest_v5(matrices2, cfg, trade_start_index=mod.WARMUP_DAYS)
    assert result2.cycles.empty
    print("V5 CI self-test passed.")


def relabel_outputs() -> None:
    out = ROOT / mod.PACK_DIRNAME
    old_summary = out / "crypto_rotation_v4_summary.json"
    if old_summary.exists():
        data = json.loads(old_summary.read_text(encoding="utf-8"))
        data["strategy"] = "CRYPTO_ROTATION_V5_CI_RESEARCH"
        data["v5_overrides"] = {
            "risk_off": "first completed non-UP BTC signal -> next-open liquidation",
            "risk_on_confirmation": [2, 4],
            "cash_cycles_excluded": True,
        }
        new_summary = out / "crypto_rotation_v5_summary.json"
        new_summary.write_text(json.dumps(data, indent=2), encoding="utf-8")

    readme = out / "README.md"
    if readme.exists():
        text = readme.read_text(encoding="utf-8")
        text = text.replace("Crypto Rotation V4", "Crypto Rotation V5 CI")
        readme.write_text(text, encoding="utf-8")

    zip_path = ROOT / f"{mod.PACK_DIRNAME}.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(out.rglob("*")):
            if path.is_file():
                zf.write(path, arcname=f"{out.name}/{path.relative_to(out)}")


def main() -> int:
    if "--self-test" in sys.argv:
        self_test()
        return 0
    rc = mod.main()
    relabel_outputs()
    return int(rc)


if __name__ == "__main__":
    raise SystemExit(main())
