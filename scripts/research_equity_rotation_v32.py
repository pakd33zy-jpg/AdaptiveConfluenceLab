#!/usr/bin/env python3
from __future__ import annotations

import itertools
import json
import math
import os
import sys
import types
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import research_equity_rotation_v31 as v31

base = v31.base
VERSION = "V32"
PACK_DIRNAME = "EquityRotationV32_research_pack"
FAST, SLOW, TREND = 63, 126, 200

# Focused structural tournament. V31 showed that equal-weight, 20-session
# rebalancing and the trend-cash family were the strongest robust neighborhood.
# V32 keeps a V31 control and tests only two general transition-risk changes.
FAMILIES = ("v31_control", "drawdown_guard", "confirmed_trend")
TOP_K = (4, 6)
REBALANCE = (20,)
WEIGHTINGS = ("equal",)
VOL_TARGETS = (10.0, 12.0, 14.0, 16.0)
QUALIFICATION_RULES = v31.QUALIFICATION_RULES


def _spy_index(m):
    return m.symbols.index("SPY")


def _market_state(m, i):
    j = _spy_index(m)
    close = float(m.closes[i, j])
    sma = float(m.sma[TREND][i, j])
    fast = float(m.momentum[FAST][i, j])
    slow = float(m.momentum[SLOW][i, j])
    vol = float(m.spy_realized_vol_pct[i])
    breadth = float(v31.v30._breadth(m, i))

    lo = max(0, i - FAST + 1)
    trail = np.asarray(m.closes[lo:i + 1, j], dtype=float)
    valid_trail = trail[np.isfinite(trail) & (trail > 0)]
    peak63 = float(valid_trail.max()) if valid_trail.size else close
    drawdown63 = close / peak63 - 1.0 if peak63 > 0 and np.isfinite(close) else 0.0

    c10 = max(0, i - 9)
    closes10 = np.asarray(m.closes[c10:i + 1, j], dtype=float)
    sma10 = np.asarray(m.sma[TREND][c10:i + 1, j], dtype=float)
    fast10 = np.asarray(m.momentum[FAST][c10:i + 1, j], dtype=float)
    valid = np.isfinite(closes10) & np.isfinite(sma10) & np.isfinite(fast10)
    if valid.any():
        confirmed = (closes10[valid] > sma10[valid]) & (fast10[valid] > 0)
        trend_confirm10 = float(confirmed.mean())
    else:
        trend_confirm10 = 0.0

    return close, sma, fast, slow, vol, breadth, drawdown63, trend_confirm10


def _risk_fraction_v32(
    family, close, sma, fast, slow, vol, breadth, drawdown63, trend_confirm10
):
    # Keep V31's selected family as a control. This prevents a new structural
    # idea from being accepted merely because every candidate got worse.
    baseline = v31._risk_fraction(
        "trend_cash", close, sma, fast, slow, vol, breadth
    )

    if family == "v31_control":
        return baseline

    if family == "drawdown_guard":
        # General transition protection: progressively cut exposure as SPY
        # falls away from its recent high, especially when fast momentum is
        # negative. No dates or fold-specific constants are used.
        if np.isfinite(drawdown63) and drawdown63 <= -0.12:
            return 0.0
        if (
            np.isfinite(drawdown63)
            and drawdown63 <= -0.08
            and np.isfinite(fast)
            and fast < 0
        ):
            return min(baseline, 0.20 if breadth < 0.50 else 0.35)
        if (
            np.isfinite(drawdown63)
            and drawdown63 <= -0.05
            and np.isfinite(fast)
            and fast < 0
        ):
            return min(baseline, 0.50)
        return baseline

    if family == "confirmed_trend":
        # Re-enter only after a broad, persistent trend is visible. This is
        # intended to reduce whipsaw around major regime transitions.
        hard_break = (
            np.isfinite(close)
            and np.isfinite(sma)
            and np.isfinite(fast)
            and close < sma
            and fast < 0
        )
        if hard_break:
            return 0.0
        if np.isfinite(drawdown63) and drawdown63 <= -0.10:
            return 0.0

        strong = (
            np.isfinite(close)
            and np.isfinite(sma)
            and np.isfinite(slow)
            and close > sma
            and slow > 0
            and trend_confirm10 >= 0.70
        )
        if strong:
            if breadth >= 0.60:
                return 1.0
            if breadth >= 0.45:
                return 0.75
            if breadth >= 0.30:
                return 0.45
            if breadth >= 0.22:
                return 0.15
            return 0.0

        transition_ok = (
            np.isfinite(fast)
            and fast > 0
            and trend_confirm10 >= 0.60
            and breadth >= 0.55
            and (not np.isfinite(drawdown63) or drawdown63 > -0.06)
        )
        return 0.30 if transition_ok else 0.0

    raise ValueError(f"Unknown V32 family: {family}")


def target_weights_v32(m, i, config):
    w = np.zeros(len(m.symbols), dtype=float)
    if i < 0:
        return w, "neutral", "cash", np.nan, 0.0

    close, sma, fast, slow, spyvol, breadth, dd63, confirm10 = _market_state(m, i)
    rf = _risk_fraction_v32(
        config.risk_profile, close, sma, fast, slow, spyvol, breadth, dd63, confirm10
    )
    if rf <= 0:
        return w, str(m.spy_regime[i]), "cash", spyvol, 0.0

    vs = (
        min(1.0, max(0.50, config.vol_target_pct / spyvol))
        if np.isfinite(spyvol) and spyvol > 0
        else 0.65
    )
    gross = min(1.0, rf * vs)

    picks = v31.v30._pick(m, i, "risk", config.top_k, config.weighting)
    if not picks:
        return w, str(m.spy_regime[i]), "cash", spyvol, 0.0

    for j, x in picks:
        w[j] = gross * float(x)

    return w, str(m.spy_regime[i]), "risk", spyvol, float(w.sum())


def exposure_and_mode_v32(m, i, config):
    if i < 0:
        return 0.0, "neutral", "cash", float("nan")

    close, sma, fast, slow, spyvol, breadth, dd63, confirm10 = _market_state(m, i)
    rf = _risk_fraction_v32(
        config.risk_profile, close, sma, fast, slow, spyvol, breadth, dd63, confirm10
    )
    vs = (
        min(1.0, max(0.50, config.vol_target_pct / spyvol))
        if np.isfinite(spyvol) and spyvol > 0
        else 0.65
    )
    exp = min(1.0, rf * vs)
    return exp, str(m.spy_regime[i]), ("risk" if exp > 0 else "cash"), spyvol


# Private backtest execution globals: do not mutate V31/V30/V29 callbacks.
_v32_backtest_globals = dict(base.run_backtest.__globals__)
_v32_backtest_globals["target_weights"] = target_weights_v32
_v32_backtest_globals["exposure_and_mode"] = exposure_and_mode_v32

run_backtest_v32 = types.FunctionType(
    base.run_backtest.__code__,
    _v32_backtest_globals,
    name="run_backtest_v32",
    argdefs=base.run_backtest.__defaults__,
    closure=base.run_backtest.__closure__,
)
run_backtest_v32.__kwdefaults__ = dict(base.run_backtest.__kwdefaults__ or {})


def configs():
    return [
        base.EquityConfig(
            momentum_days=SLOW,
            sma_days=TREND,
            rebalance_days=reb,
            top_k=top_k,
            weighting=weighting,
            risk_profile=family,
            vol_target_pct=vt,
        )
        for family, top_k, reb, weighting, vt in itertools.product(
            FAMILIES, TOP_K, REBALANCE, WEIGHTINGS, VOL_TARGETS
        )
    ]


def _evaluate_config(m, trade_start, folds, c):
    res = run_backtest_v32(m, c, trade_start_index=trade_start)
    mets = [base.segment_metrics(res, *x) for x in folds]
    scores = [base.metric_score(x) for x in mets]
    row = {**asdict(c)}
    for q, met in enumerate(mets, 1):
        for name, val in met.items():
            row[f"fold{q}_{name}"] = val
        row[f"fold{q}_score"] = scores[q - 1]
    cagrs = np.array([x["cagr_pct"] for x in mets], dtype=float)
    row["worst_fold_score"] = float(min(scores))
    row["mean_fold_score"] = float(np.mean(scores))
    row["cagr_dispersion"] = float(np.std(cagrs) / 100.0)
    row["robust_score"] = float(
        min(scores) + 0.35 * np.mean(scores) - 0.25 * np.std(cagrs) / 100.0
    )
    return c, row, res


def _qualification_diagnostics(grid):
    records = []
    survivors = grid.copy()
    funnel = []

    for q in range(1, 5):
        for metric, predicate, threshold in QUALIFICATION_RULES:
            col = f"fold{q}_{metric}"
            vals = pd.to_numeric(grid[col], errors="coerce")
            passed = vals.map(
                lambda x: bool(predicate(x)) if pd.notna(x) else False
            )
            records.append(
                {
                    "fold": q,
                    "metric": metric,
                    "threshold": threshold,
                    "failed_count": int((~passed).sum()),
                    "passed_count": int(passed.sum()),
                    "total_configs": int(len(grid)),
                    "failure_rate_pct": float((~passed).mean() * 100.0),
                }
            )

            svals = pd.to_numeric(survivors[col], errors="coerce")
            smask = svals.map(
                lambda x: bool(predicate(x)) if pd.notna(x) else False
            )
            survivors = survivors[smask]
            funnel.append(
                {
                    "fold": q,
                    "metric": metric,
                    "threshold": threshold,
                    "survivors_after_gate": int(len(survivors)),
                }
            )

    diag = pd.DataFrame(records).sort_values(
        ["failed_count", "fold", "metric"], ascending=[False, True, True]
    ).reset_index(drop=True)
    bottleneck = diag.iloc[0].to_dict() if len(diag) else {}
    return diag, pd.DataFrame(funnel), bottleneck


def select(m, trade_start):
    dates = m.dates[trade_start:]
    folds, diagnostic = base.development_splits(dates)
    cfgs = configs()
    rows, cache = [], {}

    max_workers = max(1, min(4, os.cpu_count() or 2))
    print(f"V32 focused tournament using {max_workers} workers for {len(cfgs)} configs")

    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(_evaluate_config, m, trade_start, folds, c): c for c in cfgs
        }
        for fut in as_completed(futures):
            c, row, res = fut.result()
            rows.append(row)
            cache[c] = res
            completed += 1
            if completed % 4 == 0 or completed == len(cfgs):
                print(f"V32 tournament {completed}/{len(cfgs)}")

    grid = pd.DataFrame(rows).sort_values(
        ["robust_score", "worst_fold_score"], ascending=False
    ).reset_index(drop=True)

    fail_diag, funnel, bottleneck = _qualification_diagnostics(grid)

    eligible = grid.copy()
    for q in range(1, 5):
        eligible = eligible[
            (eligible[f"fold{q}_return_pct"] > 0)
            & (eligible[f"fold{q}_profit_factor"] > 1)
            & (eligible[f"fold{q}_sharpe"] > 0)
            & (eligible[f"fold{q}_max_drawdown_pct"] > -20)
            & (eligible[f"fold{q}_cycles"] >= 6)
        ]

    row = (eligible if len(eligible) else grid).iloc[0]
    c = base.EquityConfig(
        momentum_days=int(row.momentum_days),
        sma_days=int(row.sma_days),
        rebalance_days=int(row.rebalance_days),
        top_k=int(row.top_k),
        weighting=str(row.weighting),
        risk_profile=str(row.risk_profile),
        vol_target_pct=float(row.vol_target_pct),
        cost_bps=float(row.cost_bps),
        max_liquid_assets=int(row.max_liquid_assets),
    )
    res = cache[c]

    report = {
        "config": asdict(c),
        "eligible_development_count": int(len(eligible)),
        "grid_size": int(len(grid)),
        "development_folds": [],
        "historical_diagnostic": base.segment_metrics(res, *diagnostic),
        "full": base.segment_metrics(res, dates[0], dates[-1]),
        "diagnostic_range": [x.isoformat() for x in diagnostic],
        "qualification_bottleneck": bottleneck,
    }
    for q, x in enumerate(folds, 1):
        report["development_folds"].append(
            {
                "fold": q,
                "start": x[0].isoformat(),
                "end": x[1].isoformat(),
                **base.segment_metrics(res, *x),
            }
        )

    return grid, report, res, fail_diag, funnel


def cost_stress(m, c, trade_start):
    out = []
    for mult in (1.0, 2.0, 4.0):
        cc = base.EquityConfig(**{**asdict(c), "cost_bps": c.cost_bps * mult})
        r = run_backtest_v32(m, cc, trade_start_index=trade_start)
        out.append(
            {
                "cost_multiplier": mult,
                "cost_bps": cc.cost_bps,
                **base.segment_metrics(
                    r, m.dates[trade_start], m.dates[-1]
                ),
            }
        )
    return out


def regime_breakdown(result):
    eq = result.equity.copy()
    eq["equity"] = pd.to_numeric(eq["equity"], errors="coerce")
    eq["daily_return"] = eq["equity"].pct_change()
    rows = []
    for regime, g in eq.groupby("spy_regime", dropna=False):
        r = pd.to_numeric(g["daily_return"], errors="coerce").dropna()
        if len(r) == 0:
            continue
        compounded = float(np.prod(1.0 + r) - 1.0)
        sd = float(r.std(ddof=1)) if len(r) > 2 else float("nan")
        sharpe = (
            float(r.mean() / sd * math.sqrt(252.0))
            if len(r) > 2 and math.isfinite(sd) and sd > 0
            else 0.0
        )
        rows.append(
            {
                "spy_regime": str(regime),
                "sessions": int(len(g)),
                "compounded_return_on_regime_days_pct": compounded * 100.0,
                "sharpe_on_regime_days": sharpe,
                "average_positions": float(
                    pd.to_numeric(g["positions"], errors="coerce").mean()
                ),
            }
        )
    return rows


def concentration_summary(result, config):
    eq = result.equity.copy()
    pos = pd.to_numeric(eq["positions"], errors="coerce").fillna(0)
    counts = pos.value_counts().sort_index()
    return {
        "average_positions": float(pos.mean()),
        "median_positions": float(pos.median()),
        "max_positions": int(pos.max()),
        "zero_position_sessions_pct": float((pos == 0).mean() * 100.0),
        "session_counts_by_position_count": {
            str(int(k)): int(v) for k, v in counts.items()
        },
        "selected_top_k": int(config.top_k),
        "weighting": str(config.weighting),
        "max_intended_single_holding_fraction_of_invested_sleeve":
            (1.0 / config.top_k if config.weighting == "equal" else None),
    }


def status(report, stress):
    dev = all(
        f["return_pct"] > 0
        and f["profit_factor"] > 1
        and f["sharpe"] > 0
        and f["max_drawdown_pct"] > -20
        and f["cycles"] >= 6
        for f in report["development_folds"]
    )
    full = report["full"]
    s4 = next(x for x in stress if x["cost_multiplier"] == 4.0)
    good = (
        dev
        and full["cagr_pct"] >= 8
        and full["sharpe"] >= 0.70
        and full["profit_factor"] >= 1.15
        and full["max_drawdown_pct"] > -20
        and s4["return_pct"] > 0
    )
    return (
        "HISTORICAL_PASS_REQUIRES_PAPER_FORWARD"
        if good
        else "REJECT_OR_RESEARCH_FURTHER"
    )


def main():
    print("EQUITY ROTATION V32 TRANSITION-RISK RESEARCH — RESEARCH ONLY — NO ORDERS")

    api = base.AlpacaEquityData()
    active = api.validate_assets(base.UNIVERSE)
    frames = api.daily_bars(active, start=base.DEFAULT_START)
    usable = {s: f for s, f in frames.items() if len(f) >= 300}
    if len(usable) < 20 or "SPY" not in usable:
        raise SystemExit("Insufficient ETF data")

    m = base.build_matrices(usable)
    spy = _spy_index(m)
    valid = np.flatnonzero(np.isfinite(m.sma[TREND][:, spy]))
    trade_start = max(base.WARMUP_DAYS, int(valid[0]) + 1)

    grid, report, res, fail_diag, funnel = select(m, trade_start)
    c = base.EquityConfig(**report["config"])
    stress = cost_stress(m, c, trade_start)
    regimes = regime_breakdown(res)
    concentration = concentration_summary(res, c)
    st = status(report, stress)

    out = Path(PACK_DIRNAME)
    out.mkdir(exist_ok=True)
    grid.to_csv(out / "grid_all_24.csv", index=False)
    fail_diag.to_csv(out / "qualification_failure_diagnostics.csv", index=False)
    funnel.to_csv(out / "qualification_survivor_funnel.csv", index=False)
    res.equity.to_csv(out / "selected_equity_curve.csv", index=False)
    res.cycles.to_csv(out / "selected_active_cycles.csv", index=False)
    pd.DataFrame(stress).to_csv(out / "cost_stress.csv", index=False)
    pd.DataFrame(regimes).to_csv(out / "regime_breakdown.csv", index=False)

    summary = {
        "strategy": "EQUITY_ROTATION_V32",
        "status": st,
        "orders_placed": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected": report,
        "cost_stress": stress,
        "benchmark_spy": base.benchmark_spy(m, trade_start),
        "regime_breakdown": regimes,
        "concentration": concentration,
        "methodology": {
            "grid_size": len(configs()),
            "families": list(FAMILIES),
            "v32_change":
                "Focused V31 control vs drawdown guard vs confirmed-trend re-entry",
            "reason":
                "V31 weakness concentrated in transition-era fold Sharpe; V32 tests general transition controls without date-specific rules",
            "selection_data":
                "four chronological development folds only; final 20% remains historical diagnostic and is excluded from selection",
            "signal_execution": "prior completed daily bar -> next daily open",
            "parallel_research_workers": max(1, min(4, os.cpu_count() or 2)),
            "engine_isolation":
                "private function globals; V31/V30/V29 shared bindings remain unchanged",
        },
    }

    (out / "equity_rotation_v32_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )

    z = out.with_suffix(".zip")
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zz:
        for p in out.rglob("*"):
            if p.is_file():
                zz.write(p, p.relative_to(out.parent))

    print(
        json.dumps(
            {
                "status": st,
                "selected": report["config"],
                "eligible": report["eligible_development_count"],
                "qualification_bottleneck": report["qualification_bottleneck"],
                "full": report["full"],
                "diagnostic": report["historical_diagnostic"],
                "benchmark_spy": base.benchmark_spy(m, trade_start),
            },
            indent=2,
            default=str,
        )
    )
    print("NO ORDERS WERE PLACED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
