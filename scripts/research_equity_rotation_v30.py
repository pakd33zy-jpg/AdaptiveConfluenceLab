#!/usr/bin/env python3
from __future__ import annotations

import itertools, json, math, sys, zipfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import research_equity_rotation_v29 as base

VERSION = "V30"
PACK_DIRNAME = "EquityRotationV30_research_pack"
FAST, SLOW, TREND = 63, 126, 200
FAMILIES = ("breadth", "dual_sleeve", "relative")
TOP_K = (4, 6)
REBALANCE = (10, 20)
WEIGHTINGS = ("equal", "risk")
VOL_TARGETS = (12.0, 16.0)
DEFENSIVE = frozenset(("TLT", "IEF", "LQD", "GLD"))
RISK_EXCLUDED = DEFENSIVE

def _breadth(m, i):
    close, sma = m.closes[i], m.sma[TREND][i]
    fast, slow = m.momentum[FAST][i], m.momentum[SLOW][i]
    good = total = 0
    for j, s in enumerate(m.symbols):
        if s in RISK_EXCLUDED:
            continue
        if all(np.isfinite(x) for x in (close[j], sma[j], fast[j], slow[j])):
            total += 1
            good += int(close[j] > sma[j] and slow[j] > 0 and fast[j] > -0.03)
    return good / total if total else 0.0

def _asset_vol(m, i, j, lookback=20):
    a = max(1, i-lookback+1)
    p = m.valuation_closes[a-1:i+1, j]
    p = p[np.isfinite(p) & (p > 0)]
    if len(p) < 8:
        return np.nan
    r = np.diff(np.log(p))
    if len(r) < 6:
        return np.nan
    v = float(np.std(r, ddof=1) * math.sqrt(252))
    return v if np.isfinite(v) and v > 1e-8 else np.nan

def _pick(m, i, pool, k, weighting):
    close, sma = m.closes[i], m.sma[TREND][i]
    fast, slow = m.momentum[FAST][i], m.momentum[SLOW][i]
    liq = m.dollar_volume_median[i]
    idx = [j for j,s in enumerate(m.symbols) if s in pool or pool == "risk"]
    if pool == "risk":
        idx = [j for j in idx if m.symbols[j] not in RISK_EXCLUDED]
    idx = [j for j in idx if all(np.isfinite(x) for x in (close[j], sma[j], fast[j], slow[j], liq[j]))]
    idx = sorted(idx, key=lambda j: float(liq[j]), reverse=True)[:base.DEFAULT_MAX_LIQUID_ASSETS]
    score = 0.35*fast + 0.65*slow
    idx = [j for j in idx if close[j] > sma[j] and slow[j] > 0 and score[j] > 0]
    chosen, groups = [], set()
    for j in sorted(idx, key=lambda j: float(score[j]), reverse=True):
        g = base.GROUPS.get(m.symbols[j], m.symbols[j])
        if g in groups: continue
        groups.add(g); chosen.append(j)
        if len(chosen) >= k: break
    if not chosen:
        return []
    if weighting == "equal":
        raw = np.ones(len(chosen))
    else:
        vv = np.array([_asset_vol(m, i, j) for j in chosen])
        fv = vv[np.isfinite(vv) & (vv > 0)]
        fallback = float(np.median(fv)) if len(fv) else .20
        vv = np.where(np.isfinite(vv) & (vv > 0), vv, fallback)
        raw = 1.0/vv
        raw = np.minimum(raw/raw.sum(), 2.0/len(chosen))
    raw = raw/raw.sum()
    return list(zip(chosen, raw))

def target_weights_v30(m, i, config):
    w = np.zeros(len(m.symbols))
    if i < 0:
        return w, "neutral", "cash", np.nan, 0.0
    breadth = _breadth(m, i)
    regime = str(m.spy_regime[i])
    spyvol = float(m.spy_realized_vol_pct[i])
    volscale = min(1.0, max(.55, config.vol_target_pct/spyvol)) if np.isfinite(spyvol) and spyvol > 0 else .65

    family = config.risk_profile
    if family == "breadth":
        risk_frac = 1.0 if breadth >= .60 else .70 if breadth >= .42 else .35 if breadth >= .28 else 0.0
        def_frac = 0.0 if risk_frac >= .70 else .35 if risk_frac > 0 else .55
    elif family == "dual_sleeve":
        risk_frac = .85 if breadth >= .60 else .65 if breadth >= .42 else .40 if breadth >= .25 else .20
        def_frac = 1.0-risk_frac
    elif family == "relative":
        risk_frac = 1.0 if breadth >= .50 else .75 if breadth >= .32 else .45 if breadth >= .18 else .0
        def_frac = 0.0
    else:
        raise ValueError(family)

    rp = _pick(m, i, "risk", config.top_k, config.weighting) if risk_frac else []
    dp = _pick(m, i, DEFENSIVE, 2, config.weighting) if def_frac else []
    if not rp and not dp:
        return w, regime, "cash", spyvol, 0.0
    gross = min(1.0, (risk_frac + def_frac) * volscale)
    r_alloc = gross * risk_frac / max(1e-12, risk_frac + def_frac)
    d_alloc = gross * def_frac / max(1e-12, risk_frac + def_frac)
    for j,x in rp: w[j] += r_alloc*x
    for j,x in dp: w[j] += d_alloc*x
    mode = "risk" if r_alloc >= d_alloc else "defensive"
    return w, regime, mode, spyvol, float(w.sum())

def exposure_and_mode_v30(m, i, config):
    if i < 0:
        return 0.0, "neutral", "cash", float("nan")
    b = _breadth(m, i)
    reg = str(m.spy_regime[i])
    vol = float(m.spy_realized_vol_pct[i])
    fam = config.risk_profile
    if fam == "breadth":
        mode = "risk" if b >= .28 else "cash"
        exp = 1.0 if b >= .60 else .70 if b >= .42 else .35 if b >= .28 else 0.0
    elif fam == "dual_sleeve":
        mode = "risk" if b >= .42 else "defensive"
        exp = 1.0
    elif fam == "relative":
        mode = "risk" if b >= .18 else "cash"
        exp = 1.0 if b >= .50 else .75 if b >= .32 else .45 if b >= .18 else 0.0
    else:
        raise ValueError(fam)
    if exp and np.isfinite(vol) and vol > 0:
        exp *= min(1.0, max(.55, config.vol_target_pct/vol))
    return float(exp), reg, mode, vol

base.exposure_and_mode = exposure_and_mode_v30
base.target_weights = target_weights_v30

def configs():
    return [base.EquityConfig(momentum_days=SLOW, sma_days=TREND, rebalance_days=r,
        top_k=k, weighting=wt, risk_profile=f, vol_target_pct=vt)
        for f,k,r,wt,vt in itertools.product(FAMILIES, TOP_K, REBALANCE, WEIGHTINGS, VOL_TARGETS)]

def _select(m, trade_start):
    dates = m.dates[trade_start:]
    folds, diagnostic = base.development_splits(dates)
    rows, cache = [], {}
    for n,c in enumerate(configs(),1):
        res = base.run_backtest(m,c,trade_start_index=trade_start)
        cache[c] = res
        mets = [base.segment_metrics(res,*x) for x in folds]
        scores = [base.metric_score(x) for x in mets]
        row = {**asdict(c)}
        for q,met in enumerate(mets,1):
            for name,val in met.items(): row[f"fold{q}_{name}"] = val
            row[f"fold{q}_score"] = scores[q-1]
        cagrs=np.array([x["cagr_pct"] for x in mets])
        row["worst_fold_score"]=float(min(scores))
        row["robust_score"]=float(min(scores)+.35*np.mean(scores)-.25*np.std(cagrs)/100)
        rows.append(row)
        if n % 8 == 0: print(f"V30 tournament {n}/48")
    grid=pd.DataFrame(rows).sort_values(["robust_score","worst_fold_score"],ascending=False).reset_index(drop=True)
    eligible=grid.copy()
    for q in range(1,5):
        eligible=eligible[(eligible[f"fold{q}_return_pct"]>0)&(eligible[f"fold{q}_profit_factor"]>1)&
            (eligible[f"fold{q}_sharpe"]>0)&(eligible[f"fold{q}_max_drawdown_pct"]>-20)&(eligible[f"fold{q}_cycles"]>=6)]
    row=(eligible if len(eligible) else grid).iloc[0]
    c=base.EquityConfig(momentum_days=int(row.momentum_days),sma_days=int(row.sma_days),
        rebalance_days=int(row.rebalance_days),top_k=int(row.top_k),weighting=str(row.weighting),
        risk_profile=str(row.risk_profile),vol_target_pct=float(row.vol_target_pct),
        cost_bps=float(row.cost_bps),max_liquid_assets=int(row.max_liquid_assets))
    res=cache[c]
    report={"config":asdict(c),"eligible_development_count":int(len(eligible)),"grid_size":48,
        "development_folds":[],"historical_diagnostic":base.segment_metrics(res,*diagnostic),
        "full":base.segment_metrics(res,dates[0],dates[-1]),"diagnostic_range":[x.isoformat() for x in diagnostic]}
    for q,x in enumerate(folds,1):
        met=base.segment_metrics(res,*x)
        report["development_folds"].append({"fold":q,"start":x[0].isoformat(),"end":x[1].isoformat(),**met})
    return grid,report,res

def cost_stress(m,c,trade_start):
    out=[]
    for mult in (1.0,2.0,4.0):
        cc=base.EquityConfig(**{**asdict(c),"cost_bps":c.cost_bps*mult})
        r=base.run_backtest(m,cc,trade_start_index=trade_start)
        out.append({"cost_multiplier":mult,"cost_bps":cc.cost_bps,
            **base.segment_metrics(r,m.dates[trade_start],m.dates[-1])})
    return out

def status(report,stress):
    dev=all(f["return_pct"]>0 and f["profit_factor"]>1 and f["sharpe"]>0 and f["max_drawdown_pct"]>-20 for f in report["development_folds"])
    full=report["full"]
    stressed=next(x for x in stress if x["cost_multiplier"]==4.0)
    good=dev and full["cagr_pct"]>=8 and full["sharpe"]>=.65 and full["profit_factor"]>=1.15 and full["max_drawdown_pct"]>-20 and stressed["return_pct"]>0
    return "HISTORICAL_PASS_REQUIRES_PAPER_FORWARD" if good else "REJECT_OR_RESEARCH_FURTHER"

def main():
    print("EQUITY ROTATION V30 STRUCTURAL TOURNAMENT — RESEARCH ONLY — NO ORDERS")
    api=base.AlpacaEquityData()
    active=api.validate_assets(base.UNIVERSE)
    frames=api.daily_bars(active,start=base.DEFAULT_START)
    usable={s:f for s,f in frames.items() if len(f)>=300}
    if len(usable)<20 or "SPY" not in usable: raise SystemExit("Insufficient ETF data")
    m=base.build_matrices(usable)
    spy=m.symbols.index("SPY")
    valid=np.flatnonzero(np.isfinite(m.sma[TREND][:,spy]))
    trade_start=max(base.WARMUP_DAYS,int(valid[0])+1)
    grid,report,res=_select(m,trade_start)
    c=base.EquityConfig(**report["config"])
    stress=cost_stress(m,c,trade_start)
    st=status(report,stress)
    out=Path(PACK_DIRNAME); out.mkdir(exist_ok=True)
    grid.to_csv(out/"grid_all_48.csv",index=False)
    res.equity.to_csv(out/"selected_equity_curve.csv",index=False)
    res.cycles.to_csv(out/"selected_active_cycles.csv",index=False)
    pd.DataFrame(stress).to_csv(out/"cost_stress.csv",index=False)
    summary={"strategy":"EQUITY_ROTATION_V30","status":st,"orders_placed":False,
        "generated_at_utc":datetime.now(timezone.utc).isoformat(),"selected":report,
        "cost_stress":stress,"benchmark_spy":base.benchmark_spy(m,trade_start),
        "methodology":{"one_run_structural_tournament":True,"families":list(FAMILIES),
        "grid_size":48,"selection":"4 development folds; final 20% excluded from selection",
        "signal_execution":"prior completed daily bar -> next daily open"}}
    (out/"equity_rotation_v30_summary.json").write_text(json.dumps(summary,indent=2,default=str))
    z=out.with_suffix(".zip")
    with zipfile.ZipFile(z,"w",zipfile.ZIP_DEFLATED) as zz:
        for p in out.rglob("*"):
            if p.is_file(): zz.write(p,p.relative_to(out.parent))
    print(json.dumps({"status":st,"selected":report["config"],"eligible":report["eligible_development_count"],
        "full":report["full"],"diagnostic":report["historical_diagnostic"]},indent=2,default=str))
    print("NO ORDERS WERE PLACED.")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
