#!/usr/bin/env python3
from __future__ import annotations
import itertools, json, sys, zipfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, pandas as pd
HERE=Path(__file__).resolve().parent
if str(HERE) not in sys.path: sys.path.insert(0,str(HERE))
import research_equity_rotation_v30 as v30
base=v30.base
VERSION='V31'; PACK_DIRNAME='EquityRotationV31_research_pack'; FAST,SLOW,TREND=63,126,200
FAMILIES=('trend_cash','fast_brake','two_stage'); TOP_K=(4,6); REBALANCE=(10,20); WEIGHTINGS=('equal','risk'); VOL_TARGETS=(12.0,16.0)

def _spy_state(m,i):
    j=m.symbols.index('SPY')
    return float(m.closes[i,j]),float(m.sma[TREND][i,j]),float(m.momentum[FAST][i,j]),float(m.momentum[SLOW][i,j]),float(m.spy_realized_vol_pct[i]),float(v30._breadth(m,i))

def _risk_fraction(family,close,sma,fast,slow,vol,breadth):
    broken=np.isfinite(close) and np.isfinite(sma) and np.isfinite(slow) and close<sma and slow<0
    fast_broken=np.isfinite(close) and np.isfinite(sma) and np.isfinite(fast) and close<sma and fast<0
    shock=(np.isfinite(fast) and fast<-0.06) or (np.isfinite(vol) and vol>30 and np.isfinite(fast) and fast<0)
    if family=='trend_cash':
        if broken: return 0.0
        if close>sma and slow>0: return 1.0 if breadth>=.55 else .75 if breadth>=.38 else .45 if breadth>=.22 else .20
        return .35 if breadth>=.50 else .15 if breadth>=.32 else 0.0
    if family=='fast_brake':
        if fast_broken or shock: return 0.0
        return 1.0 if breadth>=.58 else .70 if breadth>=.40 else .35 if breadth>=.25 else 0.0
    if family=='two_stage':
        if broken and shock: return 0.0
        if broken: return .20 if breadth>=.45 else 0.0
        if close<sma or slow<0: return .40 if breadth>=.45 else .20 if breadth>=.30 else 0.0
        return 1.0 if breadth>=.58 else .75 if breadth>=.40 else .45 if breadth>=.24 else .15
    raise ValueError(family)

def target_weights_v31(m,i,config):
    w=np.zeros(len(m.symbols))
    if i<0: return w,'neutral','cash',np.nan,0.0
    close,sma,fast,slow,spyvol,breadth=_spy_state(m,i)
    rf=_risk_fraction(config.risk_profile,close,sma,fast,slow,spyvol,breadth)
    if rf<=0: return w,str(m.spy_regime[i]),'cash',spyvol,0.0
    vs=min(1.0,max(.50,config.vol_target_pct/spyvol)) if np.isfinite(spyvol) and spyvol>0 else .65
    gross=min(1.0,rf*vs)
    picks=v30._pick(m,i,'risk',config.top_k,config.weighting)
    if not picks: return w,str(m.spy_regime[i]),'cash',spyvol,0.0
    for j,x in picks: w[j]=gross*x
    return w,str(m.spy_regime[i]),'risk',spyvol,float(w.sum())
def exposure_and_mode_v31(m,i,config):
    if i<0: return 0.0,'neutral','cash',float('nan')
    close,sma,fast,slow,spyvol,breadth=_spy_state(m,i)
    rf=_risk_fraction(config.risk_profile,close,sma,fast,slow,spyvol,breadth)
    vs=min(1.0,max(.50,config.vol_target_pct/spyvol)) if np.isfinite(spyvol) and spyvol>0 else .65
    exp=min(1.0,rf*vs)
    return exp,str(m.spy_regime[i]),('risk' if exp>0 else 'cash'),spyvol
base.target_weights=target_weights_v31
base.exposure_and_mode=exposure_and_mode_v31

def configs():
    return [base.EquityConfig(momentum_days=SLOW,sma_days=TREND,rebalance_days=r,top_k=k,weighting=wt,risk_profile=f,vol_target_pct=vt) for f,k,r,wt,vt in itertools.product(FAMILIES,TOP_K,REBALANCE,WEIGHTINGS,VOL_TARGETS)]

def select(m,trade_start):
    dates=m.dates[trade_start:]; folds,diag=base.development_splits(dates); rows=[]; cache={}
    for n,c in enumerate(configs(),1):
        res=base.run_backtest(m,c,trade_start_index=trade_start); cache[c]=res; mets=[base.segment_metrics(res,*x) for x in folds]; scores=[base.metric_score(x) for x in mets]
        row={**asdict(c)}
        for q,met in enumerate(mets,1):
            for name,val in met.items(): row[f'fold{q}_{name}']=val
            row[f'fold{q}_score']=scores[q-1]
        cagrs=np.array([x['cagr_pct'] for x in mets]); row['worst_fold_score']=float(min(scores)); row['robust_score']=float(min(scores)+.35*np.mean(scores)-.25*np.std(cagrs)/100); rows.append(row)
        if n%8==0: print(f'V31 tournament {n}/48')
    grid=pd.DataFrame(rows).sort_values(['robust_score','worst_fold_score'],ascending=False).reset_index(drop=True)
    eligible=grid.copy()
    for q in range(1,5): eligible=eligible[(eligible[f'fold{q}_return_pct']>0)&(eligible[f'fold{q}_profit_factor']>1)&(eligible[f'fold{q}_sharpe']>0)&(eligible[f'fold{q}_max_drawdown_pct']>-20)&(eligible[f'fold{q}_cycles']>=6)]
    row=(eligible if len(eligible) else grid).iloc[0]
    c=base.EquityConfig(momentum_days=int(row.momentum_days),sma_days=int(row.sma_days),rebalance_days=int(row.rebalance_days),top_k=int(row.top_k),weighting=str(row.weighting),risk_profile=str(row.risk_profile),vol_target_pct=float(row.vol_target_pct),cost_bps=float(row.cost_bps),max_liquid_assets=int(row.max_liquid_assets))
    res=cache[c]; report={'config':asdict(c),'eligible_development_count':int(len(eligible)),'grid_size':48,'development_folds':[],'historical_diagnostic':base.segment_metrics(res,*diag),'full':base.segment_metrics(res,dates[0],dates[-1]),'diagnostic_range':[x.isoformat() for x in diag]}
    for q,x in enumerate(folds,1): report['development_folds'].append({'fold':q,'start':x[0].isoformat(),'end':x[1].isoformat(),**base.segment_metrics(res,*x)})
    return grid,report,res

def cost_stress(m,c,trade_start):
    out=[]
    for mult in (1.0,2.0,4.0):
        cc=base.EquityConfig(**{**asdict(c),'cost_bps':c.cost_bps*mult}); r=base.run_backtest(m,cc,trade_start_index=trade_start); out.append({'cost_multiplier':mult,'cost_bps':cc.cost_bps,**base.segment_metrics(r,m.dates[trade_start],m.dates[-1])})
    return out

def status(report,stress):
    dev=all(f['return_pct']>0 and f['profit_factor']>1 and f['sharpe']>0 and f['max_drawdown_pct']>-20 and f['cycles']>=6 for f in report['development_folds']); full=report['full']; s4=next(x for x in stress if x['cost_multiplier']==4.0)
    good=dev and full['cagr_pct']>=8 and full['sharpe']>=.70 and full['profit_factor']>=1.15 and full['max_drawdown_pct']>-20 and s4['return_pct']>0
    return 'HISTORICAL_PASS_REQUIRES_PAPER_FORWARD' if good else 'REJECT_OR_RESEARCH_FURTHER'

def main():
    print('EQUITY ROTATION V31 TREND/SHOCK CASH BRAKE — RESEARCH ONLY — NO ORDERS')
    api=base.AlpacaEquityData(); active=api.validate_assets(base.UNIVERSE); frames=api.daily_bars(active,start=base.DEFAULT_START); usable={s:f for s,f in frames.items() if len(f)>=300}
    if len(usable)<20 or 'SPY' not in usable: raise SystemExit('Insufficient ETF data')
    m=base.build_matrices(usable); spy=m.symbols.index('SPY'); valid=np.flatnonzero(np.isfinite(m.sma[TREND][:,spy])); trade_start=max(base.WARMUP_DAYS,int(valid[0])+1)
    grid,report,res=select(m,trade_start); c=base.EquityConfig(**report['config']); stress=cost_stress(m,c,trade_start); st=status(report,stress)
    out=Path(PACK_DIRNAME); out.mkdir(exist_ok=True); grid.to_csv(out/'grid_all_48.csv',index=False); res.equity.to_csv(out/'selected_equity_curve.csv',index=False); res.cycles.to_csv(out/'selected_active_cycles.csv',index=False); pd.DataFrame(stress).to_csv(out/'cost_stress.csv',index=False)
    summary={'strategy':'EQUITY_ROTATION_V31','status':st,'orders_placed':False,'generated_at_utc':datetime.now(timezone.utc).isoformat(),'selected':report,'cost_stress':stress,'benchmark_spy':base.benchmark_spy(m,trade_start),'methodology':{'grid_size':48,'families':list(FAMILIES),'v31_change':'SPY trend/shock cash brake; no defensive bond sleeve; final 20% excluded from selection','signal_execution':'prior completed daily bar -> next daily open'}}
    (out/'equity_rotation_v31_summary.json').write_text(json.dumps(summary,indent=2,default=str)); z=out.with_suffix('.zip')
    with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED) as zz:
        for p in out.rglob('*'):
            if p.is_file(): zz.write(p,p.relative_to(out.parent))
    print(json.dumps({'status':st,'selected':report['config'],'eligible':report['eligible_development_count'],'full':report['full'],'diagnostic':report['historical_diagnostic']},indent=2,default=str)); print('NO ORDERS WERE PLACED.'); return 0
if __name__=='__main__': raise SystemExit(main())
