from pathlib import Path
import importlib.util, sys, numpy as np, pandas as pd

P=Path(__file__).resolve().parents[1]/"scripts"/"research_equity_rotation_v30.py"
spec=importlib.util.spec_from_file_location("v30",P); m=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=m; spec.loader.exec_module(m)

def frames(periods=1800):
    dates=pd.bdate_range("2018-01-02",periods=periods,tz="UTC")
    syms=["SPY","QQQ","IWM","DIA","XLK","XLF","XLE","XLV","SMH","IBB","KRE","XHB","VNQ",
          "RSP","MTUM","QUAL","USMV","TLT","IEF","LQD","GLD","EFA","EEM","DBC"]
    out={}
    t=np.arange(periods)
    for k,s in enumerate(syms):
        phase=k*.27
        # Multiple trend regimes plus idiosyncratic waves; deterministic and non-random.
        drift=np.where(t<450,.00045,np.where(t<800,-.00018,np.where(t<1250,.00055,.00015)))
        if s in {"TLT","IEF","LQD","GLD"}: drift=np.where(t<800,.00015,np.where(t<1250,-.00005,.00022))
        rr=drift+.0025*np.sin(t/31+phase)+.0012*np.sin(t/9+phase)
        close=100*np.exp(np.cumsum(rr))
        open_=close*np.exp(.0008*np.sin(t/7+phase))
        vol=np.full(periods,2_000_000+k*25_000,dtype=float)
        out[s]=pd.DataFrame({"date":dates,"open":open_,"high":np.maximum(open_,close)*1.002,
            "low":np.minimum(open_,close)*.998,"close":close,"volume":vol})
    return out

def test_grid_is_exactly_48_and_structural():
    cs=m.configs()
    assert len(cs)==48
    assert {c.risk_profile for c in cs}=={"breadth","dual_sleeve","relative"}
    assert {c.top_k for c in cs}=={4,6}

def test_no_orders_and_v26_untouched():
    x=P.read_text().lower()
    assert "submit_order" not in x and "/v2/orders" not in x and "paper_v26" not in x

def test_target_weights_long_only_and_bounded():
    mat=m.base.build_matrices(frames())
    c=m.configs()[0]
    for i in [300,600,900,1200,1600]:
        w,*_=m.target_weights_v30(mat,i,c)
        assert np.all(w>=-1e-12)
        assert w.sum()<=1.000001

def test_full_backtest_runs_for_each_family():
    mat=m.base.build_matrices(frames())
    for fam in m.FAMILIES:
        c=next(c for c in m.configs() if c.risk_profile==fam)
        r=m.base.run_backtest(mat,c,trade_start_index=320)
        assert len(r.equity)==len(mat.dates)-320
        assert np.isfinite(r.equity["equity"]).all()
        assert (r.equity["equity"]>0).all()

def test_diagnostic_is_excluded_from_selection_contract():
    x=P.read_text()
    assert "base.development_splits" in x
    assert "historical_diagnostic" in x
    assert "eligible_development_count" in x

def test_cost_stress_1x_2x_4x_present():
    assert "(1.0,2.0,4.0)" in P.read_text().replace(" ","")
