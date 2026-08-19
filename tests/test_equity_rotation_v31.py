from pathlib import Path
import importlib.util, sys, numpy as np, pandas as pd
P=Path(__file__).resolve().parents[1]/"scripts"/"research_equity_rotation_v31.py"
spec=importlib.util.spec_from_file_location("v31",P); m=importlib.util.module_from_spec(spec); sys.modules[spec.name]=m; spec.loader.exec_module(m)

def frames(periods=1800):
    dates=pd.bdate_range("2018-01-02",periods=periods,tz="UTC"); syms=["SPY","QQQ","IWM","DIA","XLK","XLF","XLE","XLV","SMH","IBB","KRE","XHB","VNQ","RSP","MTUM","QUAL","USMV","TLT","IEF","LQD","GLD","EFA","EEM","DBC"]
    out={}; t=np.arange(periods)
    for k,s in enumerate(syms):
        phase=k*.27; drift=np.where(t<450,.00045,np.where(t<800,-.00030,np.where(t<1250,.00055,.00015))); rr=drift+.002*np.sin(t/31+phase)+.001*np.sin(t/9+phase); close=100*np.exp(np.cumsum(rr)); open_=close*np.exp(.0008*np.sin(t/7+phase)); vol=np.full(periods,2_000_000+k*25_000,dtype=float); out[s]=pd.DataFrame({"date":dates,"open":open_,"high":np.maximum(open_,close)*1.002,"low":np.minimum(open_,close)*.998,"close":close,"volume":vol})
    return out

def test_grid_48():
    cs=m.configs(); assert len(cs)==48; assert {c.risk_profile for c in cs}=={"trend_cash","fast_brake","two_stage"}

def test_no_orders_or_v26():
    x=P.read_text().lower(); assert "submit_order" not in x and "/v2/orders" not in x and "paper_v26" not in x

def test_weights_bounded_long_only():
    mat=m.base.build_matrices(frames())
    for c in m.configs()[::16]:
        for i in (350,700,1000,1500):
            w,*_=m.target_weights_v31(mat,i,c); assert np.all(w>=-1e-12); assert w.sum()<=1.000001

def test_hard_broken_trend_goes_cash():
    assert m._risk_fraction("trend_cash",90,100,-.08,-.10,25,.40)==0.0
    assert m._risk_fraction("fast_brake",90,100,-.01,-.02,20,.60)==0.0

def test_full_backtest_each_family():
    mat=m.base.build_matrices(frames())
    for fam in m.FAMILIES:
        c=next(c for c in m.configs() if c.risk_profile==fam); r=m.run_backtest_v31(mat,c,trade_start_index=320); assert np.isfinite(r.equity["equity"]).all(); assert (r.equity["equity"]>0).all()

def test_import_does_not_override_shared_v30_bindings():
    assert m.base.target_weights is m.v30.target_weights_v30
    assert m.base.exposure_and_mode is m.v30.exposure_and_mode_v30

def test_cost_stress_contract(): assert "(1.0,2.0,4.0)" in P.read_text().replace(" ","")
