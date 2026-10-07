"""Usable export from the supplied December-2024 origin, with explicit evidence labels."""
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from sbx.data import ROOT, load_panel, national_monthly
from sbx.candidates import v4_models
from sbx.final import final_models
from sbx.intervals import future_intervals
from sbx import early_warning as EW, research_warning as W

OUT=ROOT/"outputs/research"


def main(name):
    OUT.mkdir(parents=True,exist_ok=True)
    p=load_panel();L=p.logs;o=p.periods[-1];nat=national_monthly()
    models={**final_models(),**v4_models()};pred=models[name](L,list(range(1,13)),{"origin":o,"national":nat.loc[:o]})
    frames=[]
    for h,x in pred.items():
        g=p.meta.copy();g["series"]=np.arange(len(g));g["origin"]=o;g["h"]=h
        g["target_date"]=o+pd.DateOffset(months=h);g[name]=np.exp(x)
        g["evidence"]="retrospective_h1_3" if h<=3 else "limited_h6_or_interpolation" if h<=6 else "unvalidated_extrapolation"
        g["interval_method"]="empirical_two_part_no_coverage_guarantee"
        frames.append(g)
    fut=pd.concat(frames,ignore_index=True)
    cache=OUT/"v4_pairs_0.csv.gz"
    if not cache.exists():
        raise FileNotFoundError("Run benchmarks/research_v4.py first to build observed-error calibration")
    cal=pd.read_csv(cache,parse_dates=["origin","target_date"])
    fut=future_intervals(cal,name,fut,L,p.periods)
    fut.to_csv(OUT/f"forecast_{name}_2025.csv.gz",index=False,compression={"method":"gzip","mtime":0})
    # Latest prioritization, not a calibrated forecast of a future economic crisis.
    feats=W.features(L);t=L.shape[1]-1
    y=EW.mature_labels(L,"detect");train=[s for s in range(6,t) if (y[:,s]>=0).all()]
    score=W.logistic_predict(W.stack(feats,list(feats),train),y[:,train].T.ravel(),W.stack(feats,list(feats),[t]))
    signals=p.meta.copy();signals["origin"]=o
    signals["detection_score"]=score;signals["detection_review_top2pct"]=W.top_budget(score,.02)
    signals["early_warning_score"]=feats["jump"][:,-1]
    signals["early_warning_review_top2pct"]=W.top_budget(feats["jump"][:,-1],.02)
    signals["warning_status"]="research_priority_only_low_precision"
    signals.to_csv(OUT/"signals_2024_12.csv",index=False)
    assert np.isfinite(fut[name]).all() and (fut[name]>0).all()
    assert len(fut)==len(p.meta)*12
    for level in [80,90]:
        assert (fut[f"{name}_lo{level}"]<=fut[name]).all()
        assert (fut[f"{name}_hi{level}"]>=fut[name]).all()
    print(f"Exported {len(fut)} forecasts from {o.date()}, model={name}; long horizons are unvalidated.")


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--model",default="v4_diversified",choices=list({**final_models(),**v4_models()}))
    main(parser.parse_args().model)
