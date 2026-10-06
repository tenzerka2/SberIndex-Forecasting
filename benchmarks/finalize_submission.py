"""Merge rubric-specific horizon/foundation/news results into root final tables.

Run after benchmarks/build_final.py. This does not change any model or metric; it only normalizes
already-produced outputs into FINAL_METRICS.csv / ABLATION.csv for submission.
"""
from pathlib import Path
import numpy as np, pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs"

def main():
    fm=pd.read_csv(ROOT/"FINAL_METRICS.csv")
    h=pd.read_csv(OUT/"horizons_metrics.csv")
    rows=[]
    for _,r in h.iterrows():
        if int(r.get("n_pairs",0) or 0)<=0: continue
        rows.append({
            "window":"horizons_"+str(r["subset"]),
            "h":str(int(r["h"])),
            "model":r["model"],
            "n":int(r["n_pairs"]),
            "MAE":r.get("MAE",np.nan),
            "R2_level":r.get("R2_level",np.nan),
            "R2_growth":r.get("R2_growth",np.nan),
            "wMAPE_pct":r.get("wMAPE_pct",np.nan),
            "source":"outputs/horizons_metrics.csv",
        })
    extra=pd.DataFrame(rows)
    # remove previous rubric rows on re-run
    fm=fm[~fm["window"].astype(str).str.startswith("horizons_")]
    fm=pd.concat([fm,extra],ignore_index=True,sort=False)
    fm.to_csv(ROOT/"FINAL_METRICS.csv",index=False,float_format="%.4f")

    ab=pd.read_csv(ROOT/"ABLATION.csv")
    ab=ab[~ab["block"].astype(str).str.startswith("news_")]
    nf=pd.read_csv(OUT/"news_forecast_ablation.csv")
    forecast=[]
    for w in ["exact","other","apr_nov"]:
        g=nf[nf.window.eq(w)].set_index("model")
        if {"v3","v3_news"}.issubset(g.index):
            forecast.append({
                "block":"news_forecast",
                "model":"v3_news_"+w,
                "description":f"Frozen V3 + global historical-news residual correction ({w})",
                "MAE_exact":g.loc["v3_news","MAE"] if w=="exact" else np.nan,
                "MAE_other":g.loc["v3_news","MAE"] if w=="other" else np.nan,
                "MAE_apr_nov":g.loc["v3_news","MAE"] if w=="apr_nov" else np.nan,
                "R2_growth_exact":g.loc["v3_news","R2_growth"] if w=="exact" else np.nan,
                "wMAPE_exact":g.loc["v3_news","wMAPE_pct"] if w=="exact" else np.nan,
                "dMAE_vs_V2_exact":np.nan,
                "dMAE_vs_V2_other":np.nan,
                "news_delta_vs_frozen_v3":float(g.loc["v3_news","MAE"]-g.loc["v3","MAE"]),
            })
    ew=pd.read_csv(OUT/"news_early_warning_ablation.csv")
    ewrows=[]
    for _,r in ew.iterrows():
        ewrows.append({
            "block":"news_early_warning_"+str(r["task"]),
            "model":r["model"],
            "description":"auxiliary frozen-protocol news ablation",
            "PR_AUC":r["PR_AUC"],
            "lift_vs_base_rate":r["lift_vs_base_rate"],
            "F1":r["F1"],
            "precision":r["precision"],
            "recall":r["recall"],
            "false_alarms_per_100":r["false_alarms_per_100"],
        })
    ab=pd.concat([ab,pd.DataFrame(forecast),pd.DataFrame(ewrows)],ignore_index=True,sort=False)
    ab.to_csv(ROOT/"ABLATION.csv",index=False,float_format="%.4f")
    print("finalized",len(fm),"metric rows",len(ab),"ablation rows")

if __name__=="__main__": main()
