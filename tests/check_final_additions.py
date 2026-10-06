"""Sanity checks for final rubric additions (news + real examples + horizons)."""
from pathlib import Path
import pandas as pd, json
ROOT=Path(__file__).resolve().parents[1]
def close(a,b,tol=.05): assert abs(float(a)-float(b))<=tol,(a,b)
def main():
    f=pd.read_csv(ROOT/"outputs/news_forecast_ablation.csv")
    close(f[(f.window=="exact")&(f.model=="v3")].MAE.iloc[0],719.8702)
    close(f[(f.window=="exact")&(f.model=="v3_news")].MAE.iloc[0],946.8434)
    e=pd.read_csv(ROOT/"outputs/news_early_warning_ablation.csv")
    close(e[(e.task=="detect")&(e.model=="base_news")].PR_AUC.iloc[0],0.3639,.003)
    close(e[(e.task=="predict")&(e.model=="base_news")].PR_AUC.iloc[0],0.0269,.003)
    x=pd.read_csv(ROOT/"outputs/real_examples.csv")
    assert set(x.role)=={"stable","shift","failure"}
    assert int(x[x.role=="stable"].run_id.iloc[0])==13015
    close(x[x.role=="stable"].exact_mae_v3.iloc[0],346.887,.1)
    assert int(x[x.role=="shift"].run_id.iloc[0])==406
    assert int(x[x.role=="failure"].run_id.iloc[0])==2242
    h=pd.read_csv(ROOT/"outputs/horizons_metrics.csv")
    assert {1,3,6,12}.issubset(set(h.h.dropna().astype(int)))
    print("ok final additions")
if __name__=="__main__": main()
