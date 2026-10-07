"""Time-safe global-news ablation. Frozen V3 is never re-tuned.

Forecasting: a fixed Ridge residual correction uses only global monthly news available at each
origin and only forecast errors whose targets are already observed.
Early warning: a fixed LightGBM protocol compares a compact online base feature set against the
same set plus global news. This is an auxiliary news-specific ablation, not a replacement for the
main structural-shift benchmark in early_warning_v2.py.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from sbx import backtest as B, early_warning as EW
from sbx.data import load_panel
from sbx.final import final_models
from sbx.news_global import tensor as news_tensor, monthly as news_monthly
OUT=ROOT/"outputs"
ORIGINS=pd.date_range("2024-02-01","2024-11-01",freq="MS")
WINDOWS={"exact":B.EXACT_ORIGINS,
         "other":pd.to_datetime(["2024-04-01","2024-05-01","2024-10-01","2024-11-01"]),
         "apr_nov":pd.date_range("2024-04-01","2024-11-01",freq="MS")}
NEWS_NAMES=["news_intensity","news_signed","news_natural_emergency","news_security",
            "news_enterprise_negative","news_enterprise_positive","news_infrastructure",
            "news_intensity_sum3","news_signed_sum3","news_natural_emergency_sum3",
            "news_security_sum3","news_enterprise_negative_sum3","news_enterprise_positive_sum3",
            "news_infrastructure_sum3"]

def ridge_predict(X, y, Z, alpha=10.0):
    """Standardized ridge with an unpenalized intercept; NumPy equivalent of the old pipeline."""
    X, y, Z = np.asarray(X, float), np.asarray(y, float), np.asarray(Z, float)
    mu, sd = X.mean(axis=0), np.maximum(X.std(axis=0), 1e-12)
    A, C = (X-mu)/sd, (Z-mu)/sd
    intercept = y.mean()
    beta = np.linalg.solve(A.T@A + alpha*np.eye(A.shape[1]), A.T@(y-intercept))
    return intercept+C@beta

def forecast(panel, outdir=OUT):
    d=B.run(panel,{"v3":final_models()["v3"]},ORIGINS)
    nf=news_tensor(panel.periods,len(panel.meta))
    pos={p:i for i,p in enumerate(panel.periods)}
    def X(df):
        s=df["series"].to_numpy(int); o=df["origin"].map(pos).to_numpy(int)
        return np.column_stack([*[nf[k][s,o] for k in NEWS_NAMES],df["h"].to_numpy(float)])
    frames=[]
    for origin in ORIGINS:
        te=d[d.origin.eq(origin)].copy(); tr=d[(d.origin<origin)&(d.target_date<=origin)].copy()
        te["v3_news"]=te["v3"]
        if tr.origin.nunique()>=2:
            # National news has one independent feature vector per origin/horizon.
            # Replicating it across 2,016 municipalities would dilute Ridge's penalty.
            xx = X(tr)
            z = pd.DataFrame(xx, columns=[f"x{j}" for j in range(xx.shape[1])])
            z["origin"] = tr.origin.to_numpy(); z["h"] = tr.h.to_numpy()
            z["target"] = np.log(tr.y.to_numpy()/tr.v3.to_numpy())
            agg = z.groupby(["origin", "h"], as_index=False).mean(numeric_only=True)
            corr=np.clip(ridge_predict(agg[[f"x{j}" for j in range(xx.shape[1])]],agg["target"],X(te)),-0.2,0.2)
            te["v3_news"]=te.v3*np.exp(corr)
        frames.append(te)
    z=pd.concat(frames,ignore_index=True)
    rows=[]
    for w,os_ in WINDOWS.items():
        g=z[z.origin.isin(os_)]
        for c in ["v3","v3_news"]: rows.append({"window":w,"model":c,**B.metrics(g,c)})
    out=pd.DataFrame(rows); out.to_csv(outdir/"news_forecast_ablation.csv",index=False,float_format="%.6f")
    return out

def labels(ev,task):
    n,T=ev.shape; y=np.zeros((n,T),bool)
    for t in range(T):
        y[:,t]=ev[:,max(0,t-2):t+1].any(1) if task=="detect" else (ev[:,t+1:t+4].any(1) if t+1<T else False)
    return y

def clf():
    from lightgbm import LGBMClassifier
    return LGBMClassifier(n_estimators=300,learning_rate=.03,num_leaves=15,min_child_samples=100,
        subsample=.8,subsample_freq=1,colsample_bytree=.8,reg_lambda=5,class_weight="balanced",
        random_state=42,verbosity=-1,n_jobs=4)

def best_thr(y,s):
    from sklearn.metrics import precision_recall_curve
    p,r,t=precision_recall_curve(y,s); f=2*p*r/np.maximum(p+r,1e-12)
    return float(t[int(np.argmax(f[:-1]))]) if len(t) else .5

def early_warning(panel):
    L=panel.logs; n,T=L.shape
    scores={"jump_raw":EW.score_jump(L,seasonal=False),"jump_seasonal":EW.score_jump(L,seasonal=True)}
    base=EW.feature_tensor(L,scores)
    news=news_tensor(panel.periods,n); feats={**base,**news}
    base_names=list(base); with_news=base_names+NEWS_NAMES
    FIRST=6; TEST={"detect":range(12,22),"predict":range(14,19)}; EMB={"detect":2,"predict":5}
    ev_full=EW.events(L)
    def stack(names,ts): return np.column_stack([feats[k][:,list(ts)].T.ravel() for k in names])
    def run(names,task):
        yf=labels(ev_full,task); chunks=[]
        for t in TEST[task]:
            yt=labels(EW.events(L[:,:t+1]),task); train=list(range(FIRST,t-EMB[task]+1))
            ytr=yt[:,train].T.ravel(); yte=yf[:,t]; month=np.repeat(train,n)
            Xtr,Xte=stack(names,train),stack(names,[t]); inner=train[-2]; mask=month<=inner-EMB[task]; thr=.5
            if ytr[mask].any() and ytr[~mask].any():
                m=clf().fit(Xtr[mask],ytr[mask]); thr=best_thr(ytr[~mask],m.predict_proba(Xtr[~mask])[:,1])
            m=clf().fit(Xtr,ytr); s=m.predict_proba(Xte)[:,1]; a=s>=thr
            chunks.append(pd.DataFrame({"month":t,"y":yte,"score":s,"alarm":a}))
        return pd.concat(chunks,ignore_index=True)
    def met(d):
        from sklearn.metrics import average_precision_score
        y=d.y.to_numpy(bool); a=d.alarm.to_numpy(bool); s=d.score.to_numpy()
        tp=(a&y).sum(); fp=(a&~y).sum(); fn=(~a&y).sum(); p=tp/max(tp+fp,1); r=tp/max(tp+fn,1)
        ap=average_precision_score(y,s); br=y.mean()
        return {"PR_AUC":ap,"lift_vs_base_rate":ap/br,"precision":p,"recall":r,"F1":2*p*r/max(p+r,1e-12),
                "false_alarms_per_100":100*fp/len(y),"base_rate":br,"n_rows":len(y)}
    rows=[]
    for task in ["detect","predict"]:
        for name,names in [("base",base_names),("base_news",with_news)]:
            rows.append({"task":task,"model":name,**met(run(names,task))})
    out=pd.DataFrame(rows); out.to_csv(OUT/"news_early_warning_ablation.csv",index=False,float_format="%.6f")
    return out

def main():
    p=load_panel(); _,audit=news_monthly(); (OUT/"news_audit.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2))
    print(forecast(p).to_string(index=False)); print(early_warning(p).to_string(index=False)); print(audit)
if __name__=="__main__": main()
