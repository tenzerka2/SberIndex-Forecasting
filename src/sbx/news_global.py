"""Global historical-news features for a conservative competition ablation.

The search plan was frozen before collection. Each record keeps its query_id, rank, URL, published
metadata, title and snippet. Publication time is verified conservatively as the latest plausible
date found in metadata or the URL. Records whose verified month differs from the pre-registered
query month are excluded.

These features are national/month-level and therefore identical for every municipality. They can
only add information about WHEN a macro disturbance is present, not WHERE it will hit.
"""
from __future__ import annotations
import json, re
from pathlib import Path
import numpy as np
import pandas as pd
from .data import ROOT

CORPUS = ROOT / "data" / "news" / "corpus.jsonl"
TOPICS = ["natural_emergency","security","enterprise_negative","enterprise_positive","infrastructure"]
POLARITY = {"natural_emergency":-1.0,"security":-1.0,"enterprise_negative":-1.0,"enterprise_positive":1.0,"infrastructure":-1.0}

def _dt(x):
    if not x: return None
    try:
        t=pd.Timestamp(x)
        if t.tzinfo is not None: t=t.tz_convert(None)
        return t.normalize()
    except Exception:
        return None

def _url_dates(url):
    s=str(url or ""); out=[]
    pats=[r"(?<!\d)(20(?:22|23|24|25))[/-](0?[1-9]|1[0-2])[/-]([0-2]?\d|3[01])(?!\d)",
          r"(?<!\d)(20(?:22|23|24|25))(0[1-9]|1[0-2])([0-2]\d|3[01])(?!\d)"]
    for p in pats:
        for y,m,d in re.findall(p,s):
            try: out.append(pd.Timestamp(year=int(y),month=int(m),day=int(d)))
            except ValueError: pass
    return out

def verified_date(r):
    cand=[]
    m=_dt(r.get("published"))
    if m is not None: cand.append(m)
    cand += _url_dates(r.get("url",""))
    return max(cand) if cand else None

def read_verified():
    rows=[json.loads(x) for x in CORPUS.read_text(encoding="utf-8").splitlines() if x.strip()]
    out=[]; conflicts=0; mismatches=0; missing=0
    for r in rows:
        qmonth=pd.Timestamp(r["query_id"][:7]+"-01")
        vd=verified_date(r)
        md=_dt(r.get("published")); uds=_url_dates(r.get("url",""))
        if md is not None and uds and any(d!=md for d in uds): conflicts+=1
        if vd is None:
            missing+=1; continue
        vm=vd.to_period("M").to_timestamp()
        if vm!=qmonth:
            mismatches+=1; continue
        topic=next(t for t in TOPICS if r["query_id"].endswith("_"+t))
        out.append({**r,"query_month":qmonth,"verified_date":vd,"topic":topic,
                    "w":1/np.sqrt(max(1,int(r.get("rank",1)))),"polarity":POLARITY[topic]})
    return pd.DataFrame(out), {"records":len(rows),"usable":len(out),"date_conflicts":conflicts,
                               "month_mismatches":mismatches,"missing_dates":missing}

def monthly():
    d,audit=read_verified()
    d["signed"]=d["w"]*d["polarity"]
    rows=[]
    for month,g in d.groupby("query_month"):
        row={"month":month,"articles":len(g),"intensity":g["w"].sum(),"signed":g["signed"].sum()}
        for t in TOPICS: row[t]=int((g["topic"]==t).sum())
        rows.append(row)
    m=pd.DataFrame(rows).set_index("month").sort_index()
    for c in [x for x in m.columns]:
        m[c+"_sum3"]=m[c].rolling(3,min_periods=1).sum()
    return m,audit

def tensor(periods,n_series):
    m,_=monthly()
    cols=["intensity","signed","natural_emergency","security","enterprise_negative","enterprise_positive","infrastructure",
          "intensity_sum3","signed_sum3","natural_emergency_sum3","security_sum3","enterprise_negative_sum3",
          "enterprise_positive_sum3","infrastructure_sum3"]
    out={}
    for c in cols:
        v=m[c].reindex(periods).fillna(0).to_numpy(float)
        out["news_"+c]=np.tile(v,(n_series,1))
    return out
