"""Conservative municipality-linked news without an invented region/ID crosswalk.

Uses existing archived search results, NOT a comprehensive historical news archive.
Date metadata cannot certify historical page contents. Matching is an auditable lexical
candidate link, not verified causal attribution. Homonyms and generic adjective names
remain unresolved. Planned-event language is a feature, not proof of future impact.
"""
from __future__ import annotations
import re
import numpy as np
import pandas as pd
from .geo import core, norm
from .news_global import read_verified

PLANNED = re.compile(r"планир|планов|намерен|предстоящ|будет|будут|анонс|запланир|состоится")


def tensor(panel, cutoff=None):
    docs, audit = read_verified()
    docs = docs.sort_values(["verified_date", "rank"]).drop_duplicates("url")
    if cutoff is not None:
        docs = docs[docs.verified_date <= pd.Timestamp(cutoff) + pd.offsets.MonthEnd(0)]
    keys = panel.meta.mo.map(core)
    duplicate_core = keys.duplicated(keep=False)
    patterns = []
    generic = {"центральный", "северный", "южный", "восточный", "западный", "мирный", "советский", "октябрьский", "ленинский"}
    for i, key in enumerate(keys):
        if panel.meta.homonym.iloc[i] or duplicate_core.iloc[i] or len(key) < 4 or key in generic:
            continue
        patterns.append((i, re.compile(r"(?<![а-яa-z])" + re.escape(key) + r"(?![а-яa-z])")))
    n, T = panel.values.shape
    out = {k: np.zeros((n,T)) for k in ["local_news", "local_negative", "local_positive", "local_planned"]}
    month_index = {p:i for i,p in enumerate(panel.periods)}
    links=[]
    for _, a in docs.iterrows():
        t=month_index.get(a.verified_date.to_period("M").to_timestamp())
        if t is None: continue
        txt=norm(str(a.get("title", ""))+" "+str(a.get("snippet", "")))
        for i, pattern in patterns:
            if pattern.search(txt):
                out["local_news"][i,t] += 1
                out["local_negative"][i,t] += a.polarity < 0
                out["local_positive"][i,t] += a.polarity > 0
                out["local_planned"][i,t] += bool(PLANNED.search(txt))
                links.append({"run_id":int(panel.meta.run_id.iloc[i]),"mo":panel.meta.mo.iloc[i],
                              "date":str(a.verified_date.date()),"url":a.url,"title":a.get("title", ""),
                              "link_method":"unique_exact_core_unverified", "planned_language":bool(PLANNED.search(txt))})
    for k,v in list(out.items()):
        out[k+"_sum3"] = pd.DataFrame(v.T).rolling(3,min_periods=1).sum().to_numpy().T
    audit.update(unique_articles=len(docs),linked_articles=len({r['url'] for r in links}),
                 linked_municipalities=len({r['run_id'] for r in links}),links=len(links),
                 unresolved_homonyms=int(panel.meta.homonym.sum()))
    return out, pd.DataFrame(links), audit
