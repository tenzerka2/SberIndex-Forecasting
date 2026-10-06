"""Historical news/event features with strict publication-time alignment.

Corpus: data/news/corpus.jsonl, collected from the frozen query plan in data/news/query_plan.csv.
Each record keeps search metadata only. We derive a conservative verified publication date as the
LATEST date that can be recovered from:
  1) the search-provider Published field;
  2) an explicit YYYY/MM/DD, YYYY-MM-DD or YYYYMMDD date embedded in the URL.

Using the latest candidate prevents the failure mode observed during collection where a metadata
timestamp pre-dated the article URL by weeks/months. Records whose verified month differs from the
pre-registered query month are excluded from modelling and retained only for audit.

Features are event counts / rank-weighted intensities by municipality-month. Geo matching is
deterministic and deliberately conservative: exact normalized municipality-core match has priority;
otherwise a mapped region token match broadcasts the event to municipalities in that region.
Homonymous unresolved municipalities are not linked by name.

No NLP model or query tuning is used. Event polarity is defined ex ante by query topic:
enterprise_positive=+1; natural_emergency/security/enterprise_negative/infrastructure=-1.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from .data import ROOT, Panel
from .geo import build_mapping, core, norm

CORPUS = ROOT / "data" / "news" / "corpus.jsonl"
QUERY_PLAN = ROOT / "data" / "news" / "query_plan.csv"
TOPICS = ["natural_emergency", "security", "enterprise_negative", "enterprise_positive", "infrastructure"]
POLARITY = {
    "natural_emergency": -1.0,
    "security": -1.0,
    "enterprise_negative": -1.0,
    "enterprise_positive": 1.0,
    "infrastructure": -1.0,
}
GENERIC_REGION = {
    "область", "области", "край", "края", "республика", "республики", "автономный", "автономная",
    "автономного", "округ", "округа", "город", "федерального", "значения",
}


def _parse_dt(x) -> pd.Timestamp | None:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    try:
        t = pd.Timestamp(x)
        if t.tzinfo is not None:
            t = t.tz_convert(None)
        return t.normalize()
    except Exception:
        return None


def url_dates(url: str) -> list[pd.Timestamp]:
    """Extract plausible calendar dates from URL. Only contest-era years are accepted."""
    s = str(url)
    pats = [
        r"(?<!\d)(20(?:22|23|24|25))[/-](0?[1-9]|1[0-2])[/-]([0-2]?\d|3[01])(?!\d)",
        r"(?<!\d)(20(?:22|23|24|25))(0[1-9]|1[0-2])([0-2]\d|3[01])(?!\d)",
    ]
    out = []
    for p in pats:
        for y, m, d in re.findall(p, s):
            try:
                out.append(pd.Timestamp(year=int(y), month=int(m), day=int(d)))
            except ValueError:
                pass
    return out


def verified_date(record: dict) -> tuple[pd.Timestamp | None, str]:
    """Conservative timestamp: latest valid candidate from metadata and URL."""
    cand: list[tuple[pd.Timestamp, str]] = []
    m = _parse_dt(record.get("published"))
    if m is not None:
        cand.append((m, "metadata"))
    for d in url_dates(record.get("url", "")):
        cand.append((d, "url"))
    if not cand:
        return None, "missing"
    latest = max(cand, key=lambda z: z[0])
    if len({x[0] for x in cand}) > 1:
        return latest[0], "max(metadata,url)"
    return latest


def query_month(query_id: str) -> pd.Timestamp:
    return pd.Timestamp(str(query_id)[:7] + "-01")


def topic(query_id: str) -> str:
    q = str(query_id)
    for t in TOPICS:
        if q.endswith("_" + t):
            return t
    raise ValueError(f"unknown news topic in {query_id}")


def read_corpus() -> pd.DataFrame:
    rows = [json.loads(x) for x in CORPUS.read_text(encoding="utf-8").splitlines() if x.strip()]
    d = pd.DataFrame(rows)
    d["query_month"] = d["query_id"].map(query_month)
    d["topic"] = d["query_id"].map(topic)
    vd = d.apply(lambda r: verified_date(r.to_dict()), axis=1)
    d["verified_date"] = [x[0] for x in vd]
    d["date_rule"] = [x[1] for x in vd]
    d["verified_month"] = pd.to_datetime(d["verified_date"]).dt.to_period("M").dt.to_timestamp()
    d["in_query_month"] = d["verified_month"].eq(d["query_month"])
    d["domain"] = d["url"].map(lambda u: (urlparse(str(u)).hostname or "").lower().removeprefix("www."))
    d["rank_weight"] = 1.0 / np.sqrt(d["rank"].clip(lower=1).astype(float))
    d["polarity"] = d["topic"].map(POLARITY)
    d["text"] = (d["title"].fillna("") + " " + d["snippet"].fillna("")).map(norm)
    return d


def _region_tokens(region: str | None) -> list[str]:
    if not region or pd.isna(region):
        return []
    s = norm(region)
    if s in {"москва", "санкт-петербург", "севастополь"}:
        return [s]
    toks = [x for x in re.findall(r"[а-яa-z-]+", s) if x not in GENERIC_REGION and len(x) >= 4]
    # adjective stems make "Белгородская область" match "Белгородской области"
    out = []
    for x in toks:
        for suf in ("ская", "ской", "ская", "ский", "ского", "ская", "ая", "ий", "ый", "ого"):
            if x.endswith(suf) and len(x) - len(suf) >= 4:
                x = x[: -len(suf)]
                break
        out.append(x)
    return sorted(set(out), key=len, reverse=True)


def _contains(text: str, key: str) -> bool:
    if not key or len(key) < 4:
        return False
    return re.search(rf"(?<![а-яa-z]){re.escape(norm(key))}(?![а-яa-z])", text) is not None


def article_series_links(panel: Panel, strict_month: bool = True) -> pd.DataFrame:
    """Return sparse (article_idx, series, confidence) geo links."""
    news = read_corpus()
    if strict_month:
        news = news[news["in_query_month"] & news["verified_date"].notna()].copy()
    news = news.drop_duplicates("url").reset_index(drop=True)
    geo = build_mapping(panel).set_index("run_id")
    meta = panel.meta.reset_index(drop=True).copy()
    meta["series"] = np.arange(len(meta))
    meta = meta.join(geo[["core", "region", "method"]], on="run_id")
    meta["region_tokens"] = meta["region"].map(_region_tokens)

    rows = []
    for ai, a in news.iterrows():
        txt = a["text"]
        # municipality exact match only when identity is not homonymous
        muni = np.zeros(len(meta), dtype=bool)
        for i, r in meta.iterrows():
            if not r["homonym"] and isinstance(r["core"], str) and len(r["core"]) >= 5 and _contains(txt, r["core"]):
                muni[i] = True
        if muni.any():
            for s in meta.loc[muni, "series"]:
                rows.append((ai, int(s), 1.0, "municipality"))
            continue
        # otherwise broadcast conservatively by distinctive region token
        matched_regions = set()
        for region, toks in meta.dropna(subset=["region"])[["region", "region_tokens"]].drop_duplicates("region").itertuples(index=False):
            if any((tok in txt) for tok in toks):
                matched_regions.add(region)
        for region in matched_regions:
            for s in meta.loc[meta["region"].eq(region), "series"]:
                rows.append((ai, int(s), 0.35, "region"))
    return pd.DataFrame(rows, columns=["article_idx", "series", "geo_confidence", "geo_method"]), news, meta


def feature_frame(panel: Panel, cutoff: pd.Timestamp | None = None) -> pd.DataFrame:
    """Long municipality-month feature frame. Articles after cutoff are ignored."""
    links, news, _ = article_series_links(panel)
    if cutoff is not None:
        cutoff = pd.Timestamp(cutoff) + pd.offsets.MonthEnd(0)
        allowed = set(news.index[news["verified_date"] <= cutoff])
        links = links[links["article_idx"].isin(allowed)]
        news = news.loc[sorted(allowed)].copy()
    if links.empty:
        idx = pd.MultiIndex.from_product([range(len(panel.meta)), panel.periods], names=["series", "month"])
        return pd.DataFrame(index=idx).reset_index()

    z = links.merge(news.reset_index(names="article_idx"), on="article_idx", how="left")
    z["month"] = z["verified_date"].dt.to_period("M").dt.to_timestamp()
    z["w"] = z["rank_weight"] * z["geo_confidence"]
    z["signed_w"] = z["w"] * z["polarity"]
    for t in TOPICS:
        z[f"topic_{t}"] = z["w"] * z["topic"].eq(t)

    agg = z.groupby(["series", "month"]).agg(
        news_intensity=("w", "sum"),
        news_articles=("url", "nunique"),
        news_signed=("signed_w", "sum"),
        news_sources=("domain", "nunique"),
        news_muni_matches=("geo_method", lambda x: int((x == "municipality").sum())),
    )
    for t in TOPICS:
        agg[f"news_{t}"] = z.groupby(["series", "month"])[f"topic_{t}"].sum()

    idx = pd.MultiIndex.from_product([range(len(panel.meta)), panel.periods], names=["series", "month"])
    out = agg.reindex(idx).fillna(0).reset_index()
    return out


def feature_tensor(panel: Panel, cutoff: pd.Timestamp | None = None) -> dict[str, np.ndarray]:
    """Return n_series x T matrices. Includes current month plus strictly backward rolling signals."""
    f = feature_frame(panel, cutoff).set_index(["series", "month"])
    n, T = len(panel.meta), len(panel.periods)
    base_cols = [c for c in f.columns if c.startswith("news_")]
    arr = {}
    for c in base_cols:
        x = f[c].unstack("month").reindex(index=range(n), columns=panel.periods).to_numpy(float)
        arr[c] = x
        arr[c + "_sum3"] = pd.DataFrame(x).T.rolling(3, min_periods=1).sum().T.to_numpy()
        arr[c + "_mom1"] = np.column_stack([np.zeros(n), np.diff(x, axis=1)])
    return arr


def audit() -> dict:
    d = read_corpus()
    return {
        "records": int(len(d)),
        "query_ids": int(d["query_id"].nunique()),
        "verified_dates": int(d["verified_date"].notna().sum()),
        "in_query_month": int(d["in_query_month"].sum()),
        "date_conflicts": int(d["date_rule"].eq("max(metadata,url)").sum()),
        "unique_urls": int(d["url"].nunique()),
        "months": [str(d["query_month"].min().date()), str(d["query_month"].max().date())],
        "topics": d["topic"].value_counts().to_dict(),
    }
