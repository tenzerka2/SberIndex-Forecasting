"""Frozen-model news ablation.

Tests the pre-registered historical-news corpus as an EXOGENOUS feature source without changing V3:
  A) forecasting: a time-safe pooled Ridge predicts the log residual of frozen V3 using only news
     features observed at each forecast origin. The Ridge is refit at every origin on pairs whose
     target is already observed (target_date <= current origin).
  B) early warning: the existing rolling LightGBM protocol is rerun for base vs base+news.

No query, topic, alpha, feature set or threshold is tuned on the exact window.
Outputs:
  outputs/news_audit.json
  outputs/news_forecast_ablation.csv
  outputs/news_early_warning_ablation.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))

from sbx import backtest as B  # noqa: E402
from sbx import early_warning as EW  # noqa: E402
from sbx import ew_features as F  # noqa: E402
from sbx import news as N  # noqa: E402
from sbx.data import load_panel, national_monthly, weekly_monthly  # noqa: E402
from sbx.final import final_models  # noqa: E402
import early_warning_v2 as EW2  # noqa: E402

OUT = ROOT / "outputs"
ORIGINS = pd.date_range("2024-02-01", "2024-11-01", freq="MS")
WINDOWS = {
    "exact": B.EXACT_ORIGINS,
    "other": pd.to_datetime(["2024-04-01", "2024-05-01", "2024-10-01", "2024-11-01"]),
    "apr_nov": pd.date_range("2024-04-01", "2024-11-01", freq="MS"),
}
RIDGE_ALPHA = 10.0  # fixed before seeing results
CLIP_LOG_CORRECTION = 0.20

# deliberately small / interpretable feature set fixed before ablation
NEWS_COLS = [
    "news_intensity", "news_intensity_sum3",
    "news_articles", "news_articles_sum3",
    "news_signed", "news_signed_sum3",
    "news_natural_emergency_sum3", "news_security_sum3",
    "news_enterprise_negative_sum3", "news_enterprise_positive_sum3",
    "news_infrastructure_sum3",
]


def _news_matrix(panel):
    ft = N.feature_tensor(panel)
    keep = [c for c in NEWS_COLS if c in ft]
    return ft, keep


def _x_for_rows(df: pd.DataFrame, ft: dict[str, np.ndarray], cols: list[str], periods: pd.DatetimeIndex):
    pos = {pd.Timestamp(p): j for j, p in enumerate(periods)}
    mats = []
    ss = df["series"].to_numpy(int)
    oo = df["origin"].map(pos).to_numpy(int)
    for c in cols:
        mats.append(ft[c][ss, oo])
    mats.append(df["h"].to_numpy(float))
    return np.column_stack(mats)


def forecast_ablation(panel) -> pd.DataFrame:
    nat, wk = national_monthly(), weekly_monthly()
    ctx = lambda o: {"origin": o, "national": nat[nat.index <= o], "weekly": wk[wk.index <= o]}  # noqa: E731
    allp = B.run(panel, {"v3": final_models()["v3"]}, ORIGINS, ctx_fn=ctx)
    ft, cols = _news_matrix(panel)
    frames = []
    for origin in ORIGINS:
        te = allp[allp["origin"].eq(origin)].copy()
        # At origin T, only forecast errors with realized target <= T are known.
        tr = allp[(allp["origin"] < origin) & (allp["target_date"] <= origin)].copy()
        te["v3_news"] = te["v3"]
        if len(tr) >= 3000 and tr["origin"].nunique() >= 2:
            Xtr = _x_for_rows(tr, ft, cols, panel.periods)
            Xte = _x_for_rows(te, ft, cols, panel.periods)
            ytr = np.log(tr["y"] / tr["v3"]).to_numpy()
            model = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))
            model.fit(Xtr, ytr)
            corr = np.clip(model.predict(Xte), -CLIP_LOG_CORRECTION, CLIP_LOG_CORRECTION)
            te["v3_news"] = te["v3"] * np.exp(corr)
        frames.append(te)
    d = pd.concat(frames, ignore_index=True)
    rows = []
    for w, origins in WINDOWS.items():
        g = d[d["origin"].isin(origins)]
        for m in ["v3", "v3_news"]:
            rows.append({"window": w, "model": m, **B.metrics(g, m)})
        a = (g["v3_news"] - g["y"]).abs().mean()
        b = (g["v3"] - g["y"]).abs().mean()
        rows.append({"window": w, "model": "delta_news_minus_v3", "MAE": float(a - b),
                     "R2_level": np.nan, "R2_growth": np.nan, "wMAPE_pct": np.nan, "n": len(g)})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "news_forecast_ablation.csv", index=False, float_format="%.5f")
    return out


def _labels(ev: np.ndarray, task: str) -> np.ndarray:
    return EW2.labels(ev, task)


def _stack(feats: dict[str, np.ndarray], names: list[str], ts) -> np.ndarray:
    return np.column_stack([feats[k][:, list(ts)].T.ravel() for k in names])


def _rolling_variant(L, feats: dict[str, np.ndarray], names: list[str], task: str):
    n = L.shape[0]
    ev_full = EW.events(L)
    y_full = _labels(ev_full, task)
    chunks = []
    for t in EW2.TEST[task]:
        ev_t = EW.events(L[:, : t + 1])
        y_t = _labels(ev_t, task)
        train_rows = list(range(EW2.FIRST_TRAIN_ROW, t - EW2.EMBARGO[task] + 1))
        ytr = y_t[:, train_rows].T.ravel()
        yte = y_full[:, t]
        month_tr = np.repeat(train_rows, n)
        Xtr = _stack(feats, names, train_rows)
        Xte = _stack(feats, names, [t])
        # same inner time split / threshold rule as the frozen EW benchmark
        inner_cut = train_rows[-2]
        fit_m = month_tr < inner_cut
        thr = 0.5
        if ytr[fit_m].any() and ytr[~fit_m].any():
            inner = EW2.clf().fit(Xtr[fit_m], ytr[fit_m])
            thr = EW.best_threshold(ytr[~fit_m], inner.predict_proba(Xtr[~fit_m])[:, 1])
        model = EW2.clf().fit(Xtr, ytr)
        score = model.predict_proba(Xte)[:, 1]
        chunks.append(pd.DataFrame({"month": t, "series": np.arange(n), "y": yte,
                                    "score": score, "alarm": score >= thr}))
    return pd.concat(chunks, ignore_index=True)


def early_warning_ablation(panel) -> pd.DataFrame:
    groups = F.build_groups(panel)
    base = groups["base"]
    news = N.feature_tensor(panel)
    base_names = list(base)
    news_names = [c for c in NEWS_COLS if c in news]
    feats = {**base, **news}

    rows = []
    for task in ["detect", "predict"]:
        a = _rolling_variant(panel.logs, feats, base_names, task)
        b = _rolling_variant(panel.logs, feats, base_names + news_names, task)
        for name, d in [("base", a), ("base_news", b)]:
            m = EW2.metrics(d)
            rows.append({"task": task, "model": name, **m})
        # month-cluster bootstrap via existing routine
        diff = EW2.boot_ap_diff(b, a)
        rows.append({"task": task, "model": "delta_news_minus_base",
                     "PR_AUC": diff["dPR_AUC"], "lift_vs_base_rate": np.nan,
                     "precision": np.nan, "recall": np.nan, "F1": np.nan,
                     "false_alarms_per_100": np.nan, "alarms_per_100": np.nan,
                     "positives": int(a["y"].sum()), "rows": int(len(a)),
                     "base_rate": float(a["y"].mean()), "months": int(a["month"].nunique()),
                     "ci95_lo": diff["ci95_by_month"][0], "ci95_hi": diff["ci95_by_month"][1]})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "news_early_warning_ablation.csv", index=False, float_format="%.6f")
    return out


def main():
    panel = load_panel()
    audit = N.audit()
    links, news, meta = N.article_series_links(panel)
    audit.update({
        "usable_articles_after_date_and_month_check": int(len(news)),
        "geo_links": int(len(links)),
        "linked_articles": int(links["article_idx"].nunique()) if len(links) else 0,
        "municipality_exact_links": int((links["geo_method"] == "municipality").sum()) if len(links) else 0,
        "region_links": int((links["geo_method"] == "region").sum()) if len(links) else 0,
        "mapped_series": int(meta["region"].notna().sum()),
    })
    (OUT / "news_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    f = forecast_ablation(panel)
    e = early_warning_ablation(panel)
    print("NEWS AUDIT")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print("\nFORECAST")
    print(f.to_string(index=False))
    print("\nEARLY WARNING")
    print(e.to_string(index=False))


if __name__ == "__main__":
    main()
