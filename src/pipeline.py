from __future__ import annotations

import json
import math
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import average_precision_score, mean_absolute_error, precision_recall_curve, r2_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

SEED = 42
H = [1, 3, 6, 12]
VSTART = pd.Timestamp("2024-07-01")
VEND = pd.Timestamp("2024-12-01")


def fpat(pattern: str) -> Path:
    files = sorted(RAW.glob(pattern))
    if not files:
        raise FileNotFoundError(f"Missing data file matching {pattern} in {RAW}")
    return files[0]


def load():
    muni = pd.read_csv(fpat("potrebitelskie-beznalicnye*.csv"), sep=";")
    muni["period"] = pd.to_datetime(muni["period"])

    totals = muni[muni["category_15"].eq("Все категории")]
    ambiguous = totals[totals.duplicated(["mo", "period"], keep=False)]["mo"].unique().tolist()

    clean = muni[~muni["mo"].isin(ambiguous)].copy()
    total_clean = clean[clean["category_15"].eq("Все категории")]
    cnt = total_clean.groupby("mo")["period"].nunique()
    full = cnt[cnt.eq(24)].index
    clean = clean[clean["mo"].isin(full)]

    piv = (
        clean.pivot(index=["mo", "period"], columns="category_15", values="value")
        .reset_index()
        .sort_values(["mo", "period"])
    )
    piv.columns = [str(c) for c in piv.columns]

    spend = pd.read_csv(fpat("consumer-spending_ru*.csv"), sep=";")
    spend["period"] = pd.to_datetime(spend["period"])
    spend = spend[spend["type"].eq("Всего")][["period", "value"]].rename(columns={"value": "nat_spend"})

    growth = pd.read_csv(fpat("consumer-spending-growth*.csv"), sep=";")
    growth["period"] = pd.to_datetime(growth["period"])
    growth = (
        growth[growth["type"].eq("Всего")]
        .pivot(index="period", columns="value_type", values="value")
        .reset_index()
        .rename(columns={"Номинальное": "nat_growth_nominal", "Реальное": "nat_growth_real"})
    )

    index_sa = pd.read_csv(fpat("consumper-spending-index-sa*.csv"), sep=";")
    index_sa["period"] = pd.to_datetime(index_sa["period"])
    index_sa = index_sa[index_sa["type"].eq("Всего")][["period", "value"]].rename(
        columns={"value": "nat_index_sa"}
    )

    weekly = pd.read_csv(fpat("ver-izmenenie-trat-po-kategoriyam*.csv"), sep=";")
    weekly["period"] = pd.to_datetime(weekly["period"]).dt.to_period("M").dt.to_timestamp()
    pulse = (
        weekly.groupby("period")["value"]
        .agg(pulse_mean="mean", pulse_median="median", pulse_std="std")
        .reset_index()
    )

    nat = (
        spend.merge(growth, on="period", how="outer")
        .merge(index_sa, on="period", how="outer")
        .merge(pulse, on="period", how="left")
        .sort_values("period")
    )

    meta = {
        "ambiguous_labels_excluded": len(ambiguous),
        "ambiguous_examples": ambiguous[:10],
        "full_history_municipalities": len(full),
        "all_unique_labels": totals["mo"].nunique(),
    }
    return muni, piv, nat, meta


def make_features(panel: pd.DataFrame, nat: pd.DataFrame) -> pd.DataFrame:
    d = panel.copy()
    target_col = "Все категории"
    cats = [c for c in d.columns if c not in ["mo", "period"]]
    g = d.groupby("mo", sort=False)

    d["y_lag0"] = d[target_col]
    for k in [1, 2, 3, 6, 12]:
        d[f"y_lag{k}"] = g[target_col].shift(k)

    for w in [3, 6, 12]:
        roll = g[target_col].rolling(w, min_periods=1)
        d[f"y_mean{w}"] = roll.mean().reset_index(level=0, drop=True).values
        d[f"y_std{w}"] = roll.std(ddof=0).reset_index(level=0, drop=True).fillna(0).values
        d[f"y_min{w}"] = roll.min().reset_index(level=0, drop=True).values
        d[f"y_max{w}"] = roll.max().reset_index(level=0, drop=True).values
        d[f"y_slope{w}"] = (d[target_col] - g[target_col].shift(w - 1)) / max(1, w - 1)

    for k in [1, 3, 6, 12]:
        d[f"y_relchg{k}"] = d[target_col] / g[target_col].shift(k) - 1

    for c in cats:
        if c == target_col:
            continue
        d[f"cat__{c}__chg3"] = d[c] / g[c].shift(3) - 1
        d[f"cat__{c}__ratio"] = d[c] / d[target_col]

    n = nat.copy()
    nat_cols = [x for x in n.columns if x != "period"]
    for c in nat_cols:
        for k in [1, 3, 6, 12]:
            n[f"{c}_lag{k}"] = n[c].shift(k)
        n[f"{c}_chg1"] = n[c].pct_change(fill_method=None)
        n[f"{c}_chg3"] = n[c].pct_change(3, fill_method=None)

    d = d.merge(n, on="period", how="left")
    d["origin_month"] = d["period"].dt.month
    d["origin_sin"] = np.sin(2 * np.pi * d["origin_month"] / 12)
    d["origin_cos"] = np.cos(2 * np.pi * d["origin_month"] / 12)
    d = d.rename(columns={"period": "origin_date"})

    ex = pd.concat(
        [d.assign(horizon=h, target_date=d["origin_date"] + pd.DateOffset(months=h)) for h in H],
        ignore_index=True,
    )

    target_map = d[["mo", "origin_date", target_col]].rename(
        columns={"origin_date": "target_date", target_col: "target"}
    )
    ex = ex.merge(target_map, on=["mo", "target_date"], how="left")
    ex["target_month"] = ex["target_date"].dt.month
    ex["target_sin"] = np.sin(2 * np.pi * ex["target_month"] / 12)
    ex["target_cos"] = np.cos(2 * np.pi * ex["target_month"] / 12)
    return ex.replace([np.inf, -np.inf], np.nan)


def seasonal_naive(panel: pd.DataFrame, valid: pd.DataFrame):
    lookup = panel.set_index(["mo", "period"])["Все категории"]
    out = []
    for mo, origin, target in valid[["mo", "origin_date", "target_date"]].itertuples(index=False, name=None):
        same = target - pd.DateOffset(months=12)
        value = lookup.get((mo, same), np.nan) if same <= origin else np.nan
        if pd.isna(value):
            value = lookup.get((mo, origin), np.nan)
        out.append(value)
    return out


def additive_baseline(panel: pd.DataFrame, valid: pd.DataFrame):
    wide = panel.pivot(index="period", columns="mo", values="Все категории").sort_index()
    out = []

    for origin, target, horizon in (
        valid[["origin_date", "target_date", "horizon"]].drop_duplicates().itertuples(index=False, name=None)
    ):
        hist = wide.loc[:origin]
        n = len(hist)
        t = np.arange(n, dtype=float)
        X = np.column_stack([np.ones(n), t, np.sin(2 * np.pi * t / 12), np.cos(2 * np.pi * t / 12)])
        Y = np.log1p(hist.values.astype(float))
        B = np.linalg.pinv(X.T @ X + 20 * np.eye(X.shape[1])) @ X.T @ Y

        ht = (target.to_period("M") - hist.index[0].to_period("M")).n
        xv = np.array([1.0, ht, math.sin(2 * math.pi * ht / 12), math.cos(2 * math.pi * ht / 12)])
        pred = np.expm1(xv @ B)
        last = hist.iloc[-1].values.astype(float)
        pred = np.clip(pred, 0.4 * last, 2.5 * last)
        out.extend((m, origin, target, horizon, z) for m, z in zip(wide.columns, pred))

    return pd.DataFrame(
        out, columns=["mo", "origin_date", "target_date", "horizon", "additive_baseline"]
    )


def metric(valid: pd.DataFrame, cols: list[str]):
    rows = []
    for h, g in valid.groupby("horizon"):
        for c in cols:
            q = g[["target", c]].dropna()
            y = q["target"].to_numpy()
            p = q[c].to_numpy()
            rows.append(
                {
                    "horizon_months": int(h),
                    "model": c,
                    "n": len(q),
                    "MAE": float(mean_absolute_error(y, p)),
                    "R2": float(r2_score(y, p)),
                    "wMAPE_pct": float(np.abs(y - p).sum() / np.abs(y).sum() * 100),
                    "MdAPE_pct": float(np.median(np.abs(y - p) / np.maximum(np.abs(y), 1)) * 100),
                }
            )
    return pd.DataFrame(rows)


def change_points(panel: pd.DataFrame):
    wide = panel.pivot(index="period", columns="mo", values="Все категории").sort_index()
    Y = np.log1p(wide.values.astype(float))
    D = np.diff(Y, axis=0)
    med = np.nanmedian(D, axis=0)
    scale = np.nanmedian(np.abs(D - med), axis=0) * 1.4826
    scale = np.maximum(np.where(np.isfinite(scale), scale, 0.03), 0.03)

    score = np.zeros_like(Y)
    for i in range(3, Y.shape[0] - 1):
        score[i] = np.abs(
            np.nanmedian(Y[i : min(i + 2, Y.shape[0])], axis=0) - np.nanmedian(Y[i - 3 : i], axis=0)
        ) / scale

    labels = (score >= 2.5).astype(np.int8)
    for i in range(1, labels.shape[0]):
        both = (labels[i] == 1) & (labels[i - 1] == 1)
        cur = score[i] >= score[i - 1]
        labels[i - 1, both & cur] = 0
        labels[i, both & ~cur] = 0

    return pd.DataFrame(
        {
            "period": np.repeat(wide.index.values, len(wide.columns)),
            "mo": np.tile(wide.columns.values, len(wide.index)),
            "cp_score": score.ravel(),
            "is_change": labels.ravel(),
        }
    )


def main():
    started = time.time()
    muni, panel, nat, meta = load()
    print("load", time.time() - started, meta, flush=True)

    ex = make_features(panel, nat)
    print("features", ex.shape, time.time() - started, flush=True)

    mo_map = {m: i for i, m in enumerate(sorted(ex["mo"].unique()))}
    ex["mo_code"] = ex["mo"].map(mo_map).astype("category")

    exclude = {"origin_date", "target_date", "target", "mo", "Все категории"}
    features = ["mo_code"] + [c for c in ex.columns if c not in exclude and c != "mo_code"]

    train = ex[ex["target"].notna() & (ex["target_date"] < VSTART)]
    valid = ex[ex["target"].notna() & ex["target_date"].between(VSTART, VEND)].copy()

    model = LGBMRegressor(
        n_estimators=280,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=35,
        reg_lambda=3,
        objective="regression",
        random_state=SEED,
        verbosity=-1,
    )
    model.fit(train[features], np.log1p(train["target"]), categorical_feature=["mo_code"])

    valid["gbm_log"] = np.clip(np.expm1(model.predict(valid[features])), 0, None)
    valid["seasonal_naive"] = seasonal_naive(panel, valid)
    valid = valid.merge(
        additive_baseline(panel, valid),
        on=["mo", "origin_date", "target_date", "horizon"],
        how="left",
    )

    metrics = metric(valid, ["seasonal_naive", "additive_baseline", "gbm_log"])
    metrics.to_csv(OUT / "forecast_metrics.csv", index=False)
    valid[
        [
            "mo",
            "origin_date",
            "target_date",
            "horizon",
            "target",
            "seasonal_naive",
            "additive_baseline",
            "gbm_log",
        ]
    ].to_csv(OUT / "backtest_predictions.csv", index=False)
    print(metrics.to_string(index=False), flush=True)

    cp = change_points(panel)
    cp.to_csv(OUT / "detected_change_points.csv", index=False)
    cpmap = cp.set_index(["mo", "period"])["is_change"]

    base = ex[ex["horizon"].eq(1)].drop_duplicates(["mo", "origin_date"]).copy()
    base["shock_next_3m"] = [
        int(any(cpmap.get((m, o + pd.DateOffset(months=k)), 0) == 1 for k in [1, 2, 3]))
        for m, o in base[["mo", "origin_date"]].itertuples(index=False, name=None)
    ]

    base = base[base["origin_date"] <= pd.Timestamp("2024-09-01")]
    shock_features = [c for c in features if c != "horizon"]

    train_shock = base[base["origin_date"] < pd.Timestamp("2024-05-01")]
    valid_shock = base[
        base["origin_date"].between(pd.Timestamp("2024-05-01"), pd.Timestamp("2024-09-01"))
    ].copy()

    classifier = LGBMClassifier(
        n_estimators=180,
        learning_rate=0.06,
        num_leaves=23,
        min_child_samples=45,
        class_weight="balanced",
        reg_lambda=2,
        random_state=SEED,
        verbosity=-1,
    )
    classifier.fit(
        train_shock[shock_features],
        train_shock["shock_next_3m"],
        categorical_feature=["mo_code"],
    )

    prob = classifier.predict_proba(valid_shock[shock_features])[:, 1]
    precision, recall, thresholds = precision_recall_curve(valid_shock["shock_next_3m"], prob)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-9)
    best = int(np.argmax(f1[:-1]))
    threshold = float(thresholds[best])

    pred = prob >= threshold
    truth = valid_shock["shock_next_3m"].values.astype(bool)
    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())

    shock_metrics = {
        "PR_AUC": float(average_precision_score(truth, prob)),
        "threshold": threshold,
        "precision": float(tp / max(tp + fp, 1)),
        "recall": float(tp / max(tp + fn, 1)),
        "F1": float(2 * tp / max(2 * tp + fp + fn, 1)),
        "n_val": int(len(valid_shock)),
        "positive_rate": float(truth.mean()),
    }

    valid_shock["shock_risk"] = prob
    valid_shock[["mo", "origin_date", "shock_next_3m", "shock_risk"]].to_csv(
        OUT / "shock_backtest.csv", index=False
    )
    (OUT / "shock_metrics.json").write_text(
        json.dumps(shock_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("shock", shock_metrics, flush=True)

    observed = ex[ex["target"].notna()]
    final_model = LGBMRegressor(
        n_estimators=320,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=35,
        reg_lambda=3,
        objective="regression",
        random_state=SEED,
        verbosity=-1,
    )
    final_model.fit(
        observed[features],
        np.log1p(observed["target"]),
        categorical_feature=["mo_code"],
    )

    future = ex[
        ex["origin_date"].eq(pd.Timestamp("2024-12-01")) & ex["target"].isna()
    ].copy()
    future["forecast"] = np.clip(np.expm1(final_model.predict(future[features])), 0, None)
    future[["mo", "origin_date", "target_date", "horizon", "forecast"]].to_csv(
        OUT / "future_forecasts_2025.csv", index=False
    )

    pd.DataFrame(
        {"feature": features, "importance": final_model.feature_importances_}
    ).sort_values("importance", ascending=False).to_csv(OUT / "feature_importance.csv", index=False)

    latest = ex[
        ex["horizon"].eq(1) & ex["origin_date"].eq(pd.Timestamp("2024-12-01"))
    ].drop_duplicates(["mo", "origin_date"]).copy()
    latest["shock_risk"] = classifier.predict_proba(latest[shock_features])[:, 1]
    latest[["mo", "origin_date", "shock_risk"]].sort_values(
        "shock_risk", ascending=False
    ).to_csv(OUT / "shock_risk_latest.csv", index=False)

    summary = {
        **meta,
        "municipal_period_start": str(muni["period"].min().date()),
        "municipal_period_end": str(muni["period"].max().date()),
        "validation_target_window": "2024-07..2024-12",
        "features": len(features),
        "shock_metrics": shock_metrics,
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("done", time.time() - started, summary, flush=True)


if __name__ == "__main__":
    main()
