"""Early warning v2: rolling-origin evaluation, feature-group ablation and lead-lag analysis.

Two separate tasks (never pooled):
  detect   row (i, t) positive if a structural shift started at tau in [t-2, t]
  predict  row (i, t) positive if a structural shift starts at tau in [t+1, t+3]

Rolling protocol (expanding window, retrained every test month t):
  * labels for training rows are recomputed from data <= t only (EW.events(L[:, :t+1]));
  * embargo: detect trains on rows t' <= t-2, predict on rows t' <= t-5, so every training label
    is fully observed at t;
  * the decision threshold is chosen on the last two training months by a model fitted on the
    earlier training months; the final model is then refitted on all training rows;
  * test labels come from the full data. Test months: detect 2024-01..2024-10, predict 2024-03..2024-07.

Feature groups (src/sbx/ew_features.py): base, category (municipal monthly categories), national,
weekly (national weekly categories, summary features only). National and weekly features are the
same for all municipalities in a month: they can only change WHEN alarms are raised, not WHERE.

Outputs: outputs/ew_v2_metrics.csv, outputs/ew_v2_ablation.csv, outputs/ew_v2_leadlag.csv,
         outputs/ew_v2_rate_correlations.csv, outputs/ew_v2_summary.json
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sbx import early_warning as EW  # noqa: E402
from sbx import ew_features as F  # noqa: E402
from sbx.data import ROOT, load_panel, month_index  # noqa: E402

warnings.filterwarnings("ignore")
OUT = ROOT / "outputs"
SEED = 42
FIRST_TRAIN_ROW = month_index(pd.Timestamp("2023-07-01"))
TEST = {
    "detect": range(month_index(pd.Timestamp("2024-01-01")), month_index(pd.Timestamp("2024-10-01")) + 1),
    "predict": range(month_index(pd.Timestamp("2024-03-01")), month_index(pd.Timestamp("2024-07-01")) + 1),
}
EMBARGO = {"detect": 2, "predict": 5}
WEEKLY_SUMMARY = ["w_mean", "w_disp", "w_intra_mom", "w_mean_acc", "w_disp_chg", "w_acc_max", "w_acc_disp"]
VARIANTS = {
    "sup_base": ["base"],
    "sup_base_category": ["base", "category"],
    "sup_base_national": ["base", "national"],
    "sup_base_weekly": ["base", "weekly"],
    "sup_all": ["base", "category", "national", "weekly"],
    "sup_category_only": ["category"],
}
UNSUP = ["jump_raw", "bocpd", "pelt", "cusum", "page_hinkley", "cat_jump_max"]


def labels(ev: np.ndarray, task: str) -> np.ndarray:
    n, T = ev.shape
    y = np.zeros((n, T), dtype=bool)
    for t in range(T):
        y[:, t] = ev[:, max(0, t - 2) : t + 1].any(axis=1) if task == "detect" else ev[:, t + 1 : t + 4].any(axis=1)
    return y


def clf():
    from lightgbm import LGBMClassifier

    return LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=100,
                          subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0,
                          class_weight="balanced", random_state=SEED, verbosity=-1, n_jobs=4)


def stack(feats: dict[str, np.ndarray], names: list[str], ts) -> np.ndarray:
    return np.column_stack([feats[k][:, list(ts)].T.ravel() for k in names])


def rolling(L, feats_all, groups, task):
    """Returns per-variant arrays of (month, y, score, alarm) over the test rows."""
    n, T = L.shape
    ev_full = EW.events(L)
    y_full = labels(ev_full, task)
    out = {v: [] for v in list(VARIANTS) + UNSUP}
    for t in TEST[task]:
        ev_t = EW.events(L[:, : t + 1])
        y_t = labels(ev_t, task)
        train_rows = list(range(FIRST_TRAIN_ROW, t - EMBARGO[task] + 1))
        ytr = y_t[:, train_rows].T.ravel()
        yte = y_full[:, t]
        month_tr = np.repeat(train_rows, n)
        inner_cut = train_rows[-2]
        for v, gs in VARIANTS.items():
            names = [k for g in gs for k in groups[g]]
            Xtr, Xte = stack(feats_all, names, train_rows), stack(feats_all, names, [t])
            # Inner training must itself respect label maturity at the first validation origin.
            # Using all rows < inner_cut trains on labels that need validation-period outcomes.
            fit_m = month_tr <= inner_cut - EMBARGO[task]
            thr = 0.5
            if ytr[fit_m].any() and ytr[~fit_m].any():
                inner = clf().fit(Xtr[fit_m], ytr[fit_m])
                thr = EW.best_threshold(ytr[~fit_m], inner.predict_proba(Xtr[~fit_m])[:, 1])
            s = clf().fit(Xtr, ytr).predict_proba(Xte)[:, 1]
            out[v].append(pd.DataFrame({"month": t, "series": np.arange(n), "y": yte, "score": s, "alarm": s >= thr}))
        for u in UNSUP:
            S = np.nan_to_num(feats_all[u])
            str_ = S[:, train_rows].T.ravel()
            thr = EW.best_threshold(ytr, str_) if ytr.any() else np.inf
            s = S[:, t]
            out[u].append(pd.DataFrame({"month": t, "series": np.arange(n), "y": yte, "score": s, "alarm": s >= thr}))
        print(task, "month", t, "done", flush=True)
    return {k: pd.concat(v, ignore_index=True) for k, v in out.items()}, ev_full


def metrics(df: pd.DataFrame) -> dict:
    y, a, s = df["y"].to_numpy(), df["alarm"].to_numpy(), df["score"].to_numpy()
    tp, fp, fn = int((a & y).sum()), int((a & ~y).sum()), int((~a & y).sum())
    p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    per_month = [average_precision_score(g["y"], g["score"]) for _, g in df.groupby("month") if g["y"].any()]
    base = y.mean()
    ap = average_precision_score(y, s)
    return {"PR_AUC": ap, "lift_vs_base_rate": ap / base, "PR_AUC_median_month": float(np.median(per_month)),
            "precision": p, "recall": r, "F1": 2 * p * r / max(p + r, 1e-12),
            "false_alarms_per_100": 100 * fp / len(y), "alarms_per_100": 100 * a.sum() / len(y),
            "positives": int(y.sum()), "rows": int(len(y)), "base_rate": base, "months": int(df["month"].nunique())}


def boot_ap_diff(a: pd.DataFrame, b: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    months = a["month"].unique()
    ga, gb = dict(list(a.groupby("month"))), dict(list(b.groupby("month")))
    diffs = []
    for _ in range(n_boot):
        ms = rng.choice(months, len(months), replace=True)
        A = pd.concat([ga[m] for m in ms]); Bb = pd.concat([gb[m] for m in ms])
        if A["y"].any():
            diffs.append(average_precision_score(A["y"], A["score"]) - average_precision_score(Bb["y"], Bb["score"]))
    diffs = np.array(diffs)
    return {"dPR_AUC": average_precision_score(a["y"], a["score"]) - average_precision_score(b["y"], b["score"]),
            "ci95_by_month": [float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))],
            "p_le0": float((diffs <= 0).mean())}


def event_level(df: pd.DataFrame, ev_full: np.ndarray, months) -> dict:
    """Share of shifts (start tau in test detect months) that get an alarm at tau..tau+2."""
    alarm = np.zeros_like(ev_full, dtype=bool)
    for m, g in df.groupby("month"):
        alarm[g["series"].to_numpy()[g["alarm"].to_numpy()], m] = True
    hits, delays, total = 0, [], 0
    lo, hi = min(months), max(months)
    for i, tau in zip(*np.where(ev_full)):
        if lo <= tau and tau + 2 <= hi:
            total += 1
            w = np.where(alarm[i, tau : tau + 3])[0]
            if len(w):
                hits += 1; delays.append(int(w[0]))
    return {"events": total, "detected_share": hits / max(total, 1),
            "median_delay_months": float(np.median(delays)) if delays else float("nan")}


def lead_lag(panel, L) -> pd.DataFrame:
    """Do shifts in municipal category series precede shifts in the total series?"""
    ev_tot = EW.events(L)
    C = F.category_matrices(panel)
    linked = ~np.isnan(C[F.CATEGORIES[0]][:, 0])
    ev_cat = np.zeros_like(ev_tot)
    for c in F.CATEGORIES:
        x = C[c][linked]
        e = np.zeros_like(ev_tot[linked]); e[:] = EW.events(x)
        ev_cat[linked] |= e
    rows = []
    T = L.shape[1]
    tau_idx = [(i, t) for i, t in zip(*np.where(ev_tot)) if linked[i] and 6 <= t <= T - 3]
    for lag_name, offs in [("cat_before_-3..-1", range(-3, 0)), ("cat_same_month", [0]), ("cat_after_+1..+3", range(1, 4))]:
        hit = np.mean([ev_cat[i, [t + o for o in offs if 0 <= t + o < T]].any() for i, t in tau_idx])
        rng = np.random.default_rng(0)
        ri = rng.choice(np.where(linked)[0], 20000); rt = rng.integers(6, T - 3, 20000)
        base = np.mean([ev_cat[i, [t + o for o in offs if 0 <= t + o < T]].any() for i, t in zip(ri, rt)])
        rows.append({"window": lag_name, "P(cat shift | total shift)": hit, "P(cat shift) random": base,
                     "lift": hit / max(base, 1e-12), "n_total_shifts": len(tau_idx)})
    return pd.DataFrame(rows)


def rate_correlations(L, nat, wk, periods) -> pd.DataFrame:
    """Monthly share of municipalities whose shift starts in month t vs national/weekly signals at
    t-1..t-3 (exploratory: ~15 months, no multiple-testing control beyond reporting all)."""
    ev = EW.events(L)
    rate = pd.Series(ev.mean(axis=0), index=periods)
    rate = rate[(rate.index >= "2023-07-01") & (rate.index <= "2024-10-01")]
    sig = {**{k: v[0] for k, v in nat.items() if k.endswith("_acc") or k.startswith("nat_sa")},
           **{k: v[0] for k, v in wk.items() if k in WEEKLY_SUMMARY}}
    rows = []
    for k, arr in sig.items():
        s = pd.Series(arr, index=periods)
        for lag in (1, 2, 3):
            x = s.shift(lag).reindex(rate.index)
            ok = x.notna()
            if ok.sum() >= 8:
                rho, p = spearmanr(x[ok], rate[ok])
                rows.append({"signal": k, "lag_months": lag, "n_months": int(ok.sum()), "spearman": rho, "p": p})
    return pd.DataFrame(rows).sort_values("p")


def main():
    t0 = time.time()
    panel = load_panel()
    L = panel.logs
    groups_feats = F.build_groups(panel)
    groups_feats["weekly"] = {k: v for k, v in groups_feats["weekly"].items() if k in WEEKLY_SUMMARY}
    feats_all = {k: v for g in groups_feats.values() for k, v in g.items()}
    groups = {g: list(v) for g, v in groups_feats.items()}
    print("features", {g: len(v) for g, v in groups.items()}, round(time.time() - t0, 1), "s", flush=True)

    metric_rows, abl_rows, summary = [], [], {}
    for task in ["detect", "predict"]:
        res, ev_full = rolling(L, feats_all, groups, task)
        for v, df in res.items():
            row = {"task": task, "method": v, **metrics(df)}
            if task == "detect":
                row.update(event_level(df, ev_full, TEST[task]))
            metric_rows.append(row)
        for v in VARIANTS:
            if v != "sup_base":
                abl_rows.append({"task": task, "variant": v, "vs": "sup_base", **boot_ap_diff(res[v], res["sup_base"])})
        abl_rows.append({"task": task, "variant": "sup_base", "vs": "jump_raw", **boot_ap_diff(res["sup_base"], res["jump_raw"])})
    met = pd.DataFrame(metric_rows)
    abl = pd.DataFrame(abl_rows)
    ll = lead_lag(panel, L)
    rc = rate_correlations(L, groups_feats["national"], groups_feats["weekly"], panel.periods)
    met.to_csv(OUT / "ew_v2_metrics.csv", index=False)
    abl.to_csv(OUT / "ew_v2_ablation.csv", index=False)
    ll.to_csv(OUT / "ew_v2_leadlag.csv", index=False)
    rc.to_csv(OUT / "ew_v2_rate_correlations.csv", index=False)
    summary = {"metrics": met.to_dict("records"), "ablation": abl.to_dict("records"),
               "lead_lag": ll.to_dict("records"), "top_rate_correlations": rc.head(10).to_dict("records"),
               "n_rate_correlation_tests": int(len(rc))}
    (OUT / "ew_v2_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(met.round(4).to_string(index=False))
    print(abl.round(4).to_string(index=False))
    print(ll.round(4).to_string(index=False))
    print(rc.head(10).round(3).to_string(index=False))
    print("total", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()
