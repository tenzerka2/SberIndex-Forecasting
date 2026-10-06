"""Assemble final competition artifacts from the benchmark outputs.

Reads outputs written by rolling_eval.py, category_replication.py, early_warning_eval.py,
early_warning_v2.py; additionally runs the long-horizon (h = 4..6) backtest of the finals.
Writes FINAL_METRICS.csv, ABLATION.csv (repo root) and figures/*.png.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sbx import backtest as B  # noqa: E402
from sbx import early_warning as EW  # noqa: E402
from sbx.data import ROOT, load_panel, national_monthly, weekly_monthly  # noqa: E402
from sbx.final import final_models  # noqa: E402

OUT, FIG = ROOT / "outputs", ROOT / "figures"
FIG.mkdir(exist_ok=True)
BLUE, ORANGE, AQUA, GRAY, INK, MUTED, GRID, SURF = "#2a78d6", "#eb6834", "#1baf7a", "#a3a29c", "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF, "axes.facecolor": SURF,
    "legend.frameon": False, "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
})
NAMES = {"v2_ensemble": "V2", "v3": "V3", "v3_hedge": "V3-hedge", "local_sng2": "local_sng2",
         "factor_only6": "panel factor (6m)", "common_weekly2": "weekly common (rejected)"}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=160, facecolor=SURF)
    plt.close(fig)


def final_metrics() -> pd.DataFrame:
    rows = []
    for w in ["exact", "other", "apr_nov", "feb_mar_anomaly"]:
        t = pd.read_csv(OUT / f"metrics_{w}.csv")
        for m in ["seasonal_naive", "seasonal_growth1", "local_sng2", "factor_only6", "v2_ensemble", "v3", "v3_hedge"]:
            r = t[t["model"] == m].iloc[0]
            rows.append({"window": w, "h": "1-3", "model": m, **r.drop("model").to_dict()})
    byh = pd.read_csv(OUT / "metrics_exact_by_h.csv")
    for _, r in byh[byh["model"].isin(["v2_ensemble", "v3", "v3_hedge"])].iterrows():
        rows.append({"window": "exact", **r.to_dict(), "h": str(int(r["h"]))})
    # long horizons: limited evidence (origins with T+h <= 2024-12)
    panel, nat, wk = load_panel(), national_monthly(), weekly_monthly()
    ms = {"v2_ensemble": __import__("sbx.models", fromlist=["x"]).v2_ensemble, **final_models()}
    df = B.run(panel, ms, pd.date_range("2024-02-01", "2024-08-01", freq="MS"), horizons=[4, 5, 6],
               ctx_fn=lambda o: {"origin": o, "national": nat[nat.index <= o], "weekly": wk[wk.index <= o]})
    for h, g in df.groupby("h"):
        for m in ms:
            rows.append({"window": "long_h", "h": str(h), "model": m, **B.metrics(g, m)})
    s = json.loads((OUT / "rolling_summary.json").read_text())
    if s.get("prophet_sample"):
        for m, v in s["prophet_sample"].items():
            if isinstance(v, dict):
                rows.append({"window": "exact_sample400", "h": "1-3", "model": m, **v,
                             "source": "prophet columns from cached outputs/prophet_predictions.csv.gz (REFIT_PROPHET=1 refits); other rows recomputed"})
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "FINAL_METRICS.csv", index=False, float_format="%.4f")
    return out


def ablation() -> pd.DataFrame:
    a = pd.read_csv(OUT / "ablation.csv")
    a.insert(0, "block", "forecast")
    rep = pd.read_csv(OUT / "category_replication_tests.csv")
    rep = rep[rep["window"] == "apr_nov"]
    cats = rep[rep["category"] != "Все категории"]
    for m, g in cats.groupby("model"):
        idx = a.index[a["model"] == m]
        if len(idx):
            a.loc[idx, "categories_better_than_V2_apr_nov"] = f"{int((g['dMAE'] < 0).sum())}/5"
            a.loc[idx, "median_dMAE_pct_categories_apr_nov"] = g["dMAE_pct"].median()
    ew = pd.read_csv(OUT / "ew_v2_metrics.csv")
    ew_rows = []
    for _, r in ew.iterrows():
        ew_rows.append({"block": f"early_warning_{r['task']}", "model": r["method"],
                        "description": {"sup_base": "LightGBM on total-series detectors + state",
                                        "sup_base_category": "+ municipal category features",
                                        "sup_base_national": "+ national monthly signals",
                                        "sup_base_weekly": "+ weekly national category signals",
                                        "sup_all": "all feature groups",
                                        "sup_category_only": "category features only"}.get(r["method"], "unsupervised detector"),
                        "PR_AUC": r["PR_AUC"], "lift_vs_base_rate": r["lift_vs_base_rate"], "F1": r["F1"],
                        "precision": r["precision"], "recall": r["recall"],
                        "false_alarms_per_100": r["false_alarms_per_100"],
                        "detected_share": r.get("detected_share"), "median_delay_months": r.get("median_delay_months")})
    out = pd.concat([a, pd.DataFrame(ew_rows)], ignore_index=True)
    out.to_csv(ROOT / "ABLATION.csv", index=False, float_format="%.4f")
    return out


def fig_anomaly():
    p = load_panel()
    f = pd.Series(np.median(p.logs, axis=0), index=p.periods)
    muni = 100 * (np.exp(f - f.shift(12)) - 1)
    nat = national_monthly()["nat_yoy_nominal"]
    x = p.periods[12:]
    lvl = f - np.log(national_monthly()["nat_spend"].reindex(p.periods))
    pd.DataFrame({"muni_median_yoy_pct": muni.loc[x].to_numpy(), "national_yoy_nominal_pct": nat.reindex(x).to_numpy()},
                 index=pd.Index(x.strftime("%Y-%m"), name="month")).to_csv(OUT / "common_factor_vs_national.csv", float_format="%.2f")
    pd.DataFrame({"log_muni_median_minus_log_national_demeaned_pct": 100 * (lvl - lvl.iloc[12:].mean()).to_numpy()},
                 index=pd.Index(p.periods.strftime("%Y-%m"), name="month")).to_csv(OUT / "common_level_gap.csv", float_format="%.2f")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(x, muni.loc[x], color=BLUE, lw=2, marker="o", ms=5)
    ax.plot(x, nat.reindex(x), color=ORANGE, lw=2, marker="o", ms=5)
    ax.text(x[-1], muni.iloc[-1] + 0.6, " медиана МО", color=INK, va="bottom", ha="right")
    ax.text(x[-1], nat.reindex(x).iloc[-1] - 0.6, " Россия (СберИндекс)", color=INK, va="top", ha="right")
    ax.axvspan(x[0] - pd.Timedelta(days=15), x[2] + pd.Timedelta(days=15), color=GRID, alpha=0.6, lw=0)
    ax.text(x[1], 3.5, "аномалия I кв.", ha="center", color=MUTED)
    ax.set_ylabel("рост г/г, %"); ax.set_ylim(0, 22)
    ax.set_title("Муниципальный общий фактор против национального ряда, 2024")
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%m.%y"))
    save(fig, "01_q1_anomaly.png")


def fig_mae_by_origin():
    m = pd.read_csv(OUT / "mae_by_origin.csv", index_col=0)
    x = pd.to_datetime(m.columns)
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    for k, c in [("local_sng2", GRAY), ("factor_only6", GRAY)]:
        ax.plot(x, m.loc[k], color=c, lw=1.2)
        ax.text(x[-1], m.loc[k].iloc[-1], f"  {NAMES[k]}", color=MUTED, va="center", fontsize=9)
    for k, c in [("v2_ensemble", ORANGE), ("v3", BLUE), ("v3_hedge", AQUA)]:
        ax.plot(x, m.loc[k], color=c, lw=2, marker="o", ms=5, label=NAMES[k])
    ax.axvspan(pd.Timestamp("2024-05-16"), pd.Timestamp("2024-09-15"), color=GRID, alpha=0.6, lw=0)
    ax.text(pd.Timestamp("2024-07-15"), 560, "exact-окно", ha="center", color=MUTED)
    ax.annotate("V2 почти совпадает с V3", (pd.Timestamp("2024-03-01"), 1874), xytext=(12, 8), textcoords="offset points", color=MUTED, fontsize=9)
    ax.set_ylabel("MAE, ₽"); ax.set_ylim(500, 2600)
    ax.legend(loc="upper center", ncol=3)
    ax.set_title("MAE по forecast origin (h = 1–3)")
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%m.%y"))
    save(fig, "02_mae_by_origin.png")


def fig_ablation():
    a = pd.read_csv(OUT / "ablation.csv").set_index("model")
    order = ["seasonal_growth1", "local_sng2", "factor_only6", "lowrank_factor4", "residual_lgbm", "pooled_lgbm_direct",
             "horizon_weights", "v2_ensemble", "v3", "v3_hedge"]
    a = a.loc[order]
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.barh(y + 0.2, a["MAE_exact"], height=0.38, color=BLUE, label="exact (06–09)")
    ax.barh(y - 0.2, a["MAE_other"], height=0.38, color=ORANGE, label="другие origins (04, 05, 10, 11)")
    ax.set_yticks(y, [NAMES.get(k, k) for k in order])
    for i, k in enumerate(order):
        if k in ("v2_ensemble", "v3", "v3_hedge"):
            ax.text(a.loc[k, "MAE_exact"] + 10, i + 0.2, f"{a.loc[k, 'MAE_exact']:.0f}", va="center", fontsize=9, color=INK)
            ax.text(a.loc[k, "MAE_other"] + 10, i - 0.2, f"{a.loc[k, 'MAE_other']:.0f}", va="center", fontsize=9, color=INK)
    ax.set_xlabel("MAE, ₽"); ax.set_xlim(500, 1500); ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    ax.set_title("Ablation: exact-окно против дополнительных origins")
    save(fig, "03_ablation.png")


def fig_replication():
    r = pd.read_csv(OUT / "category_replication_tests.csv")
    r = r[(r["window"] == "apr_nov") & (r["category"] != "Все категории")]
    cats = ["Здоровье", "Маркетплейсы", "Общественное питание", "Продовольствие", "Транспорт"]
    y = np.arange(len(cats))
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for off, (m, c) in zip([0.27, 0, -0.27], [("v3", BLUE), ("v3_hedge", AQUA), ("common_weekly2", GRAY)]):
        v = r[r["model"] == m].set_index("category").loc[cats, "dMAE_pct"]
        ax.barh(y + off, v, height=0.25, color=c, label=NAMES[m])
        for i, val in enumerate(v):
            ax.text(val + (0.6 if val >= 0 else -0.6), i + off, f"{val:+.1f}%", va="center",
                    ha="left" if val >= 0 else "right", fontsize=8, color=INK)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(y, cats); ax.grid(axis="y", visible=False)
    ax.set_xlabel("изменение MAE относительно V2, % (апр–ноя 2024)"); ax.set_xlim(-40, 35)
    ax.legend(loc="lower left")
    ax.set_title("Репликация на 5 нетронутых категориях")
    save(fig, "04_category_replication.png")


def fig_labels():
    sys.path.insert(0, str(ROOT / "src"))
    import pipeline as P

    _, panel_legacy, _, _ = P.load()
    legacy = P.change_points(panel_legacy)
    leg = legacy.groupby("period")["is_change"].mean() * 100
    p = load_panel()
    new = pd.Series(EW.events(p.logs).mean(axis=0) * 100, index=p.periods)
    x = p.periods[3:22]
    pd.DataFrame({"legacy_raw_level_pct": leg.reindex(x).to_numpy(), "new_panel_relative_pct": new.reindex(x).to_numpy(),
                  "legacy_n_series": panel_legacy["mo"].nunique(), "new_n_series": p.values.shape[0]},
                 index=pd.Index(x.strftime("%Y-%m"), name="month")).to_csv(OUT / "shift_label_rates.csv", float_format="%.3f")
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.plot(x, leg.reindex(x), color=ORANGE, lw=2, marker="o", ms=5, label="старая разметка (сырые уровни)")
    ax.plot(x, new.reindex(x), color=BLUE, lw=2, marker="o", ms=5, label="новая разметка (относительно панели, без сезонного эха)")
    ax.annotate("ноябрь: декабрьский пик", (pd.Timestamp("2023-11-01"), leg.loc["2023-11-01"]), xytext=(-10, 4),
                textcoords="offset points", color=MUTED, va="center", ha="right", fontsize=9)
    ax.annotate("апрель 2023: отзвук аномалии I кв.", (pd.Timestamp("2023-04-01"), new.loc["2023-04-01"]), xytext=(12, 0),
                textcoords="offset points", color=MUTED, va="center", fontsize=9)
    ax.set_ylabel("доля рядов со «сдвигом», %"); ax.set_ylim(0, 30)
    ax.legend(loc="upper right", bbox_to_anchor=(1, 0.72))
    ax.set_title("Разметка структурных сдвигов: старая кодировала сезонность")
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%m.%y"))
    save(fig, "05_shift_labels.png")


def fig_ew():
    m = pd.read_csv(OUT / "ew_v2_metrics.csv")
    order = ["page_hinkley", "cusum", "cat_jump_max", "pelt", "bocpd", "jump_raw", "sup_category_only",
             "sup_all", "sup_base_category", "sup_base_national", "sup_base_weekly", "sup_base"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    for ax, task, title in [(axes[0], "detect", "Детекция (сдвиг 0–2 мес. назад)"), (axes[1], "predict", "Упреждение (сдвиг через 1–3 мес.)")]:
        g = m[m["task"] == task].set_index("method").loc[order]
        colors = [BLUE if k == "sup_base" else GRAY for k in order]
        ax.barh(np.arange(len(order)), g["lift_vs_base_rate"], color=colors, height=0.6)
        span = g["lift_vs_base_rate"].max()
        for i, v in enumerate(g["lift_vs_base_rate"]):
            ax.text(v + 0.015 * span, i, f"{v:.1f}×", va="center", fontsize=8, color=INK)
        ax.set_xlim(0, span * 1.15)
        ax.axvline(1, color=MUTED, lw=1)
        ax.set_title(title, fontsize=11); ax.grid(axis="y", visible=False)
        ax.set_xlabel("PR-AUC / базовая частота")
    axes[0].set_yticks(np.arange(len(order)), order)
    fig.suptitle("Early warning: rolling-оценка, 2024", x=0.01, ha="left", fontweight="bold", fontsize=13)
    save(fig, "06_early_warning.png")


def fig_intervals():
    from sbx.intervals import conformal

    p = load_panel()
    df = pd.read_csv(OUT / "rolling_predictions.csv.gz", parse_dates=["origin", "target_date"])
    c = conformal(df[["series", "origin", "h", "target_date", "y", "v3"]], "v3", p.logs, p.periods)
    c = c[c["origin"] >= "2024-04-01"].dropna(subset=["v3_lo80"])
    cov = c.assign(c80=(c.y >= c.v3_lo80) & (c.y <= c.v3_hi80), c90=(c.y >= c.v3_lo90) & (c.y <= c.v3_hi90))
    g = cov.groupby("target_date")[["c80", "c90"]].mean() * 100
    g.rename(columns={"c80": "coverage80_pct", "c90": "coverage90_pct"}).assign(
        n_pairs=cov.groupby("target_date").size()).to_csv(OUT / "interval_coverage_by_month.csv", float_format="%.2f")
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.plot(g.index, g["c80"], color=BLUE, lw=2, marker="o", ms=5, label="интервал 80%")
    ax.plot(g.index, g["c90"], color=ORANGE, lw=2, marker="o", ms=5, label="интервал 90%")
    ax.axhline(80, color=BLUE, lw=1, alpha=0.5); ax.axhline(90, color=ORANGE, lw=1, alpha=0.5)
    ax.set_ylim(60, 101); ax.set_ylabel("фактическое покрытие, %"); ax.legend(loc="lower right")
    ax.set_title("V3: покрытие conformal-интервалов по целевым месяцам")
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%m.%y"))
    save(fig, "07_interval_coverage.png")


def presentation_sources():
    """Write every number quoted in the presentation that is not already in another output."""
    from sbx import models as M
    from sbx.data import read_raw_municipal

    # data audit
    raw = read_raw_municipal()
    audit = {"raw_rows": int(len(raw)), "unique_names": int(raw["mo"].nunique()), "periods": [str(raw["period"].min().date()), str(raw["period"].max().date())]}
    tot = load_panel()
    audit["complete_series_total"] = int(tot.values.shape[0])
    audit["homonym_series"] = int(tot.meta["homonym"].sum())
    audit["homonym_names"] = int(tot.meta.loc[tot.meta["homonym"], "mo"].nunique())
    names_dup = raw[raw["category_15"].eq("Все категории")].groupby("run_id")["mo"].first()
    audit["names_with_several_municipalities"] = int((names_dup.value_counts() > 1).sum())
    t = raw[raw["category_15"].eq("Все категории")].groupby("run_id").size()
    audit["incomplete_series_total"] = int((t != 24).sum())
    legacy = {}
    for c in raw["category_15"].unique():
        a = raw[raw["category_15"] == c].reset_index()
        a["blk"] = (a["mo"] != a["mo"].shift()).cumsum()
        g = a.groupby("blk").agg(runs=("run_id", "nunique"), n=("period", "size"), s=("period", "min"), e=("period", "max"))
        legacy[c] = {"legacy_merged_blocks": int((g["runs"] > 1).sum()),
                     "legacy_complete_series": int(((g["n"] == 24) & (g["s"] == "2023-01-01") & (g["e"] == "2024-12-01")).sum()),
                     "robust_complete_series": int(load_panel(c).values.shape[0])}
    audit["reconstruction_by_category"] = legacy
    from sbx.ew_features import category_matrices
    audit["category_linkable_series"] = int((~np.isnan(category_matrices(tot)["Здоровье"][:, 0])).sum())
    (OUT / "data_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    # V2 hyper-parameter grid: where does V2 rank on exact vs other origins?
    nat = national_monthly()
    ms = {**{f"L{k}": M.seasonal_growth(k) for k in (1, 2, 3, 4, 6)}, **{f"F{k}": M.panel_factor(k) for k in (1, 2, 3, 4, 6, 9, 12)}}
    df = B.run(tot, ms, pd.date_range("2024-04-01", "2024-11-01", freq="MS"), ctx_fn=lambda o: {"origin": o})
    ex = df["origin"].isin(B.EXACT_ORIGINS)
    rows = []
    for kl in (1, 2, 3, 4, 6):
        for kf in (1, 2, 3, 4, 6, 9, 12):
            for w in (0.3, 0.4, 0.5, 0.6, 0.7):
                ae = (w * df[f"L{kl}"] + (1 - w) * df[f"F{kf}"] - df["y"]).abs()
                rows.append({"local_window": kl, "factor_window": kf, "w_local": w, "MAE_exact": ae[ex].mean(), "MAE_other": ae[~ex].mean()})
    grid = pd.DataFrame(rows)
    v2 = grid[(grid.local_window == 2) & (grid.factor_window == 6) & (grid.w_local == 0.5)].iloc[0]
    grid["better_than_V2_exact"] = grid["MAE_exact"] < v2["MAE_exact"] - 1e-9
    grid["better_than_V2_other"] = grid["MAE_other"] < v2["MAE_other"] - 1e-9
    grid.to_csv(OUT / "v2_grid.csv", index=False, float_format="%.3f")
    best = grid.loc[grid["MAE_exact"].idxmin()]
    grid_summary = {"n_configs": int(len(grid)), "configs_better_than_V2_exact": int(grid["better_than_V2_exact"].sum()),
                    "configs_better_than_V2_other": int(grid["better_than_V2_other"].sum()),
                    "best_exact_config": best[["local_window", "factor_window", "w_local"]].to_dict(),
                    "best_exact_MAE_exact": float(best["MAE_exact"]), "best_exact_MAE_other": float(best["MAE_other"])}

    # V2 idiosyncratic h=1 error autocorrelation between consecutive origins (motivation of V3)
    d1 = B.run(tot, {"v2": M.v2_ensemble}, pd.date_range("2024-02-01", "2024-11-01", freq="MS"), horizons=[1])
    d1["e"] = np.log(d1["y"] / d1["v2"])
    d1["e"] -= d1.groupby("origin")["e"].transform("median")
    E = d1.pivot(index="series", columns="origin", values="e")
    ac = [{"origin_a": str(E.columns[j].date()), "origin_b": str(E.columns[j + 1].date()),
           "corr": float(np.corrcoef(E.iloc[:, j], E.iloc[:, j + 1])[0, 1])} for j in range(E.shape[1] - 1)]
    pd.DataFrame(ac).to_csv(OUT / "v2_error_autocorr.csv", index=False, float_format="%.4f")

    # 2025 forecasts: how different are the two final models?
    f = pd.read_csv(OUT / "final_forecasts_2025.csv.gz")
    gap = (f["v3_hedge"] / f["v3"] - 1).abs().groupby(f["h"]).median() * 100
    s = json.loads((OUT / "rolling_summary.json").read_text())
    summary = {"v2_grid": grid_summary,
               "v2_error_autocorr_range": [min(r["corr"] for r in ac), max(r["corr"] for r in ac)],
               "v2_error_autocorr_negative_pairs": f"{sum(r['corr'] < 0 for r in ac)}/{len(ac)}",
               "forecast_2025_median_abs_gap_v3_vs_hedge_pct_by_h": gap.round(3).to_dict(),
               "v3_vs_v2_tests": s["bootstrap_vs_v2"]["exact"]["v3"],
               "prophet_sample_source": "outputs/prophet_predictions.csv.gz (cached; refit with REFIT_PROPHET=1)"}
    (OUT / "presentation_sources.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print("presentation sources", json.dumps(summary, ensure_ascii=False, default=float)[:600])


def main():
    if "--figures-only" in sys.argv:
        for f in [fig_anomaly, fig_mae_by_origin, fig_ablation, fig_replication, fig_labels, fig_ew, fig_intervals]:
            f()
        return
    fm = final_metrics()
    ab = ablation()
    presentation_sources()
    for f in [fig_anomaly, fig_mae_by_origin, fig_ablation, fig_replication, fig_labels, fig_ew, fig_intervals]:
        f()
        print("figure", f.__name__, flush=True)
    print(fm[fm["window"].isin(["exact", "long_h"])].round(3).to_string(index=False))
    print(len(ab), "ablation rows")


if __name__ == "__main__":
    main()
