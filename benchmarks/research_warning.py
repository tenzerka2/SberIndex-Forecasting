"""Recompute early warning with prefix-stable labels. No legacy metric is overwritten."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sbx.data import ROOT, load_panel
from sbx import early_warning as EW
from sbx import research_warning as W
from sbx.ew_features import category_matrices, category_features
from sbx.news_local import tensor as local_news

OUT = ROOT / "outputs/research"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    panel = load_panel(); L = panel.logs; n, T = L.shape
    base = W.features(L)
    with np.errstate(all="ignore"):
        cat = category_features(L, category_matrices(panel))
    cat = {k: np.log1p(np.abs(np.nan_to_num(v))) for k, v in cat.items()}
    news, links, news_audit = local_news(panel)
    links.to_csv(OUT / "local_news_links.csv", index=False)
    (OUT / "local_news_audit.json").write_text(json.dumps(news_audit, ensure_ascii=False, indent=2))
    news = {k: np.log1p(v) for k,v in news.items()}
    f = {**base, **cat, **news}
    variants = {"logistic_base": list(base), "logistic_categories": list(base)+list(cat),
                "logistic_local_news":list(base)+list(news)}
    metrics, pairs = [], []
    for task, ts in [("detect", range(12, 22)), ("predict", range(14, 19))]:
        full = EW.mature_labels(L, task)
        for t in ts:
            available = EW.mature_labels(L[:, :t + 1], task)
            train = [s for s in range(6, t) if (available[:, s] >= 0).all()]
            yt = available[:, train].T.ravel()
            y = full[:, t]
            assert (y >= 0).all() and (yt >= 0).all()
            scores = {k: base[k][:, t] for k in ["jump", "seasonal_jump", "bocpd", "cusum", "page_hinkley", "slope"]}
            for name, names in variants.items():
                scores[name] = W.logistic_predict(W.stack(f, names, train), yt, W.stack(f, names, [t]))
            for name, s in scores.items():
                frame = pd.DataFrame({"task":task,"month":t,"method":name,"series":np.arange(n),"y":y,"score":s})
                for budget in [.01, .02, .05]:
                    frame[f"alarm_{int(budget*100)}"] = W.top_budget(s, budget)
                pairs.append(frame)
            print(task, str(panel.periods[t].date()), "positives", int(y.sum()), flush=True)
    pairs = pd.concat(pairs, ignore_index=True)
    for (task, name), g in pairs.groupby(["task", "method"]):
        y = g.y.to_numpy(bool); s = g.score.to_numpy()
        for budget in [1, 2, 5]:
            a = g[f"alarm_{budget}"].to_numpy(bool)
            tp, fp = int((a & y).sum()), int((a & ~y).sum())
            ap = W.average_precision(y, s)
            row = {"task":task,"method":name,"budget_pct":budget,"AP":ap,"base_rate":y.mean(),
                   "AP_lift":ap / y.mean(), "precision":tp/max(a.sum(),1),"recall":tp/max(y.sum(),1),
                   "false_alarms_per_100":100*fp/len(y),"true_positives":tp,"n_rows":len(y)}
            if task == "predict":
                ev = EW.events(L); hits = 0; total = 0; leads = []
                alarm = np.zeros_like(ev)
                for t, m in g.groupby("month"):
                    alarm[m.series.to_numpy(), t] = m[f"alarm_{budget}"].to_numpy()
                lo, hi = g.month.min(), g.month.max()
                for i, tau in zip(*np.where(ev)):
                    if tau - 3 >= lo and tau - 1 <= hi:
                        total += 1
                        hit = np.flatnonzero(alarm[i, tau - 3:tau])
                        if len(hit):
                            hits += 1; leads.append(int(3 - hit[0]))
                row.update(events_with_full_warning_window=total, events_warned=hits,
                           event_recall=hits/max(total,1), median_lead_months=float(np.median(leads)) if leads else None)
            metrics.append(row)
    result = pd.DataFrame(metrics)
    result.to_csv(OUT / "warning_metrics.csv", index=False)
    pairs.to_csv(OUT / "warning_pairs.csv.gz", index=False, compression={"method":"gzip","mtime":0})
    # Directly quantify label revisions in the old protocol versus the corrected protocol.
    drift = []
    for t in range(12, 22):
        a = EW.legacy_events(L[:, :t + 1]); b = EW.legacy_events(L)[:, :t + 1]
        x = EW.events(L[:, :t + 1]); z = EW.events(L)[:, :t + 1]
        drift.append({"origin":str(panel.periods[t].date()),
                      "legacy_mature_label_changes":int((a[:, :t - 1] != b[:, :t - 1]).sum()),
                      "stable_mature_label_changes":int((x[:, :t - 1] != z[:, :t - 1]).sum())})
    pd.DataFrame(drift).to_csv(OUT / "label_revision_audit.csv", index=False)
    print(result[result.budget_pct.eq(2)].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
