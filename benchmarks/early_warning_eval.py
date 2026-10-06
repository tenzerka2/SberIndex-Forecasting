"""Time-safe comparison of structural-change detectors and a supervised early-warning classifier.

Split (with embargo):
  calibration rows  t in 2023-09..2024-03; labels recomputed from data <= 2024-05 only
                    (an event at tau needs months up to tau+2, so only tau <= 2024-03 is known);
  test rows         t in 2024-06..2024-10; labels from the full data.
Thresholds of every method and the classifier itself are fitted on calibration rows only.
Every detector score at (i, t) uses months <= t (see sbx.early_warning).

Two tasks:
  detect   row positive if an event started in [t-2, t]   (online detection, delay <= 2)
  predict  row positive if an event starts in [t+1, t+3]  (genuine early warning)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sbx import early_warning as EW  # noqa: E402
from sbx.data import ROOT, load_panel, month_index  # noqa: E402

OUT = ROOT / "outputs"
CAL_ROWS = range(month_index(pd.Timestamp("2023-09-01")), month_index(pd.Timestamp("2024-03-01")) + 1)
CAL_CUTOFF = month_index(pd.Timestamp("2024-05-01"))
TEST_ROWS = range(month_index(pd.Timestamp("2024-06-01")), month_index(pd.Timestamp("2024-10-01")) + 1)
SEED = 42


def labels(ev: np.ndarray, task: str) -> np.ndarray:
    n, T = ev.shape
    y = np.zeros((n, T), dtype=bool)
    for t in range(T):
        if task == "detect":
            y[:, t] = ev[:, max(0, t - 2) : t + 1].any(axis=1)
        else:
            y[:, t] = ev[:, t + 1 : t + 4].any(axis=1) if t + 1 < T else False
    return y


def rows(M: np.ndarray, ts) -> np.ndarray:
    return M[:, list(ts)].T.ravel()  # month-major


def main():
    t0 = time.time()
    panel = load_panel()
    L = panel.logs
    ev_full = EW.events(L)
    ev_cal = np.zeros_like(ev_full)
    ev_cal[:, : CAL_CUTOFF + 1] = EW.events(L[:, : CAL_CUTOFF + 1])

    scores = {
        "jump_raw": EW.score_jump(L, seasonal=False),
        "jump_seasonal": EW.score_jump(L, seasonal=True),
        "cusum": EW.score_cusum(L),
        "page_hinkley": EW.score_page_hinkley(L),
        "bocpd": EW.score_bocpd(L),
        "pelt": EW.score_pelt(L),
    }
    print("scores", round(time.time() - t0, 1), "s", flush=True)
    feats = EW.feature_tensor(L, scores)
    names = list(feats)

    results = []
    for task in ["detect", "predict"]:
        y_cal_m, y_test_m = labels(ev_cal, task), labels(ev_full, task)
        cal_rows = [t for t in CAL_ROWS if (t + 2 <= CAL_CUTOFF if task == "detect" else t + 5 <= CAL_CUTOFF)]
        test_rows = [t for t in TEST_ROWS if (t + 3 + 2 <= L.shape[1] - 1 if task == "predict" else True)]
        yc, yt = rows(y_cal_m, cal_rows), rows(y_test_m, test_rows)
        for name, S in scores.items():
            sc, st = np.nan_to_num(rows(S, cal_rows)), np.nan_to_num(rows(S, test_rows))
            thr = EW.best_threshold(yc, sc)
            results.append({"task": task, "method": name, **EW.pr_metrics(yt, st, thr)})

        # supervised: LightGBM on all detector scores + state features, fitted on calibration rows.
        from lightgbm import LGBMClassifier

        Xc = np.column_stack([rows(feats[k], cal_rows) for k in names])
        Xt = np.column_stack([rows(feats[k], test_rows) for k in names])
        month_c = np.repeat(cal_rows, L.shape[0])

        def clf():
            return LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=100,
                                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0,
                                  class_weight="balanced", random_state=SEED, verbosity=-1, n_jobs=2)

        # threshold via an inner time split of the calibration period, then refit on all of it
        split = cal_rows[len(cal_rows) * 2 // 3]
        inner = clf().fit(Xc[month_c < split], yc[month_c < split])
        thr = EW.best_threshold(yc[month_c >= split], inner.predict_proba(Xc[month_c >= split])[:, 1])
        model = clf().fit(Xc, yc)
        st = model.predict_proba(Xt)[:, 1]
        results.append({"task": task, "method": "supervised_lgbm", **EW.pr_metrics(yt, st, thr)})
        imp = sorted(zip(names, model.feature_importances_), key=lambda x: -x[1])[:8]
        print(task, "top features", imp, flush=True)

    res = pd.DataFrame(results)
    res.to_csv(OUT / "early_warning_metrics.csv", index=False)
    summary = {
        "event_definition": "panel-relative level shift >= 3 robust sigma, persistent 3 months, no 12-month echo",
        "events_total": int(ev_full.sum()),
        "event_rate_test_months_pct": float(100 * ev_full[:, list(TEST_ROWS)].mean()),
        "legacy_label_rate_by_month_note": "legacy raw-level labels: 23% of series in 2023-11, 0% in 2024-05 (seasonality)",
        "results": results,
    }
    (OUT / "early_warning_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.set_option("display.width", 200)
    print(res.round(4).to_string(index=False))
    print("total", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()
