"""Perturbation test for temporal leakage.

For every model and several origins, all data after the origin (municipal panel AND national series)
is replaced by garbage. Forecasts must stay bit-identical. Also checks the series reconstruction.

Run: python tests/test_time_safety.py   (or pytest)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from sbx import backtest as B  # noqa: E402
from sbx import early_warning as EW  # noqa: E402
from sbx.data import Panel, load_panel, national_monthly, weekly_monthly  # noqa: E402

ORIGINS = pd.to_datetime(["2024-03-01", "2024-06-01", "2024-09-01"])


def _models():
    from rolling_eval import build_models

    return build_models(fast=True)


def test_series_reconstruction():
    p = load_panel()  # asserts 2016 series, no merged homonym blocks, no duplicate series
    assert p.values.shape == (2016, 24)
    assert p.meta["homonym"].sum() == 107 and p.meta.loc[p.meta["homonym"], "mo"].nunique() == 43


def test_forecasts_ignore_future():
    panel = load_panel()
    nat, wk = national_monthly(), weekly_monthly().astype(float)
    models = _models()
    rng = np.random.default_rng(0)
    for origin in ORIGINS:
        T = list(panel.periods).index(origin)
        bad_vals = panel.values.copy()
        bad_vals[:, T + 1 :] = rng.uniform(1, 1e6, size=bad_vals[:, T + 1 :].shape)
        bad_panel = Panel(values=bad_vals, periods=panel.periods, meta=panel.meta)
        bad_nat, bad_wk = nat.copy(), wk.copy()
        bad_nat.loc[bad_nat.index > origin] = rng.uniform(-1e3, 1e3, size=bad_nat.loc[bad_nat.index > origin].shape)
        bad_wk.loc[bad_wk.index > origin] = rng.uniform(-1e3, 1e3, size=bad_wk.loc[bad_wk.index > origin].shape)

        good = B.run(panel, models, [origin],
                     ctx_fn=lambda o: {"origin": o, "national": nat[nat.index <= o], "weekly": wk[wk.index <= o]})
        # exogenous frames deliberately NOT truncated: every reader must filter by ctx['origin'] itself
        bad = B.run(bad_panel, models, [origin], ctx_fn=lambda o: {"origin": o, "national": bad_nat, "weekly": bad_wk})
        for name in models:
            np.testing.assert_array_equal(good[name].to_numpy(), bad[name].to_numpy(), err_msg=f"{name} @ {origin}")


def test_national_reader_respects_origin():
    """g_national filters by ctx['origin'] itself, so even an untruncated frame cannot leak."""
    from sbx import models as M

    nat = national_monthly()
    origin = pd.Timestamp("2024-06-01")
    bad = nat.copy()
    bad.loc[bad.index > origin] = 1e9
    f = np.zeros(18)
    g = M.g_national(2)
    assert g(f, 17, {"origin": origin, "national": nat}) == g(f, 17, {"origin": origin, "national": bad})


def test_detectors_are_online():
    L = load_panel().logs
    t = 17
    bad = L.copy()
    bad[:, t + 1 :] = 0.0
    for fn in [EW.score_jump, EW.score_cusum, EW.score_page_hinkley, EW.score_bocpd]:
        a, b = fn(L)[:, : t + 1], fn(bad)[:, : t + 1]
        np.testing.assert_array_equal(np.nan_to_num(a), np.nan_to_num(b), err_msg=fn.__name__)


def test_final_models_ignore_future():
    from sbx.final import final_models

    panel = load_panel()
    nat = national_monthly()
    origin = pd.Timestamp("2024-07-01")
    T = list(panel.periods).index(origin)
    bad_vals = panel.values.copy()
    bad_vals[:, T + 1 :] = 1.0
    bad_panel = Panel(values=bad_vals, periods=panel.periods, meta=panel.meta)
    ms = final_models()
    ctx = lambda o: {"origin": o, "national": nat}  # noqa: E731  (untruncated on purpose)
    a = B.run(panel, ms, [origin], ctx_fn=ctx)
    b = B.run(bad_panel, ms, [origin], ctx_fn=ctx)
    for name in ms:
        np.testing.assert_array_equal(a[name].to_numpy(), b[name].to_numpy(), err_msg=name)


def test_category_features_online():
    from sbx import ew_features as F

    panel = load_panel()
    L = panel.logs
    C = F.category_matrices(panel)
    t = 15
    Lb = L.copy(); Lb[:, t + 1 :] = 0.0
    Cb = {k: v.copy() for k, v in C.items()}
    for v in Cb.values():
        v[:, t + 1 :] = 5.0
    with np.errstate(all="ignore"):
        import warnings

        warnings.simplefilter("ignore")
        fa, fb = F.category_features(L, C), F.category_features(Lb, Cb)
    for k in fa:
        np.testing.assert_array_equal(np.nan_to_num(fa[k][:, : t + 1]), np.nan_to_num(fb[k][:, : t + 1]), err_msg=k)


def test_intervals_use_only_observed_errors():
    """Changing outcomes of pairs whose target is after the origin must not change its intervals."""
    from sbx.intervals import conformal

    panel = load_panel()
    nat = national_monthly()
    ms = {"v2": _models()["v2_ensemble"]}
    df = B.run(panel, ms, pd.date_range("2024-02-01", "2024-09-01", freq="MS"),
               ctx_fn=lambda o: {"origin": o, "national": nat[nat.index <= o]})
    origin = pd.Timestamp("2024-07-01")
    bad = df.copy()
    bad.loc[bad["target_date"] > origin, "y"] *= 3.0
    ca = conformal(df, "v2", panel.logs, panel.periods)
    cb = conformal(bad, "v2", panel.logs, panel.periods)
    m = ca["origin"] == origin
    for c in ["v2_lo80", "v2_hi80", "v2_lo90", "v2_hi90"]:
        np.testing.assert_allclose(ca.loc[m, c].to_numpy(), cb.loc[m, c].to_numpy(), err_msg=c)


if __name__ == "__main__":
    for fn in [test_series_reconstruction, test_national_reader_respects_origin, test_detectors_are_online,
               test_category_features_online, test_intervals_use_only_observed_errors,
               test_final_models_ignore_future, test_forecasts_ignore_future]:
        fn()
        print("ok", fn.__name__)
