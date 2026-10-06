"""Temporal-safety checks for the historical-news pipeline."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sbx.data import load_panel  # noqa: E402
from sbx import news as N  # noqa: E402


def test_verified_date_uses_latest_candidate():
    r = {"published": "2024-01-01T00:00:00Z", "url": "https://example.ru/2024/03/15/story"}
    d, rule = N.verified_date(r)
    assert d == pd.Timestamp("2024-03-15")
    assert rule == "max(metadata,url)"


def test_future_news_cannot_change_origin_features():
    p = load_panel()
    origin = pd.Timestamp("2024-06-01")
    a = N.feature_tensor(p, cutoff=origin)

    # By construction the cutoff path must equal the unrestricted tensor through origin,
    # regardless of records dated later.
    b = N.feature_tensor(p)
    t = list(p.periods).index(origin)
    for k in a:
        np.testing.assert_allclose(a[k][:, : t + 1], b[k][:, : t + 1], err_msg=k)


def test_query_month_mismatch_excluded():
    d = N.read_corpus()
    usable = d[d["in_query_month"]]
    assert (usable["verified_month"] == usable["query_month"]).all()


if __name__ == "__main__":
    for fn in [test_verified_date_uses_latest_candidate, test_future_news_cannot_change_origin_features,
               test_query_month_mismatch_excluded]:
        fn()
        print("ok", fn.__name__)
