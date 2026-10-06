"""Temporal-safety checks for the frozen global-news pipeline."""
from __future__ import annotations
import json, sys, tempfile
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from sbx import news_global as N  # noqa: E402
from sbx.data import load_panel  # noqa: E402

def test_verified_date_uses_latest_candidate():
    r={"published":"2024-01-01T00:00:00Z","url":"https://example.ru/2024/03/15/story"}
    assert N.verified_date(r)==pd.Timestamp("2024-03-15")

def test_bad_dates_are_excluded():
    d,a=N.read_verified()
    assert a["records"]>=550
    assert a["month_mismatches"]>=1
    assert a["missing_dates"]>=1
    assert (d["verified_date"].dt.to_period("M").dt.to_timestamp()==d["query_month"]).all()

def test_future_article_cannot_change_past_features():
    before,_=N.monthly()
    raw=N.CORPUS.read_text(encoding="utf-8")
    fake={"query_id":"2025-01_security","rank":1,"url":"https://example.ru/2025/01/20/x",
          "published":"2025-01-20T00:00:00Z","title":"future event","snippet":"future"}
    with tempfile.TemporaryDirectory() as td:
        p=Path(td)/"corpus.jsonl"
        p.write_text(raw.rstrip()+"\n"+json.dumps(fake,ensure_ascii=False)+"\n",encoding="utf-8")
        old=N.CORPUS; N.CORPUS=p
        try: after,_=N.monthly()
        finally: N.CORPUS=old
    common=before.index.intersection(after.index)
    pd.testing.assert_frame_equal(before.loc[common],after.loc[common])

def test_tensor_has_no_forward_rolling_window():
    p=load_panel(); t=N.tensor(p.periods,len(p.meta))
    # sum3 at month t may depend only on t,t-1,t-2. Check one signal against monthly source.
    m,_=N.monthly()
    src=m["intensity"].reindex(p.periods).fillna(0)
    want=src.rolling(3,min_periods=1).sum().to_numpy()
    np.testing.assert_allclose(t["news_intensity_sum3"][0],want)

if __name__=="__main__":
    for fn in [test_verified_date_uses_latest_candidate,test_bad_dates_are_excluded,
               test_future_article_cannot_change_past_features,test_tensor_has_no_forward_rolling_window]:
        fn(); print("ok",fn.__name__)
