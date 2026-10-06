"""Data loading with explicit series-identity reconstruction and publication-time alignment.

Series identity
---------------
The municipal export has no stable municipality id and 49 municipality names are shared by
several distinct municipalities. The export is ordered as shuffled blocks: every block is one
(municipality, category) series with its months in increasing order. A series is therefore a
maximal run of rows with the same (mo, category_15) whose period strictly increases.

The previous implementation filtered to "Все категории" *before* labelling runs. That only works
as long as two homonym blocks are never separated solely by rows of other categories; otherwise
they would silently merge. Here runs are labelled on the raw file and the result is checked.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
TOTAL = "Все категории"
START = pd.Timestamp("2023-01-01")
END = pd.Timestamp("2024-12-01")
N_MONTHS = 24


def find(pattern: str) -> Path:
    files = sorted(RAW.glob(pattern))
    if not files:
        raise FileNotFoundError(f"Missing data file matching {pattern} in {RAW}")
    return files[0]


def read_raw_municipal() -> pd.DataFrame:
    df = pd.read_csv(find("potrebitelskie-beznalicnye*.csv"), sep=";")
    df["period"] = pd.to_datetime(df["period"])
    key = df["mo"].astype(str) + "|" + df["category_15"].astype(str)
    new_run = (key != key.shift()) | (df["period"] <= df["period"].shift())
    df["run_id"] = new_run.cumsum().astype(int)
    return df


@dataclass
class Panel:
    """Complete 2023-01..2024-12 series for one category, as a (series x month) matrix."""

    values: np.ndarray  # shape (n_series, 24), RUB
    periods: pd.DatetimeIndex
    meta: pd.DataFrame  # index aligned with rows: run_id, mo, homonym flag

    @property
    def logs(self) -> np.ndarray:
        return np.log(self.values)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.values, index=self.meta["run_id"], columns=self.periods)


def load_panel(category: str = TOTAL, check: bool = True) -> Panel:
    df = read_raw_municipal()
    d = df[df["category_15"].eq(category)]
    m = d.groupby("run_id").agg(
        mo=("mo", "first"), n=("period", "size"), start=("period", "min"), end=("period", "max")
    )
    full = m[(m["n"] == N_MONTHS) & (m["start"] == START) & (m["end"] == END)].copy()
    wide = d[d["run_id"].isin(full.index)].pivot(index="run_id", columns="period", values="value")
    wide = wide.loc[full.index].astype(float)
    full["homonym"] = full["mo"].duplicated(keep=False)
    full = full.reset_index()[["run_id", "mo", "homonym"]]

    if check:
        assert len(full) == 2016, f"expected 2016 complete series, got {len(full)}"
        assert not wide.duplicated().any(), "two reconstructed series are identical"
        assert np.isfinite(wide.values).all() and (wide.values > 0).all()
        if category == TOTAL:
            # The exact benchmark was built with the legacy reconstruction (filter first, then
            # contiguous name blocks). For the total category it gives the same row set; for
            # "Здоровье" and "Продовольствие" it would merge one homonym pair each (2014 series).
            a = df[df["category_15"].eq(category)].reset_index()
            a["blk"] = (a["mo"] != a["mo"].shift()).cumsum()
            assert a.groupby("blk")["run_id"].nunique().max() == 1, "legacy reconstruction merges homonyms"

    periods = pd.DatetimeIndex(wide.columns)
    assert len(periods) == N_MONTHS and periods[0] == START
    return Panel(values=wide.to_numpy(), periods=periods, meta=full)


def month_index(ts: pd.Timestamp) -> int:
    return (ts.year - START.year) * 12 + ts.month - START.month


# ---------------------------------------------------------------------------------------------
# National / weekly signals with publication-time alignment
# ---------------------------------------------------------------------------------------------
# Rule: for a forecast made at origin T (municipal data through month T is known), a national
# monthly value for month t may be used only if t <= T. Municipal SberIndex data are published
# later than the national monthly series, so this is conservative. Weekly series are dated by the
# Sunday that ends the week; a week is usable for origin T only if that Sunday is <= last day of T.
# NB: the files are a 2026 vintage, so 2024 values may include revisions (documented risk).


def national_monthly() -> pd.DataFrame:
    spend = pd.read_csv(find("consumer-spending_ru*.csv"), sep=";")
    spend["period"] = pd.to_datetime(spend["period"])
    spend = spend[spend["type"].eq("Всего")].set_index("period")["value"].rename("nat_spend")

    growth = pd.read_csv(find("consumer-spending-growth*.csv"), sep=";")
    growth["period"] = pd.to_datetime(growth["period"])
    growth = growth[growth["type"].eq("Всего")].pivot(index="period", columns="value_type", values="value")
    growth = growth.rename(columns={"Номинальное": "nat_yoy_nominal", "Реальное": "nat_yoy_real"})

    sa = pd.read_csv(find("consumper-spending-index-sa*.csv"), sep=";")
    sa["period"] = pd.to_datetime(sa["period"])
    sa = sa[sa["type"].eq("Всего")].set_index("period")["value"].rename("nat_index_sa")
    return pd.concat([spend, growth, sa], axis=1).sort_index()


def weekly_pulse() -> pd.DataFrame:
    w = pd.read_csv(find("ver-izmenenie-trat-po-kategoriyam*.csv"), sep=";")
    w["period"] = pd.to_datetime(w["period"])
    assert (w["period"].dt.dayofweek == 6).all(), "weekly dates are expected to be week-ending Sundays"
    w["month"] = w["period"].dt.to_period("M").dt.to_timestamp()
    return w.groupby("month")["value"].agg(pulse_mean="mean", pulse_median="median", pulse_std="std")


def weekly_monthly() -> pd.DataFrame:
    """Weekly category YoY (%) aggregated to months: one row per month, columns per category plus
    cross-category summaries. A week belongs to the month of its ending Sunday, so month t only
    contains weeks that ended inside t (no look-ahead into t+1).

    Columns: <category> (mean weekly YoY in the month), last_<category> (last week of the month),
    w_mean / w_median / w_disp (cross-category mean, median, std of monthly means),
    w_last_minus_mean (intra-month momentum: last-week mean minus month mean), n_weeks.
    """
    w = pd.read_csv(find("ver-izmenenie-trat-po-kategoriyam*.csv"), sep=";")
    w["period"] = pd.to_datetime(w["period"])
    assert (w["period"].dt.dayofweek == 6).all()
    w["month"] = w["period"].dt.to_period("M").dt.to_timestamp()
    mean = w.pivot_table(index="month", columns="category", values="value", aggfunc="mean")
    last = w.sort_values("period").groupby(["month", "category"])["value"].last().unstack()
    out = mean.copy()
    out.columns = [f"wk_{c}" for c in mean.columns]
    for c in last.columns:
        out[f"wklast_{c}"] = last[c]
    out["w_mean"] = mean.mean(axis=1)
    out["w_median"] = mean.median(axis=1)
    out["w_disp"] = mean.std(axis=1)
    out["w_last_minus_mean"] = last.mean(axis=1) - mean.mean(axis=1)
    out["n_weeks"] = w.groupby("month")["period"].nunique()
    return out.sort_index()
