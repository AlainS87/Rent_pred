"""Lagged COE premiums. The rental month uses the previous month's tender."""

import numpy as np
import pandas as pd

MONTH_NUMBERS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}
HOUSEHOLD_CATEGORIES = ("a", "b", "e")


def _parse_money(values: pd.Series) -> pd.Series:
    parsed = pd.to_numeric(
        values.astype("string").str.replace(r"[$,]", "", regex=True).str.strip(),
        errors="coerce",
    )
    if parsed.isna().any():
        raise ValueError("COE numeric field failed to parse")
    return parsed


def monthly_coe(coe: pd.DataFrame) -> pd.DataFrame:
    """One row per category and month. Round 2 wins; round 1 is the fallback."""
    frame = coe.copy()
    frame["category"] = frame["CATEGORY"].astype("string").str.strip().str.lower()
    month = frame["MONTH"].astype("string").str.strip().str.lower().map(MONTH_NUMBERS)
    if month.isna().any():
        raise ValueError("COE month name was not recognized")
    frame["period"] = pd.PeriodIndex.from_fields(
        year=pd.to_numeric(frame["YEAR"]).astype(int),
        month=month.astype(int),
        freq="M",
    )
    frame["round"] = pd.to_numeric(frame["ROUND"], errors="coerce")
    frame["premium"] = _parse_money(frame["QUOTA_PREMIUM"])
    frame["bids"] = _parse_money(frame["BIDS_RECEIVED"])
    frame["quota"] = _parse_money(frame["QUOTA"])
    frame = frame.sort_values(["category", "period", "round"])
    monthly = frame.groupby(["category", "period"], as_index=False).tail(1)
    return monthly.set_index(["category", "period"])


def _lookup(monthly: pd.DataFrame, category: str, periods: pd.PeriodIndex, column: str) -> np.ndarray:
    index = pd.MultiIndex.from_arrays(
        [np.repeat(category, len(periods)), periods],
        names=["category", "period"],
    )
    values = monthly.reindex(index)[column]
    return values.to_numpy(dtype=np.float64)


def add_coe_columns(rentals: pd.DataFrame, coe: pd.DataFrame) -> pd.DataFrame:
    """Append lagged COE columns. Missing source months stay null."""
    monthly = monthly_coe(coe)
    approval = pd.PeriodIndex(rentals["RENT_APPROVAL_DATE"].astype("string"), freq="M")
    out = rentals.copy()
    for category in (*HOUSEHOLD_CATEGORIES, "c"):
        premium_lag = _lookup(monthly, category, approval - 1, "premium")
        out[f"coe_{category}_premium_lag1"] = premium_lag
        if category == "c":
            continue
        premium_lag2 = _lookup(monthly, category, approval - 2, "premium")
        change = premium_lag - premium_lag2
        change[np.isnan(premium_lag) | np.isnan(premium_lag2)] = np.nan
        out[f"coe_{category}_premium_mom_change"] = change
        bids = _lookup(monthly, category, approval - 1, "bids")
        quota = _lookup(monthly, category, approval - 1, "quota")
        ratio = np.divide(bids, quota, out=np.full_like(bids, np.nan), where=quota > 0)
        ratio[np.isnan(bids) | np.isnan(quota)] = np.nan
        out[f"coe_{category}_bids_per_quota_lag1"] = ratio

    lag1 = out["coe_a_premium_lag1"].to_numpy(dtype=np.float64)
    lag2 = _lookup(monthly, "a", approval - 2, "premium")
    lag3 = _lookup(monthly, "a", approval - 3, "premium")
    stacked = np.vstack([lag1, lag2, lag3])
    roll = stacked.mean(axis=0)
    roll[np.isnan(stacked).any(axis=0)] = np.nan
    out["coe_a_premium_roll3"] = roll
    return out
