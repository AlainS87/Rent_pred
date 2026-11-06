"""Smoothed mean rents over the previous three approval months.

A row in month T uses training rents from T-1, T-2, and T-3 only. Month T is
excluded, and test rows never contribute rents. January 2021 has none of those
months in the file, so its means stay null.
"""

import numpy as np
import pandas as pd

PRIOR_STRENGTH = 20
WINDOW_MONTHS = 3

GROUP_MEAN_COLUMNS = [
    "mean_rent_postal_sector",
    "mean_rent_postal_district",
    "mean_rent_nearest_mall",
    "mean_rent_nearest_mrt",
    "mean_rent_nearest_primary",
    "mean_rent_nearest_secondary",
]

# First two digits of a Singapore postal code, mapped to the legacy postal district.
_SECTOR_DISTRICT = {}
for _sectors, _district in (
    ("01,02,03,04,05,06", "D01"),
    ("07,08", "D02"),
    ("14,15,16", "D03"),
    ("09,10", "D04"),
    ("11,12,13", "D05"),
    ("17", "D06"),
    ("18,19", "D07"),
    ("20,21", "D08"),
    ("22,23", "D09"),
    ("24,25,26,27", "D10"),
    ("28,29,30", "D11"),
    ("31,32,33", "D12"),
    ("34,35,36,37", "D13"),
    ("38,39,40,41", "D14"),
    ("42,43,44,45", "D15"),
    ("46,47,48", "D16"),
    ("49,50,81", "D17"),
    ("51,52", "D18"),
    ("53,54,55,82", "D19"),
    ("56,57", "D20"),
    ("58,59", "D21"),
    ("60,61,62,63,64", "D22"),
    ("65,66,67,68", "D23"),
    ("69,70,71", "D24"),
    ("72,73", "D25"),
    ("77,78", "D26"),
    ("75,76", "D27"),
    ("79,80", "D28"),
):
    for _sector in _sectors.split(","):
        _SECTOR_DISTRICT[_sector] = _district


def postal_sector(postal: pd.Series) -> pd.Series:
    """First two digits. Empty postal codes stay null and do not form a group."""
    text = postal.astype("string").str.replace(r"\.0$", "", regex=True)
    digits = text.str.replace(r"\D", "", regex=True)
    digits = digits.mask(digits.eq(""))
    padded = digits.str.zfill(6)
    return padded.str.slice(0, 2).where(digits.notna())


def postal_district(sector: pd.Series) -> pd.Series:
    return sector.map(_SECTOR_DISTRICT).astype("string")


def _month_start(dates: pd.Series) -> pd.Series:
    return pd.to_datetime(dates.astype("string")).dt.to_period("M").dt.to_timestamp()


def _window_totals(months: pd.Series, keys: pd.Series | None, history: pd.DataFrame, key_column: str | None):
    """Sum and count of history rents in the three months before each row."""
    if key_column is None:
        stats = (
            history.groupby("month", as_index=False)
            .agg(rent_sum=("MONTHLY_RENT", "sum"), n=("MONTHLY_RENT", "count"))
        )
    else:
        usable = history.loc[history[key_column].notna()]
        if usable.empty:
            return np.zeros(len(months)), np.zeros(len(months))
        stats = (
            usable.groupby([key_column, "month"], as_index=False)
            .agg(rent_sum=("MONTHLY_RENT", "sum"), n=("MONTHLY_RENT", "count"))
        )
        stats = stats.rename(columns={key_column: "key"})
        stats["key"] = stats["key"].astype("string").astype(object)

    query_month = pd.to_datetime(months.to_numpy())
    query_key = None if keys is None else keys.astype("string").fillna("__no_group__").astype(object).to_numpy()
    total_sum = np.zeros(len(months), dtype=np.float64)
    total_n = np.zeros(len(months), dtype=np.float64)
    periods = pd.Series(query_month).dt.to_period("M")
    for lag in range(1, WINDOW_MONTHS + 1):
        lagged = (periods - lag).dt.to_timestamp().to_numpy()
        left = pd.DataFrame({"month": lagged, "row": np.arange(len(months))})
        if query_key is not None:
            left["key"] = query_key
            merged = left.merge(stats, on=["month", "key"], how="left")
        else:
            merged = left.merge(stats, on="month", how="left")
        total_sum += merged["rent_sum"].fillna(0).to_numpy()
        total_n += merged["n"].fillna(0).to_numpy()
    return total_sum, total_n


def _smoothed(group_sum: np.ndarray, group_n: np.ndarray, global_sum: np.ndarray, global_n: np.ndarray) -> np.ndarray:
    group_n = np.nan_to_num(group_n, nan=0.0)
    global_mean = np.divide(global_sum, global_n, out=np.full_like(global_sum, np.nan), where=global_n > 0)
    group_mean = np.divide(group_sum, group_n, out=np.zeros_like(group_sum, dtype=np.float64), where=group_n > 0)
    smoothed = (group_n * group_mean + PRIOR_STRENGTH * global_mean) / (group_n + PRIOR_STRENGTH)
    return np.where(np.isnan(global_mean), np.nan, smoothed)


def _with_month(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["month"] = _month_start(out["RENT_APPROVAL_DATE"])
    return out


def _keys(frame: pd.DataFrame) -> dict[str, pd.Series]:
    sector = postal_sector(frame["postal_code"])
    return {
        "mean_rent_postal_sector": sector,
        "mean_rent_postal_district": postal_district(sector),
        "mean_rent_nearest_mall": frame["nearest_mall_name"],
        "mean_rent_nearest_mrt": frame["nearest_open_mrt_code"],
        "mean_rent_nearest_primary": frame["nearest_primary_name"],
        "mean_rent_nearest_secondary": frame["nearest_secondary_name"],
    }


def add_group_mean_columns(frame: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Append the six smoothed means. ``history`` must be the training rows."""
    if "MONTHLY_RENT" not in history.columns:
        raise ValueError("group means need MONTHLY_RENT on the training history")
    target = _with_month(frame)
    past = _with_month(history)
    keys = _keys(target)
    history_keys = _keys(past)
    for column, key in history_keys.items():
        past[column] = key
    global_sum, global_n = _window_totals(target["month"], None, past, None)
    out = frame.copy()
    for column, key in keys.items():
        group_sum, group_n = _window_totals(target["month"], key, past, column)
        out[column] = _smoothed(group_sum, group_n, global_sum, global_n)
    return out


def smoothed_group_month_table(
    history: pd.DataFrame,
    key: pd.Series,
    months: pd.Series,
) -> pd.DataFrame:
    """Smoothed mean for every group and month, including months with no row in that group.

    ``history`` supplies ``RENT_APPROVAL_DATE`` and ``MONTHLY_RENT``. ``key`` aligns with
    those rows. ``months`` is the set of approval months to show as columns.
    """
    if "MONTHLY_RENT" not in history.columns:
        raise ValueError("group means need MONTHLY_RENT on the training history")
    groups = pd.unique(key.dropna())
    month_stamps = _month_start(months).drop_duplicates().sort_values()
    grid_keys = np.repeat(groups, len(month_stamps))
    grid_months = np.tile(month_stamps.to_numpy(), len(groups))
    past = history.copy()
    past["month"] = _month_start(past["RENT_APPROVAL_DATE"])
    past["__key__"] = key.to_numpy()
    group_sum, group_n = _window_totals(
        pd.Series(grid_months),
        pd.Series(grid_keys),
        past,
        "__key__",
    )
    global_sum, global_n = _window_totals(pd.Series(grid_months), None, past, None)
    values = _smoothed(group_sum, group_n, global_sum, global_n)
    labels = pd.Series(month_stamps).dt.strftime("%Y-%m")
    return pd.DataFrame(
        values.reshape(len(groups), len(month_stamps)),
        index=pd.Index(groups, name="group"),
        columns=pd.Index(labels, name="month"),
    )


def assert_past_only(train_frame: pd.DataFrame, test_frame: pd.DataFrame) -> None:
    """January 2021 is empty, and March 2025 uses only December 2024 through February 2025."""
    january = train_frame["RENT_APPROVAL_DATE"].astype("string").eq("2021-01")
    if not train_frame.loc[january, GROUP_MEAN_COLUMNS].isna().all().all():
        raise AssertionError("2021-01 group means must be null")

    train_month = _month_start(train_frame["RENT_APPROVAL_DATE"])
    test_month = _month_start(test_frame["RENT_APPROVAL_DATE"])
    march = pd.Timestamp("2025-03-01")
    window_start = pd.Timestamp("2024-12-01")
    in_window = train_month.ge(window_start) & train_month.lt(march)
    march_train = train_frame.loc[train_month.eq(march) & train_frame["nearest_mall_name"].notna()]
    mall = None
    for candidate in march_train["nearest_mall_name"].value_counts().index:
        has_window = in_window & train_frame["nearest_mall_name"].eq(candidate)
        has_test = test_month.eq(march) & test_frame["nearest_mall_name"].eq(candidate)
        if has_window.any() and has_test.any():
            mall = candidate
            break
    if mall is None:
        raise AssertionError("no mall is available to check the previous-quarter window")

    rents = train_frame.loc[in_window & train_frame["nearest_mall_name"].eq(mall), "MONTHLY_RENT"]
    global_rents = train_frame.loc[in_window, "MONTHLY_RENT"]
    expected = (len(rents) * rents.mean() + PRIOR_STRENGTH * global_rents.mean()) / (len(rents) + PRIOR_STRENGTH)
    got_train = train_frame.loc[train_month.eq(march) & train_frame["nearest_mall_name"].eq(mall), "mean_rent_nearest_mall"]
    got_test = test_frame.loc[test_month.eq(march) & test_frame["nearest_mall_name"].eq(mall), "mean_rent_nearest_mall"]
    if not np.allclose(got_train, expected) or not np.allclose(got_test, expected):
        raise AssertionError("March 2025 group mean is not the previous three months only")
