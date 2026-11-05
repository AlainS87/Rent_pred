"""Join the rental table to the auxiliary feature table by row order."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.features.build_features import FEATURE_COLUMNS, ORIGINAL_COLUMNS
from src.pipeline.paths import DATA, FEATURE_DATA

ADDRESS_DTYPES = {"TOWN": str, "BLOCK": str, "STREET": str}
SOURCE_COLUMNS = set(ORIGINAL_COLUMNS) | {"MONTHLY_RENT"}


@dataclass
class TimeSplit:
    fit: pd.DataFrame
    search: pd.DataFrame
    confirm: pd.DataFrame
    fit_months: list[str]
    search_months: list[str]
    confirm_months: list[str]


def load_joined(split: str, columns: list[str]) -> pd.DataFrame:
    """Concatenate source and auxiliary columns. Row order is unchanged.

    ``split`` is ``train`` or ``test``. The two files must have the same
    number of rows. A column that exists in both files is an error.
    """
    source_path = DATA / ("query.csv" if split == "test" else f"{split}.csv")
    feature_path = FEATURE_DATA / f"{split}_features.csv"
    needed = list(dict.fromkeys([*columns, "RENT_APPROVAL_DATE"]))
    if "BLOCK" in needed:
        needed = list(dict.fromkeys([*needed, "TOWN", "STREET"]))
    if split == "train":
        needed = list(dict.fromkeys([*needed, "MONTHLY_RENT"]))

    source_header = set(pd.read_csv(source_path, nrows=0).columns)
    feature_header = set(pd.read_csv(feature_path, nrows=0).columns)
    unknown = [
        column
        for column in needed
        if column not in source_header and column not in feature_header
    ]
    if unknown:
        raise ValueError(f"{split} is missing columns: {unknown}")
    overlap = [column for column in needed if column in source_header and column in feature_header]
    if overlap:
        raise ValueError(f"{split} columns appear in both tables: {overlap}")

    source_cols = [column for column in needed if column in source_header]
    feature_cols = [column for column in needed if column in feature_header]
    # Always read one auxiliary column so the row-count check still runs
    # when the requested features all come from the source file.
    drop_extra = False
    if not feature_cols:
        feature_cols = [FEATURE_COLUMNS[0]]
        drop_extra = True

    source_dtype = {name: kind for name, kind in ADDRESS_DTYPES.items() if name in source_cols}
    source = pd.read_csv(source_path, usecols=source_cols, dtype=source_dtype)
    features = pd.read_csv(feature_path, usecols=feature_cols)
    if len(source) != len(features):
        raise ValueError(
            f"{split} row counts differ: source {len(source)} versus features {len(features)}"
        )
    if drop_extra:
        features = features.iloc[:, 0:0]
    joined = pd.concat([source.reset_index(drop=True), features.reset_index(drop=True)], axis=1)
    if joined.columns.duplicated().any():
        raise ValueError(f"{split} join produced duplicate column names")
    return joined


def time_split(frame: pd.DataFrame, val_months: int, max_rows: int | None) -> TimeSplit:
    """Hold out the last approval months for search and confirm.

    ``max_rows`` keeps every period and stride-samples each one. It is a
    smoke cap. The full table is used when it is omitted.
    """
    if val_months < 1:
        raise SystemExit("--val-months must be at least 1")
    work = frame.copy()
    work["_month"] = work["RENT_APPROVAL_DATE"].astype("string").str.slice(0, 7)
    months = sorted(month for month in work["_month"].dropna().unique())
    need = val_months * 2 + 1
    if len(months) < need:
        raise SystemExit(f"need at least {need} approval months, found {len(months)}")
    confirm_months = months[-val_months:]
    search_months = months[-2 * val_months : -val_months]
    fit_months = months[: -2 * val_months]
    parts = {
        "fit": work[work["_month"].isin(fit_months)],
        "search": work[work["_month"].isin(search_months)],
        "confirm": work[work["_month"].isin(confirm_months)],
    }
    if max_rows is not None:
        counts = {name: len(part) for name, part in parts.items()}
        alloc = _allocate(counts, max_rows)
        parts = {name: _stride(part, alloc.get(name, 0)) for name, part in parts.items()}
    cleaned = {}
    for name, part in parts.items():
        if part.empty:
            raise SystemExit(f"the {name} period has no rows")
        cleaned[name] = part.drop(columns="_month").reset_index(drop=True)
    return TimeSplit(
        fit=cleaned["fit"],
        search=cleaned["search"],
        confirm=cleaned["confirm"],
        fit_months=list(fit_months),
        search_months=list(search_months),
        confirm_months=list(confirm_months),
    )


def _stride(frame: pd.DataFrame, count: int) -> pd.DataFrame:
    if count >= len(frame):
        return frame
    if count <= 0:
        return frame.iloc[0:0]
    positions = np.linspace(0, len(frame) - 1, count)
    positions = np.unique(np.rint(positions).astype(int))
    return frame.iloc[positions]


def _allocate(counts: dict[str, int], max_rows: int) -> dict[str, int]:
    keys = [key for key in ("fit", "search", "confirm") if counts[key] > 0]
    if max_rows < len(keys):
        raise SystemExit("--max-rows leaves no row for a time period")
    alloc = {key: 1 for key in keys}
    remaining = max_rows - len(keys)
    capacities = {key: counts[key] - 1 for key in keys}
    while remaining > 0 and sum(capacities.values()) > 0:
        weight = sum(capacities.values())
        shares = {key: int(remaining * capacities[key] / weight) for key in keys}
        if sum(shares.values()) == 0:
            key = max(keys, key=lambda name: (capacities[name], name))
            shares[key] = 1
        for key in keys:
            take = min(shares[key], capacities[key], remaining)
            alloc[key] += take
            capacities[key] -= take
            remaining -= take
    return alloc
