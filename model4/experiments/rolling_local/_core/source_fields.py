"""Preprocess each source column according to its own kind.

The raw ``train.csv`` and ``test.csv`` columns stay unchanged. These
derived columns do not read ``MONTHLY_RENT``.
"""

import numpy as np
import pandas as pd

from experiments.rolling_local._core.keys import norm_text

SOURCE_FIELD_COLUMNS = [
    "approval_month_index",
    "town_key",
    "block_key",
    "street_key",
    "flat_type_ord",
    "flat_model_ord",
    "flat_model_key",
    "floor_area_sqm",
    "fee_amount",
    "furnished_bin",
    "lease_commence_year",
]

# Cheap to expensive. Ties in the observed median rent break on the mean rent.
FLAT_MODEL_ORDER = [
    "2 room",
    "new generation",
    "standard",
    "simplified",
    "model a2",
    "model a",
    "improved",
    "premium apartment",
    "apartment",
    "maisonette",
    "model a maisonette",
    "premium maisonette",
    "adjoined flat",
    "dbss",
    "3gen",
    "improved maisonette",
    "premium apartment loft",
    "terrace",
    "type s1",
    "type s2",
]
FLAT_MODEL_ORD = {name: index for index, name in enumerate(FLAT_MODEL_ORDER)}

FLAT_TYPE_ORD = {
    "1 room": 1,
    "2 room": 2,
    "3 room": 3,
    "4 room": 4,
    "5 room": 5,
    "executive": 6,
}


def flat_type_key(values: pd.Series) -> pd.Series:
    """Collapse ``4-room`` and ``4 room`` onto one label."""
    return norm_text(values).str.replace("-", " ", regex=False).str.replace(r"\s+", " ", regex=True).str.strip()


def flat_model_key(values: pd.Series) -> pd.Series:
    """Normalized flat-model name. The label stays a category, not a number."""
    return norm_text(values)


def flat_type_ordinal(values: pd.Series) -> pd.Series:
    mapped = flat_type_key(values).map(FLAT_TYPE_ORD)
    if mapped.isna().any():
        unknown = sorted(flat_type_key(values)[mapped.isna()].dropna().unique())
        raise ValueError(f"unknown FLAT_TYPE values: {unknown}")
    return mapped.astype("int64")


def flat_model_ordinal(values: pd.Series) -> pd.Series:
    mapped = flat_model_key(values).map(FLAT_MODEL_ORD)
    if mapped.isna().any():
        unknown = sorted(flat_model_key(values)[mapped.isna()].dropna().unique())
        raise ValueError(f"unknown FLAT_MODEL values: {unknown}")
    return mapped.astype("int64")


def approval_month_index(values: pd.Series) -> pd.Series:
    """Months since 2021-01. ``2024-09`` is 44 and ``2026-07`` is 66."""
    text = values.astype("string").str.strip()
    year = pd.to_numeric(text.str.slice(0, 4), errors="coerce")
    month = pd.to_numeric(text.str.slice(5, 7), errors="coerce")
    if year.isna().any() or month.isna().any() or (month < 1).any() or (month > 12).any():
        raise ValueError("RENT_APPROVAL_DATE is not YYYY-MM")
    return ((year - 2021) * 12 + (month - 1)).astype("int64")


def source_frame(rentals: pd.DataFrame) -> pd.DataFrame:
    """One preprocessed column per source field, aligned to ``rentals``."""
    required = [
        "RENT_APPROVAL_DATE",
        "TOWN",
        "BLOCK",
        "STREET",
        "FLAT_TYPE",
        "FLAT_MODEL",
        "FLOOR_AREA_SQM",
        "FURNISHED",
        "LEASE_COMMENCE_DATE",
        "FEE",
    ]
    missing = [column for column in required if column not in rentals.columns]
    if missing:
        raise ValueError(f"source fields are missing columns: {missing}")
    furnished = norm_text(rentals["FURNISHED"])
    if not furnished.isin(["yes", "no"]).all():
        raise ValueError("FURNISHED must be yes or no")
    return pd.DataFrame(
        {
            "approval_month_index": approval_month_index(rentals["RENT_APPROVAL_DATE"]),
            "town_key": norm_text(rentals["TOWN"]),
            "block_key": norm_text(rentals["BLOCK"]),
            "street_key": norm_text(rentals["STREET"]),
            "flat_type_ord": flat_type_ordinal(rentals["FLAT_TYPE"]),
            "flat_model_ord": flat_model_ordinal(rentals["FLAT_MODEL"]),
            "flat_model_key": flat_model_key(rentals["FLAT_MODEL"]),
            "floor_area_sqm": pd.to_numeric(rentals["FLOOR_AREA_SQM"], errors="coerce"),
            "fee_amount": pd.to_numeric(rentals["FEE"], errors="coerce"),
            "furnished_bin": furnished.map({"yes": 1, "no": 0}).astype("int64"),
            "lease_commence_year": pd.to_numeric(rentals["LEASE_COMMENCE_DATE"], errors="coerce"),
        }
    )


def add_source_fields(built: pd.DataFrame, rentals: pd.DataFrame) -> pd.DataFrame:
    """Attach source-field columns after the block join has used its own key."""
    if len(built) != len(rentals):
        raise ValueError("source fields require the same row count as the rentals")
    fields = source_frame(rentals)
    if not np.isfinite(fields[["floor_area_sqm", "fee_amount", "lease_commence_year"]].to_numpy()).all():
        raise ValueError("a numeric source field is missing")
    kept = built.drop(columns=["block_key"], errors="ignore").reset_index(drop=True)
    return pd.concat([kept, fields.reset_index(drop=True)], axis=1)
