"""Explicit input contract and history-only phase transformations."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.rolling_local._core import source_fields
from .schema import MACRO_COLUMNS as LEVELS
from .schema import BASE_COLUMNS

LOAD_COLUMNS = BASE_COLUMNS

RAW_REQUIRED = ["RENT_APPROVAL_DATE", "TOWN", "BLOCK", "STREET", "FLAT_TYPE",
                "FLAT_MODEL", "FLOOR_AREA_SQM", "MONTHLY_RENT"]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, payload):
    from .status_io import write_critical
    write_critical(path, payload)


def norm_text(values):
    return values.astype("string").str.strip().str.lower().str.replace(r"\s+", " ", regex=True)


def address_keys(frame):
    pieces = [norm_text(frame[c]) for c in ("TOWN", "street_key", "block_key")]
    for value in pieces:
        if value.isna().any() or value.str.len().eq(0).any() or value.str.contains("|", regex=False).any():
            raise ValueError("Missing/empty/ambiguous complete-address component")
    return pieces[0] + "|" + pieces[1] + "|" + pieces[2]


def load_data(raw_path, feature_path, output, *, expected_rows=None, expected_hashes=None):
    """Join explicitly supplied tables by unchanged source-row position.

    The auxiliary cache must contain the original eleven source transforms for
    alignment verification. Its historical model rank is replaced in each phase.
    All source columns required by the two fixed models must be present.
    """
    raw_path, feature_path, output = Path(raw_path), Path(feature_path), Path(output)
    expected_hashes = expected_hashes or {}
    hashes = {"raw": digest(raw_path), "features": digest(feature_path)}
    for label, want in expected_hashes.items():
        if label not in hashes or hashes[label] != want:
            raise ValueError(f"Input checksum mismatch: {label}")
    raw = pd.read_csv(raw_path, dtype={"TOWN": str, "BLOCK": str, "STREET": str})
    if feature_path.suffix.lower() == ".parquet":
        auxiliary = pd.read_parquet(feature_path)
    elif feature_path.suffix.lower() == ".csv":
        auxiliary = pd.read_csv(feature_path, float_precision="round_trip",
                                dtype={c: str for c in ("town_key", "block_key", "street_key", "flat_model_key")})
    else:
        raise ValueError("Feature table must be CSV or Parquet")
    if not len(raw) or len(raw) != len(auxiliary) or (expected_rows is not None and len(raw) != expected_rows):
        raise ValueError("Raw/feature table row count mismatch")
    if raw.columns.duplicated().any() or auxiliary.columns.duplicated().any():
        raise ValueError("Duplicate input column names")
    missing = set(RAW_REQUIRED + ["FEE", "FURNISHED", "LEASE_COMMENCE_DATE"]) - set(raw)
    if missing:
        raise ValueError(f"Missing source columns: {sorted(missing)}")
    expected = source_fields.source_frame(raw)
    alignment = {}
    for column in source_fields.SOURCE_FIELD_COLUMNS:
        if column not in auxiliary:
            raise ValueError(f"Missing alignment field: {column}")
        a, b = expected[column], auxiliary[column]
        if pd.api.types.is_numeric_dtype(a):
            same = np.isclose(a.to_numpy(float, na_value=np.nan), pd.to_numeric(b).to_numpy(float, na_value=np.nan), rtol=0, atol=0, equal_nan=True)
        else:
            same = a.astype("string").fillna("__NA__").eq(b.astype("string").fillna("__NA__")).to_numpy()
        alignment[column] = int((~same).sum())
        if alignment[column]:
            raise ValueError(f"Source/cache row alignment failed: {column}")
    columns = [c for c in LOAD_COLUMNS if c not in raw.columns] + list(LEVELS)
    if len(columns) != 27 or len(set(columns)) != 27 or set(columns) - set(auxiliary):
        raise ValueError("The fixed models require their 27 consumed auxiliary columns")
    frame = pd.concat([raw[RAW_REQUIRED].reset_index(drop=True), auxiliary[columns].reset_index(drop=True)], axis=1)
    frame.insert(0, "source_row_id", np.arange(len(frame), dtype=np.int64))
    rent, area = frame.MONTHLY_RENT.to_numpy(float), frame.floor_area_sqm.to_numpy(float)
    if not np.isfinite(rent).all() or not (rent > 0).all() or not np.isfinite(area).all() or not (area > 0).all():
        raise ValueError("Rent and floor area must be finite and positive")
    frame["month"] = pd.PeriodIndex(frame.RENT_APPROVAL_DATE.astype(str).str[:7], freq="M").astype(str)
    frame["full_address_key"] = address_keys(frame)
    output.mkdir(parents=True, exist_ok=False)
    # Preserve the original CSV round-trip compatibility check for consumed fields.
    path = output / "consumed_features.csv"
    auxiliary[columns].to_csv(path, index=False)
    roundtrip = pd.read_csv(path, float_precision="round_trip", dtype={c: str for c in columns if not pd.api.types.is_numeric_dtype(auxiliary[c])})
    pd.testing.assert_frame_equal(auxiliary[columns].reset_index(drop=True), roundtrip, check_dtype=False, rtol=1e-12, atol=1e-12)
    frame[columns] = roundtrip[columns]
    write_json(output / "manifest.json", {"rows": len(frame), "input_sha256": hashes,
               "raw_source": raw_path.name, "feature_source": feature_path.name,
               "consumed_auxiliary_columns": columns, "source_alignment_mismatches": alignment,
               "row_identity": "zero-based position in the unchanged raw table",
               "rank_replaced_in_each_phase": True, "cache_regenerated": False})
    return frame


def rank_for_phase(train, apply, folder):
    """Global monthly-rent median/mean/name rank, fitted only on phase history."""
    train, apply = train.copy(), apply.copy()
    grouped = pd.DataFrame({"name": norm_text(train.FLAT_MODEL), "rent": train.MONTHLY_RENT.to_numpy(float)}).dropna(subset=["name"]).groupby("name").rent.agg(median="median", mean="mean", rows="size").reset_index()
    grouped = grouped.sort_values(["median", "mean", "name"], kind="mergesort").reset_index(drop=True)
    if not len(grouped):
        raise ValueError("No model categories in permitted history")
    mapping = dict(zip(grouped.name.astype(str), range(len(grouped))))
    fallback = float(np.median(list(mapping.values())))
    for frame in (train, apply):
        frame["flat_model_ord"] = norm_text(frame.FLAT_MODEL).map(mapping).fillna(fallback).astype(float)
    write_json(Path(folder) / "flat_model_rank.json", {"mapping": mapping, "fallback": fallback,
               "fit_rows": len(train), "fit_end": train.month.max(),
               "order": "monthly rent median, mean, normalized name ascending; zero-based",
               "categories": grouped.to_dict("records"), "apply_labels_used": False})
    return train, apply


def enforce_phase_info(train, apply, cutoff, folder):
    """Freeze future macro inputs to the last finite permissible lag1 value."""
    train, apply = train.copy(), apply.copy()
    cutoff = str(pd.Period(cutoff, freq="M"))
    train_month = train.RENT_APPROVAL_DATE.astype(str).str[:7]
    apply_month = apply.RENT_APPROVAL_DATE.astype(str).str[:7]
    if train.empty or (train_month > cutoff).any() or (apply_month <= cutoff).any():
        raise ValueError("Phase cutoff violation")
    audit = {}
    for column in LEVELS:
        work = pd.DataFrame({"month": train_month, "value": train[column].to_numpy(float)})
        if np.isinf(work.value).any():
            raise ValueError(f"Infinite historical macro input: {column}")
        monthly = []
        for month, part in work.groupby("month", sort=True):
            finite = part.value[np.isfinite(part.value)]
            if len(finite) and (len(finite) != len(part) or finite.nunique() != 1):
                raise ValueError(f"Inconsistent monthly macro input: {column}/{month}")
            if len(finite):
                monthly.append((month, float(finite.iloc[0])))
        month, value = monthly[-1] if monthly else (None, None)
        apply[column] = np.nan if value is None else value
        audit[column] = {"input_row_month": month,
                         "reference_month": str(pd.Period(month, freq="M") - 1) if month else None,
                         "value": value, "missing_permitted_value": value is None}
    write_json(Path(folder) / "macro_freeze.json", {"cutoff": cutoff, "columns": audit,
               "future_actual_macro_used": False, "historical_training_macro_unchanged": True})
    return train, apply
