"""Predeclared temporal boundaries; no target values are used here."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from configs import config_path
from src.paths import PROJECT_ROOT as ROOT
TARGET = "MONTHLY_RENT"
ROW_ID = "row_id"
PROTOCOL_PATH = config_path("backtest/protocol.json")


def read_protocol(path: Path = PROTOCOL_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def digest_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def months(frame: pd.DataFrame) -> pd.PeriodIndex:
    return pd.PeriodIndex(frame["RENT_APPROVAL_DATE"].astype(str), freq="M")


def horizon(frame: pd.DataFrame, origin: str) -> np.ndarray:
    return months(frame).asi8 - pd.Period(origin, freq="M").ordinal


def horizon_bin(values: np.ndarray, bins: list[list[int]]) -> np.ndarray:
    out = np.full(len(values), "outside", dtype=object)
    for low, high in bins:
        out[(values >= low) & (values <= high)] = str(low) if low == high else f"{low}-{high}"
    return out


def require_identity(frame: pd.DataFrame) -> None:
    if ROW_ID not in frame or frame[ROW_ID].isna().any() or not frame[ROW_ID].is_unique:
        raise ValueError("row_id must be present, non-null, and unique")
    values = frame[ROW_ID].to_numpy()
    if not np.issubdtype(values.dtype, np.integer) or (values < 0).any():
        raise ValueError("row_id must contain nonnegative integers")


def require_history(history: pd.DataFrame, origin: str) -> None:
    require_identity(history)
    if history.empty or TARGET not in history:
        raise ValueError("fit requires a nonempty, physically truncated labeled history")
    if (months(history) > pd.Period(origin, freq="M")).any():
        raise ValueError("history contains labels after its fit boundary")
    if not np.isfinite(history[TARGET].to_numpy(dtype=float)).all():
        raise ValueError("history labels must be finite")


def require_query(query: pd.DataFrame, origin: str, max_horizon: int = 16) -> None:
    require_identity(query)
    if TARGET in query:
        raise ValueError("prediction features must not contain target labels")
    hs = horizon(query, origin)
    if len(query) == 0 or ((hs < 1) | (hs > max_horizon)).any():
        raise ValueError("query must be strictly after origin and within the declared horizon")


def history_hash(history: pd.DataFrame) -> str:
    """Canonical permitted-label identity; independent of future rows/labels."""
    require_identity(history)
    canonical = history[[ROW_ID, "RENT_APPROVAL_DATE", TARGET]].reset_index(drop=True).sort_values(ROW_ID).copy()
    canonical[TARGET] = canonical[TARGET].astype(float)
    canonical["RENT_APPROVAL_DATE"] = canonical["RENT_APPROVAL_DATE"].astype(str)
    if not np.isfinite(canonical[TARGET]).all():
        raise ValueError("non-finite historical labels")
    return hashlib.sha256(pd.util.hash_pandas_object(canonical, index=False).to_numpy().tobytes()).hexdigest()


def inner_split(history: pd.DataFrame, origin: str, config: dict):
    require_history(history, origin)
    cutoff = pd.Period(origin, freq="M") - config["inner_validation_months"]
    mask = months(history) <= cutoff
    train = history.loc[mask].copy()
    valid = history.loc[~mask].copy()
    if train.empty or valid.empty:
        raise ValueError("inner training and validation periods must both have rows")
    labels = valid[TARGET].to_numpy(dtype=float).copy()
    return train, valid.drop(columns=TARGET), labels, str(cutoff)
