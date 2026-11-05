"""Scoring alone obtains forecast-window truth; models never import this module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest.data import _read_selected_labels
from src.backtest.protocol import ROOT, ROW_ID, TARGET, horizon, horizon_bin


def attach_truth(predictions: pd.DataFrame, root: Path = ROOT) -> pd.DataFrame:
    required = {"origin", ROW_ID, "RENT_APPROVAL_DATE", "horizon"}
    if not required.issubset(predictions) or TARGET in predictions:
        raise ValueError("scoring expects identified predictions without existing truth")
    if predictions.duplicated(["origin", ROW_ID]).any():
        raise ValueError("duplicate prediction instances")
    row_ids = predictions[ROW_ID].drop_duplicates().to_numpy(dtype=np.int64)
    dates = pd.read_csv(root / "data/raw/train.csv", usecols=["RENT_APPROVAL_DATE"])["RENT_APPROVAL_DATE"]
    if (row_ids < 0).any() or (row_ids >= len(dates)).any():
        raise ValueError("prediction row IDs are outside the source table")
    expected_dates = predictions[ROW_ID].map(dates)
    if not np.array_equal(expected_dates.astype(str), predictions["RENT_APPROVAL_DATE"].astype(str)):
        raise ValueError("prediction row identity/date does not match original source")
    truth = _read_selected_labels(root / "data/raw/train.csv", row_ids)
    values = pd.Series(truth, index=row_ids)
    result = predictions.copy()
    result[TARGET] = result[ROW_ID].map(values).to_numpy(dtype=float)
    return result


def metrics(predictions: pd.DataFrame, model_columns: list[str], config: dict) -> pd.DataFrame:
    """Identical row sets for all models; pooled scores weight forecast instances."""
    if predictions.duplicated(["origin", ROW_ID]).any():
        raise ValueError("duplicate prediction instances")
    truth = predictions[TARGET].to_numpy(dtype=float)
    if not np.isfinite(truth).all():
        raise ValueError("non-finite truth")
    for model in model_columns:
        if not np.isfinite(predictions[model].to_numpy(dtype=float)).all():
            raise ValueError("every model must predict every identical evaluation row")
    for origin, part in predictions.groupby("origin", sort=True):
        if not np.array_equal(horizon(part, origin), part.horizon.to_numpy()):
            raise ValueError("horizon and origin/date disagree")
    work = predictions.copy()
    work["horizon_bin"] = horizon_bin(work.horizon.to_numpy(), config["horizon_bins"])
    if (work.horizon_bin == "outside").any():
        raise ValueError("forecast horizon outside 1..16")
    output = []
    origins = [(str(key), part) for key, part in work.groupby("origin", sort=True)] + [("pooled_instances", work)]
    for origin, subset in origins:
        pieces = [("all", subset)]
        pieces += [(str(key), part) for key, part in subset.groupby("horizon_bin", sort=False)]
        pieces += [(f"month_{key}", part) for key, part in subset.groupby("horizon", sort=True)]
        for bucket, part in pieces:
            for model in model_columns:
                error = part[model].to_numpy(dtype=float) - part[TARGET].to_numpy(dtype=float)
                sse = float(np.dot(error, error))
                output.append({"origin": origin, "horizon": bucket, "model": model, "n": len(error),
                               "sse": sse, "rmse": float(np.sqrt(sse / len(error))), "bias": float(error.mean())})
    return pd.DataFrame(output)
