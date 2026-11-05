"""Original monthly-rent metrics for one local model; no model is invoked."""
from __future__ import annotations
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .data import address_keys, norm_text, write_json

SCOPES = {"h1_16": (1, 16), "h7_16": (7, 16), "h1_6": (1, 6), "h7_12": (7, 12), "h13_16": (13, 16)}
MODEL_ID = "global_catboost_index"


def _metrics(truth, prediction):
    truth, prediction = np.asarray(truth, dtype=float), np.asarray(prediction, dtype=float)
    if truth.ndim != 1 or truth.shape != prediction.shape or not len(truth):
        raise ValueError("Metric inputs must be matching nonempty vectors")
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("Metric inputs must be finite")
    error = prediction - truth
    mse = float(np.mean(error**2))
    return {"rows": len(truth), "mse": mse, "rmse": float(np.sqrt(mse)),
            "mae": float(np.mean(np.abs(error))), "bias_pred_minus_truth": float(np.mean(error)),
            "relative_bias_percent": float(100 * np.sum(error) / np.sum(truth)),
            "overprediction_fraction": float(np.mean(prediction > truth))}


def _scopes(frame):
    return {name: _metrics(part.truth.to_numpy(), part.prediction.to_numpy())
            for name, (low, high) in SCOPES.items()
            for part in [frame.loc[frame.horizon.between(low, high)]]}


def _group_tables(frame, output, prefix=""):
    files = {}
    for dimension, column in (("horizon", "horizon"), ("town", "town"),
                              ("flat_type", "flat_type"), ("seen_address", "seen_address")):
        rows = []
        for label, part in frame.groupby(column, observed=True, sort=True):
            rows.append({"dimension": dimension, "group": label, "model": MODEL_ID,
                         **_metrics(part.truth.to_numpy(), part.prediction.to_numpy())})
        path = output / f"{prefix}metrics_by_{dimension}.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        files[dimension] = path.name
    path = output / f"{prefix}metrics_by_scope.csv"
    pd.DataFrame([{"scope": scope, "model": MODEL_ID, **values} for scope, values in _scopes(frame).items()]).to_csv(path, index=False)
    files["scope"] = path.name
    return files


def evaluate_local(source, prediction, origin, folder, *, seen_addresses, expected_rows=None):
    source = source.reset_index(drop=True)
    dates = source.RENT_APPROVAL_DATE.astype(str).str[:7]
    horizon = pd.PeriodIndex(dates, freq="M").asi8 - pd.Period(origin, freq="M").ordinal
    if set(horizon) != set(range(1, 17)) or (expected_rows is not None and len(source) != expected_rows):
        raise ValueError("Incomplete configured forecast records")
    ids = source.source_row_id.to_numpy()
    if not np.array_equal(ids, ids.astype(np.int64)) or len(set(ids)) != len(ids) or np.any(ids < 0):
        raise ValueError("Invalid original row identities")
    truth = source.MONTHLY_RENT.to_numpy(float)
    prediction = np.asarray(prediction, dtype=float).reshape(-1)
    _metrics(truth, prediction)
    address = address_keys(source)
    frame = pd.DataFrame({"source_row_id": ids, "date": dates, "origin": origin,
                          "horizon": horizon, "truth": truth, "prediction": prediction,
                          "error_pred_minus_truth": prediction-truth, "town": norm_text(source.TOWN),
                          "flat_type": norm_text(source.FLAT_TYPE).str.replace("-", " ", regex=False).str.replace(r"\s+", " ", regex=True),
                          "full_address_key": address,
                          "seen_address": np.where(address.isin(seen_addresses), "seen", "unseen")})
    folder = Path(folder)
    path = folder / "forecasts.csv"
    if path.exists() or (folder / "metrics.json").exists():
        raise FileExistsError("Completed local evaluation cannot be overwritten")
    frame.to_csv(path, index=False)
    back = pd.read_csv(path, float_precision="round_trip")
    if not np.array_equal(back.source_row_id, ids) or not np.allclose(back.prediction, prediction, rtol=1e-12, atol=1e-12):
        raise ValueError("Saved forecast identities or values changed")
    result = {"status": "complete", "model": MODEL_ID, "origin": origin, "rows": len(frame),
              "overall": _metrics(truth, prediction), "segments": _scopes(frame),
              "by_horizon": {str(h): _metrics(part.truth.to_numpy(), part.prediction.to_numpy())
                             for h, part in frame.groupby("horizon", sort=True)},
              "forecasts_file": str(path.resolve()), "forecasts_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "tables": _group_tables(frame, folder), "rows_filtered": 0,
              "evaluation_unit": "original monthly rent; equal record weights"}
    write_json(folder / "metrics.json", result)
    return result


def aggregate_results(results, output, planned_origins=None):
    output = Path(output)
    origins = list(planned_origins) if planned_origins is not None else [r["origin"] for r in results]
    complete, frames = [], []
    seen = set()
    for item in results:
        if item.get("status") != "complete":
            continue
        origin = item["origin"]
        if origin not in origins or origin in seen:
            raise ValueError("Unexpected or duplicate origin")
        seen.add(origin)
        path = Path(item["forecasts_file"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["forecasts_sha256"]:
            raise ValueError("Completed forecast artifact changed")
        frame = pd.read_csv(path, float_precision="round_trip")
        if len(frame) != item["rows"] or not frame.origin.eq(origin).all() or set(frame.horizon) != set(range(1, 17)):
            raise ValueError("Completed forecast population changed")
        recomputed = _scopes(frame)
        for scope in SCOPES:
            for field in ("mse", "rmse", "mae", "bias_pred_minus_truth", "relative_bias_percent", "overprediction_fraction"):
                if not np.isclose(recomputed[scope][field], item["segments"][scope][field], rtol=1e-10, atol=1e-10):
                    raise ValueError("Completed metrics changed")
        complete.append(item)
        frames.append(frame)
    result = {"status": "complete" if len(complete) == len(origins) else "partial" if complete else "no_complete_origins", "model": MODEL_ID,
              "planned_origins": origins, "completed_origins": [r["origin"] for r in complete],
              "missing_origins": [origin for origin in origins if origin not in seen],
              "evaluation_unit": "original monthly rent; equal record weights",
              "overlapping_windows": True, "independent_market_cycles_claimed": False}
    if complete:
        pooled = pd.concat(frames, ignore_index=True)
        result["pooled"] = _scopes(pooled)
        result["pooled_rows_with_window_overlap_retained"] = len(pooled)
        result["equal_origin"] = {}
        for scope in SCOPES:
            mse = float(np.mean([r["segments"][scope]["mse"] for r in complete]))
            result["equal_origin"][scope] = {"origins": len(complete), "mse": mse, "rmse": float(np.sqrt(mse)),
                **{field: float(np.mean([r["segments"][scope][field] for r in complete]))
                   for field in ("mae", "bias_pred_minus_truth", "relative_bias_percent", "overprediction_fraction")}}
        result["tables"] = _group_tables(pooled, output, "pooled_")
    write_json(output / "aggregate_metrics.json", result)
    lines = ["# Local rolling backtest", "", f"Status: {result['status']}", "",
             f"Completed origins: {', '.join(result['completed_origins']) or 'none'}", ""]
    if complete:
        lines += ["| Scope | Equal-origin RMSE | Pooled RMSE |", "|---|---:|---:|"]
        lines += [f"| {scope} | {result['equal_origin'][scope]['rmse']:.6f} | {result['pooled'][scope]['rmse']:.6f} |" for scope in SCOPES]
    lines += ["", "All metrics use original monthly rent. Overlapping windows are not independent market cycles.", ""]
    (output / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    return result
