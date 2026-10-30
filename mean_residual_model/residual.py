"""Fit one best genome to the gap from the town-month mean."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.pipeline.fit import load_model_module, load_preprocess_module, rmse
from src.pipeline.genome import feature_hash

MODELS = ("ridge", "xgb", "mlp", "catboost")


def load_best_genome(root: Path, model: str) -> dict:
    path = root / "models" / model / "best" / "genome.json"
    genome = json.loads(path.read_text(encoding="utf-8"))
    if not genome.get("features"):
        raise SystemExit(f"{model} best genome has no features")
    return genome


def fit_residual(
    model: str,
    genome: dict,
    fit_frame: pd.DataFrame,
    search_frame: pd.DataFrame,
    y_fit: np.ndarray,
    y_search: np.ndarray,
    seed: int,
):
    """Fit preprocess on the residual. Search residual stops the trees and the MLP."""
    module = load_model_module(model)
    preprocess = load_preprocess_module(model)
    columns = list(genome["features"])
    preparer = preprocess.fit_preprocessor(genome["preprocess"], fit_frame, y_fit, columns)
    x_fit = preprocess.transform_frame(preparer, fit_frame)
    x_search = preprocess.transform_frame(preparer, search_frame)
    bundle = module.fit_model(
        x_fit,
        y_fit,
        x_search,
        y_search,
        genome["architecture"],
        seed,
        epochs=None,
    )
    return preparer, bundle


def predict_residual(model: str, preparer, bundle, frame: pd.DataFrame) -> np.ndarray:
    module = load_model_module(model)
    preprocess = load_preprocess_module(model)
    matrix = preprocess.transform_frame(preparer, frame)
    return module.predict_model(bundle, matrix)


def save_residual(
    root: Path,
    model: str,
    genome: dict,
    preparer,
    bundle,
    metrics: dict,
) -> Path:
    folder = root / "mean_residual_model" / "D-value_models" / model
    folder.mkdir(parents=True, exist_ok=True)
    load_model_module(model).save_model(bundle, folder)
    joblib.dump(preparer, folder / "preprocessor.joblib")
    (folder / "genome.json").write_text(
        json.dumps(genome, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    payload = {
        "feature_hash": feature_hash(list(genome["features"])),
        "model": model,
        "source_genome": f"models/{model}/best/genome.json",
        **metrics,
    }
    (folder / "metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return folder


def rent_scores(
    rents: dict[str, np.ndarray],
    regional: dict[str, np.ndarray],
    residual: dict[str, np.ndarray],
) -> dict[str, float]:
    """RMSE of the sum, and of the town-month mean alone, on each period."""
    scores = {}
    for period in ("train", "search", "confirm"):
        rent = rents[period]
        level = regional[period]
        gap = residual[period]
        scores[f"{period}_rmse"] = rmse(rent, level + gap)
        scores[f"stage1_only_{period}_rmse"] = rmse(rent, level)
    return scores


def stopped_size(model: str, bundle) -> dict:
    if model == "mlp":
        return {"best_epoch": int(bundle["best_epoch"])}
    if model == "xgb":
        return {"best_iteration": int(bundle.best_iteration)}
    if model == "catboost":
        return {"best_iteration": int(bundle.get_best_iteration())}
    return {}
