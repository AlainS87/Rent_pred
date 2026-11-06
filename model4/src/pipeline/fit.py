"""Fit one genome and store its checkpoint under the feature hash."""

import importlib
import json
from pathlib import Path

import joblib
import numpy as np

from src.pipeline.genome import feature_hash, genome_hash, write_features_file
from src.pipeline.paths import MODELS


def load_model_module(model: str):
    if model not in {"mlp", "ridge", "xgb", "catboost"}:
        raise SystemExit(f"unknown model: {model}")
    return importlib.import_module(f"models.{model}.model")


def load_preprocess_module(model: str):
    return importlib.import_module(f"models.{model}.preprocess")


def rmse(y_true, y_pred) -> float:
    truth = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    if truth.shape != pred.shape:
        raise ValueError(f"prediction length {pred.shape} does not match {truth.shape}")
    score = float(np.sqrt(np.mean((truth - pred) ** 2)))
    if not np.isfinite(score):
        raise ValueError("RMSE is not finite")
    return score


def feature_folder(model: str, columns: list[str]) -> Path:
    return MODELS / model / "ckpt" / feature_hash(columns)


def run_folder(model: str, columns: list[str], run_id: str) -> Path:
    return feature_folder(model, columns) / "runs" / run_id


def fit_genome(model: str, genome: dict, fit_frame, search_frame, seed: int):
    """Fit the preprocessor on the fit months and the model on those rows."""
    module = load_model_module(model)
    preprocess = load_preprocess_module(model)
    columns = list(genome["features"])
    y_fit = fit_frame["MONTHLY_RENT"].to_numpy(dtype=np.float64)
    y_search = search_frame["MONTHLY_RENT"].to_numpy(dtype=np.float64)
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
    train_pred = module.predict_model(bundle, x_fit)
    search_pred = module.predict_model(bundle, x_search)
    return {
        "preprocessor": preparer,
        "bundle": bundle,
        "train_rmse": rmse(y_fit, train_pred),
        "search_rmse": rmse(y_search, search_pred),
        "search_pred": search_pred,
        "genome_hash": genome_hash(genome),
        "feature_hash": feature_hash(columns),
    }


def save_run(model: str, genome: dict, trial: dict, record: dict) -> Path:
    folder = run_folder(model, genome["features"], record["run_id"])
    folder.mkdir(parents=True, exist_ok=True)
    write_features_file(folder.parent.parent, genome["features"])
    (folder / "genome.json").write_text(
        json.dumps(genome, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metrics = {
        "accepted": record.get("accepted"),
        "feature_hash": record["feature_hash"],
        "genome_hash": record["genome_hash"],
        "mutation": record.get("mutation"),
        "parent_run_id": record.get("parent_run_id"),
        "run_id": record["run_id"],
        "train_rmse": record["train_rmse"],
    }
    if "search_rmse" in record:
        metrics["search_rmse"] = record["search_rmse"]
    (folder / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    joblib.dump(trial["preprocessor"], folder / "preprocessor.joblib")
    load_model_module(model).save_model(trial["bundle"], folder)
    return folder


def load_run(model: str, columns: list[str], run_id: str):
    folder = run_folder(model, columns, run_id)
    preprocessor = joblib.load(folder / "preprocessor.joblib")
    bundle = load_model_module(model).load_model(folder)
    genome = json.loads((folder / "genome.json").read_text(encoding="utf-8"))
    return preprocessor, bundle, genome


def predict_frame(model: str, preprocessor, bundle, frame, columns: list[str]) -> np.ndarray:
    del columns
    matrix = load_preprocess_module(model).transform_frame(preprocessor, frame)
    prediction = load_model_module(model).predict_model(bundle, matrix)
    from src.pipeline.preprocess import assert_prediction

    return assert_prediction(prediction, len(frame))
