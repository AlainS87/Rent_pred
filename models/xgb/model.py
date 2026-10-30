"""XGBoost architecture. Tree depth is the step that evolve changes."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

WEIGHT_NAME = "model.joblib"
TITLE = "XGBoost"

DEPTH_STEP = 2
DEPTH_MIN = 2
DEPTH_MAX = 10

MODEL_NOTE = f"""## Architecture gene

The spec is `max_depth` and `learning_rate`. Nominal columns other than the fixed date, block, and furnished overrides are native categories, so one split can place any subset of the labels on the left. Regularize lowers `max_depth` by {DEPTH_STEP} down to {DEPTH_MIN}. Capacity raises `max_depth` by {DEPTH_STEP} up to {DEPTH_MAX}.
"""


def fit_model(x_fit, y_fit, x_search, y_search, architecture: dict, seed: int, epochs: int | None = None):
    x_fit, y_fit = _prepare_pair(x_fit, y_fit)
    x_search, y_search = _prepare_pair(x_search, y_search)
    trees = int(epochs if epochs is not None else architecture.get("n_estimators", 200))
    kwargs = {
        "max_depth": int(architecture["max_depth"]),
        "learning_rate": float(architecture["learning_rate"]),
        "n_estimators": trees,
        "objective": "reg:squarederror",
        "random_state": seed,
        "n_jobs": 1,
        "tree_method": "hist",
        "enable_categorical": True,
    }
    stopping = int(architecture.get("early_stopping_rounds", 20))
    if trees > 20 and stopping > 0:
        kwargs["early_stopping_rounds"] = stopping
    model = XGBRegressor(**kwargs)
    if "early_stopping_rounds" in kwargs:
        model.fit(
            x_fit,
            y_fit,
            eval_set=[(x_search, y_search)],
            verbose=False,
        )
    else:
        model.fit(x_fit, y_fit, verbose=False)
    return model


def predict_model(bundle, features) -> np.ndarray:
    from src.pipeline.preprocess import assert_prediction

    raw = bundle.predict(features if isinstance(features, pd.DataFrame) else np.asarray(features))
    return assert_prediction(raw, len(features))


def _prepare_pair(features, target):
    from src.pipeline.preprocess import assert_fit_inputs

    if isinstance(features, pd.DataFrame):
        rents = np.asarray(target, dtype=np.float64).reshape(-1)
        if len(features) != len(rents):
            raise ValueError(f"XGBoost frame has {len(features)} rows, expected {len(rents)}")
        return features, rents
    return assert_fit_inputs(features, target)


def save_model(bundle, directory: Path) -> None:
    joblib.dump(bundle, directory / WEIGHT_NAME)


def load_model(directory: Path):
    return joblib.load(directory / WEIGHT_NAME)


def mutate_architecture(architecture: dict, mode: str) -> dict | None:
    updated = json.loads(json.dumps(architecture))
    depth = int(updated["max_depth"])
    step = -DEPTH_STEP if mode == "regularize" else DEPTH_STEP
    if mode not in {"regularize", "capacity"}:
        raise ValueError(f"unknown architecture mode: {mode}")
    nxt = min(DEPTH_MAX, max(DEPTH_MIN, depth + step))
    if nxt == depth:
        return None
    updated["max_depth"] = nxt
    return updated
