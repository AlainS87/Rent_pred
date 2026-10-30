"""CatBoost architecture. Tree depth is the step that evolve changes."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

WEIGHT_NAME = "model.joblib"
TITLE = "CatBoost"
MISSING_CATEGORY = "__missing__"

DEPTH_STEP = 2
DEPTH_MIN = 2
DEPTH_MAX = 10

MODEL_NOTE = f"""## Architecture gene

The spec is `max_depth` and `learning_rate`. Nominal columns other than the fixed date, block, and furnished overrides are native categories. Regularize lowers `max_depth` by {DEPTH_STEP} down to {DEPTH_MIN}. Capacity raises `max_depth` by {DEPTH_STEP} up to {DEPTH_MAX}.
"""


def fit_model(x_fit, y_fit, x_search, y_search, architecture: dict, seed: int, epochs: int | None = None):
    x_fit, y_fit = _prepare_pair(x_fit, y_fit)
    x_search, y_search = _prepare_pair(x_search, y_search)
    categories = _category_columns(x_fit)
    x_fit = _fill_missing_categories(x_fit)
    x_search = _fill_missing_categories(x_search)
    trees = int(epochs if epochs is not None else architecture.get("n_estimators", 200))
    stopping = int(architecture.get("early_stopping_rounds", 20))
    params = {
        "depth": int(architecture["max_depth"]),
        "learning_rate": float(architecture["learning_rate"]),
        "iterations": trees,
        "loss_function": "RMSE",
        "random_seed": seed,
        "thread_count": 1,
        "allow_writing_files": False,
    }
    use_stopping = trees > 20 and stopping > 0
    if use_stopping:
        params["early_stopping_rounds"] = stopping
    model = CatBoostRegressor(**params)
    if use_stopping:
        model.fit(
            x_fit,
            y_fit,
            eval_set=(x_search, y_search),
            cat_features=categories,
            use_best_model=True,
            verbose=False,
        )
    else:
        model.fit(x_fit, y_fit, cat_features=categories, verbose=False)
    return model


def predict_model(bundle, features) -> np.ndarray:
    from src.pipeline.preprocess import assert_prediction

    frame = features if isinstance(features, pd.DataFrame) else np.asarray(features)
    frame = _fill_missing_categories(frame)
    raw = bundle.predict(frame)
    return assert_prediction(raw, len(frame))


def _fill_missing_categories(features):
    """CatBoost rejects NaN in a category column. Unseen levels become one token."""
    if not isinstance(features, pd.DataFrame):
        return features
    frame = features.copy()
    for column in frame.columns:
        if isinstance(frame[column].dtype, pd.CategoricalDtype):
            frame[column] = frame[column].astype("string").fillna(MISSING_CATEGORY)
    return frame


def _category_columns(features) -> list[str] | None:
    if not isinstance(features, pd.DataFrame):
        return None
    names = [
        column
        for column in features.columns
        if isinstance(features[column].dtype, pd.CategoricalDtype)
    ]
    return names or None


def _prepare_pair(features, target):
    from src.pipeline.preprocess import assert_fit_inputs

    if isinstance(features, pd.DataFrame):
        rents = np.asarray(target, dtype=np.float64).reshape(-1)
        if len(features) != len(rents):
            raise ValueError(f"CatBoost frame has {len(features)} rows, expected {len(rents)}")
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
