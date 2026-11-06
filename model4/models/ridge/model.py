"""Ridge architecture. Alpha is the architecture gene."""

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import Ridge

WEIGHT_NAME = "model.joblib"
TITLE = "Ridge"

ALPHA_FACTOR = 10.0
ALPHA_MIN = 0.001
ALPHA_MAX = 10000.0

MODEL_NOTE = f"""## Architecture gene

The spec is `alpha`. Regularize multiplies alpha by {ALPHA_FACTOR:g} up to {ALPHA_MAX:g}. Capacity divides alpha by {ALPHA_FACTOR:g} down to {ALPHA_MIN:g}.
"""


def fit_model(x_fit, y_fit, x_search, y_search, architecture: dict, seed: int, epochs: int | None = None):
    from src.pipeline.preprocess import assert_fit_inputs

    del seed, epochs
    x_fit, y_fit = assert_fit_inputs(x_fit, y_fit)
    x_search, y_search = assert_fit_inputs(x_search, y_search)
    del x_search, y_search
    model = Ridge(alpha=float(architecture["alpha"]))
    model.fit(x_fit, y_fit)
    return model


def predict_model(bundle, features) -> np.ndarray:
    from src.pipeline.preprocess import assert_prediction

    raw = bundle.predict(np.asarray(features))
    return assert_prediction(raw, len(np.asarray(features)))


def save_model(bundle, directory: Path) -> None:
    joblib.dump(bundle, directory / WEIGHT_NAME)


def load_model(directory: Path):
    return joblib.load(directory / WEIGHT_NAME)


def mutate_architecture(architecture: dict, mode: str) -> dict | None:
    updated = json.loads(json.dumps(architecture))
    alpha = float(updated["alpha"])
    factor = ALPHA_FACTOR if mode == "regularize" else 1.0 / ALPHA_FACTOR
    if mode not in {"regularize", "capacity"}:
        raise ValueError(f"unknown architecture mode: {mode}")
    nxt = min(ALPHA_MAX, max(ALPHA_MIN, alpha * factor))
    nxt = float(f"{nxt:.10g}")
    if nxt == float(f"{alpha:.10g}"):
        return None
    updated["alpha"] = nxt
    return updated
