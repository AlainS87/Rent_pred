"""Strict existing-method comparators and a transparent fixed-level baseline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from src.backtest.features import PreparedFeatures, prepare_features
from src.backtest.protocol import ROW_ID, TARGET, digest_json, history_hash, inner_split, months, require_history, require_query
from src.features.group_means import GROUP_MEAN_COLUMNS
from src.pipeline.preprocess import _category_text

MEAN_HISTORY_COLUMNS = [ROW_ID, "RENT_APPROVAL_DATE", TARGET, "postal_code", "nearest_mall_name",
                        "nearest_open_mrt_code", "nearest_primary_name", "nearest_secondary_name"]


def feature_columns(model: str, config: dict) -> list[str]:
    columns = list(config["direct_features"][model])
    if not columns or len(columns) != len(set(columns)) or TARGET in columns:
        raise ValueError("invalid frozen direct-model feature list")
    return columns


def native_frame(frame: pd.DataFrame):
    result = frame.copy()
    cats = []
    for column in frame:
        if isinstance(frame[column].dtype, pd.CategoricalDtype):
            cats.append(column)
            result[column] = frame[column].astype(object).fillna("__missing__").astype(str)
        elif not np.isfinite(frame[column].to_numpy(dtype=float)).all():
            raise ValueError(f"non-finite model feature: {column}")
    return result, cats


def _params(config: dict, iterations: int) -> dict:
    opts = config["catboost"]
    return {"depth": opts["depth"], "learning_rate": opts["learning_rate"],
            "iterations": iterations, "thread_count": opts["thread_count"],
            "random_seed": config["seed"], "loss_function": "RMSE", "allow_writing_files": False}


def select_iterations(train: pd.DataFrame, validation_features: pd.DataFrame,
                      validation_targets: np.ndarray, cutoff: str, columns: list[str], config: dict):
    """Validation targets are separate, and can affect stopping only."""
    labels = np.asarray(validation_targets, dtype=float)
    if labels.shape != (len(validation_features),) or not np.isfinite(labels).all():
        raise ValueError("invalid inner-validation targets")
    x_train, x_valid, _, audit = prepare_features(train, validation_features, cutoff, columns, config)
    x_train, cats = native_frame(x_train)
    x_valid, _ = native_frame(x_valid)
    params = _params(config, config["catboost"]["iterations"])
    params["early_stopping_rounds"] = config["catboost"]["early_stopping_rounds"]
    model = CatBoostRegressor(**params)
    model.fit(x_train, train[TARGET].to_numpy(dtype=float), cat_features=cats,
              eval_set=(x_valid, labels), use_best_model=True, verbose=False)
    selected = max(1, int(model.get_best_iteration()) + 1)
    audit["selected_iterations"] = selected
    audit["executed_iterations"] = len(model.get_evals_result()["validation"]["RMSE"])
    audit["validation_targets_role"] = "early stopping only; excluded from feature generation and preprocessing"
    return selected, audit


@dataclass
class DirectBundle:
    model: CatBoostRegressor
    features: PreparedFeatures
    mean_history: pd.DataFrame
    origin: str
    floor: float
    audit: dict
    timings: dict | None = None

    def predict(self, query: pd.DataFrame) -> np.ndarray:
        require_query(query, self.origin, self.features.max_horizon)
        if any(c in query for c in GROUP_MEAN_COLUMNS):
            raise ValueError("precomputed target-derived means are forbidden inputs")
        matrix = self.features.transform(query, self.mean_history)
        matrix, _ = native_frame(matrix)
        result = np.maximum(self.floor, np.asarray(self.model.predict(matrix), dtype=float))
        if result.shape != (len(query),) or not np.isfinite(result).all():
            raise ValueError("invalid finite row-aligned prediction")
        return result

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)


def load_bundle(path: Path) -> DirectBundle:
    result = joblib.load(path)
    if not isinstance(result, DirectBundle):
        raise TypeError("artifact is not a strict direct-model bundle")
    return result


def fit_direct(history: pd.DataFrame, query: pd.DataFrame, origin: str, model_name: str,
               config: dict) -> tuple[DirectBundle, np.ndarray]:
    require_history(history, origin)
    require_query(query, origin, config["max_horizon"])
    if set(history[ROW_ID]) & set(query[ROW_ID]):
        raise ValueError("history/query row identities overlap")
    # Canonical fit order prevents input reordering from changing OOF or model fit.
    history = history.reset_index(drop=True).sort_values(ROW_ID).copy()
    train, valid_x, valid_y, cutoff = inner_split(history, origin, config)
    columns = feature_columns(model_name, config)
    started = time.perf_counter()
    selected, inner_audit = select_iterations(train, valid_x, valid_y, cutoff, columns, config)
    after_inner = time.perf_counter()
    x_train, x_query, features, outer_audit = prepare_features(history, query, origin, columns, config)
    matrix, cats = native_frame(x_train)
    after_preparation = time.perf_counter()
    model = CatBoostRegressor(**_params(config, selected))
    # No outer eval_set. The chosen iteration count is frozen before this refit.
    model.fit(matrix, history[TARGET].to_numpy(dtype=float), cat_features=cats, verbose=False)
    after_refit = time.perf_counter()
    audit = {"model": model_name, "origin": origin, "selected_iterations": selected,
             "feature_columns": columns, "config_sha256": digest_json(config),
             "allowed_history_sha256": history_hash(history),
             "inner": inner_audit, "outer": outer_audit, "outer_early_stopping": False}
    bundle = DirectBundle(model, features, history[MEAN_HISTORY_COLUMNS].copy(), origin,
                          config["prediction_floor"], audit,
                          {"inner_prepare_and_select_seconds": after_inner - started,
                           "outer_prepare_seconds": after_preparation - after_inner,
                           "outer_refit_seconds": after_refit - after_preparation})
    query_matrix, _ = native_frame(x_query)
    prediction = np.maximum(bundle.floor, np.asarray(model.predict(query_matrix), dtype=float))
    if not np.isfinite(prediction).all():
        raise ValueError("model produced non-finite predictions")
    return bundle, prediction


def fit_simple(history: pd.DataFrame, query: pd.DataFrame, origin: str, config: dict) -> np.ndarray:
    require_history(history, origin)
    require_query(query, origin, config["max_horizon"])
    period = pd.Period(origin, freq="M")
    count = config["simple_baseline"]["recent_months"]
    recent = history.loc[months(history) > period - count]
    if recent.empty:
        raise ValueError("simple baseline has no history in recent window")
    strength = config["simple_baseline"]["strength"]
    prior = float(recent[TARGET].mean())
    recent = recent.copy()
    keys_frame = query[["TOWN", "FLAT_TYPE"]].copy()
    for column in ["TOWN", "FLAT_TYPE"]:
        recent[column] = _category_text(recent[column], column)
        keys_frame[column] = _category_text(keys_frame[column], column)
    stats = recent.groupby(["TOWN", "FLAT_TYPE"], observed=True)[TARGET].agg(["sum", "count"])
    values = (stats["sum"] + strength * prior) / (stats["count"] + strength)
    keys = pd.MultiIndex.from_frame(keys_frame)
    return np.maximum(config["prediction_floor"], values.reindex(keys).fillna(prior).to_numpy(dtype=float))
