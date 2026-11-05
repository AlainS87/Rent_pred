"""Cutoff-safe rent means, stable OOF building encodings, and native inputs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.backtest.data import frame_hash
from src.backtest.protocol import ROW_ID, TARGET, digest_json, history_hash, require_history, require_query
from src.features.group_means import GROUP_MEAN_COLUMNS, add_group_mean_columns
from src.pipeline.preprocess import TabularPreprocessor, _building_key


def stable_folds(row_ids, seed: int = 5228, folds: int = 5) -> np.ndarray:
    """SplitMix64 of source row ID + seed, with no dependence on labels/order."""
    values = np.asarray(row_ids, dtype=np.uint64) + np.uint64(seed)
    values = (values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    values = (values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    values = values ^ (values >> np.uint64(31))
    return (values % np.uint64(folds)).astype(np.int64)


@dataclass
class OOFBuildingEncoder:
    strength: float = 20.0
    folds: int = 5
    seed: int = 5228
    empty_prior: float = 0.0

    def fit_transform(self, history: pd.DataFrame) -> np.ndarray:
        keys = _building_key(history).astype(str).to_numpy()
        targets = history[TARGET].to_numpy(dtype=float)
        fold = stable_folds(history[ROW_ID], self.seed, self.folds)
        table = pd.DataFrame({"key": keys, "fold": fold, "y": targets})
        group = table.groupby("key", sort=True).y.agg(["sum", "count"])
        excluded = table.groupby(["key", "fold"], sort=True).y.agg(["sum", "count"])
        fold_stats = table.groupby("fold", sort=True).y.agg(["sum", "count"])
        total_sum, total_n = float(targets.sum()), len(targets)
        index = pd.MultiIndex.from_arrays([keys, fold], names=["key", "fold"])
        group_sum = group["sum"].reindex(keys).to_numpy() - excluded["sum"].reindex(index).to_numpy()
        group_n = group["count"].reindex(keys).to_numpy() - excluded["count"].reindex(index).to_numpy()
        prior_sum = total_sum - fold_stats["sum"].reindex(fold).to_numpy()
        prior_n = total_n - fold_stats["count"].reindex(fold).to_numpy()
        prior = np.divide(prior_sum, prior_n, out=np.full(len(targets), self.empty_prior), where=prior_n > 0)
        encoded = (group_sum + self.strength * prior) / (group_n + self.strength)
        self.default_ = float(targets.mean()) if total_n else self.empty_prior
        self.mapping_ = ((group["sum"] + self.strength * self.default_) / (group["count"] + self.strength)).to_dict()
        return encoded

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        return _building_key(frame).map(self.mapping_).fillna(self.default_).to_numpy(dtype=float)

    def state(self) -> dict:
        return {"default": self.default_, "mapping": self.mapping_, "strength": self.strength,
                "folds": self.folds, "seed": self.seed, "empty_prior": self.empty_prior}


@dataclass
class PreparedFeatures:
    preprocessor: TabularPreprocessor
    encoder: OOFBuildingEncoder | None
    columns: list[str]
    boundary: str
    max_horizon: int

    def transform(self, query: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
        require_history(history, self.boundary)
        require_query(query, self.boundary, self.max_horizon)
        safe = query.copy()
        if any(c in self.columns for c in GROUP_MEAN_COLUMNS):
            safe = add_group_mean_columns(safe, history)
        matrix = self.preprocessor.transform(safe)
        if self.encoder is not None:
            matrix["BLOCK__building"] = self.encoder.transform(query)
        return matrix

    def state(self) -> dict:
        prep = self.preprocessor
        return {
            "columns": self.columns, "boundary": self.boundary, "max_horizon": self.max_horizon,
            "numeric_fill": prep.numeric_fill_, "numeric_center": prep.numeric_center_,
            "numeric_spread": prep.numeric_spread_, "categories": prep.levels_,
            "building": self.encoder.state() if self.encoder else None,
        }


def prepare_features(history: pd.DataFrame, query: pd.DataFrame, origin: str,
                     columns: list[str], config: dict):
    """Both history/query are physically bounded before entering this function."""
    require_history(history, origin)
    require_query(query, origin, config["max_horizon"])
    if any(c in history or c in query for c in GROUP_MEAN_COLUMNS):
        raise ValueError("precomputed target-derived means are forbidden inputs")
    train = history.copy()
    future = query.copy()
    needs_means = any(c in columns for c in GROUP_MEAN_COLUMNS)
    if needs_means:
        train = add_group_mean_columns(train, history)
        future = add_group_mean_columns(future, history)
    # Remove BLOCK from the legacy preprocessor to disable its self-label map.
    source_columns = [c for c in columns if c != "BLOCK"]
    prep = TabularPreprocessor(config["preprocess"])
    prep.fit(train, history[TARGET].to_numpy(dtype=float), source_columns)
    train_x, query_x = prep.transform(train), prep.transform(future)
    encoder = None
    if "BLOCK" in columns:
        opts = dict(config["building_encoding"], seed=config["seed"])
        encoder = OOFBuildingEncoder(**opts)
        train_x["BLOCK__building"] = encoder.fit_transform(history)
        query_x["BLOCK__building"] = encoder.transform(query)
    fitted = PreparedFeatures(prep, encoder, list(columns), origin, config["max_horizon"])
    audit = {
        "boundary": origin, "training_rows": len(history), "query_rows": len(query),
        "training_row_ids_sha256": frame_hash(history[[ROW_ID]]),
        "query_row_ids_sha256": frame_hash(query[[ROW_ID]]),
        "training_matrix_sha256": frame_hash(train_x), "query_matrix_sha256": frame_hash(query_x),
        "preprocessor_sha256": digest_json(fitted.state()),
        "allowed_history_sha256": history_hash(history),
        "means_all_six_rebuilt": needs_means,
        "oof_warning": "OOF regularization within permitted fit history, not row-time causal availability",
    }
    return train_x, query_x, fitted, audit
