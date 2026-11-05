"""Composition-adjusted month effects plus a static property CatBoost.

The month index is estimated jointly with hedonic controls on allowed history.
It is never replaced or level-corrected by realized forecast-window means.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
import copy

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from catboost import CatBoostRegressor
from threadpoolctl import threadpool_limits

from src.backtest.data import frame_hash
from src.backtest.models import _params, native_frame
from src.backtest.protocol import ROW_ID, TARGET, digest_json, history_hash, horizon, inner_split, months, require_history, require_query
from src.features.group_means import GROUP_MEAN_COLUMNS
from src.pipeline.preprocess import TabularPreprocessor, _building_key, _category_text

TIME_VARYING_AGES = {"building_age_years", "flat_age_from_lease", "building_age_x_central"}


def static_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["building_key"] = _building_key(frame)
    result["town_flat_type"] = _category_text(frame.TOWN, "TOWN") + "|" + _category_text(frame.FLAT_TYPE, "FLAT_TYPE")
    result["floor_area_sq"] = pd.to_numeric(frame.FLOOR_AREA_SQM, errors="coerce") ** 2
    return result


def require_static_columns(columns: list[str]) -> None:
    forbidden = {"RENT_APPROVAL_DATE", TARGET, "BLOCK", *GROUP_MEAN_COLUMNS, *TIME_VARYING_AGES}
    bad = [c for c in columns if c in forbidden or c.startswith(("coe_", "bank_basket_", "property_basket_", "mean_rent_"))]
    if bad or len(columns) != len(set(columns)):
        raise ValueError(f"forbidden or duplicate static property inputs: {bad}")


@dataclass
class HedonicMonths:
    origin: str
    config: dict

    def _numeric(self, frame: pd.DataFrame) -> np.ndarray:
        out = []
        for column in self.config["hedonic_numeric"]:
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
            params = self.numeric_[column]
            filled = np.where(np.isfinite(values), values, params["fill"])
            out.append((filled - params["center"]) / params["spread"])
        return np.column_stack(out)

    def _categories(self, frame: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({c: _category_text(frame[c], c).to_numpy() for c in self.config["hedonic_categorical"]})

    def property_matrix(self, frame: pd.DataFrame):
        prepared = static_frame(frame)
        return sparse.hstack([self.encoder_.transform(self._categories(prepared)), sparse.csr_matrix(self._numeric(prepared))], format="csr")

    def fit(self, history: pd.DataFrame):
        require_history(history, self.origin)
        require_static_columns([*self.config["hedonic_categorical"], *self.config["hedonic_numeric"]])
        data = static_frame(history)
        self.numeric_ = {}
        for column in self.config["hedonic_numeric"]:
            raw = pd.to_numeric(data[column], errors="coerce").to_numpy(dtype=float)
            finite = raw[np.isfinite(raw)]
            fill = float(np.median(finite)) if len(finite) else 0.0
            values = np.where(np.isfinite(raw), raw, fill)
            self.numeric_[column] = {"fill": fill, "center": float(values.mean()), "spread": float(values.std()) or 1.0}
        self.encoder_ = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float64)
        self.encoder_.fit(self._categories(data))
        prop = self.property_matrix(history)
        self.n_property_ = prop.shape[1]
        self.month_encoder_ = OneHotEncoder(handle_unknown="error", sparse_output=True, dtype=np.float64)
        month_text = np.asarray(months(history).astype(str)).reshape(-1, 1)
        temporal = self.month_encoder_.fit_transform(month_text)
        design = sparse.hstack([prop, temporal], format="csr")
        self.ridge_ = Ridge(alpha=self.config["ridge_alpha"], solver=self.config["ridge_solver"],
                            tol=self.config["ridge_tol"], max_iter=self.config["ridge_max_iter"], fit_intercept=True)
        with threadpool_limits(limits=1):
            self.ridge_.fit(design, history[TARGET].to_numpy(dtype=float))
        labels = self.month_encoder_.categories_[0]
        counts = pd.Series(months(history).astype(str)).value_counts().reindex(labels).to_numpy(dtype=float)
        raw_effect = np.asarray(self.ridge_.coef_[self.n_property_:], dtype=float)
        self.center_shift_ = float(np.average(raw_effect, weights=counts))
        self.property_intercept_ = float(self.ridge_.intercept_) + self.center_shift_
        self.index_ = pd.DataFrame({"month": labels, "count": counts, "raw_effect": raw_effect,
                                    "market_effect": raw_effect - self.center_shift_})
        self.allowed_history_sha256_ = history_hash(history)
        original = self.ridge_.predict(design)
        reconstructed = self.property_predict(history) + self.historical_market(history)
        self.reconstruction_max_abs_error_ = float(np.max(np.abs(original - reconstructed)))
        if not np.isfinite(reconstructed).all() or self.reconstruction_max_abs_error_ > 1e-7:
            raise ValueError("month-effect centering changed the fitted hedonic representation")
        return self

    def historical_market(self, history: pd.DataFrame) -> np.ndarray:
        values = self.index_.set_index("month").market_effect
        result = values.reindex(months(history).astype(str)).to_numpy(dtype=float)
        if not np.isfinite(result).all():
            raise ValueError("historical month effects requested outside fitted history")
        return result

    def property_predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.property_matrix(frame) @ self.ridge_.coef_[:self.n_property_]).reshape(-1) + self.property_intercept_

    def state(self) -> dict:
        return {"origin": self.origin, "config": self.config, "numeric": self.numeric_,
                "categories": [items.tolist() for items in self.encoder_.categories_],
                "property_coefficients": self.ridge_.coef_[:self.n_property_].tolist(),
                "raw_intercept": float(self.ridge_.intercept_), "center_shift": self.center_shift_,
                "property_intercept": self.property_intercept_, "index": self.index_.to_dict(orient="records"),
                "allowed_history_sha256": self.allowed_history_sha256_,
                "reconstruction_max_abs_error": self.reconstruction_max_abs_error_,
                "penalty": "alpha times squared L2 norm of all hedonic and raw month coefficients; intercept unpenalized"}


@dataclass
class MarketTrend:
    name: str
    origin: str
    options: dict

    def fit(self, index: pd.DataFrame):
        if self.name not in {"flat", "damped"}:
            raise ValueError("trend must be one of the two predeclared candidates")
        work = index.sort_values("month").tail(self.options["window"])
        if work.empty or not np.isfinite(work[["market_effect", "count"]].to_numpy()).all() or (work["count"] <= 0).any():
            raise ValueError("market index and transaction weights must be finite and nonempty")
        periods = pd.PeriodIndex(work.month, freq="M")
        if (periods > pd.Period(self.origin, freq="M")).any():
            raise ValueError("trend contains future month effects")
        x = periods.asi8 - pd.Period(self.origin, freq="M").ordinal
        y, weight = work.market_effect.to_numpy(dtype=float), work["count"].to_numpy(dtype=float)
        self.slope_ = 0.0
        self.level_ = float(np.average(y, weights=weight))
        if self.name == "damped":
            phi = self.options["phi"]
            if not 0 <= phi < 1:
                raise ValueError("damping phi must be in [0, 1)")
            mean_x = float(np.average(x, weights=weight))
            denominator = float(np.sum(weight * (x - mean_x) ** 2))
            self.slope_ = float(np.sum(weight * (x - mean_x) * (y - self.level_)) / denominator) if denominator > 0 else 0.0
            self.level_ -= self.slope_ * mean_x
        self.months_used_ = work.month.astype(str).tolist()
        return self

    def predict_horizon(self, offsets: np.ndarray) -> np.ndarray:
        offsets = np.asarray(offsets)
        if not np.issubdtype(offsets.dtype, np.integer) or (offsets < 1).any() or (offsets > 16).any():
            raise ValueError("trend forecast horizon must be an integer in 1..16")
        if self.name == "flat":
            result = np.full(len(offsets), self.level_, dtype=float)
        else:
            phi = self.options["phi"]
            increment = phi * (1 - phi ** offsets) / (1 - phi)
            result = self.level_ + self.slope_ * increment
        if not np.isfinite(result).all():
            raise ValueError("non-finite market forecast")
        return result

    def state(self) -> dict:
        return {"name": self.name, "origin": self.origin, "options": self.options,
                "level_at_origin": self.level_, "slope": self.slope_, "months_used": self.months_used_}


def prepare_property(history: pd.DataFrame, query: pd.DataFrame, adjusted_targets: np.ndarray, config: dict):
    columns = list(config["decomposed"]["property_features"])
    require_static_columns(columns)
    train, future = static_frame(history), static_frame(query)
    prep = TabularPreprocessor(config["preprocess"])
    prep.fit(train, adjusted_targets, columns)
    x_train, x_query = prep.transform(train), prep.transform(future)
    audit = {"feature_columns": columns, "numeric_fill": prep.numeric_fill_, "categories": prep.levels_,
             "training_matrix_sha256": frame_hash(x_train), "query_matrix_sha256": frame_hash(x_query)}
    return x_train, x_query, prep, audit


@dataclass
class DecomposedBundle:
    model: CatBoostRegressor
    preprocessor: TabularPreprocessor
    index: HedonicMonths
    trend: MarketTrend
    origin: str
    config: dict
    audit: dict
    timings: dict

    def predict_components(self, query: pd.DataFrame) -> pd.DataFrame:
        require_query(query, self.origin, self.config["max_horizon"])
        if any(c in query for c in GROUP_MEAN_COLUMNS):
            raise ValueError("precomputed target-derived means are forbidden inputs")
        features, _ = native_frame(self.preprocessor.transform(static_frame(query)))
        prop = np.asarray(self.model.predict(features), dtype=float)
        market = self.trend.predict_horizon(horizon(query, self.origin))
        raw = prop + market
        result = pd.DataFrame({ROW_ID: query[ROW_ID].to_numpy(), "property_component": prop,
                               "market_component": market, "unclipped_prediction": raw,
                               "prediction": np.maximum(self.config["prediction_floor"], raw)}, index=query.index)
        if not np.isfinite(result.to_numpy(dtype=float)).all():
            raise ValueError("non-finite decomposed prediction")
        return result

    def predict(self, query: pd.DataFrame) -> np.ndarray:
        return self.predict_components(query).prediction.to_numpy()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)


def fit_decomposed(history: pd.DataFrame, query: pd.DataFrame, origin: str, config: dict,
                   trend_names: list[str] | None = None, on_complete=None) -> dict[str, tuple[DecomposedBundle, pd.DataFrame]]:
    config = copy.deepcopy(config)
    require_history(history, origin)
    require_query(query, origin, config["max_horizon"])
    if any(c in history or c in query for c in GROUP_MEAN_COLUMNS):
        raise ValueError("precomputed target-derived means are forbidden inputs")
    if set(history[ROW_ID]) & set(query[ROW_ID]):
        raise ValueError("history/query row identities overlap")
    history = history.reset_index(drop=True).sort_values(ROW_ID).copy()
    names = list(config["decomposed"]["trends"]) if trend_names is None else list(trend_names)
    if not names or len(names) != len(set(names)) or any(name not in config["decomposed"]["trends"] for name in names):
        raise ValueError("undeclared trend candidate")
    started = time.perf_counter()
    train, valid_x, valid_y, cutoff = inner_split(history, origin, config)
    inner_index = HedonicMonths(cutoff, config["decomposed"]).fit(train)
    inner_y = train[TARGET].to_numpy(dtype=float) - inner_index.historical_market(train)
    inner_train, inner_valid, _, inner_preparation = prepare_property(train, valid_x, inner_y, config)
    inner_train, cats = native_frame(inner_train)
    inner_valid, _ = native_frame(inner_valid)
    inner_shared_seconds = time.perf_counter() - started
    choices, inner_audits, inner_seconds = {}, {}, {}
    for name in names:
        begin = time.perf_counter()
        trend = MarketTrend(name, cutoff, config["decomposed"]["trends"][name]).fit(inner_index.index_)
        # The validation target is used only here, after all inner features/indexes.
        validation_residual = valid_y - trend.predict_horizon(horizon(valid_x, cutoff))
        model = CatBoostRegressor(**_params(config, config["catboost"]["iterations"]),
                                  early_stopping_rounds=config["catboost"]["early_stopping_rounds"])
        model.fit(inner_train, inner_y, cat_features=cats, eval_set=(inner_valid, validation_residual),
                  use_best_model=True, verbose=False)
        choices[name] = max(1, int(model.get_best_iteration()) + 1)
        inner_audits[name] = {"boundary": cutoff, "allowed_history_sha256": history_hash(train),
                              "index_sha256": digest_json(inner_index.state()), "property": inner_preparation,
                              "trend": trend.state(), "selected_iterations": choices[name],
                              "executed_iterations": len(model.get_evals_result()["validation"]["RMSE"])}
        inner_seconds[name] = time.perf_counter() - begin
        del model
    begin = time.perf_counter()
    outer_index = HedonicMonths(origin, config["decomposed"]).fit(history)
    outer_y = history[TARGET].to_numpy(dtype=float) - outer_index.historical_market(history)
    train_x, _, prep, outer_preparation = prepare_property(history, query, outer_y, config)
    train_x, cats = native_frame(train_x)
    outer_shared_seconds = time.perf_counter() - begin
    refitted = {}
    result = {}
    for name in names:
        begin = time.perf_counter()
        iterations = choices[name]
        reused = iterations in refitted
        # Exactly identical fit data, target, seed, and all CatBoost parameters.
        # Sharing is limited to equal frozen iteration counts, never a prefix approximation.
        if not reused:
            model = CatBoostRegressor(**_params(config, iterations))
            model.fit(train_x, outer_y, cat_features=cats, verbose=False)
            refitted[iterations] = model
        trend = MarketTrend(name, origin, config["decomposed"]["trends"][name]).fit(outer_index.index_)
        audit = {"model": f"decomposed_{name}", "origin": origin, "config_sha256": digest_json(config),
                 "allowed_history_sha256": history_hash(history), "selected_iterations": iterations,
                 "inner": inner_audits[name], "outer": {"index": outer_index.state(), "property": outer_preparation,
                 "trend": trend.state()}, "outer_early_stopping": False}
        bundle = DecomposedBundle(refitted[iterations], prep, outer_index, trend, origin, config, audit,
                                   {"inner_shared_seconds": inner_shared_seconds, "inner_select_seconds": inner_seconds[name],
                                    "outer_shared_seconds": outer_shared_seconds, "outer_refit_seconds": time.perf_counter() - begin,
                                    "outer_refit_reused_identical_fit": reused})
        components = bundle.predict_components(query)
        result[name] = bundle, components
        if on_complete is not None:
            on_complete(name, bundle, components)
    return result
