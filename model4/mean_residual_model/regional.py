"""Ridge of the town-month mean rent.

Fit on one row per town and approval month through 2024-09. A flat receives
that cell's prediction. Numeric inputs are the month index and three lagged
levels, centered on the fit cells.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder

FIT_END = "2024-09"
SEARCH_MONTHS = ("2024-10", "2024-11", "2024-12")
CONFIRM_MONTHS = ("2025-01", "2025-02", "2025-03")
LEVELS = [
    "coe_c_premium_lag1",
    "bank_basket_adj_close_lag1",
    "property_basket_adj_close_lag1",
]
NUMERIC = ["month_index", *LEVELS]
EXPECTED_CELL_RMSE = {
    "search": 93.35753474370192,
    "confirm": 103.31438306463845,
}
ALPHA = 1.0


class TownMonthMean:
    """Unweighted ridge of town-month mean rent. ``alpha`` is 1."""

    def __init__(self) -> None:
        self.model: Ridge | None = None
        self.encoder: OneHotEncoder | None = None
        self.numeric: dict[str, dict[str, float]] | None = None
        self.fit_cells: int = 0

    def fit(self, frame: pd.DataFrame) -> "TownMonthMean":
        cells = cell_table(frame)
        self.numeric = _numeric_params(cells, NUMERIC)
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        area = self.encoder.fit_transform(cells[["TOWN"]].astype(str))
        train_x = np.column_stack([_numeric_block(cells, NUMERIC, self.numeric), area])
        self.model = Ridge(alpha=ALPHA)
        self.model.fit(train_x, cells["mean_rent"].to_numpy(dtype=np.float64))
        self.fit_cells = int(len(cells))
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """One prediction per row. Rows in one town-month share a value."""
        prepared = with_month(frame)
        return self._predict_table(prepared)

    def weighted_cell_rmse(self, frame: pd.DataFrame, months: tuple[str, ...]) -> float:
        prepared = with_month(frame)
        prepared = prepared[prepared["month"].isin(months)]
        known = set(self.encoder.categories_[0].astype(str))
        prepared = prepared[prepared["TOWN"].astype(str).isin(known)]
        cells = cell_table(prepared)
        prediction = self._predict_table(cells)
        return _weighted_rmse(
            cells["mean_rent"].to_numpy(dtype=np.float64),
            prediction,
            cells["rows"].to_numpy(dtype=np.float64),
        )

    def _predict_table(self, table: pd.DataFrame) -> np.ndarray:
        if self.model is None or self.encoder is None or self.numeric is None:
            raise RuntimeError("town-month ridge is not fit")
        area = self.encoder.transform(table[["TOWN"]].astype(str))
        features = np.column_stack([_numeric_block(table, NUMERIC, self.numeric), area])
        return np.asarray(self.model.predict(features), dtype=np.float64)


def save_regional(model: TownMonthMean, directory: Path, scores: dict[str, float]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "alpha": ALPHA,
            "encoder": model.encoder,
            "fit_cells": model.fit_cells,
            "model": model.model,
            "numeric": model.numeric,
        },
        directory / "model.joblib",
    )
    payload = {
        "alpha": ALPHA,
        "confirm_weighted_cell_rmse": scores["confirm"],
        "features": ["TOWN", *NUMERIC],
        "fit_cells": model.fit_cells,
        "fit_end": FIT_END,
        "search_weighted_cell_rmse": scores["search"],
    }
    (directory / "metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_regional(directory: Path) -> TownMonthMean:
    payload = joblib.load(directory / "model.joblib")
    model = TownMonthMean()
    model.model = payload["model"]
    model.encoder = payload["encoder"]
    model.numeric = payload["numeric"]
    model.fit_cells = int(payload["fit_cells"])
    return model


def assert_known_cell_scores(scores: dict[str, float]) -> None:
    """The saved ridge must reproduce the feasibility cell scores."""
    for name, expected in EXPECTED_CELL_RMSE.items():
        got = float(scores[name])
        if abs(got - expected) > 0.01:
            raise SystemExit(
                f"town-month {name} cell RMSE is {got:.4f}, expected {expected:.4f}"
            )


def cell_table(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = with_month(frame)
    grouped = prepared.groupby(["TOWN", "month"], observed=True)
    cells = grouped["MONTHLY_RENT"].agg(mean_rent="mean", rows="size").reset_index()
    month_index = prepared.groupby("month")["month_index"].first()
    cells["month_index"] = cells["month"].map(month_index)
    for column in LEVELS:
        values = prepared.groupby("month")[column].first()
        cells[column] = cells["month"].map(values)
    return cells


def with_month(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["month"] = work["RENT_APPROVAL_DATE"].astype("string").str.slice(0, 7)
    work["month_index"] = _month_index(work["month"])
    return work


def _month_index(month: pd.Series) -> pd.Series:
    year = month.str.slice(0, 4).astype(int)
    part = month.str.slice(5, 7).astype(int)
    return (year - 2021) * 12 + (part - 1)


def _numeric_params(train: pd.DataFrame, columns: list[str]) -> dict[str, dict[str, float]]:
    params = {}
    for column in columns:
        series = train[column].astype(float)
        center = float(series.median())
        spread = float(series.std(ddof=0)) or 1.0
        params[column] = {"center": center, "fill": center, "spread": spread}
    return params


def _numeric_block(
    frame: pd.DataFrame,
    columns: list[str],
    params: dict[str, dict[str, float]],
) -> np.ndarray:
    values = []
    for column in columns:
        spec = params[column]
        filled = frame[column].astype(float).fillna(spec["fill"]).to_numpy(dtype=np.float64)
        center = spec["center"]
        spread = spec["spread"]
        values.append((np.where(np.isfinite(filled), filled, center) - center) / spread)
    return np.column_stack(values)


def _weighted_rmse(truth: np.ndarray, prediction: np.ndarray, weights: np.ndarray) -> float:
    keep = np.isfinite(truth) & np.isfinite(prediction) & np.isfinite(weights)
    error = truth[keep] - prediction[keep]
    weight = weights[keep]
    return float(np.sqrt(np.average(error**2, weights=weight)))
