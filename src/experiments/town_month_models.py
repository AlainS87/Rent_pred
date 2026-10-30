"""Compare estimators for the town-month mean rent.

Same inputs as the chosen feasibility check: town, a month index, and the
three lagged level series. Fit months end at 2024-09. Search is 2024-10..12.
Confirm is 2025-01..03 and is not used to fit or to stop a model.

    python -m src.experiments.town_month_models
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.linear_model import ElasticNet, HuberRegressor, LinearRegression, Ridge
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from src.experiments.partition_month_mean import (
    CONFIRM,
    FIT_END,
    SEARCH,
    _cell_table,
    _load,
    _month_index,
    _numeric_block,
    _weighted_rmse,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "Data"
OUT = ROOT / "outputs" / "feasibility" / "town_month_models.json"
LEVELS = [
    "coe_c_premium_lag1",
    "bank_basket_adj_close_lag1",
    "property_basket_adj_close_lag1",
]
SEED = 5228


def main() -> None:
    frame = _load()
    work = frame[frame["TOWN"].notna()].copy()
    work["partition"] = work["TOWN"]
    fit = work[work["month"] <= FIT_END]
    train_cells = _cell_table(fit)
    periods = {
        "search": _period_cells(work, SEARCH, train_cells),
        "confirm": _period_cells(work, CONFIRM, train_cells),
    }
    shared = _design(train_cells, periods, interact=None)
    month_slope = _design(train_cells, periods, interact="month")
    rows = []
    rows.extend(_linear_family(shared, sample_weight=train_cells["rows"].to_numpy(dtype=np.float64)))
    rows.extend(_tree_family(shared, sample_weight=train_cells["rows"].to_numpy(dtype=np.float64)))
    rows.append(_ridge_row("ridge_town_month_slope", month_slope))
    forward = _forward_paths(train_cells)
    payload = {
        "target": "town-month mean of MONTHLY_RENT",
        "features": ["TOWN", "month_index", *LEVELS],
        "fit_end": FIT_END,
        "fit_cells": int(len(train_cells)),
        "models": rows,
        "forward_from_2025_04_no_labels": forward,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for row in rows:
        print(
            f"{row['name']}: search {row['search_weighted_cell_rmse']:.2f}, "
            f"confirm {row['confirm_weighted_cell_rmse']:.2f}"
        )
    print("forward monthly mean, first and last test month with no label")
    for name, path in forward.items():
        months = sorted(path)
        print(f"  {name}: {months[0]} {path[months[0]]:.1f} -> {months[-1]} {path[months[-1]]:.1f}")
    print(f"wrote {OUT.relative_to(ROOT).as_posix()}")


def _period_cells(frame: pd.DataFrame, months: set[str], train_cells: pd.DataFrame) -> pd.DataFrame:
    levels = set(train_cells["partition"].astype(str))
    scored = frame[frame["month"].isin(months) & frame["partition"].astype(str).isin(levels)]
    return _cell_table(scored)


def _design(train_cells: pd.DataFrame, periods: dict[str, pd.DataFrame], interact: str | None) -> dict:
    features = ["month_index", *LEVELS]
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    train_area = encoder.fit_transform(train_cells[["partition"]].astype(str))
    train_num = _numeric_block(train_cells, features, train_cells)
    blocks = {
        "train": _bind(train_num, train_area, interact),
        "y": train_cells["mean_rent"].to_numpy(dtype=np.float64),
    }
    for name, cells in periods.items():
        area = encoder.transform(cells[["partition"]].astype(str))
        number = _numeric_block(cells, features, train_cells)
        blocks[name] = {
            "x": _bind(number, area, interact),
            "y": cells["mean_rent"].to_numpy(dtype=np.float64),
            "w": cells["rows"].to_numpy(dtype=np.float64),
        }
    return blocks


def _bind(number: np.ndarray, area: np.ndarray, interact: str | None) -> np.ndarray:
    parts = [number, area]
    if interact == "month":
        parts.append(area * number[:, :1])
    return np.column_stack(parts)


def _linear_family(shared: dict, sample_weight: np.ndarray) -> list[dict]:
    specs = [
        ("ridge_a0.1", Ridge(alpha=0.1), None),
        ("ridge_a1", Ridge(alpha=1.0), None),
        ("ridge_a10", Ridge(alpha=10.0), None),
        ("ridge_a100", Ridge(alpha=100.0), None),
        ("ridge_a1_weighted", Ridge(alpha=1.0), sample_weight),
        ("ols", LinearRegression(), None),
        ("huber", HuberRegressor(epsilon=1.35, max_iter=500), None),
        ("elasticnet_a1", ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=5000), None),
    ]
    rows = []
    for name, model, weight in specs:
        kwargs = {}
        if weight is not None:
            kwargs["sample_weight"] = weight
        model.fit(shared["train"], shared["y"], **kwargs)
        rows.append(_score(name, model.predict, shared))
    return rows


def _tree_family(shared: dict, sample_weight: np.ndarray) -> list[dict]:
    specs = []
    for depth, rounds in (
        (2, 100),
        (2, 300),
        (3, 100),
        (3, 300),
        (3, 800),
        (3, 2000),
        (4, 100),
        (4, 300),
        (4, 800),
    ):
        specs.append(
            (
                f"xgb_d{depth}_n{rounds}",
                XGBRegressor(
                    n_estimators=rounds,
                    max_depth=depth,
                    learning_rate=0.05,
                    objective="reg:squarederror",
                    random_state=SEED,
                    n_jobs=1,
                    tree_method="hist",
                ),
                None,
            )
        )
        specs.append(
            (
                f"cat_d{depth}_n{rounds}",
                CatBoostRegressor(
                    iterations=rounds,
                    depth=depth,
                    learning_rate=0.05,
                    loss_function="RMSE",
                    random_seed=SEED,
                    thread_count=1,
                    allow_writing_files=False,
                    verbose=False,
                ),
                None,
            )
        )
    specs.append(
        (
            "xgb_d3_n300_weighted",
            XGBRegressor(
                n_estimators=300,
                max_depth=3,
                learning_rate=0.05,
                objective="reg:squarederror",
                random_state=SEED,
                n_jobs=1,
                tree_method="hist",
            ),
            sample_weight,
        )
    )
    specs.append(
        (
            "cat_d3_n300_weighted",
            CatBoostRegressor(
                iterations=300,
                depth=3,
                learning_rate=0.05,
                loss_function="RMSE",
                random_seed=SEED,
                thread_count=1,
                allow_writing_files=False,
                verbose=False,
            ),
            sample_weight,
        )
    )
    rows = []
    for name, model, weight in specs:
        kwargs = {}
        if weight is not None:
            kwargs["sample_weight"] = weight
        model.fit(shared["train"], shared["y"], **kwargs)
        rows.append(_score(name, model.predict, shared))
    return rows


def _ridge_row(name: str, design: dict) -> dict:
    model = Ridge(alpha=1.0)
    model.fit(design["train"], design["y"])
    return _score(name, model.predict, design)


def _forward_paths(train_cells: pd.DataFrame) -> dict:
    """Month paths on test months that have no rent label. Not an RMSE."""
    future = _future_cells(train_cells)
    design = _design(train_cells, {"future": future}, interact=None)
    weight = train_cells["rows"].to_numpy(dtype=np.float64)
    models = {
        "ridge_a1": Ridge(alpha=1.0),
        "ridge_a1_weighted": Ridge(alpha=1.0),
        "cat_d3_n800": CatBoostRegressor(
            iterations=800,
            depth=3,
            learning_rate=0.05,
            loss_function="RMSE",
            random_seed=SEED,
            thread_count=1,
            allow_writing_files=False,
            verbose=False,
        ),
    }
    paths = {}
    for name, model in models.items():
        kwargs = {"sample_weight": weight} if name.endswith("weighted") else {}
        model.fit(design["train"], design["y"], **kwargs)
        prediction = np.asarray(model.predict(design["future"]["x"]), dtype=np.float64)
        frame = future[["month"]].copy()
        frame["prediction"] = prediction
        monthly = frame.groupby("month")["prediction"].mean()
        paths[name] = {month: float(value) for month, value in monthly.items()}
    return paths


def _future_cells(train_cells: pd.DataFrame) -> pd.DataFrame:
    dates = pd.read_csv(DATA / "test.csv", usecols=["RENT_APPROVAL_DATE"])
    auxiliary = pd.read_csv(DATA / "auxiliary_features" / "test_features.csv", usecols=LEVELS)
    if len(dates) != len(auxiliary):
        raise SystemExit("test and auxiliary row counts differ")
    auxiliary = auxiliary.copy()
    auxiliary["month"] = dates["RENT_APPROVAL_DATE"].astype("string").str.slice(0, 7)
    by_month = auxiliary.groupby("month", as_index=False)[LEVELS].first()
    by_month = by_month[by_month["month"] >= "2025-04"].copy()
    by_month["month_index"] = _month_index(by_month["month"])
    towns = sorted(train_cells["partition"].astype(str).unique())
    frames = []
    for town in towns:
        part = by_month.copy()
        part["partition"] = town
        part["mean_rent"] = np.nan
        part["rows"] = 1
        frames.append(part)
    return pd.concat(frames, ignore_index=True)


def _score(name: str, predict, design: dict) -> dict:
    row = {"name": name}
    for period in ("search", "confirm"):
        block = design[period]
        prediction = np.asarray(predict(block["x"]), dtype=np.float64)
        row[f"{period}_weighted_cell_rmse"] = _weighted_rmse(block["y"], prediction, block["w"])
    return row


if __name__ == "__main__":
    main()
