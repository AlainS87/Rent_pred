"""Feasibility check: predict a partition-month mean rent from economy columns.

This does not train a production model and does not choose a partition.
Run from the project root:

    python -m src.experiments.partition_month_mean
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder

from src.features.group_means import postal_district, postal_sector

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/raw"
OUT = ROOT / "outputs" / "feasibility" / "partition_month_mean.json"

ECONOMY = [
    "coe_a_premium_lag1",
    "coe_a_premium_mom_change",
    "coe_a_bids_per_quota_lag1",
    "coe_b_premium_lag1",
    "coe_b_premium_mom_change",
    "coe_b_bids_per_quota_lag1",
    "coe_e_premium_lag1",
    "coe_e_premium_mom_change",
    "coe_e_bids_per_quota_lag1",
    "coe_a_premium_roll3",
    "coe_c_premium_lag1",
    "bank_basket_adj_close_lag1",
    "bank_basket_return_lag1",
    "property_basket_adj_close_lag1",
    "property_basket_return_lag1",
]
FIT_END = "2024-09"
SEARCH = {"2024-10", "2024-11", "2024-12"}
CONFIRM = {"2025-01", "2025-02", "2025-03"}


def main() -> None:
    frame = _load()
    economy = _economy_report(frame)
    partitions = {
        "region": frame["region"],
        "town": frame["TOWN"],
        "planning_area": frame["planning_area"],
        "postal_district": frame["postal_district"],
        "postal_sector": frame["postal_sector"],
    }
    sizes = {name: _cell_sizes(frame, labels) for name, labels in partitions.items()}
    scores = {
        name: _score_partition(frame, labels)
        for name, labels in partitions.items()
    }
    flat_type = _flat_type_inside_cells(frame, partitions["town"])
    payload = {
        "target": "MONTHLY_RENT",
        "economy": economy,
        "cell_sizes_on_fit_months": sizes,
        "mean_prediction": scores,
        "flat_type_inside_town_month": flat_type,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _print(payload)
    print(f"wrote {OUT.relative_to(ROOT).as_posix()}")


def _load() -> pd.DataFrame:
    source = pd.read_csv(
        DATA / "train.csv",
        usecols=["RENT_APPROVAL_DATE", "TOWN", "FLAT_TYPE", "MONTHLY_RENT"],
        dtype={"TOWN": str},
    )
    auxiliary = pd.read_csv(
        ROOT / "artifacts" / "features" / "train_features.csv",
        usecols=["postal_code", "region", "planning_area", "subzone", *ECONOMY],
    )
    if len(source) != len(auxiliary):
        raise SystemExit("train and auxiliary row counts differ")
    frame = pd.concat([source, auxiliary], axis=1)
    frame["month"] = frame["RENT_APPROVAL_DATE"].astype("string").str.slice(0, 7)
    frame["postal_sector"] = postal_sector(frame["postal_code"])
    frame["postal_district"] = postal_district(frame["postal_sector"])
    frame["month_index"] = _month_index(frame["month"])
    return frame


def _economy_report(frame: pd.DataFrame) -> dict:
    rows = []
    for column in ECONOMY:
        per_month = frame.groupby("month")[column].nunique(dropna=False)
        rows.append(
            {
                "column": column,
                "missing_rows": int(frame[column].isna().sum()),
                "distinct_values": int(frame[column].nunique(dropna=True)),
                "max_values_inside_one_month": int(per_month.max()),
            }
        )
    return rows


def _cell_sizes(frame: pd.DataFrame, labels: pd.Series) -> dict:
    fit = frame[frame["month"] <= FIT_END].copy()
    fit["partition"] = labels.loc[fit.index]
    known = fit["partition"].notna()
    cells = fit.loc[known].groupby(["partition", "month"], observed=True).size()
    return {
        "groups": int(fit.loc[known, "partition"].nunique()),
        "missing_rows": int((~known).sum()),
        "cells": int(len(cells)),
        "rows_per_cell_p10": float(cells.quantile(0.10)),
        "rows_per_cell_median": float(cells.median()),
        "share_cells_under_30": float((cells < 30).mean()),
    }


def _score_partition(frame: pd.DataFrame, labels: pd.Series) -> dict:
    work = frame.copy()
    work["partition"] = labels
    work = work[work["partition"].notna()].copy()
    fit = work[work["month"] <= FIT_END]
    levels = sorted(fit["partition"].astype(str).unique())
    result = {}
    for name, months in (("search", SEARCH), ("confirm", CONFIRM)):
        scored = work[work["month"].isin(months) & work["partition"].astype(str).isin(levels)]
        cells = _cell_table(scored)
        fit_means = fit.groupby("partition", observed=True)["MONTHLY_RENT"].mean()
        predictions = {
            "static_partition_mean": cells["partition"].map(fit_means).to_numpy(dtype=np.float64),
            "partition_plus_month_index": _ridge_cells(fit, cells, economy=False),
            "partition_plus_economy": _ridge_cells(fit, cells, economy=True),
            "economy_only": _ridge_cells(fit, cells, economy=True, partition=False),
        }
        truth = cells["mean_rent"].to_numpy(dtype=np.float64)
        weights = cells["rows"].to_numpy(dtype=np.float64)
        cell_scores = {
            key: _weighted_rmse(truth, value, weights) for key, value in predictions.items()
        }
        oracle = cells.set_index(["partition", "month"])["mean_rent"]
        row_oracle = scored.set_index(["partition", "month"]).index.map(oracle).to_numpy(dtype=np.float64)
        rents = scored["MONTHLY_RENT"].to_numpy(dtype=np.float64)
        result[name] = {
            "cells": int(len(cells)),
            "rows": int(len(scored)),
            "cell_mean_rmse": cell_scores,
            "row_rmse_if_given_true_cell_mean": _rmse(rents, row_oracle),
            "row_rmse_if_given_global_fit_mean": _rmse(rents, np.full(len(rents), fit["MONTHLY_RENT"].mean())),
        }
    return result


def _cell_table(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = frame.groupby(["partition", "month"], observed=True)
    cells = grouped["MONTHLY_RENT"].agg(mean_rent="mean", rows="size").reset_index()
    month_index = frame.groupby("month")["month_index"].first()
    cells["month_index"] = cells["month"].map(month_index)
    for column in ECONOMY:
        values = frame.groupby("month")[column].first()
        cells[column] = cells["month"].map(values)
    return cells


def _ridge_cells(fit: pd.DataFrame, cells: pd.DataFrame, economy: bool, partition: bool = True) -> np.ndarray:
    train_cells = _cell_table(fit)
    features = ["month_index"]
    if economy:
        features.extend(ECONOMY)
    encoder = None
    if partition:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        train_area = encoder.fit_transform(train_cells[["partition"]].astype(str))
        score_area = encoder.transform(cells[["partition"]].astype(str))
    else:
        train_area = np.zeros((len(train_cells), 0))
        score_area = np.zeros((len(cells), 0))
    train_x = np.column_stack([_numeric_block(train_cells, features, train_cells), train_area])
    score_x = np.column_stack([_numeric_block(cells, features, train_cells), score_area])
    model = Ridge(alpha=1.0)
    model.fit(train_x, train_cells["mean_rent"].to_numpy(dtype=np.float64))
    return model.predict(score_x)


def _numeric_block(frame: pd.DataFrame, columns: list[str], train: pd.DataFrame) -> np.ndarray:
    values = []
    for column in columns:
        series = frame[column].astype(float)
        fill = float(train[column].median())
        filled = series.fillna(fill).to_numpy(dtype=np.float64)
        center = float(train[column].median())
        spread = float(train[column].std(ddof=0)) or 1.0
        values.append((np.where(np.isfinite(filled), filled, center) - center) / spread)
    return np.column_stack(values)


def _flat_type_inside_cells(frame: pd.DataFrame, labels: pd.Series) -> dict:
    """How much of the within town-month remainder is the flat type."""
    work = frame[frame["month"].isin(SEARCH)].copy()
    work["partition"] = labels.loc[work.index]
    work = work[work["partition"].notna()]
    rents = work["MONTHLY_RENT"].to_numpy(dtype=np.float64)
    cell = work.groupby(["partition", "month"])["MONTHLY_RENT"].transform("mean").to_numpy(dtype=np.float64)
    both = work.groupby(["partition", "month", "FLAT_TYPE"])["MONTHLY_RENT"].transform("mean").to_numpy(dtype=np.float64)
    return {
        "search_rows": int(len(work)),
        "row_rmse_around_town_month_mean": _rmse(rents, cell),
        "row_rmse_around_town_month_flat_type_mean": _rmse(rents, both),
    }


def _month_index(month: pd.Series) -> pd.Series:
    year = month.str.slice(0, 4).astype(int)
    part = month.str.slice(5, 7).astype(int)
    return (year - 2021) * 12 + (part - 1)


def _weighted_rmse(truth: np.ndarray, prediction: np.ndarray, weights: np.ndarray) -> float:
    keep = np.isfinite(truth) & np.isfinite(prediction) & np.isfinite(weights)
    error = truth[keep] - prediction[keep]
    weight = weights[keep]
    return float(np.sqrt(np.average(error**2, weights=weight)))


def _rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    error = truth - prediction
    return float(np.sqrt(np.mean(error**2)))


def _print(payload: dict) -> None:
    print("economy columns are constant inside a month when max_values_inside_one_month is 1")
    for row in payload["economy"]:
        print(
            f"  {row['column']}: missing {row['missing_rows']}, "
            f"distinct {row['distinct_values']}, "
            f"max inside one month {row['max_values_inside_one_month']}"
        )
    print("fit-month cell sizes")
    for name, row in payload["cell_sizes_on_fit_months"].items():
        print(
            f"  {name}: groups {row['groups']}, missing rows {row['missing_rows']}, "
            f"median rows/cell {row['rows_per_cell_median']:.0f}, "
            f"p10 {row['rows_per_cell_p10']:.0f}, "
            f"share under 30 {row['share_cells_under_30']:.2%}"
        )
    for name, periods in payload["mean_prediction"].items():
        for period, row in periods.items():
            scores = row["cell_mean_rmse"]
            print(
                f"{name} {period}: cell RMSE "
                f"static {scores['static_partition_mean']:.1f}, "
                f"trend {scores['partition_plus_month_index']:.1f}, "
                f"economy {scores['partition_plus_economy']:.1f}, "
                f"economy_only {scores['economy_only']:.1f}; "
                f"row RMSE true cell mean {row['row_rmse_if_given_true_cell_mean']:.1f}, "
                f"global mean {row['row_rmse_if_given_global_fit_mean']:.1f}"
            )
    flat = payload["flat_type_inside_town_month"]
    print(
        "search town-month residual "
        f"{flat['row_rmse_around_town_month_mean']:.1f}, "
        f"after flat type {flat['row_rmse_around_town_month_flat_type_mean']:.1f}"
    )


if __name__ == "__main__":
    import argparse
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
