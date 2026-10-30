"""Build the train and test tables with the auxiliary columns.

Run from the project root:

    python -m src.features.build_features
    python -m src.features.build_features --keep-original

The default writes only the new columns, in the same row order as
``train.csv`` and ``test.csv``. Pass ``--keep-original`` to keep the
source columns in front of them. The six group-mean columns use training
rents from the previous three approval months only. No other feature reads
``MONTHLY_RENT``.
"""

import argparse
from pathlib import Path

import pandas as pd

from src.features.blocks import add_block_columns
from src.features.coe import add_coe_columns
from src.features.group_means import GROUP_MEAN_COLUMNS, add_group_mean_columns, assert_past_only
from src.features.spatial import block_spatial_features
from src.features.stocks import add_stock_columns

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "Data"
ADDRESS_DTYPES = {"TOWN": str, "BLOCK": str, "STREET": str}

FEATURE_COLUMNS = [
    "latitude",
    "longitude",
    "postal_code",
    "max_floor",
    "year_completed",
    "subzone",
    "planning_area",
    "region",
    "building_age_years",
    "lease_vs_completion_gap",
    "flat_age_from_lease",
    "dist_nearest_open_mrt_km",
    "nearest_open_mrt_code",
    "nearest_open_mrt_name",
    "nearest_open_mrt_line",
    "dist_nearest_primary_km",
    "nearest_primary_name",
    "n_primary_within_1km",
    "dist_nearest_mall_km",
    "nearest_mall_name",
    "nearest_ns_station_code",
    "nearest_ns_station_name",
    "dist_nearest_ns_km",
    "nearest_ew_station_code",
    "nearest_ew_station_name",
    "dist_nearest_ew_km",
    "nearest_ne_station_code",
    "nearest_ne_station_name",
    "dist_nearest_ne_km",
    "nearest_cc_station_code",
    "nearest_cc_station_name",
    "dist_nearest_cc_km",
    "nearest_dt_station_code",
    "nearest_dt_station_name",
    "dist_nearest_dt_km",
    "nearest_te_station_code",
    "nearest_te_station_name",
    "dist_nearest_te_km",
    "n_open_mrt_within_0_5km",
    "n_open_mrt_within_1km",
    "n_open_mrt_within_2km",
    "dist_nearest_planned_mrt_km",
    "n_planned_mrt_within_1km",
    "n_planned_mrt_within_2km",
    "nearest_open_mrt_is_interchange",
    "n_primary_within_2km",
    "dist_nearest_secondary_km",
    "nearest_secondary_name",
    "n_secondary_within_1km",
    "dist_nearest_school_km",
    "nearest_school_name",
    "n_school_within_1km",
    "n_mall_within_1km",
    "n_mall_within_2km",
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
    "dist_nearest_future_line_open_km",
    "dist_nearest_future_line_planned_km",
    "n_hdb_blocks_within_500m",
    "dist_to_town_centroid_km",
    "dist_to_cbd_km",
    "bank_basket_adj_close_lag1",
    "bank_basket_return_lag1",
    "property_basket_adj_close_lag1",
    "property_basket_return_lag1",
    "log_dist_nearest_open_mrt_km",
    "is_central_region",
    "building_age_x_central",
    *GROUP_MEAN_COLUMNS,
]

# Source predictors. MONTHLY_RENT is the target and is not a feature.
ORIGINAL_COLUMNS = [
    "RENT_APPROVAL_DATE",
    "TOWN",
    "BLOCK",
    "STREET",
    "FLAT_TYPE",
    "FLAT_MODEL",
    "FLOOR_AREA_SQM",
    "FURNISHED",
    "LEASE_COMMENCE_DATE",
    "FEE",
]

BLOCK_COLUMNS = [
    "latitude",
    "longitude",
    "postal_code",
    "max_floor",
    "year_completed",
    "subzone",
    "planning_area",
    "region",
    "building_age_years",
    "lease_vs_completion_gap",
    "flat_age_from_lease",
    "is_central_region",
    "building_age_x_central",
]

COE_COLUMNS = [
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
]

STOCK_COLUMNS = [
    "bank_basket_adj_close_lag1",
    "bank_basket_return_lag1",
    "property_basket_adj_close_lag1",
    "property_basket_return_lag1",
]

SPATIAL_COLUMNS = [
    "dist_nearest_open_mrt_km",
    "nearest_open_mrt_code",
    "nearest_open_mrt_name",
    "nearest_open_mrt_line",
    "dist_nearest_primary_km",
    "nearest_primary_name",
    "n_primary_within_1km",
    "dist_nearest_mall_km",
    "nearest_mall_name",
    "nearest_ns_station_code",
    "nearest_ns_station_name",
    "dist_nearest_ns_km",
    "nearest_ew_station_code",
    "nearest_ew_station_name",
    "dist_nearest_ew_km",
    "nearest_ne_station_code",
    "nearest_ne_station_name",
    "dist_nearest_ne_km",
    "nearest_cc_station_code",
    "nearest_cc_station_name",
    "dist_nearest_cc_km",
    "nearest_dt_station_code",
    "nearest_dt_station_name",
    "dist_nearest_dt_km",
    "nearest_te_station_code",
    "nearest_te_station_name",
    "dist_nearest_te_km",
    "n_open_mrt_within_0_5km",
    "n_open_mrt_within_1km",
    "n_open_mrt_within_2km",
    "dist_nearest_planned_mrt_km",
    "n_planned_mrt_within_1km",
    "n_planned_mrt_within_2km",
    "nearest_open_mrt_is_interchange",
    "n_primary_within_2km",
    "dist_nearest_secondary_km",
    "nearest_secondary_name",
    "n_secondary_within_1km",
    "dist_nearest_school_km",
    "nearest_school_name",
    "n_school_within_1km",
    "n_mall_within_1km",
    "n_mall_within_2km",
    "dist_nearest_future_line_open_km",
    "dist_nearest_future_line_planned_km",
    "n_hdb_blocks_within_500m",
    "dist_to_town_centroid_km",
    "dist_to_cbd_km",
    "log_dist_nearest_open_mrt_km",
]

FAMILY_COLUMNS = {
    "blocks": BLOCK_COLUMNS,
    "spatial": SPATIAL_COLUMNS,
    "coe": COE_COLUMNS,
    "stocks": STOCK_COLUMNS,
    "group_means": list(GROUP_MEAN_COLUMNS),
}


def check_feature_catalog() -> None:
    """Families partition FEATURE_COLUMNS and do not reuse source columns."""
    grouped = (
        BLOCK_COLUMNS
        + SPATIAL_COLUMNS
        + COE_COLUMNS
        + STOCK_COLUMNS
        + list(GROUP_MEAN_COLUMNS)
    )
    if len(grouped) != len(set(grouped)):
        raise AssertionError("feature families overlap")
    if set(grouped) != set(FEATURE_COLUMNS):
        missing = sorted(set(FEATURE_COLUMNS) - set(grouped))
        extra = sorted(set(grouped) - set(FEATURE_COLUMNS))
        raise AssertionError(f"feature catalog mismatch missing={missing} extra={extra}")
    overlap = set(ORIGINAL_COLUMNS) & set(FEATURE_COLUMNS)
    if overlap:
        raise AssertionError(f"source columns collide with auxiliary columns: {sorted(overlap)}")


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=ADDRESS_DTYPES)


def _nulls(frame: pd.DataFrame, column: str) -> int:
    return int(frame[column].isna().sum())


def _check(frame: pd.DataFrame, name: str, rows: int, postal: int, age: int, coe: int, property_close: int) -> None:
    if len(frame) != rows:
        raise AssertionError(f"{name} has {len(frame)} rows, expected {rows}")
    if frame["latitude"].isna().any():
        raise AssertionError(f"{name} address join missed {_nulls(frame, 'latitude')} rows")
    checks = {
        "postal_code": postal,
        "building_age_years": age,
        "coe_a_premium_lag1": coe,
        "property_basket_adj_close_lag1": property_close,
    }
    for column, expected in checks.items():
        actual = _nulls(frame, column)
        if actual != expected:
            raise AssertionError(f"{name} {column} nulls are {actual}, expected {expected}")


def assemble_features(rentals: pd.DataFrame, aux: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Attach block, spatial, COE, and stock columns. Row order matches ``rentals``."""
    built = add_block_columns(rentals, aux["blocks"])
    spatial = block_spatial_features(aux["blocks"], aux["mrt"], aux["schools"], aux["malls"])
    if not spatial["block_key"].is_unique:
        raise ValueError("spatial features are not one row per block")
    built = built.merge(spatial, on="block_key", how="left", validate="many_to_one")
    if len(built) != len(rentals):
        raise ValueError("spatial join changed the rental row count")
    built = add_coe_columns(built, aux["coe"])
    return add_stock_columns(built, aux["stocks"])


def _select_columns(built: pd.DataFrame, rentals: pd.DataFrame, keep_original: bool) -> pd.DataFrame:
    missing = [column for column in FEATURE_COLUMNS if column not in built.columns]
    if missing:
        raise ValueError(f"missing feature columns: {missing}")
    columns = list(rentals.columns) + FEATURE_COLUMNS if keep_original else FEATURE_COLUMNS
    return built[columns].reset_index(drop=True)


def build_rental_features(
    rentals: pd.DataFrame,
    aux: dict[str, pd.DataFrame],
    keep_original: bool = False,
    history: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return the new columns, optionally preceded by the source columns.

    ``history`` is the training table used for group-mean rents. When it is
    omitted, ``rentals`` itself must contain ``MONTHLY_RENT``.
    """
    built = assemble_features(rentals, aux)
    built = add_group_mean_columns(built, built if history is None else history)
    return _select_columns(built, rentals, keep_original)


def main(keep_original: bool = False) -> None:
    check_feature_catalog()
    aux = {
        "blocks": _read_csv(DATA / "auxiliary" / "sg-hdb-block.csv"),
        "mrt": pd.read_csv(DATA / "auxiliary" / "sg-mrt-stations.csv"),
        "schools": pd.read_csv(DATA / "auxiliary" / "sg-schools.csv"),
        "malls": pd.read_csv(DATA / "auxiliary" / "sg-shopping-malls.csv"),
        "coe": pd.read_csv(DATA / "auxiliary" / "sg-coe-prices.csv"),
        "stocks": pd.read_csv(DATA / "auxiliary" / "sg-stock-prices.csv"),
    }
    train_rows = _read_csv(DATA / "train.csv")
    test_rows = _read_csv(DATA / "test.csv")
    train_base = assemble_features(train_rows, aux)
    test_base = assemble_features(test_rows, aux)
    train_base = add_group_mean_columns(train_base, train_base)
    test_base = add_group_mean_columns(test_base, train_base)
    assert_past_only(train_base, test_base)
    train = _select_columns(train_base, train_rows, keep_original)
    test = _select_columns(test_base, test_rows, keep_original)
    _check(train, "train", 150_000, 5_259, 159, 3_129, 28_466)
    _check(test, "test", 50_000, 1_739, 0, 0, 0)
    if len(FEATURE_COLUMNS) != 83:
        raise AssertionError(f"expected 83 feature columns, found {len(FEATURE_COLUMNS)}")
    if not keep_original and list(train.columns) != FEATURE_COLUMNS:
        raise AssertionError("default output should contain only the new feature columns")

    out_dir = DATA / "auxiliary_features"
    out_dir.mkdir(parents=True, exist_ok=True)
    train.to_parquet(out_dir / "train_features.parquet", index=False)
    test.to_parquet(out_dir / "test_features.parquet", index=False)
    train.to_csv(out_dir / "train_features.csv", index=False)
    test.to_csv(out_dir / "test_features.csv", index=False)
    print(f"wrote {len(train)} train rows and {len(test)} test rows to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--keep-original",
        action="store_true",
        help="Keep the source CSV columns in front of the new feature columns.",
    )
    main(keep_original=parser.parse_args().keep_original)
