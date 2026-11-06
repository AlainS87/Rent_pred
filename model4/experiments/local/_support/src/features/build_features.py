"""Internal numerical support selected by the local experiments."""
import pandas as pd
from .blocks import add_block_columns
from .coe import add_coe_columns
from .group_means import GROUP_MEAN_COLUMNS, add_group_mean_columns
from .source_fields import SOURCE_FIELD_COLUMNS, add_source_fields
from .spatial import block_spatial_features
from .stocks import add_stock_columns
FEATURE_COLUMNS = ['latitude', 'longitude', 'postal_code', 'max_floor', 'year_completed', 'subzone', 'planning_area', 'region', 'building_age_years', 'lease_vs_completion_gap', 'flat_age_from_lease', 'dist_nearest_open_mrt_km', 'nearest_open_mrt_code', 'nearest_open_mrt_name', 'nearest_open_mrt_line', 'dist_nearest_primary_km', 'nearest_primary_name', 'n_primary_within_1km', 'dist_nearest_mall_km', 'nearest_mall_name', 'nearest_ns_station_code', 'nearest_ns_station_name', 'dist_nearest_ns_km', 'nearest_ew_station_code', 'nearest_ew_station_name', 'dist_nearest_ew_km', 'nearest_ne_station_code', 'nearest_ne_station_name', 'dist_nearest_ne_km', 'nearest_cc_station_code', 'nearest_cc_station_name', 'dist_nearest_cc_km', 'nearest_dt_station_code', 'nearest_dt_station_name', 'dist_nearest_dt_km', 'nearest_te_station_code', 'nearest_te_station_name', 'dist_nearest_te_km', 'n_open_mrt_within_0_5km', 'n_open_mrt_within_1km', 'n_open_mrt_within_2km', 'dist_nearest_planned_mrt_km', 'n_planned_mrt_within_1km', 'n_planned_mrt_within_2km', 'nearest_open_mrt_is_interchange', 'n_primary_within_2km', 'dist_nearest_secondary_km', 'nearest_secondary_name', 'n_secondary_within_1km', 'dist_nearest_school_km', 'nearest_school_name', 'n_school_within_1km', 'n_mall_within_1km', 'n_mall_within_2km', 'coe_a_premium_lag1', 'coe_a_premium_mom_change', 'coe_a_bids_per_quota_lag1', 'coe_b_premium_lag1', 'coe_b_premium_mom_change', 'coe_b_bids_per_quota_lag1', 'coe_e_premium_lag1', 'coe_e_premium_mom_change', 'coe_e_bids_per_quota_lag1', 'coe_a_premium_roll3', 'coe_c_premium_lag1', 'dist_nearest_future_line_open_km', 'dist_nearest_future_line_planned_km', 'n_hdb_blocks_within_500m', 'dist_to_town_centroid_km', 'dist_to_cbd_km', 'bank_basket_adj_close_lag1', 'bank_basket_return_lag1', 'property_basket_adj_close_lag1', 'property_basket_return_lag1', 'log_dist_nearest_open_mrt_km', 'is_central_region', 'building_age_x_central', *GROUP_MEAN_COLUMNS, *SOURCE_FIELD_COLUMNS]
ORIGINAL_COLUMNS = ['RENT_APPROVAL_DATE', 'TOWN', 'BLOCK', 'STREET', 'FLAT_TYPE', 'FLAT_MODEL', 'FLOOR_AREA_SQM', 'FURNISHED', 'LEASE_COMMENCE_DATE', 'FEE']

def assemble_features(rentals: pd.DataFrame, aux: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Attach block, spatial, COE, and stock columns. Row order matches ``rentals``."""
    built = add_block_columns(rentals, aux['blocks'])
    spatial = block_spatial_features(aux['blocks'], aux['mrt'], aux['schools'], aux['malls'])
    if not spatial['block_key'].is_unique:
        raise ValueError('spatial features are not one row per block')
    built = built.merge(spatial, on='block_key', how='left', validate='many_to_one')
    if len(built) != len(rentals):
        raise ValueError('spatial join changed the rental row count')
    built = add_coe_columns(built, aux['coe'])
    return add_stock_columns(built, aux['stocks'])

def _select_columns(built: pd.DataFrame, rentals: pd.DataFrame, keep_original: bool) -> pd.DataFrame:
    missing = [column for column in FEATURE_COLUMNS if column not in built.columns]
    if missing:
        raise ValueError(f'missing feature columns: {missing}')
    columns = list(rentals.columns) + FEATURE_COLUMNS if keep_original else FEATURE_COLUMNS
    return built[columns].reset_index(drop=True)

def build_rental_features(rentals: pd.DataFrame, aux: dict[str, pd.DataFrame], keep_original: bool=False, history: pd.DataFrame | None=None) -> pd.DataFrame:
    """Return the new columns, optionally preceded by the source columns.

    ``history`` is the training table used for group-mean rents. When it is
    omitted, ``rentals`` itself must contain ``MONTHLY_RENT``.
    """
    built = assemble_features(rentals, aux)
    built = add_group_mean_columns(built, built if history is None else history)
    built = add_source_fields(built, rentals)
    return _select_columns(built, rentals, keep_original)
