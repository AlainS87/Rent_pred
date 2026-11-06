"""Fixed inputs for the local indexed-rent CatBoost experiment."""

LOCAL_FEATURES = [
    "TOWN", "building_age_years", "max_floor", "year_completed", "flat_age_from_lease",
    "approval_month_index", "floor_area_sqm", "fee_amount", "lease_commence_year",
    "latitude", "longitude", "block_key", "street_key", "flat_type_ord", "flat_model_ord",
    "flat_model_key", "nearest_open_mrt_code", "dist_nearest_open_mrt_km",
    "log_dist_nearest_open_mrt_km", "nearest_mall_name", "dist_nearest_mall_km",
    "nearest_primary_name", "dist_nearest_primary_km", "nearest_secondary_name",
    "dist_nearest_secondary_km", "full_address_key",
]
BASE_COLUMNS = ["RENT_APPROVAL_DATE", *LOCAL_FEATURES[:-1]]
MACRO_COLUMNS = ["coe_c_premium_lag1", "bank_basket_adj_close_lag1", "property_basket_adj_close_lag1"]
