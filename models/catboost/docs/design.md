# CatBoost

## Architecture gene

The spec is `max_depth` and `learning_rate`. Nominal columns other than the fixed date, block, and furnished overrides are native categories. Regularize lowers `max_depth` by 2 down to 2. Capacity raises `max_depth` by 2 up to 10.

## Adjustment rules

The evolve command reads `docs/logs` and `docs/champion.json`, then proposes up to three mutants of the champion. Each mutant changes one gene: the feature subset, the preprocess spec, or the architecture. A genome already present in the logs is not trained again.

Feature changes alternate by generation. An odd generation adds the unused pool column with the largest absolute correlation to the champion residual on the search months. An even generation drops the column with the smallest permutation importance on a search sample of at most 2,000 rows. When that single-column move was already tried from the same parent, and every column of a feature family is inside the pool, an odd generation adds that family as a block and an even generation removes it. A feature set always keeps at least one column.

Preprocess changes the first knob not yet tried from this parent, in this order: numeric_impute (median, mean, zero); scale (standard, none, minmax); categorical (onehot, ordinal, target, bin); rare_min_count (10, 30, 100). The new value is the first choice in that list that differs from the current value.

Architecture regularizes when (search RMSE - train RMSE) / train RMSE is above 0.15. Otherwise it adds capacity, inside the cap in this model's note.

A mutant replaces the champion only when its search RMSE is at least 1% lower. The loop stops at the generation cap, when a generation accepts nothing, or when every proposed mutant is already in the logs. Confirm RMSE is scored once after search stops and is not an acceptance signal. Predict writes a summary and does not change the genome.

Labeled months are cut in time. Confirm is the last `val_months` approval months. Search is the `val_months` immediately before confirm. Fit is every earlier month. The preprocessor is fit on the fit months only.

## Current champion

<!-- champion:start -->
- Run: `20261030T124215167507_catboost_417a4ec8`
- Log: `models/catboost/docs/logs/20261030T124215167507_catboost_417a4ec8.json`
- Feature hash: `e8f777332831`
- Genome hash: `417a4ec8fb716101`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `latitude`, `postal_code`, `max_floor`, `region`, `dist_nearest_open_mrt_km`, `nearest_mall_name`, `nearest_ns_station_code`, `nearest_ns_station_name`, `nearest_ew_station_code`, `nearest_ew_station_name`, `dist_nearest_ew_km`, `nearest_ne_station_code`, `nearest_ne_station_name`, `dist_nearest_cc_km`, `nearest_dt_station_code`, `nearest_dt_station_name`, `dist_nearest_dt_km`, `nearest_te_station_code`, `nearest_te_station_name`, `nearest_secondary_name`, `nearest_school_name`, `dist_to_cbd_km`, `log_dist_nearest_open_mrt_km`, `is_central_region`, `building_age_x_central`, `mean_rent_postal_sector`, `mean_rent_postal_district`, `mean_rent_nearest_mall`, `mean_rent_nearest_primary`, `mean_rent_nearest_secondary`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"early_stopping_rounds": 20, "learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 462.182823
- Search RMSE: 452.618834
- Confirm RMSE: 462.238496
- Confirm log: `models/catboost/docs/logs/20261030T124215397181_catboost_417a4ec8.json`
- Row cap: full training table
<!-- champion:end -->

## Lineage

<!-- lineage:start -->
Regenerated from `docs/logs`. Do not edit the metrics in this section by hand.

### `20261030T112955113606_catboost_122f4818`

- Command: train
- Parent: `None`
- Generation: None
- Mutation: `{"axis": "train"}`
- Feature hash: `170e95b30b54`
- Genome hash: `122f4818f190de06`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"early_stopping_rounds": 20, "learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 465.383905
- Search RMSE: 460.098274
- Decision: accepted. trained as a frozen genome and set as the champion
- Log: `models/catboost/docs/logs/20261030T112955113606_catboost_122f4818.json`

### `20261030T113111382171_catboost_122f4818`

- Command: train
- Parent: `None`
- Generation: None
- Mutation: `{"axis": "train"}`
- Feature hash: `170e95b30b54`
- Genome hash: `122f4818f190de06`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"early_stopping_rounds": 20, "learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 465.383905
- Search RMSE: 460.098274
- Decision: accepted. first saved best; there was no earlier snapshot
- Log: `models/catboost/docs/logs/20261030T113111382171_catboost_122f4818.json`

### `20261030T113159370415_catboost_103f4c73`

- Command: train
- Parent: `20261030T113111382171_catboost_122f4818`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "latitude", "round": 1}`
- Feature hash: `99b4d5d7749f`
- Genome hash: `103f4c736749de52`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `latitude`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"early_stopping_rounds": 20, "learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 464.997520
- Search RMSE: 458.425842
- Decision: rejected. not promising because search RMSE 458.425842 is not at least 1% below 460.098274 from 20261030T113111382171_catboost_122f4818 after adding latitude
- Log: `models/catboost/docs/logs/20261030T113159370415_catboost_103f4c73.json`

### `20261030T113246829994_catboost_f4dbf0ca`

- Command: train
- Parent: `20261030T113111382171_catboost_122f4818`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "longitude", "round": 1}`
- Feature hash: `c27505a182db`
- Genome hash: `f4dbf0ca482d184f`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `longitude`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"early_stopping_rounds": 20, "learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 465.096373
- Search RMSE: 460.326853
- Decision: rejected. not promising because search RMSE 460.326853 is not at least 1% below 460.098274 from 20261030T113111382171_catboost_122f4818 after adding longitude
- Log: `models/catboost/docs/logs/20261030T113246829994_catboost_f4dbf0ca.json`

### `20261030T113340523715_catboost_190d9894`

- Command: train
- Parent: `20261030T113111382171_catboost_122f4818`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "postal_code", "round": 1}`
- Feature hash: `e62a952e566d`
- Genome hash: `190d9894f5257156`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `postal_code`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"early_stopping_rounds": 20, "learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 461.804311
- Search RMSE: 456.568502
- Decision: rejected. not promising because search RMSE 456.568502 is not at least 1% below 460.098274 from 20261030T113111382171_catboost_122f4818 after adding postal_code
- Log: `models/catboost/docs/logs/20261030T113340523715_catboost_190d9894.json`

### `20261030T113423508062_catboost_6f2567c1`

- Command: train
- Parent: `20261030T113111382171_catboost_122f4818`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "max_floor", "round": 1}`
- Feature hash: `6d16b7bd0382`
- Genome hash: `6f2567c1be2794bc`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `max_floor`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"early_stopping_rounds": 20, "learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 465.014406
- Search RMSE: 460.042842
- Decision: rejected. not promising because search RMSE 460.042842 is not at least 1% below 460.098274 from 20261030T113111382171_catboost_122f4818 after adding max_floor
- Log: `models/catboost/docs/logs/20261030T113423508062_catboost_6f2567c1.json`

### `20261030T113511753706_catboost_7b59c773`

- Command: train
- Parent: `20261030T113111382171_catboost_122f4818`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "year_completed", "round": 1}`
- Feature hash: `b2d2d8a929b6`
- Genome hash: `7b59c773373bcf39`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `year_completed`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"early_stopping_rounds": 20, "learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 465.174842
- Search RMSE: 460.341270
- Decision: rejected. not promising because search RMSE 460.341270 is not at least 1% below 460.098274 from 20261030T113111382171_catboost_122f4818 after adding year_completed
- Log: `models/catboost/docs/logs/20261030T113511753706_catboost_7b59c773.json`

### `20261030T113600432374_catboost_1144ca44`

- Command: train
- Parent: `20261030T113111382171_catboost_122f4818`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "subzone", "round": 1}`
- Feature hash: `19ad5218a78e`
- Genome hash: `1144ca440cc22f79`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `subzone`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"early_stopping_rounds": 20, "learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 465.320223
- Search RMSE: 460.374613
- Decision: rejected. not promising because search RMSE 460.374613 is not at least 1% below 460.098274 from 20261030T113111382171_catboost_122f4818 after adding subzone
- Log: `models/catboost/docs/logs/20261030T113600432374_catboost_1144ca44.json`

### `20261030T113650454760_catboost_54e56c07`

- Command: train
- Parent: `20261030T113111382171_catboost_122f4818`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "planning_area", "round": 1}`
- Feature hash: `212df8d837f8`
- Genome hash: `54e56c0798890395`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `planning_area`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"early_stopping_rounds": 20, "learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 465.329673
- Search RMSE: 460.535896
- Decision: rejected. not promising because search RMSE 460.535896 is not at least 1% below 460.098274 from 20261030T113111382171_catboost_122f4818 after adding planning_area
- Log: `models/catboost/docs/logs/20261030T113650454760_catboost_54e56c07.json`

### `20261030T113733408366_catboost_7f3fcd1a`

- Command: train
- Parent: `20261030T113111382171_catboost_122f4818`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "region", "round": 1}`
- Feature hash: `fd9247992a0a`
- Genome hash: `7f3fcd1a7d3c0d55`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `region`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"early_stopping_rounds": 20, "learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 465.176090
- Search RMSE: 459.670861
- Decision: rejected. not promising because search RMSE 459.670861 is not at least 1% below 460.098274 from 20261030T113111382171_catboost_122f4818 after adding region
- Log: `models/catboost/docs/logs/20261030T113733408366_catboost_7f3fcd1a.json`

### `20261030T113813716622_catboost_0d302437`

- Command: train
- Parent: `20261030T113111382171_catboost_122f4818`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "building_age_years", "round": 1}`
- Feature hash: `c234d75ff5d3`
- Genome hash: `0d3024371607ced3`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `building_age_years`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"early_stopping_rounds": 20, "learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 465.268242
- Search RMSE: 460.535066
- Decision: rejected. not promising because search RMSE 460.535066 is not at least 1% below 460.098274 from 20261030T113111382171_catboost_122f4818 after adding building_age_years
- Log: `models/catboost/docs/logs/20261030T113813716622_catboost_0d302437.json`

### `20261030T113855136745_catboost_138a3006`

- Command: train
- Parent: `20261030T113111382171_catboost_122f4818`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "lease_vs_completion_gap", "round": 1}`
- Feature hash: `d33f593517bd`
- Genome hash: `138a3006c28a664c`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `lease_vs_completion_gap`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"early_stopping_rounds": 20, "learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 465.294441
- Search RMSE: 460.461814
- Decision: rejected. not promising because search RMSE 460.461814 is not at least 1% below 460.098274 from 20261030T113111382171_catboost_122f4818 after adding lease_vs_completion_gap
- Log: `models/catboost/docs/logs/20261030T113855136745_catboost_138a3006.json`

### `20261030T113932722670_catboost_8bf45434`

- Command: train
- Parent: `20261030T113111382171_catboost_122f4818`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "flat_age_from_lease", "round": 1}`
- Feature hash: `0ae4cfca946e`
- Genome hash: `8bf454348f66d462`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `flat_age_from_lease`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"early_stopping_rounds": 20, "learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 465.298715
- Search RMSE: 460.515089
- Decision: rejected. not promising because search RMSE 460.515089 is not at least 1% below 460.098274 from 20261030T113111382171_catboost_122f4818 after adding flat_age_from_lease
- Log: `models/catboost/docs/logs/20261030T113932722670_catboost_8bf45434.json`

### `20261030T114015265077_catboost_c9a22201`

- Command: train
- Parent: `20261030T113111382171_catboost_122f4818`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "dist_nearest_open_mrt_km", "round": 1}`
- Feature hash: `0c2594d85422`
- Genome hash: `c9a2220184540bb6`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `dist_nearest_open_mrt_km`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"early_stopping_rounds": 20, "learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 464.959719
- Search RMSE: 459.907811
- Decision: rejected. not promising because search RMSE 459.907811 is not at least 1% below 460.098274 from 20261030T113111382171_catboost_122f4818 after adding dist_nearest_open_mrt_km
- Log: `models/catboost/docs/logs/20261030T114015265077_catboost_c9a22201.json`

### `20261030T114106246578_catboost_99a04b3d`

- Command: train
- Parent: `20261030T113111382171_catboost_122f4818`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_open_mrt_code", "round": 1}`
- Feature hash: `42451fae9f20`
- Genome hash: `99a04b3d8c981a46`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `nearest_open_mrt_code`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"early_stopping_rounds": 20, "learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 465.205943
- Search RMSE: 460.448722
- Decision: rejected. not promising because search RMSE 460.448722 is not at least 1% below 460.098274 from 20261030T113111382171_catboost_122f4818 after adding nearest_open_mrt_code
- Log: `models/catboost/docs/logs/20261030T114106246578_catboost_99a04b3d.json`

### `20261030T114155056889_catboost_27581e8a`

- Command: train
- Parent: `20261030T113111382171_catboost_122f4818`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_open_mrt_name", "round": 1}`
- Feature hash: `57326fd029e4`
- Genome hash: `27581e8aee59b2d0`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `nearest_open_mrt_name`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"early_stopping_rounds": 20, "learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 465.205943
- Search RMSE: 460.448722
- Decision: rejected. not promising because search RMSE 460.448722 is not at least 1% below 460.098274 from 20261030T113111382171_catboost_122f4818 after adding nearest_open_mrt_name
- Log: `models/catboost/docs/logs/20261030T114155056889_catboost_27581e8a.json`

### `20261030T114237133007_catboost_25cd4e7f`

- Command: train
- Parent: `20261030T113111382171_catboost_122f4818`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_open_mrt_line", "round": 1}`
- Feature hash: `f6e2fc2120da`
- Genome hash: `25cd4e7f9d09ab03`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `nearest_open_mrt_line`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"early_stopping_rounds": 20, "learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 465.369003
- Search RMSE: 460.170153
- Decision: rejected. not promising because search RMSE 460.170153 is not at least 1% below 460.098274 from 20261030T113111382171_catboost_122f4818 after adding nearest_open_mrt_line
- Log: `models/catboost/docs/logs/20261030T114237133007_catboost_25cd4e7f.json`

### `20261030T114315219242_catboost_3a2be077`

- Command: train
- Parent: `20261030T113111382171_catboost_122f4818`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "dist_nearest_primary_km", "round": 1}`
- Feature hash: `ec8fcfbbdc5f`
- Genome hash: `3a2be077ddd9c3b4`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `dist_nearest_primary_km`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"early_stopping_rounds": 20, "learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 465.300028
- Search RMSE: 460.819861
- Decision: rejected. not promising because search RMSE 460.819861 is not at least 1% below 460.098274 from 20261030T113111382171_catboost_122f4818 after adding dist_nearest_primary_km
- Log: `models/catboost/docs/logs/20261030T114315219242_catboost_3a2be077.json`

### `20261030T114357647442_catboost_c712ce92`

- Command: train
- Parent: `20261030T113111382171_catboost_122f4818`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_primary_name", "round": 1}`
- Feature hash: `3b600d364a2c`
- Genome hash: `c712ce92aec9ddde`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `nearest_primary_name`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"early_stopping_rounds": 20, "learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 465.284920
- Search RMSE: 460.280931
- Decision: rejected. not promising because search RMSE 460.280931 is not at least 1% below 460.098274 from 20261030T113111382171_catboost_122f4818 after adding nearest_primary_name
- Log: `models/catboost/docs/logs/20261030T114357647442_catboost_c712ce92.json`

### `20261030T114446318757_catboost_443af69b`

- Command: train
- Parent: `20261030T113111382171_catboost_122f4818`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "n_primary_within_1km", "round": 1}`
- Feature hash: `4948f0d7026e`
- Genome hash: `443af69bf70aaf05`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `n_primary_within_1km`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"early_stopping_rounds": 20, "learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 465.270204
- Search RMSE: 460.504889
- Decision: rejected. not promising because search RMSE 460.504889 is not at least 1% below 460.098274 from 20261030T113111382171_catboost_122f4818 after adding n_primary_within_1km
- Log: `models/catboost/docs/logs/20261030T114446318757_catboost_443af69b.json`

### `20261030T114533893398_catboost_4eadf21e`

- Command: train
- Parent: `20261030T113111382171_catboost_122f4818`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "dist_nearest_mall_km", "round": 1}`
- Feature hash: `eb02b9676679`
- Genome hash: `4eadf21ed777c3f7`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `dist_nearest_mall_km`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"early_stopping_rounds": 20, "learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 465.232248
- Search RMSE: 460.357157
- Decision: rejected. not promising because search RMSE 460.357157 is not at least 1% below 460.098274 from 20261030T113111382171_catboost_122f4818 after adding dist_nearest_mall_km
- Log: `models/catboost/docs/logs/20261030T114533893398_catboost_4eadf21e.json`

### `20261030T114624483175_catboost_c1cf2fd6`

- Command: train
- Parent: `20261030T113111382171_catboost_122f4818`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_mall_name", "round": 1}`
- Feature hash: `dfa1a626ebab`
- Genome hash: `c1cf2fd6bf26b137`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `nearest_mall_name`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"early_stopping_rounds": 20, "learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 465.274817
- Search RMSE: 459.971196
- Decision: rejected. not promising because search RMSE 459.971196 is not at least 1% below 460.098274 from 20261030T113111382171_catboost_122f4818 after adding nearest_mall_name
- Log: `models/catboost/docs/logs/20261030T114624483175_catboost_c1cf2fd6.json`

### `20261030T114717246642_catboost_abda7cdb`

- Command: train
- Parent: `20261030T113111382171_catboost_122f4818`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_ns_station_code", "round": 1}`
- Feature hash: `5ce89d055366`
- Genome hash: `abda7cdb93dc8d37`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `nearest_ns_station_code`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"early_stopping_rounds": 20, "learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 465.220629
- Search RMSE: 459.528760
- Decision: rejected. not promising because search RMSE 459.528760 is not at least 1% below 460.098274 from 20261030T113111382171_catboost_122f4818 after adding nearest_ns_station_code
- Log: `models/catboost/docs/logs/20261030T114717246642_catboost_abda7cdb.json`

### `20261030T114808827407_catboost_decf5545`

- Command: train
- Parent: `20261030T113111382171_catboost_122f4818`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_ns_station_name", "round": 1}`
- Feature hash: `70914731bd06`
- Genome hash: `decf5545dceca803`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `nearest_ns_station_name`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"early_stopping_rounds": 20, "learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 465.220629
- Search RMSE: 459.528760
- Decision: rejected. not promising because search RMSE 459.528760 is not at least 1% below 460.098274 from 20261030T113111382171_catboost_122f4818 after adding nearest_ns_station_name
- Log: `models/catboost/docs/logs/20261030T114808827407_catboost_decf5545.json`

### `20261030T114854187416_catboost_a0a9fe15`

- Command: train
- Parent: `20261030T113111382171_catboost_122f4818`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "dist_nearest_ns_km", "round": 1}`
- Feature hash: `8d6e42f11adc`
- Genome hash: `a0a9fe156c8de3d9`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `dist_nearest_ns_km`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"early_stopping_rounds": 20, "learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 465.207929
- Search RMSE: 460.890395
- Decision: rejected. not promising because search RMSE 460.890395 is not at least 1% below 460.098274 from 20261030T113111382171_catboost_122f4818 after adding dist_nearest_ns_km
- Log: `models/catboost/docs/logs/20261030T114854187416_catboost_a0a9fe15.json`

### `20261030T114939969146_catboost_05a6913d`

- Command: train
- Parent: `20261030T113111382171_catboost_122f4818`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_ew_station_code", "round": 1}`
- Feature hash: `033e0d7f6436`
- Genome hash: `05a6913dbc5f2617`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `nearest_ew_station_code`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"early_stopping_rounds": 20, "learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 465.041273
- Search RMSE: 459.537737
- Decision: rejected. not promising because search RMSE 459.537737 is not at least 1% below 460.098274 from 20261030T113111382171_catboost_122f4818 after adding nearest_ew_station_code
- Log: `models/catboost/docs/logs/20261030T114939969146_catboost_05a6913d.json`

### `20261030T115023092285_catboost_3009e981`

- Command: train
- Parent: `20261030T113111382171_catboost_122f4818`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_ew_station_name", "round": 1}`
- Feature hash: `cc0d61a0df40`
- Genome hash: `3009e9817ce30083`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `nearest_ew_station_name`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"early_stopping_rounds": 20, "learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 465.041273
- Search RMSE: 459.537737
- Decision: rejected. not promising because search RMSE 459.537737 is not at least 1% below 460.098274 from 20261030T113111382171_catboost_122f4818 after adding nearest_ew_station_name
- Log: `models/catboost/docs/logs/20261030T115023092285_catboost_3009e981.json`

### `20261030T115103297657_catboost_935ed8ff`

- Command: train
- Parent: `20261030T113111382171_catboost_122f4818`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "dist_nearest_ew_km", "round": 1}`
- Feature hash: `8db553894f91`
- Genome hash: `935ed8ff9696e313`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `dist_nearest_ew_km`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"early_stopping_rounds": 20, "learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 465.148642
- Search RMSE: 459.734716
- Decision: rejected. not promising because search RMSE 459.734716 is not at least 1% below 460.098274 from 20261030T113111382171_catboost_122f4818 after adding dist_nearest_ew_km
- Log: `models/catboost/docs/logs/20261030T115103297657_catboost_935ed8ff.json`

### `20261030T115144776487_catboost_834e05ba`

- Command: train
- Parent: `20261030T113111382171_catboost_122f4818`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_ne_station_code", "round": 1}`
- Feature hash: `491bf0c825f4`
- Genome hash: `834e05baa80ecb2b`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `nearest_ne_station_code`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"early_stopping_rounds": 20, "learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 465.241864
- Search RMSE: 459.950291
- Decision: rejected. not promising because search RMSE 459.950291 is not at least 1% below 460.098274 from 20261030T113111382171_catboost_122f4818 after adding nearest_ne_station_code
- Log: `models/catboost/docs/logs/20261030T115144776487_catboost_834e05ba.json`

### `20261030T115226943786_catboost_c41ab30e`

- Command: train
- Parent: `20261030T113111382171_catboost_122f4818`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_ne_station_name", "round": 1}`
- Feature hash: `0b51c9b66206`
- Genome hash: `c41ab30ebec99bca`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `nearest_ne_station_name`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"early_stopping_rounds": 20, "learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 465.241864
- Search RMSE: 459.950291
- Decision: rejected. not promising because search RMSE 459.950291 is not at least 1% below 460.098274 from 20261030T113111382171_catboost_122f4818 after adding nearest_ne_station_name
- Log: `models/catboost/docs/logs/20261030T115226943786_catboost_c41ab30e.json`

### `20261030T115306935679_catboost_28ce5071`

- Command: train
- Parent: `20261030T113111382171_catboost_122f4818`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "dist_nearest_ne_km", "round": 1}`
- Feature hash: `58386a513add`
- Genome hash: `28ce5071fc1fe00a`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `dist_nearest_ne_km`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"early_stopping_rounds": 20, "learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 465.369929
- Search RMSE: 460.585488
- Decision: rejected. not promising because search RMSE 460.585488 is not at least 1% below 460.098274 from 20261030T113111382171_catboost_122f4818 after adding dist_nearest_ne_km
- Log: `models/catboost/docs/logs/20261030T115306935679_catboost_28ce5071.json`

### `20261030T115351303533_catboost_4bd137c1`

- Command: train
- Parent: `20261030T113111382171_catboost_122f4818`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_cc_station_code", "round": 1}`
- Feature hash: `e7df40e75f9a`
- Genome hash: `4bd137c109986bcf`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `nearest_cc_station_code`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"early_stopping_rounds": 20, "learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 465.325997
- Search RMSE: 460.504378
- Decision: rejected. not promising because search RMSE 460.504378 is not at least 1% below 460.098274 from 20261030T113111382171_catboost_122f4818 after adding nearest_cc_station_code
- Log: `models/catboost/docs/logs/20261030T115351303533_catboost_4bd137c1.json`

### `20261030T115449397147_catboost_548aaf38`

- Command: train
- Parent: `20261030T113111382171_catboost_122f4818`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_cc_station_name", "round": 1}`
- Feature hash: `0e0891a68917`
- Genome hash: `548aaf3898e3f942`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `nearest_cc_station_name`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"early_stopping_rounds": 20, "learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 465.325997
- Search RMSE: 460.504378
- Decision: rejected. not promising because search RMSE 460.504378 is not at least 1% below 460.098274 from 20261030T113111382171_catboost_122f4818 after adding nearest_cc_station_name
- Log: `models/catboost/docs/logs/20261030T115449397147_catboost_548aaf38.json`

### `20261030T115540269188_catboost_f767f61c`

- Command: train
- Parent: `20261030T113111382171_catboost_122f4818`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "dist_nearest_cc_km", "round": 1}`
- Feature hash: `73966c4f212f`
- Genome hash: `f767f61cad044a51`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `dist_nearest_cc_km`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"early_stopping_rounds": 20, "learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 465.119747
- Search RMSE: 459.356235
- Decision: rejected. not promising because search RMSE 459.356235 is not at least 1% below 460.098274 from 20261030T113111382171_catboost_122f4818 after adding dist_nearest_cc_km
- Log: `models/catboost/docs/logs/20261030T115540269188_catboost_f767f61c.json`

### `20261030T115632941008_catboost_722c09a3`

- Command: train
- Parent: `20261030T113111382171_catboost_122f4818`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_dt_station_code", "round": 1}`
- Feature hash: `248dc8a22f33`
- Genome hash: `722c09a31e6bdde4`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `nearest_dt_station_code`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"early_stopping_rounds": 20, "learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 465.225684
- Search RMSE: 458.448622
- Decision: rejected. not promising because search RMSE 458.448622 is not at least 1% below 460.098274 from 20261030T113111382171_catboost_122f4818 after adding nearest_dt_station_code
- Log: `models/catboost/docs/logs/20261030T115632941008_catboost_722c09a3.json`

### `20261030T115727850731_catboost_802119c1`

- Command: train
- Parent: `20261030T113111382171_catboost_122f4818`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_dt_station_name", "round": 1}`
- Feature hash: `48c44e2e40fe`
- Genome hash: `802119c1f41e0637`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `nearest_dt_station_name`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"early_stopping_rounds": 20, "learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 465.225684
- Search RMSE: 458.448622
- Decision: rejected. not promising because search RMSE 458.448622 is not at least 1% below 460.098274 from 20261030T113111382171_catboost_122f4818 after adding nearest_dt_station_name
- Log: `models/catboost/docs/logs/20261030T115727850731_catboost_802119c1.json`

### `20261030T115811785455_catboost_025d951d`

- Command: train
- Parent: `20261030T113111382171_catboost_122f4818`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "dist_nearest_dt_km", "round": 1}`
- Feature hash: `7e6bc08d4ff7`
- Genome hash: `025d951df04857f7`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `dist_nearest_dt_km`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"early_stopping_rounds": 20, "learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 465.198639
- Search RMSE: 460.002621
- Decision: rejected. not promising because search RMSE 460.002621 is not at least 1% below 460.098274 from 20261030T113111382171_catboost_122f4818 after adding dist_nearest_dt_km
- Log: `models/catboost/docs/logs/20261030T115811785455_catboost_025d951d.json`

### `20261030T115855413698_catboost_eddda3c6`

- Command: train
- Parent: `20261030T113111382171_catboost_122f4818`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_te_station_code", "round": 1}`
- Feature hash: `4f05b6ddefc5`
- Genome hash: `eddda3c6ad95f1c7`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `nearest_te_station_code`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"early_stopping_rounds": 20, "learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 465.288304
- Search RMSE: 460.009796
- Decision: rejected. not promising because search RMSE 460.009796 is not at least 1% below 460.098274 from 20261030T113111382171_catboost_122f4818 after adding nearest_te_station_code
- Log: `models/catboost/docs/logs/20261030T115855413698_catboost_eddda3c6.json`

### `20261030T115935195777_catboost_ded6bf19`

- Command: train
- Parent: `20261030T113111382171_catboost_122f4818`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_te_station_name", "round": 1}`
- Feature hash: `1139130c8f31`
- Genome hash: `ded6bf199ed4130f`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `nearest_te_station_name`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"early_stopping_rounds": 20, "learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 465.288304
- Search RMSE: 460.009796
- Decision: rejected. not promising because search RMSE 460.009796 is not at least 1% below 460.098274 from 20261030T113111382171_catboost_122f4818 after adding nearest_te_station_name
- Log: `models/catboost/docs/logs/20261030T115935195777_catboost_ded6bf19.json`

### `20261030T120013429997_catboost_b5456884`

- Command: train
- Parent: `20261030T113111382171_catboost_122f4818`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "dist_nearest_te_km", "round": 1}`
- Feature hash: `5cef183a61db`
- Genome hash: `b54568843a69b6c7`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `dist_nearest_te_km`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"early_stopping_rounds": 20, "learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 465.125719
- Search RMSE: 460.390795
- Decision: rejected. not promising because search RMSE 460.390795 is not at least 1% below 460.098274 from 20261030T113111382171_catboost_122f4818 after adding dist_nearest_te_km
- Log: `models/catboost/docs/logs/20261030T120013429997_catboost_b5456884.json`

### `20261030T120051394239_catboost_3740ccc8`

- Command: train
- Parent: `20261030T113111382171_catboost_122f4818`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "n_open_mrt_within_0_5km", "round": 1}`
- Feature hash: `5219387275ac`
- Genome hash: `3740ccc89577a79c`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `n_open_mrt_within_0_5km`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"early_stopping_rounds": 20, "learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 465.330929
- Search RMSE: 460.347075
- Decision: rejected. not promising because search RMSE 460.347075 is not at least 1% below 460.098274 from 20261030T113111382171_catboost_122f4818 after adding n_open_mrt_within_0_5km
- Log: `models/catboost/docs/logs/20261030T120051394239_catboost_3740ccc8.json`

### `20261030T120127952879_catboost_025adeb9`

- Command: train
- Parent: `20261030T113111382171_catboost_122f4818`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "n_open_mrt_within_1km", "round": 1}`
- Feature hash: `692f243ba340`
- Genome hash: `025adeb920bbfeb9`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `n_open_mrt_within_1km`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"early_stopping_rounds": 20, "learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 465.151990
- Search RMSE: 460.176422
- Decision: rejected. not promising because search RMSE 460.176422 is not at least 1% below 460.098274 from 20261030T113111382171_catboost_122f4818 after adding n_open_mrt_within_1km
- Log: `models/catboost/docs/logs/20261030T120127952879_catboost_025adeb9.json`

### `20261030T120208632064_catboost_ef7c8e9a`

- Command: train
- Parent: `20261030T113111382171_catboost_122f4818`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "n_open_mrt_within_2km", "round": 1}`
- Feature hash: `2e76b96e95c7`
- Genome hash: `ef7c8e9a8f042549`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `n_open_mrt_within_2km`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"early_stopping_rounds": 20, "learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 465.287155
- Search RMSE: 460.424816
- Decision: rejected. not promising because search RMSE 460.424816 is not at least 1% below 460.098274 from 20261030T113111382171_catboost_122f4818 after adding n_open_mrt_within_2km
- Log: `models/catboost/docs/logs/20261030T120208632064_catboost_ef7c8e9a.json`

### `20261030T120253255423_catboost_70852e9f`

- Command: train
- Parent: `20261030T113111382171_catboost_122f4818`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "dist_nearest_planned_mrt_km", "round": 1}`
- Feature hash: `d68a3dae1af5`
- Genome hash: `70852e9fca985dac`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `dist_nearest_planned_mrt_km`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"early_stopping_rounds": 20, "learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 465.281836
- Search RMSE: 461.096229
- Decision: rejected. not promising because search RMSE 461.096229 is not at least 1% below 460.098274 from 20261030T113111382171_catboost_122f4818 after adding dist_nearest_planned_mrt_km
- Log: `models/catboost/docs/logs/20261030T120253255423_catboost_70852e9f.json`

### `20261030T120338654865_catboost_a03ffeaf`

- Command: train
- Parent: `20261030T113111382171_catboost_122f4818`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "n_planned_mrt_within_1km", "round": 1}`
- Feature hash: `4fed35fb6804`
- Genome hash: `a03ffeafd19657fb`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `n_planned_mrt_within_1km`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"early_stopping_rounds": 20, "learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 465.374114
- Search RMSE: 460.687546
- Decision: rejected. not promising because search RMSE 460.687546 is not at least 1% below 460.098274 from 20261030T113111382171_catboost_122f4818 after adding n_planned_mrt_within_1km
- Log: `models/catboost/docs/logs/20261030T120338654865_catboost_a03ffeaf.json`

### `20261030T120426390007_catboost_304a2077`

- Command: train
- Parent: `20261030T113111382171_catboost_122f4818`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "n_planned_mrt_within_2km", "round": 1}`
- Feature hash: `112eaabb4cdb`
- Genome hash: `304a207747e9ec1e`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `n_planned_mrt_within_2km`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"early_stopping_rounds": 20, "learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 465.323760
- Search RMSE: 460.974898
- Decision: rejected. not promising because search RMSE 460.974898 is not at least 1% below 460.098274 from 20261030T113111382171_catboost_122f4818 after adding n_planned_mrt_within_2km
- Log: `models/catboost/docs/logs/20261030T120426390007_catboost_304a2077.json`

### `20261030T120509226847_catboost_5d38f259`

- Command: train
- Parent: `20261030T113111382171_catboost_122f4818`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_open_mrt_is_interchange", "round": 1}`
- Feature hash: `e8afb2136c5c`
- Genome hash: `5d38f259e7dab702`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `nearest_open_mrt_is_interchange`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"early_stopping_rounds": 20, "learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 465.370525
- Search RMSE: 461.150977
- Decision: rejected. not promising because search RMSE 461.150977 is not at least 1% below 460.098274 from 20261030T113111382171_catboost_122f4818 after adding nearest_open_mrt_is_interchange
- Log: `models/catboost/docs/logs/20261030T120509226847_catboost_5d38f259.json`

### `20261030T120546914139_catboost_576432da`

- Command: train
- Parent: `20261030T113111382171_catboost_122f4818`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "n_primary_within_2km", "round": 1}`
- Feature hash: `9f58d18094df`
- Genome hash: `576432da1bb1d9f6`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `n_primary_within_2km`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"early_stopping_rounds": 20, "learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 465.297484
- Search RMSE: 460.619804
- Decision: rejected. not promising because search RMSE 460.619804 is not at least 1% below 460.098274 from 20261030T113111382171_catboost_122f4818 after adding n_primary_within_2km
- Log: `models/catboost/docs/logs/20261030T120546914139_catboost_576432da.json`

### `20261030T120624775525_catboost_ab657841`

- Command: train
- Parent: `20261030T113111382171_catboost_122f4818`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "dist_nearest_secondary_km", "round": 1}`
- Feature hash: `3b84c255aaae`
- Genome hash: `ab6578418e135058`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `dist_nearest_secondary_km`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"early_stopping_rounds": 20, "learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 465.430667
- Search RMSE: 460.451102
- Decision: rejected. not promising because search RMSE 460.451102 is not at least 1% below 460.098274 from 20261030T113111382171_catboost_122f4818 after adding dist_nearest_secondary_km
- Log: `models/catboost/docs/logs/20261030T120624775525_catboost_ab657841.json`

### `20261030T120704386182_catboost_fbc44832`

- Command: train
- Parent: `20261030T113111382171_catboost_122f4818`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_secondary_name", "round": 1}`
- Feature hash: `4932501d7490`
- Genome hash: `fbc44832844c8071`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `nearest_secondary_name`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"early_stopping_rounds": 20, "learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 465.265373
- Search RMSE: 459.957143
- Decision: rejected. not promising because search RMSE 459.957143 is not at least 1% below 460.098274 from 20261030T113111382171_catboost_122f4818 after adding nearest_secondary_name
- Log: `models/catboost/docs/logs/20261030T120704386182_catboost_fbc44832.json`

### `20261030T120741261011_catboost_4bc0bfd8`

- Command: train
- Parent: `20261030T113111382171_catboost_122f4818`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "n_secondary_within_1km", "round": 1}`
- Feature hash: `f92f056e77f5`
- Genome hash: `4bc0bfd8589e8ad8`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `n_secondary_within_1km`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"early_stopping_rounds": 20, "learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 465.366087
- Search RMSE: 460.806897
- Decision: rejected. not promising because search RMSE 460.806897 is not at least 1% below 460.098274 from 20261030T113111382171_catboost_122f4818 after adding n_secondary_within_1km
- Log: `models/catboost/docs/logs/20261030T120741261011_catboost_4bc0bfd8.json`

### `20261030T120818312057_catboost_c30a454f`

- Command: train
- Parent: `20261030T113111382171_catboost_122f4818`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "dist_nearest_school_km", "round": 1}`
- Feature hash: `482ab5dda0e9`
- Genome hash: `c30a454f6e09c716`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `dist_nearest_school_km`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"early_stopping_rounds": 20, "learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 465.334871
- Search RMSE: 460.978429
- Decision: rejected. not promising because search RMSE 460.978429 is not at least 1% below 460.098274 from 20261030T113111382171_catboost_122f4818 after adding dist_nearest_school_km
- Log: `models/catboost/docs/logs/20261030T120818312057_catboost_c30a454f.json`

### `20261030T120858556717_catboost_92bdb8c3`

- Command: train
- Parent: `20261030T113111382171_catboost_122f4818`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_school_name", "round": 1}`
- Feature hash: `e657e5902602`
- Genome hash: `92bdb8c3be9bfe58`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `nearest_school_name`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"early_stopping_rounds": 20, "learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 465.382395
- Search RMSE: 460.059030
- Decision: rejected. not promising because search RMSE 460.059030 is not at least 1% below 460.098274 from 20261030T113111382171_catboost_122f4818 after adding nearest_school_name
- Log: `models/catboost/docs/logs/20261030T120858556717_catboost_92bdb8c3.json`

### `20261030T120935477418_catboost_9d111867`

- Command: train
- Parent: `20261030T113111382171_catboost_122f4818`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "n_school_within_1km", "round": 1}`
- Feature hash: `b9907fe19ce9`
- Genome hash: `9d111867a656e1ed`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `n_school_within_1km`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"early_stopping_rounds": 20, "learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 465.424960
- Search RMSE: 460.962170
- Decision: rejected. not promising because search RMSE 460.962170 is not at least 1% below 460.098274 from 20261030T113111382171_catboost_122f4818 after adding n_school_within_1km
- Log: `models/catboost/docs/logs/20261030T120935477418_catboost_9d111867.json`

### `20261030T121012329371_catboost_114b4dbd`

- Command: train
- Parent: `20261030T113111382171_catboost_122f4818`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "n_mall_within_1km", "round": 1}`
- Feature hash: `a98c9f4224a5`
- Genome hash: `114b4dbd76489423`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `n_mall_within_1km`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"early_stopping_rounds": 20, "learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 465.299251
- Search RMSE: 460.500652
- Decision: rejected. not promising because search RMSE 460.500652 is not at least 1% below 460.098274 from 20261030T113111382171_catboost_122f4818 after adding n_mall_within_1km
- Log: `models/catboost/docs/logs/20261030T121012329371_catboost_114b4dbd.json`

### `20261030T121050268521_catboost_6f3066b0`

- Command: train
- Parent: `20261030T113111382171_catboost_122f4818`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "n_mall_within_2km", "round": 1}`
- Feature hash: `282ff6f12a0d`
- Genome hash: `6f3066b0c5c0789f`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `n_mall_within_2km`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"early_stopping_rounds": 20, "learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 465.319166
- Search RMSE: 460.907629
- Decision: rejected. not promising because search RMSE 460.907629 is not at least 1% below 460.098274 from 20261030T113111382171_catboost_122f4818 after adding n_mall_within_2km
- Log: `models/catboost/docs/logs/20261030T121050268521_catboost_6f3066b0.json`

### `20261030T121128246478_catboost_72537d9f`

- Command: train
- Parent: `20261030T113111382171_catboost_122f4818`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "coe_a_premium_lag1", "round": 1}`
- Feature hash: `ca6251adee59`
- Genome hash: `72537d9f945c4aa4`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_a_premium_lag1`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"early_stopping_rounds": 20, "learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 465.293259
- Search RMSE: 461.509469
- Decision: rejected. not promising because search RMSE 461.509469 is not at least 1% below 460.098274 from 20261030T113111382171_catboost_122f4818 after adding coe_a_premium_lag1
- Log: `models/catboost/docs/logs/20261030T121128246478_catboost_72537d9f.json`

### `20261030T121205213368_catboost_aa27f659`

- Command: train
- Parent: `20261030T113111382171_catboost_122f4818`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "coe_a_premium_mom_change", "round": 1}`
- Feature hash: `ecd53635e5ea`
- Genome hash: `aa27f659e49a68bc`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_a_premium_mom_change`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"early_stopping_rounds": 20, "learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 465.496036
- Search RMSE: 460.797914
- Decision: rejected. not promising because search RMSE 460.797914 is not at least 1% below 460.098274 from 20261030T113111382171_catboost_122f4818 after adding coe_a_premium_mom_change
- Log: `models/catboost/docs/logs/20261030T121205213368_catboost_aa27f659.json`

### `20261030T121241701802_catboost_1fdb8d61`

- Command: train
- Parent: `20261030T113111382171_catboost_122f4818`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "coe_a_bids_per_quota_lag1", "round": 1}`
- Feature hash: `88b869c77fc2`
- Genome hash: `1fdb8d61852f09f7`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_a_bids_per_quota_lag1`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"early_stopping_rounds": 20, "learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 465.409276
- Search RMSE: 460.809543
- Decision: rejected. not promising because search RMSE 460.809543 is not at least 1% below 460.098274 from 20261030T113111382171_catboost_122f4818 after adding coe_a_bids_per_quota_lag1
- Log: `models/catboost/docs/logs/20261030T121241701802_catboost_1fdb8d61.json`

### `20261030T121319205984_catboost_f5b2ae4a`

- Command: train
- Parent: `20261030T113111382171_catboost_122f4818`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "coe_b_premium_lag1", "round": 1}`
- Feature hash: `aef8257585e0`
- Genome hash: `f5b2ae4a466cb30c`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_b_premium_lag1`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"early_stopping_rounds": 20, "learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 465.306616
- Search RMSE: 461.551050
- Decision: rejected. not promising because search RMSE 461.551050 is not at least 1% below 460.098274 from 20261030T113111382171_catboost_122f4818 after adding coe_b_premium_lag1
- Log: `models/catboost/docs/logs/20261030T121319205984_catboost_f5b2ae4a.json`

### `20261030T121356360188_catboost_5a1d42ef`

- Command: train
- Parent: `20261030T113111382171_catboost_122f4818`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "coe_b_premium_mom_change", "round": 1}`
- Feature hash: `8302b3652167`
- Genome hash: `5a1d42ef40617aef`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_b_premium_mom_change`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"early_stopping_rounds": 20, "learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 465.301489
- Search RMSE: 460.912855
- Decision: rejected. not promising because search RMSE 460.912855 is not at least 1% below 460.098274 from 20261030T113111382171_catboost_122f4818 after adding coe_b_premium_mom_change
- Log: `models/catboost/docs/logs/20261030T121356360188_catboost_5a1d42ef.json`

### `20261030T121432250808_catboost_658bc817`

- Command: train
- Parent: `20261030T113111382171_catboost_122f4818`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "coe_b_bids_per_quota_lag1", "round": 1}`
- Feature hash: `4a7482aab3c7`
- Genome hash: `658bc817d3f046cf`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_b_bids_per_quota_lag1`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"early_stopping_rounds": 20, "learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 465.322419
- Search RMSE: 460.818238
- Decision: rejected. not promising because search RMSE 460.818238 is not at least 1% below 460.098274 from 20261030T113111382171_catboost_122f4818 after adding coe_b_bids_per_quota_lag1
- Log: `models/catboost/docs/logs/20261030T121432250808_catboost_658bc817.json`

### `20261030T121509736203_catboost_4969e862`

- Command: train
- Parent: `20261030T113111382171_catboost_122f4818`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "coe_e_premium_lag1", "round": 1}`
- Feature hash: `de6005d5c05c`
- Genome hash: `4969e86254398b13`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_e_premium_lag1`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"early_stopping_rounds": 20, "learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 465.228687
- Search RMSE: 461.595410
- Decision: rejected. not promising because search RMSE 461.595410 is not at least 1% below 460.098274 from 20261030T113111382171_catboost_122f4818 after adding coe_e_premium_lag1
- Log: `models/catboost/docs/logs/20261030T121509736203_catboost_4969e862.json`

### `20261030T121546705336_catboost_3cfd19ac`

- Command: train
- Parent: `20261030T113111382171_catboost_122f4818`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "coe_e_premium_mom_change", "round": 1}`
- Feature hash: `387e56d7e868`
- Genome hash: `3cfd19ac3eed65c9`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_e_premium_mom_change`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"early_stopping_rounds": 20, "learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 465.318097
- Search RMSE: 460.741360
- Decision: rejected. not promising because search RMSE 460.741360 is not at least 1% below 460.098274 from 20261030T113111382171_catboost_122f4818 after adding coe_e_premium_mom_change
- Log: `models/catboost/docs/logs/20261030T121546705336_catboost_3cfd19ac.json`

### `20261030T121625320131_catboost_ed06c43a`

- Command: train
- Parent: `20261030T113111382171_catboost_122f4818`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "coe_e_bids_per_quota_lag1", "round": 1}`
- Feature hash: `8bb848ebe345`
- Genome hash: `ed06c43a6b54412c`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_e_bids_per_quota_lag1`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"early_stopping_rounds": 20, "learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 465.401135
- Search RMSE: 460.628054
- Decision: rejected. not promising because search RMSE 460.628054 is not at least 1% below 460.098274 from 20261030T113111382171_catboost_122f4818 after adding coe_e_bids_per_quota_lag1
- Log: `models/catboost/docs/logs/20261030T121625320131_catboost_ed06c43a.json`

### `20261030T121702612879_catboost_5c05bbd7`

- Command: train
- Parent: `20261030T113111382171_catboost_122f4818`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "coe_a_premium_roll3", "round": 1}`
- Feature hash: `b09c5f9ba37b`
- Genome hash: `5c05bbd70de5caba`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_a_premium_roll3`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"early_stopping_rounds": 20, "learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 465.347594
- Search RMSE: 460.622121
- Decision: rejected. not promising because search RMSE 460.622121 is not at least 1% below 460.098274 from 20261030T113111382171_catboost_122f4818 after adding coe_a_premium_roll3
- Log: `models/catboost/docs/logs/20261030T121702612879_catboost_5c05bbd7.json`

### `20261030T121739585785_catboost_0a67ca73`

- Command: train
- Parent: `20261030T113111382171_catboost_122f4818`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "coe_c_premium_lag1", "round": 1}`
- Feature hash: `7b8f900c807f`
- Genome hash: `0a67ca732eed5f2d`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_c_premium_lag1`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"early_stopping_rounds": 20, "learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 465.285624
- Search RMSE: 460.480480
- Decision: rejected. not promising because search RMSE 460.480480 is not at least 1% below 460.098274 from 20261030T113111382171_catboost_122f4818 after adding coe_c_premium_lag1
- Log: `models/catboost/docs/logs/20261030T121739585785_catboost_0a67ca73.json`

### `20261030T121817068202_catboost_a861ed79`

- Command: train
- Parent: `20261030T113111382171_catboost_122f4818`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "dist_nearest_future_line_open_km", "round": 1}`
- Feature hash: `109871ee8d0c`
- Genome hash: `a861ed7946593d02`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `dist_nearest_future_line_open_km`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"early_stopping_rounds": 20, "learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 465.213784
- Search RMSE: 460.734273
- Decision: rejected. not promising because search RMSE 460.734273 is not at least 1% below 460.098274 from 20261030T113111382171_catboost_122f4818 after adding dist_nearest_future_line_open_km
- Log: `models/catboost/docs/logs/20261030T121817068202_catboost_a861ed79.json`

### `20261030T121855099738_catboost_02cac93c`

- Command: train
- Parent: `20261030T113111382171_catboost_122f4818`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "dist_nearest_future_line_planned_km", "round": 1}`
- Feature hash: `b96b1461ee87`
- Genome hash: `02cac93c1c774404`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `dist_nearest_future_line_planned_km`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"early_stopping_rounds": 20, "learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 465.206041
- Search RMSE: 460.439330
- Decision: rejected. not promising because search RMSE 460.439330 is not at least 1% below 460.098274 from 20261030T113111382171_catboost_122f4818 after adding dist_nearest_future_line_planned_km
- Log: `models/catboost/docs/logs/20261030T121855099738_catboost_02cac93c.json`

### `20261030T121932307092_catboost_cc21d214`

- Command: train
- Parent: `20261030T113111382171_catboost_122f4818`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "n_hdb_blocks_within_500m", "round": 1}`
- Feature hash: `95bfdabe2d04`
- Genome hash: `cc21d214f4715c84`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `n_hdb_blocks_within_500m`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"early_stopping_rounds": 20, "learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 465.285085
- Search RMSE: 460.738548
- Decision: rejected. not promising because search RMSE 460.738548 is not at least 1% below 460.098274 from 20261030T113111382171_catboost_122f4818 after adding n_hdb_blocks_within_500m
- Log: `models/catboost/docs/logs/20261030T121932307092_catboost_cc21d214.json`

### `20261030T122009703901_catboost_7ccfb554`

- Command: train
- Parent: `20261030T113111382171_catboost_122f4818`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "dist_to_town_centroid_km", "round": 1}`
- Feature hash: `cbb5a64dccd3`
- Genome hash: `7ccfb5547e5bd96c`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `dist_to_town_centroid_km`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"early_stopping_rounds": 20, "learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 465.298494
- Search RMSE: 460.846615
- Decision: rejected. not promising because search RMSE 460.846615 is not at least 1% below 460.098274 from 20261030T113111382171_catboost_122f4818 after adding dist_to_town_centroid_km
- Log: `models/catboost/docs/logs/20261030T122009703901_catboost_7ccfb554.json`

### `20261030T122046628384_catboost_7c7a3980`

- Command: train
- Parent: `20261030T113111382171_catboost_122f4818`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "dist_to_cbd_km", "round": 1}`
- Feature hash: `3a81f2034fb9`
- Genome hash: `7c7a3980e4318ea3`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `dist_to_cbd_km`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"early_stopping_rounds": 20, "learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 465.130158
- Search RMSE: 459.144276
- Decision: rejected. not promising because search RMSE 459.144276 is not at least 1% below 460.098274 from 20261030T113111382171_catboost_122f4818 after adding dist_to_cbd_km
- Log: `models/catboost/docs/logs/20261030T122046628384_catboost_7c7a3980.json`

### `20261030T122123797017_catboost_2cd94291`

- Command: train
- Parent: `20261030T113111382171_catboost_122f4818`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "bank_basket_adj_close_lag1", "round": 1}`
- Feature hash: `40ef8a2b790f`
- Genome hash: `2cd94291dc41ecc1`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `bank_basket_adj_close_lag1`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"early_stopping_rounds": 20, "learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 465.323205
- Search RMSE: 460.442672
- Decision: rejected. not promising because search RMSE 460.442672 is not at least 1% below 460.098274 from 20261030T113111382171_catboost_122f4818 after adding bank_basket_adj_close_lag1
- Log: `models/catboost/docs/logs/20261030T122123797017_catboost_2cd94291.json`

### `20261030T122204129173_catboost_232df95d`

- Command: train
- Parent: `20261030T113111382171_catboost_122f4818`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "bank_basket_return_lag1", "round": 1}`
- Feature hash: `52b8a977b3a9`
- Genome hash: `232df95d81e6587c`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `bank_basket_return_lag1`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"early_stopping_rounds": 20, "learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 465.380038
- Search RMSE: 461.364940
- Decision: rejected. not promising because search RMSE 461.364940 is not at least 1% below 460.098274 from 20261030T113111382171_catboost_122f4818 after adding bank_basket_return_lag1
- Log: `models/catboost/docs/logs/20261030T122204129173_catboost_232df95d.json`

### `20261030T122248446201_catboost_5cce6696`

- Command: train
- Parent: `20261030T113111382171_catboost_122f4818`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "property_basket_adj_close_lag1", "round": 1}`
- Feature hash: `3b1ea68b7e46`
- Genome hash: `5cce6696754849c8`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `property_basket_adj_close_lag1`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"early_stopping_rounds": 20, "learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 465.363870
- Search RMSE: 462.347280
- Decision: rejected. not promising because search RMSE 462.347280 is not at least 1% below 460.098274 from 20261030T113111382171_catboost_122f4818 after adding property_basket_adj_close_lag1
- Log: `models/catboost/docs/logs/20261030T122248446201_catboost_5cce6696.json`

### `20261030T122333191026_catboost_c2aab013`

- Command: train
- Parent: `20261030T113111382171_catboost_122f4818`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "property_basket_return_lag1", "round": 1}`
- Feature hash: `d99f88cbdcb7`
- Genome hash: `c2aab0138bc8e06d`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `property_basket_return_lag1`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"early_stopping_rounds": 20, "learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 465.346049
- Search RMSE: 460.879458
- Decision: rejected. not promising because search RMSE 460.879458 is not at least 1% below 460.098274 from 20261030T113111382171_catboost_122f4818 after adding property_basket_return_lag1
- Log: `models/catboost/docs/logs/20261030T122333191026_catboost_c2aab013.json`

### `20261030T122418388296_catboost_28353a2a`

- Command: train
- Parent: `20261030T113111382171_catboost_122f4818`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "log_dist_nearest_open_mrt_km", "round": 1}`
- Feature hash: `21c7244e6748`
- Genome hash: `28353a2a112ca1ea`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `log_dist_nearest_open_mrt_km`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"early_stopping_rounds": 20, "learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 464.959719
- Search RMSE: 459.907811
- Decision: rejected. not promising because search RMSE 459.907811 is not at least 1% below 460.098274 from 20261030T113111382171_catboost_122f4818 after adding log_dist_nearest_open_mrt_km
- Log: `models/catboost/docs/logs/20261030T122418388296_catboost_28353a2a.json`

### `20261030T122500681076_catboost_01d19b54`

- Command: train
- Parent: `20261030T113111382171_catboost_122f4818`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "is_central_region", "round": 1}`
- Feature hash: `e47e0c0ed690`
- Genome hash: `01d19b548d370552`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `is_central_region`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"early_stopping_rounds": 20, "learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 465.324428
- Search RMSE: 459.979825
- Decision: rejected. not promising because search RMSE 459.979825 is not at least 1% below 460.098274 from 20261030T113111382171_catboost_122f4818 after adding is_central_region
- Log: `models/catboost/docs/logs/20261030T122500681076_catboost_01d19b54.json`

### `20261030T122540112952_catboost_e03acd47`

- Command: train
- Parent: `20261030T113111382171_catboost_122f4818`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "building_age_x_central", "round": 1}`
- Feature hash: `a5a53d0ae4a0`
- Genome hash: `e03acd477b31bae4`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `building_age_x_central`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"early_stopping_rounds": 20, "learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 465.210556
- Search RMSE: 459.494054
- Decision: rejected. not promising because search RMSE 459.494054 is not at least 1% below 460.098274 from 20261030T113111382171_catboost_122f4818 after adding building_age_x_central
- Log: `models/catboost/docs/logs/20261030T122540112952_catboost_e03acd47.json`

### `20261030T122626239335_catboost_ba11fc49`

- Command: train
- Parent: `20261030T113111382171_catboost_122f4818`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "mean_rent_postal_sector", "round": 1}`
- Feature hash: `e7393246f8d5`
- Genome hash: `ba11fc4960929a0a`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `mean_rent_postal_sector`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"early_stopping_rounds": 20, "learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 465.588833
- Search RMSE: 460.060829
- Decision: rejected. not promising because search RMSE 460.060829 is not at least 1% below 460.098274 from 20261030T113111382171_catboost_122f4818 after adding mean_rent_postal_sector
- Log: `models/catboost/docs/logs/20261030T122626239335_catboost_ba11fc49.json`

### `20261030T122712602596_catboost_b7c1a6e3`

- Command: train
- Parent: `20261030T113111382171_catboost_122f4818`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "mean_rent_postal_district", "round": 1}`
- Feature hash: `4d4cc5ffe4d6`
- Genome hash: `b7c1a6e342d65e88`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `mean_rent_postal_district`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"early_stopping_rounds": 20, "learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 465.427144
- Search RMSE: 459.578690
- Decision: rejected. not promising because search RMSE 459.578690 is not at least 1% below 460.098274 from 20261030T113111382171_catboost_122f4818 after adding mean_rent_postal_district
- Log: `models/catboost/docs/logs/20261030T122712602596_catboost_b7c1a6e3.json`

### `20261030T122755364585_catboost_7c922927`

- Command: train
- Parent: `20261030T113111382171_catboost_122f4818`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "mean_rent_nearest_mall", "round": 1}`
- Feature hash: `471d8ee447c6`
- Genome hash: `7c922927fea04996`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `mean_rent_nearest_mall`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"early_stopping_rounds": 20, "learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 465.614112
- Search RMSE: 460.032308
- Decision: rejected. not promising because search RMSE 460.032308 is not at least 1% below 460.098274 from 20261030T113111382171_catboost_122f4818 after adding mean_rent_nearest_mall
- Log: `models/catboost/docs/logs/20261030T122755364585_catboost_7c922927.json`

### `20261030T122840720382_catboost_e5f5a639`

- Command: train
- Parent: `20261030T113111382171_catboost_122f4818`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "mean_rent_nearest_mrt", "round": 1}`
- Feature hash: `c9dd7135f0c6`
- Genome hash: `e5f5a639d2bc3dc6`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `mean_rent_nearest_mrt`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"early_stopping_rounds": 20, "learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 465.545966
- Search RMSE: 460.320635
- Decision: rejected. not promising because search RMSE 460.320635 is not at least 1% below 460.098274 from 20261030T113111382171_catboost_122f4818 after adding mean_rent_nearest_mrt
- Log: `models/catboost/docs/logs/20261030T122840720382_catboost_e5f5a639.json`

### `20261030T122926291942_catboost_389b236e`

- Command: train
- Parent: `20261030T113111382171_catboost_122f4818`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "mean_rent_nearest_primary", "round": 1}`
- Feature hash: `b23749c91b13`
- Genome hash: `389b236e0dd7434b`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `mean_rent_nearest_primary`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"early_stopping_rounds": 20, "learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 465.511994
- Search RMSE: 459.751576
- Decision: rejected. not promising because search RMSE 459.751576 is not at least 1% below 460.098274 from 20261030T113111382171_catboost_122f4818 after adding mean_rent_nearest_primary
- Log: `models/catboost/docs/logs/20261030T122926291942_catboost_389b236e.json`

### `20261030T123015319236_catboost_26e44cf9`

- Command: train
- Parent: `20261030T113111382171_catboost_122f4818`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "mean_rent_nearest_secondary", "round": 1}`
- Feature hash: `8b3befbcf20d`
- Genome hash: `26e44cf9daa8485a`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `mean_rent_nearest_secondary`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"early_stopping_rounds": 20, "learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 465.701718
- Search RMSE: 460.020997
- Decision: rejected. not promising because search RMSE 460.020997 is not at least 1% below 460.098274 from 20261030T113111382171_catboost_122f4818 after adding mean_rent_nearest_secondary
- Log: `models/catboost/docs/logs/20261030T123015319236_catboost_26e44cf9.json`

### `20261030T123149326955_catboost_7ca41a9e`

- Command: train
- Parent: `20261030T113111382171_catboost_122f4818`
- Generation: None
- Mutation: `{"axis": "finalize"}`
- Feature hash: `170e95b30b54`
- Genome hash: `7ca41a9ee3cdf8c9`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"early_stopping_rounds": 50, "learning_rate": 0.05, "max_depth": 8, "n_estimators": 2000}`
- Train RMSE: 461.003342
- Search RMSE: 456.042092
- Decision: accepted. final fit on the features of 20261030T113111382171_catboost_122f4818; this stage does not select columns
- Log: `models/catboost/docs/logs/20261030T123149326955_catboost_7ca41a9e.json`

### `20261030T123328062936_catboost_104b30bf`

- Command: submit
- Parent: `20261030T123149326955_catboost_7ca41a9e`
- Generation: None
- Mutation: `{"axis": "submit"}`
- Feature hash: `170e95b30b54`
- Genome hash: `104b30bf641c016c`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"early_stopping_rounds": 0, "learning_rate": 0.05, "max_depth": 8, "n_estimators": 367}`
- Train RMSE: 459.082217
- Decision: refit of 20261030T123149326955_catboost_7ca41a9e on every labeled month; this stage does not select columns and was not scored on held-out MONTHLY_RENT
- Log: `models/catboost/docs/logs/20261030T123328062936_catboost_104b30bf.json`

### `20261030T124215167507_catboost_417a4ec8`

- Command: train
- Parent: `20261030T113111382171_catboost_122f4818`
- Generation: None
- Mutation: `{"axis": "features", "change": "set_c", "columns": ["latitude", "postal_code", "max_floor", "region", "dist_nearest_open_mrt_km", "nearest_mall_name", "nearest_ns_station_code", "nearest_ns_station_name", "nearest_ew_station_code", "nearest_ew_station_name", "dist_nearest_ew_km", "nearest_ne_station_code", "nearest_ne_station_name", "dist_nearest_cc_km", "nearest_dt_station_code", "nearest_dt_station_name", "dist_nearest_dt_km", "nearest_te_station_code", "nearest_te_station_name", "nearest_secondary_name", "nearest_school_name", "dist_to_cbd_km", "log_dist_nearest_open_mrt_km", "is_central_region", "building_age_x_central", "mean_rent_postal_sector", "mean_rent_postal_district", "mean_rent_nearest_mall", "mean_rent_nearest_primary", "mean_rent_nearest_secondary"], "round": 1}`
- Feature hash: `e8f777332831`
- Genome hash: `417a4ec8fb716101`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `latitude`, `postal_code`, `max_floor`, `region`, `dist_nearest_open_mrt_km`, `nearest_mall_name`, `nearest_ns_station_code`, `nearest_ns_station_name`, `nearest_ew_station_code`, `nearest_ew_station_name`, `dist_nearest_ew_km`, `nearest_ne_station_code`, `nearest_ne_station_name`, `dist_nearest_cc_km`, `nearest_dt_station_code`, `nearest_dt_station_name`, `dist_nearest_dt_km`, `nearest_te_station_code`, `nearest_te_station_name`, `nearest_secondary_name`, `nearest_school_name`, `dist_to_cbd_km`, `log_dist_nearest_open_mrt_km`, `is_central_region`, `building_age_x_central`, `mean_rent_postal_sector`, `mean_rent_postal_district`, `mean_rent_nearest_mall`, `mean_rent_nearest_primary`, `mean_rent_nearest_secondary`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"early_stopping_rounds": 20, "learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 462.182823
- Search RMSE: 452.618834
- Decision: accepted. replaced best 20261030T113111382171_catboost_122f4818 search RMSE 460.098274 with 452.618834
- Log: `models/catboost/docs/logs/20261030T124215167507_catboost_417a4ec8.json`

### `20261030T124934765682_catboost_84d3c3ba`

- Command: train
- Parent: `20261030T124215167507_catboost_417a4ec8`
- Generation: None
- Mutation: `{"axis": "finalize"}`
- Feature hash: `e8f777332831`
- Genome hash: `84d3c3ba059e896c`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `latitude`, `postal_code`, `max_floor`, `region`, `dist_nearest_open_mrt_km`, `nearest_mall_name`, `nearest_ns_station_code`, `nearest_ns_station_name`, `nearest_ew_station_code`, `nearest_ew_station_name`, `dist_nearest_ew_km`, `nearest_ne_station_code`, `nearest_ne_station_name`, `dist_nearest_cc_km`, `nearest_dt_station_code`, `nearest_dt_station_name`, `dist_nearest_dt_km`, `nearest_te_station_code`, `nearest_te_station_name`, `nearest_secondary_name`, `nearest_school_name`, `dist_to_cbd_km`, `log_dist_nearest_open_mrt_km`, `is_central_region`, `building_age_x_central`, `mean_rent_postal_sector`, `mean_rent_postal_district`, `mean_rent_nearest_mall`, `mean_rent_nearest_primary`, `mean_rent_nearest_secondary`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"early_stopping_rounds": 50, "learning_rate": 0.05, "max_depth": 8, "n_estimators": 2000}`
- Train RMSE: 454.864026
- Search RMSE: 449.473470
- Decision: accepted. final fit on the features of 20261030T124215167507_catboost_417a4ec8; this stage does not select columns
- Log: `models/catboost/docs/logs/20261030T124934765682_catboost_84d3c3ba.json`

### `20261030T125529553943_catboost_33565bb8`

- Command: submit
- Parent: `20261030T124934765682_catboost_84d3c3ba`
- Generation: None
- Mutation: `{"axis": "submit"}`
- Feature hash: `e8f777332831`
- Genome hash: `33565bb85f046eb9`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `latitude`, `postal_code`, `max_floor`, `region`, `dist_nearest_open_mrt_km`, `nearest_mall_name`, `nearest_ns_station_code`, `nearest_ns_station_name`, `nearest_ew_station_code`, `nearest_ew_station_name`, `dist_nearest_ew_km`, `nearest_ne_station_code`, `nearest_ne_station_name`, `dist_nearest_cc_km`, `nearest_dt_station_code`, `nearest_dt_station_name`, `dist_nearest_dt_km`, `nearest_te_station_code`, `nearest_te_station_name`, `nearest_secondary_name`, `nearest_school_name`, `dist_to_cbd_km`, `log_dist_nearest_open_mrt_km`, `is_central_region`, `building_age_x_central`, `mean_rent_postal_sector`, `mean_rent_postal_district`, `mean_rent_nearest_mall`, `mean_rent_nearest_primary`, `mean_rent_nearest_secondary`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"early_stopping_rounds": 0, "learning_rate": 0.05, "max_depth": 8, "n_estimators": 443}`
- Train RMSE: 452.039402
- Decision: refit of 20261030T124934765682_catboost_84d3c3ba on every labeled month; this stage does not select columns and was not scored on held-out MONTHLY_RENT
- Log: `models/catboost/docs/logs/20261030T125529553943_catboost_33565bb8.json`
<!-- lineage:end -->

## Predictions

<!-- predictions:start -->
### `20261030T123150144080_catboost_predict0`

- Champion run: `20261030T123149326955_catboost_7ca41a9e`
- Rows: 50000
- Mean: 3109.3475981966913
- Std: 474.20503024144915
- Min: 1699.1460967710282
- Max: 5308.595759791538
- Output: `outputs/predictions/catboost_final.csv`
- Refit on all labeled months: False
- Log: `models/catboost/docs/logs/20261030T123150144080_catboost_predict0.json`

### `20261030T123156481117_catboost_predict0`

- Champion run: `20261030T123149326955_catboost_7ca41a9e`
- Rows: 50000
- Mean: 3109.3475981966913
- Std: 474.20503024144915
- Min: 1699.1460967710282
- Max: 5308.595759791538
- Output: `outputs/predictions/catboost_final.csv`
- Refit on all labeled months: False
- Log: `models/catboost/docs/logs/20261030T123156481117_catboost_predict0.json`

### `20261030T123328675492_catboost_predict0`

- Champion run: `20261030T123328062936_catboost_104b30bf`
- Rows: 50000
- Mean: 3136.4495507830484
- Std: 464.3252338508159
- Min: 1811.1422035067953
- Max: 5215.508066433514
- Output: `outputs/predictions/catboost_refit.csv`
- Refit on all labeled months: True
- Log: `models/catboost/docs/logs/20261030T123328675492_catboost_predict0.json`

### `20261030T124936176451_catboost_predict0`

- Champion run: `20261030T124934765682_catboost_84d3c3ba`
- Rows: 50000
- Mean: 2978.608746438637
- Std: 437.772330152874
- Min: 1538.1221939665263
- Max: 5081.127376457
- Output: `outputs/predictions/catboost_final.csv`
- Refit on all labeled months: False
- Log: `models/catboost/docs/logs/20261030T124936176451_catboost_predict0.json`

### `20261030T124943041897_catboost_predict0`

- Champion run: `20261030T124934765682_catboost_84d3c3ba`
- Rows: 50000
- Mean: 2978.608746438637
- Std: 437.772330152874
- Min: 1538.1221939665263
- Max: 5081.127376457
- Output: `outputs/predictions/catboost_final.csv`
- Refit on all labeled months: False
- Log: `models/catboost/docs/logs/20261030T124943041897_catboost_predict0.json`

### `20261030T125530690647_catboost_predict0`

- Champion run: `20261030T125529553943_catboost_33565bb8`
- Rows: 50000
- Mean: 3080.0381109945774
- Std: 440.8188032150366
- Min: 1622.2010714387823
- Max: 5106.37940571552
- Output: `outputs/predictions/catboost_refit.csv`
- Refit on all labeled months: True
- Log: `models/catboost/docs/logs/20261030T125530690647_catboost_predict0.json`
<!-- predictions:end -->

