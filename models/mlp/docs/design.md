# MLP

## Architecture gene

The spec is `hidden_sizes` and `dropout`. Regularize raises dropout by 0.1 up to 0.5. Capacity doubles the first hidden layer up to 512, then appends a layer copied from the last width, up to 4 layers.

## Adjustment rules

The evolve command reads `docs/logs` and `docs/champion.json`, then proposes up to three mutants of the champion. Each mutant changes one gene: the feature subset, the preprocess spec, or the architecture. A genome already present in the logs is not trained again.

Feature changes alternate by generation. An odd generation adds the unused pool column with the largest absolute correlation to the champion residual on the search months. An even generation drops the column with the smallest permutation importance on a search sample of at most 2,000 rows. When that single-column move was already tried from the same parent, and every column of a feature family is inside the pool, an odd generation adds that family as a block and an even generation removes it. A feature set always keeps at least one column.

Preprocess changes the first knob not yet tried from this parent, in this order: numeric_impute (median, mean, zero); scale (standard, none, minmax); categorical (onehot, ordinal, target, bin); rare_min_count (10, 30, 100). The new value is the first choice in that list that differs from the current value.

Architecture regularizes when (search RMSE - train RMSE) / train RMSE is above 0.15. Otherwise it adds capacity, inside the cap in this model's note.

A mutant replaces the champion only when its search RMSE is at least 1% lower. The loop stops at the generation cap, when a generation accepts nothing, or when every proposed mutant is already in the logs. Confirm RMSE is scored once after search stops and is not an acceptance signal. Predict writes a summary and does not change the genome.

Labeled months are cut in time. Confirm is the last `val_months` approval months. Search is the `val_months` immediately before confirm. Fit is every earlier month. The preprocessor is fit on the fit months only.

## Current champion

<!-- champion:start -->
- Run: `20261022T202913337555_mlp_3f1940b8`
- Log: `models/mlp/docs/logs/20261022T202913337555_mlp_3f1940b8.json`
- Feature hash: `8d04a6ec2a11`
- Genome hash: `3f1940b884101451`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `latitude`, `longitude`, `max_floor`, `year_completed`, `planning_area`, `region`, `building_age_years`, `lease_vs_completion_gap`, `flat_age_from_lease`, `dist_nearest_open_mrt_km`, `dist_nearest_primary_km`, `n_primary_within_1km`, `dist_nearest_mall_km`, `nearest_mall_name`, `nearest_ns_station_code`, `nearest_ns_station_name`, `dist_nearest_ns_km`, `dist_nearest_ew_km`, `nearest_ne_station_code`, `nearest_ne_station_name`, `dist_nearest_ne_km`, `nearest_cc_station_code`, `nearest_cc_station_name`, `dist_nearest_cc_km`, `dist_nearest_dt_km`, `dist_nearest_te_km`, `n_open_mrt_within_0_5km`, `n_open_mrt_within_1km`, `n_open_mrt_within_2km`, `dist_nearest_planned_mrt_km`, `n_planned_mrt_within_1km`, `n_planned_mrt_within_2km`, `nearest_open_mrt_is_interchange`, `n_primary_within_2km`, `dist_nearest_secondary_km`, `nearest_secondary_name`, `n_secondary_within_1km`, `dist_nearest_school_km`, `nearest_school_name`, `n_school_within_1km`, `n_mall_within_1km`, `n_mall_within_2km`, `coe_a_premium_lag1`, `coe_a_premium_mom_change`, `coe_a_bids_per_quota_lag1`, `coe_b_premium_lag1`, `coe_b_premium_mom_change`, `coe_b_bids_per_quota_lag1`, `coe_e_premium_lag1`, `coe_e_premium_mom_change`, `coe_e_bids_per_quota_lag1`, `coe_a_premium_roll3`, `coe_c_premium_lag1`, `dist_nearest_future_line_open_km`, `dist_nearest_future_line_planned_km`, `n_hdb_blocks_within_500m`, `dist_to_town_centroid_km`, `dist_to_cbd_km`, `bank_basket_adj_close_lag1`, `bank_basket_return_lag1`, `property_basket_adj_close_lag1`, `property_basket_return_lag1`, `log_dist_nearest_open_mrt_km`, `is_central_region`, `building_age_x_central`, `mean_rent_postal_sector`, `mean_rent_postal_district`, `mean_rent_nearest_mall`, `mean_rent_nearest_mrt`, `mean_rent_nearest_primary`, `mean_rent_nearest_secondary`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 489.836979
- Search RMSE: 467.263467
- Confirm RMSE: 480.341288
- Confirm log: `models/mlp/docs/logs/20261022T202913615928_mlp_3f1940b8.json`
- Row cap: full training table
<!-- champion:end -->

## Lineage

<!-- lineage:start -->
Regenerated from `docs/logs`. Do not edit the metrics in this section by hand.

### `20261022T183131011278_mlp_9a5f1229`

- Command: train
- Parent: `None`
- Generation: None
- Mutation: `{"axis": "train"}`
- Feature hash: `170e95b30b54`
- Genome hash: `9a5f1229ad2100b2`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 641.508832
- Search RMSE: 650.418686
- Decision: accepted. first saved best; there was no earlier snapshot
- Log: `models/mlp/docs/logs/20261022T183131011278_mlp_9a5f1229.json`

### `20261022T195459744063_mlp_09b2c331`

- Command: train
- Parent: `20261022T183131011278_mlp_9a5f1229`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "latitude", "round": 1}`
- Feature hash: `99b4d5d7749f`
- Genome hash: `09b2c331e4fc469b`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `latitude`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 514.473967
- Search RMSE: 460.283263
- Decision: accepted. promising because search RMSE 460.283263 is at least 1% below 650.418686 from 20261022T183131011278_mlp_9a5f1229 after adding latitude
- Log: `models/mlp/docs/logs/20261022T195459744063_mlp_09b2c331.json`

### `20261022T195526700361_mlp_dd22741b`

- Command: train
- Parent: `20261022T183131011278_mlp_9a5f1229`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "longitude", "round": 1}`
- Feature hash: `c27505a182db`
- Genome hash: `dd22741b81697c1d`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `longitude`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 514.300636
- Search RMSE: 462.044866
- Decision: accepted. promising because search RMSE 462.044866 is at least 1% below 650.418686 from 20261022T183131011278_mlp_9a5f1229 after adding longitude
- Log: `models/mlp/docs/logs/20261022T195526700361_mlp_dd22741b.json`

### `20261022T195551073139_mlp_d9e69418`

- Command: train
- Parent: `20261022T183131011278_mlp_9a5f1229`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "postal_code", "round": 1}`
- Feature hash: `e62a952e566d`
- Genome hash: `d9e694187515add8`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `postal_code`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 641.784051
- Search RMSE: 653.104612
- Decision: rejected. not promising because search RMSE 653.104612 is not at least 1% below 650.418686 from 20261022T183131011278_mlp_9a5f1229 after adding postal_code
- Log: `models/mlp/docs/logs/20261022T195551073139_mlp_d9e69418.json`

### `20261022T195617910708_mlp_80dc4143`

- Command: train
- Parent: `20261022T183131011278_mlp_9a5f1229`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "max_floor", "round": 1}`
- Feature hash: `6d16b7bd0382`
- Genome hash: `80dc41431c95a68d`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `max_floor`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 515.651605
- Search RMSE: 462.545421
- Decision: accepted. promising because search RMSE 462.545421 is at least 1% below 650.418686 from 20261022T183131011278_mlp_9a5f1229 after adding max_floor
- Log: `models/mlp/docs/logs/20261022T195617910708_mlp_80dc4143.json`

### `20261022T195643986307_mlp_03673fc5`

- Command: train
- Parent: `20261022T183131011278_mlp_9a5f1229`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "year_completed", "round": 1}`
- Feature hash: `b2d2d8a929b6`
- Genome hash: `03673fc53854a21a`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `year_completed`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 514.355012
- Search RMSE: 461.580401
- Decision: accepted. promising because search RMSE 461.580401 is at least 1% below 650.418686 from 20261022T183131011278_mlp_9a5f1229 after adding year_completed
- Log: `models/mlp/docs/logs/20261022T195643986307_mlp_03673fc5.json`

### `20261022T195703580490_mlp_13181839`

- Command: train
- Parent: `20261022T183131011278_mlp_9a5f1229`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "subzone", "round": 1}`
- Feature hash: `19ad5218a78e`
- Genome hash: `13181839f0d9e930`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `subzone`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 652.896946
- Search RMSE: 678.586129
- Decision: rejected. not promising because search RMSE 678.586129 is not at least 1% below 650.418686 from 20261022T183131011278_mlp_9a5f1229 after adding subzone
- Log: `models/mlp/docs/logs/20261022T195703580490_mlp_13181839.json`

### `20261022T195730666301_mlp_20e0aa21`

- Command: train
- Parent: `20261022T183131011278_mlp_9a5f1229`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "planning_area", "round": 1}`
- Feature hash: `212df8d837f8`
- Genome hash: `20e0aa21e1da6c6d`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `planning_area`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 510.075110
- Search RMSE: 458.548010
- Decision: accepted. promising because search RMSE 458.548010 is at least 1% below 650.418686 from 20261022T183131011278_mlp_9a5f1229 after adding planning_area
- Log: `models/mlp/docs/logs/20261022T195730666301_mlp_20e0aa21.json`

### `20261022T195758080948_mlp_3ff94585`

- Command: train
- Parent: `20261022T183131011278_mlp_9a5f1229`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "region", "round": 1}`
- Feature hash: `fd9247992a0a`
- Genome hash: `3ff94585b82287eb`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `region`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 518.124756
- Search RMSE: 460.788001
- Decision: accepted. promising because search RMSE 460.788001 is at least 1% below 650.418686 from 20261022T183131011278_mlp_9a5f1229 after adding region
- Log: `models/mlp/docs/logs/20261022T195758080948_mlp_3ff94585.json`

### `20261022T195825231356_mlp_2c526523`

- Command: train
- Parent: `20261022T183131011278_mlp_9a5f1229`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "building_age_years", "round": 1}`
- Feature hash: `c234d75ff5d3`
- Genome hash: `2c5265236f2344fe`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `building_age_years`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 512.661989
- Search RMSE: 459.655193
- Decision: accepted. promising because search RMSE 459.655193 is at least 1% below 650.418686 from 20261022T183131011278_mlp_9a5f1229 after adding building_age_years
- Log: `models/mlp/docs/logs/20261022T195825231356_mlp_2c526523.json`

### `20261022T195851967555_mlp_d399da0d`

- Command: train
- Parent: `20261022T183131011278_mlp_9a5f1229`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "lease_vs_completion_gap", "round": 1}`
- Feature hash: `d33f593517bd`
- Genome hash: `d399da0d3dfcc91e`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `lease_vs_completion_gap`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 514.154578
- Search RMSE: 461.757841
- Decision: accepted. promising because search RMSE 461.757841 is at least 1% below 650.418686 from 20261022T183131011278_mlp_9a5f1229 after adding lease_vs_completion_gap
- Log: `models/mlp/docs/logs/20261022T195851967555_mlp_d399da0d.json`

### `20261022T195918374266_mlp_b6798e9d`

- Command: train
- Parent: `20261022T183131011278_mlp_9a5f1229`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "flat_age_from_lease", "round": 1}`
- Feature hash: `0ae4cfca946e`
- Genome hash: `b6798e9deac30a1e`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `flat_age_from_lease`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 512.842515
- Search RMSE: 459.992040
- Decision: accepted. promising because search RMSE 459.992040 is at least 1% below 650.418686 from 20261022T183131011278_mlp_9a5f1229 after adding flat_age_from_lease
- Log: `models/mlp/docs/logs/20261022T195918374266_mlp_b6798e9d.json`

### `20261022T195944685717_mlp_0a9d6d02`

- Command: train
- Parent: `20261022T183131011278_mlp_9a5f1229`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "dist_nearest_open_mrt_km", "round": 1}`
- Feature hash: `0c2594d85422`
- Genome hash: `0a9d6d0212cb52e8`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `dist_nearest_open_mrt_km`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 514.908764
- Search RMSE: 461.727800
- Decision: accepted. promising because search RMSE 461.727800 is at least 1% below 650.418686 from 20261022T183131011278_mlp_9a5f1229 after adding dist_nearest_open_mrt_km
- Log: `models/mlp/docs/logs/20261022T195944685717_mlp_0a9d6d02.json`

### `20261022T200006802034_mlp_ba4a4046`

- Command: train
- Parent: `20261022T183131011278_mlp_9a5f1229`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_open_mrt_code", "round": 1}`
- Feature hash: `42451fae9f20`
- Genome hash: `ba4a4046e0aa83a0`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `nearest_open_mrt_code`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 640.340163
- Search RMSE: 644.949917
- Decision: rejected. not promising because search RMSE 644.949917 is not at least 1% below 650.418686 from 20261022T183131011278_mlp_9a5f1229 after adding nearest_open_mrt_code
- Log: `models/mlp/docs/logs/20261022T200006802034_mlp_ba4a4046.json`

### `20261022T200028949314_mlp_7968b412`

- Command: train
- Parent: `20261022T183131011278_mlp_9a5f1229`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_open_mrt_name", "round": 1}`
- Feature hash: `57326fd029e4`
- Genome hash: `7968b412fade2182`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `nearest_open_mrt_name`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 640.340182
- Search RMSE: 644.949973
- Decision: rejected. not promising because search RMSE 644.949973 is not at least 1% below 650.418686 from 20261022T183131011278_mlp_9a5f1229 after adding nearest_open_mrt_name
- Log: `models/mlp/docs/logs/20261022T200028949314_mlp_7968b412.json`

### `20261022T200051091330_mlp_19be1524`

- Command: train
- Parent: `20261022T183131011278_mlp_9a5f1229`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_open_mrt_line", "round": 1}`
- Feature hash: `f6e2fc2120da`
- Genome hash: `19be1524646cd1cd`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `nearest_open_mrt_line`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 642.399277
- Search RMSE: 651.209110
- Decision: rejected. not promising because search RMSE 651.209110 is not at least 1% below 650.418686 from 20261022T183131011278_mlp_9a5f1229 after adding nearest_open_mrt_line
- Log: `models/mlp/docs/logs/20261022T200051091330_mlp_19be1524.json`

### `20261022T200117471031_mlp_025467c1`

- Command: train
- Parent: `20261022T183131011278_mlp_9a5f1229`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "dist_nearest_primary_km", "round": 1}`
- Feature hash: `ec8fcfbbdc5f`
- Genome hash: `025467c190ae0670`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `dist_nearest_primary_km`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 514.915249
- Search RMSE: 462.606848
- Decision: accepted. promising because search RMSE 462.606848 is at least 1% below 650.418686 from 20261022T183131011278_mlp_9a5f1229 after adding dist_nearest_primary_km
- Log: `models/mlp/docs/logs/20261022T200117471031_mlp_025467c1.json`

### `20261022T200140263561_mlp_679cc6a4`

- Command: train
- Parent: `20261022T183131011278_mlp_9a5f1229`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_primary_name", "round": 1}`
- Feature hash: `3b600d364a2c`
- Genome hash: `679cc6a410d89ae1`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `nearest_primary_name`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 641.603197
- Search RMSE: 645.978082
- Decision: rejected. not promising because search RMSE 645.978082 is not at least 1% below 650.418686 from 20261022T183131011278_mlp_9a5f1229 after adding nearest_primary_name
- Log: `models/mlp/docs/logs/20261022T200140263561_mlp_679cc6a4.json`

### `20261022T200206891334_mlp_5ef21270`

- Command: train
- Parent: `20261022T183131011278_mlp_9a5f1229`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "n_primary_within_1km", "round": 1}`
- Feature hash: `4948f0d7026e`
- Genome hash: `5ef21270989468b9`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `n_primary_within_1km`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 515.738710
- Search RMSE: 463.532571
- Decision: accepted. promising because search RMSE 463.532571 is at least 1% below 650.418686 from 20261022T183131011278_mlp_9a5f1229 after adding n_primary_within_1km
- Log: `models/mlp/docs/logs/20261022T200206891334_mlp_5ef21270.json`

### `20261022T200233437221_mlp_dbe8b7f7`

- Command: train
- Parent: `20261022T183131011278_mlp_9a5f1229`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "dist_nearest_mall_km", "round": 1}`
- Feature hash: `eb02b9676679`
- Genome hash: `dbe8b7f7e38a4330`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `dist_nearest_mall_km`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 514.209600
- Search RMSE: 461.611308
- Decision: accepted. promising because search RMSE 461.611308 is at least 1% below 650.418686 from 20261022T183131011278_mlp_9a5f1229 after adding dist_nearest_mall_km
- Log: `models/mlp/docs/logs/20261022T200233437221_mlp_dbe8b7f7.json`

### `20261022T200255700198_mlp_3ad7e514`

- Command: train
- Parent: `20261022T183131011278_mlp_9a5f1229`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_mall_name", "round": 1}`
- Feature hash: `dfa1a626ebab`
- Genome hash: `3ad7e514d6f9ab28`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `nearest_mall_name`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 638.758627
- Search RMSE: 634.830142
- Decision: accepted. promising because search RMSE 634.830142 is at least 1% below 650.418686 from 20261022T183131011278_mlp_9a5f1229 after adding nearest_mall_name
- Log: `models/mlp/docs/logs/20261022T200255700198_mlp_3ad7e514.json`

### `20261022T200317740160_mlp_498ec99d`

- Command: train
- Parent: `20261022T183131011278_mlp_9a5f1229`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_ns_station_code", "round": 1}`
- Feature hash: `5ce89d055366`
- Genome hash: `498ec99d69c2c6fb`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `nearest_ns_station_code`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 638.528451
- Search RMSE: 641.618324
- Decision: accepted. promising because search RMSE 641.618324 is at least 1% below 650.418686 from 20261022T183131011278_mlp_9a5f1229 after adding nearest_ns_station_code
- Log: `models/mlp/docs/logs/20261022T200317740160_mlp_498ec99d.json`

### `20261022T200339172112_mlp_dc00c1d3`

- Command: train
- Parent: `20261022T183131011278_mlp_9a5f1229`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_ns_station_name", "round": 1}`
- Feature hash: `70914731bd06`
- Genome hash: `dc00c1d3c3dec9c0`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `nearest_ns_station_name`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 638.528451
- Search RMSE: 641.618324
- Decision: accepted. promising because search RMSE 641.618324 is at least 1% below 650.418686 from 20261022T183131011278_mlp_9a5f1229 after adding nearest_ns_station_name
- Log: `models/mlp/docs/logs/20261022T200339172112_mlp_dc00c1d3.json`

### `20261022T200405165820_mlp_7b8712dc`

- Command: train
- Parent: `20261022T183131011278_mlp_9a5f1229`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "dist_nearest_ns_km", "round": 1}`
- Feature hash: `8d6e42f11adc`
- Genome hash: `7b8712dc6f29276b`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `dist_nearest_ns_km`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 514.132980
- Search RMSE: 461.917873
- Decision: accepted. promising because search RMSE 461.917873 is at least 1% below 650.418686 from 20261022T183131011278_mlp_9a5f1229 after adding dist_nearest_ns_km
- Log: `models/mlp/docs/logs/20261022T200405165820_mlp_7b8712dc.json`

### `20261022T200427188681_mlp_8b08b03e`

- Command: train
- Parent: `20261022T183131011278_mlp_9a5f1229`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_ew_station_code", "round": 1}`
- Feature hash: `033e0d7f6436`
- Genome hash: `8b08b03e88bcd073`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `nearest_ew_station_code`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 643.064109
- Search RMSE: 648.042821
- Decision: rejected. not promising because search RMSE 648.042821 is not at least 1% below 650.418686 from 20261022T183131011278_mlp_9a5f1229 after adding nearest_ew_station_code
- Log: `models/mlp/docs/logs/20261022T200427188681_mlp_8b08b03e.json`

### `20261022T200449111882_mlp_d8776307`

- Command: train
- Parent: `20261022T183131011278_mlp_9a5f1229`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_ew_station_name", "round": 1}`
- Feature hash: `cc0d61a0df40`
- Genome hash: `d877630767e03494`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `nearest_ew_station_name`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 643.064109
- Search RMSE: 648.042821
- Decision: rejected. not promising because search RMSE 648.042821 is not at least 1% below 650.418686 from 20261022T183131011278_mlp_9a5f1229 after adding nearest_ew_station_name
- Log: `models/mlp/docs/logs/20261022T200449111882_mlp_d8776307.json`

### `20261022T200515580604_mlp_dd2af48d`

- Command: train
- Parent: `20261022T183131011278_mlp_9a5f1229`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "dist_nearest_ew_km", "round": 1}`
- Feature hash: `8db553894f91`
- Genome hash: `dd2af48d6c5d0f3e`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `dist_nearest_ew_km`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 515.017312
- Search RMSE: 461.769469
- Decision: accepted. promising because search RMSE 461.769469 is at least 1% below 650.418686 from 20261022T183131011278_mlp_9a5f1229 after adding dist_nearest_ew_km
- Log: `models/mlp/docs/logs/20261022T200515580604_mlp_dd2af48d.json`

### `20261022T200537018440_mlp_e8198d7a`

- Command: train
- Parent: `20261022T183131011278_mlp_9a5f1229`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_ne_station_code", "round": 1}`
- Feature hash: `491bf0c825f4`
- Genome hash: `e8198d7acf09d1bb`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `nearest_ne_station_code`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 640.615177
- Search RMSE: 639.482110
- Decision: accepted. promising because search RMSE 639.482110 is at least 1% below 650.418686 from 20261022T183131011278_mlp_9a5f1229 after adding nearest_ne_station_code
- Log: `models/mlp/docs/logs/20261022T200537018440_mlp_e8198d7a.json`

### `20261022T200558264188_mlp_cf790ef3`

- Command: train
- Parent: `20261022T183131011278_mlp_9a5f1229`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_ne_station_name", "round": 1}`
- Feature hash: `0b51c9b66206`
- Genome hash: `cf790ef38e590b69`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `nearest_ne_station_name`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 640.615177
- Search RMSE: 639.482110
- Decision: accepted. promising because search RMSE 639.482110 is at least 1% below 650.418686 from 20261022T183131011278_mlp_9a5f1229 after adding nearest_ne_station_name
- Log: `models/mlp/docs/logs/20261022T200558264188_mlp_cf790ef3.json`

### `20261022T200624995722_mlp_aeebc576`

- Command: train
- Parent: `20261022T183131011278_mlp_9a5f1229`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "dist_nearest_ne_km", "round": 1}`
- Feature hash: `58386a513add`
- Genome hash: `aeebc576a9d6e8f1`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `dist_nearest_ne_km`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 515.080350
- Search RMSE: 462.268045
- Decision: accepted. promising because search RMSE 462.268045 is at least 1% below 650.418686 from 20261022T183131011278_mlp_9a5f1229 after adding dist_nearest_ne_km
- Log: `models/mlp/docs/logs/20261022T200624995722_mlp_aeebc576.json`

### `20261022T200652056285_mlp_99b4ce4c`

- Command: train
- Parent: `20261022T183131011278_mlp_9a5f1229`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_cc_station_code", "round": 1}`
- Feature hash: `e7df40e75f9a`
- Genome hash: `99b4ce4cf66a073c`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `nearest_cc_station_code`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 505.522091
- Search RMSE: 465.574727
- Decision: accepted. promising because search RMSE 465.574727 is at least 1% below 650.418686 from 20261022T183131011278_mlp_9a5f1229 after adding nearest_cc_station_code
- Log: `models/mlp/docs/logs/20261022T200652056285_mlp_99b4ce4c.json`

### `20261022T200718744045_mlp_2195c7ac`

- Command: train
- Parent: `20261022T183131011278_mlp_9a5f1229`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_cc_station_name", "round": 1}`
- Feature hash: `0e0891a68917`
- Genome hash: `2195c7acca02a133`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `nearest_cc_station_name`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 505.522091
- Search RMSE: 465.574727
- Decision: accepted. promising because search RMSE 465.574727 is at least 1% below 650.418686 from 20261022T183131011278_mlp_9a5f1229 after adding nearest_cc_station_name
- Log: `models/mlp/docs/logs/20261022T200718744045_mlp_2195c7ac.json`

### `20261022T200745192067_mlp_fe9cd9c6`

- Command: train
- Parent: `20261022T183131011278_mlp_9a5f1229`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "dist_nearest_cc_km", "round": 1}`
- Feature hash: `73966c4f212f`
- Genome hash: `fe9cd9c680fce6df`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `dist_nearest_cc_km`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 515.086582
- Search RMSE: 460.979221
- Decision: accepted. promising because search RMSE 460.979221 is at least 1% below 650.418686 from 20261022T183131011278_mlp_9a5f1229 after adding dist_nearest_cc_km
- Log: `models/mlp/docs/logs/20261022T200745192067_mlp_fe9cd9c6.json`

### `20261022T200801907687_mlp_1f54e9f3`

- Command: train
- Parent: `20261022T183131011278_mlp_9a5f1229`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_dt_station_code", "round": 1}`
- Feature hash: `248dc8a22f33`
- Genome hash: `1f54e9f3ee1d7ff9`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `nearest_dt_station_code`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 662.173748
- Search RMSE: 689.254527
- Decision: rejected. not promising because search RMSE 689.254527 is not at least 1% below 650.418686 from 20261022T183131011278_mlp_9a5f1229 after adding nearest_dt_station_code
- Log: `models/mlp/docs/logs/20261022T200801907687_mlp_1f54e9f3.json`

### `20261022T200818421079_mlp_7c06e59d`

- Command: train
- Parent: `20261022T183131011278_mlp_9a5f1229`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_dt_station_name", "round": 1}`
- Feature hash: `48c44e2e40fe`
- Genome hash: `7c06e59dc0f5949a`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `nearest_dt_station_name`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 662.173748
- Search RMSE: 689.254527
- Decision: rejected. not promising because search RMSE 689.254527 is not at least 1% below 650.418686 from 20261022T183131011278_mlp_9a5f1229 after adding nearest_dt_station_name
- Log: `models/mlp/docs/logs/20261022T200818421079_mlp_7c06e59d.json`

### `20261022T200844173452_mlp_a1279aea`

- Command: train
- Parent: `20261022T183131011278_mlp_9a5f1229`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "dist_nearest_dt_km", "round": 1}`
- Feature hash: `7e6bc08d4ff7`
- Genome hash: `a1279aeaee2b4250`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `dist_nearest_dt_km`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 515.182026
- Search RMSE: 462.039437
- Decision: accepted. promising because search RMSE 462.039437 is at least 1% below 650.418686 from 20261022T183131011278_mlp_9a5f1229 after adding dist_nearest_dt_km
- Log: `models/mlp/docs/logs/20261022T200844173452_mlp_a1279aea.json`

### `20261022T200905569106_mlp_1cb7f63d`

- Command: train
- Parent: `20261022T183131011278_mlp_9a5f1229`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_te_station_code", "round": 1}`
- Feature hash: `4f05b6ddefc5`
- Genome hash: `1cb7f63d7d1bc55e`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `nearest_te_station_code`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 643.036760
- Search RMSE: 647.901452
- Decision: rejected. not promising because search RMSE 647.901452 is not at least 1% below 650.418686 from 20261022T183131011278_mlp_9a5f1229 after adding nearest_te_station_code
- Log: `models/mlp/docs/logs/20261022T200905569106_mlp_1cb7f63d.json`

### `20261022T200927285224_mlp_02e1986b`

- Command: train
- Parent: `20261022T183131011278_mlp_9a5f1229`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_te_station_name", "round": 1}`
- Feature hash: `1139130c8f31`
- Genome hash: `02e1986b08cffcea`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `nearest_te_station_name`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 643.036760
- Search RMSE: 647.901452
- Decision: rejected. not promising because search RMSE 647.901452 is not at least 1% below 650.418686 from 20261022T183131011278_mlp_9a5f1229 after adding nearest_te_station_name
- Log: `models/mlp/docs/logs/20261022T200927285224_mlp_02e1986b.json`

### `20261022T200953073512_mlp_940de532`

- Command: train
- Parent: `20261022T183131011278_mlp_9a5f1229`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "dist_nearest_te_km", "round": 1}`
- Feature hash: `5cef183a61db`
- Genome hash: `940de53225d6863d`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `dist_nearest_te_km`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 514.619374
- Search RMSE: 462.148627
- Decision: accepted. promising because search RMSE 462.148627 is at least 1% below 650.418686 from 20261022T183131011278_mlp_9a5f1229 after adding dist_nearest_te_km
- Log: `models/mlp/docs/logs/20261022T200953073512_mlp_940de532.json`

### `20261022T201017850540_mlp_3a632ee1`

- Command: train
- Parent: `20261022T183131011278_mlp_9a5f1229`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "n_open_mrt_within_0_5km", "round": 1}`
- Feature hash: `5219387275ac`
- Genome hash: `3a632ee14620d023`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `n_open_mrt_within_0_5km`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 515.489228
- Search RMSE: 462.606927
- Decision: accepted. promising because search RMSE 462.606927 is at least 1% below 650.418686 from 20261022T183131011278_mlp_9a5f1229 after adding n_open_mrt_within_0_5km
- Log: `models/mlp/docs/logs/20261022T201017850540_mlp_3a632ee1.json`

### `20261022T201043182150_mlp_b2269a5d`

- Command: train
- Parent: `20261022T183131011278_mlp_9a5f1229`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "n_open_mrt_within_1km", "round": 1}`
- Feature hash: `692f243ba340`
- Genome hash: `b2269a5ddf1f4931`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `n_open_mrt_within_1km`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 515.162971
- Search RMSE: 462.168172
- Decision: accepted. promising because search RMSE 462.168172 is at least 1% below 650.418686 from 20261022T183131011278_mlp_9a5f1229 after adding n_open_mrt_within_1km
- Log: `models/mlp/docs/logs/20261022T201043182150_mlp_b2269a5d.json`

### `20261022T201107894829_mlp_f29fedef`

- Command: train
- Parent: `20261022T183131011278_mlp_9a5f1229`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "n_open_mrt_within_2km", "round": 1}`
- Feature hash: `2e76b96e95c7`
- Genome hash: `f29fedefd7e87741`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `n_open_mrt_within_2km`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 515.308688
- Search RMSE: 461.755250
- Decision: accepted. promising because search RMSE 461.755250 is at least 1% below 650.418686 from 20261022T183131011278_mlp_9a5f1229 after adding n_open_mrt_within_2km
- Log: `models/mlp/docs/logs/20261022T201107894829_mlp_f29fedef.json`

### `20261022T201133248326_mlp_73c2bd8e`

- Command: train
- Parent: `20261022T183131011278_mlp_9a5f1229`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "dist_nearest_planned_mrt_km", "round": 1}`
- Feature hash: `d68a3dae1af5`
- Genome hash: `73c2bd8e9993166c`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `dist_nearest_planned_mrt_km`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 514.156557
- Search RMSE: 461.944847
- Decision: accepted. promising because search RMSE 461.944847 is at least 1% below 650.418686 from 20261022T183131011278_mlp_9a5f1229 after adding dist_nearest_planned_mrt_km
- Log: `models/mlp/docs/logs/20261022T201133248326_mlp_73c2bd8e.json`

### `20261022T201158979059_mlp_455441aa`

- Command: train
- Parent: `20261022T183131011278_mlp_9a5f1229`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "n_planned_mrt_within_1km", "round": 1}`
- Feature hash: `4fed35fb6804`
- Genome hash: `455441aa958824bc`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `n_planned_mrt_within_1km`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 514.150005
- Search RMSE: 462.089149
- Decision: accepted. promising because search RMSE 462.089149 is at least 1% below 650.418686 from 20261022T183131011278_mlp_9a5f1229 after adding n_planned_mrt_within_1km
- Log: `models/mlp/docs/logs/20261022T201158979059_mlp_455441aa.json`

### `20261022T201224447128_mlp_a44929ec`

- Command: train
- Parent: `20261022T183131011278_mlp_9a5f1229`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "n_planned_mrt_within_2km", "round": 1}`
- Feature hash: `112eaabb4cdb`
- Genome hash: `a44929ec6d9b17ec`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `n_planned_mrt_within_2km`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 514.156688
- Search RMSE: 461.992297
- Decision: accepted. promising because search RMSE 461.992297 is at least 1% below 650.418686 from 20261022T183131011278_mlp_9a5f1229 after adding n_planned_mrt_within_2km
- Log: `models/mlp/docs/logs/20261022T201224447128_mlp_a44929ec.json`

### `20261022T201249167006_mlp_9abfe2ca`

- Command: train
- Parent: `20261022T183131011278_mlp_9a5f1229`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_open_mrt_is_interchange", "round": 1}`
- Feature hash: `e8afb2136c5c`
- Genome hash: `9abfe2ca401145e6`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `nearest_open_mrt_is_interchange`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 514.223480
- Search RMSE: 461.834217
- Decision: accepted. promising because search RMSE 461.834217 is at least 1% below 650.418686 from 20261022T183131011278_mlp_9a5f1229 after adding nearest_open_mrt_is_interchange
- Log: `models/mlp/docs/logs/20261022T201249167006_mlp_9abfe2ca.json`

### `20261022T201314236084_mlp_d38b707a`

- Command: train
- Parent: `20261022T183131011278_mlp_9a5f1229`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "n_primary_within_2km", "round": 1}`
- Feature hash: `9f58d18094df`
- Genome hash: `d38b707ac5270846`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `n_primary_within_2km`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 515.447326
- Search RMSE: 463.183070
- Decision: accepted. promising because search RMSE 463.183070 is at least 1% below 650.418686 from 20261022T183131011278_mlp_9a5f1229 after adding n_primary_within_2km
- Log: `models/mlp/docs/logs/20261022T201314236084_mlp_d38b707a.json`

### `20261022T201339261898_mlp_e0f1208c`

- Command: train
- Parent: `20261022T183131011278_mlp_9a5f1229`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "dist_nearest_secondary_km", "round": 1}`
- Feature hash: `3b84c255aaae`
- Genome hash: `e0f1208c093aa8b7`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `dist_nearest_secondary_km`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 515.071091
- Search RMSE: 462.815415
- Decision: accepted. promising because search RMSE 462.815415 is at least 1% below 650.418686 from 20261022T183131011278_mlp_9a5f1229 after adding dist_nearest_secondary_km
- Log: `models/mlp/docs/logs/20261022T201339261898_mlp_e0f1208c.json`

### `20261022T201400000756_mlp_4b3bdeff`

- Command: train
- Parent: `20261022T183131011278_mlp_9a5f1229`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_secondary_name", "round": 1}`
- Feature hash: `4932501d7490`
- Genome hash: `4b3bdeff0b7753c5`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `nearest_secondary_name`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 639.079053
- Search RMSE: 641.913448
- Decision: accepted. promising because search RMSE 641.913448 is at least 1% below 650.418686 from 20261022T183131011278_mlp_9a5f1229 after adding nearest_secondary_name
- Log: `models/mlp/docs/logs/20261022T201400000756_mlp_4b3bdeff.json`

### `20261022T201425338825_mlp_39d852f7`

- Command: train
- Parent: `20261022T183131011278_mlp_9a5f1229`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "n_secondary_within_1km", "round": 1}`
- Feature hash: `f92f056e77f5`
- Genome hash: `39d852f772df11a5`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `n_secondary_within_1km`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 514.689739
- Search RMSE: 462.462436
- Decision: accepted. promising because search RMSE 462.462436 is at least 1% below 650.418686 from 20261022T183131011278_mlp_9a5f1229 after adding n_secondary_within_1km
- Log: `models/mlp/docs/logs/20261022T201425338825_mlp_39d852f7.json`

### `20261022T201450381634_mlp_fc92efe0`

- Command: train
- Parent: `20261022T183131011278_mlp_9a5f1229`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "dist_nearest_school_km", "round": 1}`
- Feature hash: `482ab5dda0e9`
- Genome hash: `fc92efe0401b4146`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `dist_nearest_school_km`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 515.236215
- Search RMSE: 462.961560
- Decision: accepted. promising because search RMSE 462.961560 is at least 1% below 650.418686 from 20261022T183131011278_mlp_9a5f1229 after adding dist_nearest_school_km
- Log: `models/mlp/docs/logs/20261022T201450381634_mlp_fc92efe0.json`

### `20261022T201517258055_mlp_729fa55e`

- Command: train
- Parent: `20261022T183131011278_mlp_9a5f1229`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_school_name", "round": 1}`
- Feature hash: `e657e5902602`
- Genome hash: `729fa55ef229cada`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `nearest_school_name`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 511.221025
- Search RMSE: 459.972595
- Decision: accepted. promising because search RMSE 459.972595 is at least 1% below 650.418686 from 20261022T183131011278_mlp_9a5f1229 after adding nearest_school_name
- Log: `models/mlp/docs/logs/20261022T201517258055_mlp_729fa55e.json`

### `20261022T201542649610_mlp_1c2c08a6`

- Command: train
- Parent: `20261022T183131011278_mlp_9a5f1229`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "n_school_within_1km", "round": 1}`
- Feature hash: `b9907fe19ce9`
- Genome hash: `1c2c08a60551f8f7`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `n_school_within_1km`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 515.294153
- Search RMSE: 463.159434
- Decision: accepted. promising because search RMSE 463.159434 is at least 1% below 650.418686 from 20261022T183131011278_mlp_9a5f1229 after adding n_school_within_1km
- Log: `models/mlp/docs/logs/20261022T201542649610_mlp_1c2c08a6.json`

### `20261022T201607551889_mlp_c59e5d4d`

- Command: train
- Parent: `20261022T183131011278_mlp_9a5f1229`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "n_mall_within_1km", "round": 1}`
- Feature hash: `a98c9f4224a5`
- Genome hash: `c59e5d4ddf30e7fc`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `n_mall_within_1km`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 514.350594
- Search RMSE: 461.899329
- Decision: accepted. promising because search RMSE 461.899329 is at least 1% below 650.418686 from 20261022T183131011278_mlp_9a5f1229 after adding n_mall_within_1km
- Log: `models/mlp/docs/logs/20261022T201607551889_mlp_c59e5d4d.json`

### `20261022T201632853593_mlp_8eaf6c5f`

- Command: train
- Parent: `20261022T183131011278_mlp_9a5f1229`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "n_mall_within_2km", "round": 1}`
- Feature hash: `282ff6f12a0d`
- Genome hash: `8eaf6c5f46e41c3a`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `n_mall_within_2km`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 514.532538
- Search RMSE: 461.973642
- Decision: accepted. promising because search RMSE 461.973642 is at least 1% below 650.418686 from 20261022T183131011278_mlp_9a5f1229 after adding n_mall_within_2km
- Log: `models/mlp/docs/logs/20261022T201632853593_mlp_8eaf6c5f.json`

### `20261022T201658371742_mlp_c6e56c27`

- Command: train
- Parent: `20261022T183131011278_mlp_9a5f1229`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "coe_a_premium_lag1", "round": 1}`
- Feature hash: `ca6251adee59`
- Genome hash: `c6e56c27bab0bf54`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_a_premium_lag1`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 489.082709
- Search RMSE: 468.157277
- Decision: accepted. promising because search RMSE 468.157277 is at least 1% below 650.418686 from 20261022T183131011278_mlp_9a5f1229 after adding coe_a_premium_lag1
- Log: `models/mlp/docs/logs/20261022T201658371742_mlp_c6e56c27.json`

### `20261022T201723812662_mlp_723efb69`

- Command: train
- Parent: `20261022T183131011278_mlp_9a5f1229`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "coe_a_premium_mom_change", "round": 1}`
- Feature hash: `ecd53635e5ea`
- Genome hash: `723efb697bae0ca3`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_a_premium_mom_change`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 515.155746
- Search RMSE: 464.906440
- Decision: accepted. promising because search RMSE 464.906440 is at least 1% below 650.418686 from 20261022T183131011278_mlp_9a5f1229 after adding coe_a_premium_mom_change
- Log: `models/mlp/docs/logs/20261022T201723812662_mlp_723efb69.json`

### `20261022T201749402408_mlp_37823e85`

- Command: train
- Parent: `20261022T183131011278_mlp_9a5f1229`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "coe_a_bids_per_quota_lag1", "round": 1}`
- Feature hash: `88b869c77fc2`
- Genome hash: `37823e85f0af241e`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_a_bids_per_quota_lag1`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 518.478499
- Search RMSE: 468.231270
- Decision: accepted. promising because search RMSE 468.231270 is at least 1% below 650.418686 from 20261022T183131011278_mlp_9a5f1229 after adding coe_a_bids_per_quota_lag1
- Log: `models/mlp/docs/logs/20261022T201749402408_mlp_37823e85.json`

### `20261022T201814509450_mlp_e62d173d`

- Command: train
- Parent: `20261022T183131011278_mlp_9a5f1229`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "coe_b_premium_lag1", "round": 1}`
- Feature hash: `aef8257585e0`
- Genome hash: `e62d173d8a59272d`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_b_premium_lag1`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 490.351944
- Search RMSE: 459.207433
- Decision: accepted. promising because search RMSE 459.207433 is at least 1% below 650.418686 from 20261022T183131011278_mlp_9a5f1229 after adding coe_b_premium_lag1
- Log: `models/mlp/docs/logs/20261022T201814509450_mlp_e62d173d.json`

### `20261022T201839376389_mlp_f63fe861`

- Command: train
- Parent: `20261022T183131011278_mlp_9a5f1229`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "coe_b_premium_mom_change", "round": 1}`
- Feature hash: `8302b3652167`
- Genome hash: `f63fe8615c6c1ae7`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_b_premium_mom_change`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 517.066130
- Search RMSE: 467.169191
- Decision: accepted. promising because search RMSE 467.169191 is at least 1% below 650.418686 from 20261022T183131011278_mlp_9a5f1229 after adding coe_b_premium_mom_change
- Log: `models/mlp/docs/logs/20261022T201839376389_mlp_f63fe861.json`

### `20261022T201904148283_mlp_479a61bc`

- Command: train
- Parent: `20261022T183131011278_mlp_9a5f1229`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "coe_b_bids_per_quota_lag1", "round": 1}`
- Feature hash: `4a7482aab3c7`
- Genome hash: `479a61bc3c1f69b1`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_b_bids_per_quota_lag1`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 510.490968
- Search RMSE: 464.610132
- Decision: accepted. promising because search RMSE 464.610132 is at least 1% below 650.418686 from 20261022T183131011278_mlp_9a5f1229 after adding coe_b_bids_per_quota_lag1
- Log: `models/mlp/docs/logs/20261022T201904148283_mlp_479a61bc.json`

### `20261022T201929300146_mlp_22d71bb7`

- Command: train
- Parent: `20261022T183131011278_mlp_9a5f1229`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "coe_e_premium_lag1", "round": 1}`
- Feature hash: `de6005d5c05c`
- Genome hash: `22d71bb72d16102a`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_e_premium_lag1`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 491.391202
- Search RMSE: 458.991130
- Decision: accepted. promising because search RMSE 458.991130 is at least 1% below 650.418686 from 20261022T183131011278_mlp_9a5f1229 after adding coe_e_premium_lag1
- Log: `models/mlp/docs/logs/20261022T201929300146_mlp_22d71bb7.json`

### `20261022T201954746864_mlp_1653cbfc`

- Command: train
- Parent: `20261022T183131011278_mlp_9a5f1229`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "coe_e_premium_mom_change", "round": 1}`
- Feature hash: `387e56d7e868`
- Genome hash: `1653cbfce602cc8f`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_e_premium_mom_change`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 516.072742
- Search RMSE: 466.776191
- Decision: accepted. promising because search RMSE 466.776191 is at least 1% below 650.418686 from 20261022T183131011278_mlp_9a5f1229 after adding coe_e_premium_mom_change
- Log: `models/mlp/docs/logs/20261022T201954746864_mlp_1653cbfc.json`

### `20261022T202018082316_mlp_772d79fa`

- Command: train
- Parent: `20261022T183131011278_mlp_9a5f1229`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "coe_e_bids_per_quota_lag1", "round": 1}`
- Feature hash: `8bb848ebe345`
- Genome hash: `772d79fa2cbde469`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_e_bids_per_quota_lag1`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 565.840811
- Search RMSE: 560.753463
- Decision: accepted. promising because search RMSE 560.753463 is at least 1% below 650.418686 from 20261022T183131011278_mlp_9a5f1229 after adding coe_e_bids_per_quota_lag1
- Log: `models/mlp/docs/logs/20261022T202018082316_mlp_772d79fa.json`

### `20261022T202042943525_mlp_9a86fa97`

- Command: train
- Parent: `20261022T183131011278_mlp_9a5f1229`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "coe_a_premium_roll3", "round": 1}`
- Feature hash: `b09c5f9ba37b`
- Genome hash: `9a86fa97b8de755b`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_a_premium_roll3`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 493.434966
- Search RMSE: 467.373546
- Decision: accepted. promising because search RMSE 467.373546 is at least 1% below 650.418686 from 20261022T183131011278_mlp_9a5f1229 after adding coe_a_premium_roll3
- Log: `models/mlp/docs/logs/20261022T202042943525_mlp_9a86fa97.json`

### `20261022T202107626897_mlp_1057797d`

- Command: train
- Parent: `20261022T183131011278_mlp_9a5f1229`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "coe_c_premium_lag1", "round": 1}`
- Feature hash: `7b8f900c807f`
- Genome hash: `1057797dc3b9f8d2`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_c_premium_lag1`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 483.422748
- Search RMSE: 458.899901
- Decision: accepted. promising because search RMSE 458.899901 is at least 1% below 650.418686 from 20261022T183131011278_mlp_9a5f1229 after adding coe_c_premium_lag1
- Log: `models/mlp/docs/logs/20261022T202107626897_mlp_1057797d.json`

### `20261022T202132833155_mlp_3890c6aa`

- Command: train
- Parent: `20261022T183131011278_mlp_9a5f1229`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "dist_nearest_future_line_open_km", "round": 1}`
- Feature hash: `109871ee8d0c`
- Genome hash: `3890c6aae117bbf8`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `dist_nearest_future_line_open_km`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 514.078253
- Search RMSE: 461.911426
- Decision: accepted. promising because search RMSE 461.911426 is at least 1% below 650.418686 from 20261022T183131011278_mlp_9a5f1229 after adding dist_nearest_future_line_open_km
- Log: `models/mlp/docs/logs/20261022T202132833155_mlp_3890c6aa.json`

### `20261022T202158191556_mlp_28ae776e`

- Command: train
- Parent: `20261022T183131011278_mlp_9a5f1229`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "dist_nearest_future_line_planned_km", "round": 1}`
- Feature hash: `b96b1461ee87`
- Genome hash: `28ae776e4f958330`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `dist_nearest_future_line_planned_km`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 514.172825
- Search RMSE: 461.979422
- Decision: accepted. promising because search RMSE 461.979422 is at least 1% below 650.418686 from 20261022T183131011278_mlp_9a5f1229 after adding dist_nearest_future_line_planned_km
- Log: `models/mlp/docs/logs/20261022T202158191556_mlp_28ae776e.json`

### `20261022T202224097631_mlp_ec65554a`

- Command: train
- Parent: `20261022T183131011278_mlp_9a5f1229`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "n_hdb_blocks_within_500m", "round": 1}`
- Feature hash: `95bfdabe2d04`
- Genome hash: `ec65554ab29837d3`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `n_hdb_blocks_within_500m`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 516.236927
- Search RMSE: 464.312813
- Decision: accepted. promising because search RMSE 464.312813 is at least 1% below 650.418686 from 20261022T183131011278_mlp_9a5f1229 after adding n_hdb_blocks_within_500m
- Log: `models/mlp/docs/logs/20261022T202224097631_mlp_ec65554a.json`

### `20261022T202249233266_mlp_a90a47b5`

- Command: train
- Parent: `20261022T183131011278_mlp_9a5f1229`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "dist_to_town_centroid_km", "round": 1}`
- Feature hash: `cbb5a64dccd3`
- Genome hash: `a90a47b565f99089`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `dist_to_town_centroid_km`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 513.954064
- Search RMSE: 461.830592
- Decision: accepted. promising because search RMSE 461.830592 is at least 1% below 650.418686 from 20261022T183131011278_mlp_9a5f1229 after adding dist_to_town_centroid_km
- Log: `models/mlp/docs/logs/20261022T202249233266_mlp_a90a47b5.json`

### `20261022T202314773737_mlp_ab0ef0e2`

- Command: train
- Parent: `20261022T183131011278_mlp_9a5f1229`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "dist_to_cbd_km", "round": 1}`
- Feature hash: `3a81f2034fb9`
- Genome hash: `ab0ef0e20fcff97f`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `dist_to_cbd_km`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 514.495358
- Search RMSE: 460.311817
- Decision: accepted. promising because search RMSE 460.311817 is at least 1% below 650.418686 from 20261022T183131011278_mlp_9a5f1229 after adding dist_to_cbd_km
- Log: `models/mlp/docs/logs/20261022T202314773737_mlp_ab0ef0e2.json`

### `20261022T202337245842_mlp_e6a40619`

- Command: train
- Parent: `20261022T183131011278_mlp_9a5f1229`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "bank_basket_adj_close_lag1", "round": 1}`
- Feature hash: `40ef8a2b790f`
- Genome hash: `e6a40619c0817255`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `bank_basket_adj_close_lag1`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 538.846745
- Search RMSE: 489.908783
- Decision: accepted. promising because search RMSE 489.908783 is at least 1% below 650.418686 from 20261022T183131011278_mlp_9a5f1229 after adding bank_basket_adj_close_lag1
- Log: `models/mlp/docs/logs/20261022T202337245842_mlp_e6a40619.json`

### `20261022T202401946631_mlp_81c99407`

- Command: train
- Parent: `20261022T183131011278_mlp_9a5f1229`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "bank_basket_return_lag1", "round": 1}`
- Feature hash: `52b8a977b3a9`
- Genome hash: `81c994076ed04354`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `bank_basket_return_lag1`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 514.070475
- Search RMSE: 464.682666
- Decision: accepted. promising because search RMSE 464.682666 is at least 1% below 650.418686 from 20261022T183131011278_mlp_9a5f1229 after adding bank_basket_return_lag1
- Log: `models/mlp/docs/logs/20261022T202401946631_mlp_81c99407.json`

### `20261022T202426960230_mlp_0a98e083`

- Command: train
- Parent: `20261022T183131011278_mlp_9a5f1229`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "property_basket_adj_close_lag1", "round": 1}`
- Feature hash: `3b1ea68b7e46`
- Genome hash: `0a98e083ff74831d`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `property_basket_adj_close_lag1`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 538.411241
- Search RMSE: 465.747838
- Decision: accepted. promising because search RMSE 465.747838 is at least 1% below 650.418686 from 20261022T183131011278_mlp_9a5f1229 after adding property_basket_adj_close_lag1
- Log: `models/mlp/docs/logs/20261022T202426960230_mlp_0a98e083.json`

### `20261022T202452552664_mlp_69d80c30`

- Command: train
- Parent: `20261022T183131011278_mlp_9a5f1229`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "property_basket_return_lag1", "round": 1}`
- Feature hash: `d99f88cbdcb7`
- Genome hash: `69d80c30b6787a93`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `property_basket_return_lag1`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 515.252795
- Search RMSE: 472.069417
- Decision: accepted. promising because search RMSE 472.069417 is at least 1% below 650.418686 from 20261022T183131011278_mlp_9a5f1229 after adding property_basket_return_lag1
- Log: `models/mlp/docs/logs/20261022T202452552664_mlp_69d80c30.json`

### `20261022T202517972264_mlp_d86262d8`

- Command: train
- Parent: `20261022T183131011278_mlp_9a5f1229`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "log_dist_nearest_open_mrt_km", "round": 1}`
- Feature hash: `21c7244e6748`
- Genome hash: `d86262d8b2223663`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `log_dist_nearest_open_mrt_km`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 515.298501
- Search RMSE: 462.147981
- Decision: accepted. promising because search RMSE 462.147981 is at least 1% below 650.418686 from 20261022T183131011278_mlp_9a5f1229 after adding log_dist_nearest_open_mrt_km
- Log: `models/mlp/docs/logs/20261022T202517972264_mlp_d86262d8.json`

### `20261022T202543846484_mlp_45109f10`

- Command: train
- Parent: `20261022T183131011278_mlp_9a5f1229`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "is_central_region", "round": 1}`
- Feature hash: `e47e0c0ed690`
- Genome hash: `45109f10e6217ef9`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `is_central_region`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 514.847728
- Search RMSE: 461.417356
- Decision: accepted. promising because search RMSE 461.417356 is at least 1% below 650.418686 from 20261022T183131011278_mlp_9a5f1229 after adding is_central_region
- Log: `models/mlp/docs/logs/20261022T202543846484_mlp_45109f10.json`

### `20261022T202609426740_mlp_e83b2e00`

- Command: train
- Parent: `20261022T183131011278_mlp_9a5f1229`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "building_age_x_central", "round": 1}`
- Feature hash: `a5a53d0ae4a0`
- Genome hash: `e83b2e00afec68e4`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `building_age_x_central`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 515.075887
- Search RMSE: 462.241600
- Decision: accepted. promising because search RMSE 462.241600 is at least 1% below 650.418686 from 20261022T183131011278_mlp_9a5f1229 after adding building_age_x_central
- Log: `models/mlp/docs/logs/20261022T202609426740_mlp_e83b2e00.json`

### `20261022T202634795628_mlp_3f4cd9d3`

- Command: train
- Parent: `20261022T183131011278_mlp_9a5f1229`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "mean_rent_postal_sector", "round": 1}`
- Feature hash: `e7393246f8d5`
- Genome hash: `3f4cd9d3e8d047c5`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `mean_rent_postal_sector`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 490.117777
- Search RMSE: 465.743165
- Decision: accepted. promising because search RMSE 465.743165 is at least 1% below 650.418686 from 20261022T183131011278_mlp_9a5f1229 after adding mean_rent_postal_sector
- Log: `models/mlp/docs/logs/20261022T202634795628_mlp_3f4cd9d3.json`

### `20261022T202700190590_mlp_432e0f40`

- Command: train
- Parent: `20261022T183131011278_mlp_9a5f1229`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "mean_rent_postal_district", "round": 1}`
- Feature hash: `4d4cc5ffe4d6`
- Genome hash: `432e0f404bdafe44`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `mean_rent_postal_district`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 489.476667
- Search RMSE: 465.437035
- Decision: accepted. promising because search RMSE 465.437035 is at least 1% below 650.418686 from 20261022T183131011278_mlp_9a5f1229 after adding mean_rent_postal_district
- Log: `models/mlp/docs/logs/20261022T202700190590_mlp_432e0f40.json`

### `20261022T202726359322_mlp_463b59e0`

- Command: train
- Parent: `20261022T183131011278_mlp_9a5f1229`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "mean_rent_nearest_mall", "round": 1}`
- Feature hash: `471d8ee447c6`
- Genome hash: `463b59e0ccb89727`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `mean_rent_nearest_mall`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 490.752181
- Search RMSE: 466.417779
- Decision: accepted. promising because search RMSE 466.417779 is at least 1% below 650.418686 from 20261022T183131011278_mlp_9a5f1229 after adding mean_rent_nearest_mall
- Log: `models/mlp/docs/logs/20261022T202726359322_mlp_463b59e0.json`

### `20261022T202752454429_mlp_e75842c9`

- Command: train
- Parent: `20261022T183131011278_mlp_9a5f1229`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "mean_rent_nearest_mrt", "round": 1}`
- Feature hash: `c9dd7135f0c6`
- Genome hash: `e75842c975529131`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `mean_rent_nearest_mrt`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 490.337431
- Search RMSE: 466.936863
- Decision: accepted. promising because search RMSE 466.936863 is at least 1% below 650.418686 from 20261022T183131011278_mlp_9a5f1229 after adding mean_rent_nearest_mrt
- Log: `models/mlp/docs/logs/20261022T202752454429_mlp_e75842c9.json`

### `20261022T202817666412_mlp_9bbbcae0`

- Command: train
- Parent: `20261022T183131011278_mlp_9a5f1229`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "mean_rent_nearest_primary", "round": 1}`
- Feature hash: `b23749c91b13`
- Genome hash: `9bbbcae081680622`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `mean_rent_nearest_primary`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 490.783786
- Search RMSE: 466.490236
- Decision: accepted. promising because search RMSE 466.490236 is at least 1% below 650.418686 from 20261022T183131011278_mlp_9a5f1229 after adding mean_rent_nearest_primary
- Log: `models/mlp/docs/logs/20261022T202817666412_mlp_9bbbcae0.json`

### `20261022T202842662431_mlp_04ef4ea5`

- Command: train
- Parent: `20261022T183131011278_mlp_9a5f1229`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "mean_rent_nearest_secondary", "round": 1}`
- Feature hash: `8b3befbcf20d`
- Genome hash: `04ef4ea5b9ef9f81`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `mean_rent_nearest_secondary`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 490.458922
- Search RMSE: 467.202636
- Decision: accepted. promising because search RMSE 467.202636 is at least 1% below 650.418686 from 20261022T183131011278_mlp_9a5f1229 after adding mean_rent_nearest_secondary
- Log: `models/mlp/docs/logs/20261022T202842662431_mlp_04ef4ea5.json`

### `20261022T202913337555_mlp_3f1940b8`

- Command: train
- Parent: `20261022T183131011278_mlp_9a5f1229`
- Generation: None
- Mutation: `{"axis": "features", "change": "set_c", "columns": ["latitude", "longitude", "max_floor", "year_completed", "planning_area", "region", "building_age_years", "lease_vs_completion_gap", "flat_age_from_lease", "dist_nearest_open_mrt_km", "dist_nearest_primary_km", "n_primary_within_1km", "dist_nearest_mall_km", "nearest_mall_name", "nearest_ns_station_code", "nearest_ns_station_name", "dist_nearest_ns_km", "dist_nearest_ew_km", "nearest_ne_station_code", "nearest_ne_station_name", "dist_nearest_ne_km", "nearest_cc_station_code", "nearest_cc_station_name", "dist_nearest_cc_km", "dist_nearest_dt_km", "dist_nearest_te_km", "n_open_mrt_within_0_5km", "n_open_mrt_within_1km", "n_open_mrt_within_2km", "dist_nearest_planned_mrt_km", "n_planned_mrt_within_1km", "n_planned_mrt_within_2km", "nearest_open_mrt_is_interchange", "n_primary_within_2km", "dist_nearest_secondary_km", "nearest_secondary_name", "n_secondary_within_1km", "dist_nearest_school_km", "nearest_school_name", "n_school_within_1km", "n_mall_within_1km", "n_mall_within_2km", "coe_a_premium_lag1", "coe_a_premium_mom_change", "coe_a_bids_per_quota_lag1", "coe_b_premium_lag1", "coe_b_premium_mom_change", "coe_b_bids_per_quota_lag1", "coe_e_premium_lag1", "coe_e_premium_mom_change", "coe_e_bids_per_quota_lag1", "coe_a_premium_roll3", "coe_c_premium_lag1", "dist_nearest_future_line_open_km", "dist_nearest_future_line_planned_km", "n_hdb_blocks_within_500m", "dist_to_town_centroid_km", "dist_to_cbd_km", "bank_basket_adj_close_lag1", "bank_basket_return_lag1", "property_basket_adj_close_lag1", "property_basket_return_lag1", "log_dist_nearest_open_mrt_km", "is_central_region", "building_age_x_central", "mean_rent_postal_sector", "mean_rent_postal_district", "mean_rent_nearest_mall", "mean_rent_nearest_mrt", "mean_rent_nearest_primary", "mean_rent_nearest_secondary"], "round": 1}`
- Feature hash: `8d04a6ec2a11`
- Genome hash: `3f1940b884101451`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `latitude`, `longitude`, `max_floor`, `year_completed`, `planning_area`, `region`, `building_age_years`, `lease_vs_completion_gap`, `flat_age_from_lease`, `dist_nearest_open_mrt_km`, `dist_nearest_primary_km`, `n_primary_within_1km`, `dist_nearest_mall_km`, `nearest_mall_name`, `nearest_ns_station_code`, `nearest_ns_station_name`, `dist_nearest_ns_km`, `dist_nearest_ew_km`, `nearest_ne_station_code`, `nearest_ne_station_name`, `dist_nearest_ne_km`, `nearest_cc_station_code`, `nearest_cc_station_name`, `dist_nearest_cc_km`, `dist_nearest_dt_km`, `dist_nearest_te_km`, `n_open_mrt_within_0_5km`, `n_open_mrt_within_1km`, `n_open_mrt_within_2km`, `dist_nearest_planned_mrt_km`, `n_planned_mrt_within_1km`, `n_planned_mrt_within_2km`, `nearest_open_mrt_is_interchange`, `n_primary_within_2km`, `dist_nearest_secondary_km`, `nearest_secondary_name`, `n_secondary_within_1km`, `dist_nearest_school_km`, `nearest_school_name`, `n_school_within_1km`, `n_mall_within_1km`, `n_mall_within_2km`, `coe_a_premium_lag1`, `coe_a_premium_mom_change`, `coe_a_bids_per_quota_lag1`, `coe_b_premium_lag1`, `coe_b_premium_mom_change`, `coe_b_bids_per_quota_lag1`, `coe_e_premium_lag1`, `coe_e_premium_mom_change`, `coe_e_bids_per_quota_lag1`, `coe_a_premium_roll3`, `coe_c_premium_lag1`, `dist_nearest_future_line_open_km`, `dist_nearest_future_line_planned_km`, `n_hdb_blocks_within_500m`, `dist_to_town_centroid_km`, `dist_to_cbd_km`, `bank_basket_adj_close_lag1`, `bank_basket_return_lag1`, `property_basket_adj_close_lag1`, `property_basket_return_lag1`, `log_dist_nearest_open_mrt_km`, `is_central_region`, `building_age_x_central`, `mean_rent_postal_sector`, `mean_rent_postal_district`, `mean_rent_nearest_mall`, `mean_rent_nearest_mrt`, `mean_rent_nearest_primary`, `mean_rent_nearest_secondary`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 489.836979
- Search RMSE: 467.263467
- Decision: accepted. replaced best 20261022T183131011278_mlp_9a5f1229 search RMSE 650.418686 with 467.263467
- Log: `models/mlp/docs/logs/20261022T202913337555_mlp_3f1940b8.json`

### `20261022T202948208229_mlp_a81266d0`

- Command: train
- Parent: `20261022T202913337555_mlp_3f1940b8`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "postal_code", "round": 2}`
- Feature hash: `dccbbcdcc940`
- Genome hash: `a81266d03a29aa65`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `latitude`, `longitude`, `max_floor`, `year_completed`, `planning_area`, `region`, `building_age_years`, `lease_vs_completion_gap`, `flat_age_from_lease`, `dist_nearest_open_mrt_km`, `dist_nearest_primary_km`, `n_primary_within_1km`, `dist_nearest_mall_km`, `nearest_mall_name`, `nearest_ns_station_code`, `nearest_ns_station_name`, `dist_nearest_ns_km`, `dist_nearest_ew_km`, `nearest_ne_station_code`, `nearest_ne_station_name`, `dist_nearest_ne_km`, `nearest_cc_station_code`, `nearest_cc_station_name`, `dist_nearest_cc_km`, `dist_nearest_dt_km`, `dist_nearest_te_km`, `n_open_mrt_within_0_5km`, `n_open_mrt_within_1km`, `n_open_mrt_within_2km`, `dist_nearest_planned_mrt_km`, `n_planned_mrt_within_1km`, `n_planned_mrt_within_2km`, `nearest_open_mrt_is_interchange`, `n_primary_within_2km`, `dist_nearest_secondary_km`, `nearest_secondary_name`, `n_secondary_within_1km`, `dist_nearest_school_km`, `nearest_school_name`, `n_school_within_1km`, `n_mall_within_1km`, `n_mall_within_2km`, `coe_a_premium_lag1`, `coe_a_premium_mom_change`, `coe_a_bids_per_quota_lag1`, `coe_b_premium_lag1`, `coe_b_premium_mom_change`, `coe_b_bids_per_quota_lag1`, `coe_e_premium_lag1`, `coe_e_premium_mom_change`, `coe_e_bids_per_quota_lag1`, `coe_a_premium_roll3`, `coe_c_premium_lag1`, `dist_nearest_future_line_open_km`, `dist_nearest_future_line_planned_km`, `n_hdb_blocks_within_500m`, `dist_to_town_centroid_km`, `dist_to_cbd_km`, `bank_basket_adj_close_lag1`, `bank_basket_return_lag1`, `property_basket_adj_close_lag1`, `property_basket_return_lag1`, `log_dist_nearest_open_mrt_km`, `is_central_region`, `building_age_x_central`, `mean_rent_postal_sector`, `mean_rent_postal_district`, `mean_rent_nearest_mall`, `mean_rent_nearest_mrt`, `mean_rent_nearest_primary`, `mean_rent_nearest_secondary`, `postal_code`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 491.164109
- Search RMSE: 470.207411
- Decision: rejected. not promising because search RMSE 470.207411 is not at least 1% below 467.263467 from 20261022T202913337555_mlp_3f1940b8 after adding postal_code
- Log: `models/mlp/docs/logs/20261022T202948208229_mlp_a81266d0.json`

### `20261022T203019117221_mlp_12c1f0b8`

- Command: train
- Parent: `20261022T202913337555_mlp_3f1940b8`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "subzone", "round": 2}`
- Feature hash: `dd0cdb27c52d`
- Genome hash: `12c1f0b8961bcd67`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `latitude`, `longitude`, `max_floor`, `year_completed`, `planning_area`, `region`, `building_age_years`, `lease_vs_completion_gap`, `flat_age_from_lease`, `dist_nearest_open_mrt_km`, `dist_nearest_primary_km`, `n_primary_within_1km`, `dist_nearest_mall_km`, `nearest_mall_name`, `nearest_ns_station_code`, `nearest_ns_station_name`, `dist_nearest_ns_km`, `dist_nearest_ew_km`, `nearest_ne_station_code`, `nearest_ne_station_name`, `dist_nearest_ne_km`, `nearest_cc_station_code`, `nearest_cc_station_name`, `dist_nearest_cc_km`, `dist_nearest_dt_km`, `dist_nearest_te_km`, `n_open_mrt_within_0_5km`, `n_open_mrt_within_1km`, `n_open_mrt_within_2km`, `dist_nearest_planned_mrt_km`, `n_planned_mrt_within_1km`, `n_planned_mrt_within_2km`, `nearest_open_mrt_is_interchange`, `n_primary_within_2km`, `dist_nearest_secondary_km`, `nearest_secondary_name`, `n_secondary_within_1km`, `dist_nearest_school_km`, `nearest_school_name`, `n_school_within_1km`, `n_mall_within_1km`, `n_mall_within_2km`, `coe_a_premium_lag1`, `coe_a_premium_mom_change`, `coe_a_bids_per_quota_lag1`, `coe_b_premium_lag1`, `coe_b_premium_mom_change`, `coe_b_bids_per_quota_lag1`, `coe_e_premium_lag1`, `coe_e_premium_mom_change`, `coe_e_bids_per_quota_lag1`, `coe_a_premium_roll3`, `coe_c_premium_lag1`, `dist_nearest_future_line_open_km`, `dist_nearest_future_line_planned_km`, `n_hdb_blocks_within_500m`, `dist_to_town_centroid_km`, `dist_to_cbd_km`, `bank_basket_adj_close_lag1`, `bank_basket_return_lag1`, `property_basket_adj_close_lag1`, `property_basket_return_lag1`, `log_dist_nearest_open_mrt_km`, `is_central_region`, `building_age_x_central`, `mean_rent_postal_sector`, `mean_rent_postal_district`, `mean_rent_nearest_mall`, `mean_rent_nearest_mrt`, `mean_rent_nearest_primary`, `mean_rent_nearest_secondary`, `subzone`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 491.321419
- Search RMSE: 469.130567
- Decision: rejected. not promising because search RMSE 469.130567 is not at least 1% below 467.263467 from 20261022T202913337555_mlp_3f1940b8 after adding subzone
- Log: `models/mlp/docs/logs/20261022T203019117221_mlp_12c1f0b8.json`

### `20261022T203049845891_mlp_943f14ad`

- Command: train
- Parent: `20261022T202913337555_mlp_3f1940b8`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_open_mrt_code", "round": 2}`
- Feature hash: `1aee49e8279c`
- Genome hash: `943f14ad942ca33b`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `latitude`, `longitude`, `max_floor`, `year_completed`, `planning_area`, `region`, `building_age_years`, `lease_vs_completion_gap`, `flat_age_from_lease`, `dist_nearest_open_mrt_km`, `dist_nearest_primary_km`, `n_primary_within_1km`, `dist_nearest_mall_km`, `nearest_mall_name`, `nearest_ns_station_code`, `nearest_ns_station_name`, `dist_nearest_ns_km`, `dist_nearest_ew_km`, `nearest_ne_station_code`, `nearest_ne_station_name`, `dist_nearest_ne_km`, `nearest_cc_station_code`, `nearest_cc_station_name`, `dist_nearest_cc_km`, `dist_nearest_dt_km`, `dist_nearest_te_km`, `n_open_mrt_within_0_5km`, `n_open_mrt_within_1km`, `n_open_mrt_within_2km`, `dist_nearest_planned_mrt_km`, `n_planned_mrt_within_1km`, `n_planned_mrt_within_2km`, `nearest_open_mrt_is_interchange`, `n_primary_within_2km`, `dist_nearest_secondary_km`, `nearest_secondary_name`, `n_secondary_within_1km`, `dist_nearest_school_km`, `nearest_school_name`, `n_school_within_1km`, `n_mall_within_1km`, `n_mall_within_2km`, `coe_a_premium_lag1`, `coe_a_premium_mom_change`, `coe_a_bids_per_quota_lag1`, `coe_b_premium_lag1`, `coe_b_premium_mom_change`, `coe_b_bids_per_quota_lag1`, `coe_e_premium_lag1`, `coe_e_premium_mom_change`, `coe_e_bids_per_quota_lag1`, `coe_a_premium_roll3`, `coe_c_premium_lag1`, `dist_nearest_future_line_open_km`, `dist_nearest_future_line_planned_km`, `n_hdb_blocks_within_500m`, `dist_to_town_centroid_km`, `dist_to_cbd_km`, `bank_basket_adj_close_lag1`, `bank_basket_return_lag1`, `property_basket_adj_close_lag1`, `property_basket_return_lag1`, `log_dist_nearest_open_mrt_km`, `is_central_region`, `building_age_x_central`, `mean_rent_postal_sector`, `mean_rent_postal_district`, `mean_rent_nearest_mall`, `mean_rent_nearest_mrt`, `mean_rent_nearest_primary`, `mean_rent_nearest_secondary`, `nearest_open_mrt_code`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 490.277824
- Search RMSE: 468.705977
- Decision: rejected. not promising because search RMSE 468.705977 is not at least 1% below 467.263467 from 20261022T202913337555_mlp_3f1940b8 after adding nearest_open_mrt_code
- Log: `models/mlp/docs/logs/20261022T203049845891_mlp_943f14ad.json`

### `20261022T203121125706_mlp_31b3ca98`

- Command: train
- Parent: `20261022T202913337555_mlp_3f1940b8`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_open_mrt_name", "round": 2}`
- Feature hash: `0def7d9238c5`
- Genome hash: `31b3ca98f7cb26de`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `latitude`, `longitude`, `max_floor`, `year_completed`, `planning_area`, `region`, `building_age_years`, `lease_vs_completion_gap`, `flat_age_from_lease`, `dist_nearest_open_mrt_km`, `dist_nearest_primary_km`, `n_primary_within_1km`, `dist_nearest_mall_km`, `nearest_mall_name`, `nearest_ns_station_code`, `nearest_ns_station_name`, `dist_nearest_ns_km`, `dist_nearest_ew_km`, `nearest_ne_station_code`, `nearest_ne_station_name`, `dist_nearest_ne_km`, `nearest_cc_station_code`, `nearest_cc_station_name`, `dist_nearest_cc_km`, `dist_nearest_dt_km`, `dist_nearest_te_km`, `n_open_mrt_within_0_5km`, `n_open_mrt_within_1km`, `n_open_mrt_within_2km`, `dist_nearest_planned_mrt_km`, `n_planned_mrt_within_1km`, `n_planned_mrt_within_2km`, `nearest_open_mrt_is_interchange`, `n_primary_within_2km`, `dist_nearest_secondary_km`, `nearest_secondary_name`, `n_secondary_within_1km`, `dist_nearest_school_km`, `nearest_school_name`, `n_school_within_1km`, `n_mall_within_1km`, `n_mall_within_2km`, `coe_a_premium_lag1`, `coe_a_premium_mom_change`, `coe_a_bids_per_quota_lag1`, `coe_b_premium_lag1`, `coe_b_premium_mom_change`, `coe_b_bids_per_quota_lag1`, `coe_e_premium_lag1`, `coe_e_premium_mom_change`, `coe_e_bids_per_quota_lag1`, `coe_a_premium_roll3`, `coe_c_premium_lag1`, `dist_nearest_future_line_open_km`, `dist_nearest_future_line_planned_km`, `n_hdb_blocks_within_500m`, `dist_to_town_centroid_km`, `dist_to_cbd_km`, `bank_basket_adj_close_lag1`, `bank_basket_return_lag1`, `property_basket_adj_close_lag1`, `property_basket_return_lag1`, `log_dist_nearest_open_mrt_km`, `is_central_region`, `building_age_x_central`, `mean_rent_postal_sector`, `mean_rent_postal_district`, `mean_rent_nearest_mall`, `mean_rent_nearest_mrt`, `mean_rent_nearest_primary`, `mean_rent_nearest_secondary`, `nearest_open_mrt_name`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 490.277828
- Search RMSE: 468.705945
- Decision: rejected. not promising because search RMSE 468.705945 is not at least 1% below 467.263467 from 20261022T202913337555_mlp_3f1940b8 after adding nearest_open_mrt_name
- Log: `models/mlp/docs/logs/20261022T203121125706_mlp_31b3ca98.json`

### `20261022T203152070201_mlp_2225ea1a`

- Command: train
- Parent: `20261022T202913337555_mlp_3f1940b8`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_open_mrt_line", "round": 2}`
- Feature hash: `66d16ee1cbd8`
- Genome hash: `2225ea1a0a3668ca`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `latitude`, `longitude`, `max_floor`, `year_completed`, `planning_area`, `region`, `building_age_years`, `lease_vs_completion_gap`, `flat_age_from_lease`, `dist_nearest_open_mrt_km`, `dist_nearest_primary_km`, `n_primary_within_1km`, `dist_nearest_mall_km`, `nearest_mall_name`, `nearest_ns_station_code`, `nearest_ns_station_name`, `dist_nearest_ns_km`, `dist_nearest_ew_km`, `nearest_ne_station_code`, `nearest_ne_station_name`, `dist_nearest_ne_km`, `nearest_cc_station_code`, `nearest_cc_station_name`, `dist_nearest_cc_km`, `dist_nearest_dt_km`, `dist_nearest_te_km`, `n_open_mrt_within_0_5km`, `n_open_mrt_within_1km`, `n_open_mrt_within_2km`, `dist_nearest_planned_mrt_km`, `n_planned_mrt_within_1km`, `n_planned_mrt_within_2km`, `nearest_open_mrt_is_interchange`, `n_primary_within_2km`, `dist_nearest_secondary_km`, `nearest_secondary_name`, `n_secondary_within_1km`, `dist_nearest_school_km`, `nearest_school_name`, `n_school_within_1km`, `n_mall_within_1km`, `n_mall_within_2km`, `coe_a_premium_lag1`, `coe_a_premium_mom_change`, `coe_a_bids_per_quota_lag1`, `coe_b_premium_lag1`, `coe_b_premium_mom_change`, `coe_b_bids_per_quota_lag1`, `coe_e_premium_lag1`, `coe_e_premium_mom_change`, `coe_e_bids_per_quota_lag1`, `coe_a_premium_roll3`, `coe_c_premium_lag1`, `dist_nearest_future_line_open_km`, `dist_nearest_future_line_planned_km`, `n_hdb_blocks_within_500m`, `dist_to_town_centroid_km`, `dist_to_cbd_km`, `bank_basket_adj_close_lag1`, `bank_basket_return_lag1`, `property_basket_adj_close_lag1`, `property_basket_return_lag1`, `log_dist_nearest_open_mrt_km`, `is_central_region`, `building_age_x_central`, `mean_rent_postal_sector`, `mean_rent_postal_district`, `mean_rent_nearest_mall`, `mean_rent_nearest_mrt`, `mean_rent_nearest_primary`, `mean_rent_nearest_secondary`, `nearest_open_mrt_line`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 490.673207
- Search RMSE: 467.754093
- Decision: rejected. not promising because search RMSE 467.754093 is not at least 1% below 467.263467 from 20261022T202913337555_mlp_3f1940b8 after adding nearest_open_mrt_line
- Log: `models/mlp/docs/logs/20261022T203152070201_mlp_2225ea1a.json`

### `20261022T203222871642_mlp_0741f971`

- Command: train
- Parent: `20261022T202913337555_mlp_3f1940b8`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_primary_name", "round": 2}`
- Feature hash: `f13e929d6dc1`
- Genome hash: `0741f971cf78be13`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `latitude`, `longitude`, `max_floor`, `year_completed`, `planning_area`, `region`, `building_age_years`, `lease_vs_completion_gap`, `flat_age_from_lease`, `dist_nearest_open_mrt_km`, `dist_nearest_primary_km`, `n_primary_within_1km`, `dist_nearest_mall_km`, `nearest_mall_name`, `nearest_ns_station_code`, `nearest_ns_station_name`, `dist_nearest_ns_km`, `dist_nearest_ew_km`, `nearest_ne_station_code`, `nearest_ne_station_name`, `dist_nearest_ne_km`, `nearest_cc_station_code`, `nearest_cc_station_name`, `dist_nearest_cc_km`, `dist_nearest_dt_km`, `dist_nearest_te_km`, `n_open_mrt_within_0_5km`, `n_open_mrt_within_1km`, `n_open_mrt_within_2km`, `dist_nearest_planned_mrt_km`, `n_planned_mrt_within_1km`, `n_planned_mrt_within_2km`, `nearest_open_mrt_is_interchange`, `n_primary_within_2km`, `dist_nearest_secondary_km`, `nearest_secondary_name`, `n_secondary_within_1km`, `dist_nearest_school_km`, `nearest_school_name`, `n_school_within_1km`, `n_mall_within_1km`, `n_mall_within_2km`, `coe_a_premium_lag1`, `coe_a_premium_mom_change`, `coe_a_bids_per_quota_lag1`, `coe_b_premium_lag1`, `coe_b_premium_mom_change`, `coe_b_bids_per_quota_lag1`, `coe_e_premium_lag1`, `coe_e_premium_mom_change`, `coe_e_bids_per_quota_lag1`, `coe_a_premium_roll3`, `coe_c_premium_lag1`, `dist_nearest_future_line_open_km`, `dist_nearest_future_line_planned_km`, `n_hdb_blocks_within_500m`, `dist_to_town_centroid_km`, `dist_to_cbd_km`, `bank_basket_adj_close_lag1`, `bank_basket_return_lag1`, `property_basket_adj_close_lag1`, `property_basket_return_lag1`, `log_dist_nearest_open_mrt_km`, `is_central_region`, `building_age_x_central`, `mean_rent_postal_sector`, `mean_rent_postal_district`, `mean_rent_nearest_mall`, `mean_rent_nearest_mrt`, `mean_rent_nearest_primary`, `mean_rent_nearest_secondary`, `nearest_primary_name`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 487.312253
- Search RMSE: 467.537604
- Decision: rejected. not promising because search RMSE 467.537604 is not at least 1% below 467.263467 from 20261022T202913337555_mlp_3f1940b8 after adding nearest_primary_name
- Log: `models/mlp/docs/logs/20261022T203222871642_mlp_0741f971.json`

### `20261022T203253065550_mlp_312f21a2`

- Command: train
- Parent: `20261022T202913337555_mlp_3f1940b8`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_ew_station_code", "round": 2}`
- Feature hash: `6085f3564aee`
- Genome hash: `312f21a2c2aed72e`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `latitude`, `longitude`, `max_floor`, `year_completed`, `planning_area`, `region`, `building_age_years`, `lease_vs_completion_gap`, `flat_age_from_lease`, `dist_nearest_open_mrt_km`, `dist_nearest_primary_km`, `n_primary_within_1km`, `dist_nearest_mall_km`, `nearest_mall_name`, `nearest_ns_station_code`, `nearest_ns_station_name`, `dist_nearest_ns_km`, `dist_nearest_ew_km`, `nearest_ne_station_code`, `nearest_ne_station_name`, `dist_nearest_ne_km`, `nearest_cc_station_code`, `nearest_cc_station_name`, `dist_nearest_cc_km`, `dist_nearest_dt_km`, `dist_nearest_te_km`, `n_open_mrt_within_0_5km`, `n_open_mrt_within_1km`, `n_open_mrt_within_2km`, `dist_nearest_planned_mrt_km`, `n_planned_mrt_within_1km`, `n_planned_mrt_within_2km`, `nearest_open_mrt_is_interchange`, `n_primary_within_2km`, `dist_nearest_secondary_km`, `nearest_secondary_name`, `n_secondary_within_1km`, `dist_nearest_school_km`, `nearest_school_name`, `n_school_within_1km`, `n_mall_within_1km`, `n_mall_within_2km`, `coe_a_premium_lag1`, `coe_a_premium_mom_change`, `coe_a_bids_per_quota_lag1`, `coe_b_premium_lag1`, `coe_b_premium_mom_change`, `coe_b_bids_per_quota_lag1`, `coe_e_premium_lag1`, `coe_e_premium_mom_change`, `coe_e_bids_per_quota_lag1`, `coe_a_premium_roll3`, `coe_c_premium_lag1`, `dist_nearest_future_line_open_km`, `dist_nearest_future_line_planned_km`, `n_hdb_blocks_within_500m`, `dist_to_town_centroid_km`, `dist_to_cbd_km`, `bank_basket_adj_close_lag1`, `bank_basket_return_lag1`, `property_basket_adj_close_lag1`, `property_basket_return_lag1`, `log_dist_nearest_open_mrt_km`, `is_central_region`, `building_age_x_central`, `mean_rent_postal_sector`, `mean_rent_postal_district`, `mean_rent_nearest_mall`, `mean_rent_nearest_mrt`, `mean_rent_nearest_primary`, `mean_rent_nearest_secondary`, `nearest_ew_station_code`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 493.477905
- Search RMSE: 471.260756
- Decision: rejected. not promising because search RMSE 471.260756 is not at least 1% below 467.263467 from 20261022T202913337555_mlp_3f1940b8 after adding nearest_ew_station_code
- Log: `models/mlp/docs/logs/20261022T203253065550_mlp_312f21a2.json`

### `20261022T203323866722_mlp_142596b7`

- Command: train
- Parent: `20261022T202913337555_mlp_3f1940b8`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_ew_station_name", "round": 2}`
- Feature hash: `28223fb1eb6e`
- Genome hash: `142596b71664c521`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `latitude`, `longitude`, `max_floor`, `year_completed`, `planning_area`, `region`, `building_age_years`, `lease_vs_completion_gap`, `flat_age_from_lease`, `dist_nearest_open_mrt_km`, `dist_nearest_primary_km`, `n_primary_within_1km`, `dist_nearest_mall_km`, `nearest_mall_name`, `nearest_ns_station_code`, `nearest_ns_station_name`, `dist_nearest_ns_km`, `dist_nearest_ew_km`, `nearest_ne_station_code`, `nearest_ne_station_name`, `dist_nearest_ne_km`, `nearest_cc_station_code`, `nearest_cc_station_name`, `dist_nearest_cc_km`, `dist_nearest_dt_km`, `dist_nearest_te_km`, `n_open_mrt_within_0_5km`, `n_open_mrt_within_1km`, `n_open_mrt_within_2km`, `dist_nearest_planned_mrt_km`, `n_planned_mrt_within_1km`, `n_planned_mrt_within_2km`, `nearest_open_mrt_is_interchange`, `n_primary_within_2km`, `dist_nearest_secondary_km`, `nearest_secondary_name`, `n_secondary_within_1km`, `dist_nearest_school_km`, `nearest_school_name`, `n_school_within_1km`, `n_mall_within_1km`, `n_mall_within_2km`, `coe_a_premium_lag1`, `coe_a_premium_mom_change`, `coe_a_bids_per_quota_lag1`, `coe_b_premium_lag1`, `coe_b_premium_mom_change`, `coe_b_bids_per_quota_lag1`, `coe_e_premium_lag1`, `coe_e_premium_mom_change`, `coe_e_bids_per_quota_lag1`, `coe_a_premium_roll3`, `coe_c_premium_lag1`, `dist_nearest_future_line_open_km`, `dist_nearest_future_line_planned_km`, `n_hdb_blocks_within_500m`, `dist_to_town_centroid_km`, `dist_to_cbd_km`, `bank_basket_adj_close_lag1`, `bank_basket_return_lag1`, `property_basket_adj_close_lag1`, `property_basket_return_lag1`, `log_dist_nearest_open_mrt_km`, `is_central_region`, `building_age_x_central`, `mean_rent_postal_sector`, `mean_rent_postal_district`, `mean_rent_nearest_mall`, `mean_rent_nearest_mrt`, `mean_rent_nearest_primary`, `mean_rent_nearest_secondary`, `nearest_ew_station_name`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 493.477905
- Search RMSE: 471.260756
- Decision: rejected. not promising because search RMSE 471.260756 is not at least 1% below 467.263467 from 20261022T202913337555_mlp_3f1940b8 after adding nearest_ew_station_name
- Log: `models/mlp/docs/logs/20261022T203323866722_mlp_142596b7.json`

### `20261022T203354791781_mlp_5b6ec458`

- Command: train
- Parent: `20261022T202913337555_mlp_3f1940b8`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_dt_station_code", "round": 2}`
- Feature hash: `dc42a824c1c7`
- Genome hash: `5b6ec4580bca5ff9`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `latitude`, `longitude`, `max_floor`, `year_completed`, `planning_area`, `region`, `building_age_years`, `lease_vs_completion_gap`, `flat_age_from_lease`, `dist_nearest_open_mrt_km`, `dist_nearest_primary_km`, `n_primary_within_1km`, `dist_nearest_mall_km`, `nearest_mall_name`, `nearest_ns_station_code`, `nearest_ns_station_name`, `dist_nearest_ns_km`, `dist_nearest_ew_km`, `nearest_ne_station_code`, `nearest_ne_station_name`, `dist_nearest_ne_km`, `nearest_cc_station_code`, `nearest_cc_station_name`, `dist_nearest_cc_km`, `dist_nearest_dt_km`, `dist_nearest_te_km`, `n_open_mrt_within_0_5km`, `n_open_mrt_within_1km`, `n_open_mrt_within_2km`, `dist_nearest_planned_mrt_km`, `n_planned_mrt_within_1km`, `n_planned_mrt_within_2km`, `nearest_open_mrt_is_interchange`, `n_primary_within_2km`, `dist_nearest_secondary_km`, `nearest_secondary_name`, `n_secondary_within_1km`, `dist_nearest_school_km`, `nearest_school_name`, `n_school_within_1km`, `n_mall_within_1km`, `n_mall_within_2km`, `coe_a_premium_lag1`, `coe_a_premium_mom_change`, `coe_a_bids_per_quota_lag1`, `coe_b_premium_lag1`, `coe_b_premium_mom_change`, `coe_b_bids_per_quota_lag1`, `coe_e_premium_lag1`, `coe_e_premium_mom_change`, `coe_e_bids_per_quota_lag1`, `coe_a_premium_roll3`, `coe_c_premium_lag1`, `dist_nearest_future_line_open_km`, `dist_nearest_future_line_planned_km`, `n_hdb_blocks_within_500m`, `dist_to_town_centroid_km`, `dist_to_cbd_km`, `bank_basket_adj_close_lag1`, `bank_basket_return_lag1`, `property_basket_adj_close_lag1`, `property_basket_return_lag1`, `log_dist_nearest_open_mrt_km`, `is_central_region`, `building_age_x_central`, `mean_rent_postal_sector`, `mean_rent_postal_district`, `mean_rent_nearest_mall`, `mean_rent_nearest_mrt`, `mean_rent_nearest_primary`, `mean_rent_nearest_secondary`, `nearest_dt_station_code`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 489.414192
- Search RMSE: 466.719309
- Decision: rejected. not promising because search RMSE 466.719309 is not at least 1% below 467.263467 from 20261022T202913337555_mlp_3f1940b8 after adding nearest_dt_station_code
- Log: `models/mlp/docs/logs/20261022T203354791781_mlp_5b6ec458.json`

### `20261022T203425655657_mlp_c1ce072b`

- Command: train
- Parent: `20261022T202913337555_mlp_3f1940b8`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_dt_station_name", "round": 2}`
- Feature hash: `63b77c8c8586`
- Genome hash: `c1ce072b11a99317`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `latitude`, `longitude`, `max_floor`, `year_completed`, `planning_area`, `region`, `building_age_years`, `lease_vs_completion_gap`, `flat_age_from_lease`, `dist_nearest_open_mrt_km`, `dist_nearest_primary_km`, `n_primary_within_1km`, `dist_nearest_mall_km`, `nearest_mall_name`, `nearest_ns_station_code`, `nearest_ns_station_name`, `dist_nearest_ns_km`, `dist_nearest_ew_km`, `nearest_ne_station_code`, `nearest_ne_station_name`, `dist_nearest_ne_km`, `nearest_cc_station_code`, `nearest_cc_station_name`, `dist_nearest_cc_km`, `dist_nearest_dt_km`, `dist_nearest_te_km`, `n_open_mrt_within_0_5km`, `n_open_mrt_within_1km`, `n_open_mrt_within_2km`, `dist_nearest_planned_mrt_km`, `n_planned_mrt_within_1km`, `n_planned_mrt_within_2km`, `nearest_open_mrt_is_interchange`, `n_primary_within_2km`, `dist_nearest_secondary_km`, `nearest_secondary_name`, `n_secondary_within_1km`, `dist_nearest_school_km`, `nearest_school_name`, `n_school_within_1km`, `n_mall_within_1km`, `n_mall_within_2km`, `coe_a_premium_lag1`, `coe_a_premium_mom_change`, `coe_a_bids_per_quota_lag1`, `coe_b_premium_lag1`, `coe_b_premium_mom_change`, `coe_b_bids_per_quota_lag1`, `coe_e_premium_lag1`, `coe_e_premium_mom_change`, `coe_e_bids_per_quota_lag1`, `coe_a_premium_roll3`, `coe_c_premium_lag1`, `dist_nearest_future_line_open_km`, `dist_nearest_future_line_planned_km`, `n_hdb_blocks_within_500m`, `dist_to_town_centroid_km`, `dist_to_cbd_km`, `bank_basket_adj_close_lag1`, `bank_basket_return_lag1`, `property_basket_adj_close_lag1`, `property_basket_return_lag1`, `log_dist_nearest_open_mrt_km`, `is_central_region`, `building_age_x_central`, `mean_rent_postal_sector`, `mean_rent_postal_district`, `mean_rent_nearest_mall`, `mean_rent_nearest_mrt`, `mean_rent_nearest_primary`, `mean_rent_nearest_secondary`, `nearest_dt_station_name`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 489.414192
- Search RMSE: 466.719309
- Decision: rejected. not promising because search RMSE 466.719309 is not at least 1% below 467.263467 from 20261022T202913337555_mlp_3f1940b8 after adding nearest_dt_station_name
- Log: `models/mlp/docs/logs/20261022T203425655657_mlp_c1ce072b.json`

### `20261022T203456274336_mlp_85fda7ce`

- Command: train
- Parent: `20261022T202913337555_mlp_3f1940b8`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_te_station_code", "round": 2}`
- Feature hash: `73f4ed658292`
- Genome hash: `85fda7cea27d90f3`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `latitude`, `longitude`, `max_floor`, `year_completed`, `planning_area`, `region`, `building_age_years`, `lease_vs_completion_gap`, `flat_age_from_lease`, `dist_nearest_open_mrt_km`, `dist_nearest_primary_km`, `n_primary_within_1km`, `dist_nearest_mall_km`, `nearest_mall_name`, `nearest_ns_station_code`, `nearest_ns_station_name`, `dist_nearest_ns_km`, `dist_nearest_ew_km`, `nearest_ne_station_code`, `nearest_ne_station_name`, `dist_nearest_ne_km`, `nearest_cc_station_code`, `nearest_cc_station_name`, `dist_nearest_cc_km`, `dist_nearest_dt_km`, `dist_nearest_te_km`, `n_open_mrt_within_0_5km`, `n_open_mrt_within_1km`, `n_open_mrt_within_2km`, `dist_nearest_planned_mrt_km`, `n_planned_mrt_within_1km`, `n_planned_mrt_within_2km`, `nearest_open_mrt_is_interchange`, `n_primary_within_2km`, `dist_nearest_secondary_km`, `nearest_secondary_name`, `n_secondary_within_1km`, `dist_nearest_school_km`, `nearest_school_name`, `n_school_within_1km`, `n_mall_within_1km`, `n_mall_within_2km`, `coe_a_premium_lag1`, `coe_a_premium_mom_change`, `coe_a_bids_per_quota_lag1`, `coe_b_premium_lag1`, `coe_b_premium_mom_change`, `coe_b_bids_per_quota_lag1`, `coe_e_premium_lag1`, `coe_e_premium_mom_change`, `coe_e_bids_per_quota_lag1`, `coe_a_premium_roll3`, `coe_c_premium_lag1`, `dist_nearest_future_line_open_km`, `dist_nearest_future_line_planned_km`, `n_hdb_blocks_within_500m`, `dist_to_town_centroid_km`, `dist_to_cbd_km`, `bank_basket_adj_close_lag1`, `bank_basket_return_lag1`, `property_basket_adj_close_lag1`, `property_basket_return_lag1`, `log_dist_nearest_open_mrt_km`, `is_central_region`, `building_age_x_central`, `mean_rent_postal_sector`, `mean_rent_postal_district`, `mean_rent_nearest_mall`, `mean_rent_nearest_mrt`, `mean_rent_nearest_primary`, `mean_rent_nearest_secondary`, `nearest_te_station_code`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 493.471900
- Search RMSE: 471.189905
- Decision: rejected. not promising because search RMSE 471.189905 is not at least 1% below 467.263467 from 20261022T202913337555_mlp_3f1940b8 after adding nearest_te_station_code
- Log: `models/mlp/docs/logs/20261022T203456274336_mlp_85fda7ce.json`

### `20261022T203527059800_mlp_1d7df401`

- Command: train
- Parent: `20261022T202913337555_mlp_3f1940b8`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_te_station_name", "round": 2}`
- Feature hash: `dc5b67ada61b`
- Genome hash: `1d7df401228feb57`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `latitude`, `longitude`, `max_floor`, `year_completed`, `planning_area`, `region`, `building_age_years`, `lease_vs_completion_gap`, `flat_age_from_lease`, `dist_nearest_open_mrt_km`, `dist_nearest_primary_km`, `n_primary_within_1km`, `dist_nearest_mall_km`, `nearest_mall_name`, `nearest_ns_station_code`, `nearest_ns_station_name`, `dist_nearest_ns_km`, `dist_nearest_ew_km`, `nearest_ne_station_code`, `nearest_ne_station_name`, `dist_nearest_ne_km`, `nearest_cc_station_code`, `nearest_cc_station_name`, `dist_nearest_cc_km`, `dist_nearest_dt_km`, `dist_nearest_te_km`, `n_open_mrt_within_0_5km`, `n_open_mrt_within_1km`, `n_open_mrt_within_2km`, `dist_nearest_planned_mrt_km`, `n_planned_mrt_within_1km`, `n_planned_mrt_within_2km`, `nearest_open_mrt_is_interchange`, `n_primary_within_2km`, `dist_nearest_secondary_km`, `nearest_secondary_name`, `n_secondary_within_1km`, `dist_nearest_school_km`, `nearest_school_name`, `n_school_within_1km`, `n_mall_within_1km`, `n_mall_within_2km`, `coe_a_premium_lag1`, `coe_a_premium_mom_change`, `coe_a_bids_per_quota_lag1`, `coe_b_premium_lag1`, `coe_b_premium_mom_change`, `coe_b_bids_per_quota_lag1`, `coe_e_premium_lag1`, `coe_e_premium_mom_change`, `coe_e_bids_per_quota_lag1`, `coe_a_premium_roll3`, `coe_c_premium_lag1`, `dist_nearest_future_line_open_km`, `dist_nearest_future_line_planned_km`, `n_hdb_blocks_within_500m`, `dist_to_town_centroid_km`, `dist_to_cbd_km`, `bank_basket_adj_close_lag1`, `bank_basket_return_lag1`, `property_basket_adj_close_lag1`, `property_basket_return_lag1`, `log_dist_nearest_open_mrt_km`, `is_central_region`, `building_age_x_central`, `mean_rent_postal_sector`, `mean_rent_postal_district`, `mean_rent_nearest_mall`, `mean_rent_nearest_mrt`, `mean_rent_nearest_primary`, `mean_rent_nearest_secondary`, `nearest_te_station_name`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001}`
- Train RMSE: 493.471900
- Search RMSE: 471.189905
- Decision: rejected. not promising because search RMSE 471.189905 is not at least 1% below 467.263467 from 20261022T202913337555_mlp_3f1940b8 after adding nearest_te_station_name
- Log: `models/mlp/docs/logs/20261022T203527059800_mlp_1d7df401.json`

### `20261030T103203158442_mlp_4afa8c53`

- Command: train
- Parent: `20261022T202913337555_mlp_3f1940b8`
- Generation: None
- Mutation: `{"axis": "finalize"}`
- Feature hash: `8d04a6ec2a11`
- Genome hash: `4afa8c53dc80de2d`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `latitude`, `longitude`, `max_floor`, `year_completed`, `planning_area`, `region`, `building_age_years`, `lease_vs_completion_gap`, `flat_age_from_lease`, `dist_nearest_open_mrt_km`, `dist_nearest_primary_km`, `n_primary_within_1km`, `dist_nearest_mall_km`, `nearest_mall_name`, `nearest_ns_station_code`, `nearest_ns_station_name`, `dist_nearest_ns_km`, `dist_nearest_ew_km`, `nearest_ne_station_code`, `nearest_ne_station_name`, `dist_nearest_ne_km`, `nearest_cc_station_code`, `nearest_cc_station_name`, `dist_nearest_cc_km`, `dist_nearest_dt_km`, `dist_nearest_te_km`, `n_open_mrt_within_0_5km`, `n_open_mrt_within_1km`, `n_open_mrt_within_2km`, `dist_nearest_planned_mrt_km`, `n_planned_mrt_within_1km`, `n_planned_mrt_within_2km`, `nearest_open_mrt_is_interchange`, `n_primary_within_2km`, `dist_nearest_secondary_km`, `nearest_secondary_name`, `n_secondary_within_1km`, `dist_nearest_school_km`, `nearest_school_name`, `n_school_within_1km`, `n_mall_within_1km`, `n_mall_within_2km`, `coe_a_premium_lag1`, `coe_a_premium_mom_change`, `coe_a_bids_per_quota_lag1`, `coe_b_premium_lag1`, `coe_b_premium_mom_change`, `coe_b_bids_per_quota_lag1`, `coe_e_premium_lag1`, `coe_e_premium_mom_change`, `coe_e_bids_per_quota_lag1`, `coe_a_premium_roll3`, `coe_c_premium_lag1`, `dist_nearest_future_line_open_km`, `dist_nearest_future_line_planned_km`, `n_hdb_blocks_within_500m`, `dist_to_town_centroid_km`, `dist_to_cbd_km`, `bank_basket_adj_close_lag1`, `bank_basket_return_lag1`, `property_basket_adj_close_lag1`, `property_basket_return_lag1`, `log_dist_nearest_open_mrt_km`, `is_central_region`, `building_age_x_central`, `mean_rent_postal_sector`, `mean_rent_postal_district`, `mean_rent_nearest_mall`, `mean_rent_nearest_mrt`, `mean_rent_nearest_primary`, `mean_rent_nearest_secondary`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 200, "hidden_sizes": [512, 256, 128], "learning_rate": 0.001, "patience": 15}`
- Train RMSE: 490.935571
- Search RMSE: 455.342206
- Decision: accepted. final fit on the features of 20261022T202913337555_mlp_3f1940b8; this stage does not select columns
- Log: `models/mlp/docs/logs/20261030T103203158442_mlp_4afa8c53.json`

### `20261030T105128475560_mlp_969afd46`

- Command: submit
- Parent: `20261030T103203158442_mlp_4afa8c53`
- Generation: None
- Mutation: `{"axis": "submit"}`
- Feature hash: `8d04a6ec2a11`
- Genome hash: `969afd46c8a5cb31`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `latitude`, `longitude`, `max_floor`, `year_completed`, `planning_area`, `region`, `building_age_years`, `lease_vs_completion_gap`, `flat_age_from_lease`, `dist_nearest_open_mrt_km`, `dist_nearest_primary_km`, `n_primary_within_1km`, `dist_nearest_mall_km`, `nearest_mall_name`, `nearest_ns_station_code`, `nearest_ns_station_name`, `dist_nearest_ns_km`, `dist_nearest_ew_km`, `nearest_ne_station_code`, `nearest_ne_station_name`, `dist_nearest_ne_km`, `nearest_cc_station_code`, `nearest_cc_station_name`, `dist_nearest_cc_km`, `dist_nearest_dt_km`, `dist_nearest_te_km`, `n_open_mrt_within_0_5km`, `n_open_mrt_within_1km`, `n_open_mrt_within_2km`, `dist_nearest_planned_mrt_km`, `n_planned_mrt_within_1km`, `n_planned_mrt_within_2km`, `nearest_open_mrt_is_interchange`, `n_primary_within_2km`, `dist_nearest_secondary_km`, `nearest_secondary_name`, `n_secondary_within_1km`, `dist_nearest_school_km`, `nearest_school_name`, `n_school_within_1km`, `n_mall_within_1km`, `n_mall_within_2km`, `coe_a_premium_lag1`, `coe_a_premium_mom_change`, `coe_a_bids_per_quota_lag1`, `coe_b_premium_lag1`, `coe_b_premium_mom_change`, `coe_b_bids_per_quota_lag1`, `coe_e_premium_lag1`, `coe_e_premium_mom_change`, `coe_e_bids_per_quota_lag1`, `coe_a_premium_roll3`, `coe_c_premium_lag1`, `dist_nearest_future_line_open_km`, `dist_nearest_future_line_planned_km`, `n_hdb_blocks_within_500m`, `dist_to_town_centroid_km`, `dist_to_cbd_km`, `bank_basket_adj_close_lag1`, `bank_basket_return_lag1`, `property_basket_adj_close_lag1`, `property_basket_return_lag1`, `log_dist_nearest_open_mrt_km`, `is_central_region`, `building_age_x_central`, `mean_rent_postal_sector`, `mean_rent_postal_district`, `mean_rent_nearest_mall`, `mean_rent_nearest_mrt`, `mean_rent_nearest_primary`, `mean_rent_nearest_secondary`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 13, "hidden_sizes": [512, 256, 128], "learning_rate": 0.001, "patience": 0}`
- Train RMSE: 478.994171
- Decision: refit of 20261030T103203158442_mlp_4afa8c53 on every labeled month; this stage does not select columns and was not scored on held-out MONTHLY_RENT
- Log: `models/mlp/docs/logs/20261030T105128475560_mlp_969afd46.json`

### `20261030T110903258555_mlp_fbff1ee6`

- Command: train
- Parent: `20261022T202913337555_mlp_3f1940b8`
- Generation: None
- Mutation: `{"axis": "use_all_features"}`
- Feature hash: `646e8a84f218`
- Genome hash: `fbff1ee65f5f2ff9`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `latitude`, `longitude`, `postal_code`, `max_floor`, `year_completed`, `subzone`, `planning_area`, `region`, `building_age_years`, `lease_vs_completion_gap`, `flat_age_from_lease`, `dist_nearest_open_mrt_km`, `nearest_open_mrt_code`, `nearest_open_mrt_name`, `nearest_open_mrt_line`, `dist_nearest_primary_km`, `nearest_primary_name`, `n_primary_within_1km`, `dist_nearest_mall_km`, `nearest_mall_name`, `nearest_ns_station_code`, `nearest_ns_station_name`, `dist_nearest_ns_km`, `nearest_ew_station_code`, `nearest_ew_station_name`, `dist_nearest_ew_km`, `nearest_ne_station_code`, `nearest_ne_station_name`, `dist_nearest_ne_km`, `nearest_cc_station_code`, `nearest_cc_station_name`, `dist_nearest_cc_km`, `nearest_dt_station_code`, `nearest_dt_station_name`, `dist_nearest_dt_km`, `nearest_te_station_code`, `nearest_te_station_name`, `dist_nearest_te_km`, `n_open_mrt_within_0_5km`, `n_open_mrt_within_1km`, `n_open_mrt_within_2km`, `dist_nearest_planned_mrt_km`, `n_planned_mrt_within_1km`, `n_planned_mrt_within_2km`, `nearest_open_mrt_is_interchange`, `n_primary_within_2km`, `dist_nearest_secondary_km`, `nearest_secondary_name`, `n_secondary_within_1km`, `dist_nearest_school_km`, `nearest_school_name`, `n_school_within_1km`, `n_mall_within_1km`, `n_mall_within_2km`, `coe_a_premium_lag1`, `coe_a_premium_mom_change`, `coe_a_bids_per_quota_lag1`, `coe_b_premium_lag1`, `coe_b_premium_mom_change`, `coe_b_bids_per_quota_lag1`, `coe_e_premium_lag1`, `coe_e_premium_mom_change`, `coe_e_bids_per_quota_lag1`, `coe_a_premium_roll3`, `coe_c_premium_lag1`, `dist_nearest_future_line_open_km`, `dist_nearest_future_line_planned_km`, `n_hdb_blocks_within_500m`, `dist_to_town_centroid_km`, `dist_to_cbd_km`, `bank_basket_adj_close_lag1`, `bank_basket_return_lag1`, `property_basket_adj_close_lag1`, `property_basket_return_lag1`, `log_dist_nearest_open_mrt_km`, `is_central_region`, `building_age_x_central`, `mean_rent_postal_sector`, `mean_rent_postal_district`, `mean_rent_nearest_mall`, `mean_rent_nearest_mrt`, `mean_rent_nearest_primary`, `mean_rent_nearest_secondary`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"batch_size": 1024, "dropout": 0.1, "epochs": 20, "hidden_sizes": [128, 64], "learning_rate": 0.001, "patience": 3}`
- Train RMSE: 486.780979
- Search RMSE: 467.388557
- Decision: accepted. all original and auxiliary columns fit together; this fit does not select columns one at a time and does not replace best/
- Log: `models/mlp/docs/logs/20261030T110903258555_mlp_fbff1ee6.json`
<!-- lineage:end -->

## Predictions

<!-- predictions:start -->
### `20261022T202915126681_mlp_predict0`

- Champion run: `20261022T202913337555_mlp_3f1940b8`
- Rows: 50000
- Mean: 3295.2716583740234
- Std: 379.95691485219334
- Min: 2015.1485595703125
- Max: 5257.7734375
- Output: `outputs/predictions/mlp_ablation.csv`
- Refit on all labeled months: False
- Log: `models/mlp/docs/logs/20261022T202915126681_mlp_predict0.json`

### `20261030T103205753272_mlp_predict0`

- Champion run: `20261030T103203158442_mlp_4afa8c53`
- Rows: 50000
- Mean: 3511.3975429858397
- Std: 496.9705034635545
- Min: 1847.533203125
- Max: 5592.7470703125
- Output: `outputs/predictions/mlp_final.csv`
- Refit on all labeled months: False
- Log: `models/mlp/docs/logs/20261030T103205753272_mlp_predict0.json`

### `20261030T104907191855_mlp_predict0`

- Champion run: `20261030T103203158442_mlp_4afa8c53`
- Rows: 50000
- Mean: 3511.3975429858397
- Std: 496.9705034635545
- Min: 1847.533203125
- Max: 5592.7470703125
- Output: `outputs/predictions/mlp_final.csv`
- Refit on all labeled months: False
- Log: `models/mlp/docs/logs/20261030T104907191855_mlp_predict0.json`

### `20261030T105130181431_mlp_predict0`

- Champion run: `20261030T105128475560_mlp_969afd46`
- Rows: 50000
- Mean: 3395.245645053711
- Std: 473.7710814432006
- Min: 1685.558837890625
- Max: 5458.87255859375
- Output: `outputs/predictions/mlp_refit.csv`
- Refit on all labeled months: True
- Log: `models/mlp/docs/logs/20261030T105130181431_mlp_predict0.json`
<!-- predictions:end -->

