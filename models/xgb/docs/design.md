# XGBoost

## Architecture gene

The spec is `max_depth` and `learning_rate`. Nominal columns other than the fixed date, block, and furnished overrides are native categories, so one split can place any subset of the labels on the left. Regularize lowers `max_depth` by 2 down to 2. Capacity raises `max_depth` by 2 up to 10.

## Adjustment rules

The evolve command reads `docs/logs` and `docs/champion.json`, then proposes up to three mutants of the champion. Each mutant changes one gene: the feature subset, the preprocess spec, or the architecture. A genome already present in the logs is not trained again.

Feature changes alternate by generation. An odd generation adds the unused pool column with the largest absolute correlation to the champion residual on the search months. An even generation drops the column with the smallest permutation importance on a search sample of at most 2,000 rows. When that single-column move was already tried from the same parent, and every column of a feature family is inside the pool, an odd generation adds that family as a block and an even generation removes it. A feature set always keeps at least one column.

Preprocess changes the first knob not yet tried from this parent, in this order: numeric_impute (median, mean, zero); scale (standard, none, minmax); categorical (onehot, ordinal, target, bin); rare_min_count (10, 30, 100). The new value is the first choice in that list that differs from the current value.

Architecture regularizes when (search RMSE - train RMSE) / train RMSE is above 0.15. Otherwise it adds capacity, inside the cap in this model's note.

A mutant replaces the champion only when its search RMSE is at least 1% lower. The loop stops at the generation cap, when a generation accepts nothing, or when every proposed mutant is already in the logs. Confirm RMSE is scored once after search stops and is not an acceptance signal. Predict writes a summary and does not change the genome.

Labeled months are cut in time. Confirm is the last `val_months` approval months. Search is the `val_months` immediately before confirm. Fit is every earlier month. The preprocessor is fit on the fit months only.

## Current champion

<!-- champion:start -->
- Run: `20261030T124013757421_xgb_824fde31`
- Log: `models/xgb/docs/logs/20261030T124013757421_xgb_824fde31.json`
- Feature hash: `ede87d75909f`
- Genome hash: `824fde31757845d5`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `longitude`, `year_completed`, `subzone`, `building_age_years`, `flat_age_from_lease`, `nearest_open_mrt_code`, `nearest_open_mrt_name`, `nearest_mall_name`, `dist_nearest_dt_km`, `n_open_mrt_within_1km`, `n_planned_mrt_within_1km`, `n_secondary_within_1km`, `n_mall_within_1km`, `coe_e_premium_lag1`, `dist_nearest_future_line_open_km`, `dist_nearest_future_line_planned_km`, `dist_to_cbd_km`, `bank_basket_adj_close_lag1`, `mean_rent_postal_sector`, `mean_rent_postal_district`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 441.235845
- Search RMSE: 459.506905
- Confirm RMSE: 466.279228
- Confirm log: `models/xgb/docs/logs/20261030T124013904622_xgb_824fde31.json`
- Row cap: full training table
<!-- champion:end -->

## Lineage

<!-- lineage:start -->
Regenerated from `docs/logs`. Do not edit the metrics in this section by hand.

### `20261022T214026587580_xgb_332e97f1`

- Command: train
- Parent: `None`
- Generation: None
- Mutation: `{"axis": "train"}`
- Feature hash: `170e95b30b54`
- Genome hash: `332e97f125f93d91`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 442.412472
- Search RMSE: 459.892220
- Decision: accepted. first saved best; there was no earlier snapshot
- Log: `models/xgb/docs/logs/20261022T214026587580_xgb_332e97f1.json`

### `20261022T214047709370_xgb_e566b00a`

- Command: train
- Parent: `20261022T214026587580_xgb_332e97f1`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "latitude", "round": 1}`
- Feature hash: `99b4d5d7749f`
- Genome hash: `e566b00a3cf39d34`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `latitude`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 441.480851
- Search RMSE: 460.425166
- Decision: rejected. not promising because search RMSE 460.425166 is not at least 1% below 459.892220 from 20261022T214026587580_xgb_332e97f1 after adding latitude
- Log: `models/xgb/docs/logs/20261022T214047709370_xgb_e566b00a.json`

### `20261022T214102818150_xgb_6c1f6a72`

- Command: train
- Parent: `20261022T214026587580_xgb_332e97f1`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "longitude", "round": 1}`
- Feature hash: `c27505a182db`
- Genome hash: `6c1f6a72745d3bb2`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `longitude`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 440.812718
- Search RMSE: 459.870787
- Decision: rejected. not promising because search RMSE 459.870787 is not at least 1% below 459.892220 from 20261022T214026587580_xgb_332e97f1 after adding longitude
- Log: `models/xgb/docs/logs/20261022T214102818150_xgb_6c1f6a72.json`

### `20261022T214122664081_xgb_22cb5e88`

- Command: train
- Parent: `20261022T214026587580_xgb_332e97f1`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "postal_code", "round": 1}`
- Feature hash: `e62a952e566d`
- Genome hash: `22cb5e8847cb60fe`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `postal_code`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 420.057735
- Search RMSE: 467.232381
- Decision: rejected. not promising because search RMSE 467.232381 is not at least 1% below 459.892220 from 20261022T214026587580_xgb_332e97f1 after adding postal_code
- Log: `models/xgb/docs/logs/20261022T214122664081_xgb_22cb5e88.json`

### `20261022T214138439626_xgb_09d97a28`

- Command: train
- Parent: `20261022T214026587580_xgb_332e97f1`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "max_floor", "round": 1}`
- Feature hash: `6d16b7bd0382`
- Genome hash: `09d97a288decb5c3`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `max_floor`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 441.785558
- Search RMSE: 460.512663
- Decision: rejected. not promising because search RMSE 460.512663 is not at least 1% below 459.892220 from 20261022T214026587580_xgb_332e97f1 after adding max_floor
- Log: `models/xgb/docs/logs/20261022T214138439626_xgb_09d97a28.json`

### `20261022T214154481486_xgb_1bbdde7a`

- Command: train
- Parent: `20261022T214026587580_xgb_332e97f1`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "year_completed", "round": 1}`
- Feature hash: `b2d2d8a929b6`
- Genome hash: `1bbdde7ad63221dd`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `year_completed`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 440.001901
- Search RMSE: 459.808343
- Decision: rejected. not promising because search RMSE 459.808343 is not at least 1% below 459.892220 from 20261022T214026587580_xgb_332e97f1 after adding year_completed
- Log: `models/xgb/docs/logs/20261022T214154481486_xgb_1bbdde7a.json`

### `20261022T214213776768_xgb_685922e6`

- Command: train
- Parent: `20261022T214026587580_xgb_332e97f1`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "subzone", "round": 1}`
- Feature hash: `19ad5218a78e`
- Genome hash: `685922e6b1019c6d`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `subzone`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 442.406171
- Search RMSE: 459.889485
- Decision: rejected. not promising because search RMSE 459.889485 is not at least 1% below 459.892220 from 20261022T214026587580_xgb_332e97f1 after adding subzone
- Log: `models/xgb/docs/logs/20261022T214213776768_xgb_685922e6.json`

### `20261022T214231090872_xgb_eaccb8f3`

- Command: train
- Parent: `20261022T214026587580_xgb_332e97f1`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "planning_area", "round": 1}`
- Feature hash: `212df8d837f8`
- Genome hash: `eaccb8f3a61d58ca`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `planning_area`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 441.231591
- Search RMSE: 460.141813
- Decision: rejected. not promising because search RMSE 460.141813 is not at least 1% below 459.892220 from 20261022T214026587580_xgb_332e97f1 after adding planning_area
- Log: `models/xgb/docs/logs/20261022T214231090872_xgb_eaccb8f3.json`

### `20261022T214248242879_xgb_75f79b2d`

- Command: train
- Parent: `20261022T214026587580_xgb_332e97f1`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "region", "round": 1}`
- Feature hash: `fd9247992a0a`
- Genome hash: `75f79b2d5a3a0e6d`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `region`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 442.412472
- Search RMSE: 459.892220
- Decision: rejected. not promising because search RMSE 459.892220 is not at least 1% below 459.892220 from 20261022T214026587580_xgb_332e97f1 after adding region
- Log: `models/xgb/docs/logs/20261022T214248242879_xgb_75f79b2d.json`

### `20261022T214304203341_xgb_81a97572`

- Command: train
- Parent: `20261022T214026587580_xgb_332e97f1`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "building_age_years", "round": 1}`
- Feature hash: `c234d75ff5d3`
- Genome hash: `81a97572b2db0469`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `building_age_years`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 441.337890
- Search RMSE: 459.555092
- Decision: rejected. not promising because search RMSE 459.555092 is not at least 1% below 459.892220 from 20261022T214026587580_xgb_332e97f1 after adding building_age_years
- Log: `models/xgb/docs/logs/20261022T214304203341_xgb_81a97572.json`

### `20261022T214320237014_xgb_51a33218`

- Command: train
- Parent: `20261022T214026587580_xgb_332e97f1`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "lease_vs_completion_gap", "round": 1}`
- Feature hash: `d33f593517bd`
- Genome hash: `51a332189dcab27d`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `lease_vs_completion_gap`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 443.138733
- Search RMSE: 460.253915
- Decision: rejected. not promising because search RMSE 460.253915 is not at least 1% below 459.892220 from 20261022T214026587580_xgb_332e97f1 after adding lease_vs_completion_gap
- Log: `models/xgb/docs/logs/20261022T214320237014_xgb_51a33218.json`

### `20261022T214334113333_xgb_b6272e32`

- Command: train
- Parent: `20261022T214026587580_xgb_332e97f1`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "flat_age_from_lease", "round": 1}`
- Feature hash: `0ae4cfca946e`
- Genome hash: `b6272e3248b8f69a`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `flat_age_from_lease`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 442.538492
- Search RMSE: 459.842593
- Decision: rejected. not promising because search RMSE 459.842593 is not at least 1% below 459.892220 from 20261022T214026587580_xgb_332e97f1 after adding flat_age_from_lease
- Log: `models/xgb/docs/logs/20261022T214334113333_xgb_b6272e32.json`

### `20261022T214348582059_xgb_cd85b4be`

- Command: train
- Parent: `20261022T214026587580_xgb_332e97f1`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "dist_nearest_open_mrt_km", "round": 1}`
- Feature hash: `0c2594d85422`
- Genome hash: `cd85b4be332d7cca`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `dist_nearest_open_mrt_km`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 440.910590
- Search RMSE: 459.947723
- Decision: rejected. not promising because search RMSE 459.947723 is not at least 1% below 459.892220 from 20261022T214026587580_xgb_332e97f1 after adding dist_nearest_open_mrt_km
- Log: `models/xgb/docs/logs/20261022T214348582059_xgb_cd85b4be.json`

### `20261022T214402157339_xgb_89984b91`

- Command: train
- Parent: `20261022T214026587580_xgb_332e97f1`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_open_mrt_code", "round": 1}`
- Feature hash: `42451fae9f20`
- Genome hash: `89984b91d9d74dcc`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `nearest_open_mrt_code`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 441.422009
- Search RMSE: 459.885938
- Decision: rejected. not promising because search RMSE 459.885938 is not at least 1% below 459.892220 from 20261022T214026587580_xgb_332e97f1 after adding nearest_open_mrt_code
- Log: `models/xgb/docs/logs/20261022T214402157339_xgb_89984b91.json`

### `20261022T214416484508_xgb_d2ca52e1`

- Command: train
- Parent: `20261022T214026587580_xgb_332e97f1`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_open_mrt_name", "round": 1}`
- Feature hash: `57326fd029e4`
- Genome hash: `d2ca52e19d92dc77`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `nearest_open_mrt_name`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 441.422009
- Search RMSE: 459.885938
- Decision: rejected. not promising because search RMSE 459.885938 is not at least 1% below 459.892220 from 20261022T214026587580_xgb_332e97f1 after adding nearest_open_mrt_name
- Log: `models/xgb/docs/logs/20261022T214416484508_xgb_d2ca52e1.json`

### `20261022T214429597559_xgb_d99dc222`

- Command: train
- Parent: `20261022T214026587580_xgb_332e97f1`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_open_mrt_line", "round": 1}`
- Feature hash: `f6e2fc2120da`
- Genome hash: `d99dc22202d65409`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `nearest_open_mrt_line`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 440.870890
- Search RMSE: 460.553732
- Decision: rejected. not promising because search RMSE 460.553732 is not at least 1% below 459.892220 from 20261022T214026587580_xgb_332e97f1 after adding nearest_open_mrt_line
- Log: `models/xgb/docs/logs/20261022T214429597559_xgb_d99dc222.json`

### `20261022T214443318429_xgb_31c3aee1`

- Command: train
- Parent: `20261022T214026587580_xgb_332e97f1`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "dist_nearest_primary_km", "round": 1}`
- Feature hash: `ec8fcfbbdc5f`
- Genome hash: `31c3aee1821a031f`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `dist_nearest_primary_km`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 441.069958
- Search RMSE: 460.147187
- Decision: rejected. not promising because search RMSE 460.147187 is not at least 1% below 459.892220 from 20261022T214026587580_xgb_332e97f1 after adding dist_nearest_primary_km
- Log: `models/xgb/docs/logs/20261022T214443318429_xgb_31c3aee1.json`

### `20261022T214456403667_xgb_5541123e`

- Command: train
- Parent: `20261022T214026587580_xgb_332e97f1`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_primary_name", "round": 1}`
- Feature hash: `3b600d364a2c`
- Genome hash: `5541123ef3e1f073`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `nearest_primary_name`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 441.650045
- Search RMSE: 460.556323
- Decision: rejected. not promising because search RMSE 460.556323 is not at least 1% below 459.892220 from 20261022T214026587580_xgb_332e97f1 after adding nearest_primary_name
- Log: `models/xgb/docs/logs/20261022T214456403667_xgb_5541123e.json`

### `20261022T214509479409_xgb_5f1b148a`

- Command: train
- Parent: `20261022T214026587580_xgb_332e97f1`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "n_primary_within_1km", "round": 1}`
- Feature hash: `4948f0d7026e`
- Genome hash: `5f1b148ad9ef880d`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `n_primary_within_1km`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 441.498684
- Search RMSE: 460.220364
- Decision: rejected. not promising because search RMSE 460.220364 is not at least 1% below 459.892220 from 20261022T214026587580_xgb_332e97f1 after adding n_primary_within_1km
- Log: `models/xgb/docs/logs/20261022T214509479409_xgb_5f1b148a.json`

### `20261022T214522125152_xgb_304db93b`

- Command: train
- Parent: `20261022T214026587580_xgb_332e97f1`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "dist_nearest_mall_km", "round": 1}`
- Feature hash: `eb02b9676679`
- Genome hash: `304db93b88a7d3ed`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `dist_nearest_mall_km`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 442.118146
- Search RMSE: 460.512741
- Decision: rejected. not promising because search RMSE 460.512741 is not at least 1% below 459.892220 from 20261022T214026587580_xgb_332e97f1 after adding dist_nearest_mall_km
- Log: `models/xgb/docs/logs/20261022T214522125152_xgb_304db93b.json`

### `20261022T214534810808_xgb_2f2e036f`

- Command: train
- Parent: `20261022T214026587580_xgb_332e97f1`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_mall_name", "round": 1}`
- Feature hash: `dfa1a626ebab`
- Genome hash: `2f2e036f404d07e6`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `nearest_mall_name`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 443.016562
- Search RMSE: 459.721673
- Decision: rejected. not promising because search RMSE 459.721673 is not at least 1% below 459.892220 from 20261022T214026587580_xgb_332e97f1 after adding nearest_mall_name
- Log: `models/xgb/docs/logs/20261022T214534810808_xgb_2f2e036f.json`

### `20261022T214548490044_xgb_b5603b92`

- Command: train
- Parent: `20261022T214026587580_xgb_332e97f1`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_ns_station_code", "round": 1}`
- Feature hash: `5ce89d055366`
- Genome hash: `b5603b925a7c5c5f`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `nearest_ns_station_code`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 442.543198
- Search RMSE: 460.778433
- Decision: rejected. not promising because search RMSE 460.778433 is not at least 1% below 459.892220 from 20261022T214026587580_xgb_332e97f1 after adding nearest_ns_station_code
- Log: `models/xgb/docs/logs/20261022T214548490044_xgb_b5603b92.json`

### `20261022T214601394284_xgb_2b8f8dcb`

- Command: train
- Parent: `20261022T214026587580_xgb_332e97f1`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_ns_station_name", "round": 1}`
- Feature hash: `70914731bd06`
- Genome hash: `2b8f8dcb5d231759`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `nearest_ns_station_name`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 442.543198
- Search RMSE: 460.778433
- Decision: rejected. not promising because search RMSE 460.778433 is not at least 1% below 459.892220 from 20261022T214026587580_xgb_332e97f1 after adding nearest_ns_station_name
- Log: `models/xgb/docs/logs/20261022T214601394284_xgb_2b8f8dcb.json`

### `20261022T214615082480_xgb_1fe7f53e`

- Command: train
- Parent: `20261022T214026587580_xgb_332e97f1`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "dist_nearest_ns_km", "round": 1}`
- Feature hash: `8d6e42f11adc`
- Genome hash: `1fe7f53e54aa4b67`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `dist_nearest_ns_km`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 442.510568
- Search RMSE: 460.463316
- Decision: rejected. not promising because search RMSE 460.463316 is not at least 1% below 459.892220 from 20261022T214026587580_xgb_332e97f1 after adding dist_nearest_ns_km
- Log: `models/xgb/docs/logs/20261022T214615082480_xgb_1fe7f53e.json`

### `20261022T214627998534_xgb_e0614443`

- Command: train
- Parent: `20261022T214026587580_xgb_332e97f1`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_ew_station_code", "round": 1}`
- Feature hash: `033e0d7f6436`
- Genome hash: `e06144435c6c7e3b`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `nearest_ew_station_code`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 443.254575
- Search RMSE: 460.032850
- Decision: rejected. not promising because search RMSE 460.032850 is not at least 1% below 459.892220 from 20261022T214026587580_xgb_332e97f1 after adding nearest_ew_station_code
- Log: `models/xgb/docs/logs/20261022T214627998534_xgb_e0614443.json`

### `20261022T214642018118_xgb_fbbccef6`

- Command: train
- Parent: `20261022T214026587580_xgb_332e97f1`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_ew_station_name", "round": 1}`
- Feature hash: `cc0d61a0df40`
- Genome hash: `fbbccef6daeac137`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `nearest_ew_station_name`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 443.254575
- Search RMSE: 460.032850
- Decision: rejected. not promising because search RMSE 460.032850 is not at least 1% below 459.892220 from 20261022T214026587580_xgb_332e97f1 after adding nearest_ew_station_name
- Log: `models/xgb/docs/logs/20261022T214642018118_xgb_fbbccef6.json`

### `20261022T214654803903_xgb_51f578c0`

- Command: train
- Parent: `20261022T214026587580_xgb_332e97f1`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "dist_nearest_ew_km", "round": 1}`
- Feature hash: `8db553894f91`
- Genome hash: `51f578c0b39ea68b`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `dist_nearest_ew_km`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 442.908071
- Search RMSE: 460.145911
- Decision: rejected. not promising because search RMSE 460.145911 is not at least 1% below 459.892220 from 20261022T214026587580_xgb_332e97f1 after adding dist_nearest_ew_km
- Log: `models/xgb/docs/logs/20261022T214654803903_xgb_51f578c0.json`

### `20261022T214707840627_xgb_5df04488`

- Command: train
- Parent: `20261022T214026587580_xgb_332e97f1`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_ne_station_code", "round": 1}`
- Feature hash: `491bf0c825f4`
- Genome hash: `5df04488871fae2a`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `nearest_ne_station_code`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 442.926668
- Search RMSE: 459.943276
- Decision: rejected. not promising because search RMSE 459.943276 is not at least 1% below 459.892220 from 20261022T214026587580_xgb_332e97f1 after adding nearest_ne_station_code
- Log: `models/xgb/docs/logs/20261022T214707840627_xgb_5df04488.json`

### `20261022T214720946821_xgb_d61b5927`

- Command: train
- Parent: `20261022T214026587580_xgb_332e97f1`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_ne_station_name", "round": 1}`
- Feature hash: `0b51c9b66206`
- Genome hash: `d61b5927a214c9e3`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `nearest_ne_station_name`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 442.926668
- Search RMSE: 459.943276
- Decision: rejected. not promising because search RMSE 459.943276 is not at least 1% below 459.892220 from 20261022T214026587580_xgb_332e97f1 after adding nearest_ne_station_name
- Log: `models/xgb/docs/logs/20261022T214720946821_xgb_d61b5927.json`

### `20261022T214733980437_xgb_d31aed17`

- Command: train
- Parent: `20261022T214026587580_xgb_332e97f1`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "dist_nearest_ne_km", "round": 1}`
- Feature hash: `58386a513add`
- Genome hash: `d31aed17ac74ec0f`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `dist_nearest_ne_km`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 441.289098
- Search RMSE: 460.171766
- Decision: rejected. not promising because search RMSE 460.171766 is not at least 1% below 459.892220 from 20261022T214026587580_xgb_332e97f1 after adding dist_nearest_ne_km
- Log: `models/xgb/docs/logs/20261022T214733980437_xgb_d31aed17.json`

### `20261022T214748262044_xgb_36041dfd`

- Command: train
- Parent: `20261022T214026587580_xgb_332e97f1`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_cc_station_code", "round": 1}`
- Feature hash: `e7df40e75f9a`
- Genome hash: `36041dfd64811995`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `nearest_cc_station_code`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 441.094144
- Search RMSE: 460.256892
- Decision: rejected. not promising because search RMSE 460.256892 is not at least 1% below 459.892220 from 20261022T214026587580_xgb_332e97f1 after adding nearest_cc_station_code
- Log: `models/xgb/docs/logs/20261022T214748262044_xgb_36041dfd.json`

### `20261022T214801700157_xgb_b7246fe9`

- Command: train
- Parent: `20261022T214026587580_xgb_332e97f1`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_cc_station_name", "round": 1}`
- Feature hash: `0e0891a68917`
- Genome hash: `b7246fe94ca6a46c`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `nearest_cc_station_name`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 441.094144
- Search RMSE: 460.256892
- Decision: rejected. not promising because search RMSE 460.256892 is not at least 1% below 459.892220 from 20261022T214026587580_xgb_332e97f1 after adding nearest_cc_station_name
- Log: `models/xgb/docs/logs/20261022T214801700157_xgb_b7246fe9.json`

### `20261022T214815367387_xgb_8250b185`

- Command: train
- Parent: `20261022T214026587580_xgb_332e97f1`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "dist_nearest_cc_km", "round": 1}`
- Feature hash: `73966c4f212f`
- Genome hash: `8250b18552af1d31`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `dist_nearest_cc_km`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 443.271171
- Search RMSE: 460.190654
- Decision: rejected. not promising because search RMSE 460.190654 is not at least 1% below 459.892220 from 20261022T214026587580_xgb_332e97f1 after adding dist_nearest_cc_km
- Log: `models/xgb/docs/logs/20261022T214815367387_xgb_8250b185.json`

### `20261022T214828291313_xgb_eaf8cc40`

- Command: train
- Parent: `20261022T214026587580_xgb_332e97f1`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_dt_station_code", "round": 1}`
- Feature hash: `248dc8a22f33`
- Genome hash: `eaf8cc4073ec3c08`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `nearest_dt_station_code`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 441.958652
- Search RMSE: 460.075610
- Decision: rejected. not promising because search RMSE 460.075610 is not at least 1% below 459.892220 from 20261022T214026587580_xgb_332e97f1 after adding nearest_dt_station_code
- Log: `models/xgb/docs/logs/20261022T214828291313_xgb_eaf8cc40.json`

### `20261022T214841964019_xgb_389eb4d3`

- Command: train
- Parent: `20261022T214026587580_xgb_332e97f1`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_dt_station_name", "round": 1}`
- Feature hash: `48c44e2e40fe`
- Genome hash: `389eb4d3092b6849`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `nearest_dt_station_name`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 441.958652
- Search RMSE: 460.075610
- Decision: rejected. not promising because search RMSE 460.075610 is not at least 1% below 459.892220 from 20261022T214026587580_xgb_332e97f1 after adding nearest_dt_station_name
- Log: `models/xgb/docs/logs/20261022T214841964019_xgb_389eb4d3.json`

### `20261022T214855043170_xgb_e3fc6302`

- Command: train
- Parent: `20261022T214026587580_xgb_332e97f1`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "dist_nearest_dt_km", "round": 1}`
- Feature hash: `7e6bc08d4ff7`
- Genome hash: `e3fc6302bdf95ea5`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `dist_nearest_dt_km`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 442.914877
- Search RMSE: 459.824249
- Decision: rejected. not promising because search RMSE 459.824249 is not at least 1% below 459.892220 from 20261022T214026587580_xgb_332e97f1 after adding dist_nearest_dt_km
- Log: `models/xgb/docs/logs/20261022T214855043170_xgb_e3fc6302.json`

### `20261022T214910673707_xgb_56b12c51`

- Command: train
- Parent: `20261022T214026587580_xgb_332e97f1`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_te_station_code", "round": 1}`
- Feature hash: `4f05b6ddefc5`
- Genome hash: `56b12c5168640738`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `nearest_te_station_code`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 441.098410
- Search RMSE: 460.350316
- Decision: rejected. not promising because search RMSE 460.350316 is not at least 1% below 459.892220 from 20261022T214026587580_xgb_332e97f1 after adding nearest_te_station_code
- Log: `models/xgb/docs/logs/20261022T214910673707_xgb_56b12c51.json`

### `20261022T214924352956_xgb_54370d82`

- Command: train
- Parent: `20261022T214026587580_xgb_332e97f1`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_te_station_name", "round": 1}`
- Feature hash: `1139130c8f31`
- Genome hash: `54370d8227b97175`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `nearest_te_station_name`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 441.098410
- Search RMSE: 460.350316
- Decision: rejected. not promising because search RMSE 460.350316 is not at least 1% below 459.892220 from 20261022T214026587580_xgb_332e97f1 after adding nearest_te_station_name
- Log: `models/xgb/docs/logs/20261022T214924352956_xgb_54370d82.json`

### `20261022T214938107940_xgb_4aaa50cf`

- Command: train
- Parent: `20261022T214026587580_xgb_332e97f1`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "dist_nearest_te_km", "round": 1}`
- Feature hash: `5cef183a61db`
- Genome hash: `4aaa50cfb10c08ff`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `dist_nearest_te_km`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 440.698066
- Search RMSE: 459.914576
- Decision: rejected. not promising because search RMSE 459.914576 is not at least 1% below 459.892220 from 20261022T214026587580_xgb_332e97f1 after adding dist_nearest_te_km
- Log: `models/xgb/docs/logs/20261022T214938107940_xgb_4aaa50cf.json`

### `20261022T214951238676_xgb_2ead0c09`

- Command: train
- Parent: `20261022T214026587580_xgb_332e97f1`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "n_open_mrt_within_0_5km", "round": 1}`
- Feature hash: `5219387275ac`
- Genome hash: `2ead0c0979f528b4`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `n_open_mrt_within_0_5km`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 440.994182
- Search RMSE: 460.016697
- Decision: rejected. not promising because search RMSE 460.016697 is not at least 1% below 459.892220 from 20261022T214026587580_xgb_332e97f1 after adding n_open_mrt_within_0_5km
- Log: `models/xgb/docs/logs/20261022T214951238676_xgb_2ead0c09.json`

### `20261022T215004447266_xgb_bc35be97`

- Command: train
- Parent: `20261022T214026587580_xgb_332e97f1`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "n_open_mrt_within_1km", "round": 1}`
- Feature hash: `692f243ba340`
- Genome hash: `bc35be97e8664897`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `n_open_mrt_within_1km`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 441.278197
- Search RMSE: 459.515748
- Decision: rejected. not promising because search RMSE 459.515748 is not at least 1% below 459.892220 from 20261022T214026587580_xgb_332e97f1 after adding n_open_mrt_within_1km
- Log: `models/xgb/docs/logs/20261022T215004447266_xgb_bc35be97.json`

### `20261022T215018643107_xgb_0f78ad63`

- Command: train
- Parent: `20261022T214026587580_xgb_332e97f1`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "n_open_mrt_within_2km", "round": 1}`
- Feature hash: `2e76b96e95c7`
- Genome hash: `0f78ad63d25ee807`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `n_open_mrt_within_2km`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 441.559889
- Search RMSE: 460.220477
- Decision: rejected. not promising because search RMSE 460.220477 is not at least 1% below 459.892220 from 20261022T214026587580_xgb_332e97f1 after adding n_open_mrt_within_2km
- Log: `models/xgb/docs/logs/20261022T215018643107_xgb_0f78ad63.json`

### `20261022T215032208256_xgb_b18ca650`

- Command: train
- Parent: `20261022T214026587580_xgb_332e97f1`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "dist_nearest_planned_mrt_km", "round": 1}`
- Feature hash: `d68a3dae1af5`
- Genome hash: `b18ca65048c15895`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `dist_nearest_planned_mrt_km`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 442.623098
- Search RMSE: 460.368522
- Decision: rejected. not promising because search RMSE 460.368522 is not at least 1% below 459.892220 from 20261022T214026587580_xgb_332e97f1 after adding dist_nearest_planned_mrt_km
- Log: `models/xgb/docs/logs/20261022T215032208256_xgb_b18ca650.json`

### `20261022T215046036390_xgb_02403083`

- Command: train
- Parent: `20261022T214026587580_xgb_332e97f1`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "n_planned_mrt_within_1km", "round": 1}`
- Feature hash: `4fed35fb6804`
- Genome hash: `0240308320c2c16a`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `n_planned_mrt_within_1km`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 442.412171
- Search RMSE: 459.891829
- Decision: rejected. not promising because search RMSE 459.891829 is not at least 1% below 459.892220 from 20261022T214026587580_xgb_332e97f1 after adding n_planned_mrt_within_1km
- Log: `models/xgb/docs/logs/20261022T215046036390_xgb_02403083.json`

### `20261022T215059222536_xgb_d1fb7395`

- Command: train
- Parent: `20261022T214026587580_xgb_332e97f1`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "n_planned_mrt_within_2km", "round": 1}`
- Feature hash: `112eaabb4cdb`
- Genome hash: `d1fb7395bf4f5b82`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `n_planned_mrt_within_2km`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 442.412472
- Search RMSE: 459.892220
- Decision: rejected. not promising because search RMSE 459.892220 is not at least 1% below 459.892220 from 20261022T214026587580_xgb_332e97f1 after adding n_planned_mrt_within_2km
- Log: `models/xgb/docs/logs/20261022T215059222536_xgb_d1fb7395.json`

### `20261022T215113701200_xgb_8e9159c4`

- Command: train
- Parent: `20261022T214026587580_xgb_332e97f1`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_open_mrt_is_interchange", "round": 1}`
- Feature hash: `e8afb2136c5c`
- Genome hash: `8e9159c4db77e38c`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `nearest_open_mrt_is_interchange`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 442.412472
- Search RMSE: 459.892220
- Decision: rejected. not promising because search RMSE 459.892220 is not at least 1% below 459.892220 from 20261022T214026587580_xgb_332e97f1 after adding nearest_open_mrt_is_interchange
- Log: `models/xgb/docs/logs/20261022T215113701200_xgb_8e9159c4.json`

### `20261022T215126823966_xgb_8ba18852`

- Command: train
- Parent: `20261022T214026587580_xgb_332e97f1`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "n_primary_within_2km", "round": 1}`
- Feature hash: `9f58d18094df`
- Genome hash: `8ba18852bd1fbf5c`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `n_primary_within_2km`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 443.094061
- Search RMSE: 460.292216
- Decision: rejected. not promising because search RMSE 460.292216 is not at least 1% below 459.892220 from 20261022T214026587580_xgb_332e97f1 after adding n_primary_within_2km
- Log: `models/xgb/docs/logs/20261022T215126823966_xgb_8ba18852.json`

### `20261022T215140612398_xgb_6a24aadc`

- Command: train
- Parent: `20261022T214026587580_xgb_332e97f1`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "dist_nearest_secondary_km", "round": 1}`
- Feature hash: `3b84c255aaae`
- Genome hash: `6a24aadcac1e5c98`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `dist_nearest_secondary_km`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 441.876593
- Search RMSE: 460.253375
- Decision: rejected. not promising because search RMSE 460.253375 is not at least 1% below 459.892220 from 20261022T214026587580_xgb_332e97f1 after adding dist_nearest_secondary_km
- Log: `models/xgb/docs/logs/20261022T215140612398_xgb_6a24aadc.json`

### `20261022T215154302221_xgb_96275f54`

- Command: train
- Parent: `20261022T214026587580_xgb_332e97f1`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_secondary_name", "round": 1}`
- Feature hash: `4932501d7490`
- Genome hash: `96275f54000d4553`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `nearest_secondary_name`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 443.105438
- Search RMSE: 460.392113
- Decision: rejected. not promising because search RMSE 460.392113 is not at least 1% below 459.892220 from 20261022T214026587580_xgb_332e97f1 after adding nearest_secondary_name
- Log: `models/xgb/docs/logs/20261022T215154302221_xgb_96275f54.json`

### `20261022T215207575483_xgb_9fecdb63`

- Command: train
- Parent: `20261022T214026587580_xgb_332e97f1`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "n_secondary_within_1km", "round": 1}`
- Feature hash: `f92f056e77f5`
- Genome hash: `9fecdb6378b8e200`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `n_secondary_within_1km`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 442.680028
- Search RMSE: 459.762652
- Decision: rejected. not promising because search RMSE 459.762652 is not at least 1% below 459.892220 from 20261022T214026587580_xgb_332e97f1 after adding n_secondary_within_1km
- Log: `models/xgb/docs/logs/20261022T215207575483_xgb_9fecdb63.json`

### `20261022T215222958358_xgb_3bbb4ac0`

- Command: train
- Parent: `20261022T214026587580_xgb_332e97f1`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "dist_nearest_school_km", "round": 1}`
- Feature hash: `482ab5dda0e9`
- Genome hash: `3bbb4ac0f1e436ef`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `dist_nearest_school_km`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 441.115902
- Search RMSE: 460.316144
- Decision: rejected. not promising because search RMSE 460.316144 is not at least 1% below 459.892220 from 20261022T214026587580_xgb_332e97f1 after adding dist_nearest_school_km
- Log: `models/xgb/docs/logs/20261022T215222958358_xgb_3bbb4ac0.json`

### `20261022T215236311965_xgb_21215967`

- Command: train
- Parent: `20261022T214026587580_xgb_332e97f1`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_school_name", "round": 1}`
- Feature hash: `e657e5902602`
- Genome hash: `2121596758e3ae19`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `nearest_school_name`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 441.560603
- Search RMSE: 460.782782
- Decision: rejected. not promising because search RMSE 460.782782 is not at least 1% below 459.892220 from 20261022T214026587580_xgb_332e97f1 after adding nearest_school_name
- Log: `models/xgb/docs/logs/20261022T215236311965_xgb_21215967.json`

### `20261022T215250700977_xgb_ae11dcee`

- Command: train
- Parent: `20261022T214026587580_xgb_332e97f1`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "n_school_within_1km", "round": 1}`
- Feature hash: `b9907fe19ce9`
- Genome hash: `ae11dceeff89cbfc`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `n_school_within_1km`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 442.907537
- Search RMSE: 460.239399
- Decision: rejected. not promising because search RMSE 460.239399 is not at least 1% below 459.892220 from 20261022T214026587580_xgb_332e97f1 after adding n_school_within_1km
- Log: `models/xgb/docs/logs/20261022T215250700977_xgb_ae11dcee.json`

### `20261022T215304329842_xgb_bfca98ef`

- Command: train
- Parent: `20261022T214026587580_xgb_332e97f1`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "n_mall_within_1km", "round": 1}`
- Feature hash: `a98c9f4224a5`
- Genome hash: `bfca98efeeb1d96c`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `n_mall_within_1km`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 441.213918
- Search RMSE: 459.656831
- Decision: rejected. not promising because search RMSE 459.656831 is not at least 1% below 459.892220 from 20261022T214026587580_xgb_332e97f1 after adding n_mall_within_1km
- Log: `models/xgb/docs/logs/20261022T215304329842_xgb_bfca98ef.json`

### `20261022T215318478591_xgb_168b8bfe`

- Command: train
- Parent: `20261022T214026587580_xgb_332e97f1`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "n_mall_within_2km", "round": 1}`
- Feature hash: `282ff6f12a0d`
- Genome hash: `168b8bfed24ad882`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `n_mall_within_2km`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 443.195910
- Search RMSE: 459.974609
- Decision: rejected. not promising because search RMSE 459.974609 is not at least 1% below 459.892220 from 20261022T214026587580_xgb_332e97f1 after adding n_mall_within_2km
- Log: `models/xgb/docs/logs/20261022T215318478591_xgb_168b8bfe.json`

### `20261022T215332427750_xgb_a0e31f8c`

- Command: train
- Parent: `20261022T214026587580_xgb_332e97f1`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "coe_a_premium_lag1", "round": 1}`
- Feature hash: `ca6251adee59`
- Genome hash: `a0e31f8c1055be20`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_a_premium_lag1`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 441.588432
- Search RMSE: 460.214052
- Decision: rejected. not promising because search RMSE 460.214052 is not at least 1% below 459.892220 from 20261022T214026587580_xgb_332e97f1 after adding coe_a_premium_lag1
- Log: `models/xgb/docs/logs/20261022T215332427750_xgb_a0e31f8c.json`

### `20261022T215347640988_xgb_bda16d10`

- Command: train
- Parent: `20261022T214026587580_xgb_332e97f1`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "coe_a_premium_mom_change", "round": 1}`
- Feature hash: `ecd53635e5ea`
- Genome hash: `bda16d10a7a1ba53`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_a_premium_mom_change`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 441.503003
- Search RMSE: 459.902386
- Decision: rejected. not promising because search RMSE 459.902386 is not at least 1% below 459.892220 from 20261022T214026587580_xgb_332e97f1 after adding coe_a_premium_mom_change
- Log: `models/xgb/docs/logs/20261022T215347640988_xgb_bda16d10.json`

### `20261022T215400949066_xgb_25c817bc`

- Command: train
- Parent: `20261022T214026587580_xgb_332e97f1`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "coe_a_bids_per_quota_lag1", "round": 1}`
- Feature hash: `88b869c77fc2`
- Genome hash: `25c817bcb5d5f46b`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_a_bids_per_quota_lag1`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 442.804438
- Search RMSE: 460.096150
- Decision: rejected. not promising because search RMSE 460.096150 is not at least 1% below 459.892220 from 20261022T214026587580_xgb_332e97f1 after adding coe_a_bids_per_quota_lag1
- Log: `models/xgb/docs/logs/20261022T215400949066_xgb_25c817bc.json`

### `20261022T215415617450_xgb_7e2e8497`

- Command: train
- Parent: `20261022T214026587580_xgb_332e97f1`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "coe_b_premium_lag1", "round": 1}`
- Feature hash: `aef8257585e0`
- Genome hash: `7e2e849757896a48`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_b_premium_lag1`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 441.282096
- Search RMSE: 460.121757
- Decision: rejected. not promising because search RMSE 460.121757 is not at least 1% below 459.892220 from 20261022T214026587580_xgb_332e97f1 after adding coe_b_premium_lag1
- Log: `models/xgb/docs/logs/20261022T215415617450_xgb_7e2e8497.json`

### `20261022T215428641114_xgb_00203866`

- Command: train
- Parent: `20261022T214026587580_xgb_332e97f1`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "coe_b_premium_mom_change", "round": 1}`
- Feature hash: `8302b3652167`
- Genome hash: `00203866f10f57c8`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_b_premium_mom_change`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 442.539692
- Search RMSE: 460.489561
- Decision: rejected. not promising because search RMSE 460.489561 is not at least 1% below 459.892220 from 20261022T214026587580_xgb_332e97f1 after adding coe_b_premium_mom_change
- Log: `models/xgb/docs/logs/20261022T215428641114_xgb_00203866.json`

### `20261022T215444015826_xgb_9fba3286`

- Command: train
- Parent: `20261022T214026587580_xgb_332e97f1`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "coe_b_bids_per_quota_lag1", "round": 1}`
- Feature hash: `4a7482aab3c7`
- Genome hash: `9fba3286273cbf75`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_b_bids_per_quota_lag1`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 442.167325
- Search RMSE: 460.454419
- Decision: rejected. not promising because search RMSE 460.454419 is not at least 1% below 459.892220 from 20261022T214026587580_xgb_332e97f1 after adding coe_b_bids_per_quota_lag1
- Log: `models/xgb/docs/logs/20261022T215444015826_xgb_9fba3286.json`

### `20261022T215457265421_xgb_7eb939e1`

- Command: train
- Parent: `20261022T214026587580_xgb_332e97f1`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "coe_e_premium_lag1", "round": 1}`
- Feature hash: `de6005d5c05c`
- Genome hash: `7eb939e179e9b056`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_e_premium_lag1`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 442.056729
- Search RMSE: 459.727285
- Decision: rejected. not promising because search RMSE 459.727285 is not at least 1% below 459.892220 from 20261022T214026587580_xgb_332e97f1 after adding coe_e_premium_lag1
- Log: `models/xgb/docs/logs/20261022T215457265421_xgb_7eb939e1.json`

### `20261022T215512592536_xgb_0277cd5e`

- Command: train
- Parent: `20261022T214026587580_xgb_332e97f1`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "coe_e_premium_mom_change", "round": 1}`
- Feature hash: `387e56d7e868`
- Genome hash: `0277cd5e68c42bb2`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_e_premium_mom_change`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 440.943706
- Search RMSE: 460.656737
- Decision: rejected. not promising because search RMSE 460.656737 is not at least 1% below 459.892220 from 20261022T214026587580_xgb_332e97f1 after adding coe_e_premium_mom_change
- Log: `models/xgb/docs/logs/20261022T215512592536_xgb_0277cd5e.json`

### `20261022T215526003232_xgb_af130c66`

- Command: train
- Parent: `20261022T214026587580_xgb_332e97f1`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "coe_e_bids_per_quota_lag1", "round": 1}`
- Feature hash: `8bb848ebe345`
- Genome hash: `af130c663773052d`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_e_bids_per_quota_lag1`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 441.524242
- Search RMSE: 460.518405
- Decision: rejected. not promising because search RMSE 460.518405 is not at least 1% below 459.892220 from 20261022T214026587580_xgb_332e97f1 after adding coe_e_bids_per_quota_lag1
- Log: `models/xgb/docs/logs/20261022T215526003232_xgb_af130c66.json`

### `20261022T215539310390_xgb_45cb2918`

- Command: train
- Parent: `20261022T214026587580_xgb_332e97f1`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "coe_a_premium_roll3", "round": 1}`
- Feature hash: `b09c5f9ba37b`
- Genome hash: `45cb2918f59649be`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_a_premium_roll3`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 441.694769
- Search RMSE: 460.389479
- Decision: rejected. not promising because search RMSE 460.389479 is not at least 1% below 459.892220 from 20261022T214026587580_xgb_332e97f1 after adding coe_a_premium_roll3
- Log: `models/xgb/docs/logs/20261022T215539310390_xgb_45cb2918.json`

### `20261022T215552721788_xgb_44b1d840`

- Command: train
- Parent: `20261022T214026587580_xgb_332e97f1`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "coe_c_premium_lag1", "round": 1}`
- Feature hash: `7b8f900c807f`
- Genome hash: `44b1d840299ad72b`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_c_premium_lag1`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 442.965984
- Search RMSE: 460.455254
- Decision: rejected. not promising because search RMSE 460.455254 is not at least 1% below 459.892220 from 20261022T214026587580_xgb_332e97f1 after adding coe_c_premium_lag1
- Log: `models/xgb/docs/logs/20261022T215552721788_xgb_44b1d840.json`

### `20261022T215607079927_xgb_6102e3a3`

- Command: train
- Parent: `20261022T214026587580_xgb_332e97f1`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "dist_nearest_future_line_open_km", "round": 1}`
- Feature hash: `109871ee8d0c`
- Genome hash: `6102e3a3f370f351`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `dist_nearest_future_line_open_km`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 441.548026
- Search RMSE: 459.882821
- Decision: rejected. not promising because search RMSE 459.882821 is not at least 1% below 459.892220 from 20261022T214026587580_xgb_332e97f1 after adding dist_nearest_future_line_open_km
- Log: `models/xgb/docs/logs/20261022T215607079927_xgb_6102e3a3.json`

### `20261022T215621043210_xgb_294ba791`

- Command: train
- Parent: `20261022T214026587580_xgb_332e97f1`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "dist_nearest_future_line_planned_km", "round": 1}`
- Feature hash: `b96b1461ee87`
- Genome hash: `294ba7910cebbee6`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `dist_nearest_future_line_planned_km`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 441.437323
- Search RMSE: 459.643518
- Decision: rejected. not promising because search RMSE 459.643518 is not at least 1% below 459.892220 from 20261022T214026587580_xgb_332e97f1 after adding dist_nearest_future_line_planned_km
- Log: `models/xgb/docs/logs/20261022T215621043210_xgb_294ba791.json`

### `20261022T215634410728_xgb_db9863d9`

- Command: train
- Parent: `20261022T214026587580_xgb_332e97f1`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "n_hdb_blocks_within_500m", "round": 1}`
- Feature hash: `95bfdabe2d04`
- Genome hash: `db9863d9b06e53f6`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `n_hdb_blocks_within_500m`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 442.452551
- Search RMSE: 460.435279
- Decision: rejected. not promising because search RMSE 460.435279 is not at least 1% below 459.892220 from 20261022T214026587580_xgb_332e97f1 after adding n_hdb_blocks_within_500m
- Log: `models/xgb/docs/logs/20261022T215634410728_xgb_db9863d9.json`

### `20261022T215648534315_xgb_179b83fc`

- Command: train
- Parent: `20261022T214026587580_xgb_332e97f1`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "dist_to_town_centroid_km", "round": 1}`
- Feature hash: `cbb5a64dccd3`
- Genome hash: `179b83fcfa84470e`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `dist_to_town_centroid_km`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 441.257244
- Search RMSE: 460.177296
- Decision: rejected. not promising because search RMSE 460.177296 is not at least 1% below 459.892220 from 20261022T214026587580_xgb_332e97f1 after adding dist_to_town_centroid_km
- Log: `models/xgb/docs/logs/20261022T215648534315_xgb_179b83fc.json`

### `20261022T215701729890_xgb_9dc833dd`

- Command: train
- Parent: `20261022T214026587580_xgb_332e97f1`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "dist_to_cbd_km", "round": 1}`
- Feature hash: `3a81f2034fb9`
- Genome hash: `9dc833dd654d262b`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `dist_to_cbd_km`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 440.812930
- Search RMSE: 459.261461
- Decision: rejected. not promising because search RMSE 459.261461 is not at least 1% below 459.892220 from 20261022T214026587580_xgb_332e97f1 after adding dist_to_cbd_km
- Log: `models/xgb/docs/logs/20261022T215701729890_xgb_9dc833dd.json`

### `20261022T215713638169_xgb_9e5ec2da`

- Command: train
- Parent: `20261022T214026587580_xgb_332e97f1`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "bank_basket_adj_close_lag1", "round": 1}`
- Feature hash: `40ef8a2b790f`
- Genome hash: `9e5ec2da42efd1b4`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `bank_basket_adj_close_lag1`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 440.881331
- Search RMSE: 459.770120
- Decision: rejected. not promising because search RMSE 459.770120 is not at least 1% below 459.892220 from 20261022T214026587580_xgb_332e97f1 after adding bank_basket_adj_close_lag1
- Log: `models/xgb/docs/logs/20261022T215713638169_xgb_9e5ec2da.json`

### `20261022T215724946302_xgb_c670bffb`

- Command: train
- Parent: `20261022T214026587580_xgb_332e97f1`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "bank_basket_return_lag1", "round": 1}`
- Feature hash: `52b8a977b3a9`
- Genome hash: `c670bffb9c96c3b3`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `bank_basket_return_lag1`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 440.624495
- Search RMSE: 460.366574
- Decision: rejected. not promising because search RMSE 460.366574 is not at least 1% below 459.892220 from 20261022T214026587580_xgb_332e97f1 after adding bank_basket_return_lag1
- Log: `models/xgb/docs/logs/20261022T215724946302_xgb_c670bffb.json`

### `20261022T215735585671_xgb_35cada51`

- Command: train
- Parent: `20261022T214026587580_xgb_332e97f1`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "property_basket_adj_close_lag1", "round": 1}`
- Feature hash: `3b1ea68b7e46`
- Genome hash: `35cada51a947e583`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `property_basket_adj_close_lag1`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 442.889334
- Search RMSE: 460.063731
- Decision: rejected. not promising because search RMSE 460.063731 is not at least 1% below 459.892220 from 20261022T214026587580_xgb_332e97f1 after adding property_basket_adj_close_lag1
- Log: `models/xgb/docs/logs/20261022T215735585671_xgb_35cada51.json`

### `20261022T215748577159_xgb_cc076192`

- Command: train
- Parent: `20261022T214026587580_xgb_332e97f1`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "property_basket_return_lag1", "round": 1}`
- Feature hash: `d99f88cbdcb7`
- Genome hash: `cc076192bb26ddbc`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `property_basket_return_lag1`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 441.171992
- Search RMSE: 460.227564
- Decision: rejected. not promising because search RMSE 460.227564 is not at least 1% below 459.892220 from 20261022T214026587580_xgb_332e97f1 after adding property_basket_return_lag1
- Log: `models/xgb/docs/logs/20261022T215748577159_xgb_cc076192.json`

### `20261022T215802547546_xgb_9a643bbb`

- Command: train
- Parent: `20261022T214026587580_xgb_332e97f1`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "log_dist_nearest_open_mrt_km", "round": 1}`
- Feature hash: `21c7244e6748`
- Genome hash: `9a643bbb7b23326c`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `log_dist_nearest_open_mrt_km`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 440.910590
- Search RMSE: 459.947723
- Decision: rejected. not promising because search RMSE 459.947723 is not at least 1% below 459.892220 from 20261022T214026587580_xgb_332e97f1 after adding log_dist_nearest_open_mrt_km
- Log: `models/xgb/docs/logs/20261022T215802547546_xgb_9a643bbb.json`

### `20261022T215816714910_xgb_80950e07`

- Command: train
- Parent: `20261022T214026587580_xgb_332e97f1`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "is_central_region", "round": 1}`
- Feature hash: `e47e0c0ed690`
- Genome hash: `80950e079a5271bc`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `is_central_region`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 442.412472
- Search RMSE: 459.892220
- Decision: rejected. not promising because search RMSE 459.892220 is not at least 1% below 459.892220 from 20261022T214026587580_xgb_332e97f1 after adding is_central_region
- Log: `models/xgb/docs/logs/20261022T215816714910_xgb_80950e07.json`

### `20261022T215829626818_xgb_408c37bd`

- Command: train
- Parent: `20261022T214026587580_xgb_332e97f1`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "building_age_x_central", "round": 1}`
- Feature hash: `a5a53d0ae4a0`
- Genome hash: `408c37bd89dbc5b1`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `building_age_x_central`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 443.570045
- Search RMSE: 460.410049
- Decision: rejected. not promising because search RMSE 460.410049 is not at least 1% below 459.892220 from 20261022T214026587580_xgb_332e97f1 after adding building_age_x_central
- Log: `models/xgb/docs/logs/20261022T215829626818_xgb_408c37bd.json`

### `20261022T215844314758_xgb_0cc5214f`

- Command: train
- Parent: `20261022T214026587580_xgb_332e97f1`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "mean_rent_postal_sector", "round": 1}`
- Feature hash: `e7393246f8d5`
- Genome hash: `0cc5214f1bd49d90`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `mean_rent_postal_sector`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 442.851342
- Search RMSE: 459.722315
- Decision: rejected. not promising because search RMSE 459.722315 is not at least 1% below 459.892220 from 20261022T214026587580_xgb_332e97f1 after adding mean_rent_postal_sector
- Log: `models/xgb/docs/logs/20261022T215844314758_xgb_0cc5214f.json`

### `20261022T215858353462_xgb_b518a6bb`

- Command: train
- Parent: `20261022T214026587580_xgb_332e97f1`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "mean_rent_postal_district", "round": 1}`
- Feature hash: `4d4cc5ffe4d6`
- Genome hash: `b518a6bbf9d23042`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `mean_rent_postal_district`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 442.585433
- Search RMSE: 459.886504
- Decision: rejected. not promising because search RMSE 459.886504 is not at least 1% below 459.892220 from 20261022T214026587580_xgb_332e97f1 after adding mean_rent_postal_district
- Log: `models/xgb/docs/logs/20261022T215858353462_xgb_b518a6bb.json`

### `20261022T215913255279_xgb_d7145bd1`

- Command: train
- Parent: `20261022T214026587580_xgb_332e97f1`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "mean_rent_nearest_mall", "round": 1}`
- Feature hash: `471d8ee447c6`
- Genome hash: `d7145bd11dd61fe3`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `mean_rent_nearest_mall`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 443.337655
- Search RMSE: 460.085271
- Decision: rejected. not promising because search RMSE 460.085271 is not at least 1% below 459.892220 from 20261022T214026587580_xgb_332e97f1 after adding mean_rent_nearest_mall
- Log: `models/xgb/docs/logs/20261022T215913255279_xgb_d7145bd1.json`

### `20261022T215927106055_xgb_e78abd5d`

- Command: train
- Parent: `20261022T214026587580_xgb_332e97f1`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "mean_rent_nearest_mrt", "round": 1}`
- Feature hash: `c9dd7135f0c6`
- Genome hash: `e78abd5d722705b7`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `mean_rent_nearest_mrt`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 442.278954
- Search RMSE: 460.453588
- Decision: rejected. not promising because search RMSE 460.453588 is not at least 1% below 459.892220 from 20261022T214026587580_xgb_332e97f1 after adding mean_rent_nearest_mrt
- Log: `models/xgb/docs/logs/20261022T215927106055_xgb_e78abd5d.json`

### `20261022T215940398328_xgb_e416c33b`

- Command: train
- Parent: `20261022T214026587580_xgb_332e97f1`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "mean_rent_nearest_primary", "round": 1}`
- Feature hash: `b23749c91b13`
- Genome hash: `e416c33be7d1a29f`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `mean_rent_nearest_primary`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 443.495318
- Search RMSE: 460.777659
- Decision: rejected. not promising because search RMSE 460.777659 is not at least 1% below 459.892220 from 20261022T214026587580_xgb_332e97f1 after adding mean_rent_nearest_primary
- Log: `models/xgb/docs/logs/20261022T215940398328_xgb_e416c33b.json`

### `20261022T215954254836_xgb_d409444c`

- Command: train
- Parent: `20261022T214026587580_xgb_332e97f1`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "mean_rent_nearest_secondary", "round": 1}`
- Feature hash: `8b3befbcf20d`
- Genome hash: `d409444c04f391cc`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `mean_rent_nearest_secondary`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 442.037817
- Search RMSE: 460.341738
- Decision: rejected. not promising because search RMSE 460.341738 is not at least 1% below 459.892220 from 20261022T214026587580_xgb_332e97f1 after adding mean_rent_nearest_secondary
- Log: `models/xgb/docs/logs/20261022T215954254836_xgb_d409444c.json`

### `20261030T103028155670_xgb_7ca41a9e`

- Command: train
- Parent: `20261022T214026587580_xgb_332e97f1`
- Generation: None
- Mutation: `{"axis": "finalize"}`
- Feature hash: `170e95b30b54`
- Genome hash: `7ca41a9ee3cdf8c9`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"early_stopping_rounds": 50, "learning_rate": 0.05, "max_depth": 8, "n_estimators": 2000}`
- Train RMSE: 424.294590
- Search RMSE: 462.145765
- Decision: accepted. final fit on the features of 20261022T214026587580_xgb_332e97f1; this stage does not select columns
- Log: `models/xgb/docs/logs/20261030T103028155670_xgb_7ca41a9e.json`

### `20261030T104900338798_xgb_fe7abcce`

- Command: submit
- Parent: `20261030T103028155670_xgb_7ca41a9e`
- Generation: None
- Mutation: `{"axis": "submit"}`
- Feature hash: `170e95b30b54`
- Genome hash: `fe7abcce9f39dc41`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"early_stopping_rounds": 0, "learning_rate": 0.05, "max_depth": 8, "n_estimators": 89}`
- Train RMSE: 425.246917
- Decision: refit of 20261030T103028155670_xgb_7ca41a9e on every labeled month; this stage does not select columns and was not scored on held-out MONTHLY_RENT
- Log: `models/xgb/docs/logs/20261030T104900338798_xgb_fe7abcce.json`

### `20261030T110148020531_xgb_643a1851`

- Command: train
- Parent: `20261022T214026587580_xgb_332e97f1`
- Generation: None
- Mutation: `{"axis": "use_all_features"}`
- Feature hash: `646e8a84f218`
- Genome hash: `643a1851182708d1`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `latitude`, `longitude`, `postal_code`, `max_floor`, `year_completed`, `subzone`, `planning_area`, `region`, `building_age_years`, `lease_vs_completion_gap`, `flat_age_from_lease`, `dist_nearest_open_mrt_km`, `nearest_open_mrt_code`, `nearest_open_mrt_name`, `nearest_open_mrt_line`, `dist_nearest_primary_km`, `nearest_primary_name`, `n_primary_within_1km`, `dist_nearest_mall_km`, `nearest_mall_name`, `nearest_ns_station_code`, `nearest_ns_station_name`, `dist_nearest_ns_km`, `nearest_ew_station_code`, `nearest_ew_station_name`, `dist_nearest_ew_km`, `nearest_ne_station_code`, `nearest_ne_station_name`, `dist_nearest_ne_km`, `nearest_cc_station_code`, `nearest_cc_station_name`, `dist_nearest_cc_km`, `nearest_dt_station_code`, `nearest_dt_station_name`, `dist_nearest_dt_km`, `nearest_te_station_code`, `nearest_te_station_name`, `dist_nearest_te_km`, `n_open_mrt_within_0_5km`, `n_open_mrt_within_1km`, `n_open_mrt_within_2km`, `dist_nearest_planned_mrt_km`, `n_planned_mrt_within_1km`, `n_planned_mrt_within_2km`, `nearest_open_mrt_is_interchange`, `n_primary_within_2km`, `dist_nearest_secondary_km`, `nearest_secondary_name`, `n_secondary_within_1km`, `dist_nearest_school_km`, `nearest_school_name`, `n_school_within_1km`, `n_mall_within_1km`, `n_mall_within_2km`, `coe_a_premium_lag1`, `coe_a_premium_mom_change`, `coe_a_bids_per_quota_lag1`, `coe_b_premium_lag1`, `coe_b_premium_mom_change`, `coe_b_bids_per_quota_lag1`, `coe_e_premium_lag1`, `coe_e_premium_mom_change`, `coe_e_bids_per_quota_lag1`, `coe_a_premium_roll3`, `coe_c_premium_lag1`, `dist_nearest_future_line_open_km`, `dist_nearest_future_line_planned_km`, `n_hdb_blocks_within_500m`, `dist_to_town_centroid_km`, `dist_to_cbd_km`, `bank_basket_adj_close_lag1`, `bank_basket_return_lag1`, `property_basket_adj_close_lag1`, `property_basket_return_lag1`, `log_dist_nearest_open_mrt_km`, `is_central_region`, `building_age_x_central`, `mean_rent_postal_sector`, `mean_rent_postal_district`, `mean_rent_nearest_mall`, `mean_rent_nearest_mrt`, `mean_rent_nearest_primary`, `mean_rent_nearest_secondary`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"early_stopping_rounds": 50, "learning_rate": 0.05, "max_depth": 6, "n_estimators": 2000}`
- Train RMSE: 418.373309
- Search RMSE: 467.691908
- Decision: accepted. all original and auxiliary columns fit together; this fit does not select columns one at a time and does not replace best/
- Log: `models/xgb/docs/logs/20261030T110148020531_xgb_643a1851.json`

### `20261030T124013757421_xgb_824fde31`

- Command: train
- Parent: `20261022T214026587580_xgb_332e97f1`
- Generation: None
- Mutation: `{"axis": "features", "change": "set_c", "columns": ["longitude", "year_completed", "subzone", "building_age_years", "flat_age_from_lease", "nearest_open_mrt_code", "nearest_open_mrt_name", "nearest_mall_name", "dist_nearest_dt_km", "n_open_mrt_within_1km", "n_planned_mrt_within_1km", "n_secondary_within_1km", "n_mall_within_1km", "coe_e_premium_lag1", "dist_nearest_future_line_open_km", "dist_nearest_future_line_planned_km", "dist_to_cbd_km", "bank_basket_adj_close_lag1", "mean_rent_postal_sector", "mean_rent_postal_district"], "round": 1}`
- Feature hash: `ede87d75909f`
- Genome hash: `824fde31757845d5`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `longitude`, `year_completed`, `subzone`, `building_age_years`, `flat_age_from_lease`, `nearest_open_mrt_code`, `nearest_open_mrt_name`, `nearest_mall_name`, `dist_nearest_dt_km`, `n_open_mrt_within_1km`, `n_planned_mrt_within_1km`, `n_secondary_within_1km`, `n_mall_within_1km`, `coe_e_premium_lag1`, `dist_nearest_future_line_open_km`, `dist_nearest_future_line_planned_km`, `dist_to_cbd_km`, `bank_basket_adj_close_lag1`, `mean_rent_postal_sector`, `mean_rent_postal_district`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}`
- Train RMSE: 441.235845
- Search RMSE: 459.506905
- Decision: accepted. replaced best 20261022T214026587580_xgb_332e97f1 search RMSE 459.892220 with 459.506905
- Log: `models/xgb/docs/logs/20261030T124013757421_xgb_824fde31.json`

### `20261030T124257708734_xgb_933157d5`

- Command: train
- Parent: `20261030T124013757421_xgb_824fde31`
- Generation: None
- Mutation: `{"axis": "finalize"}`
- Feature hash: `ede87d75909f`
- Genome hash: `933157d5623cc79d`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `longitude`, `year_completed`, `subzone`, `building_age_years`, `flat_age_from_lease`, `nearest_open_mrt_code`, `nearest_open_mrt_name`, `nearest_mall_name`, `dist_nearest_dt_km`, `n_open_mrt_within_1km`, `n_planned_mrt_within_1km`, `n_secondary_within_1km`, `n_mall_within_1km`, `coe_e_premium_lag1`, `dist_nearest_future_line_open_km`, `dist_nearest_future_line_planned_km`, `dist_to_cbd_km`, `bank_basket_adj_close_lag1`, `mean_rent_postal_sector`, `mean_rent_postal_district`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"early_stopping_rounds": 50, "learning_rate": 0.05, "max_depth": 8, "n_estimators": 2000}`
- Train RMSE: 422.274053
- Search RMSE: 461.840508
- Decision: accepted. final fit on the features of 20261030T124013757421_xgb_824fde31; this stage does not select columns
- Log: `models/xgb/docs/logs/20261030T124257708734_xgb_933157d5.json`

### `20261030T124328785808_xgb_28b1b752`

- Command: submit
- Parent: `20261030T124257708734_xgb_933157d5`
- Generation: None
- Mutation: `{"axis": "submit"}`
- Feature hash: `ede87d75909f`
- Genome hash: `28b1b752ef7f0af5`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `longitude`, `year_completed`, `subzone`, `building_age_years`, `flat_age_from_lease`, `nearest_open_mrt_code`, `nearest_open_mrt_name`, `nearest_mall_name`, `dist_nearest_dt_km`, `n_open_mrt_within_1km`, `n_planned_mrt_within_1km`, `n_secondary_within_1km`, `n_mall_within_1km`, `coe_e_premium_lag1`, `dist_nearest_future_line_open_km`, `dist_nearest_future_line_planned_km`, `dist_to_cbd_km`, `bank_basket_adj_close_lag1`, `mean_rent_postal_sector`, `mean_rent_postal_district`
- Preprocess: numeric_impute=median, scale=none, categorical=native, rare_min_count=30
- Architecture: `{"early_stopping_rounds": 0, "learning_rate": 0.05, "max_depth": 8, "n_estimators": 84}`
- Train RMSE: 423.593702
- Decision: refit of 20261030T124257708734_xgb_933157d5 on every labeled month; this stage does not select columns and was not scored on held-out MONTHLY_RENT
- Log: `models/xgb/docs/logs/20261030T124328785808_xgb_28b1b752.json`
<!-- lineage:end -->

## Predictions

<!-- predictions:start -->
### `20261030T103029132947_xgb_predict0`

- Champion run: `20261030T103028155670_xgb_7ca41a9e`
- Rows: 50000
- Mean: 3101.722121657715
- Std: 472.5534024394027
- Min: 1578.940185546875
- Max: 5890.5927734375
- Output: `outputs/predictions/xgb_final.csv`
- Refit on all labeled months: False
- Log: `models/xgb/docs/logs/20261030T103029132947_xgb_predict0.json`

### `20261030T104841401608_xgb_predict0`

- Champion run: `20261030T103028155670_xgb_7ca41a9e`
- Rows: 50000
- Mean: 3101.722121657715
- Std: 472.5534024394027
- Min: 1578.940185546875
- Max: 5890.5927734375
- Output: `outputs/predictions/xgb_final.csv`
- Refit on all labeled months: False
- Log: `models/xgb/docs/logs/20261030T104841401608_xgb_predict0.json`

### `20261030T104901075737_xgb_predict0`

- Champion run: `20261030T104900338798_xgb_fe7abcce`
- Rows: 50000
- Mean: 3125.5144426879883
- Std: 463.33541793868005
- Min: 1665.81298828125
- Max: 6402.4716796875
- Output: `outputs/predictions/xgb_refit.csv`
- Refit on all labeled months: True
- Log: `models/xgb/docs/logs/20261030T104901075737_xgb_predict0.json`

### `20261030T124258979014_xgb_predict0`

- Champion run: `20261030T124257708734_xgb_933157d5`
- Rows: 50000
- Mean: 3061.4706947265627
- Std: 463.7355126555241
- Min: 1495.2886962890625
- Max: 5870.63623046875
- Output: `outputs/predictions/xgb_final.csv`
- Refit on all labeled months: False
- Log: `models/xgb/docs/logs/20261030T124258979014_xgb_predict0.json`

### `20261030T124305735867_xgb_predict0`

- Champion run: `20261030T124257708734_xgb_933157d5`
- Rows: 50000
- Mean: 3061.4706947265627
- Std: 463.7355126555241
- Min: 1495.2886962890625
- Max: 5870.63623046875
- Output: `outputs/predictions/xgb_final.csv`
- Refit on all labeled months: False
- Log: `models/xgb/docs/logs/20261030T124305735867_xgb_predict0.json`

### `20261030T124329669944_xgb_predict0`

- Champion run: `20261030T124328785808_xgb_28b1b752`
- Rows: 50000
- Mean: 3110.6129596069336
- Std: 452.1457279028738
- Min: 1438.078125
- Max: 6318.328125
- Output: `outputs/predictions/xgb_refit.csv`
- Refit on all labeled months: True
- Log: `models/xgb/docs/logs/20261030T124329669944_xgb_predict0.json`
<!-- predictions:end -->

