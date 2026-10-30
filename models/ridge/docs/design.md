# Ridge

## Architecture gene

The spec is `alpha`. Regularize multiplies alpha by 10 up to 10000. Capacity divides alpha by 10 down to 0.001.

## Adjustment rules

The evolve command reads `docs/logs` and `docs/champion.json`, then proposes up to three mutants of the champion. Each mutant changes one gene: the feature subset, the preprocess spec, or the architecture. A genome already present in the logs is not trained again.

Feature changes alternate by generation. An odd generation adds the unused pool column with the largest absolute correlation to the champion residual on the search months. An even generation drops the column with the smallest permutation importance on a search sample of at most 2,000 rows. When that single-column move was already tried from the same parent, and every column of a feature family is inside the pool, an odd generation adds that family as a block and an even generation removes it. A feature set always keeps at least one column.

Preprocess changes the first knob not yet tried from this parent, in this order: numeric_impute (median, mean, zero); scale (standard, none, minmax); categorical (onehot, ordinal, target, bin); rare_min_count (10, 30, 100). The new value is the first choice in that list that differs from the current value.

Architecture regularizes when (search RMSE - train RMSE) / train RMSE is above 0.15. Otherwise it adds capacity, inside the cap in this model's note.

A mutant replaces the champion only when its search RMSE is at least 1% lower. The loop stops at the generation cap, when a generation accepts nothing, or when every proposed mutant is already in the logs. Confirm RMSE is scored once after search stops and is not an acceptance signal. Predict writes a summary and does not change the genome.

Labeled months are cut in time. Confirm is the last `val_months` approval months. Search is the `val_months` immediately before confirm. Fit is every earlier month. The preprocessor is fit on the fit months only.

## Current champion

<!-- champion:start -->
- Run: `20261022T185613865949_ridge_d97c6333`
- Log: `models/ridge/docs/logs/20261022T185613865949_ridge_d97c6333.json`
- Feature hash: `3dd5dfce0f43`
- Genome hash: `d97c6333368eb3dc`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_a_premium_lag1`, `coe_b_premium_lag1`, `coe_e_premium_lag1`, `coe_a_premium_roll3`, `coe_c_premium_lag1`, `bank_basket_adj_close_lag1`, `bank_basket_return_lag1`, `property_basket_adj_close_lag1`, `mean_rent_postal_sector`, `mean_rent_postal_district`, `mean_rent_nearest_mall`, `mean_rent_nearest_mrt`, `mean_rent_nearest_primary`, `mean_rent_nearest_secondary`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 475.741471
- Search RMSE: 462.061000
- Confirm RMSE: 491.271737
- Confirm log: `models/ridge/docs/logs/20261022T185614023988_ridge_d97c6333.json`
- Row cap: full training table
<!-- champion:end -->

## Lineage

<!-- lineage:start -->
Regenerated from `docs/logs`. Do not edit the metrics in this section by hand.

### `20261022T183022314474_ridge_fea0a337`

- Command: train
- Parent: `None`
- Generation: None
- Mutation: `{"axis": "train"}`
- Feature hash: `170e95b30b54`
- Genome hash: `fea0a3374b5a880f`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 489.293351
- Search RMSE: 525.267681
- Decision: accepted. first saved best; there was no earlier snapshot
- Log: `models/ridge/docs/logs/20261022T183022314474_ridge_fea0a337.json`

### `20261022T183213738486_ridge_de515bfb`

- Command: train
- Parent: `20261022T183022314474_ridge_fea0a337`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "latitude", "round": 1}`
- Feature hash: `99b4d5d7749f`
- Genome hash: `de515bfb7d490095`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `latitude`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 489.293184
- Search RMSE: 525.269306
- Decision: rejected. not promising because search RMSE 525.269306 is not at least 1% below 525.267681 from 20261022T183022314474_ridge_fea0a337 after adding latitude
- Log: `models/ridge/docs/logs/20261022T183213738486_ridge_de515bfb.json`

### `20261022T183230895922_ridge_ac143dbe`

- Command: train
- Parent: `20261022T183022314474_ridge_fea0a337`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "longitude", "round": 1}`
- Feature hash: `c27505a182db`
- Genome hash: `ac143dbecb8e4447`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `longitude`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 489.292820
- Search RMSE: 525.271536
- Decision: rejected. not promising because search RMSE 525.271536 is not at least 1% below 525.267681 from 20261022T183022314474_ridge_fea0a337 after adding longitude
- Log: `models/ridge/docs/logs/20261022T183230895922_ridge_ac143dbe.json`

### `20261022T183250934387_ridge_8f19690f`

- Command: train
- Parent: `20261022T183022314474_ridge_fea0a337`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "postal_code", "round": 1}`
- Feature hash: `e62a952e566d`
- Genome hash: `8f19690f2c4f799a`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `postal_code`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 487.050071
- Search RMSE: 524.404159
- Decision: rejected. not promising because search RMSE 524.404159 is not at least 1% below 525.267681 from 20261022T183022314474_ridge_fea0a337 after adding postal_code
- Log: `models/ridge/docs/logs/20261022T183250934387_ridge_8f19690f.json`

### `20261022T183307756165_ridge_5483ec83`

- Command: train
- Parent: `20261022T183022314474_ridge_fea0a337`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "max_floor", "round": 1}`
- Feature hash: `6d16b7bd0382`
- Genome hash: `5483ec836f3ec93b`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `max_floor`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 489.289378
- Search RMSE: 525.291737
- Decision: rejected. not promising because search RMSE 525.291737 is not at least 1% below 525.267681 from 20261022T183022314474_ridge_fea0a337 after adding max_floor
- Log: `models/ridge/docs/logs/20261022T183307756165_ridge_5483ec83.json`

### `20261022T183323454050_ridge_4b99c0b9`

- Command: train
- Parent: `20261022T183022314474_ridge_fea0a337`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "year_completed", "round": 1}`
- Feature hash: `b2d2d8a929b6`
- Genome hash: `4b99c0b9a8b79ab1`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `year_completed`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 489.288113
- Search RMSE: 525.267094
- Decision: rejected. not promising because search RMSE 525.267094 is not at least 1% below 525.267681 from 20261022T183022314474_ridge_fea0a337 after adding year_completed
- Log: `models/ridge/docs/logs/20261022T183323454050_ridge_4b99c0b9.json`

### `20261022T183340900159_ridge_24300ce5`

- Command: train
- Parent: `20261022T183022314474_ridge_fea0a337`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "subzone", "round": 1}`
- Feature hash: `19ad5218a78e`
- Genome hash: `24300ce567b712b0`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `subzone`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 489.001900
- Search RMSE: 524.885988
- Decision: rejected. not promising because search RMSE 524.885988 is not at least 1% below 525.267681 from 20261022T183022314474_ridge_fea0a337 after adding subzone
- Log: `models/ridge/docs/logs/20261022T183340900159_ridge_24300ce5.json`

### `20261022T183357996792_ridge_de437fb3`

- Command: train
- Parent: `20261022T183022314474_ridge_fea0a337`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "planning_area", "round": 1}`
- Feature hash: `212df8d837f8`
- Genome hash: `de437fb32bdd0d79`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `planning_area`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 489.281854
- Search RMSE: 525.301525
- Decision: rejected. not promising because search RMSE 525.301525 is not at least 1% below 525.267681 from 20261022T183022314474_ridge_fea0a337 after adding planning_area
- Log: `models/ridge/docs/logs/20261022T183357996792_ridge_de437fb3.json`

### `20261022T183418069180_ridge_0c1d9909`

- Command: train
- Parent: `20261022T183022314474_ridge_fea0a337`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "region", "round": 1}`
- Feature hash: `fd9247992a0a`
- Genome hash: `0c1d9909df253b13`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `region`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 489.292970
- Search RMSE: 525.258242
- Decision: rejected. not promising because search RMSE 525.258242 is not at least 1% below 525.267681 from 20261022T183022314474_ridge_fea0a337 after adding region
- Log: `models/ridge/docs/logs/20261022T183418069180_ridge_0c1d9909.json`

### `20261022T183436869349_ridge_0083faa8`

- Command: train
- Parent: `20261022T183022314474_ridge_fea0a337`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "building_age_years", "round": 1}`
- Feature hash: `c234d75ff5d3`
- Genome hash: `0083faa82b0f0dff`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `building_age_years`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 489.287966
- Search RMSE: 524.899768
- Decision: rejected. not promising because search RMSE 524.899768 is not at least 1% below 525.267681 from 20261022T183022314474_ridge_fea0a337 after adding building_age_years
- Log: `models/ridge/docs/logs/20261022T183436869349_ridge_0083faa8.json`

### `20261022T183455878739_ridge_b330fa1c`

- Command: train
- Parent: `20261022T183022314474_ridge_fea0a337`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "lease_vs_completion_gap", "round": 1}`
- Feature hash: `d33f593517bd`
- Genome hash: `b330fa1cd4f31df8`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `lease_vs_completion_gap`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 489.287910
- Search RMSE: 525.276075
- Decision: rejected. not promising because search RMSE 525.276075 is not at least 1% below 525.267681 from 20261022T183022314474_ridge_fea0a337 after adding lease_vs_completion_gap
- Log: `models/ridge/docs/logs/20261022T183455878739_ridge_b330fa1c.json`

### `20261022T183515172784_ridge_b3624879`

- Command: train
- Parent: `20261022T183022314474_ridge_fea0a337`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "flat_age_from_lease", "round": 1}`
- Feature hash: `0ae4cfca946e`
- Genome hash: `b3624879d38e9b93`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `flat_age_from_lease`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 489.293347
- Search RMSE: 525.304529
- Decision: rejected. not promising because search RMSE 525.304529 is not at least 1% below 525.267681 from 20261022T183022314474_ridge_fea0a337 after adding flat_age_from_lease
- Log: `models/ridge/docs/logs/20261022T183515172784_ridge_b3624879.json`

### `20261022T183535298370_ridge_a4e14e57`

- Command: train
- Parent: `20261022T183022314474_ridge_fea0a337`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "dist_nearest_open_mrt_km", "round": 1}`
- Feature hash: `0c2594d85422`
- Genome hash: `a4e14e57edde4b5f`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `dist_nearest_open_mrt_km`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 489.178019
- Search RMSE: 525.027567
- Decision: rejected. not promising because search RMSE 525.027567 is not at least 1% below 525.267681 from 20261022T183022314474_ridge_fea0a337 after adding dist_nearest_open_mrt_km
- Log: `models/ridge/docs/logs/20261022T183535298370_ridge_a4e14e57.json`

### `20261022T183554799137_ridge_dd05be32`

- Command: train
- Parent: `20261022T183022314474_ridge_fea0a337`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_open_mrt_code", "round": 1}`
- Feature hash: `42451fae9f20`
- Genome hash: `dd05be3213eb47ae`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `nearest_open_mrt_code`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 489.059242
- Search RMSE: 525.014468
- Decision: rejected. not promising because search RMSE 525.014468 is not at least 1% below 525.267681 from 20261022T183022314474_ridge_fea0a337 after adding nearest_open_mrt_code
- Log: `models/ridge/docs/logs/20261022T183554799137_ridge_dd05be32.json`

### `20261022T183614637606_ridge_f2b87e9b`

- Command: train
- Parent: `20261022T183022314474_ridge_fea0a337`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_open_mrt_name", "round": 1}`
- Feature hash: `57326fd029e4`
- Genome hash: `f2b87e9b3a609af9`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `nearest_open_mrt_name`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 489.059242
- Search RMSE: 525.014468
- Decision: rejected. not promising because search RMSE 525.014468 is not at least 1% below 525.267681 from 20261022T183022314474_ridge_fea0a337 after adding nearest_open_mrt_name
- Log: `models/ridge/docs/logs/20261022T183614637606_ridge_f2b87e9b.json`

### `20261022T183635279621_ridge_8d45c009`

- Command: train
- Parent: `20261022T183022314474_ridge_fea0a337`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_open_mrt_line", "round": 1}`
- Feature hash: `f6e2fc2120da`
- Genome hash: `8d45c00990c468bb`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `nearest_open_mrt_line`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 489.261623
- Search RMSE: 525.231012
- Decision: rejected. not promising because search RMSE 525.231012 is not at least 1% below 525.267681 from 20261022T183022314474_ridge_fea0a337 after adding nearest_open_mrt_line
- Log: `models/ridge/docs/logs/20261022T183635279621_ridge_8d45c009.json`

### `20261022T183655580249_ridge_9757a3e8`

- Command: train
- Parent: `20261022T183022314474_ridge_fea0a337`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "dist_nearest_primary_km", "round": 1}`
- Feature hash: `ec8fcfbbdc5f`
- Genome hash: `9757a3e8feee513c`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `dist_nearest_primary_km`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 489.292753
- Search RMSE: 525.265421
- Decision: rejected. not promising because search RMSE 525.265421 is not at least 1% below 525.267681 from 20261022T183022314474_ridge_fea0a337 after adding dist_nearest_primary_km
- Log: `models/ridge/docs/logs/20261022T183655580249_ridge_9757a3e8.json`

### `20261022T183715947822_ridge_1cb5693b`

- Command: train
- Parent: `20261022T183022314474_ridge_fea0a337`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_primary_name", "round": 1}`
- Feature hash: `3b600d364a2c`
- Genome hash: `1cb5693b7791bbcd`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `nearest_primary_name`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 488.885057
- Search RMSE: 524.999881
- Decision: rejected. not promising because search RMSE 524.999881 is not at least 1% below 525.267681 from 20261022T183022314474_ridge_fea0a337 after adding nearest_primary_name
- Log: `models/ridge/docs/logs/20261022T183715947822_ridge_1cb5693b.json`

### `20261022T183737395324_ridge_0c023ce7`

- Command: train
- Parent: `20261022T183022314474_ridge_fea0a337`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "n_primary_within_1km", "round": 1}`
- Feature hash: `4948f0d7026e`
- Genome hash: `0c023ce749417582`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `n_primary_within_1km`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 489.292677
- Search RMSE: 525.267008
- Decision: rejected. not promising because search RMSE 525.267008 is not at least 1% below 525.267681 from 20261022T183022314474_ridge_fea0a337 after adding n_primary_within_1km
- Log: `models/ridge/docs/logs/20261022T183737395324_ridge_0c023ce7.json`

### `20261022T183753515255_ridge_1545064f`

- Command: train
- Parent: `20261022T183022314474_ridge_fea0a337`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "dist_nearest_mall_km", "round": 1}`
- Feature hash: `eb02b9676679`
- Genome hash: `1545064fb14ed145`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `dist_nearest_mall_km`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 489.247876
- Search RMSE: 525.084321
- Decision: rejected. not promising because search RMSE 525.084321 is not at least 1% below 525.267681 from 20261022T183022314474_ridge_fea0a337 after adding dist_nearest_mall_km
- Log: `models/ridge/docs/logs/20261022T183753515255_ridge_1545064f.json`

### `20261022T183810911726_ridge_d1448f56`

- Command: train
- Parent: `20261022T183022314474_ridge_fea0a337`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_mall_name", "round": 1}`
- Feature hash: `dfa1a626ebab`
- Genome hash: `d1448f56bebcc6f9`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `nearest_mall_name`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 489.106264
- Search RMSE: 525.192517
- Decision: rejected. not promising because search RMSE 525.192517 is not at least 1% below 525.267681 from 20261022T183022314474_ridge_fea0a337 after adding nearest_mall_name
- Log: `models/ridge/docs/logs/20261022T183810911726_ridge_d1448f56.json`

### `20261022T183827280831_ridge_1d0573ba`

- Command: train
- Parent: `20261022T183022314474_ridge_fea0a337`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_ns_station_code", "round": 1}`
- Feature hash: `5ce89d055366`
- Genome hash: `1d0573ba17611102`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `nearest_ns_station_code`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 489.235599
- Search RMSE: 525.149117
- Decision: rejected. not promising because search RMSE 525.149117 is not at least 1% below 525.267681 from 20261022T183022314474_ridge_fea0a337 after adding nearest_ns_station_code
- Log: `models/ridge/docs/logs/20261022T183827280831_ridge_1d0573ba.json`

### `20261022T183848941551_ridge_6f697c0e`

- Command: train
- Parent: `20261022T183022314474_ridge_fea0a337`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_ns_station_name", "round": 1}`
- Feature hash: `70914731bd06`
- Genome hash: `6f697c0e3ceb4122`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `nearest_ns_station_name`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 489.235599
- Search RMSE: 525.149117
- Decision: rejected. not promising because search RMSE 525.149117 is not at least 1% below 525.267681 from 20261022T183022314474_ridge_fea0a337 after adding nearest_ns_station_name
- Log: `models/ridge/docs/logs/20261022T183848941551_ridge_6f697c0e.json`

### `20261022T183908216350_ridge_a0edc7e6`

- Command: train
- Parent: `20261022T183022314474_ridge_fea0a337`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "dist_nearest_ns_km", "round": 1}`
- Feature hash: `8d6e42f11adc`
- Genome hash: `a0edc7e6a3042de3`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `dist_nearest_ns_km`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 489.268412
- Search RMSE: 525.265155
- Decision: rejected. not promising because search RMSE 525.265155 is not at least 1% below 525.267681 from 20261022T183022314474_ridge_fea0a337 after adding dist_nearest_ns_km
- Log: `models/ridge/docs/logs/20261022T183908216350_ridge_a0edc7e6.json`

### `20261022T183923983782_ridge_38991f21`

- Command: train
- Parent: `20261022T183022314474_ridge_fea0a337`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_ew_station_code", "round": 1}`
- Feature hash: `033e0d7f6436`
- Genome hash: `38991f21d6b08fe1`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `nearest_ew_station_code`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 489.247915
- Search RMSE: 525.322298
- Decision: rejected. not promising because search RMSE 525.322298 is not at least 1% below 525.267681 from 20261022T183022314474_ridge_fea0a337 after adding nearest_ew_station_code
- Log: `models/ridge/docs/logs/20261022T183923983782_ridge_38991f21.json`

### `20261022T183939725845_ridge_d9ced85e`

- Command: train
- Parent: `20261022T183022314474_ridge_fea0a337`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_ew_station_name", "round": 1}`
- Feature hash: `cc0d61a0df40`
- Genome hash: `d9ced85e0c15575f`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `nearest_ew_station_name`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 489.247915
- Search RMSE: 525.322298
- Decision: rejected. not promising because search RMSE 525.322298 is not at least 1% below 525.267681 from 20261022T183022314474_ridge_fea0a337 after adding nearest_ew_station_name
- Log: `models/ridge/docs/logs/20261022T183939725845_ridge_d9ced85e.json`

### `20261022T183957442446_ridge_87406b1e`

- Command: train
- Parent: `20261022T183022314474_ridge_fea0a337`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "dist_nearest_ew_km", "round": 1}`
- Feature hash: `8db553894f91`
- Genome hash: `87406b1ed0e8fee4`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `dist_nearest_ew_km`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 489.293316
- Search RMSE: 525.266204
- Decision: rejected. not promising because search RMSE 525.266204 is not at least 1% below 525.267681 from 20261022T183022314474_ridge_fea0a337 after adding dist_nearest_ew_km
- Log: `models/ridge/docs/logs/20261022T183957442446_ridge_87406b1e.json`

### `20261022T184013497591_ridge_e35e7173`

- Command: train
- Parent: `20261022T183022314474_ridge_fea0a337`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_ne_station_code", "round": 1}`
- Feature hash: `491bf0c825f4`
- Genome hash: `e35e717379a957dc`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `nearest_ne_station_code`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 489.210382
- Search RMSE: 525.390804
- Decision: rejected. not promising because search RMSE 525.390804 is not at least 1% below 525.267681 from 20261022T183022314474_ridge_fea0a337 after adding nearest_ne_station_code
- Log: `models/ridge/docs/logs/20261022T184013497591_ridge_e35e7173.json`

### `20261022T184028964216_ridge_f8ee5869`

- Command: train
- Parent: `20261022T183022314474_ridge_fea0a337`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_ne_station_name", "round": 1}`
- Feature hash: `0b51c9b66206`
- Genome hash: `f8ee586935af7dd8`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `nearest_ne_station_name`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 489.210382
- Search RMSE: 525.390804
- Decision: rejected. not promising because search RMSE 525.390804 is not at least 1% below 525.267681 from 20261022T183022314474_ridge_fea0a337 after adding nearest_ne_station_name
- Log: `models/ridge/docs/logs/20261022T184028964216_ridge_f8ee5869.json`

### `20261022T184045039846_ridge_99f355a9`

- Command: train
- Parent: `20261022T183022314474_ridge_fea0a337`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "dist_nearest_ne_km", "round": 1}`
- Feature hash: `58386a513add`
- Genome hash: `99f355a904d077ca`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `dist_nearest_ne_km`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 489.286382
- Search RMSE: 525.329495
- Decision: rejected. not promising because search RMSE 525.329495 is not at least 1% below 525.267681 from 20261022T183022314474_ridge_fea0a337 after adding dist_nearest_ne_km
- Log: `models/ridge/docs/logs/20261022T184045039846_ridge_99f355a9.json`

### `20261022T184100605520_ridge_080795de`

- Command: train
- Parent: `20261022T183022314474_ridge_fea0a337`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_cc_station_code", "round": 1}`
- Feature hash: `e7df40e75f9a`
- Genome hash: `080795dea5bae839`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `nearest_cc_station_code`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 489.201482
- Search RMSE: 525.249763
- Decision: rejected. not promising because search RMSE 525.249763 is not at least 1% below 525.267681 from 20261022T183022314474_ridge_fea0a337 after adding nearest_cc_station_code
- Log: `models/ridge/docs/logs/20261022T184100605520_ridge_080795de.json`

### `20261022T184117141093_ridge_462fc9ea`

- Command: train
- Parent: `20261022T183022314474_ridge_fea0a337`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_cc_station_name", "round": 1}`
- Feature hash: `0e0891a68917`
- Genome hash: `462fc9ea81363d87`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `nearest_cc_station_name`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 489.201482
- Search RMSE: 525.249763
- Decision: rejected. not promising because search RMSE 525.249763 is not at least 1% below 525.267681 from 20261022T183022314474_ridge_fea0a337 after adding nearest_cc_station_name
- Log: `models/ridge/docs/logs/20261022T184117141093_ridge_462fc9ea.json`

### `20261022T184132671652_ridge_807fbd55`

- Command: train
- Parent: `20261022T183022314474_ridge_fea0a337`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "dist_nearest_cc_km", "round": 1}`
- Feature hash: `73966c4f212f`
- Genome hash: `807fbd55a5572e53`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `dist_nearest_cc_km`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 489.289615
- Search RMSE: 525.268876
- Decision: rejected. not promising because search RMSE 525.268876 is not at least 1% below 525.267681 from 20261022T183022314474_ridge_fea0a337 after adding dist_nearest_cc_km
- Log: `models/ridge/docs/logs/20261022T184132671652_ridge_807fbd55.json`

### `20261022T184148595046_ridge_dfd96cf7`

- Command: train
- Parent: `20261022T183022314474_ridge_fea0a337`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_dt_station_code", "round": 1}`
- Feature hash: `248dc8a22f33`
- Genome hash: `dfd96cf735bb8d04`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `nearest_dt_station_code`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 489.208980
- Search RMSE: 525.146933
- Decision: rejected. not promising because search RMSE 525.146933 is not at least 1% below 525.267681 from 20261022T183022314474_ridge_fea0a337 after adding nearest_dt_station_code
- Log: `models/ridge/docs/logs/20261022T184148595046_ridge_dfd96cf7.json`

### `20261022T184203924671_ridge_6a85e2c3`

- Command: train
- Parent: `20261022T183022314474_ridge_fea0a337`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_dt_station_name", "round": 1}`
- Feature hash: `48c44e2e40fe`
- Genome hash: `6a85e2c3ef18ad3a`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `nearest_dt_station_name`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 489.208980
- Search RMSE: 525.146933
- Decision: rejected. not promising because search RMSE 525.146933 is not at least 1% below 525.267681 from 20261022T183022314474_ridge_fea0a337 after adding nearest_dt_station_name
- Log: `models/ridge/docs/logs/20261022T184203924671_ridge_6a85e2c3.json`

### `20261022T184220060049_ridge_98539609`

- Command: train
- Parent: `20261022T183022314474_ridge_fea0a337`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "dist_nearest_dt_km", "round": 1}`
- Feature hash: `7e6bc08d4ff7`
- Genome hash: `98539609d9d98a7a`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `dist_nearest_dt_km`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 489.291389
- Search RMSE: 525.278362
- Decision: rejected. not promising because search RMSE 525.278362 is not at least 1% below 525.267681 from 20261022T183022314474_ridge_fea0a337 after adding dist_nearest_dt_km
- Log: `models/ridge/docs/logs/20261022T184220060049_ridge_98539609.json`

### `20261022T184235261722_ridge_57b02091`

- Command: train
- Parent: `20261022T183022314474_ridge_fea0a337`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_te_station_code", "round": 1}`
- Feature hash: `4f05b6ddefc5`
- Genome hash: `57b0209167561cae`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `nearest_te_station_code`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 489.241804
- Search RMSE: 525.326336
- Decision: rejected. not promising because search RMSE 525.326336 is not at least 1% below 525.267681 from 20261022T183022314474_ridge_fea0a337 after adding nearest_te_station_code
- Log: `models/ridge/docs/logs/20261022T184235261722_ridge_57b02091.json`

### `20261022T184250353867_ridge_4297c74e`

- Command: train
- Parent: `20261022T183022314474_ridge_fea0a337`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_te_station_name", "round": 1}`
- Feature hash: `1139130c8f31`
- Genome hash: `4297c74e3ead4fdd`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `nearest_te_station_name`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 489.241804
- Search RMSE: 525.326336
- Decision: rejected. not promising because search RMSE 525.326336 is not at least 1% below 525.267681 from 20261022T183022314474_ridge_fea0a337 after adding nearest_te_station_name
- Log: `models/ridge/docs/logs/20261022T184250353867_ridge_4297c74e.json`

### `20261022T184305782424_ridge_6cb8e615`

- Command: train
- Parent: `20261022T183022314474_ridge_fea0a337`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "dist_nearest_te_km", "round": 1}`
- Feature hash: `5cef183a61db`
- Genome hash: `6cb8e6158fef2155`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `dist_nearest_te_km`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 489.286222
- Search RMSE: 525.233512
- Decision: rejected. not promising because search RMSE 525.233512 is not at least 1% below 525.267681 from 20261022T183022314474_ridge_fea0a337 after adding dist_nearest_te_km
- Log: `models/ridge/docs/logs/20261022T184305782424_ridge_6cb8e615.json`

### `20261022T184322542089_ridge_56ad8472`

- Command: train
- Parent: `20261022T183022314474_ridge_fea0a337`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "n_open_mrt_within_0_5km", "round": 1}`
- Feature hash: `5219387275ac`
- Genome hash: `56ad84727cf81969`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `n_open_mrt_within_0_5km`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 489.269628
- Search RMSE: 525.157564
- Decision: rejected. not promising because search RMSE 525.157564 is not at least 1% below 525.267681 from 20261022T183022314474_ridge_fea0a337 after adding n_open_mrt_within_0_5km
- Log: `models/ridge/docs/logs/20261022T184322542089_ridge_56ad8472.json`

### `20261022T184339738873_ridge_dd392dea`

- Command: train
- Parent: `20261022T183022314474_ridge_fea0a337`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "n_open_mrt_within_1km", "round": 1}`
- Feature hash: `692f243ba340`
- Genome hash: `dd392dea3acd0113`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `n_open_mrt_within_1km`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 489.285983
- Search RMSE: 525.242196
- Decision: rejected. not promising because search RMSE 525.242196 is not at least 1% below 525.267681 from 20261022T183022314474_ridge_fea0a337 after adding n_open_mrt_within_1km
- Log: `models/ridge/docs/logs/20261022T184339738873_ridge_dd392dea.json`

### `20261022T184356532716_ridge_33e6048f`

- Command: train
- Parent: `20261022T183022314474_ridge_fea0a337`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "n_open_mrt_within_2km", "round": 1}`
- Feature hash: `2e76b96e95c7`
- Genome hash: `33e6048fcfa3f9e7`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `n_open_mrt_within_2km`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 489.286113
- Search RMSE: 525.335305
- Decision: rejected. not promising because search RMSE 525.335305 is not at least 1% below 525.267681 from 20261022T183022314474_ridge_fea0a337 after adding n_open_mrt_within_2km
- Log: `models/ridge/docs/logs/20261022T184356532716_ridge_33e6048f.json`

### `20261022T184413820155_ridge_bd54c088`

- Command: train
- Parent: `20261022T183022314474_ridge_fea0a337`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "dist_nearest_planned_mrt_km", "round": 1}`
- Feature hash: `d68a3dae1af5`
- Genome hash: `bd54c08878ab8d09`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `dist_nearest_planned_mrt_km`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 489.293095
- Search RMSE: 525.264842
- Decision: rejected. not promising because search RMSE 525.264842 is not at least 1% below 525.267681 from 20261022T183022314474_ridge_fea0a337 after adding dist_nearest_planned_mrt_km
- Log: `models/ridge/docs/logs/20261022T184413820155_ridge_bd54c088.json`

### `20261022T184430571839_ridge_709c0ab0`

- Command: train
- Parent: `20261022T183022314474_ridge_fea0a337`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "n_planned_mrt_within_1km", "round": 1}`
- Feature hash: `4fed35fb6804`
- Genome hash: `709c0ab009d300da`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `n_planned_mrt_within_1km`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 489.286406
- Search RMSE: 525.312147
- Decision: rejected. not promising because search RMSE 525.312147 is not at least 1% below 525.267681 from 20261022T183022314474_ridge_fea0a337 after adding n_planned_mrt_within_1km
- Log: `models/ridge/docs/logs/20261022T184430571839_ridge_709c0ab0.json`

### `20261022T184447755996_ridge_e6e63fbd`

- Command: train
- Parent: `20261022T183022314474_ridge_fea0a337`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "n_planned_mrt_within_2km", "round": 1}`
- Feature hash: `112eaabb4cdb`
- Genome hash: `e6e63fbd580719b7`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `n_planned_mrt_within_2km`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 489.293350
- Search RMSE: 525.267710
- Decision: rejected. not promising because search RMSE 525.267710 is not at least 1% below 525.267681 from 20261022T183022314474_ridge_fea0a337 after adding n_planned_mrt_within_2km
- Log: `models/ridge/docs/logs/20261022T184447755996_ridge_e6e63fbd.json`

### `20261022T184504340429_ridge_6ad5cf16`

- Command: train
- Parent: `20261022T183022314474_ridge_fea0a337`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_open_mrt_is_interchange", "round": 1}`
- Feature hash: `e8afb2136c5c`
- Genome hash: `6ad5cf16f0275ffe`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `nearest_open_mrt_is_interchange`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 489.288810
- Search RMSE: 525.273036
- Decision: rejected. not promising because search RMSE 525.273036 is not at least 1% below 525.267681 from 20261022T183022314474_ridge_fea0a337 after adding nearest_open_mrt_is_interchange
- Log: `models/ridge/docs/logs/20261022T184504340429_ridge_6ad5cf16.json`

### `20261022T184521094039_ridge_fa628e39`

- Command: train
- Parent: `20261022T183022314474_ridge_fea0a337`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "n_primary_within_2km", "round": 1}`
- Feature hash: `9f58d18094df`
- Genome hash: `fa628e39ebaaecc0`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `n_primary_within_2km`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 489.287457
- Search RMSE: 525.230959
- Decision: rejected. not promising because search RMSE 525.230959 is not at least 1% below 525.267681 from 20261022T183022314474_ridge_fea0a337 after adding n_primary_within_2km
- Log: `models/ridge/docs/logs/20261022T184521094039_ridge_fa628e39.json`

### `20261022T184538723452_ridge_ab2c773e`

- Command: train
- Parent: `20261022T183022314474_ridge_fea0a337`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "dist_nearest_secondary_km", "round": 1}`
- Feature hash: `3b84c255aaae`
- Genome hash: `ab2c773e496c0f10`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `dist_nearest_secondary_km`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 489.293344
- Search RMSE: 525.267086
- Decision: rejected. not promising because search RMSE 525.267086 is not at least 1% below 525.267681 from 20261022T183022314474_ridge_fea0a337 after adding dist_nearest_secondary_km
- Log: `models/ridge/docs/logs/20261022T184538723452_ridge_ab2c773e.json`

### `20261022T184556174160_ridge_6bb59560`

- Command: train
- Parent: `20261022T183022314474_ridge_fea0a337`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_secondary_name", "round": 1}`
- Feature hash: `4932501d7490`
- Genome hash: `6bb59560a9a4d472`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `nearest_secondary_name`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 488.939260
- Search RMSE: 525.114673
- Decision: rejected. not promising because search RMSE 525.114673 is not at least 1% below 525.267681 from 20261022T183022314474_ridge_fea0a337 after adding nearest_secondary_name
- Log: `models/ridge/docs/logs/20261022T184556174160_ridge_6bb59560.json`

### `20261022T184613942735_ridge_7eef92f7`

- Command: train
- Parent: `20261022T183022314474_ridge_fea0a337`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "n_secondary_within_1km", "round": 1}`
- Feature hash: `f92f056e77f5`
- Genome hash: `7eef92f73a355303`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `n_secondary_within_1km`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 489.293142
- Search RMSE: 525.262329
- Decision: rejected. not promising because search RMSE 525.262329 is not at least 1% below 525.267681 from 20261022T183022314474_ridge_fea0a337 after adding n_secondary_within_1km
- Log: `models/ridge/docs/logs/20261022T184613942735_ridge_7eef92f7.json`

### `20261022T184630901361_ridge_0bd3285f`

- Command: train
- Parent: `20261022T183022314474_ridge_fea0a337`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "dist_nearest_school_km", "round": 1}`
- Feature hash: `482ab5dda0e9`
- Genome hash: `0bd3285fcb35d793`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `dist_nearest_school_km`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 489.293358
- Search RMSE: 525.267568
- Decision: rejected. not promising because search RMSE 525.267568 is not at least 1% below 525.267681 from 20261022T183022314474_ridge_fea0a337 after adding dist_nearest_school_km
- Log: `models/ridge/docs/logs/20261022T184630901361_ridge_0bd3285f.json`

### `20261022T184649632996_ridge_a9e8f866`

- Command: train
- Parent: `20261022T183022314474_ridge_fea0a337`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_school_name", "round": 1}`
- Feature hash: `e657e5902602`
- Genome hash: `a9e8f866537dc649`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `nearest_school_name`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 488.598511
- Search RMSE: 525.409625
- Decision: rejected. not promising because search RMSE 525.409625 is not at least 1% below 525.267681 from 20261022T183022314474_ridge_fea0a337 after adding nearest_school_name
- Log: `models/ridge/docs/logs/20261022T184649632996_ridge_a9e8f866.json`

### `20261022T184706674505_ridge_e341ffb8`

- Command: train
- Parent: `20261022T183022314474_ridge_fea0a337`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "n_school_within_1km", "round": 1}`
- Feature hash: `b9907fe19ce9`
- Genome hash: `e341ffb8b3fbb00c`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `n_school_within_1km`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 489.293264
- Search RMSE: 525.272265
- Decision: rejected. not promising because search RMSE 525.272265 is not at least 1% below 525.267681 from 20261022T183022314474_ridge_fea0a337 after adding n_school_within_1km
- Log: `models/ridge/docs/logs/20261022T184706674505_ridge_e341ffb8.json`

### `20261022T184723832342_ridge_35ee6a26`

- Command: train
- Parent: `20261022T183022314474_ridge_fea0a337`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "n_mall_within_1km", "round": 1}`
- Feature hash: `a98c9f4224a5`
- Genome hash: `35ee6a2633ef7b64`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `n_mall_within_1km`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 489.280260
- Search RMSE: 525.198465
- Decision: rejected. not promising because search RMSE 525.198465 is not at least 1% below 525.267681 from 20261022T183022314474_ridge_fea0a337 after adding n_mall_within_1km
- Log: `models/ridge/docs/logs/20261022T184723832342_ridge_35ee6a26.json`

### `20261022T184741733401_ridge_ea94998a`

- Command: train
- Parent: `20261022T183022314474_ridge_fea0a337`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "n_mall_within_2km", "round": 1}`
- Feature hash: `282ff6f12a0d`
- Genome hash: `ea94998a24a1c4b8`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `n_mall_within_2km`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 489.291627
- Search RMSE: 525.307522
- Decision: rejected. not promising because search RMSE 525.307522 is not at least 1% below 525.267681 from 20261022T183022314474_ridge_fea0a337 after adding n_mall_within_2km
- Log: `models/ridge/docs/logs/20261022T184741733401_ridge_ea94998a.json`

### `20261022T184758880529_ridge_27f579dc`

- Command: train
- Parent: `20261022T183022314474_ridge_fea0a337`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "coe_a_premium_lag1", "round": 1}`
- Feature hash: `ca6251adee59`
- Genome hash: `27f579dc702001a4`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_a_premium_lag1`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 482.218418
- Search RMSE: 496.107353
- Decision: accepted. promising because search RMSE 496.107353 is at least 1% below 525.267681 from 20261022T183022314474_ridge_fea0a337 after adding coe_a_premium_lag1
- Log: `models/ridge/docs/logs/20261022T184758880529_ridge_27f579dc.json`

### `20261022T184816640019_ridge_bace7099`

- Command: train
- Parent: `20261022T183022314474_ridge_fea0a337`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "coe_a_premium_mom_change", "round": 1}`
- Feature hash: `ecd53635e5ea`
- Genome hash: `bace709965c4e653`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_a_premium_mom_change`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 489.272752
- Search RMSE: 526.335509
- Decision: rejected. not promising because search RMSE 526.335509 is not at least 1% below 525.267681 from 20261022T183022314474_ridge_fea0a337 after adding coe_a_premium_mom_change
- Log: `models/ridge/docs/logs/20261022T184816640019_ridge_bace7099.json`

### `20261022T184832897460_ridge_6f7c4995`

- Command: train
- Parent: `20261022T183022314474_ridge_fea0a337`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "coe_a_bids_per_quota_lag1", "round": 1}`
- Feature hash: `88b869c77fc2`
- Genome hash: `6f7c49957e08c53c`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_a_bids_per_quota_lag1`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 488.476227
- Search RMSE: 539.449547
- Decision: rejected. not promising because search RMSE 539.449547 is not at least 1% below 525.267681 from 20261022T183022314474_ridge_fea0a337 after adding coe_a_bids_per_quota_lag1
- Log: `models/ridge/docs/logs/20261022T184832897460_ridge_6f7c4995.json`

### `20261022T184849640147_ridge_aa0079f7`

- Command: train
- Parent: `20261022T183022314474_ridge_fea0a337`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "coe_b_premium_lag1", "round": 1}`
- Feature hash: `aef8257585e0`
- Genome hash: `aa0079f7a729ca42`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_b_premium_lag1`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 483.821852
- Search RMSE: 486.827203
- Decision: accepted. promising because search RMSE 486.827203 is at least 1% below 525.267681 from 20261022T183022314474_ridge_fea0a337 after adding coe_b_premium_lag1
- Log: `models/ridge/docs/logs/20261022T184849640147_ridge_aa0079f7.json`

### `20261022T184906143532_ridge_e15986b5`

- Command: train
- Parent: `20261022T183022314474_ridge_fea0a337`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "coe_b_premium_mom_change", "round": 1}`
- Feature hash: `8302b3652167`
- Genome hash: `e15986b5cb812343`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_b_premium_mom_change`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 489.293010
- Search RMSE: 525.285829
- Decision: rejected. not promising because search RMSE 525.285829 is not at least 1% below 525.267681 from 20261022T183022314474_ridge_fea0a337 after adding coe_b_premium_mom_change
- Log: `models/ridge/docs/logs/20261022T184906143532_ridge_e15986b5.json`

### `20261022T184922220252_ridge_e17bb87b`

- Command: train
- Parent: `20261022T183022314474_ridge_fea0a337`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "coe_b_bids_per_quota_lag1", "round": 1}`
- Feature hash: `4a7482aab3c7`
- Genome hash: `e17bb87b512a9f51`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_b_bids_per_quota_lag1`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 489.075801
- Search RMSE: 533.988240
- Decision: rejected. not promising because search RMSE 533.988240 is not at least 1% below 525.267681 from 20261022T183022314474_ridge_fea0a337 after adding coe_b_bids_per_quota_lag1
- Log: `models/ridge/docs/logs/20261022T184922220252_ridge_e17bb87b.json`

### `20261022T184939169220_ridge_3e251c6f`

- Command: train
- Parent: `20261022T183022314474_ridge_fea0a337`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "coe_e_premium_lag1", "round": 1}`
- Feature hash: `de6005d5c05c`
- Genome hash: `3e251c6fd48ffa49`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_e_premium_lag1`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 483.899956
- Search RMSE: 489.679948
- Decision: accepted. promising because search RMSE 489.679948 is at least 1% below 525.267681 from 20261022T183022314474_ridge_fea0a337 after adding coe_e_premium_lag1
- Log: `models/ridge/docs/logs/20261022T184939169220_ridge_3e251c6f.json`

### `20261022T184955158716_ridge_52ed3aa0`

- Command: train
- Parent: `20261022T183022314474_ridge_fea0a337`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "coe_e_premium_mom_change", "round": 1}`
- Feature hash: `387e56d7e868`
- Genome hash: `52ed3aa0ff2401bb`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_e_premium_mom_change`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 489.293017
- Search RMSE: 525.310213
- Decision: rejected. not promising because search RMSE 525.310213 is not at least 1% below 525.267681 from 20261022T183022314474_ridge_fea0a337 after adding coe_e_premium_mom_change
- Log: `models/ridge/docs/logs/20261022T184955158716_ridge_52ed3aa0.json`

### `20261022T185012293923_ridge_1926b1bf`

- Command: train
- Parent: `20261022T183022314474_ridge_fea0a337`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "coe_e_bids_per_quota_lag1", "round": 1}`
- Feature hash: `8bb848ebe345`
- Genome hash: `1926b1bfc90ff9d4`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_e_bids_per_quota_lag1`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 488.150899
- Search RMSE: 561.735153
- Decision: rejected. not promising because search RMSE 561.735153 is not at least 1% below 525.267681 from 20261022T183022314474_ridge_fea0a337 after adding coe_e_bids_per_quota_lag1
- Log: `models/ridge/docs/logs/20261022T185012293923_ridge_1926b1bf.json`

### `20261022T185028838641_ridge_1561f947`

- Command: train
- Parent: `20261022T183022314474_ridge_fea0a337`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "coe_a_premium_roll3", "round": 1}`
- Feature hash: `b09c5f9ba37b`
- Genome hash: `1561f9473c54c8fe`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_a_premium_roll3`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 483.233835
- Search RMSE: 510.500630
- Decision: accepted. promising because search RMSE 510.500630 is at least 1% below 525.267681 from 20261022T183022314474_ridge_fea0a337 after adding coe_a_premium_roll3
- Log: `models/ridge/docs/logs/20261022T185028838641_ridge_1561f947.json`

### `20261022T185045851682_ridge_d5203742`

- Command: train
- Parent: `20261022T183022314474_ridge_fea0a337`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "coe_c_premium_lag1", "round": 1}`
- Feature hash: `7b8f900c807f`
- Genome hash: `d5203742b1c3ba17`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_c_premium_lag1`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 478.581251
- Search RMSE: 473.731683
- Decision: accepted. promising because search RMSE 473.731683 is at least 1% below 525.267681 from 20261022T183022314474_ridge_fea0a337 after adding coe_c_premium_lag1
- Log: `models/ridge/docs/logs/20261022T185045851682_ridge_d5203742.json`

### `20261022T185102474561_ridge_350799cd`

- Command: train
- Parent: `20261022T183022314474_ridge_fea0a337`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "dist_nearest_future_line_open_km", "round": 1}`
- Feature hash: `109871ee8d0c`
- Genome hash: `350799cd56fffafe`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `dist_nearest_future_line_open_km`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 489.288477
- Search RMSE: 525.271465
- Decision: rejected. not promising because search RMSE 525.271465 is not at least 1% below 525.267681 from 20261022T183022314474_ridge_fea0a337 after adding dist_nearest_future_line_open_km
- Log: `models/ridge/docs/logs/20261022T185102474561_ridge_350799cd.json`

### `20261022T185120062018_ridge_dc3c0067`

- Command: train
- Parent: `20261022T183022314474_ridge_fea0a337`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "dist_nearest_future_line_planned_km", "round": 1}`
- Feature hash: `b96b1461ee87`
- Genome hash: `dc3c006724f2fb40`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `dist_nearest_future_line_planned_km`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 489.293301
- Search RMSE: 525.266871
- Decision: rejected. not promising because search RMSE 525.266871 is not at least 1% below 525.267681 from 20261022T183022314474_ridge_fea0a337 after adding dist_nearest_future_line_planned_km
- Log: `models/ridge/docs/logs/20261022T185120062018_ridge_dc3c0067.json`

### `20261022T185138054505_ridge_3c4c64ca`

- Command: train
- Parent: `20261022T183022314474_ridge_fea0a337`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "n_hdb_blocks_within_500m", "round": 1}`
- Feature hash: `95bfdabe2d04`
- Genome hash: `3c4c64ca07829eff`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `n_hdb_blocks_within_500m`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 489.292242
- Search RMSE: 525.288984
- Decision: rejected. not promising because search RMSE 525.288984 is not at least 1% below 525.267681 from 20261022T183022314474_ridge_fea0a337 after adding n_hdb_blocks_within_500m
- Log: `models/ridge/docs/logs/20261022T185138054505_ridge_3c4c64ca.json`

### `20261022T185154799187_ridge_a545299f`

- Command: train
- Parent: `20261022T183022314474_ridge_fea0a337`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "dist_to_town_centroid_km", "round": 1}`
- Feature hash: `cbb5a64dccd3`
- Genome hash: `a545299f86c5e2f6`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `dist_to_town_centroid_km`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 489.252009
- Search RMSE: 525.146468
- Decision: rejected. not promising because search RMSE 525.146468 is not at least 1% below 525.267681 from 20261022T183022314474_ridge_fea0a337 after adding dist_to_town_centroid_km
- Log: `models/ridge/docs/logs/20261022T185154799187_ridge_a545299f.json`

### `20261022T185212382479_ridge_3c767cc8`

- Command: train
- Parent: `20261022T183022314474_ridge_fea0a337`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "dist_to_cbd_km", "round": 1}`
- Feature hash: `3a81f2034fb9`
- Genome hash: `3c767cc880c1a5bc`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `dist_to_cbd_km`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 489.289466
- Search RMSE: 525.293544
- Decision: rejected. not promising because search RMSE 525.293544 is not at least 1% below 525.267681 from 20261022T183022314474_ridge_fea0a337 after adding dist_to_cbd_km
- Log: `models/ridge/docs/logs/20261022T185212382479_ridge_3c767cc8.json`

### `20261022T185229194394_ridge_5046f396`

- Command: train
- Parent: `20261022T183022314474_ridge_fea0a337`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "bank_basket_adj_close_lag1", "round": 1}`
- Feature hash: `40ef8a2b790f`
- Genome hash: `5046f396b57d3563`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `bank_basket_adj_close_lag1`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 487.713279
- Search RMSE: 475.429379
- Decision: accepted. promising because search RMSE 475.429379 is at least 1% below 525.267681 from 20261022T183022314474_ridge_fea0a337 after adding bank_basket_adj_close_lag1
- Log: `models/ridge/docs/logs/20261022T185229194394_ridge_5046f396.json`

### `20261022T185246603091_ridge_19b0f199`

- Command: train
- Parent: `20261022T183022314474_ridge_fea0a337`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "bank_basket_return_lag1", "round": 1}`
- Feature hash: `52b8a977b3a9`
- Genome hash: `19b0f199767722f9`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `bank_basket_return_lag1`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 489.087064
- Search RMSE: 517.979050
- Decision: accepted. promising because search RMSE 517.979050 is at least 1% below 525.267681 from 20261022T183022314474_ridge_fea0a337 after adding bank_basket_return_lag1
- Log: `models/ridge/docs/logs/20261022T185246603091_ridge_19b0f199.json`

### `20261022T185302979978_ridge_5ab2a2a8`

- Command: train
- Parent: `20261022T183022314474_ridge_fea0a337`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "property_basket_adj_close_lag1", "round": 1}`
- Feature hash: `3b1ea68b7e46`
- Genome hash: `5ab2a2a807bd1950`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `property_basket_adj_close_lag1`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 483.399674
- Search RMSE: 490.355373
- Decision: accepted. promising because search RMSE 490.355373 is at least 1% below 525.267681 from 20261022T183022314474_ridge_fea0a337 after adding property_basket_adj_close_lag1
- Log: `models/ridge/docs/logs/20261022T185302979978_ridge_5ab2a2a8.json`

### `20261022T185319995237_ridge_da6a2e81`

- Command: train
- Parent: `20261022T183022314474_ridge_fea0a337`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "property_basket_return_lag1", "round": 1}`
- Feature hash: `d99f88cbdcb7`
- Genome hash: `da6a2e810b15f38d`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `property_basket_return_lag1`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 489.229373
- Search RMSE: 524.462523
- Decision: rejected. not promising because search RMSE 524.462523 is not at least 1% below 525.267681 from 20261022T183022314474_ridge_fea0a337 after adding property_basket_return_lag1
- Log: `models/ridge/docs/logs/20261022T185319995237_ridge_da6a2e81.json`

### `20261022T185337248222_ridge_a208d2b4`

- Command: train
- Parent: `20261022T183022314474_ridge_fea0a337`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "log_dist_nearest_open_mrt_km", "round": 1}`
- Feature hash: `21c7244e6748`
- Genome hash: `a208d2b41310ffcd`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `log_dist_nearest_open_mrt_km`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 489.176514
- Search RMSE: 524.944206
- Decision: rejected. not promising because search RMSE 524.944206 is not at least 1% below 525.267681 from 20261022T183022314474_ridge_fea0a337 after adding log_dist_nearest_open_mrt_km
- Log: `models/ridge/docs/logs/20261022T185337248222_ridge_a208d2b4.json`

### `20261022T185354154811_ridge_ae34edee`

- Command: train
- Parent: `20261022T183022314474_ridge_fea0a337`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "is_central_region", "round": 1}`
- Feature hash: `e47e0c0ed690`
- Genome hash: `ae34edeecdd2e318`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `is_central_region`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 489.293018
- Search RMSE: 525.261079
- Decision: rejected. not promising because search RMSE 525.261079 is not at least 1% below 525.267681 from 20261022T183022314474_ridge_fea0a337 after adding is_central_region
- Log: `models/ridge/docs/logs/20261022T185354154811_ridge_ae34edee.json`

### `20261022T185412148565_ridge_11753da6`

- Command: train
- Parent: `20261022T183022314474_ridge_fea0a337`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "building_age_x_central", "round": 1}`
- Feature hash: `a5a53d0ae4a0`
- Genome hash: `11753da68f2d55ae`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `building_age_x_central`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 489.272883
- Search RMSE: 525.251078
- Decision: rejected. not promising because search RMSE 525.251078 is not at least 1% below 525.267681 from 20261022T183022314474_ridge_fea0a337 after adding building_age_x_central
- Log: `models/ridge/docs/logs/20261022T185412148565_ridge_11753da6.json`

### `20261022T185429851540_ridge_542a0e74`

- Command: train
- Parent: `20261022T183022314474_ridge_fea0a337`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "mean_rent_postal_sector", "round": 1}`
- Feature hash: `e7393246f8d5`
- Genome hash: `542a0e74d447b0cb`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `mean_rent_postal_sector`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 482.110801
- Search RMSE: 481.479535
- Decision: accepted. promising because search RMSE 481.479535 is at least 1% below 525.267681 from 20261022T183022314474_ridge_fea0a337 after adding mean_rent_postal_sector
- Log: `models/ridge/docs/logs/20261022T185429851540_ridge_542a0e74.json`

### `20261022T185447169955_ridge_f2679e93`

- Command: train
- Parent: `20261022T183022314474_ridge_fea0a337`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "mean_rent_postal_district", "round": 1}`
- Feature hash: `4d4cc5ffe4d6`
- Genome hash: `f2679e933a97dc06`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `mean_rent_postal_district`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 482.359348
- Search RMSE: 481.541818
- Decision: accepted. promising because search RMSE 481.541818 is at least 1% below 525.267681 from 20261022T183022314474_ridge_fea0a337 after adding mean_rent_postal_district
- Log: `models/ridge/docs/logs/20261022T185447169955_ridge_f2679e93.json`

### `20261022T185504106256_ridge_b0cf7462`

- Command: train
- Parent: `20261022T183022314474_ridge_fea0a337`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "mean_rent_nearest_mall", "round": 1}`
- Feature hash: `471d8ee447c6`
- Genome hash: `b0cf7462b23ef47b`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `mean_rent_nearest_mall`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 482.313690
- Search RMSE: 483.424112
- Decision: accepted. promising because search RMSE 483.424112 is at least 1% below 525.267681 from 20261022T183022314474_ridge_fea0a337 after adding mean_rent_nearest_mall
- Log: `models/ridge/docs/logs/20261022T185504106256_ridge_b0cf7462.json`

### `20261022T185521261673_ridge_fa58015e`

- Command: train
- Parent: `20261022T183022314474_ridge_fea0a337`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "mean_rent_nearest_mrt", "round": 1}`
- Feature hash: `c9dd7135f0c6`
- Genome hash: `fa58015ec1cdd657`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `mean_rent_nearest_mrt`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 482.386324
- Search RMSE: 483.657029
- Decision: accepted. promising because search RMSE 483.657029 is at least 1% below 525.267681 from 20261022T183022314474_ridge_fea0a337 after adding mean_rent_nearest_mrt
- Log: `models/ridge/docs/logs/20261022T185521261673_ridge_fa58015e.json`

### `20261022T185538939560_ridge_ee9ceefe`

- Command: train
- Parent: `20261022T183022314474_ridge_fea0a337`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "mean_rent_nearest_primary", "round": 1}`
- Feature hash: `b23749c91b13`
- Genome hash: `ee9ceefef71a138b`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `mean_rent_nearest_primary`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 482.540111
- Search RMSE: 484.199729
- Decision: accepted. promising because search RMSE 484.199729 is at least 1% below 525.267681 from 20261022T183022314474_ridge_fea0a337 after adding mean_rent_nearest_primary
- Log: `models/ridge/docs/logs/20261022T185538939560_ridge_ee9ceefe.json`

### `20261022T185555720506_ridge_8752d79d`

- Command: train
- Parent: `20261022T183022314474_ridge_fea0a337`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "mean_rent_nearest_secondary", "round": 1}`
- Feature hash: `8b3befbcf20d`
- Genome hash: `8752d79dc95e0f7f`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `mean_rent_nearest_secondary`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 482.639563
- Search RMSE: 485.210029
- Decision: accepted. promising because search RMSE 485.210029 is at least 1% below 525.267681 from 20261022T183022314474_ridge_fea0a337 after adding mean_rent_nearest_secondary
- Log: `models/ridge/docs/logs/20261022T185555720506_ridge_8752d79d.json`

### `20261022T185613865949_ridge_d97c6333`

- Command: train
- Parent: `20261022T183022314474_ridge_fea0a337`
- Generation: None
- Mutation: `{"axis": "features", "change": "set_c", "columns": ["coe_a_premium_lag1", "coe_b_premium_lag1", "coe_e_premium_lag1", "coe_a_premium_roll3", "coe_c_premium_lag1", "bank_basket_adj_close_lag1", "bank_basket_return_lag1", "property_basket_adj_close_lag1", "mean_rent_postal_sector", "mean_rent_postal_district", "mean_rent_nearest_mall", "mean_rent_nearest_mrt", "mean_rent_nearest_primary", "mean_rent_nearest_secondary"], "round": 1}`
- Feature hash: `3dd5dfce0f43`
- Genome hash: `d97c6333368eb3dc`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_a_premium_lag1`, `coe_b_premium_lag1`, `coe_e_premium_lag1`, `coe_a_premium_roll3`, `coe_c_premium_lag1`, `bank_basket_adj_close_lag1`, `bank_basket_return_lag1`, `property_basket_adj_close_lag1`, `mean_rent_postal_sector`, `mean_rent_postal_district`, `mean_rent_nearest_mall`, `mean_rent_nearest_mrt`, `mean_rent_nearest_primary`, `mean_rent_nearest_secondary`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 475.741471
- Search RMSE: 462.061000
- Decision: accepted. replaced best 20261022T183022314474_ridge_fea0a337 search RMSE 525.267681 with 462.061000
- Log: `models/ridge/docs/logs/20261022T185613865949_ridge_d97c6333.json`

### `20261022T185631772834_ridge_af5d3e69`

- Command: train
- Parent: `20261022T185613865949_ridge_d97c6333`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "latitude", "round": 2}`
- Feature hash: `e7712bf6acc0`
- Genome hash: `af5d3e69d0683997`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_a_premium_lag1`, `coe_b_premium_lag1`, `coe_e_premium_lag1`, `coe_a_premium_roll3`, `coe_c_premium_lag1`, `bank_basket_adj_close_lag1`, `bank_basket_return_lag1`, `property_basket_adj_close_lag1`, `mean_rent_postal_sector`, `mean_rent_postal_district`, `mean_rent_nearest_mall`, `mean_rent_nearest_mrt`, `mean_rent_nearest_primary`, `mean_rent_nearest_secondary`, `latitude`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 475.741635
- Search RMSE: 462.061173
- Decision: rejected. not promising because search RMSE 462.061173 is not at least 1% below 462.061000 from 20261022T185613865949_ridge_d97c6333 after adding latitude
- Log: `models/ridge/docs/logs/20261022T185631772834_ridge_af5d3e69.json`

### `20261022T185649511390_ridge_4e13818a`

- Command: train
- Parent: `20261022T185613865949_ridge_d97c6333`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "longitude", "round": 2}`
- Feature hash: `1787f3da9f4e`
- Genome hash: `4e13818aec038b00`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_a_premium_lag1`, `coe_b_premium_lag1`, `coe_e_premium_lag1`, `coe_a_premium_roll3`, `coe_c_premium_lag1`, `bank_basket_adj_close_lag1`, `bank_basket_return_lag1`, `property_basket_adj_close_lag1`, `mean_rent_postal_sector`, `mean_rent_postal_district`, `mean_rent_nearest_mall`, `mean_rent_nearest_mrt`, `mean_rent_nearest_primary`, `mean_rent_nearest_secondary`, `longitude`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 475.740461
- Search RMSE: 462.065570
- Decision: rejected. not promising because search RMSE 462.065570 is not at least 1% below 462.061000 from 20261022T185613865949_ridge_d97c6333 after adding longitude
- Log: `models/ridge/docs/logs/20261022T185649511390_ridge_4e13818a.json`

### `20261022T185710084035_ridge_60e4c9d1`

- Command: train
- Parent: `20261022T185613865949_ridge_d97c6333`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "postal_code", "round": 2}`
- Feature hash: `2f2a634c3925`
- Genome hash: `60e4c9d1ac888256`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_a_premium_lag1`, `coe_b_premium_lag1`, `coe_e_premium_lag1`, `coe_a_premium_roll3`, `coe_c_premium_lag1`, `bank_basket_adj_close_lag1`, `bank_basket_return_lag1`, `property_basket_adj_close_lag1`, `mean_rent_postal_sector`, `mean_rent_postal_district`, `mean_rent_nearest_mall`, `mean_rent_nearest_mrt`, `mean_rent_nearest_primary`, `mean_rent_nearest_secondary`, `postal_code`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 473.446145
- Search RMSE: 461.072390
- Decision: rejected. not promising because search RMSE 461.072390 is not at least 1% below 462.061000 from 20261022T185613865949_ridge_d97c6333 after adding postal_code
- Log: `models/ridge/docs/logs/20261022T185710084035_ridge_60e4c9d1.json`

### `20261022T185727341211_ridge_c927954e`

- Command: train
- Parent: `20261022T185613865949_ridge_d97c6333`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "max_floor", "round": 2}`
- Feature hash: `187d216ec8fb`
- Genome hash: `c927954eb965f2e7`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_a_premium_lag1`, `coe_b_premium_lag1`, `coe_e_premium_lag1`, `coe_a_premium_roll3`, `coe_c_premium_lag1`, `bank_basket_adj_close_lag1`, `bank_basket_return_lag1`, `property_basket_adj_close_lag1`, `mean_rent_postal_sector`, `mean_rent_postal_district`, `mean_rent_nearest_mall`, `mean_rent_nearest_mrt`, `mean_rent_nearest_primary`, `mean_rent_nearest_secondary`, `max_floor`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 475.736804
- Search RMSE: 462.117194
- Decision: rejected. not promising because search RMSE 462.117194 is not at least 1% below 462.061000 from 20261022T185613865949_ridge_d97c6333 after adding max_floor
- Log: `models/ridge/docs/logs/20261022T185727341211_ridge_c927954e.json`

### `20261022T185745213323_ridge_0187d580`

- Command: train
- Parent: `20261022T185613865949_ridge_d97c6333`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "year_completed", "round": 2}`
- Feature hash: `f403f593ad3f`
- Genome hash: `0187d580d068cdaa`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_a_premium_lag1`, `coe_b_premium_lag1`, `coe_e_premium_lag1`, `coe_a_premium_roll3`, `coe_c_premium_lag1`, `bank_basket_adj_close_lag1`, `bank_basket_return_lag1`, `property_basket_adj_close_lag1`, `mean_rent_postal_sector`, `mean_rent_postal_district`, `mean_rent_nearest_mall`, `mean_rent_nearest_mrt`, `mean_rent_nearest_primary`, `mean_rent_nearest_secondary`, `year_completed`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 475.736127
- Search RMSE: 462.061802
- Decision: rejected. not promising because search RMSE 462.061802 is not at least 1% below 462.061000 from 20261022T185613865949_ridge_d97c6333 after adding year_completed
- Log: `models/ridge/docs/logs/20261022T185745213323_ridge_0187d580.json`

### `20261022T185802134479_ridge_8e1591ce`

- Command: train
- Parent: `20261022T185613865949_ridge_d97c6333`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "subzone", "round": 2}`
- Feature hash: `b10a860cf566`
- Genome hash: `8e1591ce39f95b57`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_a_premium_lag1`, `coe_b_premium_lag1`, `coe_e_premium_lag1`, `coe_a_premium_roll3`, `coe_c_premium_lag1`, `bank_basket_adj_close_lag1`, `bank_basket_return_lag1`, `property_basket_adj_close_lag1`, `mean_rent_postal_sector`, `mean_rent_postal_district`, `mean_rent_nearest_mall`, `mean_rent_nearest_mrt`, `mean_rent_nearest_primary`, `mean_rent_nearest_secondary`, `subzone`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 475.432927
- Search RMSE: 461.723094
- Decision: rejected. not promising because search RMSE 461.723094 is not at least 1% below 462.061000 from 20261022T185613865949_ridge_d97c6333 after adding subzone
- Log: `models/ridge/docs/logs/20261022T185802134479_ridge_8e1591ce.json`

### `20261022T185819543785_ridge_6085acf4`

- Command: train
- Parent: `20261022T185613865949_ridge_d97c6333`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "planning_area", "round": 2}`
- Feature hash: `37abe0340f39`
- Genome hash: `6085acf4e2df3f4f`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_a_premium_lag1`, `coe_b_premium_lag1`, `coe_e_premium_lag1`, `coe_a_premium_roll3`, `coe_c_premium_lag1`, `bank_basket_adj_close_lag1`, `bank_basket_return_lag1`, `property_basket_adj_close_lag1`, `mean_rent_postal_sector`, `mean_rent_postal_district`, `mean_rent_nearest_mall`, `mean_rent_nearest_mrt`, `mean_rent_nearest_primary`, `mean_rent_nearest_secondary`, `planning_area`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 475.729619
- Search RMSE: 462.147187
- Decision: rejected. not promising because search RMSE 462.147187 is not at least 1% below 462.061000 from 20261022T185613865949_ridge_d97c6333 after adding planning_area
- Log: `models/ridge/docs/logs/20261022T185819543785_ridge_6085acf4.json`

### `20261022T185835300154_ridge_d6137c53`

- Command: train
- Parent: `20261022T185613865949_ridge_d97c6333`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "region", "round": 2}`
- Feature hash: `161e2295f0f0`
- Genome hash: `d6137c53b75b603d`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_a_premium_lag1`, `coe_b_premium_lag1`, `coe_e_premium_lag1`, `coe_a_premium_roll3`, `coe_c_premium_lag1`, `bank_basket_adj_close_lag1`, `bank_basket_return_lag1`, `property_basket_adj_close_lag1`, `mean_rent_postal_sector`, `mean_rent_postal_district`, `mean_rent_nearest_mall`, `mean_rent_nearest_mrt`, `mean_rent_nearest_primary`, `mean_rent_nearest_secondary`, `region`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 475.741146
- Search RMSE: 462.052923
- Decision: rejected. not promising because search RMSE 462.052923 is not at least 1% below 462.061000 from 20261022T185613865949_ridge_d97c6333 after adding region
- Log: `models/ridge/docs/logs/20261022T185835300154_ridge_d6137c53.json`

### `20261022T185850204042_ridge_8f0cda48`

- Command: train
- Parent: `20261022T185613865949_ridge_d97c6333`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "building_age_years", "round": 2}`
- Feature hash: `ccb0ed33d297`
- Genome hash: `8f0cda48945400bf`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_a_premium_lag1`, `coe_b_premium_lag1`, `coe_e_premium_lag1`, `coe_a_premium_roll3`, `coe_c_premium_lag1`, `bank_basket_adj_close_lag1`, `bank_basket_return_lag1`, `property_basket_adj_close_lag1`, `mean_rent_postal_sector`, `mean_rent_postal_district`, `mean_rent_nearest_mall`, `mean_rent_nearest_mrt`, `mean_rent_nearest_primary`, `mean_rent_nearest_secondary`, `building_age_years`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 475.741427
- Search RMSE: 462.108022
- Decision: rejected. not promising because search RMSE 462.108022 is not at least 1% below 462.061000 from 20261022T185613865949_ridge_d97c6333 after adding building_age_years
- Log: `models/ridge/docs/logs/20261022T185850204042_ridge_8f0cda48.json`

### `20261022T185904872580_ridge_49a7ed7e`

- Command: train
- Parent: `20261022T185613865949_ridge_d97c6333`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "lease_vs_completion_gap", "round": 2}`
- Feature hash: `3172e93c65a4`
- Genome hash: `49a7ed7e6ab5f343`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_a_premium_lag1`, `coe_b_premium_lag1`, `coe_e_premium_lag1`, `coe_a_premium_roll3`, `coe_c_premium_lag1`, `bank_basket_adj_close_lag1`, `bank_basket_return_lag1`, `property_basket_adj_close_lag1`, `mean_rent_postal_sector`, `mean_rent_postal_district`, `mean_rent_nearest_mall`, `mean_rent_nearest_mrt`, `mean_rent_nearest_primary`, `mean_rent_nearest_secondary`, `lease_vs_completion_gap`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 475.736687
- Search RMSE: 462.064554
- Decision: rejected. not promising because search RMSE 462.064554 is not at least 1% below 462.061000 from 20261022T185613865949_ridge_d97c6333 after adding lease_vs_completion_gap
- Log: `models/ridge/docs/logs/20261022T185904872580_ridge_49a7ed7e.json`

### `20261022T185919901529_ridge_dce3e970`

- Command: train
- Parent: `20261022T185613865949_ridge_d97c6333`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "flat_age_from_lease", "round": 2}`
- Feature hash: `52ba4e34ccba`
- Genome hash: `dce3e970108bb678`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_a_premium_lag1`, `coe_b_premium_lag1`, `coe_e_premium_lag1`, `coe_a_premium_roll3`, `coe_c_premium_lag1`, `bank_basket_adj_close_lag1`, `bank_basket_return_lag1`, `property_basket_adj_close_lag1`, `mean_rent_postal_sector`, `mean_rent_postal_district`, `mean_rent_nearest_mall`, `mean_rent_nearest_mrt`, `mean_rent_nearest_primary`, `mean_rent_nearest_secondary`, `flat_age_from_lease`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 475.631767
- Search RMSE: 460.161424
- Decision: rejected. not promising because search RMSE 460.161424 is not at least 1% below 462.061000 from 20261022T185613865949_ridge_d97c6333 after adding flat_age_from_lease
- Log: `models/ridge/docs/logs/20261022T185919901529_ridge_dce3e970.json`

### `20261022T185934915935_ridge_a78a1d4b`

- Command: train
- Parent: `20261022T185613865949_ridge_d97c6333`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "dist_nearest_open_mrt_km", "round": 2}`
- Feature hash: `2c745267a5df`
- Genome hash: `a78a1d4b9644c27a`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_a_premium_lag1`, `coe_b_premium_lag1`, `coe_e_premium_lag1`, `coe_a_premium_roll3`, `coe_c_premium_lag1`, `bank_basket_adj_close_lag1`, `bank_basket_return_lag1`, `property_basket_adj_close_lag1`, `mean_rent_postal_sector`, `mean_rent_postal_district`, `mean_rent_nearest_mall`, `mean_rent_nearest_mrt`, `mean_rent_nearest_primary`, `mean_rent_nearest_secondary`, `dist_nearest_open_mrt_km`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 475.649907
- Search RMSE: 461.765110
- Decision: rejected. not promising because search RMSE 461.765110 is not at least 1% below 462.061000 from 20261022T185613865949_ridge_d97c6333 after adding dist_nearest_open_mrt_km
- Log: `models/ridge/docs/logs/20261022T185934915935_ridge_a78a1d4b.json`

### `20261022T185950141961_ridge_fd9e4687`

- Command: train
- Parent: `20261022T185613865949_ridge_d97c6333`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_open_mrt_code", "round": 2}`
- Feature hash: `0a26282da6fb`
- Genome hash: `fd9e4687f4326784`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_a_premium_lag1`, `coe_b_premium_lag1`, `coe_e_premium_lag1`, `coe_a_premium_roll3`, `coe_c_premium_lag1`, `bank_basket_adj_close_lag1`, `bank_basket_return_lag1`, `property_basket_adj_close_lag1`, `mean_rent_postal_sector`, `mean_rent_postal_district`, `mean_rent_nearest_mall`, `mean_rent_nearest_mrt`, `mean_rent_nearest_primary`, `mean_rent_nearest_secondary`, `nearest_open_mrt_code`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 475.497730
- Search RMSE: 461.877277
- Decision: rejected. not promising because search RMSE 461.877277 is not at least 1% below 462.061000 from 20261022T185613865949_ridge_d97c6333 after adding nearest_open_mrt_code
- Log: `models/ridge/docs/logs/20261022T185950141961_ridge_fd9e4687.json`

### `20261022T190005628782_ridge_840003ea`

- Command: train
- Parent: `20261022T185613865949_ridge_d97c6333`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_open_mrt_name", "round": 2}`
- Feature hash: `6703f05518c8`
- Genome hash: `840003ea285baf42`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_a_premium_lag1`, `coe_b_premium_lag1`, `coe_e_premium_lag1`, `coe_a_premium_roll3`, `coe_c_premium_lag1`, `bank_basket_adj_close_lag1`, `bank_basket_return_lag1`, `property_basket_adj_close_lag1`, `mean_rent_postal_sector`, `mean_rent_postal_district`, `mean_rent_nearest_mall`, `mean_rent_nearest_mrt`, `mean_rent_nearest_primary`, `mean_rent_nearest_secondary`, `nearest_open_mrt_name`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 475.497730
- Search RMSE: 461.877277
- Decision: rejected. not promising because search RMSE 461.877277 is not at least 1% below 462.061000 from 20261022T185613865949_ridge_d97c6333 after adding nearest_open_mrt_name
- Log: `models/ridge/docs/logs/20261022T190005628782_ridge_840003ea.json`

### `20261022T190020503992_ridge_26abb975`

- Command: train
- Parent: `20261022T185613865949_ridge_d97c6333`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_open_mrt_line", "round": 2}`
- Feature hash: `0da3df1a6fcd`
- Genome hash: `26abb9758c96e350`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_a_premium_lag1`, `coe_b_premium_lag1`, `coe_e_premium_lag1`, `coe_a_premium_roll3`, `coe_c_premium_lag1`, `bank_basket_adj_close_lag1`, `bank_basket_return_lag1`, `property_basket_adj_close_lag1`, `mean_rent_postal_sector`, `mean_rent_postal_district`, `mean_rent_nearest_mall`, `mean_rent_nearest_mrt`, `mean_rent_nearest_primary`, `mean_rent_nearest_secondary`, `nearest_open_mrt_line`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 475.718252
- Search RMSE: 462.017745
- Decision: rejected. not promising because search RMSE 462.017745 is not at least 1% below 462.061000 from 20261022T185613865949_ridge_d97c6333 after adding nearest_open_mrt_line
- Log: `models/ridge/docs/logs/20261022T190020503992_ridge_26abb975.json`

### `20261022T190035438121_ridge_0a47ce40`

- Command: train
- Parent: `20261022T185613865949_ridge_d97c6333`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "dist_nearest_primary_km", "round": 2}`
- Feature hash: `6e1e9bcc39d4`
- Genome hash: `0a47ce40b118bab9`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_a_premium_lag1`, `coe_b_premium_lag1`, `coe_e_premium_lag1`, `coe_a_premium_roll3`, `coe_c_premium_lag1`, `bank_basket_adj_close_lag1`, `bank_basket_return_lag1`, `property_basket_adj_close_lag1`, `mean_rent_postal_sector`, `mean_rent_postal_district`, `mean_rent_nearest_mall`, `mean_rent_nearest_mrt`, `mean_rent_nearest_primary`, `mean_rent_nearest_secondary`, `dist_nearest_primary_km`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 475.741466
- Search RMSE: 462.060748
- Decision: rejected. not promising because search RMSE 462.060748 is not at least 1% below 462.061000 from 20261022T185613865949_ridge_d97c6333 after adding dist_nearest_primary_km
- Log: `models/ridge/docs/logs/20261022T190035438121_ridge_0a47ce40.json`

### `20261022T190051063033_ridge_4e968ead`

- Command: train
- Parent: `20261022T185613865949_ridge_d97c6333`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_primary_name", "round": 2}`
- Feature hash: `307ce63d0aef`
- Genome hash: `4e968ead446757d9`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_a_premium_lag1`, `coe_b_premium_lag1`, `coe_e_premium_lag1`, `coe_a_premium_roll3`, `coe_c_premium_lag1`, `bank_basket_adj_close_lag1`, `bank_basket_return_lag1`, `property_basket_adj_close_lag1`, `mean_rent_postal_sector`, `mean_rent_postal_district`, `mean_rent_nearest_mall`, `mean_rent_nearest_mrt`, `mean_rent_nearest_primary`, `mean_rent_nearest_secondary`, `nearest_primary_name`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 475.343764
- Search RMSE: 461.656241
- Decision: rejected. not promising because search RMSE 461.656241 is not at least 1% below 462.061000 from 20261022T185613865949_ridge_d97c6333 after adding nearest_primary_name
- Log: `models/ridge/docs/logs/20261022T190051063033_ridge_4e968ead.json`

### `20261022T190106089352_ridge_9492eaab`

- Command: train
- Parent: `20261022T185613865949_ridge_d97c6333`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "n_primary_within_1km", "round": 2}`
- Feature hash: `d4668fbbf1bf`
- Genome hash: `9492eaabd9456a53`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_a_premium_lag1`, `coe_b_premium_lag1`, `coe_e_premium_lag1`, `coe_a_premium_roll3`, `coe_c_premium_lag1`, `bank_basket_adj_close_lag1`, `bank_basket_return_lag1`, `property_basket_adj_close_lag1`, `mean_rent_postal_sector`, `mean_rent_postal_district`, `mean_rent_nearest_mall`, `mean_rent_nearest_mrt`, `mean_rent_nearest_primary`, `mean_rent_nearest_secondary`, `n_primary_within_1km`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 475.740485
- Search RMSE: 462.064265
- Decision: rejected. not promising because search RMSE 462.064265 is not at least 1% below 462.061000 from 20261022T185613865949_ridge_d97c6333 after adding n_primary_within_1km
- Log: `models/ridge/docs/logs/20261022T190106089352_ridge_9492eaab.json`

### `20261022T190120619368_ridge_fef85411`

- Command: train
- Parent: `20261022T185613865949_ridge_d97c6333`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "dist_nearest_mall_km", "round": 2}`
- Feature hash: `4b864095934e`
- Genome hash: `fef85411fcd706f2`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_a_premium_lag1`, `coe_b_premium_lag1`, `coe_e_premium_lag1`, `coe_a_premium_roll3`, `coe_c_premium_lag1`, `bank_basket_adj_close_lag1`, `bank_basket_return_lag1`, `property_basket_adj_close_lag1`, `mean_rent_postal_sector`, `mean_rent_postal_district`, `mean_rent_nearest_mall`, `mean_rent_nearest_mrt`, `mean_rent_nearest_primary`, `mean_rent_nearest_secondary`, `dist_nearest_mall_km`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 475.692078
- Search RMSE: 461.828605
- Decision: rejected. not promising because search RMSE 461.828605 is not at least 1% below 462.061000 from 20261022T185613865949_ridge_d97c6333 after adding dist_nearest_mall_km
- Log: `models/ridge/docs/logs/20261022T190120619368_ridge_fef85411.json`

### `20261022T190136284013_ridge_d89b82d3`

- Command: train
- Parent: `20261022T185613865949_ridge_d97c6333`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_mall_name", "round": 2}`
- Feature hash: `fd7239da6744`
- Genome hash: `d89b82d35b5750ee`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_a_premium_lag1`, `coe_b_premium_lag1`, `coe_e_premium_lag1`, `coe_a_premium_roll3`, `coe_c_premium_lag1`, `bank_basket_adj_close_lag1`, `bank_basket_return_lag1`, `property_basket_adj_close_lag1`, `mean_rent_postal_sector`, `mean_rent_postal_district`, `mean_rent_nearest_mall`, `mean_rent_nearest_mrt`, `mean_rent_nearest_primary`, `mean_rent_nearest_secondary`, `nearest_mall_name`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 475.530680
- Search RMSE: 461.916639
- Decision: rejected. not promising because search RMSE 461.916639 is not at least 1% below 462.061000 from 20261022T185613865949_ridge_d97c6333 after adding nearest_mall_name
- Log: `models/ridge/docs/logs/20261022T190136284013_ridge_d89b82d3.json`

### `20261022T190151159872_ridge_263daff5`

- Command: train
- Parent: `20261022T185613865949_ridge_d97c6333`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_ns_station_code", "round": 2}`
- Feature hash: `3bb50f5ffe08`
- Genome hash: `263daff56e48ddbd`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_a_premium_lag1`, `coe_b_premium_lag1`, `coe_e_premium_lag1`, `coe_a_premium_roll3`, `coe_c_premium_lag1`, `bank_basket_adj_close_lag1`, `bank_basket_return_lag1`, `property_basket_adj_close_lag1`, `mean_rent_postal_sector`, `mean_rent_postal_district`, `mean_rent_nearest_mall`, `mean_rent_nearest_mrt`, `mean_rent_nearest_primary`, `mean_rent_nearest_secondary`, `nearest_ns_station_code`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 475.689941
- Search RMSE: 461.977758
- Decision: rejected. not promising because search RMSE 461.977758 is not at least 1% below 462.061000 from 20261022T185613865949_ridge_d97c6333 after adding nearest_ns_station_code
- Log: `models/ridge/docs/logs/20261022T190151159872_ridge_263daff5.json`

### `20261022T190206220483_ridge_5b928584`

- Command: train
- Parent: `20261022T185613865949_ridge_d97c6333`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_ns_station_name", "round": 2}`
- Feature hash: `ea2190d25c07`
- Genome hash: `5b9285847c83404b`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_a_premium_lag1`, `coe_b_premium_lag1`, `coe_e_premium_lag1`, `coe_a_premium_roll3`, `coe_c_premium_lag1`, `bank_basket_adj_close_lag1`, `bank_basket_return_lag1`, `property_basket_adj_close_lag1`, `mean_rent_postal_sector`, `mean_rent_postal_district`, `mean_rent_nearest_mall`, `mean_rent_nearest_mrt`, `mean_rent_nearest_primary`, `mean_rent_nearest_secondary`, `nearest_ns_station_name`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 475.689941
- Search RMSE: 461.977758
- Decision: rejected. not promising because search RMSE 461.977758 is not at least 1% below 462.061000 from 20261022T185613865949_ridge_d97c6333 after adding nearest_ns_station_name
- Log: `models/ridge/docs/logs/20261022T190206220483_ridge_5b928584.json`

### `20261022T190221362425_ridge_7b96e57f`

- Command: train
- Parent: `20261022T185613865949_ridge_d97c6333`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "dist_nearest_ns_km", "round": 2}`
- Feature hash: `33b01b9d17b3`
- Genome hash: `7b96e57f6becdf29`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_a_premium_lag1`, `coe_b_premium_lag1`, `coe_e_premium_lag1`, `coe_a_premium_roll3`, `coe_c_premium_lag1`, `bank_basket_adj_close_lag1`, `bank_basket_return_lag1`, `property_basket_adj_close_lag1`, `mean_rent_postal_sector`, `mean_rent_postal_district`, `mean_rent_nearest_mall`, `mean_rent_nearest_mrt`, `mean_rent_nearest_primary`, `mean_rent_nearest_secondary`, `dist_nearest_ns_km`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 475.725926
- Search RMSE: 462.033877
- Decision: rejected. not promising because search RMSE 462.033877 is not at least 1% below 462.061000 from 20261022T185613865949_ridge_d97c6333 after adding dist_nearest_ns_km
- Log: `models/ridge/docs/logs/20261022T190221362425_ridge_7b96e57f.json`

### `20261022T190236626606_ridge_a1dd2fd3`

- Command: train
- Parent: `20261022T185613865949_ridge_d97c6333`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_ew_station_code", "round": 2}`
- Feature hash: `3669daa6fa2b`
- Genome hash: `a1dd2fd3674fd4a9`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_a_premium_lag1`, `coe_b_premium_lag1`, `coe_e_premium_lag1`, `coe_a_premium_roll3`, `coe_c_premium_lag1`, `bank_basket_adj_close_lag1`, `bank_basket_return_lag1`, `property_basket_adj_close_lag1`, `mean_rent_postal_sector`, `mean_rent_postal_district`, `mean_rent_nearest_mall`, `mean_rent_nearest_mrt`, `mean_rent_nearest_primary`, `mean_rent_nearest_secondary`, `nearest_ew_station_code`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 475.687772
- Search RMSE: 462.059441
- Decision: rejected. not promising because search RMSE 462.059441 is not at least 1% below 462.061000 from 20261022T185613865949_ridge_d97c6333 after adding nearest_ew_station_code
- Log: `models/ridge/docs/logs/20261022T190236626606_ridge_a1dd2fd3.json`

### `20261022T190251828634_ridge_a25b5749`

- Command: train
- Parent: `20261022T185613865949_ridge_d97c6333`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_ew_station_name", "round": 2}`
- Feature hash: `4dbac2556a6b`
- Genome hash: `a25b5749b37a763e`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_a_premium_lag1`, `coe_b_premium_lag1`, `coe_e_premium_lag1`, `coe_a_premium_roll3`, `coe_c_premium_lag1`, `bank_basket_adj_close_lag1`, `bank_basket_return_lag1`, `property_basket_adj_close_lag1`, `mean_rent_postal_sector`, `mean_rent_postal_district`, `mean_rent_nearest_mall`, `mean_rent_nearest_mrt`, `mean_rent_nearest_primary`, `mean_rent_nearest_secondary`, `nearest_ew_station_name`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 475.687772
- Search RMSE: 462.059441
- Decision: rejected. not promising because search RMSE 462.059441 is not at least 1% below 462.061000 from 20261022T185613865949_ridge_d97c6333 after adding nearest_ew_station_name
- Log: `models/ridge/docs/logs/20261022T190251828634_ridge_a25b5749.json`

### `20261022T190306768642_ridge_c31733ac`

- Command: train
- Parent: `20261022T185613865949_ridge_d97c6333`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "dist_nearest_ew_km", "round": 2}`
- Feature hash: `dbd47b001b92`
- Genome hash: `c31733acbd526a10`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_a_premium_lag1`, `coe_b_premium_lag1`, `coe_e_premium_lag1`, `coe_a_premium_roll3`, `coe_c_premium_lag1`, `bank_basket_adj_close_lag1`, `bank_basket_return_lag1`, `property_basket_adj_close_lag1`, `mean_rent_postal_sector`, `mean_rent_postal_district`, `mean_rent_nearest_mall`, `mean_rent_nearest_mrt`, `mean_rent_nearest_primary`, `mean_rent_nearest_secondary`, `dist_nearest_ew_km`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 475.741698
- Search RMSE: 462.057902
- Decision: rejected. not promising because search RMSE 462.057902 is not at least 1% below 462.061000 from 20261022T185613865949_ridge_d97c6333 after adding dist_nearest_ew_km
- Log: `models/ridge/docs/logs/20261022T190306768642_ridge_c31733ac.json`

### `20261022T190321516109_ridge_81a694ce`

- Command: train
- Parent: `20261022T185613865949_ridge_d97c6333`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_ne_station_code", "round": 2}`
- Feature hash: `fb8b1b3e225d`
- Genome hash: `81a694ce7e09f166`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_a_premium_lag1`, `coe_b_premium_lag1`, `coe_e_premium_lag1`, `coe_a_premium_roll3`, `coe_c_premium_lag1`, `bank_basket_adj_close_lag1`, `bank_basket_return_lag1`, `property_basket_adj_close_lag1`, `mean_rent_postal_sector`, `mean_rent_postal_district`, `mean_rent_nearest_mall`, `mean_rent_nearest_mrt`, `mean_rent_nearest_primary`, `mean_rent_nearest_secondary`, `nearest_ne_station_code`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 475.639648
- Search RMSE: 462.210964
- Decision: rejected. not promising because search RMSE 462.210964 is not at least 1% below 462.061000 from 20261022T185613865949_ridge_d97c6333 after adding nearest_ne_station_code
- Log: `models/ridge/docs/logs/20261022T190321516109_ridge_81a694ce.json`

### `20261022T190336269225_ridge_b5e8d085`

- Command: train
- Parent: `20261022T185613865949_ridge_d97c6333`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_ne_station_name", "round": 2}`
- Feature hash: `8f48698f8def`
- Genome hash: `b5e8d0852c5d29bd`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_a_premium_lag1`, `coe_b_premium_lag1`, `coe_e_premium_lag1`, `coe_a_premium_roll3`, `coe_c_premium_lag1`, `bank_basket_adj_close_lag1`, `bank_basket_return_lag1`, `property_basket_adj_close_lag1`, `mean_rent_postal_sector`, `mean_rent_postal_district`, `mean_rent_nearest_mall`, `mean_rent_nearest_mrt`, `mean_rent_nearest_primary`, `mean_rent_nearest_secondary`, `nearest_ne_station_name`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 475.639648
- Search RMSE: 462.210964
- Decision: rejected. not promising because search RMSE 462.210964 is not at least 1% below 462.061000 from 20261022T185613865949_ridge_d97c6333 after adding nearest_ne_station_name
- Log: `models/ridge/docs/logs/20261022T190336269225_ridge_b5e8d085.json`

### `20261022T190351265414_ridge_372ce7e2`

- Command: train
- Parent: `20261022T185613865949_ridge_d97c6333`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "dist_nearest_ne_km", "round": 2}`
- Feature hash: `ae5fd68c52ef`
- Genome hash: `372ce7e283d5e315`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_a_premium_lag1`, `coe_b_premium_lag1`, `coe_e_premium_lag1`, `coe_a_premium_roll3`, `coe_c_premium_lag1`, `bank_basket_adj_close_lag1`, `bank_basket_return_lag1`, `property_basket_adj_close_lag1`, `mean_rent_postal_sector`, `mean_rent_postal_district`, `mean_rent_nearest_mall`, `mean_rent_nearest_mrt`, `mean_rent_nearest_primary`, `mean_rent_nearest_secondary`, `dist_nearest_ne_km`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 475.739449
- Search RMSE: 462.073650
- Decision: rejected. not promising because search RMSE 462.073650 is not at least 1% below 462.061000 from 20261022T185613865949_ridge_d97c6333 after adding dist_nearest_ne_km
- Log: `models/ridge/docs/logs/20261022T190351265414_ridge_372ce7e2.json`

### `20261022T190406362023_ridge_f7148264`

- Command: train
- Parent: `20261022T185613865949_ridge_d97c6333`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_cc_station_code", "round": 2}`
- Feature hash: `eae4b2931d80`
- Genome hash: `f714826458d79b66`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_a_premium_lag1`, `coe_b_premium_lag1`, `coe_e_premium_lag1`, `coe_a_premium_roll3`, `coe_c_premium_lag1`, `bank_basket_adj_close_lag1`, `bank_basket_return_lag1`, `property_basket_adj_close_lag1`, `mean_rent_postal_sector`, `mean_rent_postal_district`, `mean_rent_nearest_mall`, `mean_rent_nearest_mrt`, `mean_rent_nearest_primary`, `mean_rent_nearest_secondary`, `nearest_cc_station_code`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 475.650199
- Search RMSE: 462.074301
- Decision: rejected. not promising because search RMSE 462.074301 is not at least 1% below 462.061000 from 20261022T185613865949_ridge_d97c6333 after adding nearest_cc_station_code
- Log: `models/ridge/docs/logs/20261022T190406362023_ridge_f7148264.json`

### `20261022T190421340146_ridge_0fc4a629`

- Command: train
- Parent: `20261022T185613865949_ridge_d97c6333`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_cc_station_name", "round": 2}`
- Feature hash: `657769448f57`
- Genome hash: `0fc4a629d392b127`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_a_premium_lag1`, `coe_b_premium_lag1`, `coe_e_premium_lag1`, `coe_a_premium_roll3`, `coe_c_premium_lag1`, `bank_basket_adj_close_lag1`, `bank_basket_return_lag1`, `property_basket_adj_close_lag1`, `mean_rent_postal_sector`, `mean_rent_postal_district`, `mean_rent_nearest_mall`, `mean_rent_nearest_mrt`, `mean_rent_nearest_primary`, `mean_rent_nearest_secondary`, `nearest_cc_station_name`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 475.650199
- Search RMSE: 462.074301
- Decision: rejected. not promising because search RMSE 462.074301 is not at least 1% below 462.061000 from 20261022T185613865949_ridge_d97c6333 after adding nearest_cc_station_name
- Log: `models/ridge/docs/logs/20261022T190421340146_ridge_0fc4a629.json`

### `20261022T190436689846_ridge_0d5da7b9`

- Command: train
- Parent: `20261022T185613865949_ridge_d97c6333`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "dist_nearest_cc_km", "round": 2}`
- Feature hash: `aa5629eeaf26`
- Genome hash: `0d5da7b9c4d431a8`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_a_premium_lag1`, `coe_b_premium_lag1`, `coe_e_premium_lag1`, `coe_a_premium_roll3`, `coe_c_premium_lag1`, `bank_basket_adj_close_lag1`, `bank_basket_return_lag1`, `property_basket_adj_close_lag1`, `mean_rent_postal_sector`, `mean_rent_postal_district`, `mean_rent_nearest_mall`, `mean_rent_nearest_mrt`, `mean_rent_nearest_primary`, `mean_rent_nearest_secondary`, `dist_nearest_cc_km`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 475.740320
- Search RMSE: 462.055323
- Decision: rejected. not promising because search RMSE 462.055323 is not at least 1% below 462.061000 from 20261022T185613865949_ridge_d97c6333 after adding dist_nearest_cc_km
- Log: `models/ridge/docs/logs/20261022T190436689846_ridge_0d5da7b9.json`

### `20261022T190452228980_ridge_eaf90fc9`

- Command: train
- Parent: `20261022T185613865949_ridge_d97c6333`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_dt_station_code", "round": 2}`
- Feature hash: `559fc5b26773`
- Genome hash: `eaf90fc90be260af`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_a_premium_lag1`, `coe_b_premium_lag1`, `coe_e_premium_lag1`, `coe_a_premium_roll3`, `coe_c_premium_lag1`, `bank_basket_adj_close_lag1`, `bank_basket_return_lag1`, `property_basket_adj_close_lag1`, `mean_rent_postal_sector`, `mean_rent_postal_district`, `mean_rent_nearest_mall`, `mean_rent_nearest_mrt`, `mean_rent_nearest_primary`, `mean_rent_nearest_secondary`, `nearest_dt_station_code`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 475.641741
- Search RMSE: 461.894774
- Decision: rejected. not promising because search RMSE 461.894774 is not at least 1% below 462.061000 from 20261022T185613865949_ridge_d97c6333 after adding nearest_dt_station_code
- Log: `models/ridge/docs/logs/20261022T190452228980_ridge_eaf90fc9.json`

### `20261022T190507797311_ridge_733b756b`

- Command: train
- Parent: `20261022T185613865949_ridge_d97c6333`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_dt_station_name", "round": 2}`
- Feature hash: `5b461f8646ed`
- Genome hash: `733b756b41fee631`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_a_premium_lag1`, `coe_b_premium_lag1`, `coe_e_premium_lag1`, `coe_a_premium_roll3`, `coe_c_premium_lag1`, `bank_basket_adj_close_lag1`, `bank_basket_return_lag1`, `property_basket_adj_close_lag1`, `mean_rent_postal_sector`, `mean_rent_postal_district`, `mean_rent_nearest_mall`, `mean_rent_nearest_mrt`, `mean_rent_nearest_primary`, `mean_rent_nearest_secondary`, `nearest_dt_station_name`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 475.641741
- Search RMSE: 461.894774
- Decision: rejected. not promising because search RMSE 461.894774 is not at least 1% below 462.061000 from 20261022T185613865949_ridge_d97c6333 after adding nearest_dt_station_name
- Log: `models/ridge/docs/logs/20261022T190507797311_ridge_733b756b.json`

### `20261022T190522961895_ridge_07daedfe`

- Command: train
- Parent: `20261022T185613865949_ridge_d97c6333`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "dist_nearest_dt_km", "round": 2}`
- Feature hash: `e91cd888cef0`
- Genome hash: `07daedfe33cc78d0`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_a_premium_lag1`, `coe_b_premium_lag1`, `coe_e_premium_lag1`, `coe_a_premium_roll3`, `coe_c_premium_lag1`, `bank_basket_adj_close_lag1`, `bank_basket_return_lag1`, `property_basket_adj_close_lag1`, `mean_rent_postal_sector`, `mean_rent_postal_district`, `mean_rent_nearest_mall`, `mean_rent_nearest_mrt`, `mean_rent_nearest_primary`, `mean_rent_nearest_secondary`, `dist_nearest_dt_km`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 475.739148
- Search RMSE: 462.058350
- Decision: rejected. not promising because search RMSE 462.058350 is not at least 1% below 462.061000 from 20261022T185613865949_ridge_d97c6333 after adding dist_nearest_dt_km
- Log: `models/ridge/docs/logs/20261022T190522961895_ridge_07daedfe.json`

### `20261022T190538516120_ridge_bcc08a50`

- Command: train
- Parent: `20261022T185613865949_ridge_d97c6333`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_te_station_code", "round": 2}`
- Feature hash: `6cd7ad6604ae`
- Genome hash: `bcc08a5096114528`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_a_premium_lag1`, `coe_b_premium_lag1`, `coe_e_premium_lag1`, `coe_a_premium_roll3`, `coe_c_premium_lag1`, `bank_basket_adj_close_lag1`, `bank_basket_return_lag1`, `property_basket_adj_close_lag1`, `mean_rent_postal_sector`, `mean_rent_postal_district`, `mean_rent_nearest_mall`, `mean_rent_nearest_mrt`, `mean_rent_nearest_primary`, `mean_rent_nearest_secondary`, `nearest_te_station_code`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 475.681142
- Search RMSE: 461.991408
- Decision: rejected. not promising because search RMSE 461.991408 is not at least 1% below 462.061000 from 20261022T185613865949_ridge_d97c6333 after adding nearest_te_station_code
- Log: `models/ridge/docs/logs/20261022T190538516120_ridge_bcc08a50.json`

### `20261022T190553717588_ridge_d2a3ad63`

- Command: train
- Parent: `20261022T185613865949_ridge_d97c6333`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_te_station_name", "round": 2}`
- Feature hash: `77270e1f38be`
- Genome hash: `d2a3ad6324de0883`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_a_premium_lag1`, `coe_b_premium_lag1`, `coe_e_premium_lag1`, `coe_a_premium_roll3`, `coe_c_premium_lag1`, `bank_basket_adj_close_lag1`, `bank_basket_return_lag1`, `property_basket_adj_close_lag1`, `mean_rent_postal_sector`, `mean_rent_postal_district`, `mean_rent_nearest_mall`, `mean_rent_nearest_mrt`, `mean_rent_nearest_primary`, `mean_rent_nearest_secondary`, `nearest_te_station_name`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 475.681142
- Search RMSE: 461.991408
- Decision: rejected. not promising because search RMSE 461.991408 is not at least 1% below 462.061000 from 20261022T185613865949_ridge_d97c6333 after adding nearest_te_station_name
- Log: `models/ridge/docs/logs/20261022T190553717588_ridge_d2a3ad63.json`

### `20261022T190609347622_ridge_1f3cceb5`

- Command: train
- Parent: `20261022T185613865949_ridge_d97c6333`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "dist_nearest_te_km", "round": 2}`
- Feature hash: `6804903b45bc`
- Genome hash: `1f3cceb5b135874c`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_a_premium_lag1`, `coe_b_premium_lag1`, `coe_e_premium_lag1`, `coe_a_premium_roll3`, `coe_c_premium_lag1`, `bank_basket_adj_close_lag1`, `bank_basket_return_lag1`, `property_basket_adj_close_lag1`, `mean_rent_postal_sector`, `mean_rent_postal_district`, `mean_rent_nearest_mall`, `mean_rent_nearest_mrt`, `mean_rent_nearest_primary`, `mean_rent_nearest_secondary`, `dist_nearest_te_km`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 475.735068
- Search RMSE: 462.033739
- Decision: rejected. not promising because search RMSE 462.033739 is not at least 1% below 462.061000 from 20261022T185613865949_ridge_d97c6333 after adding dist_nearest_te_km
- Log: `models/ridge/docs/logs/20261022T190609347622_ridge_1f3cceb5.json`

### `20261022T190624634870_ridge_98a8a320`

- Command: train
- Parent: `20261022T185613865949_ridge_d97c6333`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "n_open_mrt_within_0_5km", "round": 2}`
- Feature hash: `35cead46063f`
- Genome hash: `98a8a320afcaadff`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_a_premium_lag1`, `coe_b_premium_lag1`, `coe_e_premium_lag1`, `coe_a_premium_roll3`, `coe_c_premium_lag1`, `bank_basket_adj_close_lag1`, `bank_basket_return_lag1`, `property_basket_adj_close_lag1`, `mean_rent_postal_sector`, `mean_rent_postal_district`, `mean_rent_nearest_mall`, `mean_rent_nearest_mrt`, `mean_rent_nearest_primary`, `mean_rent_nearest_secondary`, `n_open_mrt_within_0_5km`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 475.720542
- Search RMSE: 461.959141
- Decision: rejected. not promising because search RMSE 461.959141 is not at least 1% below 462.061000 from 20261022T185613865949_ridge_d97c6333 after adding n_open_mrt_within_0_5km
- Log: `models/ridge/docs/logs/20261022T190624634870_ridge_98a8a320.json`

### `20261022T190640474785_ridge_783c02f6`

- Command: train
- Parent: `20261022T185613865949_ridge_d97c6333`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "n_open_mrt_within_1km", "round": 2}`
- Feature hash: `fc8f2eaeedeb`
- Genome hash: `783c02f6159b2a99`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_a_premium_lag1`, `coe_b_premium_lag1`, `coe_e_premium_lag1`, `coe_a_premium_roll3`, `coe_c_premium_lag1`, `bank_basket_adj_close_lag1`, `bank_basket_return_lag1`, `property_basket_adj_close_lag1`, `mean_rent_postal_sector`, `mean_rent_postal_district`, `mean_rent_nearest_mall`, `mean_rent_nearest_mrt`, `mean_rent_nearest_primary`, `mean_rent_nearest_secondary`, `n_open_mrt_within_1km`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 475.735878
- Search RMSE: 462.060891
- Decision: rejected. not promising because search RMSE 462.060891 is not at least 1% below 462.061000 from 20261022T185613865949_ridge_d97c6333 after adding n_open_mrt_within_1km
- Log: `models/ridge/docs/logs/20261022T190640474785_ridge_783c02f6.json`

### `20261022T190655489559_ridge_10d22713`

- Command: train
- Parent: `20261022T185613865949_ridge_d97c6333`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "n_open_mrt_within_2km", "round": 2}`
- Feature hash: `e8a1df49471c`
- Genome hash: `10d227133f4cadba`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_a_premium_lag1`, `coe_b_premium_lag1`, `coe_e_premium_lag1`, `coe_a_premium_roll3`, `coe_c_premium_lag1`, `bank_basket_adj_close_lag1`, `bank_basket_return_lag1`, `property_basket_adj_close_lag1`, `mean_rent_postal_sector`, `mean_rent_postal_district`, `mean_rent_nearest_mall`, `mean_rent_nearest_mrt`, `mean_rent_nearest_primary`, `mean_rent_nearest_secondary`, `n_open_mrt_within_2km`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 475.735896
- Search RMSE: 462.086844
- Decision: rejected. not promising because search RMSE 462.086844 is not at least 1% below 462.061000 from 20261022T185613865949_ridge_d97c6333 after adding n_open_mrt_within_2km
- Log: `models/ridge/docs/logs/20261022T190655489559_ridge_10d22713.json`

### `20261022T190710661066_ridge_8569fe15`

- Command: train
- Parent: `20261022T185613865949_ridge_d97c6333`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "dist_nearest_planned_mrt_km", "round": 2}`
- Feature hash: `74d25a0ba0cf`
- Genome hash: `8569fe155aa1c4ac`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_a_premium_lag1`, `coe_b_premium_lag1`, `coe_e_premium_lag1`, `coe_a_premium_roll3`, `coe_c_premium_lag1`, `bank_basket_adj_close_lag1`, `bank_basket_return_lag1`, `property_basket_adj_close_lag1`, `mean_rent_postal_sector`, `mean_rent_postal_district`, `mean_rent_nearest_mall`, `mean_rent_nearest_mrt`, `mean_rent_nearest_primary`, `mean_rent_nearest_secondary`, `dist_nearest_planned_mrt_km`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 475.741251
- Search RMSE: 462.055350
- Decision: rejected. not promising because search RMSE 462.055350 is not at least 1% below 462.061000 from 20261022T185613865949_ridge_d97c6333 after adding dist_nearest_planned_mrt_km
- Log: `models/ridge/docs/logs/20261022T190710661066_ridge_8569fe15.json`

### `20261022T190725236678_ridge_3dee39a1`

- Command: train
- Parent: `20261022T185613865949_ridge_d97c6333`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "n_planned_mrt_within_1km", "round": 2}`
- Feature hash: `d5d3f138f5d3`
- Genome hash: `3dee39a191647274`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_a_premium_lag1`, `coe_b_premium_lag1`, `coe_e_premium_lag1`, `coe_a_premium_roll3`, `coe_c_premium_lag1`, `bank_basket_adj_close_lag1`, `bank_basket_return_lag1`, `property_basket_adj_close_lag1`, `mean_rent_postal_sector`, `mean_rent_postal_district`, `mean_rent_nearest_mall`, `mean_rent_nearest_mrt`, `mean_rent_nearest_primary`, `mean_rent_nearest_secondary`, `n_planned_mrt_within_1km`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 475.733493
- Search RMSE: 462.091608
- Decision: rejected. not promising because search RMSE 462.091608 is not at least 1% below 462.061000 from 20261022T185613865949_ridge_d97c6333 after adding n_planned_mrt_within_1km
- Log: `models/ridge/docs/logs/20261022T190725236678_ridge_3dee39a1.json`

### `20261022T190740236106_ridge_a67cd9de`

- Command: train
- Parent: `20261022T185613865949_ridge_d97c6333`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "n_planned_mrt_within_2km", "round": 2}`
- Feature hash: `c3661b39bceb`
- Genome hash: `a67cd9de6b1e50c7`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_a_premium_lag1`, `coe_b_premium_lag1`, `coe_e_premium_lag1`, `coe_a_premium_roll3`, `coe_c_premium_lag1`, `bank_basket_adj_close_lag1`, `bank_basket_return_lag1`, `property_basket_adj_close_lag1`, `mean_rent_postal_sector`, `mean_rent_postal_district`, `mean_rent_nearest_mall`, `mean_rent_nearest_mrt`, `mean_rent_nearest_primary`, `mean_rent_nearest_secondary`, `n_planned_mrt_within_2km`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 475.741470
- Search RMSE: 462.060862
- Decision: rejected. not promising because search RMSE 462.060862 is not at least 1% below 462.061000 from 20261022T185613865949_ridge_d97c6333 after adding n_planned_mrt_within_2km
- Log: `models/ridge/docs/logs/20261022T190740236106_ridge_a67cd9de.json`

### `20261022T190754932542_ridge_d21a9914`

- Command: train
- Parent: `20261022T185613865949_ridge_d97c6333`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_open_mrt_is_interchange", "round": 2}`
- Feature hash: `92562c3fb51e`
- Genome hash: `d21a99143b7adb18`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_a_premium_lag1`, `coe_b_premium_lag1`, `coe_e_premium_lag1`, `coe_a_premium_roll3`, `coe_c_premium_lag1`, `bank_basket_adj_close_lag1`, `bank_basket_return_lag1`, `property_basket_adj_close_lag1`, `mean_rent_postal_sector`, `mean_rent_postal_district`, `mean_rent_nearest_mall`, `mean_rent_nearest_mrt`, `mean_rent_nearest_primary`, `mean_rent_nearest_secondary`, `nearest_open_mrt_is_interchange`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 475.737053
- Search RMSE: 462.068338
- Decision: rejected. not promising because search RMSE 462.068338 is not at least 1% below 462.061000 from 20261022T185613865949_ridge_d97c6333 after adding nearest_open_mrt_is_interchange
- Log: `models/ridge/docs/logs/20261022T190754932542_ridge_d21a9914.json`

### `20261022T190810110700_ridge_11e0e3c4`

- Command: train
- Parent: `20261022T185613865949_ridge_d97c6333`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "n_primary_within_2km", "round": 2}`
- Feature hash: `1fdf93ae44b0`
- Genome hash: `11e0e3c4d371de00`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_a_premium_lag1`, `coe_b_premium_lag1`, `coe_e_premium_lag1`, `coe_a_premium_roll3`, `coe_c_premium_lag1`, `bank_basket_adj_close_lag1`, `bank_basket_return_lag1`, `property_basket_adj_close_lag1`, `mean_rent_postal_sector`, `mean_rent_postal_district`, `mean_rent_nearest_mall`, `mean_rent_nearest_mrt`, `mean_rent_nearest_primary`, `mean_rent_nearest_secondary`, `n_primary_within_2km`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 475.737350
- Search RMSE: 462.016543
- Decision: rejected. not promising because search RMSE 462.016543 is not at least 1% below 462.061000 from 20261022T185613865949_ridge_d97c6333 after adding n_primary_within_2km
- Log: `models/ridge/docs/logs/20261022T190810110700_ridge_11e0e3c4.json`

### `20261022T190825231874_ridge_333366db`

- Command: train
- Parent: `20261022T185613865949_ridge_d97c6333`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "dist_nearest_secondary_km", "round": 2}`
- Feature hash: `726a6cf9b6b1`
- Genome hash: `333366db81023d29`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_a_premium_lag1`, `coe_b_premium_lag1`, `coe_e_premium_lag1`, `coe_a_premium_roll3`, `coe_c_premium_lag1`, `bank_basket_adj_close_lag1`, `bank_basket_return_lag1`, `property_basket_adj_close_lag1`, `mean_rent_postal_sector`, `mean_rent_postal_district`, `mean_rent_nearest_mall`, `mean_rent_nearest_mrt`, `mean_rent_nearest_primary`, `mean_rent_nearest_secondary`, `dist_nearest_secondary_km`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 475.741226
- Search RMSE: 462.065617
- Decision: rejected. not promising because search RMSE 462.065617 is not at least 1% below 462.061000 from 20261022T185613865949_ridge_d97c6333 after adding dist_nearest_secondary_km
- Log: `models/ridge/docs/logs/20261022T190825231874_ridge_333366db.json`

### `20261022T190840993432_ridge_26479d67`

- Command: train
- Parent: `20261022T185613865949_ridge_d97c6333`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_secondary_name", "round": 2}`
- Feature hash: `bb79824f56c6`
- Genome hash: `26479d67fc5d9e3d`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_a_premium_lag1`, `coe_b_premium_lag1`, `coe_e_premium_lag1`, `coe_a_premium_roll3`, `coe_c_premium_lag1`, `bank_basket_adj_close_lag1`, `bank_basket_return_lag1`, `property_basket_adj_close_lag1`, `mean_rent_postal_sector`, `mean_rent_postal_district`, `mean_rent_nearest_mall`, `mean_rent_nearest_mrt`, `mean_rent_nearest_primary`, `mean_rent_nearest_secondary`, `nearest_secondary_name`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 475.391635
- Search RMSE: 461.646836
- Decision: rejected. not promising because search RMSE 461.646836 is not at least 1% below 462.061000 from 20261022T185613865949_ridge_d97c6333 after adding nearest_secondary_name
- Log: `models/ridge/docs/logs/20261022T190840993432_ridge_26479d67.json`

### `20261022T190854685696_ridge_2bdbe655`

- Command: train
- Parent: `20261022T185613865949_ridge_d97c6333`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "n_secondary_within_1km", "round": 2}`
- Feature hash: `49f582bafe34`
- Genome hash: `2bdbe6554e46fece`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_a_premium_lag1`, `coe_b_premium_lag1`, `coe_e_premium_lag1`, `coe_a_premium_roll3`, `coe_c_premium_lag1`, `bank_basket_adj_close_lag1`, `bank_basket_return_lag1`, `property_basket_adj_close_lag1`, `mean_rent_postal_sector`, `mean_rent_postal_district`, `mean_rent_nearest_mall`, `mean_rent_nearest_mrt`, `mean_rent_nearest_primary`, `mean_rent_nearest_secondary`, `n_secondary_within_1km`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 475.741470
- Search RMSE: 462.061832
- Decision: rejected. not promising because search RMSE 462.061832 is not at least 1% below 462.061000 from 20261022T185613865949_ridge_d97c6333 after adding n_secondary_within_1km
- Log: `models/ridge/docs/logs/20261022T190854685696_ridge_2bdbe655.json`

### `20261022T190908849617_ridge_c5a7f705`

- Command: train
- Parent: `20261022T185613865949_ridge_d97c6333`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "dist_nearest_school_km", "round": 2}`
- Feature hash: `71476f2c396e`
- Genome hash: `c5a7f70526b4033e`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_a_premium_lag1`, `coe_b_premium_lag1`, `coe_e_premium_lag1`, `coe_a_premium_roll3`, `coe_c_premium_lag1`, `bank_basket_adj_close_lag1`, `bank_basket_return_lag1`, `property_basket_adj_close_lag1`, `mean_rent_postal_sector`, `mean_rent_postal_district`, `mean_rent_nearest_mall`, `mean_rent_nearest_mrt`, `mean_rent_nearest_primary`, `mean_rent_nearest_secondary`, `dist_nearest_school_km`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 475.741284
- Search RMSE: 462.065391
- Decision: rejected. not promising because search RMSE 462.065391 is not at least 1% below 462.061000 from 20261022T185613865949_ridge_d97c6333 after adding dist_nearest_school_km
- Log: `models/ridge/docs/logs/20261022T190908849617_ridge_c5a7f705.json`

### `20261022T190923426948_ridge_41bc9883`

- Command: train
- Parent: `20261022T185613865949_ridge_d97c6333`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "nearest_school_name", "round": 2}`
- Feature hash: `e0497d4a7c8c`
- Genome hash: `41bc988338c5e6e8`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_a_premium_lag1`, `coe_b_premium_lag1`, `coe_e_premium_lag1`, `coe_a_premium_roll3`, `coe_c_premium_lag1`, `bank_basket_adj_close_lag1`, `bank_basket_return_lag1`, `property_basket_adj_close_lag1`, `mean_rent_postal_sector`, `mean_rent_postal_district`, `mean_rent_nearest_mall`, `mean_rent_nearest_mrt`, `mean_rent_nearest_primary`, `mean_rent_nearest_secondary`, `nearest_school_name`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 475.056439
- Search RMSE: 462.053887
- Decision: rejected. not promising because search RMSE 462.053887 is not at least 1% below 462.061000 from 20261022T185613865949_ridge_d97c6333 after adding nearest_school_name
- Log: `models/ridge/docs/logs/20261022T190923426948_ridge_41bc9883.json`

### `20261022T190937357640_ridge_7fa5abb5`

- Command: train
- Parent: `20261022T185613865949_ridge_d97c6333`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "n_school_within_1km", "round": 2}`
- Feature hash: `148122cf5931`
- Genome hash: `7fa5abb5d7f910f0`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_a_premium_lag1`, `coe_b_premium_lag1`, `coe_e_premium_lag1`, `coe_a_premium_roll3`, `coe_c_premium_lag1`, `bank_basket_adj_close_lag1`, `bank_basket_return_lag1`, `property_basket_adj_close_lag1`, `mean_rent_postal_sector`, `mean_rent_postal_district`, `mean_rent_nearest_mall`, `mean_rent_nearest_mrt`, `mean_rent_nearest_primary`, `mean_rent_nearest_secondary`, `n_school_within_1km`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 475.741028
- Search RMSE: 462.068647
- Decision: rejected. not promising because search RMSE 462.068647 is not at least 1% below 462.061000 from 20261022T185613865949_ridge_d97c6333 after adding n_school_within_1km
- Log: `models/ridge/docs/logs/20261022T190937357640_ridge_7fa5abb5.json`

### `20261022T190951349221_ridge_3cead086`

- Command: train
- Parent: `20261022T185613865949_ridge_d97c6333`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "n_mall_within_1km", "round": 2}`
- Feature hash: `c3af3a147ade`
- Genome hash: `3cead0866e25d3c2`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_a_premium_lag1`, `coe_b_premium_lag1`, `coe_e_premium_lag1`, `coe_a_premium_roll3`, `coe_c_premium_lag1`, `bank_basket_adj_close_lag1`, `bank_basket_return_lag1`, `property_basket_adj_close_lag1`, `mean_rent_postal_sector`, `mean_rent_postal_district`, `mean_rent_nearest_mall`, `mean_rent_nearest_mrt`, `mean_rent_nearest_primary`, `mean_rent_nearest_secondary`, `n_mall_within_1km`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 475.729674
- Search RMSE: 461.993240
- Decision: rejected. not promising because search RMSE 461.993240 is not at least 1% below 462.061000 from 20261022T185613865949_ridge_d97c6333 after adding n_mall_within_1km
- Log: `models/ridge/docs/logs/20261022T190951349221_ridge_3cead086.json`

### `20261022T191005037855_ridge_3b45e0f0`

- Command: train
- Parent: `20261022T185613865949_ridge_d97c6333`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "n_mall_within_2km", "round": 2}`
- Feature hash: `bcd9d13b7f9b`
- Genome hash: `3b45e0f0e08f0d1c`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_a_premium_lag1`, `coe_b_premium_lag1`, `coe_e_premium_lag1`, `coe_a_premium_roll3`, `coe_c_premium_lag1`, `bank_basket_adj_close_lag1`, `bank_basket_return_lag1`, `property_basket_adj_close_lag1`, `mean_rent_postal_sector`, `mean_rent_postal_district`, `mean_rent_nearest_mall`, `mean_rent_nearest_mrt`, `mean_rent_nearest_primary`, `mean_rent_nearest_secondary`, `n_mall_within_2km`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 475.740538
- Search RMSE: 462.073973
- Decision: rejected. not promising because search RMSE 462.073973 is not at least 1% below 462.061000 from 20261022T185613865949_ridge_d97c6333 after adding n_mall_within_2km
- Log: `models/ridge/docs/logs/20261022T191005037855_ridge_3b45e0f0.json`

### `20261022T191019472616_ridge_0d46aa00`

- Command: train
- Parent: `20261022T185613865949_ridge_d97c6333`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "coe_a_premium_mom_change", "round": 2}`
- Feature hash: `3e39960bb2ad`
- Genome hash: `0d46aa00997b3439`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_a_premium_lag1`, `coe_b_premium_lag1`, `coe_e_premium_lag1`, `coe_a_premium_roll3`, `coe_c_premium_lag1`, `bank_basket_adj_close_lag1`, `bank_basket_return_lag1`, `property_basket_adj_close_lag1`, `mean_rent_postal_sector`, `mean_rent_postal_district`, `mean_rent_nearest_mall`, `mean_rent_nearest_mrt`, `mean_rent_nearest_primary`, `mean_rent_nearest_secondary`, `coe_a_premium_mom_change`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 475.577907
- Search RMSE: 460.881279
- Decision: rejected. not promising because search RMSE 460.881279 is not at least 1% below 462.061000 from 20261022T185613865949_ridge_d97c6333 after adding coe_a_premium_mom_change
- Log: `models/ridge/docs/logs/20261022T191019472616_ridge_0d46aa00.json`

### `20261022T191033491700_ridge_7a93ed88`

- Command: train
- Parent: `20261022T185613865949_ridge_d97c6333`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "coe_a_bids_per_quota_lag1", "round": 2}`
- Feature hash: `022f7804071c`
- Genome hash: `7a93ed8869075529`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_a_premium_lag1`, `coe_b_premium_lag1`, `coe_e_premium_lag1`, `coe_a_premium_roll3`, `coe_c_premium_lag1`, `bank_basket_adj_close_lag1`, `bank_basket_return_lag1`, `property_basket_adj_close_lag1`, `mean_rent_postal_sector`, `mean_rent_postal_district`, `mean_rent_nearest_mall`, `mean_rent_nearest_mrt`, `mean_rent_nearest_primary`, `mean_rent_nearest_secondary`, `coe_a_bids_per_quota_lag1`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 475.639233
- Search RMSE: 462.018326
- Decision: rejected. not promising because search RMSE 462.018326 is not at least 1% below 462.061000 from 20261022T185613865949_ridge_d97c6333 after adding coe_a_bids_per_quota_lag1
- Log: `models/ridge/docs/logs/20261022T191033491700_ridge_7a93ed88.json`

### `20261022T191047966820_ridge_e60966af`

- Command: train
- Parent: `20261022T185613865949_ridge_d97c6333`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "coe_b_premium_mom_change", "round": 2}`
- Feature hash: `a84736becd55`
- Genome hash: `e60966affcd12f56`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_a_premium_lag1`, `coe_b_premium_lag1`, `coe_e_premium_lag1`, `coe_a_premium_roll3`, `coe_c_premium_lag1`, `bank_basket_adj_close_lag1`, `bank_basket_return_lag1`, `property_basket_adj_close_lag1`, `mean_rent_postal_sector`, `mean_rent_postal_district`, `mean_rent_nearest_mall`, `mean_rent_nearest_mrt`, `mean_rent_nearest_primary`, `mean_rent_nearest_secondary`, `coe_b_premium_mom_change`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 475.734923
- Search RMSE: 462.125545
- Decision: rejected. not promising because search RMSE 462.125545 is not at least 1% below 462.061000 from 20261022T185613865949_ridge_d97c6333 after adding coe_b_premium_mom_change
- Log: `models/ridge/docs/logs/20261022T191047966820_ridge_e60966af.json`

### `20261022T191102092107_ridge_115f2b07`

- Command: train
- Parent: `20261022T185613865949_ridge_d97c6333`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "coe_b_bids_per_quota_lag1", "round": 2}`
- Feature hash: `b5a23b8f785f`
- Genome hash: `115f2b074eff4003`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_a_premium_lag1`, `coe_b_premium_lag1`, `coe_e_premium_lag1`, `coe_a_premium_roll3`, `coe_c_premium_lag1`, `bank_basket_adj_close_lag1`, `bank_basket_return_lag1`, `property_basket_adj_close_lag1`, `mean_rent_postal_sector`, `mean_rent_postal_district`, `mean_rent_nearest_mall`, `mean_rent_nearest_mrt`, `mean_rent_nearest_primary`, `mean_rent_nearest_secondary`, `coe_b_bids_per_quota_lag1`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 475.733826
- Search RMSE: 462.489868
- Decision: rejected. not promising because search RMSE 462.489868 is not at least 1% below 462.061000 from 20261022T185613865949_ridge_d97c6333 after adding coe_b_bids_per_quota_lag1
- Log: `models/ridge/docs/logs/20261022T191102092107_ridge_115f2b07.json`

### `20261022T191116542639_ridge_986e886c`

- Command: train
- Parent: `20261022T185613865949_ridge_d97c6333`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "coe_e_premium_mom_change", "round": 2}`
- Feature hash: `047b1999f1f7`
- Genome hash: `986e886c9ea9ad0f`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_a_premium_lag1`, `coe_b_premium_lag1`, `coe_e_premium_lag1`, `coe_a_premium_roll3`, `coe_c_premium_lag1`, `bank_basket_adj_close_lag1`, `bank_basket_return_lag1`, `property_basket_adj_close_lag1`, `mean_rent_postal_sector`, `mean_rent_postal_district`, `mean_rent_nearest_mall`, `mean_rent_nearest_mrt`, `mean_rent_nearest_primary`, `mean_rent_nearest_secondary`, `coe_e_premium_mom_change`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 475.734032
- Search RMSE: 461.978987
- Decision: rejected. not promising because search RMSE 461.978987 is not at least 1% below 462.061000 from 20261022T185613865949_ridge_d97c6333 after adding coe_e_premium_mom_change
- Log: `models/ridge/docs/logs/20261022T191116542639_ridge_986e886c.json`

### `20261022T191130570046_ridge_84feff68`

- Command: train
- Parent: `20261022T185613865949_ridge_d97c6333`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "coe_e_bids_per_quota_lag1", "round": 2}`
- Feature hash: `75025370c0d8`
- Genome hash: `84feff689c372a37`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_a_premium_lag1`, `coe_b_premium_lag1`, `coe_e_premium_lag1`, `coe_a_premium_roll3`, `coe_c_premium_lag1`, `bank_basket_adj_close_lag1`, `bank_basket_return_lag1`, `property_basket_adj_close_lag1`, `mean_rent_postal_sector`, `mean_rent_postal_district`, `mean_rent_nearest_mall`, `mean_rent_nearest_mrt`, `mean_rent_nearest_primary`, `mean_rent_nearest_secondary`, `coe_e_bids_per_quota_lag1`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 475.714402
- Search RMSE: 462.647803
- Decision: rejected. not promising because search RMSE 462.647803 is not at least 1% below 462.061000 from 20261022T185613865949_ridge_d97c6333 after adding coe_e_bids_per_quota_lag1
- Log: `models/ridge/docs/logs/20261022T191130570046_ridge_84feff68.json`

### `20261022T191145480682_ridge_28c41f88`

- Command: train
- Parent: `20261022T185613865949_ridge_d97c6333`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "dist_nearest_future_line_open_km", "round": 2}`
- Feature hash: `a9166a2c5404`
- Genome hash: `28c41f88de3cc65c`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_a_premium_lag1`, `coe_b_premium_lag1`, `coe_e_premium_lag1`, `coe_a_premium_roll3`, `coe_c_premium_lag1`, `bank_basket_adj_close_lag1`, `bank_basket_return_lag1`, `property_basket_adj_close_lag1`, `mean_rent_postal_sector`, `mean_rent_postal_district`, `mean_rent_nearest_mall`, `mean_rent_nearest_mrt`, `mean_rent_nearest_primary`, `mean_rent_nearest_secondary`, `dist_nearest_future_line_open_km`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 475.736225
- Search RMSE: 462.053177
- Decision: rejected. not promising because search RMSE 462.053177 is not at least 1% below 462.061000 from 20261022T185613865949_ridge_d97c6333 after adding dist_nearest_future_line_open_km
- Log: `models/ridge/docs/logs/20261022T191145480682_ridge_28c41f88.json`

### `20261022T191159799446_ridge_29326d84`

- Command: train
- Parent: `20261022T185613865949_ridge_d97c6333`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "dist_nearest_future_line_planned_km", "round": 2}`
- Feature hash: `c5caff1f6259`
- Genome hash: `29326d847dc37c6b`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_a_premium_lag1`, `coe_b_premium_lag1`, `coe_e_premium_lag1`, `coe_a_premium_roll3`, `coe_c_premium_lag1`, `bank_basket_adj_close_lag1`, `bank_basket_return_lag1`, `property_basket_adj_close_lag1`, `mean_rent_postal_sector`, `mean_rent_postal_district`, `mean_rent_nearest_mall`, `mean_rent_nearest_mrt`, `mean_rent_nearest_primary`, `mean_rent_nearest_secondary`, `dist_nearest_future_line_planned_km`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 475.740932
- Search RMSE: 462.064489
- Decision: rejected. not promising because search RMSE 462.064489 is not at least 1% below 462.061000 from 20261022T185613865949_ridge_d97c6333 after adding dist_nearest_future_line_planned_km
- Log: `models/ridge/docs/logs/20261022T191159799446_ridge_29326d84.json`

### `20261022T191214439797_ridge_0fd8db0b`

- Command: train
- Parent: `20261022T185613865949_ridge_d97c6333`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "n_hdb_blocks_within_500m", "round": 2}`
- Feature hash: `bd04afb9f05c`
- Genome hash: `0fd8db0b8fd84ede`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_a_premium_lag1`, `coe_b_premium_lag1`, `coe_e_premium_lag1`, `coe_a_premium_roll3`, `coe_c_premium_lag1`, `bank_basket_adj_close_lag1`, `bank_basket_return_lag1`, `property_basket_adj_close_lag1`, `mean_rent_postal_sector`, `mean_rent_postal_district`, `mean_rent_nearest_mall`, `mean_rent_nearest_mrt`, `mean_rent_nearest_primary`, `mean_rent_nearest_secondary`, `n_hdb_blocks_within_500m`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 475.739785
- Search RMSE: 462.078129
- Decision: rejected. not promising because search RMSE 462.078129 is not at least 1% below 462.061000 from 20261022T185613865949_ridge_d97c6333 after adding n_hdb_blocks_within_500m
- Log: `models/ridge/docs/logs/20261022T191214439797_ridge_0fd8db0b.json`

### `20261022T191228748444_ridge_b5e2ddeb`

- Command: train
- Parent: `20261022T185613865949_ridge_d97c6333`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "dist_to_town_centroid_km", "round": 2}`
- Feature hash: `a986ce415211`
- Genome hash: `b5e2ddeb012c4767`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_a_premium_lag1`, `coe_b_premium_lag1`, `coe_e_premium_lag1`, `coe_a_premium_roll3`, `coe_c_premium_lag1`, `bank_basket_adj_close_lag1`, `bank_basket_return_lag1`, `property_basket_adj_close_lag1`, `mean_rent_postal_sector`, `mean_rent_postal_district`, `mean_rent_nearest_mall`, `mean_rent_nearest_mrt`, `mean_rent_nearest_primary`, `mean_rent_nearest_secondary`, `dist_to_town_centroid_km`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 475.700195
- Search RMSE: 461.950274
- Decision: rejected. not promising because search RMSE 461.950274 is not at least 1% below 462.061000 from 20261022T185613865949_ridge_d97c6333 after adding dist_to_town_centroid_km
- Log: `models/ridge/docs/logs/20261022T191228748444_ridge_b5e2ddeb.json`

### `20261022T191243265353_ridge_cc8672a0`

- Command: train
- Parent: `20261022T185613865949_ridge_d97c6333`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "dist_to_cbd_km", "round": 2}`
- Feature hash: `6aab76a57505`
- Genome hash: `cc8672a082bececd`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_a_premium_lag1`, `coe_b_premium_lag1`, `coe_e_premium_lag1`, `coe_a_premium_roll3`, `coe_c_premium_lag1`, `bank_basket_adj_close_lag1`, `bank_basket_return_lag1`, `property_basket_adj_close_lag1`, `mean_rent_postal_sector`, `mean_rent_postal_district`, `mean_rent_nearest_mall`, `mean_rent_nearest_mrt`, `mean_rent_nearest_primary`, `mean_rent_nearest_secondary`, `dist_to_cbd_km`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 475.740448
- Search RMSE: 462.063917
- Decision: rejected. not promising because search RMSE 462.063917 is not at least 1% below 462.061000 from 20261022T185613865949_ridge_d97c6333 after adding dist_to_cbd_km
- Log: `models/ridge/docs/logs/20261022T191243265353_ridge_cc8672a0.json`

### `20261022T191257383856_ridge_411f8909`

- Command: train
- Parent: `20261022T185613865949_ridge_d97c6333`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "property_basket_return_lag1", "round": 2}`
- Feature hash: `cbfea443ca7a`
- Genome hash: `411f8909b71d56d1`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_a_premium_lag1`, `coe_b_premium_lag1`, `coe_e_premium_lag1`, `coe_a_premium_roll3`, `coe_c_premium_lag1`, `bank_basket_adj_close_lag1`, `bank_basket_return_lag1`, `property_basket_adj_close_lag1`, `mean_rent_postal_sector`, `mean_rent_postal_district`, `mean_rent_nearest_mall`, `mean_rent_nearest_mrt`, `mean_rent_nearest_primary`, `mean_rent_nearest_secondary`, `property_basket_return_lag1`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 475.736279
- Search RMSE: 461.228121
- Decision: rejected. not promising because search RMSE 461.228121 is not at least 1% below 462.061000 from 20261022T185613865949_ridge_d97c6333 after adding property_basket_return_lag1
- Log: `models/ridge/docs/logs/20261022T191257383856_ridge_411f8909.json`

### `20261022T191311897272_ridge_1c918d65`

- Command: train
- Parent: `20261022T185613865949_ridge_d97c6333`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "log_dist_nearest_open_mrt_km", "round": 2}`
- Feature hash: `fbbb1c035e23`
- Genome hash: `1c918d65145adbf7`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_a_premium_lag1`, `coe_b_premium_lag1`, `coe_e_premium_lag1`, `coe_a_premium_roll3`, `coe_c_premium_lag1`, `bank_basket_adj_close_lag1`, `bank_basket_return_lag1`, `property_basket_adj_close_lag1`, `mean_rent_postal_sector`, `mean_rent_postal_district`, `mean_rent_nearest_mall`, `mean_rent_nearest_mrt`, `mean_rent_nearest_primary`, `mean_rent_nearest_secondary`, `log_dist_nearest_open_mrt_km`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 475.642362
- Search RMSE: 461.709617
- Decision: rejected. not promising because search RMSE 461.709617 is not at least 1% below 462.061000 from 20261022T185613865949_ridge_d97c6333 after adding log_dist_nearest_open_mrt_km
- Log: `models/ridge/docs/logs/20261022T191311897272_ridge_1c918d65.json`

### `20261022T191325894443_ridge_a6195650`

- Command: train
- Parent: `20261022T185613865949_ridge_d97c6333`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "is_central_region", "round": 2}`
- Feature hash: `d9d7592166f3`
- Genome hash: `a6195650c292b3eb`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_a_premium_lag1`, `coe_b_premium_lag1`, `coe_e_premium_lag1`, `coe_a_premium_roll3`, `coe_c_premium_lag1`, `bank_basket_adj_close_lag1`, `bank_basket_return_lag1`, `property_basket_adj_close_lag1`, `mean_rent_postal_sector`, `mean_rent_postal_district`, `mean_rent_nearest_mall`, `mean_rent_nearest_mrt`, `mean_rent_nearest_primary`, `mean_rent_nearest_secondary`, `is_central_region`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 475.741206
- Search RMSE: 462.055340
- Decision: rejected. not promising because search RMSE 462.055340 is not at least 1% below 462.061000 from 20261022T185613865949_ridge_d97c6333 after adding is_central_region
- Log: `models/ridge/docs/logs/20261022T191325894443_ridge_a6195650.json`

### `20261022T191340466767_ridge_0a62093d`

- Command: train
- Parent: `20261022T185613865949_ridge_d97c6333`
- Generation: None
- Mutation: `{"axis": "features", "change": "add", "column": "building_age_x_central", "round": 2}`
- Feature hash: `0ca71912320c`
- Genome hash: `0a62093dde9c60dc`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_a_premium_lag1`, `coe_b_premium_lag1`, `coe_e_premium_lag1`, `coe_a_premium_roll3`, `coe_c_premium_lag1`, `bank_basket_adj_close_lag1`, `bank_basket_return_lag1`, `property_basket_adj_close_lag1`, `mean_rent_postal_sector`, `mean_rent_postal_district`, `mean_rent_nearest_mall`, `mean_rent_nearest_mrt`, `mean_rent_nearest_primary`, `mean_rent_nearest_secondary`, `building_age_x_central`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 475.723222
- Search RMSE: 462.188497
- Decision: rejected. not promising because search RMSE 462.188497 is not at least 1% below 462.061000 from 20261022T185613865949_ridge_d97c6333 after adding building_age_x_central
- Log: `models/ridge/docs/logs/20261022T191340466767_ridge_0a62093d.json`

### `20261030T103003170691_ridge_d97c6333`

- Command: train
- Parent: `20261022T185613865949_ridge_d97c6333`
- Generation: None
- Mutation: `{"axis": "finalize"}`
- Feature hash: `3dd5dfce0f43`
- Genome hash: `d97c6333368eb3dc`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_a_premium_lag1`, `coe_b_premium_lag1`, `coe_e_premium_lag1`, `coe_a_premium_roll3`, `coe_c_premium_lag1`, `bank_basket_adj_close_lag1`, `bank_basket_return_lag1`, `property_basket_adj_close_lag1`, `mean_rent_postal_sector`, `mean_rent_postal_district`, `mean_rent_nearest_mall`, `mean_rent_nearest_mrt`, `mean_rent_nearest_primary`, `mean_rent_nearest_secondary`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 475.741471
- Search RMSE: 462.061000
- Decision: accepted. final fit on the features of 20261022T185613865949_ridge_d97c6333; this stage does not select columns
- Log: `models/ridge/docs/logs/20261030T103003170691_ridge_d97c6333.json`

### `20261030T104835277133_ridge_d97c6333`

- Command: submit
- Parent: `20261030T103003170691_ridge_d97c6333`
- Generation: None
- Mutation: `{"axis": "submit"}`
- Feature hash: `3dd5dfce0f43`
- Genome hash: `d97c6333368eb3dc`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `coe_a_premium_lag1`, `coe_b_premium_lag1`, `coe_e_premium_lag1`, `coe_a_premium_roll3`, `coe_c_premium_lag1`, `bank_basket_adj_close_lag1`, `bank_basket_return_lag1`, `property_basket_adj_close_lag1`, `mean_rent_postal_sector`, `mean_rent_postal_district`, `mean_rent_nearest_mall`, `mean_rent_nearest_mrt`, `mean_rent_nearest_primary`, `mean_rent_nearest_secondary`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 473.561571
- Decision: refit of 20261030T103003170691_ridge_d97c6333 on every labeled month; this stage does not select columns and was not scored on held-out MONTHLY_RENT
- Log: `models/ridge/docs/logs/20261030T104835277133_ridge_d97c6333.json`

### `20261030T110814701751_ridge_2926f89f`

- Command: train
- Parent: `20261022T185613865949_ridge_d97c6333`
- Generation: None
- Mutation: `{"axis": "use_all_features"}`
- Feature hash: `646e8a84f218`
- Genome hash: `2926f89f185ac9fd`
- Features: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, `FEE`, `latitude`, `longitude`, `postal_code`, `max_floor`, `year_completed`, `subzone`, `planning_area`, `region`, `building_age_years`, `lease_vs_completion_gap`, `flat_age_from_lease`, `dist_nearest_open_mrt_km`, `nearest_open_mrt_code`, `nearest_open_mrt_name`, `nearest_open_mrt_line`, `dist_nearest_primary_km`, `nearest_primary_name`, `n_primary_within_1km`, `dist_nearest_mall_km`, `nearest_mall_name`, `nearest_ns_station_code`, `nearest_ns_station_name`, `dist_nearest_ns_km`, `nearest_ew_station_code`, `nearest_ew_station_name`, `dist_nearest_ew_km`, `nearest_ne_station_code`, `nearest_ne_station_name`, `dist_nearest_ne_km`, `nearest_cc_station_code`, `nearest_cc_station_name`, `dist_nearest_cc_km`, `nearest_dt_station_code`, `nearest_dt_station_name`, `dist_nearest_dt_km`, `nearest_te_station_code`, `nearest_te_station_name`, `dist_nearest_te_km`, `n_open_mrt_within_0_5km`, `n_open_mrt_within_1km`, `n_open_mrt_within_2km`, `dist_nearest_planned_mrt_km`, `n_planned_mrt_within_1km`, `n_planned_mrt_within_2km`, `nearest_open_mrt_is_interchange`, `n_primary_within_2km`, `dist_nearest_secondary_km`, `nearest_secondary_name`, `n_secondary_within_1km`, `dist_nearest_school_km`, `nearest_school_name`, `n_school_within_1km`, `n_mall_within_1km`, `n_mall_within_2km`, `coe_a_premium_lag1`, `coe_a_premium_mom_change`, `coe_a_bids_per_quota_lag1`, `coe_b_premium_lag1`, `coe_b_premium_mom_change`, `coe_b_bids_per_quota_lag1`, `coe_e_premium_lag1`, `coe_e_premium_mom_change`, `coe_e_bids_per_quota_lag1`, `coe_a_premium_roll3`, `coe_c_premium_lag1`, `dist_nearest_future_line_open_km`, `dist_nearest_future_line_planned_km`, `n_hdb_blocks_within_500m`, `dist_to_town_centroid_km`, `dist_to_cbd_km`, `bank_basket_adj_close_lag1`, `bank_basket_return_lag1`, `property_basket_adj_close_lag1`, `property_basket_return_lag1`, `log_dist_nearest_open_mrt_km`, `is_central_region`, `building_age_x_central`, `mean_rent_postal_sector`, `mean_rent_postal_district`, `mean_rent_nearest_mall`, `mean_rent_nearest_mrt`, `mean_rent_nearest_primary`, `mean_rent_nearest_secondary`
- Preprocess: numeric_impute=median, scale=standard, categorical=onehot, rare_min_count=30
- Architecture: `{"alpha": 1.0}`
- Train RMSE: 471.319357
- Search RMSE: 459.723128
- Decision: accepted. all original and auxiliary columns fit together; this fit does not select columns one at a time and does not replace best/
- Log: `models/ridge/docs/logs/20261030T110814701751_ridge_2926f89f.json`
<!-- lineage:end -->

## Predictions

<!-- predictions:start -->
### `20261022T185614941673_ridge_predict0`

- Champion run: `20261022T185613865949_ridge_d97c6333`
- Rows: 50000
- Mean: 3258.564883291744
- Std: 410.445839728446
- Min: 1586.7125565235206
- Max: 5310.970921759763
- Output: `outputs/predictions/ridge_ablation.csv`
- Refit on all labeled months: False
- Log: `models/ridge/docs/logs/20261022T185614941673_ridge_predict0.json`

### `20261030T103004551977_ridge_predict0`

- Champion run: `20261030T103003170691_ridge_d97c6333`
- Rows: 50000
- Mean: 3258.564883291744
- Std: 410.445839728446
- Min: 1586.7125565235206
- Max: 5310.970921759763
- Output: `outputs/predictions/ridge_final.csv`
- Refit on all labeled months: False
- Log: `models/ridge/docs/logs/20261030T103004551977_ridge_predict0.json`

### `20261030T104816194114_ridge_predict0`

- Champion run: `20261030T103003170691_ridge_d97c6333`
- Rows: 50000
- Mean: 3258.564883291744
- Std: 410.445839728446
- Min: 1586.7125565235206
- Max: 5310.970921759763
- Output: `outputs/predictions/ridge_final.csv`
- Refit on all labeled months: False
- Log: `models/ridge/docs/logs/20261030T104816194114_ridge_predict0.json`

### `20261030T104836155654_ridge_predict0`

- Champion run: `20261030T104835277133_ridge_d97c6333`
- Rows: 50000
- Mean: 3424.5930755348927
- Std: 434.2219260199683
- Min: 1674.7140517811556
- Max: 5577.029618601473
- Output: `outputs/predictions/ridge_refit.csv`
- Refit on all labeled months: True
- Log: `models/ridge/docs/logs/20261030T104836155654_ridge_predict0.json`
<!-- predictions:end -->

