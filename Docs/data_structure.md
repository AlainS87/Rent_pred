# CS5228 Project — Data Structure

This document summarizes the data layout and field structure for the **CS5228-2610 HDB Rental Prediction Project**.

---

## 1. Data Package Overview

The Kaggle competition provides:

```text
Data/
├── train.csv
├── test.csv
├── example-submission.csv
└── auxiliary-data/
    ├── ...
    └── ...
```

The Kaggle snapshot indicates:

- **8 CSV files** in total;
- approximately **23.6 MB**;
- **74 columns across all files**;
- `test.csv` contains about **50,000 rows** and **10 predictor columns**.

Exact auxiliary filenames should be taken from the downloaded competition package.

---

## 2. Main Supervised-Learning Tables

## 2.1 `train.csv`

Purpose:

- model training;
- validation;
- EDA;
- feature engineering.

Expected columns:

| Column | Suggested Type | Role |
|---|---|---|
| `RENT_APPROVAL_DATE` | string → datetime/period | feature |
| `TOWN` | categorical/string | feature |
| `BLOCK` | categorical/string | feature |
| `STREET` | categorical/string | feature |
| `FLAT_TYPE` | categorical/string | feature |
| `FLAT_MODEL` | categorical/string | feature |
| `FLOOR_AREA_SQM` | float | feature |
| `FURNISHED` | categorical/bool-like string | feature |
| `LEASE_COMMENCE_DATE` | integer | feature |
| `FEE` | numeric | feature |
| `MONTHLY_RENT` | numeric | **target** |

### Target

```text
MONTHLY_RENT
```

Target meaning:

> Monthly rental rate in SGD.

---

## 2.2 `test.csv`

Purpose:

- final inference;
- Kaggle submission generation.

Expected columns:

| Column | Suggested Type |
|---|---|
| `RENT_APPROVAL_DATE` | string → datetime/period |
| `TOWN` | categorical/string |
| `BLOCK` | categorical/string |
| `STREET` | categorical/string |
| `FLAT_TYPE` | categorical/string |
| `FLAT_MODEL` | categorical/string |
| `FLOOR_AREA_SQM` | float |
| `FURNISHED` | categorical/bool-like string |
| `LEASE_COMMENCE_DATE` | integer |
| `FEE` | numeric |

There is **no `MONTHLY_RENT` column** in `test.csv`.

Kaggle's displayed preview contains about **50,000 test rows**.

The visible dates span roughly:

```text
2025-03 → 2026-07
```

This should be confirmed after downloading the raw file.

---

## 2.3 `example-submission.csv`

Purpose:

- defines the required Kaggle submission schema.

Required structure:

```csv
Id,Predicted
0,2100.0
1,2300.0
2,3600.0
```

Columns:

| Column | Meaning |
|---|---|
| `Id` | Test-row identifier / submission row index |
| `Predicted` | Predicted monthly rent |

Before submission:

- row count must equal test-set row count;
- prediction order must match the test data;
- all predictions must be numeric;
- no missing values should be present.

---

# 3. Main Feature Dictionary

## `RENT_APPROVAL_DATE`

Example:

```text
2021-01
2025-03
```

Meaning:

> Year and month in which the HDB rental was approved.

Recommended derived features:

```text
approval_year
approval_month
approval_quarter
months_since_reference
```

Potential role:

- housing-market time trend;
- seasonality;
- alignment with economic indicators.

---

## `TOWN`

Examples:

```text
bishan
queenstown
clementi
jurong west
sengkang
```

Type:

```text
categorical
```

Potential information:

- broad location;
- town-level rental premium;
- proxy for connectivity and amenities.

---

## `BLOCK`

Examples:

```text
300
306A
869a
103a
```

Recommended type:

```text
string / categorical
```

Do **not** assume this is numeric because block identifiers can contain letters.

Potential use:

```text
BLOCK + STREET
```

can form a more precise building identifier.

---

## `STREET`

Examples:

```text
ubi avenue 1
woodlands drive 14
Pasir Ris Street 71
```

Type:

```text
categorical / text
```

Important preprocessing:

- lowercase;
- trim whitespace;
- normalize repeated spaces;
- standardize obvious casing inconsistencies.

Potential use:

- building lookup;
- geocoding / auxiliary block-data joins;
- street-level target encoding or categorical modeling.

---

## `FLAT_TYPE`

Examples:

```text
2-room
3-room
4-room
5-room
executive
```

Type:

```text
categorical
```

Likely strongly associated with:

- floor area;
- number of rooms;
- rental price.

Check for inconsistent naming such as:

```text
4 room
4-room
```

before encoding.

---

## `FLAT_MODEL`

Examples:

```text
model a
improved
new generation
simplified
premium apartment
maisonette
dbss
```

Type:

```text
categorical
```

Possible relation to:

- layout;
- property age;
- floor area;
- rental premium.

---

## `FLOOR_AREA_SQM`

Examples:

```text
66.0
94.0
104.0
150.0
```

Type:

```text
continuous numeric
```

Potential derived features:

```text
log_floor_area
floor_area_bucket
interaction_with_flat_type
```

The Kaggle preview shows values roughly from **31 to 215 sqm** in the displayed test distribution.

---

## `FURNISHED`

Values:

```text
yes
no
```

Type:

```text
binary categorical
```

Important observation:

The supplied Kaggle test preview displays all **50,000 test rows as `yes`**.

Before modeling:

```python
train["FURNISHED"].value_counts(dropna=False)
test["FURNISHED"].value_counts(dropna=False)
```

should be checked explicitly.

If the feature is constant in test, its usefulness and any train/test distribution mismatch should be documented.

---

## `LEASE_COMMENCE_DATE`

Examples:

```text
1969
1985
2001
2020
```

Type:

```text
integer / year
```

Recommended derived feature:

```text
flat_age_at_rental =
approval_year - lease_commence_date
```

This is usually more interpretable than the raw lease year alone.

The displayed test data spans approximately:

```text
1966 → 2022
```

---

## `FEE`

Examples from the field description:

```text
200
300
400
```

Meaning:

> Additional upfront fee associated with the rental agreement.

Type:

```text
numeric
```

Important observation:

The supplied test preview shows `FEE = 0.0` for all displayed test rows.

Verify this directly after download.

If train contains varying fees while test is constant, investigate:

- train/test distribution shift;
- whether `FEE` is meaningful at inference time;
- whether the feature should be used.

---

## `MONTHLY_RENT`

Examples:

```text
2350
3200
4000
```

Type:

```text
continuous numeric
```

Role:

```text
TARGET
```

Only available in training data.

---

# 4. Auxiliary Data

The competition's `auxiliary-data` folder contains additional datasets intended for feature engineering.

## 4.1 MRT Data

Includes locations of:

- currently operating MRT stations;
- planned MRT stations.

Potential engineered features:

```text
distance_to_nearest_mrt
distance_to_nearest_planned_mrt
num_mrt_within_1km
num_mrt_within_2km
```

Possible refinements:

- nearest line;
- interchange indicator;
- weighted accessibility score.

---

## 4.2 Shopping Mall Data

Contains mall locations.

Potential features:

```text
distance_to_nearest_mall
num_malls_within_1km
num_malls_within_2km
```

The course explicitly permits using only a subset, e.g. selected popular malls, provided the choice is justified.

---

## 4.3 School Data

Contains school locations, including primary schools.

Potential features:

```text
distance_to_nearest_primary_school
num_primary_schools_within_1km
school_density
```

---

## 4.4 HDB Block / Geospatial Data

The broader project description states that auxiliary information may include:

```text
latitude
longitude
number_of_floors
year_of_completion
other_block_attributes
```

Likely join keys:

```text
BLOCK
STREET
```

Recommended canonical building key:

```text
normalized_block + "|" + normalized_street
```

Example:

```text
869a|tampines avenue 8
```

---

## 4.5 COE Data

Contains Certificate of Entitlement prices over the same general time range as the rental data.

Potential join:

```text
RENT_APPROVAL_DATE ↔ COE month
```

Potential features:

```text
coe_price
coe_price_change
coe_rolling_mean
```

---

## 4.6 Singapore Stock-Price Data

Contains stock prices for some of the largest Singapore-based companies by market capitalization.

Potential interpretation:

> Proxy for broader economic conditions.

Potential monthly features:

```text
stock_index_proxy
monthly_return
rolling_return
rolling_volatility
```

These should be aligned carefully to `RENT_APPROVAL_DATE`.

---

# 5. Recommended Processed Data Layers

A useful internal data pipeline is:

```text
raw
  ↓
cleaned
  ↓
joined
  ↓
feature-engineered
  ↓
model-ready
```

Recommended structure:

```text
data/
├── raw/
│   ├── train.csv
│   ├── test.csv
│   ├── example-submission.csv
│   └── auxiliary/
│
├── interim/
│   ├── train_cleaned.parquet
│   ├── test_cleaned.parquet
│   ├── block_lookup.parquet
│   └── spatial_features.parquet
│
├── processed/
│   ├── train_features.parquet
│   ├── test_features.parquet
│   └── feature_metadata.json
│
└── external/
    └── ...
```

---

# 6. Recommended Canonical Cleaning Rules

For categorical strings:

```text
strip whitespace
→ lowercase
→ collapse repeated spaces
→ standardize known naming variants
```

Example:

```text
"Pasir Ris Street 71"
→
"pasir ris street 71"
```

Recommended fields:

```text
TOWN
BLOCK
STREET
FLAT_TYPE
FLAT_MODEL
FURNISHED
```

Do not modify raw files directly. Preserve `data/raw/` as immutable input.

---

# 7. Recommended Derived Features

## Temporal

```text
approval_year
approval_month
approval_quarter
months_since_2021_01
```

## Property age

```text
flat_age =
approval_year - lease_commence_date
```

## Property structure

```text
floor_area_sqm
flat_type
flat_model
floor_area × flat_type
```

## Location

```text
latitude
longitude
town
distance_to_nearest_mrt
distance_to_nearest_mall
distance_to_nearest_school
amenity_counts
```

## Building identity

```text
block_street_key
```

## Economic

```text
coe_price
stock_market_proxy
monthly_returns
rolling_market_features
```

---

# 8. Data Validation Checks

Run these checks before modeling.

## Schema

- [ ] Train contains `MONTHLY_RENT`.
- [ ] Test does not contain `MONTHLY_RENT`.
- [ ] Train/test predictor columns match.
- [ ] Expected dtypes are enforced.

## Missing values

- [ ] Missing values counted for every column.
- [ ] Missingness compared between train/test.

## Categories

- [ ] Train-only categories identified.
- [ ] Test-only categories identified.
- [ ] Casing normalized.
- [ ] Category spelling / formatting reviewed.

## Numeric ranges

- [ ] `FLOOR_AREA_SQM` validated.
- [ ] `LEASE_COMMENCE_DATE` validated.
- [ ] `FEE` distribution validated.
- [ ] `MONTHLY_RENT` outliers inspected.

## Temporal

- [ ] Train/test date ranges compared.
- [ ] Potential temporal distribution shift documented.

## Auxiliary joins

- [ ] Join-key coverage measured.
- [ ] Duplicate join keys checked.
- [ ] Missing geolocation after join quantified.

---

# 9. Train/Test Distribution Shift to Check

The Kaggle preview already suggests several points worth checking.

### Time

The visible test set is concentrated in later dates:

```text
2025-03 → 2026-07
```

If much of training is earlier, validation should mimic this temporal shift where possible.

### `FURNISHED`

Displayed test preview:

```text
yes: 50,000
no: 0
```

### `FEE`

Displayed test preview:

```text
0.0 for all 50,000 rows
```

These should be verified directly from the downloaded CSVs.

A standard random split may overestimate real test performance if strong temporal or feature-distribution shift exists.

---

# 10. Evaluation Structure

Competition metric:

```text
RMSE
```

Formula:

\[
RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat y_i)^2}
\]

Because large errors are squared, RMSE is especially sensitive to large misses.

Recommended evaluation outputs:

```text
overall RMSE
RMSE by town
RMSE by flat type
RMSE by price/rent range
RMSE by time period
largest-error examples
```

---

# 11. Final Modeling Matrix

A processed training table may conceptually look like:

```text
RENT_APPROVAL_YEAR
RENT_APPROVAL_MONTH
TOWN
FLAT_TYPE
FLAT_MODEL
FLOOR_AREA_SQM
FLAT_AGE
LATITUDE
LONGITUDE
DIST_NEAREST_MRT
DIST_NEAREST_MALL
DIST_NEAREST_SCHOOL
COE_PRICE
ECONOMIC_FEATURES
...
MONTHLY_RENT
```

while the processed test table contains the same feature columns without:

```text
MONTHLY_RENT
```

The exact final feature set should be selected through validation rather than assumed in advance.
