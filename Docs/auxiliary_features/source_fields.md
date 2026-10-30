# Source-field features

Implemented in [src/features/source_fields.py](../../src/features/source_fields.py). Each source column keeps the transform that matches its own kind. These columns do not read `MONTHLY_RENT`. The raw `train.csv` and `test.csv` columns stay in place so the time split and the row-order join still read them.

## Motivation

`4-room` and `4 room` are the same flat type, and test streets such as `Whampoa Drive` differ from the lowercase streets in the training file only by case. Flat type has a size order. Flat model has a price order and is also a set of names that a tree can split on directly. Town, block, and street have no price order.

## Implementation

1. `approval_month_index` is the number of months since 2021-01. `2024-09` is 44 and `2026-07` is 66.
2. `lease_commence_year` keeps the four-digit commencement year. It is not a month index.
3. `town_key`, `block_key`, and `street_key` strip, lowercase, and collapse spaces. `block_key` here is the block identifier only. The `BLOCK|STREET` join key used while building spatial columns is not this feature.
4. `flat_type_ord` maps 1-room through 5-room to 1 through 5 and executive to 6, after turning hyphens into spaces.
5. `flat_model_ord` is a fixed rank from 0 through 19, cheap to expensive. `flat_model_key` is the normalized model name and stays a category.
6. `floor_area_sqm` and `fee_amount` are floats. They are not standardized on the whole table.
7. `furnished_bin` is 1 for `yes` and 0 for `no`.

## Tradeoffs

The model-name rank is a fixed list, not a rank recomputed on the search or confirm months. CatBoost receives `flat_model_key` so a split can be a subset of model names. The mixture and the numeric MLP receive `flat_model_ord` only. A shrunk block encoding still has to be fit on fit-month residuals, so it is not stored in this table.

## Columns

| Column | Source | Kind |
|---|---|---|
| `approval_month_index` | `RENT_APPROVAL_DATE` | month index |
| `town_key` | `TOWN` | normalized name |
| `block_key` | `BLOCK` | normalized identifier |
| `street_key` | `STREET` | normalized name |
| `flat_type_ord` | `FLAT_TYPE` | size rank 1–6 |
| `flat_model_ord` | `FLAT_MODEL` | price rank 0–19 |
| `flat_model_key` | `FLAT_MODEL` | normalized category |
| `floor_area_sqm` | `FLOOR_AREA_SQM` | float |
| `fee_amount` | `FEE` | float |
| `furnished_bin` | `FURNISHED` | 0/1 |
| `lease_commence_year` | `LEASE_COMMENCE_DATE` | year |
