# Block attributes and ages

Implemented in [src/features/blocks.py](../../src/features/blocks.py) and [src/features/keys.py](../../src/features/keys.py). Priorities here are P0, except `is_central_region` and `building_age_x_central`, which are P2. These priorities were not tested against `MONTHLY_RENT`.

## Motivation

The rental tables have an address but no coordinates, height, or completion year. The block table has all three, plus official areas. Ages are the elapsed years from those dates to the approval year, so the same block is older in 2026 than in 2021.

## Implementation

1. `block_key = norm_text(BLOCK) + "|" + norm_text(STREET)`. Keep letters in `BLOCK`.
2. Left-join `Data/auxiliary/sg-hdb-block.csv` on that key. The key is unique on all 9,464 blocks. Stop if the rental row count changes.
3. Turn postal `nil` into null. Turn `MAX_FLOOR` or `YEAR_COMPLETED` of `-1` into null. Do not fill those nulls.
4. Parse `approval_year` from `RENT_APPROVAL_DATE`. Subtract. Do not read `MONTHLY_RENT`.

## Formula

```text
building_age_years       = approval_year - year_completed
lease_vs_completion_gap  = LEASE_COMMENCE_DATE - year_completed
flat_age_from_lease      = approval_year - LEASE_COMMENCE_DATE
is_central_region        = 1 if region == "central region" else 0
building_age_x_central   = building_age_years * is_central_region
```

If `year_completed` is null, the two ages that use it are null. If either factor of the product is null, the product is null. Negative ages are kept.

## Why this fits the tables

`BLOCK|STREET` matches every rental row, so coordinates do not need a geocoder. Completion year and lease year are different fields already on the joined row. Region is one of five strings on the block file. School regions use another vocabulary and are not joined here.

## Tradeoffs

The join cannot duplicate rows, and the sentinel `-1` cannot become a fake year. One coordinate still stands for the whole building, and `max_floor` is the top floor of the block, not the rented unit. `region` already contains the Central Region label, so the 0/1 flag adds nothing for a model that can use the category. k-means labels and postal-code joins were rejected: cluster ids are unnamed, and 285 postal codes are `NIL`.

## Under-considered points

Ages use calendar years, so January and December of the same approval year match. A gap between lease year and completion year is not renovation. The newest completion year in the file is 2021.

## Columns

| Column | Priority | Rule | Null train / test rows |
|---|---|---|---|
| `latitude`, `longitude` | P0 | Block `LATITUDE` / `LONGITUDE` | 0 / 0. Source coordinates have 0 nulls |
| `postal_code` | P0 | `nil` becomes null | 5,259 / 1,739, from 285 blocks |
| `max_floor` | P0 | `-1` becomes null. Valid range 2–50 | 159 / 0, from 11 blocks |
| `year_completed` | P0 | `-1` becomes null. Valid range 1949–2021 | 159 / 0, same 11 blocks |
| `subzone`, `planning_area`, `region` | P0 | `norm_text` of the block field | 0 / 0. About 156 subzones, 31 planning areas, 5 regions |
| `building_age_years` | P0 | Approval year minus completion year | 159 / 0. Non-null ages are at least 0 |
| `lease_vs_completion_gap` | P0 | Lease year minus completion year | 159 / 0. Other rows may be negative |
| `flat_age_from_lease` | P0 | Main tables only | 0 / 0. Negative on 1 train row and 0 test rows |
| `is_central_region` | P2 | Exact string `central region` (1,712 blocks) | 0 / 0. Non-central rows are 0, not null |
| `building_age_x_central` | P2 | Product of age and the flag | 159 / 0. Outside the Central Region the product is 0 when age is known |
