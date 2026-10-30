# Auxiliary feature specifications

The constructed columns are documented in six files, matching the modules that build them. Column names and filters also match [../auxiliary_new_features.md](../auxiliary_new_features.md). Motivations and rejected options are summarised in [../design_tradeoffs.md](../design_tradeoffs.md).

These notes were not used to screen columns by correlation with `MONTHLY_RENT`. A priority is a data-quality and mechanism choice, not an RMSE rank.

Shared rules, used by every family:

- `block_key = norm_text(BLOCK) + "|" + norm_text(STREET)`, with letters kept in `BLOCK`.
- The block join matches 150,000 of 150,000 train rows and 50,000 of 50,000 test rows. The output row count stays the same.
- Distances are Haversine kilometres with Earth radius 6371.
- Block, distance, COE, and stock columns do not read `MONTHLY_RENT`. The six group-mean columns read training rents from the previous three approval months only.

| Document | Module | Columns |
|---|---|---|
| [blocks.md](blocks.md) | [src/features/blocks.py](../../src/features/blocks.py) | Coordinates, postal code, height, completion year, official areas, ages, Central Region flag |
| [spatial.md](spatial.md) | [src/features/spatial.py](../../src/features/spatial.py) | MRT, schools, malls, block density, town centroid, Raffles Place, log distance |
| [coe.md](coe.md) | [src/features/coe.py](../../src/features/coe.py) | Lagged COE premiums, changes, bid ratios, and the category-A three-month mean |
| [stocks.md](stocks.md) | [src/features/stocks.py](../../src/features/stocks.py) | Lagged bank and property baskets |
| [group_means.md](group_means.md) | [src/features/group_means.py](../../src/features/group_means.py) | Smoothed mean rent by postal area and by nearest mall, station, and school |
| [source_fields.md](source_fields.md) | [src/features/source_fields.py](../../src/features/source_fields.py) | Source columns preprocessed by their own kind |

`src/features/build_features.py` writes the new columns, in source-row order, to `Data/auxiliary_features/train_features.csv` and `Data/auxiliary_features/test_features.csv`, and to the Parquet files with the same names. Pass `--keep-original` to store the source columns in front of them.
