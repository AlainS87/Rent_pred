# Spatial features

Implemented in [src/features/spatial.py](../../src/features/spatial.py). Most columns are P0 or P1. Future-line distances, block density, the town centroid, Raffles Place, and the log distance are P2. Priorities were not tested against `MONTHLY_RENT`.

Block, MRT, school, and mall coordinates have 0 nulls and 0 zeros. After the address join in [blocks.md](blocks.md), these columns are non-null on all 150,000 train rows and all 50,000 test rows. A radius count of 0 is a real zero.

## Motivation

Stations, schools, and malls have coordinates but no block number, so they cannot be string-joined to a rental. Distances run from the block point. Separate filters keep today's trunk lines, planned stations, primary schools, and malls from being mixed into one "nearest facility" number.

## Implementation

1. Compute features once on the 9,464 blocks, then left-join them on `block_key`.
2. Haversine uses Earth radius 6371 km and returns kilometres.
3. The nearest facility is the minimum distance. If two distances differ by less than `1e-6` km, keep the lexicographically smallest `CODE`, or the smallest `norm_text(NAME)` when the table has no code.
4. `line_prefix` is the leading letters of `CODE`, uppercased. `trunk_open` is `STATUS == open` and prefix in `{NS, NE, CC, DT, TE, EW, CG}`.
5. Counts use `distance <= radius`. The 500 m block count also requires `distance > 0`, so a block is not its own neighbour.
6. Do not read `MONTHLY_RENT`.

## Formula

```text
distance = Haversine(block, facility)
count    = number of facilities with distance <= radius
log_dist_nearest_open_mrt_km = ln(dist_nearest_open_mrt_km + 0.001)
nearest_open_mrt_line = EW if the winning prefix is EW or CG, else the prefix
dist_to_cbd_km = Haversine(block, 1.283889, 103.851389)
```

The CBD point is `ns26` / `ew14` (Raffles Place) and is hardcoded. Town centroids are the arithmetic mean of block latitudes and longitudes inside each `TOWN`, using every block, not only blocks that appear in train.

An interchange name has at least two distinct `CODE` values in the full MRT table, including planned rows. About 44 names qualify. `nearest_open_mrt_is_interchange` is 1 when the nearest trunk station has such a name.

## Why this fits the tables

The MRT file has 18 code prefixes, not 18 equal lines. NS, EW, NE, CC, DT, and TE each have about 17–37 open stations. Open `CG` stations (the Changi Airport branch, 2 stations) fold into EW. `BP`, `PE`, `PTC`, `RTS`, `STC`, and `DE` have one or two stations and are not columns. `CR`, `JE`, `JS`, `JW`, and `CP` are the future-line pair, because some of those rows are already marked `open`.

Schools have no line. Primary is `MAINLEVEL_CODE == "primary"` (179 schools). Secondary is `secondary (s1-s5)` or `secondary (s1-s4)` (133 schools). All 337 schools supply the unrestricted school columns. Malls are 113 names and coordinates, with no line and no tier. `norm_text(NAME)` is unique for schools and for malls, so the stored name is the later join key. MRT `CODE` is unique; the station name is only for reading.

## Tradeoffs

One distance function and one count function cover every subset, so a line or a radius is only a filter. Planned stations cannot shrink `dist_nearest_open_mrt_km`. The cost is correlation: the six line distances agree near interchanges, and an interchange increments a radius count once per `CODE`. Haversine was kept instead of SVY21. On this island the projection gain is a few metres, while about 20 MRT longitudes have only 1–3 decimal places (about 110 m to 1 km) and about 22 more have 4–5 places (about 1–11 m). k-means and per-mall-cluster distances were rejected because those labels have no stable name.

## Under-considered points

The MRT file has no opening date, so a station marked `open` may have opened after January 2021. Schools and malls have no open or close date. Great-circle distance is not walk time. `MAX_FLOOR` is not used in the 500 m count, so a tall block and a short block both count as 1. The school table has no university. `nus high school of mathematics and science` is a mixed-level school, not the National University of Singapore. There are 12 planned stations, so most planned-station counts are 0.

## Columns

| Column | Priority | Filter or parameter |
|---|---|---|
| `dist_nearest_open_mrt_km`, `nearest_open_mrt_code`, `nearest_open_mrt_name`, `nearest_open_mrt_line` | P0 | `trunk_open`. Name is `norm_text(NAME)` of the winning code. Line is `NS`, `EW`, `NE`, `CC`, `DT`, or `TE` |
| `dist_nearest_primary_km`, `nearest_primary_name`, `n_primary_within_1km` | P0 | 179 primary schools. The count radius is 1 km. Mixed level `p1-s4` (3 schools) is excluded |
| `dist_nearest_mall_km`, `nearest_mall_name` | P0 | All 113 malls |
| `nearest_{line}_station_code`, `nearest_{line}_station_name`, `dist_nearest_{line}_km` | P1 | `{line}` is `ns`, `ew`, `ne`, `cc`, `dt`, `te`. EW also includes open `CG`. Each line has its own nearest station |
| `n_open_mrt_within_0_5km`, `n_open_mrt_within_1km`, `n_open_mrt_within_2km` | P1 | `trunk_open` inside 0.5, 1, or 2 km, counted by `CODE` |
| `dist_nearest_planned_mrt_km`, `n_planned_mrt_within_1km`, `n_planned_mrt_within_2km` | P1 | All 12 `STATUS == planned` rows, any prefix |
| `nearest_open_mrt_is_interchange` | P1 | 1 or 0 from the nearest trunk station's name. Not null when the station is not an interchange |
| `n_primary_within_2km` | P1 | Same 179 primary schools, radius 2 km |
| `dist_nearest_secondary_km`, `nearest_secondary_name`, `n_secondary_within_1km` | P1 | 133 secondary schools. The count radius is 1 km |
| `dist_nearest_school_km`, `nearest_school_name`, `n_school_within_1km` | P1 | All 337 schools |
| `n_mall_within_1km`, `n_mall_within_2km` | P1 | All 113 malls |
| `dist_nearest_future_line_open_km`, `dist_nearest_future_line_planned_km` | P2 | Prefixes `CR`, `JE`, `JS`, `JW`, `CP`, split by `open` and `planned` |
| `n_hdb_blocks_within_500m` | P2 | Other blocks from the full block file, radius 0.5 km |
| `dist_to_town_centroid_km` | P2 | Mean coordinate of blocks in the same town |
| `dist_to_cbd_km` | P2 | Fixed Raffles Place coordinate |
| `log_dist_nearest_open_mrt_km` | P2 | `ln(trunk distance + 0.001)` |

Source coordinates do not create nulls in this family. A null appears only if the block join fails or a filter matches no facilities. On the current files every filter above is non-empty, so these columns have 0 nulls on train and on test.
