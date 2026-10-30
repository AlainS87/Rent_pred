# Auxiliary Feature Design Tradeoffs

Recorded: 3 October 2026.

This file records why the auxiliary features are designed this way. Column names, filters, and formulas live in [auxiliary_new_features.md](auxiliary_new_features.md). Use this file for motivation and limitations in the progress report or the final report. The conclusions come from table structure, join checks, and error magnitude. They have not been tested against `MONTHLY_RENT`, and they do not show whether these features reduce RMSE.

In the report, write that the features were chosen from data quality and mechanism. Do not write that they have been shown to predict rent, and do not write that SVY21 distances are already used in a model. The specified distance formula is still Haversine.

New tradeoffs are appended in [Section 9](#9-log-of-later-tradeoffs).

---

## 1. How rental rows join to blocks

The rental tables have no coordinates. MRT stations, schools, and malls have coordinates but no block number or street, so they cannot be string-joined to a rental row. The block table has both an address and coordinates, so it is the hub: join the rental row to a block, then compute distances from the block coordinates.

Three candidate keys were checked.

| Candidate key | Check | Decision |
|---|---|---|
| Normalized `BLOCK\|STREET` | Unique on all 9,464 block rows. Match rate is 100% on 150,000 train rows and 50,000 test rows, with no town-name conflicts | Adopted |
| `TOWN\|BLOCK\|STREET` | Also unique, and it matches the same rows as the key above | Not used as an extra key. Town is already brought in from the block table |
| `POSTAL_CODE` | 285 rows are `NIL`, and every duplicate postal code is that one value | Not a join key. A non-null postal code can still be kept as a block attribute |

`BLOCK` stays a string. Block identifiers can contain letters, such as `306A`. Casting them to integers would merge distinct blocks onto one key.

The join must preserve row count. If one rental row becomes many rows because of a duplicate key, later training rows are inflated.

---

## 2. MRT: why not one column per code prefix

The first idea was: Singapore has n MRT lines, and each block keeps the nearest station and distance on every line. After checking `CODE` prefixes, the 18 prefixes cannot be treated equally.

The open trunk lines with enough stations are NS, EW, NE, CC, DT, and TE, each with about 17–37 stations. Each of those lines gets a nearest-station code, name, and distance. `CG` is the Changi Airport branch of the East-West Line, with 2 open stations and 1 planned station. It is folded into EW and does not get its own columns.

`BP`, `PE`, `PTC`, `RTS`, `STC`, and `DE` have only 1–2 stations. A nearest-distance column on those prefixes is almost constant and does not describe a corridor, so they are not separate columns.

`CR`, `JE`, `JS`, `JW`, and `CP` include planned stations, or they correspond to lines that are not treated as existing trunk lines. Some of those rows already have `STATUS = open`. Putting them into "nearest open MRT" could place a 2021 rental next to a station that was not yet in service. They are kept as separate future-line distances, at a lower priority than the six trunk lines.

One physical station can have several `CODE` values. About 44 station names map to more than one code. Per-line distance is computed once per code. An interchange is not split into two coordinates. Whether the nearest trunk station is an interchange is marked by whether its name maps to at least two codes in the full station table.

The file has no opening date, only `open` and `planned`. A station marked `open` may still have opened after 2021. In the report this is a limitation: trunk-line distance can be optimistic for early rentals. Opening dates were not invented.

Schools and malls have no line field, so the per-line pattern is not copied onto them.

---

## 3. Schools and malls: what the grouping is

Schools are grouped by level, not by line.

- 179 primary schools. Nearest distance, plus counts within 1 km and 2 km. One kilometre is a commonly discussed primary-school distance. It is not a column in this table.
- 133 secondary schools in the narrow sense, only `secondary (s1-s5)` and `secondary (s1-s4)`.
- All 337 schools also get a nearest distance and a 1 km count, so junior colleges and mixed-level schools are covered.

The mixed level `mixed level (p1-s4)` has only 3 schools. Those schools are not added to the primary set, so one school is not counted twice.

The school table contains no university. The only name containing "university" is `nus high school of mathematics and science`. Its level is `mixed level (s1-jc2)`. It is a high school, not the National University of Singapore. A university-proximity feature needs external university coordinates. It cannot be derived from these 337 rows.

There are 113 malls, with a name and coordinates only. The table has no tier, floor area, or line. The features are therefore nearest mall, count within 1 km, and count within 2 km. Malls are not clustered so that each cluster can have its own nearest-mall distance: a cluster has no stable name and cannot be joined to a later manual tier table.

A radius count stores only the count, plus the name of the single nearest mall. If one regional mall and two neighbourhood malls sit inside 1 km, the current columns do not separate them. Expressing "a top-tier mall is nearby" requires a manual tier table first, then a separate count of tiers inside the radius. That step has not been done.

---

## 4. Names and codes: no tier yet, but the row must stay joinable

Mall tier, school type, and station importance can be classified by hand later. The auxiliary tables do not contain those tiers, so the features do not store labels such as "top-tier" or "neighbourhood".

The join key is whichever identifier is actually unique in that table.

| Object | What the table contains | Column a later classification table joins on |
|---|---|---|
| Mall | No code. After `norm_text(NAME)`, all 113 names are still distinct | `nearest_mall_name` |
| School | No code. All 337 school names stay unique under the same normalization | `nearest_primary_name`, `nearest_secondary_name`, `nearest_school_name` |
| MRT | All 243 `CODE` values are unique. There are only about 192 names, and names repeat | `nearest_open_mrt_code`, and each line's `nearest_*_station_code`. The station name is kept for reading |

School `REGION` values are `north`, `south`, `east`, and `west`. Block `REGION` values are the other set, such as `central region` and `north-east region`. Those two encodings cannot be joined on the region string. School `MRT_STATIONS` is free text with inconsistent spelling, so it is not a join key either.

---

## 5. Area and density: why there is no clustering

The block table already has official areas: about 156 `SUBZONE` values, about 31 `PLANNING_AREA` values, and 5 `REGION` values. Those names are stable and can be interpreted directly.

k-means cluster ids were not adopted. The number of clusters would be another choice, the labels are not place names, and the clusters repeat the official areas above. Clustering on rent would also leak the target into the features. The density features do not use rent: the number of other blocks within 500 m, distance to the centroid of blocks in the same town, and distance to a fixed Raffles Place point.

Raffles Place uses the coordinates of `ns26` and `ew14` in the MRT table. Both rows share latitude `1.283889` and longitude `103.851389`. The reference point is hardcoded so the implementation does not search station names fuzzily at runtime. It represents that one central point, not other employment centres such as Jurong or Tampines.

`MAX_FLOOR` is the top floor of the whole building, not the floor of the rented unit. High-floor and low-floor units in the same building receive the same value. `MAX_FLOOR = -1` and `YEAR_COMPLETED = -1` occur on the same 11 blocks, which correspond to 159 train rows and 0 test rows. `-1` is treated as missing. Valid completion years run from 1949 to 2021. After the sentinel is removed, approval year minus completion year is not negative.

Lease commencement year comes from the main table and can differ from completion year. Both building age and "lease year minus completion year" are kept. Their difference shows that the two years measure different things. It should not be described as renovation or reconstruction.

---

## 6. Distance formula: Haversine, a local plane, and SVY21

### 6.1 Three formulas compared

Every facility coordinate is in Singapore. Block latitudes run from about 1.270 to 1.457 and longitudes from about 103.685 to 103.988. The island span is about 21 km north–south and 34 km east–west. The farthest pair of points is about 40 km apart.

| Formula | Meaning | Error on this extent |
|---|---|---|
| Haversine, radius 6371 km | Great-circle distance on a sphere. This is the formula in the current spec | About 0.5% long relative to the ellipsoid on a north–south path, roughly 5 m per km |
| Local plane | Scale longitude by `cos(reference latitude)` at 1.35° N, then use Pythagoras | Differs from Haversine by metres or less |
| Euclidean distance after SVY21 (EPSG:3414) | Singapore's official projection, in metres | The projection itself is accurate to about centimetres relative to ground distance |

SVY21 is the more accurate survey formula. Haversine uses one radius for both axes, so a 1 km north–south distance can be off by about 5 m, and a 40 km cross-island distance can be off by on the order of 100–200 m. A local plane is simpler to compute, but it is not the official projection either.

### 6.2 How fine the stored coordinates are

Changing the projection only reduces formula error. It cannot recover a position that was already rounded away. In Singapore, decimal places correspond to roughly 110 m (3 places), 11 m (4), 1 m (5), and 0.1 m (6).

| Table | How coordinates are written | What that means for SVY21 |
|---|---|---|
| Blocks | Most latitudes and longitudes have 6 decimal places, about 0.1 m | Fine enough for SVY21's few-metre advantage to show up |
| Schools and malls | Many decimal places | Numerically sufficient. Digits past the 7th are floating-point tails, not centimetre surveying |
| MRT | About 200 of 243 stations have longitude to 6 or more decimal places | Between these stations and a block, SVY21 is more accurate by a few metres |
| Coarser MRT stations | About 20 longitudes have only 1–3 decimal places, about 110 m to 1 km. About 22 more have 4–5 places, about 1–11 m | Position error is already larger than the few metres the formula can recover |

Examples: Bartley is stored as `1.3425, 103.88`, so longitude is known only to about 1 km. Aviation Park has latitude `1.37`, also about 1 km. For these stations, neither Haversine nor SVY21 produces a metre-level nearest distance.

All four coordinate tables have no nulls and no `(0, 0)`, and the values lie in Singapore. The data are therefore sufficient to transform EPSG:4326 to EPSG:3414. What they are not sufficient for is a distance that is more trustworthy than Haversine for every MRT station after that transform.

### 6.3 Decision

Distance columns keep using Haversine, Earth radius 6371 km, output in kilometres. Every distance column uses that same formula.

The reasons are:

1. Inside a 40 km extent, the gap between Haversine and a local plane is smaller than the uncertainty already present in the coordinates and facility points.
2. The part SVY21 improves is a few metres per kilometre. The nearest station changes only when two facilities differ by a few metres as well. Ties are already broken by lexicographic code or name, not by the projection.
3. A 1 km or 2 km count changes only when a facility sits on the radius boundary.
4. Some MRT coordinates are coarser than 11 m, and some are coarser than 1 km. For those stations SVY21 does not add precision, but it does add a projection dependency.

If a later step must align with government maps metre by metre, transform the same WGS84 latitudes and longitudes to EPSG:3414 and then use planar distance. That would be a separate implementation, not the current feature definition. If the report mentions this comparison, say that Haversine was kept because coordinate precision already dominates the projection gain, not because SVY21 was never considered.

---

## 7. Economic and time features

COE is a certificate-of-entitlement bidding price. It is not a consumer-price index, a wage, or a general consumption level. The whole island has one set of numbers each month, so COE cannot separate two flats. It is kept because it can be aligned by month and used as a proxy for credit conditions and vehicle demand.

The price fields contain `$` and thousands separators and must be parsed first. Each category has at most two rounds in a month. The monthly price uses round 2 by default, because it is the later exercise in that month. Round 1 is used only when round 2 is absent. A rental month uses the previous month's price, so a round that may not yet have been published in the rental month is not attached. The approval field is only a year and month, so there is no evidence that round 2 of the same month was already public.

Categories A, B, and E get a lagged price, a month-on-month change, and bids divided by quota. A three-month mean is built only for category A. Category C is commercial vehicles and keeps only one lagged price. December 2020 is not in the file, so the lagged price for January 2021 is missing. That missing value comes from the lag rule. It is not filled with the January 2021 price. The test set ends in July 2026, so the one-month lag uses June 2026, which is complete.

The stock file has 78 symbols, including listings outside Singapore. Some series are short, and 85 price rows are missing. There is not one column per symbol. Two equal-weight baskets are used, and all three symbols must have a month-end adjusted close in the month being used:

- Banks: `D05.SI`, `O39.SI`, `U11.SI`
- Property: `C09.SI`, `U14.SI`, `9CI.SI`

The baskets are also lagged by one month. Stock history starts in January 2021, so the return for the first one or two approval months can be missing. Both baskets are still island-wide time series. They do not vary across flats in the same month. Their priority is below block location and MRT distance.

---

## 8. Three layers to keep separate in the report

1. **Facts already checked.** The address key is one-to-one and matches completely. Coordinates are not missing. MRT prefix counts and `open` / `planned` counts are known. The school table has no university. The mall table has no tier and no code. Eleven blocks use the `-1` sentinel. Some MRT coordinates have only 1–4 decimal places.
2. **Design choices made from those facts, not yet tested on rent.** Six trunk lines instead of 18 prefixes. Primary schools counted at 1 km. Names or codes kept for a later manual tier. COE lagged by one month. Distance uses Haversine rather than SVY21.
3. **Work not done.** Feature columns have not been written into the training table. RMSE with and without these columns has not been compared. Mall tiers and school tiers have not been labelled by hand. University coordinates have not been added. MRT opening dates have not been added.

If the third layer appears in the report, put it under limitations or next steps, not under results.

---

## 9. Log of later tradeoffs

Append every later design tradeoff below this line. Do not rewrite Sections 1–8 except to correct a factual error. Each entry uses this template:

```text
### YYYY-MM-DD — short title

- Options considered:
- Evidence checked:
- Decision:
- What the report may say:
- What the report must not say:
```

### 2026-10-03 — one specification file per output column

- Options considered: one markdown file per output column, or one file per feature family with a shared formula.
- Evidence checked: the catalog lists 77 columns. Many share Haversine, the block join, or the COE lag, but the line, radius, school level, or COE category changes the column. Missing-row counts also differ: postal `NIL` hits 5,259 train rows, the `-1` completion sentinel hits 159 train rows, and the property-stock basket lag hits 28,466 train close values.
- Decision: write `Docs/auxiliary_features/<column>.md` for each of the 77 columns, plus an index. Each file repeats the shared join and distance rules and states that column's filter, formula, and missingness. The Chinese catalog is unchanged except for a pointer to the folder.
- What the report may say: each proposed column has a standalone note of how it would be built and where source gaps would leave it null.
- What the report must not say: that splitting the notes changed any feature definition, or that these columns were tested against `MONTHLY_RENT`.

### 2026-10-03 — feature code is shared by algorithm, not split by column

- Options considered: one Python file per output column (and a second file per column for plots), or a few modules that share the join, Haversine, COE lag, and stock-basket rules, with plots grouped by family.
- Evidence checked: 77 columns reuse the same address key, the same distance function, and the same one-month lag. The per-column notes already state the filter that changes (line, radius, school level, or COE category).
- Decision: implement `src/features/` (`keys`, `blocks`, `spatial`, `coe`, `stocks`, `build_features`) and four plot scripts under `src/viz/`. Plots read the written parquet files and do not use `MONTHLY_RENT`.
- What the report may say: the 77 columns are produced by shared functions, and the figures describe those columns.
- What the report must not say: that a separate script was validated for each column, or that the figures show these columns predict rent.

### 2026-10-03 — feature notes merged to match the modules

- Options considered: keep one markdown file per output column (77 files), or replace them with four family notes matching `blocks`, `spatial`, `coe`, and `stocks`.
- Evidence checked: the 77 files repeated the same join, Haversine, and lag text. The code that builds the columns is already four modules. Column-specific filters and null counts still differ and fit in one table per family.
- Decision: keep `Docs/auxiliary_features/blocks.md`, `spatial.md`, `coe.md`, and `stocks.md`, plus the index. Delete the per-column files. Formulas and measured null counts stay. The earlier one-file-per-column log entry is left in place.
- What the report may say: the feature notes follow the same split as the code, and each column's filter and missing-row count is in that family's table.
- What the report must not say: that merging the notes changed a feature definition, or that the columns were tested against `MONTHLY_RENT`.

### 2026-10-03 — saved feature tables omit the source columns

- Options considered: write the original rental columns plus the 77 new columns, or write only the 77 new columns and attach the source columns later by row order.
- Evidence checked: `train.csv` and `test.csv` already hold the source fields. The feature build does not reorder rows. `MONTHLY_RENT` is not an input to any new column.
- Decision: `build_rental_features(..., keep_original=False)` is the default. It writes `Data/auxiliary_features/train_features.parquet` and `test_features.parquet` with only the 77 new columns, in the same order as the source CSVs. `--keep-original` keeps the source columns in front.
- What the report may say: the feature files line up with the source files by row, and training can concatenate them.
- What the report must not say: that the feature file itself contains `MONTHLY_RENT`, or that this split was chosen because of a rent correlation test.

### 2026-10-03 — feature tables are CSV

- Options considered: Parquet, which is smaller, or CSV, which opens as a table in the editor and in Excel.
- Evidence checked: the default table is 150,000 rows by 77 columns. CSV is what the source rental files already use, and row order is the join key back to those files.
- Decision: write `train_features.csv` and `test_features.csv` under `Data/auxiliary_features/`. The plot scripts read those CSVs.
- What the report may say: the feature tables are plain CSV and line up with the source files by row.
- What the report must not say: that the file format was chosen because it changed predictive performance.

### 2026-10-03 — group-mean rents use only earlier training rows

- Options considered: store postal sector and postal district as columns; or store only the mean rent of each group. For the mean itself: all training rents, or rents from months strictly before the row.
- Evidence checked: a six-digit postal code's first two digits are the postal sector, and those sectors map onto districts D01–D28. The last four digits repeat the building. `nearest_mall_name`, `nearest_open_mrt_code`, `nearest_primary_name`, and `nearest_secondary_name` already identify the groups. Train approval months start at `2021-01`. Test has no `MONTHLY_RENT`.
- Decision: add `mean_rent_postal_sector`, `mean_rent_postal_district`, `mean_rent_nearest_mall`, `mean_rent_nearest_mrt`, `mean_rent_nearest_primary`, and `mean_rent_nearest_secondary`. Do not store the sector or district labels. Each value uses training rents from earlier months only, shrunk toward the earlier island-wide mean with strength 20. `2021-01` stays null. Primary and secondary schools are separate groups.
- What the report may say: these columns are target encodings of earlier rents in the same postal area or nearest facility, with a fixed shrinkage toward the earlier island-wide mean.
- What the report must not say: that the mean is a property of the mall, station, or school, that same-month or test rents were included, or that the columns were kept because a correlation with rent was measured.

### 2026-10-03 — group means use the previous three months

- Options considered: all training rents strictly before the approval month, or only the three calendar months immediately before it. The island-wide fallback can use the same window or the full history.
- Evidence checked: approval dates are year-month from `2021-01` onward. Rents over 2021--2025 are not a stable level, so a cumulative mean keeps old rents inside a later feature. A single month would make `n` small for rare malls and schools. The previous three months are a quarter-length window and still exclude month `T`.
- Decision: `mean_rent_*` for month `T` uses training rows in `T-1`, `T-2`, and `T-3` only. Shrinkage strength stays 20, and `global_mean` is the mean of that same window. `2021-01` stays null. February uses January only.
- What the report may say: the group means are recent local rent levels, shrunk toward the same quarter's island-wide mean when the group is thin.
- What the report must not say: that the mean still accumulates every rent since 2021, or that this window was chosen because it reduced RMSE.

### 2026-10-03 — group-mean figures use the formula, and each map has its own scale

- Options considered: heatmaps only where a training rental exists in that group and month; or a cell for every group and month whenever the previous three months can be evaluated. For the Singapore maps: one shared colour scale across months, or a separate minimum and maximum on each map.
- Evidence checked: the feature for month `T` is defined from `T-1`, `T-2`, and `T-3` and does not require a rental in month `T`. A shared scale made early years pale and later years dark, so within-map differences were hard to see. The basemap is an OpenStreetMap tile mosaic. Landmarks are eight stations already in `sg-mrt-stations.csv`.
- Decision: `heatmap_*.png` fills a cell from the formula even if that month has no rental in the group. Blank cells remain only when the window has no training rents. `outputs/figures/maps/` draws June and December maps for malls, primary schools, and open MRT stations. Each file sets its own colour limits. December 2025 is omitted.
- What the report may say: the figures describe the previous-quarter group means. A map region of one colour is one facility's nearest-facility area. Colour bars are not comparable across files.
- What the report must not say: that a darker region in 2024 than in 2021, across two files, shows a rent increase, or that these figures selected features by RMSE.

### 2026-10-03 — English feature summary does not select columns

- Options considered: a Chinese note compiled with XeLaTeX, or an English note compiled with pdfLaTeX; a summary that ranks features by a model score, or a summary that only records the current design and a future test plan.
- Evidence checked: the feature definitions, join results, and leakage rules are already written in the auxiliary catalogs and in this log. No model has been trained.
- Decision: write `Docs/auxiliary_feature_summary.tex` in English for the default Overleaf pdfLaTeX compiler. It separates implemented columns from planned checks with a linear model, XGBoost, and an MLP.
- What the report may say: the note describes how auxiliary tables are joined and how those columns would later be tested on a time split.
- What the report must not say: that the summary measured which columns reduce RMSE, or that a model score was used to keep or drop a feature.

### 2026-10-03 — summary is one section and cites the local figures

- Options considered: a long note with separate sections for every feature family and no figures, or one section that covers the join, the limits, and the planned checks, with the figures already saved under `images/`.
- Evidence checked: the nine PNG files in `images/` match the plots written from the feature table. The note had not included them, and the previous draft said not to call `\includegraphics`.
- Decision: rewrite `Docs/report/auxiliary_feature_summary.tex` as a single section. Cite `images/*.png` with the `graphicx` package. Keep pdfLaTeX. The figures remain descriptive; they are not model results.
- What the report may say: the note shows how auxiliary tables are joined, states the limits of those columns, and outlines a later time-split comparison of a linear model, XGBoost, and an MLP.
- What the report must not say: that a figure shows a column predicts rent, or that the shorter note dropped a feature from the model.

### 2026-10-22 — evolving baselines keep a confirm month untouched

- Options considered: one random holdout that both accepts mutations and is the reported score; a single time holdout used the same way; or two later blocks of approval months, with only the earlier block allowed to accept a change. The searchable state could be the architecture alone, or the architecture plus the feature subset and the in-model impute, scale, encoding, and rare-level threshold. Checkpoint folders could be one directory per run, or one directory per feature set.
- Evidence checked: `train.csv` approval months run from `2021-01` through `2025-03`, and `test.csv` has no `MONTHLY_RENT`. The auxiliary feature file is row-aligned with the rental file and is not concatenated on disk. The report note already names a linear model, XGBoost, and an MLP, and it already says to train on earlier months. String fields differ in case and in `FLAT_TYPE` hyphenation (`4 room` versus `4-room`). No full-table RMSE has been measured.
- Decision: `python -m src.pipeline.run` joins the rental and feature tables by row order. `ridge`, `xgb`, and `mlp` live under `models/`. `evolve` proposes one change at a time to the feature subset, the preprocess gene, or the architecture, and accepts it only when search RMSE falls by at least 1%. Search is the three approval months before the last three. Confirm is the last three months and is scored once after search stops. The same feature set shares `models/<model>/ckpt/<feature_hash>/`, with `features.txt` listing the columns. String lowercasing and the `FLAT_TYPE` hyphen fix are fixed cleanup, not genes. `--max-rows` stride-samples each period so a smoke run can finish; that cap is written on the log. A later evolve resumes the champion only when the pool, row cap, month cut, and epoch cap match that run.
- What the report may say: the pipeline can change features, preprocessing, and architecture from its own logs, and the confirm months are not used to accept those changes. A feature set has its own checkpoint folder and a file that names the columns.
- What the report must not say: that a `--max-rows` smoke run measured which features or models reduce RMSE on the full table, or that the confirm RMSE was used to choose the genome.

### 2026-10-22 — nominal fields are encoded inside the model preprocessor

- Options considered: leave one-hot and ordinal encoding in front of the model; or select the genome columns inside each model's preprocessor and also offer target encoding and frequency bins. Target means could use every labeled month, or only the fit months.
- Evidence checked: `TOWN`, `FLAT_TYPE`, block identifiers, and the auxiliary name and code columns are nominal. `test.csv` has no `MONTHLY_RENT`. The group-mean columns already shrink earlier rents toward a global mean with strength 20. No full-table comparison of one-hot, target encoding, and bins has been run.
- Decision: `models/<model>/preprocess.py` selects the requested columns and encodes nominal fields as `onehot`, `ordinal`, `target`, or `bin`. Target encoding is the fit-month mean, shrunk toward the fit-month global mean with strength 20. Unseen levels use that global mean. Frequency bins use 10 bins from the fit months, and an unseen level gets bin 10. The matrix passed to ridge, XGBoost, and the MLP is 2-D `float64` and finite. Default genomes stay `onehot`.
- What the report may say: nominal columns are turned into numbers inside the model, and target encoding does not read rents from the search months, the confirm months, or the test set.
- What the report must not say: that target encoding or bins lowered full-table RMSE, or that a capped train-and-predict run chose the encoding.

### 2026-10-22 — full time split keeps block, COE, and model-specific extras

- Options considered: keep the capped three-column smoke runs; retrain `original` with the default budgets and then raise the MLP learning rate from 0.001 to 0.01; add `blocks`, `spatial`, `coe`, `stocks`, and `group_means` one at a time; encode nominal fields as one-hot, target means, or frequency bins.
- Evidence checked: fit months are 2021-01 through 2024-09 (132,802 rows), search is 2024-10 through 2024-12 (8,274 rows), and confirm is 2025-01 through 2025-03 (8,924 rows). On `original`, search RMSE was 655.83 for ridge, 571.45 for XGBoost with 200 trees, and 896.24 for the MLP with 20 epochs. The MLP was not near the capped predict-zero search RMSE of 3243.56. A family was kept only when search RMSE fell by at least 1% from the best set so far. Ridge then reached 547.44 with `blocks`, 545.56 with `spatial` added, 467.01 with `coe`, 468.02 with `stocks`, and 458.29 with `group_means` (confirm 466.11). XGBoost reached 539.54 with `blocks`, 534.60 with `spatial`, 460.08 with `coe`, 454.12 with `stocks` (confirm 518.48), and 454.43 with `group_means`. On the XGBoost set `original,blocks,coe,stocks`, target encoding had search RMSE 702.89 and bins had 456.22.
- Decision: leave the MLP learning rate at 0.001. Keep `blocks` and `coe` for both ridge and XGBoost. Keep `group_means` for ridge and drop `spatial` and `stocks` there. Keep `stocks` for XGBoost and drop `spatial` and `group_means` there. The prediction file uses the XGBoost one-hot genome because its search RMSE, 454.12, is below ridge's 458.29. Target encoding and bins stay off. The file is `outputs/predictions/xgb_original_blocks_coe_stocks.csv`.
- What the report may say: on this time split, block attributes and lagged COE prices lowered search RMSE for ridge and XGBoost; group-mean rents did so for ridge, and the stock basket did so for XGBoost. One-hot encoding beat target encoding and frequency bins on the XGBoost feature set under the 1% search rule. Confirm RMSE for that XGBoost run is 518.48, which was recorded and not used to accept the genome.
- What the report must not say: that the capped 4,000-row or one-epoch runs chose the features, that spatial columns were tested and found useless beyond this 1% rule, or that the lower ridge confirm RMSE of 466.11 replaced the XGBoost prediction.

### 2026-10-22 — auxiliary columns are screened by a ridge add-one before a model confirms them

- Options considered: retrain XGBoost and the MLP once for every auxiliary column; or fit ridge on the current best feature set plus one column, and retrain the model only for columns whose ridge search RMSE falls by at least 1%. Name and code columns could be one-hot encoded, or target-encoded so each adds one numeric column.
- Evidence checked: the auxiliary table has 83 columns. Ridge's best set already contained the block, COE, and group-mean columns, and none of the other 53 columns lowered that ridge search RMSE of 458.29 by 1%. XGBoost's best set was screened for 55 columns and none cleared the same ridge bar against its base. The MLP base was the ten original fields, search RMSE 896.24. Seventeen columns cleared the ridge bar. Added one at a time on the MLP, `coe_a_premium_roll3`, `coe_a_premium_lag1`, `property_basket_adj_close_lag1`, `coe_e_bids_per_quota_lag1`, and `postal_code` each cut the MLP's own search RMSE by at least 1%. `coe_b_premium_mom_change` then moved search RMSE from 508.82 to 504.28, which is below the previous snapshot but short of the next 1% cut.
- Decision: `models/<model>/best` is replaced only when the new search RMSE is lower, and a run with `--max-rows` does not replace an uncapped snapshot. `ablate` writes one JSON per column under `models/<model>/ablation`. Name and code columns are screened with target encoding; the MLP retrain still uses that model's one-hot settings. The MLP snapshot is now those ten original fields plus the five columns that cleared 1% and `coe_b_premium_mom_change`, search RMSE 504.28, confirm RMSE 553.08, in `outputs/predictions/mlp_ablation.csv`. Ridge and XGBoost snapshots are unchanged.
- What the report may say: a ridge add-one on the search months ranked auxiliary columns, and only the MLP's own later fits decided which of those columns entered the MLP. Confirm RMSE is stored and was not the acceptance signal.
- What the report must not say: that a column which failed the 1% ridge screen has been shown to be useless for every model, or that the MLP confirm RMSE of 553.08 was the reason the snapshot was kept.

### 2026-10-22 — approval month, block number, and furnished flag are not one-hot

- Options considered: one-hot all three with the other nominal columns; encode `RENT_APPROVAL_DATE` as a month index since `2021-01`, or as a calendar month plus a year; one-hot `BLOCK`, one-hot `TOWN|BLOCK|STREET`, target-encode the block number, target-encode the building key, or drop `BLOCK`; one-hot `FURNISHED`, always emit 0/1, or omit it when the fit months have a single value.
- Evidence checked: train approval months run from `2021-01` through `2025-03` (51 months), and the fit window used by the current champions ends at `2024-09`. Test months run from `2025-03` through `2026-07`. Every train row and every test row has `FURNISHED` equal to `yes`. The block table has 9,464 buildings, `TOWN|BLOCK|STREET` is unique there, and the same block number occurs in more than one town. This choice was not scored against `MONTHLY_RENT`.
- Decision: these three columns stay in the feature list and ignore the categorical gene. `RENT_APPROVAL_DATE` is months since `2021-01` and then follows numeric impute and scale. `BLOCK` is the fit-month mean rent of `TOWN|BLOCK|STREET`, shrunk toward that period's global mean with strength 20; an unseen building gets the global mean. `FURNISHED` is 1 for `yes` and 0 for `no`, and is omitted when the fit months contain only one of those values. `genome_hash` includes the token `date-month,block-building-target,furnished-binary`, so a logged one-hot run is not reused as this matrix. No model was retrained.
- What the report may say: the approval month is a count of months since January 2021, the block column is a shrunk fit-month mean rent of the town, block, and street, and a furnished flag with only one fit-month value is left out of the matrix.
- What the report must not say: that these encodings lowered search RMSE, or that the saved ridge, XGBoost, or MLP champions already use them.

### 2026-10-22 — auxiliary search restarts from the original columns

- Options considered: keep the family-based snapshots (ridge search RMSE 458.29, XGBoost 454.12) as set A; or delete those runs and retrain the ten `train.csv` columns with the current preprocessor before adding auxiliary columns one at a time.
- Evidence checked: the 458.29 and 454.12 runs were fit while approval month, block number, and furnished were still one-hot. The current preprocessor changes those three columns in code, so a later set B did not share their matrix. The cleared artifacts were `models/<model>/best`, `ckpt`, `ablation`, `docs/champion.json`, `docs/logs`, and `outputs/predictions`. Retraining `original` with the current preprocessor gave search RMSE 525.27 for ridge (`20261022T183022314474_ridge_fea0a337`, confirm 576.78), 460.05 for XGBoost (`20261022T183100953808_xgb_e28e4bcc`, confirm 464.92), and 650.42 for the MLP (`20261022T183131011278_mlp_9a5f1229`, confirm 664.81). Each of those runs is the first snapshot. `FURNISHED` is requested and omitted because every fit row is `yes`.
- Decision: those three original-column runs are the baselines. Auxiliary columns are added one at a time from `FEATURE_COLUMNS` with the same architecture and preprocess. A column is promising only when its search RMSE is at least 1% below that round's set A. Set C, the baseline plus every promising column, replaces the snapshot only when its search RMSE is lower. The old family scores and the ridge add-one screen are not evidence for these rounds.
- What the report may say: the baseline for the current preprocessor is a model trained on the original rental columns, and auxiliary columns are tested individually against that same matrix.
- What the report must not say: that 458.29 or 454.12 is the baseline for this preprocessor, or that a whole feature family was kept or dropped by the new one-column rounds.

### 2026-10-22 — one-column rounds from the original-column baselines

- Options considered: stop after the original-column baselines; or run the set-B then set-C loop on all 83 auxiliary columns for ridge, XGBoost, and the MLP.
- Evidence checked: set A search RMSE was 525.27 for ridge, 460.05 for XGBoost, and 650.42 for the MLP, all on the ten original columns with the current preprocessor. Ridge round 1 found 14 promising columns and set C reached search RMSE 462.06 (confirm 491.27). Round 2 found none. XGBoost round 1 found none, so its snapshot stays the original-column run, search RMSE 460.05 (confirm 464.92). The MLP round 1 found 71 promising columns and set C reached search RMSE 467.26 (confirm 480.34). Round 2 found none.
- Decision: ridge best is `20261022T185613865949_ridge_d97c6333`. XGBoost best is `20261022T183100953808_xgb_e28e4bcc`. MLP best is `20261022T202913337555_mlp_3f1940b8`. The column lists are in each `models/<model>/ablation/rounds.json`. Predictions for the two replaced snapshots are `outputs/predictions/ridge_ablation.csv` and `outputs/predictions/mlp_ablation.csv`.
- What the report may say: each auxiliary column was tested by itself against the original-column model, and a column entered the combined set only after a 1% search-RMSE drop. The combined set replaced the snapshot only when its own search RMSE was lower.
- What the report must not say: that the 71 MLP columns each still help once the others are present, or that confirm RMSE chose the snapshot.

### 2026-10-22 — XGBoost nominal columns are native categories

- Options considered: keep one-hot dummy columns for XGBoost; or mark nominal columns as pandas categories and let a tree partition the labels. Integer codes used as a numeric feature were rejected because a threshold split follows the arbitrary code order.
- Evidence checked: XGBoost 3.4 accepts `enable_categorical=True` with category dtype. The one-hot original-column run had search RMSE 460.05. After clearing the XGBoost snapshot, checkpoint, ablation, champion, and logs, the same ten columns with native categories scored search RMSE 459.89 and confirm RMSE 466.43 (`20261022T214026587580_xgb_332e97f1`). The 1% bar on that baseline is 455.29. The lowest single-column probe was `dist_to_cbd_km` at 459.26. No column was promising, so set C was not trained.
- Decision: XGBoost `categorical` is `native`. Ridge and the MLP stay on the shared one-hot, ordinal, target, and bin choices, and their mutation list does not include `native`. Date, block, and furnished overrides are unchanged. The XGBoost best snapshot remains the original-column native run.
- What the report may say: XGBoost splits town, street, flat type, flat model, and auxiliary names or codes as categories learned on the fit months, and an unseen level is missing.
- What the report must not say: that the one-hot search RMSE of 460.05 is the baseline for this encoding, or that a single auxiliary column lowered XGBoost search RMSE by 1%.

### 2026-10-30 — larger budgets refit the chosen columns only

- Options considered: leave prediction on the search-budget snapshot in `best/`; or refit those same columns with a wider network or more trees, write `models/<model>/final`, and point `predict` at that folder when it exists.
- Evidence checked: the column lists were copied from the current best snapshots. Ridge kept feature hash `3dd5dfce0f43` (24 columns) and genome hash `d97c6333368eb3dc`, because `alpha` stayed 1. XGBoost kept feature hash `170e95b30b54` (the ten original columns, native categories). The MLP kept feature hash `8d04a6ec2a11` (81 columns, one-hot). `best/` and `docs/champion.json` still point at `20261022T185613865949_ridge_d97c6333`, `20261022T214026587580_xgb_332e97f1`, and `20261022T202913337555_mlp_3f1940b8`. Scored against `MONTHLY_RENT` on the same time split: ridge final `20261030T103003170691_ridge_d97c6333` has search RMSE 462.0610 and confirm RMSE 491.2717, matching its best snapshot. XGBoost final `20261030T103028155670_xgb_7ca41a9e` used depth 8, learning rate 0.05, a cap of 2000 trees, and patience 50; the saved model kept boosting round 88 and its search-set score matches the log. Search RMSE is 462.1458 and confirm RMSE is 467.3685, against the depth-6 best of search 459.8922 and confirm 466.4331. MLP final `20261030T103203158442_mlp_4afa8c53` used widths 512, 256, and 128, dropout 0.1, learning rate 0.001, at most 200 epochs, and patience 15. Search RMSE is 455.3422 and confirm RMSE is 465.5154, against the search-budget best of search 467.2635 and confirm 480.3413. Each prediction file has 50,000 rows: `outputs/predictions/ridge_final.csv`, `outputs/predictions/xgb_final.csv`, and `outputs/predictions/mlp_final.csv`.
- Decision: the larger budgets are for this final fit only. `train` and `ablate` keep the small budgets: MLP widths 128 and 64 for 20 epochs with patience 3, XGBoost depth 6 and 200 trees with patience 20, and Ridge `alpha` 1. This stage does not add or drop a column and does not replace `best/`. Search RMSE selects the stopped MLP epoch and the stopped XGBoost round. Confirm RMSE is stored and does not choose that iterate. `predict` reads `final/` when that folder exists.
- What the report may say: each submitted model is a refit of the columns already stored in `best/`, with search RMSE selecting the stopped weights.
- What the report must not say: that this stage selected or dropped a column, that the larger XGBoost budget lowered search RMSE relative to the depth-6 native snapshot, or that confirm RMSE chose the stopped iterate.

### 2026-10-30 — Kaggle files from final weights and from an all-label refit

- Options considered: upload only the saved `final/` weights; or also fit that same genome on every labeled month, including the confirm months, and upload that prediction as a second file.
- Evidence checked: `submit` rewrote `outputs/predictions/ridge_final.csv`, `xgb_final.csv`, and `mlp_final.csv` from the saved final weights, and wrote `ridge_refit.csv`, `xgb_refit.csv`, and `mlp_refit.csv`. Each file has header `Id,Predicted`, ids `0` through `49999`, and 50,000 finite predictions. Feature hashes stayed `3dd5dfce0f43`, `170e95b30b54`, and `8d04a6ec2a11`. `best/`, `final/`, and `docs/champion.json` still point at the runs from before this stage. The refits are `20261030T104835277133_ridge_d97c6333` (train RMSE 473.5616, same genome hash `d97c6333368eb3dc`), `20261030T104900338798_xgb_fe7abcce` (89 trees, early stopping off, train RMSE 425.2469), and `20261030T105128475560_mlp_969afd46` (13 epochs, patience off, train RMSE 478.9942). The MLP epoch is the one recovered on the original fit/search split; that recovery was not saved as a champion. These refits use every labeled month, so they were not scored on held-out `MONTHLY_RENT`.
- Decision: each model has two Kaggle files. `*_final.csv` is the search-stopped final fit. `*_refit.csv` uses the same columns and the round or epoch that search RMSE already chose, with the preprocessor and the weights fit on all 150,000 labeled rows. This stage does not add or drop a column and does not replace `best/` or `final/`.
- What the report may say: the two files for a model differ by whether the confirm months were included in the fit.
- What the report must not say: that search RMSE or confirm RMSE chose the refit, or that this stage selected a column. The refit train RMSE is an in-sample score on all labeled rows.

### 2026-10-30 — XGBoost fit on every original and auxiliary column

- Options considered: keep the native original-column XGBoost snapshot; or fit all 10 original columns and all 83 auxiliary columns together, growing trees until search RMSE stops falling, and store that run beside the snapshot.
- Evidence checked: `use-all` fit `20261030T110148020531_xgb_643a1851` on 93 requested columns (feature hash `646e8a84f218`), native categories, depth 6, learning rate 0.05, a cap of 2000 trees, and patience 50. Search RMSE chose boosting round 98 (99 trees). The cap was not hit. Search RMSE is 467.6919 and confirm RMSE is 472.3582. The native original-column snapshot `20261022T214026587580_xgb_332e97f1` has search RMSE 459.8922. `best/` and `final/` still point at `20261022T214026587580_xgb_332e97f1` and `20261030T103028155670_xgb_7ca41a9e`. This run was scored against `MONTHLY_RENT` on the same time split. Confirm RMSE did not choose the round.
- Decision: the all-column fit stays in `models/xgb/use_all_features`. It does not replace `best/` or `final/`. Columns were fit together, not chosen one at a time.
- What the report may say: all 93 requested columns were fit together, and search RMSE chose the stopped round.
- What the report must not say: that the one-column screen selected these columns, that confirm RMSE chose the round, or that using every auxiliary column improved XGBoost. Search RMSE 467.6919 is higher than the original-column snapshot's 459.8922.

### 2026-10-30 — ridge and the MLP fit on every original and auxiliary column

- Options considered: keep each model's selected-column snapshot; or fit all 93 requested columns together under that model's search-budget architecture and store the run beside the snapshot.
- Evidence checked: both fits use feature hash `646e8a84f218` and one-hot encoding. Ridge `20261030T110814701751_ridge_2926f89f` uses `alpha` 1. Search RMSE is 459.7231 and confirm RMSE is 511.4417, against the 24-column snapshot `20261022T185613865949_ridge_d97c6333` at search RMSE 462.0610. The MLP `20261030T110903258555_mlp_fbff1ee6` uses widths 128 and 64, at most 20 epochs, and patience 3. Search RMSE kept epoch 19. The epoch cap was not hit. Search RMSE is 467.3886 and confirm RMSE is 481.6540, against the 81-column snapshot `20261022T202913337555_mlp_3f1940b8` at search RMSE 467.2635. `best/` and `final/` still point at those earlier runs. Both fits were scored against `MONTHLY_RENT` on the same time split. Confirm RMSE did not choose either fit.
- Decision: the all-column fits stay in `models/ridge/use_all_features` and `models/mlp/use_all_features`. They do not replace `best/` or `final/`. Columns were fit together, not chosen one at a time.
- What the report may say: all 93 requested columns were fit together under the search-budget architecture. Ridge search RMSE 459.7231 is lower than its 24-column snapshot.
- What the report must not say: that the one-column screen selected these columns, that confirm RMSE chose either fit, or that using every column improved the MLP. The MLP search RMSE 467.3886 is higher than its 81-column snapshot's 467.2635.

### 2026-10-30 — CatBoost follows the original, ablation, final, and submit path

- Options considered: one-hot nominal columns, as ridge and the MLP do; or pass them to CatBoost as categories. Integer codes were rejected because a threshold split would follow the arbitrary code order.
- Evidence checked: date, block, and furnished stay on the shared overrides. Other nominal columns are native categories. CatBoost rejects a missing category, so an unseen level is the string `__missing__`. The original-column run `20261030T113111382171_catboost_122f4818` scored search RMSE 460.0983 and confirm RMSE 467.6470 on `MONTHLY_RENT`. The 1% bar on that baseline is 455.50. All 83 auxiliary probes missed it. The lowest was `postal_code` at 456.5685, so set C was not trained and `best/` stayed the original columns. Final `20261030T123149326955_catboost_7ca41a9e` uses depth 8, learning rate 0.05, and at most 2000 iterations with patience 50. Search RMSE chose 367 iterations. Search RMSE is 456.0421 and confirm RMSE is 463.6901. `outputs/predictions/catboost_final.csv` and `outputs/predictions/catboost_refit.csv` each have 50,000 rows. The refit uses those 367 iterations with early stopping off and was not scored on held-out `MONTHLY_RENT`.
- Decision: CatBoost nominal columns are native categories. The feature set remains the ten original columns. The larger budget is the final fit only and does not add or drop a column. Later models follow the same order: original-column `train`, one-column `ablate`, `finalize`, then `submit`.
- What the report may say: CatBoost was trained on the original columns, no auxiliary column cleared the 1% search-RMSE bar, and search RMSE selected the stopped iteration of the larger final model.
- What the report must not say: that confirm RMSE chose the feature set, or that an auxiliary column was kept. No probe cleared 1%, and set C was not trained.

### 2026-10-30 — tree models keep any search-RMSE drop

- Options considered: keep the 1% probe bar for every model; or, for XGBoost and CatBoost only, treat a probe as promising when its logged search RMSE is strictly below set A, then train that combination once.
- Evidence checked: the round-1 probe files were not refit. Against XGBoost `20261022T214026587580_xgb_332e97f1` (search RMSE 459.8922), 20 probes were lower. Set C `20261030T124013757421_xgb_824fde31` scored search RMSE 459.5069 and confirm RMSE 466.2792, so it replaced `best/`. Against CatBoost `20261030T113111382171_catboost_122f4818` (search RMSE 460.0983), 30 probes were lower. Set C `20261030T124215167507_catboost_417a4ec8` scored search RMSE 452.6188 and confirm RMSE 462.2385, so it replaced `best/`. Both combinations were scored against `MONTHLY_RENT`. Confirm RMSE did not choose them. The old probe files still say `promising: false` because those files record the 1% screen. Final XGBoost `20261030T124257708734_xgb_933157d5` on the 30 columns scored search RMSE 461.8405 and confirm RMSE 467.6501. Final CatBoost `20261030T124934765682_catboost_84d3c3ba` on the 40 columns scored search RMSE 449.4735 and confirm RMSE 458.3408. The prediction files were rewritten from those final weights.
- Decision: ridge and the MLP still require a 1% drop. XGBoost and CatBoost accept any strictly lower search RMSE on a probe. Set C still replaces `best/` only when its own search RMSE is lower. The larger final budget does not add or drop columns.
- What the report may say: these columns were grouped because each probe was lower on its own, and each combination replaced the snapshot because its own search RMSE was lower.
- What the report must not say: that each of those columns still helps inside the combination, that confirm RMSE chose the snapshot, that any of these probes cleared the 1% bar, or that the larger XGBoost budget lowered search RMSE. The final XGBoost search RMSE 461.8405 is higher than the set C snapshot's 459.5069.

### 2026-10-30 — town and three level series for the partition-month mean

- Options considered: five regions, 26 towns, 31 planning areas, 24 postal districts, or 53 postal sectors. For the economy series, all 15 COE and stock columns, the month-to-month changes and bid ratios, every COE premium, or a short list of levels.
- Evidence checked: `python -m src.experiments.partition_month_mean` on `MONTHLY_RENT`. Town has no missing rows and a median of 110 fit-month rows per town-month; 6% of those cells have fewer than 30 rows. Postal sectors have a median of 34 and 44% of cells under 30, and 4,685 rows have no postal code. A ridge of town indicators plus a month index has weighted cell RMSE 291.65 on 2024-10..12 and 366.09 on 2025-01..03. Adding `coe_c_premium_lag1` and `bank_basket_adj_close_lag1` lowers that to 94.37 and 112.85. Adding `property_basket_adj_close_lag1` lowers it further to 93.36 and 103.31. The four COE premiums are correlated about 0.93 with each other. All 15 columns score 104.81 and 229.70, so the extra change, bid-ratio, and return columns make the next quarter worse. Month-to-month COE changes and bid ratios alone do not beat the month index.
- Decision: the partition for this check is `TOWN`. The economy columns are `coe_c_premium_lag1`, `bank_basket_adj_close_lag1`, and `property_basket_adj_close_lag1`. This is not yet a feature in the training pipeline.
- What the report may say: town-month mean rent is tracked by the town, a month index, one COE premium level, and the two basket price levels.
- What the report must not say: that COE categories A, B, and E add a separate signal beyond category C, that the month-to-month changes or bid ratios lowered the cell RMSE, or that this choice has been wired into ridge, XGBoost, CatBoost, or the MLP.

### 2026-10-30 — estimators for the town-month mean

- Options considered: the unweighted ridge already scored (`alpha` 1); other linear fits (ordinary least squares, Huber, elastic net, a row-weighted ridge, and a ridge with a separate month slope per town); XGBoost; CatBoost. Depth and iteration count were a fixed preset list. No preset used the confirm quarter to stop.
- Evidence checked: `python -m src.experiments.town_month_models` on the town-month mean of `MONTHLY_RENT`. Features stay `TOWN`, `month_index`, `coe_c_premium_lag1`, `bank_basket_adj_close_lag1`, and `property_basket_adj_close_lag1`. Fit months end at 2024-09 (1,170 cells). The unweighted ridge reproduces weighted cell RMSE 93.36 on 2024-10..12 and 103.31 on 2025-01..03. The row-weighted ridge scores 90.63 and 95.17. CatBoost depth 3, learning rate 0.05, 800 iterations scores 84.10 and 77.66. The same CatBoost at 2,000 iterations scores 81.77 and 99.09. XGBoost depth 3, 300 trees, weighted by cell size, scores 90.63 and 84.76; at 2,000 trees its search RMSE rises to 106.87. A separate month slope per town scores 102.30 and 111.98. Months 2025-04 through 2026-07 have economy columns and no `MONTHLY_RENT`. On that unlabeled path the ridge's cross-town mean goes from 3175 to 3609, and the 800-iteration CatBoost goes from 3130 to 3031.
- Decision: on the two scored quarters, CatBoost depth 3 with 800 iterations has the lower cell RMSE that still holds up on the next quarter. Two thousand iterations lower the search quarter and raise the next quarter, so that preset is not the level model. The unlabeled path through 2026-07 is not a reason to replace the ridge, because the tree no longer follows the month index and there is no rent to score. Neither estimator is in the training pipeline.
- What the report may say: with the same town and three level series, an 800-iteration depth-3 CatBoost has weighted cell RMSE 84.10 and 77.66, below the ridge's 93.36 and 103.31. XGBoost and a town-specific month slope do not beat that pair of scores.
- What the report must not say: that 2,000 iterations is the better CatBoost because its search RMSE is 81.77, that the 2025-04..2026-07 path was scored against rent, or that this estimator has been wired into ridge, XGBoost, CatBoost, or the MLP.

### 2026-10-30 — town-month ridge plus a residual from each best genome

- Options considered: keep predicting `MONTHLY_RENT` with the four raw-rent snapshots; or predict it as an unweighted town-month ridge plus a residual. The residual models considered are the four best genomes copied as they stand. A new ablation, a larger final budget, and an all-label refit were not used.
- Evidence checked: `python -m mean_residual_model.run` on `MONTHLY_RENT`. The ridge is fit on 1,170 town-month cells through 2024-09 with `TOWN`, `month_index`, `coe_c_premium_lag1`, `bank_basket_adj_close_lag1`, and `property_basket_adj_close_lag1`. Weighted cell RMSE is 93.3575 on 2024-10..12 and 103.3144 on 2025-01..03. Giving every flat that cell prediction leaves row RMSE 612.9319 and 630.4496. Each residual model copies `models/<model>/best/genome.json` and is fit to `MONTHLY_RENT` minus that prediction. `BLOCK` uses the residual. Search stops XGBoost at boosting round 63 and the MLP at epoch 10. CatBoost reaches iteration 199, the last step of its 200-iteration cap. Sum RMSE on the search quarter is ridge 461.1604, XGBoost 454.4929, MLP 461.7834, and CatBoost 451.2565. Confirm RMSE is 487.5892, 465.6292, 483.0766, and 466.2242. The raw-rent best snapshots remain at search RMSE 462.0610, 459.5069, 467.2635, and 452.6188. `models/<model>/best` and `final` were not replaced. Test files `outputs/predictions/mean_residual_<model>.csv` use the search-stopped weights and were not scored on held-out rent.
- Decision: the rent prediction in `mean_residual_model` is the town-month ridge plus the residual from that model's copied best genome. This does not replace the raw-rent snapshots.
- What the report may say: the town-month ridge reproduces the cell scores above, and each residual sum has a lower search RMSE than the raw-rent best snapshot of the same genome. Search RMSE stopped the XGBoost round and the MLP epoch.
- What the report must not say: that confirm RMSE chose a model, that `best/` or `final/` changed, that the test files refit 2025-01..03, or that CatBoost stopped before its 200-iteration cap.

### 2026-10-30 — one residual model per town

- Options considered: one residual model for all towns, already fit from each best genome; or 26 residual models, one fit on each town's own rows with that same genome. The town-month ridge stays shared.
- Evidence checked: `python -m mean_residual_model.run_by_town` on `MONTHLY_RENT`. Every town appears in the fit, search, and confirm months. Bukit Timah has 393 fit rows, 23 search rows, and 22 confirm rows. Sum RMSE for the 26-town residuals is ridge search 462.5479 and confirm 489.6274, XGBoost 458.9201 and 473.8054, MLP 494.2216 and 508.3543, CatBoost 450.4965 and 468.2633. The single residual models score search 461.1604, 454.4929, 461.7834, and 451.2565, with confirm 487.5892, 465.6292, 483.0766, and 466.2242. Confirm RMSE did not choose the split. `outputs/predictions/mean_residual_<model>.csv` still comes from the single residual model. The 26-town predictions are `outputs/predictions/mean_residual_by_town_<model>.csv`.
- Decision: the residual stays one model per algorithm. Ridge, XGBoost, and the MLP have higher search RMSE when split by town. CatBoost's 26-town search RMSE is lower by 0.76 and its confirm RMSE is higher by about 2, so that split does not replace the single CatBoost residual.
- What the report may say: separate town residual models were scored on the same genomes, and the single residual model has the lower search RMSE for ridge, XGBoost, and the MLP.
- What the report must not say: that the 26-town split replaced `mean_residual_<model>.csv`, or that confirm RMSE chose the single CatBoost residual. CatBoost's 26-town search RMSE is 450.4965.
