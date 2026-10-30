# Group-mean rent features

Implemented in [src/features/group_means.py](../../src/features/group_means.py). These six columns are target encodings. They were not used to screen features against `MONTHLY_RENT`, and they are not attributes of a mall, station, or school.

## Motivation

Flats that share a postal area, or that share the same nearest mall, station, or school, sit in the same local rental market. The mean of rents from the previous quarter in that group tracks the recent market. A mean over all earlier years would mix 2021 rents into a 2025 row. The postal sector and postal district are used only as grouping keys. They are not stored as columns.

## Implementation

1. Build the block and nearest-facility columns first.
2. Postal sector is the first two digits of `postal_code`, after restoring a dropped leading zero to six digits. An empty postal code has no sector and no district.
3. Postal district maps those two digits onto D01–D28. For example, 01–06 are D01, 22–23 are D09, and 51–52 are D18.
4. For a row in approval month T, use `MONTHLY_RENT` from training rows in T-1, T-2, and T-3 only. Test rows never contribute rents. Month T does not count. Months before that window do not count.
5. Shrink the group mean toward the island-wide mean of those same three months.

## Formula

```text
smoothed = (n * group_mean + 20 * global_mean) / (n + 20)
```

`n` is the number of training rows in the group inside that three-month window. `global_mean` is the mean of all training rows in the same window. If `n` is 0, the value is `global_mean`. If month T is `2021-01`, none of T-1, T-2, or T-3 exist in the file, so the value stays null. February 2021 uses January only, and March 2021 uses January and February. It is not filled with a later island-wide mean, and it is not filled with rents from before the window.

## Why this fits the available data

The feature table already has `nearest_mall_name`, `nearest_open_mrt_code`, `nearest_primary_name`, and `nearest_secondary_name`. The station code is unique; the station name is not, so the MRT mean uses the code. Primary and secondary schools stay separate. The rental month is on every source row, so the lag can be applied to both train and test. Test labels are not in the file, so the history is always the training set.

## Tradeoffs

The three-month window keeps a rent out of its own feature and drops rent levels from much earlier years. Shrinkage stops a handful of rents in a thin quarter from becoming the whole group mean. The cost is that `2021-01` is null, rare malls and schools often have a small `n` inside one quarter, and those rows fall back toward that quarter's island-wide mean. Using every training rent before T was rejected because 2021 rents would still pull down a 2025 mean.

## Under-considered points

The mall mean is the mean of rentals for which that mall is the nearest mall, not the mean of every flat inside a 1 km radius. A flat can be near two malls and still count for only one. The same is true for stations and schools. Postal district boundaries are the legacy 28-district map, not URA planning areas. The window is three calendar months, not a named fiscal quarter. Shrinkage strength 20 is a fixed choice, not a tuned hyperparameter.

## Columns

| Column | Group key | Nulls |
|---|---|---|
| `mean_rent_postal_sector` | First two postal digits | `2021-01` is null. Empty postal codes use that quarter's island-wide mean once the window contains any training rent |
| `mean_rent_postal_district` | D01–D28 from those digits | Same rule. An unmapped sector has no district group |
| `mean_rent_nearest_mall` | `nearest_mall_name` | `2021-01` is null |
| `mean_rent_nearest_mrt` | `nearest_open_mrt_code` | `2021-01` is null |
| `mean_rent_nearest_primary` | `nearest_primary_name` | `2021-01` is null |
| `mean_rent_nearest_secondary` | `nearest_secondary_name` | `2021-01` is null |

## Figures

`src/viz/plot_group_means.py` writes `outputs/figures/heatmap_*.png`. Each cell is the formula for that group and approval month. The cell is drawn even when that month has no rental whose nearest facility is the group. It is blank only when `T-1`, `T-2`, and `T-3` contain no training rents, which is the whole `2021-01` column.

`src/viz/plot_group_mean_maps.py` writes `outputs/figures/maps/`. For every June and December whose three-month window overlaps the training months, it draws three maps: nearest mall, nearest primary school, and nearest open MRT station. The background is an OpenStreetMap tile mosaic. Eight landmarks are marked from the station table: Raffles Place, Orchard, Jurong East, Woodlands, Tampines, Changi Airport, HarbourFront, and Punggol. Each land cell takes the colour of its single nearest facility, so one facility's service area is one colour. Each map uses its own minimum and maximum rent for the colour bar. Colours cannot be compared across months or across the three facility types. December 2025 is omitted because September--November 2025 are outside the training rents. The school maps use primary schools, because primary and secondary means are separate columns.
