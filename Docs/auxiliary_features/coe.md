# COE features

Implemented in [src/features/coe.py](../../src/features/coe.py). Categories A, B, and E are P1. Category C is P2. The priority was not tested against `MONTHLY_RENT`.

Every flat approved in the same month receives the same value. COE is a vehicle-quota price, not a wage or a consumer-price index.

## Motivation

The rental file has a year-month, and the COE file has a price for that month. Last month's tender is a dated measure of credit and vehicle demand that block features cannot show. The change and the bids-per-quota ratio describe the jump and the pressure, not only the level.

## Implementation

1. Parse `QUOTA_PREMIUM`, `BIDS_RECEIVED`, and `QUOTA` by removing `$` and commas. A failed parse raises; it is not stored as 0.
2. Map `jan`–`dec` to months 1–12.
3. For each category and month, use `ROUND == 2`. If round 2 is absent, use round 1.
4. For a rental in month `T`, read month `T-1`. Do not use `PREVAILING_QUOTA_PREMIUM`, and do not fill a missing month with the rental month's own price.
5. Do not read `MONTHLY_RENT`.

## Formula

```text
coe_{cat}_premium_lag1        = premium(cat, T-1)
coe_{cat}_premium_mom_change  = premium(cat, T-1) - premium(cat, T-2)
coe_{cat}_bids_per_quota_lag1 = bids(cat, T-1) / quota(cat, T-1)
coe_a_premium_roll3           = mean(premium(a, T-1), premium(a, T-2), premium(a, T-3))
```

`{cat}` for the first three formulas is `a`, `b`, or `e`. The ratio is null when quota is 0 or the month is missing. The three-month mean is null unless all three months exist. Only category `a` has that mean. Category `c` has only `coe_c_premium_lag1`.

## Why this fits the tables

The file is a monthly panel from 2021 through mid-2026, with up to two rounds per category. June 2026 has two rounds for A, B, C, and E, so the July 2026 test lag is covered. December 2020 is not in the file, so the lag rule leaves January 2021 empty instead of borrowing January's price. The approval field is only `YYYY-MM`, so attaching the same month's round 2 would assume a publication date the table does not show.

## Tradeoffs

The lag is the same rule for train and test, and round 2 is the later tender in an already finished month. The series cannot separate two flats approved together. A single tender can move the dollar change by thousands. Averaging both rounds, or building the full change-and-ratio family for category C, was rejected. Category C is commercial vehicles and stays one column.

## Under-considered points

Quota is a policy choice, so the ratio can jump because the quota shrank. A negative month-on-month change is a price fall and is kept. The three-month window can hide a policy break. None of these columns is household income.

## Columns

| Column | Priority | Null train / test rows |
|---|---|---|
| `coe_a_premium_lag1`, `coe_b_premium_lag1`, `coe_e_premium_lag1`, `coe_c_premium_lag1` | P1, except `coe_c_premium_lag1` at P2 | 3,129 / 0. The 3,129 rows are approval month `2021-01` |
| `coe_a_premium_mom_change`, `coe_b_premium_mom_change`, `coe_e_premium_mom_change` | P1 | 5,809 / 0. Those rows are `2021-01` (3,129) and `2021-02` (2,680) |
| `coe_a_bids_per_quota_lag1`, `coe_b_bids_per_quota_lag1`, `coe_e_bids_per_quota_lag1` | P1 | 3,129 / 0, same gap as the lagged premium |
| `coe_a_premium_roll3` | P1 | 9,514 / 0. Those rows are `2021-01` through `2021-03` (3,129 + 2,680 + 3,705) |

The raw COE fields have 0 nulls. These nulls come from the lag, not from blank cells. Do not average the months that happen to exist.
