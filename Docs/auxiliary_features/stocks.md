# Stock-basket features

Implemented in [src/features/stocks.py](../../src/features/stocks.py). All four columns are P2. The priority was not tested against `MONTHLY_RENT`.

Every flat approved in the same month receives the same value. The baskets are equity proxies, not HDB rents.

## Motivation

Block features do not move with the approval month. A short, fixed set of local shares gives a lagged market level and a lagged return. Banks stand in for credit conditions. The property names stand in for listed property companies. They are separate baskets because the businesses differ.

## Implementation

1. Read `Data/auxiliary/sg-stock-prices.csv`. Drop rows whose `ADJUSTED_CLOSE` is null. The file has 85 such rows overall.
2. For each symbol and calendar month, keep the adjusted close on the latest date in that month.
3. A basket month is valid only when all three symbols have that close. Do not average the symbols that remain.
4. For rental month `T`, use month `T-1`.
5. Do not read `MONTHLY_RENT`.

Symbols:

| Basket | Symbols |
|---|---|
| Banks | `D05.SI` DBS, `O39.SI` OCBC, `U11.SI` UOB |
| Property | `C09.SI` City Developments, `U14.SI` UOL, `9CI.SI` CapitaLand Investment |

## Formula

```text
month_return(symbol, M) = month_end_close(M) / month_end_close(M-1) - 1
{name}_basket_adj_close_lag1 = mean of the three month-end closes in month T-1
{name}_basket_return_lag1    = mean of the three month returns in month T-1
```

`{name}` is `bank` or `property`. The return is null unless all three closes exist in both `T-1` and `T-2`. Negative returns are kept.

## Why this fits the tables

The file has 78 symbols, including short overseas listings. It is not the Straits Times Index. The three bank symbols have a month-end close for every month from `2021-01` through `2026-07`. `9CI.SI` has no month-end close from `2021-01` through `2021-08`, so the property basket is missing there instead of quietly becoming a two-name average. History starts in January 2021, so a lag into 2020 is null.

## Tradeoffs

Membership is fixed, equal-weighted, and uses adjusted closes. One number per month cannot separate flats. Replacing `9CI.SI` with the mall trust `C38U.SI`, or building one column per symbol, was rejected. Using the rental month's own unfinished closes was rejected for the same reason as the COE same-month tender.

## Under-considered points

A bank-specific or developer-specific event can still move the three-name mean. The month-end print can be the last session of a short month. These series are not inflation, wages, or a residential price index. Do not fill a missing close with 0, which would invent a huge return. Do not copy the bank return into the early property gap.

## Columns

| Column | Null train / test rows | Why those rows are null |
|---|---|---|
| `bank_basket_adj_close_lag1` | 3,129 / 0 | Approval month `2021-01` looks up `2020-12` |
| `bank_basket_return_lag1` | 5,809 / 0 | `2021-01` and `2021-02`. The first return the file can form is February 2021 |
| `property_basket_adj_close_lag1` | 28,466 / 0 | `9CI.SI` is missing through `2021-08`, and `2021-01` also looks up 2020. Null approvals run through `2021-09` |
| `property_basket_return_lag1` | 31,565 / 0 | The first complete three-name return is `2021-10`, so the lag stays null through approval month `2021-10` |

Test months start at `2025-03`, after both gaps, so all four test columns have 0 nulls.
