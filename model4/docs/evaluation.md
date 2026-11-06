# Evaluation contracts

The main unit is individual monthly rent in SGD. Area-normalised or residual
predictions are converted back before scoring. RMSE is the square root of mean
squared prediction-minus-truth error; bias is prediction minus truth. Monthly
mean error and individual-rental error are different measurements.

## Fixed split

Fit: 2021-01..2024-09 (132802 rows). Validation: 2024-10..2024-12 (8274 rows).
Confirmation: 2025-01..2025-03 (8924 rows), repeatedly viewed during development.
Reported changes are cumulative, not an independent factorial ablation. Missing
scores stay null and failed experiments stay failed.

## Three-origin local protocol

Origins: 2023-05, 2022-03, 2022-10. The preceding three history months select a
global CatBoost tree count. Model and index are refit through the origin, then
scored on all prescribed records in the next 16 months. Outer rent labels never
select features, capacities or stopping sizes.

The index uses within-property log-rent month effects, a six-month slope and
0.9 damping. The global model learns rent/index with normalised squared-index
weights. Statistics, ranks and index estimates use each phase's permitted
history. Future macro columns hold the last finite permitted lag1 value without
a second lag.

Equal-origin RMSE takes the square root of mean origin MSE. Pooled metrics
combine all records, retaining overlap. Overlap does not provide independent
market cycles.

## Auxiliary diagnostics

The descriptive rental series chains adjacent-month median rent/sqm changes in
town/flat-type cells with at least five observations in both months. Equal cell
weights partly limit composition changes; this is not a fully quality-adjusted
official rental index.

Lagged three-month signal momentum is compared with future six/twelve-month
rent growth. The common six-month analysis has 33 overlapping monthly windows.
Partial correlations control only one linear rental-momentum statistic, not
causality or forecast gains. At origin 2022-03 the property basket has no matured
six-month relation; full-period associations were not coefficients known then.
