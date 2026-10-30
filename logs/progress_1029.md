# Progress on 2026-10-30

Search RMSE is the selection number below. Confirm RMSE is recorded and did not choose a model. Fit months end at 2024-09. Search is 2024-10 through 2024-12. Confirm is 2025-01 through 2025-03.

## Town-month mean

Source: `mean_residual_model/reginal_mean_model/trials/metrics.json`. The scored target is the town-month mean of `MONTHLY_RENT`. The unweighted ridge uses the town, the month index, `coe_c_premium_lag1`, `bank_basket_adj_close_lag1`, and `property_basket_adj_close_lag1`.

| Model | Search cell RMSE | Confirm cell RMSE | Forward mean, 2025-04 to 2026-07 |
| --- | ---: | ---: | --- |
| Ridge, alpha 1 | 93.36 | 103.31 | 3175 to 3609 |
| Ridge, town-specific slopes | 93.90 | 113.51 | 3175 to 3608 |
| Ridge, degree 2 | 123.40 | 197.02 | 2869 to -293 |
| MLP 64 | 178.07 | 191.45 | 3258 to 3597 |
| MLP 64-32 | 179.78 | 189.47 | 3241 to 3546 |
| MLP 128-64-32 | 184.37 | 197.27 | 3247 to 3582 |
| RNN, hidden 16, 1 layer | 93.25 | 88.04 | 3192 to 3274 |
| LSTM, hidden 16, 1 layer | 92.85 | 113.88 | 3091 to 2982 |
| LSTM, hidden 16, 2 layers | 89.00 | 86.39 | 3144 to 3048 |
| LSTM, hidden 16, 3 layers, 4 inputs | 88.75 | 83.39 | 3132 to 3109 |
| LSTM, hidden 16, 3 layers, 16 inputs | 87.98 | 85.26 | 3136 to 3104 |

The 4-input three-layer LSTM is the region mean used by the later residual stack. Its unlabeled path still falls, so it did not replace the ridge file used by `mean_residual_<model>.csv`. The 16-input network has a lower search cell RMSE and a higher confirm cell RMSE.

## Residuals on the ridge mean

Giving every flat the ridge cell prediction leaves search row RMSE 612.93 and confirm 630.45.

Shared residual, copied from each `models/<model>/best` genome:

| Model | Search | Confirm |
| --- | ---: | ---: |
| Ridge | 461.16 | 487.59 |
| XGBoost | 454.49 | 465.63 |
| MLP | 461.78 | 483.08 |
| CatBoost | 451.26 | 466.22 |

Twenty-six separate residual models on the same genomes and the same ridge mean: ridge 462.55 / 489.63, XGBoost 458.92 / 473.81, MLP 494.22 / 508.35, CatBoost 450.50 / 468.26. The single residual stays in `mean_residual_<model>.csv`. CatBoost's split search RMSE is lower and its confirm RMSE is higher.

## Residuals on the 4-input three-layer LSTM

The LSTM alone, given to every flat, has search row RMSE 612.25 and confirm 627.49.

Shared residual:

| Model | Search | Confirm |
| --- | ---: | ---: |
| Ridge | 459.80 | 470.97 |
| XGBoost | 454.63 | 462.04 |
| MLP | 460.28 | 469.62 |
| CatBoost | 451.71 | 460.31 |

One larger model per town on within-town flat and amenity columns:

| Model | Search | Confirm |
| --- | ---: | ---: |
| Ridge | 457.52 | 469.96 |
| XGBoost | 461.48 | 469.88 |
| MLP | 456.71 | 470.33 |
| CatBoost | 451.26 | 461.43 |

## Location gap

Source: `outputs/feasibility/location_premium.json` and `location_surface.json`. These scores are row RMSE of `MONTHLY_RENT` after adding a location rule to the saved LSTM mean. They are not the stack.

Across 76 components, the fit-month dollar gap and the search-quarter gap correlate 0.890 (mean absolute error 69.1). The ratio correlates 0.878 (mean absolute error 0.0227).

| Rule | Search | Confirm |
| --- | ---: | ---: |
| Constant dollar gap | 598.14 | 614.15 |
| Constant ratio | 597.80 | 613.66 |
| Stored time slope | 596.89 | 614.25 |
| Shrunk gap | 598.28 | 614.23 |
| Shrunk ratio | 597.88 | 613.69 |
| Own-town 3D surface, start at 3 | 578.11 | 593.19 |
| Density blend across towns | 579.76 | 594.42 |

Starting the same own-town 3D surface at 10, and dropping a component count when a hard assignment has fewer than 30 fit rows, scores search 566.55 and confirm 580.58 (`location_surface_k10.json`). The current saved mixtures in the stack start at 20 and use that same drop rule. The saved `location_surface.json` does not contain a 300m or 500m edge-blend score.

## Stack

On 2026-10-30, `mean_residual_model/multi_stage_model(lstm+catboost+mlp)/metrics.json` recorded search RMSE 451.80, confirm 462.82, stopped at epoch 36. The region mean was the saved 4-input three-layer LSTM. CatBoost was the frozen full model for each town. The Gaussian feature was the own-town 3D conditional residual. The MLP widths were 128 and 64. The prediction was the region mean plus the MLP output. That `metrics.json` was overwritten on 2026-10-31. The three numbers above are not in the current file. The continuation is `logs/progress_1031.md`.

Reloading those weights, without refitting, reproduced the stored search RMSE exactly (451.803206). A full CatBoost refit started later was killed before Hougang finished. That run is not a result.

`mlp_deep/metrics.json` records a wider MLP, widths 512-256-128-64, search 448.88 and confirm 461.87, stopped at epoch 29. It is not the stack prediction.

## Maps

`src/viz/gaussian_towns/` has one figure per town. The surface is the predicted gap from the region mean. December 2024 flats are coloured by rent minus that town's true December mean. Both means are printed on the figure. The figures were not scored against `MONTHLY_RENT`.
