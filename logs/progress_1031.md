# Progress on 2026-10-31

Search RMSE is the selection number below. Confirm RMSE is recorded and did not choose a model or an epoch. Fit months end at 2024-09. Search is 2024-10 through 2024-12. Confirm is 2025-01 through 2025-03. Seed is 5228.

`logs/progress_1029.md` records the dollar town-month trials, the residuals on the ridge mean and on the 4-input three-layer LSTM, and the location-gap checks. Its stack paragraph records search RMSE 451.80, confirm 462.82, and epoch 36. That `metrics.json` was later overwritten, so those three numbers are not in the current file. A CatBoost refit killed before Hougang finished is not a result.

Choices among these setups are also in `Docs/design_tradeoffs.md`. A later entry supersedes an earlier one when they describe the current file.

## Source columns

`src/features/source_fields.py` adds eleven derived columns to the auxiliary tables. The tables now have 94 columns. Raw `Data/train.csv` and `Data/test.csv` were not overwritten. `MONTHLY_RENT` is not a feature.

Dates become an approval-month index or a commencement year. Town, block, and street are normalized names, not price ranks. Flat type is the size rank 1 through 6. Flat model keeps the fixed rank 0 through 19 and the normalized name `flat_model_key`. Area and fee stay floats. Furnished becomes 0 or 1. `FURNISHED` is `yes` on every row and `FEE` is 0. CatBoost can split on `flat_model_key` as a category. The mixture and the MLP use `flat_model_ord`.

`models/<model>/best` and `final` were not refit on these columns.

## Region mean after the dollar trials

The dollar table is in `logs/progress_1029.md`. The rows below are the later fits. A per-square-metre cell RMSE is not comparable to a dollar cell RMSE.

| Model | Target | Search cell RMSE | Confirm cell RMSE | Forward mean, 2025-04 to 2026-07 | Weights |
| --- | --- | ---: | ---: | --- | --- |
| LSTM, hidden 16, 3 layers, linear bypass | town-month mean of monthly rent | 170.56 | 261.29 | 3339 to 3821 | overwritten |
| LSTM, hidden 16, 3 layers, linear bypass | town-month mean of rent per square metre | 1.9646 | 3.4044 | 38.17 to 43.30 | `reginal_mean_model/lstm/h16_l3_linear/` |
| RNN, hidden 16, 3 layers | town-month mean of rent per square metre | 0.9707 | 1.0897 | 35.99 to 35.79 | `reginal_mean_model/rnn/h16_l3/` |

The dollar bypass fit-month cell RMSE was 83.73. Its search cell RMSE is higher than the saved 4-input LSTM's 88.75. Those dollar weights were then overwritten by the per-square-metre refit in the same folder. The dollar scores remain in `Docs/design_tradeoffs.md`.

The current region mean is the three-layer RNN. It reads the month index and the three lag-1 economy levels. It does not read the previous month's predicted or true rent. Hidden state is carried by running the sequence from 2021. `rnn/h16_l1` and `lstm/h16_l3` were not overwritten. The ridge file used by `mean_residual_<model>.csv` was not replaced.

Public HDB approved rents for 2025-04 through 2026-07 were an external check only. They were not training labels. The unweighted mean of town means moves from about 3179 to about 3280. Among the dollar means, the one-layer RNN (3192 to 3274) was the closest path. The downloaded check file was deleted.

## Town CatBoost and mixture on the RNN gap

`python "mean_residual_model/multi_stage_model(lstm+catboost+mlp)/run.py" --residuals-only` fits both models to `MONTHLY_RENT / floor_area_sqm` minus the RNN town-month unit mean. They do not see each other's output. The RNN was not refit. This pass did not fit the MLP.

Source: `mean_residual_model/multi_stage_model(lstm+catboost+mlp)/residuals_metrics.json`.

| Model | Search unit RMSE | Confirm unit RMSE | Search rent RMSE | Confirm rent RMSE |
| --- | ---: | ---: | ---: | ---: |
| CatBoost | 5.3811 | 5.4562 | 455.06 | 469.19 |
| Mixture | 5.7707 | 5.8380 | 490.02 | 502.48 |

Rent RMSE adds the gap back to the RNN unit mean and multiplies by floor area. Each town's mixture starts at 20 full-covariance components and drops the count by one when a hard-assigned component has fewer than 30 fit rows. The kept counts run from 3 to 19 and are listed in that file. The features are scaled latitude, scaled longitude, town-standardized flat-type and flat-model ranks, and the per-square-metre residual. This is not the earlier 3D feasibility surface in `outputs/feasibility/`.

## MLP

`python "mean_residual_model/multi_stage_model(lstm+catboost+mlp)/run.py" --mlp-only` trains only the MLP. The RNN, CatBoost, and mixture weights stay as saved.

The network is Linear(19, 128), ReLU, Dropout 0.1, Linear(128, 64), ReLU, Dropout 0.1, Linear(64, 1). Adam learning rate is 0.001, the batch size is 1024, and patience is 10. The target is z-scored `MONTHLY_RENT` on the fit months.

Three inputs are converted to dollars before the network: the RNN unit mean, the CatBoost unit gap, and the mixture unit gap, each multiplied by `FLOOR_AREA_SQM`. Floor area is also its own input. The other inputs are fee, lease, age, month index, latitude, longitude, flat-type rank, flat-model rank, maximum floor, year completed, flat age, and the four amenity distances. The network output is monthly rent. Nothing is multiplied by area after the network.

Source: `mean_residual_model/multi_stage_model(lstm+catboost+mlp)/metrics.json`.

| Split | RMSE |
| --- | ---: |
| Train | 460.70 |
| Search | 451.53 |
| Confirm | 468.74 |

Search RMSE stopped the run at epoch 8. Fit-row CatBoost features are the saved full model's predictions on its own fit rows, because the out-of-fold vectors were not stored. Search, confirm, and test CatBoost features are that same full model applied to those rows.

The test file is `outputs/predictions/mean_residual_multistage.csv`. It has 50,000 rows, header `Id,Predicted`, and ids 0 through 49999. Predicted rent runs from 1522.37 to 6322.51, with mean 3106.94. There are no missing values.

## Scores that are not the current test file

The linear-bypass per-square-metre LSTM, with a 128-64 MLP that predicted a unit price and then multiplied by area, had search rent RMSE 464.07 and confirm 504.02, stopped at epoch 11. Those two numbers are in `Docs/design_tradeoffs.md`. The stack `metrics.json` no longer contains them.

`mlp_deep/metrics.json` is a wider network, widths 512-256-128-64, dropout 0.3, patience 40, on the dollar 4-input LSTM mean. Search RMSE is 448.88 and confirm is 461.87, stopped at epoch 29. It is not the stack prediction.

The town heatmaps in `src/viz/gaussian_towns/` were drawn for the earlier 3D conditional mixture. They were not redrawn for the per-square-metre mixture, and `src/viz/plot_gaussian_towns.py` still expects that earlier kind.

## Current prediction

The file `outputs/predictions/mean_residual_multistage.csv` is the three-layer per-square-metre RNN, plus the town CatBoost gap and the town mixture gap on that same unit residual, passed through the 128-64 MLP after those three terms have been multiplied by floor area. `outputs/predictions/mean_residual_<model>.csv` still comes from the earlier ridge mean plus one shared residual. Production `best` and `final` snapshots were not replaced.
