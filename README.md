# CS5228 HDB monthly rent

[中文](README.zh.md)

Predict `MONTHLY_RENT` for Singapore HDB flats. The metric is RMSE. Run every command from the project root after installing `requirements.txt`. Seed is `5228`.

Column formulas live in [Docs/auxiliary_new_features.md](Docs/auxiliary_new_features.md). The command reference lives in [Docs/training_pipeline.md](Docs/training_pipeline.md). Choices among alternatives are logged in [Docs/design_tradeoffs.md](Docs/design_tradeoffs.md).

```mermaid
flowchart LR
  features[Feature files]
  train[Train original columns]
  ablate[Ablate auxiliary columns]
  finalize[Larger budget]
  submit[Test predictions]
  features --> train --> ablate --> finalize --> submit
```

A model added under `models/<model>/` is finished only after these four commands, in order: `train`, `ablate`, `finalize`, `submit`. `--model` is `ridge`, `xgb`, `mlp`, or `catboost`.

## 1. Feature preparation

```text
python -m src.features.build_features
```

This reads `Data/train.csv`, `Data/test.csv`, and the source tables under `Data/auxiliary/`. It writes `Data/auxiliary_features/train_features.csv` and `test_features.csv`, plus the parquet copies, in the same row order as the rental files. The auxiliary files stay unmerged on disk. Training joins them later by row order.

The auxiliary table has 83 columns. `MONTHLY_RENT` is not one of them. The six group-mean columns use training rents from the previous three approval months only.

## 2. Initial models

```text
python -m src.pipeline.run train --model <model> --features original
```

`original` is the ten columns in `train.csv`: `RENT_APPROVAL_DATE`, `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `FURNISHED`, `LEASE_COMMENCE_DATE`, and `FEE`. The command fits those columns and writes the first `models/<model>/best/` snapshot. A later run replaces that snapshot only when its search RMSE is strictly lower.

### 2.1 How a model is scored

`--val-months 3` cuts the labeled months by approval date.

| Period | Months | Role |
|---|---|---|
| Fit | through 2024-09 | Fit the preprocessor and the weights |
| Search | 2024-10 through 2024-12 | Stop trees and the MLP, accept a feature set, and replace `best/` |
| Confirm | 2025-01 through 2025-03 | Recorded once after search has stopped |

The preprocessor is fit on the fit months only. Search RMSE replaces `best/` only when the new score is strictly lower. Confirm RMSE is written after that and is not an acceptance signal. A run that used `--max-rows` stride-samples each period. It is not a result about the full training table.

## 3. Ablation

```text
python -m src.pipeline.run ablate --model <model>
```

Ablation starts from the columns already in `models/<model>/best`. It probes each auxiliary column that is missing from that snapshot. It does not start from all 83 columns.

Ridge and the MLP call a column promising only when its search RMSE is at least 1% below that round's set A. XGBoost and CatBoost call it promising when the search RMSE is strictly lower. Those probes do not change `best/`.

Set C is the current columns plus every promising column, trained once. Set C replaces `best/` only when its own search RMSE is lower. The loop stops when a round has no promising column or set C does not win.

## 4. Larger budget

```text
python -m src.pipeline.run finalize --model <model>
```

Finalize keeps the columns already in `best/` and writes `models/<model>/final/`. It does not add or drop columns. Search RMSE selects the stopped iterate.

| Model | Budget |
|---|---|
| Ridge | `alpha` 1 |
| MLP | widths 512, 256, and 128; at most 200 epochs; patience 15 |
| XGBoost | depth 8; learning rate 0.05; at most 2000 trees; patience 50 |
| CatBoost | depth 8; learning rate 0.05; at most 2000 iterations; patience 50 |

## 5. Test prediction

```text
python -m src.pipeline.run submit --model <model>
```

Submit joins `Data/test.csv` to `Data/auxiliary_features/test_features.csv` by row order. It writes two files:

- `outputs/predictions/<model>_final.csv` from the saved `final/` weights
- `outputs/predictions/<model>_refit.csv` after fitting every labeled month with the stopped size frozen and early stopping off

The header is `Id,Predicted`. Ids run from `0` through `49999`, and each file has 50,000 rows. The refit is not scored on held-out rent. `predict` reads `final/` when that folder exists.

`evolve` and `use-all` are extra commands. They do not replace the train, ablate, finalize, submit sequence. `use-all` fits every original and auxiliary column into `models/<model>/use_all_features` and does not replace `best/` or `final/`.

## Current progress: town-month mean plus residual

[mean_residual_model](mean_residual_model/README.md) is a separate experiment. It does not replace `models/<model>/best` or `final/`.

The rent of one flat is a town-month mean plus a residual. The mean is a ridge (`alpha` 1) of the average `MONTHLY_RENT` in each `TOWN` and approval month, fit on cells through 2024-09. Its inputs are the town, the month index since 2021-01, `coe_c_premium_lag1`, `bank_basket_adj_close_lag1`, and `property_basket_adj_close_lag1`. Weighted cell RMSE is 93.36 on the search quarter and 103.31 on the confirm quarter.

Ridge, XGBoost, the MLP, and CatBoost each copy `models/<model>/best/genome.json` and fit `MONTHLY_RENT` minus that mean. The sum's search RMSE is 461.16, 454.49, 461.78, and 451.26. Test files are `outputs/predictions/mean_residual_<model>.csv`. Those weights are fit through 2024-09. Labeled months 2025-01 through 2025-03 are not refit into them.

```text
python -m mean_residual_model.run
python -m mean_residual_model.run_by_town
```

The second command fits the same genomes separately on each town's rows. Ridge, XGBoost, and the MLP then have a higher search RMSE than the single residual model. CatBoost's 26-town search RMSE is 450.50, against the single model's 451.26, and its confirm RMSE is higher, so `mean_residual_catboost.csv` stays the single model. The 26-town predictions are `outputs/predictions/mean_residual_by_town_<model>.csv`.
