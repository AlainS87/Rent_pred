# Town-month mean plus residual

The predicted rent of one flat is the town-month mean plus a residual.

`reginal_mean_model` is a ridge (`alpha` 1) of the mean `MONTHLY_RENT` in each `TOWN` and approval month. It is fit on cells through 2024-09. The inputs are the town, the month index since 2021-01, `coe_c_premium_lag1`, `bank_basket_adj_close_lag1`, and `property_basket_adj_close_lag1`. Every flat in that town and month receives the same mean. A town absent from the fit months keeps only the shared month and economy part.

`D-value_models` holds one residual model for ridge, XGBoost, the MLP, and CatBoost. Each copies the features, preprocess, and architecture from `models/<model>/best/genome.json`. The target is `MONTHLY_RENT` minus the town-month prediction. `BLOCK` is the shrunk mean of that residual on the fit months. Search months 2024-10..12 stop the trees and the MLP. Confirm months 2025-01..03 are recorded and do not choose a model.

Run from the project root:

    python -m mean_residual_model.run

The command does not replace `models/<model>/best` or `final`, and it does not add or drop columns. Test predictions are `outputs/predictions/mean_residual_<model>.csv` with header `Id,Predicted`. They use the search-stopped weights fit through 2024-09. Labeled months 2025-01..03 are not refit into those files.

`python -m mean_residual_model.run_by_town` fits those same genomes separately on each town's rows. The town-month ridge is unchanged. On the search quarter, the single residual model is lower for ridge, XGBoost, and the MLP. CatBoost's 26-town search RMSE is lower and its confirm RMSE is higher, so the single CatBoost residual stays in `mean_residual_catboost.csv`. The 26-town predictions are `outputs/predictions/mean_residual_by_town_<model>.csv`. Weights for each town are under `D-value_models/<model>/towns/`.

## Later stack

The later experiment is `multi_stage_model(lstm+catboost+mlp)`. The dated scores are in `logs/progress_1031.md`. The current test file is `outputs/predictions/mean_residual_multistage.csv`. Its region mean is the three-layer RNN in `reginal_mean_model/rnn/h16_l3/`, trained on the town-month mean of rent per square metre. Each town then has its own CatBoost and its own Gaussian mixture on that unit gap. The 128-64 MLP sees those three terms after they have been multiplied by floor area, and its output is monthly rent. This file does not replace `mean_residual_<model>.csv`.
