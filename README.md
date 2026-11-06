# CS5228 HDB monthly rent


Predict `MONTHLY_RENT` for Singapore HDB flats. The metric is RMSE, in SGD. This file is the map of the code: which directory a command belongs to, and which model it trains.

Two write-ups sit in `Docs/merged/`. [report.tex](Docs/merged/report.tex) ([report.pdf](Docs/merged/report.pdf)) is the full report. [report_reduced.tex](Docs/merged/report_reduced.tex) ([report_reduced.pdf](Docs/merged/report_reduced.pdf)) is the shorter overview. The full report keeps the section-level argument, figures, and appendix. The overview keeps the question, the four models, and the scores.

Three code trees live in this checkout. They do not share a Python path.

| Tree | Working directory | What it is |
|---|---|---|
| This repository root | project root | Feature table, ridge / XGBoost / MLP / CatBoost pipeline, and the multistage stack (Model 3) |
| [model4](model4/README.md) | `model4/` | Standalone forecasting project, including Model 4 |
| [catboost_historical](catboost_historical/README.md) | `catboost_historical/` | Historical CatBoost line, including the F1 submission |

Run a command only from the directory named above. `model4/` contains its own `src/` and `models/`. Starting those modules from the repository root imports the wrong package.

## Data and splits

Labelled approvals run from January 2021 through March 2025 (150,000 rows in `Data/train.csv`). The unlabelled competition file is `Data/test.csv` (50,000 rows, header `Id,Predicted`, ids 0 through 49999). Auxiliary source tables are under `Data/auxiliary/`.

For Model 3 the labelled months are cut by approval month:

| Name | Months | Role |
|---|---|---|
| Training set | through September 2024 | Fit parameters |
| Validation set | October–December 2024 | The only split that stops or chooses the model |
| Test set | January–March 2025 | Scored after that choice; does not choose |
| Private test set | unlabelled file after the labelled period | Competition score; does not choose a model |

Model 1 and Model 2 use other labelled windows. The report still calls those scores validation RMSE. They are not the October–December 2024 validation set. A private-test RMSE is not a validation RMSE and is not a test RMSE.

Seed `5228` is the default for the root pipeline and for Model 3 and Model 4. The historical CatBoost line uses seed 42. A run that passes `--max-rows` is not a result about the full training table.

## Models in the report

| Report name | Idea | Where the code is | Score the report records |
|---|---|---|---|
| Model 1 | CatBoost on cleaned rental fields, after nested town and flat-type means | Written up in the report. The root pipeline below is the maintained CatBoost entry, not a replay of that six-month baseline | Validation RMSE 454.29 on October 2024–March 2025, after a fit through September 2024 |
| Model 2, `add_mrt` | CatBoost after adding auxiliary families; `add_mrt` is the best of those configurations on the development pooled RMSE | The comparison is in the report. This checkout does not contain the 1,878-column rebuilt candidate builder | Validation RMSE 499.53 on November 2023–March 2025, after a fit through October 2023. Private-test RMSE 432.50 |
| Model 3 | Per-town RNN mean of rent per square metre, then a per-town CatBoost and a per-town Gaussian mixture on the unit gap, then one shared MLP | `mean_residual_model/` | Validation RMSE 448.24 at epoch 8. Test RMSE 462.87. An earlier private-test submission scored 451.26 |
| Model 4 | One CatBoost for every town, with a property-adjusted rental time index | `model4/experiments/local/` | Validation RMSE 444.47 and test RMSE 450.89 on the same months as Model 3. Private-test RMSE 441.46, the latest submission, not the best |
| F1 | CatBoost with block-level historical rents, 1600 iterations, plus a forward market correction | `catboost_historical/` | Private-test RMSE 432.50394 |

The report's best private-test RMSE is 432.50, named `add_mrt` in the comparison table. The F1 table in the same report records 432.50394 for block history at 1600 iterations. Those are the published competition figures. They did not choose Model 3's epoch. Gold Miner (Group 12) is rank 21 after eight submissions. The latest submission, 441.46, did not replace 432.50.

## Root: install and features

From the project root, after `pip install -r requirements.txt`:

```text
python -m src.features.build_features
```

This reads `Data/train.csv`, `Data/test.csv`, and `Data/auxiliary/`. It writes `Data/auxiliary_features/train_features.csv` and `test_features.csv` in the same row order. It does not sort. The auxiliary table has 94 columns. The six group-mean columns use labelled rents from the previous three approval months only. No other auxiliary column reads `MONTHLY_RENT`. Column formulas are in [Docs/auxiliary_new_features.md](Docs/auxiliary_new_features.md).

## Root: ridge, XGBoost, MLP, and CatBoost

`--model` is `ridge`, `xgb`, `mlp`, or `catboost`. A model under `models/<model>/` is finished only after these four commands, in order. `ablate` adds auxiliary columns that are missing from `models/<model>/best`. It does not start from the full column list.

```text
python -m src.pipeline.run train --model <model> --features original
python -m src.pipeline.run ablate --model <model>
python -m src.pipeline.run finalize --model <model>
python -m src.pipeline.run submit --model <model>
```

`train` fits the ten source columns and writes the first `best/` snapshot. `ablate` probes one missing auxiliary column at a time. Ridge and the MLP call a column promising only when its validation RMSE is at least 1% below that round's current set. XGBoost and CatBoost call it promising when the validation RMSE is strictly lower. Those probes do not change `best/`. The current columns plus every promising column replace `best/` only when that combined fit has a lower validation RMSE. The loop stops when a round has no promising column or the combined fit does not win. The test RMSE is recorded and does not accept a column. The method is also in the report appendix.

`finalize` retrains the columns already in `best/` with a larger budget and writes `final/`. It does not add or drop columns. `submit` joins `Data/test.csv` to the auxiliary test table by row order and writes `outputs/predictions/<model>_final.csv` from `final/`, then `outputs/predictions/<model>_refit.csv` after fitting every labelled month with the stopped size frozen.

Other commands, which do not replace that sequence:

```text
python -m src.pipeline.run test --model <model>
python -m src.pipeline.run predict --model <model> --out outputs/predictions/<model>.csv
python -m src.pipeline.run evolve --model <model> --features original,blocks --generations 4
python -m src.pipeline.run use-all --model <model>
```

`use-all` fits every original and auxiliary column into `models/<model>/use_all_features` and does not replace `best/` or `final/`.

## Root: town-month mean plus a residual

This is an earlier experiment. It does not replace `models/<model>/best` or `final/`.

```text
python -m mean_residual_model.run
python -m mean_residual_model.run_by_town
```

The first command predicts a town-month mean with a ridge, then fits one residual model per algorithm on `MONTHLY_RENT` minus that mean. Test files are `outputs/predictions/mean_residual_<model>.csv`. The second command fits those genomes separately in each town. Details are in [mean_residual_model/README.md](mean_residual_model/README.md).

## Root: Model 3, the multistage stack

The region mean is a per-town 3-layer RNN of rent per square metre. Each town then has a CatBoost and a Gaussian mixture on the unit gap. Those two do not see each other. One shared MLP receives the RNN mean, both gaps, floor area, and the flat's other features, and outputs monthly rent. The validation set stops the CatBoost iteration and the MLP epoch.

```text
python "mean_residual_model/multi_stage_model(lstm+catboost+mlp)/run.py"
python "mean_residual_model/multi_stage_model(lstm+catboost+mlp)/run.py" --residuals-only
python "mean_residual_model/multi_stage_model(lstm+catboost+mlp)/run.py" --mlp-only
python "mean_residual_model/multi_stage_model(lstm+catboost+mlp)/run.py" --frozen-catboost
python "mean_residual_model/multi_stage_model(lstm+catboost+mlp)/run.py" --predict
```

`--residuals-only` fits the town CatBoost and the mixture and does not fit the MLP. `--mlp-only` trains the MLP on saved gaps. `--frozen-catboost` reloads each town CatBoost and refits the mixture and the MLP. `--predict` writes the test file from saved weights and does not refit. The search-stopped test file is `outputs/predictions/mean_residual_multistage.csv`.

Later checks keep those weights and write separate files:

```text
python -m mean_residual_model.refit.run
python -u "mean_residual_model/aligned_preprocess/run.py"
python -u "mean_residual_model/catboost_head/run.py"
```

`refit` fits the same stack on every labelled month through March 2025, with the stopped sizes frozen, and writes `outputs/predictions/mean_residual_multistage_refit.csv`. It has no validation or test RMSE. `aligned_preprocess` retrains the town CatBoost, the mixture, and the MLP with a building id, calendar fields, and lease age; it does not replace the search-stopped file. `catboost_head` freezes the RNN, the town CatBoost, and the mixture, and fits one CatBoost in place of the MLP. Its test file is `outputs/predictions/mean_residual_catboost_head.csv`.

The RNN checkpoint used by the stack is trained with:

```text
python -m mean_residual_model.reginal_mean_model.rnn.train_l3
```

That command scores the hold-out and writes `mean_residual_model/reginal_mean_model/rnn/h16_l3/`. The refit command trains a separate copy under `mean_residual_model/refit/rnn/` and does not overwrite that folder.

## model4: Model 4 and the forecasting project

Install and run from `model4/`, not from the repository root. Its own README is [model4/README.md](model4/README.md). Python 3.11 or 3.12:

```text
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[models,test]"
```

Model 4's global CatBoost and the rental time index are fixed-split scripts. Each takes `prepare`, then `run`:

```text
python -m experiments.local.global_monthly_catboost prepare
python -m experiments.local.global_monthly_catboost run
python -m experiments.local.temporal_index_trial prepare
python -m experiments.local.temporal_index_trial run
```

The time-index script is the configuration the report calls Model 4: depth 8, learning rate 0.05, at most 1000 iterations, seed 5228, early stopping of 50 rounds on October–December 2024. Its private-test file is the 441.46 submission. It does not replace the 432.50 result.

Other entry points in that directory, each with `--help`:

```text
python -m src.features.build_features --help
python -m src.pipeline.run --help
python -m src.backtest.run --help
python -m src.research_round2.run --help
python -m src.research_round3.run --help
python -m src.research_round4.run --help
python -m experiments.local.bootstrap prepare-features
python -m experiments.rolling_local validate --config experiments/rolling_local/example_config.json
python -m experiments.rolling_local run --config experiments/rolling_local/example_config.json --output artifacts/rolling-local-example
python -m analysis.external_signals --help
```

`model4/src/pipeline/run.py` is not `src/pipeline/run.py` at the repository root. Raw inputs for this tree default to `model4/data/raw/`, which is separate from `Data/` at the root.

## catboost_historical: F1

Install and run from `catboost_historical/`. Its own README is [catboost_historical/README.md](catboost_historical/README.md). It reads `../Data` and does not import the root `src/` or `models/`.

```text
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m hdb_rent.final validate --data-dir ../Data --iterations 1600 --threads 4 --fold-preset strict --output-dir outputs/f1_validation
python -m hdb_rent.final train --data-dir ../Data --iterations 1600 --threads 4 --model-dir outputs/f1
python -m hdb_rent.final predict --data-dir ../Data --model-dir outputs/f1 --output outputs/f1/submission_F1_1600.csv
python -m hdb_rent.final submit --data-dir ../Data --iterations 1600 --threads 4 --model-dir outputs/f1 --output outputs/f1/submission_F1_1600.csv
```

`validate` scores chronological folds and does not write a submission. `train` fits the saved bundle. `predict` applies a saved bundle. `submit` trains and predicts. F1 is 1600 iterations, depth 6, learning rate 0.055, seed 42, with block-history features and a damped market correction. The report's Kaggle table gives this submission private-test RMSE 432.50394.

History variants and plots, still from `catboost_historical/`:

```text
python -m experiments.historical --data-dir ../Data --mode validate --fold-preset strict --variants F0,F1,F2,F3,F4 --iterations 1600 --threads 4 --output-dir outputs/history_strict
python -m experiments.temporal --data-dir ../Data --mode validate --fold-preset strict --iterations 800,1200,1600,2000 --threads 4 --output-dir outputs/controlled_sweep
python -m experiments.baseline --data-dir ../Data --mode both --iterations 180 --depth 6 --threads 4 --output-dir outputs/baseline
python -m unittest discover -s tests -v
```

## Report

From `Docs/merged/`, compile each file twice:

```text
pdflatex -interaction=nonstopmode report.tex
pdflatex -interaction=nonstopmode report_reduced.tex
```

`report.pdf` is the full report. `report_reduced.pdf` is the overview. An earlier copy of the full report remains at `Docs/report/merged/report.tex`.
