# Historical CatBoost for Singapore HDB rental prediction

This standalone subproject preserves the original CatBoost experiments and final
**F1 1600** prediction pipeline. Its reported Kaggle Public Leaderboard RMSE is
**432.50394**. It reuses the parent repository's datasets and does not depend on
the teammate's training framework in `src/` or `models/`.

F1 consists of property/geographic feature engineering, strictly prior-month
block rental statistics, a CatBoost estimator, and a forward market correction.
The estimator alone is insufficient to reproduce a submission.

## Layout

```text
catboost_historical/
├── README.md
├── requirements.txt                 # installs the tested lock file
├── requirements-lock.txt            # exact direct and transitive versions
├── hdb_rent/
│   ├── data.py                      # parent-data paths, hashes, environment
│   ├── base.py                      # original 34 features and model factory
│   ├── temporal.py                  # seasonality and market correction
│   ├── historical_features.py       # original F0–F4 rental history
│   ├── enhanced_history.py          # experimental G1–G4 history
│   ├── evaluation.py
│   ├── submission.py
│   └── final.py                     # validate / train / predict / submit CLI
├── experiments/
│   ├── baseline.py
│   ├── temporal.py                  # controlled A–D and iteration sweeps
│   ├── legacy_temporal.py           # original A–E workflow, retained once
│   ├── historical.py                # F0–F4
│   └── enhanced.py                  # F1 versus G1–G4
├── diagnostics/official_matching.py
├── scripts/
│   ├── plot_results.py
│   ├── compare_submissions.py
│   └── verify_original.py           # optional local reference comparison
├── tests/
├── results/                         # small historical records and verification
├── outputs/                         # ignored generated artifacts
└── cs-5228-2610-project 2/          # ignored, untouched local reference
```

The reference directory and `_migration_source/` are ignored. Neither is needed
for normal training, validation, prediction, plotting, or unit tests. Normal
modelling entry points import only this subproject. The optional
`verify_original` utility explicitly imports the preserved reference for comparison.

## Setup

Use a checkout containing this subproject. After this work is published to the
repository, a fresh-clone setup is:

```bash
git clone https://github.com/AlainS87/Rent_pred.git
cd Rent_pred/catboost_historical
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
```

If the subproject is published on a separate branch, check out that branch before
entering `catboost_historical/`. These local changes have not been committed or
pushed as part of the implementation.

All remaining commands run **inside `catboost_historical/` with this environment
activated**. No global Python installation needs modification.

The tested environment is Python **3.12.5**, macOS 14.8.7, arm64:

| Package | Tested version | Purpose |
|---|---:|---|
| numpy | 2.5.4 | numerical operations |
| pandas | 2.3.3 | CSV and feature processing |
| scipy | 1.18.1 | geographic nearest-neighbour queries |
| scikit-learn | 1.9.1 | evaluation metrics |
| catboost | 1.2.10 | regression; matches the archived estimator version |
| matplotlib | 3.11.2 | plots |
| requests | 2.34.2 | optional official-data download diagnostic |

`requirements-lock.txt` includes all 28 tested third-party distributions.
Installation on other platforms was not tested. The saved pipeline checks core
package versions; Python differences produce a warning. Matching a seed alone
does not guarantee identical results across library versions and platforms.

## Existing data

```text
../Data/
├── train.csv                        # 150,000 labelled rows, 2021-01–2025-03
├── test.csv                         # 50,000 rows, 2025-03–2026-07
└── auxiliary/
    ├── sg-hdb-block.csv
    ├── sg-mrt-stations.csv
    ├── sg-schools.csv
    └── sg-shopping-malls.csv
```

`train.csv` must include `MONTHLY_RENT`; `test.csv` must not. Both require the
original property columns and `RENT_APPROVAL_DATE` in `YYYY-MM` format. Supply
the competition CSV files at these paths if they are absent from your checkout.
Datasets are not copied into this subproject. All six required inputs were
byte-identical to the preserved reference at implementation time.

Every modelling CLI accepts `--data-dir ../Data`; this is also the default,
resolved relative to the package location. Auxiliary files are read from
`<data-dir>/auxiliary/`. The old flat auxiliary layout is also supported.

COE, stock prices, and official HDB rental records are not F1 inputs.

## Final F1 configuration

The original feature order and arithmetic are retained: 34 original features,
then `month_sin`, `month_cos`, then:

```text
hr_block_n
hr_block_median
hr_block_mean
hr_block_std
hr_block_age_months
```

There are 41 features, including the original 11 categorical columns. The exact
list, dtypes, and categorical names are saved in each trained bundle manifest.
Original normalization, missing numeric values, population standard deviation,
geographic calculations, and CatBoost defaults are retained.

| Setting | Value |
|---|---:|
| Iterations | 1600, fixed, no early stopping |
| Depth | 6 |
| Learning rate | 0.055 |
| L2 leaf regularization | 5 |
| Loss / evaluation metric | RMSE / RMSE |
| Seed | 42 |
| CPU threads | 4 |
| Trend strength | 0.5 |
| Trend lookback | 12 months |
| Trend decay | 6 months |
| Monthly log-slope cap | 0.012 |

The market index uses the original 20-round composition-adjusted town/flat-type
log-rent calculation and damped local extrapolation. Corrections apply only to
months strictly later than the last training month. March 2025 test rows receive
a multiplier of 1. Historical features for those rows still exclude March rents.

## Chronological validation

Run F1 with the original strict fold preset:

```bash
python -m hdb_rent.final validate --data-dir ../Data --iterations 1600 --threads 4 --fold-preset strict --output-dir outputs/f1_validation
```

| Role | Validation months | Training labels |
|---|---|---|
| Development | 2023-07–2023-12 | strictly before 2023-07 |
| Development | 2024-01–2024-06 | strictly before 2024-01 |
| Development | 2024-07–2024-12 | strictly before 2024-07 |
| Diagnostic | 2025-01–2025-03 | strictly before 2025-01 |

The development score is the equal-weight mean of the three fold RMSEs, not a
pooled score. The 2025-Q1 period was inspected in earlier experiments and is not
an independent unseen holdout. Validation uses no early stopping for F1.

For a single custom-period check:

```bash
python -m hdb_rent.final validate --data-dir ../Data --folds 2025-01:2025-03 --output-dir outputs/f1_q1
```

The inherited custom-fold implementation labels custom periods `development`;
the Q1 score remains a previously inspected diagnostic regardless of that label.
The `legacy` preset uses overlapping 18-, 12-, and 3-month windows ending in
2025-03. Do not treat those overlapping observations as independent pooled OOF
predictions.

## Full training and submission

Train on all 150,000 labels, saving the estimator and its pipeline metadata:

```bash
python -m hdb_rent.final train --data-dir ../Data --iterations 1600 --threads 4 --model-dir outputs/f1
```

Generate the submission from the saved bundle, without fitting another model:

```bash
python -m hdb_rent.final predict --data-dir ../Data --model-dir outputs/f1 --output outputs/f1/submission_F1_1600.csv
```

Or perform training followed by saved-model inference in one command:

```bash
python -m hdb_rent.final submit --data-dir ../Data --iterations 1600 --threads 4 --model-dir outputs/f1 --output outputs/f1/submission_F1_1600.csv
```

The bundle is `model.cbm` plus `manifest.json`. The manifest records configuration,
resolved estimator parameters, feature order/dtypes, categorical columns,
training dates and counts, environment, and SHA-256 hashes of code, estimator,
training CSV, geographic inputs, and the test input used during training.

Inference requires the bundle, this package, and the matching parent training
and geographic CSVs. It verifies their identities, rebuilds the original history
from training labels, and recomputes the original market correction. No labels
are serialized or duplicated. Retain the same CSV bytes; even editing line
endings invalidates the reference checksum. Inference rejects changed code,
core dependency versions, reference data, estimator bytes, or feature schema.
Bundles should be treated as immutable outputs; use another model directory for
another configuration.

Training retains original joint train/test preprocessing, so `train` also needs
`test.csv`, even though its labels are never read. `predict --test-file NAME`
can select another test CSV in the data directory; the output report records its
hash and whether it is the same input used at training. The default expected
submission size is exactly 50,000. The `--expected-rows` override is intended for
small test fixtures only.

The submission is read back and checked for exactly `Id,Predicted`, ordered IDs
0–49999, 50,000 rows, and finite predictions. Input row order is preserved without
sorting. A neighbouring JSON report records configuration, input identity,
environment, and inference runtime. A non-default iteration setting is recorded
in metadata; set a corresponding output filename yourself.

## Reproduce the experiments

These commands implement the recorded configurations; they write fresh results
to ignored `outputs/`, leaving historical `results/` unchanged.

Original baseline, including chronological validation and full-data prediction:

```bash
python -m experiments.baseline --data-dir ../Data --mode both --iterations 180 --depth 6 --threads 4 --output-dir outputs/baseline
```

Controlled A–D seasonality/trend ablation and 800/1200/1600/2000 iteration sweep:

```bash
python -m experiments.temporal --data-dir ../Data --mode validate --fold-preset strict --iterations 800,1200,1600,2000 --trend-strength 0.5 --threads 4 --output-dir outputs/controlled_sweep
```

A uses the original features; B adds cyclic month features; C applies the market
correction to A; D applies it to B. The original optimization trains the maximum
iteration count once per feature set and evaluates tree prefixes for smaller
counts. All variants at a given count share hyperparameters and seed.

Recorded overlapping long-horizon sweep:

```bash
python -m experiments.temporal --data-dir ../Data --mode validate --fold-preset legacy --iterations 1200,1600,2000 --trend-strength 0.5 --threads 4 --output-dir outputs/long_horizon
```

F0–F4 historical ablation:

```bash
python -m experiments.historical --data-dir ../Data --mode validate --fold-preset strict --variants F0,F1,F2,F3,F4 --iterations 1600 --trend-strength 0.5 --threads 4 --output-dir outputs/history_strict
```

F0 is D1600; F1 adds block history; F2 adds street/town history; F3 adds rolling
3/6/12-month statistics and spreads; F4 adds nearby other-block comparables.

F1 versus enhanced G1–G4:

```bash
python -m experiments.enhanced --data-dir ../Data --mode validate --fold-preset strict --variants F1,G1,G2,G3,G4 --iterations 1600 --trend-strength 0.5 --threads 4 --output-dir outputs/g_strict
```

The G tiers cumulatively add normalized block history, rolling normalized block
history, hierarchical street/town shrinkage, and nearby normalized comparables.
They remain experiments; the selected F1 model does not include G features.

The older A–E workflow remains available with `python -m experiments.legacy_temporal
--help`. E uses longer training than A–D; it must not be interpreted as a
controlled same-capacity comparison. Its original defaults are preserved.

Plot fresh controlled-sweep results:

```bash
MPLCONFIGDIR=outputs/matplotlib python -m scripts.plot_results --input-dir outputs/controlled_sweep
```

## Recorded results

The Public Leaderboard values below were supplied by the project author. They
are not present in the historical CSV/JSON records and were not re-evaluated by
uploading submissions during this implementation.

| Submitted model | Kaggle Public Leaderboard RMSE |
|---|---:|
| Initial CatBoost | 441.78158 |
| D1200 | 439.55517 |
| D1600 | 438.28697 |
| **F1 1600** | **432.50394** |

The following are actual local chronological records, not leaderboard scores:

| Controlled model, 1200 iterations | Development mean RMSE | 2025-Q1 diagnostic RMSE |
|---|---:|---:|
| A | 519.59757 | 451.39216 |
| B | 519.54890 | 451.31575 |
| C | 518.37978 | 450.83549 |
| D | 517.95431 | 450.71961 |

| D iterations | Strict development mean RMSE | 2025-Q1 diagnostic RMSE |
|---|---:|---:|
| 800 | 518.51354 | 451.95234 |
| 1200 | 517.95431 | 450.71961 |
| 1600 | 517.89381 | 449.80102 |
| 2000 | 517.88433 | 449.68020 |

| Historical model, 1600 iterations | Development mean RMSE | 2025-Q1 diagnostic RMSE |
|---|---:|---:|
| F0 | 517.89381 | 449.80102 |
| **F1** | **514.82873** | **444.28655** |
| F2 | 519.01292 | 445.79905 |
| F3 | 528.69577 | 445.43949 |
| F4 | 535.78300 | 445.62850 |
| G1 | 515.16907 | 444.57476 |
| G2 | 515.01463 | 443.82278 |
| G3 | 516.39382 | 444.88241 |
| G4 | 517.35174 | 444.97480 |

F1 has the lowest recorded development mean among the F/G history variants.
G2's slightly lower Q1 diagnostic score did not replace F1. The strict temporal
sweep selected D2000, whereas the overlapping long-horizon sweep selected D1600
(mean RMSE 473.14167); these use different validation windows and are not directly
comparable. The final F1 configuration remains the author's selected model.

The older baseline records contain Q1 RMSE 459.57820 at 180 iterations and
451.38415 for the 1200-iteration baseline run, which used early stopping and
retained best iteration 1197 (zero based). These should not be substituted for
the controlled fixed-iteration scores. The exact run corresponding to the
author-reported initial leaderboard score is not established by those records.

Validation RMSE evaluates known historical labels held out chronologically.
Kaggle Public Leaderboard RMSE evaluates the competition's undisclosed public
test subset. They cover different samples, horizons, and market conditions.
Matching reference predictions does not independently verify the leaderboard.

Small original CSV/JSON files are copied byte-for-byte into `results/`.
`results/provenance.json` maps all 44 files to their original paths and hashes.
See `results/README.md` and `results/verification/` for verification evidence.

## Leakage prevention and limitations

For a query in month M, F1 uses labelled transactions with month **strictly less
than M**, within the same town/block/street/flat-type group. The current row,
all same-month transactions, and later labels are excluded. Empty history has
count zero and missing-valued statistics, as in the original code.

Validation history contains only training-fold labels. Validation predictions
never update that reference with held-out rents. F1 counts and prices consequently
stay fixed after the cutoff, while time-since-last-rental increases. F3 rolling
windows age relative to the query month; the enhanced G windows are instead
anchored at the fold cutoff. This distinction is preserved, not redesigned.

Market effects are fitted only on permitted training labels. Centered smoothing
inside that already-observed history does not read held-out months. CatBoost's
original categorical treatment is retained. Geographic data are static public
covariates and may not represent the exact historical opening dates of amenities.

The future test horizon extends beyond the labelled period. Damped extrapolation
and sparse block histories may fail under market changes. Repeated examination
of Q1 and tuning against the Public Leaderboard risk overfitting. Geographic
distances are the original approximate planar distances. No additional data or
algorithmic enhancements were introduced to address these limitations.

## Official HDB matching diagnostic

This optional utility compares existing predictions with public records. Those
records omit flat identifiers, so even single-candidate matches are not verified
Kaggle labels. Group-mean errors are proxy metrics and must not be reported as
competition RMSE. The diagnostic never changes model predictions.

With an independently supplied official CSV:

```bash
python -m diagnostics.official_matching --test-file ../Data/test.csv --train-file ../Data/train.csv --pred-file outputs/f1/submission_F1_1600.csv --official-csv outputs/official_data.csv --output-dir outputs/official_check
```

`--download` is an optional network operation; it was not exercised during this
implementation. No official rental CSV is committed or imported into F1.

## Verification and resource requirements

```bash
python -m unittest discover -s tests -v
python -m compileall -q hdb_rent experiments diagnostics scripts tests
```

Tests use small synthetic CSVs, run actual CatBoost fits and saved-model reloads,
exercise all main experiment CLIs, and reject altered historical reference data.
They also test month exclusions, cutoff anchoring, trend boundaries, feature
order, submission schemas, and the four adapted official-matching tests. Smoke
tests alone do not establish full F1 numerical equivalence.

For the local preserved source only, compare all 200,000 feature rows exactly:

```bash
python -m scripts.verify_original --data-dir ../Data --output outputs/verification/feature_parity.json
```

Compare a generated submission with the local reference:

```bash
python -m scripts.compare_submissions --reference 'cs-5228-2610-project 2/outputs/submit_F1/submission_F1_1600.csv' --candidate outputs/f1/submission_F1_1600.csv --output outputs/verification/reference_comparison.json
```

These two optional commands require the ignored reference directory and are not
fresh-clone prerequisites. The reference is deliberately excluded from Git.

The implementation first ran the unmodified original F1 on all 150,000 labels
in the dedicated environment. Its 50,000 predictions differed from the archived
submission by at most 1.8189894035458565e-12 (mean 4.428784450283274e-14).
Detailed refactored-run results and measured timings are in
`results/verification/README.md`. Tests and training use the same environment.

F1 materializes 200,000 rows and 41 features and runs on CPU with four threads.
On the tested machine, full training took 92.1 seconds and saved-model inference
took 7.0 seconds. The dedicated environment occupies approximately 513 MiB and
the generated estimator approximately 187.4 MiB; both are ignored by Git.
Reserve several GB of RAM; an 8 GB or larger machine is a reasonable planning
estimate, not a measured minimum. Full sweeps are substantially more expensive:
the strict temporal sweep fits eight 2000-tree models, and each complete F or G
ablation fits twenty 1600-tree models, plus feature computation. Allow tens of
minutes to hours depending on hardware. Full ablation reruns are distinguished
from historical records in the verification report; their runtimes are not
claimed as measured here.

Models, submissions, environments, and large intermediate outputs remain in
ignored directories. No commit, push, merge, or pull request is performed by
these commands.
