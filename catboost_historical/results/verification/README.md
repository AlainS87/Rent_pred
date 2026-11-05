# Implementation verification — 5 November 2026

All training and tests used the dedicated local `.venv`, Python 3.12.5,
macOS 14.8.7 arm64, NumPy 2.5.4, pandas 2.3.3, SciPy 1.18.1,
scikit-learn 1.9.1, and CatBoost 1.2.10. The complete tested environment is pinned
in `../../requirements-lock.txt`. The archived estimator also reports CatBoost
1.2.10; the archived Python and other package versions were not recorded.

## Full-data reproduction

Before restructuring, the unmodified original implementation was run with:

```bash
PYTHONDONTWRITEBYTECODE=1 python 'cs-5228-2610-project 2/historical_ablation.py' --data-dir 'cs-5228-2610-project 2' --mode submit --submit-variant F1 --iterations 1600 --threads 4 --output-dir outputs/original_f1
```

This optional reference command uses the existing flat-layout source datasets;
no copies were created. They are byte-identical to the parent inputs. Bytecode
writing was disabled to keep the original directory unchanged. The four original
official-matching tests also passed in this environment before restructuring.

After packaging, the following commands used the parent inputs directly:

```bash
python -m hdb_rent.final submit --data-dir ../Data --iterations 1600 --threads 4 --model-dir outputs/f1 --output outputs/f1/submission_F1_1600.csv
python -m hdb_rent.final predict --data-dir ../Data --model-dir outputs/f1 --output outputs/f1/submission_reloaded.csv
python -m hdb_rent.final validate --data-dir ../Data --iterations 1600 --threads 4 --fold-preset strict --output-dir outputs/f1_validation
```

Each full-data fit used all 150,000 training labels. Each submission comparison
below covers all **50,000** predictions in the original test row order.

| Comparison | Maximum absolute difference | Mean absolute difference | Evidence |
|---|---:|---:|---|
| Original full retraining vs archived submission | 1.8189894035458565e-12 | 4.428784450283274e-14 | `original_retraining.json` |
| Refactored full retraining vs original full retraining | **0** | **0** | `refactored_vs_original.json` |
| Refactored full retraining vs archived submission | 1.8189894035458565e-12 | 4.428784450283274e-14 | `refactored_vs_reference.json` |
| Separate-process saved-model inference vs first refactored submission | **0** | **0** | `reloaded_vs_first.json` |

The original and refactored generated CSV files have identical SHA-256 hashes.
The archived CSV differs only at a scale consistent with floating-point rounding.
Because the complete archived numerical environment is unavailable, the tiny
archival discrepancy cannot be attributed uniquely to serialization or numerical
library differences. Rebuilding features with the archived estimator also matches
its archived submission within 1.8189894035458565e-12
(`original_saved_model.json`). No substantive prediction difference was found.

All submission checks passed: exactly `Id,Predicted`, 50,000 rows, IDs ordered
0 through 49999, and finite/nonmissing predictions. The current test-file hash
matches the original input. The inference report binds predictions to that hash;
ID equality alone would not prove that the correct test input was used.

Full training took **92.1366 seconds**, including input/feature handling and
bundle creation. First saved-model inference took approximately **7.0 seconds**;
the independent reload took approximately **8.5 seconds**. These observations
include concurrent verification work and are not controlled hardware benchmarks.
The generated estimator is 196,524,368 bytes (approximately 187.4 MiB) and is ignored.

## Full chronological validation

The packaged F1 was rerun at 1600 iterations on all four strict folds:

| Fold | RMSE |
|---|---:|
| 2023 H2 development | 576.1308845879627 |
| 2024 H1 development | 512.3789583407276 |
| 2024 H2 development | 455.9763583364238 |
| 2025 Q1 diagnostic | 444.2865470821469 |
| Equal-weight development mean | **514.8287337550381** |

All four RMSE values match the recorded F1 values exactly after CSV parsing.
The maximum discrepancy across RMSE, MAE, and bias is 1.1368683772161603e-13.
`chronological_comparison.json` provides the comparison; `f1_chronological/`
contains the new run's small score tables and configuration.
The Q1 diagnostic is previously inspected and is not an independent holdout.

## Other verification

- **13 tests passed**, no failures, errors, or skips. The recorded test run took
  approximately 10 seconds. See `tests.json`.
- All **21 runtime modules** imported, syntax compilation succeeded, and
  `pip check` reported no broken requirements. See `imports.json`.
- All **200,000 rows × 41 features**, their dtypes, target values, and original
  train/test row orders match exactly between original and packaged pipelines.
  See `feature_parity.json`.
- A syntax-tree audit found **38 numerical definitions identical**, permitting
  only the explicit auxiliary-path lookup adaptation. This includes original
  F/G history generation and the principal ablation routines.
  See `numerical_code_audit.json`.
- All main experiment CLIs ran with small synthetic CSVs: baseline, legacy A–E,
  controlled A–D/iteration sweep, F0–F4, and F1/G1–G4. This checks execution paths;
  it does not establish full-data equivalence for every experiment.
- The plotting utility generated all three controlled-sweep plots from copies
  of historical score tables in `outputs/plot_verification/`. The development
  plot was visually inspected. Historical result directories were not modified.
- All **97 preserved reference files** are unchanged. All **44 historical
  CSV/JSON copies** retain their source SHA-256 hashes. See `integrity.json` and
  `../provenance.json`.
- The trained configuration, resolved estimator defaults, feature schema,
  environment and all required input/code identities are retained in
  `f1_bundle_manifest.json`. The inference checks are in `f1_inference.json`.

## Not performed

The full multi-model A–D iteration sweeps, F0–F4 sweep, and G1–G4 sweep were not
rerun on all labels. Their original recorded results are preserved separately.
Official-data download and full public-record matching were not rerun. No Kaggle
submission was uploaded, and the four author-reported Public Leaderboard scores
were not independently reevaluated. No second platform or fresh remote clone
was tested. Numerical reproduction above applies to this recorded environment.

No commit, push, merge, or pull request was made. Generated estimators, predictions,
the environment, and the original reference directory remain ignored by Git.
