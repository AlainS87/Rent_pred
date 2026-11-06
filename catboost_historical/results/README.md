# Result provenance

These are historical records, not newly manufactured experimental outcomes.
All 44 original CSV/JSON files were copied byte-for-byte from the local preserved
reference. `provenance.json` records their original relative paths, destination
paths, and SHA-256 hashes. Original manifests may mention the old file layout;
those historical strings are intentionally preserved.

| Directory | Meaning |
|---|---|
| `baseline/initial/` | 180-iteration baseline metrics |
| `baseline/cb_1200/` | older 1200-iteration baseline metrics with early stopping |
| `controlled_sweep/` | strict A–D sweep at 800/1200/1600/2000 iterations |
| `long_horizon/` | overlapping historical folds at 1200/1600/2000 iterations |
| `history_strict/` | F0–F4 at 1600 iterations |
| `g_strict/` | F1 versus G1–G4 at 1600 iterations |
| `final_f1/` | original final-submission manifest and summary, without predictions |
| `official_matching/` | public-record diagnostic summaries, not Kaggle RMSE |
| `E1200/`, `E1600/`, `ablation/` | small market-index records from older submissions |
| `verification/` | new implementation checks and complete prediction comparisons |

`leaderboard.csv` contains the four actual Public Leaderboard values supplied by
the project author. They were not found in the original machine-readable records
and were not verified by submitting to Kaggle during this work.

Large predictions, row-level official matches, weights, source datasets, and
generated images are excluded. Fresh experiment commands write to `outputs/`.
Historical metrics, new verification measurements, and public leaderboard scores
must remain distinguishable.
