# Contributing

Keep each experimental change separate from its baseline. Record the input
schema, source version, date cutoffs, feature changes, model capacity, device,
seed and compute budget. Never use an outer evaluation target to select a model.

Use a dedicated output directory under `artifacts/`. Do not commit private or
row-level data, trained weights, credentials, environment directories, or runtime
logs. Public research results should contain aggregate metrics and a description
of their evaluation population.

Before proposing changes, run `python scripts/validate_repository.py` and the
relevant tests. Some model tests fit estimators; review a test before running it
when training is not authorised. Report exactly what was executed.

Describe changes to preprocessing, data cutoffs and future-covariate assumptions
explicitly. Keep unsuccessful candidates in the research record; a missing score
must remain missing rather than being replaced by zero.
