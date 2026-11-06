"""Mutation rules shared by the evolve loop and each model's design note.

The prose in ``rules_text`` is written into ``models/<model>/docs/design.md``
on every train, test, evolve, and predict. Change the constants and the
prose together.
"""

SEARCH_GAIN = 0.01
OVERFIT_GAP = 0.15
PERMUTATION_ROWS = 2000

PREPROCESS_KNOBS = {
    "numeric_impute": ("median", "mean", "zero"),
    "scale": ("standard", "none", "minmax"),
    "categorical": ("onehot", "ordinal", "target", "bin"),
    "rare_min_count": (10, 30, 100),
}


def rules_text() -> str:
    """Stable design rules. The numbers come from the constants above."""
    knob_bits = []
    for name, values in PREPROCESS_KNOBS.items():
        shown = ", ".join(str(value) for value in values)
        knob_bits.append(f"{name} ({shown})")
    knobs = "; ".join(knob_bits)
    gain = f"{SEARCH_GAIN:.0%}"
    return f"""## Adjustment rules

The evolve command reads `docs/logs` and `docs/champion.json`, then proposes up to three mutants of the champion. Each mutant changes one gene: the feature subset, the preprocess spec, or the architecture. A genome already present in the logs is not trained again.

Feature changes alternate by generation. An odd generation adds the unused pool column with the largest absolute correlation to the champion residual on the search months. An even generation drops the column with the smallest permutation importance on a search sample of at most {PERMUTATION_ROWS:,} rows. When that single-column move was already tried from the same parent, and every column of a feature family is inside the pool, an odd generation adds that family as a block and an even generation removes it. A feature set always keeps at least one column.

Preprocess changes the first knob not yet tried from this parent, in this order: {knobs}. The new value is the first choice in that list that differs from the current value.

Architecture regularizes when (search RMSE - train RMSE) / train RMSE is above {OVERFIT_GAP}. Otherwise it adds capacity, inside the cap in this model's note.

A mutant replaces the champion only when its search RMSE is at least {gain} lower. The loop stops at the generation cap, when a generation accepts nothing, or when every proposed mutant is already in the logs. Confirm RMSE is scored once after search stops and is not an acceptance signal. Predict writes a summary and does not change the genome.

Labeled months are cut in time. Confirm is the last `val_months` approval months. Search is the `val_months` immediately before confirm. Fit is every earlier month. The preprocessor is fit on the fit months only.
"""
