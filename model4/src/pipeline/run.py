"""Train, test, predict, or evolve one baseline.

Run from the project root:

    python -m src.pipeline.run evolve  --model mlp --features original,blocks --generations 4
    python -m src.pipeline.run train   --model xgb --features original
    python -m src.pipeline.run test    --model mlp
    python -m src.pipeline.run predict --model mlp --out artifacts/predictions/mlp.csv
    python -m src.pipeline.run use-all --model xgb
"""

import argparse
import sys
from pathlib import Path

from src.pipeline.catalog import resolve_features
from src.pipeline.evolve import evolve_model, predict_model, test_model, train_model
from src.pipeline.records import refresh_design


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or evolve a rental baseline.")
    parser.add_argument(
        "command",
        choices=["train", "test", "predict", "evolve", "ablate", "finalize", "use-all"],
    )
    parser.add_argument("--model", required=True, choices=["ridge", "xgb", "mlp", "catboost"])
    parser.add_argument("--features", default="", help="Column names and presets, comma-separated.")
    parser.add_argument("--feature-file", default="", help="One column name or preset per line.")
    parser.add_argument("--genome", default="", help="JSON file whose preprocess and architecture overlay the defaults.")
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--val-months", type=int, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--out", default="")
    parser.add_argument("--refit-all", action="store_true")
    parser.add_argument("--seed", type=int, default=5228)
    args = parser.parse_args()

    for name in ("ridge", "xgb", "mlp", "catboost"):
        refresh_design(name)

    if args.command == "finalize":
        from src.pipeline.finalize import finalize_model

        finalize_model(args.model, args.seed)
        return
    if args.command == "use-all":
        from src.pipeline.use_all import use_all_features

        use_all_features(args.model, args.seed)
        return
    if args.command == "ablate":
        from src.pipeline.ablation import ablate_model

        ablate_model(args.model, args.seed, args.epochs)
        return
    if args.command in {"train", "evolve"}:
        features = resolve_features(args.features, args.feature_file)
        val_months = 3 if args.val_months is None else args.val_months
        if args.command == "train":
            train_model(args.model, features, val_months, args.max_rows, args.epochs, args.seed, args.genome)
        else:
            evolve_model(
                args.model,
                features,
                args.generations,
                val_months,
                args.max_rows,
                args.epochs,
                args.seed,
            )
        return
    if args.command == "test":
        test_model(args.model, args.val_months, args.max_rows, args.seed)
        return
    predict_model(args.model, args.out, args.refit_all, args.seed)


if __name__ == "__main__":
    main()
