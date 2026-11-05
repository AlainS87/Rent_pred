"""Fit one residual model per town, using each best genome.

The town-month ridge stays the shared level. Each town then gets its own
ridge, XGBoost, MLP, and CatBoost for the gap from that level.

    python -m mean_residual_model.run_by_town
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from mean_residual_model.regional import load_regional
from mean_residual_model.residual import (
    MODELS,
    fit_residual,
    load_best_genome,
    predict_residual,
    rent_scores,
    stopped_size,
)
from mean_residual_model.run import (
    REGIONAL_DIR,
    ROOT,
    SEED,
    _check_months,
    _columns,
    _write_predictions,
)
from src.pipeline.data import load_joined, time_split
from src.pipeline.fit import load_model_module
from src.pipeline.genome import feature_hash


def main() -> None:
    genomes = {model: load_best_genome(ROOT, model) for model in MODELS}
    columns = _columns(genomes)
    frame = load_joined("train", columns)
    split = time_split(frame, 3, None)
    _check_months(split)
    regional = load_regional(REGIONAL_DIR)
    periods = {"train": split.fit, "search": split.search, "confirm": split.confirm}
    level = {name: regional.predict(part) for name, part in periods.items()}
    rents = {
        name: part["MONTHLY_RENT"].to_numpy(dtype=np.float64) for name, part in periods.items()
    }
    y_fit = rents["train"] - level["train"]
    y_search = rents["search"] - level["search"]
    towns = _towns(periods)
    print(f"towns {len(towns)}")

    test = load_joined("test", columns)
    test_level = regional.predict(test)
    for model in MODELS:
        print(f"fitting {model} residual by town")
        gap, test_gap, town_metrics = _fit_towns(
            model,
            genomes[model],
            periods,
            y_fit,
            y_search,
            test,
            towns,
        )
        metrics = rent_scores(rents, level, gap)
        metrics["towns"] = town_metrics
        metrics["rows"] = {name: int(len(part)) for name, part in periods.items()}
        _save_summary(model, genomes[model], metrics)
        shared = _shared_search(model)
        print(
            f"{model} by town sum RMSE train {metrics['train_rmse']:.4f}, "
            f"search {metrics['search_rmse']:.4f}, confirm {metrics['confirm_rmse']:.4f}"
            f" (shared-model search {shared:.4f})"
        )
        prediction = test_level + test_gap
        output = ROOT / "outputs" / "predictions" / f"mean_residual_by_town_{model}.csv"
        _write_predictions(output, prediction)
        print(f"wrote {output.relative_to(ROOT).as_posix()}")


def _fit_towns(
    model: str,
    genome: dict,
    periods: dict[str, pd.DataFrame],
    y_fit: np.ndarray,
    y_search: np.ndarray,
    test: pd.DataFrame,
    towns: list[str],
):
    gap = {name: np.full(len(part), np.nan) for name, part in periods.items()}
    test_gap = np.full(len(test), np.nan)
    town_metrics = []
    module = load_model_module(model)
    for town in towns:
        masks = {
            name: part["TOWN"].astype(str).to_numpy() == town for name, part in periods.items()
        }
        fit_rows = int(masks["train"].sum())
        search_rows = int(masks["search"].sum())
        if fit_rows == 0 or search_rows == 0:
            raise SystemExit(f"{town} has fit {fit_rows} and search {search_rows}")
        fit_frame = periods["train"].loc[masks["train"]].reset_index(drop=True)
        search_frame = periods["search"].loc[masks["search"]].reset_index(drop=True)
        preparer, bundle = fit_residual(
            model,
            genome,
            fit_frame,
            search_frame,
            y_fit[masks["train"]],
            y_search[masks["search"]],
            SEED,
        )
        for name, part in periods.items():
            if int(masks[name].sum()) == 0:
                continue
            piece = part.loc[masks[name]].reset_index(drop=True)
            gap[name][masks[name]] = predict_residual(model, preparer, bundle, piece)
        test_mask = test["TOWN"].astype(str).to_numpy() == town
        if int(test_mask.sum()):
            piece = test.loc[test_mask].reset_index(drop=True)
            test_gap[test_mask] = predict_residual(model, preparer, bundle, piece)
        folder = _town_dir(model, town)
        folder.mkdir(parents=True, exist_ok=True)
        module.save_model(bundle, folder)
        _dump_preprocessor(preparer, folder)
        record = {
            "confirm_rows": int(masks["confirm"].sum()),
            "fit_rows": fit_rows,
            "search_rows": search_rows,
            "test_rows": int(test_mask.sum()),
            "town": town,
            **stopped_size(model, bundle),
        }
        town_metrics.append(record)
        print(f"  {town}: fit {fit_rows}, search {search_rows}, confirm {record['confirm_rows']}")
    for name, values in gap.items():
        if not np.isfinite(values).all():
            raise SystemExit(f"{model} left a {name} row without a town residual")
    missing = ~np.isfinite(test_gap)
    if missing.any():
        names = sorted(test.loc[missing, "TOWN"].astype(str).unique())
        raise SystemExit(f"{model} test towns without a residual model: {names}")
    return gap, test_gap, town_metrics


def _towns(periods: dict[str, pd.DataFrame]) -> list[str]:
    fit_towns = set(periods["train"]["TOWN"].astype(str).unique())
    search_towns = set(periods["search"]["TOWN"].astype(str).unique())
    confirm_towns = set(periods["confirm"]["TOWN"].astype(str).unique())
    if fit_towns != search_towns or fit_towns != confirm_towns:
        raise SystemExit("fit, search, and confirm do not cover the same towns")
    return sorted(fit_towns)


def _town_dir(model: str, town: str) -> Path:
    return ROOT / "artifacts" / "mean_residual" / "residuals" / model / "towns" / _slug(town)


def _slug(town: str) -> str:
    text = town.strip().lower().replace("/", "_").replace(" ", "_")
    return "".join(char if char.isalnum() or char == "_" else "_" for char in text)


def _dump_preprocessor(preparer, folder: Path) -> None:
    import joblib

    joblib.dump(preparer, folder / "preprocessor.joblib")


def _save_summary(model: str, genome: dict, metrics: dict) -> None:
    folder = ROOT / "artifacts" / "mean_residual" / "residuals" / model / "towns"
    folder.mkdir(parents=True, exist_ok=True)
    payload = {
        "feature_hash": feature_hash(list(genome["features"])),
        "model": model,
        "source_genome": f"models/{model}/best/genome.json",
        "split": "one residual model per TOWN",
        **metrics,
    }
    (folder / "metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _shared_search(model: str) -> float:
    path = ROOT / "artifacts" / "mean_residual" / "residuals" / model / "metrics.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return float(payload["search_rmse"])


if __name__ == "__main__":
    import argparse
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
