"""Train the town-month ridge and the four residual models, then predict test.

    python -m mean_residual_model.run
"""

from pathlib import Path

import numpy as np
import pandas as pd

from mean_residual_model.regional import (
    CONFIRM_MONTHS,
    FIT_END,
    LEVELS,
    SEARCH_MONTHS,
    TownMonthMean,
    assert_known_cell_scores,
    save_regional,
)
from mean_residual_model.residual import (
    MODELS,
    fit_residual,
    load_best_genome,
    predict_residual,
    rent_scores,
    save_residual,
    stopped_size,
)
from src.pipeline.data import load_joined, time_split

ROOT = Path(__file__).resolve().parents[1]
SEED = 5228
TEST_ROWS = 50000
REGIONAL_DIR = ROOT / "mean_residual_model" / "reginal_mean_model"


def main() -> None:
    genomes = {model: load_best_genome(ROOT, model) for model in MODELS}
    columns = _columns(genomes)
    frame = load_joined("train", columns)
    split = time_split(frame, 3, None)
    _check_months(split)
    regional = TownMonthMean().fit(split.fit)
    cell_scores = {
        "search": regional.weighted_cell_rmse(split.search, SEARCH_MONTHS),
        "confirm": regional.weighted_cell_rmse(split.confirm, CONFIRM_MONTHS),
    }
    assert_known_cell_scores(cell_scores)
    save_regional(regional, REGIONAL_DIR, cell_scores)
    print(
        f"town-month ridge cells {regional.fit_cells}, "
        f"search {cell_scores['search']:.4f}, confirm {cell_scores['confirm']:.4f}"
    )

    periods = {"train": split.fit, "search": split.search, "confirm": split.confirm}
    level = {name: regional.predict(part) for name, part in periods.items()}
    rents = {name: part["MONTHLY_RENT"].to_numpy(dtype=np.float64) for name, part in periods.items()}
    y_fit = rents["train"] - level["train"]
    y_search = rents["search"] - level["search"]
    fitted = {}
    for model in MODELS:
        print(f"fitting {model} residual")
        preparer, bundle = fit_residual(
            model,
            genomes[model],
            split.fit,
            split.search,
            y_fit,
            y_search,
            SEED,
        )
        gap = {
            name: predict_residual(model, preparer, bundle, part)
            for name, part in periods.items()
        }
        metrics = rent_scores(rents, level, gap)
        metrics.update(stopped_size(model, bundle))
        metrics["rows"] = {name: int(len(part)) for name, part in periods.items()}
        save_residual(ROOT, model, genomes[model], preparer, bundle, metrics)
        fitted[model] = (preparer, bundle)
        print(
            f"{model} sum RMSE train {metrics['train_rmse']:.4f}, "
            f"search {metrics['search_rmse']:.4f}, confirm {metrics['confirm_rmse']:.4f}"
        )

    test = load_joined("test", columns)
    test_level = regional.predict(test)
    for model in MODELS:
        preparer, bundle = fitted[model]
        prediction = test_level + predict_residual(model, preparer, bundle, test)
        output = ROOT / "outputs" / "predictions" / f"mean_residual_{model}.csv"
        _write_kaggle(output, prediction)
        print(f"wrote {output.relative_to(ROOT).as_posix()}")


def _columns(genomes: dict[str, dict]) -> list[str]:
    columns = ["RENT_APPROVAL_DATE", "TOWN", *LEVELS]
    for genome in genomes.values():
        columns.extend(genome["features"])
    return list(dict.fromkeys(columns))


def _check_months(split) -> None:
    if split.fit_months[-1] != FIT_END:
        raise SystemExit(f"fit months end at {split.fit_months[-1]}, expected {FIT_END}")
    if tuple(split.search_months) != SEARCH_MONTHS:
        raise SystemExit(f"search months are {split.search_months}")
    if tuple(split.confirm_months) != CONFIRM_MONTHS:
        raise SystemExit(f"confirm months are {split.confirm_months}")


def _write_kaggle(path: Path, prediction: np.ndarray) -> None:
    values = np.asarray(prediction, dtype=np.float64)
    if len(values) != TEST_ROWS or not np.isfinite(values).all():
        raise SystemExit(f"test prediction has length {len(values)} and must be finite")
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame({"Id": np.arange(TEST_ROWS), "Predicted": values})
    table.to_csv(path, index=False)
    saved = pd.read_csv(path)
    if list(saved.columns) != ["Id", "Predicted"]:
        raise SystemExit(f"{path} header is {list(saved.columns)}")
    ids = saved["Id"].to_numpy()
    if len(saved) != TEST_ROWS or not np.array_equal(ids, np.arange(TEST_ROWS)):
        raise SystemExit(f"{path} ids are not 0..{TEST_ROWS - 1}")
    if not np.isfinite(pd.to_numeric(saved["Predicted"], errors="coerce")).all():
        raise SystemExit(f"{path} has a missing or non-numeric prediction")


if __name__ == "__main__":
    main()
