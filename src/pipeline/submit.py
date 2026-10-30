"""Write Kaggle files from ``final/`` and from an all-label refit of that genome.

The refit keeps the XGBoost round and the MLP epoch already chosen on the
search months. It does not write a search RMSE, and it does not replace
``best/`` or ``final/``.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.data import load_joined, time_split
from src.pipeline.evolve import _new_run_id, predict_model
from src.pipeline.finalize import load_final_weights
from src.pipeline.fit import fit_genome, predict_frame, save_run
from src.pipeline.paths import ROOT, rel
from src.pipeline.records import design_path, log_path_for, read_logs, refresh_design, write_log

TEST_ROWS = 50000


def submit_model(model: str, seed: int = 5228) -> dict:
    loaded = load_final_weights(model)
    if loaded is None:
        raise SystemExit(f"models/{model}/final does not exist; finalize the model first")
    _preprocessor, bundle, genome, parent_run_id = loaded
    final_path = f"outputs/predictions/{model}_final.csv"
    print(f"submit {model} from {parent_run_id}")
    predict_model(model, final_path, False, seed)
    _check_kaggle(ROOT / final_path)

    refit_genome = _refit_genome(model, genome, bundle, seed)
    columns = list(refit_genome["features"])
    frame = load_joined("train", columns)
    print(f"fit submit on {len(frame)} labeled rows")
    trial = fit_genome(model, refit_genome, frame, frame, seed)
    existing = {row["run_id"] for row in read_logs(model)}
    run_id = _new_run_id(model, trial["genome_hash"], existing)
    record = {
        "accepted": None,
        "command": "submit",
        "design_path": rel(design_path(model)),
        "feature_hash": trial["feature_hash"],
        "genome": json.loads(json.dumps(refit_genome)),
        "genome_hash": trial["genome_hash"],
        "labeled_rows": int(len(frame)),
        "log_path": log_path_for(model, run_id),
        "mutation": {"axis": "submit"},
        "parent_run_id": parent_run_id,
        "reason": (
            f"refit of {parent_run_id} on every labeled month; "
            "this stage does not select columns and was not scored on held-out MONTHLY_RENT"
        ),
        "run_id": run_id,
        "train_rmse": trial["train_rmse"],
    }
    save_run(model, refit_genome, trial, record)
    write_log(model, record)
    print(f"  {run_id} train RMSE {trial['train_rmse']:.4f}")

    refit_path = ROOT / "outputs" / "predictions" / f"{model}_refit.csv"
    prediction = _write_kaggle(model, trial["preprocessor"], trial["bundle"], columns, refit_path)
    _log_prediction(model, refit_genome, run_id, refit_path, prediction, refit_all=True)
    refresh_design(model)
    print(f"wrote {len(prediction)} predictions to {rel(refit_path)}")
    return record


def _refit_genome(model: str, genome: dict, bundle, seed: int) -> dict:
    copied = json.loads(json.dumps(genome))
    if model == "xgb":
        iteration = getattr(bundle, "best_iteration", None)
        if iteration is None:
            raise SystemExit("final XGBoost model has no best_iteration")
        trees = int(iteration) + 1
        copied["architecture"]["n_estimators"] = trees
        copied["architecture"]["early_stopping_rounds"] = 0
        print(f"refit xgb for {trees} trees with early stopping off")
    elif model == "catboost":
        iteration = bundle.get_best_iteration()
        if iteration is None:
            raise SystemExit("final CatBoost model has no best iteration")
        trees = int(iteration) + 1
        copied["architecture"]["n_estimators"] = trees
        copied["architecture"]["early_stopping_rounds"] = 0
        print(f"refit catboost for {trees} iterations with early stopping off")
    elif model == "mlp":
        epoch = _recover_mlp_epoch(genome, seed)
        copied["architecture"]["epochs"] = int(epoch)
        copied["architecture"]["patience"] = 0
        print(f"refit mlp for {epoch} epochs with patience off")
    elif model == "ridge":
        print("refit ridge on every labeled row")
    else:
        raise SystemExit(f"unknown model: {model}")
    return copied


def _recover_mlp_epoch(genome: dict, seed: int) -> int:
    """Fit the final MLP once on the original split and read the kept epoch.

    The weights from this pass are discarded. They are not a champion.
    """
    frame = load_joined("train", list(genome["features"]))
    split = time_split(frame, 3, None)
    print("recover mlp epoch on the original fit/search split")
    trial = fit_genome("mlp", genome, split.fit, split.search, seed)
    epoch = int(trial["bundle"]["best_epoch"])
    if epoch < 1:
        raise SystemExit("MLP recovery did not record an epoch")
    print(f"recovered mlp best_epoch {epoch}")
    return epoch


def _write_kaggle(model: str, preprocessor, bundle, columns: list[str], output: Path) -> np.ndarray:
    test = load_joined("test", columns)
    prediction = predict_frame(model, preprocessor, bundle, test, columns)
    output.parent.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame({"Id": np.arange(len(prediction)), "Predicted": prediction})
    table.to_csv(output, index=False)
    _check_kaggle(output)
    return prediction


def _check_kaggle(path: Path) -> None:
    table = pd.read_csv(path)
    if list(table.columns) != ["Id", "Predicted"]:
        raise SystemExit(f"{path} header is {list(table.columns)}")
    if len(table) != TEST_ROWS:
        raise SystemExit(f"{path} has {len(table)} rows, expected {TEST_ROWS}")
    ids = table["Id"].to_numpy()
    if ids.dtype.kind not in {"i", "u"} or not np.array_equal(ids, np.arange(TEST_ROWS)):
        raise SystemExit(f"{path} ids are not 0..{TEST_ROWS - 1}")
    values = pd.to_numeric(table["Predicted"], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise SystemExit(f"{path} has a missing or non-numeric prediction")


def _log_prediction(model, genome, used_run, output: Path, prediction: np.ndarray, refit_all: bool) -> None:
    from src.pipeline.genome import feature_hash, genome_hash

    existing = {row["run_id"] for row in read_logs(model)}
    run_id = _new_run_id(model, "predict00", existing)
    record = {
        "champion_run_id": used_run,
        "command": "predict",
        "design_path": rel(design_path(model)),
        "feature_hash": feature_hash(list(genome["features"])),
        "genome": json.loads(json.dumps(genome)),
        "genome_hash": genome_hash(genome),
        "log_path": log_path_for(model, run_id),
        "n": int(len(prediction)),
        "output": rel(output),
        "prediction_max": float(np.max(prediction)),
        "prediction_mean": float(np.mean(prediction)),
        "prediction_min": float(np.min(prediction)),
        "prediction_std": float(np.std(prediction)),
        "refit_all": bool(refit_all),
        "run_id": run_id,
    }
    write_log(model, record)
