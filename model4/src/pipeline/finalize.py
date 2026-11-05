"""Train a larger model on the features already chosen in ``best/``.

This stage does not add or drop columns. Search RMSE picks the stopped
iterate. Confirm RMSE is recorded and does not choose that iterate.
"""

import json
import shutil
from pathlib import Path

import joblib

from src.pipeline.best import load_best, seed_best_from_champion
from src.pipeline.evolve import predict_model, train_model
from src.pipeline.fit import load_model_module
from src.pipeline.genome import write_features_file
from src.pipeline.paths import MODELS, ROOT
from src.pipeline.records import refresh_design

FINAL_ARCHITECTURE = {
    "mlp": {
        "hidden_sizes": [512, 256, 128],
        "dropout": 0.1,
        "learning_rate": 0.001,
        "epochs": 200,
        "batch_size": 1024,
        "patience": 15,
    },
    "xgb": {
        "max_depth": 8,
        "learning_rate": 0.05,
        "n_estimators": 2000,
        "early_stopping_rounds": 50,
    },
    "catboost": {
        "max_depth": 8,
        "learning_rate": 0.05,
        "n_estimators": 2000,
        "early_stopping_rounds": 50,
    },
    "ridge": {"alpha": 1.0},
}


def final_dir(model: str) -> Path:
    return MODELS / model / "final"


def load_final(model: str) -> dict | None:
    path = final_dir(model) / "record.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_final_weights(model: str):
    """Return the preprocessor, weights, genome, and run id stored in ``final/``."""
    folder = final_dir(model)
    record = load_final(model)
    if record is None:
        return None
    preprocessor = joblib.load(folder / "preprocessor.joblib")
    bundle = load_model_module(model).load_model(folder)
    genome = json.loads((folder / "genome.json").read_text(encoding="utf-8"))
    return preprocessor, bundle, genome, record["run_id"]


def finalize_model(model: str, seed: int = 5228) -> dict:
    seed_best_from_champion(model)
    best = load_best(model)
    if best is None:
        raise SystemExit(f"models/{model}/best does not exist; train a model first")
    features = list(best["champion"]["genome"]["features"])
    locked = {
        "architecture": FINAL_ARCHITECTURE[model],
        "preprocess": best["champion"]["genome"]["preprocess"],
    }
    print(f"finalize {model} on {len(features)} features from {best['run_id']}")
    trained = train_model(
        model,
        features,
        best["champion"].get("val_months") or 3,
        None,
        None,
        seed,
        locked=locked,
        commit_best=False,
        parent_run_id=best["run_id"],
        mutation={"axis": "finalize"},
        reason=(
            f"final fit on the features of {best['run_id']}; "
            "this stage does not select columns"
        ),
    )
    _copy_final(model, trained)
    refresh_design(model)
    predict_model(model, f"artifacts/predictions/{model}_final.csv", False, seed)
    print(
        f"final {model} search RMSE {trained['search_rmse']:.4f} "
        f"confirm RMSE {trained.get('confirm_rmse')}"
    )
    return trained


def _copy_final(model: str, champion: dict) -> None:
    folder = final_dir(model)
    staging = folder.with_name("final_staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    features = list(champion["genome"]["features"])
    source = MODELS / model / "ckpt" / champion["feature_hash"] / "runs" / champion["run_id"]
    weight = load_model_module(model).WEIGHT_NAME
    for name in ("genome.json", "metrics.json", "preprocessor.joblib", weight):
        shutil.copy2(source / name, staging / name)
    write_features_file(staging, features)
    shutil.copy2(ROOT / champion["log_path"], staging / "train_log.json")
    if champion.get("confirm_log"):
        shutil.copy2(ROOT / champion["confirm_log"], staging / "confirm_log.json")
    metrics = json.loads((staging / "metrics.json").read_text(encoding="utf-8"))
    parent = champion.get("parent_run_id") or metrics.get("parent_run_id")
    payload = {
        "confirm_log": champion.get("confirm_log"),
        "confirm_rmse": champion.get("confirm_rmse"),
        "feature_hash": champion["feature_hash"],
        "genome_hash": champion["genome_hash"],
        "log_path": champion["log_path"],
        "parent_run_id": parent,
        "run_id": champion["run_id"],
        "search_rmse": champion["search_rmse"],
        "source_best_run_id": parent,
        "train_rmse": champion["train_rmse"],
    }
    (staging / "record.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if folder.exists():
        shutil.rmtree(folder)
    staging.rename(folder)
