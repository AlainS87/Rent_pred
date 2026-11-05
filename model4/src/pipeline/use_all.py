"""Fit one model on every original and auxiliary column.

Search RMSE chooses the stopped iterate. The snapshot is written to
``models/<model>/use_all_features`` and does not replace ``best/`` or ``final/``.
"""

import json
import shutil
from pathlib import Path

from src.pipeline.best import load_best
from src.pipeline.catalog import resolve_features
from src.pipeline.evolve import train_model
from src.pipeline.fit import load_model_module
from src.pipeline.genome import load_default_genome, write_features_file
from src.pipeline.paths import MODELS, ROOT
from src.pipeline.records import refresh_design

TREE_CAP = 2000
EPOCH_CAP = 20
ARCHITECTURE = {
    "xgb": {
        "max_depth": 6,
        "learning_rate": 0.05,
        "n_estimators": TREE_CAP,
        "early_stopping_rounds": 50,
    },
    "mlp": {
        "hidden_sizes": [128, 64],
        "dropout": 0.1,
        "learning_rate": 0.001,
        "epochs": EPOCH_CAP,
        "batch_size": 1024,
        "patience": 3,
    },
    "ridge": {"alpha": 1.0},
}


def use_all_features(model: str, seed: int = 5228) -> dict:
    if model not in ARCHITECTURE:
        raise SystemExit(f"unknown model: {model}")
    features = resolve_features("all", "")
    default = load_default_genome(model)
    best = load_best(model)
    parent = None if best is None else best["run_id"]
    print(f"use-all {model} on {len(features)} columns")
    trained = train_model(
        model,
        features,
        3,
        None,
        None,
        seed,
        locked={
            "architecture": ARCHITECTURE[model],
            "preprocess": default["preprocess"],
        },
        commit_best=False,
        parent_run_id=parent,
        mutation={"axis": "use_all_features"},
        reason=(
            "all original and auxiliary columns fit together; "
            "this fit does not select columns one at a time and does not replace best/"
        ),
    )
    folder = _copy_use_all(model, trained)
    refresh_design(model)
    record = json.loads((folder / "record.json").read_text(encoding="utf-8"))
    _print_record(model, record)
    return record


def _print_record(model: str, record: dict) -> None:
    print(
        f"use-all search RMSE {record['search_rmse']:.4f} "
        f"confirm RMSE {record['confirm_rmse']}"
    )
    if model == "xgb":
        print(f"best_iteration {record['best_iteration']} trees {record['trees_kept']}")
    elif model == "mlp":
        print(f"best_epoch {record['best_epoch']}")
    if record.get("cap_hit"):
        print("the cap was hit and the curve had not flattened")


def _copy_use_all(model: str, champion: dict) -> Path:
    folder = MODELS / model / "use_all_features"
    staging = folder.with_name("use_all_features_staging")
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
    payload = {
        "confirm_log": champion.get("confirm_log"),
        "confirm_rmse": champion.get("confirm_rmse"),
        "feature_hash": champion["feature_hash"],
        "genome_hash": champion["genome_hash"],
        "log_path": champion["log_path"],
        "parent_run_id": champion.get("parent_run_id"),
        "run_id": champion["run_id"],
        "search_rmse": champion["search_rmse"],
        "train_rmse": champion["train_rmse"],
    }
    payload.update(_stopped_iterate(model, staging))
    metrics = json.loads((staging / "metrics.json").read_text(encoding="utf-8"))
    if payload["parent_run_id"] is None:
        payload["parent_run_id"] = metrics.get("parent_run_id")
    (staging / "record.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if folder.exists():
        shutil.rmtree(folder)
    staging.rename(folder)
    return folder


def _stopped_iterate(model: str, folder: Path) -> dict:
    if model == "ridge":
        return {}
    bundle = load_model_module(model).load_model(folder)
    if model == "xgb":
        iteration = getattr(bundle, "best_iteration", None)
        if iteration is None:
            raise SystemExit("XGBoost use-all model has no best_iteration")
        best_iteration = int(iteration)
        cap_hit = best_iteration == TREE_CAP - 1
        payload = {
            "best_iteration": best_iteration,
            "cap_hit": cap_hit,
            "trees_kept": best_iteration + 1,
        }
    else:
        epoch = bundle.get("best_epoch") if isinstance(bundle, dict) else None
        if epoch is None:
            raise SystemExit("MLP use-all model has no best_epoch")
        best_epoch = int(epoch)
        cap_hit = best_epoch == EPOCH_CAP
        payload = {"best_epoch": best_epoch, "cap_hit": cap_hit}
    if payload["cap_hit"]:
        payload["cap_note"] = "the cap was hit and the curve had not flattened"
    return payload
