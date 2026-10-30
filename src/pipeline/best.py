"""Keep the run with the lowest search RMSE in ``models/<model>/best``."""

import json
import shutil
from pathlib import Path

from src.pipeline.fit import load_model_module, run_folder
from src.pipeline.genome import write_features_file
from src.pipeline.paths import MODELS, ROOT
from src.pipeline.records import load_champion, refresh_design, write_champion, write_log


def best_dir(model: str) -> Path:
    return MODELS / model / "best"


def load_best(model: str) -> dict | None:
    path = best_dir(model) / "record.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def seed_best_from_champion(model: str) -> dict | None:
    """Copy the current champion into ``best`` when that folder is empty."""
    if load_best(model) is not None:
        return load_best(model)
    champion = load_champion(model)
    if champion is None:
        return None
    _copy_snapshot(model, champion, previous_run_id=None)
    print(f"seeded {model} best from {champion['run_id']} search RMSE {champion['search_rmse']:.4f}")
    return load_best(model)


def publish_best(model: str, train_record: dict, champion: dict) -> bool:
    """Replace ``best`` only when this run has a lower search RMSE.

    A capped run does not replace an uncapped best. The compared run stays in
    ``docs/logs`` either way.
    """
    current = load_best(model)
    if current is None:
        _copy_snapshot(model, champion, previous_run_id=None)
        train_record["accepted"] = True
        train_record["reason"] = "first saved best; there was no earlier snapshot"
        write_log(model, train_record)
        refresh_design(model)
        print(f"saved first best {champion['run_id']}")
        return True
    if _capped_versus_full(train_record, current):
        _keep_current(model, train_record, current, "a capped run does not replace an uncapped best")
        return False
    if train_record["search_rmse"] < current["search_rmse"]:
        previous = current["run_id"]
        _copy_snapshot(model, champion, previous_run_id=previous)
        train_record["accepted"] = True
        train_record["parent_run_id"] = train_record.get("parent_run_id") or previous
        train_record["reason"] = (
            f"replaced best {previous} search RMSE {current['search_rmse']:.6f} "
            f"with {train_record['search_rmse']:.6f}"
        )
        write_log(model, train_record)
        refresh_design(model)
        print(f"replaced best {previous} with {champion['run_id']}")
        return True
    _keep_current(
        model,
        train_record,
        current,
        (
            f"kept best {current['run_id']} search RMSE {current['search_rmse']:.6f}; "
            f"this run search RMSE {train_record['search_rmse']:.6f} is not lower"
        ),
    )
    return False


def _capped_versus_full(train_record: dict, current: dict) -> bool:
    return train_record.get("max_rows") is not None and current.get("max_rows") is None


def _keep_current(model: str, train_record: dict, current: dict, reason: str) -> None:
    train_record["accepted"] = False
    train_record["parent_run_id"] = train_record.get("parent_run_id") or current["run_id"]
    train_record["reason"] = reason
    write_log(model, train_record)
    write_champion(model, current["champion"])
    refresh_design(model)
    print(reason)


def _copy_snapshot(model: str, champion: dict, previous_run_id: str | None) -> None:
    folder = best_dir(model)
    staging = folder.with_name("best_staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    features = list(champion["genome"]["features"])
    source = run_folder(model, features, champion["run_id"])
    weight = load_model_module(model).WEIGHT_NAME
    for name in ("genome.json", "metrics.json", "preprocessor.joblib", weight):
        shutil.copy2(source / name, staging / name)
    write_features_file(staging, features)
    _copy_log(champion["log_path"], staging / "train_log.json")
    if champion.get("confirm_log"):
        _copy_log(champion["confirm_log"], staging / "confirm_log.json")
    payload = {
        "champion": champion,
        "confirm_log": champion.get("confirm_log"),
        "confirm_rmse": champion.get("confirm_rmse"),
        "feature_hash": champion["feature_hash"],
        "genome_hash": champion["genome_hash"],
        "log_path": champion["log_path"],
        "max_rows": champion.get("max_rows"),
        "previous_run_id": previous_run_id,
        "run_id": champion["run_id"],
        "search_rmse": champion["search_rmse"],
        "train_rmse": champion["train_rmse"],
    }
    (staging / "record.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if folder.exists():
        shutil.rmtree(folder)
    staging.rename(folder)


def _copy_log(relative: str, dest: Path) -> None:
    shutil.copy2(ROOT / relative, dest)
