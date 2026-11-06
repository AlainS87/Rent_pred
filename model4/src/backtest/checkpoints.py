"""Per-model immutable contracts and checksum-verified resumable checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
import os

import joblib
import pandas as pd

from src.backtest.data import frame_hash, sha256, source_hashes
from src.backtest.protocol import ROOT, TARGET, digest_json, history_hash
from src.backtest.resources import runtime_metadata
from src.backtest.execution import atomic_json, replace_file


def implementation_hashes(root: Path = ROOT) -> dict:
    paths = sorted((root / "src/backtest").glob("*.py"))
    paths += sorted((root / "src/features").glob("*.py"))
    paths += [root / "src/pipeline/preprocess.py", root / "src/pipeline/rules.py",
              root / "src/paths.py", root / "configs/__init__.py", root / "pyproject.toml"]
    if (root / "scripts/backtest_launch.ps1").exists():
        paths.append(root / "scripts/backtest_launch.ps1")
    return {str(path.relative_to(root)).replace("\\", "/"): sha256(path) for path in paths}


def environment_snapshot(root: Path = ROOT) -> dict:
    runtime = runtime_metadata()
    return {"implementation_sha256": implementation_hashes(root), "source_sha256": source_hashes(root, "train"),
            "python": runtime["python"], "executable": runtime["executable"], "packages": runtime["packages"]}


def fit_contract(model_name: str, origin: str, history: pd.DataFrame, query: pd.DataFrame,
                 config: dict, environment: dict) -> dict:
    value = {"version": 1, "model": model_name, "origin": origin, "config": config,
             "config_sha256": digest_json(config), "environment": environment,
             "allowed_history_sha256": history_hash(history),
             "history_predictors_sha256": frame_hash(history.drop(columns=TARGET)),
             "query_predictors_sha256": frame_hash(query), "history_rows": len(history), "query_rows": len(query)}
    # Serializing creates a deep immutable-by-convention snapshot, not references
    # to a caller's subsequently mutable configuration dictionaries.
    value = json.loads(json.dumps(value, allow_nan=False))
    value["contract_sha256"] = digest_json(value)
    return value


def _json_atomic(path: Path, value):
    atomic_json(path, value)


def begin_checkpoint(directory: Path, contract: dict):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "contract.json"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous != contract:
            raise ValueError("checkpoint configuration/history/query/code/dependency/source contract mismatch")
    else:
        _json_atomic(path, contract)


def save_checkpoint(directory: Path, contract: dict, bundle, predictions: pd.DataFrame, observations: dict):
    begin_checkpoint(directory, contract)
    if (directory / "complete.json").exists():
        raise ValueError("refusing to overwrite a completed checkpoint")
    if bundle is not None:
        if bundle.origin != contract["origin"] or bundle.audit["model"] != contract["model"] or bundle.audit["allowed_history_sha256"] != contract["allowed_history_sha256"] or bundle.audit["config_sha256"] != contract["config_sha256"]:
            raise ValueError("bundle does not belong to checkpoint's origin/history/config")
        temporary = directory / "bundle.joblib.partial"
        joblib.dump(bundle, temporary)
        replace_file(temporary, directory / "bundle.joblib")
    temporary = directory / "predictions.parquet.partial"
    predictions.to_parquet(temporary, index=False)
    replace_file(temporary, directory / "predictions.parquet")
    _json_atomic(directory / "observations.json", observations)
    files = ["predictions.parquet", "observations.json"] + (["bundle.joblib"] if bundle is not None else [])
    if bundle is not None:
        _json_atomic(directory / "audit.json", bundle.audit)
        files.append("audit.json")
        if hasattr(bundle, "index"):
            temporary = directory / "market_index.csv.partial"
            bundle.index.index_.to_csv(temporary, index=False)
            replace_file(temporary, directory / "market_index.csv")
            files.append("market_index.csv")
    _json_atomic(directory / "complete.json", {"contract_sha256": contract["contract_sha256"],
                                               "artifact_sha256": {file: sha256(directory / file) for file in files}})


def load_checkpoint(directory: Path, expected_contract: dict):
    if not directory.exists():
        return None
    begin_checkpoint(directory, expected_contract)
    path = directory / "complete.json"
    if not path.exists():
        return None  # same-contract incomplete work may be recomputed, never accepted
    complete = json.loads(path.read_text(encoding="utf-8"))
    if complete["contract_sha256"] != expected_contract["contract_sha256"]:
        raise ValueError("checkpoint completion contract mismatch")
    for file, expected in complete["artifact_sha256"].items():
        if sha256(directory / file) != expected:
            raise ValueError("checkpoint artifact checksum mismatch")
    bundle = joblib.load(directory / "bundle.joblib") if "bundle.joblib" in complete["artifact_sha256"] else None
    if bundle is not None and (bundle.origin != expected_contract["origin"] or bundle.audit["model"] != expected_contract["model"] or bundle.audit["allowed_history_sha256"] != expected_contract["allowed_history_sha256"] or bundle.audit["config_sha256"] != expected_contract["config_sha256"]):
        raise ValueError("checkpoint bundle origin/history/config mismatch")
    predictions = pd.read_parquet(directory / "predictions.parquet")
    observations = json.loads((directory / "observations.json").read_text(encoding="utf-8"))
    return bundle, predictions, observations
