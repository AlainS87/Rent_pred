"""Validate complete development evidence before freezing one trend."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest.data import frame_hash, history_at, read_predictors, sha256
from src.backtest.execution import atomic_json
from src.backtest.frozen import DEVELOPMENT_MODELS, read_manifest
from src.backtest.protocol import TARGET, digest_json, history_hash
from src.backtest.scoring import attach_truth, metrics


def write_origin_receipt(folder: Path, summary: dict):
    files = ["predictions_without_truth.parquet", "predictions.parquet", "metrics.csv", "origin_summary.json"]
    files += [f"{model}/{file}" for model in summary["models"] for file in ["contract.json", "complete.json"]]
    value = {"manifest_sha256": summary["manifest_sha256"], "origin": summary["origin"], "phase": summary["phase"],
             "artifact_sha256": {name: sha256(folder / name) for name in files}}
    atomic_json(folder / "origin_complete.json", value)


def validate_origin(run_dir: Path, manifest: dict, phase: str, origin: str, models: list[str]):
    folder = run_dir / phase / origin
    receipt = json.loads((folder / "origin_complete.json").read_text(encoding="utf-8"))
    if receipt["manifest_sha256"] != manifest["manifest_sha256"] or receipt["origin"] != origin or receipt["phase"] != phase:
        raise ValueError("origin receipt is not part of the frozen run")
    required_files = {"predictions_without_truth.parquet", "predictions.parquet", "metrics.csv", "origin_summary.json"}
    required_files.update(f"{model}/{file}" for model in models for file in ["contract.json", "complete.json"])
    if set(receipt["artifact_sha256"]) != required_files:
        raise ValueError("origin receipt is missing required artifacts")
    for name, expected in receipt["artifact_sha256"].items():
        if sha256(folder / name) != expected:
            raise ValueError("origin prediction/metric/summary checksum mismatch")
    summary = json.loads((folder / "origin_summary.json").read_text(encoding="utf-8"))
    if summary["config_sha256"] != manifest["protocol_sha256"] or summary["models"] != models or summary["sampling"] is not None:
        raise ValueError("origin configuration/model set/sampling differs from frozen full experiment")
    if summary["manifest_sha256"] != manifest["manifest_sha256"] or summary["origin"] != origin or summary["phase"] != phase:
        raise ValueError("origin summary identity differs from frozen run")
    data_root = Path(manifest["dataset_root"])
    current_history = history_at(read_predictors(data_root, "train"), origin, data_root)
    if history_hash(current_history) != summary["allowed_history_sha256"]:
        raise ValueError("allowed historical targets changed after origin completion")
    frame = pd.read_parquet(folder / "predictions.parquet")
    if not {TARGET, *models}.issubset(frame):
        raise ValueError("origin is missing a required candidate/truth column")
    identity = frame[["origin", "row_id", "RENT_APPROVAL_DATE", "horizon"]]
    window = manifest["windows"][origin]
    if len(frame) != window["query_rows"] or frame_hash(identity) != window["identity_sha256"] or frame.duplicated(["origin", "row_id"]).any():
        raise ValueError("incomplete or misaligned origin prediction rows")
    if phase == "development" and set(frame.horizon) != set(range(1, 17)):
        raise ValueError("development forecast does not contain all 16 months")
    if not np.isfinite(frame[[TARGET, *models]].to_numpy(dtype=float)).all():
        raise ValueError("a candidate is missing finite predictions or truth")
    for model in models:
        model_dir = folder / model
        contract = json.loads((model_dir / "contract.json").read_text(encoding="utf-8"))
        plain = {k: v for k, v in contract.items() if k != "contract_sha256"}
        if digest_json(plain) != contract["contract_sha256"]:
            raise ValueError("model checkpoint contract checksum mismatch")
        if (contract["model"] != model or contract["origin"] != origin or contract["config"] != manifest["protocol"]
            or contract["config_sha256"] != manifest["protocol_sha256"] or contract["environment"] != manifest["environment"]
            or contract["allowed_history_sha256"] != summary["allowed_history_sha256"]
            or contract["query_rows"] != window["query_rows"] or contract["history_rows"] != window["history_rows"]):
            raise ValueError("model checkpoint differs from frozen configuration/history/rows")
        complete = json.loads((model_dir / "complete.json").read_text(encoding="utf-8"))
        if complete["contract_sha256"] != contract["contract_sha256"]:
            raise ValueError("model completion contract mismatch")
        required = {"predictions.parquet", "observations.json"}
        if model != "simple_recent":
            required.update(["bundle.joblib", "audit.json"])
        if not required.issubset(complete["artifact_sha256"]):
            raise ValueError("model checkpoint is missing required fitted artifacts")
        for file, expected in complete["artifact_sha256"].items():
            if sha256(model_dir / file) != expected:
                raise ValueError("model checkpoint artifact checksum mismatch")
        individual = pd.read_parquet(model_dir / "predictions.parquet")
        if frame_hash(individual[["origin", "row_id", "RENT_APPROVAL_DATE", "horizon"]]) != window["identity_sha256"]:
            raise ValueError("individual candidate rows do not match the full frozen window")
        if not np.array_equal(individual.prediction.to_numpy(), frame[model].to_numpy()):
            raise ValueError("combined candidate predictions differ from fitted checkpoint")
    # Re-read only the declared phase's scoring rows. Models never import this.
    truth = attach_truth(frame.drop(columns=TARGET), Path(manifest["dataset_root"]))[TARGET].to_numpy()
    np.testing.assert_array_equal(frame[TARGET].to_numpy(), truth)
    recalculated = metrics(frame, models, manifest["protocol"])
    saved = pd.read_csv(folder / "metrics.csv")
    pd.testing.assert_frame_equal(recalculated.reset_index(drop=True), saved.reset_index(drop=True), check_exact=False, rtol=1e-12, atol=1e-6)
    return frame, receipt


def derive_selection(run_dir: Path, manifest: dict) -> dict:
    pooled = {name: {"sse": 0.0, "n": 0} for name in ["flat", "damped"]}
    evidence = {}
    if len(manifest["protocol"]["development_origins"]) != 2:
        raise ValueError("selection requires both predeclared development origins")
    for origin in manifest["protocol"]["development_origins"]:
        frame, receipt = validate_origin(run_dir, manifest, "development", origin, DEVELOPMENT_MODELS)
        evidence[origin] = receipt
        for name in pooled:
            error = frame[f"decomposed_{name}"].to_numpy(dtype=float) - frame[TARGET].to_numpy(dtype=float)
            pooled[name]["sse"] += float(np.dot(error, error))
            pooled[name]["n"] += len(error)
    if pooled["flat"]["n"] != pooled["damped"]["n"] or not pooled["flat"]["n"]:
        raise ValueError("trend candidates do not cover the identical complete development rows")
    choice = min(["flat", "damped"], key=lambda name: (pooled[name]["sse"] / pooled[name]["n"], name != "flat"))
    value = {"manifest_sha256": manifest["manifest_sha256"], "config_sha256": manifest["protocol_sha256"],
             "selected_trend": choice, "pooled_development": pooled, "development_evidence": evidence,
             "selection_rule": manifest["selection_rule"]}
    value["selection_sha256"] = digest_json(value)
    return value


def select_and_freeze(run_dir: Path, manifest: dict) -> dict:
    expected = derive_selection(run_dir, manifest)
    path = run_dir / "selected_trend.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != expected:
            raise ValueError("existing selection was tampered or development evidence changed")
    else:
        atomic_json(path, expected)
    return expected


def verify_selection(run_dir: Path, manifest: dict, expected_hash: str) -> dict:
    actual = json.loads((run_dir / "selected_trend.json").read_text(encoding="utf-8"))
    plain = {key: item for key, item in actual.items() if key != "selection_sha256"}
    if digest_json(plain) != actual.get("selection_sha256") or actual["selection_sha256"] != expected_hash:
        raise ValueError("selection checksum/pinned hash mismatch")
    if actual != derive_selection(run_dir, manifest):
        raise ValueError("selection does not match both complete frozen development windows")
    return actual
