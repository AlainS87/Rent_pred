"""Target-free run plan and pinned manifests, without mutable fit defaults."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.backtest.checkpoints import environment_snapshot
from src.backtest.data import frame_hash, query_at
from src.backtest.execution import atomic_json, utc_now
from src.backtest.protocol import ROOT, ROW_ID, digest_json, horizon, months, read_protocol

DIRECT_MODELS = ["simple_recent", "direct_legacy40", "direct_all93"]
DEVELOPMENT_MODELS = [*DIRECT_MODELS, "decomposed_flat", "decomposed_damped"]


def prediction_identity(base: pd.DataFrame, origin: str, config: dict) -> pd.DataFrame:
    query = query_at(base, origin, config["max_horizon"])
    result = query[[ROW_ID, "RENT_APPROVAL_DATE"]].copy()
    result.insert(0, "origin", origin)
    result["horizon"] = horizon(query, origin)
    return result.reset_index(drop=True)


def make_manifest(config: dict, base: pd.DataFrame, environment: dict, dataset_root: Path = ROOT) -> dict:
    if len(config["development_origins"]) != 2 or len(set(config["development_origins"])) != 2:
        raise ValueError("exactly two distinct predeclared development origins are required")
    last_development = max(pd.Period(origin, freq="M") + 16 for origin in config["development_origins"])
    first_evaluation = min(pd.Period(origin, freq="M") + 1 for origin in config["evaluation_origins"])
    if last_development >= first_evaluation or config["max_horizon"] != 16:
        raise ValueError("development/selection labels must precede formal forecast labels, at horizon 16")
    windows = {}
    for phase, origins in [("development", config["development_origins"]), ("evaluation", config["evaluation_origins"])]:
        for origin in origins:
            identity = prediction_identity(base, origin, config)
            if identity.empty:
                raise ValueError("frozen origin has no eligible forecast rows")
            if phase == "development" and set(identity.horizon) != set(range(1, 17)):
                raise ValueError("both development origins require every month of the complete 16-month window")
            windows[origin] = {"phase": phase, "query_rows": len(identity), "identity_sha256": frame_hash(identity),
                               "history_rows": int((months(base) <= pd.Period(origin, freq="M")).sum())}
    value = {"version": 1, "created_utc": utc_now(), "protocol": config,
             "protocol_sha256": digest_json(config), "environment": environment, "windows": windows,
             "dataset_root": str(dataset_root.resolve()), "development_models": DEVELOPMENT_MODELS,
             "selection_rule": "minimum pooled development SSE/n; exact ties choose flat; both complete origins and candidates required"}
    value["manifest_sha256"] = digest_json(value)
    return value


def read_manifest(run_dir: Path, expected_hash: str) -> dict:
    value = json.loads((run_dir / "frozen_protocol.json").read_text(encoding="utf-8"))
    plain = {key: item for key, item in value.items() if key != "manifest_sha256"}
    if digest_json(plain) != value.get("manifest_sha256") or value["manifest_sha256"] != expected_hash:
        raise ValueError("frozen manifest checksum/pinned hash mismatch")
    if digest_json(value["protocol"]) != value["protocol_sha256"]:
        raise ValueError("frozen configuration checksum mismatch")
    return value


def verify_runtime(manifest: dict):
    if manifest["dataset_root"] != str(ROOT.resolve()):
        raise ValueError("production runner requires the approved workspace data root")
    if manifest["environment"] != environment_snapshot() or manifest["protocol_sha256"] != digest_json(read_protocol()):
        raise ValueError("frozen code/source/dependencies/protocol differ from current runtime")


def freeze_run(run_dir: Path):
    from src.backtest.data import load_cache
    if run_dir.exists():
        raise ValueError("freeze requires a new, empty run directory")
    config, environment = read_protocol(), environment_snapshot()
    manifest = make_manifest(config, load_cache("train"), environment)
    run_dir.mkdir(parents=True)
    atomic_json(run_dir / "frozen_protocol.json", manifest)
    return manifest
