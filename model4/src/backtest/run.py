"""Stage-2 cache and explicitly sampled development smoke commands only."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.backtest.data import build_cache, history_at, load_cache, query_at, source_hashes, sha256
from src.backtest.models import fit_direct, fit_simple, load_bundle
from src.backtest.protocol import ROOT, ROW_ID, horizon, read_protocol
from src.backtest.resources import ResourceMeter, runtime_metadata


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def monthly_sample(frame: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    """Stable systematic sample within months, retaining temporal coverage."""
    if len(frame) <= max_rows:
        return frame.copy()
    pieces = []
    for _, group in frame.reset_index(drop=True).sort_values(ROW_ID).groupby("RENT_APPROVAL_DATE", sort=True):
        n = max(1, int(np.floor(max_rows * len(group) / len(frame))))
        positions = np.unique(np.rint(np.linspace(0, len(group) - 1, n)).astype(int))
        pieces.append(group.iloc[positions])
    return pd.concat(pieces).sort_values(ROW_ID).copy()


def smoke(args):
    protocol = read_protocol()
    if args.origin not in protocol["development_origins"]:
        raise ValueError("stage-2 smoke is limited to the predeclared development origins")
    config = copy.deepcopy(protocol)
    config["catboost"]["iterations"] = args.iterations
    config["catboost"]["early_stopping_rounds"] = min(20, protocol["catboost"]["early_stopping_rounds"])
    output = Path(args.output)
    if output.exists():
        raise ValueError("refusing to overwrite an existing run directory")
    output.mkdir(parents=True)
    implementation_paths = sorted((ROOT / "src/backtest").glob("*.py")) + [ROOT / "configs/backtest/protocol.json", ROOT / "models/catboost/default_genome.json", ROOT / "pyproject.toml"]
    implementation = {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in implementation_paths}
    sources = source_hashes(ROOT, "train")
    started = datetime.now(timezone.utc).isoformat()
    base = load_cache("train")
    history = monthly_sample(history_at(base, args.origin), args.max_train_rows)
    query = monthly_sample(query_at(base, args.origin, config["max_horizon"]), args.max_query_rows)
    predictions = query[[ROW_ID, "RENT_APPROVAL_DATE"]].copy()
    predictions.insert(0, "origin", args.origin)
    predictions["horizon"] = horizon(query, args.origin)
    observations = {}
    with ResourceMeter() as meter:
        predictions["simple_recent"] = fit_simple(history, query, args.origin, config)
    observations["simple_recent"] = meter.result
    for model_name in ("direct_legacy40", "direct_all93"):
        print(f"sampled smoke: {model_name}, history={len(history)}, query={len(query)}, max_iterations={args.iterations}", flush=True)
        with ResourceMeter() as meter:
            bundle, pred = fit_direct(history, query, args.origin, model_name, config)
            path = output / f"{model_name}.joblib"
            bundle.save(path)
            loaded = load_bundle(path).predict(query)
            np.testing.assert_allclose(pred, loaded, rtol=0, atol=0)
        predictions[model_name] = pred
        observations[model_name] = {**meter.result, "selected_iterations": bundle.audit["selected_iterations"],
                                    "inner_executed_iterations": bundle.audit["inner"]["executed_iterations"],
                                    "fit_phase_timings": bundle.timings,
                                    "saved_reload_exact_match": True}
        write_json(output / f"{model_name}_audit.json", bundle.audit)
        del bundle
    predictions.to_parquet(output / "predictions_without_truth.parquet", index=False)
    # This import is deliberately after every fit; scoring is separate from model code.
    from src.backtest.scoring import attach_truth, metrics
    scored = attach_truth(predictions)
    scored.to_parquet(output / "predictions.parquet", index=False)
    scores = metrics(scored, ["simple_recent", "direct_legacy40", "direct_all93"], config)
    scores.to_csv(output / "metrics.csv", index=False)
    write_json(output / "run.json", {"scope": "sampled development smoke; not definitive validation or candidate selection",
               "origin": args.origin, "history_rows": len(history), "query_rows": len(query),
               "config": config, "formal_protocol_unchanged": protocol, "resources": observations,
               "source_sha256": sources, "implementation_sha256": implementation,
               "raw_train_sha256_for_audit_only": sha256(ROOT / "data/raw/train.csv"),
               "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
               "runtime": runtime_metadata()})
    print(scores.loc[(scores.origin == args.origin) & (scores.horizon == "all")].to_string(index=False), flush=True)
    print(json.dumps(observations, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    cache = subs.add_parser("rebuild-cache")
    cache.add_argument("--output", default=str(ROOT / "artifacts/backtest/cache"))
    check = subs.add_parser("check-cache")
    check.add_argument("--output", default=str(ROOT / "artifacts/backtest/cache"))
    run = subs.add_parser("smoke")
    run.add_argument("--origin", default="2022-02")
    run.add_argument("--max-train-rows", type=int, default=12000)
    run.add_argument("--max-query-rows", type=int, default=3000)
    run.add_argument("--iterations", type=int, default=100)
    run.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "smoke":
        smoke(args)
    elif args.command == "rebuild-cache":
        with ResourceMeter() as meter:
            report = build_cache(cache_dir=Path(args.output))
        write_json(Path(args.output) / "build_runtime.json", {"resources": meter.result, "runtime": runtime_metadata()})
        print(json.dumps({key: {"rows": value["rows"], "columns": len(value["columns"]), "identity_sha256": value["identity_sha256"]}
                          for key, value in report.items()}, indent=2))
        print(json.dumps(meter.result, indent=2))
    else:
        for split in ("train", "test"):
            frame = load_cache(split, cache_dir=Path(args.output))
            print(f"cache verified: {split}, {len(frame)} rows, {len(frame.columns)} columns")


if __name__ == "__main__":
    main()
