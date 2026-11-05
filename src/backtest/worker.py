"""One frozen origin per independent process; no CLI model/row/budget overrides."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import traceback

from src.backtest.execution import Attempt, LockBusy, RunLock, atomic_json, replace_file, utc_now


def run_origin(args, attempt):
    import numpy as np
    from src.backtest.checkpoints import begin_checkpoint, fit_contract, load_checkpoint, save_checkpoint
    from src.backtest.data import frame_hash, history_at, load_cache, query_at
    from src.backtest.decomposed import fit_decomposed
    from src.backtest.frozen import DIRECT_MODELS, DEVELOPMENT_MODELS, prediction_identity, read_manifest, verify_runtime
    from src.backtest.models import fit_direct, fit_simple
    from src.backtest.protocol import history_hash
    from src.backtest.resources import ResourceMeter, memory_mib
    from src.backtest.scoring import attach_truth, metrics
    from src.backtest.selection import validate_origin, verify_selection, write_origin_receipt

    root = Path(args.run_dir).resolve()
    manifest = read_manifest(root, args.manifest_hash)
    verify_runtime(manifest)
    config = manifest["protocol"]
    if args.origin not in manifest["windows"] or manifest["windows"][args.origin]["phase"] != args.phase:
        raise ValueError("origin/phase is not part of the frozen run")
    if args.phase == "evaluation":
        if not args.selection_hash:
            raise ValueError("formal worker requires a pinned frozen development selection")
        choice = verify_selection(root, manifest, args.selection_hash)["selected_trend"]
        names, trend_names = [*DIRECT_MODELS, f"decomposed_{choice}"], [choice]
    else:
        names, trend_names = list(DEVELOPMENT_MODELS), ["flat", "damped"]
    folder = root / args.phase / args.origin
    folder.mkdir(parents=True, exist_ok=True)
    attempt.mirrors.append(folder / "status.json")
    attempt.update("running", phase=args.phase, origin=args.origin, models=names, stage="load_allowed_history")
    base = load_cache("train")
    history, query = history_at(base, args.origin), query_at(base, args.origin, config["max_horizon"])
    template = prediction_identity(base, args.origin, config)
    planned = manifest["windows"][args.origin]
    if len(history) != planned["history_rows"] or len(query) != planned["query_rows"] or frame_hash(template) != planned["identity_sha256"]:
        raise ValueError("full history/query rows disagree with frozen target-free plan")
    history_sha = history_hash(history)
    completed_before = (folder / "origin_complete.json").exists()
    if completed_before:
        validate_origin(root, manifest, args.phase, args.origin, names)
    outcomes = {}

    def contract(name):
        return fit_contract(name, args.origin, history, query, config, manifest["environment"])

    def ensure_unchanged():
        verify_runtime(manifest)
        if history_hash(history_at(base, args.origin)) != history_sha:
            raise ValueError("permitted historical labels changed during worker")
        if args.phase == "evaluation":
            verify_selection(root, manifest, args.selection_hash)

    def prepare(name):
        restored = load_checkpoint(folder / name, contract(name))
        if restored is not None:
            bundle, predictions, observations = restored
            if not np.array_equal(predictions.row_id.to_numpy(), query.row_id.to_numpy()):
                raise ValueError("checkpoint prediction row alignment differs")
            outcomes[name] = predictions
            print(f"restored {args.phase}/{args.origin}/{name}", flush=True)
            del bundle
            return False
        if completed_before:
            raise ValueError("completed origin is missing a complete model checkpoint")
        begin_checkpoint(folder / name, contract(name))
        return True

    def publish(name, bundle, predictions, observations):
        ensure_unchanged()
        save_checkpoint(folder / name, contract(name), bundle, predictions, observations)
        loaded, saved, _ = load_checkpoint(folder / name, contract(name))
        if loaded is not None:
            np.testing.assert_array_equal(loaded.predict(query), saved.prediction.to_numpy())
        outcomes[name] = saved
        attempt.update("running", stage="model_checkpoint_complete", model=name, completed_models=list(outcomes))
        print(f"completed {args.phase}/{args.origin}/{name}: selected={observations.get('selected_iterations')}", flush=True)

    for name in DIRECT_MODELS:
        if not prepare(name):
            continue
        ensure_unchanged()
        attempt.update("running", stage="fit_direct", model=name, completed_models=list(outcomes))
        print(f"fit {name}: history={len(history)} query={len(query)} cap={config['catboost']['iterations']}", flush=True)
        with ResourceMeter() as meter:
            if name == "simple_recent":
                bundle, values = None, fit_simple(history, query, args.origin, config)
            else:
                bundle, values = fit_direct(history, query, args.origin, name, config)
        prediction = template.copy()
        prediction["prediction"] = values
        observations = {"resources": meter.result, "selected_iterations": bundle.audit["selected_iterations"] if bundle else None,
                        "inner_executed_iterations": bundle.audit["inner"]["executed_iterations"] if bundle else None,
                        "fit_phase_timings": bundle.timings if bundle else None}
        publish(name, bundle, prediction, observations)
        del bundle
    needed = [name for name in trend_names if prepare(f"decomposed_{name}")]
    if needed:
        ensure_unchanged()
        attempt.update("running", stage="fit_decomposed_batch", model=None, active_candidates=needed, completed_models=list(outcomes))
        def finished(name, bundle, components):
            prediction = template.copy()
            for column in ["prediction", "property_component", "market_component", "unclipped_prediction"]:
                prediction[column] = components[column].to_numpy()
            observations = {"selected_iterations": bundle.audit["selected_iterations"],
                            "inner_executed_iterations": bundle.audit["inner"]["executed_iterations"],
                            "fit_phase_timings": bundle.timings, "memory_at_publish": memory_mib()}
            publish(f"decomposed_{name}", bundle, prediction, observations)
        with ResourceMeter() as meter:
            fitted = fit_decomposed(history, query, args.origin, config, needed, on_complete=finished)
        atomic_json(attempt.path / "decomposed_batch_resources.json", meter.result)
        del fitted
    ensure_unchanged()
    if completed_before:
        attempt.finish(0, stage="restored_complete_origin", restored_models=list(outcomes))
        return
    predictions = template.copy()
    for name in names:
        predictions[name] = outcomes[name].prediction.to_numpy()
        for component in ["property_component", "market_component", "unclipped_prediction"]:
            if component in outcomes[name]:
                predictions[f"{name}__{component}"] = outcomes[name][component].to_numpy()
    def parquet_atomic(frame, path):
        temporary = path.with_suffix(path.suffix + ".partial")
        frame.to_parquet(temporary, index=False)
        replace_file(temporary, path)
    attempt.update("running", stage="score_completed_predictions", completed_models=list(outcomes))
    parquet_atomic(predictions, folder / "predictions_without_truth.parquet")
    scored = attach_truth(predictions, Path(manifest["dataset_root"]))
    parquet_atomic(scored, folder / "predictions.parquet")
    scores = metrics(scored, names, config)
    temporary = folder / "metrics.csv.partial"
    scores.to_csv(temporary, index=False)
    replace_file(temporary, folder / "metrics.csv")
    summary = {"manifest_sha256": manifest["manifest_sha256"], "config_sha256": manifest["protocol_sha256"],
               "origin": args.origin, "phase": args.phase, "models": names, "sampling": None,
               "allowed_history_sha256": history_sha, "history_rows": len(history), "query_rows": len(query),
               "completed_utc": utc_now(), "completion_attempt": attempt.attempt_id}
    atomic_json(folder / "origin_summary.json", summary)
    write_origin_receipt(folder, summary)
    validate_origin(root, manifest, args.phase, args.origin, names)
    attempt.finish(0, stage="origin_complete", completed_models=names)
    print(scores.loc[(scores.origin == args.origin) & (scores.horizon == "all")].to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--manifest-hash", required=True)
    parser.add_argument("--phase", choices=["development", "evaluation"], required=True)
    parser.add_argument("--origin", required=True)
    parser.add_argument("--selection-hash")
    parser.add_argument("--dispatch-id")
    args = parser.parse_args()
    root = Path(args.run_dir).resolve()
    try:
        from src.backtest.protocol import ROOT
        if not root.is_relative_to((ROOT / "artifacts/backtest").resolve()) or not (root / "frozen_protocol.json").is_file():
            raise ValueError("worker requires an owned frozen run within artifacts/backtest")
        with RunLock(root / "worker.lock"):
            attempt = Attempt(root / "attempts/worker", root / "worker_status.json", "worker",
                              phase=args.phase, origin=args.origin, command=sys.argv, dispatch_id=args.dispatch_id)
            try:
                run_origin(args, attempt)
            except BaseException as error:
                attempt.fail(error)
                raise
    except LockBusy as error:
        print(str(error), file=sys.stderr, flush=True)
        return 3
    except BaseException:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
