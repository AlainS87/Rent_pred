"""Checkpointed development-only experiments; formal evaluation is not exposed."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import traceback

import numpy as np
import pandas as pd

from src.backtest.checkpoints import begin_checkpoint, environment_snapshot, fit_contract, load_checkpoint, save_checkpoint
from src.backtest.data import history_at, load_cache, query_at
from src.backtest.decomposed import fit_decomposed
from src.backtest.models import fit_direct, fit_simple
from src.backtest.protocol import ROOT, ROW_ID, digest_json, history_hash, horizon, read_protocol
from src.backtest.resources import ResourceMeter, memory_mib, runtime_metadata
from src.backtest.run import monthly_sample, write_json


def run(args):
    protocol = read_protocol()
    if args.origin not in protocol["development_origins"]:
        raise ValueError("stage 3 permits development origins only; formal evaluation requires later review")
    config = copy.deepcopy(protocol)
    if args.smoke:
        config["catboost"]["iterations"] = args.iterations or 100
        config["catboost"]["early_stopping_rounds"] = min(20, config["catboost"]["early_stopping_rounds"])
    elif args.iterations is not None:
        raise ValueError("iteration overrides require explicit --smoke")
    output = Path(args.output).resolve()
    if not output.is_relative_to((ROOT / "artifacts/backtest").resolve()):
        raise ValueError("new experiment outputs must stay in artifacts/backtest")
    if output.exists() and not args.resume:
        raise ValueError("existing run requires explicit --resume; completed checkpoints are never overwritten")
    output.mkdir(parents=True, exist_ok=True)
    environment = environment_snapshot()
    spec = {"scope": "sampled development smoke" if args.smoke else "complete development origin, all eligible rows and unchanged 2000-iteration cap",
            "origin": args.origin, "config": config, "environment": environment,
            "source_protocol_sha256": digest_json(protocol),
            "sampling": {"history": args.max_train_rows, "query": args.max_query_rows} if args.smoke else None}
    spec_path = output / "run_spec.json"
    if spec_path.exists():
        if json.loads(spec_path.read_text(encoding="utf-8")) != spec:
            raise ValueError("run configuration/code/source/dependency snapshot mismatch on resume")
    else:
        write_json(spec_path, spec)
    base = load_cache("train")
    history = history_at(base, args.origin)
    query = query_at(base, args.origin, config["max_horizon"])
    if args.smoke:
        history, query = monthly_sample(history, args.max_train_rows), monthly_sample(query, args.max_query_rows)
    template = query[[ROW_ID, "RENT_APPROVAL_DATE"]].copy()
    template.insert(0, "origin", args.origin)
    template["horizon"] = horizon(query, args.origin)
    started, wall = datetime.now(timezone.utc).isoformat(), time.perf_counter()
    outcomes = {}

    def contract(name):
        return fit_contract(name, args.origin, history, query, config, environment)

    def ensure_unchanged():
        if environment_snapshot() != environment or digest_json(read_protocol()) != spec["source_protocol_sha256"]:
            raise ValueError("implementation/source/dependency/config changed during run")
        current = history_at(base, args.origin)
        current = current.loc[current.row_id.isin(history.row_id)]
        if history_hash(current) != history_hash(history):
            raise ValueError("permitted historical labels changed during run")

    def prepare(name):
        expected = contract(name)
        folder = output / name
        restored = load_checkpoint(folder, expected)
        if restored is not None:
            bundle, predictions, observations = restored
            if not np.array_equal(predictions.row_id.to_numpy(), query.row_id.to_numpy()):
                raise ValueError("checkpoint prediction row order mismatch")
            outcomes[name] = {"predictions": predictions, "observations": observations, "resumed": True}
            print(f"restored {name}: selected={observations.get('selected_iterations')}", flush=True)
            return False
        begin_checkpoint(folder, expected)
        write_json(output / "status.json", {"state": "running", "model": name, "origin": args.origin})
        return True

    def publish(name, bundle, predictions, observations):
        ensure_unchanged()
        save_checkpoint(output / name, contract(name), bundle, predictions, observations)
        restored, saved_predictions, _ = load_checkpoint(output / name, contract(name))
        if bundle is not None:
            np.testing.assert_array_equal(restored.predict(query), saved_predictions.prediction.to_numpy())
        outcomes[name] = {"predictions": saved_predictions, "observations": observations, "resumed": False}
        print(f"completed {name}: selected={observations.get('selected_iterations')}", flush=True)

    try:
        for name in ["simple_recent", "direct_legacy40", "direct_all93"]:
            if not prepare(name):
                continue
            ensure_unchanged()
            print(f"fit {name}: origin={args.origin}, history={len(history)}, query={len(query)}, cap={config['catboost']['iterations']}", flush=True)
            with ResourceMeter() as meter:
                if name == "simple_recent":
                    bundle, pred = None, fit_simple(history, query, args.origin, config)
                else:
                    bundle, pred = fit_direct(history, query, args.origin, name, config)
            predictions = template.copy()
            predictions["prediction"] = pred
            observations = {"resources": meter.result, "selected_iterations": bundle.audit["selected_iterations"] if bundle else None,
                            "inner_executed_iterations": bundle.audit["inner"]["executed_iterations"] if bundle else None,
                            "fit_phase_timings": bundle.timings if bundle else None}
            publish(name, bundle, predictions, observations)
            del bundle
        needed = [name for name in config["decomposed"]["trends"] if prepare(f"decomposed_{name}")]
        if needed:
            ensure_unchanged()
            print(f"fit decomposed {needed}: origin={args.origin}, history={len(history)}, query={len(query)}, cap={config['catboost']['iterations']}", flush=True)
            def finished(name, bundle, components):
                predictions = template.copy()
                for column in ["property_component", "market_component", "unclipped_prediction", "prediction"]:
                    predictions[column] = components[column].to_numpy()
                observations = {"selected_iterations": bundle.audit["selected_iterations"],
                                "inner_executed_iterations": bundle.audit["inner"]["executed_iterations"],
                                "fit_phase_timings": bundle.timings, "memory_at_publish": memory_mib()}
                publish(f"decomposed_{name}", bundle, predictions, observations)
            with ResourceMeter() as meter:
                decomposed = fit_decomposed(history, query, args.origin, config, needed, on_complete=finished)
            write_json(output / "decomposed_batch_resources.json", meter.result)
            del decomposed
        ensure_unchanged()
        predictions = template.copy()
        for name, entry in outcomes.items():
            predictions[name] = entry["predictions"].prediction.to_numpy()
            for component in ["property_component", "market_component", "unclipped_prediction"]:
                if component in entry["predictions"]:
                    predictions[f"{name}__{component}"] = entry["predictions"][component].to_numpy()
        predictions.to_parquet(output / "predictions_without_truth.parquet", index=False)
        from src.backtest.scoring import attach_truth, metrics
        scored = attach_truth(predictions)
        scored.to_parquet(output / "predictions.parquet", index=False)
        scores = metrics(scored, list(outcomes), config)
        scores.to_csv(output / "metrics.csv", index=False)
        write_json(output / "run.json", {**spec, "history_rows": len(history), "query_rows": len(query),
                   "allowed_history_sha256": history_hash(history), "started_utc": started,
                   "finished_utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.perf_counter() - wall,
                   "runtime": runtime_metadata(), "models": {name: {key: value for key, value in entry.items() if key != "predictions"} for name, entry in outcomes.items()}})
        write_json(output / "status.json", {"state": "complete", "origin": args.origin, "exit_status": 0})
        print(scores.loc[(scores.origin == args.origin) & (scores.horizon == "all")].to_string(index=False), flush=True)
    except BaseException as error:
        write_json(output / "status.json", {"state": "failed", "origin": args.origin, "exit_status": 1,
                                           "exception": repr(error), "traceback": traceback.format_exc()})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origin", default="2022-02")
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--max-train-rows", type=int, default=12000)
    parser.add_argument("--max-query-rows", type=int, default=3000)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
