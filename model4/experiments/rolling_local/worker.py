"""One fresh, serial local rolling backtest. No import-time model or data operations."""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

from .status_io import append_record, write_critical, write_status


def execute(config, output, started):
    compute_deadline = started + config["budget"]["wall_seconds"] - config["budget"]["report_reserve_seconds"]
    def deadline():
        if time.time() >= compute_deadline:
            raise TimeoutError("Configured model-computation budget exhausted")
    deadline()
    import numpy as np
    import pandas as pd
    import psutil
    from threadpoolctl import threadpool_info, threadpool_limits
    from .data import load_data, rank_for_phase, enforce_phase_info, address_keys
    from .metrics import evaluate_local, aggregate_results
    from .local_model import run_local
    assigned = config["resources"]["cpu_threads"]
    process = psutil.Process()
    if not hasattr(process, "cpu_affinity"):
        raise RuntimeError("CPU affinity control is required on this operating system")
    process.cpu_affinity(sorted(process.cpu_affinity())[:assigned])
    controller = threadpool_limits(limits=assigned)
    counts = {}
    n = len(config["origins"])
    limits = {"catboost": 2*n, "index": 2*n}
    context = {}
    def event(stage, **fields):
        deadline()
        if stage == "fit_enter":
            kind = fields.get("kind")
            if kind not in limits:
                raise ValueError(f"Unexpected fitted model: {kind}")
            counts[kind] = counts.get(kind, 0) + 1
            if counts[kind] > limits[kind]:
                raise RuntimeError(f"Model-fit allowance exceeded: {kind}")
        record = {"stage": stage, "epoch": time.time(), **context, **fields,
                  "fit_counts": dict(counts), "elapsed_seconds": time.time()-started}
        append_record(output / "events.jsonl", record)
        write_status(output / "status.json", {"status": "running", **record},
                       deadline_epoch=compute_deadline, warning_path=output / "status_io_warnings.jsonl")
    write_critical(output / "environment.json", {"python": sys.version, "numpy": np.__version__,
                   "pandas": pd.__version__, "device": "cpu",
                   "cpu_affinity": process.cpu_affinity(), "threadpools": threadpool_info()})
    results = []
    try:
        frame = load_data(config["raw_data"], config["feature_data"], output / "data",
                          expected_rows=config.get("expected_source_rows"), expected_hashes=config.get("input_sha256"))
        for origin in config["origins"]:
            deadline()
            period = pd.Period(origin, freq="M")
            inner_cutoff, outer_end = str(period - 3), str(period + 16)
            full = frame.loc[frame.month <= origin].copy().reset_index(drop=True)
            inner = frame.loc[frame.month <= inner_cutoff].copy().reset_index(drop=True)
            validation = frame.loc[(frame.month > inner_cutoff) & (frame.month <= origin)].copy().reset_index(drop=True)
            truth = frame.loc[(frame.month > origin) & (frame.month <= outer_end)].copy().reset_index(drop=True)
            outer = truth.drop(columns="MONTHLY_RENT").copy()
            root = output / "origins" / origin
            root.mkdir(parents=True, exist_ok=False)
            phases = {}
            for label, train, apply, cutoff in (("inner", inner, validation, inner_cutoff), ("full", full, outer, origin)):
                shared = root / "shared" / label
                shared.mkdir(parents=True)
                train, apply = rank_for_phase(train, apply, shared)
                phases[label] = (*enforce_phase_info(train, apply, cutoff, shared), cutoff)
            selected = None
            for phase in ("inner", "full"):
                context.update(origin=origin, phase=phase)
                train, apply, cutoff = phases[phase]
                dest = root / phase
                dest.mkdir(parents=True)
                prediction, selected = run_local(train.copy(), apply.copy(), dest, cutoff, selected, event, deadline)
                prediction = np.asarray(prediction, dtype=float).reshape(-1)
                if len(prediction) != len(apply) or not np.isfinite(prediction).all():
                    raise ValueError("Incomplete or nonfinite phase predictions")
                if phase == "full":
                    pd.DataFrame({"source_row_id": apply.source_row_id, "prediction": prediction}).to_csv(root / "outer_predictions.csv", index=False)
            result = evaluate_local(truth, prediction, origin, root,
                                   seen_addresses=set(address_keys(full)),
                                   expected_rows=config.get("expected_outer_rows", {}).get(origin))
            results.append(result)
            write_critical(output / "completed_results.json", results)
            event("origin_complete", completed_origins=len(results))
        aggregate = aggregate_results(results, output, planned_origins=config["origins"])
        deadline()
        write_critical(output / "result.json", {"status": "complete", "fit_counts": counts,
                       "elapsed_seconds": time.time()-started, "aggregate": aggregate})
    except BaseException as error:
        write_critical(output / "failure.json", {"status": "failed", **context, "fit_counts": counts,
                       "error_type": type(error).__name__, "error": str(error),
                       "traceback": traceback.format_exc(), "automatic_restart": False})
        raise
    finally:
        controller.restore_original_limits()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-spec", required=True)
    args = parser.parse_args()
    spec = json.loads(Path(args.run_spec).read_text(encoding="utf-8"))
    execute(spec["config"], Path(spec["output"]), spec["started_epoch"])


if __name__ == "__main__":
    main()
