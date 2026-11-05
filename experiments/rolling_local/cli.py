"""Portable entry point; help and configuration validation do not train models."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .config import read_config
from .status_io import write_critical


def run(config, output):
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("A fresh empty output directory is required; no automatic recovery")
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with (output / "launch_guard.json").open("x", encoding="utf-8") as guard:
        json.dump({"pid": os.getpid(), "started_epoch": started, "automatic_restart": False}, guard)
    compute_end = started + config["budget"]["wall_seconds"] - config["budget"]["report_reserve_seconds"]
    spec = {"config": config, "output": str(output), "started_epoch": started,
            "compute_deadline_epoch": compute_end,
            "hard_deadline_epoch": started + config["budget"]["wall_seconds"]}
    path = output / "run_spec.json"
    write_critical(path, spec)
    env = dict(os.environ)
    threads = str(config["resources"]["cpu_threads"])
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = threads
    env["CUDA_VISIBLE_DEVICES"] = ""
    with (output / "worker.log").open("w", encoding="utf-8") as log:
        child = subprocess.Popen([sys.executable, "-m", "experiments.rolling_local.worker", "--run-spec", str(path)],
                                 stdout=log, stderr=subprocess.STDOUT, env=env,
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        write_critical(output / "worker_identity.json", {"pid": child.pid, "started_epoch": started})
        stopped = False
        try:
            while child.poll() is None:
                if time.time() >= compute_end:
                    stopped = True
                    child.kill()
                    child.wait(timeout=10)
                    break
                time.sleep(min(0.5, max(0.01, compute_end-time.time())))
        except BaseException:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=10)
            raise
    results_file = output / "completed_results.json"
    results = json.loads(results_file.read_text(encoding="utf-8")) if results_file.exists() else []
    report_status = "already_complete"
    if not (output / "aggregate_metrics.json").exists():
        remaining = spec["hard_deadline_epoch"] - time.time() - 1.0
        report_status = "no_report_time_remaining"
        if remaining > 0:
            # Reporting consumes the reserve and reads saved results only. It is
            # supervised separately so a report cannot extend the total budget.
            with (output / "report.log").open("w", encoding="utf-8") as log:
                reporter = subprocess.Popen([sys.executable, "-m", "experiments.rolling_local.report", "--run-spec", str(path)],
                                            stdout=log, stderr=subprocess.STDOUT, env=env,
                                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                try:
                    reporter.wait(timeout=remaining)
                    report_status = "complete" if reporter.returncode == 0 else "failed"
                except subprocess.TimeoutExpired:
                    reporter.kill()
                    reporter.wait(timeout=10)
                    report_status = "stopped_deadline"
    write_critical(output / "supervisor_result.json", {"status": "stopped_deadline" if stopped else "complete" if child.returncode == 0 else "failed",
                   "returncode": child.returncode, "completed_origins": [p["origin"] for p in results],
                   "planned_origins": config["origins"], "elapsed_seconds": time.time()-started,
                   "automatic_restart": False, "report_status": report_status})
    return 124 if stopped else child.returncode


def main(argv=None):
    parser = argparse.ArgumentParser(description="CPU local rental forecasting backtest with history-only capacity selection.")
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("validate", help="Validate configuration only; no data preparation or fitting")
    check.add_argument("--config", required=True)
    execute = sub.add_parser("run", help="Explicitly execute one fresh configured local backtest")
    execute.add_argument("--config", required=True)
    execute.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    config = read_config(args.config)
    if args.command == "validate":
        print(json.dumps({"valid": True, "origins": config["origins"], "config_sha256": config["config_sha256"], "models_fitted": 0}))
        return 0
    return run(config, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
