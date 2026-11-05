"""Aggregate completed prediction files only; never fits or invokes a model."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-spec", required=True)
    args = parser.parse_args()
    spec = json.loads(Path(args.run_spec).read_text(encoding="utf-8"))
    output = Path(spec["output"])
    results_file = output / "completed_results.json"
    results = json.loads(results_file.read_text(encoding="utf-8")) if results_file.exists() else []
    from .metrics import aggregate_results
    aggregate_results(results, output, planned_origins=spec["config"]["origins"])


if __name__ == "__main__":
    main()
