"""Read committed local research aggregates without importing model code."""
import argparse
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=["table", "json"], default="table")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fixed = json.loads((root/"results/fixed_split/metrics.json").read_text(encoding="utf-8"))
    origins = json.loads((root/"results/rolling_local/origin_results.json").read_text(encoding="utf-8"))
    aggregate = json.loads((root/"results/rolling_local/aggregate_results.json").read_text(encoding="utf-8"))
    data = {"fixed_split":fixed, "rolling_origins":[{"origin":x["origin"], "rows":x["rows"], **x["overall"]} for x in origins],
            "equal_origin":aggregate["equal_origin"]["h1_16"]}
    if args.format == "json":
        print(json.dumps(data, indent=2, ensure_ascii=True, allow_nan=False))
        return
    print("FIXED SPLIT | validation / repeatedly viewed confirmation | monthly rent SGD")
    for row in fixed:
        a,b=row["validation_rmse"],row["confirmation_rmse"]
        a="-" if a is None else f"{a:.4f}"
        b="-" if b is None else f"{b:.4f}"
        print(f"{row['experiment']:<28} {a:>10} {b:>10}  {row['status']}")
    print("\nLOCAL ROLLING | global CatBoost + time index | 16 months")
    for row in data["rolling_origins"]:
        print(f"{row['origin']}  RMSE={row['rmse']:.4f}  bias={row['bias_pred_minus_truth']:.4f}")
    print(f"Equal-origin RMSE={data['equal_origin']['rmse_equal_origin']:.4f}")
    print("\nStudies differ in periods, horizons and information assumptions. See README.md#results.")

if __name__ == "__main__":
    main()
