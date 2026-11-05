"""Portable paths and explicit feature preparation for local experiments.

Importing this module does not read data, fit models, or create artifacts.
"""
from pathlib import Path
import json
import os
import sys
from src.paths import PROJECT_ROOT as REPO_ROOT, DATA_ROOT, ARTIFACT_ROOT as OUTPUT_ROOT, LOCAL_FEATURE_ROOT

CODE_DIR = Path(__file__).resolve().parent
SUPPORT_ROOT = CODE_DIR / "_support"
ARTIFACT_ROOT = OUTPUT_ROOT / "local"
BASELINE_DIR = Path(os.environ.get("RENTAL_BASELINE_DIR", str(ARTIFACT_ROOT / "baseline"))).expanduser().resolve()


def support_path(relative):
    """Resolve an internal library file or a user-supplied input role."""
    value = Path(relative)
    data_names = {"train.csv", "test.csv", "query.csv", "train_features.csv", "test_features.csv", "query_features.csv"}
    if value.parts and value.parts[0].lower() == "data":
        value = Path(*value.parts[1:])
        data_role = True
    else:
        data_role = bool(value.parts and (value.parts[0] in {"auxiliary", "auxiliary_features"} or value.name in data_names))
    if data_role:
        aliases = {"test.csv": "query.csv", "test_features.csv": "query_features.csv"}
        if value.name in aliases:
            value = value.with_name(aliases[value.name])
        if value.parts and value.parts[0] == "auxiliary_features":
            return LOCAL_FEATURE_ROOT / Path(*value.parts[1:])
        if value.name in {"train_features.csv", "query_features.csv"}:
            return LOCAL_FEATURE_ROOT / value
        return DATA_ROOT / value
    return SUPPORT_ROOT / value


def asset_path(relative):
    value = Path(relative)
    if value.is_absolute():
        return value
    if not value.parts:
        return ARTIFACT_ROOT
    first = value.parts[0]
    rest = Path(*value.parts[1:])
    if first == "support":
        return support_path(rest)
    if first.lower() == "data":
        return support_path(value)
    if first == "code":
        return CODE_DIR / rest
    if first == "artifacts":
        return ARTIFACT_ROOT / rest
    if first == "baseline_alignment":
        return BASELINE_DIR / rest
    if value.suffix in {".py", ".ps1"} or str(value) == "protocol.json":
        return CODE_DIR / value
    return ARTIFACT_ROOT / value


def asset_key(path):
    value = Path(path).resolve()
    for root, prefix in [(LOCAL_FEATURE_ROOT, "Data/auxiliary_features"), (BASELINE_DIR, "baseline_alignment"), (ARTIFACT_ROOT, "artifacts"), (SUPPORT_ROOT, "support"), (CODE_DIR, "code"), (DATA_ROOT, "Data")]:
        if value.is_relative_to(root):
            return Path(prefix) / value.relative_to(root)
    raise ValueError("Asset is outside configured research roots")


AUXILIARY_FILES = {"blocks": "sg-hdb-block.csv", "mrt": "sg-mrt-stations.csv", "schools": "sg-schools.csv", "malls": "sg-shopping-malls.csv", "coe": "sg-coe-prices.csv", "stocks": "sg-stock-prices.csv"}


def check_inputs():
    """Check declared input headers; do not create a source inventory."""
    import pandas as pd
    required = [DATA_ROOT / "train.csv", DATA_ROOT / "query.csv", *(DATA_ROOT / "auxiliary" / name for name in AUXILIARY_FILES.values())]
    missing = [str(path.relative_to(DATA_ROOT)) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing input files: " + ", ".join(missing))
    columns = {path.relative_to(DATA_ROOT).as_posix(): list(pd.read_csv(path, nrows=0).columns) for path in required}
    expected = {"RENT_APPROVAL_DATE", "TOWN", "BLOCK", "STREET", "FLAT_TYPE", "FLAT_MODEL", "FLOOR_AREA_SQM", "FURNISHED", "LEASE_COMMENCE_DATE", "FEE"}
    for role in ("train.csv", "query.csv"):
        absent = expected - set(columns[role])
        if absent:
            raise ValueError(f"{role}: missing input fields {sorted(absent)}")
    if "MONTHLY_RENT" not in columns["train.csv"] or "MONTHLY_RENT" in columns["query.csv"]:
        raise ValueError("Training needs MONTHLY_RENT; query must be unlabeled")
    return {"status": "headers_checked", "columns": columns, "models_fitted": 0}


def prepare_features():
    """Build the 94 auxiliary features without fitting a forecasting model."""
    import pandas as pd
    from experiments.local._support.src.features.build_features import build_rental_features
    check_inputs()
    dtype = {"TOWN": str, "BLOCK": str, "STREET": str}
    aux = {key: pd.read_csv(DATA_ROOT / "auxiliary" / name, dtype=dtype) for key, name in AUXILIARY_FILES.items()}
    train = pd.read_csv(DATA_ROOT / "train.csv", dtype=dtype)
    query = pd.read_csv(DATA_ROOT / "query.csv", dtype=dtype)
    folder = LOCAL_FEATURE_ROOT
    if folder.exists():
        raise FileExistsError("Refusing to replace existing generated feature tables")
    training_features = build_rental_features(train, aux)
    history = pd.concat([train.reset_index(drop=True), training_features.reset_index(drop=True)], axis=1)
    query_features = build_rental_features(query, aux, history=history)
    if len(training_features) != len(train) or len(query_features) != len(query):
        raise ValueError("Feature preparation changed row identities")
    if "MONTHLY_RENT" in training_features or "MONTHLY_RENT" in query_features:
        raise ValueError("Target unexpectedly included in feature cache")
    folder.mkdir(parents=True)
    training_features.to_csv(folder / "train_features.csv", index=False)
    query_features.to_csv(folder / "query_features.csv", index=False)
    return {"status": "features_prepared", "rows": {"train": len(train), "query": len(query)}, "features": len(training_features.columns), "models_fitted": 0}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["check-inputs", "prepare-features"])
    args = parser.parse_args()
    print(json.dumps(prepare_features() if args.action == "prepare-features" else check_inputs()))
