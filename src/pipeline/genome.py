"""Genome identity: features, preprocess, and architecture."""

import hashlib
import json
from pathlib import Path

from src.pipeline.paths import MODELS
from src.pipeline.preprocess import COLUMN_OVERRIDES
from src.pipeline.rules import PREPROCESS_KNOBS


def feature_hash(columns: list[str]) -> str:
    """12 hex chars of SHA-256 over the sorted column names."""
    payload = "\n".join(sorted(columns)).encode()
    return hashlib.sha256(payload).hexdigest()[:12]


def genome_hash(genome: dict) -> str:
    """16 hex chars of the features, preprocess, architecture, and fixed column overrides."""
    payload = {
        "architecture": genome["architecture"],
        "column_overrides": COLUMN_OVERRIDES,
        "features": sorted(genome["features"]),
        "preprocess": {
            "categorical": genome["preprocess"]["categorical"],
            "numeric_impute": genome["preprocess"]["numeric_impute"],
            "rare_min_count": int(genome["preprocess"]["rare_min_count"]),
            "scale": genome["preprocess"]["scale"],
        },
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def load_default_genome(model: str) -> dict:
    path = MODELS / model / "default_genome.json"
    genome = json.loads(path.read_text(encoding="utf-8"))
    _require_genome(genome)
    return genome


def seed_genome(default: dict, pool: list[str]) -> dict:
    """Clip the default feature list to the pool. An empty clip uses the pool."""
    pool_set = set(pool)
    features = [column for column in default["features"] if column in pool_set]
    if not features:
        features = list(pool)
    if not features:
        raise SystemExit("the feature pool is empty")
    genome = {
        "features": features,
        "preprocess": dict(default["preprocess"]),
        "architecture": json.loads(json.dumps(default["architecture"])),
    }
    _require_genome(genome)
    return genome


def apply_epochs(genome: dict, epochs: int | None, model: str) -> dict:
    """Bake a CLI epoch cap into the architecture so the genome hash changes."""
    if epochs is None:
        return genome
    if epochs < 1:
        raise SystemExit("--epochs must be at least 1")
    copied = json.loads(json.dumps(genome))
    if model == "mlp":
        copied["architecture"]["epochs"] = int(epochs)
    elif model in {"xgb", "catboost"}:
        copied["architecture"]["n_estimators"] = int(epochs)
    return copied


def write_features_file(folder: Path, columns: list[str]) -> None:
    """Explain the columns that this checkpoint folder used."""
    digest = feature_hash(columns)
    lines = [
        f"# feature_hash: {digest}",
        "# sha256 prefix of the sorted column names, joined by newlines",
        *sorted(columns),
        "",
    ]
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "features.txt").write_text("\n".join(lines), encoding="utf-8")


def _require_genome(genome: dict) -> None:
    if not genome.get("features"):
        raise SystemExit("a genome needs at least one feature")
    preprocess = genome.get("preprocess") or {}
    for knob, choices in PREPROCESS_KNOBS.items():
        if knob not in preprocess:
            raise SystemExit(f"preprocess is missing {knob}")
        value = preprocess[knob]
        if knob == "rare_min_count":
            value = int(value)
            preprocess[knob] = value
        if knob == "categorical" and value == "native":
            continue
        if value not in choices:
            raise SystemExit(f"preprocess {knob}={value} is outside {list(choices)}")
    if "architecture" not in genome:
        raise SystemExit("a genome needs an architecture")
