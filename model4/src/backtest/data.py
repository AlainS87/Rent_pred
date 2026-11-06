"""Source-hashed, target-free caches and selective historical-label access.

The model API never receives a full labeled source table. Future labels are
parsed only in the separate scoring module. Non-history rows are skipped when
reading the target column, while all predictors may be known in advance.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.build_features import FEATURE_COLUMNS, ORIGINAL_COLUMNS, assemble_features
from src.features.group_means import GROUP_MEAN_COLUMNS
from src.backtest.protocol import ROOT, ROW_ID, TARGET, months, require_identity

NON_TARGET_FEATURES = [c for c in FEATURE_COLUMNS if c not in GROUP_MEAN_COLUMNS]
BASE_COLUMNS = [ROW_ID, *ORIGINAL_COLUMNS, *NON_TARGET_FEATURES]
ADDRESS_DTYPES = {"TOWN": str, "BLOCK": str, "STREET": str}
AUXILIARY = {
    "blocks": "sg-hdb-block.csv", "mrt": "sg-mrt-stations.csv",
    "schools": "sg-schools.csv", "malls": "sg-shopping-malls.csv",
    "coe": "sg-coe-prices.csv", "stocks": "sg-stock-prices.csv",
}


def sha256(path: Path) -> str:
    state = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            state.update(chunk)
    return state.hexdigest()


def identity_hash(frame: pd.DataFrame) -> str:
    """Hash all original predictor values in source-row order, independent of index."""
    values = frame[[ROW_ID, *ORIGINAL_COLUMNS]].reset_index(drop=True)
    return hashlib.sha256(pd.util.hash_pandas_object(values, index=False).to_numpy().tobytes()).hexdigest()


def frame_hash(frame: pd.DataFrame) -> str:
    state = hashlib.sha256()
    state.update(json.dumps(frame.columns.tolist()).encode())
    state.update(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes())
    return state.hexdigest()


def source_hashes(root: Path, split: str) -> dict:
    # A target-free cache must not become invalid merely because unavailable
    # future labels changed. Hash the full predictor projection for validity;
    # the raw file's byte checksum is retained separately as audit provenance.
    projected = {f"data/raw/{'query' if split == 'test' else split}.csv#all_predictor_rows": identity_hash(read_predictors(root, split))}
    paths = [root / "data/raw/auxiliary" / name for name in AUXILIARY.values()]
    # Feature-generating code is part of cache provenance, not just input data.
    paths.extend(sorted((root / "src/features").glob("*.py")))
    paths.append(root / "src/backtest/data.py")
    return {**projected, **{str(p.relative_to(root)).replace("\\", "/"): sha256(p) for p in paths}}


def read_predictors(root: Path, split: str) -> pd.DataFrame:
    frame = pd.read_csv(root / "data/raw" / ("query.csv" if split == "test" else f"{split}.csv"), usecols=ORIGINAL_COLUMNS, dtype=ADDRESS_DTYPES)
    frame.insert(0, ROW_ID, np.arange(len(frame), dtype=np.int64))
    return frame


def build_cache(root: Path = ROOT, cache_dir: Path | None = None) -> dict:
    cache_dir = cache_dir or root / "artifacts/backtest/cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    aux = {
        name: pd.read_csv(root / "data/raw/auxiliary" / path, dtype=ADDRESS_DTYPES if name == "blocks" else None)
        for name, path in AUXILIARY.items()
    }
    report = {}
    for split in ("train", "test"):
        raw = read_predictors(root, split)
        built = assemble_features(raw, aux)[BASE_COLUMNS].reset_index(drop=True)
        if len(built) != len(raw) or identity_hash(built) != identity_hash(raw):
            raise ValueError("non-target reconstruction changed source row identity/order")
        if set(GROUP_MEAN_COLUMNS + [TARGET]) & set(built):
            raise ValueError("target-derived or target column entered cache")
        path = cache_dir / f"{split}_base.parquet"
        built.to_parquet(path, index=False)
        manifest = {
            "version": 1, "split": split, "rows": len(built), "columns": built.columns.tolist(),
            "identity_sha256": identity_hash(raw), "parquet_sha256": sha256(path),
            "source_sha256": source_hashes(root, split),
            "raw_file_sha256_at_build_for_audit_only": sha256(root / "data/raw" / ("query.csv" if split == "test" else f"{split}.csv")),
            "cache_validity_uses": "full predictor projection plus original auxiliary/code hashes; excludes all labels",
            "precomputed_target_means_read": False,
            "feature_generation": "assemble_features(raw predictors, original auxiliary sources)",
        }
        (cache_dir / f"{split}_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        report[split] = manifest
    return report


def load_cache(split: str, root: Path = ROOT, cache_dir: Path | None = None) -> pd.DataFrame:
    cache_dir = cache_dir or root / "artifacts/backtest/cache"
    manifest = json.loads((cache_dir / f"{split}_manifest.json").read_text(encoding="utf-8"))
    path = cache_dir / f"{split}_base.parquet"
    if manifest["source_sha256"] != source_hashes(root, split):
        raise ValueError("cache source/code hashes no longer match")
    if manifest["parquet_sha256"] != sha256(path):
        raise ValueError("cached parquet checksum mismatch")
    frame = pd.read_parquet(path)
    if frame.columns.tolist() != BASE_COLUMNS or manifest["columns"] != BASE_COLUMNS:
        raise ValueError("cache schema includes missing, unexpected, or target-derived columns")
    require_identity(frame)
    raw = read_predictors(root, split)
    if len(frame) != manifest["rows"] or len(frame) != len(raw):
        raise ValueError("cache row count mismatch")
    if identity_hash(frame) != identity_hash(raw) or identity_hash(raw) != manifest["identity_sha256"]:
        raise ValueError("cache row identities/order do not match original source")
    return frame


def _read_selected_labels(path: Path, row_ids: np.ndarray) -> np.ndarray:
    """Read only requested CSV rows, then restore the caller's explicit row order."""
    requested = np.asarray(row_ids, dtype=np.int64)
    if len(np.unique(requested)) != len(requested) or (requested < 0).any():
        raise ValueError("label requests require unique nonnegative row IDs")
    selected = np.sort(requested)
    allowed = set(selected.tolist())
    table = pd.read_csv(path, usecols=[TARGET], skiprows=lambda line: line > 0 and line - 1 not in allowed)
    if len(table) != len(selected):
        raise ValueError("requested target row IDs are not present")
    values = pd.Series(table[TARGET].to_numpy(dtype=float), index=selected)
    return values.loc[requested].to_numpy()


def history_at(base: pd.DataFrame, origin: str, root: Path = ROOT) -> pd.DataFrame:
    if TARGET in base or any(c in base for c in GROUP_MEAN_COLUMNS):
        raise ValueError("history source must be a target-free base cache")
    history = base.loc[months(base) <= pd.Period(origin, freq="M")].copy()
    history[TARGET] = _read_selected_labels(root / "data/raw/train.csv", history[ROW_ID].to_numpy())
    return history


def query_at(base: pd.DataFrame, origin: str, max_horizon: int = 16) -> pd.DataFrame:
    if TARGET in base or any(c in base for c in GROUP_MEAN_COLUMNS):
        raise ValueError("query source must be a target-free base cache")
    offsets = months(base).asi8 - pd.Period(origin, freq="M").ordinal
    return base.loc[(offsets >= 1) & (offsets <= max_horizon)].copy()
