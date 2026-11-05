"""Contracts for resource lookup, cache separation and import isolation."""
import json
import subprocess
import sys
from pathlib import Path
import importlib
import pytest

from configs import config_path
from experiments.local import bootstrap
from src.paths import CORE_FEATURE_ROOT, LOCAL_FEATURE_ROOT


def test_all_bundled_protocols_are_accessible_away_from_checkout(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for family in ("backtest", "research_round2", "research_round3", "research_round4"):
        assert isinstance(json.loads(config_path(f"{family}/protocol.json").read_text(encoding="utf-8")), dict)


def test_shared_path_and_resource_loaders_are_in_frozen_code_identity():
    from src.backtest.checkpoints import implementation_hashes
    from src.research_round2.run import source_paths
    hashes = implementation_hashes()
    paths = source_paths()
    root = Path(__file__).resolve().parents[1]
    for relative in ("src/paths.py", "configs/__init__.py"):
        assert relative in hashes
        assert root / relative in paths


def test_local_feature_aliases_share_one_explicit_cache():
    assert CORE_FEATURE_ROOT != LOCAL_FEATURE_ROOT
    for spelling in ("Data/auxiliary_features/test_features.csv", "auxiliary_features/query_features.csv", "query_features.csv"):
        resolved = bootstrap.support_path(Path(spelling))
        assert resolved == LOCAL_FEATURE_ROOT / "query_features.csv"
        assert bootstrap.asset_path(bootstrap.asset_key(resolved)) == resolved


def test_core_and_local_imports_are_independent_of_order(tmp_path):
    groups = [
        ["src.pipeline.data", "models.catboost.preprocess", "mean_residual_model.regional"],
        ["experiments.local._support.src.pipeline.data", "experiments.local._support.models.catboost.preprocess", "experiments.local.temporal_index"],
    ]
    root = Path(__file__).resolve().parents[1]
    for ordered in (groups[0] + groups[1], groups[1] + groups[0]):
        code = """
import importlib, json, pathlib, sys
sys.path.insert(0, sys.argv[1])
before = list(sys.path)
for name in json.loads(sys.argv[2]):
    module = importlib.import_module(name)
    assert module.__name__ == name
assert sys.path == before, 'Import modified module search paths'
assert not any(name == '_support' or name.startswith('_support.') for name in sys.modules)
"""
        subprocess.run([sys.executable, "-c", code, str(root), json.dumps(ordered)], cwd=tmp_path, check=True, timeout=30)


def test_local_module_imports_do_not_read_data_or_change_search_paths(monkeypatch):
    pytest.importorskip("torch")
    import pandas as pd

    def forbidden(*args, **kwargs):
        pytest.fail("Importing local research modules must not read datasets")

    monkeypatch.setattr(pd, "read_csv", forbidden)
    before = list(sys.path)
    folder = Path(__file__).resolve().parents[1] / "experiments/local"
    for path in sorted(folder.glob("*.py")):
        importlib.import_module(f"experiments.local.{path.stem}")
    assert sys.path == before

    from experiments.local._support import load_stack
    first, second = load_stack("_first_check"), load_stack("_second_check")
    original = second.HERE
    first.HERE = Path("isolated-test-output")
    assert second.HERE == original
