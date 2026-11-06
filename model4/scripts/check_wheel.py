"""Build a wheel and check its resources/imports in a separate install directory.

Uses the current interpreter's dependencies; never trains or reads rental data.
"""
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="rental-wheel-") as temporary:
        work = Path(temporary)
        wheels, installed = work / "wheels", work / "installed"
        subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation", "--wheel-dir", str(wheels), str(root)], check=True)
        wheel, = wheels.glob("*.whl")
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--no-compile", "--target", str(installed), str(wheel)], check=True)
        code = r'''
import importlib, json, pathlib, runpy, sys
installed = pathlib.Path(sys.argv[1]).resolve()
sys.path.insert(0, str(installed))
for name in ('configs', 'src.paths', 'analysis.external_signals', 'experiments.local.bootstrap', 'experiments.local.temporal_index', 'models.catboost.preprocess'):
    module = importlib.import_module(name)
    assert pathlib.Path(module.__file__).resolve().is_relative_to(installed), name
from configs import config_path
for family in ('backtest', 'research_round2', 'research_round3', 'research_round4'):
    assert json.loads(config_path(f'{family}/protocol.json').read_text(encoding='utf-8'))
for name in ('bin', 'target'):
    assert json.loads(config_path(f'encodings/{name}.json').read_text(encoding='utf-8'))
from src.paths import PROJECT_ROOT, CORE_FEATURE_ROOT, LOCAL_FEATURE_ROOT
assert PROJECT_ROOT == pathlib.Path.cwd()
assert CORE_FEATURE_ROOT != LOCAL_FEATURE_ROOT
assert not CORE_FEATURE_ROOT.is_relative_to(installed)
for model in ('ridge', 'xgb', 'catboost', 'mlp'):
    assert json.loads((installed / 'models' / model / 'default_genome.json').read_text(encoding='utf-8'))
for relative in ('experiments/local/protocol.json', 'experiments/historical_data/sources.json', 'experiments/rolling_local/example_config.json'):
    assert json.loads((installed / relative).read_text(encoding='utf-8'))
from src.backtest.protocol import read_protocol
assert read_protocol()
for module in ('experiments.rolling_local', 'analysis.external_signals', 'experiments.local.bootstrap'):
    sys.modules.pop(module, None)
    sys.argv = [module, '--help']
    try:
        runpy.run_module(module, run_name='__main__', alter_sys=True)
    except SystemExit as exc:
        assert exc.code == 0
print('Wheel resources, isolated imports, writable paths and help checks passed; no model fitted.')
'''
        environment = os.environ.copy()
        for name in ("PYTHONPATH", "RENTAL_PROJECT_DIR", "RENTAL_ARTIFACTS_DIR", "RENTAL_DATA_DIR", "RENTAL_LOCAL_FEATURES_DIR"):
            environment.pop(name, None)
        subprocess.run([sys.executable, "-I", "-c", code, str(installed)], cwd=work, env=environment, check=True, timeout=60)


if __name__ == "__main__":
    main()
