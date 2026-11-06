"""Paths and identities for the existing parent-repository datasets."""
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
import platform

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_DIR.parent / 'Data'
OUTPUT_DIR = PROJECT_DIR / 'outputs'
AUXILIARY_FILES = ('sg-hdb-block.csv', 'sg-mrt-stations.csv',
                   'sg-schools.csv', 'sg-shopping-malls.csv')
CORE_PACKAGES = ('numpy', 'pandas', 'scipy', 'scikit-learn', 'catboost')


def auxiliary_path(data_dir, name):
    """Prefer Data/auxiliary; also accept the original flat input layout."""
    folder = Path(data_dir)
    nested = folder / 'auxiliary' / name
    return nested if nested.is_file() else folder / name


def file_hash(path):
    digest = sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def input_hashes(data_dir, train_file='train.csv'):
    files = {'training_reference': Path(data_dir) / train_file}
    files.update({name: auxiliary_path(data_dir, name) for name in AUXILIARY_FILES})
    return {name: file_hash(path) for name, path in files.items()}


def environment():
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'packages': {name: version(name) for name in CORE_PACKAGES}}


def implementation_hashes():
    files = list((PROJECT_DIR / 'hdb_rent').glob('*.py'))
    files.append(PROJECT_DIR / 'experiments/historical.py')
    return {str(p.relative_to(PROJECT_DIR)): file_hash(p) for p in sorted(files)}
