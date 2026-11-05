"""Portable paths only; importing this module does not create artifacts."""
from pathlib import Path
import os
CODE_DIR = Path(__file__).resolve().parent
PROJECT = CODE_DIR.parents[1]
DATA_DIR = Path(os.environ.get('RENTAL_DATA_DIR', str(PROJECT / 'data' / 'raw'))).expanduser().resolve()
ANALYSIS = Path(os.environ.get('RENTAL_ARTIFACTS_DIR', str(PROJECT / 'artifacts'))).expanduser().resolve() / 'early_analysis'
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('OMP_NUM_THREADS', '4')
