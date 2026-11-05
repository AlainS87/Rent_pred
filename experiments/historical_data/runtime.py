"""Portable historical-source catalog and retrieval output paths."""
from pathlib import Path
import os
CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parents[1]
ARTIFACT_DIR = Path(os.environ.get('RENTAL_ARTIFACTS_DIR', str(REPO_ROOT / 'artifacts'))).expanduser().resolve() / 'historical_data'
