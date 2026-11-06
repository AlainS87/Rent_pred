"""Internal numerical support selected by the local experiments."""
import os
from pathlib import Path
from .regional import CONFIRM_MONTHS, FIT_END, SEARCH_MONTHS
from src.paths import PROJECT_ROOT as ROOT
SEED = 5228
REGIONAL_DIR = Path(os.environ.get('RENTAL_ARTIFACTS_DIR', str(ROOT / 'artifacts'))).expanduser().resolve() / 'local' / 'baseline' / 'regional'

def _check_months(split) -> None:
    if split.fit_months[-1] != FIT_END:
        raise SystemExit(f'fit months end at {split.fit_months[-1]}, expected {FIT_END}')
    if tuple(split.search_months) != SEARCH_MONTHS:
        raise SystemExit(f'search months are {split.search_months}')
    if tuple(split.confirm_months) != CONFIRM_MONTHS:
        raise SystemExit(f'confirm months are {split.confirm_months}')
