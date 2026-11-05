"""Separate installed source resources from user-owned data and outputs."""
import os
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[1]
_default_project = SOURCE_ROOT if (SOURCE_ROOT / "pyproject.toml").is_file() else Path.cwd()
PROJECT_ROOT = Path(os.environ.get("RENTAL_PROJECT_DIR", str(_default_project))).expanduser().resolve()
DATA_ROOT = Path(os.environ.get("RENTAL_DATA_DIR", str(PROJECT_ROOT / "data/raw"))).expanduser().resolve()
ARTIFACT_ROOT = Path(os.environ.get("RENTAL_ARTIFACTS_DIR", str(PROJECT_ROOT / "artifacts"))).expanduser().resolve()
CORE_FEATURE_ROOT = ARTIFACT_ROOT / "features" / "core83"
LOCAL_FEATURE_ROOT = Path(os.environ.get("RENTAL_LOCAL_FEATURES_DIR", str(ARTIFACT_ROOT / "features/local94"))).expanduser().resolve()
