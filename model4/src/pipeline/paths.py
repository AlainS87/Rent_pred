"""Repository-relative source, input, and generated-artifact paths."""
from pathlib import Path
from src.paths import PROJECT_ROOT as ROOT, DATA_ROOT as DATA, CORE_FEATURE_ROOT as FEATURE_DATA, ARTIFACT_ROOT, SOURCE_ROOT
MODEL_DEFINITIONS = SOURCE_ROOT / "models"
MODELS = ARTIFACT_ROOT / "models"
def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()
