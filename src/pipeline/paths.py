"""Project paths used by the training pipeline."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "Data"
MODELS = ROOT / "models"


def rel(path: Path) -> str:
    """POSIX path relative to the project root, for logs and design notes."""
    return path.resolve().relative_to(ROOT).as_posix()
