"""Bundled research protocols and encoding recipes."""
from pathlib import Path


def config_path(relative: str) -> Path:
    """Locate an installed configuration independently of the working directory."""
    root = Path(__file__).resolve().parent
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise FileNotFoundError(f"Unknown bundled configuration: {relative}")
    return path
