"""Resolve feature names and presets against the auxiliary catalog."""

from pathlib import Path

from src.features.build_features import (
    FAMILY_COLUMNS,
    FEATURE_COLUMNS,
    ORIGINAL_COLUMNS,
    check_feature_catalog,
)

PRESETS = {
    "original": list(ORIGINAL_COLUMNS),
    **{name: list(columns) for name, columns in FAMILY_COLUMNS.items()},
    "all": list(ORIGINAL_COLUMNS) + list(FEATURE_COLUMNS),
}

_CATALOG_ORDER = list(ORIGINAL_COLUMNS) + list(FEATURE_COLUMNS)
_KNOWN = set(_CATALOG_ORDER)


def resolve_features(text: str, feature_file: str) -> list[str]:
    """Expand presets and column names into catalog order.

    ``text`` is a comma-separated list. ``feature_file`` is one name per line.
    Blank lines and ``#`` comments in the file are ignored.
    """
    check_feature_catalog()
    tokens: list[str] = []
    if text.strip():
        tokens.extend(part.strip() for part in text.split(",") if part.strip())
    if feature_file.strip():
        lines = Path(feature_file).read_text(encoding="utf-8").splitlines()
        for line in lines:
            item = line.split("#", 1)[0].strip()
            if item:
                tokens.append(item)
    if not tokens:
        raise SystemExit("pass --features or --feature-file")

    selected: list[str] = []
    for token in tokens:
        if token == "MONTHLY_RENT":
            raise SystemExit("MONTHLY_RENT is the target and cannot be a feature")
        if token in PRESETS:
            selected.extend(PRESETS[token])
            continue
        if token in _KNOWN:
            selected.append(token)
            continue
        raise SystemExit(f"unknown feature or preset: {token}")

    wanted = set(selected)
    return [column for column in _CATALOG_ORDER if column in wanted]
