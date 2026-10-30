"""Address-key helpers shared by every feature builder."""

import pandas as pd


def norm_text(values: pd.Series) -> pd.Series:
    """Strip, lowercase, and collapse repeated spaces."""
    return (
        values.astype("string")
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )


def block_key(block: pd.Series, street: pd.Series) -> pd.Series:
    """Normalized BLOCK|STREET. Block identifiers stay strings, including letters."""
    return norm_text(block) + "|" + norm_text(street)
