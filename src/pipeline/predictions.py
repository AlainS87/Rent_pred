"""Generic finite, row-preserving prediction output."""
from pathlib import Path
import numpy as np
import pandas as pd
def write_predictions(path: Path, prediction, row_ids=None) -> None:
    values = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("predictions must be finite")
    ids = np.arange(len(values)) if row_ids is None else np.asarray(row_ids)
    if len(ids) != len(values) or pd.Series(ids).isna().any() or not pd.Series(ids).is_unique:
        raise ValueError("row identifiers must be non-null, unique, and aligned")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"row_id": ids, "prediction": values}).to_csv(path, index=False)
