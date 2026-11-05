"""Kaggle submission construction and strict validation."""
from pathlib import Path
import numpy as np
import pandas as pd

def make_submission(predictions, expected_rows):
    """Kaggle format: Id (0-based test row index), Predicted (monthly rent)."""
    values = np.asarray(predictions, dtype=float).reshape(-1)
    if len(values) != expected_rows or not np.isfinite(values).all():
        raise ValueError('Submission predictions must match test row count and be finite')
    submission = pd.DataFrame({
        'Id': np.arange(expected_rows, dtype=np.int64),
        'Predicted': values,
    })
    if submission.columns.tolist() != ['Id', 'Predicted']:
        raise RuntimeError('Invalid Kaggle submission schema')
    return submission


def validate_submission(frame, expected_rows=50000):
    """Require exact columns, original zero-based row IDs and finite predictions."""
    if frame.columns.tolist() != ['Id', 'Predicted']:
        raise ValueError('Submission columns must be exactly Id,Predicted')
    if len(frame) != expected_rows:
        raise ValueError(f'Expected {expected_rows} predictions, found {len(frame)}')
    if not np.array_equal(frame['Id'].to_numpy(), np.arange(expected_rows)):
        raise ValueError('Submission IDs must preserve test row order, from 0 to N-1')
    values = pd.to_numeric(frame['Predicted'], errors='raise').to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError('Predictions must be finite and nonmissing')
    return {'prediction_count': len(frame), 'columns': frame.columns.tolist(),
            'ids_in_original_row_order': True, 'finite_predictions': True}


def write_submission(predictions, path, expected_rows=50000):
    result = make_submission(predictions, expected_rows)
    validate_submission(result, expected_rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False)
    return validate_submission(pd.read_csv(path), expected_rows)
