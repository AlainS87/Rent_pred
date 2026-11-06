"""Internal numerical support selected by the local experiments."""
import numpy as np

def rmse(y_true, y_pred) -> float:
    truth = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    if truth.shape != pred.shape:
        raise ValueError(f'prediction length {pred.shape} does not match {truth.shape}')
    score = float(np.sqrt(np.mean((truth - pred) ** 2)))
    if not np.isfinite(score):
        raise ValueError('RMSE is not finite')
    return score
