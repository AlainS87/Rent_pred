"""Original RMSE, MAE and signed bias metrics."""
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def score(y, pred):
    values = np.asarray(pred, dtype=float)
    truth = np.asarray(y, dtype=float)
    if values.shape != truth.shape or not np.isfinite(values).all():
        raise ValueError('Invalid prediction shape or values')
    return {'rmse': float(np.sqrt(mean_squared_error(truth, values))),
            'mae': float(mean_absolute_error(truth, values)),
            'bias_pred_minus_actual': float(np.mean(values - truth)),
            'n': int(len(truth))}
