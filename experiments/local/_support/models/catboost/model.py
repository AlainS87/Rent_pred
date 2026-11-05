"""Internal numerical support selected by the local experiments."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
WEIGHT_NAME = 'model.joblib'
MISSING_CATEGORY = '__missing__'

def fit_model(x_fit, y_fit, x_search, y_search, architecture: dict, seed: int, epochs: int | None=None):
    x_fit, y_fit = _prepare_pair(x_fit, y_fit)
    x_search, y_search = _prepare_pair(x_search, y_search)
    categories = _category_columns(x_fit)
    x_fit = _fill_missing_categories(x_fit)
    x_search = _fill_missing_categories(x_search)
    trees = int(epochs if epochs is not None else architecture.get('n_estimators', 200))
    stopping = int(architecture.get('early_stopping_rounds', 20))
    params = {'depth': int(architecture['max_depth']), 'learning_rate': float(architecture['learning_rate']), 'iterations': trees, 'loss_function': 'RMSE', 'random_seed': seed, 'thread_count': 1, 'allow_writing_files': False}
    use_stopping = trees > 20 and stopping > 0
    if use_stopping:
        params['early_stopping_rounds'] = stopping
    model = CatBoostRegressor(**params)
    if use_stopping:
        model.fit(x_fit, y_fit, eval_set=(x_search, y_search), cat_features=categories, use_best_model=True, verbose=False)
    else:
        model.fit(x_fit, y_fit, cat_features=categories, verbose=False)
    return model

def predict_model(bundle, features) -> np.ndarray:
    from ...src.pipeline.preprocess import assert_prediction
    frame = features if isinstance(features, pd.DataFrame) else np.asarray(features)
    frame = _fill_missing_categories(frame)
    raw = bundle.predict(frame)
    return assert_prediction(raw, len(frame))

def _fill_missing_categories(features):
    """CatBoost rejects NaN in a category column. Unseen levels become one token."""
    if not isinstance(features, pd.DataFrame):
        return features
    frame = features.copy()
    for column in frame.columns:
        if isinstance(frame[column].dtype, pd.CategoricalDtype):
            frame[column] = frame[column].astype('string').fillna(MISSING_CATEGORY)
    return frame

def _category_columns(features) -> list[str] | None:
    if not isinstance(features, pd.DataFrame):
        return None
    names = [column for column in features.columns if isinstance(features[column].dtype, pd.CategoricalDtype)]
    return names or None

def _prepare_pair(features, target):
    from ...src.pipeline.preprocess import assert_fit_inputs
    if isinstance(features, pd.DataFrame):
        rents = np.asarray(target, dtype=np.float64).reshape(-1)
        if len(features) != len(rents):
            raise ValueError(f'CatBoost frame has {len(features)} rows, expected {len(rents)}')
        return (features, rents)
    return assert_fit_inputs(features, target)

def save_model(bundle, directory: Path) -> None:
    joblib.dump(bundle, directory / WEIGHT_NAME)

def load_model(directory: Path):
    return joblib.load(directory / WEIGHT_NAME)
