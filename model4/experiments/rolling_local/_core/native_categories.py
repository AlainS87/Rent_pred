"""Native-category identification and missing tokens for the local model."""
import pandas as pd
MISSING_CATEGORY = "__missing__"

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
