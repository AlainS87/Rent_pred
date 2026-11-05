"""Internal numerical support selected by the local experiments."""
import numpy as np
import pandas as pd
FIT_END = '2024-09'
SEARCH_MONTHS = ('2024-10', '2024-11', '2024-12')
CONFIRM_MONTHS = ('2025-01', '2025-02', '2025-03')
LEVELS = ['coe_c_premium_lag1', 'bank_basket_adj_close_lag1', 'property_basket_adj_close_lag1']

def cell_table(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = with_month(frame)
    grouped = prepared.groupby(['TOWN', 'month'], observed=True)
    cells = grouped['MONTHLY_RENT'].agg(mean_rent='mean', rows='size').reset_index()
    month_index = prepared.groupby('month')['month_index'].first()
    cells['month_index'] = cells['month'].map(month_index)
    for column in LEVELS:
        values = prepared.groupby('month')[column].first()
        cells[column] = cells['month'].map(values)
    return cells

def with_month(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work['month'] = work['RENT_APPROVAL_DATE'].astype('string').str.slice(0, 7)
    work['month_index'] = _month_index(work['month'])
    return work

def _month_index(month: pd.Series) -> pd.Series:
    year = month.str.slice(0, 4).astype(int)
    part = month.str.slice(5, 7).astype(int)
    return (year - 2021) * 12 + (part - 1)

def _numeric_params(train: pd.DataFrame, columns: list[str]) -> dict[str, dict[str, float]]:
    params = {}
    for column in columns:
        series = train[column].astype(float)
        center = float(series.median())
        spread = float(series.std(ddof=0)) or 1.0
        params[column] = {'center': center, 'fill': center, 'spread': spread}
    return params

def _numeric_block(frame: pd.DataFrame, columns: list[str], params: dict[str, dict[str, float]]) -> np.ndarray:
    values = []
    for column in columns:
        spec = params[column]
        filled = frame[column].astype(float).fillna(spec['fill']).to_numpy(dtype=np.float64)
        center = spec['center']
        spread = spec['spread']
        values.append((np.where(np.isfinite(filled), filled, center) - center) / spread)
    return np.column_stack(values)

def _weighted_rmse(truth: np.ndarray, prediction: np.ndarray, weights: np.ndarray) -> float:
    keep = np.isfinite(truth) & np.isfinite(prediction) & np.isfinite(weights)
    error = truth[keep] - prediction[keep]
    weight = weights[keep]
    return float(np.sqrt(np.average(error ** 2, weights=weight)))
