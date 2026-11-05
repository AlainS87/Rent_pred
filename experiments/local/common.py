"""Thin temporal adapter around the internal support preprocessing functions."""
from experiments.local.bootstrap import Path, CODE_DIR, DATA_ROOT, SUPPORT_ROOT, ARTIFACT_ROOT, asset_path, asset_key, support_path
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent
SUPPORT = asset_path(Path('support'))
from experiments.local._support import load_stack
stack = load_stack('_common_stack')
from experiments.local._support.mean_residual_model.regional import LEVELS, _numeric_block, _numeric_params, cell_table, with_month
from experiments.local._support.mean_residual_model.reginal_mean_model.lstm.linear import _as_unit_rent
from experiments.local._support.models.catboost.preprocess import fit_preprocessor, transform_frame
from experiments.local._support.models.catboost.model import _category_columns, _fill_missing_categories
from experiments.local._support.src.pipeline.data import load_joined
from experiments.local._support.src.features.build_features import FEATURE_COLUMNS, ORIGINAL_COLUMNS
from experiments.local._support.src.features.group_means import GROUP_MEAN_COLUMNS
CONFIG = json.loads(asset_path(Path('protocol.json')).read_text())
STATIC_KEYS = ['floor_area', 'fee', 'lease', 'age', 'month_index', 'latitude', 'longitude', 'flat_type_ord', 'flat_model_ord', 'max_floor', 'year_completed', 'flat_age', 'dist_mrt', 'dist_mall', 'dist_primary', 'dist_secondary']
FEATURES = {'original': stack.CAT_FEATURES, 'expanded': stack.CAT_FEATURES + CONFIG['expanded_features']}

def month_add(month, offset):
    return str(pd.Period(month, freq='M') + offset)

def load_data():
    columns = list(dict.fromkeys([*ORIGINAL_COLUMNS, *FEATURE_COLUMNS]))
    train = load_joined('train', columns)
    test = load_joined('query', columns)
    train['row_id'] = np.arange(len(train))
    test['row_id'] = np.arange(len(test))
    return (train, test)

def before(frame, cutoff):
    return frame.loc[frame.RENT_APPROVAL_DATE.astype(str).str[:7] <= cutoff].copy().reset_index(drop=True)

def interval(frame, start, end):
    m = frame.RENT_APPROVAL_DATE.astype(str).str[:7]
    return frame.loc[(m > start) & (m <= end)].copy().reset_index(drop=True)

def ranked_frames(fit, others):
    stats = fit.groupby('flat_model_key').MONTHLY_RENT.agg(['median', 'mean'])
    order = stats.reset_index().sort_values(['median', 'mean', 'flat_model_key'])
    mapping = {name: rank for rank, name in enumerate(order.flat_model_key)}
    fallback = float(np.median(list(mapping.values())))
    out = []
    for part in [fit, *others]:
        p = part.copy()
        p['flat_model_ord'] = p.flat_model_key.map(mapping).fillna(fallback).astype(float)
        out.append(p)
    return (out, mapping)

def plain_block(frame):
    block = {key: np.full(len(frame), np.nan, dtype=np.float64) for key in STATIC_KEYS}
    stack._copy_plain(block, frame)
    block['latitude'][:] = frame.latitude.to_numpy(dtype=np.float64)
    block['longitude'][:] = frame.longitude.to_numpy(dtype=np.float64)
    return block

def static_matrix(block):
    return np.column_stack([block[k] for k in STATIC_KEYS]).astype(np.float64)

def finite(values, name):
    a = np.asarray(values, dtype=np.float64)
    if not np.isfinite(a).all():
        raise ValueError(f'{name}: {int((~np.isfinite(a)).sum())} non-finite values')
    return a

def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    tmp.replace(path)

def cutoffs_needed():
    base = set()
    for origin in [CONFIG['development']['origin'], CONFIG['evaluation']['origin']]:
        for endpoint in [origin, month_add(origin, -3)]:
            base.add(endpoint)
            for delta in [-12, -8, -4]:
                base.add(month_add(endpoint, delta))
    return sorted(base | {month_add(c, -3) for c in base})
