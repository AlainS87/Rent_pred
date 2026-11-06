"""Smoothed local-market history with strictly earlier-month training labels.

Entire months are queried before any label from that month is incorporated.
Validation labels are never inspected; the validation history is frozen at the
training cutoff.  Entity encodings are local-market proxies, not school quality
or an estimate of a causal facility effect.
"""
from __future__ import annotations
from runtime import DATA_DIR, CODE_DIR, ANALYSIS
import re
from typing import Any
import numpy as np
import pandas as pd

def default_groups(frame: pd.DataFrame) -> dict[str, list[str]]:
    """Return explicit requested groups; unavailable columns remain auditable."""

    def pick(*names: str) -> str:
        return next((name for name in names if name in frame), names[0])
    return {'address': ['address_key'], 'town_flat_type': ['town', 'flat_type'], 'street_flat_type': ['street', 'flat_type'], 'postal_sector': [pick('hdb_postal_sector', 'postal_sector')], 'nearest_snapshot_open_mrt': [pick('mrt_status_open_nearest_name', 'mrt_status_open_nearest_code', 'mrt_nearest_name', 'mrt_nearest_code')], 'nearest_mall': ['mall_nearest_name'], 'nearest_school': ['school_nearest_name'], 'nearest_primary_school': [pick('school_primary_nearest_name', 'school_level_primary_nearest_name')], 'nearest_secondary_school': [pick('school_secondary_nearest_name', 'school_level_secondary_nearest_name')]}

def _keys(frame: pd.DataFrame, columns: list[str]) -> list[tuple | None]:
    normalized = pd.DataFrame(index=frame.index)
    valid = pd.Series(True, index=frame.index)
    for column in columns:
        values = frame[column].astype('string').str.strip().str.lower().str.replace('\\s+', ' ', regex=True)
        valid &= values.notna() & values.ne('').fillna(False)
        normalized[column] = values
    return [tuple(values) if is_valid else None for values, is_valid in zip(normalized.itertuples(index=False, name=None), valid.to_numpy())]

def construct_history(raw_train: pd.DataFrame, raw_valid: pd.DataFrame, columns: dict[str, list[str]] | None, cutoff: str, smoothing: float=20) -> tuple[pd.DataFrame, pd.DataFrame, list[dict], dict[str, Any]]:
    """Encode group histories without same-month or held-out label leakage.

    Mean = (group past sum + alpha * global past mean) / (past count + alpha).
    A previously unseen or missing group receives the global past mean, count
    zero and seen flag zero.  Before the first observed month its mean is NaN.
    Three columns per group expose mean, log1p count and coverage explicitly.

    Validation months must be strictly after ``cutoff``.  Predicting a partial
    test month also present among training labels requires a separate call with
    a cutoff preceding that target month; otherwise same-month labels would be
    available to this feature despite the approval-time interpretation.
    """
    if smoothing <= 0 or not np.isfinite(smoothing):
        raise ValueError('smoothing must be a positive finite constant')
    for name, frame in (('train', raw_train), ('valid', raw_valid)):
        if 'month' not in frame:
            raise ValueError(f'{name} lacks month')
        if not frame.index.is_unique:
            raise ValueError(f'{name} row index must be unique for safe alignment')
        months = frame['month'].astype(str)
        if not months.str.fullmatch('\\d{4}-(0[1-9]|1[0-2])').all():
            raise ValueError(f'{name} month must be YYYY-MM')
    if 'target' not in raw_train:
        raise ValueError('train lacks target')
    train_month = raw_train['month'].astype(str)
    valid_month = raw_valid['month'].astype(str)
    if (train_month > cutoff).any():
        raise ValueError('raw_train contains labels after cutoff')
    if (valid_month <= cutoff).any():
        raise ValueError('validation must follow cutoff; same-month test rows need an earlier cutoff')
    requested = default_groups(raw_train) if columns is None else dict(columns)
    target = pd.to_numeric(raw_train['target'], errors='coerce').to_numpy(dtype=float)
    usable_target = np.isfinite(target)
    groups: dict[str, list[str]] = {}
    skipped = []
    used_prefixes: set[str] = set()
    for name, cols in requested.items():
        absent_train = [col for col in cols if col not in raw_train]
        absent_valid = [col for col in cols if col not in raw_valid]
        if not cols or absent_train or absent_valid:
            skipped.append({'group': name, 'requested_columns': cols, 'missing_train': absent_train, 'missing_valid': absent_valid, 'reason': 'Missing entity columns; no approximation from distance or targets'})
            continue
        prefix = 'te_' + re.sub('[^a-z0-9]+', '_', name.lower()).strip('_')
        if prefix in used_prefixes:
            raise ValueError(f'group names produce duplicate prefix: {name}')
        used_prefixes.add(prefix)
        groups[name] = cols
    train_keys = {name: _keys(raw_train, cols) for name, cols in groups.items()}
    valid_keys = {name: _keys(raw_valid, cols) for name, cols in groups.items()}
    train_data: dict[str, np.ndarray] = {}
    valid_data: dict[str, np.ndarray] = {}
    lineage: list[dict] = []
    output_columns: dict[str, list[str]] = {}
    for name, cols in groups.items():
        prefix = 'te_' + re.sub('[^a-z0-9]+', '_', name.lower()).strip('_')
        names = [prefix + '_smoothed_mean', prefix + '_log1p_count', prefix + '_seen_before']
        output_columns[name] = names
        for feature, description in zip(names, (f'(past group sum + {smoothing:g} * global past mean)/(past count + {smoothing:g}); unseen/missing group falls back to global past mean', 'log(1 + count of valid strictly earlier-month group labels); unseen/missing group has zero count', '1 if at least one valid strictly earlier-month group label exists, else 0')):
            train_data[feature] = np.full(len(raw_train), np.nan)
            valid_data[feature] = np.full(len(raw_valid), np.nan)
            lineage.append({'feature': feature, 'family': 'historical_target_encoding', 'source_file': 'train.csv', 'source_attributes': '|'.join(['MONTHLY_RENT', 'RENT_APPROVAL_DATE', *cols]), 'construction': description, 'availability': 'Training query before updating entire month T; validation frozen training history through cutoff, no held-out labels'})
    histories: dict[str, dict[tuple, tuple[float, int]]] = {name: {} for name in groups}
    global_sum, global_count = (0.0, 0)
    latest_history_month: str | None = None
    train_source = np.full(len(raw_train), None, dtype=object)
    monthly_coverage = []

    def query(keys: list[tuple | None], name: str, mean: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        state = histories[name]
        stats = [state.get(key, (0.0, 0)) if key is not None else (0.0, 0) for key in keys]
        sums = np.array([value[0] for value in stats], dtype=float)
        counts = np.array([value[1] for value in stats], dtype=float)
        encoded = (sums + smoothing * mean) / (counts + smoothing)
        return (encoded, np.log1p(counts), (counts > 0).astype(float))
    months_array = train_month.to_numpy()
    for month in sorted(train_month.unique()):
        positions = np.flatnonzero(months_array == month)
        global_mean = global_sum / global_count if global_count else np.nan
        train_source[positions] = latest_history_month
        for name in groups:
            keys = [train_keys[name][pos] for pos in positions]
            encoded, log_count, seen = query(keys, name, global_mean)
            for feature, values in zip(output_columns[name], (encoded, log_count, seen)):
                train_data[feature][positions] = values
            monthly_coverage.append({'dataset': 'train', 'month': month, 'group': name, 'rows': len(positions), 'seen_fraction': float(seen.mean()), 'available_global_past_labels': global_count, 'latest_label_month': latest_history_month})
        for name in groups:
            state = histories[name]
            for pos in positions:
                key = train_keys[name][pos]
                if usable_target[pos] and key is not None:
                    previous_sum, previous_count = state.get(key, (0.0, 0))
                    state[key] = (previous_sum + float(target[pos]), previous_count + 1)
        valid_positions = positions[usable_target[positions]]
        global_sum += float(target[valid_positions].sum())
        global_count += len(valid_positions)
        if len(valid_positions):
            latest_history_month = month
    frozen_mean = global_sum / global_count if global_count else np.nan
    for name in groups:
        encoded, log_count, seen = query(valid_keys[name], name, frozen_mean)
        for feature, values in zip(output_columns[name], (encoded, log_count, seen)):
            valid_data[feature] = values
        for month in sorted(valid_month.unique()):
            positions = np.flatnonzero(valid_month.to_numpy() == month)
            monthly_coverage.append({'dataset': 'valid', 'month': month, 'group': name, 'rows': len(positions), 'seen_fraction': float(seen[positions].mean()), 'available_global_past_labels': global_count, 'latest_label_month': latest_history_month})
    train_hist = pd.DataFrame(train_data, index=raw_train.index)
    valid_hist = pd.DataFrame(valid_data, index=raw_valid.index)
    train_hist['history_source_month_max'] = train_source
    valid_hist['history_source_month_max'] = latest_history_month
    audit = {'cutoff': cutoff, 'smoothing_fixed': float(smoothing), 'used_groups': groups, 'skipped_groups': skipped, 'group_feature_columns': output_columns, 'feature_columns': [col for names in output_columns.values() for col in names], 'metadata_columns': ['history_source_month_max'], 'monthly_coverage': monthly_coverage, 'training_rows': len(raw_train), 'validation_rows': len(raw_valid), 'valid_training_labels': int(usable_target.sum()), 'ignored_nonfinite_training_labels': int((~usable_target).sum()), 'same_month_labels_used_for_training_features': False, 'validation_labels_read': False, 'validation_history_updates': False, 'validation_frozen_global_mean': float(frozen_mean) if np.isfinite(frozen_mean) else None, 'interpretation': 'Smoothed past local-market rent proxies; facility identity does not identify school quality or a causal amenity effect', 'mrt_availability': {name: 'Nearest entity marked open in supplied snapshot; no historical opening chronology inferred' if any(('mrt_status_open_nearest' in col for col in cols)) else 'Fallback nearest supplied snapshot entity can include planned stations; no historical opening chronology inferred' for name, cols in groups.items() if any(('mrt' in col.lower() for col in cols))}, 'first_month_policy': 'Mean NaN, history count 0, seen flag 0 when no earlier global labels exist'}
    return (train_hist, valid_hist, lineage, audit)
