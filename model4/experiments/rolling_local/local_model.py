"""Index-adjusted CatBoost phase; no work runs on import."""
from __future__ import annotations
import json
import os
from pathlib import Path
import time
from experiments.rolling_local._core import temporal_index as index_module
from experiments.rolling_local._core.keys import norm_text
from experiments.rolling_local._core import catboost_preprocess as cat_preprocess
from experiments.rolling_local._core import native_categories as cat_model



FEATURES = ['TOWN', 'building_age_years', 'max_floor', 'year_completed', 'flat_age_from_lease', 'approval_month_index', 'floor_area_sqm', 'fee_amount', 'lease_commence_year', 'latitude', 'longitude', 'block_key', 'street_key', 'flat_type_ord', 'flat_model_ord', 'flat_model_key', 'nearest_open_mrt_code', 'dist_nearest_open_mrt_km', 'log_dist_nearest_open_mrt_km', 'nearest_mall_name', 'dist_nearest_mall_km', 'nearest_primary_name', 'dist_nearest_primary_km', 'nearest_secondary_name', 'dist_nearest_secondary_km', 'full_address_key']

PREPROCESS = {'categorical': 'native', 'numeric_impute': 'median', 'rare_min_count': 30, 'scale': 'none'}

PARAMS = {'depth': 8, 'learning_rate': 0.05, 'iterations': 1000, 'loss_function': 'RMSE', 'random_seed': 5228, 'thread_count': 8, 'allow_writing_files': False, 'early_stopping_rounds': 50, 'ctr_target_border_count': 8}

def _write_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
    os.replace(temporary, path)

def _add_address(frame, norm_text):
    """Exact label-free TOWN|street_key|block_key definition from approved source."""
    result = frame.copy()
    keys = [norm_text(result[name]) for name in ('TOWN', 'street_key', 'block_key')]
    for key in keys:
        if not (key.notna().all() and key.str.len().gt(0).all()):
            raise ValueError('Missing/empty complete-address component')
        if key.str.contains('|', regex=False).any():
            raise ValueError('Complete-address component contains separator')
    result['full_address_key'] = keys[0] + '|' + keys[1] + '|' + keys[2]
    return result

def _finite(np, values, label: str, positive: bool=False):
    result = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.isfinite(result).all() or (positive and (not (result > 0).all())):
        raise ValueError(f'Invalid {label}')
    return result

class _DeadlineCallback:

    def __init__(self, deadline):
        self.deadline = deadline

    def after_iteration(self, info):
        self.deadline()
        return True

def run_local(train_df, apply_df, folder: Path, cutoff: str, selected: dict | None, event, deadline):
    """Fit one global_catboost_index phase and return row-preserving predictions and selection.

    selected=None: train through S; apply_df is the next three labeled months,
    used for weighted early stopping. Returned selection['K'] is tree_count_.
    selected=dict: train through T with exactly selected['K'] trees; apply_df
    contains sixteen future months and must not contain MONTHLY_RENT.
    """
    deadline()
    import joblib
    import numpy as np
    import pandas as pd
    from catboost import CatBoostRegressor, Pool
    deadline()
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for name in ('index.joblib', 'preprocessor.joblib', 'model.joblib', 'selection.json', 'fit_audit.json', 'prediction.csv', 'fit_target_audit.csv'):
        if (folder / name).exists():
            raise FileExistsError(f'Refusing global_catboost_index overwrite/retry: {folder / name}')
    inner = selected is None
    mode = 'inner_selection' if inner else 'full_fixed_refit'
    if not len(train_df) or not len(apply_df):
        raise ValueError('global_catboost_index requires nonempty train/apply frames')
    if 'MONTHLY_RENT' not in train_df:
        raise ValueError('Missing training labels')
    if not inner and 'MONTHLY_RENT' in apply_df:
        raise ValueError('Outer future labels must be removed before run_local')
    if inner and 'MONTHLY_RENT' not in apply_df:
        raise ValueError('Inner selection requires validation labels')
    for name, frame in (('train', train_df), ('apply', apply_df)):
        if 'source_row_id' not in frame or frame.source_row_id.isna().any():
            raise ValueError(f'Missing {name} source_row_id')
        if frame.source_row_id.duplicated().any():
            raise ValueError(f'Duplicate {name} source_row_id')
    if set(train_df.source_row_id) & set(apply_df.source_row_id):
        raise ValueError('Training and application row IDs overlap')
    train_month = train_df.RENT_APPROVAL_DATE.astype(str).str[:7]
    apply_month = apply_df.RENT_APPROVAL_DATE.astype(str).str[:7]
    if train_month.max() != cutoff or not train_month.le(cutoff).all():
        raise ValueError('global_catboost_index training cutoff mismatch')
    expected_months = [str(pd.Period(cutoff, freq='M') + h) for h in range(1, (3 if inner else 16) + 1)]
    if sorted(apply_month.unique()) != expected_months:
        raise ValueError('global_catboost_index application months do not match approved horizons')
    train = _add_address(train_df, norm_text)
    application = _add_address(apply_df, norm_text)
    missing = [name for name in FEATURES if name not in train or name not in application]
    if missing:
        raise ValueError(f'Missing global_catboost_index features: {missing}')
    y_train = _finite(np, train.MONTHLY_RENT, 'fit rent', positive=True)
    y_apply = _finite(np, application.MONTHLY_RENT, 'validation rent', positive=True) if inner else None
    history = index_module.prepare_rows(train)
    deadline()
    event('fit_enter', kind='index', phase=mode, cutoff=cutoff, rows=len(train))
    start = time.monotonic()
    index = index_module.fit_index(history)
    deadline()
    if index['fit_end'] != cutoff or index['fit_rows'] != len(train):
        raise ValueError('Index history identity mismatch')
    event('fit_complete', kind='index', phase=mode, cutoff=cutoff, elapsed_seconds=time.monotonic() - start, fit_rows=int(index['fit_rows']), eligible_fit_rows=int(index['eligible_fit_rows']), property_groups=int(index['property_groups']), slope=float(index['slope']))
    joblib.dump(index, folder / 'index.joblib')
    fit_scale = _finite(np, index_module.scales(index, train.RENT_APPROVAL_DATE), 'fit index', positive=True)
    apply_scale = _finite(np, index_module.scales(index, application.RENT_APPROVAL_DATE), 'apply index', positive=True)
    if not np.allclose(fit_scale[train_month.to_numpy() == cutoff], 1.0, rtol=0.0, atol=1e-12):
        raise ValueError('Index is not normalized to I(cutoff)=1')
    normalized_train = _finite(np, y_train / fit_scale, 'normalized fit target', positive=True)
    fit_weight = _finite(np, fit_scale ** 2 / np.mean(fit_scale ** 2), 'fit weights', positive=True)
    apply_weight = _finite(np, apply_scale ** 2 / np.mean(apply_scale ** 2), 'validation weights', positive=True) if inner else None
    pd.DataFrame({'source_row_id': train.source_row_id.to_numpy(), 'date': train.RENT_APPROVAL_DATE.to_numpy(), 'truth': y_train, 'time_scale': fit_scale, 'normalized_target': normalized_train, 'weight': fit_weight}).to_csv(folder / 'fit_target_audit.csv', index=False)
    if inner:
        pd.DataFrame({'source_row_id': application.source_row_id.to_numpy(), 'date': application.RENT_APPROVAL_DATE.to_numpy(), 'truth': y_apply, 'time_scale': apply_scale, 'normalized_target': y_apply / apply_scale, 'weight': apply_weight}).to_csv(folder / 'validation_target_audit.csv', index=False)
    deadline()
    preparer = cat_preprocess.fit_preprocessor(PREPROCESS, train, normalized_train, FEATURES)
    x_train = cat_preprocess.transform_frame(preparer, train.drop(columns='MONTHLY_RENT'))
    x_apply = cat_preprocess.transform_frame(preparer, application.drop(columns='MONTHLY_RENT', errors='ignore'))
    if preparer.selected_columns_ != FEATURES or len(x_train.columns) != 26 or set(x_train.columns) != set(FEATURES) or (list(x_train.columns) != list(x_apply.columns)):
        raise ValueError('global_catboost_index 26-column matrix identity mismatch')
    if preparer.include_building_ or preparer.target_values_:
        raise ValueError('Unexpected target encoding in native global_catboost_index preprocessor')
    category_columns = cat_model._category_columns(x_train)
    x_train = cat_model._fill_missing_categories(x_train)
    x_apply = cat_model._fill_missing_categories(x_apply)
    joblib.dump(preparer, folder / 'preprocessor.joblib')
    _write_json(folder / 'preprocessor_statistics.json', {'cutoff': cutoff, 'fit_rows': len(train), 'features': FEATURES, 'output_columns': list(preparer.output_columns_), 'spec': dict(preparer.spec), 'numeric_fill': {key: float(value) for key, value in preparer.numeric_fill_.items()}, 'numeric_center': {key: float(value) for key, value in preparer.numeric_center_.items()}, 'numeric_spread': {key: float(value) for key, value in preparer.numeric_spread_.items()}, 'category_level_counts': {key: len(value) for key, value in preparer.levels_.items()}, 'target_encoding_used': False})
    params = dict(PARAMS)
    if not inner:
        if selected.get('model') != 'global_catboost_index':
            raise ValueError('Wrong model capacity selection')
        expected_selection_cutoff = str(pd.Period(cutoff, freq='M') - 3)
        if selected.get('cutoff') != expected_selection_cutoff:
            raise ValueError("global_catboost_index K was not selected at this origin's S=T-3")
        trees = int(selected['K'])
        if not 1 <= trees <= PARAMS['iterations']:
            raise ValueError('Invalid fixed global_catboost_index K')
        params['iterations'] = trees
        params.pop('early_stopping_rounds')
    fit_pool = Pool(x_train, normalized_train, weight=fit_weight, cat_features=category_columns)
    validation_pool = Pool(x_apply, y_apply / apply_scale, weight=apply_weight, cat_features=category_columns) if inner else None
    model = CatBoostRegressor(**params)
    deadline()
    event('fit_enter', kind='catboost', phase=mode, cutoff=cutoff, rows=len(train), validation_rows=len(application) if inner else 0, maximum_trees=int(params['iterations']))
    start = time.monotonic()
    if inner:
        model.fit(fit_pool, eval_set=validation_pool, use_best_model=True, verbose=False, callbacks=[_DeadlineCallback(deadline)])
    else:
        model.fit(fit_pool, use_best_model=False, verbose=False, callbacks=[_DeadlineCallback(deadline)])
    deadline()
    trees = int(model.tree_count_)
    if trees < 1 or (not inner and trees != int(selected['K'])):
        raise ValueError('CatBoost kept tree count violates fixed capacity')
    best_iteration = model.get_best_iteration()
    best_iteration = int(best_iteration) if best_iteration is not None else None
    event('fit_complete', kind='catboost', phase=mode, cutoff=cutoff, elapsed_seconds=time.monotonic() - start, trees=trees, best_iteration=best_iteration)
    joblib.dump(model, folder / 'model.joblib')
    deadline()
    normalized_prediction = _finite(np, model.predict(x_apply), 'normalized predictions')
    prediction = _finite(np, normalized_prediction * apply_scale, 'monthly predictions')
    if len(prediction) != len(application):
        raise ValueError('global_catboost_index prediction length mismatch')
    eval_history = model.get_evals_result()
    selection = {'model': 'global_catboost_index', 'K': trees, 'phase': mode, 'cutoff': cutoff, 'selected_from_cutoff': cutoff if inner else selected['cutoff'], 'features': FEATURES, 'target': 'MONTHLY_RENT / I(month)', 'weight': 'I(month)^2 / mean(I(month)^2)', 'validation_weight_passed': inner, 'validation_months': expected_months if inner else [], 'early_stopping': inner, 'tree_cap_hit': trees == PARAMS['iterations']}
    if inner:
        rent_rmse = float(np.sqrt(np.mean((prediction - y_apply) ** 2)))
        weighted_rmse = float(np.sqrt(np.average((normalized_prediction - y_apply / apply_scale) ** 2, weights=apply_weight)))
        factor = float(np.sqrt(np.mean(apply_scale ** 2)))
        recorded_minimum = float(min(eval_history['validation']['RMSE']))
        manual_difference = abs(rent_rmse - weighted_rmse * factor)
        library_difference = abs(rent_rmse - recorded_minimum * factor)
        if manual_difference >= 0.001 or library_difference >= 0.001:
            raise ValueError(f'Weighted monthly-RMSE conversion failed: {manual_difference}, {library_difference}')
        selection.update(validation_rent_rmse=rent_rmse, validation_weighted_normalized_rmse=weighted_rmse, catboost_minimum_validation_rmse=recorded_minimum, metric_conversion_factor=factor, manual_metric_conversion_abs_error=manual_difference, library_metric_conversion_abs_error=library_difference)
    audit = {'mode': mode, 'cutoff': cutoff, 'train_rows': len(train), 'apply_rows': len(application), 'params': model.get_params(), 'resolved_params': model.get_all_params(), 'features': model.feature_names_, 'categorical_indices': model.get_cat_feature_indices(), 'evaluation_history': eval_history, 'selection': selection, 'index': {key: int(index[key]) if key in ('fit_rows', 'eligible_fit_rows', 'property_groups', 'slope_months') else float(index[key]) if key in ('slope', 'damping', 'normal_equation_max_month_residual_sum', 'normal_equation_max_property_residual_sum', 'time_gram_condition_number') else index[key] for key in ('months', 'fit_end', 'fit_rows', 'eligible_fit_rows', 'property_groups', 'slope', 'damping', 'slope_months', 'normal_equation_max_month_residual_sum', 'normal_equation_max_property_residual_sum', 'time_gram_condition_number')}, 'source_row_order_preserved': True, 'no_old_weights_loaded': True, 'outer_labels_supplied': False if not inner else None}
    _write_json(folder / 'selection.json', selection)
    _write_json(folder / 'fit_audit.json', audit)
    table = pd.DataFrame({'source_row_id': application.source_row_id.to_numpy(), 'date': application.RENT_APPROVAL_DATE.to_numpy(), 'time_scale': apply_scale, 'normalized_prediction': normalized_prediction, 'prediction': prediction})
    if inner:
        table['truth'] = y_apply
    table.to_csv(folder / 'prediction.csv', index=False)
    deadline()
    event('local_phase_complete', phase=mode, cutoff=cutoff, prediction_rows=len(prediction), trees=trees)
    return (prediction, selection)
