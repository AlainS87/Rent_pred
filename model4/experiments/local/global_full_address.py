"""One global, direct-monthly-rent CatBoost candidate. No retry or other fits."""
from experiments.local.bootstrap import Path, CODE_DIR, DATA_ROOT, SUPPORT_ROOT, ARTIFACT_ROOT, asset_path, asset_key, support_path
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
from experiments.local.spatial_history_io import append_json, write_json
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
SOURCE = asset_path(Path('support'))
OUT = asset_path(Path('global_full_address_01'))
PRIOR = asset_path(Path('global_monthly_ctr8_01'))
BASE_FEATURES = ['TOWN', 'building_age_years', 'max_floor', 'year_completed', 'flat_age_from_lease', 'approval_month_index', 'floor_area_sqm', 'fee_amount', 'lease_commence_year', 'latitude', 'longitude', 'block_key', 'street_key', 'flat_type_ord', 'flat_model_ord', 'flat_model_key', 'nearest_open_mrt_code', 'dist_nearest_open_mrt_km', 'log_dist_nearest_open_mrt_km', 'nearest_mall_name', 'dist_nearest_mall_km', 'nearest_primary_name', 'dist_nearest_primary_km', 'nearest_secondary_name', 'dist_nearest_secondary_km']
EXTRA = 'full_address_key'
FEATURES = [*BASE_FEATURES, EXTRA]
PREPROCESS = {'categorical': 'native', 'numeric_impute': 'median', 'rare_min_count': 30, 'scale': 'none'}
PARAMS = {'depth': 8, 'learning_rate': 0.05, 'iterations': 1000, 'loss_function': 'RMSE', 'random_seed': 5228, 'thread_count': 8, 'allow_writing_files': False, 'early_stopping_rounds': 50, 'ctr_target_border_count': 8}
LAST_STAGE = 'not_started'

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()

def record(stage, **fields):
    global LAST_STAGE
    event = {'stage': stage, 'pid': os.getpid(), 'epoch': time.time(), **fields}
    append_json(OUT / 'events.jsonl', event)
    if not write_json(OUT / 'status.json', event, required=False):
        append_json(OUT / 'events.jsonl', {'stage': 'status_snapshot_skipped', 'pid': os.getpid(), 'epoch': time.time()})
    LAST_STAGE = stage
    print(json.dumps(event, allow_nan=False), flush=True)

def check_original_constants():
    path = support_path(Path('stack.py'))
    module = ast.parse(path.read_text(encoding='utf-8'))
    nodes = {node.targets[0].id: node.value for node in module.body if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)}
    cat = ast.literal_eval(nodes['CAT_FEATURES'])
    loaded = []
    for node in nodes['LOAD_COLUMNS'].elts:
        loaded.extend(cat if isinstance(node, ast.Starred) else [ast.literal_eval(node)])
    assert BASE_FEATURES == [name for name in loaded if name != 'RENT_APPROVAL_DATE']
    prior = read(PRIOR / 'protocol.json')
    assert prior['features'] == BASE_FEATURES and prior['params'] == PARAMS
    assert prior['preprocess'] == PREPROCESS
    assert PREPROCESS == ast.literal_eval(nodes['PREPROCESS'])
    assert ast.literal_eval(nodes['CAT_ARCH']) == {'max_depth': 8, 'learning_rate': 0.05, 'n_estimators': 1000, 'early_stopping_rounds': 50}

def add_address(frame):
    """Label-free native category; preserve all original columns."""
    from experiments.local._support.src.features.keys import norm_text
    result = frame.copy()
    keys = [norm_text(frame[name]) for name in ('TOWN', 'street_key', 'block_key')]
    for key in keys:
        assert key.notna().all() and key.str.len().gt(0).all()
        assert not key.str.contains('|', regex=False).any()
    result[EXTRA] = keys[0] + '|' + keys[1] + '|' + keys[2]
    assert result[EXTRA].notna().all()
    return result

def validate_preparer(preparer, previous):
    assert preparer.selected_columns_ == [*previous.selected_columns_, EXTRA]
    assert preparer.categorical_columns_ == [*previous.categorical_columns_, EXTRA]
    assert preparer.output_columns_ == [*previous.output_columns_, EXTRA]
    for name, value in previous.__dict__.items():
        if name in ('selected_columns_', 'categorical_columns_', 'output_columns_'):
            continue
        if name == 'levels_':
            assert set(preparer.levels_) == set(previous.levels_) | {EXTRA}
            assert all((preparer.levels_[key] == levels for key, levels in previous.levels_.items()))
        else:
            assert getattr(preparer, name) == value, name
    return True

def validate_fit_audit(audit):
    prior = read(PRIOR / 'fit_audit.json')
    assert audit['params'] == prior['params'] == PARAMS
    assert audit['resolved_params'] == prior['resolved_params']
    assert audit['features'] == [*prior['features'], EXTRA]
    assert audit['categorical_indices'] == [*prior['categorical_indices'], len(BASE_FEATURES)]
    assert audit['training_weights'] == audit['early_stopping_weights'] == 'none'
    return {'only_feature_added': EXTRA, 'all_constructor_and_resolved_parameters_unchanged': True}

def fixed_search_diagnostics():
    import numpy as np
    import pandas as pd
    previous = pd.read_csv(PRIOR / 'search_predictions.csv')
    current = pd.read_csv(OUT / 'search_predictions.csv')
    assert np.array_equal(previous.source_row_id, current.source_row_id)
    assert np.array_equal(previous.truth, current.truth)
    old_error = previous.prediction.to_numpy() - previous.truth.to_numpy()
    new_error = current.prediction.to_numpy() - current.truth.to_numpy()
    worst = np.zeros(len(current), dtype=bool)
    worst[np.argsort(np.abs(old_error))[-828:]] = True
    masks = {'current_global_worst_10pct': worst, 'other_90pct': ~worst, 'current_global_overprediction_ge_1000': old_error >= 1000, 'truth_at_most_1500': previous.truth.to_numpy() <= 1500}
    return {name: {'rows': int(mask.sum()), 'current_global_rmse': float(np.sqrt(np.mean(old_error[mask] ** 2))), 'candidate_rmse': float(np.sqrt(np.mean(new_error[mask] ** 2)))} for name, mask in masks.items()}

def prepare():
    check_original_constants()
    if OUT.exists():
        raise RuntimeError('Refusing to overwrite or restart this candidate')
    previous = read(PRIOR / 'results.json')
    assert previous['search']['rmse'] == 445.3804031850469
    assert previous['confirm']['rmse'] == 452.87416556913826
    prior_protocol = read(PRIOR / 'protocol.json')
    assert digest(PRIOR / 'protocol.json') == previous['protocol_sha256']
    frozen = dict(prior_protocol['frozen_sha256'])
    for name in ['protocol.json', 'results.json', 'selection_frozen.json', 'model.joblib', 'preprocessor.joblib', 'fit_audit.json', 'data_audit.json', 'search_predictions.csv', 'confirm_predictions.csv', 'search_metrics.json', 'confirmation_metrics.json', 'completion_verification.json']:
        relative = 'global_monthly_ctr8_01/' + name
        frozen[relative] = digest(asset_path(Path(relative)))
    for name in ['global_full_address.py', 'verify_global_full_address.py', 'global_original_blend_50_50_01/results.json']:
        frozen[name] = digest(asset_path(Path(name)))
    protocol = {**prior_protocol, 'created_epoch': time.time(), 'candidate': 'global_ctr8_plus_explicit_full_address', 'features': FEATURES, 'params': PARAMS, 'frozen_sha256': frozen, 'search_gate_rmse': previous['search']['rmse'], 'previous_best': {'search_rmse': previous['search']['rmse'], 'confirm_rmse': previous['confirm']['rmse']}, 'only_change': 'Add one native categorical full_address_key; no other predictor or parameter changes', 'new_joint_address_feature': True, 'full_address_definition': "norm_text(TOWN) + '|' + norm_text(street_key) + '|' + norm_text(block_key)", 'address_uses_labels': False, 'original_features': BASE_FEATURES, 'search_gate_authorization': 'User explicitly selected 445.3804/current global model; use full precision 445.3804031850469', 'historical_blend_comparison_only': {'search_rmse': 442.9406238420111, 'confirm_rmse': 454.36702064571455}, 'authorization': 'One explicit-address candidate, CPU8, 900 seconds including preparation and gated one-time Confirm; no retry', 'supplementary_diagnostics': 'Fixed current CTR8 Search worst-error groups; report only, no alternate selection rule'}
    protocol.pop('one_parameter_change', None)
    OUT.mkdir()
    write_json(OUT / 'protocol.json', protocol)
    print(json.dumps({'prepared': str(OUT), 'features': len(FEATURES), 'only_new_feature': EXTRA, 'search_gate_rmse': protocol['search_gate_rmse'], 'frozen_files': len(frozen), 'training_started': False}))

def check_frozen(protocol):
    for relative, expected in protocol['frozen_sha256'].items():
        if digest(asset_path(Path(relative))) != expected:
            raise RuntimeError(f'Frozen source/data/baseline/protocol changed: {relative}')

def run():
    with (OUT / 'worker_identity.json').open('x', encoding='utf-8') as handle:
        json.dump({'pid': os.getpid(), 'epoch': time.time()}, handle)
    budget = read(OUT / 'budget.json')
    protocol = read(OUT / 'protocol.json')
    protocol_hash = digest(OUT / 'protocol.json')
    assert protocol['features'] == FEATURES and protocol['params'] == PARAMS
    assert budget['limit_seconds'] == 900

    def deadline():
        if time.time() >= budget['deadline_epoch']:
            raise TimeoutError('The authorized 900-second wall-time budget has expired')
    record('preflight_start', limit_seconds=900)
    check_original_constants()
    check_frozen(protocol)
    deadline()
    import joblib
    import numpy as np
    import pandas as pd
    from catboost import CatBoostRegressor
    from experiments.local._support.models.catboost.preprocess import fit_preprocessor, transform_frame
    from experiments.local._support.models.catboost.model import _fill_missing_categories, _category_columns
    from experiments.local._support.src.pipeline.data import load_joined, time_split

    def finite(values, name):
        array = np.asarray(values, dtype=float)
        if not np.isfinite(array).all():
            raise ValueError(f'Nonfinite {name}')
        return array

    def metrics(truth, prediction, dates):
        truth, prediction = (finite(truth, 'truth'), finite(prediction, 'prediction'))
        errors = prediction - truth
        months = np.asarray(dates).astype(str)
        months = np.array([value[:7] for value in months])
        return {'rows': len(truth), 'rmse': float(np.sqrt(np.mean(errors ** 2))), 'mae': float(np.mean(np.abs(errors))), 'bias_pred_minus_truth': float(np.mean(errors)), 'by_month': {month: {'rows': int((months == month).sum()), 'rmse': float(np.sqrt(np.mean(errors[months == month] ** 2)))} for month in sorted(set(months))}}
    cache = joblib.load(asset_path(Path('feature_ablation_01/training_cache.joblib')))
    fit, search = (cache['parts']['train'].copy(), cache['parts']['search'].copy())
    assert len(fit) == 132802 and len(search) == 8274
    assert fit.RENT_APPROVAL_DATE.astype(str).str[:7].max() == '2024-09'
    assert sorted(search.RENT_APPROVAL_DATE.astype(str).str[:7].unique()) == protocol['search_months']
    assert np.array_equal(cache['search_row_ids'], np.arange(132802, 141076))
    needed = ['RENT_APPROVAL_DATE', *BASE_FEATURES, 'MONTHLY_RENT']
    source_path = support_path(Path('Data/train.csv'))
    auxiliary_path = support_path(Path('Data/auxiliary_features/train_features.csv'))
    source_names = set(pd.read_csv(source_path, nrows=0).columns)
    source_columns = [name for name in needed if name in source_names]
    auxiliary_columns = [name for name in needed if name not in source_names]
    source = pd.read_csv(source_path, nrows=141076, usecols=source_columns, dtype={'TOWN': str})
    auxiliary = pd.read_csv(auxiliary_path, nrows=141076, usecols=auxiliary_columns)
    joined = pd.concat([source, auxiliary], axis=1)
    pd.testing.assert_frame_equal(fit[needed].reset_index(drop=True), joined.iloc[:132802][needed].reset_index(drop=True), check_dtype=False, check_exact=True)
    pd.testing.assert_frame_equal(search[needed].reset_index(drop=True), joined.iloc[132802:][needed].reset_index(drop=True), check_dtype=False, check_exact=True)
    y_fit, y_search = (finite(fit.MONTHLY_RENT, 'fit truth'), finite(search.MONTHLY_RENT, 'Search truth'))
    original = pd.read_csv(asset_path(Path('feature_ablation_01/baseline_search_predictions.csv')))
    assert np.array_equal(original.source_row_id.to_numpy(), cache['search_row_ids'])
    assert np.array_equal(original.truth.to_numpy(), y_search)
    assert abs(metrics(y_search, original.prediction, search.RENT_APPROVAL_DATE)['rmse'] - protocol['baseline']['search_rmse']) < 1e-09
    fit, search = (add_address(fit), add_address(search))
    preparer = fit_preprocessor(PREPROCESS, fit, y_fit, FEATURES)
    fit_x = transform_frame(preparer, fit.drop(columns='MONTHLY_RENT'))
    search_x = transform_frame(preparer, search.drop(columns='MONTHLY_RENT'))
    categories = _category_columns(fit_x)
    assert len(fit_x.columns) == 26 and set(fit_x.columns) == set(FEATURES)
    previous_preparer = joblib.load(PRIOR / 'preprocessor.joblib')
    validate_preparer(preparer, previous_preparer)
    for frame_part, matrix in [(fit, fit_x), (search, search_x)]:
        prior_matrix = transform_frame(previous_preparer, frame_part.drop(columns='MONTHLY_RENT'))
        pd.testing.assert_frame_equal(matrix[previous_preparer.output_columns_], prior_matrix, check_exact=True)
    assert EXTRA in categories
    address_audit = {'construction_uses_labels': False, 'original_25_transformed_features_exactly_equal': True, 'train_unique_addresses': int(fit[EXTRA].nunique()), 'search_unique_addresses': int(search[EXTRA].nunique()), 'search_unseen_address_rows': int((~search[EXTRA].isin(set(fit[EXTRA]))).sum()), 'no_null_or_empty_address': bool(fit[EXTRA].notna().all() and search[EXTRA].notna().all())}
    assert fit[['TOWN', 'street_key', 'block_key']].drop_duplicates().shape[0] == fit[EXTRA].nunique()
    assert not preparer.include_building_ and (not preparer.target_values_)
    write_json(OUT / 'data_audit.json', {'train_rows': len(fit), 'search_rows': len(search), 'original_inputs_and_labels_exactly_equal': True, 'train_source_row_id_range': [0, 132801], 'search_source_row_id_range': [132802, 141075], 'features': FEATURES, 'model_columns': list(fit_x.columns), 'categories': categories, 'numeric_fit_medians': preparer.numeric_fill_, 'confirm_read_for_fitting': False, 'address_audit': address_audit})
    joblib.dump(preparer, OUT / 'preprocessor.joblib')
    deadline()
    record('catboost_fit_enter', train_rows=len(fit), search_rows=len(search), features=26, params=PARAMS, direct_monthly_target=True, weights='none', fit_call_number=1)
    model = CatBoostRegressor(**PARAMS)
    model.fit(_fill_missing_categories(fit_x), y_fit, eval_set=(_fill_missing_categories(search_x), y_search), cat_features=categories, use_best_model=True, verbose=50)
    record('catboost_fit_complete', trees=model.tree_count_, best_iteration=model.get_best_iteration())
    deadline()
    joblib.dump(model, OUT / 'model.joblib')
    fit_audit = {'params': model.get_params(), 'resolved_params': model.get_all_params(), 'features': model.feature_names_, 'categorical_indices': model.get_cat_feature_indices(), 'trees': model.tree_count_, 'best_iteration': model.get_best_iteration(), 'fit_calls': 1, 'target': 'MONTHLY_RENT', 'training_weights': 'none', 'early_stopping_weights': 'none', 'eval_history': model.get_evals_result()}
    fit_audit['one_feature_change_verification'] = validate_fit_audit(fit_audit)
    write_json(OUT / 'fit_audit.json', fit_audit)
    prediction = finite(model.predict(_fill_missing_categories(search_x)), 'Search predictions')
    search_csv = OUT / 'search_predictions.csv'
    pd.DataFrame({'source_row_id': cache['search_row_ids'], 'date': search.RENT_APPROVAL_DATE.to_numpy(), 'truth': y_search, 'prediction': prediction}).to_csv(search_csv, index=False)
    search_metrics = metrics(y_search, prediction, search.RENT_APPROVAL_DATE)
    score = search_metrics['rmse']
    search_metrics.update(delta_from_baseline=score - protocol['baseline']['search_rmse'], delta_from_previous_best=score - protocol['previous_best']['search_rmse'])
    search_metrics['fixed_prior_error_groups'] = fixed_search_diagnostics()
    search_metrics['delta_from_historical_blend_search'] = score - protocol['historical_blend_comparison_only']['search_rmse']
    write_json(OUT / 'search_metrics.json', search_metrics)
    passed = score < protocol['search_gate_rmse']
    selection = {'candidate': protocol['candidate'], 'passed_search_gate': passed, 'search_rmse': score, 'threshold': protocol['search_gate_rmse'], 'epoch': time.time(), 'protocol_sha256': protocol_hash, 'confirm_evaluations_before_freeze': 0, 'candidate_sha256': {name: digest(OUT / name) for name in ['model.joblib', 'preprocessor.joblib', 'search_predictions.csv', 'fit_audit.json', 'data_audit.json']}}
    write_json(OUT / 'selection_frozen.json', selection)
    selection_hash = digest(OUT / 'selection_frozen.json')
    record('selection_frozen', passed_search_gate=passed, search_rmse=score)
    result = {'status': 'completed', 'candidate': protocol['candidate'], 'search': search_metrics, 'confirm': None, 'candidate_confirm_evaluations': 0, 'passed_search_gate': passed, 'baseline': protocol['baseline'], 'previous_best': protocol['previous_best'], 'fit_calls': 1, 'protocol_sha256': protocol_hash, 'selection_sha256': selection_hash, 'search_predictions_csv': str(search_csv), 'confirm_predictions_csv': None, 'original_labels_unchanged': True, 'original_split_unchanged': True}
    if passed:
        deadline()
        with (OUT / 'confirm_started.json').open('x', encoding='utf-8') as handle:
            json.dump({'epoch': time.time(), 'pid': os.getpid(), 'selection_sha256': selection_hash}, handle)
        record('confirm_prediction_started', evaluation_number=1)
        frame = load_joined('train', ['RENT_APPROVAL_DATE', *BASE_FEATURES])
        split = time_split(frame, 3, None)
        assert [len(split.fit), len(split.search), len(split.confirm)] == [132802, 8274, 8924]
        assert split.fit_months[-1] == '2024-09' and split.search_months == protocol['search_months']
        assert split.confirm_months == protocol['confirm_months']
        row_ids = np.flatnonzero(frame.RENT_APPROVAL_DATE.astype(str).str[:7].isin(protocol['confirm_months']).to_numpy())
        assert np.array_equal(row_ids, np.arange(141076, 150000))
        confirm_input = add_address(split.confirm.drop(columns='MONTHLY_RENT'))
        confirm_x = transform_frame(preparer, confirm_input)
        prior_confirm_x = transform_frame(previous_preparer, confirm_input)
        pd.testing.assert_frame_equal(confirm_x[previous_preparer.output_columns_], prior_confirm_x, check_exact=True)
        confirm_prediction = finite(model.predict(_fill_missing_categories(confirm_x)), 'Confirm predictions')
        confirm_truth = finite(split.confirm.MONTHLY_RENT, 'Confirm truth')
        confirm_path = OUT / 'confirm_predictions.csv'
        pd.DataFrame({'source_row_id': row_ids, 'date': split.confirm.RENT_APPROVAL_DATE.to_numpy(), 'truth': confirm_truth, 'prediction': confirm_prediction}).to_csv(confirm_path, index=False)
        confirm_metrics = metrics(confirm_truth, confirm_prediction, split.confirm.RENT_APPROVAL_DATE)
        confirm_metrics.update(delta_from_baseline=confirm_metrics['rmse'] - protocol['baseline']['confirm_rmse'], delta_from_previous_best=confirm_metrics['rmse'] - protocol['previous_best']['confirm_rmse'])
        write_json(OUT / 'confirmation_metrics.json', confirm_metrics)
        improved = confirm_metrics['rmse'] < protocol['previous_best']['confirm_rmse']
        result.update(confirm=confirm_metrics, candidate_confirm_evaluations=1, confirm_predictions_csv=str(confirm_path), confirm_predictions_sha256=digest(confirm_path), improved_over_previous_best_on_both_periods=improved, conclusion='Improved both original periods; iterative validation, not an untouched test set.' if improved else 'Search improved but Confirm did not beat the previous best; optimization not established. No reselection.')
        record('confirm_complete', rmse=confirm_metrics['rmse'])
    else:
        result.update(improved_over_previous_best_on_both_periods=False, conclusion='Search did not beat previous best; Confirm not evaluated. No retries.')
    deadline()
    check_frozen(protocol)
    assert digest(OUT / 'protocol.json') == protocol_hash
    assert digest(OUT / 'selection_frozen.json') == selection_hash
    for relative, expected in selection['candidate_sha256'].items():
        assert digest(OUT / relative) == expected
    deadline()
    result.update(source_baseline_and_previous_results_unchanged=True, exit_code=0, elapsed_seconds=time.time() - budget['started_epoch'])
    write_json(OUT / 'results.json', result)
    record('completed', search_rmse=score, confirm_rmse=result['confirm']['rmse'] if passed else None, conclusion=result['conclusion'])
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['prepare', 'run'])
    args = parser.parse_args()
    try:
        prepare() if args.phase == 'prepare' else run()
    except BaseException as exc:
        traceback.print_exc()
        if args.phase == 'run' and OUT.exists():
            prior_stage = LAST_STAGE
            write_json(OUT / 'failure_report.json', {'status': 'failed', 'error': repr(exc), 'last_successful_stage': prior_stage, 'epoch': time.time(), 'exit_code': 1}, required=False)
            record('failed', error=repr(exc), last_successful_stage=prior_stage)
        raise
