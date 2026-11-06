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
OUT = asset_path(Path('global_monthly_01'))
FEATURES = ['TOWN', 'building_age_years', 'max_floor', 'year_completed', 'flat_age_from_lease', 'approval_month_index', 'floor_area_sqm', 'fee_amount', 'lease_commence_year', 'latitude', 'longitude', 'block_key', 'street_key', 'flat_type_ord', 'flat_model_ord', 'flat_model_key', 'nearest_open_mrt_code', 'dist_nearest_open_mrt_km', 'log_dist_nearest_open_mrt_km', 'nearest_mall_name', 'dist_nearest_mall_km', 'nearest_primary_name', 'dist_nearest_primary_km', 'nearest_secondary_name', 'dist_nearest_secondary_km']
PREPROCESS = {'categorical': 'native', 'numeric_impute': 'median', 'rare_min_count': 30, 'scale': 'none'}
PARAMS = {'depth': 8, 'learning_rate': 0.05, 'iterations': 1000, 'loss_function': 'RMSE', 'random_seed': 5228, 'thread_count': 8, 'allow_writing_files': False, 'early_stopping_rounds': 50}
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
    assert FEATURES == [name for name in loaded if name != 'RENT_APPROVAL_DATE']
    assert PREPROCESS == ast.literal_eval(nodes['PREPROCESS'])
    assert ast.literal_eval(nodes['CAT_ARCH']) == {'max_depth': 8, 'learning_rate': 0.05, 'n_estimators': 1000, 'early_stopping_rounds': 50}

def prepare():
    check_original_constants()
    if OUT.exists():
        raise RuntimeError('Refusing to overwrite or restart this candidate')
    prior = read(asset_path(Path('frozen_mlp_area_confirm_01/results.json')))
    assert prior['search_rmse'] == 451.042648255612
    frozen = dict(read(asset_path(Path('feature_ablation_01/protocol.json')))['frozen_sha256'])
    for name in ['feature_ablation_01/protocol.json', 'frozen_mlp_area_confirm_01/results.json', 'feature_ablation_01/baseline_search_predictions.csv', 'global_monthly_catboost.py', 'spatial_history_io.py']:
        frozen[name] = digest(asset_path(Path(name)))
    OUT.mkdir()
    protocol = {'status': 'frozen_before_training', 'created_epoch': time.time(), 'candidate': 'global_catboost_direct_monthly_rent', 'candidate_count': 1, 'fit_end': '2024-09', 'search_months': ['2024-10', '2024-11', '2024-12'], 'confirm_months': ['2025-01', '2025-02', '2025-03'], 'rows': {'train': 132802, 'search': 8274, 'confirm': 8924}, 'target': 'original MONTHLY_RENT, unchanged', 'features': FEATURES, 'preprocess': PREPROCESS, 'params': PARAMS, 'weights': 'none', 'max_wall_seconds': 900, 'automatic_retry': False, 'budget_includes': 'imports, integrity checks, preparation, fit, Search, gated Confirm, saving', 'structure': 'One global CatBoost; no RNN/GMM/MLP fitting or inference; original baseline remains frozen', 'new_history_features': False, 'new_joint_address_feature': False, 'source_feature_values': 'Original unscaled load_joined values; training-only native category maps and medians', 'search_gate_rmse': prior['search_rmse'], 'gate_rule': 'strictly less than prior best Search', 'confirm_max_evaluations': 1, 'confirm_reselection': False, 'confirm_interpretation': 'Original split iterative validation; previously inspected, not an untouched test set', 'baseline': prior['baseline'], 'previous_best': {'search_rmse': prior['search_rmse'], 'confirm_rmse': prior['confirm']['rmse']}, 'frozen_sha256': frozen}
    write_json(OUT / 'protocol.json', protocol)
    print(json.dumps({'prepared': str(OUT), 'features': len(FEATURES), 'frozen_files': len(frozen), 'training_started': False}), flush=True)

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
    needed = ['RENT_APPROVAL_DATE', *FEATURES, 'MONTHLY_RENT']
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
    preparer = fit_preprocessor(PREPROCESS, fit, y_fit, FEATURES)
    fit_x = transform_frame(preparer, fit.drop(columns='MONTHLY_RENT'))
    search_x = transform_frame(preparer, search.drop(columns='MONTHLY_RENT'))
    categories = _category_columns(fit_x)
    assert len(fit_x.columns) == 25 and set(fit_x.columns) == set(FEATURES)
    assert not preparer.include_building_ and (not preparer.target_values_)
    write_json(OUT / 'data_audit.json', {'train_rows': len(fit), 'search_rows': len(search), 'original_inputs_and_labels_exactly_equal': True, 'train_source_row_id_range': [0, 132801], 'search_source_row_id_range': [132802, 141075], 'features': FEATURES, 'model_columns': list(fit_x.columns), 'categories': categories, 'numeric_fit_medians': preparer.numeric_fill_, 'confirm_read_for_fitting': False})
    joblib.dump(preparer, OUT / 'preprocessor.joblib')
    deadline()
    record('catboost_fit_enter', train_rows=len(fit), search_rows=len(search), features=25, params=PARAMS, direct_monthly_target=True, weights='none', fit_call_number=1)
    model = CatBoostRegressor(**PARAMS)
    model.fit(_fill_missing_categories(fit_x), y_fit, eval_set=(_fill_missing_categories(search_x), y_search), cat_features=categories, use_best_model=True, verbose=50)
    record('catboost_fit_complete', trees=model.tree_count_, best_iteration=model.get_best_iteration())
    deadline()
    joblib.dump(model, OUT / 'model.joblib')
    write_json(OUT / 'fit_audit.json', {'params': model.get_params(), 'resolved_params': model.get_all_params(), 'features': model.feature_names_, 'categorical_indices': model.get_cat_feature_indices(), 'trees': model.tree_count_, 'best_iteration': model.get_best_iteration(), 'fit_calls': 1, 'target': 'MONTHLY_RENT', 'training_weights': 'none', 'early_stopping_weights': 'none', 'eval_history': model.get_evals_result()})
    prediction = finite(model.predict(_fill_missing_categories(search_x)), 'Search predictions')
    search_csv = OUT / 'search_predictions.csv'
    pd.DataFrame({'source_row_id': cache['search_row_ids'], 'date': search.RENT_APPROVAL_DATE.to_numpy(), 'truth': y_search, 'prediction': prediction}).to_csv(search_csv, index=False)
    search_metrics = metrics(y_search, prediction, search.RENT_APPROVAL_DATE)
    score = search_metrics['rmse']
    search_metrics.update(delta_from_baseline=score - protocol['baseline']['search_rmse'], delta_from_previous_best=score - protocol['previous_best']['search_rmse'])
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
        frame = load_joined('train', ['RENT_APPROVAL_DATE', *FEATURES])
        split = time_split(frame, 3, None)
        assert [len(split.fit), len(split.search), len(split.confirm)] == [132802, 8274, 8924]
        assert split.fit_months[-1] == '2024-09' and split.search_months == protocol['search_months']
        assert split.confirm_months == protocol['confirm_months']
        row_ids = np.flatnonzero(frame.RENT_APPROVAL_DATE.astype(str).str[:7].isin(protocol['confirm_months']).to_numpy())
        assert np.array_equal(row_ids, np.arange(141076, 150000))
        confirm_x = transform_frame(preparer, split.confirm.drop(columns='MONTHLY_RENT'))
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
