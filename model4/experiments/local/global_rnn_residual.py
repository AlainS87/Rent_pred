"""One frozen-RNN trend plus global monthly-residual CatBoost candidate."""
from experiments.local.bootstrap import Path, CODE_DIR, DATA_ROOT, SUPPORT_ROOT, ARTIFACT_ROOT, asset_path, asset_key, support_path
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from experiments.local.global_full_address import BASE_FEATURES, FEATURES, PARAMS, PREPROCESS, add_address, digest, read
from experiments.local.spatial_history_io import append_json, write_json
ROOT = Path(__file__).resolve().parent
SOURCE = asset_path(Path('support'))
PRIOR = asset_path(Path('global_full_address_01'))
OUT = asset_path(Path('global_rnn_residual_01'))
RNN = asset_path(Path('baseline_alignment/regional/rnn/h16_l3'))
LAST_STAGE = 'not_started'
TARGET = 'MONTHLY_RENT - frozen_rnn_unit_level * floor_area_sqm'

def check_frozen(protocol):
    for relative, expected in protocol['frozen_sha256'].items():
        if digest(asset_path(Path(relative))) != expected:
            raise RuntimeError(f'Frozen file changed: {relative}')

def record(stage, **fields):
    global LAST_STAGE
    event = {'stage': stage, 'pid': os.getpid(), 'epoch': time.time(), **fields}
    append_json(OUT / 'events.jsonl', event)
    write_json(OUT / 'status.json', event, required=False)
    LAST_STAGE = stage
    print(json.dumps(event, allow_nan=False), flush=True)

def prepare():
    if OUT.exists():
        raise RuntimeError('Refusing to overwrite or restart this candidate')
    prior_protocol = read(PRIOR / 'protocol.json')
    prior = read(PRIOR / 'results.json')
    assert digest(PRIOR / 'protocol.json') == prior['protocol_sha256']
    assert prior['search']['rmse'] == 444.9106685519473
    assert prior['confirm']['rmse'] == 452.71911314177345
    assert prior_protocol['features'] == FEATURES and prior_protocol['params'] == PARAMS
    check_frozen(prior_protocol)
    frozen = dict(prior_protocol['frozen_sha256'])
    for name in ['protocol.json', 'results.json', 'selection_frozen.json', 'model.joblib', 'preprocessor.joblib', 'fit_audit.json', 'data_audit.json', 'search_predictions.csv', 'confirm_predictions.csv', 'search_metrics.json', 'confirmation_metrics.json', 'completion_verification.json']:
        relative = 'global_full_address_01/' + name
        frozen[relative] = digest(asset_path(Path(relative)))
    for name in ['global_rnn_residual.py', 'verify_global_rnn_residual.py']:
        frozen[name] = digest(asset_path(Path(name)))
    protocol = {'status': 'frozen_before_training', 'created_epoch': time.time(), 'candidate': 'frozen_original_rnn_plus_global_monthly_residual_catboost', 'candidate_count': 1, 'fit_end': '2024-09', 'search_months': prior_protocol['search_months'], 'confirm_months': prior_protocol['confirm_months'], 'rows': prior_protocol['rows'], 'features': FEATURES, 'params': PARAMS, 'preprocess': PREPROCESS, 'preprocessor_reused_byte_for_byte': 'global_full_address_01/preprocessor.joblib', 'observed_target': 'Original MONTHLY_RENT unchanged', 'training_target': TARGET, 'prediction': 'frozen_rnn_unit_level * floor_area_sqm + global_catboost_residual', 'evaluation': 'Unweighted original monthly-rent RMSE', 'early_stopping': 'Unweighted residual RMSE, mathematically equal to final monthly-rent RMSE', 'only_change': 'Replace direct monthly-rent fitting by additive frozen RNN trend and monthly residual fitting', 'rnn_weights_and_scaling': 'baseline_alignment/regional/rnn/h16_l3; frozen, no fitting', 'rnn_inputs': read(RNN / 'config.json')['columns'], 'rnn_fit_search_source': 'Verified feature_ablation_01/training_cache.joblib mu arrays', 'rnn_confirmation': 'One gated forward pass using original weights and label-free prefix timeline', 'gmm_mlp': 'Unchanged baseline artifacts; not used by this candidate', 'fit_count': 1, 'rnn_fit_count': 0, 'weights': 'none', 'max_wall_seconds': 900, 'automatic_retry': False, 'budget_includes': 'imports, integrity checks, data preparation, fit, Search, gated Confirm, saving', 'search_gate_rmse': prior['search']['rmse'], 'gate_rule': 'strictly below current accepted best Search', 'confirm_max_evaluations': 1, 'confirm_reselection': False, 'confirm_interpretation': 'Original split iterative validation; not an untouched test set', 'long_horizon_accuracy_validated': False, 'long_horizon_limitation': 'Fixed Confirm covers months 4-6 after fit; no labeled 2025-04 to 2026-07 evaluation', 'baseline': prior_protocol['baseline'], 'previous_best': {'search_rmse': prior['search']['rmse'], 'confirm_rmse': prior['confirm']['rmse']}, 'authorization': 'User approved trying the time-trend direction; one candidate under existing 900-second/no-retry rules', 'frozen_sha256': frozen}
    OUT.mkdir()
    write_json(OUT / 'protocol.json', protocol)
    print(json.dumps({'prepared': str(OUT), 'search_gate_rmse': protocol['search_gate_rmse'], 'frozen_files': len(frozen), 'training_started': False}))

def rnn_prefix(nrows):
    """Original forward implementation; no rental labels or later rows are read."""
    import numpy as np
    import pandas as pd
    from experiments.local._support.mean_residual_model.regional import with_month
    from experiments.local._support.mean_residual_model.reginal_mean_model.lstm.model import _predict_timeline
    from experiments.local._support.mean_residual_model.reginal_mean_model.rnn.model import TownRNN
    config = read(RNN / 'config.json')
    extra = [column for column in config['columns'] if column != 'month_index']
    dates = pd.read_csv(support_path(Path('Data/train.csv')), nrows=nrows, usecols=['RENT_APPROVAL_DATE', 'TOWN'], dtype={'TOWN': str})
    economy = pd.read_csv(support_path(Path('Data/auxiliary_features/train_features.csv')), nrows=nrows, usecols=extra)
    work = with_month(pd.concat([dates, economy], axis=1))
    cells = work[['TOWN', 'month', 'month_index']].drop_duplicates().copy()
    assert not cells.duplicated(['TOWN', 'month']).any()
    for column in extra:
        cells[column] = cells['month'].map(work.groupby('month')[column].first())
    prediction = _predict_timeline(RNN, cells[['TOWN', 'month', *config['columns']]], TownRNN)
    assert all((np.isfinite(value) for value in prediction.values()))
    return prediction

def map_trend(frame, prediction):
    import numpy as np
    return np.array([prediction[str(town), str(month)[:7]] for town, month in zip(frame.TOWN, frame.RENT_APPROVAL_DATE)], dtype=float)

def run():
    with (OUT / 'worker_identity.json').open('x', encoding='utf-8') as handle:
        json.dump({'pid': os.getpid(), 'epoch': time.time()}, handle)
    budget, protocol = (read(OUT / 'budget.json'), read(OUT / 'protocol.json'))
    protocol_hash = digest(OUT / 'protocol.json')
    assert budget['limit_seconds'] == 900
    assert protocol['features'] == FEATURES and protocol['params'] == PARAMS

    def deadline():
        if time.time() >= budget['deadline_epoch']:
            raise TimeoutError('Authorized 900-second wall-time expired')
    record('preflight_start', limit_seconds=900)
    check_frozen(protocol)
    deadline()
    import joblib
    import numpy as np
    import pandas as pd
    import torch
    from catboost import CatBoostRegressor
    from experiments.local._support.models.catboost.preprocess import transform_frame
    from experiments.local._support.models.catboost.model import _fill_missing_categories, _category_columns
    from experiments.local._support.src.pipeline.data import load_joined, time_split
    torch.set_num_threads(8)
    torch.set_num_interop_threads(2)

    def finite(values, name):
        values = np.asarray(values, dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f'Nonfinite {name}')
        return values

    def metrics(truth, prediction, dates):
        error = finite(prediction, 'prediction') - finite(truth, 'truth')
        months = np.array([str(value)[:7] for value in dates])
        return {'rows': len(error), 'rmse': float(np.sqrt(np.mean(error ** 2))), 'mae': float(np.mean(np.abs(error))), 'bias_pred_minus_truth': float(error.mean()), 'by_month': {month: {'rows': int((months == month).sum()), 'rmse': float(np.sqrt(np.mean(error[months == month] ** 2)))} for month in sorted(set(months))}}
    cache = joblib.load(asset_path(Path('feature_ablation_01/training_cache.joblib')))
    fit, search = (cache['parts']['train'].copy(), cache['parts']['search'].copy())
    assert [len(fit), len(search)] == [132802, 8274]
    assert fit.RENT_APPROVAL_DATE.astype(str).str[:7].max() == '2024-09'
    assert sorted(search.RENT_APPROVAL_DATE.astype(str).str[:7].unique()) == protocol['search_months']
    assert np.array_equal(cache['search_row_ids'], np.arange(132802, 141076))
    needed = ['RENT_APPROVAL_DATE', *BASE_FEATURES, 'MONTHLY_RENT']
    names = set(pd.read_csv(support_path(Path('Data/train.csv')), nrows=0).columns)
    raw = pd.read_csv(support_path(Path('Data/train.csv')), nrows=141076, usecols=[name for name in needed if name in names], dtype={'TOWN': str})
    auxiliary = pd.read_csv(support_path(Path('Data/auxiliary_features/train_features.csv')), nrows=141076, usecols=[name for name in needed if name not in names])
    joined = pd.concat([raw, auxiliary], axis=1)
    for frame, start, end in [(fit, 0, 132802), (search, 132802, 141076)]:
        pd.testing.assert_frame_equal(frame[needed].reset_index(drop=True), joined.iloc[start:end][needed].reset_index(drop=True), check_dtype=False, check_exact=True)
    y_fit, y_search = (finite(fit.MONTHLY_RENT, 'fit truth'), finite(search.MONTHLY_RENT, 'Search truth'))
    prefix = rnn_prefix(141076)
    trend_audit = {'rnn_fits': 0, 'prefix_uses_rental_labels': False, 'prefix_last_month': '2024-12', 'confirm_loaded_before_gate': False, 'max_abs_cached_mu_difference': {}}
    offsets = {}
    for name, frame, truth in [('train', fit, y_fit), ('search', search, y_search)]:
        mu = finite(cache['store'][name]['mu'], f'cached RNN {name}')
        reconstructed = finite(map_trend(frame, prefix), f'prefix RNN {name}')
        difference = float(np.max(np.abs(mu - reconstructed)))
        assert difference <= 2e-05, (name, difference)
        trend_audit['max_abs_cached_mu_difference'][name] = difference
        area = finite(frame.floor_area_sqm, 'floor area')
        assert np.all(area > 0)
        offsets[name] = finite(mu * area, 'monthly trend')
        rows = np.arange(0, 132802) if name == 'train' else cache['search_row_ids']
        pd.DataFrame({'source_row_id': rows, 'rnn_unit_level': mu, 'floor_area_sqm': area, 'trend_component': offsets[name], 'truth': truth, 'residual_target': truth - offsets[name]}).to_csv(OUT / f'{name}_target_audit.csv', index=False)
    record('frozen_rnn_verified', **trend_audit)
    fit, search = (add_address(fit), add_address(search))
    shutil.copyfile(PRIOR / 'preprocessor.joblib', OUT / 'preprocessor.joblib')
    preparer = joblib.load(OUT / 'preprocessor.joblib')
    assert preparer.selected_columns_ == FEATURES and preparer.spec == PREPROCESS
    assert not preparer.target_values_ and (not preparer.include_building_)
    fit_x = transform_frame(preparer, fit.drop(columns='MONTHLY_RENT'))
    search_x = transform_frame(preparer, search.drop(columns='MONTHLY_RENT'))
    categories = _category_columns(fit_x)
    assert len(fit_x.columns) == 26 and set(fit_x.columns) == set(FEATURES)
    write_json(OUT / 'data_audit.json', {'train_rows': len(fit), 'search_rows': len(search), 'original_inputs_and_labels_exactly_equal': True, 'train_source_row_id_range': [0, 132801], 'search_source_row_id_range': [132802, 141075], 'features': FEATURES, 'model_columns': list(fit_x.columns), 'categories': categories, 'trend_audit': trend_audit, 'preprocessor_sha256': digest(OUT / 'preprocessor.joblib'), 'confirm_read_for_fitting': False})
    residual_fit, residual_search = (y_fit - offsets['train'], y_search - offsets['search'])
    deadline()
    record('catboost_fit_enter', train_rows=len(fit), search_rows=len(search), features=26, params=PARAMS, target=TARGET, weights='none', fit_call_number=1)
    model = CatBoostRegressor(**PARAMS)
    model.fit(_fill_missing_categories(fit_x), residual_fit, eval_set=(_fill_missing_categories(search_x), residual_search), cat_features=categories, use_best_model=True, verbose=50)
    record('catboost_fit_complete', trees=model.tree_count_, best_iteration=model.get_best_iteration())
    deadline()
    joblib.dump(model, OUT / 'model.joblib')
    prior_audit = read(PRIOR / 'fit_audit.json')
    assert model.get_params() == prior_audit['params'] == PARAMS
    assert model.get_all_params() == prior_audit['resolved_params']
    assert model.feature_names_ == prior_audit['features']
    assert model.get_cat_feature_indices() == prior_audit['categorical_indices']
    fit_audit = {'params': model.get_params(), 'resolved_params': model.get_all_params(), 'features': model.feature_names_, 'categorical_indices': model.get_cat_feature_indices(), 'trees': model.tree_count_, 'best_iteration': model.get_best_iteration(), 'fit_calls': 1, 'target': TARGET, 'training_weights': 'none', 'early_stopping_weights': 'none', 'eval_history': model.get_evals_result(), 'constructor_and_resolved_parameters_unchanged': True}
    write_json(OUT / 'fit_audit.json', fit_audit)
    residual_prediction = finite(model.predict(_fill_missing_categories(search_x)), 'Search residual prediction')
    prediction = finite(offsets['search'] + residual_prediction, 'Search monthly prediction')
    search_path = OUT / 'search_predictions.csv'
    pd.DataFrame({'source_row_id': cache['search_row_ids'], 'date': search.RENT_APPROVAL_DATE.to_numpy(), 'truth': y_search, 'prediction': prediction, 'trend_component': offsets['search'], 'residual_prediction': residual_prediction}).to_csv(search_path, index=False)
    search_metrics = metrics(y_search, prediction, search.RENT_APPROVAL_DATE)
    score = search_metrics['rmse']
    assert abs(score - min(fit_audit['eval_history']['validation']['RMSE'])) < 0.0001
    search_metrics.update(delta_from_baseline=score - protocol['baseline']['search_rmse'], delta_from_previous_best=score - protocol['previous_best']['search_rmse'])
    write_json(OUT / 'search_metrics.json', search_metrics)
    passed = score < protocol['search_gate_rmse']
    selection = {'candidate': protocol['candidate'], 'passed_search_gate': passed, 'search_rmse': score, 'threshold': protocol['search_gate_rmse'], 'epoch': time.time(), 'protocol_sha256': protocol_hash, 'confirm_evaluations_before_freeze': 0, 'candidate_sha256': {name: digest(OUT / name) for name in ['model.joblib', 'preprocessor.joblib', 'search_predictions.csv', 'fit_audit.json', 'data_audit.json', 'train_target_audit.csv', 'search_target_audit.csv']}}
    write_json(OUT / 'selection_frozen.json', selection)
    selection_hash = digest(OUT / 'selection_frozen.json')
    record('selection_frozen', passed_search_gate=passed, search_rmse=score)
    result = {'status': 'completed', 'candidate': protocol['candidate'], 'search': search_metrics, 'confirm': None, 'candidate_confirm_evaluations': 0, 'passed_search_gate': passed, 'baseline': protocol['baseline'], 'previous_best': protocol['previous_best'], 'fit_calls': 1, 'rnn_fit_calls': 0, 'protocol_sha256': protocol_hash, 'selection_sha256': selection_hash, 'search_predictions_csv': str(search_path), 'confirm_predictions_csv': None, 'long_horizon_accuracy_validated': False, 'original_labels_unchanged': True, 'original_split_unchanged': True}
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
        rows = np.flatnonzero(frame.RENT_APPROVAL_DATE.astype(str).str[:7].isin(protocol['confirm_months']).to_numpy())
        assert np.array_equal(rows, np.arange(141076, 150000))
        confirm_input = add_address(split.confirm.drop(columns='MONTHLY_RENT'))
        confirm_mu = finite(map_trend(confirm_input, rnn_prefix(150000)), 'Confirm RNN')
        confirm_offset = finite(confirm_mu * confirm_input.floor_area_sqm.to_numpy(float), 'Confirm trend')
        confirm_x = transform_frame(preparer, confirm_input)
        confirm_residual = finite(model.predict(_fill_missing_categories(confirm_x)), 'Confirm residual')
        confirm_prediction = finite(confirm_offset + confirm_residual, 'Confirm monthly prediction')
        truth = finite(split.confirm.MONTHLY_RENT, 'Confirm truth')
        confirm_path = OUT / 'confirm_predictions.csv'
        pd.DataFrame({'source_row_id': rows, 'date': split.confirm.RENT_APPROVAL_DATE.to_numpy(), 'truth': truth, 'prediction': confirm_prediction, 'rnn_unit_level': confirm_mu, 'floor_area_sqm': confirm_input.floor_area_sqm.to_numpy(float), 'trend_component': confirm_offset, 'residual_prediction': confirm_residual}).to_csv(confirm_path, index=False)
        confirmation = metrics(truth, confirm_prediction, split.confirm.RENT_APPROVAL_DATE)
        confirmation.update(delta_from_baseline=confirmation['rmse'] - protocol['baseline']['confirm_rmse'], delta_from_previous_best=confirmation['rmse'] - protocol['previous_best']['confirm_rmse'])
        write_json(OUT / 'confirmation_metrics.json', confirmation)
        improved = confirmation['rmse'] < protocol['previous_best']['confirm_rmse']
        result.update(confirm=confirmation, candidate_confirm_evaluations=1, confirm_predictions_csv=str(confirm_path), confirm_predictions_sha256=digest(confirm_path), improved_over_previous_best_on_both_periods=improved, conclusion='Improved both original periods; long-horizon accuracy remains unverified.' if improved else 'Search improved but Confirm did not beat previous best; optimization not established. No reselection.')
        record('confirm_complete', rmse=confirmation['rmse'])
    else:
        result.update(improved_over_previous_best_on_both_periods=False, conclusion='Search did not beat previous best; Confirm not evaluated. No retry.')
    deadline()
    check_frozen(protocol)
    assert digest(OUT / 'protocol.json') == protocol_hash
    assert digest(OUT / 'selection_frozen.json') == selection_hash
    for name, expected in selection['candidate_sha256'].items():
        assert digest(OUT / name) == expected
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
            last = LAST_STAGE
            write_json(OUT / 'failure_report.json', {'status': 'failed', 'error': repr(exc), 'last_successful_stage': last, 'epoch': time.time(), 'exit_code': 1}, required=False)
            record('failed', error=repr(exc), last_successful_stage=last)
        raise
