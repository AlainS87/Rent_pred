"""One time-index diagnostic and at most one gated global CatBoost fit."""
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
OUT = asset_path(Path('temporal_index_trial_01'))
LAST_STAGE = 'not_started'
ORIGINS = [('2022-09', '2024-07', 22), ('2023-05', '2024-09', 16)]

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
        raise RuntimeError('Refusing overwrite/retry')
    prior = read(PRIOR / 'results.json')
    prior_protocol = read(PRIOR / 'protocol.json')
    check_frozen(prior_protocol)
    assert digest(PRIOR / 'protocol.json') == prior['protocol_sha256']
    frozen = dict(prior_protocol['frozen_sha256'])
    for name in ['protocol.json', 'results.json', 'selection_frozen.json', 'model.joblib', 'preprocessor.joblib', 'fit_audit.json', 'data_audit.json', 'search_predictions.csv', 'confirm_predictions.csv', 'search_metrics.json', 'confirmation_metrics.json', 'completion_verification.json']:
        relative = 'global_full_address_01/' + name
        frozen[relative] = digest(asset_path(Path(relative)))
    for name in ['temporal_index.py', 'temporal_index_trial.py', 'verify_temporal_index_trial.py', 'temporal_structure_audit_01/audit.json']:
        frozen[name] = digest(asset_path(Path(name)))
    protocol = {'status': 'frozen_before_testing', 'created_epoch': time.time(), 'candidate': 'property_controlled_time_index_damped_forecast_then_conditional_global_catboost', 'candidate_count': 1, 'max_wall_seconds': 900, 'automatic_retry': False, 'budget_includes': 'imports, checks, historical index fits/backtests, conditional single CatBoost, Search, gated Confirm, saving', 'original_fit_end': '2024-09', 'rows': prior_protocol['rows'], 'search_months': prior_protocol['search_months'], 'confirm_months': prior_protocol['confirm_months'], 'baseline': prior_protocol['baseline'], 'previous_best': {'search_rmse': prior['search']['rmse'], 'confirm_rmse': prior['confirm']['rmse']}, 'search_gate_rmse': prior['search']['rmse'], 'confirm_max_evaluations': 1, 'confirm_reselection': False, 'confirm_interpretation': 'Original iterative validation, not untouched test data', 'index': {'model': 'log MONTHLY_RENT = property fixed effect + month fixed effect', 'property_key': ['TOWN', 'STREET', 'BLOCK', 'FLAT_TYPE', 'FLAT_MODEL', 'FLOOR_AREA_SQM'], 'normalization': 'Existing internal support text/type normalization', 'eligible_properties': 'At least two distinct observation months before each origin', 'fitting': 'Unregularized within-property OLS, first month effect fixed at zero', 'current_month_forecast_level': 'Last estimated month effect', 'slope': 'OLS slope of the last six estimated log-index months', 'damping_phi': 0.9, 'formula': 'b(T+h)=b(T)+slope*0.9*(1-0.9**h)/(1-0.9)', 'rationale': 'Recent six months responds to observed slowdown; damping prevents extrapolating the same growth indefinitely. Fixed hypothesis, no parameter grid.', 'macroeconomic_inputs': 'None; future external indicator values are not needed', 'other_hypothesis': 'Flat benchmark b(T+h)=b(T)', 'candidate_forecast_count': 1, 'benchmark_count': 1}, 'historical_origins': [{'origin': o, 'evaluation_end': e, 'max_horizon': h} for o, e, h in ORIGINS], 'historical_validation': {'labels': 'Only within original fit ending 2024-09; no Search/Confirm labels', 'property_effects': 'Fit before each origin and frozen; future unseen properties excluded from this index diagnostic only', 'observed_index': '100*exp(mean(log rent - frozen property effect) - origin month effect)', 'forecast_metric': 'Equal-month RMSE in index points, not individual-rent RMSE', 'gate': {'minimum_monthly_eligible_coverage': 0.7, 'each_origin_h7_plus_not_worse': True, 'pooled_h7_plus_minimum_rmse_reduction': 0.05, 'maximum_each_origin_h1_6_rmse_increase': 0.05}, 'cautions': ['Observable property groups are not actual unit IDs', 'Overlapping horizons are not independent', 'Historical patterns already inspected; this is a retrospective diagnostic', 'Does not replace original baseline split']}, 'conditional_catboost': {'execute_only_if_index_gate_passes': True, 'features': FEATURES, 'params': PARAMS, 'preprocess': PREPROCESS, 'preprocessor': 'Reuse current best byte-for-byte', 'max_fits': 1, 'target': 'Original MONTHLY_RENT / I(month), I normalized to 1 at 2024-09', 'weights': 'I(month)^2 normalized to mean 1 separately in fit and Search', 'prediction': 'I(month) * CatBoost normalized-monthly-rent prediction', 'evaluation': 'Original unweighted monthly-rent RMSE; early-stop weighted normalized RMSE is proportional by a fixed Search factor', 'original_labels': 'Unchanged source truth; derived training target only', 'confirm': 'At most once, only if Search strictly improves 444.9106685519473'}, 'frozen_sha256': frozen}
    OUT.mkdir()
    write_json(OUT / 'protocol.json', protocol)
    print(json.dumps({'prepared': str(OUT), 'frozen_files': len(frozen), 'training_started': False}))

def run():
    with (OUT / 'worker_identity.json').open('x', encoding='utf-8') as handle:
        json.dump({'pid': os.getpid(), 'epoch': time.time()}, handle)
    protocol, budget = (read(OUT / 'protocol.json'), read(OUT / 'budget.json'))
    protocol_hash = digest(OUT / 'protocol.json')

    def deadline():
        if time.time() >= budget['deadline_epoch']:
            raise TimeoutError('900-second cumulative test budget expired')
    record('preflight_start', limit_seconds=900)
    check_frozen(protocol)
    import joblib
    import numpy as np
    import pandas as pd
    from experiments.local.temporal_index import prepare_rows, backtest, gate, fit_index, scales
    raw = pd.read_csv(support_path(Path('Data/train.csv')), nrows=132802, usecols=['RENT_APPROVAL_DATE', 'MONTHLY_RENT', 'TOWN', 'STREET', 'BLOCK', 'FLAT_TYPE', 'FLAT_MODEL', 'FLOOR_AREA_SQM'], dtype={'TOWN': str, 'STREET': str, 'BLOCK': str, 'FLAT_TYPE': str, 'FLAT_MODEL': str})
    assert len(raw) == 132802 and raw.RENT_APPROVAL_DATE.str[:7].max() == '2024-09'
    raw['source_row_id'] = np.arange(len(raw))
    rows = prepare_rows(raw)
    tables = []
    for origin, end, h in ORIGINS:
        deadline()
        record('historical_index_fit_started', origin=origin, max_horizon=h)
        history = rows.loc[rows.month <= origin].copy()
        future = rows.loc[(rows.month > origin) & (rows.month <= end)].copy()
        index, monthly, evaluation_rows = backtest(history, future, origin, h)
        slug = origin.replace('-', '_')
        joblib.dump(index, OUT / f'index_{slug}.joblib')
        monthly.to_csv(OUT / f'backtest_{slug}.csv', index=False)
        evaluation_rows.to_csv(OUT / f'evaluation_rows_{slug}.csv', index=False)
        tables.append(monthly)
        record('historical_index_fit_complete', origin=origin, fit_rows=index['fit_rows'], eligible_fit_rows=index['eligible_fit_rows'], property_groups=index['property_groups'], slope_per_month=index['slope'], minimum_future_coverage=float(monthly.coverage.min()))
    combined = pd.concat(tables, ignore_index=True)
    combined.to_csv(OUT / 'historical_backtest_predictions.csv', index=False)
    diagnosis = gate(combined)
    write_json(OUT / 'historical_metrics.json', diagnosis)
    history_selection = {'epoch': time.time(), 'passed': diagnosis['passed'], 'protocol_sha256': protocol_hash, 'conditions': diagnosis['conditions'], 'historical_metrics_sha256': digest(OUT / 'historical_metrics.json'), 'catboost_fits_before_freeze': 0, 'confirm_evaluations_before_freeze': 0}
    write_json(OUT / 'time_selection_frozen.json', history_selection)
    history_hash = digest(OUT / 'time_selection_frozen.json')
    record('time_gate_frozen', passed=diagnosis['passed'], conditions=diagnosis['conditions'])
    result = {'status': 'completed', 'candidate': protocol['candidate'], 'time_gate_passed': diagnosis['passed'], 'historical_metrics': diagnosis, 'historical_index_fits': 2, 'catboost_fits': 0, 'search': None, 'confirm': None, 'candidate_confirm_evaluations': 0, 'baseline': protocol['baseline'], 'previous_best': protocol['previous_best'], 'protocol_sha256': protocol_hash, 'time_selection_sha256': history_hash, 'source_labels_unchanged': True, 'original_split_unchanged': True, 'historical_backtest_csv': str(OUT / 'historical_backtest_predictions.csv')}
    if not diagnosis['passed']:
        result['conclusion'] = 'Damped time-index hypothesis failed the locked historical gate. No CatBoost fitting, Search or Confirm evaluation. No retry.'
    else:
        deadline()
        record('final_time_index_fit_started')
        index = fit_index(rows)
        joblib.dump(index, OUT / 'final_time_index.joblib')
        result['historical_index_fits'] = 3
        record('final_time_index_fit_complete', fit_rows=index['fit_rows'], fit_end=index['fit_end'])
        run_catboost(index, protocol, result, deadline)
    deadline()
    check_frozen(protocol)
    assert digest(OUT / 'protocol.json') == protocol_hash
    assert digest(OUT / 'time_selection_frozen.json') == history_hash
    result.update(exit_code=0, elapsed_seconds=time.time() - budget['started_epoch'], frozen_files_unchanged=True)
    write_json(OUT / 'results.json', result)
    record('completed', time_gate_passed=result['time_gate_passed'], catboost_fits=result['catboost_fits'], conclusion=result['conclusion'])

def run_catboost(index, protocol, result, deadline):
    import joblib
    import numpy as np
    import pandas as pd
    from catboost import CatBoostRegressor, Pool
    from experiments.local.temporal_index import scales
    from experiments.local._support.models.catboost.preprocess import transform_frame
    from experiments.local._support.models.catboost.model import _fill_missing_categories, _category_columns
    from experiments.local._support.src.pipeline.data import load_joined, time_split

    def finite(values):
        values = np.asarray(values, dtype=float)
        assert np.isfinite(values).all()
        return values

    def metrics(y, prediction, dates):
        error = finite(prediction) - finite(y)
        months = np.array([str(x)[:7] for x in dates])
        return {'rows': len(y), 'rmse': float(np.sqrt(np.mean(error ** 2))), 'mae': float(np.mean(np.abs(error))), 'bias_pred_minus_truth': float(error.mean()), 'by_month': {m: {'rows': int((months == m).sum()), 'rmse': float(np.sqrt(np.mean(error[months == m] ** 2)))} for m in sorted(set(months))}}
    cache = joblib.load(asset_path(Path('feature_ablation_01/training_cache.joblib')))
    fit, search = (cache['parts']['train'].copy(), cache['parts']['search'].copy())
    assert [len(fit), len(search)] == [132802, 8274]
    assert np.array_equal(cache['search_row_ids'], np.arange(132802, 141076))
    needed = ['RENT_APPROVAL_DATE', *BASE_FEATURES, 'MONTHLY_RENT']
    source_names = set(pd.read_csv(support_path(Path('Data/train.csv')), nrows=0).columns)
    src = pd.read_csv(support_path(Path('Data/train.csv')), nrows=141076, usecols=[x for x in needed if x in source_names], dtype={'TOWN': str})
    aux = pd.read_csv(support_path(Path('Data/auxiliary_features/train_features.csv')), nrows=141076, usecols=[x for x in needed if x not in source_names])
    joined = pd.concat([src, aux], axis=1)
    for frame, start, end in [(fit, 0, 132802), (search, 132802, 141076)]:
        pd.testing.assert_frame_equal(frame[needed].reset_index(drop=True), joined.iloc[start:end][needed].reset_index(drop=True), check_dtype=False, check_exact=True)
    y_fit, y_search = (finite(fit.MONTHLY_RENT), finite(search.MONTHLY_RENT))
    fit_scale, search_scale = (scales(index, fit.RENT_APPROVAL_DATE), scales(index, search.RENT_APPROVAL_DATE))
    fit_weight, search_weight = (fit_scale ** 2 / np.mean(fit_scale ** 2), search_scale ** 2 / np.mean(search_scale ** 2))
    for label, ids, y, scale, weight in [('train', np.arange(132802), y_fit, fit_scale, fit_weight), ('search', cache['search_row_ids'], y_search, search_scale, search_weight)]:
        pd.DataFrame({'source_row_id': ids, 'truth': y, 'time_scale': scale, 'normalized_target': y / scale, 'weight': weight}).to_csv(OUT / f'{label}_target_audit.csv', index=False)
    fit, search = (add_address(fit), add_address(search))
    shutil.copyfile(PRIOR / 'preprocessor.joblib', OUT / 'preprocessor.joblib')
    preparer = joblib.load(OUT / 'preprocessor.joblib')
    assert preparer.selected_columns_ == FEATURES and preparer.spec == PREPROCESS
    fit_x = transform_frame(preparer, fit.drop(columns='MONTHLY_RENT'))
    search_x = transform_frame(preparer, search.drop(columns='MONTHLY_RENT'))
    categories = _category_columns(fit_x)
    assert len(fit_x.columns) == 26 and set(fit_x.columns) == set(FEATURES)
    write_json(OUT / 'data_audit.json', {'train_rows': len(fit), 'search_rows': len(search), 'original_inputs_and_labels_exactly_equal': True, 'confirm_read_for_fitting': False, 'features': FEATURES, 'model_columns': list(fit_x.columns), 'categories': categories, 'index_fit_end': index['fit_end'], 'preprocessor_sha256': digest(OUT / 'preprocessor.joblib')})
    deadline()
    record('catboost_fit_enter', fit_call_number=1, params=PARAMS, features=26)
    model = CatBoostRegressor(**PARAMS)
    model.fit(Pool(_fill_missing_categories(fit_x), y_fit / fit_scale, weight=fit_weight, cat_features=categories), eval_set=Pool(_fill_missing_categories(search_x), y_search / search_scale, weight=search_weight, cat_features=categories), use_best_model=True, verbose=50)
    record('catboost_fit_complete', trees=model.tree_count_, best_iteration=model.get_best_iteration())
    deadline()
    joblib.dump(model, OUT / 'model.joblib')
    prior_audit = read(PRIOR / 'fit_audit.json')
    assert model.get_params() == prior_audit['params'] == PARAMS
    assert model.feature_names_ == prior_audit['features']
    resolved_diff = {key: [prior_audit['resolved_params'].get(key), value] for key, value in model.get_all_params().items() if prior_audit['resolved_params'].get(key) != value}
    assert not resolved_diff, resolved_diff
    audit = {'params': model.get_params(), 'resolved_params': model.get_all_params(), 'features': model.feature_names_, 'categorical_indices': model.get_cat_feature_indices(), 'fit_calls': 1, 'trees': model.tree_count_, 'best_iteration': model.get_best_iteration(), 'eval_history': model.get_evals_result(), 'target': 'MONTHLY_RENT / time_scale', 'weight': 'time_scale^2 / mean(time_scale^2)', 'search_metric_conversion_factor': float(np.sqrt(np.mean(search_scale ** 2)))}
    write_json(OUT / 'fit_audit.json', audit)
    normalized = finite(model.predict(_fill_missing_categories(search_x)))
    prediction = normalized * search_scale
    pd.DataFrame({'source_row_id': cache['search_row_ids'], 'date': search.RENT_APPROVAL_DATE.to_numpy(), 'truth': y_search, 'prediction': prediction, 'time_scale': search_scale, 'normalized_prediction': normalized}).to_csv(OUT / 'search_predictions.csv', index=False)
    search_metrics = metrics(y_search, prediction, search.RENT_APPROVAL_DATE)
    score = search_metrics['rmse']
    expected = min(audit['eval_history']['validation']['RMSE']) * audit['search_metric_conversion_factor']
    assert abs(score - expected) < 0.001, (score, expected)
    search_metrics.update(delta_from_baseline=score - protocol['baseline']['search_rmse'], delta_from_previous_best=score - protocol['previous_best']['search_rmse'])
    write_json(OUT / 'search_metrics.json', search_metrics)
    passed = score < protocol['search_gate_rmse']
    selection = {'epoch': time.time(), 'passed_search_gate': passed, 'search_rmse': score, 'threshold': protocol['search_gate_rmse'], 'confirm_evaluations_before_freeze': 0, 'protocol_sha256': result['protocol_sha256'], 'candidate_sha256': {name: digest(OUT / name) for name in ['model.joblib', 'preprocessor.joblib', 'final_time_index.joblib', 'fit_audit.json', 'data_audit.json', 'train_target_audit.csv', 'search_target_audit.csv', 'search_predictions.csv']}}
    write_json(OUT / 'selection_frozen.json', selection)
    selection_hash = digest(OUT / 'selection_frozen.json')
    result.update(catboost_fits=1, search=search_metrics, passed_search_gate=passed, selection_sha256=selection_hash, search_predictions_csv=str(OUT / 'search_predictions.csv'), confirm_predictions_csv=None)
    record('selection_frozen', passed_search_gate=passed, search_rmse=score)
    if passed:
        deadline()
        with (OUT / 'confirm_started.json').open('x', encoding='utf-8') as handle:
            json.dump({'epoch': time.time(), 'pid': os.getpid(), 'selection_sha256': selection_hash}, handle)
        record('confirm_prediction_started', evaluation_number=1)
        frame = load_joined('train', ['RENT_APPROVAL_DATE', *BASE_FEATURES])
        split = time_split(frame, 3, None)
        assert [len(split.fit), len(split.search), len(split.confirm)] == [132802, 8274, 8924]
        assert split.fit_months[-1] == '2024-09' and split.search_months == protocol['search_months'] and (split.confirm_months == protocol['confirm_months'])
        ids = np.flatnonzero(frame.RENT_APPROVAL_DATE.astype(str).str[:7].isin(protocol['confirm_months']))
        assert np.array_equal(ids, np.arange(141076, 150000))
        confirm = add_address(split.confirm.drop(columns='MONTHLY_RENT'))
        factor = scales(index, confirm.RENT_APPROVAL_DATE)
        norm = finite(model.predict(_fill_missing_categories(transform_frame(preparer, confirm))))
        pred = norm * factor
        y = finite(split.confirm.MONTHLY_RENT)
        path = OUT / 'confirm_predictions.csv'
        pd.DataFrame({'source_row_id': ids, 'date': confirm.RENT_APPROVAL_DATE.to_numpy(), 'truth': y, 'prediction': pred, 'time_scale': factor, 'normalized_prediction': norm}).to_csv(path, index=False)
        cm = metrics(y, pred, confirm.RENT_APPROVAL_DATE)
        cm.update(delta_from_baseline=cm['rmse'] - protocol['baseline']['confirm_rmse'], delta_from_previous_best=cm['rmse'] - protocol['previous_best']['confirm_rmse'])
        write_json(OUT / 'confirmation_metrics.json', cm)
        improved = cm['rmse'] < protocol['previous_best']['confirm_rmse']
        result.update(confirm=cm, candidate_confirm_evaluations=1, confirm_predictions_csv=str(path), improved_both_original_periods=improved, conclusion='Improved both original periods; true distant-test accuracy remains unknown.' if improved else 'Search improved but Confirm regressed; optimization not established, no reselection.')
        record('confirm_complete', rmse=cm['rmse'])
    else:
        result.update(improved_both_original_periods=False, conclusion='Historical index gate passed but Search did not beat current best; Confirm not evaluated. No retry.')
    assert digest(OUT / 'selection_frozen.json') == selection_hash
    for name, expected in selection['candidate_sha256'].items():
        assert digest(OUT / name) == expected
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['prepare', 'run'])
    args = parser.parse_args()
    try:
        prepare() if args.phase == 'prepare' else run()
    except BaseException as exc:
        traceback.print_exc()
        if args.phase == 'run' and OUT.exists():
            previous = LAST_STAGE
            write_json(OUT / 'failure_report.json', {'status': 'failed', 'error': repr(exc), 'last_successful_stage': previous, 'epoch': time.time(), 'exit_code': 1}, required=False)
            record('failed', error=repr(exc), last_successful_stage=previous)
        raise
