"""Authorized single candidate: three historical features, frozen RNN/GMM/MLP."""
from experiments.local.bootstrap import Path, CODE_DIR, DATA_ROOT, SUPPORT_ROOT, ARTIFACT_ROOT, asset_path, asset_key, support_path
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback
sys.dont_write_bytecode = True
from experiments.local import feature_ablation as shared
ROOT = Path(__file__).resolve().parent
OUT = asset_path(Path('spatial_history_01'))
BASE = asset_path(Path('baseline_alignment'))
PREVIOUS = asset_path(Path('feature_ablation_01'))
EXTRA = ['past_monthly_rent_mean', 'past_record_count', 'past_monthly_rent_std']

def record(stage, **values):
    event = {'stage': stage, 'pid': os.getpid(), 'epoch': time.time(), **values}
    with (OUT / 'events.jsonl').open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(event, allow_nan=False) + '\n')
    shared.save(OUT / 'status.json', event)
    print(json.dumps(event), flush=True)

def deadline():
    if time.time() >= shared.read(OUT / 'supervisor.json')['deadline_epoch']:
        raise TimeoutError('Authorized 900-second total wall-clock budget exhausted')

def check_frozen():
    protocol = shared.read(OUT / 'protocol.json')
    for relative, expected in protocol['frozen_sha256'].items():
        if shared.digest(asset_path(Path(relative))) != expected:
            raise RuntimeError(f'Frozen file changed: {relative}')
    return protocol

def prepare():
    if OUT.exists():
        raise RuntimeError('Refusing to overwrite or repeat an existing candidate')
    prior = shared.verify_frozen()
    best = shared.read(asset_path(Path('frozen_mlp_area_confirm_01/results.json')))
    assert best['search_rmse'] == 451.042648255612
    extras = [Path(__file__).resolve(), asset_path(Path('spatial_history_features.py')), PREVIOUS / 'protocol.json', PREVIOUS / 'selection_frozen.json', asset_path(Path('frozen_mlp_area_confirm_01/results.json')), asset_path(Path('frozen_mlp_area_confirm_01/protocol.json'))]
    frozen = dict(prior['frozen_sha256'])
    frozen.update({asset_key(path).as_posix(): shared.digest(path) for path in extras})
    protocol = {'status': 'frozen_before_feature_preparation_and_training', 'candidate': 'original_catboost_plus_three_spatial_history_features_frozen_mlp', 'fit_end': '2024-09', 'search_months': ['2024-10', '2024-11', '2024-12'], 'confirm_months': ['2025-01', '2025-02', '2025-03'], 'rows': {'train': 132802, 'search': 8274, 'confirm': 8924}, 'original_features': prior['original_catboost_features'], 'added_features': EXTRA, 'history': {'same_flat_type': True, 'radius_meters': 500, 'earth_radius_km': 6371.0088, 'training': 'Each month T uses only original fit rows from T-6 through T-1, never same-month labels', 'search_and_confirm': 'Reference rows permanently fixed to 2024-04 through 2024-09', 'statistics': 'Unweighted mean, original record count, sample standard deviation ddof=1', 'missing': 'count=0; mean NaN for no history; std NaN for fewer than two records; original fit-only median imputation', 'external_hdb_records_used': False, 'deduplicated_or_relabelled': False}, 'preprocess': prior['preprocess'], 'seed': 5228, 'threads': 8, 'device': 'CPU', 'catboost': 'Exact original per-town get_params; unweighted original unit-residual RMSE; 1000 trees maximum and early stopping 50', 'other_models': 'Original baseline_alignment RNN/GMM/MLP weights and 19-input MLP normalizers frozen', 'baseline': prior['baseline'], 'previous_best': {'search_rmse': best['search_rmse'], 'confirm_rmse': best['confirm']['rmse']}, 'search_gate_rmse': best['search_rmse'], 'selection': 'Strictly lower final Search RMSE than existing best permits exactly one Confirm evaluation; freeze all candidate hashes first; never reselect using Confirm', 'confirm_caveat': 'Previously inspected original Confirm, an iterative validation period, not an untouched test set', 'metric': 'Unweighted monthly-rent RMSE on unchanged true rents', 'hard_limit_seconds': 900, 'budget_includes_feature_preparation': True, 'candidate_count': 1, 'automatic_retries': 0, 'frozen_sha256': frozen}
    OUT.mkdir()
    shared.save(OUT / 'protocol.json', protocol)
    record('protocol_prepared', training=False, production_features_prepared=False)

def runtime():
    global np, pd, torch, joblib, CatBoostRegressor, Pool, cb_module, features
    stack = shared.runtime()
    import numpy as np
    import pandas as pd
    import torch
    import joblib
    from catboost import CatBoostRegressor, Pool
    import experiments.local._support.models.catboost.model as cb_module
    from experiments.local import spatial_history_features as features
    from sklearn.mixture import GaussianMixture

    def forbidden(*args, **kwargs):
        raise RuntimeError("Only this candidate's CatBoost may be trained")
    stack._fit_mlp = forbidden
    stack._fit_mixture = forbidden
    GaussianMixture.fit = forbidden
    torch.optim.Adam.step = forbidden
    return stack

def run():
    with (OUT / 'run_started.json').open('x') as handle:
        json.dump({'pid': os.getpid(), 'epoch': time.time()}, handle)
    record('runtime_start', training=False)
    stack = runtime()
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=8):
        execute(stack)

def execute(stack):
    protocol = check_frozen()
    protocol_hash = shared.digest(OUT / 'protocol.json')
    deadline()
    cache = joblib.load(PREVIOUS / 'training_cache.joblib')
    assert set(cache['parts']) == set(cache['store']) == {'train', 'search'}
    parts, store = (cache['parts'], cache['store'])
    assert [len(parts[n]) for n in ('train', 'search')] == [132802, 8274]
    assert stack.CAT_FEATURES == protocol['original_features']
    assert stack.PREPROCESS == protocol['preprocess']
    with torch.inference_mode():
        original = shared.saved_mlp_predictions(stack, BASE / 'stack', store['search'])
    y_search = parts['search'].MONTHLY_RENT.to_numpy(float)
    np.testing.assert_allclose(stack.rmse(y_search, original), protocol['baseline']['search_rmse'], atol=1e-09, rtol=0)
    record('history_feature_preparation_start', training=False)
    train_feature, month_audit = features.training_history(parts['train'], deadline)
    search_feature = features.frozen_history(parts['train'], parts['search'].drop(columns='MONTHLY_RENT'))
    assert list(train_feature) == list(search_feature) == EXTRA
    joblib.dump({'train': train_feature, 'search': search_feature}, OUT / 'history_features.joblib')
    shared.save(OUT / 'history_audit.json', {'training_months': month_audit, 'search_reference_start': '2024-04', 'search_reference_end': '2024-09', 'search_reference_rows': 17960, 'no_search_or_confirm_rent_used': True, 'no_external_rent_used': True, 'training_feature_rows': len(train_feature), 'search_feature_rows': len(search_feature), 'search_at_least_five_peers': int((search_feature[EXTRA[1]] >= 5).sum()), 'original_labels_unchanged': True, 'sample_std_ddof': 1})
    for name, feat in (('train', train_feature), ('search', search_feature)):
        for col in EXTRA:
            parts[name][col] = feat[col].to_numpy()
    original_features = list(stack.CAT_FEATURES)
    stack.CAT_FEATURES = original_features + EXTRA
    assert len(stack.CAT_FEATURES) == 19
    record('history_feature_preparation_complete', training=False, total_columns=19)
    deadline()
    for town_index, town in enumerate(cache['towns'], 1):
        deadline()
        masks = {name: part.TOWN.astype(str).to_numpy() == town for name, part in parts.items()}
        fit, search = [parts[name].loc[masks[name]].reset_index(drop=True) for name in ('train', 'search')]
        stack._scale_coords(fit, [search])
        reference_folder = BASE / 'stack/catboost' / stack.town_slug(town)
        reference = stack.load_model(reference_folder)
        base_preparer = joblib.load(reference_folder / 'preprocessor.joblib')
        parameters = reference.get_params()
        assert parameters['thread_count'] == 8 and parameters['random_seed'] == 5228
        residuals = {name: frame.MONTHLY_RENT.to_numpy(float) / frame.floor_area_sqm.to_numpy(float) - store[name]['mu'][masks[name]] for name, frame in (('train', fit), ('search', search))}
        preparer = stack.fit_preprocessor(stack.PREPROCESS, fit, residuals['train'], stack.CAT_FEATURES)
        pools = {}
        for name, frame in (('train', fit), ('search', search)):
            matrix = stack.transform_frame(preparer, frame)
            baseline_matrix = stack.transform_frame(base_preparer, frame)
            pd.testing.assert_frame_equal(matrix[baseline_matrix.columns], baseline_matrix, check_exact=True)
            shared.finite(matrix.select_dtypes(include=[np.number]).to_numpy(), f'{town} {name} features')
            pools[name] = Pool(cb_module._fill_missing_categories(matrix), label=residuals[name], cat_features=cb_module._category_columns(matrix))
        record('catboost_town_start', town=town, index=town_index, fit_rows=len(fit), search_rows=len(search))
        model = CatBoostRegressor(**parameters)
        model.fit(pools['train'], eval_set=pools['search'], use_best_model=True, verbose=False)
        if model.get_all_params() != reference.get_all_params():
            delta = {k: [reference.get_all_params().get(k), v] for k, v in model.get_all_params().items() if reference.get_all_params().get(k) != v}
            raise RuntimeError(f'CatBoost resolved parameters differ: {town} {delta}')
        assert set(model.feature_names_) == set(stack.CAT_FEATURES)
        folder = OUT / 'catboost' / stack.town_slug(town)
        folder.mkdir(parents=True)
        stack.save_model(model, folder)
        joblib.dump(preparer, folder / 'preprocessor.joblib')
        store['search']['t'][masks['search']] = shared.finite(model.predict(pools['search']), town)
        shared.save(folder / 'fit_audit.json', {'params': parameters, 'resolved_params': model.get_all_params(), 'same_parameters_as_baseline': True, 'features': model.feature_names_, 'original_feature_values_exactly_unchanged': True, 'training_weights': 'none', 'trees': model.tree_count_, 'best_iteration': model.get_best_iteration(), 'new_feature_fit_medians': {c: preparer.numeric_fill_[c] for c in EXTRA}, 'eval_history': model.get_evals_result()})
        record('catboost_town_complete', town=town, completed=town_index, trees=model.tree_count_)
    deadline()
    record('frozen_mlp_search_prediction', training=False)
    with torch.inference_mode():
        prediction = shared.saved_mlp_predictions(stack, BASE / 'stack', store['search'])
    score = stack.rmse(y_search, prediction)
    search_csv = OUT / 'search_predictions.csv'
    pd.DataFrame({'source_row_id': cache['search_row_ids'], 'truth': y_search, 'prediction': prediction}).to_csv(search_csv, index=False)
    search_metrics = {'rows': len(y_search), 'rmse': score, 'catboost_plus_rnn_rmse': stack.rmse(y_search, (store['search']['mu'] + store['search']['t']) * store['search']['floor_area']), 'bias_pred_minus_truth': float(np.mean(prediction - y_search)), 'delta_from_baseline': score - protocol['baseline']['search_rmse'], 'delta_from_previous_best': score - protocol['previous_best']['search_rmse']}
    worst = np.zeros(len(y_search), dtype=bool)
    worst[np.argsort(np.abs(original - y_search))[-828:]] = True
    search_metrics['diagnostic_groups'] = {label: {'rows': int(mask.sum()), 'baseline_rmse': stack.rmse(y_search[mask], original[mask]), 'candidate_rmse': stack.rmse(y_search[mask], prediction[mask])} for label, mask in {'baseline_worst_10pct': worst, 'other_90pct': ~worst, 'rent_at_most_1500': y_search <= 1500, 'rent_above_1500': y_search > 1500}.items()}
    shared.save(OUT / 'search_metrics.json', search_metrics)
    passed = score < protocol['search_gate_rmse']
    frozen_paths = list((OUT / 'catboost').rglob('model.joblib')) + list((OUT / 'catboost').rglob('preprocessor.joblib'))
    frozen_paths += [OUT / 'history_features.joblib', OUT / 'history_audit.json', search_csv]
    selection = {'candidate': protocol['candidate'], 'passed_search_gate': passed, 'search_rmse': score, 'threshold': protocol['search_gate_rmse'], 'protocol_sha256': protocol_hash, 'confirm_evaluations_before_freeze': 0, 'epoch': time.time(), 'candidate_sha256': {p.relative_to(OUT).as_posix(): shared.digest(p) for p in frozen_paths}}
    shared.save(OUT / 'selection_frozen.json', selection)
    record('selection_frozen', passed_search_gate=passed, search_rmse=score)
    result = {'status': 'completed', 'candidate': protocol['candidate'], 'search': search_metrics, 'confirm': None, 'baseline': protocol['baseline'], 'previous_best': protocol['previous_best'], 'passed_search_gate': passed, 'candidate_confirm_evaluations': 0, 'mlp_rnn_gmm_retrained': False, 'original_labels_unchanged': True, 'protocol_sha256': protocol_hash, 'selection_sha256': shared.digest(OUT / 'selection_frozen.json')}
    if passed:
        deadline()
        with (OUT / 'confirm_started.json').open('x') as handle:
            json.dump({'epoch': time.time(), 'pid': os.getpid(), 'selection_sha256': result['selection_sha256']}, handle)
        record('confirm_prediction_started', candidate_confirm_evaluations=1)
        frame, split, towns = shared.inputs(stack)
        assert split.fit_months[-1] == '2024-09' and split.search_months == protocol['search_months']
        assert split.confirm_months == protocol['confirm_months']
        confirm = split.confirm.drop(columns='MONTHLY_RENT')
        confirm_feature = features.frozen_history(split.fit, confirm)
        joblib.dump(confirm_feature, OUT / 'confirm_history_features.joblib')
        for col in EXTRA:
            confirm[col] = confirm_feature[col].to_numpy()
        confirm_parts = {'confirm': confirm}
        confirm_store = stack._empty_store(confirm_parts, confirm.iloc[:0])
        confirm_store.pop('test')
        with torch.inference_mode():
            shared.model_predictions(stack, OUT, confirm_store, confirm_parts, split.fit.drop(columns='MONTHLY_RENT'), towns)
            confirm_prediction = shared.saved_mlp_predictions(stack, BASE / 'stack', confirm_store['confirm'])
        truth = split.confirm.MONTHLY_RENT.to_numpy(float)
        confirm_score = stack.rmse(truth, confirm_prediction)
        row_ids = np.flatnonzero(frame.RENT_APPROVAL_DATE.astype(str).str[:7].isin(protocol['confirm_months']).to_numpy())
        path = OUT / 'confirm_predictions.csv'
        pd.DataFrame({'source_row_id': row_ids, 'date': split.confirm.RENT_APPROVAL_DATE.to_numpy(), 'truth': truth, 'prediction': confirm_prediction}).to_csv(path, index=False)
        confirmation = {'rows': len(truth), 'rmse': confirm_score, 'delta_from_baseline': confirm_score - protocol['baseline']['confirm_rmse'], 'delta_from_previous_best': confirm_score - protocol['previous_best']['confirm_rmse'], 'bias_pred_minus_truth': float(np.mean(confirm_prediction - truth)), 'by_month': {month: stack.rmse(truth[split.confirm.RENT_APPROVAL_DATE.astype(str).str[:7] == month], confirm_prediction[split.confirm.RENT_APPROVAL_DATE.astype(str).str[:7] == month]) for month in protocol['confirm_months']}}
        shared.save(OUT / 'confirmation_metrics.json', confirmation)
        result.update(confirm=confirmation, candidate_confirm_evaluations=1, confirm_predictions_sha256=shared.digest(path))
        result['improved_over_previous_best_on_both_periods'] = confirm_score < protocol['previous_best']['confirm_rmse']
        result['conclusion'] = 'Improved both periods versus previous best; iterative validation, not an untouched test set.' if result['improved_over_previous_best_on_both_periods'] else 'Search improved but Confirm did not beat previous best; optimization not established; no reselection.'
    else:
        result.update(improved_over_previous_best_on_both_periods=False, conclusion='Search did not beat previous best; Confirm not evaluated; no retries.')
    deadline()
    check_frozen()
    assert shared.digest(OUT / 'protocol.json') == protocol_hash
    for relative, expected in selection['candidate_sha256'].items():
        assert shared.digest(OUT / relative) == expected
    result.update(source_baseline_and_previous_results_unchanged=True, exit_code=0, elapsed_seconds=time.time() - shared.read(OUT / 'supervisor.json')['started_epoch'], search_predictions_csv=str(search_csv), confirm_predictions_csv=str(OUT / 'confirm_predictions.csv') if passed else None)
    shared.save(OUT / 'results.json', result)
    record('completed', search_rmse=score, confirm=result['confirm'], conclusion=result['conclusion'])
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['prepare', 'run'])
    args = parser.parse_args()
    try:
        prepare() if args.phase == 'prepare' else run()
    except BaseException as exc:
        traceback.print_exc()
        if OUT.exists():
            record('failed', phase=args.phase, error=repr(exc))
        raise
