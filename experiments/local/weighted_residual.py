"""One original-feature CatBoost candidate: area-squared loss weights only."""
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
OUT = asset_path(Path('weighted_residual_01'))
BASE = asset_path(Path('baseline_alignment'))
PREVIOUS = asset_path(Path('feature_ablation_01'))

def record(stage, **fields):
    item = {'stage': stage, 'pid': os.getpid(), 'epoch': time.time(), **fields}
    with (OUT / 'events.jsonl').open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(item, allow_nan=False) + '\n')
    shared.save(OUT / 'status.json', item)
    print(json.dumps(item), flush=True)

def runtime():
    global np, pd, torch, joblib, Pool, CatBoostRegressor, cb_module
    stack = shared.runtime()
    import numpy as np
    import pandas as pd
    import torch
    import joblib
    from catboost import Pool, CatBoostRegressor
    import experiments.local._support.models.catboost.model as cb_module
    from sklearn.mixture import GaussianMixture

    def forbidden(*args, **kwargs):
        raise RuntimeError('Only the authorized CatBoost fitting is permitted')
    stack._fit_mlp = forbidden
    stack._fit_mixture = forbidden
    GaussianMixture.fit = forbidden
    torch.optim.Adam.step = forbidden
    return stack

def town_data(stack, cache, town):
    parts, store = (cache['parts'], cache['store'])
    masks = {name: part.TOWN.astype(str).to_numpy() == town for name, part in parts.items()}
    fit, search = [parts[name].loc[masks[name]].reset_index(drop=True) for name in ('train', 'search')]
    stack._scale_coords(fit, [search])
    folder = BASE / 'stack/catboost' / stack.town_slug(town)
    preparer = joblib.load(folder / 'preprocessor.joblib')
    assert preparer.selected_columns_ == stack.CAT_FEATURES and len(stack.CAT_FEATURES) == 16
    original_model = stack.load_model(folder)
    params = original_model.get_params()
    assert params['thread_count'] == 8 and params['random_seed'] == stack.SEED
    pools, residual, weights, frames = ({}, {}, {}, {'train': fit, 'search': search})
    normalizer = float(np.mean(fit.floor_area_sqm.to_numpy(float) ** 2))
    assert np.isfinite(normalizer) and normalizer > 0
    for name, part in frames.items():
        area = shared.finite(part.floor_area_sqm.to_numpy(float), f'{town} {name} area')
        assert (area > 0).all()
        residual[name] = shared.finite(part.MONTHLY_RENT.to_numpy(float) / area - store[name]['mu'][masks[name]], f'{town} {name} residual')
        weights[name] = shared.finite(area ** 2 / normalizer, f'{town} {name} weights')
        matrix = stack.transform_frame(preparer, part)
        categories = cb_module._category_columns(matrix)
        matrix = cb_module._fill_missing_categories(matrix)
        shared.finite(matrix.select_dtypes(include=[np.number]).to_numpy(), f'{town} {name} numeric matrix')
        pools[name] = Pool(matrix, label=residual[name], cat_features=categories, weight=weights[name])
        np.testing.assert_allclose(pools[name].get_weight(), weights[name], rtol=1e-07, atol=1e-08)
    return (masks, frames, pools, residual, weights, preparer, original_model, params, normalizer)

def check_frozen():
    shared.verify_frozen()
    protocol = shared.read(OUT / 'protocol.json')
    for relative, expected in protocol['frozen_sha256'].items():
        if shared.digest(asset_path(Path(relative))) != expected:
            raise RuntimeError(f'Frozen file changed: {relative}')
    return protocol

def prepare(stack):
    if OUT.exists():
        raise RuntimeError('Refusing to replace an existing weighted candidate')
    baseline_protocol = shared.verify_frozen()
    cache = joblib.load(PREVIOUS / 'training_cache.joblib')
    assert set(cache['parts']) == {'train', 'search'}
    assert [len(cache['parts'][name]) for name in ('train', 'search')] == [132802, 8274]
    checks = []
    for town in cache['towns']:
        masks, frames, pools, residual, weights, preparer, model, params, normalizer = town_data(stack, cache, town)
        prediction = model.predict(pools['search'])
        np.testing.assert_allclose(prediction, cache['store']['search']['t'][masks['search']], rtol=0, atol=1e-10)
        error = prediction - residual['search']
        area = frames['search'].floor_area_sqm.to_numpy(float)
        monthly_mse = float(np.mean((area * error) ** 2))
        weighted_unit_mse = float(np.average(error ** 2, weights=weights['search']))
        np.testing.assert_allclose(weighted_unit_mse * np.mean(area ** 2), monthly_mse, rtol=1e-12)
        checks.append({'town': town, 'fit_rows': len(frames['train']), 'search_rows': len(frames['search']), 'fit_weight_mean': float(weights['train'].mean()), 'fit_area_squared_mean': normalizer, 'fit_weight_range': [float(weights['train'].min()), float(weights['train'].max())], 'baseline_pool_predictions_match': True, 'loss_identity_passed': True})
    best_path = asset_path(Path('frozen_mlp_area_confirm_01/results.json'))
    best = shared.read(best_path)
    assert best['passed_predeclared_criterion'] and best['training_performed'] is False
    frozen_paths = [Path(__file__).resolve(), PREVIOUS / 'protocol.json', PREVIOUS / 'results.json', PREVIOUS / 'selection_frozen.json', best_path, asset_path(Path('frozen_mlp_area_confirm_01/protocol.json'))]
    OUT.mkdir()
    protocol = {'status': 'frozen_before_training', 'candidate': 'area_squared_loss_original_features_frozen_mlp', 'user_scope': 'One area-squared-weight CatBoost candidate, original 16 features and split, frozen original RNN/GMM/MLP; 900 seconds total, no retries.', 'fit_end': '2024-09', 'search_months': ['2024-10', '2024-11', '2024-12'], 'confirm_months': ['2025-01', '2025-02', '2025-03'], 'rows': {'train': 132802, 'search': 8274, 'confirm': 8924}, 'features': stack.CAT_FEATURES, 'seed': stack.SEED, 'threads': 8, 'device': 'CPU', 'model_parameters': 'Exact get_params() from each saved baseline_alignment CatBoost; verify resolved parameters after fitting.', 'weights': 'train and Search: floor_area_sqm^2 / mean(fit-town floor_area_sqm^2); strictly positive; same fit-derived normalizer applied to Search.', 'preprocessor': 'Read-only original per-town baseline preprocessors; no fitting or altered columns.', 'loss_and_early_stopping': 'Original RMSE with weighted train and Search Pool objects; early stopping unchanged at 50 rounds and 1000 maximum trees.', 'final_metric': 'Unweighted monthly-rent RMSE after frozen original 19-input MLP and original normalizers.', 'frozen_other_models': str(BASE), 'baseline': baseline_protocol['baseline'], 'previous_best': {'name': best['candidate'], 'search_rmse': best['search_rmse'], 'confirm_rmse': best['confirm']['rmse']}, 'search_gate_rmse': min(baseline_protocol['baseline']['search_rmse'], best['search_rmse']), 'selection': 'Freeze after one complete Search score. Only a strict improvement over the existing best Search enters one Confirm evaluation. Never tune or reselect using Confirm.', 'confirm_caveat': 'This original Confirm period has been inspected in prior rounds; report this as iterative validation, not a new untouched holdout.', 'hard_limit_seconds': 900, 'candidate_count': 1, 'automatic_retries': 0, 'frozen_sha256': {asset_key(p).as_posix(): shared.digest(p) for p in frozen_paths}}
    shared.save(OUT / 'protocol.json', protocol)
    shared.save(OUT / 'preflight.json', {'status': 'passed', 'training_performed': False, 'confirm_in_training_cache': False, 'towns': checks, 'features': stack.CAT_FEATURES})
    record('preflight_passed', training=False, towns=len(checks), search_gate=protocol['search_gate_rmse'])

def run(stack):
    started = time.time()
    protocol = check_frozen()
    with (OUT / 'run_started.json').open('x', encoding='utf-8') as handle:
        json.dump({'pid': os.getpid(), 'epoch': started}, handle)
    protocol_hash = shared.digest(OUT / 'protocol.json')
    cache = joblib.load(PREVIOUS / 'training_cache.joblib')
    assert set(cache['parts']) == set(cache['store']) == {'train', 'search'}
    assert stack.CAT_FEATURES == protocol['features'] and len(stack.CAT_FEATURES) == 16
    store, parts = (cache['store'], cache['parts'])
    record('training_start', features=stack.CAT_FEATURES, hard_limit_seconds=900, mlp_training=False)
    for index, town in enumerate(cache['towns'], 1):
        data = town_data(stack, cache, town)
        masks, frames, pools, residual, weights, preparer, reference, params, normalizer = data
        record('catboost_town_start', town=town, index=index)
        model = CatBoostRegressor(**params)
        model.fit(pools['train'], eval_set=pools['search'], use_best_model=True, verbose=False)
        if model.get_all_params() != reference.get_all_params():
            differences = {k: [reference.get_all_params().get(k), v] for k, v in model.get_all_params().items() if reference.get_all_params().get(k) != v}
            raise RuntimeError(f'Resolved CatBoost parameters changed: {town} {differences}')
        folder = OUT / 'catboost' / stack.town_slug(town)
        folder.mkdir(parents=True)
        stack.save_model(model, folder)
        joblib.dump(preparer, folder / 'preprocessor.joblib')
        for name in parts:
            store[name]['t'][masks[name]] = shared.finite(model.predict(pools[name]), f'{town} {name}')
        search_prediction = store['search']['t'][masks['search']]
        weighted_rmse = float(np.sqrt(np.average((search_prediction - residual['search']) ** 2, weights=weights['search'])))
        best_metric = model.get_best_score()['validation']['RMSE']
        np.testing.assert_allclose(weighted_rmse, best_metric, rtol=1e-05, atol=1e-05)
        shared.save(folder / 'fit_audit.json', {'params': params, 'resolved_params': model.get_all_params(), 'same_resolved_parameters_as_baseline': True, 'features': model.feature_names_, 'fit_weight_normalizer': normalizer, 'trees': model.tree_count_, 'best_iteration': model.get_best_iteration(), 'weighted_search_rmse': weighted_rmse, 'logged_best_weighted_search_rmse': best_metric, 'eval_history': model.get_evals_result()})
        record('catboost_town_complete', town=town, completed=index, trees=model.tree_count_)
    record('frozen_mlp_search_prediction', training=False)
    with torch.inference_mode():
        predictions = shared.saved_mlp_predictions(stack, BASE / 'stack', store['search'])
    truth = parts['search'].MONTHLY_RENT.to_numpy(float)
    score = stack.rmse(truth, predictions)
    cat_score = stack.rmse(truth, (store['search']['mu'] + store['search']['t']) * store['search']['floor_area'])
    search_csv = OUT / 'search_predictions.csv'
    pd.DataFrame({'source_row_id': cache['search_row_ids'], 'truth': truth, 'prediction': predictions}).to_csv(search_csv, index=False)
    search = {'rows': len(truth), 'rmse': score, 'catboost_plus_rnn_rmse': cat_score, 'bias_pred_minus_truth': float(np.mean(predictions - truth)), 'delta_from_baseline': score - protocol['baseline']['search_rmse'], 'delta_from_previous_best': score - protocol['previous_best']['search_rmse']}
    shared.save(OUT / 'search_metrics.json', search)
    passed = score < protocol['search_gate_rmse']
    frozen = {'candidate': protocol['candidate'], 'passed_search_gate': passed, 'search_rmse': score, 'threshold': protocol['search_gate_rmse'], 'confirm_evaluations_before_freeze': 0, 'protocol_sha256': protocol_hash, 'epoch': time.time()}
    shared.save(OUT / 'selection_frozen.json', frozen)
    record('selection_frozen', passed_search_gate=passed, search_rmse=score)
    result = {'status': 'completed', 'candidate': protocol['candidate'], 'search': search, 'confirm': None, 'baseline': protocol['baseline'], 'previous_best': protocol['previous_best'], 'passed_search_gate': passed, 'protocol_sha256': protocol_hash, 'selection_sha256': shared.digest(OUT / 'selection_frozen.json'), 'mlp_rnn_gmm_retrained': False, 'candidate_confirm_evaluations': 0, 'search_predictions_sha256': shared.digest(search_csv)}
    if passed:
        record('confirm_prediction_started', candidate_confirm_evaluations=1)
        frame, split, towns = shared.inputs(stack)
        assert split.fit_months[-1] == protocol['fit_end'] and split.search_months == protocol['search_months']
        assert split.confirm_months == protocol['confirm_months']
        confirm_input = split.confirm.drop(columns='MONTHLY_RENT')
        confirm_parts = {'confirm': confirm_input}
        confirm_store = stack._empty_store(confirm_parts, confirm_input.iloc[:0])
        confirm_store.pop('test')
        with torch.inference_mode():
            shared.model_predictions(stack, OUT, confirm_store, confirm_parts, split.fit.drop(columns='MONTHLY_RENT'), towns)
            confirm_prediction = shared.saved_mlp_predictions(stack, BASE / 'stack', confirm_store['confirm'])
        target = split.confirm.MONTHLY_RENT.to_numpy(float)
        confirm_score = stack.rmse(target, confirm_prediction)
        ids = np.flatnonzero(frame.RENT_APPROVAL_DATE.astype(str).str[:7].isin(protocol['confirm_months']).to_numpy())
        path = OUT / 'confirm_predictions.csv'
        pd.DataFrame({'source_row_id': ids, 'date': split.confirm.RENT_APPROVAL_DATE.to_numpy(), 'truth': target, 'prediction': confirm_prediction}).to_csv(path, index=False)
        confirm = {'rows': len(target), 'rmse': confirm_score, 'bias_pred_minus_truth': float(np.mean(confirm_prediction - target)), 'delta_from_baseline': confirm_score - protocol['baseline']['confirm_rmse'], 'delta_from_previous_best': confirm_score - protocol['previous_best']['confirm_rmse']}
        shared.save(OUT / 'confirmation_metrics.json', confirm)
        result.update(confirm=confirm, candidate_confirm_evaluations=1, confirm_predictions_sha256=shared.digest(path))
        result['improved_over_previous_best_on_both_periods'] = confirm_score < protocol['previous_best']['confirm_rmse']
        result['conclusion'] = 'Improved both original periods versus previous best; iterative validation only.' if result['improved_over_previous_best_on_both_periods'] else 'Search improved but Confirm did not beat previous best; no tuning or reselection.'
    else:
        result['improved_over_previous_best_on_both_periods'] = False
        result['conclusion'] = 'Search did not beat previous best; Confirm not evaluated; no retries or additional candidates.'
    check_frozen()
    assert shared.digest(OUT / 'protocol.json') == protocol_hash
    result.update(source_baseline_and_previous_results_unchanged=True, elapsed_seconds=time.time() - started, exit_code=0)
    shared.save(OUT / 'results.json', result)
    record('completed', search_rmse=score, confirm=result['confirm'], conclusion=result['conclusion'])
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['prepare', 'run'])
    phase = parser.parse_args().phase
    try:
        stack = runtime()
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=8):
            prepare(stack) if phase == 'prepare' else run(stack)
    except BaseException as exc:
        traceback.print_exc()
        if OUT.exists():
            record('failed', phase=phase, error=repr(exc))
        raise
