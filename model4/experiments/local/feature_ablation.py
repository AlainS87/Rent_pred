"""Two authorized, single-feature CatBoost ablations on the original split.

Original exported source and baseline_alignment artifacts are read-only.
Confirm is scored only for the Search-selected candidate, after selection freezes.
"""
from experiments.local.bootstrap import Path, CODE_DIR, DATA_ROOT, SUPPORT_ROOT, ARTIFACT_ROOT, asset_path, asset_key, support_path
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
SOURCE = asset_path(Path('support'))
BASELINE = asset_path(Path('baseline_alignment'))
OUT = asset_path(Path('feature_ablation_01'))
CANDIDATES = {'area_only': 'floor_area_sqm', 'age_only': 'building_age_years'}

def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()

def record(stage, **fields):
    event = {'stage': stage, 'pid': os.getpid(), 'utc_epoch': time.time(), **fields}
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'events.jsonl').open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(event, allow_nan=False) + '\n')
    save(OUT / 'status.json', event)
    print(json.dumps(event), flush=True)

def runtime():
    global np, pd, torch, joblib
    import numpy as np
    import pandas as pd
    import torch
    import joblib
    import experiments.local._support.models.catboost.model as cb_module
    from experiments.local._support.mean_residual_model.reginal_mean_model.rnn import model as rnn_model
    torch.set_num_threads(8)
    torch.set_num_interop_threads(2)
    original_regressor = cb_module.CatBoostRegressor

    def threaded_regressor(**kwargs):
        kwargs['thread_count'] = 8
        return original_regressor(**kwargs)
    cb_module.CatBoostRegressor = threaded_regressor
    rnn_model.REGIONAL_DIR = BASELINE / 'regional'
    from experiments.local._support import load_stack
    stack = load_stack('_ablation_stack')
    return stack

def inputs(stack):
    frame = stack.load_joined('train', stack.LOAD_COLUMNS)
    split = stack.time_split(frame, 3, None)
    stack._check_months(split)
    assert [len(split.fit), len(split.search), len(split.confirm)] == [132802, 8274, 8924]
    assert len(frame) == 150000
    periods = {'train': split.fit, 'search': split.search, 'confirm': split.confirm}
    towns = stack._towns(periods)
    assert len(towns) == 26
    return (frame, split, towns)

def finite(values, name):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f'Nonfinite values in {name}')
    return values

def model_predictions(stack, model_root, store, parts, fit_frame, towns):
    """Apply frozen RNN/GMM and given CatBoosts; no fitting and no label scoring."""
    for name, part in parts.items():
        store[name]['mu'][:] = finite(stack.predict_saved_rnn3(part), f'RNN {name}')
    for town in towns:
        fit = fit_frame.loc[fit_frame.TOWN.astype(str) == town].reset_index(drop=True)
        masks = {name: part.TOWN.astype(str).to_numpy() == town for name, part in parts.items()}
        local = {name: part.loc[masks[name]].reset_index(drop=True) for name, part in parts.items()}
        stack._scale_coords(fit, list(local.values()))
        model_folder = model_root / 'catboost' / stack.town_slug(town)
        model = stack.load_model(model_folder)
        preparer = joblib.load(model_folder / 'preprocessor.joblib')
        mixture = joblib.load(BASELINE / 'stack/gmm' / f'{stack.town_slug(town)}.joblib')
        for name, part in local.items():
            stack._write_coords(store, name, masks[name], part)
            stack._write_mixture(store, name, masks[name], part, mixture)
            store[name]['t'][masks[name]] = stack.predict_model(model, stack.transform_frame(preparer, part))

def saved_mlp_predictions(stack, folder, block):
    payload = torch.load(folder / 'mlp/model.pt', map_location='cpu', weights_only=False)
    matrix = stack._matrix(block, center=payload['x_center'], spread=payload['x_spread'])
    finite(matrix, 'saved MLP inputs')
    assert matrix.shape[1] == 19
    network = stack._MLP(19)
    network.load_state_dict(payload['state'])
    return finite(stack._mlp_predict(network, block, payload['y_center'], payload['y_spread'], payload['x_center'], payload['x_spread']), 'saved MLP outputs')

def verify_frozen():
    protocol = read(OUT / 'protocol.json')
    for relative, expected in protocol['frozen_sha256'].items():
        actual = digest(asset_path(Path(relative)))
        if actual != expected:
            raise RuntimeError(f'Frozen source/data/baseline/cache changed: {relative}')
    return protocol

def prepare(stack):
    if (OUT / 'protocol.json').exists():
        raise RuntimeError('Refusing to overwrite an existing ablation protocol')
    record('preflight_start', training=False)
    frame, split, towns = inputs(stack)
    parts = {'train': split.fit, 'search': split.search}
    store = stack._empty_store(parts, split.fit.iloc[:0])
    store.pop('test')
    model_predictions(stack, BASELINE / 'stack', store, parts, split.fit, towns)
    for name in parts:
        finite(store[name]['mu'], f'RNN {name}')
        finite(store[name]['gmm_gap'], f'GMM {name}')
        finite(store[name]['t'], f'baseline CatBoost {name}')
    train_x, center, spread = stack._matrix(store['train'], fit=True)
    finite(train_x, 'baseline train matrix')
    finite(stack._matrix(store['search'], center=center, spread=spread), 'baseline Search matrix')
    assert train_x.shape == (132802, 19)
    baseline = read(BASELINE / 'results.json')
    predictions = saved_mlp_predictions(stack, BASELINE / 'stack', store['search'])
    score = stack.rmse(split.search.MONTHLY_RENT.to_numpy(float), predictions)
    if abs(score - baseline['metrics']['search_rmse']) > 1e-08:
        raise RuntimeError(f'Reconstructed baseline Search changed: {score}')
    row_month = frame.RENT_APPROVAL_DATE.astype(str).str[:7]
    search_row_ids = np.flatnonzero(row_month.isin(split.search_months).to_numpy())
    pd.DataFrame({'source_row_id': search_row_ids, 'truth': split.search.MONTHLY_RENT, 'prediction': predictions}).to_csv(OUT / 'baseline_search_predictions.csv', index=False)
    feature_checks = {}
    for label, column in CANDIDATES.items():
        feature_checks[label] = {}
        for town in towns:
            fit = split.fit.loc[split.fit.TOWN.astype(str) == town].reset_index(drop=True)
            search = split.search.loc[split.search.TOWN.astype(str) == town].reset_index(drop=True)
            stack._scale_coords(fit, [search])
            y = fit.MONTHLY_RENT.to_numpy(float) / fit.floor_area_sqm.to_numpy(float)
            preparer = stack.fit_preprocessor(stack.PREPROCESS, fit, y, [*stack.CAT_FEATURES, column])
            for name, data in (('train', fit), ('search', search)):
                transformed = stack.transform_frame(preparer, data)
                finite(transformed.select_dtypes(include=[np.number]).to_numpy(), f'{label} {town} {name}')
            feature_checks[label][town] = {'fit_rows': len(fit), 'search_rows': len(search)}
    joblib.dump({'parts': parts, 'store': store, 'towns': towns, 'search_row_ids': search_row_ids}, OUT / 'training_cache.joblib')
    frozen_files = set(SUPPORT_ROOT.rglob('*.py'))
    frozen_files.update((support_path(Path('Data/auxiliary_features') / f'{part}_features.csv') for part in ('train', 'test')))
    frozen_files.update((path for path in BASELINE.rglob('*') if path.is_file()))
    frozen_files.update([Path(__file__).resolve(), OUT / 'training_cache.joblib'])
    protocol = {'status': 'frozen_before_candidate_training', 'candidates': CANDIDATES, 'candidate_order': list(CANDIDATES), 'fit_end': '2024-09', 'search_months': split.search_months, 'confirm_months': split.confirm_months, 'rows': {'train': 132802, 'search': 8274, 'confirm': 8924}, 'original_catboost_features': stack.CAT_FEATURES, 'catboost_architecture': stack.CAT_ARCH, 'preprocess': stack.PREPROCESS, 'seed': stack.SEED, 'threads': 8, 'device': 'CPU', 'mlp': 'Unmodified internal support _fit_mlp, _MLP, _matrix; 19 inputs; original early stopping on Search', 'reuse': 'Only baseline_alignment RNN/GMM; no old failed models or temporal-run models', 'baseline': baseline['metrics'], 'selection': 'Minimum Search RMSE including baseline; exact tie keeps baseline then candidate order. Freeze before candidate Confirm scoring.', 'confirmation': 'Only frozen selected candidate; failure to improve Confirm is reported, never used to reselect or tune.', 'per_candidate_seconds': 900, 'total_seconds': 1800, 'automatic_retries': 0, 'scope': 'Two independent one-column CatBoost additions; no combined candidate, other tuning, production refit or external publication', 'frozen_sha256': {asset_key(path).as_posix(): digest(path) for path in sorted(frozen_files)}}
    save(OUT / 'protocol.json', protocol)
    save(OUT / 'preflight.json', {'status': 'passed', 'training_performed': False, 'baseline_search_recomputed': score, 'feature_checks': feature_checks, 'confirm_in_training_cache': False, 'matrix_columns': 19})
    record('preflight_passed', training=False, baseline_search=score, candidates=list(CANDIDATES))

def train_candidate(stack, label):
    started = time.time()
    protocol = verify_frozen()
    folder = OUT / label
    if folder.exists():
        raise RuntimeError(f'Refusing to rerun candidate {label}')
    folder.mkdir()
    stack.HERE = folder
    stack.CAT_FEATURES = [*protocol['original_catboost_features'], CANDIDATES[label]]
    assert stack.CAT_ARCH == protocol['catboost_architecture']
    assert stack.PREPROCESS == protocol['preprocess']
    cache = joblib.load(OUT / 'training_cache.joblib')
    parts, store = (cache['parts'], cache['store'])
    assert set(parts) == {'train', 'search'} and set(store) == {'train', 'search'}
    residuals = {name: part.MONTHLY_RENT.to_numpy(float) / part.floor_area_sqm.to_numpy(float) - store[name]['mu'] for name, part in parts.items()}
    record('candidate_start', candidate=label, added_feature=CANDIDATES[label], columns=stack.CAT_FEATURES)
    for index, town in enumerate(cache['towns'], 1):
        masks = {name: part.TOWN.astype(str).to_numpy() == town for name, part in parts.items()}
        fit, search = [parts[name].loc[masks[name]].reset_index(drop=True) for name in ('train', 'search')]
        stack._scale_coords(fit, [search])
        record('catboost_town_start', candidate=label, town=town, town_index=index)
        model, preparer = stack._catboost(fit, residuals['train'][masks['train']], search, residuals['search'][masks['search']])
        model_folder = folder / 'catboost' / stack.town_slug(town)
        model_folder.mkdir(parents=True)
        stack.save_model(model, model_folder)
        joblib.dump(preparer, model_folder / 'preprocessor.joblib')
        if index == 1:
            save(folder / 'catboost_resolved_parameters.json', model.get_all_params())
        for name, data in (('train', fit), ('search', search)):
            store[name]['t'][masks[name]] = finite(stack.predict_model(model, stack.transform_frame(preparer, data)), f'{label} CatBoost {name} {town}')
        record('catboost_town_complete', candidate=label, town=town, completed=index, trees=model.tree_count_)
    train_x, center, spread = stack._matrix(store['train'], fit=True)
    finite(train_x, 'candidate real MLP train matrix')
    finite(stack._matrix(store['search'], center=center, spread=spread), 'candidate real MLP Search matrix')
    assert train_x.shape == (132802, 19)
    rents = {name: finite(part.MONTHLY_RENT.to_numpy(float), name) for name, part in parts.items()}
    record('mlp_start', candidate=label, columns=19, confirm_available=False)
    network, epoch, center, spread = stack._fit_mlp(store, rents)
    stack._save_mlp(network, epoch, center, spread, store)
    predictions = finite(stack._mlp_predict(network, store['search'], center, spread), 'candidate Search predictions')
    score = stack.rmse(rents['search'], predictions)
    catboost_score = stack.rmse(rents['search'], (store['search']['mu'] + store['search']['t']) * store['search']['floor_area'])
    pd.DataFrame({'source_row_id': cache['search_row_ids'], 'truth': rents['search'], 'prediction': predictions}).to_csv(folder / 'search_predictions.csv', index=False)
    verify_frozen()
    result = {'status': 'search_completed', 'candidate': label, 'added_feature': CANDIDATES[label], 'search_rmse': score, 'catboost_search_rmse': catboost_score, 'best_epoch': epoch, 'baseline_search_delta': score - protocol['baseline']['search_rmse'], 'confirm_scored': False, 'elapsed_seconds': time.time() - started}
    save(folder / 'search_metrics.json', result)
    record('candidate_complete', **result)

def finalize(stack):
    protocol = verify_frozen()
    metrics = {name: read(OUT / name / 'search_metrics.json') for name in CANDIDATES}
    scores = {'baseline': protocol['baseline']['search_rmse'], **{name: item['search_rmse'] for name, item in metrics.items()}}
    winner = min(scores, key=scores.get)
    selection_path = OUT / 'selection_frozen.json'
    if selection_path.exists():
        raise RuntimeError('Selection was already frozen; no rerun allowed')
    selection = {'chosen': winner, 'search_rmse': scores, 'criterion': protocol['selection'], 'candidate_confirm_scored': False, 'frozen_epoch': time.time(), 'protocol_sha256': digest(OUT / 'protocol.json')}
    save(selection_path, selection)
    record('selection_frozen', chosen=winner, search_rmse=scores)
    result = {'status': 'completed', 'search': metrics, 'baseline': protocol['baseline'], 'chosen_on_search': winner, 'selection_sha256': digest(selection_path), 'candidate_confirm': None}
    if winner == 'baseline':
        result['conclusion'] = 'Neither candidate improved Search; baseline retained; no candidate Confirm scoring.'
    else:
        record('selected_confirm_start', candidate=winner)
        frame, split, towns = inputs(stack)
        parts = {'confirm': split.confirm}
        store = stack._empty_store(parts, split.fit.iloc[:0])
        folder = OUT / winner
        model_predictions(stack, folder, store, parts, split.fit, towns)
        confirm_pred = saved_mlp_predictions(stack, folder, store['confirm'])
        confirm_score = stack.rmse(split.confirm.MONTHLY_RENT.to_numpy(float), confirm_pred)
        month = frame.RENT_APPROVAL_DATE.astype(str).str[:7]
        ids = np.flatnonzero(month.isin(split.confirm_months).to_numpy())
        pd.DataFrame({'source_row_id': ids, 'truth': split.confirm.MONTHLY_RENT, 'prediction': confirm_pred}).to_csv(folder / 'confirm_predictions.csv', index=False)
        confirmation = {'candidate': winner, 'confirm_rmse': confirm_score, 'delta_from_local_baseline': confirm_score - protocol['baseline']['confirm_rmse'], 'selection_frozen_before_scoring': True}
        save(folder / 'confirmation_metrics.json', confirmation)
        result['candidate_confirm'] = confirmation
        result['conclusion'] = 'Selected candidate improved both Search and Confirm versus the local control.' if confirmation['delta_from_local_baseline'] < 0 else 'Search improvement did not transfer to Confirm; optimization not established; no reselection.'
    verify_frozen()
    result['frozen_source_data_baseline_verified'] = True
    save(OUT / 'results.json', result)
    record('completed', chosen=winner, conclusion=result['conclusion'])
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['prepare', *CANDIDATES, 'finalize'])
    phase = parser.parse_args().phase
    try:
        stack = runtime()
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=8):
            if phase == 'prepare':
                prepare(stack)
            elif phase == 'finalize':
                finalize(stack)
            else:
                train_candidate(stack, phase)
    except BaseException as exc:
        traceback.print_exc()
        record('failed', phase=phase, error=repr(exc))
        raise
