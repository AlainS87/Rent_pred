"""Fit and evaluate the regional RNN, conditional residuals and final MLP."""
from experiments.local.bootstrap import Path, CODE_DIR, DATA_ROOT, SUPPORT_ROOT, ARTIFACT_ROOT, asset_path, asset_key, support_path
import json
import os
from pathlib import Path
import sys
import time
import traceback
HERE = Path(__file__).resolve().parent
OUT = asset_path(Path('baseline_alignment'))
SOURCE = asset_path(Path('support'))

def record(stage, **fields):
    value = {'stage': stage, 'pid': os.getpid(), 'utc_epoch': time.time(), **fields}
    print(json.dumps(value), flush=True)
    with (OUT / 'events.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(value) + '\n')
    (OUT / 'status.json').write_text(json.dumps(value, indent=2), encoding='utf-8')

def fit_stacked_mlp(stack, store, periods):
    """Fit the original full-training-input MLP and score labeled periods only."""
    rents = {name: part.MONTHLY_RENT.to_numpy(dtype=float) for name, part in periods.items()}
    network, epoch, center, spread = stack._fit_mlp(store, rents)
    stack._save_mlp(network, epoch, center, spread, store)
    metrics = {name + '_rmse': stack.rmse(rents[name], stack._mlp_predict(network, store[name], center, spread)) for name in periods}
    metrics.update(best_epoch=epoch, catboost='frozen_full_model', gmm='conditional_location_flat_per_sqm', mean_model='rnn_h16_l3_per_sqm', mlp_input='per_sqm_terms_times_area', mlp_stopped_on_search=True, target='monthly_rent')
    (stack.HERE / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n', encoding='utf-8')
    return metrics

def execute():
    import joblib
    import numpy as np
    import pandas as pd
    import torch
    from threadpoolctl import threadpool_limits
    from experiments.local import train_town_rnn as train_l3
    from experiments.local._support.mean_residual_model.reginal_mean_model.rnn import model as rnn_model
    import experiments.local._support.models.catboost.model as cb_module
    from experiments.local._support.src.pipeline.data import time_split
    from experiments.local._support import load_stack
    stack = load_stack('_baseline_stack')
    torch.set_num_threads(8)
    torch.set_num_interop_threads(2)
    original_regressor = cb_module.CatBoostRegressor

    def threaded_regressor(**kwargs):
        kwargs['thread_count'] = 8
        return original_regressor(**kwargs)
    cb_module.CatBoostRegressor = threaded_regressor
    train_l3.FOLDER = OUT / 'regional/rnn/h16_l3'
    rnn_model.REGIONAL_DIR = OUT / 'regional'
    stack.HERE = OUT / 'stack'
    stack.ROOT = OUT
    stack.HERE.mkdir(parents=True, exist_ok=True)
    frame = stack.load_joined('train', stack.LOAD_COLUMNS)
    split = time_split(frame, 3, None)
    stack._check_months(split)
    periods = {'train': split.fit, 'search': split.search, 'confirm': split.confirm}
    test = stack.load_joined('query', stack.LOAD_COLUMNS)
    assert [len(periods[k]) for k in ['train', 'search', 'confirm']] == [132802, 8274, 8924]
    record('baseline_inputs_loaded', rows={k: len(v) for k, v in periods.items()}, test_rows=len(test), threads=8)
    with threadpool_limits(limits=8):
        record('rnn_start', epochs=50)
        train_l3.main()
        rnn_metrics = json.loads((train_l3.FOLDER / 'metrics.json').read_text())
        record('rnn_trained', metrics=rnn_metrics)
        levels = {k: stack.predict_saved_rnn3(v) for k, v in periods.items()}
        residual = {k: v.MONTHLY_RENT.to_numpy(float) / v.floor_area_sqm.to_numpy(float) - levels[k] for k, v in periods.items()}
        for key, value in residual.items():
            assert np.isfinite(value).all(), f'Nonfinite real residual {key}'
        store = stack._empty_store(periods, test)
        for k in periods:
            store[k]['mu'][:] = levels[k]
        store['test']['mu'][:] = stack.predict_saved_rnn3(test)
        for town in stack._towns(periods):
            record('residual_town_start', town=town)
            masks = {k: v.TOWN.astype(str).to_numpy() == town for k, v in periods.items()}
            tm = test.TOWN.astype(str).to_numpy() == town
            ft, se, co = [periods[k].loc[masks[k]].reset_index(drop=True) for k in ['train', 'search', 'confirm']]
            tt = test.loc[tm].reset_index(drop=True)
            stack._scale_coords(ft, [se, co, tt])
            yf, ys = (residual['train'][masks['train']], residual['search'][masks['search']])
            model, preparer = stack._catboost(ft, yf, se, ys)
            folder = stack.HERE / 'catboost' / stack.town_slug(town)
            folder.mkdir(parents=True, exist_ok=True)
            stack.save_model(model, folder)
            joblib.dump(preparer, folder / 'preprocessor.joblib')
            if town == 'ang mo kio':
                params = model.get_all_params()
                (OUT / 'catboost_resolved_parameters.json').write_text(json.dumps(params, indent=2), encoding='utf-8')
            mixture = stack._fit_mixture(ft, yf)
            gpath = stack.HERE / 'gmm' / f'{stack.town_slug(town)}.joblib'
            gpath.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(mixture, gpath)
            for name, part, mask in [('train', ft, masks['train']), ('search', se, masks['search']), ('confirm', co, masks['confirm']), ('test', tt, tm)]:
                if not len(part):
                    continue
                stack._write_coords(store, name, mask, part)
                store[name]['t'][mask] = stack.predict_model(model, stack.transform_frame(preparer, part))
                stack._write_mixture(store, name, mask, part, mixture)
            record('residual_town_complete', town=town, trees=int(model.tree_count_), gmm_components=mixture['n_components'])
        residual_scores = {}
        for name in ['search', 'confirm']:
            for key, label in [('t', 'catboost'), ('gmm_gap', 'gmm')]:
                pred = (store[name]['mu'] + store[name][key]) * store[name]['floor_area']
                residual_scores[f'{label}_{name}'] = stack.rmse(periods[name].MONTHLY_RENT.to_numpy(float), pred)
        (OUT / 'residual_metrics.json').write_text(json.dumps(residual_scores, indent=2), encoding='utf-8')
        record('residuals_checked', **residual_scores)
        original_matrix = stack._matrix

        def checked_matrix(*args, **kwargs):
            value = original_matrix(*args, **kwargs)
            matrix = value[0] if kwargs.get('fit', False) else value
            assert np.isfinite(matrix).all() and matrix.shape[1] == 19
            return value
        stack._matrix = checked_matrix
        record('mlp_start', inputs='real outputs from this local baseline fit', threads=8)
        fit_stacked_mlp(stack, store, periods)
        metrics = json.loads((stack.HERE / 'metrics.json').read_text())
        result = {'status': 'completed', 'metrics': metrics, 'rnn': rnn_metrics, 'residuals': residual_scores, 'fresh_weights_only': True, 'cpu_threads': 8, 'scope': 'single historical split; no optimization search', 'duration_seconds': time.time() - START}
        (OUT / 'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        record(result['status'], metrics=metrics, duration_seconds=result['duration_seconds'])
if __name__ == '__main__':
    import argparse as _cli_argparse
    _cli_argparse.ArgumentParser(description=__doc__).parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    START = time.time()
    try:
        execute()
    except BaseException as exc:
        traceback.print_exc()
        record('failed', error=repr(exc))
        raise
