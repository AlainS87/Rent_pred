"""Fresh GPU temporal backtest using the teammate's RNN/residual/GMM/MLP stack."""
from experiments.local.bootstrap import Path, CODE_DIR, DATA_ROOT, SUPPORT_ROOT, ARTIFACT_ROOT, asset_path, asset_key, support_path
import argparse
import hashlib
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from catboost import CatBoostRegressor
from threadpoolctl import threadpool_limits
from experiments.local.common import *
from experiments.local.common import _as_unit_rent, _numeric_block, _numeric_params, _category_columns, _fill_missing_categories
from experiments.local._support.mean_residual_model.reginal_mean_model.rnn.model import TownRNN
RUN = ARTIFACT_ROOT / 'temporal_stack'
DEVICE = 'cuda:0'
THREADS = CONFIG['cpu_threads']
TRAIN = TEST = TIMELINE = None
RNN_CACHE = {}
BASE_CACHE = {}
STARTED = time.time()

def event(stage, **fields):
    record = {'time_utc': datetime.now(timezone.utc).isoformat(), 'pid': os.getpid(), 'stage': stage, **fields}
    print(json.dumps(record, ensure_ascii=False, allow_nan=False), flush=True)
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN / 'events.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
    save_json(RUN / 'status.json', {'status': 'running', **record})

def rent(frame):
    return finite(frame.MONTHLY_RENT.to_numpy(float), 'actual rent')

def make_timeline(train, test):
    covars = pd.concat([train[['TOWN', 'RENT_APPROVAL_DATE', *LEVELS]], test[['TOWN', 'RENT_APPROVAL_DATE', *LEVELS]]])
    return with_month(covars).groupby(['TOWN', 'month'], as_index=False)[['month_index', *LEVELS]].first()

def rnn(cutoff):
    if cutoff in RNN_CACHE:
        return RNN_CACHE[cutoff]
    folder = RUN / 'components' / cutoff / 'rnn'
    folder.mkdir(parents=True, exist_ok=True)
    result_path = folder / 'predictions.csv'
    if (folder / 'complete.json').exists():
        result = pd.read_csv(result_path).set_index(['TOWN', 'month']).prediction
        RNN_CACHE[cutoff] = result
        return result
    fit = before(TRAIN, cutoff)
    cells = cell_table(_as_unit_rent(fit))
    cols = ['month_index', *LEVELS]
    numeric = _numeric_params(cells, cols)
    target = cells.mean_rent.to_numpy(float)
    center, spread = (float(target.mean()), float(target.std()) or 1.0)
    rows = []
    torch.manual_seed(CONFIG['seed'])
    for town, part in cells.groupby('TOWN', sort=True):
        ordered = part.sort_values('month_index')
        x = torch.tensor(finite(_numeric_block(ordered, cols, numeric), 'RNN input'), dtype=torch.float32, device=DEVICE).unsqueeze(0)
        y = torch.tensor((ordered.mean_rent.to_numpy(float) - center) / spread, dtype=torch.float32, device=DEVICE)
        net = TownRNN(3, 4).to(DEVICE)
        opt = torch.optim.Adam(net.parameters(), lr=0.01)
        net.train()
        for epoch in range(CONFIG['rnn_epochs']):
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(net(x).squeeze(0), y)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite RNN loss')
            loss.backward()
            opt.step()
        torch.save({'state': {k: v.detach().cpu() for k, v in net.state_dict().items()}, 'cutoff': cutoff}, folder / f'{stack.town_slug(town)}.pt')
        timeline = TIMELINE.loc[TIMELINE.TOWN == town].sort_values('month_index')
        tx = torch.tensor(finite(_numeric_block(timeline, cols, numeric), 'RNN forecast input'), dtype=torch.float32, device=DEVICE).unsqueeze(0)
        net.eval()
        with torch.no_grad():
            pred = net(tx).squeeze(0).cpu().numpy() * spread + center
        finite(pred, 'RNN actual predictions')
        rows.extend(({'TOWN': town, 'month': m, 'prediction': float(v)} for m, v in zip(timeline.month, pred)))
        event('rnn_town_complete', cutoff=cutoff, town=town, epochs=CONFIG['rnn_epochs'], device=DEVICE, loss=float(loss.detach().cpu()))
    result = pd.DataFrame(rows)
    result.to_csv(result_path, index=False)
    save_json(folder / 'complete.json', {'cutoff': cutoff, 'numeric': numeric, 'target_center': center, 'target_spread': spread, 'fit_rows': len(fit), 'towns': 26})
    output = result.set_index(['TOWN', 'month']).prediction
    RNN_CACHE[cutoff] = output
    return output

def level(series, frame):
    index = pd.MultiIndex.from_arrays([frame.TOWN.astype(str), frame.RENT_APPROVAL_DATE.astype(str).str[:7]], names=['TOWN', 'month'])
    return finite(series.reindex(index).to_numpy(float), 'aligned RNN levels')

def catfit(fit, yfit, valid, yvalid, columns, iterations, select):
    prep = fit_preprocessor(stack.PREPROCESS, fit, finite(yfit, 'real CB residual'), columns)
    xf = transform_frame(prep, fit)
    cats = _category_columns(xf)
    xf = _fill_missing_categories(xf)
    params = dict(iterations=int(iterations), depth=8, learning_rate=0.05, loss_function='RMSE', random_seed=CONFIG['seed'], task_type='GPU', devices='0', thread_count=THREADS, allow_writing_files=False)
    if select:
        params['early_stopping_rounds'] = CONFIG['catboost_patience']
    net = CatBoostRegressor(**params)
    if select:
        xv = _fill_missing_categories(transform_frame(prep, valid))
        net.fit(xf, yfit, cat_features=cats, eval_set=(xv, yvalid), use_best_model=True, verbose=False)
    else:
        net.fit(xf, yfit, cat_features=cats, verbose=False)
    return (net, prep)

def base(cutoff):
    if cutoff in BASE_CACHE:
        return BASE_CACHE[cutoff]
    folder = RUN / 'components' / cutoff / 'base'
    folder.mkdir(parents=True, exist_ok=True)
    level_final = rnn(cutoff)
    tune_end = month_add(cutoff, -3)
    level_tune = rnn(tune_end)
    fit_raw = before(TRAIN, cutoff)
    (fit,), ranking = ranked_frames(fit_raw, [])
    tune_raw, valid_raw = (before(TRAIN, tune_end), interval(TRAIN, tune_end, cutoff))
    (tune, valid), _ = ranked_frames(tune_raw, [valid_raw])
    models = {}
    for town in sorted(fit.TOWN.unique()):
        path = folder / f'{stack.town_slug(town)}.joblib'
        if path.exists():
            models[town] = joblib.load(path)
            continue
        ft = fit.loc[fit.TOWN == town].copy().reset_index(drop=True)
        tt = tune.loc[tune.TOWN == town].copy().reset_index(drop=True)
        vv = valid.loc[valid.TOWN == town].copy().reset_index(drop=True)
        if not len(tt) or not len(vv):
            raise ValueError(f'Town missing in inner CB split: {cutoff} {town}')
        coord = {c: {'center': float(ft[c].mean()), 'spread': float(ft[c].std(ddof=0)) or 1.0} for c in ['latitude', 'longitude']}
        stack._scale_coords(ft, [])
        stack._scale_coords(tt, [vv])
        yf = finite(rent(ft) / ft.floor_area_sqm.to_numpy(float) - level(level_final, ft), 'full residual')
        yt = finite(rent(tt) / tt.floor_area_sqm.to_numpy(float) - level(level_tune, tt), 'tune residual')
        yv = finite(rent(vv) / vv.floor_area_sqm.to_numpy(float) - level(level_tune, vv), 'inner residual')
        event('gmm_start', cutoff=cutoff, town=town, rows=len(ft), cpu_threads=THREADS)
        with threadpool_limits(limits=THREADS):
            mixture = stack._fit_mixture(ft, yf)
        entry = {'coord': coord, 'mixture': mixture, 'cat': {}}
        for name, cols in FEATURES.items():
            event('catboost_selection_start', cutoff=cutoff, town=town, variant=name, rows=len(tt), device='GPU')
            selected, _ = catfit(tt, yt, vv, yv, cols, CONFIG['catboost_max_iterations'], True)
            trees = max(1, int(selected.tree_count_))
            del selected
            event('catboost_refit_start', cutoff=cutoff, town=town, variant=name, rows=len(ft), iterations=trees, device='GPU')
            net, prep = catfit(ft, yf, None, None, cols, trees, False)
            entry['cat'][name] = {'model': net, 'preparer': prep, 'trees': trees}
        temp = path.with_suffix('.tmp')
        joblib.dump(entry, temp)
        temp.replace(path)
        models[town] = entry
        event('base_town_complete', cutoff=cutoff, town=town, rows=len(ft), gmm_components=mixture['n_components'])
    result = {'cutoff': cutoff, 'ranking': ranking, 'models': models, 'level': level_final}
    save_json(folder / 'complete.json', {'cutoff': cutoff, 'fit_rows': len(fit), 'towns': 26, 'ranking': ranking})
    BASE_CACHE[cutoff] = result
    return result

def predict_block(bundle, frame, feature_variant):
    p = frame.copy()
    ranking = bundle['ranking']
    p['flat_model_ord'] = p.flat_model_key.map(ranking).fillna(float(np.median(list(ranking.values())))).astype(float)
    block = plain_block(p)
    block['mu'] = level(bundle['level'], p)
    block['t'] = np.full(len(p), np.nan)
    block['gmm_gap'] = np.full(len(p), np.nan)
    for town, item in bundle['models'].items():
        mask = (p.TOWN == town).to_numpy()
        if not mask.any():
            continue
        part = p.loc[mask].copy()
        for col, stats in item['coord'].items():
            part[col] = (part[col].astype(float) - stats['center']) / stats['spread']
        stack._write_coords({'apply': block}, 'apply', mask, part)
        cb = item['cat'][feature_variant]
        block['t'][mask] = stack.predict_model(cb['model'], transform_frame(cb['preparer'], part))
        stack._write_mixture({'apply': block}, 'apply', mask, part, item['mixture'])
    for key in ['mu', 't', 'gmm_gap']:
        finite(block[key], f'real component {key}')
    return block

def stack_training(end, variant):
    fv = 'original' if variant == 'original_stack' else 'expanded'
    if variant != 'expanded_temporal_stack':
        data = before(TRAIN, end)
        return (predict_block(base(end), data, fv), rent(data))
    blocks, ys = ([], [])
    for delta in [-12, -8, -4]:
        start = month_add(end, delta)
        finish = month_add(start, 4)
        data = interval(TRAIN, start, finish)
        blocks.append(predict_block(base(start), data, fv))
        ys.append(rent(data))
        event('temporal_stack_block_ready', model_cutoff=start, prediction_end=finish, rows=len(data))
    return ({key: np.concatenate([b[key] for b in blocks]) for key in blocks[0]}, np.concatenate(ys))

def fit_mlp(block, y, valid_block=None, valid_y=None, epochs=None, tag=''):
    x, center, spread = stack._matrix(block, fit=True)
    finite(x, 'real 19-column MLP fit matrix')
    finite(center, 'MLP fit medians')
    finite(spread, 'MLP fit scales')
    yc, ys = (float(y.mean()), float(y.std()) or 1.0)
    torch.manual_seed(CONFIG['seed'])
    net = stack._MLP(19).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=0.001)
    tx = torch.tensor(x, dtype=torch.float32, device=DEVICE)
    ty = torch.tensor((y - yc) / ys, dtype=torch.float32, device=DEVICE)
    vx = None
    if valid_block is not None:
        vx = torch.tensor(finite(stack._matrix(valid_block, center=center, spread=spread), 'real 19-column MLP validation matrix'), dtype=torch.float32, device=DEVICE)
    event('mlp_real_inputs_passed', tag=tag, train_rows=len(y), columns=19, validation_rows=0 if valid_y is None else len(valid_y), device=DEVICE)
    generator = torch.Generator().manual_seed(CONFIG['seed'])
    best, epoch_best, state, stall = (float('inf'), 0, None, 0)
    for epoch in range(1, (epochs or CONFIG['mlp_max_epochs']) + 1):
        net.train()
        order = torch.randperm(len(y), generator=generator).to(DEVICE)
        for start in range(0, len(y), 1024):
            ids = order[start:start + 1024]
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(net(tx[ids]), ty[ids])
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite MLP loss')
            loss.backward()
            opt.step()
        if vx is not None:
            net.eval()
            with torch.no_grad():
                p = net(vx).cpu().numpy() * ys + yc
            score = float(np.sqrt(np.mean((valid_y - finite(p, 'MLP inner predictions')) ** 2)))
            if score < best - 1e-06:
                best, epoch_best, stall = (score, epoch, 0)
                state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            else:
                stall += 1
            if epoch == 1 or epoch % 10 == 0:
                event('mlp_inner_epoch', tag=tag, epoch=epoch, rmse=score, best_epoch=epoch_best)
            if stall >= CONFIG['mlp_patience']:
                break
        else:
            epoch_best = epoch
    if state is not None:
        net.load_state_dict(state)
    payload = {'state': {k: v.detach().cpu() for k, v in net.state_dict().items()}, 'center': center, 'spread': spread, 'y_center': yc, 'y_spread': ys, 'epoch': epoch_best, 'inner_rmse': best if state is not None else None}
    return payload

def mlp_predict(payload, block):
    x = finite(stack._matrix(block, center=payload['center'], spread=payload['spread']), 'MLP inference matrix')
    net = stack._MLP(19).to(DEVICE)
    net.load_state_dict(payload['state'])
    net.eval()
    with torch.no_grad():
        p = net(torch.tensor(x, dtype=torch.float32, device=DEVICE)).cpu().numpy()
    return finite(p * payload['y_spread'] + payload['y_center'], 'MLP monthly rent prediction')

def model_at(origin, variant):
    folder = RUN / 'stacks' / origin / variant
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'model.pt'
    if path.exists():
        return torch.load(path, map_location='cpu', weights_only=False)
    inner_end = month_add(origin, -3)
    train_block, y = stack_training(inner_end, variant)
    valid = interval(TRAIN, inner_end, origin)
    fv = 'original' if variant == 'original_stack' else 'expanded'
    valid_block = predict_block(base(inner_end), valid, fv)
    selected = fit_mlp(train_block, y, valid_block, rent(valid), tag=f'{origin}/{variant}/select')
    save_json(folder / 'selection.json', {'inner_fit_end': inner_end, 'inner_validation_end': origin, 'epoch': selected['epoch'], 'inner_rmse': selected['inner_rmse']})
    full_block, full_y = stack_training(origin, variant)
    result = fit_mlp(full_block, full_y, epochs=selected['epoch'], tag=f'{origin}/{variant}/refit')
    tmp = path.with_suffix('.tmp')
    torch.save(result, tmp)
    tmp.replace(path)
    event('stack_complete', origin=origin, variant=variant, fit_rows=len(full_y), epochs=result['epoch'])
    return result

def metrics(frame, pred, origin):
    err = pred - rent(frame)
    horizons = frame.RENT_APPROVAL_DATE.astype(str).str[:7].map(lambda m: pd.Period(m, 'M').ordinal - pd.Period(origin, 'M').ordinal).to_numpy()

    def summarize(mask):
        e = err[mask]
        return {'rows': len(e), 'rmse': float(np.sqrt(np.mean(e ** 2))), 'bias': float(e.mean()), 'sse': float(np.sum(e ** 2))}
    out = summarize(np.ones(len(err), dtype=bool))
    out['horizons'] = {}
    for lo, hi in [(1, 3), (4, 6), (7, 12), (13, 15)]:
        mask = (horizons >= lo) & (horizons <= hi)
        if mask.any():
            out['horizons'][f'{lo}-{hi}'] = summarize(mask)
    out['towns'] = {str(t): summarize((frame.TOWN == t).to_numpy()) for t in sorted(frame.TOWN.unique())}
    return (out, horizons)

def evaluate(name):
    settings = CONFIG[name]
    origin, end = (settings['origin'], settings['end'])
    data = interval(TRAIN, origin, end)
    scores = {}
    for variant in CONFIG['variants']:
        model = model_at(origin, variant)
        block = predict_block(base(origin), data, 'original' if variant == 'original_stack' else 'expanded')
        pred = mlp_predict(model, block)
        summary, horizons = metrics(data, pred, origin)
        scores[variant] = summary
        rows = data[['row_id', 'RENT_APPROVAL_DATE', 'TOWN', 'MONTHLY_RENT']].copy()
        rows['origin'], rows['horizon'], rows['prediction'] = (origin, horizons, pred)
        rows['rnn_plus_cb'] = (block['mu'] + block['t']) * block['floor_area']
        rows['rnn_plus_gmm'] = (block['mu'] + block['gmm_gap']) * block['floor_area']
        rows.to_csv(RUN / f'{name}_{variant}_predictions.csv', index=False)
        event('evaluation_variant_complete', split=name, origin=origin, variant=variant, rmse=summary['rmse'], rows=len(data))
        save_json(RUN / f'{name}_metrics.json', scores)
    return scores

def execute():
    global TRAIN, TEST, TIMELINE
    RUN.mkdir(parents=True, exist_ok=True)
    prep = json.loads(asset_path(Path('preparation.json')).read_text())
    digest = hashlib.sha256(asset_path(Path('protocol.json')).read_bytes()).hexdigest()
    assert prep['status'] == 'passed_observed_inputs' and prep['protocol_sha256'] == digest
    for split in ['train', 'test']:
        current = hashlib.sha256(support_path(Path('Data/auxiliary_features') / f'{split}_features.csv').read_bytes()).hexdigest()
        assert current == prep['csv_sha256'][split], 'Prepared feature CSV changed after acceptance'
    assert torch.cuda.is_available(), 'CUDA required; do not silently fall back to CPU'
    torch.set_num_threads(THREADS)
    torch.set_num_interop_threads(2)
    manifest = {'protocol_sha256': digest, 'run_name': RUN.name, 'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'adapter_sha256': hashlib.sha256(asset_path(Path('common.py')).read_bytes()).hexdigest()}
    if (RUN / 'manifest.json').exists():
        assert json.loads((RUN / 'manifest.json').read_text()) == manifest, 'Resume source/protocol mismatch'
    else:
        assert not (RUN / 'components').exists(), 'Refuse unknown existing models'
        save_json(RUN / 'manifest.json', manifest)
    import catboost, sklearn, scipy
    save_json(RUN / 'runtime.json', {'python': sys.version, 'executable': sys.executable, 'torch': torch.__version__, 'catboost': catboost.__version__, 'pandas': pd.__version__, 'numpy': np.__version__, 'sklearn': sklearn.__version__, 'scipy': scipy.__version__, 'gpu': torch.cuda.get_device_name(0), 'cpu_threads': THREADS})
    TRAIN, TEST = load_data()
    TIMELINE = make_timeline(TRAIN, TEST)
    for cutoff in cutoffs_needed():
        altered = TRAIN.copy()
        mask = altered.RENT_APPROVAL_DATE.astype(str).str[:7] > cutoff
        altered.loc[mask, 'MONTHLY_RENT'] = altered.loc[mask, 'MONTHLY_RENT'] * 1000 + 999999
        pd.testing.assert_frame_equal(before(TRAIN, cutoff), before(altered, cutoff))
    save_json(RUN / 'causality_checks.json', {'future_label_perturbation': 'passed for every fit cutoff', 'fit_cutoffs': cutoffs_needed(), 'rnn_timeline_contains_target': False, 'group_mean_features_used': False})
    event('training_start', train_rows=len(TRAIN), test_rows=len(TEST), gpu=torch.cuda.get_device_name(0))
    dev_origin = CONFIG['development']['origin']
    model_at(dev_origin, 'original_stack')
    event('first_full_stack_complete', origin=dev_origin)
    development = evaluate('development')
    winner = min(CONFIG['variants'], key=lambda v: development[v]['rmse'])
    save_json(RUN / 'selection_frozen.json', {'chosen_variant': winner, 'reason': 'lowest development RMSE', 'development_rmse': development[winner]['rmse'], 'evaluation_not_used': True})
    evaluation = evaluate('evaluation')
    result = {'status': 'completed', 'chosen_on_development': winner, 'development': development, 'evaluation': evaluation, 'elapsed_seconds': time.time() - STARTED, 'window_overlap': CONFIG['overlap']}
    save_json(RUN / 'results.json', result)
    event('completed', chosen_variant=winner, elapsed_seconds=result['elapsed_seconds'])
    save_json(RUN / 'status.json', {'status': 'completed', 'pid': os.getpid(), 'chosen_variant': winner, 'exit_code': 0})
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, default=RUN)
    args = parser.parse_args()
    RUN = args.run_dir.resolve()
    try:
        execute()
    except BaseException as exc:
        traceback.print_exc()
        save_json(RUN / 'status.json', {'status': 'failed', 'pid': os.getpid(), 'error': repr(exc), 'exit_code': 1, 'time_utc': datetime.now(timezone.utc).isoformat()})
        raise
