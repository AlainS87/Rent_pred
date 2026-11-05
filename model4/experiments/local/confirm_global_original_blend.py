"""Freeze and evaluate one equal-weight blend; no fitting or weight adjustment."""
from experiments.local.bootstrap import Path, CODE_DIR, DATA_ROOT, SUPPORT_ROOT, ARTIFACT_ROOT, asset_path, asset_key, support_path
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback
from experiments.local.global_monthly_catboost import read, digest, check_frozen
from experiments.local.spatial_history_io import append_json, write_json
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
OUT = asset_path(Path('global_original_blend_50_50_01'))
GLOBAL = asset_path(Path('global_monthly_ctr8_01'))
BASELINE = asset_path(Path('baseline_alignment'))
WEIGHTS = {'global_ctr8': 0.5, 'original_local_stack': 0.5}
LAST_STAGE = 'not_started'

def record(stage, **fields):
    global LAST_STAGE
    value = {'stage': stage, 'pid': os.getpid(), 'epoch': time.time(), **fields}
    append_json(OUT / 'events.jsonl', value)
    write_json(OUT / 'status.json', value, required=False)
    LAST_STAGE = stage
    print(json.dumps(value), flush=True)

def prepare():
    import numpy as np
    import pandas as pd
    if OUT.exists():
        raise RuntimeError('Refusing to overwrite this one-shot evaluation')
    prior = read(GLOBAL / 'results.json')
    prior_protocol = read(GLOBAL / 'protocol.json')
    check_frozen(prior_protocol)
    current = pd.read_csv(GLOBAL / 'search_predictions.csv')
    local = pd.read_csv(asset_path(Path('feature_ablation_01/baseline_search_predictions.csv')))
    assert np.array_equal(current.source_row_id, np.arange(132802, 141076))
    assert np.array_equal(current.source_row_id, local.source_row_id)
    assert np.array_equal(current.truth, local.truth)
    blend = WEIGHTS['global_ctr8'] * current.prediction.to_numpy() + WEIGHTS['original_local_stack'] * local.prediction.to_numpy()
    assert np.isfinite(blend).all()
    score = float(np.sqrt(np.mean((blend - current.truth.to_numpy()) ** 2)))
    assert abs(score - 442.9406238420111) < 1e-09
    assert score < prior['search']['rmse'] == 445.3804031850469
    frozen = dict(prior_protocol['frozen_sha256'])
    for filename in ['protocol.json', 'results.json', 'selection_frozen.json', 'model.joblib', 'preprocessor.joblib', 'search_predictions.csv', 'confirm_predictions.csv', 'search_metrics.json', 'confirmation_metrics.json', 'completion_verification.json', 'diagnosis/combination_feasibility.json']:
        path = GLOBAL / filename
        frozen[asset_key(path).as_posix()] = digest(path)
    for name in ['feature_ablation.py', 'confirm_global_original_blend.py', 'verify_global_original_blend.py', 'spatial_history_io.py']:
        frozen[name] = digest(asset_path(Path(name)))
    OUT.mkdir(exist_ok=False)
    table = pd.DataFrame({'source_row_id': current.source_row_id, 'date': current.date, 'truth': current.truth, 'global_prediction': current.prediction, 'original_prediction': local.prediction, 'prediction': blend})
    table.to_csv(OUT / 'search_predictions.csv', index=False)
    search_metrics = {'rows': len(table), 'rmse': score, 'delta_from_current_global': score - prior['search']['rmse'], 'delta_from_local_baseline': score - prior['baseline']['search_rmse']}
    write_json(OUT / 'search_metrics.json', search_metrics)
    protocol = {'status': 'frozen_before_confirm', 'candidate': 'global_ctr8_and_original_local_stack_equal_weight', 'weights': WEIGHTS, 'weights_tunable': False, 'selection_basis': 'Previously computed Search score; user explicitly chose this 50/50 blend', 'fit_end': '2024-09', 'search_months': ['2024-10', '2024-11', '2024-12'], 'confirm_months': ['2025-01', '2025-02', '2025-03'], 'rows': {'fit': 132802, 'search': 8274, 'confirm': 8924}, 'current_global': {'search_rmse': prior['search']['rmse'], 'confirm_rmse': prior['confirm']['rmse']}, 'local_baseline': prior['baseline'], 'original_component': 'Fixed local baseline_alignment RNN/GMM/CatBoost/MLP and saved normalizers; not the area variant', 'global_component': 'Reuse immutable global_monthly_ctr8_01 saved predictions; no second global inference', 'training_allowed': False, 'candidate_confirm_evaluations_max': 1, 'automatic_retry': False, 'confirm_weight_adjustment_or_reselection': False, 'confirm_interpretation': 'Original split iterative validation, not a new untouched test set', 'criterion': 'Report whether both Search and Confirm strictly improve over current global CTR8; if Confirm regresses, report improvement not established without changing weights', 'frozen_sha256': frozen, 'frozen_epoch': time.time(), 'search_predictions_sha256': digest(OUT / 'search_predictions.csv'), 'search_metrics_sha256': digest(OUT / 'search_metrics.json')}
    write_json(OUT / 'protocol.json', protocol)
    selection = {'candidate': protocol['candidate'], 'weights': WEIGHTS, 'search_rmse': score, 'search_gate_rmse': prior['search']['rmse'], 'passed_search_gate': True, 'protocol_sha256': digest(OUT / 'protocol.json'), 'epoch': time.time(), 'candidate_confirm_evaluations_before_freeze': 0, 'search_predictions_sha256': protocol['search_predictions_sha256']}
    write_json(OUT / 'selection_frozen.json', selection)
    record('selection_frozen', search_rmse=score, weights=WEIGHTS, training_performed=False)

def run():
    started = time.time()
    protocol = read(OUT / 'protocol.json')
    selection = read(OUT / 'selection_frozen.json')
    assert protocol['weights'] == selection['weights'] == WEIGHTS
    assert digest(OUT / 'protocol.json') == selection['protocol_sha256']
    selection_hash = digest(OUT / 'selection_frozen.json')
    assert selection['passed_search_gate'] and selection['search_rmse'] < selection['search_gate_rmse']
    check_frozen(protocol)
    with (OUT / 'confirm_started.json').open('x', encoding='utf-8') as handle:
        json.dump({'epoch': time.time(), 'pid': os.getpid(), 'weights': WEIGHTS, 'selection_sha256': selection_hash}, handle, indent=2)
    record('confirm_evaluation_started', candidate_confirm_evaluations=1, weights=WEIGHTS)
    from experiments.local import feature_ablation as experiment
    stack = experiment.runtime()
    import numpy as np
    import pandas as pd
    import torch
    from catboost import CatBoostRegressor
    from sklearn.mixture import GaussianMixture
    from threadpoolctl import threadpool_limits

    def forbidden(*args, **kwargs):
        raise RuntimeError('Training or preprocessor fitting is forbidden for this frozen blend')
    CatBoostRegressor.fit = forbidden
    GaussianMixture.fit = forbidden
    for name in ['_fit_mlp', '_fit_mixture', '_fit_town', '_catboost', 'fit_model', 'fit_preprocessor']:
        setattr(stack, name, forbidden)
    torch.optim.Adam.step = forbidden
    with threadpool_limits(limits=8), torch.inference_mode():
        frame, split, towns = experiment.inputs(stack)
        assert split.fit_months[-1] == protocol['fit_end']
        assert split.search_months == protocol['search_months']
        assert split.confirm_months == protocol['confirm_months']
        ids = np.flatnonzero(frame.RENT_APPROVAL_DATE.astype(str).str[:7].isin(protocol['confirm_months']).to_numpy())
        assert np.array_equal(ids, np.arange(141076, 150000))
        confirm = split.confirm.drop(columns='MONTHLY_RENT')
        fit = split.fit.drop(columns='MONTHLY_RENT')
        parts = {'confirm': confirm}
        store = stack._empty_store(parts, confirm.iloc[:0])
        store.pop('test')
        global_saved = pd.read_csv(GLOBAL / 'confirm_predictions.csv')
        assert np.array_equal(global_saved.source_row_id, ids)
        assert np.array_equal(global_saved.date.astype(str), split.confirm.RENT_APPROVAL_DATE.astype(str))
        record('original_component_inference_started', original_stack_inference_number=1, training_performed=False)
        experiment.model_predictions(stack, BASELINE / 'stack', store, parts, fit, towns)
        local_prediction = experiment.saved_mlp_predictions(stack, BASELINE / 'stack', store['confirm'])
        truth = split.confirm.MONTHLY_RENT.to_numpy(float)
        assert np.array_equal(global_saved.truth.to_numpy(), truth)
        assert np.isfinite(local_prediction).all() and len(local_prediction) == len(truth) == 8924
        global_prediction = global_saved.prediction.to_numpy(float)
        prediction = WEIGHTS['global_ctr8'] * global_prediction + WEIGHTS['original_local_stack'] * local_prediction
        assert np.isfinite(prediction).all()
        table = pd.DataFrame({'source_row_id': ids, 'date': split.confirm.RENT_APPROVAL_DATE.to_numpy(), 'truth': truth, 'global_prediction': global_prediction, 'original_prediction': local_prediction, 'prediction': prediction})
        table.to_csv(OUT / 'confirm_predictions.csv', index=False)
        local_score = float(np.sqrt(np.mean((local_prediction - truth) ** 2)))
        global_score = float(np.sqrt(np.mean((global_prediction - truth) ** 2)))
        error = prediction - truth
        score = float(np.sqrt(np.mean(error ** 2)))
        assert abs(local_score - protocol['local_baseline']['confirm_rmse']) < 1e-08
        assert abs(global_score - protocol['current_global']['confirm_rmse']) < 1e-09
        months = table.date.astype(str).str[:7].to_numpy()
        confirmation = {'rows': len(table), 'rmse': score, 'mae': float(np.mean(np.abs(error))), 'bias_pred_minus_truth': float(error.mean()), 'original_component_rmse': local_score, 'global_component_rmse': global_score, 'delta_from_current_global': score - global_score, 'delta_from_local_baseline': score - local_score, 'by_month': {month: {'rows': int((months == month).sum()), 'rmse': float(np.sqrt(np.mean(error[months == month] ** 2))), 'global_rmse': float(np.sqrt(np.mean((global_prediction[months == month] - truth[months == month]) ** 2)))} for month in protocol['confirm_months']}}
        write_json(OUT / 'confirmation_metrics.json', confirmation)
    check_frozen(protocol)
    assert digest(OUT / 'protocol.json') == selection['protocol_sha256']
    assert digest(OUT / 'selection_frozen.json') == selection_hash
    assert digest(OUT / 'search_predictions.csv') == protocol['search_predictions_sha256']
    assert digest(OUT / 'search_metrics.json') == protocol['search_metrics_sha256']
    improved = score < protocol['current_global']['confirm_rmse']
    result = {'status': 'completed', 'exit_code': 0, 'candidate': protocol['candidate'], 'weights': WEIGHTS, 'search': read(OUT / 'search_metrics.json'), 'confirm': confirmation, 'current_global': protocol['current_global'], 'local_baseline': protocol['local_baseline'], 'improved_both_periods_over_current_global': improved, 'conclusion': 'Improved both original periods versus current global model; iterative validation only.' if improved else 'Search improved but Confirm did not improve over current global model. Optimization not established; weights unchanged and no reselection.', 'candidate_confirm_evaluations': 1, 'original_component_confirm_inferences': 1, 'global_component_confirm_inferences': 0, 'training_performed': False, 'weight_tuning_performed': False, 'source_baseline_labels_split_and_model_hashes_unchanged': True, 'protocol_sha256': selection['protocol_sha256'], 'selection_sha256': selection_hash, 'search_predictions_csv': str((OUT / 'search_predictions.csv').resolve()), 'confirm_predictions_csv': str((OUT / 'confirm_predictions.csv').resolve()), 'confirm_predictions_sha256': digest(OUT / 'confirm_predictions.csv'), 'elapsed_seconds': time.time() - started}
    write_json(OUT / 'results.json', result)
    record('completed', confirm_rmse=score, improved_both_periods=improved, exit_code=0, elapsed_seconds=result['elapsed_seconds'])
    print(json.dumps(result, indent=2), flush=True)
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
            write_json(OUT / 'failure_report.json', {'status': 'failed', 'error': repr(exc), 'last_successful_stage': prior_stage, 'pid': os.getpid(), 'exit_code': 1, 'epoch': time.time()}, required=False)
            record('failed', error=repr(exc), last_successful_stage=prior_stage, exit_code=1)
        raise
