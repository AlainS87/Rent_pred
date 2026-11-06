"""Authorized one-parameter ablation of the frozen global monthly CatBoost."""
from experiments.local.bootstrap import Path, CODE_DIR, DATA_ROOT, SUPPORT_ROOT, ARTIFACT_ROOT, asset_path, asset_key, support_path
import argparse
import json
from pathlib import Path
import time
import traceback
from experiments.local import global_monthly_catboost as original
from experiments.local.spatial_history_io import write_json
ROOT = original.ROOT
PRIOR = asset_path(Path('global_monthly_01'))
OUT = asset_path(Path('global_monthly_ctr8_01'))
FEATURES = list(original.FEATURES)
PREPROCESS = dict(original.PREPROCESS)
ORIGINAL_PARAMS = dict(original.PARAMS)
PARAMS = {**ORIGINAL_PARAMS, 'ctr_target_border_count': 8}
read, digest = (original.read, original.digest)
check_frozen = original.check_frozen
original.OUT = OUT
original.PARAMS = PARAMS
BASE_CHECK = original.check_original_constants

def check_constants():
    BASE_CHECK()
    prior = read(PRIOR / 'protocol.json')
    assert FEATURES == prior['features'] and PREPROCESS == prior['preprocess']
    assert ORIGINAL_PARAMS == prior['params']
    assert PARAMS == {**prior['params'], 'ctr_target_border_count': 8}
    assert read(PRIOR / 'fit_audit.json')['resolved_params']['ctr_target_border_count'] == 1

def validate_fit_audit(audit):
    previous = read(PRIOR / 'fit_audit.json')
    assert audit['params'] == PARAMS
    assert audit['features'] == previous['features']
    assert audit['categorical_indices'] == previous['categorical_indices']
    assert audit['training_weights'] == audit['early_stopping_weights'] == 'none'
    assert audit['target'] == 'MONTHLY_RENT' and audit['fit_calls'] == 1
    resolved = audit['resolved_params']
    expected = dict(previous['resolved_params'])
    expected['ctr_target_border_count'] = 8
    for name in ('simple_ctr', 'combinations_ctr'):
        expected[name] = [item.replace('TargetBorderCount=1:', 'TargetBorderCount=8:') for item in expected[name]]
    if resolved != expected:
        changes = {key: {'expected': expected.get(key), 'actual': resolved.get(key)} for key in set(expected) | set(resolved) if expected.get(key) != resolved.get(key)}
        raise AssertionError(f'Unexpected resolved parameter changes: {changes}')
    return {'constructor_only_change': 'ctr_target_border_count: 1 -> 8', 'resolved_changes': [key for key in expected if expected[key] != previous['resolved_params'][key]], 'other_resolved_parameters_exactly_unchanged': True}

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
    masks = {'prior_global_worst_10pct': worst, 'other_90pct': ~worst, 'prior_global_overprediction_ge_1000': old_error >= 1000, 'truth_at_most_1500': previous.truth.to_numpy() <= 1500}
    return {name: {'rows': int(mask.sum()), 'prior_global_rmse': float(np.sqrt(np.mean(old_error[mask] ** 2))), 'candidate_rmse': float(np.sqrt(np.mean(new_error[mask] ** 2)))} for name, mask in masks.items()}

def checked_write(path, value, **kwargs):
    path = Path(path)
    if path.parent == OUT and path.name == 'fit_audit.json':
        value['one_parameter_change_verification'] = validate_fit_audit(value)
    elif path.parent == OUT and path.name == 'search_metrics.json':
        value['fixed_prior_error_groups'] = fixed_search_diagnostics()
    return write_json(path, value, **kwargs)
original.check_original_constants = check_constants
original.write_json = checked_write

def prepare():
    check_constants()
    if OUT.exists():
        raise RuntimeError('Refusing to overwrite or restart this candidate')
    previous = read(PRIOR / 'results.json')
    assert previous['status'] == 'completed'
    assert previous['search']['rmse'] == 447.7752932883708
    assert previous['confirm']['rmse'] == 455.21281745811854
    old_protocol = read(PRIOR / 'protocol.json')
    assert digest(PRIOR / 'protocol.json') == previous['protocol_sha256']
    frozen = dict(old_protocol['frozen_sha256'])
    for filename in ['protocol.json', 'results.json', 'selection_frozen.json', 'fit_audit.json', 'data_audit.json', 'model.joblib', 'preprocessor.joblib', 'search_predictions.csv', 'confirm_predictions.csv', 'search_metrics.json', 'confirmation_metrics.json', 'completion_verification.json', 'diagnosis/next_step_proposal.json']:
        name = 'global_monthly_01/' + filename
        frozen[name] = digest(asset_path(Path(name)))
    for name in ['global_monthly_ctr8.py', 'verify_global_monthly_ctr8_result.py']:
        frozen[name] = digest(asset_path(Path(name)))
    protocol = {**old_protocol, 'created_epoch': time.time(), 'candidate': 'global_monthly_catboost_ctr_target_border_count_8', 'params': PARAMS, 'frozen_sha256': frozen, 'search_gate_rmse': previous['search']['rmse'], 'previous_best': {'search_rmse': previous['search']['rmse'], 'confirm_rmse': previous['confirm']['rmse']}, 'one_parameter_change': {'ctr_target_border_count': {'previous': 1, 'candidate': 8}}, 'supplementary_diagnostics': 'Fixed previous global model worst 10%, other 90%, prior overprediction>=1000, truth<=1500; reporting only, not alternate selection rules', 'authorization': 'User approved the previously proposed single ctr_target_border_count=8 candidate; one fit, CPU8, 900 seconds, no retry'}
    OUT.mkdir()
    write_json(OUT / 'protocol.json', protocol)
    print(json.dumps({'prepared': str(OUT), 'features': len(FEATURES), 'frozen_files': len(frozen), 'only_parameter_change': protocol['one_parameter_change'], 'training_started': False}), flush=True)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['prepare', 'run'])
    args = parser.parse_args()
    try:
        prepare() if args.phase == 'prepare' else original.run()
    except BaseException as exc:
        traceback.print_exc()
        if args.phase == 'run' and OUT.exists():
            prior_stage = original.LAST_STAGE
            write_json(OUT / 'failure_report.json', {'status': 'failed', 'error': repr(exc), 'last_successful_stage': prior_stage, 'epoch': time.time(), 'exit_code': 1}, required=False)
            original.record('failed', error=repr(exc), last_successful_stage=prior_stage)
        raise
