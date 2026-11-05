"""One authorized Confirm evaluation of frozen baseline MLP plus area CatBoost.

No fitting, parameter updates, model selection, other candidate evaluations,
external publication/test predictions, or writes to any source model artifact.
"""
from experiments.local.bootstrap import Path, CODE_DIR, DATA_ROOT, SUPPORT_ROOT, ARTIFACT_ROOT, asset_path, asset_key, support_path
import json
import os
from pathlib import Path
import sys
import time
import traceback
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
OUT = asset_path(Path('frozen_mlp_area_confirm_01'))
ABLATION = asset_path(Path('feature_ablation_01'))
BASELINE = asset_path(Path('baseline_alignment'))

def main():
    from experiments.local import feature_ablation as experiment
    stack = experiment.runtime()
    import numpy as np
    import pandas as pd
    import torch
    from catboost import CatBoostRegressor
    from sklearn.mixture import GaussianMixture
    from threadpoolctl import threadpool_limits

    def forbidden(*args, **kwargs):
        raise RuntimeError('Training is forbidden in this Confirm-only evaluation')
    CatBoostRegressor.fit = forbidden
    GaussianMixture.fit = forbidden
    stack._fit_mlp = forbidden
    stack._fit_mixture = forbidden
    stack._catboost = forbidden
    stack.fit_model = forbidden
    stack.fit_preprocessor = forbidden
    torch.optim.Adam.step = forbidden
    original_protocol = experiment.verify_frozen()
    original_selection = experiment.read(ABLATION / 'selection_frozen.json')
    assert experiment.digest(ABLATION / 'protocol.json') == original_selection['protocol_sha256']
    analysis_path = ABLATION / 'diagnosis/analysis.json'
    analysis = experiment.read(analysis_path)
    assert analysis['training_performed'] is False and analysis['confirm_accessed'] is False
    known_search = analysis['frozen_mlp_cross_input_metrics']['baseline']['area_only']['search']['rmse']
    assert abs(known_search - 451.042648255612) < 1e-09
    baseline_metrics = original_protocol['baseline']
    area_folder = ABLATION / 'area_only'
    files = list((area_folder / 'catboost').rglob('*.joblib'))
    assert len(files) == 52
    files += [BASELINE / 'stack/mlp/model.pt', ABLATION / 'protocol.json', ABLATION / 'selection_frozen.json', ABLATION / 'results.json', analysis_path, Path(__file__).resolve()]
    frozen = {asset_key(p).as_posix(): experiment.digest(p) for p in files}
    OUT.mkdir(exist_ok=False)
    started = time.time()
    experiment.save(OUT / 'status.json', {'stage': 'freezing_protocol', 'pid': os.getpid(), 'epoch': started})
    protocol = {'status': 'frozen_before_confirm_prediction', 'chosen_combination': 'baseline_alignment MLP/normalizers + area_only CatBoost + baseline_alignment RNN/GMM', 'authorization': 'User explicitly requested this combination and one original Confirm evaluation, without training or changing the baseline/split.', 'fit_end': '2024-09', 'search_months': ['2024-10', '2024-11', '2024-12'], 'confirm_months': ['2025-01', '2025-02', '2025-03'], 'rows': {'fit': 132802, 'search': 8274, 'confirm': 8924}, 'search_rmse_from_previous_diagnostic': known_search, 'baseline_metrics': baseline_metrics, 'mlp_checkpoint': str(BASELINE / 'stack/mlp/model.pt'), 'mlp_normalizers': 'Use x_center, x_spread, y_center, y_spread unchanged from that baseline checkpoint.', 'catboost_directory': str(area_folder / 'catboost'), 'preprocessing': 'Original exported loader, split, fit-period coordinate scaling, saved area preprocessor transform, original 19-column MLP matrix.', 'criterion': 'Search and Confirm RMSE must both be strictly lower than the unchanged local baseline; otherwise report not established. Do not tune or reselect after Confirm.', 'candidate_confirm_evaluations': 1, 'training_allowed': False, 'automatic_retries': 0, 'test_prediction_requested': False, 'support_and_baseline_manifest': str(ABLATION / 'protocol.json'), 'frozen_sha256': frozen, 'frozen_epoch': time.time()}
    experiment.save(OUT / 'protocol.json', protocol)
    protocol_hash = experiment.digest(OUT / 'protocol.json')
    print(json.dumps({'stage': 'protocol_frozen', 'pid': os.getpid(), 'protocol_sha256': protocol_hash, 'search_rmse_already_known': known_search}), flush=True)
    with threadpool_limits(limits=8), torch.inference_mode():
        frame, split, towns = experiment.inputs(stack)
        assert split.fit_months[-1] == protocol['fit_end']
        assert split.search_months == protocol['search_months']
        assert split.confirm_months == protocol['confirm_months']
        confirm_input = split.confirm.drop(columns='MONTHLY_RENT')
        fit_input = split.fit.drop(columns='MONTHLY_RENT')
        parts = {'confirm': confirm_input}
        store = stack._empty_store(parts, confirm_input.iloc[:0])
        store.pop('test')
        prediction_started = time.time()
        experiment.save(OUT / 'status.json', {'stage': 'confirm_prediction_started', 'pid': os.getpid(), 'epoch': prediction_started, 'candidate_confirm_evaluations': 1})
        experiment.model_predictions(stack, area_folder, store, parts, fit_input, towns)
        predictions = experiment.saved_mlp_predictions(stack, BASELINE / 'stack', store['confirm'])
        assert len(predictions) == 8924 and np.isfinite(predictions).all()
        month = frame.RENT_APPROVAL_DATE.astype(str).str[:7]
        row_ids = np.flatnonzero(month.isin(protocol['confirm_months']).to_numpy())
        truth = split.confirm.MONTHLY_RENT.to_numpy(float)
        assert len(row_ids) == len(truth) == len(predictions)
        assert np.isfinite(truth).all()
        error = predictions - truth
        score = stack.rmse(truth, predictions)
        metrics = {'rows': len(truth), 'rmse': score, 'mae': float(np.mean(np.abs(error))), 'bias_pred_minus_truth': float(np.mean(error))}
        table = pd.DataFrame({'source_row_id': row_ids, 'date': split.confirm.RENT_APPROVAL_DATE.to_numpy(), 'truth': truth, 'prediction': predictions})
        predictions_path = OUT / 'confirm_predictions.csv'
        table.to_csv(predictions_path, index=False)
        monthly = {}
        for value in protocol['confirm_months']:
            mask = table.date.astype(str).str[:7].to_numpy() == value
            monthly[value] = {'rows': int(mask.sum()), 'rmse': float(np.sqrt(np.mean(error[mask] ** 2))), 'bias_pred_minus_truth': float(error[mask].mean())}
    experiment.verify_frozen()
    for relative, expected in frozen.items():
        if experiment.digest(asset_path(Path(relative))) != expected:
            raise RuntimeError(f'Frozen input changed: {relative}')
    assert experiment.digest(OUT / 'protocol.json') == protocol_hash
    improved = known_search < baseline_metrics['search_rmse'] and score < baseline_metrics['confirm_rmse']
    result = {'status': 'completed', 'pid': os.getpid(), 'exit_code': 0, 'candidate': 'frozen_baseline_mlp_with_area_catboost', 'search_rmse': known_search, 'search_source': 'Previously recorded Search diagnostic; no new Search tuning.', 'confirm': metrics, 'confirm_by_month': monthly, 'baseline': {'search_rmse': baseline_metrics['search_rmse'], 'confirm_rmse': baseline_metrics['confirm_rmse']}, 'delta_from_local_baseline': {'search_rmse': known_search - baseline_metrics['search_rmse'], 'confirm_rmse': score - baseline_metrics['confirm_rmse']}, 'passed_predeclared_criterion': improved, 'conclusion': 'Improved Search and original Confirm versus the local control.' if improved else 'Improvement not established on both original periods; no tuning or reselection performed.', 'candidate_confirm_evaluations': 1, 'training_performed': False, 'weights_updated': False, 'model_and_source_hashes_unchanged': True, 'original_selection_unchanged': True, 'all_predictions_finite': True, 'protocol_sha256': protocol_hash, 'protocol_frozen_before_prediction': protocol['frozen_epoch'] < prediction_started, 'confirm_predictions_csv': str(predictions_path.resolve()), 'confirm_predictions_sha256': experiment.digest(predictions_path), 'elapsed_seconds': time.time() - started, 'finished_epoch': time.time()}
    experiment.save(OUT / 'results.json', result)
    experiment.save(OUT / 'status.json', {'stage': 'completed', 'pid': os.getpid(), 'exit_code': 0, 'confirm_rmse': score, 'passed_predeclared_criterion': improved})
    print(json.dumps(result, indent=2), flush=True)
if __name__ == '__main__':
    import argparse as _cli_argparse
    _cli_argparse.ArgumentParser(description=__doc__).parse_args()
    if OUT.exists():
        raise SystemExit('Refusing a second evaluation: the one-shot output directory already exists')
    try:
        main()
    except BaseException as exc:
        traceback.print_exc()
        if OUT.exists():
            (OUT / 'failure.json').write_text(json.dumps({'error': repr(exc), 'pid': os.getpid(), 'epoch': time.time()}, indent=2), encoding='utf-8')
        raise
