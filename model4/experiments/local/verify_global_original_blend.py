"""Verify saved blend rows and fixed weights without any model inference."""
from experiments.local.bootstrap import Path, CODE_DIR, DATA_ROOT, SUPPORT_ROOT, ARTIFACT_ROOT, asset_path, asset_key, support_path

def main():
    import json
    import numpy as np
    import pandas as pd
    from experiments.local.confirm_global_original_blend import ROOT, OUT, GLOBAL, WEIGHTS, read, digest, check_frozen
    from experiments.local.spatial_history_io import write_json
    protocol = read(OUT / 'protocol.json')
    selection = read(OUT / 'selection_frozen.json')
    result = read(OUT / 'results.json')
    started = read(OUT / 'confirm_started.json')
    events = [json.loads(line) for line in (OUT / 'events.jsonl').read_text().splitlines()]
    assert result['status'] == 'completed' and result['exit_code'] == 0
    assert protocol['weights'] == selection['weights'] == result['weights'] == started['weights'] == WEIGHTS
    assert protocol['frozen_epoch'] < selection['epoch'] < started['epoch']
    assert digest(OUT / 'protocol.json') == selection['protocol_sha256'] == result['protocol_sha256']
    assert digest(OUT / 'selection_frozen.json') == result['selection_sha256'] == started['selection_sha256']
    assert digest(OUT / 'search_predictions.csv') == protocol['search_predictions_sha256'] == selection['search_predictions_sha256']
    assert digest(OUT / 'search_metrics.json') == protocol['search_metrics_sha256']
    assert digest(OUT / 'confirm_predictions.csv') == result['confirm_predictions_sha256']
    check_frozen(protocol)
    assert sum((event['stage'] == 'confirm_evaluation_started' for event in events)) == 1
    assert sum((event['stage'] == 'original_component_inference_started' for event in events)) == 1
    assert events[-1]['stage'] == 'completed' and (not any((event['stage'] == 'failed' for event in events)))
    assert result['candidate_confirm_evaluations'] == result['original_component_confirm_inferences'] == 1
    assert result['global_component_confirm_inferences'] == 0
    assert not result['training_performed'] and (not result['weight_tuning_performed'])
    raw = pd.read_csv(asset_path(Path('support/Data/train.csv')), usecols=['RENT_APPROVAL_DATE', 'MONTHLY_RENT'])
    checks = {}
    for stage, start, end in [('search', 132802, 141076), ('confirm', 141076, 150000)]:
        saved = pd.read_csv(OUT / f'{stage}_predictions.csv')
        global_saved = pd.read_csv(GLOBAL / f'{stage}_predictions.csv')
        assert len(saved) == end - start and saved.source_row_id.is_unique
        assert np.array_equal(saved.source_row_id.to_numpy(), np.arange(start, end))
        assert np.array_equal(saved.truth.to_numpy(), raw.MONTHLY_RENT.iloc[start:end].to_numpy())
        assert np.array_equal(saved.date.astype(str).to_numpy(), raw.RENT_APPROVAL_DATE.iloc[start:end].astype(str).to_numpy())
        assert np.isfinite(saved[['truth', 'global_prediction', 'original_prediction', 'prediction']].to_numpy(float)).all()
        assert np.allclose(saved.global_prediction.to_numpy(), global_saved.prediction.to_numpy(), rtol=0, atol=1e-09)
        expected = 0.5 * saved.global_prediction.to_numpy() + 0.5 * saved.original_prediction.to_numpy()
        assert np.max(np.abs(expected - saved.prediction.to_numpy())) < 1e-09
        error = saved.prediction.to_numpy() - saved.truth.to_numpy()
        score = float(np.sqrt(np.mean(error ** 2)))
        assert abs(score - result[stage]['rmse']) < 1e-09
        original_score = float(np.sqrt(np.mean((saved.original_prediction - saved.truth) ** 2)))
        assert abs(original_score - protocol['local_baseline'][stage + '_rmse']) < 1e-08
        if stage == 'search':
            local = pd.read_csv(asset_path(Path('feature_ablation_01/baseline_search_predictions.csv')))
            assert np.allclose(saved.original_prediction.to_numpy(), local.prediction.to_numpy(), rtol=0, atol=1e-09)
            assert abs(score - selection['search_rmse']) < 1e-09 and score < selection['search_gate_rmse']
        else:
            for month, metrics in result['confirm']['by_month'].items():
                mask = saved.date.str[:7].to_numpy() == month
                assert int(mask.sum()) == metrics['rows']
                assert abs(float(np.sqrt(np.mean(error[mask] ** 2))) - metrics['rmse']) < 1e-09
        checks[stage] = {'rows': len(saved), 'rmse': score, 'source_row_ids_dates_truth_exact': True, 'finite': True, 'weights_exactly_50_50': True, 'original_component_matches_fixed_baseline_rmse': True}
    assert result['improved_both_periods_over_current_global'] == (checks['search']['rmse'] < protocol['current_global']['search_rmse'] and checks['confirm']['rmse'] < protocol['current_global']['confirm_rmse'])
    report = {'status': 'passed', 'scope': 'Saved artifacts only; no training or model inference', 'weights_frozen_before_confirm': True, 'candidate_confirm_evaluations': 1, 'frozen_source_and_model_files_verified': len(protocol['frozen_sha256']), 'source_baseline_labels_and_split_unchanged': True, 'predictions': checks, 'exit_code': 0, 'elapsed_seconds': result['elapsed_seconds']}
    write_json(OUT / 'completion_verification.json', report)
    print(json.dumps(report, indent=2))
if __name__ == '__main__':
    import argparse as _cli_argparse
    _cli_argparse.ArgumentParser(description=__doc__).parse_args()
    main()
