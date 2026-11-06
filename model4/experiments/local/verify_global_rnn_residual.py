"""Independent saved-artifact audit. No fit or prediction calls."""
from experiments.local.bootstrap import Path, CODE_DIR, DATA_ROOT, SUPPORT_ROOT, ARTIFACT_ROOT, asset_path, asset_key, support_path

def main():
    import json
    import sys
    from pathlib import Path
    ROOT = Path(__file__).resolve().parent
    import joblib
    import numpy as np
    import pandas as pd
    from experiments.local.global_rnn_residual import OUT, PRIOR, FEATURES, PARAMS, PREPROCESS, TARGET, check_frozen, read, digest
    from experiments.local.spatial_history_io import write_json
    supervisor = read(OUT / 'supervisor.json')
    assert supervisor['status'] == 'completed' and supervisor['exit_code'] == 0
    assert supervisor['elapsed_seconds'] < 900
    result, protocol = (read(OUT / 'results.json'), read(OUT / 'protocol.json'))
    selection, audit = (read(OUT / 'selection_frozen.json'), read(OUT / 'fit_audit.json'))
    data = read(OUT / 'data_audit.json')
    assert result['status'] == 'completed' and result['exit_code'] == 0 and (result['elapsed_seconds'] < 900)
    assert digest(OUT / 'protocol.json') == result['protocol_sha256'] == selection['protocol_sha256']
    assert digest(OUT / 'selection_frozen.json') == result['selection_sha256']
    check_frozen(protocol)
    for name, expected in selection['candidate_sha256'].items():
        assert digest(OUT / name) == expected, name
    assert protocol['search_gate_rmse'] == 444.9106685519473
    assert protocol['features'] == FEATURES and protocol['params'] == PARAMS
    assert protocol['training_target'] == TARGET
    assert result['rnn_fit_calls'] == 0 and result['fit_calls'] == 1
    assert selection['confirm_evaluations_before_freeze'] == 0
    assert result['passed_search_gate'] == selection['passed_search_gate']
    passed = selection['passed_search_gate']
    assert passed == (selection['search_rmse'] < protocol['search_gate_rmse'])
    events = [json.loads(line) for line in (OUT / 'events.jsonl').read_text().splitlines()]
    assert sum((event['stage'] == 'catboost_fit_enter' for event in events)) == 1
    assert sum((event['stage'] == 'catboost_fit_complete' for event in events)) == 1
    assert events[-1]['stage'] == 'completed' and (not any((event['stage'] == 'failed' for event in events)))
    assert all((event['pid'] == supervisor['training_python_pid'] for event in events))
    model, prep = (joblib.load(OUT / 'model.joblib'), joblib.load(OUT / 'preprocessor.joblib'))
    prior_audit = read(PRIOR / 'fit_audit.json')
    assert model.get_params() == audit['params'] == PARAMS == prior_audit['params']
    assert model.get_all_params() == audit['resolved_params'] == prior_audit['resolved_params']
    assert model.feature_names_ == audit['features'] == data['model_columns'] == prior_audit['features']
    assert model.get_cat_feature_indices() == audit['categorical_indices'] == prior_audit['categorical_indices']
    assert model.tree_count_ == audit['trees'] <= 1000 and model.get_best_iteration() == audit['best_iteration']
    assert audit['target'] == TARGET and audit['fit_calls'] == 1
    assert audit['training_weights'] == audit['early_stopping_weights'] == 'none'
    assert digest(OUT / 'preprocessor.joblib') == digest(PRIOR / 'preprocessor.joblib') == data['preprocessor_sha256']
    assert prep.selected_columns_ == FEATURES and prep.spec == PREPROCESS
    assert not prep.target_values_ and (not prep.include_building_)
    assert data['original_inputs_and_labels_exactly_equal'] and (not data['confirm_read_for_fitting'])
    assert data['train_rows'] == 132802 and data['search_rows'] == 8274
    assert not data['trend_audit']['prefix_uses_rental_labels']
    assert all((value <= 2e-05 for value in data['trend_audit']['max_abs_cached_mu_difference'].values()))
    cache = joblib.load(asset_path(Path('feature_ablation_01/training_cache.joblib')))
    source = pd.read_csv(asset_path(Path('support/Data/train.csv')), usecols=['RENT_APPROVAL_DATE', 'MONTHLY_RENT'], nrows=150000 if passed else 141076)
    for stage, start, end in [('train', 0, 132802), ('search', 132802, 141076)]:
        targets = pd.read_csv(OUT / f'{stage}_target_audit.csv')
        assert len(targets) == end - start and np.array_equal(targets.source_row_id, np.arange(start, end))
        assert np.array_equal(targets.truth, source.iloc[start:end].MONTHLY_RENT)
        assert np.allclose(targets.rnn_unit_level, cache['store'][stage]['mu'], rtol=0, atol=1e-12)
        assert np.array_equal(targets.floor_area_sqm, cache['parts'][stage].floor_area_sqm)
        assert np.isfinite(targets.to_numpy(float)).all()
        assert np.allclose(targets.trend_component, targets.rnn_unit_level * targets.floor_area_sqm, rtol=0, atol=1e-09)
        assert np.allclose(targets.residual_target, targets.truth - targets.trend_component, rtol=0, atol=1e-09)
    checks = {}
    for stage, start, end in [('search', 132802, 141076)] + ([('confirm', 141076, 150000)] if passed else []):
        path = OUT / f'{stage}_predictions.csv'
        saved = pd.read_csv(path)
        assert np.array_equal(saved.source_row_id, np.arange(start, end)) and saved.source_row_id.is_unique
        assert len(saved) == end - start
        assert np.array_equal(saved.truth, source.iloc[start:end].MONTHLY_RENT)
        assert np.array_equal(saved.date.astype(str), source.iloc[start:end].RENT_APPROVAL_DATE.astype(str))
        assert np.isfinite(saved.select_dtypes(include=[np.number]).to_numpy()).all()
        assert np.allclose(saved.prediction, saved.trend_component + saved.residual_prediction, rtol=0, atol=1e-09)
        if stage == 'search':
            assert np.allclose(saved.trend_component, pd.read_csv(OUT / 'search_target_audit.csv').trend_component, rtol=0, atol=1e-09)
        else:
            assert np.allclose(saved.trend_component, saved.rnn_unit_level * saved.floor_area_sqm, rtol=0, atol=1e-09)
        errors = saved.prediction.to_numpy() - saved.truth.to_numpy()
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        recorded = read(OUT / ('search_metrics.json' if stage == 'search' else 'confirmation_metrics.json'))
        assert abs(rmse - recorded['rmse']) < 1e-09
        assert result[stage] == recorded and recorded['rows'] == len(saved)
        assert abs(recorded['delta_from_baseline'] - (rmse - protocol['baseline'][stage + '_rmse'])) < 1e-09
        assert abs(recorded['delta_from_previous_best'] - (rmse - protocol['previous_best'][stage + '_rmse'])) < 1e-09
        for month, values in recorded['by_month'].items():
            mask = saved.date.str[:7].to_numpy() == month
            assert int(mask.sum()) == values['rows']
            assert abs(float(np.sqrt(np.mean(errors[mask] ** 2))) - values['rmse']) < 1e-09
        checks[stage] = {'rows': len(saved), 'row_ids_and_truth_exact': True, 'finite': True, 'rmse': rmse, 'csv': str(path), 'sha256': digest(path)}
    assert abs(selection['search_rmse'] - checks['search']['rmse']) < 1e-09
    assert abs(min(audit['eval_history']['validation']['RMSE']) - checks['search']['rmse']) < 0.0001
    if passed:
        started = read(OUT / 'confirm_started.json')
        assert selection['epoch'] < started['epoch'] and started['selection_sha256'] == result['selection_sha256']
        assert result['candidate_confirm_evaluations'] == 1
        assert sum((event['stage'] == 'confirm_prediction_started' for event in events)) == 1
        assert sum((event['stage'] == 'confirm_complete' for event in events)) == 1
        assert result['confirm_predictions_sha256'] == checks['confirm']['sha256']
        assert result['improved_over_previous_best_on_both_periods'] == (checks['confirm']['rmse'] < protocol['previous_best']['confirm_rmse'])
    else:
        assert result['confirm'] is None and result['candidate_confirm_evaluations'] == 0
        assert result['confirm_predictions_csv'] is None and (not result['improved_over_previous_best_on_both_periods'])
        for name in ['confirm_started.json', 'confirm_predictions.csv', 'confirmation_metrics.json']:
            assert not (OUT / name).exists(), name
        assert not any((event['stage'].startswith('confirm_') for event in events))
    assert not result['long_horizon_accuracy_validated']
    report = {'status': 'passed', 'verification_scope': 'Saved artifacts only; no fitting or prediction', 'frozen_files_verified': len(protocol['frozen_sha256']), 'one_candidate_one_fit': True, 'rnn_fits': 0, 'source_baseline_split_labels_unchanged': True, 'preprocessor_byte_identical': True, 'features_constructor_resolved_parameters_unchanged': True, 'residual_target_identity_verified': True, 'unweighted_monthly_rent_rmse_verified': True, 'strict_search_gate_verified': True, 'confirm_evaluations': result['candidate_confirm_evaluations'], 'predictions': checks, 'elapsed_seconds': supervisor['elapsed_seconds'], 'exit_code': supervisor['exit_code'], 'long_horizon_accuracy_validated': False, 'protocol_sha256': digest(OUT / 'protocol.json'), 'selection_sha256': digest(OUT / 'selection_frozen.json')}
    write_json(OUT / 'completion_verification.json', report)
    print(json.dumps(report, indent=2))
if __name__ == '__main__':
    import argparse as _cli_argparse
    _cli_argparse.ArgumentParser(description=__doc__).parse_args()
    main()
