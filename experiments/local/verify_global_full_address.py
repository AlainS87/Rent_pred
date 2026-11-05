"""Audit saved artifacts only: no fitting and no model.predict calls."""
from experiments.local.bootstrap import Path, CODE_DIR, DATA_ROOT, SUPPORT_ROOT, ARTIFACT_ROOT, asset_path, asset_key, support_path

def main():
    import json
    from pathlib import Path
    import sys
    ROOT = Path(__file__).resolve().parent
    import joblib
    import numpy as np
    import pandas as pd
    from experiments.local.global_full_address import OUT, FEATURES, PARAMS, PREPROCESS, read, digest, check_frozen
    from experiments.local.spatial_history_io import write_json
    supervisor = read(OUT / 'supervisor.json')
    assert supervisor['status'] == 'completed' and supervisor['exit_code'] == 0
    assert supervisor['elapsed_seconds'] < 900
    result = read(OUT / 'results.json')
    protocol = read(OUT / 'protocol.json')
    selection = read(OUT / 'selection_frozen.json')
    fit_audit = read(OUT / 'fit_audit.json')
    data_audit = read(OUT / 'data_audit.json')
    events = [json.loads(line) for line in (OUT / 'events.jsonl').read_text().splitlines()]
    assert result['status'] == 'completed' and result['exit_code'] == 0
    assert result['elapsed_seconds'] < 900
    assert digest(OUT / 'protocol.json') == result['protocol_sha256'] == selection['protocol_sha256']
    assert digest(OUT / 'selection_frozen.json') == result['selection_sha256']
    check_frozen(protocol)
    for relative, expected in selection['candidate_sha256'].items():
        assert digest(OUT / relative) == expected, relative
    assert protocol['search_gate_rmse'] == 445.3804031850469
    assert protocol['features'] == FEATURES and protocol['params'] == PARAMS
    assert result['passed_search_gate'] == selection['passed_search_gate']
    assert selection['passed_search_gate'] == (selection['search_rmse'] < protocol['search_gate_rmse'])
    assert sum((event['stage'] == 'catboost_fit_enter' for event in events)) == 1
    assert sum((event['stage'] == 'catboost_fit_complete' for event in events)) == 1
    assert all((event['pid'] == supervisor['training_python_pid'] for event in events))
    assert not any((event['stage'] == 'failed' for event in events))
    assert events[-1]['stage'] == 'completed'
    model = joblib.load(OUT / 'model.joblib')
    preparer = joblib.load(OUT / 'preprocessor.joblib')
    assert model.get_params() == PARAMS == fit_audit['params']
    assert model.get_all_params() == fit_audit['resolved_params']
    assert set(model.feature_names_) == set(FEATURES) and len(model.feature_names_) == 26
    assert model.feature_names_ == fit_audit['features'] == data_audit['model_columns']
    assert model.get_cat_feature_indices() == fit_audit['categorical_indices']
    assert fit_audit['fit_calls'] == result['fit_calls'] == 1
    assert fit_audit['target'] == 'MONTHLY_RENT'
    assert fit_audit['training_weights'] == fit_audit['early_stopping_weights'] == 'none'
    assert preparer.selected_columns_ == FEATURES and preparer.spec == PREPROCESS
    assert not preparer.include_building_ and (not preparer.target_values_)
    assert model.tree_count_ == fit_audit['trees'] <= 1000
    assert model.get_best_iteration() == fit_audit['best_iteration']
    assert data_audit['original_inputs_and_labels_exactly_equal']
    assert data_audit['train_rows'] == 132802 and data_audit['search_rows'] == 8274
    passed = selection['passed_search_gate']
    source = pd.read_csv(asset_path(Path('support/Data/train.csv')), usecols=['RENT_APPROVAL_DATE', 'MONTHLY_RENT'], nrows=150000 if passed else 141076)
    auxiliary = pd.read_csv(asset_path(Path('support/Data/auxiliary_features/train_features.csv')), usecols=preparer.numeric_columns_, nrows=132802)
    for name in preparer.numeric_columns_:
        values = auxiliary[name].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        expected = float(np.median(values)) if len(values) else 0.0
        assert preparer.numeric_fill_[name] == expected, name
        assert preparer.numeric_center_[name] == 0 and preparer.numeric_spread_[name] == 1
    checks = {}
    for stage, start, end in [('search', 132802, 141076)] + ([('confirm', 141076, 150000)] if passed else []):
        path = OUT / f'{stage}_predictions.csv'
        saved = pd.read_csv(path)
        assert np.array_equal(saved.source_row_id.to_numpy(), np.arange(start, end))
        assert saved.source_row_id.is_unique and len(saved) == end - start
        assert np.array_equal(saved.truth.to_numpy(), source.iloc[start:end].MONTHLY_RENT.to_numpy())
        assert np.array_equal(saved.date.astype(str).to_numpy(), source.iloc[start:end].RENT_APPROVAL_DATE.astype(str).to_numpy())
        assert np.isfinite(saved[['truth', 'prediction']].to_numpy(float)).all()
        error = saved.prediction.to_numpy(float) - saved.truth.to_numpy(float)
        rmse = float(np.sqrt(np.mean(error ** 2)))
        recorded = read(OUT / ('search_metrics.json' if stage == 'search' else 'confirmation_metrics.json'))
        assert abs(rmse - recorded['rmse']) < 1e-09
        assert result[stage] == recorded and recorded['rows'] == len(saved)
        assert abs(recorded['delta_from_baseline'] - (rmse - protocol['baseline'][stage + '_rmse'])) < 1e-09
        assert abs(recorded['delta_from_previous_best'] - (rmse - protocol['previous_best'][stage + '_rmse'])) < 1e-09
        assert abs(recorded['bias_pred_minus_truth'] - float(error.mean())) < 1e-09
        for month, details in recorded['by_month'].items():
            mask = saved.date.str[:7].to_numpy() == month
            assert int(mask.sum()) == details['rows']
            assert abs(float(np.sqrt(np.mean(error[mask] ** 2))) - details['rmse']) < 1e-09
        checks[stage] = {'rows': len(saved), 'row_ids_and_truth_exact': True, 'finite': True, 'rmse': rmse, 'prediction_min': float(saved.prediction.min()), 'prediction_max': float(saved.prediction.max()), 'sha256': digest(path), 'csv': str(path)}
    assert abs(selection['search_rmse'] - checks['search']['rmse']) < 1e-09
    assert abs(min(fit_audit['eval_history']['validation']['RMSE']) - checks['search']['rmse']) < 1e-05
    if passed:
        started = read(OUT / 'confirm_started.json')
        assert selection['epoch'] < started['epoch']
        assert started['selection_sha256'] == result['selection_sha256']
        assert result['candidate_confirm_evaluations'] == 1
        assert sum((event['stage'] == 'confirm_prediction_started' for event in events)) == 1
        assert sum((event['stage'] == 'confirm_complete' for event in events)) == 1
        assert result['confirm_predictions_sha256'] == checks['confirm']['sha256']
        assert result['improved_over_previous_best_on_both_periods'] == (checks['confirm']['rmse'] < protocol['previous_best']['confirm_rmse'])
    else:
        assert result['confirm'] is None and result['candidate_confirm_evaluations'] == 0
        assert result['confirm_predictions_csv'] is None
        for name in ['confirm_started.json', 'confirm_predictions.csv', 'confirmation_metrics.json']:
            assert not (OUT / name).exists(), name
        assert not any((event['stage'].startswith('confirm_') for event in events))
    report = {'status': 'passed', 'verification_scope': 'Saved artifacts only; no fitting or prediction', 'frozen_files_verified': len(protocol['frozen_sha256']), 'one_candidate_one_fit': True, 'source_baseline_split_labels_unchanged': True, 'training_only_imputation_verified': True, 'constructor_and_resolved_params_verified': True, 'strict_search_gate_verified': True, 'confirm_evaluations': result['candidate_confirm_evaluations'], 'predictions': checks, 'elapsed_seconds': supervisor['elapsed_seconds'], 'exit_code': supervisor['exit_code'], 'protocol_sha256': digest(OUT / 'protocol.json'), 'selection_sha256': digest(OUT / 'selection_frozen.json')}
    from experiments.local.global_full_address import validate_fit_audit, fixed_search_diagnostics
    report['one_feature_change_verification'] = validate_fit_audit(fit_audit)
    previous_preparer = joblib.load(asset_path(Path('global_monthly_ctr8_01/preprocessor.joblib')))
    from experiments.local.global_full_address import validate_preparer, add_address, EXTRA
    assert validate_preparer(preparer, previous_preparer)
    report['original_preprocessor_attributes_unchanged_with_one_address_category_added'] = True
    assert result['search']['fixed_prior_error_groups'] == fixed_search_diagnostics()
    report['fixed_prior_error_groups'] = result['search']['fixed_prior_error_groups']
    raw_address = pd.read_csv(asset_path(Path('support/Data/train.csv')), usecols=['TOWN'], nrows=141076)
    aux_address = pd.read_csv(asset_path(Path('support/Data/auxiliary_features/train_features.csv')), usecols=['street_key', 'block_key'], nrows=141076)
    address = add_address(pd.concat([raw_address, aux_address], axis=1))[EXTRA]
    assert set(preparer.levels_[EXTRA]) == set(address.iloc[:132802])
    assert int(address.iloc[:132802].nunique()) == data_audit['address_audit']['train_unique_addresses']
    assert int((~address.iloc[132802:].isin(set(address.iloc[:132802]))).sum()) == data_audit['address_audit']['search_unseen_address_rows']
    assert data_audit['address_audit']['original_25_transformed_features_exactly_equal']
    report['explicit_address_training_only_categories_verified'] = True
    report['address_audit'] = data_audit['address_audit']
    write_json(OUT / 'completion_verification.json', report)
    print(json.dumps(report, indent=2))
if __name__ == '__main__':
    import argparse as _cli_argparse
    _cli_argparse.ArgumentParser(description=__doc__).parse_args()
    main()
