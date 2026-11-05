"""Verify saved index fits, historical predictions, gates, and optional CatBoost outputs.

No estimator fit or predict calls. Historical metrics are reconstructed from raw
rent rows, saved coefficients and saved forecast formulas.
"""
from experiments.local.bootstrap import Path, CODE_DIR, DATA_ROOT, SUPPORT_ROOT, ARTIFACT_ROOT, asset_path, asset_key, support_path

def main():
    import json
    import sys
    from pathlib import Path
    ROOT = Path(__file__).resolve().parent
    import joblib
    import numpy as np
    import pandas as pd
    from experiments.local.temporal_index_trial import OUT, PRIOR, ORIGINS, FEATURES, PARAMS, read, digest, check_frozen
    from experiments.local.temporal_index import KEYS, prepare_rows, gate, scales
    from experiments.local.spatial_history_io import write_json
    protocol = read(OUT / 'protocol.json')
    result = read(OUT / 'results.json')
    supervisor = read(OUT / 'supervisor.json')
    assert supervisor['status'] == 'completed' and supervisor['exit_code'] == 0 and (supervisor['elapsed_seconds'] < 900)
    assert result['status'] == 'completed' and result['exit_code'] == 0 and (result['elapsed_seconds'] < 900)
    assert digest(OUT / 'protocol.json') == result['protocol_sha256']
    assert digest(OUT / 'time_selection_frozen.json') == result['time_selection_sha256']
    check_frozen(protocol)
    time_selection = read(OUT / 'time_selection_frozen.json')
    assert time_selection['protocol_sha256'] == result['protocol_sha256']
    assert time_selection['historical_metrics_sha256'] == digest(OUT / 'historical_metrics.json')
    assert time_selection['catboost_fits_before_freeze'] == time_selection['confirm_evaluations_before_freeze'] == 0
    events = [json.loads(line) for line in (OUT / 'events.jsonl').read_text().splitlines()]
    assert events[-1]['stage'] == 'completed' and (not any((event['stage'] == 'failed' for event in events)))
    assert all((event['pid'] == supervisor['training_python_pid'] for event in events))
    assert sum((event['stage'] == 'historical_index_fit_started' for event in events)) == 2
    assert sum((event['stage'] == 'historical_index_fit_complete' for event in events)) == 2
    raw = pd.read_csv(asset_path(Path('support/Data/train.csv')), nrows=132802, usecols=['RENT_APPROVAL_DATE', 'MONTHLY_RENT', *KEYS], dtype={name: str for name in KEYS if name != 'FLOOR_AREA_SQM'})
    raw['source_row_id'] = np.arange(len(raw))
    rows = prepare_rows(raw)
    tables = []
    origin_checks = {}
    for origin, end, h in ORIGINS:
        slug = origin.replace('-', '_')
        index = joblib.load(OUT / f'index_{slug}.joblib')
        history = rows.loc[rows.month <= origin].copy()
        future = rows.loc[(rows.month > origin) & (rows.month <= end)].copy()
        assert index['fit_end'] == origin and index['fit_rows'] == len(history)
        assert index['months'] == sorted(history.month.unique())
        assert index['damping'] == 0.9 and index['slope_months'] == 6
        distinct = history.groupby(KEYS, observed=True).month.nunique()
        eligible = distinct.index[distinct >= 2]
        assert set(index['property_effect'].index) == set(eligible)
        hk = pd.MultiIndex.from_frame(history[KEYS])
        alpha = index['property_effect'].reindex(hk).to_numpy(float)
        b = history.month.map(dict(zip(index['months'], index['month_effect']))).to_numpy(float)
        hist_ok = np.isfinite(alpha)
        residual = history.log_rent.to_numpy() - alpha - b
        hist = history.loc[hist_ok, KEYS + ['month']].copy()
        hist['error'] = residual[hist_ok]
        assert hist.groupby(KEYS, observed=True).error.sum().abs().max() < 1e-06
        assert hist.groupby('month').error.sum().abs().max() < 1e-06
        assert int(hist_ok.sum()) == index['eligible_fit_rows']
        xx = np.arange(6, dtype=float) - 2.5
        slope = float(np.dot(xx, index['month_effect'][-6:]) / np.dot(xx, xx))
        assert abs(slope - index['slope']) < 1e-12
        saved_rows = pd.read_csv(OUT / f'evaluation_rows_{slug}.csv')
        assert np.array_equal(saved_rows.source_row_id, future.source_row_id)
        assert np.allclose(saved_rows.log_rent, future.log_rent, rtol=0, atol=1e-12)
        future_alpha = index['property_effect'].reindex(pd.MultiIndex.from_frame(future[KEYS])).to_numpy(float)
        assert np.allclose(saved_rows.property_effect, future_alpha, rtol=0, atol=1e-12, equal_nan=True)
        assert np.array_equal(saved_rows.eligible, np.isfinite(future_alpha))
        observed = future.log_rent.to_numpy() - future_alpha
        assert np.allclose(saved_rows.observed_time_component, observed, rtol=0, atol=1e-12, equal_nan=True)
        future['observed'] = observed
        expected = future.groupby('month').observed.mean()
        table = pd.read_csv(OUT / f'backtest_{slug}.csv')
        assert np.array_equal(table.month, expected.index)
        assert np.array_equal(table.horizon, np.arange(1, h + 1))
        assert np.allclose(table.observed_index, 100 * np.exp(expected.to_numpy() - index['month_effect'][-1]), rtol=0, atol=1e-09)
        assert np.allclose(table.flat_index, 100, rtol=0, atol=1e-12)
        expected_damped = 100 * np.exp(slope * 0.9 * (1 - 0.9 ** np.arange(1, h + 1)) / 0.1)
        assert np.allclose(table.damped_index, expected_damped, rtol=0, atol=1e-09)
        assert np.isfinite(table.select_dtypes(include=[np.number]).to_numpy()).all()
        assert np.allclose(table.coverage, table.eligible_rows / table.total_rows, rtol=0, atol=1e-12)
        tables.append(table)
        origin_checks[origin] = {'history_rows': len(history), 'history_months': len(index['months']), 'eligible_history_properties': len(eligible), 'future_months': h, 'minimum_future_coverage': float(table.coverage.min()), 'history_only_coefficients_verified': True, 'raw_evaluation_rows_and_forecast_formula_verified': True, 'slope_log_index_per_month': slope}
    combined = pd.concat(tables, ignore_index=True)
    saved_combined = pd.read_csv(OUT / 'historical_backtest_predictions.csv')
    pd.testing.assert_frame_equal(combined, saved_combined, check_exact=False, atol=1e-12, rtol=0)
    recomputed = gate(combined)
    recorded = read(OUT / 'historical_metrics.json')
    assert recomputed['conditions'] == recorded['conditions'] == time_selection['conditions']
    assert recomputed['passed'] == recorded['passed'] == time_selection['passed'] == result['time_gate_passed']
    for origin in recorded['by_origin']:
        for period in recorded['by_origin'][origin]:
            for key, value in recorded['by_origin'][origin][period].items():
                assert abs(value - recomputed['by_origin'][origin][period][key]) < 1e-09
    for period in recorded['pooled']:
        for key, value in recorded['pooled'][period].items():
            assert abs(value - recomputed['pooled'][period][key]) < 1e-09
    checks = {}
    if not result['time_gate_passed']:
        assert result['historical_index_fits'] == 2 and result['catboost_fits'] == 0
        assert result['search'] is None and result['confirm'] is None and (result['candidate_confirm_evaluations'] == 0)
        assert not any((event['stage'].startswith('catboost_') or event['stage'].startswith('confirm_') for event in events))
        for name in ['model.joblib', 'final_time_index.joblib', 'search_predictions.csv', 'confirm_predictions.csv', 'search_metrics.json', 'confirmation_metrics.json', 'selection_frozen.json', 'confirm_started.json']:
            assert not (OUT / name).exists(), name
    else:
        assert result['historical_index_fits'] == 3 and result['catboost_fits'] == 1
        assert sum((event['stage'] == 'catboost_fit_enter' for event in events)) == 1
        assert sum((event['stage'] == 'catboost_fit_complete' for event in events)) == 1
        selection = read(OUT / 'selection_frozen.json')
        assert digest(OUT / 'selection_frozen.json') == result['selection_sha256']
        for name, expected in selection['candidate_sha256'].items():
            assert digest(OUT / name) == expected
        assert selection['threshold'] == 444.9106685519473
        assert selection['passed_search_gate'] == (selection['search_rmse'] < selection['threshold'])
        index = joblib.load(OUT / 'final_time_index.joblib')
        assert index['fit_end'] == '2024-09' and index['fit_rows'] == 132802
        model = joblib.load(OUT / 'model.joblib')
        fit_audit = read(OUT / 'fit_audit.json')
        prior_audit = read(PRIOR / 'fit_audit.json')
        assert model.get_params() == PARAMS == fit_audit['params'] == prior_audit['params']
        assert model.get_all_params() == fit_audit['resolved_params'] == prior_audit['resolved_params']
        assert model.feature_names_ == fit_audit['features'] == prior_audit['features']
        assert digest(OUT / 'preprocessor.joblib') == digest(PRIOR / 'preprocessor.joblib')
        passed = selection['passed_search_gate']
        labels = pd.read_csv(asset_path(Path('support/Data/train.csv')), usecols=['RENT_APPROVAL_DATE', 'MONTHLY_RENT'], nrows=150000 if passed else 141076)
        for name, start, end in [('train', 0, 132802), ('search', 132802, 141076)]:
            target = pd.read_csv(OUT / f'{name}_target_audit.csv')
            assert np.array_equal(target.source_row_id, np.arange(start, end))
            assert np.array_equal(target.truth, labels.iloc[start:end].MONTHLY_RENT)
            scale = scales(index, labels.iloc[start:end].RENT_APPROVAL_DATE)
            assert np.allclose(target.time_scale, scale, rtol=0, atol=1e-12)
            assert np.allclose(target.normalized_target, target.truth / scale, rtol=0, atol=1e-08)
            assert np.allclose(target.weight, scale ** 2 / np.mean(scale ** 2), rtol=0, atol=1e-12)
        for name, start, end in [('search', 132802, 141076)] + ([('confirm', 141076, 150000)] if passed else []):
            pred = pd.read_csv(OUT / f'{name}_predictions.csv')
            assert np.array_equal(pred.source_row_id, np.arange(start, end)) and pred.source_row_id.is_unique
            assert np.array_equal(pred.truth, labels.iloc[start:end].MONTHLY_RENT)
            assert np.array_equal(pred.date, labels.iloc[start:end].RENT_APPROVAL_DATE)
            assert np.isfinite(pred.select_dtypes(include=[np.number]).to_numpy()).all()
            assert np.allclose(pred.prediction, pred.time_scale * pred.normalized_prediction, rtol=0, atol=1e-08)
            assert np.allclose(pred.time_scale, scales(index, pred.date), rtol=0, atol=1e-12)
            rmse = float(np.sqrt(np.mean((pred.prediction - pred.truth) ** 2)))
            recorded = read(OUT / ('search_metrics.json' if name == 'search' else 'confirmation_metrics.json'))
            assert abs(rmse - recorded['rmse']) < 1e-09 and result[name] == recorded
            checks[name] = {'rows': len(pred), 'rmse': rmse, 'rows_truth_finite_verified': True, 'csv': str(OUT / f'{name}_predictions.csv')}
        factor = fit_audit['search_metric_conversion_factor']
        assert abs(min(fit_audit['eval_history']['validation']['RMSE']) * factor - checks['search']['rmse']) < 0.001
        if passed:
            started = read(OUT / 'confirm_started.json')
            assert selection['epoch'] < started['epoch'] and started['selection_sha256'] == result['selection_sha256']
            assert result['candidate_confirm_evaluations'] == 1
            assert sum((event['stage'] == 'confirm_prediction_started' for event in events)) == 1
            assert sum((event['stage'] == 'confirm_complete' for event in events)) == 1
        else:
            assert result['candidate_confirm_evaluations'] == 0 and result['confirm'] is None
            assert not (OUT / 'confirm_started.json').exists() and (not (OUT / 'confirm_predictions.csv').exists())
    report = {'status': 'passed', 'verification_scope': 'Saved coefficients, raw rows and metrics; no fitting or model inference', 'frozen_files_verified': len(protocol['frozen_sha256']), 'original_baseline_labels_and_split_unchanged': True, 'history_only_time_index_verification': origin_checks, 'strict_time_gate_verified': True, 'time_gate_passed': result['time_gate_passed'], 'catboost_fits': result['catboost_fits'], 'candidate_confirm_evaluations': result['candidate_confirm_evaluations'], 'predictions': checks, 'elapsed_seconds': supervisor['elapsed_seconds'], 'exit_code': supervisor['exit_code'], 'long_horizon_metric_is_index_points_not_monthly_rent_rmse': True}
    write_json(OUT / 'completion_verification.json', report)
    print(json.dumps(report, indent=2))
if __name__ == '__main__':
    import argparse as _cli_argparse
    _cli_argparse.ArgumentParser(description=__doc__).parse_args()
    main()
