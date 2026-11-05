"""One predeclared vacancy correction; frozen CatBoost, gated saved-output scoring."""
from experiments.local.bootstrap import Path, CODE_DIR, DATA_ROOT, SUPPORT_ROOT, ARTIFACT_ROOT, asset_path, asset_key, support_path
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
from experiments.local.spatial_history_io import append_json, write_json
ROOT = Path(__file__).resolve().parent
OUT = asset_path(Path('vacancy_index_01'))
PRIOR = asset_path(Path('temporal_index_trial_01'))
SOURCE = asset_path(Path('support'))
ORIGINS = [('2022-09', '2024-07', 22), ('2023-05', '2024-09', 16)]
LAST_STAGE = 'not_started'
LAST_SUCCESSFUL_STAGE = 'not_started'
RESULT = {}

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()

def check_frozen(protocol):
    for relative, expected in protocol['frozen_sha256'].items():
        assert digest(asset_path(Path(relative))) == expected, relative

def record(stage, **fields):
    global LAST_STAGE, LAST_SUCCESSFUL_STAGE
    event = {'stage': stage, 'pid': os.getpid(), 'epoch': time.time(), **fields}
    append_json(OUT / 'events.jsonl', event)
    write_json(OUT / 'status.json', event, required=False)
    LAST_STAGE = stage
    if stage.endswith(('_complete', '_frozen')) or stage == 'completed':
        LAST_SUCCESSFUL_STAGE = stage
    print(json.dumps(event, allow_nan=False), flush=True)

def prepare():
    assert not OUT.exists(), 'Refusing overwrite or another attempt'
    protected = read(asset_path(Path('external_release_audit_01/results.json')))['frozen_sha256']
    check_frozen({'frozen_sha256': protected})
    frozen = dict(protected)
    for relative in ['vacancy_index_trial.py', 'temporal_index.py', 'spatial_history_io.py', 'external_release_audit_01/ura_quarterly_releases.csv']:
        frozen[relative] = digest(asset_path(Path(relative)))
    for name in ['protocol.json', 'results.json', 'preprocessor.joblib', 'model.joblib', 'final_time_index.joblib', 'index_2022_09.joblib', 'index_2023_05.joblib', 'backtest_2022_09.csv', 'backtest_2023_05.csv', 'search_predictions.csv', 'confirm_predictions.csv', 'completion_verification.json']:
        relative = 'temporal_index_trial_01/' + name
        frozen[relative] = digest(asset_path(Path(relative)))
    check_frozen({'frozen_sha256': frozen})
    protocol = {'status': 'frozen_before_first_candidate_test', 'candidate': 'single_sign_constrained_shrunk_vacancy_correction_to_frozen_time_forecast', 'candidate_count': 1, 'max_wall_seconds': 900, 'automatic_retry': False, 'budget_includes': 'Worker imports, all rolling index fits, calibration, historical tests, gated Search/Confirm, verification and writing', 'original_fit_end': '2024-09', 'rows': {'train': 132802, 'search': 8274, 'confirm': 8924}, 'search_months': ['2024-10', '2024-11', '2024-12'], 'confirm_months': ['2025-01', '2025-02', '2025-03'], 'baseline': {'search_rmse': 451.45286816840405, 'confirm_rmse': 467.11280438228}, 'current_best': {'search_rmse': 444.469123087004, 'confirm_rmse': 450.8945887042277}, 'search_gate_rmse': 444.4691, 'search_gate_operator': 'strictly_less_than', 'confirm_max_evaluations': 1, 'confirm_reselection': False, 'catboost_fits': 0, 'catboost_new_inference_calls': 0, 'frozen_housing_outputs': 'Use verified saved normalized_prediction from temporal_index_trial_01; rescale only', 'vacancy': {'source': 'external_release_audit_01/ura_quarterly_releases.csv', 'used_fields': ['quarter', 'release_date', 'vacancy_pct'], 'scope': 'URA completed private residential units excluding EC; auxiliary proxy, not HDB vacancy', 'x': 'Latest released vacancy quarter-on-quarter change in percentage points, divided by 3', 'availability': 'Latest release_date <= origin month-end; no later releases at any forecast horizon', 'future_inputs': 'Origin x held as initial signal; its effect decays with existing phi=0.9', 'no_other_external_features': True}, 'formula': {'phi': 0.9, 'kernel': 'K(h)=(1-0.9**h)/(1-0.9)', 'baseline': 'b_T + slope_last6*0.9*K(h)', 'candidate': 'baseline + beta_T*x_T*K(h)', 'coefficient': 'beta_T=min(0, sum(z*e)/(2*sum(z*z))), where z=x_t*K(h), e=observed_log_index(t+h)-baseline_log_forecast(t,h)', 'shrinkage': 'Fixed 50% shrinkage of the one-coefficient through-origin least-squares estimate toward zero; no grid or tuning', 'zero_variance_fallback': 'beta=0, preserve incumbent', 'calibration_origins': 'Every month from 2021-06 whose complete next six months end on/before T', 'calibration_horizons': [1, 2, 3, 4, 5, 6], 'calibration_weights': 'Every origin has exactly six equal-weight horizon observations', 'index_vintages': 'Fit property/month effects only through each calibration origin; known-property future index proxy uses frozen property effects', 'intercept': False, 'rationale': 'One parameter, expected negative sign and strong shrinkage limit flexibility; same damping as incumbent avoids adding a new decay parameter'}, 'historical_origins': [{'origin': o, 'evaluation_end': e, 'max_horizon': h} for o, e, h in ORIGINS], 'historical_gate': {'reference': 'Current damped time-index forecast, not flat benchmark', 'minimum_monthly_coverage': 0.7, 'each_origin_h7_plus_not_worse': True, 'pooled_h7_plus_minimum_rmse_reduction': 0.05, 'maximum_each_origin_h1_6_rmse_increase': 0.05, 'failure_action': 'Stop candidate; Search and Confirm not evaluated'}, 'metrics': 'Historical index-point RMSE separately from original unweighted individual monthly-rent RMSE', 'cautions': ['Historical windows overlap and were inspected before; retrospective supplemental validation', 'Confirm has been viewed before; original-split iterative validation, not untouched test', 'Historical local baseline scores belong to their recorded run', 'Only local validation metrics are retained'], 'frozen_sha256': frozen}
    OUT.mkdir()
    write_json(OUT / 'protocol.json', protocol)
    print(json.dumps({'prepared': str(OUT), 'protected_files': len(frozen), 'worker_started': False}))

def summarize(table):
    import numpy as np
    masks = {'h1_6': table.horizon <= 6, 'h7_16': table.horizon.between(7, 16), 'h17_22': table.horizon.between(17, 22), 'h7_plus': table.horizon >= 7, 'all': np.ones(len(table), bool)}
    for h in [1, 3, 6, 12, 16, 22]:
        if table.horizon.max() >= h:
            masks[f'through_h{h}'] = table.horizon <= h
    result = {}
    for name, mask in masks.items():
        part = table.loc[mask]
        if not len(part):
            continue
        old = float(np.sqrt(np.mean((part.baseline_index - part.observed_index) ** 2)))
        new = float(np.sqrt(np.mean((part.candidate_index - part.observed_index) ** 2)))
        result[name] = {'months': len(part), 'baseline_rmse_index_points': old, 'candidate_rmse_index_points': new, 'delta': new - old}
    return result

def history_gate(combined):
    by = {o: summarize(p) for o, p in combined.groupby('origin')}
    pooled = summarize(combined)
    conditions = {'each_month_coverage_at_least_70pct': bool((combined.coverage >= 0.7).all()), 'each_origin_h7_plus_not_worse': all((v['h7_plus']['delta'] <= 0 for v in by.values())), 'pooled_h7_plus_at_least_5pct_better': pooled['h7_plus']['candidate_rmse_index_points'] <= 0.95 * pooled['h7_plus']['baseline_rmse_index_points'], 'each_origin_h1_6_no_more_than_5pct_worse': all((v['h1_6']['candidate_rmse_index_points'] <= 1.05 * v['h1_6']['baseline_rmse_index_points'] for v in by.values()))}
    return {'passed': all(conditions.values()), 'conditions': conditions, 'by_origin': by, 'pooled': pooled, 'unit': 'Index points with origin=100; NOT individual monthly-rent RMSE'}

def run():
    global RESULT
    with (OUT / 'worker_identity.json').open('x', encoding='utf-8') as f:
        json.dump({'pid': os.getpid(), 'epoch': time.time()}, f)
    budget = read(OUT / 'budget.json')
    protocol = read(OUT / 'protocol.json')
    protocol_hash = digest(OUT / 'protocol.json')

    def deadline():
        if time.time() >= budget['deadline_epoch']:
            raise TimeoutError('900-second cumulative candidate budget expired')
    record('preflight_started')
    check_frozen(protocol)
    import joblib
    import numpy as np
    import pandas as pd
    from experiments.local.temporal_index import KEYS, prepare_rows, fit_index, forecast
    raw = pd.read_csv(support_path(Path('Data/train.csv')), nrows=132802, usecols=['RENT_APPROVAL_DATE', 'MONTHLY_RENT', *KEYS], dtype={k: str for k in KEYS if k != 'FLOOR_AREA_SQM'})
    assert len(raw) == 132802 and raw.RENT_APPROVAL_DATE.str[:7].max() == '2024-09'
    raw['source_row_id'] = np.arange(len(raw))
    rows = prepare_rows(raw)
    macro = pd.read_csv(asset_path(Path(protocol['vacancy']['source'])), usecols=['quarter', 'release_date', 'vacancy_pct'])
    macro['release_date'] = pd.to_datetime(macro.release_date)
    macro['vacancy_change_pp'] = macro.vacancy_pct.diff()
    macro['previous_vacancy_pct'] = macro.vacancy_pct.shift(1)
    assert macro.release_date.is_monotonic_increasing

    def available(origin):
        end = pd.Period(origin, freq='M').end_time
        a = macro.loc[macro.release_date <= end].iloc[-1]
        assert pd.notna(a.vacancy_change_pp)
        return {'origin': origin, 'quarter': str(a.quarter), 'release_date': a.release_date.strftime('%Y-%m-%d'), 'vacancy_pct': float(a.vacancy_pct), 'previous_vacancy_pct': float(a.previous_vacancy_pct), 'vacancy_change_pp': float(a.vacancy_change_pp), 'x': float(a.vacancy_change_pp) / 3}

    def kernel(h):
        h = np.asarray(h, dtype=float)
        return (1 - 0.9 ** h) / (1 - 0.9)
    (OUT / 'calibration_indices').mkdir()
    (OUT / 'calibration_tables').mkdir()
    index_cache = {}
    calibration = {}
    fit_count = 0

    def obtain_index(origin):
        nonlocal fit_count
        if origin in index_cache:
            return index_cache[origin]
        deadline()
        if origin in ['2022-09', '2023-05']:
            path = PRIOR / f"index_{origin.replace('-', '_')}.joblib"
            index = joblib.load(path)
            origin_source = 'verified_incumbent_saved_index'
        elif origin == '2024-09':
            path = PRIOR / 'final_time_index.joblib'
            index = joblib.load(path)
            origin_source = 'verified_incumbent_saved_index'
        else:
            hist = rows.loc[rows.month <= origin].copy()
            record('rolling_index_fit_started', origin=origin, fit_rows=len(hist))
            index = fit_index(hist)
            fit_count += 1
            path = OUT / 'calibration_indices' / f'{origin}.joblib'
            joblib.dump(index, path)
            origin_source = 'new_history_only_index'
        assert index['fit_end'] == origin and index['months'][-1] == origin
        assert index['fit_rows'] == int((rows.month <= origin).sum())
        index_cache[origin] = (index, path, origin_source)
        return index_cache[origin]

    def evaluate_proxy(index, origin, end):
        future = rows.loc[(rows.month > origin) & (rows.month <= end)].copy()
        alpha = index['property_effect'].reindex(pd.MultiIndex.from_frame(future[KEYS])).to_numpy(float)
        future['eligible'] = np.isfinite(alpha)
        future['observed_log_component'] = future.log_rent.to_numpy() - alpha
        table = future.groupby('month').agg(total_rows=('eligible', 'size'), eligible_rows=('eligible', 'sum'), observed_log_component=('observed_log_component', 'mean')).reset_index()
        table['coverage'] = table.eligible_rows / table.total_rows
        table['origin'] = origin
        table['horizon'] = pd.PeriodIndex(table.month, freq='M').asi8 - pd.Period(origin, freq='M').ordinal
        assert np.array_equal(table.horizon, np.arange(1, len(table) + 1))
        table['origin_log_index'] = float(index['month_effect'][-1])
        table['baseline_log_index'] = forecast(index, table.horizon, 'damped')
        table['observed_index'] = 100 * np.exp(table.observed_log_component - table.origin_log_index)
        table['baseline_index'] = 100 * np.exp(table.baseline_log_index - table.origin_log_index)
        return table

    def fit_correction(origin):
        deadline()
        last = pd.Period(origin, freq='M') - 6
        origins = pd.period_range('2021-06', last, freq='M').astype(str)
        pieces = []
        for cal_origin in origins:
            deadline()
            if cal_origin not in calibration:
                index, path, kind = obtain_index(cal_origin)
                end = str(pd.Period(cal_origin, freq='M') + 6)
                t = evaluate_proxy(index, cal_origin, end)
                assert len(t) == 6 and t.month.max() <= origin
                a = available(cal_origin)
                t['x'] = a['x']
                t['quarter'] = a['quarter']
                t['release_date'] = a['release_date']
                t['z'] = a['x'] * kernel(t.horizon)
                t['error_log'] = t.observed_log_component - t.baseline_log_index
                t['index_path'] = str(asset_key(path))
                t['index_sha256'] = digest(path)
                t['index_source'] = kind
                assert np.isfinite(t[['z', 'error_log', 'observed_index', 'baseline_index']].to_numpy()).all()
                assert (t.coverage >= 0.7).all()
                t.to_csv(OUT / 'calibration_tables' / f'{cal_origin}.csv', index=False)
                calibration[cal_origin] = t
                record('calibration_origin_complete', origin=cal_origin, target_end=end, index_fits=fit_count)
            pieces.append(calibration[cal_origin])
        samples = pd.concat(pieces, ignore_index=True)
        assert samples.month.max() <= origin
        assert samples.groupby('origin').size().eq(6).all()
        zz = float(np.dot(samples.z, samples.z))
        ze = float(np.dot(samples.z, samples.error_log))
        unshrunk = ze / zz if zz > 0 else 0.0
        beta = min(0.0, unshrunk / 2)
        a = available(origin)
        slug = origin.replace('-', '_')
        samples.to_csv(OUT / f'coefficient_samples_{slug}.csv', index=False)
        coeff = {'origin': origin, 'beta': beta, 'unconstrained_beta': unshrunk, 'shrink_factor': 0.5, 'sign_constraint': 'beta<=0', 'sum_z_squared': zz, 'sum_z_error': ze, 'calibration_origins': len(origins), 'calibration_rows': len(samples), 'calibration_target_end': str(samples.month.max()), 'calibration_origin_start': str(origins[0]), 'calibration_origin_end': str(origins[-1]), 'latest_available_vacancy': a, 'protocol_sha256': protocol_hash, 'samples_sha256': digest(OUT / f'coefficient_samples_{slug}.csv')}
        write_json(OUT / f'coefficient_{slug}.json', coeff)
        record('coefficient_frozen', origin=origin, beta=beta, calibration_origins=len(origins), latest_vacancy_change_pp=a['vacancy_change_pp'])
        return coeff

    def verify_coefficient(coeff):
        origin = coeff['origin']
        slug = origin.replace('-', '_')
        path = OUT / f'coefficient_samples_{slug}.csv'
        assert digest(path) == coeff['samples_sha256']
        s = pd.read_csv(path)
        assert s.month.max() <= origin and s.groupby('origin').size().eq(6).all()
        for cal_origin, g in s.groupby('origin'):
            a = available(cal_origin)
            assert g.release_date.eq(a['release_date']).all()
            assert np.allclose(g.x, a['x'], rtol=0, atol=1e-14)
            assert np.allclose(g.z, g.x * kernel(g.horizon), rtol=0, atol=1e-13)
            assert np.allclose(g.error_log, g.observed_log_component - g.baseline_log_index, rtol=0, atol=1e-13)
        zz = float(np.dot(s.z, s.z))
        ze = float(np.dot(s.z, s.error_log))
        expected = min(0.0, ze / (2 * zz)) if zz else 0.0
        assert abs(expected - coeff['beta']) < 1e-12
        return {'origin': origin, 'beta': coeff['beta'], 'calibration_origins': coeff['calibration_origins'], 'publication_alignment_and_cutoff_verified': True}
    RESULT.update(status='running', candidate=protocol['candidate'], catboost_fits=0, catboost_inference_calls=0, search=None, confirm=None, candidate_confirm_evaluations=0, baseline=protocol['baseline'], current_best=protocol['current_best'], original_split_unchanged=True, source_labels_unchanged=True, protocol_sha256=protocol_hash)
    record('preflight_complete', fit_rows=len(rows), frozen_files=len(protocol['frozen_sha256']))
    tables = []
    coefficients = []
    for origin, end, horizon in ORIGINS:
        coeff = fit_correction(origin)
        coefficients.append(coeff)
        index, _, _ = obtain_index(origin)
        t = evaluate_proxy(index, origin, end)
        assert len(t) == horizon
        prior_t = pd.read_csv(PRIOR / f"backtest_{origin.replace('-', '_')}.csv")
        assert np.array_equal(t.month, prior_t.month)
        assert np.allclose(t.observed_index, prior_t.observed_index, rtol=0, atol=1e-09)
        assert np.allclose(t.baseline_index, prior_t.damped_index, rtol=0, atol=1e-09)
        adjustment = coeff['beta'] * coeff['latest_available_vacancy']['x'] * kernel(t.horizon)
        t['candidate_log_index'] = t.baseline_log_index + adjustment
        t['candidate_index'] = 100 * np.exp(t.candidate_log_index - t.origin_log_index)
        t['beta'] = coeff['beta']
        t['origin_vacancy_x'] = coeff['latest_available_vacancy']['x']
        assert np.isfinite(t.candidate_index).all()
        t.to_csv(OUT / f"historical_{origin.replace('-', '_')}.csv", index=False)
        tables.append(t)
        record('historical_origin_complete', origin=origin, beta=coeff['beta'], horizons=horizon)
    combined = pd.concat(tables, ignore_index=True)
    combined.to_csv(OUT / 'historical_predictions.csv', index=False)
    historical = history_gate(combined)
    write_json(OUT / 'historical_metrics.json', historical)
    time_selection = {'epoch': time.time(), 'passed': historical['passed'], 'conditions': historical['conditions'], 'protocol_sha256': protocol_hash, 'historical_metrics_sha256': digest(OUT / 'historical_metrics.json'), 'coefficient_sha256': {c['origin']: digest(OUT / f"coefficient_{c['origin'].replace('-', '_')}.json") for c in coefficients}, 'search_evaluations_before_freeze': 0, 'confirm_evaluations_before_freeze': 0}
    write_json(OUT / 'time_selection_frozen.json', time_selection)
    RESULT.update(time_gate_passed=historical['passed'], historical_metrics=historical, time_selection_sha256=digest(OUT / 'time_selection_frozen.json'), historical_predictions_csv=str(OUT / 'historical_predictions.csv'))
    record('historical_gate_frozen', passed=historical['passed'], conditions=historical['conditions'])

    def score(name, coeff):
        deadline()
        start, end = (132802, 141076) if name == 'search' else (141076, 150000)
        labels = pd.read_csv(support_path(Path('Data/train.csv')), nrows=end, usecols=['RENT_APPROVAL_DATE', 'MONTHLY_RENT']).iloc[start:end]
        p = pd.read_csv(PRIOR / f'{name}_predictions.csv')
        assert len(p) == end - start and np.array_equal(p.source_row_id, np.arange(start, end))
        assert p.source_row_id.is_unique and np.array_equal(p.truth, labels.MONTHLY_RENT)
        assert np.array_equal(p.date, labels.RENT_APPROVAL_DATE)
        assert np.allclose(p.prediction, p.time_scale * p.normalized_prediction, rtol=0, atol=1e-08)
        h = pd.PeriodIndex(p.date, freq='M').asi8 - pd.Period('2024-09', freq='M').ordinal
        index, _, _ = obtain_index('2024-09')
        old_scale = np.exp(forecast(index, h, 'damped') - index['month_effect'][-1])
        assert np.allclose(p.time_scale, old_scale, rtol=0, atol=1e-12)
        new_scale = old_scale * np.exp(coeff['beta'] * coeff['latest_available_vacancy']['x'] * kernel(h))
        pred = p.normalized_prediction.to_numpy() * new_scale
        candidate = pd.DataFrame({'source_row_id': p.source_row_id, 'date': p.date, 'truth': p.truth, 'prediction': pred, 'time_scale': new_scale, 'normalized_prediction': p.normalized_prediction, 'previous_prediction': p.prediction, 'previous_time_scale': p.time_scale})
        assert np.isfinite(candidate.select_dtypes(include=[np.number]).to_numpy()).all()
        error = pred - p.truth.to_numpy()
        metrics = {'rows': len(p), 'rmse': float(np.sqrt(np.mean(error ** 2))), 'mae': float(np.mean(np.abs(error))), 'bias_pred_minus_truth': float(error.mean()), 'by_month': {}}
        for m in sorted(p.date.unique()):
            mask = p.date.eq(m).to_numpy()
            metrics['by_month'][str(m)] = {'rows': int(mask.sum()), 'rmse': float(np.sqrt(np.mean(error[mask] ** 2)))}
        metrics['delta_from_current_best'] = metrics['rmse'] - protocol['current_best'][name + '_rmse']
        metrics['delta_from_fixed_baseline'] = metrics['rmse'] - protocol['baseline'][name + '_rmse']
        candidate.to_csv(OUT / f'{name}_predictions.csv', index=False)
        write_json(OUT / ('search_metrics.json' if name == 'search' else 'confirmation_metrics.json'), metrics)
        saved = pd.read_csv(OUT / f'{name}_predictions.csv')
        assert np.array_equal(saved.source_row_id, np.arange(start, end))
        assert np.array_equal(saved.truth, labels.MONTHLY_RENT)
        assert abs(float(np.sqrt(np.mean((saved.prediction - saved.truth) ** 2))) - metrics['rmse']) < 1e-09
        assert np.allclose(saved.prediction, saved.normalized_prediction * saved.time_scale, rtol=0, atol=1e-08)
        RESULT[name] = metrics
        RESULT[name + '_predictions_csv'] = str(OUT / f'{name}_predictions.csv')
        record(name + '_complete', rmse=metrics['rmse'], rows=len(p))
    if historical['passed']:
        coeff = fit_correction('2024-09')
        coefficients.append(coeff)
        score('search', coeff)
        passed = RESULT['search']['rmse'] < protocol['search_gate_rmse']
        selection = {'epoch': time.time(), 'passed_search_gate': passed, 'threshold': protocol['search_gate_rmse'], 'search_rmse': RESULT['search']['rmse'], 'protocol_sha256': protocol_hash, 'coefficient_sha256': digest(OUT / 'coefficient_2024_09.json'), 'search_predictions_sha256': digest(OUT / 'search_predictions.csv'), 'time_selection_sha256': RESULT['time_selection_sha256'], 'confirm_evaluations_before_freeze': 0}
        write_json(OUT / 'selection_frozen.json', selection)
        RESULT.update(passed_search_gate=passed, selection_sha256=digest(OUT / 'selection_frozen.json'))
        record('search_gate_frozen', passed=passed, rmse=RESULT['search']['rmse'], threshold=444.4691)
        if passed:
            deadline()
            with (OUT / 'confirm_started.json').open('x', encoding='utf-8') as f:
                json.dump({'epoch': time.time(), 'selection_sha256': RESULT['selection_sha256']}, f)
            record('confirm_evaluation_started')
            score('confirm', coeff)
            RESULT['candidate_confirm_evaluations'] = 1
            RESULT['conclusion'] = 'Candidate improved Search and Confirm.' if RESULT['confirm']['rmse'] < protocol['current_best']['confirm_rmse'] else 'Confirm regressed or tied; optimization not established. Do not change winner or tune.'
        else:
            RESULT['conclusion'] = 'Search did not beat 444.4691. Confirm not evaluated; no retry.'
    else:
        RESULT['conclusion'] = 'Historical long-horizon gate failed. Search and Confirm not evaluated; candidate rejected without retry.'
    deadline()
    checks = [verify_coefficient(c) for c in coefficients]
    assert history_gate(pd.read_csv(OUT / 'historical_predictions.csv'))['conditions'] == historical['conditions']
    check_frozen(protocol)
    assert digest(OUT / 'protocol.json') == protocol_hash
    assert digest(OUT / 'time_selection_frozen.json') == RESULT['time_selection_sha256']
    for c in coefficients:
        slug = c['origin'].replace('-', '_')
        if c['origin'] in time_selection['coefficient_sha256']:
            assert digest(OUT / f'coefficient_{slug}.json') == time_selection['coefficient_sha256'][c['origin']]
    if RESULT['confirm'] is None:
        assert not (OUT / 'confirm_started.json').exists() and (not (OUT / 'confirm_predictions.csv').exists())
    if RESULT['search'] is None:
        assert not (OUT / 'search_predictions.csv').exists() and (not (OUT / 'selection_frozen.json').exists())
    verification = {'status': 'passed', 'scope': 'Saved coefficients, release dates, raw source labels, predictions and hashes; no refit or new model inference', 'coefficient_checks': checks, 'frozen_file_count': len(protocol['frozen_sha256']), 'all_frozen_files_unchanged': True, 'catboost_fits': 0, 'catboost_inference_calls': 0, 'confirm_evaluations': RESULT['candidate_confirm_evaluations'], 'original_rows': {'train': 132802, 'search': 8274, 'confirm': 8924}, 'baseline_labels_and_split_unchanged': True}
    write_json(OUT / 'completion_verification.json', verification)
    deadline()
    RESULT.update(status='completed', exit_code=0, elapsed_seconds=time.time() - budget['started_epoch'], rolling_index_fits=fit_count, calibration_origin_count=len(calibration), frozen_files_unchanged=True, completion_verification_sha256=digest(OUT / 'completion_verification.json'))
    write_json(OUT / 'results.json', RESULT)
    record('completed', elapsed_seconds=RESULT['elapsed_seconds'], conclusion=RESULT['conclusion'])
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'run'])
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare()
    else:
        try:
            run()
        except BaseException as exc:
            last = LAST_STAGE
            traceback.print_exc()
            failure = {'status': 'failed', 'error': repr(exc), 'last_successful_stage': LAST_SUCCESSFUL_STAGE, 'last_entered_stage': last, 'exit_code': 1, 'partial_results': RESULT, 'pid': os.getpid(), 'epoch': time.time()}
            write_json(OUT / 'failure.json', failure, required=False)
            record('failed', error=repr(exc), last_successful_stage=LAST_SUCCESSFUL_STAGE, last_entered_stage=last, exit_code=1)
            raise SystemExit(1)
