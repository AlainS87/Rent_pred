"""Saved-artifact attribution only. No estimator fitting, predictions, or tuning."""
from experiments.local.bootstrap import Path, CODE_DIR, DATA_ROOT, SUPPORT_ROOT, ARTIFACT_ROOT, asset_path, asset_key, support_path
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent
SOURCE = asset_path(Path('support'))
TRIAL = asset_path(Path('temporal_index_trial_01'))
OUT = asset_path(Path('time_overestimate_diagnosis_01'))
import joblib
import numpy as np
import pandas as pd
from experiments.local.temporal_index import KEYS, prepare_rows

def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def observed_components(index, history, future):
    """Retrospective outcome summaries; not deployable corrections."""
    hk = pd.MultiIndex.from_frame(history[KEYS])
    fk = pd.MultiIndex.from_frame(future[KEYS])
    hist = history.loc[hk.isin(index['property_effect'].index)].copy()
    work = future.copy()
    work['property_effect'] = index['property_effect'].reindex(fk).to_numpy(float)
    work = work.loc[work.property_effect.notna()].copy()
    work['observed_time_component'] = work.log_rent - work.property_effect
    work['stratum'] = work.TOWN.astype(str) + '|' + work.FLAT_TYPE.astype(str)
    hist['stratum'] = hist.TOWN.astype(str) + '|' + hist.FLAT_TYPE.astype(str)
    hist_weights = hist.stratum.value_counts(normalize=True)
    reference = float(index['month_effect'][-1])
    output = []
    for month, part in work.groupby('month'):
        property_means = part.groupby(KEYS, observed=True).observed_time_component.mean()
        strata = part.groupby('stratum').observed_time_component.mean()
        weights = hist_weights.reindex(strata.index)
        assert weights.notna().all()
        fixed = float(np.average(strata.to_numpy(), weights=weights.to_numpy()))
        output.append({'month': month, 'eligible_rows': len(part), 'total_rows': int((future.month == month).sum()), 'coverage': len(part) / int((future.month == month).sum()), 'reference_log_effect': reference, 'row_weighted_observed_index': float(100 * np.exp(part.observed_time_component.mean() - reference)), 'equal_property_observed_index': float(100 * np.exp(property_means.mean() - reference)), 'fixed_town_type_observed_index': float(100 * np.exp(fixed - reference)), 'fixed_strata_historical_mass_present': float(weights.sum())})
    return (work, pd.DataFrame(output))

def signed_summary(frame, forecast_col, observed_col):
    error = frame[forecast_col].to_numpy() - frame[observed_col].to_numpy()
    return {'months': len(frame), 'bias_index_points': float(error.mean()), 'rmse_index_points': float(np.sqrt(np.mean(error ** 2))), 'overprediction_months': int((error > 0).sum()), 'mean_relative_overprediction_pct': float(np.mean(frame[forecast_col] / frame[observed_col] - 1) * 100)}

def main():
    assert not OUT.exists(), 'Refusing to overwrite this diagnosis'
    original_protocol = read(TRIAL / 'protocol.json')
    frozen = dict(original_protocol['frozen_sha256'])
    for name in ['protocol.json', 'results.json', 'completion_verification.json', 'model.joblib', 'preprocessor.joblib', 'final_time_index.joblib', 'index_2022_09.joblib', 'index_2023_05.joblib', 'backtest_2022_09.csv', 'backtest_2023_05.csv', 'historical_metrics.json', 'time_selection_frozen.json', 'selection_frozen.json', 'search_predictions.csv', 'confirm_predictions.csv', 'search_metrics.json', 'confirmation_metrics.json']:
        relative = 'temporal_index_trial_01/' + name
        frozen[relative] = digest(asset_path(Path(relative)))
    for name, expected in frozen.items():
        assert digest(asset_path(Path(name))) == expected, name
    raw = pd.read_csv(support_path(Path('Data/train.csv')), usecols=['RENT_APPROVAL_DATE', 'MONTHLY_RENT', *KEYS], dtype={key: str for key in KEYS if key != 'FLOOR_AREA_SQM'})
    raw['source_row_id'] = np.arange(len(raw))
    rows = prepare_rows(raw)
    historical = []
    historical_tables = []
    for origin, end in [('2022-09', '2024-07'), ('2023-05', '2024-09')]:
        slug = origin.replace('-', '_')
        index = joblib.load(TRIAL / f'index_{slug}.joblib')
        history = rows.loc[rows.month <= origin]
        future = rows.loc[(rows.month > origin) & (rows.month <= end)]
        _, observed = observed_components(index, history, future)
        saved = pd.read_csv(TRIAL / f'backtest_{slug}.csv')
        combined = saved.merge(observed, on='month', how='left', validate='one_to_one', suffixes=('', '_audit'))
        assert np.allclose(combined.observed_index, combined.row_weighted_observed_index, rtol=0, atol=1e-09)
        groups = {}
        for label, mask in {'h1_6': combined.horizon <= 6, 'h7_16': combined.horizon.between(7, 16), 'h17_plus': combined.horizon >= 17, 'h7_plus': combined.horizon >= 7}.items():
            part = combined.loc[mask]
            if not len(part):
                continue
            groups[label] = {name: signed_summary(part, 'damped_index', name) for name in ['row_weighted_observed_index', 'equal_property_observed_index', 'fixed_town_type_observed_index']}
        growth = []
        for start, end_h in [(0, 6), (6, 16)] + ([(16, 22)] if len(combined) == 22 else []):
            for name in ['observed_index', 'damped_index']:
                first = 100.0 if start == 0 else float(combined.loc[combined.horizon == start, name].iloc[0])
                last = float(combined.loc[combined.horizon == end_h, name].iloc[0])
                growth.append({'from_horizon': start, 'to_horizon': end_h, 'series': name, 'cumulative_growth_pct': (last / first - 1) * 100, 'average_monthly_log_growth_pct': np.log(last / first) / (end_h - start) * 100})
        historical.append({'origin': origin, 'slope_log_index_per_month': index['slope'], 'phi': index['damping'], 'growth_half_life_months': float(np.log(0.5) / np.log(index['damping'])), 'index_forecast_only_no_catboost': True, 'signed_errors': groups, 'growth_segments': growth, 'last_month': combined.iloc[-1][['month', 'horizon', 'observed_index', 'damped_index']].to_dict(), 'minimum_fixed_strata_historical_mass_present': float(combined.fixed_strata_historical_mass_present.min())})
        combined['origin'] = origin
        historical_tables.append(combined)
    index = joblib.load(TRIAL / 'final_time_index.joblib')
    history = rows.loc[rows.month <= '2024-09']
    future = rows.loc[rows.month.between('2024-10', '2025-03')]
    eligible, observed = observed_components(index, history, future)
    proxy = observed.set_index('month').row_weighted_observed_index / 100
    predictions = []
    for stage, start, end in [('search', 132802, 141076), ('confirm', 141076, 150000)]:
        saved = pd.read_csv(TRIAL / f'{stage}_predictions.csv')
        assert np.array_equal(saved.source_row_id, np.arange(start, end))
        assert np.array_equal(saved.truth, raw.iloc[start:end].MONTHLY_RENT)
        assert np.allclose(saved.prediction, saved.normalized_prediction * saved.time_scale, rtol=0, atol=1e-08)
        saved['stage'] = stage
        predictions.append(saved)
    predictions = pd.concat(predictions, ignore_index=True)
    predictions['month'] = predictions.date.str[:7]
    predictions['error'] = predictions.prediction - predictions.truth
    original_summary = {stage: {'rows': len(part), 'rmse': float(np.sqrt(np.mean(part.error ** 2))), 'bias': float(part.error.mean()), 'mae': float(part.error.abs().mean())} for stage, part in predictions.groupby('stage')}
    eligibility = set(eligible.source_row_id)
    primary = predictions.loc[predictions.source_row_id.isin(eligibility)].copy()
    primary['observed_index_proxy'] = primary.month.map(proxy)
    primary['time_component'] = primary.normalized_prediction * (primary.time_scale - primary.observed_index_proxy)
    primary['remaining_component'] = primary.error - primary.time_component
    assert np.allclose(primary.error, primary.time_component + primary.remaining_component, rtol=0, atol=1e-09)
    period_components = {}
    for stage, part in primary.groupby('stage'):
        error = part.error.to_numpy()
        time_component = part.time_component.to_numpy()
        remainder = part.remaining_component.to_numpy()
        e2 = float(np.mean(error ** 2))
        t2 = float(np.mean(time_component ** 2))
        r2 = float(np.mean(remainder ** 2))
        cross = float(2 * np.mean(time_component * remainder))
        assert abs(e2 - t2 - r2 - cross) < 1e-08
        period_components[stage] = {'eligible_rows': len(part), 'coverage': len(part) / original_summary[stage]['rows'], 'original_mean_error_on_eligible_rows': float(error.mean()), 'time_proxy_component_mean': float(time_component.mean()), 'remaining_component_mean': float(remainder.mean()), 'mse_identity': {'original_mse': e2, 'time_proxy_term': t2, 'remaining_term': r2, 'cross_term': cross}, 'note': 'Components are correlated and proxy-dependent; squared terms are not additive attribution percentages.'}
    monthly = []
    for month, part in primary.groupby('month'):
        assert part.time_scale.nunique() == 1
        proxy_row = observed.loc[observed.month == month].iloc[0]
        variants = {}
        for column in ['row_weighted_observed_index', 'equal_property_observed_index', 'fixed_town_type_observed_index']:
            value = float(proxy_row[column] / 100)
            time_part = part.normalized_prediction.to_numpy() * (part.time_scale.to_numpy() - value)
            variants[column] = {'observed_scale_proxy': value, 'mean_time_component': float(time_part.mean()), 'mean_remaining_component': float(np.mean(part.error - time_part))}
        monthly.append({'month': month, 'stage': part.stage.iloc[0], 'horizon': int(pd.Period(month, 'M').ordinal - pd.Period('2024-09', 'M').ordinal), 'eligible_rows': len(part), 'all_rows': int((predictions.month == month).sum()), 'forecast_scale': float(part.time_scale.iloc[0]), 'observed_scale_proxy': float(proxy_row.row_weighted_observed_index / 100), 'mean_error_eligible': float(part.error.mean()), 'mean_time_component': float(part.time_component.mean()), 'mean_remaining_component': float(part.remaining_component.mean()), 'sensitivity': variants})
    rent_segments = {}
    for stage, part in predictions.groupby('stage'):
        masks = {'rent_at_most_1500': part.truth <= 1500, 'rent_1500_to_4000': part.truth.between(1500, 4000, inclusive='right'), 'rent_above_4000': part.truth > 4000}
        rent_segments[stage] = {}
        total_sse = float(np.sum(part.error ** 2))
        for name, mask in masks.items():
            segment = part.loc[mask]
            rent_segments[stage][name] = {'rows': len(segment), 'mean_error': float(segment.error.mean()), 'rmse': float(np.sqrt(np.mean(segment.error ** 2))), 'fraction_of_stage_sse': float(np.sum(segment.error ** 2) / total_sse)}
    model = joblib.load(TRIAL / 'model.joblib')
    borders = model.get_borders()
    month_borders = borders[model.feature_names_.index('approval_month_index')]
    metadata = {'approval_month_max_threshold': max(month_borders), 'fit_last_month_index': 44, 'all_search_confirm_and_test_months_above_max_month_split': True, 'interpretation': 'The numeric calendar-month predictor cannot create continued post-cutoff growth in the tree. Age and other inputs can still change the housing component.', 'historical_house_models_at_2022_09_or_2023_05_available_in_this_trial': False}
    forecast_scale = []
    for h in [1, 3, 6, 12, 16, 22]:
        log_change = index['slope'] * 0.9 * (1 - 0.9 ** h) / 0.1
        forecast_scale.append({'horizon': h, 'current_locked_index_scale': float(np.exp(log_change)), 'growth_pct': float(np.expm1(log_change) * 100)})
    report = {'training_performed': False, 'new_model_inference': False, 'new_candidate_evaluations': False, 'new_confirm_inference': False, 'parameters_changed': False, 'truth_or_original_split_changed': False, 'historical_time_only_diagnosis': historical, 'current_original_saved_prediction_metrics': original_summary, 'current_period_proxy_attribution': period_components, 'current_monthly_proxy_attribution': monthly, 'saved_prediction_error_by_rent_band': rent_segments, 'house_model_metadata': metadata, 'current_locked_index_future_scale': forecast_scale, 'interpretation_limits': ['The historical overshoot chart is an index-only forecast; it cannot have been caused by CatBoost.', 'Observed indices are property-adjusted log-rent proxies, not a known true market price index or exact same-unit repeat rents.', 'Current-period attribution uses realized outcomes for diagnosis only; no oracle-scaled alternative is scored, selected, or saved as a candidate.', 'The housing remainder includes property prediction errors, observation noise and proxy error; it is not a causal decomposition.', 'No end-to-end 16/22-month house-model backtest was conducted; actual distant-test overprediction remains unknown.', 'Only two overlapping historical origins have been examined; do not claim population-level or untouched-test certainty.'], 'protected_input_sha256': frozen}
    for name, expected in frozen.items():
        assert digest(asset_path(Path(name))) == expected, name
    report['all_protected_inputs_unchanged'] = True
    OUT.mkdir()
    pd.concat(historical_tables, ignore_index=True).to_csv(OUT / 'historical_index_error_audit.csv', index=False)
    observed.to_csv(OUT / 'current_observed_index_proxy.csv', index=False)
    primary[['source_row_id', 'stage', 'month', 'error', 'time_component', 'remaining_component']].to_csv(OUT / 'saved_prediction_error_decomposition.csv', index=False)
    (OUT / 'diagnosis.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'historical': [{'origin': r['origin'], 'signed_errors': r['signed_errors'], 'growth_segments': r['growth_segments']} for r in historical], 'current_period_components': period_components, 'current_monthly_components': monthly, 'current_locked_index_future_scale': forecast_scale, 'rent_segments': rent_segments, 'output': str(OUT / 'diagnosis.json')}, indent=2))
if __name__ == '__main__':
    import argparse as _cli_argparse
    _cli_argparse.ArgumentParser(description=__doc__).parse_args()
    main()
