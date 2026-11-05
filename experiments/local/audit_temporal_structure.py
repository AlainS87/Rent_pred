"""Descriptive time audit only: no fitting, prediction, or Confirm scoring."""
from experiments.local.bootstrap import Path, CODE_DIR, DATA_ROOT, SUPPORT_ROOT, ARTIFACT_ROOT, asset_path, asset_key, support_path
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parent
OUT = asset_path(Path('temporal_structure_audit_01'))
SOURCE = asset_path(Path('support/Data'))

def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def window(frame, begin, end):
    return frame.loc[frame.month.between(begin, end)]

def matched(first, second, minimum):
    keys = ['TOWN', 'STREET', 'BLOCK', 'FLAT_TYPE', 'FLAT_MODEL', 'FLOOR_AREA_SQM']
    a = first.groupby(keys, observed=True).MONTHLY_RENT.agg(['median', 'size'])
    b = second.groupby(keys, observed=True).MONTHLY_RENT.agg(['median', 'size'])
    joined = a.join(b, how='inner', lsuffix='_a', rsuffix='_b')
    joined = joined.loc[(joined.size_a >= minimum) & (joined.size_b >= minimum)].copy()
    growth = joined.median_b / joined.median_a - 1
    weights = np.minimum(joined.size_a, joined.size_b)
    return {'minimum_records_per_group_per_period': minimum, 'matched_groups': len(joined), 'first_period_matched_rows': int(joined.size_a.sum()), 'second_period_matched_rows': int(joined.size_b.sum()), 'median_group_rent_change': float((joined.median_b - joined.median_a).median()), 'median_group_growth_pct': float(growth.median() * 100), 'matched_count_weighted_mean_growth_pct': float(np.average(growth, weights=weights) * 100), 'growth_q25_pct': float(growth.quantile(0.25) * 100), 'growth_q75_pct': float(growth.quantile(0.75) * 100), 'higher_rent_fraction': float((growth > 0).mean())}

def main():
    if OUT.exists():
        assert not read(OUT / 'audit.json')['training_performed']
    paths = [support_path(Path('train.csv')), support_path(Path('test.csv')), support_path(Path('auxiliary_features/train_features.csv')), support_path(Path('auxiliary_features/test_features.csv')), asset_path(Path('baseline_alignment/regional/rnn/h16_l3/metrics.json')), asset_path(Path('baseline_alignment/regional/rnn/h16_l3/config.json')), asset_path(Path('global_full_address_01/search_predictions.csv')), asset_path(Path('global_rnn_residual_01/search_predictions.csv')), asset_path(Path('global_full_address_01/protocol.json')), asset_path(Path('global_rnn_residual_01/protocol.json'))]
    before = {str(asset_key(path)): digest(path) for path in paths}
    fit = pd.read_csv(support_path(Path('train.csv')), nrows=132802, usecols=['RENT_APPROVAL_DATE', 'MONTHLY_RENT', 'TOWN', 'STREET', 'BLOCK', 'FLAT_TYPE', 'FLAT_MODEL', 'FLOOR_AREA_SQM'], dtype={'TOWN': str, 'STREET': str, 'BLOCK': str, 'FLAT_TYPE': str, 'FLAT_MODEL': str})
    assert len(fit) == 132802
    fit['month'] = fit.RENT_APPROVAL_DATE.astype(str).str[:7]
    assert fit.month.max() == '2024-09'
    raw_type_labels = {year: sorted(part.FLAT_TYPE.unique()) for year, part in fit.groupby(fit.month.str[:4])}
    from experiments.local._support.src.features.source_fields import flat_type_key
    from experiments.local._support.src.features.keys import norm_text
    fit['FLAT_TYPE'] = flat_type_key(fit.FLAT_TYPE)
    for column in ['TOWN', 'STREET', 'BLOCK', 'FLAT_MODEL']:
        fit[column] = norm_text(fit[column])
    fit['unit_rent'] = fit.MONTHLY_RENT / fit.FLOOR_AREA_SQM
    fit['quarter'] = pd.PeriodIndex(fit.month, freq='M').asfreq('Q').astype(str)
    monthly = fit.groupby('month').agg(rows=('MONTHLY_RENT', 'size'), mean_rent=('MONTHLY_RENT', 'mean'), median_rent=('MONTHLY_RENT', 'median'), mean_unit_rent=('unit_rent', 'mean'), median_unit_rent=('unit_rent', 'median'), mean_area=('FLOOR_AREA_SQM', 'mean')).reset_index()
    quarters = fit.groupby('quarter').agg(rows=('MONTHLY_RENT', 'size'), mean_rent=('MONTHLY_RENT', 'mean'), median_rent=('MONTHLY_RENT', 'median'), mean_unit_rent=('unit_rent', 'mean'), mean_area=('FLOOR_AREA_SQM', 'mean')).reset_index()
    comparisons = [('2021 Jan-Sep to 2022 Jan-Sep', '2021-01', '2021-09', '2022-01', '2022-09'), ('2022 Jan-Sep to 2023 Jan-Sep', '2022-01', '2022-09', '2023-01', '2023-09'), ('2023 Jan-Sep to 2024 Jan-Sep', '2023-01', '2023-09', '2024-01', '2024-09'), ('2023 Apr-Sep to 2023 Oct-2024 Mar', '2023-04', '2023-09', '2023-10', '2024-03'), ('2023 Oct-2024 Mar to 2024 Apr-Sep', '2023-10', '2024-03', '2024-04', '2024-09')]
    matched_results = []
    for name, start_a, end_a, start_b, end_b in comparisons:
        a, b = (window(fit, start_a, end_a), window(fit, start_b, end_b))
        matched_results.append({'comparison': name, 'raw_mean_rent_growth_pct': float((b.MONTHLY_RENT.mean() / a.MONTHLY_RENT.mean() - 1) * 100), 'raw_mean_unit_rent_growth_pct': float((b.unit_rent.mean() / a.unit_rent.mean() - 1) * 100), 'first_rows': len(a), 'second_rows': len(b), 'matching_sensitivity': [matched(a, b, minimum) for minimum in [1, 2, 3]]})
    stratified = []
    for year in [2021, 2022, 2023]:
        a = window(fit, f'{year}-01', f'{year}-09')
        b = window(fit, f'{year + 1}-01', f'{year + 1}-09')
        keys = ['TOWN', 'FLAT_TYPE']
        aa = a.groupby(keys).MONTHLY_RENT.agg(['mean', 'size'])
        bb = b.groupby(keys).MONTHLY_RENT.agg(['mean', 'size'])
        pair = aa.join(bb, how='inner', lsuffix='_a', rsuffix='_b')
        wa, wb = (pair.size_a / pair.size_a.sum(), pair.size_b / pair.size_b.sum())
        within = float(((wa + wb) / 2 * (pair.mean_b - pair.mean_a)).sum())
        mix = float(((pair.mean_a + pair.mean_b) / 2 * (wb - wa)).sum())
        total = float((wb * pair.mean_b).sum() - (wa * pair.mean_a).sum())
        assert abs(total - within - mix) < 1e-08
        stratified.append({'years': [year, year + 1], 'common_town_type_groups': len(pair), 'mean_rent_change_common_support': total, 'within_town_type_component': within, 'town_type_mix_component': mix, 'caution': 'Within-town-type component still includes unobserved house and tenancy mix changes.'})
    levels = ['coe_c_premium_lag1', 'bank_basket_adj_close_lag1', 'property_basket_adj_close_lag1']
    macro = {}
    macro_monthly = {}
    for stage, limit in [('train', 132802), ('test', None)]:
        dates = pd.read_csv(support_path(Path(f'{stage}.csv')), usecols=['RENT_APPROVAL_DATE'], nrows=limit)
        values = pd.read_csv(support_path(Path(f'auxiliary_features/{stage}_features.csv')), usecols=levels, nrows=limit)
        values['month'] = dates.RENT_APPROVAL_DATE.astype(str).str[:7]
        grouped = values.groupby('month')[levels].first()
        counts = values.groupby('month')[levels].nunique(dropna=True)
        assert (counts <= 1).all().all()
        macro_monthly[stage] = grouped.reset_index()
        macro[stage] = {name: {'missing_rows': int(values[name].isna().sum()), 'missing_months': list(grouped.index[grouped[name].isna()]), 'min': float(values[name].min()), 'max': float(values[name].max()), 'distinct_monthly_values': int(grouped[name].nunique())} for name in levels}
    for name in levels:
        minimum, maximum = (macro['train'][name]['min'], macro['train'][name]['max'])
        values = macro_monthly['test'][name]
        macro['test'][name]['months_outside_training_range'] = macro_monthly['test'].loc[(values < minimum) | (values > maximum), 'month'].tolist()
    old = pd.read_csv(asset_path(Path('global_full_address_01/search_predictions.csv')))
    new = pd.read_csv(asset_path(Path('global_rnn_residual_01/search_predictions.csv')))
    assert np.array_equal(old.source_row_id, new.source_row_id) and np.array_equal(old.truth, new.truth)
    assert np.array_equal(old.date, new.date)
    search_info = pd.read_csv(support_path(Path('train.csv')), nrows=141076, usecols=['TOWN', 'RENT_APPROVAL_DATE']).iloc[132802:].reset_index(drop=True)
    eo, en = (old.prediction.to_numpy() - old.truth.to_numpy(), new.prediction.to_numpy() - new.truth.to_numpy())
    worst = np.zeros(len(old), bool)
    worst[np.argsort(np.abs(eo))[-828:]] = True
    comparisons_search = {}
    for name, mask in {'all': np.ones(len(old), bool), 'prior_worst_10pct': worst, 'other_90pct': ~worst}.items():
        comparisons_search[name] = {'rows': int(mask.sum()), 'direct_rmse': float(np.sqrt(np.mean(eo[mask] ** 2))), 'rnn_residual_rmse': float(np.sqrt(np.mean(en[mask] ** 2))), 'delta_sse': float(np.sum(en[mask] ** 2 - eo[mask] ** 2))}
    for label, error in [('direct', eo), ('rnn_residual', en)]:
        tmp = search_info.copy()
        tmp['month'] = tmp.RENT_APPROVAL_DATE.astype(str).str[:7]
        tmp['error'] = error
        cells = tmp.groupby(['TOWN', 'month']).error.agg(['mean', 'size'])
        comparisons_search[label] = {'rmse': float(np.sqrt(np.mean(error ** 2))), 'mae': float(np.mean(np.abs(error))), 'bias': float(error.mean()), 'overall_bias_squared': float(error.mean() ** 2), 'town_month_bias_mse_component': float(np.sum(cells['mean'] ** 2 * cells['size']) / len(error)), 'total_mse': float(np.mean(error ** 2))}
    rnn_metrics = read(asset_path(Path('baseline_alignment/regional/rnn/h16_l3/metrics.json')))
    first, last = (rnn_metrics['forward_unit_2025_04'], rnn_metrics['forward_unit_2026_07'])
    report = {'training_performed': False, 'new_model_inference': False, 'confirm_scored': False, 'rental_labels_scope': 'Original fit rows for time analysis; existing saved Search CSVs for last-candidate audit', 'fit_rows': len(fit), 'fit_distinct_months': int(fit.month.nunique()), 'raw_flat_type_labels_by_year': raw_type_labels, 'flat_type_labels_normalized_with_existing_support_function': True, 'matched_group_definition': 'Same town, street, block, flat type, flat model, and exact floor area; not same unit identifier', 'matched_changes': matched_results, 'town_type_composition_decomposition': stratified, 'quarterly_training_statistics': quarters.to_dict(orient='records'), 'macro_input_coverage': macro, 'macro_timing_caution': 'One-month lag is not known at a fixed forecast origin for distant future months; dataset supplies those inputs, original eligibility/provenance must be distinguished from operational ex-ante forecasting.', 'saved_original_rnn_forward_endpoints': {'aggregation': 'Unweighted mean of 26 town unit-rent predictions', '2025-04': first, '2026-07': last, 'change_pct': (last / first - 1) * 100, 'caution': 'Endpoints alone do not establish each town path or why the model is almost flat. These are predictions, not observed future rents.'}, 'last_candidate_existing_search_audit': comparisons_search, 'supplementary_long_horizon_backtest_feasibility_only': [{'origin': '2022-09', 'training_months_available': 21, 'horizon_16_end': '2024-01', 'horizon_22_end': '2024-07'}, {'origin': '2022-11', 'training_months_available': 23, 'horizon_16_end': '2024-03', 'horizon_22_end': '2024-09'}, {'origin': '2023-05', 'training_months_available': 29, 'horizon_16_end': '2024-09', 'horizon_22_end': None}], 'backtest_cautions': ['Not run; requires prospective approval and locked rules.', 'Keep original fit/Search/Confirm unchanged; these would be supplemental folds inside original fit only.', 'Current saved RNN/CatBoost are trained through 2024-09 and cannot be evaluated as if trained at earlier origins.', 'Overlapping historical forecast windows are not independent evidence.', 'Cheap monthly-index backtests do not replace end-to-end monthly-rent RMSE verification.'], 'input_sha256': before}
    assert before == {str(asset_key(path)): digest(path) for path in paths}
    report['all_read_inputs_unchanged'] = True
    OUT.mkdir(exist_ok=True)
    monthly.to_csv(OUT / 'training_monthly_statistics.csv', index=False)
    for stage, table in macro_monthly.items():
        table.to_csv(OUT / f'{stage}_monthly_macro_inputs.csv', index=False)
    (OUT / 'audit.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'audit': str(OUT / 'audit.json'), 'matched': [{'comparison': row['comparison'], **row['matching_sensitivity'][1]} for row in matched_results], 'macro': macro, 'rnn_endpoints': report['saved_original_rnn_forward_endpoints'], 'search': comparisons_search, 'composition': stratified}, indent=2))
if __name__ == '__main__':
    import argparse as _cli_argparse
    _cli_argparse.ArgumentParser(description=__doc__).parse_args()
    main()
