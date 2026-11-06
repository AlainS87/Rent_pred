"""Descriptive macro co-movement only; no fitting, selection or prediction."""
from experiments.local.bootstrap import Path, CODE_DIR, DATA_ROOT, SUPPORT_ROOT, ARTIFACT_ROOT, asset_path, asset_key, support_path
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent
SOURCE = asset_path(Path('support'))
OUT = asset_path(Path('time_overestimate_diagnosis_01/external_signals'))
import joblib
import numpy as np
import pandas as pd

def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()

def association(x, y):
    pair = pd.concat([x.rename('x'), y.rename('y')], axis=1).dropna()
    if len(pair) < 5 or pair.x.std() == 0 or pair.y.std() == 0:
        return {'n': len(pair), 'pearson': None, 'spearman': None}
    return {'n': len(pair), 'pearson': float(pair.x.corr(pair.y)), 'spearman': float(pair.x.corr(pair.y, method='spearman'))}

def main():
    assert not OUT.exists()
    paths = [support_path(Path('Data/train.csv')), support_path(Path('Data/auxiliary_features/train_features.csv')), asset_path(Path('temporal_index_trial_01/final_time_index.joblib')), asset_path(Path('temporal_index_trial_01/model.joblib'))]
    hashes = {str(asset_key(p)): digest(p) for p in paths}
    names = ['bank_basket_adj_close_lag1', 'bank_basket_return_lag1', 'property_basket_adj_close_lag1', 'property_basket_return_lag1', 'coe_a_premium_lag1', 'coe_b_premium_lag1', 'coe_c_premium_lag1', 'coe_e_premium_lag1']
    dates = pd.read_csv(support_path(Path('Data/train.csv')), nrows=132802, usecols=['RENT_APPROVAL_DATE'])
    aux = pd.read_csv(support_path(Path('Data/auxiliary_features/train_features.csv')), nrows=132802, usecols=names)
    aux['month'] = dates.RENT_APPROVAL_DATE.astype(str).str[:7]
    assert aux.month.max() == '2024-09'
    assert (aux.groupby('month')[names].nunique() <= 1).all().all()
    monthly = aux.groupby('month')[names].first()
    index = joblib.load(asset_path(Path('temporal_index_trial_01/final_time_index.joblib')))
    effect = pd.Series(index['month_effect'], index=index['months'])
    assert list(monthly.index) == index['months']
    monthly['rental_log_index'] = effect
    monthly['rental_index'] = 100 * np.exp(effect - effect.iloc[0])
    monthly['rental_monthly_log_change'] = effect.diff()
    series = {'bank_stocks': {'level': monthly.bank_basket_adj_close_lag1, 'change': monthly.bank_basket_return_lag1, 'change_definition': 'Previous-month mean constituent stock return; original auxiliary feature'}, 'property_stocks': {'level': monthly.property_basket_adj_close_lag1, 'change': monthly.property_basket_return_lag1, 'change_definition': 'Previous-month mean constituent stock return; original auxiliary feature'}}
    for category in ['a', 'b', 'c', 'e']:
        level = monthly[f'coe_{category}_premium_lag1']
        series['coe_' + category] = {'level': level, 'change': np.log(level).diff(), 'change_definition': 'Log monthly change in previous-month premium'}
    periods = {'all_fit': np.ones(len(monthly), bool), '2021_2022': monthly.index <= '2022-12', '2023_2024': monthly.index >= '2023-01'}
    records = []
    for name, values in series.items():
        for lag in [1, 3, 6]:
            xlevel = np.log(values['level']).shift(lag - 1)
            xchange = values['change'].shift(lag - 1)
            for period, mask in periods.items():
                level = association(xlevel.loc[mask], monthly.rental_log_index.loc[mask])
                changes = association(xchange.loc[mask], monthly.rental_monthly_log_change.loc[mask])
                records.append({'signal': name, 'actual_signal_months_before_rent_month': lag, 'period': period, 'level_n': level['n'], 'level_pearson': level['pearson'], 'level_spearman': level['spearman'], 'change_n': changes['n'], 'change_pearson': changes['pearson'], 'change_spearman': changes['spearman']})
    table = pd.DataFrame(records)
    quarters = pd.period_range('2021Q1', '2024Q3', freq='Q').astype(str)
    hdb = pd.Series([142.2, 146.4, 150.6, 155.7, 159.5, 163.9, 168.1, 171.9, 173.6, 176.2, 178.5, 180.4, 183.7, 187.9, 192.9], index=quarters)
    quarter = monthly.copy()
    quarter['quarter'] = pd.PeriodIndex(quarter.index, freq='M').asfreq('Q').astype(str)
    quarter = quarter.groupby('quarter').rental_log_index.mean().to_frame()
    quarter['hdb_resale_price_index'] = hdb
    quarter['rental_log_change'] = quarter.rental_log_index.diff()
    quarter['hdb_log_change'] = np.log(quarter.hdb_resale_price_index).diff()
    housing = []
    for lag in [0, 1, 2]:
        housing.append({'hdb_quarters_before_rent_quarter': lag, 'levels': association(np.log(quarter.hdb_resale_price_index).shift(lag), quarter.rental_log_index), 'changes': association(quarter.hdb_log_change.shift(lag), quarter.rental_log_change), 'availability': 'Contemporaneous descriptive only' if lag == 0 else 'Prior-quarter final RPI is not known on the first day of the next quarter; release-date alignment still required' if lag == 1 else 'Conservative two-quarter lag; historical release vintage must still be checked before any predictive use'})
    report = {'training_performed': False, 'new_predictions': False, 'confirm_used_for_signal_screening': False, 'rental_label_scope': 'Existing index estimated from original fit through 2024-09 only', 'months': len(monthly), 'statistical_independent_time_points_not_rental_rows': len(monthly), 'fixed_screening_scope': {'signals': list(series), 'monthly_lags': [1, 3, 6], 'periods': list(periods), 'no_best_lag_selection': True}, 'macroeconomic_associations': records, 'hdb_quarterly_associations': housing, 'sources': {'stocks_and_coe': 'Frozen internal support Data/auxiliary_features/train_features.csv; original source code maps prior-month observations', 'hdb_rpi': 'https://www.hdb.gov.sg/-/media/doc/EAPG-CSC/3Q2024-RPI-Table.ashx', 'ura_pre_cutoff_rental_market': 'https://www.ura.gov.sg/news/media/pr24-38/', 'ura_release_timing_example': 'https://www.ura.gov.sg/news/media/pr24-50/', 'sora_definition_and_publication': 'https://www.mas.gov.sg/monetary-policy/sora'}, 'source_availability_notes': ['URA 2Q2024 release is dated 2024-07-26, before original 2024-09 fit cutoff; it reports a third consecutive private-rent quarterly decline.', 'URA 3Q2024 release is dated 2024-10-25 and cannot be treated as available at 2024-09.', 'HDB resale RPI is a quarterly sale-price index, not the HDB rental target; the Q3 2024 value is used here descriptively, not as a September-available predictor.', 'SORA is published the next business day at 09:00; no SORA time series has been loaded or statistically screened in this analysis.'], 'limitations': ['Level correlations can reflect shared nonstationary trends; changes and stability across subperiods matter.', 'Only 45 monthly observations and 15 quarters; correlations are exploratory, not causal or predictive proof.', 'Multiple signal/lag comparisons are all reported; no p-value, winner or training candidate is selected.', 'The existing rent index uses all original-fit observations. These retrospective correlations are not rolling-origin out-of-sample evidence.', 'A one-month lag is insufficient to provide inputs for a fixed-origin 16/22-month forecast unless future macro paths are supplied or independently forecast.', 'Private housing price/rent/supply is a related market; transfer to HDB rent must be tested.', 'Latest revised external histories are not historical publication vintages; release-time alignment is required before modeling.'], 'protected_inputs_sha256': hashes}
    assert hashes == {str(asset_key(p)): digest(p) for p in paths}
    report['all_protected_inputs_unchanged'] = True
    OUT.mkdir()
    monthly.to_csv(OUT / 'fit_monthly_macro_and_rent_index.csv')
    table.to_csv(OUT / 'macro_correlations_all_prespecified_lags.csv', index=False)
    quarter.to_csv(OUT / 'hdb_price_and_rental_index_by_quarter.csv')
    (OUT / 'external_signal_audit.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(table.loc[table.actual_signal_months_before_rent_month == 1, ['signal', 'period', 'level_pearson', 'change_pearson', 'change_n']].to_string(index=False))
    print('ALL FIXED CHANGE LAGS', table.loc[table.period == 'all_fit', ['signal', 'actual_signal_months_before_rent_month', 'change_pearson', 'change_n']].to_string(index=False))
    print('HDB', json.dumps(housing, indent=2))
    print('Report', OUT / 'external_signal_audit.json')
if __name__ == '__main__':
    import argparse as _cli_argparse
    _cli_argparse.ArgumentParser(description=__doc__).parse_args()
    main()
