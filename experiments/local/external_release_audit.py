"""Release-aligned descriptive audit. No model fitting, prediction, or selection.

Partial correlations use linear projections for descriptive adjustment only.
The saved full-fit rental index is retrospective, not a real-time vintage.
"""
from experiments.local.bootstrap import Path, CODE_DIR, DATA_ROOT, SUPPORT_ROOT, ARTIFACT_ROOT, asset_path, asset_key, support_path
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parent
OUT = asset_path(Path('external_release_audit_01'))

def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def corr(x, y, control):
    p = pd.concat([x.rename('x'), y.rename('y'), control.rename('control')], axis=1).dropna()
    result = {'n': len(p), 'pearson': None, 'spearman': None, 'partial_pearson_own_recent_growth': None}
    if len(p) < 5 or p.x.std() == 0 or p.y.std() == 0:
        return result
    result.update(pearson=float(p.x.corr(p.y)), spearman=float(p.x.corr(p.y, method='spearman')))
    design = np.column_stack([np.ones(len(p)), p.control])
    xr = p.x.to_numpy() - design @ np.linalg.lstsq(design, p.x.to_numpy(), rcond=None)[0]
    yr = p.y.to_numpy() - design @ np.linalg.lstsq(design, p.y.to_numpy(), rcond=None)[0]
    if np.std(xr) > 1e-10 and np.std(yr) > 1e-10:
        result['partial_pearson_own_recent_growth'] = float(np.corrcoef(xr, yr)[0, 1])
    return result

def main():
    assert not (OUT / 'results.json').exists(), 'Completed audit must not be overwritten'
    frozen = json.loads(asset_path(Path('temporal_index_trial_01/protocol.json')).read_text())['frozen_sha256']
    for p in ['temporal_index_trial_01/model.joblib', 'temporal_index_trial_01/final_time_index.joblib', 'temporal_index_trial_01/search_predictions.csv', 'temporal_index_trial_01/confirm_predictions.csv', 'temporal_index_trial_01/protocol.json']:
        frozen[p] = sha(asset_path(Path(p)))
    mismatches = [p for p, expected in frozen.items() if sha(asset_path(Path(p))) != expected]
    assert not mismatches, mismatches
    before = {p: sha(OUT / p) for p in ['protocol.json', 'ura_quarterly_releases.csv', 'sora_monthly.csv']}
    monthly = pd.read_csv(asset_path(Path('time_overestimate_diagnosis_01/external_signals/fit_monthly_macro_and_rent_index.csv')))
    assert len(monthly) == 45 and monthly.month.max() == '2024-09'
    monthly['origin_date'] = pd.PeriodIndex(monthly.month, freq='M').to_timestamp(how='end').normalize()
    quarterly = pd.read_csv(OUT / 'ura_quarterly_releases.csv')
    quarterly['release_date'] = pd.to_datetime(quarterly.release_date)
    assert quarterly.release_date.is_monotonic_increasing and quarterly.quarter.is_unique
    assert (quarterly.release_date > pd.PeriodIndex(quarterly.quarter, freq='Q').to_timestamp(how='end')).all()
    quarterly['private_rent_growth_change'] = quarterly.private_rent_qoq_pct.diff()
    quarterly['vacancy_change_pp'] = quarterly.vacancy_pct.diff()
    quarterly['net_stock_addition_change'] = quarterly.net_stock_addition.diff()
    joined = pd.merge_asof(monthly, quarterly, left_on='origin_date', right_on='release_date', direction='backward')
    assert (joined.release_date <= joined.origin_date).all()
    assert joined.iloc[-1].quarter == '2024Q2'
    sora = pd.read_csv(OUT / 'sora_monthly.csv').set_index('month').sora_3m_eop_pct
    assert np.isfinite(sora).all() and sora.index.max() == '2024-09'
    prior_month = (pd.PeriodIndex(joined.month, freq='M') - 1).astype(str)
    joined['sora_prior_month_pct'] = sora.reindex(prior_month).to_numpy()
    joined['sora_prior_month_change_3m_pp'] = sora.diff(3).reindex(prior_month).to_numpy()
    hdb = pd.read_csv(asset_path(Path('time_overestimate_diagnosis_01/external_signals/hdb_price_and_rental_index_by_quarter.csv'))).set_index('quarter')
    quarter_lag2 = (pd.PeriodIndex(joined.month, freq='M').asfreq('Q') - 2).astype(str)
    joined['hdb_price_growth_lag2q'] = hdb.hdb_log_change.reindex(quarter_lag2).to_numpy()
    b = joined.rental_log_index
    joined['own_trailing6_monthly_log_growth'] = (b - b.shift(6)) / 6
    for horizon in [3, 6, 12]:
        joined[f'future{horizon}_monthly_log_growth'] = (b.shift(-horizon) - b) / horizon
        joined[f'future{horizon}_growth_minus_trailing6'] = joined[f'future{horizon}_monthly_log_growth'] - joined.own_trailing6_monthly_log_growth
    signals = ['private_rent_qoq_pct', 'private_rent_growth_change', 'vacancy_pct', 'vacancy_change_pp', 'net_stock_addition', 'net_stock_addition_change', 'sora_prior_month_pct', 'sora_prior_month_change_3m_pp', 'hdb_price_growth_lag2q', 'bank_basket_return_lag1', 'property_basket_return_lag1']
    periods = {'all': np.ones(len(joined), bool), 'through_2022': joined.month <= '2022-12', 'from_2023': joined.month >= '2023-01'}
    records = []
    for cadence in ['monthly', 'quarter_end_only']:
        base = np.ones(len(joined), bool) if cadence == 'monthly' else joined.origin_date.dt.month.isin([3, 6, 9, 12])
        for period, mask in periods.items():
            part = joined.loc[base & mask]
            for horizon in [3, 6, 12]:
                for signal in signals:
                    for target in [f'future{horizon}_monthly_log_growth', f'future{horizon}_growth_minus_trailing6']:
                        stats = corr(part[signal], part[target], part.own_trailing6_monthly_log_growth)
                        valid = part[[signal, target, 'own_trailing6_monthly_log_growth', 'quarter']].dropna()
                        records.append({'cadence': cadence, 'period': period, 'horizon_months': horizon, 'signal': signal, 'target': target, 'distinct_ura_releases': int(valid.quarter.nunique()), **stats})
    stats = pd.DataFrame(records)
    snapshots = joined.loc[joined.month.isin(['2022-09', '2023-05', '2024-09']), ['month', 'quarter', 'release_date', 'private_rent_qoq_pct', 'vacancy_pct', 'net_stock_addition', 'sora_prior_month_pct']].copy()
    snapshots['release_date'] = snapshots.release_date.dt.strftime('%Y-%m-%d')
    supply = [{'origin': '2022-09', 'release_date': '2022-07-22', 'scope': 'private homes including EC', 'announced_remainder_2022': 7195, 'announced_2023': 19958, 'completed_first_half_2022': 3501, 'source': 'https://www.ura.gov.sg/news/media/pr22-30/'}, {'origin': '2023-05', 'release_date': '2023-04-28', 'scope': 'private homes including EC', 'announced_remainder_2023': 15221, 'completed_first_quarter_2023': 3785, 'announced_2024_2025_total': 21421, 'source': 'https://www.ura.gov.sg/news/media/pr23-12/'}, {'origin': '2024-09', 'release_date': '2024-07-26', 'scope': 'private homes including EC', 'announced_remainder_2024': 7675, 'announced_2025_2027_total': 26801, 'source': 'https://www.ura.gov.sg/news/media/pr24-38/'}]
    unchanged = all((sha(asset_path(Path(p))) == expected for p, expected in frozen.items()))
    assert unchanged and all((sha(OUT / p) == v for p, v in before.items()))
    result = {'training_performed': False, 'new_model_predictions': False, 'confirm_used': False, 'search_used': False, 'original_split_or_labels_changed': False, 'protected_file_count': len(frozen), 'protected_files_unchanged': unchanged, 'analysis_inputs_sha256': before, 'monthly_points': len(joined), 'ura_release_count': len(quarterly), 'publication_timing_check_passed': True, 'frozen_origin_snapshots': snapshots.to_dict('records'), 'historically_announced_supply': supply, 'sora_source': 'https://data.gov.sg/datasets/d_5fe5a4bb4a1ecc4d8a56a095832e2b24/view', 'correlations_file': 'associations_all_prespecified.csv', 'limitations': ['Retrospective associations, not forecast RMSE or proof of additional predictive value.', 'Saved rental index is estimated using all original-fit history; monthly index vintages are not real-time.', '45 months span one main boom/slowdown cycle; adjacent and overlapping outcomes are dependent.', 'Quarter-end sensitivity reduces quarterly repetition but leaves very few observations.', 'No p-values, no winner, no best lag or coefficient selected.', 'URA features refer to private homes excluding EC, not HDB. Supply forecasts separately include EC.', 'Net completed stock additions include removals/reclassifications; they are not gross completions.', 'URA figures are transcribed from dated releases. SORA/HDB latest historical vintages are not all verified against original publications.', 'HDB resale price uses a conservative two-quarter lag; this is not full release-vintage certification.', 'Long forecasts must use origin-known signals or origin-known expected supply; cannot fill with later realized macro values.', 'Announced supply is uncertain and calendar-year based; no monthly delivery path invented.'], 'frozen_sha256': frozen}
    joined.to_csv(OUT / 'release_aligned_monthly.csv', index=False)
    stats.to_csv(OUT / 'associations_all_prespecified.csv', index=False)
    (OUT / 'results.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    subset = stats.loc[(stats.cadence == 'monthly') & (stats.horizon_months == 6) & (stats.target == 'future6_monthly_log_growth')]
    print(subset[['signal', 'period', 'n', 'distinct_ura_releases', 'pearson', 'partial_pearson_own_recent_growth']].to_string(index=False))
    print('SNAPSHOTS', snapshots.to_json(orient='records'))
    print('Protected files unchanged:', len(frozen))
if __name__ == '__main__':
    import argparse as _cli_argparse
    _cli_argparse.ArgumentParser(description=__doc__).parse_args()
    main()
