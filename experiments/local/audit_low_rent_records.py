"""Source matching and descriptive feature audit. No estimator fit or new predictions."""
from experiments.local.bootstrap import Path, CODE_DIR, DATA_ROOT, SUPPORT_ROOT, ARTIFACT_ROOT, asset_path, asset_key, support_path
from collections import Counter
import json
from pathlib import Path
import re
import sys
import time
sys.dont_write_bytecode = True
from experiments.local import feature_ablation as shared
ROOT = Path(__file__).resolve().parent
OUT = asset_path(Path('feature_ablation_01/diagnosis'))

def text_key(value):
    return re.sub('\\s+', ' ', str(value).strip().upper())

def street_key(value):
    replacements = {'ROAD': 'RD', 'STREET': 'ST', 'AVENUE': 'AVE', 'DRIVE': 'DR', 'CENTRAL': 'CTRL', 'CRESCENT': 'CRES', 'PLACE': 'PL', 'JALAN': 'JLN', 'LORONG': 'LOR', 'BUKIT': 'BT', 'UPPER': 'UPP', 'COMMONWEALTH': "C'WEALTH", 'NORTH': 'NTH', 'SOUTH': 'STH', 'GARDENS': 'GDNS', 'TANJONG': 'TG', 'CLOSE': 'CL', 'PARK': 'PK', 'HEIGHTS': 'HTS', 'SAINT': 'ST.', 'KAMPONG': 'KG', 'TERRACE': 'TER', 'MARKET': 'MKT'}
    return ' '.join((replacements.get(token, token) for token in text_key(value).split()))

def main():
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    started = time.time()
    shared.verify_frozen()
    cache = joblib.load(asset_path(Path('feature_ablation_01/training_cache.joblib')))
    assert set(cache['parts']) == {'train', 'search'}
    fit, search = (cache['parts']['train'], cache['parts']['search'])
    n_fit, n_search = (len(fit), len(search))
    ids = cache['search_row_ids']
    np.testing.assert_array_equal(ids, np.arange(n_fit, n_fit + n_search))
    raw = pd.read_csv(asset_path(Path('support/Data/train.csv')), nrows=n_fit + n_search, dtype={'TOWN': str, 'BLOCK': str, 'STREET': str})
    assert raw.RENT_APPROVAL_DATE.max() == '2024-12'
    np.testing.assert_array_equal(raw.MONTHLY_RENT.iloc[:n_fit], fit.MONTHLY_RENT)
    np.testing.assert_array_equal(raw.MONTHLY_RENT.iloc[n_fit:], search.MONTHLY_RENT)
    raw['source_row_id'] = np.arange(len(raw))
    raw['month'] = raw.RENT_APPROVAL_DATE.astype(str).str[:7]
    raw['town_norm'] = raw.TOWN.map(text_key)
    raw['block_norm'] = raw.BLOCK.map(text_key)
    raw['street_norm'] = raw.STREET.map(street_key)
    raw['type_norm'] = raw.FLAT_TYPE.map(text_key).str.replace(' ', '-', regex=False)
    raw['model_norm'] = raw.FLAT_MODEL.map(text_key)
    raw['building_key'] = raw.town_norm + '|' + raw.block_norm + '|' + raw.street_norm
    raw['low_1500'] = raw.MONTHLY_RENT <= 1500
    raw_fit, raw_search = (raw.iloc[:n_fit].copy(), raw.iloc[n_fit:].copy().reset_index(drop=True))
    baseline = pd.read_csv(asset_path(Path('feature_ablation_01/baseline_search_predictions.csv')))
    np.testing.assert_array_equal(baseline.source_row_id, ids)
    np.testing.assert_array_equal(baseline.truth, raw_search.MONTHLY_RENT)
    errors = baseline.prediction.to_numpy(float) - baseline.truth.to_numpy(float)
    sse = float(np.sum(errors ** 2))
    low = raw_search.low_1500.to_numpy()
    tail = np.zeros(n_search, dtype=bool)
    tail[np.argsort(np.abs(errors))[-828:]] = True
    over_tail = tail & (errors > 0)
    official_records, urls = ([], [])
    for month in ('2024-10', '2024-11', '2024-12'):
        payload = shared.read(OUT / f'hdb_public_{month}.json')
        official_records.extend(payload['response']['result']['records'])
        urls.append(payload['url'])
    official = pd.DataFrame(official_records)
    assert set(official.rent_approval_date) == {'2024-10', '2024-11', '2024-12'}
    official['key'] = list(zip(official.rent_approval_date, official.town.map(text_key), official.block.map(text_key), official.street_name.map(street_key), official.flat_type.map(text_key), official.monthly_rent.astype(float)))
    local_keys = list(zip(raw_search.month, raw_search.town_norm, raw_search.block_norm, raw_search.street_norm, raw_search.type_norm, raw_search.MONTHLY_RENT.astype(float)))
    official_counts, local_counts = (Counter(official.key), Counter(local_keys))
    exact = np.array([key in official_counts for key in local_keys])
    mapped_ids = official.groupby('key')['_id'].apply(list).to_dict()
    source_rows = raw_search[['source_row_id', 'RENT_APPROVAL_DATE', 'TOWN', 'BLOCK', 'STREET', 'FLAT_TYPE', 'MONTHLY_RENT']].copy()
    source_rows['official_key_found'] = exact
    source_rows['official_record_ids'] = [json.dumps(mapped_ids.get(key, [])) for key in local_keys]
    source_rows['local_key_multiplicity'] = [local_counts[key] for key in local_keys]
    source_rows['official_key_multiplicity'] = [official_counts[key] for key in local_keys]
    source_rows.loc[low | over_tail].to_csv(OUT / 'low_rent_source_matches.csv', index=False)

    def stats(mask):
        mask = np.asarray(mask, bool)
        n = int(mask.sum())
        if not n:
            return {'rows': 0}
        return {'rows': n, 'low_1500_rows': int(np.sum(low & mask)), 'low_1500_rate': float(np.mean(low[mask])), 'severely_overpredicted_rows': int(np.sum(over_tail & mask)), 'sse_share': float(np.sum(errors[mask] ** 2) / sse), 'mean_error_pred_minus_truth': float(np.mean(errors[mask]))}
    temporal = []
    raw['quarter'] = pd.to_datetime(raw.month).dt.to_period('Q').astype(str)
    for quarter, frame in raw.groupby('quarter'):
        temporal.append({'quarter': quarter, 'rows': len(frame), 'low_1500_rows': int(frame.low_1500.sum()), 'low_1500_rate': float(frame.low_1500.mean()), 'rent_median': float(frame.MONTHLY_RENT.median())})
    current_fit = raw_fit.month >= '2024-04'
    fit6 = raw_fit.loc[current_fit].copy()
    categorical = {}
    for col in ('town_norm', 'type_norm', 'model_norm'):
        train = fit6.groupby(col).low_1500.agg(rows='size', low_rows='sum', rate='mean')
        se = raw_search.groupby(col).low_1500.agg(search_rows='size', search_low_rows='sum', search_rate='mean')
        joined = train.join(se, how='outer').fillna(0).reset_index()
        categorical[col] = json.loads(joined.sort_values('search_low_rows', ascending=False).to_json(orient='records'))
    history = {}
    for name, columns in (('same_building', ['building_key']), ('same_building_flat_type', ['building_key', 'type_norm']), ('same_building_type_area', ['building_key', 'type_norm', 'FLOOR_AREA_SQM']), ('same_town_type_area', ['town_norm', 'type_norm', 'FLOOR_AREA_SQM'])):
        historical = fit6.groupby(columns).agg(past_rows=('MONTHLY_RENT', 'size'), past_low_rows=('low_1500', 'sum'), past_mean=('MONTHLY_RENT', 'mean'), past_median=('MONTHLY_RENT', 'median'), past_std=('MONTHLY_RENT', 'std'))
        joined = raw_search[columns].merge(historical.reset_index(), on=columns, how='left', validate='many_to_one', sort=False)
        assert len(joined) == n_search
        counts = joined.past_rows.fillna(0).to_numpy()
        low_counts = joined.past_low_rows.fillna(0).to_numpy()
        has_low = low_counts > 0
        enough = counts >= 5
        history[name] = {'past_window': '2024-04 through 2024-09; fixed for all Search months', 'had_at_least_one_low_rent': stats(has_low), 'had_no_low_rent': stats(~has_low), 'at_least_five_peers_and_had_low_rent': stats(enough & has_low), 'at_least_five_peers_without_low_rent': stats(enough & ~has_low), 'by_search_month': [{'month': m, 'had_past_low': stats(has_low & (raw_search.month == m).to_numpy()), 'no_past_low': stats(~has_low & (raw_search.month == m).to_numpy())} for m in ('2024-10', '2024-11', '2024-12')]}
        difference = joined.past_mean.to_numpy(float) - baseline.prediction.to_numpy(float)
        bins = np.asarray(pd.cut(difference, [-np.inf, -300, -100, 100, 300, np.inf]).astype(str), dtype=str)
        history[name]['history_mean_minus_baseline_prediction_bins_at_least_5_peers'] = [{'bin': label, **stats(enough & (bins == label))} for label in sorted(set(bins)) if label != 'nan']
    aux = pd.read_csv(asset_path(Path('support/Data/auxiliary_features/train_features.csv')), nrows=n_fit + n_search)
    assert aux.shape == (n_fit + n_search, 94)
    from sklearn.neighbors import BallTree
    fit_coordinates = aux.iloc[:n_fit].loc[current_fit.to_numpy(), ['latitude', 'longitude']].to_numpy(float)
    search_coordinates = aux.iloc[n_fit:][['latitude', 'longitude']].to_numpy(float)
    past_counts = np.zeros(n_search, dtype=int)
    past_means = np.full(n_search, np.nan)
    past_low_counts = np.zeros(n_search, dtype=int)
    for flat_type in sorted(raw_search.type_norm.unique()):
        ft = (fit6.type_norm.to_numpy() == flat_type) & np.isfinite(fit_coordinates).all(axis=1)
        se = (raw_search.type_norm.to_numpy() == flat_type) & np.isfinite(search_coordinates).all(axis=1)
        if not ft.any() or not se.any():
            continue
        index = BallTree(np.radians(fit_coordinates[ft]), metric='haversine')
        neighbors = index.query_radius(np.radians(search_coordinates[se]), r=0.5 / 6371.0088)
        past_rents = fit6.MONTHLY_RENT.to_numpy(float)[ft]
        for position, matches in zip(np.flatnonzero(se), neighbors):
            past_counts[position] = len(matches)
            if len(matches):
                past_means[position] = past_rents[matches].mean()
                past_low_counts[position] = np.sum(past_rents[matches] <= 1500)
    enough = past_counts >= 5
    spatial_difference = past_means - baseline.prediction.to_numpy(float)
    spatial_bins = np.asarray(pd.cut(spatial_difference, [-np.inf, -300, -100, 100, 300, np.inf]).astype(str), dtype=str)
    spatial_history = {'radius_meters': 500, 'same_flat_type': True, 'past_window': '2024-04 through 2024-09; no Search or public-source rent enters the lookup', 'at_least_five_peers': stats(enough), 'fewer_than_five_peers': stats(~enough), 'historical_mean_minus_baseline_bins': [{'bin': label, **stats(enough & (spatial_bins == label)), 'monthly': [{'month': month, **stats(enough & (spatial_bins == label) & (raw_search.month == month).to_numpy())} for month in ('2024-10', '2024-11', '2024-12')]} for label in sorted(set(spatial_bins)) if label != 'nan'], 'had_past_low_rent': stats(past_low_counts > 0), 'no_past_low_rent': stats(past_low_counts == 0), 'caution': 'Neighbor means are descriptive features, not evaluated replacement predictions. Radius is fixed, not selected by RMSE.'}
    raw_cols = ['RENT_APPROVAL_DATE', 'TOWN', 'BLOCK', 'STREET', 'FLAT_TYPE', 'FLAT_MODEL', 'FLOOR_AREA_SQM', 'FURNISHED', 'LEASE_COMMENCE_DATE', 'FEE']
    expanded = pd.concat([raw_search[raw_cols].reset_index(drop=True), aux.iloc[n_fit:].reset_index(drop=True)], axis=1)
    expanded_group = expanded.groupby(list(expanded), dropna=False, observed=True).ngroup().to_numpy()
    group_frame = pd.DataFrame({'group': expanded_group, 'low': low, 'rent': raw_search.MONTHLY_RENT, 'source_row_id': ids})
    group_stats = group_frame.groupby('group').agg(rows=('rent', 'size'), low_rows=('low', 'sum'), rents=('rent', 'nunique'))
    mixed = group_stats.index[(group_stats.low_rows > 0) & (group_stats.low_rows < group_stats.rows)]
    low_mixed = low & np.isin(expanded_group, mixed)
    excluded = [col for col in aux if col.startswith(('mean_rent_', 'coe_', 'bank_', 'property_'))]
    excluded += ['approval_month_index', 'postal_code']
    numeric = []
    for col in aux.select_dtypes(include=[np.number]):
        if col in excluded:
            continue
        x_train = aux.iloc[:n_fit].loc[current_fit.to_numpy(), col].to_numpy(float)
        x_search = aux.iloc[n_fit:][col].to_numpy(float)
        good_t, good_s = (np.isfinite(x_train), np.isfinite(x_search))
        y_train = fit6.low_1500.to_numpy()
        if np.unique(x_train[good_t]).size < 2 or np.unique(x_search[good_s]).size < 2:
            continue
        at = float(roc_auc_score(y_train[good_t], x_train[good_t]))
        ass = float(roc_auc_score(low[good_s], x_search[good_s]))
        numeric.append({'feature': col, 'fit_recent_signed_auc': at, 'search_signed_auc': ass, 'direction_agrees': bool((at - 0.5) * (ass - 0.5) > 0), 'fit_oriented_search_auc': ass if at >= 0.5 else 1 - ass, 'search_missing': int(np.sum(~good_s))})
    numeric.sort(key=lambda row: -row['fit_oriented_search_auc'])
    pd.DataFrame(numeric).to_csv(OUT / 'low_rent_numeric_associations.csv', index=False)
    constants = {col: {'fit': fit6[col].unique().tolist(), 'search': raw_search[col].unique().tolist()} for col in ('FURNISHED', 'FEE')}
    diagnosis = shared.read(OUT / 'large_error_analysis.json')
    report = {'training_performed': False, 'new_model_predictions_created': False, 'confirm_accessed': False, 'baseline_or_split_changed': False, 'labels_changed': False, 'thresholds': {'low_rent_primary': 'MONTHLY_RENT <= 1500; carried over from the preceding audit, not optimized', 'broader_descriptive_band': 'MONTHLY_RENT <= 2500; not a candidate selection criterion'}, 'source_provenance': {'dataset_url': 'https://data.gov.sg/datasets/d_c9f57187485a850908655db0e8cfe651/view', 'api_urls': urls, 'official_search_rows': len(official), 'all_search_matching_rows': int(exact.sum()), 'low_1500_matching_rows': int(np.sum(exact & low)), 'low_1500_total_rows': int(low.sum()), 'severely_overpredicted_matching_rows': int(np.sum(exact & over_tail)), 'local_key_counts_exceed_official': [{'key': list(key), 'local': n, 'official': official_counts[key]} for key, n in local_counts.items() if n > official_counts[key]], 'unmatched_local_rows': json.loads(source_rows.loc[~exact].to_json(orient='records')), 'limitation': 'Matching proves the values occur in the official owner-declared release, not that each contract value is accurate or that same-condition rows are the same unit.'}, 'low_1500': stats(low), 'rent_at_most_2500': stats(raw_search.MONTHLY_RENT <= 2500), 'quarterly_counts': temporal, 'category_rates_recent_fit_vs_search': categorical, 'history_associations': history, 'spatial_history_associations': spatial_history, 'identical_full_input_mixed_low_high': {'groups': len(mixed), 'low_rows': int(low_mixed.sum()), 'all_rows_in_mixed_groups': int(np.isin(expanded_group, mixed).sum()), 'low_row_share': float(np.mean(low_mixed[low])), 'caution': 'Cannot distinguish individual rows in these groups from existing predictors; this does not prove all other low rents are learnable or establish an error floor.'}, 'numeric_marginal_associations': numeric, 'constant_fields': constants, 'feature_availability': {'actual_unit_number': False, 'actual_unit_floor': False, 'max_floor_is_building_height_not_unit_floor': True, 'existing_mean_rent_columns': 'Computed from the preceding three months of the full labeled file. Using them for Nov/Dec Search would include earlier Search labels; they cannot be copied into a new fixed-cutoff candidate unchanged.'}, 'limitations': ['All association screens are descriptive and exploratory; they do not prove multivariate predictive accuracy or a gain in rent RMSE.', 'Official records were used only for provenance; no official rent was added as a feature or label.', 'No averaging, deduplication, label correction, fitting, extra Confirm evaluation or candidate selection occurred.']}
    shared.verify_frozen()
    report['source_and_baseline_hashes_unchanged'] = True
    report['elapsed_seconds'] = time.time() - started
    shared.save(OUT / 'low_rent_audit.json', report)
    compact = {key: report[key] for key in ('low_1500', 'rent_at_most_2500', 'identical_full_input_mixed_low_high', 'constant_fields', 'elapsed_seconds')}
    compact['source_provenance'] = {key: value for key, value in report['source_provenance'].items() if key not in ('api_urls', 'unmatched_local_rows', 'local_key_counts_exceed_official')}
    compact['source_unmatched'] = report['source_provenance']['unmatched_local_rows'][:10]
    compact['quarterly_counts'] = temporal[-6:]
    compact['top_numeric_associations'] = numeric[:12]
    compact['category_rates'] = {key: value[:8] for key, value in categorical.items()}
    compact['history_associations'] = history
    print(json.dumps(compact, indent=2, allow_nan=False), flush=True)
if __name__ == '__main__':
    import argparse as _cli_argparse
    _cli_argparse.ArgumentParser(description=__doc__).parse_args()
    main()
