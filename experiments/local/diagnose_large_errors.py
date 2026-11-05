"""Describe existing Search errors; never fit, rescore Confirm or select models."""
from experiments.local.bootstrap import Path, CODE_DIR, DATA_ROOT, SUPPORT_ROOT, ARTIFACT_ROOT, asset_path, asset_key, support_path
import json
from pathlib import Path
import sys
import time
sys.dont_write_bytecode = True
from experiments.local import feature_ablation as shared
ROOT = Path(__file__).resolve().parent
OUT = asset_path(Path('feature_ablation_01/diagnosis'))

def main():
    import joblib
    import numpy as np
    import pandas as pd
    started = time.time()
    shared.verify_frozen()
    cache = joblib.load(asset_path(Path('feature_ablation_01/training_cache.joblib')))
    assert set(cache['parts']) == {'train', 'search'}
    fit, search = [cache['parts'][key].copy() for key in ('train', 'search')]
    assert len(fit) == 132802 and len(search) == 8274
    saved = pd.read_csv(asset_path(Path('feature_ablation_01/baseline_search_predictions.csv')))
    np.testing.assert_array_equal(saved.source_row_id, cache['search_row_ids'])
    y = search.MONTHLY_RENT.to_numpy(float)
    np.testing.assert_array_equal(saved.truth, y)
    p = saved.prediction.to_numpy(float)
    e = p - y
    assert np.isfinite(e).all()
    np.testing.assert_allclose(np.sqrt(np.mean(e ** 2)), 451.45286816840405, atol=1e-09, rtol=0)
    tail = np.zeros(len(y), dtype=bool)
    tail[np.argsort(np.abs(e))[-int(np.ceil(0.1 * len(y))):]] = True
    total_sse = float(np.sum(e ** 2))
    raw = pd.read_csv(asset_path(Path('support/Data/train.csv')), usecols=lambda c: c != 'MONTHLY_RENT', dtype={'BLOCK': str, 'STREET': str, 'TOWN': str})
    months = raw.RENT_APPROVAL_DATE.astype(str).str[:7]
    raw_fit = raw.loc[months <= '2024-09'].reset_index(drop=True)
    raw_search = raw.iloc[cache['search_row_ids']].reset_index(drop=True)
    assert len(raw_fit) == len(fit)
    assert set(raw_search.RENT_APPROVAL_DATE.astype(str).str[:7]) == {'2024-10', '2024-11', '2024-12'}
    for raw_part, part in ((raw_fit, fit), (raw_search, search)):
        np.testing.assert_array_equal(raw_part.RENT_APPROVAL_DATE.astype(str), part.RENT_APPROVAL_DATE.astype(str))
        np.testing.assert_array_equal(raw_part.TOWN.str.lower(), part.TOWN.astype(str).str.lower())
        np.testing.assert_allclose(raw_part.FLOOR_AREA_SQM.astype(float), part.floor_area_sqm.astype(float))
    del raw, months
    b = cache['store']['search']
    stage = {'rnn': b['mu'] * b['floor_area'], 'rnn_catboost': (b['mu'] + b['t']) * b['floor_area'], 'rnn_gmm': (b['mu'] + b['gmm_gap']) * b['floor_area'], 'final_mlp': p}
    cb_error = stage['rnn_catboost'] - y

    def describe(mask):
        n = int(mask.sum())
        if not n:
            return {'rows': 0}
        return {'rows': n, 'tail_rows': int(np.sum(tail & mask)), 'tail_rate': float(np.mean(tail[mask])), 'sse_share': float(np.sum(e[mask] ** 2) / total_sse), 'bias_prediction_minus_truth': float(e[mask].mean()), 'mean_truth': float(y[mask].mean()), 'mean_prediction': float(p[mask].mean()), 'stage_rmse': {name: float(np.sqrt(np.mean((pred[mask] - y[mask]) ** 2))) for name, pred in stage.items()}}
    labels = {'town': search.TOWN.astype(str), 'month': search.RENT_APPROVAL_DATE.astype(str).str[:7], 'flat_type': raw_search.FLAT_TYPE.astype(str), 'flat_model': raw_search.FLAT_MODEL.astype(str), 'furnished': raw_search.FURNISHED.astype(str), 'area': pd.cut(search.floor_area_sqm, [-np.inf, 60, 90, 120, np.inf]).astype(str), 'age': pd.cut(search.building_age_years, [-np.inf, 20, 40, np.inf]).astype(str), 'truth_rent_descriptive_only': pd.cut(y, [-np.inf, 1500, 2500, 3500, 4500, np.inf]).astype(str)}
    groups = {}
    for name, values in labels.items():
        values = np.asarray(values)
        groups[name] = sorted([{'group': value, **describe(values == value)} for value in sorted(set(values))], key=lambda item: -item['sse_share'])
    cases = raw_search.copy()
    cases.insert(0, 'source_row_id', cache['search_row_ids'])
    cases['truth'] = y
    cases['baseline_prediction'] = p
    cases['baseline_error'] = e
    cases['is_worst_10pct'] = tail
    for name, pred in stage.items():
        cases[f'{name}_prediction'] = pred
    support = {}
    for name, cols, recent in (('same_building_type_area_all_fit', ['TOWN', 'BLOCK', 'STREET', 'FLAT_TYPE', 'FLOOR_AREA_SQM'], False), ('same_building_type_area_recent_6mo', ['TOWN', 'BLOCK', 'STREET', 'FLAT_TYPE', 'FLOOR_AREA_SQM'], True), ('same_town_type_area_recent_6mo', ['TOWN', 'FLAT_TYPE', 'FLOOR_AREA_SQM'], True)):
        peers = raw_fit.copy()
        peers['rent'] = fit.MONTHLY_RENT.to_numpy(float)
        if recent:
            peers = peers.loc[peers.RENT_APPROVAL_DATE.astype(str).str[:7] >= '2024-04']
        agg = peers.groupby(cols, dropna=False).rent.agg(count='size', median='median', minimum='min', maximum='max')
        bounds = peers.groupby(cols, dropna=False).rent.quantile([0.1, 0.9]).unstack()
        agg['q10'], agg['q90'] = (bounds[0.1], bounds[0.9])
        matched = raw_search[cols].merge(agg.reset_index(), on=cols, how='left', validate='many_to_one', sort=False)
        counts = matched['count'].fillna(0).to_numpy()
        enough = counts >= 5
        outside = (y < matched.q10.to_numpy()) | (y > matched.q90.to_numpy())
        pred_inside = (p >= matched.q10.to_numpy()) & (p <= matched.q90.to_numpy())
        support[name] = {'no_peers': describe(counts == 0), 'one_to_four_peers': describe((counts > 0) & ~enough), 'at_least_five_peers': describe(enough), 'tail_with_at_least_five_peers': int(np.sum(tail & enough)), 'tail_truth_outside_historical_10_90_percentiles_with_at_least_five_peers': int(np.sum(tail & enough & outside)), 'tail_truth_outside_but_prediction_inside_with_at_least_five_peers': int(np.sum(tail & enough & outside & pred_inside)), 'caution': 'Historical spread and drift are not evidence that any rent label is incorrect.'}
        if name == 'same_building_type_area_recent_6mo':
            for col in ('count', 'median', 'q10', 'q90'):
                cases[f'recent_building_peers_{col}'] = matched[col].to_numpy()
    observed_columns = [col for col in search if col != 'MONTHLY_RENT']
    observed_groups = search.groupby(observed_columns, dropna=False, observed=True).ngroup().to_numpy()
    observed = pd.DataFrame({'group': observed_groups, 'truth': y, 'prediction': p, 'tail': tail, 'source_row_id': cache['search_row_ids']})
    obs_summary = observed.groupby('group').agg(rows=('truth', 'size'), min_rent=('truth', 'min'), max_rent=('truth', 'max'), min_prediction=('prediction', 'min'), max_prediction=('prediction', 'max'), tail_rows=('tail', 'sum'))
    conflicting = obs_summary[(obs_summary.rows > 1) & (obs_summary.max_rent > obs_summary.min_rent)].copy()
    conflicting['rent_range'] = conflicting.max_rent - conflicting.min_rent
    examples = []
    for group_id, row in conflicting[conflicting.tail_rows > 0].sort_values('rent_range', ascending=False).head(5).iterrows():
        group_cases = cases.loc[observed_groups == group_id]
        examples.append({'same_model_inputs': True, 'rent_range': float(row.rent_range), 'prediction_range': float(row.max_prediction - row.min_prediction), 'cases': json.loads(group_cases[['source_row_id', 'RENT_APPROVAL_DATE', 'TOWN', 'BLOCK', 'STREET', 'FLAT_TYPE', 'FLOOR_AREA_SQM', 'FURNISHED', 'truth', 'baseline_prediction']].to_json(orient='records'))})
    ids = cache['search_row_ids']
    np.testing.assert_array_equal(ids, np.arange(len(fit), len(fit) + len(search)))
    source_before_confirm = pd.read_csv(asset_path(Path('support/Data/train.csv')), nrows=int(ids[-1]) + 1, dtype={'BLOCK': str, 'STREET': str, 'TOWN': str})
    assert source_before_confirm.RENT_APPROVAL_DATE.astype(str).str[:7].max() == '2024-12'
    np.testing.assert_array_equal(source_before_confirm.MONTHLY_RENT.iloc[:len(fit)], fit.MONTHLY_RENT)
    np.testing.assert_array_equal(source_before_confirm.MONTHLY_RENT.iloc[ids], y)
    source_search = source_before_confirm.iloc[ids].reset_index(drop=True)
    auxiliary_search = pd.read_csv(asset_path(Path('support/Data/auxiliary_features/train_features.csv')), skiprows=range(1, int(ids[0]) + 1), nrows=len(search))
    assert len(auxiliary_search) == len(search) and len(auxiliary_search.columns) == 94
    raw_predictors = [col for col in source_search if col != 'MONTHLY_RENT']
    expanded = pd.concat([source_search[raw_predictors], auxiliary_search], axis=1)
    expanded_ids = expanded.groupby(list(expanded.columns), dropna=False, observed=True).ngroup().to_numpy()
    expanded_stats = pd.DataFrame({'group': expanded_ids, 'truth': y, 'tail': tail}).groupby('group').agg(rows=('truth', 'size'), distinct_rents=('truth', 'nunique'), tail_rows=('tail', 'sum'))
    for example in examples:
        positions = [int(case['source_row_id']) - int(ids[0]) for case in example['cases']]
        varying = expanded.iloc[positions].nunique(dropna=False)
        example['varying_raw_or_all_94_auxiliary_predictors'] = varying[varying > 1].index.tolist()
    feature_quality = {}
    for col in search.columns:
        if col == 'MONTHLY_RENT':
            continue
        missing = search[col].isna().to_numpy()
        if missing.any():
            feature_quality[col] = describe(missing)
    all_stage_same_sign = np.logical_and.reduce([(pred - y) * e > 0 for pred in stage.values()])
    report = {'training_performed': False, 'confirm_labels_or_predictions_read': False, 'new_inference_performed': False, 'baseline_or_split_changed': False, 'search_rows': len(y), 'tail_definition': 'Largest 828 absolute errors of the fixed local baseline on original Search; selection is descriptive, not a deployable feature.', 'overall': describe(np.ones(len(y), bool)), 'worst_10pct': describe(tail), 'tail_underpredicted': describe(tail & (e < 0)), 'tail_overpredicted': describe(tail & (e > 0)), 'groups': groups, 'historical_support': support, 'stage_attribution': {'tail_catboost_already_abs_error_over_700': int(np.sum(tail & (np.abs(cb_error) > 700))), 'tail_catboost_abs_error_under_350': int(np.sum(tail & (np.abs(cb_error) < 350))), 'tail_final_improved_abs_error_vs_catboost': int(np.sum(tail & (np.abs(e) < np.abs(cb_error)))), 'tail_all_stages_error_same_direction': int(np.sum(tail & all_stage_same_sign)), 'tail_mlp_minus_catboost_sse': float(np.sum(e[tail] ** 2 - cb_error[tail] ** 2)), 'caution': 'Conditioning on final-model worst errors affects comparisons; this diagnoses these rows, not overall model superiority.'}, 'identical_model_input_examples': examples, 'raw_label_row_alignment_verified': True, 'all_raw_plus_94_auxiliary_inputs': {'groups_with_identical_inputs_but_different_rents': int(((expanded_stats.rows > 1) & (expanded_stats.distinct_rents > 1)).sum()), 'tail_rows_in_these_groups': int(expanded_stats.loc[(expanded_stats.rows > 1) & (expanded_stats.distinct_rents > 1), 'tail_rows'].sum()), 'caution': 'Demonstrates observational ambiguity, not incorrect labels or the achievable error floor.'}, 'raw_furnished_counts_fit': raw_fit.FURNISHED.value_counts(dropna=False).to_dict(), 'raw_furnished_counts_search': raw_search.FURNISHED.value_counts(dropna=False).to_dict(), 'missing_input_groups': feature_quality, 'unseen_categories': {col: describe(~search[col].isin(fit[col]).to_numpy()) for col in ('block_key', 'street_key', 'flat_model_key')}, 'limitations': ['Error concentration is descriptive association, not proof of causal mechanism.', 'Historical peer quantiles use fit labels only; they are not newly evaluated predictions.', 'No rows removed, labels corrected, thresholds tuned, winners changed or Confirm rescored.', 'Potential unobserved property/contract attributes cannot be identified from these files alone.']}
    shared.verify_frozen()
    report['source_and_baseline_hashes_unchanged'] = True
    report['elapsed_seconds'] = time.time() - started
    cases = cases.assign(absolute_error=np.abs(e)).sort_values('absolute_error', ascending=False)
    cases.to_csv(OUT / 'search_errors_ranked.csv', index=False)
    report['case_csv'] = str(OUT / 'search_errors_ranked.csv')
    shared.save(OUT / 'large_error_analysis.json', report)
    summary = {key: report[key] for key in ('overall', 'worst_10pct', 'tail_underpredicted', 'tail_overpredicted', 'stage_attribution', 'raw_furnished_counts_fit', 'raw_furnished_counts_search', 'unseen_categories', 'elapsed_seconds')}
    summary['groups'] = {name: rows[:6] for name, rows in groups.items()}
    summary['historical_support'] = support
    summary['identical_model_input_examples'] = examples[:2]
    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)
if __name__ == '__main__':
    import argparse as _cli_argparse
    _cli_argparse.ArgumentParser(description=__doc__).parse_args()
    main()
