"""Analyze existing Search errors only; no fitting or Confirm access."""
from experiments.local.bootstrap import Path, CODE_DIR, DATA_ROOT, SUPPORT_ROOT, ARTIFACT_ROOT, asset_path, asset_key, support_path
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
from experiments.local import feature_ablation as shared
ROOT = Path(__file__).resolve().parent
RUN = asset_path(Path('feature_ablation_01'))

def main():
    stack = shared.runtime()
    import joblib
    import numpy as np
    import pandas as pd
    import torch
    from catboost import CatBoostRegressor
    from sklearn.mixture import GaussianMixture
    from threadpoolctl import threadpool_limits

    def no_fit(*args, **kwargs):
        raise RuntimeError('Only read-only Search diagnostics are authorized')
    CatBoostRegressor.fit = no_fit
    GaussianMixture.fit = no_fit
    stack._fit_mlp = no_fit
    torch.optim.Adam.step = no_fit
    shared.verify_frozen()
    cache = joblib.load(RUN / 'training_cache.joblib')
    assert set(cache['parts']) == {'train', 'search'}
    search = cache['parts']['search']
    fit = cache['parts']['train']
    block = {key: value.copy() for key, value in cache['store']['search'].items()}
    with threadpool_limits(limits=8), torch.inference_mode():
        for town in cache['towns']:
            mask = search.TOWN.astype(str).to_numpy() == town
            ft = fit.loc[fit.TOWN.astype(str) == town].reset_index(drop=True)
            se = search.loc[mask].reset_index(drop=True)
            stack._scale_coords(ft, [se])
            folder = RUN / 'area_only/catboost' / stack.town_slug(town)
            model = stack.load_model(folder)
            preparer = joblib.load(folder / 'preprocessor.joblib')
            block['t'][mask] = stack.predict_model(model, stack.transform_frame(preparer, se))
        best = shared.saved_mlp_predictions(stack, asset_path(Path('baseline_alignment/stack')), block)
    analysis = shared.read(RUN / 'diagnosis/analysis.json')
    y = search.MONTHLY_RENT.to_numpy(float)
    np.testing.assert_allclose(np.sqrt(np.mean((best - y) ** 2)), 451.042648255612, rtol=0, atol=1e-09)
    paths = {'baseline': RUN / 'baseline_search_predictions.csv', 'area_retrained_mlp': RUN / 'area_only/search_predictions.csv', 'age_retrained_mlp': RUN / 'age_only/search_predictions.csv', 'weighted_frozen_mlp': asset_path(Path('weighted_residual_01/search_predictions.csv'))}
    predictions = {'best_area_frozen_mlp': best}
    for name, path in paths.items():
        table = pd.read_csv(path)
        np.testing.assert_array_equal(table.source_row_id.to_numpy(), cache['search_row_ids'])
        np.testing.assert_array_equal(table.truth.to_numpy(), y)
        predictions[name] = table.prediction.to_numpy()
    baseline = predictions['baseline']
    e0 = baseline - y
    top_count = int(np.ceil(0.1 * len(y)))
    top = np.argsort(np.abs(e0))[-top_count:]
    report = {'training_performed': False, 'confirm_accessed': False, 'baseline_changed': False, 'rows': len(y), 'comparison': {}, 'baseline_error_concentration': {}, 'observational_error_groups': {}}
    for name, p in predictions.items():
        error = p - y
        change = p - baseline
        assert np.isfinite(error).all()
        report['comparison'][name] = {'rmse': float(np.sqrt(np.mean(error ** 2))), 'prediction_correlation_with_baseline': float(np.corrcoef(p, baseline)[0, 1]), 'error_correlation_with_baseline': float(np.corrcoef(error, e0)[0, 1]), 'prediction_change_rmse': float(np.sqrt(np.mean(change ** 2))), 'median_absolute_prediction_change': float(np.median(np.abs(change))), 'baseline_worst_10pct_rmse': float(np.sqrt(np.mean(error[top] ** 2))), 'baseline_worst_10pct_sse_change': float(np.sum(error[top] ** 2 - e0[top] ** 2)), 'total_sse_change': float(np.sum(error ** 2 - e0 ** 2))}
    sse = e0 ** 2
    report['baseline_error_concentration'] = {'worst_10pct_rows': top_count, 'worst_10pct_sse_share': float(np.sum(sse[top]) / np.sum(sse)), 'rows_abs_error_over_1000': int(np.sum(np.abs(e0) > 1000)), 'abs_error_over_1000_sse_share': float(np.sum(sse[np.abs(e0) > 1000]) / np.sum(sse)), 'absolute_error_quantiles': dict(zip(['50%', '75%', '90%', '95%', '99%'], np.quantile(np.abs(e0), [0.5, 0.75, 0.9, 0.95, 0.99]).tolist()))}
    group_columns = [c for c in search.columns if c != 'MONTHLY_RENT']
    grouped = search.groupby(group_columns, dropna=False, observed=True).MONTHLY_RENT.agg(['size', 'var'])
    repeated = grouped[grouped['size'] > 1]
    within_sse = float(((grouped['size'] - 1) * grouped['var'].fillna(0)).sum())
    report['observational_error_groups']['identical_observed_inputs'] = {'input_columns': group_columns, 'total_groups': len(grouped), 'repeated_groups': len(repeated), 'rows_in_repeated_groups': int(repeated['size'].sum()), 'groups_with_different_rents': int((repeated['var'] > 0).sum()), 'empirical_within_identical_input_sse': within_sse, 'descriptive_rmse_lower_bound_on_this_sample': float(np.sqrt(within_sse / len(search))), 'caution': 'Uses Search label variation descriptively; most singleton groups have zero observed within-group error. This is not an achievable validation score, not an irreducible population error estimate and not a prediction model.'}
    errors = pd.DataFrame({'town': search.TOWN.astype(str), 'month': search.RENT_APPROVAL_DATE.astype(str).str[:7], 'error': e0})
    agg = errors.groupby(['town', 'month']).error.agg(['size', 'mean'])
    report['observational_error_groups']['town_month_bias_sse_share'] = float(np.sum(agg['size'] * agg['mean'] ** 2) / np.sum(sse))
    source_header = pd.read_csv(asset_path(Path('support/Data/train.csv')), nrows=0).columns.tolist()
    report['raw_source_columns'] = source_header
    report['stage_gain'] = {'area': {'catboost_rmse_reduction': 455.0577162431999 - 453.01686957640345, 'frozen_final_rmse_reduction': 451.45286816840405 - 451.042648255612}, 'weighting': {'catboost_rmse_reduction': 455.0577162431999 - 454.6075902766782, 'frozen_final_rmse_reduction': 451.45286816840405 - 451.38977051538865}}
    shared.verify_frozen()
    report['source_and_baseline_hashes_unchanged'] = True
    path = RUN / 'diagnosis/gain_limits.json'
    shared.save(path, report)
    print(json.dumps(report, indent=2), flush=True)
if __name__ == '__main__':
    import argparse as _cli_argparse
    _cli_argparse.ArgumentParser(description=__doc__).parse_args()
    main()
