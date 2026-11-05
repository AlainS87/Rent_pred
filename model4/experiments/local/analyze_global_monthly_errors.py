"""Descriptive audit of saved predictions. No training, inference, or tuning."""
from experiments.local.bootstrap import Path, CODE_DIR, DATA_ROOT, SUPPORT_ROOT, ARTIFACT_ROOT, asset_path, asset_key, support_path

def main():
    import json
    import math
    import sys
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import joblib
    ROOT = Path(__file__).resolve().parent
    from experiments.local.global_monthly_catboost import FEATURES, OUT, read, digest, check_frozen
    from experiments.local.spatial_history_io import write_json
    DEST = OUT / 'diagnosis'
    DEST.mkdir(exist_ok=True)
    protocol = read(OUT / 'protocol.json')
    selection = read(OUT / 'selection_frozen.json')
    check_frozen(protocol)
    protected = {name: digest(OUT / name) for name in ['protocol.json', 'selection_frozen.json', 'results.json', 'model.joblib', 'preprocessor.joblib', 'search_predictions.csv', 'confirm_predictions.csv']}
    for name, sha in selection['candidate_sha256'].items():
        assert digest(OUT / name) == sha
    raw = pd.read_csv(asset_path(Path('support/Data/train.csv')), dtype={'TOWN': str, 'BLOCK': str, 'STREET': str})
    aux = pd.read_csv(asset_path(Path('support/Data/auxiliary_features/train_features.csv')))
    assert len(raw) == len(aux) == 150000
    assert not set(raw).intersection(aux)
    all_rows = pd.concat([raw, aux], axis=1)
    all_rows['source_row_id'] = np.arange(len(raw))
    all_rows['building_key'] = all_rows[['TOWN', 'block_key', 'street_key']].astype(str).agg('|'.join, axis=1)
    all_rows['building_type_key'] = all_rows[['building_key', 'flat_type_ord']].astype(str).agg('|'.join, axis=1)
    all_rows['building_type_area_key'] = all_rows[['building_type_key', 'floor_area_sqm']].astype(str).agg('|'.join, axis=1)
    all_rows['town_block_key'] = all_rows[['TOWN', 'block_key']].astype(str).agg('|'.join, axis=1)
    train = all_rows.iloc[:132802].copy()
    recent = train[train.RENT_APPROVAL_DATE >= '2024-04'].copy()
    assert train.RENT_APPROVAL_DATE.max() == '2024-09' and len(recent) == 17960
    alias_counts = train.groupby('town_block_key').building_key.nunique()
    block_global_aliases = train.groupby('block_key').building_key.nunique()
    history = {}
    for window, source in [('all', train), ('recent6', recent)]:
        for key in ['building_key', 'building_type_key', 'building_type_area_key']:
            source = source.assign(low=source.MONTHLY_RENT <= 1500)
            history[window + '_' + key] = source.groupby(key).agg(count=('MONTHLY_RENT', 'size'), mean=('MONTHLY_RENT', 'mean'), std=('MONTHLY_RENT', 'std'), low_rate=('low', 'mean'), low_count=('low', 'sum'))

    def summary(frame, total_sse=None):
        err = frame.error.to_numpy(float)
        if len(err) == 0:
            return {'rows': 0}
        sse = float(np.dot(err, err))
        result = {'rows': len(frame), 'rmse': float(np.sqrt(sse / len(err))), 'mae': float(np.mean(np.abs(err))), 'bias_pred_minus_truth': float(err.mean()), 'sse': sse, 'truth_mean': float(frame.truth.mean()), 'prediction_mean': float(frame.prediction.mean()), 'abs_error_ge_1000': int((np.abs(err) >= 1000).sum())}
        if total_sse is not None:
            result['sse_share'] = sse / total_sse
        if 'baseline_prediction' in frame:
            baseline_error = frame.baseline_prediction.to_numpy(float) - frame.truth.to_numpy(float)
            result['baseline_rmse'] = float(np.sqrt(np.mean(baseline_error ** 2)))
            result['sse_delta_from_baseline'] = sse - float(np.dot(baseline_error, baseline_error))
        return result

    def group_table(frame, columns, total_sse):
        entries = []
        for key, part in frame.groupby(columns, observed=True, dropna=False):
            if not isinstance(key, tuple):
                key = (key,)
            entries.append({**dict(zip(columns, [str(k) for k in key])), **summary(part, total_sse)})
        return sorted(entries, key=lambda x: x['sse'], reverse=True)

    def identical_audit(frame):
        grouped = frame.groupby(FEATURES, dropna=False, observed=True, sort=False)
        groups = grouped.agg(n=('truth', 'size'), ymin=('truth', 'min'), ymax=('truth', 'max'), mean=('truth', 'mean'), yvar=('truth', 'var'), prediction_span=('prediction', lambda s: s.max() - s.min()))
        repeated = groups.n > 1
        variation = groups.ymax > groups.ymin
        within = float(((groups.n - 1) * groups.yvar.fillna(0)).sum())
        duplicate_mask = frame.duplicated(FEATURES, keep=False)
        return {'groups': len(groups), 'repeated_groups': int(repeated.sum()), 'rows_in_repeated_groups': int(groups.loc[repeated, 'n'].sum()), 'different_rent_groups': int(variation.sum()), 'max_rent_spread': float((groups.ymax - groups.ymin).max()), 'rows_abs_error_ge_1000_in_repeated_groups': int((duplicate_mask & (frame.error.abs() >= 1000)).sum()), 'max_prediction_spread_identical_inputs': float(groups.prediction_span.max()), 'observed_within_identical_inputs_sse': within, 'share_of_current_sse': within / float((frame.error ** 2).sum()), 'warning': 'Descriptive sample lower bound only. Singleton rows contribute zero; not population irreducible error or an achievable forecast.'}
    report = {'training_performed': False, 'new_inference_performed': False, 'prediction_correction_performed': False, 'analysis_primary': 'Search; Confirm is a descriptive follow-up using existing saved predictions, not a new evaluation', 'periods': {}, 'historical_window': {'end': '2024-09', 'recent_start': '2024-04', 'rows': len(recent)}, 'cautions': ['Target-defined error groups are retrospective and are not deployable routing rules.', 'Correlations and group biases do not prove causation or establish improvement.', 'No Search/Confirm labels are added to training or historical lookup tables.']}
    frames = {}
    for stage, start, end in [('search', 132802, 141076), ('confirm', 141076, 150000)]:
        saved = pd.read_csv(OUT / f'{stage}_predictions.csv')
        assert np.array_equal(saved.source_row_id, np.arange(start, end))
        frame = all_rows.iloc[start:end].copy().reset_index(drop=True)
        assert np.array_equal(saved.truth, frame.MONTHLY_RENT)
        frame['truth'], frame['prediction'] = (saved.truth.to_numpy(), saved.prediction.to_numpy())
        frame['error'] = frame.prediction - frame.truth
        frame['absolute_error'] = frame.error.abs()
        if stage == 'search':
            base = pd.read_csv(asset_path(Path('feature_ablation_01/baseline_search_predictions.csv')))
            assert np.array_equal(base.source_row_id, frame.source_row_id)
            assert np.array_equal(base.truth, frame.truth)
            frame['baseline_prediction'] = base.prediction.to_numpy()
        total_sse = float((frame.error ** 2).sum())
        assert abs(math.sqrt(total_sse / len(frame)) - read(OUT / 'results.json')[stage]['rmse']) < 1e-09
        frame['same_town_block_multi_streets'] = frame.town_block_key.map(alias_counts).fillna(0) > 1
        frame['same_block_multi_buildings_globally'] = frame.block_key.map(block_global_aliases).fillna(0) > 1
        for name, table in history.items():
            key = name[name.index('_') + 1:]
            for col in table.columns:
                frame[name + '_' + col] = frame[key].map(table[col])
        for key in ['building_key', 'building_type_key', 'building_type_area_key']:
            frame[key + '_recent_count_bin'] = pd.cut(frame['recent6_' + key + '_count'].fillna(0), [-1, 0, 2, 5, 10, np.inf], labels=['0', '1-2', '3-5', '6-10', '11+'])
        frame['rent_bin'] = pd.cut(frame.truth, [-np.inf, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, np.inf])
        frame['predicted_rent_bin'] = pd.cut(frame.prediction, [-np.inf, 2500, 3000, 3500, 4000, 4500, np.inf])
        frame['area_bin'] = pd.cut(frame.floor_area_sqm, [-np.inf, 60, 80, 100, 120, 140, np.inf])
        frame['age_bin'] = pd.cut(frame.building_age_years, [-np.inf, 10, 20, 30, 40, 50, np.inf])
        frame['distance_cbd_bin'] = pd.cut(frame.dist_to_cbd_km, [-np.inf, 5, 10, 15, 20, np.inf])
        frame['history_has_low'] = frame.recent6_building_type_key_low_count.fillna(0) > 0
        frame['history_std_bin'] = pd.cut(frame.recent6_building_type_key_std, [-np.inf, 250, 500, 750, np.inf])
        largest = frame.nlargest(math.ceil(len(frame) * 0.1), 'absolute_error')
        masks = {'worst_10pct': frame.index.isin(largest.index), 'abs_error_ge_1000': frame.absolute_error >= 1000, 'large_overprediction_ge_1000': frame.error >= 1000, 'large_underprediction_ge_1000': frame.error <= -1000, 'truth_le_1500': frame.truth <= 1500, 'truth_gt_5000': frame.truth > 5000}
        period = {'overall': summary(frame), 'concentration': {name: summary(frame[mask], total_sse) for name, mask in masks.items()}, 'by_town': group_table(frame, ['TOWN'], total_sse), 'by_month': group_table(frame, ['RENT_APPROVAL_DATE'], total_sse), 'by_flat_type': group_table(frame, ['FLAT_TYPE'], total_sse), 'by_rent': group_table(frame, ['rent_bin'], total_sse), 'by_predicted_rent': group_table(frame, ['predicted_rent_bin'], total_sse), 'by_area': group_table(frame, ['area_bin'], total_sse), 'by_age': group_table(frame, ['age_bin'], total_sse), 'by_distance_cbd': group_table(frame, ['distance_cbd_bin'], total_sse), 'by_address_alias': group_table(frame, ['same_town_block_multi_streets'], total_sse), 'by_recent_history_count': group_table(frame, ['building_type_key_recent_count_bin'], total_sse), 'by_history_std': group_table(frame, ['history_std_bin'], total_sse), 'by_history_low': group_table(frame, ['history_has_low'], total_sse), 'identical_observed_inputs': identical_audit(frame), 'constant_bias_sse_share': float(len(frame) * frame.error.mean() ** 2 / total_sse)}
        period['town_mean_bias_sse_share'] = sum((item['rows'] * item['bias_pred_minus_truth'] ** 2 for item in period['by_town'])) / total_sse
        period['town_type_mean_bias_sse_share'] = sum((item['rows'] * item['bias_pred_minus_truth'] ** 2 for item in group_table(frame, ['TOWN', 'FLAT_TYPE'], total_sse))) / total_sse
        period['low_rent_history_detectability'] = {'low_rent_rows': int((frame.truth <= 1500).sum()), 'low_rent_with_any_past_low_same_building_type': int(((frame.truth <= 1500) & frame.history_has_low).sum()), 'all_rows_with_any_past_low_same_building_type': int(frame.history_has_low.sum()), 'recent_history_missing_rows': int(frame.recent6_building_type_key_count.isna().sum())}
        period['worst_rows_by_town'] = group_table(largest, ['TOWN'], total_sse)
        report['periods'][stage] = period
        keep = ['source_row_id', 'RENT_APPROVAL_DATE', 'TOWN', 'BLOCK', 'STREET', 'FLAT_TYPE', 'FLAT_MODEL', 'floor_area_sqm', 'building_age_years', 'truth', 'prediction', 'error', 'absolute_error', 'building_key', 'recent6_building_type_key_count', 'recent6_building_type_key_mean', 'recent6_building_type_key_std', 'recent6_building_type_key_low_count', 'all_building_type_key_count', 'same_town_block_multi_streets']
        frame.sort_values('absolute_error', ascending=False)[keep].to_csv(DEST / f'{stage}_errors_ranked.csv', index=False)
        for name, value in period.items():
            if name.startswith('by_'):
                pd.DataFrame(value).to_csv(DEST / f'{stage}_{name}.csv', index=False)
        frames[stage] = frame
    model = joblib.load(OUT / 'model.joblib')
    report['cached_model_feature_importance'] = sorted([{'feature': name, 'importance': float(value)} for name, value in zip(model.feature_names_, model.feature_importances_)], key=lambda item: item['importance'], reverse=True)
    report['resolved_ctr_target_border_count'] = model.get_all_params()['ctr_target_border_count']
    report['training_label_rent_distribution'] = {'all_fit': train.MONTHLY_RENT.describe(percentiles=[0.01, 0.1, 0.5, 0.9, 0.99]).to_dict(), 'recent_fit': recent.MONTHLY_RENT.describe(percentiles=[0.01, 0.1, 0.5, 0.9, 0.99]).to_dict()}
    report['input_information_limits'] = {'furnished_unique': sorted(raw.FURNISHED.astype(str).unique()), 'fee_unique': sorted(raw.FEE.unique().tolist()), 'unit_number_present': False, 'actual_unit_floor_present': False, 'max_floor_is_building_height_not_unit_floor': True, 'identical_25_features_do_not_imply_same_physical_unit_or_dirty_labels': True}
    check_frozen(protocol)
    assert all((digest(OUT / name) == sha for name, sha in protected.items()))
    report['source_baseline_candidate_and_predictions_unchanged'] = True
    report['protected_candidate_sha256'] = protected

    def clean(value):
        if isinstance(value, dict):
            return {str(k): clean(v) for k, v in value.items()}
        if isinstance(value, list):
            return [clean(v) for v in value]
        if isinstance(value, (float, np.floating)) and (not math.isfinite(value)):
            return None
        if isinstance(value, np.generic):
            return value.item()
        return value
    write_json(DEST / 'error_audit.json', clean(report))
    print(json.dumps(clean({'periods': {stage: {key: val for key, val in data.items() if key in ['overall', 'concentration', 'identical_observed_inputs', 'low_rent_history_detectability', 'constant_bias_sse_share', 'town_mean_bias_sse_share', 'town_type_mean_bias_sse_share']} for stage, data in report['periods'].items()}, 'top_search_towns': report['periods']['search']['by_town'][:8], 'feature_importance': report['cached_model_feature_importance'][:12]}), indent=2))
if __name__ == '__main__':
    import argparse as _cli_argparse
    _cli_argparse.ArgumentParser(description=__doc__).parse_args()
    main()
