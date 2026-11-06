"""Inference-only diagnostics. No training, Confirm scoring, or model selection."""
from experiments.local.bootstrap import Path, CODE_DIR, DATA_ROOT, SUPPORT_ROOT, ARTIFACT_ROOT, asset_path, asset_key, support_path
import copy
import json
from pathlib import Path
import sys
import time
sys.dont_write_bytecode = True
from experiments.local import feature_ablation as experiment
ROOT = Path(__file__).resolve().parent
RUN = asset_path(Path('feature_ablation_01'))
OUTPUT = RUN / 'diagnosis'

def main():
    start = time.time()
    stack = experiment.runtime()
    import joblib
    import numpy as np
    import pandas as pd
    import torch
    from catboost import CatBoostRegressor
    from threadpoolctl import threadpool_limits

    def prohibited(*args, **kwargs):
        raise RuntimeError('Training is forbidden in this diagnostic')
    CatBoostRegressor.fit = prohibited
    stack._fit_mlp = prohibited
    stack._fit_mixture = prohibited
    torch.optim.Adam.step = prohibited
    protocol = experiment.verify_frozen()
    cache = joblib.load(RUN / 'training_cache.joblib')
    parts = cache['parts']
    assert set(parts) == {'train', 'search'}
    model_roots = {'baseline': asset_path(Path('baseline_alignment/stack')), 'area_only': RUN / 'area_only', 'age_only': RUN / 'age_only'}
    blocks = {'baseline': cache['store']}
    resolved = {name: experiment.read(path.parent / 'catboost_resolved_parameters.json') if name == 'baseline' else experiment.read(path / 'catboost_resolved_parameters.json') for name, path in model_roots.items()}
    assert resolved['baseline'] == resolved['area_only'] == resolved['age_only']
    importance, tree_counts, all_parameter_differences, selected_columns = ({}, {}, {}, {})
    baseline_parameters_by_town = {}
    for label, folder in model_roots.items():
        if label != 'baseline':
            blocks[label] = copy.deepcopy(cache['store'])
        importance[label], tree_counts[label], selected_columns[label] = ([], [], [])
        all_parameter_differences[label] = []
        for town in cache['towns']:
            model_folder = folder / 'catboost' / stack.town_slug(town)
            model = stack.load_model(model_folder)
            preparer = joblib.load(model_folder / 'preprocessor.joblib')
            parameters = model.get_all_params()
            if label == 'baseline':
                baseline_parameters_by_town[town] = parameters
            else:
                diffs = {key: [baseline_parameters_by_town[town].get(key), value] for key, value in parameters.items() if baseline_parameters_by_town[town].get(key) != value}
                if diffs:
                    all_parameter_differences[label].append({'town': town, 'differences': diffs})
            selected_columns[label].append({'town': town, 'columns': preparer.selected_columns_})
            tree_counts[label].append({'town': town, 'trees': int(model.tree_count_)})
            masks = {name: part.TOWN.astype(str).to_numpy() == town for name, part in parts.items()}
            fit, search = [parts[name].loc[masks[name]].reset_index(drop=True) for name in ('train', 'search')]
            stack._scale_coords(fit, [search])
            for name, part in (('train', fit), ('search', search)):
                prediction = stack.predict_model(model, stack.transform_frame(preparer, part))
                if label == 'baseline':
                    np.testing.assert_array_equal(prediction, blocks[label][name]['t'][masks[name]])
                else:
                    blocks[label][name]['t'][masks[name]] = prediction
            values = model.feature_importances_
            ranks = sorted(zip(model.feature_names_, values), key=lambda item: -item[1])
            feature = experiment.CANDIDATES.get(label)
            importance[label].append({'town': town, 'fit_rows': len(fit), 'top5': ranks[:5], 'added_feature_importance': dict(ranks).get(feature)})
        print(json.dumps({'stage': 'saved_catboost_predictions_loaded', 'candidate': label}), flush=True)

    def metrics(y, p):
        difference = np.asarray(p, dtype=float) - np.asarray(y, dtype=float)
        assert np.isfinite(difference).all()
        return {'rows': len(y), 'rmse': float(np.sqrt(np.mean(difference ** 2))), 'mae': float(np.mean(np.abs(difference))), 'bias_pred_minus_truth': float(difference.mean())}
    truth = {name: part.MONTHLY_RENT.to_numpy(float) for name, part in parts.items()}
    cat_predictions, mlp_predictions, stage_metrics = ({}, {}, {})
    for label in model_roots:
        cat_predictions[label], stage_metrics[label] = ({}, {})
        for name in parts:
            block = blocks[label][name]
            cat_predictions[label][name] = (block['mu'] + block['t']) * block['floor_area']
            stage_metrics[label][name] = {'catboost_plus_rnn': metrics(truth[name], cat_predictions[label][name])}
    cross_predictions, cross_metrics, checkpoint_details = ({}, {}, {})
    for mlp_label, folder in model_roots.items():
        checkpoint = torch.load(folder / 'mlp/model.pt', map_location='cpu', weights_only=False)
        checkpoint_details[mlp_label] = {'best_epoch': checkpoint['epoch'], 'catboost_input_center': float(checkpoint['x_center'][1]), 'catboost_input_spread': float(checkpoint['x_spread'][1])}
        base_checkpoint = torch.load(model_roots['baseline'] / 'mlp/model.pt', map_location='cpu', weights_only=False)
        unchanged = [index for index in range(19) if index != 1]
        np.testing.assert_array_equal(checkpoint['x_center'][unchanged], base_checkpoint['x_center'][unchanged])
        np.testing.assert_array_equal(checkpoint['x_spread'][unchanged], base_checkpoint['x_spread'][unchanged])
        assert checkpoint['y_center'] == base_checkpoint['y_center']
        assert checkpoint['y_spread'] == base_checkpoint['y_spread']
        cross_predictions[mlp_label], cross_metrics[mlp_label] = ({}, {})
        for input_label in model_roots:
            cross_predictions[mlp_label][input_label], cross_metrics[mlp_label][input_label] = ({}, {})
            for name in parts:
                with torch.inference_mode():
                    p = experiment.saved_mlp_predictions(stack, folder, blocks[input_label][name])
                cross_predictions[mlp_label][input_label][name] = p
                cross_metrics[mlp_label][input_label][name] = metrics(truth[name], p)
                if mlp_label == input_label:
                    stage_metrics[mlp_label][name]['final_mlp'] = metrics(truth[name], p)
        mlp_predictions[mlp_label] = cross_predictions[mlp_label][mlp_label]
        saved_csv = RUN / 'baseline_search_predictions.csv' if mlp_label == 'baseline' else folder / 'search_predictions.csv'
        table = pd.read_csv(saved_csv)
        np.testing.assert_array_equal(table.source_row_id.to_numpy(), cache['search_row_ids'])
        np.testing.assert_allclose(mlp_predictions[mlp_label]['search'], table.prediction.to_numpy(), atol=1e-09, rtol=0)
        print(json.dumps({'stage': 'frozen_mlp_inference_complete', 'checkpoint': mlp_label}), flush=True)
    group_results, error_changes, feature_summary = ({}, {}, {})
    for name, part in parts.items():
        feature_summary[name] = {}
        for column in experiment.CANDIDATES.values():
            values = part[column].to_numpy(float)
            good = values[np.isfinite(values)]
            feature_summary[name][column] = {'missing_rows': int(np.sum(~np.isfinite(values))), 'quantiles': np.quantile(good, [0, 0.25, 0.5, 0.75, 1]).tolist()}
    part = parts['search']
    age = part.building_age_years.to_numpy(float)
    groups = {'town': part.TOWN.astype(str).to_numpy(), 'month': part.RENT_APPROVAL_DATE.astype(str).str[:7].to_numpy(), 'area': np.asarray(pd.cut(part.floor_area_sqm, [-np.inf, 60, 90, 120, np.inf], labels=['<=60', '60-90', '90-120', '>120']).astype(str)), 'age': np.where(np.isfinite(age), np.asarray(pd.cut(age, [-np.inf, 20, 40, np.inf], labels=['<=20', '20-40', '>40']).astype(str)), 'missing')}
    base_errors = mlp_predictions['baseline']['search'] - truth['search']
    for group_name, labels in groups.items():
        group_results[group_name] = []
        for value in sorted(set(labels)):
            mask = labels == value
            item = {'group': str(value), 'rows': int(mask.sum()), 'variants': {}}
            for label in model_roots:
                item['variants'][label] = {'catboost_plus_rnn': metrics(truth['search'][mask], cat_predictions[label]['search'][mask]), 'final_mlp': metrics(truth['search'][mask], mlp_predictions[label]['search'][mask]), 'delta_mlp_sse': float(np.sum((mlp_predictions[label]['search'][mask] - truth['search'][mask]) ** 2 - base_errors[mask] ** 2))}
            group_results[group_name].append(item)
    for label in experiment.CANDIDATES:
        change = mlp_predictions[label]['search'] - mlp_predictions['baseline']['search']
        delta_squared = (base_errors + change) ** 2 - base_errors ** 2
        order = np.argsort(delta_squared)[::-1]
        improved = np.abs(base_errors + change) < np.abs(base_errors)
        cb_change = cat_predictions[label]['search'] - cat_predictions['baseline']['search']
        error_changes[label] = {'prediction_change_rmse': float(np.sqrt(np.mean(change ** 2))), 'prediction_change_mean': float(change.mean()), 'rows_improved': int(improved.sum()), 'rows_worsened': int(np.sum(np.abs(base_errors + change) > np.abs(base_errors))), 'total_delta_sse': float(delta_squared.sum()), 'mse_cross_term': float(2 * np.mean(base_errors * change)), 'mse_change_squared': float(np.mean(change ** 2)), 'catboost_mlp_change_correlation': float(np.corrcoef(change, cb_change)[0, 1]), 'largest_regressions': [{'source_row_id': int(cache['search_row_ids'][i]), 'town': str(part.TOWN.iloc[i]), 'area': float(part.floor_area_sqm.iloc[i]), 'truth': float(truth['search'][i]), 'baseline_prediction': float(mlp_predictions['baseline']['search'][i]), 'candidate_prediction': float(mlp_predictions[label]['search'][i]), 'delta_sse': float(delta_squared[i])} for i in order[:10]]}
    clusters = np.array([f'{t}/{m}' for t, m in zip(groups['town'], groups['month'])])
    cluster_names = sorted(set(clusters))
    sizes = np.array([np.sum(clusters == value) for value in cluster_names])
    sse = {label: np.array([np.sum((mlp_predictions[label]['search'][clusters == value] - truth['search'][clusters == value]) ** 2) for value in cluster_names]) for label in model_roots}
    rng = np.random.default_rng(5228)
    draws = rng.integers(0, len(cluster_names), size=(2000, len(cluster_names)))
    denominator = sizes[draws].sum(axis=1)
    baseline_rmse = np.sqrt(sse['baseline'][draws].sum(axis=1) / denominator)
    bootstrap = {label: {'delta_rmse_percentile_95': np.quantile(np.sqrt(sse[label][draws].sum(axis=1) / denominator) - baseline_rmse, [0.025, 0.975]).tolist()} for label in experiment.CANDIDATES}
    experiment.verify_frozen()
    report = {'training_performed': False, 'confirm_accessed': False, 'selection_changed': False, 'stage_metrics': stage_metrics, 'frozen_mlp_cross_input_metrics': cross_metrics, 'checkpoint_details': checkpoint_details, 'feature_summary': feature_summary, 'all_town_resolved_parameter_differences': all_parameter_differences, 'selected_columns': selected_columns, 'tree_counts': tree_counts, 'catboost_importance': importance, 'search_groups': group_results, 'search_error_changes': error_changes, 'bootstrap': {'method': 'paired town-month cluster bootstrap, 2000 draws, 78 observed clusters; only three Search months, not proof of population significance', 'results': bootstrap}, 'limitation': 'Per-epoch learning curves and repeated seeds were not saved; no new fits are allowed, so exact optimization-path causality is unverified.', 'frozen_source_data_baseline_hashes_passed': True, 'elapsed_seconds': time.time() - start}
    OUTPUT.mkdir(exist_ok=True)
    experiment.save(OUTPUT / 'analysis.json', report)
    print(json.dumps({'stage_metrics': stage_metrics, 'cross_input_search_rmse': {mlp: {inp: val['search']['rmse'] for inp, val in row.items()} for mlp, row in cross_metrics.items()}, 'checkpoint_details': checkpoint_details, 'feature_summary': feature_summary, 'bootstrap': bootstrap, 'parameter_differences': all_parameter_differences, 'output': str(OUTPUT / 'analysis.json'), 'elapsed_seconds': time.time() - start}, indent=2))
if __name__ == '__main__':
    import argparse as _cli_argparse
    _cli_argparse.ArgumentParser(description=__doc__).parse_args()
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=8):
        main()
