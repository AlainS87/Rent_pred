"""Locked temporal experiments. Selection never reads confirmation labels."""
from runtime import DATA_DIR, CODE_DIR, ANALYSIS
import runtime
from runtime import ANALYSIS
import json
import hashlib
import time
import joblib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from catboost import CatBoostRegressor
SPLITS = [{'name': 'development_1', 'cutoff': '2022-04', 'start': '2022-05', 'end': '2022-10', 'role': 'development'}, {'name': 'development_2', 'cutoff': '2022-10', 'start': '2022-11', 'end': '2023-04', 'role': 'development'}, {'name': 'development_3', 'cutoff': '2023-04', 'start': '2023-05', 'end': '2023-10', 'role': 'development'}, {'name': 'confirmation_17m', 'cutoff': '2023-10', 'start': '2023-11', 'end': '2025-03', 'role': 'confirmation'}]
PARAMS = dict(iterations=240, depth=6, learning_rate=0.065, l2_leaf_reg=8, loss_function='RMSE', random_seed=5228, thread_count=4, verbose=False, allow_writing_files=False)
SPATIAL_FAMILIES = ['block_structure', 'block_location', 'mrt', 'schools', 'malls']
SCENARIOS = ['fixed_cutoff', 'provided_lagged_aux']

def metric(y, p):
    error = np.asarray(p) - np.asarray(y)
    return {'rmse': float(np.sqrt(np.mean(error ** 2))), 'mae': float(np.mean(np.abs(error)))}

def prepare_frame(frame):
    result = frame.copy()
    categorical = []
    for c in result:
        if not pd.api.types.is_numeric_dtype(result[c]):
            result[c] = result[c].fillna('__MISSING__').astype(str)
            categorical.append(c)
        else:
            result[c] = pd.to_numeric(result[c], errors='coerce').astype(np.float32)
    return (result, categorical)

def select_nonstock_columns(train, families):
    selected = {}
    rows = []
    for family, cols in families.items():
        if family in ['stocks', 'stocks_adjusted_sensitivity']:
            continue
        keep = []
        seen = {}
        for c in cols:
            coverage = float(train[c].notna().mean())
            distinct = int(train[c].nunique(dropna=True))
            reason = 'retained'
            if distinct < 2:
                reason = 'constant_or_unobserved_in_training_window'
            elif coverage < 0.5:
                reason = 'training_coverage_below_50pct'
            else:
                h = int(pd.util.hash_pandas_object(train[c], index=False).sum())
                if h in seen and train[c].equals(train[seen[h]]):
                    reason = 'exact_duplicate_of_' + seen[h]
                else:
                    seen[h] = c
                    keep.append(c)
            rows.append({'family': family, 'feature': c, 'training_coverage': coverage, 'training_unique': distinct, 'decision': reason})
        selected[family] = keep
    return (selected, rows)

def prepare_fold(raw, families, split, scenario):
    train = raw.loc[raw['month'] <= split['cutoff']].copy()
    valid = raw.loc[raw['month'].between(split['start'], split['end'])].copy()
    assert train['month'].max() < valid['month'].min()
    monthly_source = pd.read_pickle(ANALYSIS / 'features' / 'macro_monthly.pkl')
    macro_cols = [c for c in families.get('coe', []) + families.get('stocks', []) if c in raw]
    if scenario == 'fixed_cutoff':
        anchor = valid.loc[valid['month'] == split['start'], macro_cols].iloc[0]
        valid.loc[:, macro_cols] = np.tile(anchor.to_numpy(), (len(valid), 1))
        if 'source_month_max' in valid:
            valid['source_month_max'] = split['cutoff']
    selected, decisions = select_nonstock_columns(train, families)
    stock_cols = [c for c in families.get('stocks', []) if pd.api.types.is_numeric_dtype(monthly_source[c]) and 'adjusted' not in c.lower() and ('adj_' not in c.lower())]
    train_months = sorted(train['month'].unique())
    monthly_train = monthly_source.loc[train_months, stock_cols]
    eligible = [c for c in stock_cols if monthly_train[c].notna().mean() >= 0.5 and monthly_train[c].nunique(dropna=True) > 1]
    components = []
    preprocessing = {'split': split, 'scenario': scenario, 'stock_candidates': eligible, 'main_stock_adjusted_prices': False}
    if eligible:
        mt = monthly_train[eligible].astype(float)
        lower = mt.quantile(0.01)
        upper = mt.quantile(0.99)
        imputer = SimpleImputer(strategy='median', keep_empty_features=True)
        scaler = StandardScaler()
        fitted = scaler.fit_transform(imputer.fit_transform(mt.clip(lower, upper, axis=1)))
        n = min(6, len(mt) - 1, len(eligible))
        pca = PCA(n_components=n, random_state=5228)
        pca.fit(fitted)
        preprocessing.update(stock_imputer=imputer, stock_scaler=scaler, stock_pca=pca, stock_lower_clip=lower, stock_upper_clip=upper)
        for f in [train, valid]:
            unique_months = sorted(f['month'].unique())
            requested = [split['start']] * len(unique_months) if f is valid and scenario == 'fixed_cutoff' else unique_months
            panel = monthly_source[eligible].reindex(requested).astype(float)
            transformed = pca.transform(scaler.transform(imputer.transform(panel.clip(lower, upper, axis=1))))
            transformed = pd.DataFrame(transformed, index=unique_months)
            for j in range(n):
                name = f'stock_pca_{j + 1}'
                f[name] = f['month'].map(transformed[j]).astype(np.float32)
        components = [f'stock_pca_{j + 1}' for j in range(n)]
        loading = pd.DataFrame(pca.components_.T, index=eligible, columns=components)
        loading.to_csv(ANALYSIS / 'results' / f"stock_pca_loadings_{split['name']}_{scenario}.csv")
        decisions.extend(({'family': 'stocks', 'feature': c, 'training_coverage': float(monthly_train[c].notna().mean()), 'training_unique': int(monthly_train[c].nunique(dropna=True)), 'decision': 'contributes_to_training_month_PCA' if c in eligible else 'excluded_from_main_PCA_missing_constant_or_adjusted'} for c in stock_cols))
    aggregate = []
    for group in ['all_stock', 'banks', 'property']:
        aggregate.extend([c for c in eligible if group in c.lower()][:6])
    selected['stocks'] = aggregate + components
    adjusted_cols = families.get('stocks_adjusted_sensitivity', [])
    adjusted = monthly_source.loc[train_months, adjusted_cols]
    adjusted_cols = [c for c in adjusted if adjusted[c].notna().mean() >= 0.5 and adjusted[c].nunique(dropna=True) > 1]
    adjusted_components = []
    if adjusted_cols:
        imputer_adj = SimpleImputer(strategy='median')
        scaler_adj = StandardScaler()
        mat = scaler_adj.fit_transform(imputer_adj.fit_transform(adjusted[adjusted_cols]))
        pca_adj = PCA(n_components=min(2, len(train_months) - 1, len(adjusted_cols)), random_state=5228).fit(mat)
        preprocessing.update(adjusted_candidates=adjusted_cols, adjusted_imputer=imputer_adj, adjusted_scaler=scaler_adj, adjusted_pca=pca_adj)
        for f in [train, valid]:
            unique = sorted(f['month'].unique())
            requested = [split['start']] * len(unique) if f is valid and scenario == 'fixed_cutoff' else unique
            panel = monthly_source[adjusted_cols].reindex(requested)
            value = pca_adj.transform(scaler_adj.transform(imputer_adj.transform(panel)))
            component_frame = pd.DataFrame(value, index=unique)
            for j in range(value.shape[1]):
                col = f'stock_adjusted_sensitivity_pca_{j + 1}'
                f[col] = f['month'].map(component_frame[j]).astype(np.float32)
        adjusted_components = [f'stock_adjusted_sensitivity_pca_{j + 1}' for j in range(pca_adj.n_components_)]
    decisions.extend(({'family': 'stocks_adjusted_sensitivity', 'feature': c, 'training_coverage': float(adjusted[c].notna().mean()), 'training_unique': int(adjusted[c].nunique(dropna=True)), 'decision': 'contributes_to_retrospective_sensitivity_PCA' if c in adjusted_cols else 'excluded_missing_or_constant_in_training_window'} for c in adjusted))
    selected['stocks_adjusted_sensitivity'] = adjusted_components
    for r in decisions:
        r.update(split=split['name'], scenario=scenario)
    if split['role'] == 'confirmation':
        joblib.dump(preprocessing, ANALYSIS / 'results' / f'confirmation_preprocessing_{scenario}.joblib')
        modeled_lineage = []
        for name in components:
            modeled_lineage.append({'feature': name, 'family': 'stocks', 'source_features': '|'.join(eligible), 'construction': 'PCA fitted on one observation per training month; see stored loadings and preprocessing', 'cutoff': split['cutoff'], 'scenario': scenario})
        for name in adjusted_components:
            modeled_lineage.append({'feature': name, 'family': 'stocks_adjusted_sensitivity', 'source_features': '|'.join(adjusted_cols), 'construction': 'Retrospective adjusted-close PCA sensitivity only', 'cutoff': split['cutoff'], 'scenario': scenario})
        pd.DataFrame(modeled_lineage).to_csv(ANALYSIS / 'results' / f'modeled_pca_lineage_{scenario}.csv', index=False)
    return (train, valid, selected, decisions)

def configurations(selected):
    original = selected['original']
    dwelling = [c for c in original if c not in ['block', 'street', 'address_key']]
    aux = [f for f in SPATIAL_FAMILIES + ['coe', 'stocks'] if f in selected]
    full = list(dict.fromkeys(original + [c for f in aux for c in selected[f]]))
    configs = {'original_dwelling': dwelling, 'original_address': original}
    for f in aux:
        configs['add_' + f] = list(dict.fromkeys(original + selected[f]))
    configs['full_aux'] = full
    if selected.get('stocks_adjusted_sensitivity'):
        configs['full_plus_adjusted_sensitivity'] = full + selected['stocks_adjusted_sensitivity']
    for f in aux:
        removed = set(selected[f])
        configs['remove_' + f] = [c for c in full if c not in removed]
    return configs

def ridge_fit_predict(train, valid, cols):
    tr, _ = prepare_frame(train[cols])
    va, _ = prepare_frame(valid[cols])
    cat = [c for c in tr if not pd.api.types.is_numeric_dtype(tr[c])]
    num = [c for c in tr if c not in cat]
    transform = ColumnTransformer([('numeric', make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler()), num), ('category', OneHotEncoder(handle_unknown='ignore', min_frequency=5, sparse_output=True), cat)], sparse_threshold=1.0)
    model = make_pipeline(transform, Ridge(alpha=100, solver='lsqr'))
    model.fit(tr, train['target'])
    return (model.predict(va), model)

def contribution_table(scores):
    rows = []
    for (split, scenario), part in scores.loc[scores['model'] == 'catboost'].groupby(['split', 'scenario']):
        scores_map = part.set_index('configuration')['rmse'].to_dict()
        for f in SPATIAL_FAMILIES + ['coe', 'stocks']:
            required = ['original_address', 'full_aux', 'add_' + f, 'remove_' + f]
            if not all((c in scores_map for c in required)):
                continue
            rows.append({'split': split, 'scenario': scenario, 'family': f, 'addition_rmse_gain': scores_map['original_address'] - scores_map['add_' + f], 'removal_rmse_cost': scores_map['remove_' + f] - scores_map['full_aux'], 'original_rmse': scores_map['original_address'], 'full_rmse': scores_map['full_aux']})
    return pd.DataFrame(rows)

def save_outputs(scores, monthly, selection, predictions):
    out = ANALYSIS / 'results'
    score_frame = pd.DataFrame(scores)
    score_frame.to_csv(out / 'scores.csv', index=False)
    pd.DataFrame(monthly).to_csv(out / 'monthly_scores.csv', index=False)
    pd.DataFrame(selection).to_csv(out / 'model_feature_selection.csv', index=False)
    contribution_table(score_frame).to_csv(out / 'family_contributions.csv', index=False)
    if predictions:
        pd.concat(predictions, ignore_index=True).to_csv(out / 'final_confirmation_predictions.csv.gz', index=False, compression='gzip')

def main():
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    out = ANALYSIS / 'results'
    out.mkdir(exist_ok=True)
    raw = pd.read_pickle(ANALYSIS / 'features' / 'train_features.pkl')
    families = json.loads((ANALYSIS / 'features' / 'feature_families.json').read_text())
    config = {'splits': SPLITS, 'scenarios': SCENARIOS, 'catboost_parameters': PARAMS, 'ridge_alpha': 100, 'target': 'MONTHLY_RENT', 'ignored_attributes': ['FURNISHED', 'FEE'], 'selection_policy': 'Training-only coverage/constant/exact-duplicate filters; stock PCA fitted to training months only, max 6 components; adjusted prices excluded from main models.', 'confirmation_policy': 'All configurations and parameters locked before confirmation evaluation; final scores are reported without retuning.', 'fixed_cutoff_policy': 'All validation macro descriptors frozen at cutoff+1 month features (lagged from completed cutoff month); no later actual macro observation.', 'provided_aux_policy': 'Each validation month uses preceding completed month auxiliary observations; this conditions forecasts on supplied future macro data.', 'static_snapshot_limit': 'Spatial files have no complete opening histories; all spatial results are conditional on supplied static snapshots.'}
    config_bytes = json.dumps(config, indent=2).encode()
    (out / 'experiment_config.json').write_bytes(config_bytes)
    (out / 'configuration_lock_sha256.txt').write_text(hashlib.sha256(config_bytes).hexdigest())
    scores = []
    monthly = []
    selection = []
    predictions = []
    development_best = {}
    for split in SPLITS:
        for scenario in SCENARIOS:
            train, valid, selected, decisions = prepare_fold(raw, families, split, scenario)
            selection.extend(decisions)
            configs = configurations(selected)
            y = valid['target'].to_numpy()
            mean_pred = np.repeat(train['target'].mean(), len(valid))
            scores.append(dict(split=split['name'], role=split['role'], scenario=scenario, model='mean', configuration='mean', features=0, training_rows=len(train), validation_rows=len(valid), seconds=0, **metric(y, mean_pred)))
            cache = {}
            for name, cols in configs.items():
                started = time.time()
                signature = tuple(cols)
                contains_macro = any((c in selected.get('coe', []) + selected.get('stocks', []) for c in cols))
                shared_key = (split['name'], name)
                if scenario == 'provided_lagged_aux' and (not contains_macro) and (shared_key in development_best):
                    pred, model = development_best[shared_key]
                else:
                    tr, cat = prepare_frame(train[cols])
                    va, _ = prepare_frame(valid[cols])
                    model = CatBoostRegressor(**PARAMS)
                    model.fit(tr, train['target'], cat_features=cat)
                    pred = model.predict(va)
                    if not contains_macro:
                        development_best[shared_key] = (pred, model)
                score = dict(split=split['name'], role=split['role'], scenario=scenario, model='catboost', configuration=name, features=len(cols), training_rows=len(train), validation_rows=len(valid), seconds=round(time.time() - started, 2), **metric(y, pred))
                scores.append(score)
                print(json.dumps(score), flush=True)
                prediction_frame = pd.DataFrame({'row_id': valid['row_id'].to_numpy(), 'month': valid['month'].to_numpy(), 'address_key': valid['address_key'].to_numpy(), 'town': valid['town'].to_numpy(), 'flat_type': valid['flat_type'].to_numpy(), 'target': y, 'prediction': pred, 'model': 'catboost', 'configuration': name, 'scenario': scenario, 'split': split['name'], 'seen_address': valid['address_key'].isin(train['address_key']).to_numpy()})
                for month, p in prediction_frame.groupby('month'):
                    monthly.append({'split': split['name'], 'scenario': scenario, 'model': 'catboost', 'configuration': name, 'month': month, 'horizon_month': int((pd.Period(month, 'M') - pd.Period(split['cutoff'], 'M')).n), 'rows': len(p), **metric(p['target'], p['prediction'])})
                if split['role'] == 'confirmation' and name in ['original_address', 'full_aux'] + [f'remove_{f}' for f in SPATIAL_FAMILIES + ['coe', 'stocks']]:
                    predictions.append(prediction_frame)
                if split['role'] == 'confirmation' and name == 'full_aux':
                    model.save_model(str(out / f'confirmation_model_{scenario}.cbm'))
                    (out / f'confirmation_features_{scenario}.json').write_text(json.dumps(cols, indent=2))
                    rng = np.random.default_rng(5228)
                    sample = valid.sample(min(2500, len(valid)), random_state=5228)
                    x, cat = prepare_frame(sample[cols])
                    from catboost import Pool
                    shap = model.get_feature_importance(Pool(x, cat_features=cat), type='ShapValues')[:, :-1]
                    importance = pd.DataFrame({'feature': cols, 'mean_abs_shap': np.abs(shap).mean(axis=0), 'mean_shap': shap.mean(axis=0)})
                    importance.to_csv(out / f'shap_importance_{scenario}.csv', index=False)
                save_outputs(scores, monthly, selection, predictions)
            for name in ['original_address', 'full_aux']:
                start = time.time()
                pred, _ = ridge_fit_predict(train, valid, configs[name])
                score = dict(split=split['name'], role=split['role'], scenario=scenario, model='ridge', configuration=name, features=len(configs[name]), training_rows=len(train), validation_rows=len(valid), seconds=round(time.time() - start, 2), **metric(y, pred))
                scores.append(score)
                print(json.dumps(score), flush=True)
            save_outputs(scores, monthly, selection, predictions)
    bootstrap = []
    all_pred = pd.concat(predictions, ignore_index=True)
    rng = np.random.default_rng(5228)
    for scenario, p in all_pred.groupby('scenario'):
        pivot = p.pivot(index='row_id', columns='configuration', values='prediction')
        meta = p.drop_duplicates('row_id').set_index('row_id').reindex(pivot.index)
        true = meta['target'].to_numpy()
        loss = (pivot.to_numpy() - true[:, None]) ** 2
        for unit in ['address_key', 'month']:
            labels = meta[unit].to_numpy()
            unique = np.unique(labels)
            loss_frame = pd.DataFrame(loss, index=labels)
            grouped_loss = loss_frame.groupby(level=0).sum().reindex(unique).to_numpy()
            grouped_count = pd.Series(1, index=labels).groupby(level=0).sum().reindex(unique).to_numpy()
            draws = []
            for b in range(300):
                if unit == 'month':
                    starts = rng.integers(0, len(unique) - 2, size=int(np.ceil(len(unique) / 3)))
                    picked = np.concatenate([np.arange(s, s + 3) for s in starts])[:len(unique)]
                else:
                    picked = rng.integers(0, len(unique), size=len(unique))
                draws.append(np.sqrt(grouped_loss[picked].sum(axis=0) / grouped_count[picked].sum()))
            draws = np.asarray(draws)
            lookup = {c: i for i, c in enumerate(pivot.columns)}
            for family in SPATIAL_FAMILIES + ['coe', 'stocks']:
                change = draws[:, lookup['remove_' + family]] - draws[:, lookup['full_aux']]
                bootstrap.append({'scenario': scenario, 'resampling_unit': unit, 'family': family, 'comparison': 'remove_family_minus_full_RMSE', 'low_95': float(np.quantile(change, 0.025)), 'high_95': float(np.quantile(change, 0.975)), 'median': float(np.median(change)), 'resamples': 300})
    pd.DataFrame(bootstrap).to_csv(out / 'clustered_family_intervals.csv', index=False)
    subgroup = []
    for (scenario, name), part in all_pred.loc[all_pred['configuration'].isin(['original_address', 'full_aux'])].groupby(['scenario', 'configuration']):
        part = part.copy()
        part['rent_quartile'] = pd.qcut(part['target'], 4, duplicates='drop').astype(str)
        for key in ['month', 'town', 'flat_type', 'seen_address', 'rent_quartile']:
            for value, p in part.groupby(key):
                subgroup.append({'scenario': scenario, 'configuration': name, 'grouping': key, 'group': str(value), 'rows': len(p), **metric(p['target'], p['prediction'])})
    pd.DataFrame(subgroup).to_csv(out / 'subgroup_scores.csv', index=False)
    spatial_scores = []
    rng = np.random.default_rng(5228)
    unique_blocks = np.sort(raw['address_key'].unique())
    held_blocks = set(rng.choice(unique_blocks, size=int(0.15 * len(unique_blocks)), replace=False))
    split = SPLITS[-1]
    spatial_raw = raw.loc[(raw['month'] > split['cutoff']) | ~raw['address_key'].isin(held_blocks)]
    for scenario in SCENARIOS:
        train, valid, selected, _ = prepare_fold(spatial_raw, families, split, scenario)
        valid = valid.loc[valid['address_key'].isin(held_blocks)]
        assert not set(valid['address_key']) & set(train['address_key'])
        configs = configurations(selected)
        for name in ['original_address', 'full_aux']:
            cols = configs[name]
            tr, cat = prepare_frame(train[cols])
            va, _ = prepare_frame(valid[cols])
            model = CatBoostRegressor(**PARAMS).fit(tr, train['target'], cat_features=cat)
            pred = model.predict(va)
            score = {'scenario': scenario, 'configuration': name, 'training_rows': len(train), 'validation_rows': len(valid), 'heldout_blocks': len(held_blocks), 'training_address_overlap': 0, **metric(valid['target'], pred)}
            spatial_scores.append(score)
            print('spatial_holdout ' + json.dumps(score), flush=True)
    pd.DataFrame(spatial_scores).to_csv(out / 'spatial_holdout_scores.csv', index=False)
    print('All locked temporal comparisons completed.', flush=True)
if __name__ == '__main__':
    import argparse as _cli_argparse
    _cli_argparse.ArgumentParser(description=__doc__).parse_args()
    main()
