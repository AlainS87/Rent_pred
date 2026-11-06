"""Verify final research artifacts without retraining or changing locked inputs.

Run only after primary, history and report generation have completed. A failed
assertion leaves existing artifacts untouched. Success writes one audit JSON.
"""
from runtime import DATA_DIR, CODE_DIR, ANALYSIS
import sys
sys.dont_write_bytecode = True
import runtime
from runtime import PROJECT, ANALYSIS
import hashlib
import json
import time
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
FAMILIES = ['block_structure', 'block_location', 'mrt', 'schools', 'malls', 'coe', 'stocks']
SCENARIOS = ['fixed_cutoff', 'provided_lagged_aux']
EVIDENCE = []

def _check(condition, description, **details):
    assert bool(condition), description + (': ' + str(details) if details else '')
    EVIDENCE.append(dict(check=description, passed=True, **details))

def _hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def _metrics(target, prediction):
    error = np.asarray(prediction, dtype=float) - np.asarray(target, dtype=float)
    return (float(np.sqrt(np.mean(error ** 2))), float(np.mean(np.abs(error))))

def _match_metrics(actual, expected, description):
    _check(np.allclose(actual, expected, rtol=0, atol=1e-07), description, recomputed=[float(value) for value in actual], saved=[float(value) for value in expected])

def _verify_preservation():
    """Verify current locked research inputs; no publication-layout dependencies."""
    locks = json.loads((ANALYSIS / 'results' / 'feature_input_lock_sha256.json').read_text())
    for name, expected in locks.items():
        _check(_hash(ANALYSIS / name) == expected, 'locked feature input SHA256 unchanged', file=name)

def _verify_features(raw_train, raw_test, rental, test, monthly, families):
    coverage = pd.read_csv(ANALYSIS / 'audit' / 'source_attribute_coverage.csv')
    _check(len(coverage) == 63 and (not coverage.duplicated(['source_file', 'attribute']).any()), 'all 63 source attribute positions have unique coverage rows')
    expected = {(filename, column) for filename in ['train.csv'] + [path.name for path in (DATA_DIR / 'auxiliary').glob('*.csv')] for column in pd.read_csv(DATA_DIR / ('auxiliary/' + filename if filename.startswith('sg-') else filename), nrows=0).columns if column != 'MONTHLY_RENT'}
    _check(set(zip(coverage.source_file, coverage.attribute)) == expected, 'coverage rows match actual raw headers')
    ignored = coverage.loc[coverage.role == 'ignored_by_user', 'attribute']
    _check(set(ignored) == {'FURNISHED', 'FEE'} and len(ignored) == 2, 'only two explicitly ignored attributes')
    concepts = [column for columns in families.values() for column in columns]
    lineage = pd.read_csv(ANALYSIS / 'audit' / 'feature_lineage.csv')
    _check(len(concepts) == len(set(concepts)) == 1878, 'all 1878 conceptual candidates uniquely assigned')
    _check(set(concepts) == set(lineage.feature) and len(lineage) == 1878 and (not lineage.feature.duplicated().any()), 'every conceptual candidate has one construction lineage')
    declared_family = {column: family for family, columns in families.items() for column in columns}
    _check(all((declared_family[row.feature] == row.family for row in lineage.itertuples())), 'construction lineage family agrees with family manifest')
    _check(lineage[['source_file', 'source_attributes', 'construction', 'availability']].notna().all().all(), 'all lineage provenance fields populated')
    forbidden = {'row_id', 'month', 'target', 'MONTHLY_RENT', 'source_month_max', 'history_source_month_max'}
    _check(not forbidden.intersection(concepts) and (not any(('source_month' in name or 'availability' in name for name in concepts))), 'row identity, target and provenance metadata excluded from model families')
    _check(not {'furnished', 'fee', 'FURNISHED', 'FEE'}.intersection(concepts), 'ignored constants absent from model families')
    _check(all((column in rental or column in monthly for column in concepts)), 'every candidate is stored in rental or independent monthly layer')
    _check(all((column in test or column in monthly for column in concepts)), 'candidate storage schema consistent in test')
    for name, source, frame in [('train', raw_train, rental), ('test', raw_test, test)]:
        _check(len(frame) == len(source) and np.array_equal(frame.row_id.to_numpy(), np.arange(len(source))), 'row_id preserves original source order', dataset=name)
        _check(np.array_equal(frame.month.astype(str).to_numpy(), source.RENT_APPROVAL_DATE.astype(str).to_numpy()), 'rental approval months preserve original row order', dataset=name)
    _check(np.array_equal(rental.target.to_numpy(), raw_train.MONTHLY_RENT.to_numpy()), 'training targets equal untouched original source')
    _check('target' not in test, 'test contains no target')
    _check(monthly.index.is_unique and monthly.index.name == 'month', 'monthly feature panel has unique calendar index')
    prior = (pd.PeriodIndex(monthly.index, freq='M') - 1).astype(str)
    _check(monthly.source_month_max.astype(str).le(prior).all(), 'monthly provenance never later than preceding month')
    for family in ['coe', 'stocks', 'stocks_adjusted_sensitivity']:
        _check(set(families[family]).issubset(monthly.columns), 'macro candidates reside in independent monthly panel', family=family)

def _verify_score_grid(scores, monthly_scores, splits):
    configs = {'original_dwelling', 'original_address', 'full_aux', 'full_plus_adjusted_sensitivity'}
    configs.update(('add_' + family for family in FAMILIES))
    configs.update(('remove_' + family for family in FAMILIES))
    _check(len(scores) == 4 * 2 * 21, 'primary score table contains 168 locked comparison rows')
    _check(not scores.duplicated(['split', 'scenario', 'model', 'configuration']).any(), 'primary score keys unique')
    _check(not monthly_scores.duplicated(['split', 'scenario', 'model', 'configuration', 'month']).any(), 'primary monthly score keys unique')
    expected_month_keys = {(split['name'], scenario, 'catboost', configuration, str(month)) for split in splits for scenario in SCENARIOS for configuration in configs for month in pd.period_range(split['start'], split['end'], freq='M')}
    actual_month_keys = set(monthly_scores[['split', 'scenario', 'model', 'configuration', 'month']].itertuples(index=False, name=None))
    _check(actual_month_keys == expected_month_keys and len(monthly_scores) == len(expected_month_keys), 'complete predefined primary monthly comparison grid present', rows=len(monthly_scores))
    cutoff_by_split = {split['name']: pd.Period(split['cutoff'], freq='M') for split in splits}
    _check(all((row.horizon_month == pd.Period(row.month, freq='M').ordinal - cutoff_by_split[row.split].ordinal for row in monthly_scores.itertuples())), 'every primary monthly horizon agrees with cutoff chronology')
    for split in splits:
        for scenario in SCENARIOS:
            part = scores.loc[(scores.split == split['name']) & (scores.scenario == scenario)]
            _check(len(part) == 21 and part.model.value_counts().to_dict() == {'catboost': 18, 'ridge': 2, 'mean': 1}, 'each fold/scenario has mean one, CatBoost eighteen, Ridge two', split=split['name'], scenario=scenario)
            _check(set(part.loc[part.model == 'catboost', 'configuration']) == configs, 'every predefined CatBoost configuration evaluated', split=split['name'], scenario=scenario)
            _check(set(part.loc[part.model == 'ridge', 'configuration']) == {'original_address', 'full_aux'}, 'both interpretable Ridge baselines evaluated', split=split['name'], scenario=scenario)
    cat = scores.loc[scores.model == 'catboost'].set_index(['split', 'scenario', 'configuration'])
    for keys, part in monthly_scores.groupby(['split', 'scenario', 'configuration']):
        expected = cat.loc[keys]
        _check(int(part.rows.sum()) == int(expected.validation_rows), 'monthly counts reconstruct validation rows', keys=list(keys))
        recomputed = (np.sqrt(np.average(part.rmse.to_numpy() ** 2, weights=part.rows)), np.average(part.mae.to_numpy(), weights=part.rows))
        _match_metrics(recomputed, (expected.rmse, expected.mae), 'monthly errors reconstruct complete-fold errors')
    contributions = pd.read_csv(ANALYSIS / 'results' / 'family_contributions.csv')
    _check(len(contributions) == 7 * 4 * 2 and (not contributions.duplicated(['split', 'scenario', 'family']).any()), '56 unique family addition/removal comparisons present')
    for row in contributions.itertuples():
        base = cat.loc[(row.split, row.scenario, 'original_address'), 'rmse']
        full = cat.loc[(row.split, row.scenario, 'full_aux'), 'rmse']
        add = cat.loc[(row.split, row.scenario, 'add_' + row.family), 'rmse']
        remove = cat.loc[(row.split, row.scenario, 'remove_' + row.family), 'rmse']
        _check(np.allclose([base - add, remove - full, base, full], [row.addition_rmse_gain, row.removal_rmse_cost, row.original_rmse, row.full_rmse], atol=1e-08, rtol=0), 'family effects recompute from locked scores', family=row.family, split=row.split, scenario=row.scenario)

def _verify_predictions(path, scores, monthly_scores, raw_train, expected_ids, history=False):
    predictions = pd.read_csv(path, compression='gzip', float_precision='round_trip', dtype={'row_id': 'int32', 'configuration': 'category', 'scenario': 'category', 'split': 'category'})
    keys = ['split', 'scenario', 'configuration']
    expected_configs = {'original_address_plus_history', 'full_aux_plus_history'} if history else {'original_address', 'full_aux', *('remove_' + family for family in FAMILIES)}
    _check(set(predictions.configuration) == expected_configs and set(predictions.scenario) == set(SCENARIOS), 'confirmation predictions cover expected stored configurations', history=history)
    _check(set(predictions.split) == {'confirmation_17m'}, 'only confirmation rows in stored predictions', history=history)
    _check(len(predictions) == len(expected_ids) * len(expected_configs) * 2, 'stored prediction total complete', history=history)
    score_lookup = scores.loc[scores.model == 'catboost'].set_index(keys)
    month_lookup = monthly_scores.set_index(keys + ['month'])
    for group, part in predictions.groupby(keys, observed=True, sort=True):
        row_ids = part.row_id.to_numpy(dtype=int)
        _check(len(part) == 50096 and np.array_equal(row_ids, expected_ids), 'each confirmation prediction subset preserves all 50096 original rows', history=history, keys=list(group))
        _check(np.array_equal(part.month.to_numpy(), raw_train.RENT_APPROVAL_DATE.to_numpy()[row_ids]) and np.array_equal(part.target.to_numpy(), raw_train.MONTHLY_RENT.to_numpy()[row_ids]), 'prediction target and month match untouched source by row_id', history=history, keys=list(group))
        score = score_lookup.loc[group]
        _match_metrics(_metrics(part.target, part.prediction), (score.rmse, score.mae), 'confirmation score recomputed from saved predictions')
        for month, values in part.groupby('month', sort=True):
            saved = month_lookup.loc[*group, month]
            _check(int(saved.rows) == len(values), 'confirmation monthly row count matches prediction file', history=history)
            _match_metrics(_metrics(values.target, values.prediction), (saved.rmse, saved.mae), 'confirmation monthly errors recomputed')
    return predictions

def _verify_history(scores, splits):
    _check(len(scores) == 16 and (not scores.duplicated(['split', 'scenario', 'configuration']).any()), 'sixteen unique history comparisons completed')
    for split in splits:
        for scenario in SCENARIOS:
            part = scores.loc[(scores.split == split['name']) & (scores.scenario == scenario)]
            _check(len(part) == 2 and set(part.configuration) == {'original_address_plus_history', 'full_aux_plus_history'} and part.history_features.eq(27).all(), 'each history fold/scenario has two comparisons and 27 features', split=split['name'], scenario=scenario)
        audit = json.loads((ANALYSIS / 'results' / f"history_audit_{split['name']}.json").read_text())
        _check(audit['cutoff'] == split['cutoff'] and len(audit['used_groups']) == 9 and (len(audit['feature_columns']) == 27) and (len(set(audit['feature_columns'])) == 27), 'history audit has nine groups and 27 distinct feature columns', split=split['name'])
        _check(audit['smoothing_fixed'] == 20 and audit['same_month_labels_used_for_training_features'] is False and (audit['validation_labels_read'] is False) and (audit['validation_history_updates'] is False), 'history audit records strictly past training and frozen validation labels', split=split['name'])
        for row in audit['monthly_coverage']:
            latest = row['latest_label_month']
            legal = latest is None or (latest < row['month'] if row['dataset'] == 'train' else latest <= split['cutoff'])
            assert legal, 'history label provenance exceeds query time'
        _check(True, 'all history label source months satisfy availability cutoff', split=split['name'])
    lineage = pd.read_csv(ANALYSIS / 'results' / 'history_feature_lineage.csv')
    _check(len(lineage) == 4 * 27 and lineage.groupby('split').feature.nunique().eq(27).all(), 'all history features traceable within every fold')
    monthly_scores = pd.read_csv(ANALYSIS / 'results' / 'history_monthly_scores.csv')
    expected_month_keys = {(split['name'], scenario, configuration, str(month)) for split in splits for scenario in SCENARIOS for configuration in ['original_address_plus_history', 'full_aux_plus_history'] for month in pd.period_range(split['start'], split['end'], freq='M')}
    _check(set(monthly_scores[['split', 'scenario', 'configuration', 'month']].itertuples(index=False, name=None)) == expected_month_keys and len(monthly_scores) == len(expected_month_keys), 'complete predefined history monthly comparison grid present')
    lookup = scores.set_index(['split', 'scenario', 'configuration'])
    for keys, part in monthly_scores.groupby(['split', 'scenario', 'configuration']):
        score = lookup.loc[keys]
        _check(int(part.rows.sum()) == int(score.validation_rows), 'history monthly row counts reconstruct full fold', keys=list(keys))
        _match_metrics((np.sqrt(np.average(part.rmse.to_numpy() ** 2, weights=part.rows)), np.average(part.mae.to_numpy(), weights=part.rows)), (score.rmse, score.mae), 'history monthly errors reconstruct full-fold errors')
    tests = json.loads((ANALYSIS / 'results' / 'history_encoding_verification.json').read_text())
    test_keys = ['same_month_and_future_training_label_perturbation', 'validation_label_perturbation', 'first_month_is_cold_start', 'manual_alpha20_formula', 'unseen_group_global_past_fallback', 'identical_group_validation_history_frozen']
    _check(all((tests[key] is True for key in test_keys)) and tests['real_group_count'] == 9, 'independent leakage perturbation and smoothing verification passed')
    for scenario in SCENARIOS:
        frame = pd.read_csv(ANALYSIS / 'features' / f'history_confirmation_features_{scenario}.csv.gz', compression='gzip')
        columns = [name for name in frame if name.startswith('te_')]
        _check(len(frame) == 50096 and len(columns) == 27 and frame.history_source_month_max.eq('2023-10').all(), 'history confirmation snapshot has 27 columns and frozen October 2023 provenance', scenario=scenario)

def _replay_primary_models(rental, monthly, families, predictions, expected_ids):
    rng = np.random.default_rng(5228)
    sample_ids = np.sort(rng.choice(expected_ids, size=500, replace=False))
    base = rental.set_index('row_id').loc[sample_ids].copy()
    unique_months = sorted(rental.loc[rental.month.between('2023-11', '2025-03'), 'month'].unique())
    _check(len(unique_months) == 17, 'replay spans the complete seventeen-month confirmation panel')
    for scenario in SCENARIOS:
        artifact = joblib.load(ANALYSIS / 'results' / f'confirmation_preprocessing_{scenario}.joblib')
        _check(artifact['scenario'] == scenario and artifact['split']['cutoff'] == '2023-10' and (artifact['main_stock_adjusted_prices'] is False), 'saved preprocessing identifies correct scenario/cutoff', scenario=scenario)
        eligible = artifact['stock_candidates']
        _check(set(eligible).issubset(families['stocks']) and (not any(('adjusted' in column for column in eligible))), 'main stock PCA excludes retrospective adjusted prices', scenario=scenario)
        training_months = sorted(rental.loc[rental.month <= '2023-10', 'month'].unique())
        fitted = monthly.loc[training_months, eligible].astype(float)
        lower, upper = (artifact['stock_lower_clip'], artifact['stock_upper_clip'])
        _check(lower.index.tolist() == upper.index.tolist() == eligible and artifact['stock_imputer'].feature_names_in_.tolist() == eligible, 'saved stock candidate order agrees with clipping/imputer state', scenario=scenario)
        _check(np.allclose(lower.to_numpy(), fitted.quantile(0.01).to_numpy(), atol=1e-12, rtol=0, equal_nan=True) and np.allclose(upper.to_numpy(), fitted.quantile(0.99).to_numpy(), atol=1e-12, rtol=0, equal_nan=True), 'saved stock clipping thresholds fit training months exclusively', scenario=scenario)
        _check(np.allclose(artifact['stock_imputer'].statistics_, fitted.clip(lower, upper, axis=1).median().to_numpy(), atol=1e-12, rtol=0, equal_nan=True), 'saved stock imputer medians fit training months exclusively', scenario=scenario)
        observed = artifact['stock_imputer'].transform(fitted.clip(lower, upper, axis=1))
        _check(np.allclose(artifact['stock_scaler'].mean_, observed.mean(axis=0), atol=1e-12, rtol=0) and np.allclose(artifact['stock_scaler'].var_, observed.var(axis=0), atol=1e-12, rtol=0) and (int(artifact['stock_scaler'].n_samples_seen_) == 34) and (artifact['stock_pca'].n_samples_ == 34), 'stock scaling/PCA uses 34 independent training months', scenario=scenario)
        requested = ['2023-11'] * len(unique_months) if scenario == 'fixed_cutoff' else unique_months
        source = monthly.loc[requested, eligible].astype(float)
        value = artifact['stock_pca'].transform(artifact['stock_scaler'].transform(artifact['stock_imputer'].transform(source.clip(lower, upper, axis=1))))
        columns = [f'stock_pca_{j + 1}' for j in range(value.shape[1])]
        loadings = pd.read_csv(ANALYSIS / 'results' / f'stock_pca_loadings_confirmation_17m_{scenario}.csv', index_col=0, float_precision='round_trip')
        _check(loadings.index.tolist() == eligible and loadings.columns.tolist() == columns and np.allclose(loadings.to_numpy(), artifact['stock_pca'].components_.T, atol=1e-12, rtol=0), 'saved confirmation PCA loadings match actual preprocessing state', scenario=scenario)
        modeled = pd.read_csv(ANALYSIS / 'results' / f'modeled_pca_lineage_{scenario}.csv').set_index('feature')
        _check(set(columns).issubset(modeled.index) and all((modeled.loc[column, 'source_features'].split('|') == eligible for column in columns)) and modeled.loc[columns, 'cutoff'].eq('2023-10').all() and modeled.loc[columns, 'scenario'].eq(scenario).all(), 'modeled PCA features trace to exact fold-fitted candidate list', scenario=scenario)
        components = pd.DataFrame(value.astype(np.float32), index=unique_months, columns=columns)
        frame = base.copy()
        for column in columns:
            frame[column] = frame.month.map(components[column]).astype(np.float32)
        macro_columns = [column for column in families['coe'] + families['stocks'] if column in base]
        panel = monthly.loc[requested, macro_columns].copy()
        panel.index = unique_months
        if scenario == 'fixed_cutoff':
            anchor = rental.loc[rental.month == '2023-11', macro_columns].iloc[0]
            frame.loc[:, macro_columns] = np.tile(anchor.to_numpy(), (len(frame), 1))
            frame['source_month_max'] = '2023-10'
            pca_inputs = value.astype(np.float32)
            prepared_macro = frame[macro_columns + columns].to_numpy(dtype=np.float32)
            _check(np.allclose(pca_inputs, np.tile(pca_inputs[0], (len(value), 1)), atol=0, rtol=0, equal_nan=True) and np.allclose(prepared_macro, np.tile(prepared_macro[0], (len(frame), 1)), atol=0, rtol=0, equal_nan=True) and np.allclose(panel.to_numpy(), np.tile(panel.iloc[0].to_numpy(), (len(panel), 1)), atol=0, rtol=0, equal_nan=True) and frame.source_month_max.eq('2023-10').all(), 'fixed-cutoff COE, aggregates and actual float32 PCA inputs frozen across all seventeen future months', scenario=scenario, maximum_float64_PCA_roundoff=float(np.max(np.abs(value - np.tile(value[0], (len(value), 1))))), maximum_float32_PCA_input_difference=float(np.max(np.abs(pca_inputs - np.tile(pca_inputs[0], (len(value), 1))))))
        else:
            prior = (pd.PeriodIndex(unique_months, freq='M') - 1).astype(str)
            _check(monthly.loc[requested, 'source_month_max'].astype(str).to_numpy().tolist() == prior.tolist(), 'provided auxiliary scenario uses only each preceding completed month', scenario=scenario)
            for column in macro_columns:
                _check(np.allclose(frame[column].to_numpy(dtype=np.float32), frame.month.map(panel[column]).to_numpy(dtype=np.float32), atol=0, rtol=0, equal_nan=True), 'rental aggregate values match provided monthly panel', feature=column)
        feature_columns = json.loads((ANALYSIS / 'results' / f'confirmation_features_{scenario}.json').read_text())
        _check(not any(('adjusted' in column or 'source_month' in column or column in {'target', 'month', 'row_id'} for column in feature_columns)), 'saved main model feature list excludes target and adjusted/provenance columns', scenario=scenario)
        from run_experiments import prepare_frame
        prepared, categorical = prepare_frame(frame[feature_columns])
        model = CatBoostRegressor()
        model.load_model(str(ANALYSIS / 'results' / f'confirmation_model_{scenario}.cbm'))
        _check(model.feature_names_ == feature_columns, 'saved model feature order matches feature manifest', scenario=scenario)
        reproduced = model.predict(Pool(prepared, cat_features=categorical), thread_count=1)
        saved = predictions.loc[(predictions.scenario == scenario) & (predictions.configuration == 'full_aux')].set_index('row_id').loc[sample_ids, 'prediction'].to_numpy()
        difference = float(np.max(np.abs(reproduced - saved)))
        _check(difference <= 1e-06, 'saved model/preprocessing replay reproduces five hundred predictions', scenario=scenario, replay_rows=500, maximum_absolute_difference=difference)

def main():
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    EVIDENCE.clear()
    started = time.perf_counter()
    _verify_preservation()
    out = ANALYSIS / 'results'
    raw_train = pd.read_csv(DATA_DIR / 'train.csv')
    raw_test = pd.read_csv(DATA_DIR / 'query.csv')
    rental = pd.read_pickle(ANALYSIS / 'features' / 'train_features.pkl')
    test = pd.read_pickle(ANALYSIS / 'features' / 'test_features.pkl')
    monthly = pd.read_pickle(ANALYSIS / 'features' / 'macro_monthly.pkl')
    families = json.loads((ANALYSIS / 'features' / 'feature_families.json').read_text())
    _verify_features(raw_train, raw_test, rental, test, monthly, families)
    del test, raw_test
    config_bytes = (out / 'experiment_config.json').read_bytes()
    _check(hashlib.sha256(config_bytes).hexdigest() == (out / 'configuration_lock_sha256.txt').read_text().strip(), 'primary experiment configuration lock unchanged')
    config = json.loads(config_bytes)
    _check(config['scenarios'] == SCENARIOS and len(config['splits']) == 4, 'primary temporal experiment definitions complete')
    scores = pd.read_csv(out / 'scores.csv')
    monthly_scores = pd.read_csv(out / 'monthly_scores.csv')
    _verify_score_grid(scores, monthly_scores, config['splits'])
    for split in config['splits']:
        training_mask = raw_train.RENT_APPROVAL_DATE.le(split['cutoff'])
        validation_mask = raw_train.RENT_APPROVAL_DATE.between(split['start'], split['end'])
        rows = scores.loc[scores.split == split['name']]
        _check(rows.training_rows.eq(int(training_mask.sum())).all() and rows.validation_rows.eq(int(validation_mask.sum())).all(), 'all model fold sizes agree with untouched source chronology', split=split['name'])
        for row in rows.loc[rows.model == 'mean'].itertuples():
            prediction = np.repeat(raw_train.loc[training_mask, 'MONTHLY_RENT'].mean(), validation_mask.sum())
            _match_metrics(_metrics(raw_train.loc[validation_mask, 'MONTHLY_RENT'], prediction), (row.rmse, row.mae), 'mean baseline recomputed from strictly training labels')
    expected_ids = np.flatnonzero(raw_train.RENT_APPROVAL_DATE.between('2023-11', '2025-03').to_numpy())
    _check(len(expected_ids) == 50096, 'independent confirmation period contains 50096 original rows')
    predictions = _verify_predictions(out / 'final_confirmation_predictions.csv.gz', scores, monthly_scores, raw_train, expected_ids)
    history_bytes = (out / 'history_experiment_lock.json').read_bytes()
    _check(hashlib.sha256(history_bytes).hexdigest() == (out / 'history_lock_sha256.txt').read_text().strip(), 'history experiment configuration lock unchanged')
    history_scores = pd.read_csv(out / 'history_scores.csv')
    _verify_history(history_scores, config['splits'])
    _verify_predictions(out / 'history_confirmation_predictions.csv.gz', history_scores, pd.read_csv(out / 'history_monthly_scores.csv'), raw_train, expected_ids, history=True)
    notes = json.loads((out / 'association_notes.json').read_text())
    _check(notes['development_cutoff'] == '2023-10' and notes['development_rows'] == 99904 and (notes['development_months'] == 34) and (notes['confirmation_labels_used'] is False) and (notes['model_column_selection_uses_this_screen'] is False), 'conditional association screens restricted to development and excluded from model selection')
    _replay_primary_models(rental, monthly, families, predictions, expected_ids)
    for name, expected in json.loads((out / 'feature_input_lock_sha256.json').read_text()).items():
        _check(_hash(ANALYSIS / name) == expected, 'verification leaves locked input unchanged', file=name)
    result = {'status': 'passed', 'checks_passed': len(EVIDENCE), 'retrained_models': 0, 'model_replay_rows_per_scenario': 500, 'prediction_replay_tolerance': 1e-06, 'primary_scores': len(scores), 'history_scores': len(history_scores), 'conceptual_feature_count': 1878, 'confirmation_rows_per_configuration': 50096, 'elapsed_seconds': round(time.perf_counter() - started, 3), 'checks': EVIDENCE}
    (ANALYSIS / 'audit' / 'verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({key: value for key, value in result.items() if key != 'checks'}), flush=True)
if __name__ == '__main__':
    import argparse as _cli_argparse
    _cli_argparse.ArgumentParser(description=__doc__).parse_args()
    main()
