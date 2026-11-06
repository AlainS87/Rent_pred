"""Independent saved-artifact audit; never fit or make a second Confirm prediction."""
from experiments.local.bootstrap import Path, CODE_DIR, DATA_ROOT, SUPPORT_ROOT, ARTIFACT_ROOT, asset_path, asset_key, support_path
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
from experiments.local import feature_ablation as shared
ROOT = Path(__file__).resolve().parent
OUT = asset_path(Path('spatial_history_01'))

def main():
    import numpy as np
    import pandas as pd
    import joblib
    from experiments.local._support.mean_residual_model.reginal_mean_model.sequence_store import town_slug
    from experiments.local import spatial_history_features as feature
    from catboost import CatBoostRegressor

    def forbidden(*args, **kwargs):
        raise RuntimeError('No fitting or prediction in completion verification')
    CatBoostRegressor.fit = forbidden
    CatBoostRegressor.predict = forbidden
    result = shared.read(OUT / 'results.json')
    resumed = (OUT / 'resume_protocol.json').exists()
    supervisor = shared.read(OUT / ('supervisor_resume.json' if resumed else 'supervisor.json'))
    assert result['status'] == supervisor['status'] == 'completed'
    assert result['exit_code'] == supervisor['exit_code'] == 0
    assert result['elapsed_seconds'] < 900
    protocol = shared.read(OUT / 'protocol.json')
    selection = shared.read(OUT / 'selection_frozen.json')
    assert shared.digest(OUT / 'protocol.json') == result['protocol_sha256'] == selection['protocol_sha256']
    assert shared.digest(OUT / 'selection_frozen.json') == result['selection_sha256']
    for relative, expected in protocol['frozen_sha256'].items():
        assert shared.digest(asset_path(Path(relative))) == expected, relative
    if resumed:
        resume = shared.read(OUT / 'resume_protocol.json')
        assert shared.digest(OUT / 'resume_protocol.json') == result['resume_protocol_sha256']
        assert resume['original_protocol_sha256'] == result['protocol_sha256']
        assert result['reused_completed_towns'] == 10 and result['newly_trained_towns'] == 16
        assert supervisor['cumulative_elapsed_seconds'] < 900
        assert result['prior_elapsed_seconds'] == resume['prior_elapsed_seconds']
        np.testing.assert_allclose(result['elapsed_seconds'], result['prior_elapsed_seconds'] + result['resume_elapsed_seconds'], atol=0.01, rtol=0)
        for relative, expected in resume['frozen_sha256'].items():
            assert shared.digest(asset_path(Path(relative))) == expected, relative
    for relative, expected in selection['candidate_sha256'].items():
        assert shared.digest(OUT / relative) == expected, relative
    cache = joblib.load(asset_path(Path('feature_ablation_01/training_cache.joblib')))
    assert set(cache['parts']) == {'train', 'search'}
    history = joblib.load(OUT / 'history_features.joblib')
    assert set(history) == {'train', 'search'}
    all_features = protocol['original_features'] + protocol['added_features']
    assert len(all_features) == 19 and protocol['added_features'] == feature.FEATURES
    fits = []
    for town in cache['towns']:
        folder = OUT / 'catboost' / town_slug(town)
        original_folder = asset_path(Path('baseline_alignment/stack/catboost') / town_slug(town))
        model = joblib.load(folder / 'model.joblib')
        original = joblib.load(original_folder / 'model.joblib')
        prep = joblib.load(folder / 'preprocessor.joblib')
        old = joblib.load(original_folder / 'preprocessor.joblib')
        audit = shared.read(folder / 'fit_audit.json')
        assert model.get_params() == original.get_params() == audit['params']
        assert model.get_all_params() == original.get_all_params() == audit['resolved_params']
        assert prep.selected_columns_ == all_features
        assert set(model.feature_names_) == set(all_features)
        assert prep.spec == old.spec == protocol['preprocess']
        for field in ('numeric_fill_', 'numeric_center_', 'numeric_spread_', 'levels_'):
            for name, value in getattr(old, field).items():
                assert getattr(prep, field)[name] == value
        assert audit['training_weights'] == 'none' and audit['original_feature_values_exactly_unchanged']
        fits.append({'town': town, 'features': len(model.feature_names_), 'trees': model.tree_count_, 'original_parameters_and_original_column_preprocessing_equal': True})

    def verify_csv(path, expected_ids, truth, expected_rmse):
        frame = pd.read_csv(path)
        np.testing.assert_array_equal(frame.source_row_id.to_numpy(), expected_ids)
        np.testing.assert_array_equal(frame.truth.to_numpy(), truth)
        assert frame.source_row_id.is_unique and len(frame) == len(expected_ids)
        assert np.isfinite(frame.prediction.to_numpy()).all()
        rmse = float(np.sqrt(np.mean((frame.prediction.to_numpy() - truth) ** 2)))
        np.testing.assert_allclose(rmse, expected_rmse, atol=1e-09, rtol=0)
        return {'rows': len(frame), 'finite': True, 'unique_original_row_ids': True, 'true_rents_exact': True, 'recomputed_rmse': rmse}
    passed = result['passed_search_gate']
    assert passed == selection['passed_search_gate'] == (result['search']['rmse'] < 451.042648255612)
    raw_count = 150000 if passed else 141076
    raw = pd.read_csv(asset_path(Path('support/Data/train.csv')), nrows=raw_count, usecols=['RENT_APPROVAL_DATE', 'MONTHLY_RENT'])
    month = raw.RENT_APPROVAL_DATE.astype(str).str[:7]
    fit_raw = raw.loc[month <= '2024-09', 'MONTHLY_RENT'].to_numpy(float)
    np.testing.assert_array_equal(cache['parts']['train'].MONTHLY_RENT, fit_raw)
    search_ids = np.flatnonzero(month.isin(protocol['search_months']).to_numpy())
    search_check = verify_csv(OUT / 'search_predictions.csv', search_ids, raw.MONTHLY_RENT.to_numpy(float)[search_ids], result['search']['rmse'])
    confirm_check = None
    if passed:
        start = shared.read(OUT / 'confirm_started.json')
        assert start['epoch'] >= selection['epoch']
        assert result['candidate_confirm_evaluations'] == 1
        ids = np.flatnonzero(month.isin(protocol['confirm_months']).to_numpy())
        confirm_check = verify_csv(OUT / 'confirm_predictions.csv', ids, raw.MONTHLY_RENT.to_numpy(float)[ids], result['confirm']['rmse'])
    else:
        assert result['candidate_confirm_evaluations'] == 0 and result['confirm'] is None
        for name in ('confirm_started.json', 'confirm_predictions.csv', 'confirmation_metrics.json', 'confirm_history_features.joblib'):
            assert not (OUT / name).exists(), name
    events = [json.loads(line) for line in (OUT / 'events.jsonl').read_text().splitlines()]
    assert sum((e['stage'] == 'runtime_start' for e in events)) == 1
    assert sum((e['stage'] == 'catboost_town_complete' for e in events)) == 26
    completed = [e for e in events if e['stage'] == 'catboost_town_complete']
    assert len({e['town'] for e in completed}) == 26
    if resumed:
        assert sum((e['stage'] == 'resume_runtime_start' for e in events)) == 1
        reused = [e for e in events if e['stage'] == 'catboost_town_reused']
        entered = [e for e in events if e['stage'] == 'catboost_fit_enter']
        verified = [e for e in events if e['stage'] == 'all_saved_models_verified']
        assert len(reused) == len({e['town'] for e in reused}) == 10
        assert len(entered) == len({e['town'] for e in entered}) == 16
        assert not {e['town'] for e in reused} & {e['town'] for e in entered}
        assert len(verified) == 1 and min((e['epoch'] for e in entered)) >= verified[0]['epoch']
        assert sum((e['stage'] == 'catboost_town_start' for e in events)) == 27
        assert shared.read(OUT / 'resume_model_verification.json')['all_ten_verified_before_new_fits']
    else:
        assert sum((e['stage'] == 'catboost_town_start' for e in events)) == 26
    assert sum((e['stage'] == 'confirm_prediction_started' for e in events)) == int(passed)
    fit = cache['parts']['train']
    fit_month = pd.PeriodIndex(fit.RENT_APPROVAL_DATE.astype(str).str[:7], freq='M')
    brute_checks = []
    for part_name in ('train', 'search'):
        queries = cache['parts'][part_name]
        for index in np.unique(np.linspace(0, len(queries) - 1, 10, dtype=int)):
            row = queries.iloc[index]
            if part_name == 'train':
                m = pd.Period(str(row.RENT_APPROVAL_DATE)[:7], freq='M')
                mask = (fit_month >= m - 6) & (fit_month < m)
            else:
                mask = (fit_month >= pd.Period('2024-04', freq='M')) & (fit_month <= pd.Period('2024-09', freq='M'))
            mask &= fit.flat_type_ord.to_numpy() == row.flat_type_ord
            ref = fit.loc[mask]
            lat = np.radians(ref.latitude.to_numpy(float))
            lon = np.radians(ref.longitude.to_numpy(float))
            qlat, qlon = np.radians([float(row.latitude), float(row.longitude)])
            a = np.sin((lat - qlat) / 2) ** 2 + np.cos(lat) * np.cos(qlat) * np.sin((lon - qlon) / 2) ** 2
            distance = 2 * 6371.0088 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
            rent = ref.MONTHLY_RENT.to_numpy(float)[distance <= 0.5]
            expected = [float(rent.mean()) if len(rent) else np.nan, len(rent), float(rent.std(ddof=1)) if len(rent) > 1 else np.nan]
            np.testing.assert_allclose(history[part_name].iloc[index].to_numpy(), expected, atol=1e-08, rtol=1e-10, equal_nan=True)
            brute_checks.append({'part': part_name, 'row': int(index), 'count': len(rent), 'passed': True})
    verification = {'status': 'passed', 'training_performed': False, 'new_predictions_performed': False, 'search': search_check, 'confirm': confirm_check, 'confirm_evaluations_in_candidate': int(passed), 'frozen_hashes_checked': len(protocol['frozen_sha256']), 'candidate_hashes_checked': len(selection['candidate_sha256']), 'per_town_checks': fits, 'independent_spatial_checks': brute_checks, 'all_original_labels_and_splits_unchanged': True, 'wall_time_under_900_seconds': True, 'supervisor_exit_code': supervisor['exit_code'], 'elapsed_seconds': result['elapsed_seconds'], 'authorized_resume': resumed, 'saved_ten_models_not_retrained': resumed, 'cumulative_supervised_seconds': supervisor.get('cumulative_elapsed_seconds', supervisor.get('elapsed_seconds'))}
    shared.save(OUT / 'completion_verification.json', verification)
    print(json.dumps({'verification': 'passed', 'elapsed_seconds': result['elapsed_seconds'], 'search': search_check, 'confirm': confirm_check, 'result': result}, indent=2))
if __name__ == '__main__':
    import argparse as _cli_argparse
    _cli_argparse.ArgumentParser(description=__doc__).parse_args()
    main()
