"""Validate all real rows through the internal preprocessing before any training."""
from experiments.local.bootstrap import Path, CODE_DIR, DATA_ROOT, SUPPORT_ROOT, ARTIFACT_ROOT, asset_path, asset_key, support_path
import hashlib
import json
import pickle
import time
import numpy as np
import pandas as pd
from experiments.local.common import *
from experiments.local.common import _as_unit_rent, _numeric_params, _numeric_block, _category_columns, _fill_missing_categories

def main():
    started = time.time()
    train, test = load_data()
    assert 'MONTHLY_RENT' not in test
    assert train.row_id.is_unique and test.row_id.is_unique
    assert set(train.TOWN) == set(test.TOWN) and train.TOWN.nunique() == 26
    for split, data in [('train', train), ('test', test)]:
        raw = pd.read_csv(support_path(Path('Data') / f'{split}.csv'), dtype={'TOWN': str, 'BLOCK': str, 'STREET': str})
        pd.testing.assert_frame_equal(data[list(raw.columns)], raw, check_dtype=False)
        assert (data.floor_area_sqm > 0).all()
    assert not set(GROUP_MEAN_COLUMNS) & set(FEATURES['expanded'])
    assert 'BLOCK' not in FEATURES['expanded']
    all_rows = pd.concat([train.drop(columns='MONTHLY_RENT'), test], ignore_index=True)
    timeline = with_month(all_rows).groupby(['TOWN', 'month'], as_index=False)[['month_index', *LEVELS]].first()
    report = {'status': 'running', 'rows': {'train': len(train), 'test': len(test)}, 'towns': 26, 'feature_count': len(FEATURE_COLUMNS), 'row_identity': 'all source fields and original row order equal raw CSVs', 'support_feature_builder': 'Reference build_rental_features pure API or supplied compatible caches; this preparation checks source-row alignment and actual numeric transformations without claiming execution of removed legacy entry-point assertions', 'unused_target_features': GROUP_MEAN_COLUMNS, 'flat_model_ranking': 'refitted from each allowed training subset; no holdout targets', 'model_generated_inputs': 'NOT YET VALIDATED: no model trained; validate real residuals and three outputs during run', 'missing_by_split': {s: {c: int(n) for c, n in d.isna().sum().items() if n} for s, d in [('train', train), ('test', test)]}, 'cutoffs': []}
    for cutoff in cutoffs_needed():
        fit = before(train, cutoff)
        assert len(fit) and fit.TOWN.nunique() == 26
        (fit, apply), rank = ranked_frames(fit, [all_rows])
        cells = cell_table(_as_unit_rent(fit))
        numeric = _numeric_params(cells, ['month_index', *LEVELS])
        finite(_numeric_block(cells, ['month_index', *LEVELS], numeric), 'RNN train')
        finite(_numeric_block(timeline, ['month_index', *LEVELS], numeric), 'RNN full timeline')
        towns = []
        for town in sorted(fit.TOWN.unique()):
            ft = fit.loc[fit.TOWN == town].copy().reset_index(drop=True)
            ap = apply.loc[apply.TOWN == town].copy().reset_index(drop=True)
            stack._scale_coords(ft, [ap])
            ord_stats = stack._ordinal_stats(ft)
            for label, data in [('fit', ft), ('all_rows', ap)]:
                coords = np.column_stack([data.latitude, data.longitude, *stack._scaled_ordinals(data, ord_stats)])
                finite(coords, f'{town} GMM four observed covariates {label}')
            xfit, xapply = (static_matrix(plain_block(ft)), static_matrix(plain_block(ap)))
            center = np.nanmedian(xfit, axis=0)
            filled = np.where(np.isfinite(xfit), xfit, center)
            spread = np.std(filled, axis=0)
            spread[spread == 0] = 1
            finite((filled - center) / spread, 'MLP 16 observed fit attributes')
            finite((np.where(np.isfinite(xapply), xapply, center) - center) / spread, 'MLP 16 observed apply attributes')
            item = {'town': town, 'fit_rows': len(ft), 'apply_rows': len(ap), 'mlp_attribute_missing_before_fill': int(np.isnan(xapply).sum()), 'catboost': {}}
            for name, columns in FEATURES.items():
                preparer = fit_preprocessor(stack.PREPROCESS, ft, ft.MONTHLY_RENT / ft.floor_area_sqm, columns)
                assert not preparer.include_building_ and (not preparer.target_values_)
                state = pickle.dumps(preparer)
                unseen = {}
                for label, data in [('fit', ft), ('all_rows', ap)]:
                    matrix = transform_frame(preparer, data)
                    cats = _category_columns(matrix)
                    clean = _fill_missing_categories(matrix)
                    for col in clean:
                        if col not in (cats or []):
                            finite(clean[col], f'{town} {name} {label} {col}')
                    assert not clean.isna().any().any()
                    if label == 'all_rows':
                        unseen = {c: int(matrix[c].isna().sum()) for c in cats or []}
                assert pickle.dumps(preparer) == state, 'Transform changed fitted state'
                item['catboost'][name] = {'columns': len(columns), 'unseen_categories_mapped_to_missing_token': unseen}
            towns.append(item)
        report['cutoffs'].append({'fit_end': cutoff, 'fit_rows': len(fit), 'apply_rows': len(all_rows), 'rnn_cells': len(cells), 'towns': towns})
        print(f'PREP PASSED cutoff={cutoff} fit_rows={len(fit)} apply_rows={len(all_rows)} towns=26', flush=True)
        save_json(asset_path(Path('preparation.json')), report)
    report['status'] = 'passed_observed_inputs'
    report['elapsed_seconds'] = time.time() - started
    report['protocol_sha256'] = hashlib.sha256(asset_path(Path('protocol.json')).read_bytes()).hexdigest()
    report['csv_sha256'] = {s: hashlib.sha256(support_path(Path('Data/auxiliary_features') / f'{s}_features.csv').read_bytes()).hexdigest() for s in ['train', 'test']}
    save_json(asset_path(Path('preparation.json')), report)
    print(json.dumps({'status': report['status'], 'cutoffs': len(report['cutoffs']), 'elapsed_seconds': report['elapsed_seconds']}), flush=True)
if __name__ == '__main__':
    import argparse as _cli_argparse
    _cli_argparse.ArgumentParser(description=__doc__).parse_args()
    main()
