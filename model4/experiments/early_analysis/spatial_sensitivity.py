"""Coordinate representation sensitivity, without changing locked features.

Fewer than four recorded decimals is a precision flag, not evidence that a
coordinate is wrong. The exclusion is an intentionally separate stress test.
"""
from runtime import DATA_DIR, CODE_DIR, ANALYSIS
import runtime
from runtime import PROJECT, ANALYSIS
import json
import re
import time
import numpy as np
import pandas as pd
from spatial import _prepare_facilities, _haversine, normalize
from build_features import original_features
from associations import CUTOFF, _control_design, _residualizer, _cluster_effect

def _decimal_digits(value):
    match = re.fullmatch('[+-]?\\d+(?:\\.(\\d*))?', str(value).strip())
    return len(match.group(1) or '') if match else None

def main():
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    audit_dir, out = (ANALYSIS / 'audit', ANALYSIS / 'results')
    audit_dir.mkdir(exist_ok=True)
    out.mkdir(exist_ok=True)
    raw = pd.read_csv(DATA_DIR / 'auxiliary' / 'sg-mrt-stations.csv', dtype=str, keep_default_na=False)
    precision = raw.copy()
    precision['latitude_decimal_digits'] = raw['LATITUDE'].map(_decimal_digits)
    precision['longitude_decimal_digits'] = raw['LONGITUDE'].map(_decimal_digits)
    precision['coarse_precision_flag'] = precision['latitude_decimal_digits'].lt(4) | precision['longitude_decimal_digits'].lt(4)
    precision['definition'] = 'either original coordinate string has fewer than four decimal places; not a correctness judgment'
    precision.to_csv(audit_dir / 'mrt_coordinate_precision.csv', index=False)
    coarse_by_entity = {}
    for _, row in precision.iterrows():
        key = (normalize(row['NAME']), float(row['LATITUDE']), float(row['LONGITUDE']))
        coarse_by_entity[key] = coarse_by_entity.get(key, False) or bool(row['coarse_precision_flag'])
    mrt, entity_audit = _prepare_facilities(raw, multi_line_station=True)
    coarse = np.array([coarse_by_entity[row['NAME'], row['LATITUDE'], row['LONGITUDE']] for _, row in mrt.iterrows()], dtype=bool)
    open_mask = mrt['STATUS'].map(lambda text: 'open' in (text or '').split('|')).to_numpy(dtype=bool)
    precise_open_mask = open_mask & ~coarse
    if not precise_open_mask.any():
        raise ValueError('No open entity remains in the coordinate-precision sensitivity')
    blocks = pd.read_pickle(ANALYSIS / 'features' / 'block_features.pkl')
    distances = _haversine(blocks[['hdb_latitude', 'hdb_longitude']].to_numpy(dtype=float), mrt[['LATITUDE', 'LONGITUDE']].to_numpy(dtype=float))
    open_ids, precise_ids = (np.flatnonzero(open_mask), np.flatnonzero(precise_open_mask))
    baseline_argmin = open_ids[distances[:, open_mask].argmin(axis=1)]
    sensitivity_argmin = precise_ids[distances[:, precise_open_mask].argmin(axis=1)]
    baseline = distances[np.arange(len(blocks)), baseline_argmin]
    sensitivity = distances[np.arange(len(blocks)), sensitivity_argmin]
    assert np.all(sensitivity >= baseline - 1e-12)
    assert np.allclose(baseline, blocks['mrt_status_open_nearest_km'].to_numpy(), atol=1e-12)
    absolute_change = np.abs(sensitivity - baseline)
    changed = absolute_change > 1e-09
    nearest_entity_changed = baseline_argmin != sensitivity_argmin
    nearest_coarse = coarse[baseline_argmin]
    rental = pd.read_csv(DATA_DIR / 'train.csv', dtype={'BLOCK': 'string'})
    controls, _ = original_features(rental)
    for column in controls.select_dtypes(include=['float64']).columns:
        controls[column] = controls[column].astype('float32')
    controls['month'] = rental['RENT_APPROVAL_DATE'].astype(str)
    all_keys = controls['address_key']
    lookup = pd.DataFrame({'baseline': baseline, 'sensitivity': sensitivity, 'delta': absolute_change, 'changed': changed, 'nearest_changed': nearest_entity_changed, 'nearest_coarse': nearest_coarse}, index=blocks.index)
    aligned = lookup.reindex(all_keys)
    assert len(aligned) == len(rental) and aligned['baseline'].notna().all()
    dev_mask = controls['month'].le(CUTOFF)
    dev = controls.loc[dev_mask].copy()
    target = pd.to_numeric(rental.loc[dev_mask, 'MONTHLY_RENT']).to_numpy(dtype=float)
    design = _control_design(dev, include_month=True)
    residualize = _residualizer(design)
    y = residualize(target)
    addresses = dev['address_key'].to_numpy()
    existing = pd.read_csv(out / 'controlled_spatial_associations.csv').set_index('feature')
    existing_beta = float(existing.loc['mrt_status_open_nearest_km', 'coefficient_sgd_per_unit'])
    raw_open = raw['STATUS'].map(normalize).eq('open')
    summary = {'raw_rows': len(raw), 'station_entities': len(mrt), 'coarse_raw_rows': int(precision['coarse_precision_flag'].sum()), 'coarse_entities': int(coarse.sum()), 'coarse_open_raw_rows': int((precision['coarse_precision_flag'] & raw_open).sum()), 'coarse_open_entities': int((coarse & open_mask).sum()), 'open_entities_include_all': int(open_mask.sum()), 'open_entities_exclude_coarse': int(precise_open_mask.sum()), 'coarse_nearest_block_fraction': float(nearest_coarse.mean()), 'coarse_nearest_rental_fraction': float(aligned['nearest_coarse'].mean())}
    rows = []
    for name, column in [('include_all', 'baseline'), ('exclude_coarse', 'sensitivity')]:
        x = aligned.loc[dev_mask.to_numpy(), column].to_numpy(dtype=np.float32).astype(float)
        effect = _cluster_effect(residualize(x), y, addresses, design.shape[1])
        if name == 'include_all':
            assert abs(effect['coefficient_sgd_per_unit'] - existing_beta) < 1e-07
        rows.append({'scenario': name, 'development_cutoff': CUTOFF, 'coefficient_sgd_per_km': effect['coefficient_sgd_per_unit'], 'ci_low_95': effect['ci_low_95'], 'ci_high_95': effect['ci_high_95'], 'address_clusters': effect['address_clusters'], 'development_rows': len(dev), 'changed_block_fraction': float(changed.mean()) if name == 'exclude_coarse' else 0.0, 'nearest_entity_changed_block_fraction': float(nearest_entity_changed.mean()) if name == 'exclude_coarse' else 0.0, 'changed_rental_fraction_150000': float(aligned['changed'].mean()) if name == 'exclude_coarse' else 0.0, 'mean_abs_distance_change_km': float(absolute_change.mean()) if name == 'exclude_coarse' else 0.0, 'median_abs_distance_change_km': float(np.median(absolute_change)) if name == 'exclude_coarse' else 0.0, 'p90_abs_distance_change_km': float(np.quantile(absolute_change, 0.9)) if name == 'exclude_coarse' else 0.0, 'max_abs_distance_change_km': float(absolute_change.max()) if name == 'exclude_coarse' else 0.0, 'affected_blocks_mean_abs_change_km': float(absolute_change[changed].mean()) if name == 'exclude_coarse' and changed.any() else 0.0, 'affected_blocks_median_abs_change_km': float(np.median(absolute_change[changed])) if name == 'exclude_coarse' and changed.any() else 0.0, 'affected_blocks_p90_abs_change_km': float(np.quantile(absolute_change[changed], 0.9)) if name == 'exclude_coarse' and changed.any() else 0.0, 'mean_abs_rental_distance_change_km': float(aligned['delta'].mean()) if name == 'exclude_coarse' else 0.0, **summary})
    table = pd.DataFrame(rows)
    table.to_csv(out / 'spatial_coordinate_sensitivity.csv', index=False)
    schools = pd.read_csv(audit_dir / 'spatial_schools_entities.csv')
    same_coordinate = schools.groupby(['LATITUDE', 'LONGITUDE'])['NAME'].agg(list)
    shared = same_coordinate[same_coordinate.map(len).gt(1)]
    raw_schools = pd.read_csv(DATA_DIR / 'auxiliary' / 'sg-schools.csv', dtype=str)
    school_entity_keys = set(zip(raw_schools['NAME'].map(normalize), pd.to_numeric(raw_schools['LATITUDE']), pd.to_numeric(raw_schools['LONGITUDE'])))
    assert len(schools) == len(school_entity_keys) == 337
    notes = {'precision_definition': 'original LATITUDE or LONGITUDE string has fewer than 4 decimal places', 'interpretation': 'low recorded precision does not establish an erroneous location; exclusion is a separate stress test, not corrected truth', 'representation_limit': 'a shortened decimal string can omit trailing zeroes; decimal count alone does not establish positional accuracy', 'exclusion_limit': 'excluding entities changes the facility universe as well as coordinate precision; resulting distances are not corrected locations', 'locked_main_features_and_models_changed': False, 'main_model_retrained': False, 'distance_definition': 'nearest supplied snapshot-open station entity; straight-line haversine kilometres', 'controls': 'same dwelling/town/month controls and building-cluster sandwich intervals as associations.py', 'development_cutoff': CUTOFF, 'confirmation_labels_used': False, 'static_snapshot_limit': 'open describes the provided snapshot, not historical operations', 'baseline_coefficient_validation': 'recomputed include_all coefficient matches controlled_spatial_associations.csv within 1e-7 SGD/km', 'delta_reference': 'exclude_coarse minus include_all; distance changes calculated for all supplied HDB buildings and separately weighted across 150000 unchanged rental rows', 'precision_latitude_decimal_distribution': precision['latitude_decimal_digits'].value_counts().sort_index().to_dict(), 'precision_longitude_decimal_distribution': precision['longitude_decimal_digits'].value_counts().sort_index().to_dict(), 'schools_unique_name_coordinate_entities': len(schools), 'schools_shared_coordinate_groups': len(shared), 'schools_shared_coordinate_entity_names': shared.to_list(), 'school_preservation': 'different named schools on identical coordinates remain separate entities', 'station_entity_definition': entity_audit['entity_definition'], 'summary': summary, 'elapsed_seconds': round(time.perf_counter() - start, 3)}
    (out / 'spatial_sensitivity_notes.json').write_text(json.dumps(notes, indent=2), encoding='utf-8')
    print(table.to_json(orient='records'), flush=True)
if __name__ == '__main__':
    import argparse as _cli_argparse
    _cli_argparse.ArgumentParser(description=__doc__).parse_args()
    main()
