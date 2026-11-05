"""Rebuild all inputs from untouched source CSVs, preserving rental row order."""
from runtime import DATA_DIR, CODE_DIR, ANALYSIS
import runtime
from runtime import PROJECT, ANALYSIS
from pathlib import Path
import hashlib
import json
import re
import numpy as np
import pandas as pd

def norm(series):
    return series.astype('string').str.strip().str.lower().str.replace('\\s+', ' ', regex=True)

def address_key(frame):
    return norm(frame['TOWN']) + '|' + norm(frame['BLOCK']) + '|' + norm(frame['STREET'])

def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def original_features(raw):
    dates = pd.to_datetime(raw['RENT_APPROVAL_DATE'], format='%Y-%m')
    flat = norm(raw['FLAT_TYPE']).str.replace('-', ' ', regex=False).str.replace('\\s+', ' ', regex=True)
    data = {'town': norm(raw['TOWN']).astype(object), 'block': norm(raw['BLOCK']).astype(object), 'street': norm(raw['STREET']).astype(object), 'address_key': address_key(raw).astype(object), 'flat_type': flat.astype(object), 'flat_model': norm(raw['FLAT_MODEL']).astype(object), 'floor_area_sqm': pd.to_numeric(raw['FLOOR_AREA_SQM'], errors='coerce'), 'lease_commence_year': pd.to_numeric(raw['LEASE_COMMENCE_DATE'], errors='coerce'), 'approval_month_index': (dates.dt.year - 2021) * 12 + dates.dt.month - 1, 'approval_year': dates.dt.year, 'approval_month_sin': np.sin(2 * np.pi * dates.dt.month / 12), 'approval_month_cos': np.cos(2 * np.pi * dates.dt.month / 12)}
    f = pd.DataFrame(data, index=raw.index)
    f['log_floor_area'] = np.log(f['floor_area_sqm'].where(f['floor_area_sqm'] > 0))
    f['lease_age'] = dates.dt.year - f['lease_commence_year']
    f['lease_age'] = f['lease_age'].where(f['lease_age'] >= 0)
    original = list(f.columns)
    return (f, original)

def build_features():
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    from spatial import build_spatial
    from macro import build_macro
    data_dir = DATA_DIR
    out = ANALYSIS / 'features'
    audit_dir = ANALYSIS / 'audit'
    out.mkdir(exist_ok=True)
    audit_dir.mkdir(exist_ok=True)
    train = pd.read_csv(data_dir / 'train.csv', dtype={'BLOCK': 'string'})
    test = pd.read_csv(data_dir / 'query.csv', dtype={'BLOCK': 'string'})
    source_hashes = {str(p.relative_to(DATA_DIR)): file_hash(p) for p in sorted(data_dir.rglob('*')) if p.is_file()}
    print('Constructing HDB, MRT, school and mall features...', flush=True)
    blocks, spatial_families, spatial_lineage, spatial_audit = build_spatial(data_dir, audit_dir)
    assert blocks.index.is_unique and blocks.index.name == 'address_key'
    print('Constructing COE and all-stock features...', flush=True)
    monthly, macro_families, macro_lineage, macro_audit = build_macro(data_dir, audit_dir)
    monthly.index = monthly.index.astype(str)
    monthly.index.name = 'month'
    assert monthly.index.is_unique
    families = dict(spatial_families)
    families.update(macro_families)
    original_sources = {'town': 'TOWN', 'block': 'BLOCK', 'street': 'STREET', 'address_key': 'TOWN|BLOCK|STREET', 'flat_type': 'FLAT_TYPE', 'flat_model': 'FLAT_MODEL', 'floor_area_sqm': 'FLOOR_AREA_SQM', 'lease_commence_year': 'LEASE_COMMENCE_DATE', 'approval_month_index': 'RENT_APPROVAL_DATE', 'approval_year': 'RENT_APPROVAL_DATE', 'approval_month_sin': 'RENT_APPROVAL_DATE', 'approval_month_cos': 'RENT_APPROVAL_DATE', 'log_floor_area': 'FLOOR_AREA_SQM', 'lease_age': 'RENT_APPROVAL_DATE|LEASE_COMMENCE_DATE'}
    original_lineage = [{'feature': feature, 'family': 'original', 'source_file': 'train.csv', 'source_attributes': attrs, 'construction': 'Normalized original attribute / approval-time transformation; no target data', 'availability': 'rental row attribute'} for feature, attrs in original_sources.items()]
    lineage = original_lineage + spatial_lineage + macro_lineage
    built = {}
    join_audit = []
    for name, raw in [('train', train), ('test', test)]:
        f, base = original_features(raw)
        aligned_blocks = blocks.reindex(f['address_key']).reset_index(drop=True)
        aligned_blocks.index = raw.index
        assert len(aligned_blocks) == len(raw)
        if 'address_key' in aligned_blocks:
            aligned_blocks = aligned_blocks.drop(columns='address_key')
        compact_macro_cols = macro_families.get('coe', []) + [c for c in macro_families.get('stocks', []) if any((t in c.lower() for t in ['all_stock', 'banks', 'property']))]
        if 'source_month_max' in monthly:
            compact_macro_cols.append('source_month_max')
        aligned_macro = monthly[compact_macro_cols].reindex(raw['RENT_APPROVAL_DATE']).reset_index(drop=True)
        aligned_macro.index = raw.index
        overlap = set(f.columns) & set(aligned_blocks.columns) | set(f.columns) & set(aligned_macro.columns) | set(aligned_blocks.columns) & set(aligned_macro.columns)
        assert not overlap, overlap
        f = pd.concat([f, aligned_blocks, aligned_macro], axis=1)
        completion = next((c for c in ['year_completed', 'hdb_year_completed', 'block_year_completed'] if c in f), None)
        if completion:
            f['building_age'] = f['approval_year'] - pd.to_numeric(f[completion], errors='coerce')
            f['building_age'] = f['building_age'].where(f['building_age'] >= 0)
            f['lease_completion_gap'] = f['lease_commence_year'] - pd.to_numeric(f[completion], errors='coerce')
            if name == 'train':
                families.setdefault('block_structure', []).extend(['building_age', 'lease_completion_gap'])
                for feature in ['building_age', 'lease_completion_gap']:
                    lineage.append({'feature': feature, 'family': 'block_structure', 'source_file': 'train.csv|sg-hdb-block.csv', 'source_attributes': 'RENT_APPROVAL_DATE|LEASE_COMMENCE_DATE|YEAR_COMPLETED', 'construction': 'approval-time age / lease minus completion year; no negative age', 'availability': 'static supplied building snapshot plus rental date'})
        interactions = []
        if 'building_age' in f and 'hdb_region' in f:
            f['building_age_x_central_region'] = f['building_age'] * f['hdb_region'].eq('central region').astype(float)
            interactions.append(('building_age_x_central_region', 'block_structure', 'train.csv|sg-hdb-block.csv', 'RENT_APPROVAL_DATE|YEAR_COMPLETED|REGION', 'building age times supplied Central Region indicator'))
        if 'school_level_primary_nearest_km' in f:
            f['primary_distance_x_large_flat'] = f['school_level_primary_nearest_km'] * f['floor_area_sqm'].ge(100).astype(float)
            interactions.append(('primary_distance_x_large_flat', 'schools', 'train.csv|sg-schools.csv', 'FLOOR_AREA_SQM|MAINLEVEL_CODE|LATITUDE|LONGITUDE', 'nearest primary distance times area >= 100 square metres; fixed candidate threshold'))
        if 'mrt_status_open_nearest_km' in f and 'cbd_raffles_place_distance_km' in f:
            f['mrt_open_distance_x_cbd_distance'] = f['mrt_status_open_nearest_km'] * f['cbd_raffles_place_distance_km']
            interactions.append(('mrt_open_distance_x_cbd_distance', 'mrt', 'sg-mrt-stations.csv|sg-hdb-block.csv', 'STATUS|LATITUDE|LONGITUDE|NAME', 'snapshot-open MRT distance times Raffles Place distance'))
        if name == 'train':
            for feature, family, source, attrs, rule in interactions:
                families[family].append(feature)
                lineage.append({'feature': feature, 'family': family, 'source_file': source, 'source_attributes': attrs, 'construction': rule, 'availability': 'rental row and provided static spatial snapshots'})
        f.insert(0, 'row_id', np.arange(len(raw)))
        f.insert(1, 'month', raw['RENT_APPROVAL_DATE'].astype(str))
        if name == 'train':
            f['target'] = pd.to_numeric(raw['MONTHLY_RENT'], errors='raise')
        for col in f.select_dtypes(include=['float64']).columns:
            f[col] = f[col].astype('float32')
        assert f['row_id'].tolist() == list(range(len(raw)))
        assert not set(['FURNISHED', 'FEE', 'furnished', 'fee']) & set(f.columns)
        numeric = f.select_dtypes(include=[np.number])
        assert not np.isinf(numeric.to_numpy(dtype=float)).any()
        f.to_pickle(out / f'{name}_features.pkl')
        built[name] = f
        match = f['address_key'].isin(blocks.index)
        assert match.all(), f'Unmatched addresses in {name}; inspect before modeling'
        join_audit.append({'dataset': name, 'rows_before': len(raw), 'rows_after': len(f), 'matched_rows': int(match.sum()), 'match_rate': float(match.mean()), 'months': raw['RENT_APPROVAL_DATE'].nunique(), 'first_month': raw['RENT_APPROVAL_DATE'].min(), 'last_month': raw['RENT_APPROVAL_DATE'].max()})
        raw.groupby('RENT_APPROVAL_DATE').size().rename('rows').to_csv(audit_dir / f'{name}_monthly_counts.csv')
    families['original'] = base
    families = {k: list(dict.fromkeys(v)) for k, v in families.items()}
    absent = {k: [c for c in v if c not in (monthly if k in macro_families else built['train'])] for k, v in families.items()}
    assert not any(absent.values()), absent
    accounted = set((c for v in families.values() for c in v))
    extra = set(built['train'].columns) - accounted - {'row_id', 'month', 'target'}
    metadata = sorted((c for c in extra if 'source_month' in c or 'availability' in c))
    unexplained = extra - set(metadata)
    assert not unexplained, f'Unassigned predictor columns: {unexplained}'
    (out / 'feature_families.json').write_text(json.dumps(families, indent=2), encoding='utf-8')
    blocks.to_pickle(out / 'block_features.pkl')
    monthly.to_pickle(out / 'macro_monthly.pkl')
    pd.DataFrame(lineage).to_csv(audit_dir / 'feature_lineage.csv', index=False)
    pd.DataFrame(join_audit).to_csv(audit_dir / 'join_audit.csv', index=False)
    coverage = []
    source_files = [data_dir / 'train.csv'] + sorted((data_dir / 'auxiliary').glob('*.csv'))
    for path in source_files:
        raw = pd.read_csv(path)
        for attr in raw:
            if attr == 'MONTHLY_RENT':
                continue
            matches = [r for r in lineage if path.name in str(r['source_file']).split('|') and attr in str(r['source_attributes']).split('|')]
            if path.name == 'train.csv':
                status = 'ignored_by_user' if attr in {'FURNISHED', 'FEE'} else 'original_predictor_and_construction'
                reason = 'User explicitly requested exclusion.' if attr in {'FURNISHED', 'FEE'} else 'Original attributes retained after normalization; date and lease ages constructed.'
            else:
                status = 'constructed' if matches else 'identity_or_metadata_audited'
                reason = 'Used in candidate construction; see lineage.' if matches else 'Entity/join/metadata role; not interpreted as an independent causal factor.'
            coverage.append({'source_file': path.name, 'attribute': attr, 'raw_rows': len(raw), 'raw_non_null': int(raw[attr].notna().sum()), 'raw_unique': int(raw[attr].nunique(dropna=True)), 'role': status, 'candidate_feature_count': len(matches), 'candidate_features': '|'.join((r['feature'] for r in matches)), 'reason': reason})
    assert len(coverage) == 63
    pd.DataFrame(coverage).to_csv(audit_dir / 'source_attribute_coverage.csv', index=False)
    feature_coverage = []
    for family, cols in families.items():
        for col in cols:
            train_values = built['train'][col] if col in built['train'] else monthly[col].reindex(train['RENT_APPROVAL_DATE'])
            test_values = built['test'][col] if col in built['test'] else monthly[col].reindex(test['RENT_APPROVAL_DATE'])
            feature_coverage.append({'family': family, 'feature': col, 'storage_layer': 'rental_rows' if col in built['train'] else 'monthly_panel', 'dtype': str(train_values.dtype), 'train_non_null_fraction': float(train_values.notna().mean()), 'test_non_null_fraction': float(test_values.notna().mean()), 'train_unique': int(train_values.nunique(dropna=True))})
    pd.DataFrame(feature_coverage).to_csv(audit_dir / 'feature_coverage.csv', index=False)
    summary = {'source_hashes': source_hashes, 'ignored_attributes': ['FURNISHED', 'FEE'], 'source_attribute_positions': 63, 'active_source_attribute_positions': 61, 'feature_families': {k: len(v) for k, v in families.items()}, 'joins': join_audit, 'spatial': spatial_audit, 'macro': macro_audit, 'metadata_columns': metadata, 'raw_training_exact_duplicate_rows': int(train.duplicated().sum()), 'note': 'Raw CSV files never modified; repeated rental rows retained.'}
    (audit_dir / 'build_summary.json').write_text(json.dumps(summary, indent=2, default=str), encoding='utf-8')
    print(json.dumps({'train_shape': built['train'].shape, 'test_shape': built['test'].shape, 'families': summary['feature_families'], 'source_attributes': len(coverage)}, default=str), flush=True)
    return (built['train'], built['test'], families)
if __name__ == '__main__':
    import argparse as _cli_argparse
    _cli_argparse.ArgumentParser(description=__doc__).parse_args()
    build_features()
