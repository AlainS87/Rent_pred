"""Construct auditable block-level geographic features from the supplied CSVs.

Distances are straight-line haversine distances in kilometres. Facilities are
deduplicated by their normalized name and coordinates: station line records
become one physical entity, but different schools at a shared campus remain
different entities. No historical opening dates or school-quality labels are
invented. The public entry point does not read any previous project outputs.
"""
from __future__ import annotations
from runtime import DATA_DIR, CODE_DIR, ANALYSIS
import json
import re
import time
from pathlib import Path
from urllib.parse import urlsplit
import numpy as np
import pandas as pd
EARTH_RADIUS_KM = 6371.0088
RADII_KM = (0.5, 1.0, 2.0)
STATIC_AVAILABILITY = 'provided static snapshot; historical validity dates unavailable'
MRT_AVAILABILITY = 'provided static snapshot; STATUS is snapshot status; opening dates unavailable'

def normalize(value: object) -> str | None:
    """Normalize join keys and categories, preserving unknowns as None."""
    if value is None or pd.isna(value):
        return None
    text = re.sub('\\s+', ' ', str(value).strip().lower())
    return None if text in {'', 'na', 'nan', 'null', 'none', 'nil'} else text

def _slug(value: str) -> str:
    return re.sub('[^a-z0-9]+', '_', value.lower()).strip('_')

def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding='utf-8-sig')

def _haversine(block_coords: np.ndarray, facility_coords: np.ndarray) -> np.ndarray:
    """Return pairwise distances; callers chunk large block-block matrices."""
    left = np.radians(block_coords)
    right = np.radians(facility_coords)
    lat_delta = right[None, :, 0] - left[:, None, 0]
    lon_delta = right[None, :, 1] - left[:, None, 1]
    hav = np.sin(lat_delta / 2.0) ** 2
    hav += np.cos(left[:, None, 0]) * np.cos(right[None, :, 0]) * np.sin(lon_delta / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(hav, 0.0, 1.0)))

def _valid_coords(frame: pd.DataFrame) -> np.ndarray:
    return (frame['LATITUDE'].between(-90.0, 90.0) & frame['LONGITUDE'].between(-180.0, 180.0)).to_numpy()

def _prepare_facilities(raw: pd.DataFrame, *, multi_line_station: bool=False) -> tuple[pd.DataFrame, dict]:
    """Combine repeat records of one entity without merging shared campuses."""
    frame = raw.copy()
    for column in frame:
        if column in {'LATITUDE', 'LONGITUDE'}:
            frame[column] = pd.to_numeric(frame[column], errors='coerce').astype(float)
        else:
            frame[column] = frame[column].map(normalize)
    valid = _valid_coords(frame) & frame['NAME'].notna().to_numpy()
    invalid_count = int((~valid).sum())
    frame = frame.loc[valid].copy()
    keys = ['NAME', 'LATITUDE', 'LONGITUDE']
    records: list[dict] = []
    conflicts: list[dict] = []
    for entity_key, group in frame.groupby(keys, sort=True, dropna=False):
        row = dict(zip(keys, entity_key))
        for column in frame.columns.difference(keys):
            values = sorted(set(group[column].dropna()))
            row[column] = '|'.join(values) if values else None
            if len(values) > 1 and (not (multi_line_station and column in {'CODE', 'STATUS'})):
                conflicts.append({'name': row['NAME'], 'attribute': column, 'values': values})
        if multi_line_station:
            codes = (row.get('CODE') or '').split('|')
            prefixes = sorted({m.group(0) for code in codes if (m := re.match('[a-z]+', code))})
            row['LINE_PREFIXES'] = '|'.join(prefixes)
            row['LINE_COUNT'] = float(len(prefixes))
            row['CODE_COUNT'] = float(sum((bool(code) for code in codes)))
            row['INTERCHANGE'] = float(len(prefixes) > 1)
        records.append(row)
    entities = pd.DataFrame.from_records(records).reset_index(drop=True)
    audit = {'raw_rows': int(len(raw)), 'valid_rows': int(valid.sum()), 'invalid_coordinate_or_name_rows': invalid_count, 'entities': int(len(entities)), 'duplicate_entity_rows_combined': int(valid.sum() - len(entities)), 'shared_coordinate_groups': int(entities.groupby(['LATITUDE', 'LONGITUDE']).size().gt(1).sum()), 'attribute_conflicts': conflicts, 'entity_definition': 'normalized NAME + numeric LATITUDE + numeric LONGITUDE'}
    return (entities, audit)

def _bus_text_tokens(value: str | None) -> tuple[float, float]:
    """Count unique route-like tokens, explicitly a heuristic text descriptor."""
    value = normalize(value)
    if value is None:
        return (np.nan, 1.0)
    text = re.sub('\\b(?:blk|block|ave|avenue|street)\\s+\\d+\\b', ' ', value)
    text = re.sub('\\bat\\s+\\d+\\s+[a-z ]+(?:campus|hill|crescent)\\b', ' ', text)
    tokens = set(re.findall('(?<![a-z0-9])(?:\\d{1,3}[a-z]{0,2}(?:/[a-z])?|nr\\d{1,2})(?![a-z0-9])', text))
    return (float(len(tokens)), 0.0)

def _school_mrt_mentions(value: str | None, mrt: pd.DataFrame) -> tuple[float, float]:
    """Count distinct supplied station entities recognized in school text."""
    value = normalize(value)
    if value is None:
        return (np.nan, 1.0)
    text = re.sub('\\b([a-z]{1,3})\\s+(\\d{1,2})\\b', '\\1\\2', value)
    codes = set(re.findall('\\b[a-z]{1,3}\\d{1,2}\\b', text))
    matched: set[int] = set()
    for idx, row in mrt.iterrows():
        base_name = re.sub('\\s+(?:mrt/lrt|mrt|lrt)\\s+station$', '', row['NAME'])
        name_found = bool(re.search('(?<![a-z0-9])' + re.escape(base_name) + '(?![a-z0-9])', text))
        code_found = bool(codes.intersection((row['CODE'] or '').split('|')))
        if name_found or code_found:
            matched.add(int(idx))
    return (float(len(matched)), 0.0)

def _url_metadata(value: str | None) -> tuple[str | None, float]:
    value = normalize(value)
    if value is None:
        return (None, np.nan)
    parsed = urlsplit(value if '://' in value else 'https://' + value)
    return (parsed.hostname or None, float(value.startswith('https://')))

def build_spatial(data_dir: Path, output_dir: Path) -> tuple[pd.DataFrame, dict[str, list[str]], list[dict], dict]:
    """Build geographic candidates, source lineage and independent audit files.

    Features are computed once per supplied HDB block. The caller joins rows
    using ``TOWN|BLOCK|STREET`` after applying the same normalization, and
    constructs age relative to each rental approval month separately.
    """
    start = time.perf_counter()
    data_dir, output_dir = (Path(data_dir), Path(output_dir))
    auxiliary = data_dir / 'auxiliary'
    source_files = {'hdb': 'sg-hdb-block.csv', 'mrt': 'sg-mrt-stations.csv', 'schools': 'sg-schools.csv', 'malls': 'sg-shopping-malls.csv'}
    raw = {source: _read_csv(auxiliary / filename) for source, filename in source_files.items()}
    hdb = raw['hdb'].copy()
    for column in hdb:
        if column in {'LATITUDE', 'LONGITUDE', 'MAX_FLOOR', 'YEAR_COMPLETED'}:
            hdb[column] = pd.to_numeric(hdb[column], errors='coerce').astype(float)
            if column in {'MAX_FLOOR', 'YEAR_COMPLETED'}:
                hdb.loc[hdb[column] < 0, column] = np.nan
        else:
            hdb[column] = hdb[column].map(normalize)
    if hdb[['TOWN', 'BLOCK', 'STREET']].isna().any().any():
        raise ValueError('HDB join-key fields contain unknown values')
    address_keys = hdb['TOWN'] + '|' + hdb['BLOCK'] + '|' + hdb['STREET']
    if address_keys.duplicated().any():
        raise ValueError('Normalized HDB keys are not unique; a many-to-many join would be unsafe')
    hdb.index = pd.Index(address_keys, name='address_key')
    block_coords = hdb[['LATITUDE', 'LONGITUDE']].to_numpy(dtype=float)
    if not _valid_coords(hdb).all():
        raise ValueError('HDB contains invalid coordinates; geographic features require explicit resolution')
    features: dict[str, np.ndarray] = {}
    families: dict[str, list[str]] = {'block_structure': [], 'block_location': [], 'mrt': [], 'schools': [], 'malls': []}
    lineage: list[dict] = []
    audit: dict = {'distance_unit': 'km', 'distance_definition': 'straight-line haversine; not network travel distance or travel time', 'static_snapshot_warning': STATIC_AVAILABILITY, 'hdb': {'raw_rows': len(hdb), 'unique_address_keys': int(address_keys.nunique()), 'postal_unknown_blocks': int(hdb['POSTAL_CODE'].isna().sum()), 'max_floor_unknown_blocks': int(hdb['MAX_FLOOR'].isna().sum()), 'year_completed_unknown_blocks': int(hdb['YEAR_COMPLETED'].isna().sum()), 'sentinel_policy': 'NIL and negative building metadata are unknown; no missingness indicators added', 'floor_warning': "MAX_FLOOR describes the building, not the rented unit's floor"}}

    def add(name: str, values: object, family: str, source: str, attributes: str, construction: str, availability: str=STATIC_AVAILABILITY) -> None:
        if name in features:
            raise ValueError(f'Duplicate constructed feature name: {name}')
        array = np.asarray(values)
        if len(array) != len(hdb):
            raise ValueError(f'Feature {name} is not aligned with the HDB block rows')
        if array.dtype.kind in 'biufc':
            array = array.astype(float)
        else:
            array = np.asarray([None if pd.isna(value) else str(value) for value in array], dtype=object)
        features[name] = array
        families[family].append(name)
        lineage.append({'feature': name, 'family': family, 'source_file': source, 'source_attributes': attributes, 'construction': construction, 'availability': availability})

    def geographic_summary(distances: np.ndarray, mask: np.ndarray, prefix: str, family: str, source: str, attrs: str, description: str, availability: str=STATIC_AVAILABILITY) -> np.ndarray | None:
        selected = distances[:, mask]
        if selected.shape[1] == 0:
            add(prefix + '_nearest_km', np.full(len(hdb), np.nan), family, source, attrs, 'no matching supplied entities; nearest distance unknown', availability)
            for radius in RADII_KM:
                radius_name = str(radius).replace('.', 'p')
                add(prefix + '_count_' + radius_name + 'km', np.zeros(len(hdb)), family, source, attrs, f'number of {description} within {radius} km; no entities in snapshot', availability)
            return None
        k = min(3, selected.shape[1])
        nearest = np.sort(np.partition(selected, kth=k - 1, axis=1)[:, :k], axis=1)
        add(prefix + '_nearest_km', nearest[:, 0], family, source, attrs, f'nearest {description}; haversine km', availability)
        add(prefix + '_top3_mean_km', nearest.mean(axis=1), family, source, attrs, f'mean distance to the closest min(3, available entities) {description}; haversine km', availability)
        for radius in RADII_KM:
            radius_name = str(radius).replace('.', 'p')
            add(prefix + '_count_' + radius_name + 'km', (selected <= radius).sum(axis=1), family, source, attrs, f'number of distinct {description} within {radius} km', availability)
        return np.flatnonzero(mask)[selected.argmin(axis=1)]
    hdb_source = source_files['hdb']
    for column in ['MAX_FLOOR', 'YEAR_COMPLETED']:
        add(column.lower(), hdb[column].to_numpy(), 'block_structure', hdb_source, column, 'raw building metadata; negative sentinel converted to unknown')
    for column in ['POSTAL_CODE', 'SUBZONE', 'PLANNING_AREA', 'REGION', 'LATITUDE', 'LONGITUDE']:
        add('hdb_' + column.lower(), hdb[column].to_numpy(), 'block_location', hdb_source, column, 'normalized source category or raw numeric coordinate')
    postal_sectors = [value[:2] if isinstance(value, str) and re.fullmatch('\\d{6}', value) else None for value in hdb['POSTAL_CODE']]
    add('hdb_postal_sector', postal_sectors, 'block_location', hdb_source, 'POSTAL_CODE', 'first two digits of a valid six-digit postal code; identifier geography proxy')
    centroids = hdb.groupby('TOWN')[['LATITUDE', 'LONGITUDE']].mean()
    town_coords = centroids.loc[hdb['TOWN']].to_numpy(dtype=float)
    lat1, lon1 = np.radians(block_coords).T
    lat2, lon2 = np.radians(town_coords).T
    hv = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    add('hdb_town_centroid_distance_km', 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(hv, 0, 1))), 'block_location', hdb_source, 'TOWN|LATITUDE|LONGITUDE', 'distance to unweighted centroid of supplied buildings within the same TOWN')
    neighbor_counts = {radius: np.zeros(len(hdb), dtype=float) for radius in (0.5, 1.0)}
    for first in range(0, len(hdb), 384):
        last = min(first + 384, len(hdb))
        matrix = _haversine(block_coords[first:last], block_coords)
        matrix[np.arange(last - first), np.arange(first, last)] = np.inf
        for radius in neighbor_counts:
            neighbor_counts[radius][first:last] = (matrix <= radius).sum(axis=1)
    for radius, values in neighbor_counts.items():
        add('hdb_other_blocks_count_' + str(radius).replace('.', 'p') + 'km', values, 'block_location', hdb_source, 'TOWN|BLOCK|STREET|LATITUDE|LONGITUDE', f'other distinct supplied buildings within {radius} km; excludes the current building')
    mrt, mrt_audit = _prepare_facilities(raw['mrt'], multi_line_station=True)
    audit['mrt'] = mrt_audit
    audit['mrt']['raw_status_counts'] = raw['mrt']['STATUS'].value_counts().to_dict()
    audit['mrt']['status_caution'] = MRT_AVAILABILITY
    audit['mrt']['line_prefix_caution'] = 'observed CODE prefixes, not externally verified line taxonomy'
    raffles = mrt.loc[mrt['NAME'].eq('raffles place mrt station')]
    if raffles.empty:
        raise ValueError('Raffles Place station not found; CBD reference must not be invented')
    cbd_coord = raffles[['LATITUDE', 'LONGITUDE']].iloc[[0]].to_numpy(dtype=float)
    audit['cbd_reference'] = {'name': 'raffles place mrt station', 'latitude': float(cbd_coord[0, 0]), 'longitude': float(cbd_coord[0, 1]), 'source_file': source_files['mrt']}
    add('cbd_raffles_place_distance_km', _haversine(block_coords, cbd_coord)[:, 0], 'block_location', hdb_source + '|' + source_files['mrt'], 'LATITUDE|LONGITUDE|NAME', 'distance to supplied Raffles Place station coordinates as an explicit CBD reference')
    mrt_distances = _haversine(block_coords, mrt[['LATITUDE', 'LONGITUDE']].to_numpy(dtype=float))
    nearest_mrt = geographic_summary(mrt_distances, np.ones(len(mrt), dtype=bool), 'mrt', 'mrt', source_files['mrt'], 'NAME|CODE|LATITUDE|LONGITUDE', 'station entities', MRT_AVAILABILITY)
    assert nearest_mrt is not None
    for column in ['CODE', 'NAME', 'STATUS', 'LATITUDE', 'LONGITUDE', 'SUBZONE', 'PLANNING_AREA', 'REGION', 'LINE_PREFIXES', 'LINE_COUNT', 'CODE_COUNT', 'INTERCHANGE']:
        attrs = 'CODE' if column in {'LINE_PREFIXES', 'LINE_COUNT', 'CODE_COUNT', 'INTERCHANGE'} else column
        add('mrt_nearest_' + column.lower(), mrt.iloc[nearest_mrt][column].to_numpy(), 'mrt', source_files['mrt'], attrs + '|NAME|LATITUDE|LONGITUDE', 'attribute of closest station entity; multi-line station records combined', MRT_AVAILABILITY)
    line_prefixes = sorted({value for text in mrt['LINE_PREFIXES'] for value in text.split('|') if value})
    for prefix in line_prefixes:
        mask = mrt['LINE_PREFIXES'].map(lambda value: prefix in value.split('|')).to_numpy(dtype=bool)
        geographic_summary(mrt_distances, mask, 'mrt_line_' + prefix, 'mrt', source_files['mrt'], 'CODE|NAME|LATITUDE|LONGITUDE', f'station entities containing observed CODE prefix {prefix}', MRT_AVAILABILITY)
    for status in sorted({value for text in mrt['STATUS'].dropna() for value in text.split('|')}):
        mask = mrt['STATUS'].map(lambda value: status in (value or '').split('|')).to_numpy(dtype=bool)
        nearest_status = geographic_summary(mrt_distances, mask, 'mrt_status_' + _slug(status), 'mrt', source_files['mrt'], 'STATUS|NAME|LATITUDE|LONGITUDE', f'station entities with snapshot STATUS={status}', MRT_AVAILABILITY)
        if nearest_status is not None:
            for attr in ['NAME', 'CODE']:
                add('mrt_status_' + _slug(status) + '_nearest_' + attr.lower(), mrt.iloc[nearest_status][attr].to_numpy(), 'mrt', source_files['mrt'], 'STATUS|' + attr + '|LATITUDE|LONGITUDE', 'identity of nearest station with this snapshot status', MRT_AVAILABILITY)
    schools, school_audit = _prepare_facilities(raw['schools'])
    audit['schools'] = school_audit
    audit['schools']['region_caution'] = 'school north/south/east/west REGION is not the HDB five-region classification'
    audit['schools']['url_caution'] = 'URL/domain/HTTPS describe source identity and metadata, not quality or ranking'
    audit['schools']['bus_text_caution'] = 'route-like token counts are approximate; prose can contain address numbers'
    audit['schools']['mrt_text_caution'] = 'mentions recognized only against supplied MRT entities; not walking distance or service history'
    bus_tokens = [_bus_text_tokens(value) for value in schools['BUS_LINES']]
    mrt_mentions = [_school_mrt_mentions(value, mrt) for value in schools['MRT_STATIONS']]
    url_values = [_url_metadata(value) for value in schools['URL']]
    schools['BUS_TEXT_TOKEN_COUNT'] = [value[0] for value in bus_tokens]
    schools['BUS_TEXT_LENGTH'] = schools['BUS_LINES'].map(lambda value: float(len(value)) if isinstance(value, str) else np.nan)
    schools['MRT_TEXT_STATION_MENTIONS'] = [value[0] for value in mrt_mentions]
    schools['MRT_TEXT_LENGTH'] = schools['MRT_STATIONS'].map(lambda value: float(len(value)) if isinstance(value, str) else np.nan)
    schools['URL_DOMAIN'] = [value[0] for value in url_values]
    schools['URL_HTTPS_EXPLICIT'] = [value[1] for value in url_values]
    audit['schools']['mrt_text_present_but_no_supplied_station_match'] = int((schools['MRT_STATIONS'].notna() & schools['MRT_TEXT_STATION_MENTIONS'].eq(0)).sum())
    school_distances = _haversine(block_coords, schools[['LATITUDE', 'LONGITUDE']].to_numpy(dtype=float))
    nearest_school = geographic_summary(school_distances, np.ones(len(schools), dtype=bool), 'school', 'schools', source_files['schools'], 'NAME|LATITUDE|LONGITUDE', 'distinct school entities (shared campuses retained)')
    assert nearest_school is not None
    nearest_columns = {'NAME': 'NAME', 'STREET': 'STREET', 'POSTAL_CODE': 'POSTAL_CODE', 'PLANNING_AREA': 'PLANNING_AREA', 'REGION': 'REGION', 'TYPE_CODE': 'TYPE_CODE', 'NATRUE_CODE': 'NATRUE_CODE', 'SESSION_CODE': 'SESSION_CODE', 'MAINLEVEL_CODE': 'MAINLEVEL_CODE', 'LATITUDE': 'LATITUDE', 'LONGITUDE': 'LONGITUDE', 'URL_DOMAIN': 'URL', 'URL_HTTPS_EXPLICIT': 'URL', 'BUS_TEXT_TOKEN_COUNT': 'BUS_LINES', 'BUS_TEXT_LENGTH': 'BUS_LINES', 'MRT_TEXT_STATION_MENTIONS': 'MRT_STATIONS|NAME|CODE', 'MRT_TEXT_LENGTH': 'MRT_STATIONS'}
    for column, attrs in nearest_columns.items():
        source = source_files['schools']
        if column == 'MRT_TEXT_STATION_MENTIONS':
            source += '|' + source_files['mrt']
        add('school_nearest_' + column.lower(), schools.iloc[nearest_school][column].to_numpy(), 'schools', source, attrs + '|NAME|LATITUDE|LONGITUDE', 'attribute or text metadata descriptor of nearest school; no quality label inferred')
    for column, label in [('MAINLEVEL_CODE', 'level'), ('TYPE_CODE', 'type'), ('NATRUE_CODE', 'nature'), ('SESSION_CODE', 'session')]:
        for category in sorted(schools[column].dropna().unique()):
            mask = schools[column].eq(category).to_numpy(dtype=bool)
            nearest_category = geographic_summary(school_distances, mask, 'school_' + label + '_' + _slug(category), 'schools', source_files['schools'], column + '|NAME|LATITUDE|LONGITUDE', f'school entities with {column}={category}')
            if column == 'MAINLEVEL_CODE' and nearest_category is not None:
                add('school_level_' + _slug(category) + '_nearest_name', schools.iloc[nearest_category]['NAME'].to_numpy(), 'schools', source_files['schools'], 'MAINLEVEL_CODE|NAME|LATITUDE|LONGITUDE', 'nearest identity within observed main level; local identity rather than school quality')
    for stage, pattern in [('primary', 'primary|p1'), ('secondary', 'secondary|s1')]:
        mask = schools['MAINLEVEL_CODE'].fillna('').str.contains(pattern, regex=True).to_numpy(dtype=bool)
        if mask.any():
            nearest_stage = np.flatnonzero(mask)[school_distances[:, mask].argmin(axis=1)]
            add('school_' + stage + '_nearest_name', schools.iloc[nearest_stage]['NAME'].to_numpy(), 'schools', source_files['schools'], 'MAINLEVEL_CODE|NAME|LATITUDE|LONGITUDE', 'nearest school name in stage, including supplied mixed-level categories')
    malls, mall_audit = _prepare_facilities(raw['malls'])
    audit['malls'] = mall_audit
    mall_distances = _haversine(block_coords, malls[['LATITUDE', 'LONGITUDE']].to_numpy(dtype=float))
    nearest_mall = geographic_summary(mall_distances, np.ones(len(malls), dtype=bool), 'mall', 'malls', source_files['malls'], 'NAME|LATITUDE|LONGITUDE', 'shopping-mall entities')
    assert nearest_mall is not None
    for column in ['NAME', 'LATITUDE', 'LONGITUDE']:
        add('mall_nearest_' + column.lower(), malls.iloc[nearest_mall][column].to_numpy(), 'malls', source_files['malls'], 'NAME|LATITUDE|LONGITUDE', 'attribute of closest supplied shopping-mall entity')
    columns = {name: pd.Series(values, index=hdb.index, dtype=object) if values.dtype.kind == 'O' else values for name, values in features.items()}
    block_features = pd.DataFrame(columns, index=hdb.index)
    block_features.index.name = 'address_key'
    coverage = []
    for source, frame in raw.items():
        filename = source_files[source]
        for attribute in frame.columns:
            used = [item['feature'] for item in lineage if filename in item['source_file'].split('|') and attribute in item['source_attributes'].split('|')]
            key_only = source == 'hdb' and attribute in {'TOWN', 'BLOCK', 'STREET'}
            coverage.append({'source_file': filename, 'attribute': attribute, 'role': 'join key and derived geographic features' if key_only else 'constructed feature source', 'feature_count': len(used), 'features': '|'.join(used), 'note': 'not copied as redundant predictor; preserved in address_key' if key_only else ''})
            if not used and (not key_only):
                raise AssertionError(f'Unreviewed source attribute: {filename}:{attribute}')
    audit['source_attribute_coverage'] = coverage
    audit['candidate_count'] = len(block_features.columns)
    audit['families'] = {name: len(columns) for name, columns in families.items()}
    audit['elapsed_seconds'] = round(time.perf_counter() - start, 3)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(lineage).to_csv(output_dir / 'spatial_feature_lineage.csv', index=False)
    pd.DataFrame(coverage).to_csv(output_dir / 'spatial_source_attribute_coverage.csv', index=False)
    for label, entities in [('mrt', mrt), ('schools', schools), ('malls', malls)]:
        entities.to_csv(output_dir / f'spatial_{label}_entities.csv', index=False)
    block_features.to_csv(output_dir / 'spatial_block_features.csv', index=True)
    (output_dir / 'spatial_audit.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding='utf-8')
    return (block_features, families, lineage, audit)
