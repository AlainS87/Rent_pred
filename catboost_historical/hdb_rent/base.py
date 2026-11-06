"""Original property/geographic features and CatBoost factory; arithmetic unchanged."""
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from scipy.spatial import cKDTree
from .data import auxiliary_path
from .submission import make_submission

CAT = ['TOWN', 'BLOCK', 'STREET', 'FLAT_TYPE', 'FLAT_MODEL', 'FURNISHED',
       'address', 'SUBZONE', 'PLANNING_AREA', 'REGION', 'area_type']

def norm(s):
    return s.fillna('').astype(str).str.strip().str.lower().str.replace(r'\s+', ' ', regex=True)

def geo_nearest(frame, resource, prefix, k=1):
    dest = resource[['LATITUDE', 'LONGITUDE']].apply(pd.to_numeric, errors='coerce').dropna().to_numpy()
    xy = frame[['LATITUDE', 'LONGITUDE']].to_numpy(float)
    # Singapore bounding box: convert degrees to approximate km
    factor = np.array([111.19, 107.5])
    valid = np.isfinite(xy).all(axis=1)
    d = np.full((len(frame), k), np.nan)
    if len(dest):
        tree = cKDTree(dest * factor)
        if valid.any():
            v, _ = tree.query(xy[valid] * factor, k=k)
            d[valid] = np.asarray(v).reshape(-1, k)
    for i in range(k):
        frame[f'{prefix}_dist_{i+1}_km'] = d[:, i]

def features(data, data_dir):
    df = data.copy().reset_index(drop=True)
    df['TOWN'] = norm(df['TOWN'])
    df['BLOCK'] = norm(df['BLOCK'])
    df['STREET'] = norm(df['STREET'])
    df['FLAT_TYPE'] = norm(df['FLAT_TYPE']).str.replace(' ', '-', regex=False)
    df['FLAT_MODEL'] = norm(df['FLAT_MODEL'])
    df['FURNISHED'] = norm(df['FURNISHED'])
    df['address'] = df['BLOCK'] + ' ' + df['STREET']
    dates = pd.to_datetime(df['RENT_APPROVAL_DATE'] + '-01')
    df['year'] = dates.dt.year
    df['month'] = dates.dt.month
    df['time_index'] = (df['year'] - 2021) * 12 + df['month']
    df['quarter'] = dates.dt.quarter
    df['lease_age'] = df['year'] - pd.to_numeric(df['LEASE_COMMENCE_DATE'], errors='coerce')
    df['area_type'] = df['TOWN'] + ' ' + df['FLAT_TYPE']
    hdb = pd.read_csv(auxiliary_path(data_dir, 'sg-hdb-block.csv'))
    for key in ('TOWN', 'BLOCK', 'STREET'):
        hdb[key] = norm(hdb[key])
    hdb = hdb.drop_duplicates(['TOWN', 'BLOCK', 'STREET'])
    df = df.merge(hdb, on=['TOWN', 'BLOCK', 'STREET'], how='left', validate='m:1')
    for col in ('SUBZONE', 'PLANNING_AREA', 'REGION'):
        df[col] = norm(df[col])
    for col in ('LATITUDE', 'LONGITUDE', 'MAX_FLOOR', 'YEAR_COMPLETED', 'FLOOR_AREA_SQM', 'FEE'):
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['building_age'] = df['year'] - df['YEAR_COMPLETED']
    df['lease_to_build'] = pd.to_numeric(df['LEASE_COMMENCE_DATE']) - df['YEAR_COMPLETED']
    df['distance_to_cbd_km'] = np.sqrt(((df['LATITUDE'] - 1.2836) * 111.19) ** 2 +
                                       ((df['LONGITUDE'] - 103.8515) * 107.5) ** 2)
    for file, prefix, k in [('sg-mrt-stations.csv', 'mrt', 3),
                            ('sg-schools.csv', 'school', 2),
                            ('sg-shopping-malls.csv', 'mall', 2)]:
        other = pd.read_csv(auxiliary_path(data_dir, file))
        if 'STATUS' in other.columns:
            other = other[norm(other['STATUS']) == 'open']
        geo_nearest(df, other, prefix, k=k)
    df['area_x_age'] = df['FLOOR_AREA_SQM'] * df['lease_age']
    df['area_per_room'] = df['FLOOR_AREA_SQM'] / pd.to_numeric(df['FLAT_TYPE'].str.extract(r'(\d+)')[0], errors='coerce').replace(0, np.nan)
    df[CAT] = df[CAT].fillna('unknown').astype(str)
    exclude = {'RENT_APPROVAL_DATE', 'MONTHLY_RENT', 'POSTAL_CODE', 'LEASE_COMMENCE_DATE'}
    columns = [c for c in df if c not in exclude and (c in CAT or pd.api.types.is_numeric_dtype(df[c]))]
    return df[columns].copy(), df

def model(iterations, seed, threads, depth, lr):
    return CatBoostRegressor(iterations=iterations, depth=depth, learning_rate=lr,
                             loss_function='RMSE', eval_metric='RMSE',
                             l2_leaf_reg=5, random_seed=seed, thread_count=threads,
                             verbose=False, allow_writing_files=False)
