"""Small, deterministic inputs covering causal history, geography and forecast months."""
from pathlib import Path
import numpy as np
import pandas as pd


def make_data(folder):
    folder = Path(folder)
    aux = folder / 'auxiliary'
    aux.mkdir(parents=True, exist_ok=True)
    rows = []
    for month_index, month in enumerate(pd.period_range('2022-01', '2025-03', freq='M')):
        for block in range(1, 5):
            for repeat in range(3):
                rows.append(dict(RENT_APPROVAL_DATE=str(month), TOWN='Test Town', BLOCK=str(block),
                                 STREET='Test Street', FLAT_TYPE='4-room', FLAT_MODEL='Model A',
                                 FLOOR_AREA_SQM=85 + block + repeat, FURNISHED='yes',
                                 LEASE_COMMENCE_DATE=1990 + block, FEE=0,
                                 MONTHLY_RENT=1800 + month_index * 23 + block * 35 + repeat * 17))
    train = pd.DataFrame(rows)
    test = train.drop(columns='MONTHLY_RENT').iloc[:20].copy()
    test['RENT_APPROVAL_DATE'] = ['2024-03', '2024-04', '2025-07', '2024-06'] * 5
    train.to_csv(folder / 'train.csv', index=False)
    test.to_csv(folder / 'test.csv', index=False)
    blocks = pd.DataFrame([dict(TOWN='Test Town', BLOCK=str(b), STREET='Test Street',
                                POSTAL_CODE=f'10000{b}', LATITUDE=1.3 + b * .0005,
                                LONGITUDE=103.8 + b * .0005, MAX_FLOOR=12, YEAR_COMPLETED=1985,
                                SUBZONE='Subzone', PLANNING_AREA='Test Area', REGION='East')
                           for b in range(1, 5)])
    blocks.to_csv(aux / 'sg-hdb-block.csv', index=False)
    places = pd.DataFrame({'LATITUDE': [1.3, 1.31, 1.32], 'LONGITUDE': [103.8, 103.81, 103.82]})
    for name in ['sg-mrt-stations.csv', 'sg-schools.csv', 'sg-shopping-malls.csv']:
        places.to_csv(aux / name, index=False)
    return train, test
