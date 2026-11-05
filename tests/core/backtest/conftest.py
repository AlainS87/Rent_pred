import copy
import uuid
import shutil

import numpy as np
import pandas as pd
import pytest

from src.backtest.data import NON_TARGET_FEATURES
from src.backtest.protocol import ROOT, read_protocol
from src.pipeline.preprocess import CATEGORICAL_NAMES


@pytest.fixture
def tiny_config():
    config = copy.deepcopy(read_protocol())
    config["catboost"].update(depth=3, iterations=16, early_stopping_rounds=4, thread_count=1)
    return config


@pytest.fixture
def checkpoint_path():
    # This Windows AppContainer permits temp writes but denies temp renames.
    # Checkpoint atomic-replace tests therefore use a unique workspace location.
    base = (ROOT / "artifacts/backtest/test_checkpoints").resolve()
    path = (base / uuid.uuid4().hex).resolve()
    if path.parent != base:
        raise ValueError("Test checkpoint path escaped its dedicated directory")
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path)


@pytest.fixture
def labeled_fixture():
    periods = pd.period_range("2021-01", "2024-04", freq="M")
    month = np.repeat(np.arange(len(periods)), 12)
    within = np.tile(np.arange(12), len(periods))
    count = len(month)
    town = np.where(within % 2 == 0, "town a", "town b")
    group = (within // 2) % 3
    frame = pd.DataFrame({
        "row_id": np.arange(count, dtype=np.int64),
        "RENT_APPROVAL_DATE": np.repeat(periods.astype(str), 12),
        "TOWN": town, "BLOCK": (within % 4).astype(str),
        "STREET": np.where(within % 2 == 0, "street a", "street b"),
        "FLAT_TYPE": np.where(group == 0, "3 room", "4-room"),
        "FLAT_MODEL": np.where(group == 0, "standard", "model a"),
        "FLOOR_AREA_SQM": 65.0 + group * 25.0,
        "FURNISHED": "yes", "LEASE_COMMENCE_DATE": 1980 + within, "FEE": 0.0,
    })
    for i, column in enumerate(NON_TARGET_FEATURES):
        if column in CATEGORICAL_NAMES or column.endswith(("_name", "_code", "_line")):
            frame[column] = np.where(within % 2 == 0, f"{column}_a", f"{column}_b")
        else:
            frame[column] = (i + 1.0) + group * 0.01
    frame["postal_code"] = np.where(within % 2 == 0, "530001", "650002")
    frame.loc[frame.row_id % 37 == 0, "latitude"] = np.nan
    frame["MONTHLY_RENT"] = (1700 + month * 25 + group * 300 + (within % 2) * 100 + (within % 5) * 13).astype(float)
    # Exercise positional alignment rather than relying on a default DataFrame index.
    frame.index = frame.row_id * 7 + 11
    return frame
