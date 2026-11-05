"""Past-only optimization features: cutoff, fallback, identity and horizon checks."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.backtest.protocol import TARGET, months
import src.research_round2.snapshots as snapshot_module
from src.research_round2.snapshots import Snapshot, SnapshotStore, keys, training_rows

# Every test below constructs at least one real hedonic Ridge snapshot.
pytestmark = pytest.mark.training


@pytest.fixture(scope="module")
def config():
    return json.loads((Path(__file__).resolve().parents[3] / "configs/research_round2/protocol.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def source():
    rows = []
    for number, month in enumerate(pd.period_range("2021-01", "2024-06", freq="M")):
        for group in range(3):
            rows.append({
                "row_id": len(rows), "RENT_APPROVAL_DATE": str(month),
                "TOWN": " Alpha " if group < 2 else "Beta", "BLOCK": "10" if group != 1 else "11",
                "STREET": "First Street" if group < 2 else "Second Street",
                "FLAT_TYPE": "3-room" if group != 1 else "4 ROOM", "FLAT_MODEL": "Model A",
                "FLOOR_AREA_SQM": 70.0 + 15.0 * group, "FURNISHED": "yes",
                "LEASE_COMMENCE_DATE": 1990.0 + group * 10, "FEE": 0.0,
                "latitude": 1.30 + group * 0.01, "longitude": 103.80 + group * 0.01,
                "max_floor": 12.0 + group, "dist_nearest_open_mrt_km": 0.5 + group * 0.1,
                "dist_to_cbd_km": 8.0 + group, TARGET: 1500.0 + 30.0 * number + 120.0 * group + (number % 3) * 5,
            })
    return pd.DataFrame(rows)


def query_at(source, cutoff, h=1):
    query = source.iloc[:3].drop(columns=TARGET).copy().reset_index(drop=True)
    query["row_id"] = np.arange(10000, 10000 + len(query))
    query["RENT_APPROVAL_DATE"] = str(pd.Period(cutoff, freq="M") + h)
    return query


def compare_transforms(left, right, query, hs):
    left_frames, left_bases = left.transform(query, hs)
    right_frames, right_bases = right.transform(query, hs)
    for name in ("raw", "adjusted"):
        assert_frame_equal(left_frames[name], right_frames[name])
        np.testing.assert_array_equal(left_bases[name], right_bases[name])


def test_future_labels_do_not_change_snapshot_index_or_features(source, config):
    cutoff = "2022-06"
    changed = source.copy()
    changed.loc[months(changed) > pd.Period(cutoff, freq="M"), TARGET] += 9e6
    original = SnapshotStore(source, config).get(cutoff)
    modified = SnapshotStore(changed, config).get(cutoff)
    assert original.allowed_history_sha256 == modified.allowed_history_sha256
    assert_frame_equal(original.index, modified.index)
    for key in original.tables:
        assert_frame_equal(original.tables[key], modified.tables[key])
    compare_transforms(original, modified, query_at(source, cutoff), np.ones(3, dtype=int))
    assert original.hedonic_audit["cpu_threads"] == 4
    assert original.hedonic_audit["reconstruction_max_abs_error"] < 1e-7


def test_own_and_all_same_month_labels_excluded_from_training_features(source, config):
    cutoff, target_month = "2022-11", "2022-10"
    changed = source.copy()
    changed.loc[changed.RENT_APPROVAL_DATE == target_month, TARGET] += 9e6
    original_frames, original_meta, _ = training_rows(source, cutoff, config, SnapshotStore(source, config))
    changed_frames, changed_meta, _ = training_rows(changed, cutoff, config, SnapshotStore(changed, config))
    use = original_meta.target_month == target_month
    assert set(original_meta.loc[use, "horizon"]) == {1, 3, 6, 12, 16}
    assert (original_meta.loc[use, "snapshot"] < target_month).all()
    for variant in ("raw", "adjusted"):
        assert_frame_equal(original_frames[variant].loc[use], changed_frames[variant].loc[use])
        np.testing.assert_array_equal(original_meta.loc[use, f"{variant}_base"], changed_meta.loc[use, f"{variant}_base"])
    np.testing.assert_allclose(changed_meta.loc[use, "target"] - original_meta.loc[use, "target"], 9e6)


def test_inner_validation_labels_do_not_enter_inner_training_or_query_features(source, config):
    cutoff = "2023-06"
    changed = source.copy()
    changed.loc[months(changed) > pd.Period(cutoff, freq="M"), TARGET] *= 100.0
    original_frames, original_meta, _ = training_rows(source, cutoff, config, SnapshotStore(source, config))
    changed_frames, changed_meta, _ = training_rows(changed, cutoff, config, SnapshotStore(changed, config))
    assert (original_meta.target_month <= cutoff).all()
    assert_frame_equal(original_meta, changed_meta)
    for name in original_frames:
        assert_frame_equal(original_frames[name], changed_frames[name])
    query = query_at(source, cutoff)
    query.loc[1, "RENT_APPROVAL_DATE"] = "2023-08"
    query.loc[2, "RENT_APPROVAL_DATE"] = "2023-09"
    compare_transforms(SnapshotStore(source, config).get(cutoff), SnapshotStore(changed, config).get(cutoff), query, np.array([1, 2, 3]))


def test_hierarchical_shrink_and_missing_building_fallback(source, config):
    cutoff = "2022-06"
    snapshot = SnapshotStore(source, config).get(cutoff)
    query = query_at(source, cutoff).iloc[[0]].copy()
    frame, bases = snapshot.transform(query, np.array([1]))
    parent = snapshot.fallback["raw"]
    for level, (key, _) in keys(query).items():
        row = snapshot.tables[("12", level)].loc[key.iloc[0]]
        parent = row.raw / row.n if level == "global" else (row.raw + 20 * parent) / (row.n + 20)
        assert frame["raw"][f"price_12_{level}"].iloc[0] == pytest.approx(parent, abs=2e-4)
    assert bases["raw"][0] == pytest.approx(parent)

    for missing in (None, "999"):
        query["BLOCK"] = missing
        for variant, result in snapshot.transform(query, np.array([1]))[0].items():
            assert result["count_12_building"].iloc[0] == 0
            assert result["known_12_building"].iloc[0] == 0
            assert result["recency_12_building"].iloc[0] == -1
            assert result["price_12_building"].iloc[0] == result["price_12_street"].iloc[0]


def test_same_block_number_in_other_town_is_distinct_and_unknown_parent_falls_back(source, config):
    snapshot = SnapshotStore(source, config).get("2022-06")
    query = query_at(source, "2022-06")
    identifiers, valid = keys(query)["building"]
    assert valid.all() and identifiers.iloc[0] != identifiers.iloc[2]
    query = query.iloc[[0]].copy()
    query["TOWN"] = "Never Observed Town"
    frames, _ = snapshot.transform(query, np.array([1]))
    for frame in frames.values():
        assert frame["count_12_town"].iloc[0] == 0
        assert frame["price_12_town"].iloc[0] == frame["price_12_flat"].iloc[0]
        assert frame["price_12_building"].iloc[0] == frame["price_12_flat"].iloc[0]
    query["FLAT_TYPE"] = "never observed flat"
    frames, _ = snapshot.transform(query, np.array([1]))
    for frame in frames.values():
        assert frame["count_12_flat"].iloc[0] == 0
        assert frame["price_12_flat"].iloc[0] == frame["price_12_global"].iloc[0]


def test_h1_h16_anchor_frozen_and_only_forecast_market_base_changes(source, config):
    snapshot = SnapshotStore(source, config).get("2022-06")
    query = pd.concat([query_at(source, "2022-06", 1).iloc[[0]], query_at(source, "2022-06", 16).iloc[[0]]], ignore_index=True)
    query["row_id"] = [10000, 10001]
    frames, bases = snapshot.transform(query, np.array([1, 16]))
    assert list(frames["raw"]) == list(frames["adjusted"])
    for variant, frame in frames.items():
        derived = frame.filter(regex=r"^(price|count|recency|known)_")
        np.testing.assert_array_equal(derived.iloc[0].to_numpy(), derived.iloc[1].to_numpy())
        assert np.isfinite(derived.to_numpy()).all()
    assert bases["raw"][0] == bases["raw"][1]
    assert bases["adjusted"][1] - bases["adjusted"][0] == pytest.approx(np.diff(snapshot.trend.predict_horizon(np.array([1, 16])))[0])


@pytest.mark.parametrize("damage", ["truth", "date", "zero", "float"])
def test_transform_rejects_truth_or_inconsistent_horizon(source, config, damage):
    snapshot = SnapshotStore(source, config).get("2022-06")
    query = query_at(source, "2022-06")
    offsets = np.ones(3, dtype=int)
    if damage == "truth":
        query[TARGET] = 1000.0
    elif damage == "date":
        query.loc[0, "RENT_APPROVAL_DATE"] = "2022-08"
    elif damage == "zero":
        offsets[0] = 0
    else:
        offsets = offsets.astype(float)
    with pytest.raises(ValueError):
        snapshot.transform(query, offsets)


def test_snapshot_training_identity_all_far_horizons_and_equal_h_mass(source, config):
    cutoff = "2023-06"
    _, meta, audit = training_rows(source, cutoff, config, SnapshotStore(source, config))
    assert not meta.duplicated(["snapshot", "horizon", "row_id"]).any()
    assert (meta.snapshot >= config["snapshot_start"]).all()
    assert (meta.snapshot < meta.target_month).all()
    assert (meta.target_month <= cutoff).all()
    assert (pd.PeriodIndex(meta.target_month, freq="M").asi8 - pd.PeriodIndex(meta.snapshot, freq="M").asi8 == meta.horizon).all()
    source_dates = source.set_index("row_id").RENT_APPROVAL_DATE
    assert (meta.row_id.map(source_dates) == meta.target_month).all()
    mass = meta.groupby("horizon").weight.sum()
    assert set(mass.index) == {1, 3, 6, 12, 16}
    np.testing.assert_allclose(mass.to_numpy(), np.repeat(len(meta) / 5, 5))
    assert meta.weight.mean() == pytest.approx(1.0)
    assert audit["snapshot_count"] == meta.snapshot.nunique()
    assert audit["unique_target_rows"] == meta.row_id.nunique()
    assert audit["instances"] == len(meta)


def test_same_cutoff_cache_reuses_own_history_only_not_later_origin(source, config, tmp_path, monkeypatch):
    cutoff = "2022-06"
    first = SnapshotStore(source, config, tmp_path, "test").get(cutoff)
    changed = source.copy()
    changed.loc[months(changed) > pd.Period(cutoff, freq="M"), TARGET] += 9e6
    def forbidden(*_args, **_kwargs):
        raise AssertionError("same cutoff, same allowed history must not refit")
    monkeypatch.setattr(Snapshot, "fit", forbidden)
    second = SnapshotStore(changed, config, tmp_path, "test").get(cutoff)
    compare_transforms(first, second, query_at(source, cutoff), np.ones(3, dtype=int))
    assert len(list(tmp_path.glob("*/complete.json"))) == 1


def test_snapshot_cache_detects_changed_weights(source, config, tmp_path):
    SnapshotStore(source, config, tmp_path, "test").get("2022-06")
    payload = next(tmp_path.glob("*/snapshot.joblib"))
    with payload.open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        SnapshotStore(source, config, tmp_path, "test").get("2022-06")


def test_owned_store_isolates_caller_mutation_and_hashes_each_cutoff_once(source, config, tmp_path, monkeypatch):
    cutoff = "2022-06"
    expected = SnapshotStore(source, config).get(cutoff)
    caller_history = source.copy(deep=True)
    caller_config = copy.deepcopy(config)
    store = SnapshotStore(caller_history, caller_config, tmp_path, "owned")
    # Mutate caller objects before the first request: an owned store must retain
    # its construction-time data and nested configuration, not merely cache an
    # already-fitted snapshot.
    caller_history.loc[0, TARGET] += 9e6
    caller_history.loc[0, "FLOOR_AREA_SQM"] += 30.0
    caller_config["strength"] = 2000.0
    caller_config["hedonic"]["ridge_alpha"] = 2000.0
    original_hash = snapshot_module.history_hash
    calls = []
    def counted_hash(frame):
        calls.append(len(frame))
        return original_hash(frame)
    monkeypatch.setattr(snapshot_module, "history_hash", counted_hash)

    first = store.get(cutoff)
    first_call_count = len(calls)
    assert first_call_count > 0
    assert first.allowed_history_sha256 == expected.allowed_history_sha256
    assert first.config == config
    assert_frame_equal(first.index, expected.index)
    compare_transforms(first, expected, query_at(source, cutoff), np.ones(3, dtype=int))
    assert store.get(cutoff) is first
    assert len(calls) == first_call_count

    # A newly constructed store must re-check the newly supplied permitted
    # history and must not alias the old disk cache just because s is identical.
    changed = SnapshotStore(caller_history, config, tmp_path, "owned").get(cutoff)
    assert changed.allowed_history_sha256 != first.allowed_history_sha256
    assert len(calls) > first_call_count
    receipts = [json.loads(path.read_text(encoding="utf-8")) for path in tmp_path.glob("*/complete.json")]
    assert len(receipts) == 2
    assert {item["contract"]["labels"] for item in receipts} == {first.allowed_history_sha256, changed.allowed_history_sha256}
