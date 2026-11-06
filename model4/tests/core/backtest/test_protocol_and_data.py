import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.backtest.data import BASE_COLUMNS, _read_selected_labels, history_at, load_cache, query_at
from src.backtest.models import fit_simple
from src.backtest.protocol import ROOT, TARGET, horizon, horizon_bin, months, read_protocol
from src.backtest.scoring import attach_truth, metrics


def test_horizon_edges_and_development_isolation():
    config = read_protocol()
    origin = "2023-09"
    offsets = np.array([1, 3, 4, 6, 7, 12, 13, 15, 16])
    dates = [str(pd.Period(origin, freq="M") + int(offset)) for offset in offsets]
    frame = pd.DataFrame({"RENT_APPROVAL_DATE": dates})
    np.testing.assert_array_equal(horizon(frame, origin), offsets)
    assert horizon_bin(offsets, config["horizon_bins"]).tolist() == ["1-3", "1-3", "4-6", "4-6", "7-12", "7-12", "13-15", "13-15", "16"]
    last_dev = max(pd.Period(o, freq="M") + config["max_horizon"] for o in config["development_origins"])
    first_eval = min(pd.Period(o, freq="M") + 1 for o in config["evaluation_origins"])
    assert str(last_dev) == "2023-09"
    assert last_dev < first_eval
    assert pd.Period("2026-07", freq="M").ordinal - pd.Period("2025-03", freq="M").ordinal == 16


def test_history_reader_selects_only_allowed_target_rows(tmp_path, labeled_fixture):
    data = tmp_path / "data/raw"
    data.mkdir(parents=True)
    raw = labeled_fixture.reset_index(drop=True).copy()
    # Simulate unparsable future targets. A reader that loads all targets will fail.
    raw[TARGET] = raw[TARGET].astype(object)
    raw.loc[months(raw) > pd.Period("2022-12", freq="M"), TARGET] = "FORBIDDEN_FUTURE_LABEL"
    raw.to_csv(data / "train.csv", index=False)
    base = labeled_fixture.drop(columns=TARGET)
    history = history_at(base, "2022-12", tmp_path)
    expected = labeled_fixture.loc[months(labeled_fixture) <= pd.Period("2022-12", freq="M"), TARGET]
    np.testing.assert_array_equal(history[TARGET].to_numpy(), expected.to_numpy())
    query = query_at(base, "2022-12")
    assert TARGET not in query
    assert horizon(query, "2022-12").max() == 16


def test_selected_label_reader_restores_order(tmp_path):
    path = tmp_path / "labels.csv"
    pd.DataFrame({TARGET: [2000, 2100, 2300, 2400]}).to_csv(path, index=False)
    np.testing.assert_array_equal(_read_selected_labels(path, np.array([3, 0, 2])), [2400, 2000, 2300])


def test_scoring_attaches_truth_by_source_row_and_rejects_wrong_date(tmp_path):
    data = tmp_path / "data/raw"
    data.mkdir(parents=True)
    pd.DataFrame({"RENT_APPROVAL_DATE": ["2023-10", "2024-01"], TARGET: [2000, 2500]}).to_csv(data / "train.csv", index=False)
    predictions = pd.DataFrame({"origin": ["2023-09", "2023-09"], "row_id": [1, 0],
                                "RENT_APPROVAL_DATE": ["2024-01", "2023-10"], "horizon": [4, 1], "a": [2501, 2001]})
    result = attach_truth(predictions, tmp_path)
    np.testing.assert_array_equal(result[TARGET], [2500, 2000])
    with pytest.raises(ValueError, match="identity/date"):
        attach_truth(predictions.assign(RENT_APPROVAL_DATE=["2023-10", "2024-01"]), tmp_path)


def test_scoring_is_sample_weighted_and_identical_rows_required():
    frame = pd.DataFrame({"origin": ["2023-09"] * 3, "row_id": [1, 2, 3],
                          "RENT_APPROVAL_DATE": ["2023-10", "2024-01", "2025-01"],
                          "horizon": [1, 4, 16], TARGET: [100, 100, 100], "a": [101, 102, 106], "b": [100, 100, 100]})
    result = metrics(frame, ["a", "b"], read_protocol())
    pooled = result.loc[(result.origin == "pooled_instances") & (result.horizon == "all") & (result.model == "a")].iloc[0]
    assert pooled.rmse == pytest.approx(np.sqrt(41 / 3))
    assert pooled.bias == 3
    assert pooled.n == 3
    with pytest.raises(ValueError, match="every model"):
        metrics(frame.assign(b=[100, np.nan, 100]), ["a", "b"], read_protocol())
    with pytest.raises(ValueError, match="duplicate"):
        metrics(pd.concat([frame, frame.iloc[[0]]]), ["a", "b"], read_protocol())
    with pytest.raises(ValueError, match="disagree"):
        metrics(frame.assign(horizon=[1, 4, 15]), ["a", "b"], read_protocol())


def test_simple_baseline_is_frozen_and_unknown_group_falls_back(labeled_fixture, tiny_config):
    origin = "2022-12"
    history = labeled_fixture.loc[months(labeled_fixture) <= pd.Period(origin, freq="M")]
    query = query_at(labeled_fixture.drop(columns=TARGET), origin).copy()
    pred = fit_simple(history, query, origin, tiny_config)
    pairs = pd.DataFrame({"town": query.TOWN.to_numpy(), "flat": query.FLAT_TYPE.to_numpy(), "pred": pred})
    assert pairs.groupby(["town", "flat"]).pred.nunique().max() == 1
    unknown = query.iloc[[0]].copy()
    unknown["TOWN"] = "unknown"
    recent = history.loc[months(history) > pd.Period(origin, freq="M") - 3, TARGET]
    np.testing.assert_array_equal(fit_simple(history, unknown, origin, tiny_config), [recent.mean()])


def test_cache_tampering_source_and_row_identity_are_rejected(tmp_path, labeled_fixture):
    """Use a tiny synthetic manifest to exercise both provenance and alignment guards."""
    from src.backtest import data as module
    cache = tmp_path / "cache"
    cache.mkdir(parents=True)
    source = tmp_path / "data/raw"
    source.mkdir(parents=True)
    frame = labeled_fixture[BASE_COLUMNS].reset_index(drop=True).iloc[:5].copy()
    frame.to_csv(source / "train.csv", index=False)
    path = cache / "train_base.parquet"
    frame.to_parquet(path, index=False)
    # Monkeypatch only source/code hashing, not row-identity verification.
    original = module.source_hashes
    module.source_hashes = lambda root, split: {"source": module.sha256(root / "data/raw/train.csv")}
    try:
        manifest = {"rows": len(frame), "columns": BASE_COLUMNS, "parquet_sha256": module.sha256(path),
                    "identity_sha256": module.identity_hash(module.read_predictors(tmp_path, "train")),
                    "source_sha256": module.source_hashes(tmp_path, "train")}
        mpath = cache / "train_manifest.json"
        mpath.write_text(json.dumps(manifest), encoding="utf-8")
        load_cache("train", tmp_path, cache)
        # Valid parquet checksum alone cannot conceal row reordering.
        frame.iloc[::-1].to_parquet(path, index=False)
        manifest["parquet_sha256"] = module.sha256(path)
        mpath.write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(ValueError, match="identities"):
            load_cache("train", tmp_path, cache)
        with (source / "train.csv").open("a") as stream:
            stream.write("\n")
        with pytest.raises(ValueError, match="source/code"):
            load_cache("train", tmp_path, cache)
    finally:
        module.source_hashes = original
