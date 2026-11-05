import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.backtest.checkpoints import fit_contract, load_checkpoint, save_checkpoint
from src.backtest.models import feature_columns, fit_direct
from src.backtest.protocol import TARGET, history_hash, horizon_bin, inner_split, months


def bounded(frame):
    allowed = months(frame) <= pd.Period("2022-12", freq="M")
    return frame.loc[allowed].copy(), frame.loc[~allowed].drop(columns=TARGET).copy()


@pytest.mark.training
def test_configuration_is_passed_without_hidden_protocol_or_genome_reads(labeled_fixture, tiny_config, monkeypatch):
    train, query = bounded(labeled_fixture)
    config = copy.deepcopy(tiny_config)
    config["inner_validation_months"] = 2
    config["direct_features"]["direct_legacy40"] = ["FLOOR_AREA_SQM", "TOWN"]
    original_read = Path.read_text
    def guarded(path, *args, **kwargs):
        if str(path).replace("\\", "/").endswith(("configs/backtest/protocol.json", "models/catboost/final/genome.json")):
            raise AssertionError("implicit disk configuration read")
        return original_read(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", guarded)
    assert inner_split(train, "2022-12", config)[3] == "2022-10"
    assert horizon_bin(np.array([1, 16]), [[1, 16]]).tolist() == ["1-16", "1-16"]
    assert feature_columns("direct_legacy40", config) == ["FLOOR_AREA_SQM", "TOWN"]
    bundle, pred = fit_direct(train, query, "2022-12", "direct_legacy40", config)
    assert bundle.audit["inner"]["boundary"] == "2022-10"
    assert bundle.audit["feature_columns"] == ["FLOOR_AREA_SQM", "TOWN"]
    assert np.isfinite(pred).all()


def test_history_label_hash_is_future_invariant_and_changes_for_permitted_label(labeled_fixture):
    train, _ = bounded(labeled_fixture)
    original = history_hash(train)
    changed = labeled_fixture.copy()
    changed.loc[months(changed) > pd.Period("2022-12", freq="M"), TARGET] = 1e9
    assert history_hash(bounded(changed)[0]) == original
    train.iloc[0, train.columns.get_loc(TARGET)] += 1
    assert history_hash(train) != original
    assert history_hash(train.sample(frac=1, random_state=3)) == history_hash(train)


@pytest.mark.training
def test_checkpoint_reload_is_exact_and_does_not_refit(labeled_fixture, tiny_config, checkpoint_path, monkeypatch):
    train, query = bounded(labeled_fixture)
    bundle, pred = fit_direct(train, query, "2022-12", "direct_legacy40", tiny_config)
    contract = fit_contract("direct_legacy40", "2022-12", train, query, tiny_config, {"code": "frozen"})
    frame = query[["row_id", "RENT_APPROVAL_DATE"]].copy()
    frame["prediction"] = pred
    save_checkpoint(checkpoint_path, contract, bundle, frame, {"selected_iterations": bundle.audit["selected_iterations"]})
    from src.backtest import models
    monkeypatch.setattr(models, "fit_direct", lambda *args: (_ for _ in ()).throw(AssertionError("refit during restore")))
    restored, rows, observed = load_checkpoint(checkpoint_path, contract)
    pd.testing.assert_frame_equal(rows, frame.reset_index(drop=True))
    np.testing.assert_array_equal(restored.predict(query), pred)
    assert observed["selected_iterations"] == bundle.audit["selected_iterations"]
    with (checkpoint_path / "predictions.parquet").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="checksum"):
        load_checkpoint(checkpoint_path, contract)


@pytest.mark.parametrize("change", ["config", "history", "query", "code", "origin"])
def test_resume_rejects_changed_contract(change, labeled_fixture, tiny_config, checkpoint_path):
    train, query = bounded(labeled_fixture)
    original = fit_contract("simple_recent", "2022-12", train, query, tiny_config, {"code": "a"})
    predictions = query[["row_id"]].copy()
    predictions["prediction"] = 2000.
    save_checkpoint(checkpoint_path, original, None, predictions, {})
    config, history, future, environment, origin = copy.deepcopy(tiny_config), train.copy(), query.copy(), {"code": "a"}, "2022-12"
    if change == "config": config["catboost"]["depth"] += 1
    elif change == "history": history.iloc[0, history.columns.get_loc(TARGET)] += 1
    elif change == "query": future.iloc[0, future.columns.get_loc("FLOOR_AREA_SQM")] += 1
    elif change == "code": environment["code"] = "b"
    else: origin = "2022-11"
    changed = fit_contract("simple_recent", origin, history, future, config, environment)
    with pytest.raises(ValueError, match="contract mismatch"):
        load_checkpoint(checkpoint_path, changed)
