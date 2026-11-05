import copy
import json

import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace

from src.backtest.data import sha256
from src.backtest.checkpoints import fit_contract, save_checkpoint
from src.backtest.execution import atomic_json
from src.backtest.frozen import DEVELOPMENT_MODELS, make_manifest, prediction_identity, read_manifest
from src.backtest.protocol import TARGET, digest_json, history_hash, months
from src.backtest.scoring import metrics
from src.backtest.selection import derive_selection, select_and_freeze, verify_selection, write_origin_receipt


@pytest.fixture
def frozen_evidence(checkpoint_path, labeled_fixture, tiny_config):
    root = checkpoint_path
    dataset = root / "fixture_source"
    (dataset / "data/raw").mkdir(parents=True)
    raw = labeled_fixture.reset_index(drop=True).copy()
    raw.to_csv(dataset / "data/raw/train.csv", index=False)
    config = copy.deepcopy(tiny_config)
    config["evaluation_origins"] = ["2023-09"]
    config["last_label_month"] = "2024-04"
    base = raw.drop(columns=TARGET)
    manifest = make_manifest(config, base, {"fixture": True}, dataset)
    atomic_json(root / "frozen_protocol.json", manifest)
    return root, dataset, raw, base, config, manifest


def write_development(evidence, origin, tied=False, missing_row=False):
    root, dataset, raw, base, config, manifest = evidence
    folder = root / "development" / origin
    folder.mkdir(parents=True)
    frame = prediction_identity(base, origin, config)
    for model in DEVELOPMENT_MODELS:
        offset = 10 if model == "decomposed_flat" or tied else 5 if model == "decomposed_damped" else 20
        frame[model] = raw.set_index("row_id").loc[frame.row_id, TARGET].to_numpy() + offset
    frame.to_parquet(folder / "predictions_without_truth.parquet", index=False)
    frame[TARGET] = raw.set_index("row_id").loc[frame.row_id, TARGET].to_numpy()
    if missing_row:
        frame = frame.iloc[1:].copy()
    frame.to_parquet(folder / "predictions.parquet", index=False)
    metrics(frame, DEVELOPMENT_MODELS, config).to_csv(folder / "metrics.csv", index=False)
    history = raw.loc[months(raw) <= pd.Period(origin, freq="M")]
    summary = {"manifest_sha256": manifest["manifest_sha256"], "config_sha256": manifest["protocol_sha256"],
               "origin": origin, "phase": "development", "models": DEVELOPMENT_MODELS,
               "sampling": None, "allowed_history_sha256": history_hash(history)}
    query = base.loc[(months(base) > pd.Period(origin, freq="M")) & (months(base) <= pd.Period(origin, freq="M") + 16)]
    for model in DEVELOPMENT_MODELS:
        contract = fit_contract(model, origin, history, query, config, manifest["environment"])
        individual = frame[["origin", "row_id", "RENT_APPROVAL_DATE", "horizon"]].copy()
        individual["prediction"] = frame[model].to_numpy()
        audit = {"model": model, "config_sha256": manifest["protocol_sha256"],
                 "allowed_history_sha256": summary["allowed_history_sha256"], "selected_iterations": 1}
        fixture_bundle = None if model == "simple_recent" else SimpleNamespace(origin=origin, audit=audit)
        save_checkpoint(folder / model, contract, fixture_bundle, individual, {})
    atomic_json(folder / "origin_summary.json", summary)
    write_origin_receipt(folder, summary)


def test_both_complete_windows_choose_pooled_sse_and_exact_tie_flat(frozen_evidence):
    root, _, _, _, config, manifest = frozen_evidence
    for origin in config["development_origins"]:
        write_development(frozen_evidence, origin)
    selected = select_and_freeze(root, manifest)
    assert selected["selected_trend"] == "damped"
    assert selected["pooled_development"]["flat"]["n"] == 384
    assert selected["pooled_development"]["damped"]["sse"] == 384 * 25
    assert verify_selection(root, manifest, selected["selection_sha256"]) == selected
    assert select_and_freeze(root, manifest) == selected


def test_exact_tie_chooses_flat(frozen_evidence):
    root, _, _, _, config, manifest = frozen_evidence
    for origin in config["development_origins"]:
        write_development(frozen_evidence, origin, tied=True)
    assert derive_selection(root, manifest)["selected_trend"] == "flat"


def test_missing_window_or_candidate_row_cannot_select(frozen_evidence):
    root, _, _, _, config, manifest = frozen_evidence
    write_development(frozen_evidence, config["development_origins"][0])
    with pytest.raises(FileNotFoundError):
        derive_selection(root, manifest)
    write_development(frozen_evidence, config["development_origins"][1], missing_row=True)
    with pytest.raises(ValueError, match="incomplete|misaligned"):
        derive_selection(root, manifest)


def test_manifest_and_selection_tampering_are_rejected_even_with_resigned_wrong_choice(frozen_evidence):
    root, _, _, _, config, manifest = frozen_evidence
    for origin in config["development_origins"]:
        write_development(frozen_evidence, origin)
    choice = select_and_freeze(root, manifest)
    wrong = copy.deepcopy(choice)
    wrong["selected_trend"] = "flat"
    atomic_json(root / "selected_trend.json", wrong)
    with pytest.raises(ValueError, match="checksum"):
        verify_selection(root, manifest, choice["selection_sha256"])
    wrong["selection_sha256"] = digest_json({k: v for k, v in wrong.items() if k != "selection_sha256"})
    atomic_json(root / "selected_trend.json", wrong)
    with pytest.raises(ValueError, match="complete frozen"):
        verify_selection(root, manifest, wrong["selection_sha256"])
    altered = copy.deepcopy(manifest)
    altered["protocol"]["catboost"]["iterations"] = 1
    atomic_json(root / "frozen_protocol.json", altered)
    with pytest.raises(ValueError, match="checksum"):
        read_manifest(root, manifest["manifest_sha256"])


def test_metrics_must_correspond_to_predictions_and_current_history(frozen_evidence):
    root, dataset, raw, _, config, manifest = frozen_evidence
    for origin in config["development_origins"]:
        write_development(frozen_evidence, origin)
    folder = root / "development" / config["development_origins"][0]
    table = pd.read_csv(folder / "metrics.csv")
    table.loc[0, "rmse"] += 1
    table.to_csv(folder / "metrics.csv", index=False)
    summary = json.loads((folder / "origin_summary.json").read_text())
    write_origin_receipt(folder, summary)  # valid file checksums cannot hide incorrect scoring
    with pytest.raises(AssertionError):
        derive_selection(root, manifest)
    table.loc[0, "rmse"] -= 1
    table.to_csv(folder / "metrics.csv", index=False)
    write_origin_receipt(folder, summary)
    raw.loc[0, TARGET] += 1
    raw.to_csv(dataset / "data/raw/train.csv", index=False)
    with pytest.raises(ValueError, match="historical targets"):
        derive_selection(root, manifest)


def test_one_candidate_missing_one_prediction_cannot_select(frozen_evidence):
    root, _, _, _, config, manifest = frozen_evidence
    for origin in config["development_origins"]:
        write_development(frozen_evidence, origin)
    folder = root / "development" / config["development_origins"][1]
    frame = pd.read_parquet(folder / "predictions.parquet")
    frame.loc[0, "decomposed_damped"] = np.nan
    frame.to_parquet(folder / "predictions.parquet", index=False)
    write_origin_receipt(folder, json.loads((folder / "origin_summary.json").read_text()))
    with pytest.raises(ValueError, match="finite predictions"):
        derive_selection(root, manifest)


@pytest.mark.training
def test_actual_origin_worker_fits_then_restores_without_overwriting_first_attempt(frozen_evidence, monkeypatch):
    from src.backtest import data, frozen, models, decomposed, worker
    import sys
    root, dataset, raw, base, config, manifest = frozen_evidence
    original_history_at = data.history_at
    monkeypatch.setattr(data, "load_cache", lambda split: base.copy())
    monkeypatch.setattr(data, "history_at", lambda frame, origin, source=dataset: original_history_at(frame, origin, source))
    def fixture_runtime(value):
        assert value["environment"] == {"fixture": True}
        assert value["protocol_sha256"] == digest_json(config)
    monkeypatch.setattr(frozen, "verify_runtime", fixture_runtime)
    monkeypatch.setattr(sys, "argv", ["worker", "--run-dir", str(root), "--manifest-hash", manifest["manifest_sha256"],
                                    "--phase", "development", "--origin", "2022-02"])
    assert worker.main() == 0
    folder = root / "development/2022-02"
    original_receipt = (folder / "origin_complete.json").read_bytes()
    originals = {name: (folder / name / "observations.json").read_bytes() for name in DEVELOPMENT_MODELS}
    attempts = list((root / "attempts/worker").iterdir())
    assert len(attempts) == 1
    original_finished = (attempts[0] / "finished.json").read_bytes()
    def forbidden(*args, **kwargs):
        raise AssertionError("a valid completed origin must not train/select again")
    monkeypatch.setattr(models, "fit_direct", forbidden)
    monkeypatch.setattr(models, "fit_simple", forbidden)
    monkeypatch.setattr(decomposed, "fit_decomposed", forbidden)
    assert worker.main() == 0
    assert (folder / "origin_complete.json").read_bytes() == original_receipt
    assert (attempts[0] / "finished.json").read_bytes() == original_finished
    assert len(list((root / "attempts/worker").iterdir())) == 2
    for name in DEVELOPMENT_MODELS:
        assert (folder / name / "observations.json").read_bytes() == originals[name]
    assert json.loads((root / "worker_status.json").read_text())["stage"] == "restored_complete_origin"


def test_readonly_status_works_while_locks_are_held_without_writing(frozen_evidence):
    import subprocess
    import sys
    from src.backtest.execution import RunLock
    from src.backtest.protocol import ROOT
    root, _, _, _, _, manifest = frozen_evidence
    atomic_json(root / "status.json", {"state": "running", "sentinel": "keep"})
    before = (root / "status.json").read_bytes()
    with RunLock(root / "supervisor.lock"), RunLock(root / "worker.lock"):
        result = subprocess.run([sys.executable, "-m", "src.backtest.supervisor", "status", "--run-dir", str(root),
                                 "--manifest-hash", manifest["manifest_sha256"]], cwd=ROOT, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    assert json.loads(result.stdout)["supervisor"] == {"state": "running", "sentinel": "keep"}
    assert (root / "status.json").read_bytes() == before
    assert not (root / "attempts").exists()


@pytest.mark.parametrize("selection_state", ["missing", "tampered"])
def test_formal_worker_refuses_unfrozen_or_tampered_selection_before_any_fit(frozen_evidence, monkeypatch, selection_state):
    import sys
    from src.backtest import frozen, models, decomposed, worker
    root, _, _, _, config, manifest = frozen_evidence
    monkeypatch.setattr(frozen, "verify_runtime", lambda value: None)  # isolated synthetic dataset only
    calls = []
    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError("formal fit before selection validation")
    monkeypatch.setattr(models, "fit_direct", forbidden)
    monkeypatch.setattr(models, "fit_simple", forbidden)
    monkeypatch.setattr(decomposed, "fit_decomposed", forbidden)
    arguments = ["worker", "--run-dir", str(root), "--manifest-hash", manifest["manifest_sha256"],
                 "--phase", "evaluation", "--origin", "2023-09"]
    if selection_state == "tampered":
        for origin in config["development_origins"]:
            write_development(frozen_evidence, origin)
        selected = select_and_freeze(root, manifest)
        altered = dict(selected, selected_trend="flat")
        atomic_json(root / "selected_trend.json", altered)
        arguments += ["--selection-hash", selected["selection_sha256"]]
    monkeypatch.setattr(sys, "argv", arguments)
    assert worker.main() == 1
    assert not calls
    state = json.loads((root / "worker_status.json").read_text())
    assert state["state"] == "failed" and state["exit_status"] == 1
    assert not (root / "evaluation").exists()
