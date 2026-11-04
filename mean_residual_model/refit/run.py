"""Fit the multistage stack on every labeled month and predict the test file.

The search-chosen RNN epoch count, CatBoost iteration, and MLP epoch stay
frozen. This command does not write a search or confirm RMSE, and it does not
replace the weights selected on 2024-10..12.

    python -m mean_residual_model.refit.run
"""

import importlib.util
import json
from pathlib import Path

import joblib
import numpy as np
import torch

from mean_residual_model.reginal_mean_model.rnn.model import predict_saved_rnn3
from mean_residual_model.reginal_mean_model.rnn.train_l3 import EPOCHS, LAYERS, fit_rnn
from mean_residual_model.reginal_mean_model.sequence_store import town_slug
from mean_residual_model.run import ROOT, _write_kaggle
from src.pipeline.data import load_joined
from src.pipeline.fit import rmse

HERE = Path(__file__).resolve().parent
STACK_PATH = HERE.parent / "multi_stage_model(lstm+catboost+mlp)" / "run.py"
LABELED_END = "2025-03"
RNN_DIR = HERE / "rnn" / "h16_l3"


def _stack():
    spec = importlib.util.spec_from_file_location("multistage_stack", STACK_PATH)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {STACK_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _frozen_trees(stack, town: str) -> int:
    folder = stack.HERE / "catboost" / town_slug(town)
    if not (folder / "model.joblib").is_file():
        raise SystemExit(f"missing saved CatBoost for {town}: {folder}")
    bundle = stack.load_model(folder)
    iteration = bundle.get_best_iteration()
    if iteration is None:
        raise SystemExit(f"{town} CatBoost has no best iteration")
    return int(iteration) + 1


def _fit_town(stack, town: str, frame, residual, test, store, trees: int) -> dict:
    mask = frame["TOWN"].astype(str).to_numpy() == town
    test_mask = test["TOWN"].astype(str).to_numpy() == town
    fit = frame.loc[mask].reset_index(drop=True)
    test_town = test.loc[test_mask].reset_index(drop=True)
    y_fit = residual[mask]
    stack._scale_coords(fit, [test_town])
    stack._write_coords(store, "train", mask, fit)
    if len(test_town):
        stack._write_coords(store, "test", test_mask, test_town)
    architecture = dict(stack.CAT_ARCH)
    architecture["n_estimators"] = trees
    architecture["early_stopping_rounds"] = 0
    preparer = stack.fit_preprocessor(stack.PREPROCESS, fit, y_fit, stack.CAT_FEATURES)
    transformed = stack.transform_frame(preparer, fit)
    bundle = stack.fit_model(transformed, y_fit, transformed, y_fit, architecture, stack.SEED)
    folder = HERE / "catboost" / town_slug(town)
    folder.mkdir(parents=True, exist_ok=True)
    stack.save_model(bundle, folder)
    joblib.dump(preparer, folder / "preprocessor.joblib")
    store["train"]["t"][mask] = stack.predict_model(bundle, transformed)
    if len(test_town):
        store["test"]["t"][test_mask] = stack.predict_model(
            bundle, stack.transform_frame(preparer, test_town)
        )
    mixture = stack._fit_mixture(fit, y_fit)
    gmm_path = HERE / "gmm" / f"{town_slug(town)}.joblib"
    gmm_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(mixture, gmm_path)
    stack._write_mixture(store, "train", mask, fit, mixture)
    if len(test_town):
        stack._write_mixture(store, "test", test_mask, test_town, mixture)
    print(f"  trees {trees}, mixture k {mixture['n_components']}", flush=True)
    return {
        "n_components": mixture["n_components"],
        "town": town,
        "trees": trees,
    }


def _save_mlp(network, epoch: int, center, spread) -> None:
    folder = HERE / "mlp"
    folder.mkdir(parents=True, exist_ok=True)
    x_center, y_center = center
    x_spread, y_spread = spread
    torch.save(
        {
            "epoch": epoch,
            "state": {key: value.detach().cpu() for key, value in network.state_dict().items()},
            "x_center": x_center,
            "x_spread": x_spread,
            "y_center": y_center,
            "y_spread": y_spread,
        },
        folder / "model.pt",
    )


def main() -> None:
    stack = _stack()
    frame = load_joined("train", stack.LOAD_COLUMNS)
    months = sorted(frame["RENT_APPROVAL_DATE"].astype("string").str.slice(0, 7).unique())
    if months[-1] != LABELED_END:
        raise SystemExit(f"labeled months end at {months[-1]}, expected {LABELED_END}")
    test = load_joined("test", stack.LOAD_COLUMNS)
    towns = sorted(frame["TOWN"].astype(str).unique())
    missing = sorted(set(test["TOWN"].astype(str)) - set(towns))
    if missing:
        raise SystemExit(f"test towns absent from the labeled months: {missing}")
    epoch = int(json.loads((stack.HERE / "metrics.json").read_text(encoding="utf-8"))["best_epoch"])
    trees = {town: _frozen_trees(stack, town) for town in towns}
    print(f"labeled rows {len(frame)}, mlp epochs {epoch}", flush=True)

    rnn_scores = fit_rnn(RNN_DIR, None, False)
    if rnn_scores["fit_through"] != LABELED_END or rnn_scores["epochs"] != EPOCHS:
        raise SystemExit(f"refit RNN record is {rnn_scores}")
    if LAYERS != 3:
        raise SystemExit(f"refit RNN layers are {LAYERS}")

    level = predict_saved_rnn3(frame, RNN_DIR)
    test_level = predict_saved_rnn3(test, RNN_DIR)
    units, _areas = stack._unit_and_area(frame)
    residual = units - level
    rents = frame["MONTHLY_RENT"].to_numpy(dtype=np.float64)
    store = stack._empty_store({"train": frame}, test)
    store["train"]["mu"][:] = level
    store["test"]["mu"][:] = test_level
    town_records = []
    for town in towns:
        print(f"refit {town}", flush=True)
        town_records.append(_fit_town(stack, town, frame, residual, test, store, trees[town]))

    network, trained_epochs, center, spread = stack._fit_mlp_epochs(store, {"train": rents}, epoch)
    _save_mlp(network, trained_epochs, center, spread)
    train_prediction = stack._mlp_predict(network, store["train"], center, spread)
    test_prediction = stack._mlp_predict(network, store["test"], center, spread)
    output = ROOT / "outputs" / "predictions" / "mean_residual_multistage_refit.csv"
    _write_kaggle(output, test_prediction)
    scores = {
        "labeled_rows": int(len(frame)),
        "fit_through": LABELED_END,
        "holdout_scored": False,
        "rnn_epochs": EPOCHS,
        "rnn_layers": LAYERS,
        "mlp_epochs": trained_epochs,
        "mlp_train_rmse": rmse(rents, train_prediction),
        "mlp_train_rmse_is_held_out": False,
        "catboost_features_on_labeled_rows": "in_sample",
        "towns": town_records,
        "prediction": output.relative_to(ROOT).as_posix(),
    }
    path = HERE / "metrics.json"
    path.write_text(json.dumps(scores, indent=2) + "\n", encoding="utf-8")
    print(
        f"refit train RMSE {scores['mlp_train_rmse']:.4f} "
        f"on {scores['labeled_rows']} labeled rows, epoch {trained_epochs}"
    )
    print(f"wrote {output.relative_to(ROOT).as_posix()}")
    print(f"wrote {path.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
