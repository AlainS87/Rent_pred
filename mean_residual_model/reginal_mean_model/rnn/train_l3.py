"""Fit a 4-input three-layer RNN on the town-month mean rent per square metre.

    python -m mean_residual_model.reginal_mean_model.rnn.train_l3
"""

import json

import numpy as np
import torch
from torch import nn

from mean_residual_model.reginal_mean_model.lstm.linear import _as_unit_rent
from mean_residual_model.reginal_mean_model.lstm.model import _history_timeline, _predict_timeline
from mean_residual_model.reginal_mean_model.rnn.model import TownRNN
from mean_residual_model.reginal_mean_model.sequence_store import load_config, save_town_states
from mean_residual_model.regional import (
    CONFIRM_MONTHS,
    FIT_END,
    LEVELS,
    SEARCH_MONTHS,
    _numeric_block,
    _numeric_params,
    _weighted_rmse,
    cell_table,
    with_month,
)
from mean_residual_model.run import REGIONAL_DIR, SEED
from src.pipeline.data import load_joined

COLUMNS = ["month_index", *LEVELS]
LAYERS = 3
EPOCHS = 50
LEARNING_RATE = 0.01
FOLDER = REGIONAL_DIR / "rnn" / "h16_l3"
LINEAR_LSTM_SEARCH = 1.9645978411632352
LINEAR_LSTM_CONFIRM = 3.4044068287564584


def fit_rnn(folder, fit_end: str | None, score_holdout: bool) -> dict:
    """Fit one three-layer RNN per town.

    ``fit_end`` keeps months through that approval month. ``None`` uses every
    labeled month. Holdout cell RMSE is written only when ``score_holdout`` is
    true, and only for the 2024-09 cutoff.
    """
    torch.manual_seed(SEED)
    frame = load_joined("train", ["RENT_APPROVAL_DATE", "TOWN", "floor_area_sqm", *LEVELS])
    prepared = with_month(_as_unit_rent(frame))
    if fit_end is None:
        fit_rows = prepared
    else:
        fit_rows = prepared.loc[prepared["month"].astype(str) <= fit_end]
    if fit_rows.empty:
        raise SystemExit("the RNN fit months have no rows")
    cells = cell_table(fit_rows)
    numeric = _numeric_params(cells, COLUMNS)
    target = cells["mean_rent"].to_numpy(dtype=np.float64)
    center = float(np.mean(target))
    spread = float(np.std(target)) or 1.0
    states = {}
    for town, part in cells.groupby(cells["TOWN"].astype(str)):
        ordered = part.sort_values("month_index")
        features = torch.tensor(
            _numeric_block(ordered, COLUMNS, numeric),
            dtype=torch.float32,
        ).unsqueeze(0)
        rent = torch.tensor(
            (ordered["mean_rent"].to_numpy(dtype=np.float64) - center) / spread,
            dtype=torch.float32,
        )
        network = TownRNN(LAYERS, len(COLUMNS))
        optimizer = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE)
        network.train()
        for _epoch in range(EPOCHS):
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(network(features).squeeze(0), rent)
            loss.backward()
            optimizer.step()
        states[str(town)] = {key: value.detach().cpu() for key, value in network.state_dict().items()}
        print(f"fit {town}", flush=True)
    save_town_states(folder, LAYERS, COLUMNS, numeric, center, spread, states)
    months = sorted(fit_rows["month"].astype(str).unique())
    scores = {
        "name": "rnn_h16_l3",
        "layers": LAYERS,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "target": "town_month_mean_rent_per_sqm",
        "fit_through": months[-1],
        "holdout_scored": score_holdout,
    }
    if score_holdout:
        config = load_config(folder)
        series = _history_timeline(config)
        prediction = _predict_timeline(folder, series, TownRNN)
        labeled = cell_table(prepared)
        labeled["prediction"] = [
            prediction[(str(town), str(month))]
            for town, month in zip(labeled["TOWN"], labeled["month"])
        ]
        scores["linear_lstm_search_cell_rmse"] = LINEAR_LSTM_SEARCH
        scores["linear_lstm_confirm_cell_rmse"] = LINEAR_LSTM_CONFIRM
        for label, month_list in (("search", SEARCH_MONTHS), ("confirm", CONFIRM_MONTHS)):
            part = labeled.loc[labeled["month"].astype(str).isin(month_list)]
            scores[f"{label}_weighted_cell_rmse"] = _weighted_rmse(
                part["mean_rent"].to_numpy(dtype=np.float64),
                part["prediction"].to_numpy(dtype=np.float64),
                part["rows"].to_numpy(dtype=np.float64),
            )
        forward = series.copy()
        forward["prediction"] = [
            prediction[(str(town), str(month))]
            for town, month in zip(forward["TOWN"], forward["month"])
        ]
        for month in ("2025-04", "2026-07"):
            part = forward.loc[forward["month"].astype(str) == month, "prediction"]
            scores[f"forward_unit_{month.replace('-', '_')}"] = float(part.mean())
    (folder / "metrics.json").write_text(json.dumps(scores, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(scores, indent=2), flush=True)
    return scores


def main() -> None:
    fit_rnn(FOLDER, FIT_END, True)


if __name__ == "__main__":
    main()
