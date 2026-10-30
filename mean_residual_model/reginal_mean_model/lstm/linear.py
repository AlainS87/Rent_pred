"""Fit the 4-input three-layer LSTM with a linear input bypass.

    python -m mean_residual_model.reginal_mean_model.lstm.linear
"""

import json

import numpy as np
import pandas as pd
import torch
from torch import nn

from mean_residual_model.reginal_mean_model.lstm.model import (
    TownLSTMLinear,
    _history_timeline,
    _predict_timeline,
)
from mean_residual_model.reginal_mean_model.sequence_store import load_config, save_town_states
from mean_residual_model.regional import (
    FIT_END,
    LEVELS,
    SEARCH_MONTHS,
    CONFIRM_MONTHS,
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
FOLDER = REGIONAL_DIR / "lstm" / "h16_l3_linear"


def _as_unit_rent(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["MONTHLY_RENT"] = work["MONTHLY_RENT"] / work["floor_area_sqm"]
    return work


def main() -> None:
    torch.manual_seed(SEED)
    frame = load_joined("train", ["RENT_APPROVAL_DATE", "TOWN", "floor_area_sqm", *LEVELS])
    prepared = with_month(_as_unit_rent(frame))
    fit_rows = prepared.loc[prepared["month"].astype(str) <= FIT_END]
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
        network = TownLSTMLinear(LAYERS, len(COLUMNS))
        optimizer = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE)
        network.train()
        for _epoch in range(EPOCHS):
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(network(features).squeeze(0), rent)
            loss.backward()
            optimizer.step()
        states[str(town)] = {key: value.detach().cpu() for key, value in network.state_dict().items()}
        print(f"fit {town}", flush=True)
    save_town_states(FOLDER, LAYERS, COLUMNS, numeric, center, spread, states)
    config = load_config(FOLDER)
    series = _history_timeline(config)
    prediction = _predict_timeline(FOLDER, series, TownLSTMLinear)
    labeled = cell_table(prepared)
    labeled["prediction"] = [
        prediction[(str(town), str(month))]
        for town, month in zip(labeled["TOWN"], labeled["month"])
    ]
    scores = {
        "name": "lstm_h16_l3_linear",
        "layers": LAYERS,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "target": "town_month_mean_rent_per_sqm",
        "baseline_search_cell_rmse_dollars": 88.75068269775853,
    }
    for label, months in (("search", SEARCH_MONTHS), ("confirm", CONFIRM_MONTHS)):
        part = labeled.loc[labeled["month"].astype(str).isin(months)]
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
    (FOLDER / "metrics.json").write_text(json.dumps(scores, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(scores, indent=2), flush=True)


if __name__ == "__main__":
    main()
