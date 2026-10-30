"""Load the saved 4-input three-layer town LSTM. No refit."""

import numpy as np
import pandas as pd
import torch
from torch import nn

from mean_residual_model.regional import LEVELS, _numeric_block, cell_table, with_month
from mean_residual_model.run import REGIONAL_DIR

FORWARD_START = "2025-04"
FORWARD_END = "2026-07"


class TownLSTM(nn.Module):
    def __init__(self, layers: int, n_inputs: int) -> None:
        super().__init__()
        self.core = nn.LSTM(n_inputs, 16, num_layers=layers, batch_first=True)
        self.head = nn.Linear(16, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        output, _state = self.core(features)
        return self.head(output).squeeze(-1)


class TownLSTMLinear(nn.Module):
    """Three-layer LSTM plus a linear map of the same inputs.

    The hidden state still passes through tanh. ``a + b x`` does not.
    """

    def __init__(self, layers: int, n_inputs: int) -> None:
        super().__init__()
        self.core = nn.LSTM(n_inputs, 16, num_layers=layers, batch_first=True)
        self.head = nn.Linear(16, 1, bias=False)
        self.skip = nn.Linear(n_inputs, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        output, _state = self.core(features)
        return (self.head(output) + self.skip(features)).squeeze(-1)


def predict_saved_l3(frame: pd.DataFrame) -> np.ndarray:
    """Town-month mean from the saved 4-input three-layer LSTM. No refit."""
    folder = REGIONAL_DIR / "lstm" / "h16_l3"
    return _predict_saved(folder, frame, TownLSTM)


def predict_saved_l3_linear(frame: pd.DataFrame) -> np.ndarray:
    """Town-month mean from the saved linear-bypass three-layer LSTM. No refit."""
    folder = REGIONAL_DIR / "lstm" / "h16_l3_linear"
    return _predict_saved(folder, frame, TownLSTMLinear)


def _predict_saved(folder, frame: pd.DataFrame, network_cls: type[nn.Module]) -> np.ndarray:
    from mean_residual_model.reginal_mean_model.sequence_store import load_config

    config = load_config(folder)
    prepared = with_month(frame)
    series = _history_timeline(config)
    prediction = _predict_timeline(folder, series, network_cls)
    values = [
        prediction[(str(town), str(month))]
        for town, month in zip(prepared["TOWN"], prepared["month"])
    ]
    return np.array(values, dtype=np.float64)


def _history_timeline(config: dict) -> pd.DataFrame:
    from src.pipeline.data import load_joined

    columns = list(config["columns"])
    if columns == ["month_index", *LEVELS]:
        train = load_joined("train", ["RENT_APPROVAL_DATE", "TOWN", *LEVELS])
        cells = cell_table(train)
    else:
        train = load_joined("train", ["RENT_APPROVAL_DATE", "TOWN", *columns])
        cells = _economy_cells(train, columns)
    extra = [name for name in columns if name != "month_index"]
    test = load_joined("test", ["RENT_APPROVAL_DATE", "TOWN", *extra])
    future = _future_economy(test, train, columns)
    future = future.loc[future["month"].astype(str) >= FORWARD_START]
    keep = ["TOWN", "month", *columns]
    table = pd.concat([cells[keep], future[keep]], ignore_index=True)
    return table.drop_duplicates(["TOWN", "month"]).reset_index(drop=True)


def _predict_timeline(folder, series: pd.DataFrame, network_cls: type[nn.Module]) -> dict[tuple[str, str], float]:
    from mean_residual_model.reginal_mean_model.sequence_store import load_config, load_state

    config = load_config(folder)
    columns = list(config["columns"])
    center = float(config["target_center"])
    spread = float(config["target_spread"])
    prediction = {}
    for slug, town in config["towns"].items():
        network = network_cls(int(config["layers"]), len(columns))
        network.load_state_dict(load_state(folder, slug))
        network.eval()
        ordered = series.loc[series["TOWN"].astype(str) == str(town)].sort_values("month_index")
        features = torch.tensor(
            _numeric_block(ordered, columns, config["numeric"]),
            dtype=torch.float32,
        ).unsqueeze(0)
        with torch.no_grad():
            values = network(features).squeeze(0).numpy() * spread + center
        for month, value in zip(ordered["month"], values, strict=True):
            prediction[(str(town), str(month))] = float(value)
    return prediction


def _economy_cells(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    prepared = with_month(frame)
    grouped = prepared.groupby(["TOWN", "month"], observed=True)
    cells = grouped["MONTHLY_RENT"].agg(mean_rent="mean", rows="size").reset_index()
    month_index = prepared.groupby("month")["month_index"].first()
    cells["month_index"] = cells["month"].map(month_index)
    for column in columns:
        if column == "month_index":
            continue
        values = prepared.groupby("month")[column].first()
        cells[column] = cells["month"].map(values)
    return cells


def _future_economy(test: pd.DataFrame, fit: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    work = with_month(test)
    level_columns = [column for column in columns if column != "month_index"]
    by_month = work.groupby("month", as_index=False)[level_columns].first()
    by_month = by_month.loc[by_month["month"].astype(str) >= FORWARD_START].copy()
    frames = []
    for town in sorted(fit["TOWN"].astype(str).unique()):
        part = by_month.copy()
        part["TOWN"] = town
        part["RENT_APPROVAL_DATE"] = part["month"].astype(str) + "-01"
        frames.append(part)
    future = with_month(pd.concat(frames, ignore_index=True))
    months = set(future["month"].astype(str))
    if FORWARD_START not in months or FORWARD_END not in months:
        raise SystemExit(f"forward months missing {FORWARD_START} or {FORWARD_END}")
    return future
