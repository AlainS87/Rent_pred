"""Per-town RNN for the town-month mean rent per square metre."""

import torch
from torch import nn

from mean_residual_model.reginal_mean_model.lstm.model import _predict_saved
from mean_residual_model.run import REGIONAL_DIR


class TownRNN(nn.Module):
    """RNN whose input is the month index and three lagged economy levels.

    The previous month's rent is not an input. The hidden state carries the
    earlier months.
    """

    def __init__(self, layers: int, n_inputs: int) -> None:
        super().__init__()
        self.core = nn.RNN(n_inputs, 16, num_layers=layers, batch_first=True)
        self.head = nn.Linear(16, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        output, _state = self.core(features)
        return self.head(output).squeeze(-1)


def predict_saved_rnn3(frame, folder=None):
    """Town-month unit price from a saved 4-input three-layer RNN. No refit."""
    if folder is None:
        folder = REGIONAL_DIR / "rnn" / "h16_l3"
    return _predict_saved(folder, frame, TownRNN)
