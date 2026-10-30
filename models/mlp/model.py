"""MLP architecture. Hidden sizes and dropout are the architecture gene."""

from pathlib import Path

import numpy as np
import torch
from torch import nn

WEIGHT_NAME = "model.pt"
TITLE = "MLP"

DROPOUT_STEP = 0.1
DROPOUT_CAP = 0.5
WIDTH_CAP = 512
DEPTH_CAP = 4

MODEL_NOTE = f"""## Architecture gene

The spec is `hidden_sizes` and `dropout`. Regularize raises dropout by {DROPOUT_STEP} up to {DROPOUT_CAP}. Capacity doubles the first hidden layer up to {WIDTH_CAP}, then appends a layer copied from the last width, up to {DEPTH_CAP} layers.
"""


class RentMLP(nn.Module):
    def __init__(self, n_features: int, hidden_sizes: list[int], dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        width = n_features
        for hidden in hidden_sizes:
            layers.append(nn.Linear(width, int(hidden)))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            width = int(hidden)
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def fit_model(x_fit, y_fit, x_search, y_search, architecture: dict, seed: int, epochs: int | None = None):
    from src.pipeline.preprocess import assert_fit_inputs

    x_fit, y_fit = assert_fit_inputs(x_fit, y_fit)
    torch.manual_seed(seed)
    hidden = [int(size) for size in architecture["hidden_sizes"]]
    dropout = float(architecture["dropout"])
    learning_rate = float(architecture.get("learning_rate", 0.001))
    epoch_count = int(epochs if epochs is not None else architecture.get("epochs", 20))
    batch_size = int(architecture.get("batch_size", 1024))
    patience = int(architecture.get("patience", 3))
    model = RentMLP(x_fit.shape[1], hidden, dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    generator = torch.Generator()
    generator.manual_seed(seed)
    features = torch.tensor(np.asarray(x_fit), dtype=torch.float32)
    target = torch.tensor(np.asarray(y_fit), dtype=torch.float32)
    search_x = None
    search_y = None
    if patience > 0:
        x_search, y_search = assert_fit_inputs(x_search, y_search)
        search_x = torch.tensor(np.asarray(x_search), dtype=torch.float32)
        search_y = np.asarray(y_search, dtype=np.float64)
    best_state = None
    best_rmse = float("inf")
    best_epoch = 0
    stall = 0
    for epoch in range(1, epoch_count + 1):
        model.train()
        order = torch.randperm(features.shape[0], generator=generator)
        for start in range(0, features.shape[0], batch_size):
            batch = order[start : start + batch_size]
            optimizer.zero_grad()
            loss = loss_fn(model(features[batch]), target[batch])
            loss.backward()
            optimizer.step()
        if patience <= 0:
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            best_epoch = epoch
            continue
        prediction = _predict_tensor(model, search_x)
        score = float(np.sqrt(np.mean((search_y - prediction) ** 2)))
        if score < best_rmse - 1e-6:
            best_rmse = score
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            best_epoch = epoch
            stall = 0
        else:
            stall += 1
            if epoch_count > 1 and stall >= patience:
                break
    if best_state is None or best_epoch < 1:
        raise RuntimeError("MLP training produced no weights")
    model.load_state_dict(best_state)
    return {
        "state": best_state,
        "architecture": {"hidden_sizes": hidden, "dropout": dropout},
        "best_epoch": best_epoch,
        "n_features": int(x_fit.shape[1]),
    }


def predict_model(bundle: dict, features) -> np.ndarray:
    from src.pipeline.preprocess import assert_prediction

    model = RentMLP(
        bundle["n_features"],
        bundle["architecture"]["hidden_sizes"],
        bundle["architecture"]["dropout"],
    )
    model.load_state_dict(bundle["state"])
    raw = _predict_tensor(model, torch.tensor(np.asarray(features), dtype=torch.float32))
    return assert_prediction(raw, len(np.asarray(features)))


def save_model(bundle: dict, directory: Path) -> None:
    torch.save(bundle, directory / WEIGHT_NAME)


def load_model(directory: Path) -> dict:
    return torch.load(directory / WEIGHT_NAME, map_location="cpu", weights_only=False)


def mutate_architecture(architecture: dict, mode: str) -> dict | None:
    """Apply one regularize or capacity step. Return None at the cap."""
    updated = json_copy(architecture)
    hidden = [int(size) for size in updated["hidden_sizes"]]
    dropout = round(float(updated["dropout"]), 2)
    if mode == "regularize":
        nxt = round(min(DROPOUT_CAP, dropout + DROPOUT_STEP), 2)
        if nxt == dropout:
            return None
        updated["dropout"] = nxt
        updated["hidden_sizes"] = hidden
        return updated
    if mode == "capacity":
        if hidden[0] < WIDTH_CAP:
            hidden[0] = min(WIDTH_CAP, hidden[0] * 2)
            updated["hidden_sizes"] = hidden
            updated["dropout"] = dropout
            return updated
        if len(hidden) < DEPTH_CAP:
            updated["hidden_sizes"] = hidden + [hidden[-1]]
            updated["dropout"] = dropout
            return updated
        return None
    raise ValueError(f"unknown architecture mode: {mode}")


def _predict_tensor(model: RentMLP, features: torch.Tensor) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(features).numpy().astype(np.float64)


def json_copy(payload: dict) -> dict:
    import json

    return json.loads(json.dumps(payload))
