"""Heatmaps of the six previous-quarter group-mean rents.

A cell is the smoothed mean the formula would assign to that group in that
approval month. The month itself contributes no rents. A group with no rental
in that month is still drawn when the previous three months can be evaluated.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.features.group_means import postal_district, postal_sector, smoothed_group_month_table

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "Data" / "auxiliary_features" / "train_features.csv"
RENTALS = ROOT / "Data" / "train.csv"
OUT = ROOT / "outputs" / "figures"

PLOTS = (
    ("postal_district", "mean_rent_postal_district", "Postal district", "heatmap_postal_district.png"),
    ("postal_sector", "mean_rent_postal_sector", "Postal sector", "heatmap_postal_sector.png"),
    ("nearest_mall_name", "mean_rent_nearest_mall", "Nearest mall", "heatmap_nearest_mall.png"),
    ("nearest_open_mrt_code", "mean_rent_nearest_mrt", "Nearest open MRT code", "heatmap_nearest_mrt.png"),
    ("nearest_primary_name", "mean_rent_nearest_primary", "Nearest primary school", "heatmap_nearest_primary.png"),
    ("nearest_secondary_name", "mean_rent_nearest_secondary", "Nearest secondary school", "heatmap_nearest_secondary.png"),
)


def _load() -> pd.DataFrame:
    columns = [
        "postal_code",
        "nearest_mall_name",
        "nearest_open_mrt_code",
        "nearest_primary_name",
        "nearest_secondary_name",
    ]
    frame = pd.read_csv(TRAIN, usecols=columns)
    rentals = pd.read_csv(RENTALS, usecols=["RENT_APPROVAL_DATE", "MONTHLY_RENT"])
    if len(rentals) != len(frame):
        raise ValueError("train.csv and train feature rows are not aligned")
    frame.insert(0, "RENT_APPROVAL_DATE", rentals["RENT_APPROVAL_DATE"].to_numpy())
    frame.insert(1, "MONTHLY_RENT", rentals["MONTHLY_RENT"].to_numpy())
    frame["postal_sector"] = postal_sector(frame["postal_code"])
    frame["postal_district"] = postal_district(frame["postal_sector"])
    return frame


def _heatmap(frame: pd.DataFrame, group: str, ylabel: str, path: Path) -> None:
    table = smoothed_group_month_table(frame, frame[group], frame["RENT_APPROVAL_DATE"])
    table = table.dropna(how="all")
    table = table.loc[table.mean(axis=1).sort_values().index]

    height = max(3.2, 0.18 * len(table))
    fig, ax = plt.subplots(figsize=(11, height))
    image = np.ma.masked_invalid(table.to_numpy(dtype=float))
    panel = ax.imshow(image, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_xticks(np.arange(table.shape[1]))
    ax.set_xticklabels(table.columns, rotation=90, fontsize=6)
    ax.set_yticks(np.arange(table.shape[0]))
    ax.set_yticklabels(table.index.astype(str), fontsize=6)
    ax.set_xlabel("Approval month")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Previous-quarter smoothed mean rent by {ylabel.lower()}")
    bar = fig.colorbar(panel, ax=ax, fraction=0.03, pad=0.02)
    bar.set_label("SGD")
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    frame = _load()
    OUT.mkdir(parents=True, exist_ok=True)
    for group, _value, ylabel, filename in PLOTS:
        _heatmap(frame, group, ylabel, OUT / filename)
        print(f"wrote {OUT / filename}")


if __name__ == "__main__":
    main()
