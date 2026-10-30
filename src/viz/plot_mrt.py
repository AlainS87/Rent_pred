"""MRT distance and station-count figures. Does not read MONTHLY_RENT."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "Data" / "auxiliary_features" / "train_features.csv"
OUT = ROOT / "outputs" / "figures"
LINE_DISTANCES = [
    "dist_nearest_ns_km",
    "dist_nearest_ew_km",
    "dist_nearest_ne_km",
    "dist_nearest_cc_km",
    "dist_nearest_dt_km",
    "dist_nearest_te_km",
]
COUNTS = [
    "n_open_mrt_within_0_5km",
    "n_open_mrt_within_1km",
    "n_open_mrt_within_2km",
]


def main() -> None:
    columns = ["dist_nearest_open_mrt_km", *LINE_DISTANCES, *COUNTS]
    frame = pd.read_csv(TRAIN, usecols=columns)
    OUT.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(frame["dist_nearest_open_mrt_km"].dropna(), bins=40, color="#4C78A8")
    ax.set_xlabel("Kilometres")
    ax.set_title("Distance to nearest open trunk station")
    fig.tight_layout()
    fig.savefig(OUT / "mrt_nearest_trunk.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.boxplot([frame[column].dropna() for column in LINE_DISTANCES], tick_labels=["NS", "EW", "NE", "CC", "DT", "TE"])
    ax.set_ylabel("Kilometres")
    ax.set_title("Distance to nearest station on each trunk line")
    fig.tight_layout()
    fig.savefig(OUT / "mrt_line_distances.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for column, label in zip(COUNTS, ["0.5 km", "1 km", "2 km"]):
        ax.hist(frame[column], bins=range(0, int(frame[column].max()) + 2), histtype="step", label=label)
    ax.set_xlabel("Open trunk-line codes")
    ax.set_title("Stations within a fixed radius")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "mrt_radius_counts.png", dpi=120)
    plt.close(fig)
    print(f"wrote MRT figures to {OUT}")


if __name__ == "__main__":
    main()
