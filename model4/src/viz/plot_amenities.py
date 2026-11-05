"""School and mall distance figures. Does not read MONTHLY_RENT."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "artifacts" / "features" / "train_features.csv"
OUT = ROOT / "outputs" / "figures"


def main() -> None:
    columns = [
        "dist_nearest_primary_km",
        "dist_nearest_secondary_km",
        "dist_nearest_mall_km",
        "n_primary_within_1km",
        "n_secondary_within_1km",
        "n_mall_within_1km",
    ]
    frame = pd.read_csv(TRAIN, usecols=columns)
    OUT.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    specs = [
        ("dist_nearest_primary_km", "Nearest primary school"),
        ("dist_nearest_secondary_km", "Nearest secondary school"),
        ("dist_nearest_mall_km", "Nearest mall"),
    ]
    for ax, (column, title) in zip(axes, specs):
        ax.hist(frame[column].dropna(), bins=40, color="#4C78A8")
        ax.set_xlabel("Kilometres")
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(OUT / "amenity_distances.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for column, label in [
        ("n_primary_within_1km", "Primary schools"),
        ("n_secondary_within_1km", "Secondary schools"),
        ("n_mall_within_1km", "Malls"),
    ]:
        ax.hist(frame[column], bins=range(0, int(frame[column].max()) + 2), histtype="step", label=label)
    ax.set_xlabel("Count within 1 km")
    ax.set_title("Amenities within 1 km")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "amenity_counts_1km.png", dpi=120)
    plt.close(fig)
    print(f"wrote amenity figures to {OUT}")


if __name__ == "__main__":
    import argparse
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
