"""Block age, height, and location figures. Does not read MONTHLY_RENT."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "artifacts" / "features" / "train_features.csv"
OUT = ROOT / "outputs" / "figures"


def main() -> None:
    frame = pd.read_csv(TRAIN, usecols=["latitude", "longitude", "region", "building_age_years", "max_floor"])
    OUT.mkdir(parents=True, exist_ok=True)
    blocks = frame.drop_duplicates(["latitude", "longitude"])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(frame["building_age_years"].dropna(), bins=30, color="#4C78A8")
    axes[0].set_title("Building age at approval")
    axes[0].set_xlabel("Years")
    axes[1].hist(frame["max_floor"].dropna(), bins=30, color="#F58518")
    axes[1].set_title("Block maximum floor")
    axes[1].set_xlabel("Floors")
    fig.tight_layout()
    fig.savefig(OUT / "blocks_age_floor.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 7))
    for region, part in blocks.groupby("region"):
        ax.scatter(part["longitude"], part["latitude"], s=4, alpha=0.7, label=region)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("HDB blocks by region")
    ax.legend(markerscale=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "blocks_region_map.png", dpi=120)
    plt.close(fig)
    print(f"wrote block figures to {OUT}")


if __name__ == "__main__":
    import argparse
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
