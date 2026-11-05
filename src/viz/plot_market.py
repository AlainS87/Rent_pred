"""Monthly COE and stock-basket series. Does not read MONTHLY_RENT."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "artifacts" / "features" / "train_features.csv"
RENTALS = ROOT / "data/raw" / "train.csv"
OUT = ROOT / "outputs" / "figures"
SERIES = [
    "coe_a_premium_lag1",
    "coe_b_premium_lag1",
    "coe_e_premium_lag1",
    "bank_basket_adj_close_lag1",
    "property_basket_adj_close_lag1",
]


def main() -> None:
    features = pd.read_csv(TRAIN, usecols=SERIES)
    dates = pd.read_csv(RENTALS, usecols=["RENT_APPROVAL_DATE"])
    if len(dates) != len(features):
        raise ValueError("train.csv and train feature rows are not aligned")
    frame = features.copy()
    frame.insert(0, "RENT_APPROVAL_DATE", dates["RENT_APPROVAL_DATE"].to_numpy())
    monthly = frame.groupby("RENT_APPROVAL_DATE", as_index=False)[SERIES].first()
    monthly = monthly.sort_values("RENT_APPROVAL_DATE")
    OUT.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 4))
    for column, label in [
        ("coe_a_premium_lag1", "Category A"),
        ("coe_b_premium_lag1", "Category B"),
        ("coe_e_premium_lag1", "Category E"),
    ]:
        ax.plot(monthly["RENT_APPROVAL_DATE"], monthly[column], label=label)
    ax.set_title("Lagged COE premium by approval month")
    ax.set_ylabel("SGD")
    ax.tick_params(axis="x", labelrotation=60, labelsize=7)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "coe_monthly.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(monthly["RENT_APPROVAL_DATE"], monthly["bank_basket_adj_close_lag1"], label="Banks")
    ax.plot(monthly["RENT_APPROVAL_DATE"], monthly["property_basket_adj_close_lag1"], label="Property")
    ax.set_title("Lagged equal-weight adjusted close")
    ax.tick_params(axis="x", labelrotation=60, labelsize=7)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "stock_baskets_monthly.png", dpi=120)
    plt.close(fig)
    print(f"wrote market figures to {OUT}")


if __name__ == "__main__":
    import argparse
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
