"""Singapore map of the May 2024 town mean of rent per square metre.

    python -m src.viz.plot_town_unit_may2024

Each land cell takes the colour of the town of the nearest HDB block.
The colour is the unweighted mean of MONTHLY_RENT / FLOOR_AREA_SQM for
approvals in 2024-05. It is not a model forecast.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.features.spatial import haversine_km
from src.viz.plot_group_mean_maps import LAT, LON, OUT, ROOT, _basemap, _draw_coast, _land_mask

MONTH = "2024-05"
CHUNK = 400


def _town_means() -> pd.Series:
    rentals = pd.read_csv(
        ROOT / "Data" / "train.csv",
        usecols=["RENT_APPROVAL_DATE", "TOWN", "MONTHLY_RENT", "FLOOR_AREA_SQM"],
    )
    month = rentals["RENT_APPROVAL_DATE"].astype("string").str.slice(0, 7)
    towns = sorted(rentals["TOWN"].astype(str).unique())
    may = rentals.loc[month.eq(MONTH)].copy()
    present = set(may["TOWN"].astype(str))
    missing = [town for town in towns if town not in present]
    if missing:
        raise SystemExit(f"{MONTH} has no rows for {missing}")
    area = may["FLOOR_AREA_SQM"].to_numpy(dtype=np.float64)
    if np.any(area <= 0):
        raise SystemExit(f"{MONTH} has a non-positive floor area")
    may["unit"] = may["MONTHLY_RENT"].to_numpy(dtype=np.float64) / area
    return may.groupby(may["TOWN"].astype(str))["unit"].mean()


def _blocks(means: pd.Series) -> pd.DataFrame:
    blocks = pd.read_csv(
        ROOT / "Data" / "auxiliary" / "sg-hdb-block.csv",
        usecols=["TOWN", "LATITUDE", "LONGITUDE"],
    )
    blocks["TOWN"] = blocks["TOWN"].astype(str)
    blocks = blocks.loc[blocks["TOWN"].isin(means.index)].copy()
    blocks["LATITUDE"] = pd.to_numeric(blocks["LATITUDE"], errors="coerce")
    blocks["LONGITUDE"] = pd.to_numeric(blocks["LONGITUDE"], errors="coerce")
    blocks = blocks.dropna(subset=["LATITUDE", "LONGITUDE"])
    if blocks.empty:
        raise SystemExit("no block coordinates for the May towns")
    return blocks.reset_index(drop=True)


def _owner(blocks: pd.DataFrame, land: np.ndarray) -> np.ndarray:
    lat_grid, lon_grid = np.meshgrid(LAT, LON, indexing="ij")
    best = np.full(lat_grid.shape, np.inf)
    owner = np.full(lat_grid.shape, -1, dtype=np.int32)
    blat = blocks["LATITUDE"].to_numpy(dtype=np.float64)
    blon = blocks["LONGITUDE"].to_numpy(dtype=np.float64)
    for start in range(0, len(blocks), CHUNK):
        stop = min(start + CHUNK, len(blocks))
        distance = haversine_km(
            lat_grid[:, :, None],
            lon_grid[:, :, None],
            blat[None, None, start:stop],
            blon[None, None, start:stop],
        )
        local = distance.argmin(axis=2)
        local_distance = distance.min(axis=2)
        better = local_distance < best
        best = np.where(better, local_distance, best)
        owner = np.where(better, local + start, owner)
    return np.ma.masked_where(~land, owner)


def main() -> None:
    means = _town_means()
    blocks = _blocks(means)
    colours = blocks["TOWN"].map(means).to_numpy(dtype=np.float64)
    land = _land_mask()
    owner = _owner(blocks, land)
    painted = np.ma.masked_where(owner.mask, colours[owner.filled(0)])
    basemap, extent = _basemap()
    finite = painted.compressed()
    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    ax.imshow(basemap, extent=extent, origin="upper", zorder=0)
    panel = ax.pcolormesh(
        LON,
        LAT,
        painted,
        cmap="YlOrRd",
        shading="nearest",
        vmin=vmin,
        vmax=vmax,
        alpha=0.58,
        zorder=2,
    )
    _draw_coast(ax)
    centroids = blocks.groupby("TOWN")[["LONGITUDE", "LATITUDE"]].mean()
    for town, row in centroids.iterrows():
        ax.annotate(
            str(town).title(),
            (row["LONGITUDE"], row["LATITUDE"]),
            fontsize=6,
            ha="center",
            va="center",
            color="black",
            zorder=5,
            bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none", "pad": 0.4},
        )
    ax.set_aspect("equal")
    ax.set_xlim(LON[0], LON[-1])
    ax.set_ylim(LAT[0], LAT[-1])
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Town mean unit rent, {MONTH}")
    bar = fig.colorbar(panel, ax=ax, fraction=0.04, pad=0.02)
    bar.set_label("SGD per square metre")
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"town_unit_mean_{MONTH}.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(
        f"wrote {path.relative_to(ROOT).as_posix()} "
        f"towns {len(means)} range {vmin:.2f} to {vmax:.2f}"
    )


if __name__ == "__main__":
    main()
