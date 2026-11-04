"""Heatmap of each town's saved per-square-metre Gaussian surface.

    python -m src.viz.plot_gaussian_towns

The mixture files are not refit. The surface is the conditional residual
at that town's mean flat-type rank and mean flat-model rank. A cell more
than 400m from a fit-month flat is left blank. December 2024 flats are
drawn on top, coloured by their unit-rent gap from that town's true
December mean.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import MaxNLocator
from sklearn.mixture import GaussianMixture

from mean_residual_model.reginal_mean_model.rnn.model import predict_saved_rnn3
from mean_residual_model.run import _check_months
from src.features.spatial import haversine_km
from src.pipeline.data import load_joined, time_split

ROOT = Path(__file__).resolve().parents[2]
STAGE = ROOT / "mean_residual_model" / "multi_stage_model(lstm+catboost+mlp)"
OUT = Path(__file__).resolve().parent / "gaussian_towns"
MASK_KM = 0.4
STEP_DEG = 0.0007
DECEMBER = "2024-12"


def main() -> None:
    frame = load_joined(
        "train",
        ["RENT_APPROVAL_DATE", "TOWN", "latitude", "longitude", "floor_area_sqm"],
    )
    split = time_split(frame, 3, None)
    _check_months(split)
    fit = split.fit
    month = frame["RENT_APPROVAL_DATE"].astype("string").str.slice(0, 7)
    december = frame.loc[month == DECEMBER].copy()
    december["mu"] = predict_saved_rnn3(december)
    december["unit"] = december["MONTHLY_RENT"].to_numpy(dtype=np.float64) / december["floor_area_sqm"].to_numpy(
        dtype=np.float64
    )
    towns = sorted(fit["TOWN"].astype(str).unique())
    surfaces = []
    point_gaps = []
    for town in towns:
        part = fit.loc[fit["TOWN"].astype(str) == town]
        item = _surface(town, part)
        dec = december.loc[december["TOWN"].astype(str) == town]
        if len(dec) == 0:
            item["december"] = None
        else:
            unit = dec["unit"].to_numpy(dtype=np.float64)
            true_mean = float(unit.mean())
            gaps = unit - true_mean
            item["december"] = {
                "gaps": gaps,
                "lat": dec["latitude"].astype(float).to_numpy(),
                "lon": dec["longitude"].astype(float).to_numpy(),
                "pred_mean": float(dec["mu"].mean()),
                "true_mean": true_mean,
            }
            point_gaps.append(gaps)
        surfaces.append(item)
    pieces = [item["values"].compressed() for item in surfaces]
    if point_gaps:
        pieces.append(np.concatenate(point_gaps))
    limit = float(np.percentile(np.abs(np.concatenate(pieces)), 98))
    limit = max(limit, 0.05)
    OUT.mkdir(parents=True, exist_ok=True)
    for item in surfaces:
        _draw(item, limit)
        print(f"wrote {item['town']} k {item['k']}")
    print(f"wrote {len(surfaces)} heatmaps to {OUT}")


def _surface(town: str, part: pd.DataFrame) -> dict:
    path = STAGE / "gmm" / f"{town_slug(town)}.joblib"
    mixture = joblib.load(path)
    if mixture.get("kind") != "conditional_location_flat_per_sqm":
        raise SystemExit(f"{path.name} is not a per-square-metre location mixture")
    if mixture["gmm"].means_.shape[1] != 5:
        raise SystemExit(f"{path.name} does not have location, flat ranks, and a residual")
    lat = part["latitude"].astype(float).to_numpy()
    lon = part["longitude"].astype(float).to_numpy()
    lat_center = float(lat.mean())
    lon_center = float(lon.mean())
    lat_spread = float(lat.std()) or 1.0
    lon_spread = float(lon.std()) or 1.0
    places = part.drop_duplicates(["latitude", "longitude"])
    place_lat = places["latitude"].astype(float).to_numpy()
    place_lon = places["longitude"].astype(float).to_numpy()
    pad_lat = MASK_KM / 111.32
    pad_lon = MASK_KM / (111.32 * np.cos(np.radians(lat_center)))
    grid_lat = np.arange(lat.min() - pad_lat, lat.max() + pad_lat + STEP_DEG, STEP_DEG)
    grid_lon = np.arange(lon.min() - pad_lon, lon.max() + pad_lon + STEP_DEG, STEP_DEG)
    mesh_lat, mesh_lon = np.meshgrid(grid_lat, grid_lon, indexing="ij")
    scaled_lat = (mesh_lat.ravel() - lat_center) / lat_spread
    scaled_lon = (mesh_lon.ravel() - lon_center) / lon_spread
    mean_rank = np.zeros(scaled_lat.shape, dtype=np.float64)
    scaled = np.column_stack([scaled_lat, scaled_lon, mean_rank, mean_rank])
    height = _conditional(mixture["gmm"], scaled).reshape(mesh_lat.shape)
    nearest = _nearest_km(mesh_lat.ravel(), mesh_lon.ravel(), place_lat, place_lon)
    values = np.ma.masked_where(nearest.reshape(mesh_lat.shape) > MASK_KM, height)
    if values.count() == 0:
        raise SystemExit(f"{town} heatmap has no cells inside {MASK_KM} km")
    return {
        "k": int(mixture["n_components"]),
        "lat": mesh_lat,
        "lon": mesh_lon,
        "town": town,
        "values": values,
    }


def _nearest_km(lat: np.ndarray, lon: np.ndarray, place_lat: np.ndarray, place_lon: np.ndarray) -> np.ndarray:
    nearest = np.full(len(lat), np.inf)
    for start in range(0, len(place_lat), 250):
        stop = start + 250
        distance = haversine_km(lat[:, None], lon[:, None], place_lat[start:stop], place_lon[start:stop])
        nearest = np.minimum(nearest, distance.min(axis=1))
    return nearest


def _draw(item: dict, limit: float) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    panel = ax.pcolormesh(
        item["lon"],
        item["lat"],
        item["values"],
        cmap="RdBu_r",
        norm=norm,
        shading="auto",
    )
    december = item["december"]
    if december is None:
        note = f"{DECEMBER}: no flats"
    else:
        ax.scatter(
            december["lon"],
            december["lat"],
            c=december["gaps"],
            cmap="RdBu_r",
            norm=norm,
            s=18,
            edgecolors="black",
            linewidths=0.3,
            zorder=3,
        )
        note = (
            f"{DECEMBER} true mean {december['true_mean']:.2f} SGD/sqm\n"
            f"RNN predicted region mean {december['pred_mean']:.2f} SGD/sqm\n"
            "points: unit rent minus true mean\n"
            "surface: predicted gap at mean flat"
        )
    ax.text(
        0.02,
        0.98,
        note,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.88, "edgecolor": "none"},
    )
    ax.set_aspect(1.0 / np.cos(np.radians(float(np.mean(item["lat"])))))
    ax.ticklabel_format(useOffset=False)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"{item['town']}  ({item['k']} components)")
    fig.colorbar(panel, ax=ax, label="Gap (SGD / sqm / month)")
    fig.tight_layout()
    path = OUT / f"{town_slug(item['town'])}.png"
    temporary = path.with_suffix(".tmp.png")
    fig.savefig(temporary, dpi=140)
    plt.close(fig)
    temporary.replace(path)


def town_slug(town: str) -> str:
    text = town.strip().lower().replace("/", "_").replace(" ", "_")
    return "".join(char if char.isalnum() or char == "_" else "_" for char in text)


def _conditional(gmm: GaussianMixture, coords: np.ndarray) -> np.ndarray:
    """Conditional residual mean given scaled location and flat ranks."""
    width = coords.shape[1]
    log_terms = []
    heights = []
    for index in range(gmm.n_components):
        mean = gmm.means_[index]
        cov = gmm.covariances_[index]
        location_cov = cov[:width, :width] + 1e-8 * np.eye(width)
        solved = np.linalg.solve(location_cov, (coords - mean[:width]).T).T
        heights.append(mean[width] + solved @ cov[width, :width])
        sign, logdet = np.linalg.slogdet(location_cov)
        quad = np.sum((coords - mean[:width]) * solved, axis=1)
        log_density = -0.5 * (width * np.log(2.0 * np.pi) + logdet + quad)
        if sign <= 0:
            log_density = np.full(len(coords), -1e30)
        log_terms.append(np.log(float(gmm.weights_[index]) + 1e-300) + log_density)
    stacked = np.column_stack(log_terms)
    mix = np.exp(stacked - stacked.max(axis=1, keepdims=True))
    mix_sum = mix.sum(axis=1)
    responsibility = mix / mix_sum[:, None]
    return np.sum(responsibility * np.column_stack(heights), axis=1)


if __name__ == "__main__":
    main()
