"""Singapore maps of previous-quarter mean rent by nearest facility.

June and December of each year get three maps: nearest mall, nearest primary
school, and nearest open trunk MRT station. Every land cell takes the colour
of its single nearest facility, so each facility's service area is one colour.
The colour is the same smoothed mean used in the feature columns.
"""

import json
import math
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath
from PIL import Image

from src.features.group_means import (
    _month_start,
    _smoothed,
    _window_totals,
    smoothed_group_month_table,
)
from src.features.keys import norm_text
from src.features.spatial import TRUNK_OPEN_PREFIXES, haversine_km

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "Data" / "auxiliary_features" / "train_features.csv"
RENTALS = ROOT / "Data" / "train.csv"
AUX = ROOT / "Data" / "auxiliary"
OUTLINE = ROOT / "Data" / "external" / "singapore_boundary.geojson"
TILES = ROOT / "Data" / "external" / "tiles" / "osm"
OUT = ROOT / "outputs" / "figures" / "maps"
ZOOM = 12

LON = np.linspace(103.60, 104.10, 280)
LAT = np.linspace(1.20, 1.49, 190)

LANDMARKS = (
    ("Raffles Place", 103.851389, 1.283889),
    ("Orchard", 103.831389, 1.303056),
    ("Jurong East", 103.742222, 1.333333),
    ("Woodlands", 103.786483, 1.437094),
    ("Tampines", 103.943889, 1.355),
    ("Changi Airport", 103.988333, 1.3575),
    ("HarbourFront", 103.822222, 1.265278),
    ("Punggol", 103.9025, 1.405278),
)


def _rings():
    geo = json.loads(OUTLINE.read_text(encoding="utf-8"))
    geometry = geo["features"][0]["geometry"]
    polygons = geometry["coordinates"] if geometry["type"] == "MultiPolygon" else [geometry["coordinates"]]
    return [polygon[0] for polygon in polygons]


def _land_mask() -> np.ndarray:
    lon, lat = np.meshgrid(LON, LAT)
    points = np.column_stack([lon.ravel(), lat.ravel()])
    inside = np.zeros(len(points), dtype=bool)
    for ring in _rings():
        inside |= MplPath(ring).contains_points(points)
    return inside.reshape(lat.shape)


def _tile_range():
    def tile_xy(lon, lat):
        n = 2 ** ZOOM
        x = (lon + 180.0) / 360.0 * n
        lat_rad = math.radians(lat)
        y = (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n
        return x, y

    x0, y_north = tile_xy(LON[0], LAT[-1])
    x1, y_south = tile_xy(LON[-1], LAT[0])
    return math.floor(x0), math.floor(x1), math.floor(y_north), math.floor(y_south)


def _tile_bounds(x: int, y: int):
    n = 2 ** ZOOM

    def latitude(row: float) -> float:
        return math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * row / n))))

    west = x / n * 360.0 - 180.0
    east = (x + 1) / n * 360.0 - 180.0
    north = latitude(y)
    south = latitude(y + 1)
    return west, south, east, north


def _basemap():
    x0, x1, y0, y1 = _tile_range()
    tiles = []
    for y in range(y0, y1 + 1):
        row = []
        for x in range(x0, x1 + 1):
            path = TILES / f"{ZOOM}_{x}_{y}.png"
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                url = f"https://tile.openstreetmap.org/{ZOOM}/{x}/{y}.png"
                request = urllib.request.Request(
                    url,
                    headers={"User-Agent": "CS5228-HDB-rent-maps/1.0 (academic project)"},
                )
                path.write_bytes(urllib.request.urlopen(request, timeout=30).read())
            row.append(Image.open(path).convert("RGB"))
        tiles.append(row)
    width, height = tiles[0][0].size
    mosaic = Image.new("RGB", (width * len(tiles[0]), height * len(tiles)))
    for row_i, row in enumerate(tiles):
        for col_i, tile in enumerate(row):
            mosaic.paste(tile, (col_i * width, row_i * height))
    west, _, _, north = _tile_bounds(x0, y0)
    _, south, east, _ = _tile_bounds(x1, y1)
    return np.asarray(mosaic), (west, east, south, north)


def _draw_coast(ax) -> None:
    for ring in _rings():
        xs = [point[0] for point in ring]
        ys = [point[1] for point in ring]
        ax.plot(xs, ys, color="#4a4a4a", linewidth=0.4, zorder=3)


def _history() -> pd.DataFrame:
    features = pd.read_csv(
        TRAIN,
        usecols=[
            "nearest_mall_name",
            "nearest_open_mrt_code",
            "nearest_primary_name",
        ],
    )
    rentals = pd.read_csv(RENTALS, usecols=["RENT_APPROVAL_DATE", "MONTHLY_RENT"])
    if len(rentals) != len(features):
        raise ValueError("train.csv and train feature rows are not aligned")
    features.insert(0, "RENT_APPROVAL_DATE", rentals["RENT_APPROVAL_DATE"].to_numpy())
    features.insert(1, "MONTHLY_RENT", rentals["MONTHLY_RENT"].to_numpy())
    return features


def _target_months(history: pd.DataFrame) -> list[str]:
    present = set(_month_start(history["RENT_APPROVAL_DATE"]).dt.strftime("%Y-%m"))
    chosen = []
    for year in range(2021, 2027):
        for month in (6, 12):
            label = f"{year}-{month:02d}"
            period = pd.Period(label, freq="M")
            window = {(period - lag).strftime("%Y-%m") for lag in (1, 2, 3)}
            if window & present:
                chosen.append(label)
    return chosen


def _global_for(history: pd.DataFrame, month: str) -> float:
    past = history.copy()
    past["month"] = _month_start(past["RENT_APPROVAL_DATE"])
    query = pd.Series([pd.Timestamp(month)])
    total_sum, total_n = _window_totals(query, None, past, None)
    value = _smoothed(np.zeros(1), np.zeros(1), total_sum, total_n)
    return float(value[0])


def _facilities(kind: str) -> pd.DataFrame:
    if kind == "mall":
        frame = pd.read_csv(AUX / "sg-shopping-malls.csv")
        frame["key"] = norm_text(frame["NAME"])
        return frame[["key", "LATITUDE", "LONGITUDE"]]
    if kind == "school":
        frame = pd.read_csv(AUX / "sg-schools.csv")
        frame = frame.loc[frame["MAINLEVEL_CODE"].astype("string").eq("primary")].copy()
        frame["key"] = norm_text(frame["NAME"])
        return frame[["key", "LATITUDE", "LONGITUDE"]]
    frame = pd.read_csv(AUX / "sg-mrt-stations.csv")
    prefix = frame["CODE"].astype("string").str.extract(r"^([A-Za-z]+)", expand=False).str.upper()
    status = frame["STATUS"].astype("string").str.strip().str.lower()
    frame = frame.loc[status.eq("open") & prefix.isin(TRUNK_OPEN_PREFIXES)].copy()
    frame["key"] = frame["CODE"].astype("string")
    return frame[["key", "LATITUDE", "LONGITUDE"]]


def _paint(facilities: pd.DataFrame, colour: np.ndarray, land: np.ndarray) -> np.ndarray:
    fac_lat = facilities["LATITUDE"].to_numpy(dtype=float)
    fac_lon = facilities["LONGITUDE"].to_numpy(dtype=float)
    lon, lat = np.meshgrid(LON, LAT)
    distance = haversine_km(
        lat[:, :, None],
        lon[:, :, None],
        fac_lat[None, None, :],
        fac_lon[None, None, :],
    )
    nearest = distance.argmin(axis=2)
    painted = np.asarray(colour, dtype=float)[nearest]
    return np.ma.masked_where(~land, painted)


def main() -> None:
    history = _history()
    months = _target_months(history)
    land = _land_mask()
    basemap, extent = _basemap()
    kinds = (
        ("mall", "nearest_mall_name", "Nearest mall"),
        ("school", "nearest_primary_name", "Nearest primary school"),
        ("mrt", "nearest_open_mrt_code", "Nearest open MRT station"),
    )
    OUT.mkdir(parents=True, exist_ok=True)
    for kind, column, title in kinds:
        facilities = _facilities(kind)
        table = smoothed_group_month_table(history, history[column], pd.Series(months))
        grids = []
        labels = []
        for month in months:
            fallback = _global_for(history, month)
            if np.isnan(fallback):
                continue
            looked_up = table[month] if month in table.columns else pd.Series(dtype=float)
            values = facilities["key"].map(looked_up).fillna(fallback).to_numpy(dtype=float)
            grids.append(_paint(facilities, values, land))
            labels.append(month)
        for grid, month in zip(grids, labels):
            finite = grid.compressed()
            vmin = float(np.min(finite))
            vmax = float(np.max(finite))
            if vmin == vmax:
                vmax = vmin + 1.0
            fig, ax = plt.subplots(figsize=(7.4, 5.4))
            ax.imshow(basemap, extent=extent, origin="upper", zorder=0)
            panel = ax.pcolormesh(
                LON,
                LAT,
                grid,
                cmap="YlOrRd",
                shading="nearest",
                vmin=vmin,
                vmax=vmax,
                alpha=0.58,
                zorder=2,
            )
            _draw_coast(ax)
            for name, lon, lat in LANDMARKS:
                ax.scatter(lon, lat, s=14, color="black", zorder=4)
                ax.annotate(
                    name,
                    (lon, lat),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=7,
                    color="black",
                    zorder=5,
                    bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 0.6},
                )
            ax.set_aspect("equal")
            ax.set_xlim(LON[0], LON[-1])
            ax.set_ylim(LAT[0], LAT[-1])
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.set_title(f"{title} mean rent, {month}")
            bar = fig.colorbar(panel, ax=ax, fraction=0.04, pad=0.02)
            bar.set_label("SGD")
            fig.tight_layout()
            path = OUT / f"map_{kind}_{month}.png"
            fig.savefig(path, dpi=140)
            plt.close(fig)
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
