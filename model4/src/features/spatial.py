"""Distances and radius counts from block coordinates to auxiliary points."""

import numpy as np
import pandas as pd

from src.features.keys import block_key, norm_text

EARTH_RADIUS_KM = 6371.0
TIE_EPS_KM = 1e-6
CBD_LAT = 1.283889
CBD_LON = 103.851389
TRUNK_OPEN_PREFIXES = {"NS", "NE", "CC", "DT", "TE", "EW", "CG"}
FUTURE_PREFIXES = {"CR", "JE", "JS", "JW", "CP"}
LINE_SPECS = (
    ("ns", {"NS"}),
    ("ew", {"EW", "CG"}),
    ("ne", {"NE"}),
    ("cc", {"CC"}),
    ("dt", {"DT"}),
    ("te", {"TE"}),
)


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle kilometres. Inputs are decimal degrees and may broadcast."""
    lat1 = np.radians(np.asarray(lat1, dtype=np.float64))
    lon1 = np.radians(np.asarray(lon1, dtype=np.float64))
    lat2 = np.radians(np.asarray(lat2, dtype=np.float64))
    lon2 = np.radians(np.asarray(lon2, dtype=np.float64))
    dphi = lat2 - lat1
    dlmb = lon2 - lon1
    a = np.sin(dphi / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlmb / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _pairwise(lat, lon, other_lat, other_lon, chunk: int = 500):
    other_lat = np.asarray(other_lat, dtype=np.float64)
    other_lon = np.asarray(other_lon, dtype=np.float64)
    for start in range(0, len(lat), chunk):
        end = min(start + chunk, len(lat))
        dist = haversine_km(
            lat[start:end, None],
            lon[start:end, None],
            other_lat[None, :],
            other_lon[None, :],
        )
        yield start, end, dist


def nearest(lat, lon, other_lat, other_lon, tie_keys: np.ndarray):
    """Minimum Haversine. Ties within 1e-6 km use the lexicographically smallest key."""
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    tie_keys = np.asarray(tie_keys)
    if len(other_lat) == 0:
        empty = np.full(len(lat), np.nan)
        return empty, np.full(len(lat), -1, dtype=int)
    best_dist = np.empty(len(lat), dtype=np.float64)
    best_index = np.empty(len(lat), dtype=int)
    for start, end, dist in _pairwise(lat, lon, other_lat, other_lon):
        min_dist = dist.min(axis=1)
        for row_i, row in enumerate(dist):
            candidates = np.flatnonzero(row <= min_dist[row_i] + TIE_EPS_KM)
            order = np.argsort(tie_keys[candidates], kind="mergesort")
            chosen = candidates[order[0]]
            best_dist[start + row_i] = row[chosen]
            best_index[start + row_i] = chosen
    return best_dist, best_index


def count_within(lat, lon, other_lat, other_lon, radius_km: float, exclude_self: bool = False) -> np.ndarray:
    """Count other points at Haversine distance <= radius_km. Zero is a real zero."""
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    counts = np.zeros(len(lat), dtype=np.int64)
    if len(other_lat) == 0:
        return counts
    for start, end, dist in _pairwise(lat, lon, other_lat, other_lon):
        inside = dist <= radius_km
        if exclude_self:
            inside = inside & (dist > 0)
        counts[start:end] = inside.sum(axis=1)
    return counts


def _line_prefix(codes: pd.Series) -> pd.Series:
    return codes.astype("string").str.extract(r"^([A-Za-z]+)", expand=False).str.upper()


def _prepare_mrt(mrt: pd.DataFrame) -> pd.DataFrame:
    stations = mrt.copy()
    stations["line_prefix"] = _line_prefix(stations["CODE"])
    stations["name_key"] = norm_text(stations["NAME"])
    stations["code"] = stations["CODE"].astype("string")
    stations["status"] = stations["STATUS"].astype("string").str.strip().str.lower()
    return stations


def _subset(stations: pd.DataFrame, prefixes: set[str], status: str) -> pd.DataFrame:
    return stations[stations["line_prefix"].isin(prefixes) & stations["status"].eq(status)]


def block_spatial_features(blocks: pd.DataFrame, mrt: pd.DataFrame, schools: pd.DataFrame, malls: pd.DataFrame) -> pd.DataFrame:
    """One row per block. Callers join the result back onto rental rows."""
    origin = pd.DataFrame(
        {
            "block_key": block_key(blocks["BLOCK"], blocks["STREET"]),
            "latitude": pd.to_numeric(blocks["LATITUDE"], errors="coerce"),
            "longitude": pd.to_numeric(blocks["LONGITUDE"], errors="coerce"),
            "town": norm_text(blocks["TOWN"]),
        }
    )
    lat = origin["latitude"].to_numpy()
    lon = origin["longitude"].to_numpy()
    stations = _prepare_mrt(mrt)
    trunk = _subset(stations, TRUNK_OPEN_PREFIXES, "open")
    code_counts = stations.groupby("name_key")["code"].nunique()
    interchange_names = set(code_counts[code_counts >= 2].index)

    dist, index = nearest(
        lat, lon, trunk["LATITUDE"].to_numpy(), trunk["LONGITUDE"].to_numpy(), trunk["code"].to_numpy()
    )
    chosen = trunk.iloc[index].reset_index(drop=True)
    prefix = chosen["line_prefix"].astype("string")
    line = prefix.mask(prefix.isin(["EW", "CG"]), "EW")
    out = origin[["block_key"]].copy()
    out["dist_nearest_open_mrt_km"] = dist
    out["nearest_open_mrt_code"] = chosen["code"].to_numpy()
    out["nearest_open_mrt_name"] = chosen["name_key"].to_numpy()
    out["nearest_open_mrt_line"] = line.to_numpy()
    out["nearest_open_mrt_is_interchange"] = (
        chosen["name_key"].isin(interchange_names).astype(np.int64).to_numpy()
    )
    out["log_dist_nearest_open_mrt_km"] = np.log(dist + 0.001)

    for slug, prefixes in LINE_SPECS:
        subset = _subset(stations, prefixes, "open")
        line_dist, line_index = nearest(
            lat,
            lon,
            subset["LATITUDE"].to_numpy(),
            subset["LONGITUDE"].to_numpy(),
            subset["code"].to_numpy(),
        )
        picked = subset.iloc[line_index].reset_index(drop=True)
        out[f"nearest_{slug}_station_code"] = picked["code"].to_numpy()
        out[f"nearest_{slug}_station_name"] = picked["name_key"].to_numpy()
        out[f"dist_nearest_{slug}_km"] = line_dist

    trunk_lat = trunk["LATITUDE"].to_numpy()
    trunk_lon = trunk["LONGITUDE"].to_numpy()
    out["n_open_mrt_within_0_5km"] = count_within(lat, lon, trunk_lat, trunk_lon, 0.5)
    out["n_open_mrt_within_1km"] = count_within(lat, lon, trunk_lat, trunk_lon, 1.0)
    out["n_open_mrt_within_2km"] = count_within(lat, lon, trunk_lat, trunk_lon, 2.0)

    planned = stations[stations["status"].eq("planned")]
    planned_dist, _ = nearest(
        lat,
        lon,
        planned["LATITUDE"].to_numpy(),
        planned["LONGITUDE"].to_numpy(),
        planned["code"].to_numpy(),
    )
    out["dist_nearest_planned_mrt_km"] = planned_dist
    out["n_planned_mrt_within_1km"] = count_within(
        lat, lon, planned["LATITUDE"].to_numpy(), planned["LONGITUDE"].to_numpy(), 1.0
    )
    out["n_planned_mrt_within_2km"] = count_within(
        lat, lon, planned["LATITUDE"].to_numpy(), planned["LONGITUDE"].to_numpy(), 2.0
    )

    future_open = _subset(stations, FUTURE_PREFIXES, "open")
    future_planned = _subset(stations, FUTURE_PREFIXES, "planned")
    future_open_dist, _ = nearest(
        lat,
        lon,
        future_open["LATITUDE"].to_numpy(),
        future_open["LONGITUDE"].to_numpy(),
        future_open["code"].to_numpy(),
    )
    future_planned_dist, _ = nearest(
        lat,
        lon,
        future_planned["LATITUDE"].to_numpy(),
        future_planned["LONGITUDE"].to_numpy(),
        future_planned["code"].to_numpy(),
    )
    out["dist_nearest_future_line_open_km"] = future_open_dist
    out["dist_nearest_future_line_planned_km"] = future_planned_dist

    school = schools.copy()
    school["name_key"] = norm_text(school["NAME"])
    school["level"] = school["MAINLEVEL_CODE"].astype("string")
    primary = school[school["level"].eq("primary")]
    secondary = school[school["level"].isin(["secondary (s1-s5)", "secondary (s1-s4)"])]

    def _school_nearest(frame: pd.DataFrame):
        return nearest(
            lat,
            lon,
            frame["LATITUDE"].to_numpy(),
            frame["LONGITUDE"].to_numpy(),
            frame["name_key"].to_numpy(),
        )

    primary_dist, primary_index = _school_nearest(primary)
    secondary_dist, secondary_index = _school_nearest(secondary)
    any_dist, any_index = _school_nearest(school)
    out["dist_nearest_primary_km"] = primary_dist
    out["nearest_primary_name"] = primary.iloc[primary_index]["name_key"].to_numpy()
    out["n_primary_within_1km"] = count_within(
        lat, lon, primary["LATITUDE"].to_numpy(), primary["LONGITUDE"].to_numpy(), 1.0
    )
    out["n_primary_within_2km"] = count_within(
        lat, lon, primary["LATITUDE"].to_numpy(), primary["LONGITUDE"].to_numpy(), 2.0
    )
    out["dist_nearest_secondary_km"] = secondary_dist
    out["nearest_secondary_name"] = secondary.iloc[secondary_index]["name_key"].to_numpy()
    out["n_secondary_within_1km"] = count_within(
        lat, lon, secondary["LATITUDE"].to_numpy(), secondary["LONGITUDE"].to_numpy(), 1.0
    )
    out["dist_nearest_school_km"] = any_dist
    out["nearest_school_name"] = school.iloc[any_index]["name_key"].to_numpy()
    out["n_school_within_1km"] = count_within(
        lat, lon, school["LATITUDE"].to_numpy(), school["LONGITUDE"].to_numpy(), 1.0
    )

    mall = malls.copy()
    mall["name_key"] = norm_text(mall["NAME"])
    mall_dist, mall_index = nearest(
        lat, lon, mall["LATITUDE"].to_numpy(), mall["LONGITUDE"].to_numpy(), mall["name_key"].to_numpy()
    )
    out["dist_nearest_mall_km"] = mall_dist
    out["nearest_mall_name"] = mall.iloc[mall_index]["name_key"].to_numpy()
    out["n_mall_within_1km"] = count_within(
        lat, lon, mall["LATITUDE"].to_numpy(), mall["LONGITUDE"].to_numpy(), 1.0
    )
    out["n_mall_within_2km"] = count_within(
        lat, lon, mall["LATITUDE"].to_numpy(), mall["LONGITUDE"].to_numpy(), 2.0
    )

    out["n_hdb_blocks_within_500m"] = count_within(lat, lon, lat, lon, 0.5, exclude_self=True)

    centroid = origin.groupby("town")[["latitude", "longitude"]].mean()
    town_lat = origin["town"].map(centroid["latitude"]).to_numpy()
    town_lon = origin["town"].map(centroid["longitude"]).to_numpy()
    out["dist_to_town_centroid_km"] = haversine_km(lat, lon, town_lat, town_lon)
    out["dist_to_cbd_km"] = haversine_km(lat, lon, CBD_LAT, CBD_LON)
    return out
