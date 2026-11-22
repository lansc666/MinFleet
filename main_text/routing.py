"""Population-weighted detour ratios using travel times and a fixed speed."""

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

SPEED_MS = 50 / 3.6
MIN_DIST = 50.0
POP_POWER = 0.6
DIST_POWER = 1.2


def haversine_vectorized(coords_array):
    """Return pairwise distances in metres for (latitude, longitude) nodes."""
    lat = np.radians(coords_array[:, 0])
    lon = np.radians(coords_array[:, 1])
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return 6371000.0 * c


def match_population_to_nodes(nodes_dict, shp_path):
    """Return node-ordered populations from nearest spatial matches, or None."""
    num_nodes = len(nodes_dict)
    # Coordinates are (latitude, longitude), ordered by integer node ID.
    node_coords = np.array([nodes_dict[i] for i in range(num_nodes)])

    try:
        gdf = gpd.read_file(shp_path)
        if "value" not in gdf.columns:
            return None

        shp_coords = np.array(list(zip(gdf.geometry.y, gdf.geometry.x)))
        shp_values = np.nan_to_num(gdf["value"].values, nan=0.0)
    except Exception:
        return None

    tree = cKDTree(shp_coords)
    distances, indices = tree.query(node_coords, k=1)

    return shp_values[indices]


def process_city(city, latlon_path, matrix_path, shp_path):
    """Return unweighted and population-weighted detour ratios, or None."""
    try:
        nodes = pd.read_pickle(latlon_path)
        nodes_clean = {}
        for k, v in nodes.items():
            try:
                nodes_clean[k] = (float(v[0]), float(v[1]))
            except:
                # Use zero coordinates when numeric conversion fails.
                nodes_clean[k] = (0.0, 0.0)

        num_nodes = len(nodes_clean)

        raw_mat = pd.read_pickle(matrix_path)

        if isinstance(raw_mat, pd.DataFrame):
            raw_mat = raw_mat.values

        try:
            time_matrix = np.array(raw_mat, dtype=float)
        except:
            # Coerce nonnumeric entries to NaN if direct conversion fails.
            time_matrix = (
                pd.DataFrame(raw_mat).apply(pd.to_numeric, errors="coerce").values
            )

        if time_matrix.shape[0] != num_nodes:
            return None

        population_array = match_population_to_nodes(nodes_clean, shp_path)
        if population_array is None:
            return None

        coords_arr = np.array([nodes_clean[i] for i in range(num_nodes)])

        dist_straight = haversine_vectorized(coords_arr)

        # Convert durations in seconds to distance equivalents at 50 km/h.
        dist_road = time_matrix * SPEED_MS

        denom_mdr = dist_straight.copy()
        # Exclude near-zero straight-line distances from the ratio.
        denom_mdr[denom_mdr < 1e-6] = np.nan
        ratio_matrix = dist_road / denom_mdr

        pop_i = population_array[:, None]
        pop_j = population_array[None, :]

        # Weights are proportional to (P_i P_j)^0.6 / D_ij^1.2.
        pop_term = (pop_i**POP_POWER) * (pop_j**POP_POWER)

        dist_term = dist_straight**DIST_POWER
        # Floor the distance-power term at 1.
        dist_term[dist_term < 1.0] = 1.0

        weight_matrix = pop_term / dist_term

        np.fill_diagonal(weight_matrix, 0)

        # Retain pairs beyond 50 m with 0.5 < ratio < 10 and positive weights.
        mask = (
            (~np.isnan(ratio_matrix))
            & (dist_straight > MIN_DIST)
            & (ratio_matrix > 0.5)
            & (ratio_matrix < 10.0)
            & (weight_matrix > 0)
            & (~np.isnan(weight_matrix))
        )

        valid_ratios = ratio_matrix[mask]
        valid_weights = weight_matrix[mask]

        if valid_ratios.size == 0:
            return None

        simple_mdr = np.mean(valid_ratios)
        weighted_mdr = np.average(valid_ratios, weights=valid_weights)

        return simple_mdr, weighted_mdr

    except Exception as e:
        print(f"Could not compute detour ratios for {city}: {e}")
        return None
