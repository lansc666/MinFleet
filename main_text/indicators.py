"""Compute nine urban structural indicators from city grids and travel times."""

from pathlib import Path
import argparse
import json
import sys
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.ndimage import label
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
from main_text.geometry import (
    raster,
    polygon_moments,
    exposed_perimeter,
    weighted_ei,
    cell_area_km2,
    patch_geometry,
)
from main_text.routing import process_city
from main_text import FEATURES
from utils.paths import DATA, make_output


def morphology(mask, left, top, px, py, epsg):
    r, c = np.where(mask)
    rr = np.column_stack([r, r + 1, r + 1, r])
    cc = np.column_stack([c, c, c + 1, c + 1])
    projection = Transformer.from_crs(4326, epsg, always_xy=True)
    x, y = projection.transform(left + cc * px, top - rr * py)
    vertices = np.stack([x, y], axis=-1)
    area, _ = polygon_moments(vertices)
    perimeter = exposed_perimeter(mask, vertices)
    ei, excluded, n8 = weighted_ei(mask)
    labs, n4 = label(mask)
    row_areas = cell_area_km2(top - np.arange(mask.shape[0]) * py, py, px)
    patch_areas = np.bincount(labs[mask], weights=row_areas[r], minlength=n4 + 1)[1:]
    division = float(1 - np.sum((patch_areas / patch_areas.sum()) ** 2))
    return dict(
        area_m2=area,
        perimeter_m=perimeter,
        EI=ei,
        LDI=division,
        patches_8=n8,
        patches_4=n4,
        EI_excluded_cells=excluded,
    )


def hotspot_count(projected):
    values = projected["value"].to_numpy(dtype=float)
    selected = values > np.quantile(values, 0.5)
    xy = np.column_stack([projected.geometry.x, projected.geometry.y])
    labels = (
        DBSCAN(eps=1500, min_samples=3).fit_predict(xy[selected])
        if selected.any()
        else np.array([], int)
    )
    return int(len(set(labels)) - (1 if -1 in labels else 0))


def normalized_travel_time(matrix, coords):
    """Return DNTT in min/km from all ordered non-self OD pairs."""
    # All ordered non-self OD pairs, without the detour-ratio filter.
    n = len(matrix)
    lat = np.radians(coords[:, 0])
    lon = np.radians(coords[:, 1])
    total_time, total_distance = 0.0, 0.0
    for start in range(0, n, 128):
        end = min(start + 128, n)
        origins = np.arange(start, end)
        z = (
            np.sin((lat[origins, None] - lat[None, :]) / 2) ** 2
            + np.cos(lat[origins, None])
            * np.cos(lat[None, :])
            * np.sin((lon[origins, None] - lon[None, :]) / 2) ** 2
        )
        z = np.clip(z, 0, 1)
        distance = 6371 * 2 * np.arctan2(np.sqrt(z), np.sqrt(1 - z))
        mask = origins[:, None] != np.arange(n)[None, :]
        if not (distance[mask] > 0).all():
            raise ValueError("Distinct grid nodes must have positive distances.")
        total_time += (matrix[start:end] / 60)[mask].sum()
        total_distance += distance[mask].sum()
    return float(total_time / total_distance)


def compute(country, roster, output):
    country, output = Path(country), Path(output)
    reference = pd.read_csv(roster, float_precision="round_trip")
    missing = sorted({"city", "province", "eta"} - set(reference))
    if missing:
        raise ValueError(f"The city table is missing columns: {missing}")
    if not reference.city.is_unique or reference[["city", "province", "eta"]].isna().any().any():
        raise ValueError("Each city must have a unique identifier, province and eta.")
    available = [
        (country / row.province / f"{row.city}_clip.tif").is_file()
        for row in reference.itertuples(index=False)
    ]
    skipped = reference.loc[np.logical_not(available), "city"].tolist()
    reference = reference.loc[available]
    if reference.empty:
        raise ValueError("No city has a clipped population raster.")
    if skipped:
        print(f"Skipping {len(skipped)} cities without clipped population rasters.", flush=True)
    rows, metadata = [], []
    for index, row in enumerate(reference.itertuples(index=False), 1):
        shp = country / row.province / f"{row.city}_center.shp"
        tif = shp.with_name(f"{row.city}_clip.tif")
        for path in [shp, tif]:
            if not path.is_file():
                raise FileNotFoundError(path)
        values, mask, left, top, px, py = raster(tif)
        points = gpd.read_file(shp)
        epsg = 32600 + int((points.geometry.x.iloc[0] + 180) // 6) + 1
        projected = points.to_crs(epsg=epsg)
        xy = np.column_stack([projected.geometry.x, projected.geometry.y])
        pop = points["value"].to_numpy(dtype=float)
        pr, pc = np.where(mask & (values > 0))
        tree = cKDTree(np.column_stack([points.geometry.x, points.geometry.y]))
        error, idx = tree.query(
            np.column_stack([left + (pc + 0.5) * px, top - (pr + 0.5) * py])
        )
        assert len(idx) == len(points) == len(np.unique(idx)) and error.max() < 1e-7
        assert np.allclose(values[pr, pc], pop[idx], atol=1e-6, rtol=1e-7)
        shape = morphology(mask, left, top, px, py, epsg)
        total_pop = float(values[mask].sum())
        assert np.isclose(total_pop, pop.sum(), rtol=1e-10)
        center = np.average(xy, axis=0, weights=pop)
        weighted_distance = float(
            np.average(np.linalg.norm(xy - center, axis=1), weights=pop)
        )
        route_path = country / "routes" / f"{row.city}_routes_con.pkl"
        node_path = country / "routes" / f"{row.city}_latlon.pkl"
        matrix = (
            pd.DataFrame(pd.read_pickle(route_path))
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=float)
        )
        assert (
            matrix.shape == (len(pop), len(pop))
            and np.isfinite(matrix).all()
            and (matrix >= 0).all()
        )
        nodes = pd.read_pickle(node_path)
        coords = np.array([nodes[j] for j in range(len(nodes))], dtype=float)
        error, idx = tree.query(coords[:, ::-1])
        assert error.max() < 1e-7 and len(np.unique(idx)) == len(pop)
        row_means = (matrix.sum(axis=1) - np.diag(matrix)) / (len(matrix) - 1)
        adi = float(np.std(row_means) / (row_means.mean() + 1e-6))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            detour = process_city(row.city, str(node_path), str(route_path), str(shp))
        if detour is None:
            raise ValueError(f"No valid detour ratio for {row.city}")
        geometry_row = pd.Series(
            dict(
                city=row.city,
                province=row.province,
                raster_path=str(tif.resolve()),
                raster_source="external_raster",
                utm_epsg=epsg,
                **shape,
            )
        )
        lsi = patch_geometry(geometry_row, country)["LSI"]
        record = dict(
            city=row.city,
            province=row.province,
            POP=total_pop,
            eta=row.eta,
            POPDEN=total_pop / (shape["area_m2"] / 1e6),
            C_COUNT=hotspot_count(projected),
            PPI=weighted_distance / np.sqrt(shape["area_m2"] / np.pi),
            EI=shape["EI"],
            LSI=lsi,
            LDI=shape["LDI"],
            DNTT=normalized_travel_time(matrix, coords),
            WMDR=round(detour[1], 4),
            ADI=adi,
        )
        rows.append(record)
        metadata.append(
            dict(city=row.city, province=row.province, utm_epsg=epsg, **shape)
        )
        if index % 100 == 0 or index == len(reference):
            print(f"Indicators: {index}/{len(reference)} cities", flush=True)
    data = pd.DataFrame(rows).sort_values("POP", kind="stable").reset_index(drop=True)
    assert np.isfinite(data[FEATURES + ["POP", "eta"]]).all().all()
    data.to_csv(output / "indicators.csv", index=False)
    pd.DataFrame(metadata).to_csv(output / "geometry.csv", index=False)
    (output / "settings.json").write_text(
        json.dumps(
            dict(
                C_COUNT_eps_m=1500,
                C_COUNT_min_samples=3,
                C_COUNT_population_quantile=0.5,
                DNTT_unit="min/km",
                LSI_connectivity=8,
                LDI_connectivity=4,
                skipped_cities_without_rasters=skipped,
            ),
            indent=2,
        ),
        "utf-8",
    )
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--country-data", type=Path, default=DATA / "country",
        help="City centre layers, clipped population rasters and saved routing inputs.",
    )
    parser.add_argument("--city-inputs", type=Path, default=DATA / "city_indicators.csv")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = make_output("main_text", "indicators", args.output)
    compute(args.country_data, args.city_inputs, output)
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
