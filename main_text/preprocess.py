"""Prepare population grids and vehicle connectors from supplied inputs."""

import argparse
from pathlib import Path
import pickle
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import shapes
from rasterio.mask import mask
from shapely.geometry import shape

from utils.paths import make_output


def clip_population(raster_path, boundary_path, output):
    """Clip to the supplied city boundary while retaining the source mask."""
    boundaries = gpd.read_file(boundary_path)
    with rasterio.open(raster_path) as source:
        projected = boundaries.to_crs(source.crs) if source.crs else boundaries
        image, transform = mask(source, projected.geometry, crop=True)
        metadata = source.meta.copy()
        metadata.update(
            driver="GTiff",
            height=image.shape[1],
            width=image.shape[2],
            transform=transform,
        )
        with rasterio.open(output, "w", **metadata) as target:
            target.write(image)


def population_grid(city, raster_path, boundary_path, roads_path, output):
    """Keep centres inside the boundary for population polygons intersecting roads."""
    with rasterio.open(raster_path) as source:
        values = source.read(1).astype(np.float32)
        valid = source.read_masks(1) > 0
        records = [
            {"value": float(value), "geometry": shape(geometry)}
            for geometry, value in shapes(
                values, mask=valid, transform=source.transform, connectivity=4
            )
            if np.isfinite(value) and value > 0
        ]
        polygons = gpd.GeoDataFrame(records, geometry="geometry", crs=source.crs)
    polygons = polygons.to_crs(4326)
    polygons["Id"] = np.arange(len(polygons))
    centres = polygons[["Id", "value", "geometry"]].copy()
    # Define routing nodes at polygon centroids in geographic coordinates.
    centres["geometry"] = gpd.GeoSeries(
        [geometry.centroid for geometry in polygons.geometry], crs=4326
    )
    boundary = gpd.read_file(boundary_path).to_crs(4326).dissolve()[["geometry"]]
    roads = gpd.read_file(roads_path).to_crs(4326)
    if "osm_id" not in roads:
        raise ValueError("The road layer must contain the osm_id field.")
    inside = gpd.sjoin(centres, boundary, predicate="within", how="inner")
    road_join = gpd.sjoin(polygons, roads, predicate="intersects", how="inner")
    counts = road_join.groupby("Id")["osm_id"].count().rename("road_count")
    centres = inside.merge(counts, on="Id", how="inner")
    centres["Id"] = np.arange(len(centres))
    polygons.to_file(output / f"{city}_pop.shp")
    centres[["Id", "value", "geometry"]].to_file(output / f"{city}_center.shp")
    coordinates = {
        int(index): (float(geometry.y), float(geometry.x))
        for index, geometry in enumerate(centres.geometry)
    }
    with (output / f"{city}_latlon.pkl").open("wb") as stream:
        pickle.dump(coordinates, stream)


def prepare_connectors(city, matrix_path, output):
    """Write the dense durations and connectors shorter than 600 seconds."""
    with Path(matrix_path).open("rb") as stream:
        routes = pickle.load(stream)
    count = len(routes)
    connectors = {index: {} for index in range(count)}
    dense = np.empty((count, count), dtype=float)
    # Connector indexing is next origin, then previous destination.
    for origin in range(count):
        for destination in range(count):
            value = routes[origin][destination]
            dense[origin, destination] = 1e5 if value is None else float(value)
            previous = routes[destination][origin]
            if previous is not None and previous < 600:
                connectors[origin][destination] = previous
    for suffix, values in [("con", dense), ("sim", connectors)]:
        with (output / f"{city}_routes_{suffix}.pkl").open("wb") as stream:
            pickle.dump(values, stream)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=["clip-raster", "population-grid", "connectors"]
    )
    parser.add_argument("--city", required=True)
    parser.add_argument("--raster", type=Path)
    parser.add_argument("--boundary", type=Path)
    parser.add_argument("--roads", type=Path)
    parser.add_argument("--matrix", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    required = {
        "clip-raster": ["raster", "boundary"],
        "population-grid": ["raster", "boundary", "roads"],
        "connectors": ["matrix"],
    }[args.stage]
    for name in required:
        path = getattr(args, name)
        if path is None:
            parser.error(f"--{name} is required for {args.stage}")
        if not path.is_file():
            raise FileNotFoundError(path)
    if Path(args.city).name != args.city:
        parser.error("City must be a filename key without path separators.")
    output = make_output("main_text", args.stage, args.output)
    if args.stage == "clip-raster":
        clip_population(args.raster, args.boundary, output / f"{args.city}_clip.tif")
    elif args.stage == "population-grid":
        population_grid(args.city, args.raster, args.boundary, args.roads, output)
    else:
        prepare_connectors(args.city, args.matrix, output)
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
