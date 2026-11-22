"""Raster footprints, patch geometry and urban-form indicators."""

from pathlib import Path
import hashlib
import numpy as np
from PIL import Image
from pyproj import Transformer
from scipy.ndimage import label


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def polygon_moments(vertices):
    """Exact area moments of non-overlapping planar polygon cells, including interiors."""
    v = np.asarray(vertices, dtype=float)
    # Shift before integration to avoid cancellation at large projected coordinates.
    v = v - v.reshape(-1, 2).mean(axis=0)
    x, y = v[:, :, 0], v[:, :, 1]
    xn, yn = np.roll(x, -1, axis=1), np.roll(y, -1, axis=1)
    cross = x * yn - xn * y
    sign = np.sign(cross.sum(axis=1))
    cross = cross * sign[:, None]
    area = cross.sum() / 2
    mx = ((x + xn) * cross).sum() / 6
    my = ((y + yn) * cross).sum() / 6
    mxx = ((x * x + x * xn + xn * xn) * cross).sum() / 12
    myy = ((y * y + y * yn + yn * yn) * cross).sum() / 12
    mxy = ((2 * x * y + x * yn + xn * y + 2 * xn * yn) * cross).sum() / 24
    center = np.array([mx, my]) / area
    cov = np.array([[mxx, mxy], [mxy, myy]]) / area - np.outer(center, center)
    eigenvalues = np.linalg.eigvalsh(cov)
    assert area > 0 and eigenvalues[0] > 0
    return float(area), float(np.sqrt(eigenvalues[1] / eigenvalues[0]))


def exposed_perimeter(mask, vertices):
    """Count each occupied/background cell edge once, including holes and outer edges."""
    r, c = np.nonzero(mask)
    padded = np.pad(mask, 1)
    # Cell corners TL, BL, BR, TR; corresponding left, bottom, right, top neighbors.
    adjacent = np.column_stack(
        [padded[r + 1, c], padded[r + 2, c + 1], padded[r + 1, c + 2], padded[r, c + 1]]
    )
    length = np.linalg.norm(np.roll(vertices, -1, axis=1) - vertices, axis=2)
    return float(length[~adjacent].sum())


def cell_area_km2(latitude_north, pixel_y_deg, pixel_x_deg):
    earth_radius_km = 6371.0088
    north = np.deg2rad(latitude_north)
    south = np.deg2rad(latitude_north - pixel_y_deg)
    return (
        earth_radius_km**2 * np.deg2rad(pixel_x_deg) * (np.sin(north) - np.sin(south))
    )


def raster(path):
    with Image.open(path) as im:
        values = np.array(im, dtype=float)
        px, py, _ = im.tag_v2[33550]
        tie = im.tag_v2[33922]
        nd = float(im.tag_v2.get(42113, -99999))
    valid = np.isfinite(values) & (values != nd)
    assert valid.any() and (values[valid] >= 0).all()
    return (
        values,
        valid,
        float(tie[3] - tie[0] * px),
        float(tie[4] + tie[1] * py),
        float(px),
        float(py),
    )


def weighted_ei(mask):
    # Use cell-centre covariance within each eight-connected patch.
    # Exclude patches with minor-axis variance at most 1e-12 from the mean.
    labels, n = label(mask, structure=np.ones((3, 3), dtype=int))
    total, weighted, excluded = 0, 0.0, 0
    for k in range(1, n + 1):
        coords = np.column_stack(np.where(labels == k)).astype(float)
        centered = coords - coords.mean(axis=0)
        eigen = np.maximum(np.linalg.eigvalsh(centered.T @ centered / len(coords)), 0)
        if eigen[0] <= 1e-12:
            excluded += len(coords)
            continue
        total += len(coords)
        weighted += len(coords) * np.sqrt(eigen[1] / eigen[0])
    assert total > 0
    return weighted / total, excluded, n


def weighted_shape(areas, perimeters):
    areas = np.asarray(areas, dtype=float)
    perimeters = np.asarray(perimeters, dtype=float)
    whole = perimeters.sum() / (2 * np.sqrt(np.pi * areas.sum()))
    patch = np.sum(perimeters * np.sqrt(areas)) / (2 * np.sqrt(np.pi) * areas.sum())
    return float(whole), float(patch)


def patch_geometry(row, raster_root):
    path = Path(raster_root) / row.raster_path
    with Image.open(path) as im:
        values = np.array(im, dtype=float)
        px, py, _ = im.tag_v2[33550]
        tie = im.tag_v2[33922]
        nodata = float(im.tag_v2.get(42113, -99999))
    mask = np.isfinite(values) & (values != nodata)
    assert mask.any() and (values[mask] >= 0).all()
    left, top = float(tie[3] - tie[0] * px), float(tie[4] + tie[1] * py)
    r, c = np.where(mask)
    rr = np.column_stack([r, r + 1, r + 1, r])
    cc = np.column_stack([c, c, c + 1, c + 1])
    transformer = Transformer.from_crs(4326, int(row.utm_epsg), always_xy=True)
    x, y = transformer.transform(left + cc * px, top - rr * py)
    vertices = np.stack([x, y], axis=-1)
    # Independent local-cell integration avoids cancellation at UTM coordinate scale.
    local = vertices - vertices[:, :1, :]
    next_local = np.roll(local, -1, axis=1)
    cell_areas = (
        np.abs(
            np.sum(
                local[:, :, 0] * next_local[:, :, 1]
                - next_local[:, :, 0] * local[:, :, 1],
                axis=1,
            )
        )
        / 2
    )
    padded = np.pad(mask, 1)
    adjacent = np.column_stack(
        [padded[r + 1, c], padded[r + 2, c + 1], padded[r + 1, c + 2], padded[r, c + 1]]
    )
    edge_lengths = np.linalg.norm(np.roll(vertices, -1, axis=1) - vertices, axis=2)
    cell_perimeters = np.sum(np.where(adjacent, 0.0, edge_lengths), axis=1)
    labels, n_patches = label(mask, structure=np.ones((3, 3), dtype=int))
    ids = labels[r, c]
    areas = np.bincount(ids, weights=cell_areas, minlength=n_patches + 1)[1:]
    perimeters = np.bincount(ids, weights=cell_perimeters, minlength=n_patches + 1)[1:]
    assert (areas > 0).all()
    whole, patch = weighted_shape(areas, perimeters)
    assert np.isclose(areas.sum(), row.area_m2, rtol=1e-9, atol=1e-4)
    assert np.isclose(perimeters.sum(), row.perimeter_m, rtol=1e-12, atol=1e-6)
    assert n_patches == int(row.patches_8)
    assert patch <= whole + 1e-10
    return {
        "city": row.city,
        "province": row.province,
        "LSI": patch,
        "patch_count_8": n_patches,
        "valid_cells": int(mask.sum()),
        "valid_zero_cells": int(np.sum(mask & (values == 0))),
        "raster_source": row.raster_source,
        "raster_path": row.raster_path,
        "raster_sha256": sha(path),
    }
