"""Simplified 100-year flood depth generation utilities."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely.geometry import box, LineString

from utils import GeocodeUtils


_DEF_BBOX_METERS = 63.6  # ~1 acre square


def _acre_bbox(lat: float, lon: float, size_m: float = _DEF_BBOX_METERS) -> Tuple[float, float, float, float]:
    """Return a small square bounding box around a point."""
    lat_deg = size_m / 111_000
    lon_deg = size_m / (111_000 * np.cos(np.radians(lat)))
    min_lat = lat - lat_deg / 2
    max_lat = lat + lat_deg / 2
    min_lon = lon - lon_deg / 2
    max_lon = lon + lon_deg / 2
    return min_lon, min_lat, max_lon, max_lat


def generate(address: str, bbox_m: float = _DEF_BBOX_METERS) -> str:
    """Create a dummy flood depth GeoTIFF for an address.

    The implementation is intentionally lightweight and does not fetch real NFHL
    or DEM data, but mimics the expected processing steps so that the API can
    serve an example product without external dependencies.
    """
    geocoder = GeocodeUtils()
    lat, lon = geocoder.geocode_address(address)
    min_lon, min_lat, max_lon, max_lat = _acre_bbox(lat, lon, bbox_m)

    width = height = int(bbox_m)
    transform = from_origin(min_lon, max_lat, (max_lon - min_lon) / width, (max_lat - min_lat) / height)

    # Dummy DEM surface
    dem = np.linspace(100.0, 101.0, width * height).reshape((height, width)).astype("float32")

    # Simplified BFE lines and flood zone covering entire bbox
    bfe_surface = np.full_like(dem, 103.0, dtype="float32")
    zone_poly = box(min_lon, min_lat, max_lon, max_lat)

    mask = rasterize([zone_poly], out_shape=dem.shape, transform=transform, fill=0, default_value=1).astype(bool)

    depth = bfe_surface - dem
    depth[~mask] = np.nan

    temp_dir = Path(tempfile.mkdtemp(prefix="flood_depth_"))
    output = temp_dir / "flood_depth.tif"

    with rasterio.open(
        output,
        "w",
        driver="GTiff",
        height=depth.shape[0],
        width=depth.shape[1],
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        nodata=np.nan,
    ) as dst:
        dst.write(depth, 1)

    return str(output)
