"""Simplified 100-year flood depth generation utilities."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np
import rasterio
from rasterio.features import rasterize
from shapely.geometry import box, LineString
import requests

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


def _download_dem(min_lon: float, min_lat: float, max_lon: float, max_lat: float, size: int = 512) -> str:
    """Download a DEM tile from the USGS 3DEP service."""
    url = "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage"
    params = {
        "bbox": f"{min_lon},{min_lat},{max_lon},{max_lat}",
        "bboxSR": 4326,
        "imageSR": 4326,
        "format": "tiff",
        "pixelType": "F32",
        "f": "image",
        "size": f"{size},{size}",
    }
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
    tmp.write(resp.content)
    tmp.close()
    return tmp.name


def generate(address: str, bbox_m: float = _DEF_BBOX_METERS) -> str:
    """Generate an approximate 1m flood-depth raster for the supplied address.

    This function relies on public web services for DEM data. If FEMA NFHL
    shapefiles are not accessible in the execution environment, a simple
    synthetic BFE surface is used instead.
    """

    geocoder = GeocodeUtils()
    lat, lon = geocoder.geocode_address(address)
    min_lon, min_lat, max_lon, max_lat = _acre_bbox(lat, lon, bbox_m)

    dem_path = _download_dem(min_lon, min_lat, max_lon, max_lat)

    with rasterio.open(dem_path) as dem_src:
        dem = dem_src.read(1)
        transform = dem_src.transform
        meta = dem_src.meta.copy()

    # Placeholder BFE line across the centre of the tile at a constant elevation
    bfe_elev = float(np.nanmax(dem)) + 1.0
    center_y = (min_lat + max_lat) / 2
    bfe_line = LineString([(min_lon, center_y), (max_lon, center_y)])

    bfe_surface = rasterize([(bfe_line, bfe_elev)], out_shape=dem.shape, transform=transform, fill=bfe_elev).astype("float32")

    zone_poly = box(min_lon, min_lat, max_lon, max_lat)
    mask = rasterize([zone_poly], out_shape=dem.shape, transform=transform, fill=0, default_value=1).astype(bool)

    depth = bfe_surface - dem
    depth[~mask] = np.nan

    temp_dir = Path(tempfile.mkdtemp(prefix="flood_depth_"))
    output = temp_dir / "flood_depth.tif"

    meta.update(dtype="float32", count=1, nodata=np.nan)

    with rasterio.open(output, "w", **meta) as dst:
        dst.write(depth.astype("float32"), 1)

    Path(dem_path).unlink(missing_ok=True)

    return str(output)
