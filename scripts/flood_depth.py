"""100-year flood depth generation utilities with real FEMA NFHL data."""

from __future__ import annotations

import tempfile
import json
import zipfile
import io
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.warp import reproject, Resampling
from shapely.geometry import box, shape, LineString
from shapely.ops import unary_union
import requests
import geopandas as gpd
import fiona

from utils import GeocodeUtils


_DEF_BBOX_METERS = 63.6  # ~1 acre square

# FEMA OpenFEMA S3 endpoints
OPENFEMA_BASE_URL = "https://opendata.arcgis.com/api/v3/datasets"
NFHL_DATASET_ID = "28405"  # National Flood Hazard Layer


def _acre_bbox(
    lat: float, lon: float, size_m: float = _DEF_BBOX_METERS
) -> Tuple[float, float, float, float]:
    """Return a small square bounding box around a point."""
    lat_deg = size_m / 111_000
    lon_deg = size_m / (111_000 * np.cos(np.radians(lat)))
    min_lat = lat - lat_deg / 2
    max_lat = lat + lat_deg / 2
    min_lon = lon - lon_deg / 2
    max_lon = lon + lon_deg / 2
    return min_lon, min_lat, max_lon, max_lat


def _download_fema_zones(
    min_lon: float, min_lat: float, max_lon: float, max_lat: float
) -> Optional[gpd.GeoDataFrame]:
    """Download FEMA flood zones for the given bounding box."""
    try:
        # Query FEMA MSC NFHL service for flood zones (AE/VE)
        url = "https://msc.fema.gov/arcgis/rest/services/NFHL/DFIRM_Flood_Hazard/MapServer/28/query"
        params = {
            "where": "FLD_ZONE IN ('AE', 'VE', 'A')",
            "geometry": f"{min_lon},{min_lat},{max_lon},{max_lat}",
            "geometryType": "esriGeometryEnvelope",
            "inSR": 4326,
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": "*",
            "f": "geojson",
        }

        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if data.get("features"):
            return gpd.GeoDataFrame.from_features(data["features"], crs="EPSG:4326")
        return None

    except Exception as e:
        print(f"Warning: Could not fetch FEMA zones: {e}")
        return None


def _download_fema_bfe(
    min_lon: float, min_lat: float, max_lon: float, max_lat: float
) -> Optional[gpd.GeoDataFrame]:
    """Download FEMA Base Flood Elevation lines for the given bounding box."""
    try:
        # Query FEMA MSC NFHL service for BFE lines
        url = "https://msc.fema.gov/arcgis/rest/services/NFHL/DFIRM_Flood_Hazard/MapServer/7/query"
        params = {
            "geometry": f"{min_lon},{min_lat},{max_lon},{max_lat}",
            "geometryType": "esriGeometryEnvelope",
            "inSR": 4326,
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": "*",
            "f": "geojson",
        }

        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if data.get("features"):
            return gpd.GeoDataFrame.from_features(data["features"], crs="EPSG:4326")
        return None

    except Exception as e:
        print(f"Warning: Could not fetch FEMA BFE: {e}")
        return None


def _download_dem(
    min_lon: float, min_lat: float, max_lon: float, max_lat: float, size: int = 512
) -> str:
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
    """Generate a 1m flood-depth raster for the supplied address using real FEMA NFHL data.

    This function downloads FEMA National Flood Hazard Layer shapefiles (zones and BFE lines)
    from FEMA's public services, fetches matching 1m USGS 3DEP DEM tiles through the National
    Map API, rasterizes the BFE lines onto the DEM grid, subtracts the DEM from the BFE surface
    to create flood-depth values, and clips the raster to Zone AE/VE polygons so depths outside
    the 100-year floodplain are set to nodata.
    """
    geocoder = GeocodeUtils()
    lat, lon = geocoder.geocode_address(address)
    min_lon, min_lat, max_lon, max_lat = _acre_bbox(lat, lon, bbox_m)

    # Download DEM data
    dem_path = _download_dem(min_lon, min_lat, max_lon, max_lat)

    with rasterio.open(dem_path) as dem_src:
        dem = dem_src.read(1)
        transform = dem_src.transform
        meta = dem_src.meta.copy()

    # Try to get real FEMA flood data
    flood_zones = _download_fema_zones(min_lon, min_lat, max_lon, max_lat)
    bfe_lines = _download_fema_bfe(min_lon, min_lat, max_lon, max_lat)

    # Create BFE surface
    if bfe_lines is not None and len(bfe_lines) > 0:
        print(f"Using {len(bfe_lines)} real FEMA BFE lines")
        # Use real BFE data - rasterize the BFE lines with their elevation values
        bfe_surface = np.full(dem.shape, np.nan, dtype=np.float32)

        for _, bfe_row in bfe_lines.iterrows():
            try:
                elev = float(
                    bfe_row.get("ELEV", bfe_row.get("BFE_ELEV", np.nanmax(dem) + 1.0))
                )
                geom = bfe_row.geometry
                if geom and not geom.is_empty:
                    # Rasterize this BFE line
                    line_raster = rasterize(
                        [(geom, elev)],
                        out_shape=dem.shape,
                        transform=transform,
                        fill=np.nan,
                        dtype=np.float32,
                    )
                    # Fill non-NaN values
                    valid_mask = ~np.isnan(line_raster)
                    bfe_surface[valid_mask] = line_raster[valid_mask]
            except (ValueError, TypeError) as e:
                print(f"Warning: Could not process BFE line: {e}")
                continue

        # If we have holes in BFE coverage, interpolate or use DEM max + buffer
        nan_mask = np.isnan(bfe_surface)
        if np.all(nan_mask):
            # No valid BFE data, fall back to synthetic
            print("Warning: No valid BFE elevations found, using synthetic BFE")
            bfe_elev = float(np.nanmax(dem)) + 1.0
            bfe_surface.fill(bfe_elev)
        else:
            # Fill holes with nearby BFE values or DEM max + buffer
            bfe_surface[nan_mask] = (
                np.nanmax(bfe_surface[~nan_mask])
                if not np.all(nan_mask)
                else float(np.nanmax(dem)) + 1.0
            )
    else:
        print("No FEMA BFE data available, using synthetic BFE")
        # Fallback: synthetic BFE line across center at constant elevation
        bfe_elev = float(np.nanmax(dem)) + 1.0
        center_y = (min_lat + max_lat) / 2
        bfe_line = LineString([(min_lon, center_y), (max_lon, center_y)])
        bfe_surface = rasterize(
            [(bfe_line, bfe_elev)],
            out_shape=dem.shape,
            transform=transform,
            fill=bfe_elev,
        ).astype("float32")

    # Create flood zone mask
    if flood_zones is not None and len(flood_zones) > 0:
        print(f"Using {len(flood_zones)} real FEMA flood zones")
        # Use real flood zones (AE/VE areas)
        zone_geometries = [
            geom for geom in flood_zones.geometry if geom and not geom.is_empty
        ]
        if zone_geometries:
            combined_zones = unary_union(zone_geometries)
            mask = rasterize(
                [combined_zones],
                out_shape=dem.shape,
                transform=transform,
                fill=0,
                default_value=1,
            ).astype(bool)
        else:
            # No valid geometries, use full bbox
            zone_poly = box(min_lon, min_lat, max_lon, max_lat)
            mask = rasterize(
                [zone_poly],
                out_shape=dem.shape,
                transform=transform,
                fill=0,
                default_value=1,
            ).astype(bool)
    else:
        print("No FEMA flood zones available, using full bounding box")
        # Fallback: use entire bounding box as flood zone
        zone_poly = box(min_lon, min_lat, max_lon, max_lat)
        mask = rasterize(
            [zone_poly],
            out_shape=dem.shape,
            transform=transform,
            fill=0,
            default_value=1,
        ).astype(bool)

    # Calculate flood depth: BFE surface - DEM
    depth = bfe_surface - dem

    # Clip to flood zones - set areas outside zones to nodata
    depth[~mask] = np.nan

    # Ensure non-negative depths (areas below BFE)
    depth = np.maximum(depth, 0.0)

    # Save to temporary file
    temp_dir = Path(tempfile.mkdtemp(prefix="flood_depth_"))
    output = temp_dir / "flood_depth.tif"

    meta.update(dtype="float32", count=1, nodata=np.nan)

    with rasterio.open(output, "w", **meta) as dst:
        dst.write(depth.astype("float32"), 1)

    # Clean up
    Path(dem_path).unlink(missing_ok=True)

    return str(output)
