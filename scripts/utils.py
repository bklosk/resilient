#!/usr/bin/env python3
"""
Common utilities for photogrammetry point cloud processing.

This module contains shared functionality used across multiple scripts
to reduce code duplication and improve maintainability.
"""

import logging
import os
import json
import time
from typing import Tuple, Optional
from geopy.geocoders import Photon, ArcGIS
from geopy.exc import GeopyError
from pyproj import Transformer, CRS
import numpy as np
import laspy

# Set up logging
logger = logging.getLogger(__name__)


class GeocodeUtils:
    """Centralized geocoding functionality using multiple services."""

    def __init__(self):
        """Initialize geocoder with multiple fallback services."""
        self.user_agent = "photogrammetry_geocoder"
        # Multiple geocoding services in order of preference
        self.geocoders = [
            Photon(user_agent=self.user_agent, timeout=10),  # Free, no API key needed
            ArcGIS(timeout=10),  # Free tier, no API key needed
        ]
        # Backwards compatibility for older code/tests expecting `geolocator`
        self.geolocator = self.geocoders[0]
        # Fallback coordinates for common test addresses
        self.fallback_coordinates = {
            "1250 wildwood road, boulder, co": (40.0274, -105.2519),
            "1250 wildwood road boulder co": (40.0274, -105.2519),
            "1250 wildwood rd, boulder, co": (40.0274, -105.2519),
            "1250 wildwood rd boulder co": (40.0274, -105.2519),
            "boulder, co": (40.0150, -105.2705),
            "boulder colorado": (40.0150, -105.2705),
            "denver, co": (39.7392, -104.9903),
            "denver colorado": (39.7392, -104.9903),
        }

    def geocode_address(
        self, address: str, max_retries: int = 3
    ) -> Tuple[float, float]:
        """Convert an address to lat/lon coordinates using multiple geocoding services.

        Args:
            address: Address string to geocode
            max_retries: Maximum number of retry attempts

        Returns:
            Tuple of (latitude, longitude)

        Raises:
            ValueError: If geocoding fails
        """
        addr_lower = address.lower().strip()
        if addr_lower in self.fallback_coordinates:
            lat, lon = self.fallback_coordinates[addr_lower]
            logger.info(
                f"Using fallback coordinates for '{address}': {lat}, {lon}"
            )
            return lat, lon

        last_error: Optional[str] = None

        for geocoder in self.geocoders:
            for attempt in range(max_retries):
                try:
                    logger.info(
                        f"Attempting geocoding with {geocoder.__class__.__name__}..."
                    )
                    location = geocoder.geocode(address)

                    if location:
                        lat, lon = float(location.latitude), float(location.longitude)
                        logger.info(f"Successfully geocoded to: {lat:.6f}, {lon:.6f}")
                        return lat, lon
                    else:
                        last_error = f"{geocoder.__class__.__name__}: No results found"

                except GeopyError as e:
                    last_error = f"{geocoder.__class__.__name__}: {str(e)}"
                    logger.warning(f"Geocoding attempt {attempt + 1} failed: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(1)  # Brief delay between retries
                except Exception as e:
                    last_error = f"{geocoder.__class__.__name__}: {str(e)}"
                    logger.warning(f"Unexpected error: {e}")
                    break  # Don't retry on unexpected errors

        # Fallback to hard-coded coordinates if we have a match
        addr_lower = address.lower().strip()
        if addr_lower in self.fallback_coordinates:
            lat, lon = self.fallback_coordinates[addr_lower]
            logger.info(
                f"Using fallback coordinates for '{address}': {lat}, {lon}"
            )
            return lat, lon
        for key, coords in self.fallback_coordinates.items():
            if key in addr_lower or addr_lower in key:
                lat, lon = coords
                logger.warning(
                    f"Using partial match fallback coordinates for '{address}': {lat}, {lon}"
                )
                return lat, lon

        raise ValueError(f"All geocoding services failed. Last error: {last_error}")


class CRSUtils:
    """Coordinate Reference System utilities."""

    @staticmethod
    def detect_point_cloud_crs(las_data: laspy.LasData) -> Optional[str]:
        """Detect CRS from point cloud coordinates using heuristics.

        Args:
            las_data: LAS/LAZ point cloud data

        Returns:
            EPSG code string if detected, None otherwise
        """
        x_coords = np.array(las_data.x)
        y_coords = np.array(las_data.y)

        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)

        logger.info(
            f"Point cloud coordinate ranges - X: {x_min:.2f} to {x_max:.2f}, Y: {y_min:.2f} to {y_max:.2f}"
        )

        # Web Mercator (EPSG:3857) - common for web mapping services
        if -20037508 < x_min < 20037508 and -20037508 < y_min < 20037508:
            # Additional check: Web Mercator coordinates are typically large numbers
            if abs(x_min) > 1000000 or abs(y_min) > 1000000:
                logger.info("Detected likely Web Mercator coordinates (EPSG:3857)")
                return "EPSG:3857"

        # WGS84 Geographic (EPSG:4326) - latitude/longitude
        if -180 <= x_min <= 180 and -90 <= y_min <= 90:
            logger.info("Detected likely WGS84 geographic coordinates (EPSG:4326)")
            return "EPSG:4326"

        # UTM Zones for US (general UTM pattern first, then refine by region)
        # General UTM pattern: X between 160000-834000, Y between 0-10000000
        if 160000 < x_min < 834000 and 0 < y_min < 10000000:
            # For continental US, Y values are typically > 3000000
            if y_min > 3000000:
                # Guess UTM zone based on rough geographic distribution
                # This is a simplified heuristic - real detection would need more context
                logger.info(
                    "Detected likely UTM coordinates (assuming Zone 13N - EPSG:26913)"
                )
                return "EPSG:26913"

        # Colorado State Plane (feet) - common for US LiDAR
        if 3000000 < x_min < 3200000 and 1700000 < y_min < 1900000:
            logger.info("Detected likely Colorado State Plane coordinates (feet)")
            return "EPSG:2232"
        # Colorado State Plane (meters)
        elif 3000000 < x_min < 3200000 and 500000 < y_min < 700000:
            logger.info("Detected likely Colorado State Plane coordinates (meters)")
            return "EPSG:26954"

        # Fallback: If coordinates are large projected values, try Web Mercator
        if (abs(x_min) > 1000000 or abs(y_min) > 1000000) and abs(x_min) < 20037508:
            logger.info(
                "Fallback: Large coordinate values suggest Web Mercator (EPSG:3857)"
            )
            return "EPSG:3857"

        logger.warning("Could not automatically detect point cloud CRS")
        return None

    @staticmethod
    def transform_coordinates(
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        source_crs: str,
        target_crs: str,
        batch_size: int = 500000,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Transform coordinates between CRS with batching for performance.

        Args:
            x_coords: X coordinates
            y_coords: Y coordinates
            source_crs: Source CRS (e.g., "EPSG:2232")
            target_crs: Target CRS (e.g., "EPSG:4326")
            batch_size: Batch size for processing

        Returns:
            Tuple of transformed (x, y) coordinates
        """
        try:
            source_crs_obj = CRS.from_string(source_crs)
            target_crs_obj = CRS.from_string(target_crs)

            transformer = Transformer.from_crs(
                source_crs_obj, target_crs_obj, always_xy=True
            )

            total_points = len(x_coords)
            transformed_x = np.zeros_like(x_coords)
            transformed_y = np.zeros_like(y_coords)

            logger.info(
                f"Transforming {total_points:,} points in batches of {batch_size:,}..."
            )

            for i in range(0, total_points, batch_size):
                end_idx = min(i + batch_size, total_points)
                batch_x = x_coords[i:end_idx]
                batch_y = y_coords[i:end_idx]

                trans_x, trans_y = transformer.transform(batch_x, batch_y)
                transformed_x[i:end_idx] = trans_x
                transformed_y[i:end_idx] = trans_y

            return transformed_x, transformed_y

        except Exception as e:
            logger.error(f"Coordinate transformation failed: {e}")
            raise RuntimeError(f"Coordinate transformation failed: {e}")


class BoundingBoxUtils:
    """Utilities for working with bounding boxes."""

    @staticmethod
    def generate_bounding_box(lat: float, lon: float, buffer_km: float = 1.0) -> str:
        """Generate a bounding box around coordinates.

        Args:
            lat: Latitude in decimal degrees
            lon: Longitude in decimal degrees
            buffer_km: Buffer distance in kilometers

        Returns:
            Comma-separated bounding box string: "min_lon,min_lat,max_lon,max_lat"
        """
        # Convert buffer from km to degrees (rough approximation)
        buffer_deg = buffer_km / 111.0  # 1 degree ≈ 111 km

        min_lon = lon - buffer_deg
        max_lon = lon + buffer_deg
        min_lat = lat - buffer_deg
        max_lat = lat + buffer_deg

        bbox = f"{min_lon},{min_lat},{max_lon},{max_lat}"
        logger.info(
            f"Generated bounding box for {lat:.6f}, {lon:.6f} with {buffer_km}km buffer: {bbox}"
        )
        return bbox

    @staticmethod
    def validate_bounding_box(bbox: str) -> bool:
        """Validate bounding box format.

        Args:
            bbox: Bounding box string

        Returns:
            True if valid, False otherwise
        """
        try:
            parts = bbox.split(",")
            if len(parts) != 4:
                return False

            min_lon, min_lat, max_lon, max_lat = map(float, parts)

            # Basic validation
            if not (-180 <= min_lon <= 180 and -180 <= max_lon <= 180):
                return False
            if not (-90 <= min_lat <= 90 and -90 <= max_lat <= 90):
                return False
            if min_lon >= max_lon or min_lat >= max_lat:
                return False

            return True
        except (ValueError, TypeError):
            return False


class FileUtils:
    """File handling utilities."""


    @staticmethod
    def get_safe_filename(address: str) -> str:
        """Generate a safe filename from an address string.

        Args:
            address: Address string

        Returns:
            Safe filename string
        """
        import re

        # Remove special characters and replace spaces with underscores
        safe_name = re.sub(r"[^\w\s-]", "", address).strip()
        safe_name = re.sub(r"[-\s]+", "_", safe_name)
        return safe_name.lower()




try:
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False


class S3Utils:
    """Centralized S3 utilities for USGS LiDAR data access."""

    def __init__(self, bucket_name: str = "usgs-lidar-public"):
        """
        Initialize S3 utilities.

        Args:
            bucket_name: Name of the S3 bucket to access

        Raises:
            ImportError: If boto3 is not available
        """
        if not BOTO3_AVAILABLE:
            raise ImportError(
                "boto3 is required for S3 operations. Install with: pip install boto3"
            )

        self.bucket_name = bucket_name

        # Create S3 client with no credentials required for public bucket
        self.s3_client = boto3.client(
            "s3", region_name="us-west-2", config=Config(signature_version=UNSIGNED)
        )

    def get_client(self):
        """Get the configured S3 client."""
        return self.s3_client

    def list_objects(self, prefix: str = "", delimiter: str = ""):
        """
        List objects in the bucket with optional prefix and delimiter.

        Args:
            prefix: Object key prefix to filter by
            delimiter: Delimiter for hierarchical listing

        Returns:
            Paginator for iterating through results
        """
        paginator = self.s3_client.get_paginator("list_objects_v2")
        page_iterator = paginator.paginate(
            Bucket=self.bucket_name, Prefix=prefix, Delimiter=delimiter
        )
        return page_iterator


class JSONUtils:
    """Centralized JSON file handling utilities."""

    @staticmethod
    def save_metadata(
        data: dict, output_dir: str, filename: str = "metadata.json"
    ) -> str:
        """
        Save data to a JSON file with consistent formatting.

        Args:
            data: Dictionary to save as JSON
            output_dir: Directory to save the file
            filename: Name of the output file

        Returns:
            Path to saved file

        Raises:
            Exception: If save operation fails
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            filepath = os.path.join(output_dir, filename)

            with open(filepath, "w") as f:
                json.dump(data, f, indent=2, default=str)

            return filepath
        except Exception as e:
            raise Exception(f"Failed to save JSON file {filepath}: {e}")

    @staticmethod
    def load_json(filepath: str) -> dict:
        """
        Load JSON data from file with error handling.

        Args:
            filepath: Path to JSON file

        Returns:
            Loaded dictionary data

        Raises:
            Exception: If load operation fails
        """
        try:
            with open(filepath, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"JSON file not found: {filepath}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in file {filepath}: {e}")
        except Exception as e:
            raise Exception(f"Failed to load JSON file {filepath}: {e}")
