#!/usr/bin/env python3
"""
Common utilities for photogrammetry point cloud processing.

This module contains shared functionality used across multiple scripts
to reduce code duplication and improve maintainability.
"""

import logging
import os
import json
import requests
from typing import Tuple, Optional
from geopy.geocoders import Nominatim
from geopy.exc import GeopyError
from pyproj import Transformer, CRS
import numpy as np
import laspy

# Set up logging
logger = logging.getLogger(__name__)


class GeocodeUtils:
    """Centralized geocoding functionality with fallback support."""

    def __init__(self):
        self.geolocator = Nominatim(
            user_agent="photogrammetry_processor", timeout=10  # 10 second timeout
        )
        self.fallback_coordinates = {
            # Common test addresses for development
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
        """Convert an address to lat/lon coordinates with fallback support.

        Args:
            address: Address string to geocode
            max_retries: Maximum number of retry attempts

        Returns:
            Tuple of (latitude, longitude)

        Raises:
            ValueError: If geocoding fails after all retries and no fallback available
        """
        address_lower = address.lower().strip()
        last_error: Optional[str] = None

        for attempt in range(max_retries):
            try:
                location = self.geolocator.geocode(address, country_codes="us")
                if location:
                    logger.info(
                        f"Geocoded '{address}' to {location.latitude:.6f}, {location.longitude:.6f}"
                    )
                    return float(location.latitude), float(location.longitude)
                last_error = "No results returned"
            except GeopyError as e:
                last_error = str(e)
                logger.warning(
                    f"Geocoding attempt {attempt + 1} failed for '{address}': {e}"
                )

        # Fallback to hardcoded coordinates if available
        if address_lower in self.fallback_coordinates:
            lat, lon = self.fallback_coordinates[address_lower]
            logger.info(
                f"Using fallback coordinates for '{address}': {lat}, {lon}"
            )
            return lat, lon

        for fallback_addr, coords in self.fallback_coordinates.items():
            if fallback_addr in address_lower or address_lower in fallback_addr:
                lat, lon = coords
                logger.warning(
                    f"Using partial match fallback coordinates for '{address}': {lat}, {lon}"
                )
                return lat, lon

        raise ValueError(
            f"Could not geocode address '{address}': {last_error or 'unknown error'}"
        )


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
    def validate_point_cloud_file(file_path: str) -> bool:
        """Validate that a point cloud file exists and is readable.

        Args:
            file_path: Path to point cloud file

        Returns:
            True if valid, False otherwise
        """
        try:
            from pathlib import Path

            path = Path(file_path)

            if not path.exists():
                logger.error(f"Point cloud file not found: {file_path}")
                return False

            if not path.suffix.lower() in [".las", ".laz"]:
                logger.error(f"Unsupported file format: {path.suffix}")
                return False

            # Try to read header
            las_data = laspy.read(str(path))
            num_points = len(las_data.points)
            if num_points == 0:
                logger.error(f"Point cloud file is empty: {file_path}")
                return False

            logger.info(f"Validated point cloud file: {num_points:,} points")
            return True

        except Exception as e:
            logger.error(f"Point cloud file validation failed: {e}")
            return False

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


class HTTPUtils:
    """Centralized HTTP request utilities with consistent error handling."""

    @staticmethod
    def post_json(url: str, json_data: dict, timeout: int = 30) -> dict:
        """
        Make a POST request with JSON data and return parsed response.

        Args:
            url: URL to send request to
            json_data: Data to send as JSON
            timeout: Request timeout in seconds

        Returns:
            Parsed JSON response

        Raises:
            Exception: If request fails
        """
        try:
            response = requests.post(url, json=json_data, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"HTTP POST request failed to {url}: {e}")

    @staticmethod
    def download_file(
        url: str,
        output_path: str,
        chunk_size: int = 1024 * 1024,
        timeout: int = 60,
        use_parallel: bool = True,
        num_connections: int = 4,
    ) -> None:
        """
        Download a file from URL to local path with streaming and progress indication.

        Supports parallel downloading for large files and resume capability.

        Args:
            url: URL to download from
            output_path: Local path to save file
            chunk_size: Size of chunks to download (default: 1MB for better performance)
            timeout: Request timeout in seconds
            use_parallel: Whether to use parallel downloading for large files
            num_connections: Number of parallel connections to use

        Raises:
            Exception: If download fails
        """
        import time
        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed

        try:
            # Create directory if it doesn't exist
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            # Get file size first for progress tracking
            print(f"Checking file size for: {url}")
            head_response = requests.head(url, timeout=timeout)
            head_response.raise_for_status()

            total_size = int(head_response.headers.get("content-length", 0))
            print(f"File size: {total_size:,} bytes ({total_size/(1024*1024):.1f} MB)")

            # Check if server supports range requests
            supports_ranges = head_response.headers.get("accept-ranges") == "bytes"

            # Use parallel download for large files if supported
            if (
                use_parallel and supports_ranges and total_size > 50 * 1024 * 1024
            ):  # 50MB threshold
                print(f"Using parallel download with {num_connections} connections")
                return HTTPUtils._parallel_download(
                    url, output_path, total_size, num_connections, timeout
                )

            # Check if we should resume download
            downloaded_size = 0
            resume_header = {}
            if os.path.exists(output_path):
                downloaded_size = os.path.getsize(output_path)
                if downloaded_size < total_size:
                    if supports_ranges:
                        resume_header = {"Range": f"bytes={downloaded_size}-"}
                        print(f"Resuming download from {downloaded_size:,} bytes")
                    else:
                        print("Server doesn't support range requests, starting over")
                        downloaded_size = 0
                elif downloaded_size == total_size:
                    print(f"File already fully downloaded: {output_path}")
                    return
                else:
                    # File is larger than expected, start over
                    downloaded_size = 0
                    resume_header = {}

            # Start download with streaming
            print("Starting download...")
            response = requests.get(
                url, stream=True, timeout=timeout, headers=resume_header
            )
            response.raise_for_status()

            # Open file in appropriate mode
            mode = "ab" if downloaded_size > 0 else "wb"

            with open(output_path, mode) as f:
                start_time = time.time()
                bytes_downloaded = downloaded_size
                last_progress_time = start_time

                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        bytes_downloaded += len(chunk)

                        # Show progress every 2 seconds
                        current_time = time.time()
                        if current_time - last_progress_time >= 2.0:
                            if total_size > 0:
                                progress = (bytes_downloaded / total_size) * 100
                                elapsed = current_time - start_time
                                if elapsed > 0:
                                    speed = (
                                        (bytes_downloaded - downloaded_size)
                                        / elapsed
                                        / 1024
                                        / 1024
                                    )  # MB/s
                                    eta = (
                                        (total_size - bytes_downloaded)
                                        / (speed * 1024 * 1024)
                                        if speed > 0
                                        else 0
                                    )
                                    print(
                                        f"Progress: {progress:.1f}% ({bytes_downloaded:,}/{total_size:,} bytes) - {speed:.1f} MB/s - ETA: {eta:.0f}s"
                                    )
                                else:
                                    print(
                                        f"Progress: {progress:.1f}% ({bytes_downloaded:,}/{total_size:,} bytes)"
                                    )
                            else:
                                elapsed = current_time - start_time
                                if elapsed > 0:
                                    speed = (
                                        (bytes_downloaded - downloaded_size)
                                        / elapsed
                                        / 1024
                                        / 1024
                                    )  # MB/s
                                    print(
                                        f"Downloaded: {bytes_downloaded:,} bytes - {speed:.1f} MB/s"
                                    )
                                else:
                                    print(f"Downloaded: {bytes_downloaded:,} bytes")
                            last_progress_time = current_time

                # Final progress
                elapsed = time.time() - start_time
                if elapsed > 0:
                    avg_speed = (
                        (bytes_downloaded - downloaded_size) / elapsed / 1024 / 1024
                    )
                    print(
                        f"Download completed: {bytes_downloaded:,} bytes in {elapsed:.1f}s (avg: {avg_speed:.1f} MB/s)"
                    )

        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to download file from {url}: {e}")

    @staticmethod
    def _parallel_download(
        url: str, output_path: str, total_size: int, num_connections: int, timeout: int
    ) -> None:
        """
        Download a file using multiple parallel connections.
        """
        import time
        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Calculate chunk size per connection
        chunk_size = total_size // num_connections
        ranges = []

        for i in range(num_connections):
            start = i * chunk_size
            end = start + chunk_size - 1
            if i == num_connections - 1:  # Last chunk gets remainder
                end = total_size - 1
            ranges.append((start, end))

        print(f"Downloading {num_connections} chunks in parallel...")

        # Temporary files for each chunk
        temp_files = [f"{output_path}.part{i}" for i in range(num_connections)]

        def download_chunk(chunk_info):
            chunk_idx, start, end = chunk_info
            temp_file = temp_files[chunk_idx]

            headers = {"Range": f"bytes={start}-{end}"}
            response = requests.get(url, headers=headers, stream=True, timeout=timeout)
            response.raise_for_status()

            with open(temp_file, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

            return chunk_idx, end - start + 1

        start_time = time.time()

        # Download chunks in parallel
        with ThreadPoolExecutor(max_workers=num_connections) as executor:
            futures = [
                executor.submit(download_chunk, (i, start, end))
                for i, (start, end) in enumerate(ranges)
            ]

            completed_bytes = 0
            for future in as_completed(futures):
                chunk_idx, chunk_bytes = future.result()
                completed_bytes += chunk_bytes
                progress = (completed_bytes / total_size) * 100
                elapsed = time.time() - start_time
                speed = completed_bytes / elapsed / 1024 / 1024 if elapsed > 0 else 0
                print(f"Parallel progress: {progress:.1f}% - {speed:.1f} MB/s")

        # Combine all chunks into final file
        print("Combining chunks...")
        with open(output_path, "wb") as outfile:
            for temp_file in temp_files:
                with open(temp_file, "rb") as infile:
                    outfile.write(infile.read())
                os.remove(temp_file)  # Clean up temp file

        elapsed = time.time() - start_time
        avg_speed = total_size / elapsed / 1024 / 1024 if elapsed > 0 else 0
        print(
            f"Parallel download completed: {total_size:,} bytes in {elapsed:.1f}s (avg: {avg_speed:.1f} MB/s)"
        )

    @staticmethod
    def get_json(url: str, timeout: int = 30) -> dict:
        """
        Make a GET request and return parsed JSON response.

        Args:
            url: URL to send request to
            timeout: Request timeout in seconds

        Returns:
            Parsed JSON response

        Raises:
            Exception: If request fails
        """
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"HTTP GET request failed to {url}: {e}")


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
