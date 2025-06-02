#!/usr/bin/env python3
"""
NAIP Orthophoto Fetcher using USGS NAIPPlus Service

This module fetches NAIP (National Agriculture Imagery Program) orthophotos
for a given address using the USGS NAIPPlus ImageServer with clipped exports.
The bounding box is automatically sized to approximately one acre.

Usage:
    from get_orthophoto import NAIPFetcher

    fetcher = NAIPFetcher()
    output_path, metadata = fetcher.get_orthophoto_for_address("1250 Wildwood Road, Boulder, CO")
"""

import json
import os
import sys
import math
from datetime import datetime
from typing import Dict, Optional, Tuple
import requests
from utils import GeocodeUtils, FileUtils


class NAIPFetcher:
    """Class to fetch NAIP orthophotos using USGS NAIPPlus ImageServer."""

    def __init__(self):
        """Initialize the NAIP fetcher."""
        self.service_url = "https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPPlus/ImageServer/exportImage"

    def calculate_acre_bbox(
        self, latitude: float, longitude: float
    ) -> Tuple[float, float, float, float]:
        """
        Calculate a bounding box of approximately one acre around the given coordinates.

        Args:
            latitude: Latitude in decimal degrees
            longitude: Longitude in decimal degrees

        Returns:
            Tuple of (min_lon, min_lat, max_lon, max_lat)
        """
        # One acre = 4,047 square meters
        # For a square area: side length = sqrt(4047) ≈ 63.6 meters
        # Convert to degrees (rough approximation)

        # Latitude: 1 degree ≈ 111,000 meters
        lat_degrees = 63.6 / 111000

        # Longitude varies by latitude: 1 degree longitude = cos(lat) * 111,000 meters
        lon_degrees = 63.6 / (111000 * math.cos(math.radians(latitude)))

        # Create square bounding box
        min_lat = latitude - lat_degrees / 2
        max_lat = latitude + lat_degrees / 2
        min_lon = longitude - lon_degrees / 2
        max_lon = longitude + lon_degrees / 2

        print(f"Generated ~1 acre bounding box:")
        print(f"  Center: {latitude:.6f}, {longitude:.6f}")
        print(f"  Bounds: {min_lon:.6f}, {min_lat:.6f}, {max_lon:.6f}, {max_lat:.6f}")
        print(
            f"  Size: ~{lat_degrees*111000:.1f}m x {lon_degrees*111000*math.cos(math.radians(latitude)):.1f}m"
        )

        return min_lon, min_lat, max_lon, max_lat

    def export_image(
        self,
        min_lon: float,
        min_lat: float,
        max_lon: float,
        max_lat: float,
        output_path: str,
        image_size: str = "5000,5000",
    ) -> Dict:
        """
        Export a clipped NAIP image from the USGS NAIPPlus service.

        Args:
            min_lon: Minimum longitude
            min_lat: Minimum latitude
            max_lon: Maximum longitude
            max_lat: Maximum latitude
            output_path: Path to save the exported image
            image_size: Image dimensions as "width,height" string

        Returns:
            Dictionary containing export metadata

        Raises:
            Exception: If export fails
        """
        bbox = ",".join(map(str, [min_lon, min_lat, max_lon, max_lat]))

        params = {
            "bbox": bbox,
            "bboxSR": 4326,  # WGS84 coordinate system
            "size": image_size,  # Output image size in pixels
            "imageSR": 4326,  # Output coordinate system
            "format": "tiff",  # Output format
            "f": "image",  # Response format
        }

        print(f"Exporting NAIP image:")
        print(f"  Bounding box: {bbox}")
        print(f"  Image size: {image_size}")
        print(f"  Output: {output_path}")

        try:
            # Make request to USGS NAIPPlus service
            response = requests.get(self.service_url, params=params, timeout=60)
            response.raise_for_status()

            # Check if response is actually an image
            content_type = response.headers.get("content-type", "")
            if not content_type.startswith("image/"):
                # Service might return JSON error
                try:
                    error_data = response.json()
                    raise Exception(f"Service error: {error_data}")
                except:
                    raise Exception(f"Unexpected response type: {content_type}")

            # Save the image
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(response.content)

            file_size = len(response.content)
            print(f"Successfully exported {file_size:,} bytes to: {output_path}")

            # Create metadata
            metadata = {
                "service": "USGS NAIPPlus",
                "service_url": self.service_url,
                "bbox": {
                    "min_longitude": min_lon,
                    "min_latitude": min_lat,
                    "max_longitude": max_lon,
                    "max_latitude": max_lat,
                },
                "bbox_string": bbox,
                "coordinate_system": "EPSG:4326",
                "image_size": image_size,
                "format": "tiff",
                "file_size_bytes": file_size,
                "export_time": datetime.now().isoformat(),
                "output_path": output_path,
            }

            return metadata

        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to export image from USGS service: {e}")
        except Exception as e:
            raise Exception(f"Image export failed: {e}")

    def save_metadata(
        self, metadata: Dict, output_dir: str, filename: str = "naip_metadata.json"
    ):
        """
        Save metadata to a JSON file.

        Args:
            metadata: Metadata dictionary to save
            output_dir: Directory to save the file
            filename: Name of the output file
        """
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)

        with open(filepath, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Metadata saved to: {filepath}")
        return filepath

    def get_orthophoto_for_address(
        self, address: str, output_dir: str = "../data", image_size: str = "5000,5000"
    ) -> Tuple[str, Dict]:
        """
        Get NAIP orthophoto for a given address using USGS NAIPPlus service.

        Args:
            address: Street address to geocode and find imagery for
            output_dir: Directory to save downloaded files
            image_size: Image dimensions as "width,height" string

        Returns:
            Tuple of (output_path, metadata)

        Raises:
            Exception: If geocoding or image export fails
        """
        # Geocode the address
        print(f"Processing address: {address}")
        geocoder = GeocodeUtils()
        latitude, longitude = geocoder.geocode_address(address)

        # Calculate acre-sized bounding box
        min_lon, min_lat, max_lon, max_lat = self.calculate_acre_bbox(
            latitude, longitude
        )

        # Generate safe filename
        safe_name = FileUtils.get_safe_filename(address)
        filename = f"naip_orthophoto_{safe_name}.tif"
        output_path = os.path.join(output_dir, filename)

        # Export the image
        metadata = self.export_image(
            min_lon, min_lat, max_lon, max_lat, output_path, image_size
        )

        # Add address info to metadata
        metadata["address"] = address
        metadata["geocoded_coordinates"] = {
            "latitude": latitude,
            "longitude": longitude,
        }

        # Save metadata
        self.save_metadata(metadata, output_dir)

        return output_path, metadata


def get_orthophoto_for_address(
    address: str, output_dir: str = "../data", image_size: str = "5000,5000"
) -> Tuple[str, Dict]:
    """
    Convenience function to get NAIP orthophoto for an address.

    Args:
        address: Street address to geocode and find imagery for
        output_dir: Directory to save downloaded files
        image_size: Image dimensions as "width,height" string

    Returns:
        Tuple of (output_path, metadata)
    """
    fetcher = NAIPFetcher()
    return fetcher.get_orthophoto_for_address(address, output_dir, image_size)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Fetch NAIP orthophoto for a given address using USGS NAIPPlus service"
    )
    parser.add_argument(
        "address",
        help="Street address to geocode and find imagery for (e.g., '1250 Wildwood Road, Boulder, CO')",
    )
    parser.add_argument(
        "--output-dir",
        default="../data",
        help="Directory to save downloaded files (default: ../data)",
    )
    parser.add_argument(
        "--image-size",
        default="5000,5000",
        help="Image size as 'width,height' (default: 5000,5000)",
    )

    args = parser.parse_args()

    try:
        output_path, metadata = get_orthophoto_for_address(
            args.address, args.output_dir, args.image_size
        )
        print(f"\nSuccess!")
        print(f"Image saved to: {output_path}")
        print(f"File size: {metadata.get('file_size_bytes', 0):,} bytes")
        bbox = metadata.get("bbox", {})
        print(
            f"Bounding box: {bbox.get('min_longitude', 'N/A'):.6f}, {bbox.get('min_latitude', 'N/A'):.6f} to {bbox.get('max_longitude', 'N/A'):.6f}, {bbox.get('max_latitude', 'N/A'):.6f}"
        )
        print(f"Image size: {metadata.get('image_size', 'N/A')}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
