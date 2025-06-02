#!/usr/bin/env python3
"""
NAIP Orthophoto Fetcher using Microsoft Planetary Computer STAC API

This module fetches NAIP (National Agriculture Imagery Program) orthophotos
for a given address using the Microsoft Planetary Computer STAC API.

Usage:
    from get_orthophoto import NAIPFetcher

    fetcher = NAIPFetcher()
    orthophoto_url, metadata = fetcher.get_orthophoto_for_address("1250 Wildwood Road, Boulder, CO")
"""

import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import requests
from utils import GeocodeUtils, BoundingBoxUtils, HTTPUtils, JSONUtils


class NAIPFetcher:
    """Class to fetch NAIP orthophotos using Microsoft Planetary Computer STAC API."""

    def __init__(self):
        """Initialize the NAIP fetcher."""
        self.stac_api_url = "https://planetarycomputer.microsoft.com/api/stac/v1"
        self.collection = "naip"

    def search_naip_items(
        self,
        latitude: float,
        longitude: float,
        bbox_size: float = 0.01,
        limit: int = 10,
    ) -> List[Dict]:
        """
        Search for NAIP items that intersect with the given coordinates.

        Args:
            latitude: Latitude in decimal degrees
            longitude: Longitude in decimal degrees
            bbox_size: Size of bounding box around the point (degrees)
            limit: Maximum number of items to return

        Returns:
            List of STAC items matching the search criteria

        Raises:
            Exception: If the API request fails
        """
        # Create bounding box around the point
        bbox_utils = BoundingBoxUtils()
        bbox = bbox_utils.create_bbox_from_point(latitude, longitude, bbox_size)

        search_params = {
            "collections": [self.collection],
            "bbox": bbox,
            "limit": limit,
            "sortby": [{"field": "datetime", "direction": "desc"}],
        }

        print(
            f"Searching for NAIP imagery at coordinates: {latitude:.6f}, {longitude:.6f}"
        )
        print(f"Using bounding box: {bbox}")

        try:
            search_results = HTTPUtils.post_json(
                f"{self.stac_api_url}/search", search_params, timeout=30
            )
            items = search_results.get("features", [])

            print(f"Found {len(items)} NAIP items")
            return items

        except Exception as e:
            raise Exception(f"Failed to search NAIP items: {e}")

    def get_best_item(self, items: List[Dict]) -> Optional[Dict]:
        """
        Select the best NAIP item from the search results.

        Prioritizes the most recent imagery with the highest resolution.

        Args:
            items: List of STAC items

        Returns:
            The best item, or None if no suitable item found
        """
        if not items:
            return None

        # Sort by date (most recent first)
        sorted_items = sorted(
            items,
            key=lambda x: x.get("properties", {}).get("datetime", ""),
            reverse=True,
        )

        return sorted_items[0]

    def get_download_url(self, item: Dict) -> str:
        """
        Get the download URL for the orthophoto image.

        Args:
            item: STAC item containing asset information

        Returns:
            URL to download the orthophoto

        Raises:
            Exception: If no suitable image asset is found
        """
        assets = item.get("assets", {})

        # Look for the main image asset (usually 'image' or 'rendered_preview')
        preferred_assets = ["image", "rendered_preview", "visual"]

        for asset_key in preferred_assets:
            if asset_key in assets:
                asset = assets[asset_key]
                if "href" in asset:
                    return asset["href"]

        # If no preferred asset found, try the first available image asset
        for asset_key, asset in assets.items():
            if asset.get("type", "").startswith("image/"):
                if "href" in asset:
                    return asset["href"]

        raise Exception("No suitable image asset found in STAC item")

    def extract_metadata(self, item: Dict) -> Dict:
        """
        Extract useful metadata from a STAC item.

        Args:
            item: STAC item

        Returns:
            Dictionary containing extracted metadata
        """
        properties = item.get("properties", {})

        metadata = {
            "id": item.get("id"),
            "datetime": properties.get("datetime"),
            "collection": item.get("collection"),
            "gsd": properties.get("gsd"),  # Ground sample distance
            "proj:epsg": properties.get("proj:epsg"),
            "naip:year": properties.get("naip:year"),
            "naip:state": properties.get("naip:state"),
            "platform": properties.get("platform"),
            "instruments": properties.get("instruments"),
            "bbox": item.get("bbox"),
            "geometry": item.get("geometry"),
        }

        # Clean up None values
        metadata = {k: v for k, v in metadata.items() if v is not None}

        return metadata

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
        filepath = JSONUtils.save_metadata(metadata, output_dir, filename)
        print(f"Metadata saved to: {filepath}")

    def download_orthophoto(self, url: str, output_path: str):
        """
        Download the orthophoto from the given URL.

        Args:
            url: URL to download from
            output_path: Local path to save the file

        Raises:
            Exception: If download fails
        """
        print(f"Downloading orthophoto from: {url}")
        print(f"Saving to: {output_path}")

        try:
            HTTPUtils.download_file(url, output_path, timeout=60)
            print(f"Successfully downloaded orthophoto to: {output_path}")

        except Exception as e:
            raise Exception(f"Failed to download orthophoto: {e}")

    def get_orthophoto_for_address(
        self, address: str, output_dir: str = "../data", download: bool = True
    ) -> Tuple[str, Dict]:
        """
        Get NAIP orthophoto for a given address.

        Args:
            address: Street address to geocode and find imagery for
            output_dir: Directory to save downloaded files
            download: Whether to download the actual image file

        Returns:
            Tuple of (download_url, metadata)

        Raises:
            Exception: If geocoding or NAIP search fails
        """
        # Geocode the address
        print(f"Processing address: {address}")
        geocoder = GeocodeUtils()
        latitude, longitude = geocoder.geocode_address(address)

        # Search for NAIP items
        items = self.search_naip_items(latitude, longitude)

        if not items:
            raise Exception(
                f"No NAIP imagery found for coordinates: {latitude:.6f}, {longitude:.6f}"
            )

        # Get the best item
        best_item = self.get_best_item(items)
        if not best_item:
            raise Exception("No suitable NAIP item found")

        # Extract metadata
        metadata = self.extract_metadata(best_item)
        print(f"Selected NAIP image from {metadata.get('naip:year', 'unknown year')}")

        # Get download URL
        download_url = self.get_download_url(best_item)

        # Save metadata
        self.save_metadata(metadata, output_dir)

        # Download image if requested
        if download:
            filename = f"naip_orthophoto_{metadata.get('id', 'unknown')}.tif"
            output_path = os.path.join(output_dir, filename)
            self.download_orthophoto(download_url, output_path)

        return download_url, metadata


def get_orthophoto_for_address(
    address: str, output_dir: str = "../data"
) -> Tuple[str, Dict]:
    """
    Convenience function to get NAIP orthophoto for an address.

    Args:
        address: Street address to geocode and find imagery for
        output_dir: Directory to save downloaded files

    Returns:
        Tuple of (download_url, metadata)
    """
    fetcher = NAIPFetcher()
    return fetcher.get_orthophoto_for_address(address, output_dir)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python get_orthophoto.py '<address>'")
        print("Example: python get_orthophoto.py '1250 Wildwood Road, Boulder, CO'")
        sys.exit(1)

    address = sys.argv[1]
    try:
        url, metadata = get_orthophoto_for_address(address)
        print(f"\nSuccess!")
        print(f"Download URL: {url}")
        print(f"Image year: {metadata.get('naip:year', 'Unknown')}")
        print(f"State: {metadata.get('naip:state', 'Unknown')}")
        print(f"Ground sample distance: {metadata.get('gsd', 'Unknown')} meters")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
