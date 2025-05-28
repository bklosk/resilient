#!/usr/bin/env python3
"""
Point Cloud Fetcher Script

This script accepts an address, converts it to coordinates using geopy,
generates a bounding box, and fetches LiDAR point cloud (LPC) LAZ datasets
from the USGS National Map API.

Usage:
    python get_point_cloud.py "1250 Wildwood Road, Boulder, CO"
"""

import sys
import argparse
import json
import os
import math
from typing import Tuple, List, Dict, Optional
import requests
from geocode import Geocoder


class PointCloudFetcher:
    """Class to handle LiDAR data fetching."""

    def __init__(self):
        """Initialize the fetcher."""
        self.api_base_url = "https://tnmaccess.nationalmap.gov/api/v1/products"

    def generate_bounding_box(
        self, lat: float, lon: float, buffer_km: float = 1.0
    ) -> str:
        """
        Generate a bounding box around the given coordinates.

        Args:
            lat: Latitude in decimal degrees
            lon: Longitude in decimal degrees
            buffer_km: Buffer distance in kilometers (default 1km)

        Returns:
            Comma-delimited string: "west,south,east,north"
        """
        # Approximate degrees per kilometer
        # 1 degree latitude ≈ 111 km
        # 1 degree longitude ≈ 111 km * cos(latitude)
        lat_buffer = buffer_km / 111.0
        lon_buffer = buffer_km / (111.0 * abs(math.cos(math.radians(lat))))

        west = lon - lon_buffer
        east = lon + lon_buffer
        south = lat - lat_buffer
        north = lat + lat_buffer

        bbox = f"{west:.6f},{south:.6f},{east:.6f},{north:.6f}"
        print(f"Generated bounding box: {bbox}")
        return bbox

    def search_lidar_products(self, bbox: str) -> List[Dict]:
        """
        Search for LiDAR products in the given bounding box.

        Args:
            bbox: Bounding box as "west,south,east,north"

        Returns:
            List of product dictionaries
        """
        params = {
            "datasets": "Lidar Point Cloud (LPC)",
            "bbox": bbox,
            "format": "JSON",
            "max": 50,  # Limit results
        }

        print(f"Searching for LiDAR products in bbox: {bbox}")

        try:
            response = requests.get(self.api_base_url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            if "items" not in data:
                print("No 'items' field in response")
                return []

            products = data["items"]
            print(f"Found {len(products)} LiDAR products")

            return products

        except requests.exceptions.RequestException as e:
            print(f"Error searching for products: {e}")
            return []

    def filter_laz_products(self, products: List[Dict]) -> List[Dict]:
        """
        Filter products to only include LAZ files.

        Args:
            products: List of product dictionaries

        Returns:
            List of LAZ product dictionaries
        """
        laz_products = []

        for product in products:
            # Check if the download URL contains .laz
            download_url = product.get("downloadURL", "")
            if download_url.lower().endswith(".laz"):
                laz_products.append(product)

        print(f"Filtered to {len(laz_products)} LAZ products")
        return laz_products

    def download_point_cloud(
        self, product: Dict, output_dir: str = "data"
    ) -> Optional[str]:
        """
        Download a point cloud file.

        Args:
            product: Product dictionary containing download information
            output_dir: Directory to save the file

        Returns:
            Path to downloaded file or None if failed
        """
        download_url = product.get("downloadURL")
        if not download_url:
            print("No download URL found in product")
            return None

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Extract filename from URL
        filename = download_url.split("/")[-1]
        if not filename.endswith(".laz"):
            filename += ".laz"

        output_path = os.path.join(output_dir, filename)

        print(f"Downloading {filename}...")
        print(f"URL: {download_url}")

        try:
            response = requests.get(download_url, stream=True, timeout=60)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))

            with open(output_path, "wb") as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(f"\rProgress: {percent:.1f}%", end="", flush=True)

            print(f"\nSuccessfully downloaded: {output_path}")
            return output_path

        except requests.exceptions.RequestException as e:
            print(f"Error downloading file: {e}")
            return None

    def print_product_info(self, products: List[Dict]) -> None:
        """Print information about found products."""
        if not products:
            print("No products found")
            return

        print(f"\nFound {len(products)} products:")
        print("-" * 80)

        for i, product in enumerate(products, 1):
            title = product.get("title", "Unknown Title")
            date = product.get("dateCreated", "Unknown Date")
            size = product.get("sizeInBytes", 0)
            size_mb = size / (1024 * 1024) if size else 0
            url = product.get("downloadURL", "No URL")

            print(f"{i}. {title}")
            print(f"   Date: {date}")
            print(f"   Size: {size_mb:.1f} MB")
            print(f"   URL: {url}")
            print()


def main():
    """Main function to run the point cloud fetcher."""
    parser = argparse.ArgumentParser(
        description="Fetch LiDAR point cloud data for an address"
    )
    parser.add_argument("address", help="Address to geocode and fetch data for")
    parser.add_argument(
        "--buffer",
        type=float,
        default=1.0,
        help="Buffer distance in kilometers (default: 1.0)",
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Output directory for downloaded files (default: data)",
    )
    parser.add_argument(
        "--download", action="store_true", help="Download the first available LAZ file"
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only list available products, do not download",
    )

    args = parser.parse_args()

    try:
        # Initialize the geocoder
        geocoder = Geocoder()

        # Initialize the fetcher
        fetcher = PointCloudFetcher()

        # Geocode the address
        lat, lon = geocoder.geocode_address(args.address)

        # Generate bounding box
        bbox = fetcher.generate_bounding_box(lat, lon, args.buffer)

        # Search for LiDAR products
        products = fetcher.search_lidar_products(bbox)

        if not products:
            print("No LiDAR products found for this location")
            return

        # Filter to LAZ products
        laz_products = fetcher.filter_laz_products(products)

        if not laz_products:
            print("No LAZ products found for this location")
            # Show all products for reference
            fetcher.print_product_info(products)
            return

        # Print product information
        fetcher.print_product_info(laz_products)

        # Download if requested
        if args.download and not args.list_only:
            if laz_products:
                print("Downloading first LAZ product...")
                downloaded_file = fetcher.download_point_cloud(
                    laz_products[0], args.output_dir
                )
                if downloaded_file:
                    print(f"Point cloud data saved to: {downloaded_file}")
                else:
                    print("Download failed")
        elif not args.list_only:
            print("\nUse --download flag to download the first LAZ file")
            print("Use --list-only flag to only show available products")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
