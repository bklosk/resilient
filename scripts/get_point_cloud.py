#!/usr/bin/env python3
"""
Improved Point Cloud Data Retrieval Script

This script efficiently finds and downloads USGS 3DEP LiDAR point cloud data
for any given address using a spatial index approach. The script geocodes addresses,
finds intersecting datasets, and provides options to list or download EPT data.

Usage:
    python get_point_cloud.py "1250 Wildwood Road, Boulder, CO" --list-only
    python get_point_cloud.py "1250 Wildwood Road, Boulder, CO" --download
"""

import requests
import json
import os
import argparse
import logging
from typing import List, Dict, Optional, Tuple
from pyproj import Transformer

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PointCloudDatasetFinder:
    def __init__(self, spatial_index_path: str = None):
        # Use centralized utilities
        from utils import GeocodeUtils, BoundingBoxUtils, S3Utils, JSONUtils

        self.geocode_utils = GeocodeUtils()
        self.bbox_utils = BoundingBoxUtils()
        self.s3_utils = S3Utils()
        self.json_utils = JSONUtils()

        self.bucket_name = "usgs-lidar-public"
        # Use centralized S3 client
        self.s3_client = self.s3_utils.get_client()

        # Default spatial index path
        if spatial_index_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            self.spatial_index_path = os.path.join(
                os.path.dirname(script_dir), "data", "spatial_index.json"
            )
        else:
            self.spatial_index_path = spatial_index_path

        # Set up coordinate transformer (WGS84 to Web Mercator)
        self.transformer = Transformer.from_crs(
            "EPSG:4326", "EPSG:3857", always_xy=True
        )

        # Load spatial index
        self.datasets = self._load_spatial_index()

    def _load_spatial_index(self) -> List[Dict]:
        """Load the pre-computed spatial index of dataset bounds."""
        try:
            logger.info(f"Loading spatial index from: {self.spatial_index_path}")
            data = self.json_utils.load_json(self.spatial_index_path)
            datasets = data.get("datasets", [])
            logger.info(f"Loaded spatial index with {len(datasets)} datasets")
            return datasets
        except Exception as e:
            raise RuntimeError(f"Failed to load spatial index: {e}")

    def geocode_address(self, address: str) -> Tuple[float, float]:
        """Convert an address to lat/lon coordinates using centralized utils."""
        try:
            lat, lon = self.geocode_utils.geocode_address(address)
            return lat, lon
        except ValueError as e:
            raise ValueError(f"Geocoding failed: {e}")

    def find_datasets_for_location(self, lat: float, lon: float) -> List[Dict]:
        """Find USGS 3DEP datasets that contain the given coordinates using spatial index."""
        logger.info(f"Searching for datasets covering location: {lat}, {lon}")

        # Transform coordinates to Web Mercator (EPSG:3857)
        x, y = self.transformer.transform(lon, lat)
        logger.info(f"Transformed coordinates: {x}, {y} (Web Mercator)")

        matching_datasets = []

        for dataset in self.datasets:
            try:
                bounds = dataset.get("bounds", [])
                if len(bounds) >= 4:  # Ensure we have at least 2D bounds
                    # Bounds format: [xmin, ymin, zmin, xmax, ymax, zmax] or [xmin, ymin, xmax, ymax]
                    xmin, ymin = bounds[0], bounds[1]
                    if len(bounds) == 6:  # 3D bounds
                        xmax, ymax = bounds[3], bounds[4]
                    else:  # 2D bounds
                        xmax, ymax = bounds[2], bounds[3]

                    # Check if point is within bounds (with small buffer for edge cases)
                    buffer = 100  # 100 meter buffer in Web Mercator
                    if (
                        xmin - buffer <= x <= xmax + buffer
                        and ymin - buffer <= y <= ymax + buffer
                    ):

                        matching_datasets.append(dataset)
                        logger.info(
                            f"Dataset '{dataset['name']}' covers the location (points: {dataset.get('points', 'unknown')})"
                        )

            except Exception as e:
                logger.warning(
                    f"Could not check bounds for dataset '{dataset.get('name', 'unknown')}': {e}"
                )
                continue

        logger.info(f"Found {len(matching_datasets)} datasets covering the location")
        return matching_datasets

    def get_ept_metadata(self, dataset_name: str) -> Optional[Dict]:
        """Get EPT metadata for a specific dataset from S3."""
        try:
            # EPT metadata is stored as ept.json in the dataset folder
            ept_json_key = f"{dataset_name}/ept.json"
            logger.info(f"Fetching EPT metadata from S3: {ept_json_key}")

            response = self.s3_client.get_object(
                Bucket=self.bucket_name, Key=ept_json_key
            )
            metadata = json.loads(response["Body"].read().decode("utf-8"))
            logger.info(f"Retrieved EPT metadata for dataset: {dataset_name}")
            return metadata

        except self.s3_client.exceptions.NoSuchKey:
            logger.warning(f"EPT metadata not found for dataset: {dataset_name}")
            return None
        except Exception as e:
            logger.warning(f"Failed to get EPT metadata for {dataset_name}: {e}")
            return None

    def download_dataset(
        self, dataset_name: str, output_dir: str = "downloads"
    ) -> bool:
        """Download EPT point cloud data for a dataset from S3."""
        try:
            # Create output directory
            dataset_dir = os.path.join(output_dir, dataset_name.replace("/", "_"))
            os.makedirs(dataset_dir, exist_ok=True)

            # First, get and save EPT metadata
            logger.info(f"Downloading EPT metadata for {dataset_name}")
            metadata = self.get_ept_metadata(dataset_name)
            if not metadata:
                logger.error(f"Could not retrieve metadata for {dataset_name}")
                return False

            # Save metadata using centralized JSON utilities
            metadata_path = self.json_utils.save_metadata(
                metadata, dataset_dir, "ept.json"
            )
            logger.info(f"Saved EPT metadata to: {metadata_path}")

            # Get hierarchy information to understand data structure
            hierarchy_key = f"{dataset_name}/ept-hierarchy/0-0-0-0.json"
            try:
                response = self.s3_client.get_object(
                    Bucket=self.bucket_name, Key=hierarchy_key
                )
                hierarchy = json.loads(response["Body"].read().decode("utf-8"))

                hierarchy_path = self.json_utils.save_metadata(
                    hierarchy, dataset_dir, "hierarchy.json"
                )
                logger.info(f"Saved hierarchy data to: {hierarchy_path}")

            except Exception as e:
                logger.warning(f"Could not download hierarchy data: {e}")

            # Download some sample EPT data tiles
            # Start with the root tile and a few high-level tiles
            sample_tiles = [
                "0-0-0-0.laz",  # Root tile
                "1-0-0-0.laz",  # Level 1 tiles
                "1-0-0-1.laz",
                "1-0-1-0.laz",
                "1-0-1-1.laz",
                "1-1-0-0.laz",
                "1-1-0-1.laz",
                "1-1-1-0.laz",
                "1-1-1-1.laz",
            ]

            downloaded_count = 0
            max_downloads = 5  # Limit downloads for testing

            for tile in sample_tiles:
                if downloaded_count >= max_downloads:
                    break

                tile_key = f"{dataset_name}/ept-data/{tile}"
                tile_path = os.path.join(dataset_dir, tile)

                try:
                    logger.info(f"Downloading tile: {tile}")
                    self.s3_client.download_file(self.bucket_name, tile_key, tile_path)

                    file_size = os.path.getsize(tile_path)
                    logger.info(f"Downloaded {tile} ({file_size:,} bytes)")
                    downloaded_count += 1

                except Exception as e:
                    logger.warning(f"Could not download tile {tile}: {e}")
                    continue

            logger.info(
                f"Successfully downloaded {downloaded_count} data tiles to: {dataset_dir}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to download dataset {dataset_name}: {e}")
            return False

    def list_available_datasets(self, datasets: List[Dict]) -> None:
        """Display information about available datasets."""
        if not datasets:
            print("No datasets found for the specified location.")
            return

        print(f"\nFound {len(datasets)} datasets covering the location:")
        print("=" * 80)

        # Sort datasets by point count (descending)
        sorted_datasets = sorted(
            datasets, key=lambda x: x.get("points", 0), reverse=True
        )

        for i, dataset in enumerate(sorted_datasets, 1):
            name = dataset.get("name", "Unknown")
            points = dataset.get("points", 0)
            bounds = dataset.get("bounds", [])

            print(f"{i:2d}. {name}")
            print(f"    Points: {points:,}")

            if bounds:
                if len(bounds) == 6:
                    xmin, ymin, zmin, xmax, ymax, zmax = bounds
                    print(
                        f"    Bounds: X[{xmin:.0f}, {xmax:.0f}] Y[{ymin:.0f}, {ymax:.0f}] Z[{zmin:.1f}, {zmax:.1f}]"
                    )
                else:
                    print(f"    Bounds: {bounds}")

            # Show coordinate system info if available
            srs = dataset.get("srs", {})
            if srs:
                authority = srs.get("authority", "Unknown")
                horizontal = srs.get("horizontal", "Unknown")
                print(f"    CRS: {authority}:{horizontal}")

            print()

    def select_best_dataset(self, datasets: List[Dict], address: str) -> Dict:
        """Select the most recent dataset, preferring in-state options."""
        state = address.strip()[-2:].upper()  # Last 2 chars as state

        # Sort by: in-state first, then by year (most recent), then by points
        def sort_key(d):
            name = d.get("name", "")
            in_state = state in name.upper()
            year = max(
                [int(x) for x in name.split("_") if x.isdigit() and len(x) == 4],
                default=0,
            )
            return (not in_state, -year, -d.get("points", 0))

        return sorted(datasets, key=sort_key)[0]

    def generate_bounding_box(
        self, lat: float, lon: float, buffer_km: float = 1.0
    ) -> str:
        """Generate a bounding box around coordinates for LiDAR search using centralized utils."""
        return self.bbox_utils.generate_bounding_box(lat, lon, buffer_km)

    def search_lidar_products(self, bbox: str) -> List[Dict]:
        """Search for LiDAR products within a bounding box.

        Args:
            bbox: Bounding box string "min_lon,min_lat,max_lon,max_lat"

        Returns:
            List of matching datasets
        """
        try:
            bbox_parts = bbox.split(",")
            if len(bbox_parts) != 4:
                raise ValueError(f"Invalid bounding box format: {bbox}")

            min_lon, min_lat, max_lon, max_lat = map(float, bbox_parts)
            logger.info(f"Searching for LiDAR products in bbox: {bbox}")

            # Find center point for dataset search
            center_lat = (min_lat + max_lat) / 2
            center_lon = (min_lon + max_lon) / 2

            # Use existing find_datasets_for_location method
            datasets = self.find_datasets_for_location(center_lat, center_lon)

            logger.info(f"Found {len(datasets)} LiDAR products")
            return datasets

        except Exception as e:
            logger.error(f"Error searching LiDAR products: {e}")
            return []

    def filter_laz_products(self, products: List[Dict]) -> List[Dict]:
        """Filter products to only include LAZ format data.

        Args:
            products: List of LiDAR products

        Returns:
            Filtered list containing only LAZ products
        """
        # For now, assume all products support LAZ format since USGS 3DEP data is available as LAZ
        # In a more complete implementation, this would check format availability
        laz_products = [p for p in products if p.get("name")]  # Basic validation

        logger.info(
            f"Filtered to {len(laz_products)} LAZ products from {len(products)} total"
        )
        return laz_products

    def download_point_cloud(self, product: Dict, output_dir: str) -> str:
        """Download point cloud data for a specific product.

        Args:
            product: Product dictionary containing dataset information
            output_dir: Directory to save downloaded files

        Returns:
            Path to downloaded LAZ file
        """
        try:
            dataset_name = product.get("name")
            if not dataset_name:
                raise ValueError("Product missing name field")

            logger.info(f"Downloading point cloud data for: {dataset_name}")

            # Create output directory
            os.makedirs(output_dir, exist_ok=True)

            # For now, use the existing download_dataset method
            # In production, this would implement more targeted LAZ file download
            success = self.download_dataset(dataset_name, output_dir)

            if success:
                # Look for downloaded LAZ files
                dataset_dir = os.path.join(output_dir, dataset_name.replace("/", "_"))
                laz_files = []
                if os.path.exists(dataset_dir):
                    for root, dirs, files in os.walk(dataset_dir):
                        laz_files.extend(
                            [os.path.join(root, f) for f in files if f.endswith(".laz")]
                        )

                if laz_files:
                    # Return the first LAZ file found
                    downloaded_file = laz_files[0]
                    logger.info(f"Downloaded point cloud: {downloaded_file}")
                    return downloaded_file
                else:
                    raise RuntimeError("No LAZ files found after download")
            else:
                raise RuntimeError("Dataset download failed")

        except Exception as e:
            logger.error(f"Error downloading point cloud: {e}")
            raise


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Find and download USGS 3DEP LiDAR point cloud data for an address"
    )
    parser.add_argument(
        "address",
        help="Address to search for (e.g., '1250 Wildwood Road, Boulder, CO')",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only list available datasets, don't download",
    )
    parser.add_argument(
        "--download", action="store_true", help="Download the first available dataset"
    )
    parser.add_argument(
        "--output-dir",
        default="downloads",
        help="Directory to save downloaded files (default: downloads)",
    )
    parser.add_argument(
        "--spatial-index",
        default=None,
        help="Path to spatial index JSON file (default: ../data/spatial_index.json)",
    )

    args = parser.parse_args()

    try:
        # Initialize the dataset finder
        finder = PointCloudDatasetFinder(spatial_index_path=args.spatial_index)

        # Geocode the address
        lat, lon = finder.geocode_address(args.address)

        # Find datasets covering this location
        datasets = finder.find_datasets_for_location(lat, lon)

        if not datasets:
            print("No LiDAR datasets found covering this location.")
            return

        # List available datasets
        finder.list_available_datasets(datasets)

        # Download if requested
        if args.download and not args.list_only:
            if datasets:
                best_dataset = finder.select_best_dataset(datasets, args.address)
                dataset_name = best_dataset["name"]
                print(f"\nSelected best dataset: {dataset_name}")

                success = finder.download_dataset(dataset_name, args.output_dir)
                if success:
                    print(f"Download completed successfully!")
                    print(
                        f"Files saved to: {args.output_dir}/{dataset_name.replace('/', '_')}"
                    )
                else:
                    print("Download failed. Check the logs for details.")
            else:
                print("No datasets available to download.")
        elif not args.list_only and not args.download:
            print("\nUse --download to download the first dataset")
            print("Use --list-only to only show available datasets")

    except Exception as e:
        logger.error(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
