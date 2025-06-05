#!/usr/bin/env python3
"""
Data Fetching Module

Handles downloading point cloud and orthophoto data from external sources.
"""

import logging
import time
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class DataFetcher:
    """Handles fetching point cloud and orthophoto data."""

    def __init__(self, output_dir: str):
        """
        Initialize data fetcher.

        Args:
            output_dir: Directory for downloaded files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def fetch_point_cloud_data(
        self,
        pc_fetcher,  # PointCloudDatasetFinder
        lat: float,
        lon: float,
        ortho_bounds: Optional[Dict] = None,
        ortho_crs: Optional[str] = None,
    ) -> str:
        """
        Fetch point cloud data for given coordinates with retry logic.

        Args:
            pc_fetcher: Point cloud fetcher instance
            lat: Latitude
            lon: Longitude
            ortho_bounds: Optional orthophoto bounds for better dataset selection
            ortho_crs: Optional orthophoto CRS

        Returns:
            Path to downloaded point cloud file
        """
        logger.info("Fetching point cloud data...")

        try:
            # Generate bounding box for point cloud search
            bbox = pc_fetcher.generate_bounding_box(lat, lon, buffer_km=1.0)
            logger.info(f"Search area: {bbox}")

            # Search for point cloud data
            logger.info("Searching for LiDAR data...")
            products = pc_fetcher.search_lidar_products(bbox)

            if not products:
                logger.error(
                    f"No LiDAR data found for coordinates {lat:.6f}, {lon:.6f}"
                )
                raise RuntimeError(
                    f"No LiDAR data found for location {lat:.6f}, {lon:.6f}. This area may not have available point cloud data."
                )

            logger.info(f"Found {len(products)} LiDAR products")

            laz_products = pc_fetcher.filter_laz_products(products)

            if not laz_products:
                logger.error("No LAZ format LiDAR data found")
                raise RuntimeError(
                    "No LAZ format LiDAR data found. Only LAZ files are supported for processing."
                )

            logger.info(f"Found {len(laz_products)} LAZ products")

            # Select the best dataset using improved selection logic
            logger.info("Selecting best dataset based on location and recency...")
            if ortho_bounds and ortho_crs:
                logger.info("Using orthophoto-aware dataset selection...")
                best_product = pc_fetcher.select_best_dataset_for_orthophoto(
                    laz_products, ortho_bounds, ortho_crs, lat, lon
                )
            else:
                best_product = pc_fetcher.select_best_dataset_for_location(
                    laz_products, lat, lon
                )

            logger.info(f"Selected dataset: {best_product.get('name', 'Unknown')}")

            # Try downloading point cloud with retry logic
            max_retries = 3
            retry_delay = 5  # seconds

            for attempt in range(max_retries):
                try:
                    product_title = best_product.get(
                        "title", best_product.get("name", "Unknown product")
                    )
                    logger.info(
                        f"Attempting to download (attempt {attempt + 1}/{max_retries}): {product_title}"
                    )

                    downloaded_pc = pc_fetcher.download_point_cloud(
                        best_product, str(self.output_dir), ortho_bounds, ortho_crs
                    )

                    if downloaded_pc and Path(downloaded_pc).exists():
                        logger.info(
                            f"Point cloud downloaded successfully: {downloaded_pc}"
                        )
                        return downloaded_pc
                    else:
                        logger.warning(
                            f"Download attempt {attempt + 1} failed - file not created"
                        )

                except Exception as download_error:
                    logger.warning(
                        f"Download attempt {attempt + 1} failed: {str(download_error)}"
                    )

                    # Check if it's a timeout or connection error
                    if any(
                        keyword in str(download_error).lower()
                        for keyword in [
                            "timeout",
                            "connection",
                            "max retries",
                            "httpsconnectionpool",
                        ]
                    ):
                        logger.info(
                            f"Network error detected, will retry in {retry_delay} seconds..."
                        )
                        if attempt < max_retries - 1:
                            time.sleep(retry_delay)
                            retry_delay *= 2  # Exponential backoff
                            continue

                    # If it's the last attempt or not a network error, raise
                    if attempt == max_retries - 1:
                        raise

            # If we get here, all retries failed
            raise RuntimeError(
                f"Failed to download point cloud after {max_retries} attempts. "
                "The USGS server may be experiencing issues. Please try again later."
            )

        except Exception as e:
            logger.error(f"Point cloud fetch failed: {str(e)}")
            # Re-raise with more context
            if any(
                msg in str(e)
                for msg in ["No LiDAR data found", "No LAZ format", "after", "attempts"]
            ):
                raise  # Re-raise our custom error messages as-is
            else:
                raise RuntimeError(f"Failed to fetch point cloud data: {str(e)}")

    def fetch_orthophoto_data(
        self, ortho_fetcher, address: str, lat: float, lon: float
    ) -> str:
        """
        Fetch orthophoto data for given address/coordinates with improved error handling.

        Args:
            ortho_fetcher: Orthophoto fetcher instance
            address: Street address (fallback)
            lat: Latitude
            lon: Longitude

        Returns:
            Path to downloaded orthophoto file
        """
        logger.info("Fetching orthophoto data...")

        try:
            # Try fetching orthophoto using coordinates
            ortho_path, ortho_metadata = ortho_fetcher.get_orthophoto_for_address(
                address, str(self.output_dir)
            )

            logger.info(f"Orthophoto downloaded: {ortho_path}")

            # Verify the file exists and is valid
            if Path(ortho_path).exists():
                try:
                    import rasterio

                    with rasterio.open(ortho_path) as test_ds:
                        logger.info(
                            f"Orthophoto validation successful: {test_ds.width}x{test_ds.height}"
                        )
                        return ortho_path
                except Exception as e:
                    logger.warning(f"Downloaded orthophoto failed validation: {e}")
                    # File exists but is invalid, try to find alternative
            else:
                logger.warning(f"Expected orthophoto file not found: {ortho_path}")

            # If the expected file doesn't exist or is invalid, search for any orthophoto in the directory
            output_dir = Path(self.output_dir)
            if output_dir.exists():
                logger.info("Searching for alternative orthophoto files...")

                # Look for any TIFF files that might be orthophotos
                tiff_patterns = ["*.tif", "*.tiff", "*naip*.tif", "*orthophoto*.tif"]
                found_files = []

                for pattern in tiff_patterns:
                    found_files.extend(list(output_dir.glob(pattern)))

                for candidate_file in found_files:
                    try:
                        import rasterio

                        with rasterio.open(str(candidate_file)) as test_ds:
                            if test_ds.width > 0 and test_ds.height > 0:
                                logger.info(
                                    f"Found alternative valid orthophoto: {candidate_file}"
                                )
                                return str(candidate_file)
                    except Exception:
                        continue

            # If we still haven't found a valid file, raise an error
            raise RuntimeError(f"No valid orthophoto file found after download attempt")

        except Exception as e:
            logger.error(f"Failed to fetch orthophoto: {e}")
            raise
