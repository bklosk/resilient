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
import math
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

    def _score_dataset(
        self,
        dataset: Dict,
        coords: Tuple[float, float],
        extras: Dict,
    ) -> Dict:
        """Calculate common scoring metrics for a dataset."""
        lat, lon = coords
        name = dataset.get("name", "")
        bounds = dataset.get("bounds", [])
        points = dataset.get("points", 0)

        distance_score = float("inf")
        dataset_area = 0
        target_coords = extras.get("target_coords")
        if target_coords and bounds and len(bounds) >= 4:
            if len(bounds) == 6:
                xmin, ymin, xmax, ymax = bounds[0], bounds[1], bounds[3], bounds[4]
            else:
                xmin, ymin, xmax, ymax = bounds[0], bounds[1], bounds[2], bounds[3]

            center_x = (xmin + xmax) / 2
            center_y = (ymin + ymax) / 2
            dataset_area = (xmax - xmin) * (ymax - ymin)
            tx, ty = target_coords
            distance_score = math.sqrt((tx - center_x) ** 2 + (ty - center_y) ** 2)

        year = 0
        for part in name.split("_"):
            if part.isdigit() and len(part) == 4:
                try:
                    year = int(part)
                    break
                except ValueError:
                    continue

        recency_score = year if year > 0 else 1900
        quality_score = math.log10(max(points, 1))

        specificity_bonus = 0
        if dataset_area > 0:
            area_km2 = dataset_area / 1_000_000
            specificity_bonus = max(0, 100 - math.log10(max(area_km2, 1)) * 20)

        region_bonus = 0
        name_upper = name.upper()
        front_range_keywords = [
            "DRCOG",
            "DENVER",
            "METRO",
            "FRONT",
            "BOULDER",
            "JEFFCO",
            "ADAMS",
        ]
        broad_keywords = [
            "NWCO",
            "SWCO",
            "NECO",
            "SECO",
            "CENTRAL",
            "WESTERN",
            "EASTERN",
            "NORTHERN",
            "SOUTHERN",
        ]

        if 39.5 <= lat <= 40.5 and -105.5 <= lon <= -104.5:
            if any(k in name_upper for k in front_range_keywords):
                region_bonus = 500
            elif any(k in name_upper for k in broad_keywords):
                region_bonus = -200

        state_bonus = 0
        if 39 <= lat <= 41 and -109 <= lon <= -102:
            if "CO" in name_upper:
                state_bonus = 300

        return {
            "dataset": dataset,
            "name": name,
            "points": points,
            "distance_score": distance_score,
            "dataset_area": dataset_area,
            "year": year,
            "recency_score": recency_score,
            "quality_score": quality_score,
            "specificity_bonus": specificity_bonus,
            "region_bonus": region_bonus,
            "state_bonus": state_bonus,
        }

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

    def _prepare_download(
        self, dataset_name: str, output_dir: str
    ) -> Tuple[str, Optional[Dict]]:
        """Create dataset directory and fetch EPT metadata."""
        dataset_dir = os.path.join(output_dir, dataset_name.replace("/", "_"))
        os.makedirs(dataset_dir, exist_ok=True)

        logger.info(f"Downloading EPT metadata for {dataset_name}")
        metadata = self.get_ept_metadata(dataset_name)
        if not metadata:
            logger.error(f"Could not retrieve metadata for {dataset_name}")
            return dataset_dir, None

        metadata_path = self.json_utils.save_metadata(metadata, dataset_dir, "ept.json")
        logger.info(f"Saved EPT metadata to: {metadata_path}")

        return dataset_dir, metadata

    def download_dataset(
        self, dataset_name: str, output_dir: str = "downloads"
    ) -> bool:
        """Download EPT point cloud data for a dataset from S3."""
        try:
            dataset_dir, metadata = self._prepare_download(dataset_name, output_dir)
            if not metadata:
                return False

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

            # Load hierarchy to get actual available tiles
            hierarchy = {}
            hierarchy_key = f"{dataset_name}/ept-hierarchy/0-0-0-0.json"
            try:
                response = self.s3_client.get_object(
                    Bucket=self.bucket_name, Key=hierarchy_key
                )
                hierarchy = json.loads(response["Body"].read().decode("utf-8"))
                logger.info(f"Loaded hierarchy with {len(hierarchy)} tiles")
            except Exception as e:
                logger.warning(f"Could not download hierarchy data: {e}")
                # Fall back to simple tile list if hierarchy not available
                sample_tiles = ["0-0-0-0.laz"]
                hierarchy = {"0-0-0-0": 1000}  # Dummy entry

            # Select best tiles from available hierarchy, prioritizing higher levels with good point counts
            if hierarchy:
                # Group tiles by level
                tiles_by_level = {}
                for tile_name, point_count in hierarchy.items():
                    parts = tile_name.split("-")
                    if len(parts) >= 1:
                        try:
                            level = int(parts[0])
                            if level not in tiles_by_level:
                                tiles_by_level[level] = []
                            tiles_by_level[level].append((tile_name, point_count))
                        except ValueError:
                            continue

                # Select tiles from different levels, prioritizing higher levels for better density
                sample_tiles = []
                max_downloads = 50  # Increased from 25
                total_points = 0
                max_target_points = 5_000_000  # Increased from 2M to 5M points

                # Always include root tile
                if "0-0-0-0" in hierarchy:
                    sample_tiles.append("0-0-0-0.laz")

                # Add tiles from levels 2-6, prioritizing by point count and filtering sparse tiles
                for level in [
                    3,
                    4,
                    5,
                    2,
                    6,
                ]:  # Reordered to prioritize levels 3-5 first
                    if level in tiles_by_level and len(sample_tiles) < max_downloads:
                        # Sort tiles by point count (descending) and take the best ones
                        level_tiles = sorted(
                            tiles_by_level[level], key=lambda x: x[1], reverse=True
                        )
                        for tile_name, point_count in level_tiles:
                            if len(sample_tiles) >= max_downloads:
                                break
                            if total_points + point_count > max_target_points:
                                break
                            # Filter out very sparse tiles (< 1000 points) unless we need more coverage
                            if point_count < 1000 and len(sample_tiles) > 10:
                                continue
                            sample_tiles.append(f"{tile_name}.laz")
                            total_points += point_count

                logger.info(
                    f"Selected {len(sample_tiles)} tiles from hierarchy (targeting {total_points:,} points)"
                )
            else:
                # Fallback to basic tiles if hierarchy is not available
                sample_tiles = ["0-0-0-0.laz"]

            downloaded_count = 0

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

    def download_dataset_with_orthophoto_bounds(
        self, dataset_name: str, output_dir: str, ortho_bounds: Dict, ortho_crs: str
    ) -> bool:
        """Download EPT point cloud data for a dataset, focusing on tiles that overlap with orthophoto bounds.

        Args:
            dataset_name: Name of the dataset to download
            output_dir: Directory to save downloaded files
            ortho_bounds: Orthophoto bounds dict with left, right, bottom, top
            ortho_crs: Orthophoto coordinate reference system

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info(
                f"Downloading point cloud data with geographic filtering for: {dataset_name}"
            )
            logger.info(f"Target area bounds: {ortho_bounds} (CRS: {ortho_crs})")

            dataset_dir, metadata = self._prepare_download(dataset_name, output_dir)
            if not metadata:
                return False

            # Get dataset CRS and bounds from metadata
            dataset_srs = metadata.get("srs", {})
            dataset_bounds = metadata.get("bounds", [])

            if not dataset_bounds or len(dataset_bounds) < 4:
                logger.warning(
                    "Dataset bounds not found in metadata, falling back to sample tiles"
                )
                return self.download_dataset(dataset_name, output_dir)

            dataset_crs = None
            if dataset_srs:
                # Try to construct CRS from metadata
                authority = dataset_srs.get("authority")
                horizontal = dataset_srs.get("horizontal")
                if authority and horizontal:
                    dataset_crs = f"{authority}:{horizontal}"
                    logger.info(f"Dataset CRS: {dataset_crs}")

            # Transform orthophoto bounds to dataset CRS for comparison
            if dataset_crs and dataset_crs != ortho_crs:
                try:
                    from pyproj import Transformer, CRS

                    transformer = Transformer.from_crs(
                        CRS.from_string(ortho_crs),
                        CRS.from_string(dataset_crs),
                        always_xy=True,
                    )

                    # Transform orthophoto corners to dataset CRS
                    ortho_left_ds, ortho_bottom_ds = transformer.transform(
                        ortho_bounds["left"], ortho_bounds["bottom"]
                    )
                    ortho_right_ds, ortho_top_ds = transformer.transform(
                        ortho_bounds["right"], ortho_bounds["top"]
                    )

                    target_bounds = {
                        "left": ortho_left_ds,
                        "right": ortho_right_ds,
                        "bottom": ortho_bottom_ds,
                        "top": ortho_top_ds,
                    }

                    logger.info(f"Target area bounds in dataset CRS: {target_bounds}")

                except Exception as e:
                    logger.warning(
                        f"Could not transform target bounds to dataset CRS: {e}"
                    )
                    target_bounds = ortho_bounds
            else:
                target_bounds = ortho_bounds

            # Check if target area overlaps with dataset bounds
            ds_xmin, ds_ymin = dataset_bounds[0], dataset_bounds[1]
            ds_xmax, ds_ymax = dataset_bounds[3], dataset_bounds[4]

            overlap_x = not (
                target_bounds["right"] < ds_xmin or target_bounds["left"] > ds_xmax
            )
            overlap_y = not (
                target_bounds["top"] < ds_ymin or target_bounds["bottom"] > ds_ymax
            )

            if not (overlap_x and overlap_y):
                logger.warning("Target area does not overlap with dataset bounds")
                logger.warning(
                    f"Dataset bounds: X[{ds_xmin:.0f}, {ds_xmax:.0f}] Y[{ds_ymin:.0f}, {ds_ymax:.0f}]"
                )
                logger.warning(
                    f"Target bounds: X[{target_bounds['left']:.0f}, {target_bounds['right']:.0f}] Y[{target_bounds['bottom']:.0f}, {target_bounds['top']:.0f}]"
                )
                logger.info("Falling back to sample tiles from dataset...")
                return self.download_dataset(dataset_name, output_dir)

            # Download tiles with geographic targeting
            # Start with deeper levels for finer resolution in the target area
            successful_files = self._download_geographically_filtered_tiles(
                dataset_name, dataset_dir, target_bounds, dataset_bounds, metadata
            )

            if successful_files:
                logger.info(
                    f"Successfully downloaded {len(successful_files)} geographically filtered tiles"
                )
                logger.info(f"Files saved to: {dataset_dir}")
                return True
            else:
                logger.warning("Geographic filtering failed, trying fallback approach")
                return self.download_dataset(dataset_name, output_dir)

        except Exception as e:
            logger.error(
                f"Failed to download dataset {dataset_name} with geographic bounds: {e}"
            )
            return False

    def _download_geographically_filtered_tiles(
        self,
        dataset_name: str,
        dataset_dir: str,
        target_bounds: Dict,
        dataset_bounds: List,
        metadata: Dict,
    ) -> List[str]:
        """Download EPT tiles that specifically cover the target geographic area.

        Args:
            dataset_name: Name of the dataset
            dataset_dir: Directory to save files
            target_bounds: Target area bounds in dataset CRS
            dataset_bounds: Full dataset bounds
            metadata: EPT metadata

        Returns:
            List of successfully downloaded file paths
        """
        successful_files = []

        try:
            # Load hierarchy data to know which tiles exist
            hierarchy = {}
            hierarchy_key = f"{dataset_name}/ept-hierarchy/0-0-0-0.json"
            try:
                response = self.s3_client.get_object(
                    Bucket=self.bucket_name, Key=hierarchy_key
                )
                hierarchy = json.loads(response["Body"].read().decode("utf-8"))
                logger.info(f"Loaded hierarchy with {len(hierarchy)} tiles")
            except Exception as e:
                logger.warning(f"Could not download hierarchy data: {e}")
                # Try to load from local file if it exists
                hierarchy_path = os.path.join(dataset_dir, "hierarchy.json")
                if os.path.exists(hierarchy_path):
                    with open(hierarchy_path, "r") as f:
                        hierarchy = json.load(f)
                    logger.info(
                        f"Loaded hierarchy from local file with {len(hierarchy)} tiles"
                    )

            if not hierarchy:
                logger.error("No hierarchy data available for geographic filtering")
                return successful_files

            # Calculate target area size for appropriate tile level selection
            target_width = target_bounds["right"] - target_bounds["left"]
            target_height = target_bounds["top"] - target_bounds["bottom"]
            target_area = target_width * target_height

            # Dataset dimensions
            ds_width = dataset_bounds[3] - dataset_bounds[0]  # xmax - xmin
            ds_height = dataset_bounds[4] - dataset_bounds[1]  # ymax - ymin
            ds_area = ds_width * ds_height

            # Find available tile levels in hierarchy
            available_levels = set()
            for tile_name in hierarchy.keys():
                parts = tile_name.split("-")
                if len(parts) >= 1:
                    try:
                        level = int(parts[0])
                        available_levels.add(level)
                    except ValueError:
                        continue

            available_levels = sorted(available_levels)
            logger.info(f"Available tile levels: {available_levels}")

            # Calculate what tile level we need for good coverage
            # Each level subdivides by 2 in each dimension (4x area subdivision)
            area_ratio = target_area / ds_area
            calculated_level = max(
                0, min(6, int(-math.log2(area_ratio) / 2) + 1)
            )  # More conservative calculation

            logger.info(
                f"Target area: {target_area:.0f} sq units ({target_width:.0f} x {target_height:.0f})"
            )
            logger.info(
                f"Dataset area: {ds_area:.0f} sq units, area ratio: {area_ratio:.6f}"
            )
            logger.info(f"Calculated ideal tile level: {calculated_level}")

            # Use available levels, starting from calculated level and working down/up
            levels_to_try = []

            # Start with calculated level if available
            if calculated_level in available_levels:
                levels_to_try.append(calculated_level)

            # Add nearby levels, prioritizing higher resolution (higher numbers)
            for offset in [1, -1, 2, -2, 3, -3]:
                level = calculated_level + offset
                if level in available_levels and level not in levels_to_try:
                    levels_to_try.append(level)

            # Add any remaining available levels
            for level in reversed(available_levels):  # Higher levels first
                if level not in levels_to_try:
                    levels_to_try.append(level)

            # Limit to reasonable range (levels 2-6 are usually most useful)
            levels_to_try = [l for l in levels_to_try if 2 <= l <= 8][
                :6
            ]  # Max 6 levels for better coverage

            if not levels_to_try:
                # Fallback to any available levels
                levels_to_try = [l for l in available_levels if l >= 2][:5]

            logger.info(f"Trying tile levels in order: {levels_to_try}")

            for level in levels_to_try:
                logger.info(f"Downloading tiles at level {level}...")
                level_files = self._download_tiles_at_level(
                    dataset_name,
                    dataset_dir,
                    target_bounds,
                    dataset_bounds,
                    level,
                    hierarchy,
                )
                successful_files.extend(level_files)

                # Check if we have good coverage based on total points, not just tile count
                # Calculate current total points from all downloaded files
                total_downloaded_points = 0
                for file_path in successful_files:
                    try:
                        import laspy

                        las_data = laspy.read(file_path)
                        total_downloaded_points += len(las_data.points)
                    except:
                        pass  # Skip files we can't read

                # Target: at least 500K points for good coverage, but allow up to 5M for dense areas
                min_target_points = 500_000
                max_target_points = 5_000_000

                logger.info(
                    f"Current coverage: {len(successful_files)} tiles, {total_downloaded_points:,} points"
                )

                # Stop if we have sufficient coverage or too many points
                if total_downloaded_points >= min_target_points:
                    if total_downloaded_points >= max_target_points:
                        logger.info(
                            f"Reached maximum target points ({max_target_points:,}), stopping"
                        )
                        break
                    elif len(successful_files) >= 10:  # Also stop if we have many tiles
                        logger.info(
                            f"Good coverage achieved: {total_downloaded_points:,} points from {len(successful_files)} tiles"
                        )
                        break

                # Continue if no tiles found at this level
                if not level_files:
                    logger.info(
                        f"No suitable tiles found at level {level}, trying next level"
                    )
                    continue

            # Merge tiles if we have multiple small ones
            if len(successful_files) > 1:
                merged_file = self._merge_laz_files(successful_files, dataset_dir)
                if merged_file:
                    return [merged_file]

            return successful_files

        except Exception as e:
            logger.error(f"Error in geographic filtering: {e}")
            return []

    def _download_tiles_at_level(
        self,
        dataset_name: str,
        dataset_dir: str,
        target_bounds: Dict,
        dataset_bounds: List,
        level: int,
        hierarchy: Dict,
    ) -> List[str]:
        """Download all tiles at a specific level that intersect with target bounds."""
        successful_files = []

        try:
            # Filter hierarchy for tiles at this level
            level_tiles = {
                k: v for k, v in hierarchy.items() if k.startswith(f"{level}-")
            }

            if not level_tiles:
                logger.info(f"No tiles available at level {level}")
                return successful_files

            logger.info(f"Found {len(level_tiles)} tiles at level {level}")

            # Calculate tile grid dimensions at this level
            tiles_per_side = 2**level

            # Dataset bounds
            ds_xmin, ds_ymin = dataset_bounds[0], dataset_bounds[1]
            ds_xmax, ds_ymax = dataset_bounds[3], dataset_bounds[4]

            # Tile dimensions
            tile_width = (ds_xmax - ds_xmin) / tiles_per_side
            tile_height = (ds_ymax - ds_ymin) / tiles_per_side

            logger.info(
                f"Level {level}: {tiles_per_side}x{tiles_per_side} grid, tile size: {tile_width:.0f}x{tile_height:.0f}"
            )

            # Find target tile range based on bounds
            start_col = max(0, int((target_bounds["left"] - ds_xmin) / tile_width))
            end_col = min(
                tiles_per_side - 1, int((target_bounds["right"] - ds_xmin) / tile_width)
            )
            start_row = max(0, int((target_bounds["bottom"] - ds_ymin) / tile_height))
            end_row = min(
                tiles_per_side - 1, int((target_bounds["top"] - ds_ymin) / tile_height)
            )

            logger.info(
                f"Target tile range: cols {start_col}-{end_col}, rows {start_row}-{end_row}"
            )

            downloaded_count = 0
            max_tiles = 100  # Increased limit for better geographic coverage
            max_points = 10_000_000  # 10M point limit to prevent huge downloads
            total_points = 0

            # Download tiles that exist in hierarchy and intersect target area
            for tile_name, point_count in level_tiles.items():
                if downloaded_count >= max_tiles:
                    logger.info(f"Reached maximum tile limit ({max_tiles})")
                    break

                if total_points + point_count > max_points:
                    logger.info(
                        f"Approaching point limit ({max_points:,}), stopping downloads"
                    )
                    break

                # Parse tile coordinates
                parts = tile_name.split("-")
                if len(parts) != 4:
                    continue

                try:
                    level_num, col, row, z = map(int, parts)

                    # Check if tile intersects with target area
                    if start_col <= col <= end_col and start_row <= row <= end_row:
                        tile_key = f"{dataset_name}/ept-data/{tile_name}.laz"
                        tile_path = os.path.join(dataset_dir, f"{tile_name}.laz")

                        try:
                            logger.debug(
                                f"Downloading tile: {tile_name} ({point_count:,} points)"
                            )
                            self.s3_client.download_file(
                                self.bucket_name, tile_key, tile_path
                            )

                            file_size = os.path.getsize(tile_path)
                            logger.info(
                                f"Downloaded {tile_name}.laz ({file_size:,} bytes, {point_count:,} points)"
                            )
                            successful_files.append(tile_path)
                            downloaded_count += 1
                            total_points += point_count

                        except Exception as e:
                            logger.debug(f"Tile {tile_name} not available: {e}")

                except ValueError:
                    continue

            logger.info(
                f"Downloaded {downloaded_count} tiles at level {level} with {total_points:,} total points"
            )

        except Exception as e:
            logger.error(f"Error downloading tiles at level {level}: {e}")

        return successful_files

    def _merge_laz_files(self, laz_files: List[str], output_dir: str) -> Optional[str]:
        """Merge multiple LAZ files into a single file for processing."""
        try:
            import laspy
            import numpy as np

            if len(laz_files) <= 1:
                return laz_files[0] if laz_files else None

            logger.info(f"Merging {len(laz_files)} LAZ files...")

            # Read all files and combine points
            all_points = []
            total_points = 0

            for laz_file in laz_files:
                try:
                    las_data = laspy.read(laz_file)
                    if len(las_data.points) > 0:
                        all_points.append(las_data)
                        total_points += len(las_data.points)
                        logger.debug(
                            f"Read {len(las_data.points):,} points from {os.path.basename(laz_file)}"
                        )
                except Exception as e:
                    logger.warning(f"Failed to read {laz_file}: {e}")
                    continue

            if not all_points:
                logger.error("No valid points found in any LAZ file")
                return None

            logger.info(f"Total points to merge: {total_points:,}")

            # Create merged file using the first file as template
            header = all_points[0].header
            merged_las = laspy.LasData(header)

            # Combine coordinates and attributes
            x_coords = np.concatenate([las.x for las in all_points])
            y_coords = np.concatenate([las.y for las in all_points])
            z_coords = np.concatenate([las.z for las in all_points])

            merged_las.x = x_coords
            merged_las.y = y_coords
            merged_las.z = z_coords

            # Copy other attributes if they exist
            for attr in [
                "intensity",
                "return_number",
                "number_of_returns",
                "classification",
            ]:
                if hasattr(all_points[0], attr):
                    try:
                        values = np.concatenate(
                            [getattr(las, attr) for las in all_points]
                        )
                        setattr(merged_las, attr, values)
                    except Exception as e:
                        logger.debug(f"Could not merge attribute {attr}: {e}")

            # Save merged file
            merged_path = os.path.join(output_dir, "merged_tiles.laz")
            merged_las.write(merged_path)

            file_size = os.path.getsize(merged_path) / (1024 * 1024)  # MB
            logger.info(
                f"Merged file saved: {merged_path} ({file_size:.1f} MB, {total_points:,} points)"
            )

            return merged_path

        except Exception as e:
            logger.error(f"Failed to merge LAZ files: {e}")
            return None

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

    def select_best_dataset_for_location(
        self, datasets: List[Dict], lat: float, lon: float
    ) -> Dict:
        """
        Select the best dataset for a given location using improved criteria.

        Prioritizes:
        1. Geographic proximity (closest distance to dataset center)
        2. Regional specificity (metropolitan vs. broad regional datasets)
        3. Recency (newer datasets preferred)
        4. Data quality (more points = better)
        5. State/region matching

        Args:
            datasets: List of available datasets
            lat: Target latitude
            lon: Target longitude

        Returns:
            Best matching dataset
        """
        from pyproj import Transformer, CRS

        if not datasets:
            raise ValueError("No datasets provided")

        logger.info(
            f"Evaluating {len(datasets)} datasets for location {lat:.6f}, {lon:.6f}"
        )

        # Transform target coordinates to Web Mercator for distance calculations
        wgs84_to_mercator = Transformer.from_crs(
            CRS.from_epsg(4326), CRS.from_epsg(3857), always_xy=True
        )
        target_x, target_y = wgs84_to_mercator.transform(lon, lat)

        scored_datasets = []

        for dataset in datasets:
            metrics = self._score_dataset(
                dataset,
                (lat, lon),
                {"target_coords": (target_x, target_y)},
            )

            distance_score = metrics["distance_score"]
            dataset_area = metrics["dataset_area"]
            recency_score = metrics["recency_score"]
            quality_score = metrics["quality_score"]
            specificity_bonus = metrics["specificity_bonus"]
            region_bonus = metrics["region_bonus"]
            state_bonus = metrics["state_bonus"]

            distance_penalty = (
                distance_score / 1000 if distance_score != float("inf") else 1000
            )
            recency_bonus = (recency_score - 2000) * 8
            quality_bonus = quality_score * 3

            composite_score = (
                -distance_penalty * 2
                + recency_bonus
                + quality_bonus
                + specificity_bonus
                + region_bonus
                + state_bonus
            )

            metrics.update(
                {
                    "score": composite_score,
                    "distance_km": (
                        distance_score / 1000
                        if distance_score != float("inf")
                        else float("inf")
                    ),
                    "area_km2": dataset_area / 1_000_000 if dataset_area > 0 else 0,
                }
            )

            scored_datasets.append(metrics)

            logger.debug(
                f"Dataset {metrics['name']}: distance={distance_score/1000:.1f}km, area={dataset_area/1_000_000:.0f}km², "
                f"year={metrics['year']}, points={metrics['points']:,}, specificity={specificity_bonus:.1f}, "
                f"region={region_bonus:.1f}, score={composite_score:.1f}"
            )

        # Sort by score (highest first)
        scored_datasets.sort(key=lambda x: x["score"], reverse=True)

        # Log top candidates with enhanced details
        logger.info("Top 5 dataset candidates:")
        for i, item in enumerate(scored_datasets[:5]):
            logger.info(f"  {i+1}. {item['name']}")
            logger.info(
                f"     Distance: {item['distance_km']:.1f}km, Area: {item['area_km2']:.0f}km², "
                f"Year: {item['year']}, Points: {item['points']:,}"
            )
            logger.info(
                f"     Bonuses - Specificity: {item['specificity_bonus']:.1f}, "
                f"Region: {item['region_bonus']:.1f}"
            )
            logger.info(f"     Total Score: {item['score']:.1f}")
            logger.info("")  # Empty line for readability

        best_dataset = scored_datasets[0]["dataset"]
        logger.info(f"Selected best dataset: {best_dataset.get('name', 'Unknown')}")

        return best_dataset

    def select_best_dataset_for_orthophoto(
        self,
        datasets: List[Dict],
        ortho_bounds: Dict,
        ortho_crs: str,
        lat: float,
        lon: float,
    ) -> Dict:
        """
        Select the best dataset for a given location considering orthophoto coverage.

        This method prioritizes datasets that overlap with the orthophoto bounds,
        ensuring better coordinate alignment between point cloud and orthophoto.

        Args:
            datasets: List of available datasets
            ortho_bounds: Orthophoto bounds dict with keys: left, right, bottom, top
            ortho_crs: Orthophoto coordinate reference system
            lat: Target latitude
            lon: Target longitude

        Returns:
            Best matching dataset
        """
        from pyproj import Transformer, CRS

        if not datasets:
            raise ValueError("No datasets provided")

        logger.info(
            f"Evaluating {len(datasets)} datasets for orthophoto coverage at {lat:.6f}, {lon:.6f}"
        )
        logger.info(f"Orthophoto bounds: {ortho_bounds} (CRS: {ortho_crs})")

        # Transform orthophoto bounds to Web Mercator for comparison
        try:
            ortho_to_mercator = Transformer.from_crs(
                CRS.from_string(ortho_crs), CRS.from_epsg(3857), always_xy=True
            )

            # Transform corners of orthophoto to Web Mercator
            left_merc, bottom_merc = ortho_to_mercator.transform(
                ortho_bounds["left"], ortho_bounds["bottom"]
            )
            right_merc, top_merc = ortho_to_mercator.transform(
                ortho_bounds["right"], ortho_bounds["top"]
            )

            ortho_bounds_mercator = {
                "left": left_merc,
                "right": right_merc,
                "bottom": bottom_merc,
                "top": top_merc,
            }

            logger.info(f"Orthophoto bounds in Web Mercator: {ortho_bounds_mercator}")

        except Exception as e:
            logger.warning(f"Could not transform orthophoto bounds: {e}")
            # Fall back to original method
            return self.select_best_dataset_for_location(datasets, lat, lon)

        scored_datasets = []

        ortho_center = (
            (ortho_bounds_mercator["left"] + ortho_bounds_mercator["right"]) / 2,
            (ortho_bounds_mercator["bottom"] + ortho_bounds_mercator["top"]) / 2,
        )

        for dataset in datasets:
            name = dataset.get("name", "")
            bounds = dataset.get("bounds", [])
            overlap_score = 0
            distance_score = float("inf")

            if bounds and len(bounds) >= 4:
                if len(bounds) == 6:
                    ds_xmin, ds_ymin, ds_xmax, ds_ymax = (
                        bounds[0],
                        bounds[1],
                        bounds[3],
                        bounds[4],
                    )
                else:
                    ds_xmin, ds_ymin, ds_xmax, ds_ymax = (
                        bounds[0],
                        bounds[1],
                        bounds[2],
                        bounds[3],
                    )

                overlap_left = max(ds_xmin, ortho_bounds_mercator["left"])
                overlap_right = min(ds_xmax, ortho_bounds_mercator["right"])
                overlap_bottom = max(ds_ymin, ortho_bounds_mercator["bottom"])
                overlap_top = min(ds_ymax, ortho_bounds_mercator["top"])

                if overlap_left < overlap_right and overlap_bottom < overlap_top:
                    overlap_area = (overlap_right - overlap_left) * (
                        overlap_top - overlap_bottom
                    )
                    ortho_area = (
                        ortho_bounds_mercator["right"] - ortho_bounds_mercator["left"]
                    ) * (ortho_bounds_mercator["top"] - ortho_bounds_mercator["bottom"])
                    overlap_percentage = (
                        overlap_area / ortho_area if ortho_area > 0 else 0
                    )
                    overlap_score = overlap_percentage * 10000
                    logger.debug(
                        f"Dataset {name}: overlap {overlap_percentage:.1%} of orthophoto"
                    )
                else:
                    ds_center_x = (ds_xmin + ds_xmax) / 2
                    ds_center_y = (ds_ymin + ds_ymax) / 2
                    distance_score = math.sqrt(
                        (ds_center_x - ortho_center[0]) ** 2
                        + (ds_center_y - ortho_center[1]) ** 2
                    )

            metrics = self._score_dataset(
                dataset,
                (lat, lon),
                {"target_coords": ortho_center},
            )

            dataset_area = metrics["dataset_area"]
            recency_score = metrics["recency_score"]
            quality_score = metrics["quality_score"]
            specificity_bonus = metrics["specificity_bonus"]
            region_bonus = metrics["region_bonus"]
            state_bonus = metrics["state_bonus"]

            if overlap_score > 0:
                distance_penalty = (
                    distance_score / 1000 if distance_score != float("inf") else 0
                )
                recency_bonus = (recency_score - 2000) * 8
                quality_bonus = quality_score * 3

                composite_score = (
                    overlap_score * 10
                    + -distance_penalty * 2
                    + recency_bonus
                    + quality_bonus
                    + specificity_bonus
                    + region_bonus
                    + state_bonus
                )
            else:
                distance_penalty = distance_score / 1000
                composite_score = (
                    -(distance_penalty * 10)
                    + (recency_score - 2000) * 10
                    + quality_score * 5
                    + state_bonus
                )

            metrics.update(
                {
                    "score": composite_score,
                    "overlap_score": overlap_score,
                    "distance_km": (
                        distance_score / 1000
                        if distance_score != float("inf")
                        else float("inf")
                    ),
                    "area_km2": dataset_area / 1_000_000 if dataset_area > 0 else 0,
                }
            )

            scored_datasets.append(metrics)
        # Sort by score (highest first)
        scored_datasets.sort(key=lambda x: x["score"], reverse=True)

        # Log top candidates with enhanced details
        logger.info("Top 3 dataset candidates (considering orthophoto overlap):")
        for i, item in enumerate(scored_datasets[:3]):
            logger.info(f"  {i+1}. {item['name']}")
            if item["overlap_score"] > 0:
                logger.info(
                    f"     Overlap: {item['overlap_score']/100:.1f}% of orthophoto"
                )
            else:
                logger.info(
                    f"     Distance: {item['distance_km']:.1f}km from orthophoto center"
                )
            logger.info(
                f"     Area: {item['area_km2']:.0f}km², Year: {item['year']}, Points: {item['points']:,}"
            )
            logger.info(
                f"     Bonuses - Specificity: {item['specificity_bonus']:.1f}, "
                f"Region: {item['region_bonus']:.1f}"
            )
            logger.info(f"     Score: {item['score']:.1f}")
            logger.info("")  # Empty line for readability

        best_dataset = scored_datasets[0]["dataset"]
        logger.info(f"Selected best dataset: {best_dataset.get('name', 'Unknown')}")

        return best_dataset

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

    def download_point_cloud(
        self,
        product: Dict,
        output_dir: str,
        ortho_bounds: Optional[Dict] = None,
        ortho_crs: Optional[str] = None,
    ) -> str:
        """Download point cloud data for a specific product.

        Args:
            product: Product dictionary containing dataset information
            output_dir: Directory to save downloaded files
            ortho_bounds: Optional orthophoto bounds for smart tile selection
            ortho_crs: Optional orthophoto CRS

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

            # Use orthophoto-aware download if bounds are provided
            if ortho_bounds and ortho_crs:
                success = self.download_dataset_with_orthophoto_bounds(
                    dataset_name, output_dir, ortho_bounds, ortho_crs
                )
            else:
                # Fall back to existing method
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
                    # If we have orthophoto bounds, test each file to find the best overlap
                    if ortho_bounds and ortho_crs and len(laz_files) > 1:
                        logger.info(
                            f"Testing {len(laz_files)} downloaded tiles for orthophoto overlap..."
                        )
                        best_file = self._find_best_overlapping_tile(
                            laz_files, ortho_bounds, ortho_crs
                        )
                        if best_file:
                            logger.info(
                                f"Selected best overlapping tile: {os.path.basename(best_file)}"
                            )
                            return best_file
                        else:
                            logger.warning(
                                "No overlapping tiles found, using first available"
                            )

                    # Return the first LAZ file found (fallback behavior)
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

    def _find_best_overlapping_tile(
        self, laz_files: List[str], ortho_bounds: Dict, ortho_crs: str
    ) -> Optional[str]:
        """Find the LAZ tile with the best overlap with orthophoto bounds.

        Args:
            laz_files: List of LAZ file paths to test
            ortho_bounds: Orthophoto bounds dict
            ortho_crs: Orthophoto CRS

        Returns:
            Path to best overlapping tile, or None if no overlap found
        """
        try:
            import laspy
            import numpy as np
            from pyproj import Transformer, CRS

            best_file = None
            best_overlap_score = 0
            best_distance = float("inf")

            logger.info("Testing tile overlaps with orthophoto...")

            for laz_file in laz_files:
                try:
                    # Quick read of tile to get bounds
                    las_data = laspy.read(laz_file)

                    if len(las_data.points) == 0:
                        logger.debug(
                            f"Skipping empty tile: {os.path.basename(laz_file)}"
                        )
                        continue

                    # Get point cloud bounds
                    pc_x_min, pc_x_max = float(np.min(las_data.x)), float(
                        np.max(las_data.x)
                    )
                    pc_y_min, pc_y_max = float(np.min(las_data.y)), float(
                        np.max(las_data.y)
                    )

                    # Detect coordinate system (same logic as main colorization)
                    if abs(pc_x_min) > 1000000 or abs(pc_y_min) > 1000000:
                        # Likely projected coordinates (Web Mercator or UTM)
                        if abs(pc_x_min) > 10000000:  # Web Mercator range
                            pc_crs = "EPSG:3857"
                        else:  # Assume UTM
                            pc_crs = "EPSG:26913"  # Common for Colorado
                    else:
                        # Likely geographic coordinates
                        pc_crs = "EPSG:4326"

                    # Transform orthophoto bounds to point cloud CRS for comparison
                    if pc_crs != ortho_crs:
                        transformer = Transformer.from_crs(
                            CRS.from_string(ortho_crs),
                            CRS.from_string(pc_crs),
                            always_xy=True,
                        )
                        ortho_left_pc, ortho_bottom_pc = transformer.transform(
                            ortho_bounds["left"], ortho_bounds["bottom"]
                        )
                        ortho_right_pc, ortho_top_pc = transformer.transform(
                            ortho_bounds["right"], ortho_bounds["top"]
                        )
                    else:
                        ortho_left_pc = ortho_bounds["left"]
                        ortho_right_pc = ortho_bounds["right"]
                        ortho_bottom_pc = ortho_bounds["bottom"]
                        ortho_top_pc = ortho_bounds["top"]

                    # Check overlap
                    overlap_x = not (
                        pc_x_max < ortho_left_pc or pc_x_min > ortho_right_pc
                    )
                    overlap_y = not (
                        pc_y_max < ortho_bottom_pc or pc_y_min > ortho_top_pc
                    )

                    # Calculate distance between centers
                    pc_center_x = (pc_x_min + pc_x_max) / 2
                    pc_center_y = (pc_y_min + pc_y_max) / 2
                    ortho_center_x = (ortho_left_pc + ortho_right_pc) / 2
                    ortho_center_y = (ortho_bottom_pc + ortho_top_pc) / 2

                    distance = (
                        (pc_center_x - ortho_center_x) ** 2
                        + (pc_center_y - ortho_center_y) ** 2
                    ) ** 0.5

                    overlap_score = 0
                    if overlap_x and overlap_y:
                        # Calculate actual overlap area
                        overlap_left = max(pc_x_min, ortho_left_pc)
                        overlap_right = min(pc_x_max, ortho_right_pc)
                        overlap_bottom = max(pc_y_min, ortho_bottom_pc)
                        overlap_top = min(pc_y_max, ortho_top_pc)

                        if (
                            overlap_left < overlap_right
                            and overlap_bottom < overlap_top
                        ):
                            overlap_area = (overlap_right - overlap_left) * (
                                overlap_top - overlap_bottom
                            )
                            ortho_area = (ortho_right_pc - ortho_left_pc) * (
                                ortho_top_pc - ortho_bottom_pc
                            )
                            overlap_score = (
                                overlap_area / ortho_area if ortho_area > 0 else 0
                            )

                    tile_name = os.path.basename(laz_file)
                    logger.debug(
                        f"Tile {tile_name}: overlap={overlap_x and overlap_y}, score={overlap_score:.3f}, distance={distance:.0f}m, points={len(las_data.points)}"
                    )

                    # Select best tile based on overlap score, then distance
                    if overlap_score > best_overlap_score:
                        best_file = laz_file
                        best_overlap_score = overlap_score
                        best_distance = distance
                        logger.info(
                            f"New best tile: {tile_name} (overlap score: {overlap_score:.3f})"
                        )
                    elif (
                        overlap_score == best_overlap_score and distance < best_distance
                    ):
                        best_file = laz_file
                        best_distance = distance
                        logger.info(
                            f"New best tile by distance: {tile_name} (distance: {distance:.0f}m)"
                        )

                except Exception as e:
                    logger.debug(
                        f"Error testing tile {os.path.basename(laz_file)}: {e}"
                    )
                    continue

            if best_file:
                logger.info(
                    f"Best tile found: {os.path.basename(best_file)} (overlap score: {best_overlap_score:.3f}, distance: {best_distance:.0f}m)"
                )
            else:
                logger.warning("No overlapping tiles found")

            return best_file

        except Exception as e:
            logger.error(f"Error finding best overlapping tile: {e}")
            return None


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
