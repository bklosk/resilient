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
        import math
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
            name = dataset.get("name", "")
            bounds = dataset.get("bounds", [])
            points = dataset.get("points", 0)

            # Calculate geographic proximity score
            distance_score = float("inf")  # Default to infinite distance
            dataset_area = 0  # Dataset coverage area

            if bounds and len(bounds) >= 4:
                # Calculate dataset center and area in Web Mercator
                if len(bounds) == 6:  # [xmin, ymin, zmin, xmax, ymax, zmax]
                    xmin, ymin, xmax, ymax = bounds[0], bounds[1], bounds[3], bounds[4]
                else:  # [xmin, ymin, xmax, ymax]
                    xmin, ymin, xmax, ymax = bounds[0], bounds[1], bounds[2], bounds[3]

                center_x = (xmin + xmax) / 2
                center_y = (ymin + ymax) / 2
                dataset_area = (xmax - xmin) * (ymax - ymin)  # Area in square meters

                # Calculate distance in meters
                distance = math.sqrt(
                    (target_x - center_x) ** 2 + (target_y - center_y) ** 2
                )
                distance_score = distance  # Lower is better

            # Extract year from dataset name
            year = 0
            for part in name.split("_"):
                if part.isdigit() and len(part) == 4:
                    try:
                        year = int(part)
                        break
                    except ValueError:
                        continue

            # Calculate recency score (higher is better)
            current_year = 2025  # Update as needed
            recency_score = year if year > 0 else 1900

            # Calculate quality score based on point count (log scale to prevent overwhelming)
            quality_score = math.log10(max(points, 1))

            # Regional specificity bonus - prefer smaller, more focused datasets over broad regional ones
            # This helps choose metropolitan datasets (like DRCOG) over state-wide datasets (like NWCO)
            specificity_bonus = 0
            if dataset_area > 0:
                # Convert area to km²
                area_km2 = dataset_area / 1_000_000
                # Prefer datasets with smaller coverage areas (more specific)
                # Use logarithmic scale to prevent overwhelming the score
                specificity_bonus = max(0, 100 - math.log10(max(area_km2, 1)) * 20)

            # Regional keyword matching for better geographic classification
            region_bonus = 0
            name_upper = name.upper()

            # Check for Front Range/Denver metro area keywords (good for Boulder)
            front_range_keywords = [
                "DRCOG",
                "DENVER",
                "METRO",
                "FRONT",
                "BOULDER",
                "JEFFCO",
                "ADAMS",
            ]
            # Check for broader regional keywords (less specific)
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

            # Bonus for Front Range/metro area datasets when in Colorado Front Range
            if 39.5 <= lat <= 40.5 and -105.5 <= lon <= -104.5:  # Front Range area
                if any(keyword in name_upper for keyword in front_range_keywords):
                    region_bonus = 500  # Strong preference for metro datasets
                elif any(keyword in name_upper for keyword in broad_keywords):
                    region_bonus = -200  # Slight penalty for broad regional datasets

            # State/region matching
            state_bonus = 0
            if (
                lat >= 39 and lat <= 41 and lon >= -109 and lon <= -102
            ):  # Colorado bounds
                if "CO" in name_upper:
                    state_bonus = 300  # Moderate bonus for Colorado datasets

            # Composite score with improved weighting
            # Normalize distance to km and weight factors appropriately
            distance_penalty = (
                distance_score / 1000 if distance_score != float("inf") else 1000
            )
            recency_bonus = (recency_score - 2000) * 8  # Reduced weight for recency
            quality_bonus = quality_score * 3  # Reduced weight for quality

            composite_score = (
                -distance_penalty * 2  # Distance is most important (doubled weight)
                + recency_bonus
                + quality_bonus
                + specificity_bonus
                + region_bonus
                + state_bonus
            )

            scored_datasets.append(
                {
                    "dataset": dataset,
                    "score": composite_score,
                    "distance_km": (
                        distance_score / 1000
                        if distance_score != float("inf")
                        else float("inf")
                    ),
                    "area_km2": dataset_area / 1_000_000 if dataset_area > 0 else 0,
                    "year": year,
                    "points": points,
                    "name": name,
                    "specificity_bonus": specificity_bonus,
                    "region_bonus": region_bonus,
                }
            )

            logger.debug(
                f"Dataset {name}: distance={distance_score/1000:.1f}km, area={dataset_area/1_000_000:.0f}km², "
                f"year={year}, points={points:,}, specificity={specificity_bonus:.1f}, "
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
        import math
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

        for dataset in datasets:
            name = dataset.get("name", "")
            bounds = dataset.get("bounds", [])
            points = dataset.get("points", 0)

            # Calculate overlap with orthophoto
            overlap_score = 0
            distance_score = float("inf")

            if bounds and len(bounds) >= 4:
                # Dataset bounds in Web Mercator
                if len(bounds) == 6:  # [xmin, ymin, zmin, xmax, ymax, zmax]
                    ds_xmin, ds_ymin, ds_xmax, ds_ymax = (
                        bounds[0],
                        bounds[1],
                        bounds[3],
                        bounds[4],
                    )
                else:  # [xmin, ymin, xmax, ymax]
                    ds_xmin, ds_ymin, ds_xmax, ds_ymax = (
                        bounds[0],
                        bounds[1],
                        bounds[2],
                        bounds[3],
                    )

                # Calculate overlap area
                overlap_left = max(ds_xmin, ortho_bounds_mercator["left"])
                overlap_right = min(ds_xmax, ortho_bounds_mercator["right"])
                overlap_bottom = max(ds_ymin, ortho_bounds_mercator["bottom"])
                overlap_top = min(ds_ymax, ortho_bounds_mercator["top"])

                if overlap_left < overlap_right and overlap_bottom < overlap_top:
                    # There is overlap
                    overlap_area = (overlap_right - overlap_left) * (
                        overlap_top - overlap_bottom
                    )
                    ortho_area = (
                        ortho_bounds_mercator["right"] - ortho_bounds_mercator["left"]
                    ) * (ortho_bounds_mercator["top"] - ortho_bounds_mercator["bottom"])
                    overlap_percentage = (
                        overlap_area / ortho_area if ortho_area > 0 else 0
                    )
                    overlap_score = overlap_percentage * 10000  # Scale up for scoring

                    logger.debug(
                        f"Dataset {name}: overlap {overlap_percentage:.1%} of orthophoto"
                    )
                else:
                    # No overlap, calculate distance between centers
                    ds_center_x = (ds_xmin + ds_xmax) / 2
                    ds_center_y = (ds_ymin + ds_ymax) / 2
                    ortho_center_x = (
                        ortho_bounds_mercator["left"] + ortho_bounds_mercator["right"]
                    ) / 2
                    ortho_center_y = (
                        ortho_bounds_mercator["bottom"] + ortho_bounds_mercator["top"]
                    ) / 2

                    distance = math.sqrt(
                        (ds_center_x - ortho_center_x) ** 2
                        + (ds_center_y - ortho_center_y) ** 2
                    )
                    distance_score = distance

            # Extract year from dataset name
            year = 0
            for part in name.split("_"):
                if part.isdigit() and len(part) == 4:
                    try:
                        year = int(part)
                        break
                    except ValueError:
                        continue

            # Calculate scores
            recency_score = year if year > 0 else 1900
            quality_score = math.log10(max(points, 1))

            # Calculate dataset area for specificity bonus
            dataset_area = 0
            if bounds and len(bounds) >= 4:
                if len(bounds) == 6:  # [xmin, ymin, zmin, xmax, ymax, zmax]
                    xmin, ymin, xmax, ymax = bounds[0], bounds[1], bounds[3], bounds[4]
                else:  # [xmin, ymin, xmax, ymax]
                    xmin, ymin, xmax, ymax = bounds[0], bounds[1], bounds[2], bounds[3]
                dataset_area = (xmax - xmin) * (ymax - ymin)  # Area in square meters

            # Regional specificity bonus - prefer smaller, more focused datasets
            specificity_bonus = 0
            if dataset_area > 0:
                area_km2 = dataset_area / 1_000_000
                specificity_bonus = max(0, 100 - math.log10(max(area_km2, 1)) * 20)

            # Regional keyword matching for better geographic classification
            region_bonus = 0
            name_upper = name.upper()

            # Check for Front Range/Denver metro area keywords (good for Boulder)
            front_range_keywords = [
                "DRCOG",
                "DENVER",
                "METRO",
                "FRONT",
                "BOULDER",
                "JEFFCO",
                "ADAMS",
            ]
            # Check for broader regional keywords (less specific)
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

            # Bonus for Front Range/metro area datasets when in Colorado Front Range
            if 39.5 <= lat <= 40.5 and -105.5 <= lon <= -104.5:  # Front Range area
                if any(keyword in name_upper for keyword in front_range_keywords):
                    region_bonus = 500  # Strong preference for metro datasets
                elif any(keyword in name_upper for keyword in broad_keywords):
                    region_bonus = -200  # Penalty for broad regional datasets

            # State/region bonus
            state_bonus = 0
            if (
                lat >= 39 and lat <= 41 and lon >= -109 and lon <= -102
            ):  # Colorado bounds
                if "CO" in name.upper():
                    state_bonus = 300

            # Composite score - balance overlap with geographic appropriateness
            if overlap_score > 0:
                # If there's overlap, use balanced scoring that considers geographic appropriateness
                distance_penalty = (
                    distance_score / 1000 if distance_score != float("inf") else 0
                )
                recency_bonus = (recency_score - 2000) * 8
                quality_bonus = (
                    quality_score * 3
                )  # Reduced weight to prevent domination

                composite_score = (
                    overlap_score * 10  # Moderate weight for overlap
                    + -distance_penalty * 2  # Distance penalty (doubled weight)
                    + recency_bonus
                    + quality_bonus
                    + specificity_bonus
                    + region_bonus
                    + state_bonus
                )
            else:
                # If no overlap, use distance-based scoring (heavily penalized)
                distance_penalty = distance_score / 1000
                composite_score = (
                    -(distance_penalty * 10)
                    + (recency_score - 2000) * 10
                    + quality_score * 5
                    + state_bonus
                )

            scored_datasets.append(
                {
                    "dataset": dataset,
                    "score": composite_score,
                    "overlap_score": overlap_score,
                    "distance_km": (
                        distance_score / 1000
                        if distance_score != float("inf")
                        else float("inf")
                    ),
                    "area_km2": dataset_area / 1_000_000 if dataset_area > 0 else 0,
                    "year": year,
                    "points": points,
                    "name": name,
                    "specificity_bonus": specificity_bonus,
                    "region_bonus": region_bonus,
                }
            )

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
