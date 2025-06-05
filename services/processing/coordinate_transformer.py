#!/usr/bin/env python3
"""
Coordinate Transformation Module

Handles coordinate system detection and transformation between different CRS.
"""

import logging
from typing import Tuple, Optional
import numpy as np
from tqdm import tqdm

from pyproj import Transformer, CRS
import laspy

import sys
import os
from pathlib import Path

# Add the services directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.utils import CRSUtils

logger = logging.getLogger(__name__)


class CoordinateTransformer:
    """Handles coordinate transformations between different CRS."""

    @staticmethod
    def detect_point_cloud_crs(las_data: laspy.LasData) -> Optional[str]:
        """Detect point cloud CRS using centralized utilities."""
        return CRSUtils.detect_point_cloud_crs(las_data)

    @staticmethod
    def transform_point_cloud_to_ortho_crs(
        las_data: laspy.LasData, ortho_crs: str, source_crs: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform point cloud coordinates to orthophoto CRS with optimized batching.

        Args:
            las_data: Point cloud data
            ortho_crs: Target CRS (orthophoto CRS)
            source_crs: Source CRS (if known)

        Returns:
            Tuple of (transformed_x, transformed_y)
        """
        x_coords = las_data.x
        y_coords = las_data.y

        # Detect source CRS if not provided
        if source_crs is None:
            source_crs = CoordinateTransformer.detect_point_cloud_crs(las_data)

        if source_crs is None:
            # Enhanced fallback: Analyze coordinate ranges to guess CRS
            x_min, x_max = np.min(x_coords), np.max(x_coords)
            y_min, y_max = np.min(y_coords), np.max(y_coords)

            logger.warning(f"CRS detection failed. Analyzing coordinate ranges:")
            logger.warning(
                f"X: {x_min:.2f} to {x_max:.2f}, Y: {y_min:.2f} to {y_max:.2f}"
            )

            # Heuristic: Large negative X values and large positive Y values suggest Web Mercator
            if (
                x_min < -1000000
                and abs(x_min) < 20037508
                and y_min > 1000000
                and y_max < 20037508
            ):
                source_crs = "EPSG:3857"
                logger.info(
                    f"Fallback: Assuming Web Mercator (EPSG:3857) based on coordinate ranges"
                )
            # Geographic coordinates
            elif -180 <= x_min <= 180 and -90 <= y_min <= 90:
                source_crs = "EPSG:4326"
                logger.info(
                    f"Fallback: Assuming WGS84 (EPSG:4326) based on coordinate ranges"
                )
            # UTM-like coordinates
            elif 100000 < x_min < 900000 and 1000000 < y_min < 10000000:
                source_crs = "EPSG:26913"  # Assume UTM Zone 13N for western US
                logger.info(
                    f"Fallback: Assuming UTM Zone 13N (EPSG:26913) based on coordinate ranges"
                )
            else:
                raise ValueError(
                    f"Cannot determine point cloud CRS. Coordinate ranges: X[{x_min:.0f}, {x_max:.0f}], Y[{y_min:.0f}, {y_max:.0f}]"
                )

        logger.info(f"Transforming coordinates: {source_crs} -> {ortho_crs}")

        try:
            # Validate CRS
            source_crs_obj = CRS.from_string(source_crs)
            target_crs_obj = CRS.from_string(ortho_crs)

            # Create transformer
            transformer = Transformer.from_crs(
                source_crs_obj, target_crs_obj, always_xy=True
            )

            # Optimized batch size for faster processing
            batch_size = 500000  # Increased from 100k
            total_points = len(x_coords)

            transformed_x = np.zeros_like(x_coords)
            transformed_y = np.zeros_like(y_coords)

            logger.info(
                f"Transforming {total_points:,} points in batches of {batch_size:,}..."
            )

            for i in tqdm(range(0, total_points, batch_size), desc="Transforming"):
                end_idx = min(i + batch_size, total_points)

                batch_x = x_coords[i:end_idx]
                batch_y = y_coords[i:end_idx]

                trans_x, trans_y = transformer.transform(batch_x, batch_y)

                transformed_x[i:end_idx] = trans_x
                transformed_y[i:end_idx] = trans_y

            return transformed_x, transformed_y

        except Exception as e:
            logger.error(f"Transformation failed: {e}")
            raise RuntimeError(f"Coordinate transformation failed: {e}")
