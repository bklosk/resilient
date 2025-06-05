#!/usr/bin/env python3
"""
Point Cloud I/O Module

Handles loading, saving, and basic operations for point cloud files.
"""

import logging
from pathlib import Path
from typing import Optional
import numpy as np

try:
    import laspy
except ImportError:
    print("ERROR: laspy not found. Install with: pip install laspy lazrs")
    raise

logger = logging.getLogger(__name__)


class PointCloudIO:
    """Handles point cloud file I/O operations."""

    @staticmethod
    def load_point_cloud(file_path: str) -> laspy.LasData:
        """
        Load point cloud with comprehensive error handling.

        Args:
            file_path: Path to LAZ/LAS file

        Returns:
            Loaded point cloud data
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Point cloud file not found: {file_path}")

        try:
            logger.info(f"Loading point cloud: {file_path}")
            las_data = laspy.read(str(file_path))

            # Basic statistics
            num_points = len(las_data.points)
            logger.info(f"Loaded {num_points:,} points")

            # Coordinate bounds
            x_min, x_max = np.min(las_data.x), np.max(las_data.x)
            y_min, y_max = np.min(las_data.y), np.max(las_data.y)
            z_min, z_max = np.min(las_data.z), np.max(las_data.z)

            logger.info(f"Point cloud bounds:")
            logger.info(f"  X: {x_min:.2f} to {x_max:.2f}")
            logger.info(f"  Y: {y_min:.2f} to {y_max:.2f}")
            logger.info(f"  Z: {z_min:.2f} to {z_max:.2f}")

            return las_data

        except Exception as e:
            raise RuntimeError(f"Failed to load point cloud: {e}")

    @staticmethod
    def save_colorized_point_cloud(
        las_data: laspy.LasData,
        colors: np.ndarray,
        valid_mask: np.ndarray,
        output_path: str,
        preserve_original_colors: bool = True,
    ):
        """
        Save trimmed and colorized point cloud to file with optimized performance.
        Only saves points that fall within the orthophoto bounds.

        Args:
            las_data: Original point cloud data
            colors: RGB color array for all points
            valid_mask: Boolean mask indicating which points have valid colors
            output_path: Output file path
            preserve_original_colors: Whether to preserve existing colors as backup
        """
        output_path = Path(output_path)
        logger.info(f"Saving trimmed colorized point cloud to: {output_path}")

        # Filter data to only include points within orthophoto bounds
        valid_points_count = np.sum(valid_mask)
        total_points_count = len(las_data.points)

        logger.info(
            f"Trimming point cloud: {valid_points_count:,}/{total_points_count:,} points "
            f"({100*valid_points_count/total_points_count:.1f}%) within orthophoto bounds"
        )

        # Create new header based on original header
        original_header = las_data.header

        # Point formats that support RGB colors: 2, 3, 5, 7, 8, 10
        rgb_supported_formats = {2, 3, 5, 7, 8, 10}

        # Determine the appropriate point format for colors
        if original_header.point_format.id not in rgb_supported_formats:
            logger.info(
                f"Converting from point format {original_header.point_format.id} to format 2 to support colors"
            )
            new_point_format = laspy.PointFormat(2)
        else:
            new_point_format = original_header.point_format

        # Create new header with the same version but potentially different point format
        header = laspy.LasHeader(
            version=original_header.version, point_format=new_point_format
        )

        # Copy important properties from original header
        header.x_scale = original_header.x_scale
        header.y_scale = original_header.y_scale
        header.z_scale = original_header.z_scale
        header.x_offset = original_header.x_offset
        header.y_offset = original_header.y_offset
        header.z_offset = original_header.z_offset

        # Copy VLRs (including CRS information) from original header
        if hasattr(original_header, "vlrs") and original_header.vlrs:
            header.vlrs = original_header.vlrs.copy()

        # Create new LAS file with filtered data
        colorized_las = laspy.LasData(header)

        # Copy only the valid points
        colorized_las.x = las_data.x[valid_mask]
        colorized_las.y = las_data.y[valid_mask]
        colorized_las.z = las_data.z[valid_mask]

        # Copy other attributes if they exist (only for valid points)
        attributes_to_copy = [
            "intensity",
            "return_number",
            "number_of_returns",
            "classification",
            "scan_angle_rank",
            "user_data",
            "point_source_id",
        ]

        for attr_name in attributes_to_copy:
            if hasattr(las_data, attr_name):
                setattr(
                    colorized_las, attr_name, getattr(las_data, attr_name)[valid_mask]
                )

        # Set colors efficiently (only for valid points)
        valid_colors = colors[valid_mask]
        colorized_las.red = valid_colors[:, 0]
        colorized_las.green = valid_colors[:, 1]
        colorized_las.blue = valid_colors[:, 2]

        # Save to file
        logger.info("Writing trimmed LAZ file...")
        colorized_las.write(str(output_path))

        # File statistics
        file_size = output_path.stat().st_size / (1024 * 1024)  # MB
        logger.info(f"Saved trimmed colorized point cloud ({file_size:.1f} MB)")
        logger.info(
            f"Trimmed from {total_points_count:,} to {valid_points_count:,} points"
        )

    @staticmethod
    def save_point_cloud(las_data: laspy.LasData, output_path: str):
        """
        Save point cloud data to file.

        Args:
            las_data: Point cloud data to save
            output_path: Output file path

        Raises:
            RuntimeError: If save operation fails
        """
        try:
            output_path = Path(output_path)
            logger.info(f"Saving point cloud to: {output_path}")

            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Write the point cloud
            las_data.write(str(output_path))

            # Verify file was created
            if not output_path.exists():
                raise RuntimeError(f"File was not created: {output_path}")

            logger.info(
                f"Successfully saved {len(las_data.points):,} points to {output_path}"
            )

        except Exception as e:
            logger.error(f"Failed to save point cloud to {output_path}: {e}")
            raise RuntimeError(f"Failed to save point cloud: {e}")
