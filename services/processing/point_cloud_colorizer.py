"""
Point Cloud Colorization Engine

This module handles the core colorization logic for mapping orthophoto colors
to point cloud coordinates with optimized performance and automatic error handling.
"""

import logging
import numpy as np
import rasterio
import laspy
from typing import Tuple, Optional
from pathlib import Path
from rasterio.transform import from_bounds

from .coordinate_transformer import CoordinateTransformer
from ..data.corrected_orthophoto_downloader import CorrectedOrthophotoDownloader

logger = logging.getLogger(__name__)


class PointCloudColorizer:
    """Handles the core colorization process for point clouds using orthophoto data."""

    def __init__(self, output_dir: Path, create_diagnostics: bool = True):
        """
        Initialize the colorizer.

        Args:
            output_dir: Output directory for temporary files
            create_diagnostics: Whether to create diagnostic plots
        """
        self.output_dir = Path(output_dir)
        self.create_diagnostics = create_diagnostics
        self.transformer = CoordinateTransformer()
        self.downloader = CorrectedOrthophotoDownloader(output_dir)

    def colorize_point_cloud(
        self,
        las_data: laspy.LasData,
        ortho_dataset: rasterio.DatasetReader,
        source_crs: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Colorize point cloud using orthophoto with optimized performance.

        Args:
            las_data: Point cloud data
            ortho_dataset: Orthophoto dataset
            source_crs: Source CRS override

        Returns:
            Tuple of (RGB color array (N, 3) with uint16 values, valid_mask boolean array)
        """
        ortho_crs = str(ortho_dataset.crs) if ortho_dataset.crs else None
        if ortho_crs is None:
            raise ValueError("Orthophoto has no CRS information")

        # Transform coordinates
        transformed_x, transformed_y = (
            self.transformer.transform_point_cloud_to_ortho_crs(
                las_data, ortho_crs, source_crs
            )
        )

        # Enhanced debugging: Print coordinate ranges and bounds
        logger.info("=== COORDINATE ANALYSIS ===")
        logger.info(f"Orthophoto CRS: {ortho_crs}")
        logger.info(f"Orthophoto bounds: {ortho_dataset.bounds}")

        # Point cloud bounds in transformed coordinates
        pc_x_min, pc_x_max = np.min(transformed_x), np.max(transformed_x)
        pc_y_min, pc_y_max = np.min(transformed_y), np.max(transformed_y)
        logger.info(
            f"Point cloud bounds (transformed): X[{pc_x_min:.2f}, {pc_x_max:.2f}], Y[{pc_y_min:.2f}, {pc_y_max:.2f}]"
        )

        # Calculate distances between centers
        ortho_center_x = (ortho_dataset.bounds.left + ortho_dataset.bounds.right) / 2
        ortho_center_y = (ortho_dataset.bounds.bottom + ortho_dataset.bounds.top) / 2
        pc_center_x = (pc_x_min + pc_x_max) / 2
        pc_center_y = (pc_y_min + pc_y_max) / 2

        distance_x = abs(ortho_center_x - pc_center_x)
        distance_y = abs(ortho_center_y - pc_center_y)
        logger.info(
            f"Distance between centers: X={distance_x:.2f}m, Y={distance_y:.2f}m"
        )

        # Check for overlap
        overlap_x = not (
            pc_x_max < ortho_dataset.bounds.left
            or pc_x_min > ortho_dataset.bounds.right
        )
        overlap_y = not (
            pc_y_max < ortho_dataset.bounds.bottom
            or pc_y_min > ortho_dataset.bounds.top
        )
        logger.info(f"Bounds overlap: X={overlap_x}, Y={overlap_y}")

        # If there's no overlap, try alternative CRS transformations
        if not (overlap_x and overlap_y):
            logger.warning(
                "No coordinate overlap detected. Attempting CRS correction..."
            )
            self._handle_coordinate_mismatch(
                las_data,
                ortho_dataset,
                pc_x_min,
                pc_x_max,
                pc_y_min,
                pc_y_max,
                ortho_center_x,
                ortho_center_y,
                pc_center_x,
                pc_center_y,
                ortho_crs,
                source_crs,
            )

        # Create diagnostic plot if requested or if there's coordinate issues
        if self.create_diagnostics or not (overlap_x and overlap_y):
            from alignment_diagnostics import AlignmentDiagnostics

            diagnostics = AlignmentDiagnostics(self.output_dir)
            diagnostics.create_alignment_diagnostic(
                las_data, ortho_dataset, transformed_x, transformed_y
            )

        # Convert to pixel coordinates and extract colors
        colors, valid_mask = self._extract_colors_from_orthophoto(
            las_data,
            ortho_dataset,
            transformed_x,
            transformed_y,
            pc_x_min,
            pc_x_max,
            pc_y_min,
            pc_y_max,
            source_crs,
        )

        return colors, valid_mask

    def _handle_coordinate_mismatch(
        self,
        las_data,
        ortho_dataset,
        pc_x_min,
        pc_x_max,
        pc_y_min,
        pc_y_max,
        ortho_center_x,
        ortho_center_y,
        pc_center_x,
        pc_center_y,
        ortho_crs,
        source_crs,
    ):
        """Handle coordinate system mismatches with detailed diagnostics."""

        # Calculate how far off the data is
        distance_from_center = (
            (ortho_center_x - pc_center_x) ** 2 + (ortho_center_y - pc_center_y) ** 2
        ) ** 0.5

        logger.error(f"SIGNIFICANT COORDINATE MISMATCH DETECTED!")
        logger.error(f"Point cloud center: ({pc_center_x:.6f}, {pc_center_y:.6f})")
        logger.error(f"Orthophoto center: ({ortho_center_x:.6f}, {ortho_center_y:.6f})")
        logger.error(
            f"Distance between centers: {distance_from_center:.6f} degrees ({distance_from_center * 111320:.1f} meters)"
        )

        if distance_from_center > 0.001:  # More than ~100 meters apart
            logger.error(
                "The orthophoto and point cloud appear to be from different locations!"
            )
            logger.error(
                "SOLUTION: Use the correct bounding box for orthophoto download:"
            )

            # Calculate the correct bounding box
            buffer_deg = 0.001  # ~100 meter buffer
            correct_bbox = {
                "west": pc_x_min - buffer_deg,
                "south": pc_y_min - buffer_deg,
                "east": pc_x_max + buffer_deg,
                "north": pc_y_max + buffer_deg,
            }

            bbox_string = f"{correct_bbox['west']:.8f},{correct_bbox['south']:.8f},{correct_bbox['east']:.8f},{correct_bbox['north']:.8f}"
            logger.error(f"Correct bounding box: {bbox_string}")
            logger.error(
                f'Download command: python -m services.data.get_orthophoto --bbox "{bbox_string}"'
            )

            # For this run, we'll proceed with a warning, but results will be poor
            logger.warning(
                "Continuing with current data, but expect very low coverage..."
            )

        # Try transforming orthophoto bounds to point cloud's original CRS for comparison
        self._analyze_crs_compatibility(las_data, ortho_dataset, ortho_crs, source_crs)

    def _analyze_crs_compatibility(
        self, las_data, ortho_dataset, ortho_crs, source_crs
    ):
        """Analyze CRS compatibility between point cloud and orthophoto."""

        original_x_coords = las_data.x
        original_y_coords = las_data.y
        orig_x_min, orig_x_max = np.min(original_x_coords), np.max(original_x_coords)
        orig_y_min, orig_y_max = np.min(original_y_coords), np.max(original_y_coords)

        # Detect or assume point cloud CRS
        pc_crs = source_crs or self.transformer.detect_point_cloud_crs(las_data)
        if pc_crs is None:
            pc_crs = "EPSG:3857"  # Default fallback

        logger.info(f"Original point cloud CRS: {pc_crs}")
        logger.info(
            f"Original point cloud bounds: X[{orig_x_min:.2f}, {orig_x_max:.2f}], Y[{orig_y_min:.2f}, {orig_y_max:.2f}]"
        )

        # Try transforming orthophoto bounds to point cloud CRS
        try:
            from pyproj import Transformer, CRS

            ortho_to_pc_transformer = Transformer.from_crs(
                CRS.from_string(ortho_crs), CRS.from_string(pc_crs), always_xy=True
            )

            # Transform the four corners of the orthophoto bounds
            ortho_left, ortho_bottom = ortho_to_pc_transformer.transform(
                ortho_dataset.bounds.left, ortho_dataset.bounds.bottom
            )
            ortho_right, ortho_top = ortho_to_pc_transformer.transform(
                ortho_dataset.bounds.right, ortho_dataset.bounds.top
            )

            logger.info(
                f"Orthophoto bounds in point cloud CRS: [{ortho_left:.2f}, {ortho_bottom:.2f}, {ortho_right:.2f}, {ortho_top:.2f}]"
            )

            # Check overlap in original coordinate space
            ortho_pc_overlap_x = not (
                orig_x_max < ortho_left or orig_x_min > ortho_right
            )
            ortho_pc_overlap_y = not (
                orig_y_max < ortho_bottom or orig_y_min > ortho_top
            )

            logger.info(
                f"Overlap in original PC CRS: X={ortho_pc_overlap_x}, Y={ortho_pc_overlap_y}"
            )

            if not (ortho_pc_overlap_x and ortho_pc_overlap_y):
                logger.error(
                    "Point cloud and orthophoto do not overlap in any tested coordinate system!"
                )
                logger.error("This suggests they are from different geographic areas.")

        except Exception as e:
            logger.warning(f"Could not transform orthophoto bounds to PC CRS: {e}")

    def _extract_colors_from_orthophoto(
        self,
        las_data,
        ortho_dataset,
        transformed_x,
        transformed_y,
        pc_x_min,
        pc_x_max,
        pc_y_min,
        pc_y_max,
        source_crs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract colors from orthophoto for point cloud coordinates."""

        logger.info("Converting to pixel coordinates...")

        # DEBUG: Examine the geotransform matrix
        logger.info(f"DEBUG - Orthophoto transform matrix: {ortho_dataset.transform}")
        self._debug_transform_matrix(ortho_dataset)

        # Calculate pixel coordinates
        rows, cols = self._calculate_pixel_coordinates(
            ortho_dataset, transformed_x, transformed_y
        )
        pixel_cols = np.array(cols, dtype=np.int32)
        pixel_rows = np.array(rows, dtype=np.int32)

        # DEBUG: Check coordinate transformation results
        self._debug_coordinate_transformation(
            transformed_x, transformed_y, pixel_cols, pixel_rows, ortho_dataset
        )

        # Find valid pixels
        valid_mask = (
            (pixel_cols >= 0)
            & (pixel_cols < ortho_dataset.width)
            & (pixel_rows >= 0)
            & (pixel_rows < ortho_dataset.height)
        )

        num_valid = np.sum(valid_mask)
        total_points = len(las_data.points)

        logger.info(
            f"Valid points for colorization: {num_valid:,}/{total_points:,} "
            f"({100*num_valid/total_points:.1f}%)"
        )

        # Check if coverage is too low and attempt auto-correction
        colors, valid_mask = self._handle_low_coverage(
            las_data,
            ortho_dataset,
            source_crs,
            valid_mask,
            num_valid,
            total_points,
            pc_x_min,
            pc_x_max,
            pc_y_min,
            pc_y_max,
            pixel_cols,
            pixel_rows,
        )

        if colors is not None:
            return colors, valid_mask

        # Extract colors for valid points
        return self._extract_pixel_colors(
            ortho_dataset, valid_mask, pixel_cols, pixel_rows, total_points
        )

    def _debug_transform_matrix(self, ortho_dataset):
        """Debug orthophoto transform matrix."""

        logger.info(f"DEBUG - Transform breakdown:")
        logger.info(f"  Pixel width (X resolution): {ortho_dataset.transform[0]}")
        logger.info(f"  Row rotation: {ortho_dataset.transform[1]}")
        logger.info(f"  Upper-left X coordinate: {ortho_dataset.transform[2]}")
        logger.info(f"  Column rotation: {ortho_dataset.transform[3]}")
        logger.info(f"  Pixel height (Y resolution): {ortho_dataset.transform[4]}")
        logger.info(f"  Upper-left Y coordinate: {ortho_dataset.transform[5]}")

        # Calculate expected transform based on bounds and image dimensions
        expected_pixel_width = (
            ortho_dataset.bounds.right - ortho_dataset.bounds.left
        ) / ortho_dataset.width
        expected_pixel_height = (
            ortho_dataset.bounds.top - ortho_dataset.bounds.bottom
        ) / ortho_dataset.height

        logger.info(f"DEBUG - Expected pixel dimensions:")
        logger.info(f"  Expected pixel width: {expected_pixel_width}")
        logger.info(f"  Expected pixel height: {expected_pixel_height}")
        logger.info(f"  Actual pixel width: {ortho_dataset.transform[0]}")
        logger.info(f"  Actual pixel height: {abs(ortho_dataset.transform[4])}")

    def _calculate_pixel_coordinates(self, ortho_dataset, transformed_x, transformed_y):
        """Calculate pixel coordinates with transform validation."""

        # Check if transform seems reasonable
        expected_pixel_width = (
            ortho_dataset.bounds.right - ortho_dataset.bounds.left
        ) / ortho_dataset.width
        expected_pixel_height = (
            ortho_dataset.bounds.top - ortho_dataset.bounds.bottom
        ) / ortho_dataset.height

        transform_issues = []
        if (
            abs(ortho_dataset.transform[0] - expected_pixel_width)
            > expected_pixel_width * 0.1
        ):
            transform_issues.append("X pixel size mismatch")
        if (
            abs(abs(ortho_dataset.transform[4]) - expected_pixel_height)
            > expected_pixel_height * 0.1
        ):
            transform_issues.append("Y pixel size mismatch")

        if transform_issues:
            logger.warning(
                f"Potential transform issues detected: {', '.join(transform_issues)}"
            )
            logger.warning("Attempting to create corrected transform...")

            # Create a corrected transform based on bounds
            corrected_transform = from_bounds(
                ortho_dataset.bounds.left,
                ortho_dataset.bounds.bottom,
                ortho_dataset.bounds.right,
                ortho_dataset.bounds.top,
                ortho_dataset.width,
                ortho_dataset.height,
            )

            logger.info(f"DEBUG - Corrected transform: {corrected_transform}")

            # Try with corrected transform
            rows, cols = rasterio.transform.rowcol(
                corrected_transform, transformed_x, transformed_y
            )
            logger.info("Using corrected transform for pixel coordinate calculation")
        else:
            rows, cols = rasterio.transform.rowcol(
                ortho_dataset.transform, transformed_x, transformed_y
            )

        return rows, cols

    def _debug_coordinate_transformation(
        self, transformed_x, transformed_y, pixel_cols, pixel_rows, ortho_dataset
    ):
        """Debug coordinate transformation results."""

        logger.info(f"DEBUG - Coordinate transformation samples (first 5):")
        for i in range(min(5, len(transformed_x))):
            logger.info(
                f"  Point {i}: World({transformed_x[i]:.6f}, {transformed_y[i]:.6f}) -> Pixel({pixel_cols[i]}, {pixel_rows[i]})"
            )

        logger.info(f"DEBUG - Pixel coordinate ranges:")
        logger.info(
            f"  Columns: min={np.min(pixel_cols)}, max={np.max(pixel_cols)} (image width: {ortho_dataset.width})"
        )
        logger.info(
            f"  Rows: min={np.min(pixel_rows)}, max={np.max(pixel_rows)} (image height: {ortho_dataset.height})"
        )

    def _handle_low_coverage(
        self,
        las_data,
        ortho_dataset,
        source_crs,
        valid_mask,
        num_valid,
        total_points,
        pc_x_min,
        pc_x_max,
        pc_y_min,
        pc_y_max,
        pixel_cols,
        pixel_rows,
    ):
        """Handle low coverage scenarios with auto-correction."""

        # Check if coverage is too low (less than 10% of points)
        coverage_threshold = 0.10  # 10%
        current_coverage = num_valid / total_points

        if num_valid == 0 or current_coverage < coverage_threshold:
            if num_valid == 0:
                logger.error("No points fall within orthophoto bounds!")
            else:
                logger.warning(
                    f"Low coverage detected: only {current_coverage:.1%} of points are within orthophoto bounds"
                )

            logger.info(
                "Attempting to download corrected orthophoto with proper bounds..."
            )

            # Calculate point cloud bounds for corrected orthophoto
            pc_bounds = {
                "west": pc_x_min,
                "east": pc_x_max,
                "south": pc_y_min,
                "north": pc_y_max,
            }

            try:
                # Download corrected orthophoto
                corrected_ortho_path = self.downloader.download_corrected_orthophoto(
                    pc_bounds
                )

                logger.info(
                    "Successfully downloaded corrected orthophoto. Retrying colorization..."
                )

                # Reload with corrected orthophoto and retry colorization
                with rasterio.open(corrected_ortho_path) as corrected_dataset:
                    # Recursive call with corrected orthophoto
                    return self.colorize_point_cloud(
                        las_data, corrected_dataset, source_crs
                    )

            except Exception as e:
                logger.error(f"Failed to download corrected orthophoto: {e}")
                logger.error(
                    "Proceeding with original orthophoto but expect poor results..."
                )

                if num_valid == 0:
                    raise ValueError(
                        "No points within orthophoto bounds and auto-correction failed"
                    )

        return None, valid_mask

    def _extract_pixel_colors(
        self, ortho_dataset, valid_mask, pixel_cols, pixel_rows, total_points
    ):
        """Extract actual color values from orthophoto pixels."""

        # Initialize color array
        colors = np.zeros((total_points, 3), dtype=np.uint16)

        # Only extract colors for valid points - massive performance improvement
        logger.info("Extracting colors from orthophoto (optimized)...")

        valid_cols = pixel_cols[valid_mask]
        valid_rows = pixel_rows[valid_mask]

        if ortho_dataset.count >= 3:
            # Read RGB bands
            logger.info("Reading RGB bands for valid pixels only...")

            with rasterio.Env():
                red_band = ortho_dataset.read(1)
                green_band = ortho_dataset.read(2)
                blue_band = ortho_dataset.read(3)

                red_values = red_band[valid_rows, valid_cols]
                green_values = green_band[valid_rows, valid_cols]
                blue_values = blue_band[valid_rows, valid_cols]

                self._debug_raw_pixel_values(red_values, green_values, blue_values)

        elif ortho_dataset.count == 1:
            # Grayscale
            logger.info("Reading grayscale band for valid pixels only...")
            gray_band = ortho_dataset.read(1)
            gray_values = gray_band[valid_rows, valid_cols]
            red_values = green_values = blue_values = gray_values

            # DEBUG: Check raw grayscale values
            logger.info(
                f"DEBUG - Raw grayscale value samples (first 5): {gray_values[:5]}"
            )
            logger.info(
                f"DEBUG - Raw grayscale value range: min={np.min(gray_values)}, max={np.max(gray_values)}"
            )

        else:
            raise ValueError(f"Unsupported number of bands: {ortho_dataset.count}")

        # Apply color scaling based on data type
        self._apply_color_scaling(
            ortho_dataset, colors, valid_mask, red_values, green_values, blue_values
        )

        # DEBUG: Check final color values and statistics
        self._debug_final_colors(colors, valid_mask)

        logger.info("Point cloud colorization complete")
        return colors, valid_mask

    def _debug_raw_pixel_values(self, red_values, green_values, blue_values):
        """Debug raw pixel values from orthophoto."""

        logger.info(f"DEBUG - Raw pixel value samples (first 5):")
        logger.info(f"  Red: {red_values[:5]}")
        logger.info(f"  Green: {green_values[:5]}")
        logger.info(f"  Blue: {blue_values[:5]}")
        logger.info(f"DEBUG - Raw pixel value ranges:")
        logger.info(f"  Red: min={np.min(red_values)}, max={np.max(red_values)}")
        logger.info(f"  Green: min={np.min(green_values)}, max={np.max(green_values)}")
        logger.info(f"  Blue: min={np.min(blue_values)}, max={np.max(blue_values)}")

    def _apply_color_scaling(
        self, ortho_dataset, colors, valid_mask, red_values, green_values, blue_values
    ):
        """Apply appropriate color scaling based on orthophoto data type."""

        # Optimized scaling using vectorized operations
        dtype_str = str(ortho_dataset.dtypes[0])
        logger.info(f"DEBUG - Orthophoto data type: {dtype_str}")

        if "uint8" in dtype_str:
            # Scale from 0-255 to 0-65535 using vectorized multiplication
            scale_factor = 257  # 65535 / 255
            colors[valid_mask, 0] = (red_values * scale_factor).astype(np.uint16)
            colors[valid_mask, 1] = (green_values * scale_factor).astype(np.uint16)
            colors[valid_mask, 2] = (blue_values * scale_factor).astype(np.uint16)
            logger.info(f"DEBUG - Applied uint8 scaling (factor: {scale_factor})")

        elif "uint16" in dtype_str:
            # Direct assignment for uint16
            colors[valid_mask, 0] = red_values.astype(np.uint16)
            colors[valid_mask, 1] = green_values.astype(np.uint16)
            colors[valid_mask, 2] = blue_values.astype(np.uint16)
            logger.info("DEBUG - Applied uint16 direct assignment")

        elif "float" in dtype_str:
            # Scale from 0-1 range to 0-65535
            colors[valid_mask, 0] = (red_values * 65535).astype(np.uint16)
            colors[valid_mask, 1] = (green_values * 65535).astype(np.uint16)
            colors[valid_mask, 2] = (blue_values * 65535).astype(np.uint16)
            logger.info("DEBUG - Applied float scaling (factor: 65535)")

        else:
            logger.warning(f"Unknown data type {dtype_str}, using direct conversion")
            colors[valid_mask, 0] = red_values.astype(np.uint16)
            colors[valid_mask, 1] = green_values.astype(np.uint16)
            colors[valid_mask, 2] = blue_values.astype(np.uint16)
            logger.info("DEBUG - Applied direct conversion")

    def _debug_final_colors(self, colors, valid_mask):
        """Debug final color values and statistics."""

        # DEBUG: Check final color values
        logger.info(f"DEBUG - Final color values (first 5 points):")
        for i in range(min(5, len(colors))):
            if valid_mask[i]:
                logger.info(
                    f"  Point {i}: R={colors[i,0]}, G={colors[i,1]}, B={colors[i,2]}"
                )

        # DEBUG: Check color statistics
        valid_colors = colors[valid_mask]
        logger.info(f"DEBUG - Color statistics for {len(valid_colors)} valid points:")
        logger.info(
            f"  Red: min={np.min(valid_colors[:,0])}, max={np.max(valid_colors[:,0])}, mean={np.mean(valid_colors[:,0]):.1f}"
        )
        logger.info(
            f"  Green: min={np.min(valid_colors[:,1])}, max={np.max(valid_colors[:,1])}, mean={np.mean(valid_colors[:,1]):.1f}"
        )
        logger.info(
            f"  Blue: min={np.min(valid_colors[:,2])}, max={np.max(valid_colors[:,2])}, mean={np.mean(valid_colors[:,2]):.1f}"
        )

        # DEBUG: Count non-zero colors
        non_zero_red = np.sum(valid_colors[:, 0] > 0)
        non_zero_green = np.sum(valid_colors[:, 1] > 0)
        non_zero_blue = np.sum(valid_colors[:, 2] > 0)
        logger.info(
            f"DEBUG - Non-zero color counts: R={non_zero_red}, G={non_zero_green}, B={non_zero_blue}"
        )

    def colorize(
        self, point_cloud_path: str, orthophoto_path: str, output_path: str = None
    ) -> str:
        """
        Colorize a point cloud using an orthophoto.

        Args:
            point_cloud_path: Path to input point cloud file
            orthophoto_path: Path to orthophoto file
            output_path: Optional output path (auto-generated if not provided)

        Returns:
            Path to colorized point cloud file
        """
        from .point_cloud_io import PointCloudIO
        from .orthophoto_io import OrthophotoIO

        logger.info(f"Colorizing point cloud: {point_cloud_path}")
        logger.info(f"Using orthophoto: {orthophoto_path}")

        # Load input files
        las_data = PointCloudIO.load_point_cloud(point_cloud_path)
        ortho_dataset = OrthophotoIO.load_orthophoto(orthophoto_path)

        # Perform colorization
        colors, valid_mask = self.colorize_point_cloud(las_data, ortho_dataset)

        # Generate output path if not provided
        if output_path is None:
            input_name = Path(point_cloud_path).stem
            output_path = self.output_dir / f"{input_name}_colorized.laz"
        else:
            output_path = Path(output_path)

        # Save colorized point cloud
        PointCloudIO.save_colorized_point_cloud(
            las_data, colors, valid_mask, str(output_path)
        )

        logger.info(f"Colorized point cloud saved to: {output_path}")
        return str(output_path)
