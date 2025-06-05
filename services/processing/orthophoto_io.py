#!/usr/bin/env python3
"""
Orthophoto I/O Module

Handles loading, validation, and basic operations for orthophoto files.
"""

import logging
from pathlib import Path
import rasterio

logger = logging.getLogger(__name__)


class OrthophotoIO:
    """Handles orthophoto file I/O operations."""

    @staticmethod
    def load_orthophoto(file_path: str) -> rasterio.DatasetReader:
        """
        Load orthophoto with validation and fallback for filename mismatches.

        Args:
            file_path: Path to orthophoto file

        Returns:
            Rasterio dataset
        """
        file_path = Path(file_path)

        # If the exact file doesn't exist, try to find orthophoto files in the same directory
        if not file_path.exists():
            parent_dir = file_path.parent
            logger.warning(f"Orthophoto file not found: {file_path}")

            if parent_dir.exists():
                logger.info(
                    f"Searching for orthophoto files in directory: {parent_dir}"
                )

                # Look for any TIFF files that might be orthophotos
                tiff_patterns = ["*.tif", "*.tiff", "*naip*.tif", "*orthophoto*.tif"]
                found_files = []

                for pattern in tiff_patterns:
                    found_files.extend(list(parent_dir.glob(pattern)))

                if found_files:
                    # Use the first valid orthophoto file found
                    for candidate_file in found_files:
                        try:
                            logger.info(f"Trying candidate file: {candidate_file}")
                            with rasterio.open(str(candidate_file)) as test_ds:
                                if test_ds.width > 0 and test_ds.height > 0:
                                    logger.info(
                                        f"Found valid orthophoto: {candidate_file}"
                                    )
                                    file_path = candidate_file
                                    break
                        except Exception:
                            continue
                    else:
                        raise FileNotFoundError(
                            f"No valid orthophoto files found in {parent_dir}"
                        )
                else:
                    raise FileNotFoundError(
                        f"No orthophoto files found in {parent_dir}"
                    )
            else:
                raise FileNotFoundError(f"Orthophoto file not found: {file_path}")

        try:
            logger.info(f"Loading orthophoto: {file_path}")
            dataset = rasterio.open(str(file_path))

            # Basic information
            logger.info(f"Orthophoto info:")
            logger.info(f"  Size: {dataset.width} x {dataset.height}")
            logger.info(f"  Bands: {dataset.count}")
            logger.info(f"  CRS: {dataset.crs}")
            logger.info(f"  Bounds: {dataset.bounds}")

            if dataset.crs is None:
                logger.warning("Orthophoto has no CRS information")

            return dataset

        except Exception as e:
            raise RuntimeError(f"Failed to load orthophoto: {e}")

    @staticmethod
    def get_orthophoto_bounds(file_path: str):
        """
        Extract bounds and CRS from orthophoto file.

        Args:
            file_path: Path to orthophoto file

        Returns:
            Tuple of (bounds_dict, crs_string)
        """
        try:
            with rasterio.open(file_path) as ortho_dataset:
                ortho_bounds = {
                    "left": ortho_dataset.bounds.left,
                    "right": ortho_dataset.bounds.right,
                    "bottom": ortho_dataset.bounds.bottom,
                    "top": ortho_dataset.bounds.top,
                }
                ortho_crs = str(ortho_dataset.crs)
                logger.info(f"Orthophoto bounds extracted: {ortho_bounds}")
                logger.info(f"Orthophoto CRS: {ortho_crs}")
                return ortho_bounds, ortho_crs
        except Exception as e:
            logger.warning(f"Could not extract orthophoto bounds: {e}")
            raise
