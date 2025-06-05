#!/usr/bin/env python3
"""
Corrected Orthophoto Downloader Module

Handles downloading corrected orthophotos when point cloud bounds don't match.
"""

import logging
import json
from pathlib import Path
from typing import Dict
import numpy as np
import requests

logger = logging.getLogger(__name__)


class CorrectedOrthophotoDownloader:
    """Downloads corrected orthophotos with proper bounds for point clouds."""

    def __init__(self, output_dir: Path):
        """
        Initialize downloader.

        Args:
            output_dir: Directory for downloaded files
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)

    def download_corrected_orthophoto(
        self, pc_bounds: Dict[str, float], output_path: str = None
    ) -> str:
        """
        Download orthophoto with bounds that properly cover the point cloud.

        Args:
            pc_bounds: Point cloud bounds in WGS84 {'west': x_min, 'east': x_max, 'south': y_min, 'north': y_max}
            output_path: Optional path for output file

        Returns:
            Path to downloaded orthophoto
        """
        if output_path is None:
            output_path = str(self.output_dir / "corrected_orthophoto.tif")

        logger.info(f"Downloading corrected orthophoto with bounds: {pc_bounds}")

        # Add 10% buffer to ensure full coverage
        width_deg = pc_bounds["east"] - pc_bounds["west"]
        height_deg = pc_bounds["north"] - pc_bounds["south"]
        buffer_x = width_deg * 0.1
        buffer_y = height_deg * 0.1

        min_lon = pc_bounds["west"] - buffer_x
        min_lat = pc_bounds["south"] - buffer_y
        max_lon = pc_bounds["east"] + buffer_x
        max_lat = pc_bounds["north"] + buffer_y

        bbox = f"{min_lon},{min_lat},{max_lon},{max_lat}"

        # Calculate appropriate image size (aim for ~1m resolution)
        width_m = (
            width_deg
            * 111320
            * np.cos(np.radians((pc_bounds["north"] + pc_bounds["south"]) / 2))
        )
        height_m = height_deg * 111320

        # Target ~2 meter per pixel for reasonable file size, but cap at 2000px
        target_width = min(max(int(width_m / 2), 200), 2000)
        target_height = min(max(int(height_m / 2), 200), 2000)
        image_size = f"{target_width},{target_height}"

        logger.info(f"Requesting image size: {target_width} x {target_height}")
        logger.info(f"Coverage area: {width_m:.0f}m x {height_m:.0f}m")

        # Use USGS NAIPPlus ImageServer (same as get_orthophoto.py)
        service_url = "https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPPlus/ImageServer/exportImage"

        # Try multiple sizes in case the requested size is too large
        fallback_sizes = ["2048,2048", "1024,1024", "512,512"]
        sizes_to_try = [image_size] + fallback_sizes

        # Remove duplicates while preserving order
        seen = set()
        unique_sizes = []
        for size in sizes_to_try:
            if size not in seen:
                seen.add(size)
                unique_sizes.append(size)

        last_error = None

        for attempt, size in enumerate(unique_sizes):
            params = {
                "bbox": bbox,
                "bboxSR": 4326,  # WGS84 coordinate system
                "size": size,  # Output image size in pixels
                "imageSR": 4326,  # Output coordinate system
                "format": "tiff",  # Output format
                "f": "image",  # Response format
            }

            try:
                logger.info(
                    f"Making request to NAIP service (attempt {attempt + 1}/{len(unique_sizes)}):"
                )
                logger.info(f"  Bounding box: {bbox}")
                logger.info(f"  Image size: {size}")

                response = requests.get(service_url, params=params, timeout=120)
                response.raise_for_status()

                file_size = len(response.content)

                # Check if response is actually an image by examining content
                if file_size < 1000:  # Very small response is likely an error
                    try:
                        error_data = response.json()
                        error_msg = error_data.get("error", {}).get(
                            "message", "Unknown service error"
                        )
                        if (
                            "size limit" in error_msg.lower()
                            and attempt < len(unique_sizes) - 1
                        ):
                            logger.info(
                                f"  Size {size} too large, trying smaller size..."
                            )
                            last_error = Exception(f"NAIP service error: {error_msg}")
                            continue
                        else:
                            raise Exception(f"NAIP service error: {error_msg}")
                    except json.JSONDecodeError:
                        pass  # Not JSON, continue with normal processing

                # Additional check: if content type suggests image but size is suspiciously small
                content_type = response.headers.get("content-type", "")
                if content_type.startswith("image/") and file_size < 1000:
                    if attempt < len(unique_sizes) - 1:
                        logger.info(
                            f"  Received very small image ({file_size} bytes), trying smaller size..."
                        )
                        last_error = Exception(
                            f"Received very small image ({file_size} bytes), likely an error response"
                        )
                        continue
                    else:
                        raise Exception(
                            f"Received very small image ({file_size} bytes), likely an error response"
                        )

                # Save the image
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                Path(output_path).write_bytes(response.content)

                logger.info(
                    f"Successfully downloaded corrected orthophoto: {output_path}"
                )
                logger.info(f"File size: {file_size / 1024 / 1024:.1f} MB")
                if size != image_size:
                    logger.info(
                        f"Note: Used fallback size {size} instead of requested {image_size}"
                    )

                # Save metadata
                metadata = {
                    "bbox": bbox,
                    "bbox_array": [min_lon, min_lat, max_lon, max_lat],
                    "image_size": [int(size.split(",")[0]), int(size.split(",")[1])],
                    "crs": "EPSG:4326",
                    "source": "USGS NAIPPlus - Auto-corrected",
                    "point_cloud_bounds": pc_bounds,
                    "service_url": service_url,
                    "request_params": params,
                }

                metadata_path = str(Path(output_path).with_suffix(".json"))
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)

                return output_path

            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}")
                last_error = e
                if attempt < len(unique_sizes) - 1:
                    continue
                else:
                    break

        # If we get here, all attempts failed
        logger.error(f"All download attempts failed. Last error: {last_error}")
        raise Exception(
            f"Failed to download corrected orthophoto after {len(unique_sizes)} attempts: {last_error}"
        )
