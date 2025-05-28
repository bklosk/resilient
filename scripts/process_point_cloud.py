#!/usr/bin/env python3
"""
Point Cloud Processing Script: Advanced Colorization

This script provides robust colorization of LiDAR point clouds using orthophotos.
It intelligently handles coordinate transformations, CRS matching, and provides
comprehensive diagnostics.

Features:
- Automatic CRS detection and transformation
- Robust coordinate system handling for common LiDAR/orthophoto combinations
- Intelligent fallback mechanisms for coordinate transformation
- Comprehensive diagnostic outputs
- Support for various orthophoto formats (NAIP, satellite imagery)
- Quality assessment and validation

Usage:
    python process_point_cloud.py --address "1250 Wildwood Road, Boulder, CO"
    python process_point_cloud.py --input_pc data/point_cloud.laz --input_ortho data/orthophoto.tif
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import warnings

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Geospatial libraries
import rasterio
from rasterio.warp import transform_bounds, reproject, Resampling
from rasterio.transform import from_bounds
from pyproj import Transformer, CRS
from pyproj.exceptions import CRSError

# Point cloud processing
try:
    import laspy
except ImportError:
    print("ERROR: laspy not found. Install with: pip install laspy lazrs")
    sys.exit(1)

# Import local modules
from geocode import Geocoder
from get_point_cloud import PointCloudFetcher
from get_orthophoto import NAIPFetcher

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


class PointCloudColorizer:
    """
    Advanced point cloud colorization using orthophotos.
    """

    def __init__(self, output_dir: str = "data"):
        """
        Initialize the colorizer.
        
        Args:
            output_dir: Directory for outputs and diagnostics
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Common CRS for different regions
        self.common_crs_mappings = {
            'colorado': {
                'utm': 'EPSG:26913',  # UTM Zone 13N
                'state_plane': 'EPSG:2232',  # NAD83 Colorado Central ftUS
                'state_plane_m': 'EPSG:26954',  # NAD83 Colorado Central meters
            },
            'california': {
                'utm_10': 'EPSG:26910',  # UTM Zone 10N
                'utm_11': 'EPSG:26911',  # UTM Zone 11N
                'state_plane': 'EPSG:2227',  # NAD83 California Zone 3 ftUS
            }
        }

    def load_point_cloud(self, file_path: str) -> laspy.LasData:
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

    def load_orthophoto(self, file_path: str) -> rasterio.DatasetReader:
        """
        Load orthophoto with validation.
        
        Args:
            file_path: Path to orthophoto file
            
        Returns:
            Rasterio dataset
        """
        file_path = Path(file_path)
        if not file_path.exists():
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

    def detect_point_cloud_crs(self, las_data: laspy.LasData) -> Optional[str]:
        """
        Intelligently detect point cloud CRS.
        
        Args:
            las_data: Point cloud data
            
        Returns:
            CRS string or None if detection fails
        """
        # Try header CRS first
        if (hasattr(las_data, 'header') and 
            hasattr(las_data.header, 'crs') and 
            las_data.header.crs):
            crs_str = str(las_data.header.crs)
            logger.info(f"Point cloud CRS from header: {crs_str}")
            return crs_str
        
        # Try to infer from coordinate ranges
        x_coords = las_data.x
        y_coords = las_data.y
        
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        
        # Check coordinate ranges to guess CRS
        if (3000000 < x_min < 3200000 and 1700000 < y_min < 1900000):
            # Likely Colorado State Plane (feet)
            logger.info("Detected likely Colorado State Plane coordinates (feet)")
            return "EPSG:2232"
        elif (400000 < x_min < 800000 and 4000000 < y_min < 5000000):
            # Likely UTM Zone 13N (meters)
            logger.info("Detected likely UTM Zone 13N coordinates")
            return "EPSG:26913"
        elif (3000000 < x_min < 3200000 and 500000 < y_min < 700000):
            # Likely Colorado State Plane (meters)
            logger.info("Detected likely Colorado State Plane coordinates (meters)")
            return "EPSG:26954"
        
        logger.warning("Could not automatically detect point cloud CRS")
        return None

    def transform_point_cloud_to_ortho_crs(
        self, 
        las_data: laspy.LasData, 
        ortho_crs: str,
        source_crs: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform point cloud coordinates to orthophoto CRS.
        
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
            source_crs = self.detect_point_cloud_crs(las_data)
        
        if source_crs is None:
            raise ValueError("Cannot determine point cloud CRS")
        
        logger.info(f"Transforming coordinates: {source_crs} -> {ortho_crs}")
        
        try:
            # Validate CRS
            source_crs_obj = CRS.from_string(source_crs)
            target_crs_obj = CRS.from_string(ortho_crs)
            
            # Create transformer
            transformer = Transformer.from_crs(
                source_crs_obj, target_crs_obj, always_xy=True
            )
            
            # Transform coordinates in batches for large datasets
            batch_size = 100000
            total_points = len(x_coords)
            
            transformed_x = np.zeros_like(x_coords)
            transformed_y = np.zeros_like(y_coords)
            
            logger.info(f"Transforming {total_points:,} points...")
            
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

    def create_alignment_diagnostic(
        self,
        las_data: laspy.LasData,
        ortho_dataset: rasterio.DatasetReader,
        transformed_x: np.ndarray,
        transformed_y: np.ndarray,
        output_name: str = "alignment_diagnostic.png"
    ):
        """
        Create diagnostic plot showing point cloud and orthophoto alignment.
        
        Args:
            las_data: Original point cloud data
            ortho_dataset: Orthophoto dataset
            transformed_x: Transformed X coordinates
            transformed_y: Transformed Y coordinates
            output_name: Output filename
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot 1: Overview with bounds
        ortho_bounds = ortho_dataset.bounds
        
        # Sample points for visualization
        sample_size = min(5000, len(transformed_x))
        sample_indices = np.random.choice(len(transformed_x), sample_size, replace=False)
        
        sample_x = transformed_x[sample_indices]
        sample_y = transformed_y[sample_indices]
        
        # Orthophoto bounds
        rect = plt.Rectangle(
            (ortho_bounds.left, ortho_bounds.bottom),
            ortho_bounds.right - ortho_bounds.left,
            ortho_bounds.top - ortho_bounds.bottom,
            linewidth=2, edgecolor='red', facecolor='none',
            label='Orthophoto Bounds'
        )
        ax1.add_patch(rect)
        
        # Point cloud
        ax1.scatter(sample_x, sample_y, s=1, c='blue', alpha=0.6, 
                   label=f'Point Cloud ({sample_size:,} points)')
        
        ax1.set_title('Coordinate Alignment Overview')
        ax1.set_xlabel(f'X ({ortho_dataset.crs})')
        ax1.set_ylabel(f'Y ({ortho_dataset.crs})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Intersection analysis
        # Find points within orthophoto bounds
        within_bounds = (
            (sample_x >= ortho_bounds.left) & 
            (sample_x <= ortho_bounds.right) &
            (sample_y >= ortho_bounds.bottom) & 
            (sample_y <= ortho_bounds.top)
        )
        
        points_in_bounds = np.sum(within_bounds)
        
        ax2.scatter(sample_x[~within_bounds], sample_y[~within_bounds], 
                   s=1, c='red', alpha=0.6, label=f'Outside bounds ({np.sum(~within_bounds):,})')
        ax2.scatter(sample_x[within_bounds], sample_y[within_bounds], 
                   s=1, c='green', alpha=0.6, label=f'Within bounds ({points_in_bounds:,})')
        
        ax2.add_patch(plt.Rectangle(
            (ortho_bounds.left, ortho_bounds.bottom),
            ortho_bounds.right - ortho_bounds.left,
            ortho_bounds.top - ortho_bounds.bottom,
            linewidth=2, edgecolor='black', facecolor='none'
        ))
        
        ax2.set_title('Point Distribution Analysis')
        ax2.set_xlabel(f'X ({ortho_dataset.crs})')
        ax2.set_ylabel(f'Y ({ortho_dataset.crs})')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / output_name
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Diagnostic plot saved: {output_path}")
        logger.info(f"Points within orthophoto bounds: {points_in_bounds:,}/{sample_size:,} "
                   f"({100*points_in_bounds/sample_size:.1f}%)")

    def colorize_point_cloud(
        self,
        las_data: laspy.LasData,
        ortho_dataset: rasterio.DatasetReader,
        source_crs: Optional[str] = None
    ) -> np.ndarray:
        """
        Colorize point cloud using orthophoto.
        
        Args:
            las_data: Point cloud data
            ortho_dataset: Orthophoto dataset
            source_crs: Source CRS override
            
        Returns:
            RGB color array (N, 3) with uint16 values
        """
        ortho_crs = str(ortho_dataset.crs) if ortho_dataset.crs else None
        if ortho_crs is None:
            raise ValueError("Orthophoto has no CRS information")
        
        # Transform coordinates
        transformed_x, transformed_y = self.transform_point_cloud_to_ortho_crs(
            las_data, ortho_crs, source_crs
        )
        
        # Create diagnostic plot
        self.create_alignment_diagnostic(
            las_data, ortho_dataset, transformed_x, transformed_y
        )
        
        # Convert to pixel coordinates
        logger.info("Converting to pixel coordinates...")
        rows, cols = rasterio.transform.rowcol(
            ortho_dataset.transform, transformed_x, transformed_y
        )
        
        pixel_cols = np.array(cols, dtype=np.int32)
        pixel_rows = np.array(rows, dtype=np.int32)
        
        # Find valid pixels
        valid_mask = (
            (pixel_cols >= 0) & (pixel_cols < ortho_dataset.width) &
            (pixel_rows >= 0) & (pixel_rows < ortho_dataset.height)
        )
        
        num_valid = np.sum(valid_mask)
        total_points = len(las_data.points)
        
        logger.info(f"Valid points for colorization: {num_valid:,}/{total_points:,} "
                   f"({100*num_valid/total_points:.1f}%)")
        
        if num_valid == 0:
            logger.error("No points fall within orthophoto bounds!")
            raise ValueError("No points within orthophoto bounds")
        
        # Initialize color array
        colors = np.zeros((total_points, 3), dtype=np.uint16)
        
        # Read orthophoto bands
        logger.info("Extracting colors from orthophoto...")
        
        if ortho_dataset.count >= 3:
            # RGB bands
            red_band = ortho_dataset.read(1)
            green_band = ortho_dataset.read(2) 
            blue_band = ortho_dataset.read(3)
            
            # Extract colors for valid points
            valid_cols = pixel_cols[valid_mask]
            valid_rows = pixel_rows[valid_mask]
            
            red_values = red_band[valid_rows, valid_cols]
            green_values = green_band[valid_rows, valid_cols]
            blue_values = blue_band[valid_rows, valid_cols]
            
        elif ortho_dataset.count == 1:
            # Grayscale
            gray_band = ortho_dataset.read(1)
            valid_cols = pixel_cols[valid_mask]
            valid_rows = pixel_rows[valid_mask]
            
            gray_values = gray_band[valid_rows, valid_cols]
            red_values = green_values = blue_values = gray_values
            
        else:
            raise ValueError(f"Unsupported number of bands: {ortho_dataset.count}")
        
        # Scale values to uint16 range
        dtype_str = str(ortho_dataset.dtypes[0])
        
        if 'uint8' in dtype_str:
            # Scale from 0-255 to 0-65535
            scale_factor = 257  # 65535 / 255
            red_scaled = red_values.astype(np.uint16) * scale_factor
            green_scaled = green_values.astype(np.uint16) * scale_factor
            blue_scaled = blue_values.astype(np.uint16) * scale_factor
            
        elif 'uint16' in dtype_str:
            # Already in correct range
            red_scaled = red_values.astype(np.uint16)
            green_scaled = green_values.astype(np.uint16)
            blue_scaled = blue_values.astype(np.uint16)
            
        elif 'float' in dtype_str:
            # Assume 0-1 range, scale to 0-65535
            red_scaled = (red_values * 65535).astype(np.uint16)
            green_scaled = (green_values * 65535).astype(np.uint16)
            blue_scaled = (blue_values * 65535).astype(np.uint16)
            
        else:
            logger.warning(f"Unknown data type {dtype_str}, using direct conversion")
            red_scaled = red_values.astype(np.uint16)
            green_scaled = green_values.astype(np.uint16)
            blue_scaled = blue_values.astype(np.uint16)
        
        # Assign colors to valid points
        colors[valid_mask, 0] = red_scaled
        colors[valid_mask, 1] = green_scaled
        colors[valid_mask, 2] = blue_scaled
        
        logger.info("Point cloud colorization complete")
        
        return colors

    def save_colorized_point_cloud(
        self,
        las_data: laspy.LasData,
        colors: np.ndarray,
        output_path: str,
        preserve_original_colors: bool = True
    ):
        """
        Save colorized point cloud to file.
        
        Args:
            las_data: Original point cloud data
            colors: RGB color array
            output_path: Output file path
            preserve_original_colors: Whether to preserve existing colors as backup
        """
        output_path = Path(output_path)
        logger.info(f"Saving colorized point cloud to: {output_path}")
        
        # Create new LAS data with colors
        header = las_data.header
        
        # Point formats that support RGB colors: 2, 3, 5, 7, 8, 10
        rgb_supported_formats = {2, 3, 5, 7, 8, 10}
        
        # Ensure point format supports colors
        if header.point_format.id not in rgb_supported_formats:
            logger.info(f"Converting from point format {header.point_format.id} to format 2 to support colors")
            header.point_format = laspy.PointFormat(2)
        
        # Create new LAS file
        colorized_las = laspy.LasData(header)
        
        # Copy all point data
        colorized_las.x = las_data.x
        colorized_las.y = las_data.y
        colorized_las.z = las_data.z
        
        # Copy other attributes if they exist
        for attr_name in ['intensity', 'return_number', 'number_of_returns', 
                         'classification', 'scan_angle_rank', 'user_data', 'point_source_id']:
            if hasattr(las_data, attr_name):
                setattr(colorized_las, attr_name, getattr(las_data, attr_name))
        
        # Set colors
        colorized_las.red = colors[:, 0]
        colorized_las.green = colors[:, 1]
        colorized_las.blue = colors[:, 2]
        
        # Save to file
        colorized_las.write(str(output_path))
        
        # File statistics
        file_size = output_path.stat().st_size / (1024 * 1024)  # MB
        logger.info(f"Saved colorized point cloud ({file_size:.1f} MB)")

    def process_from_address(self, address: str) -> str:
        """
        Complete workflow: fetch data and colorize point cloud for an address.
        
        Args:
            address: Street address
            
        Returns:
            Path to colorized point cloud file
        """
        logger.info(f"Processing address: {address}")
        
        # Initialize fetchers
        geocoder = Geocoder()
        pc_fetcher = PointCloudFetcher()
        ortho_fetcher = NAIPFetcher()
        
        try:
            # Geocode address
            lat, lon = geocoder.geocode_address(address)
            logger.info(f"Coordinates: {lat:.6f}, {lon:.6f}")
            
            # Generate bounding box for point cloud search
            bbox = pc_fetcher.generate_bounding_box(lat, lon, buffer_km=1.0)
            
            # Search for point cloud data
            logger.info("Searching for LiDAR data...")
            products = pc_fetcher.search_lidar_products(bbox)
            
            if not products:
                raise RuntimeError("No LiDAR data found for this location")
            
            laz_products = pc_fetcher.filter_laz_products(products)
            
            if not laz_products:
                raise RuntimeError("No LAZ format LiDAR data found")
            
            # Download point cloud
            pc_output_path = self.output_dir / "point_cloud.laz"
            downloaded_pc = pc_fetcher.download_point_cloud(
                laz_products[0], str(self.output_dir)
            )
            
            if not downloaded_pc:
                raise RuntimeError("Failed to download point cloud")
            
            # Load the point cloud to get its actual bounds
            logger.info("Analyzing downloaded point cloud bounds...")
            las_data = self.load_point_cloud(downloaded_pc)
            
            # Get point cloud bounds in geographic coordinates
            pc_crs = self.detect_point_cloud_crs(las_data)
            if pc_crs:
                try:
                    # Transform point cloud bounds to lat/lon for orthophoto search
                    from pyproj import Transformer
                    transformer = Transformer.from_crs(pc_crs, "EPSG:4326", always_xy=True)
                    
                    # Get corner coordinates
                    x_min, x_max = np.min(las_data.x), np.max(las_data.x)
                    y_min, y_max = np.min(las_data.y), np.max(las_data.y)
                    
                    # Transform corners to lat/lon
                    lon_min, lat_min = transformer.transform(x_min, y_min)
                    lon_max, lat_max = transformer.transform(x_max, y_max)
                    
                    # Use center of point cloud for orthophoto search
                    center_lat = (lat_min + lat_max) / 2
                    center_lon = (lon_min + lon_max) / 2
                    
                    logger.info(f"Using point cloud center: {center_lat:.6f}, {center_lon:.6f}")
                    
                    # Create a temporary address-like string for the center point
                    temp_address = f"{center_lat:.6f}, {center_lon:.6f}"
                    
                    # Get orthophoto using point cloud center coordinates
                    logger.info("Fetching orthophoto for point cloud area...")
                    
                    # Directly use the search method with center coordinates
                    items = ortho_fetcher.search_naip_items(center_lat, center_lon)
                    if items:
                        best_item = ortho_fetcher.get_best_item(items)
                        if best_item:
                            metadata = ortho_fetcher.extract_metadata(best_item)
                            download_url = ortho_fetcher.get_download_url(best_item)
                            
                            # Save metadata
                            ortho_fetcher.save_metadata(metadata, str(self.output_dir))
                            
                            # Download image
                            filename = f"naip_orthophoto_{metadata.get('id', 'unknown')}.tif"
                            output_path = str(self.output_dir / filename)
                            ortho_fetcher.download_orthophoto(download_url, output_path)
                            
                            logger.info(f"Orthophoto matched to point cloud area")
                        else:
                            raise Exception("No suitable NAIP item found for point cloud area")
                    else:
                        raise Exception("No NAIP imagery found for point cloud area")
                    
                except Exception as e:
                    logger.warning(f"Failed to use point cloud bounds for orthophoto search: {e}")
                    logger.info("Falling back to original address-based search...")
                    ortho_url, ortho_metadata = ortho_fetcher.get_orthophoto_for_address(
                        address, str(self.output_dir), download=True
                    )
            else:
                # Fallback to address-based search
                logger.info("Fetching orthophoto...")
                ortho_url, ortho_metadata = ortho_fetcher.get_orthophoto_for_address(
                    address, str(self.output_dir), download=True
                )
            
            # Find the downloaded orthophoto file
            ortho_files = list(self.output_dir.glob("naip_orthophoto_*.tif"))
            if not ortho_files:
                raise RuntimeError("Orthophoto download failed")
            
            ortho_path = ortho_files[0]
            
            # Process the data
            return self.process_files(str(downloaded_pc), str(ortho_path))
            
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            raise

    def process_files(
        self, 
        point_cloud_path: str, 
        orthophoto_path: str,
        output_name: Optional[str] = None
    ) -> str:
        """
        Process existing point cloud and orthophoto files.
        
        Args:
            point_cloud_path: Path to point cloud file
            orthophoto_path: Path to orthophoto file
            output_name: Output filename (optional)
            
        Returns:
            Path to colorized point cloud file
        """
        logger.info("Starting point cloud colorization...")
        
        # Load data
        las_data = self.load_point_cloud(point_cloud_path)
        ortho_dataset = self.load_orthophoto(orthophoto_path)
        
        try:
            # Colorize
            colors = self.colorize_point_cloud(las_data, ortho_dataset)
            
            # Generate output filename
            if output_name is None:
                pc_name = Path(point_cloud_path).stem
                output_name = f"{pc_name}_colorized.laz"
            
            output_path = self.output_dir / output_name
            
            # Save result
            self.save_colorized_point_cloud(las_data, colors, str(output_path))
            
            # Create summary report
            self.create_summary_report(
                point_cloud_path, orthophoto_path, str(output_path), colors
            )
            
            return str(output_path)
            
        finally:
            ortho_dataset.close()

    def create_summary_report(
        self,
        pc_path: str,
        ortho_path: str,
        output_path: str,
        colors: np.ndarray
    ):
        """
        Create a summary report of the colorization process.
        
        Args:
            pc_path: Input point cloud path
            ortho_path: Input orthophoto path
            output_path: Output point cloud path
            colors: Color array
        """
        report = {
            'input_point_cloud': str(pc_path),
            'input_orthophoto': str(ortho_path),
            'output_point_cloud': str(output_path),
            'processing_stats': {
                'total_points': len(colors),
                'colorized_points': np.sum(np.any(colors > 0, axis=1)),
                'colorization_rate': float(np.sum(np.any(colors > 0, axis=1)) / len(colors))
            },
            'color_stats': {
                'mean_red': float(np.mean(colors[:, 0])),
                'mean_green': float(np.mean(colors[:, 1])),
                'mean_blue': float(np.mean(colors[:, 2])),
                'max_red': int(np.max(colors[:, 0])),
                'max_green': int(np.max(colors[:, 1])),
                'max_blue': int(np.max(colors[:, 2]))
            }
        }
        
        report_path = self.output_dir / 'colorization_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Summary report saved: {report_path}")
        logger.info(f"Colorization rate: {report['processing_stats']['colorization_rate']:.1%}")


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Advanced Point Cloud Colorization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process by address (downloads data automatically)
  python process_point_cloud.py --address "1250 Wildwood Road, Boulder, CO"
  
  # Process existing files
  python process_point_cloud.py --input_pc data/point_cloud.laz --input_ortho data/orthophoto.tif
  
  # Specify output directory
  python process_point_cloud.py --address "123 Main St" --output_dir results/
        """
    )
    
    # Input options
    parser.add_argument(
        '--address',
        type=str,
        help='Address to process (automatically downloads data)'
    )
    
    # File input arguments (both required together if using files)
    parser.add_argument(
        '--input_pc',
        type=str,
        help='Path to input point cloud file (LAZ/LAS)'
    )
    parser.add_argument(
        '--input_ortho',
        type=str,
        help='Path to input orthophoto file'
    )
    
    # Output options
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data',
        help='Output directory (default: data)'
    )
    parser.add_argument(
        '--output_name',
        type=str,
        help='Output filename (default: auto-generated)'
    )
    
    # Processing options
    parser.add_argument(
        '--source_crs',
        type=str,
        help='Override source CRS for point cloud (e.g., EPSG:2232)'
    )
    
    args = parser.parse_args()
    
    # Validate input arguments
    if args.address and (args.input_pc or args.input_ortho):
        parser.error("Cannot use --address with --input_pc or --input_ortho")
    elif not args.address and (not args.input_pc or not args.input_ortho):
        parser.error("Must provide either --address OR both --input_pc and --input_ortho")
    
    try:
        # Initialize colorizer
        colorizer = PointCloudColorizer(args.output_dir)
        
        if args.address:
            # Process by address
            output_path = colorizer.process_from_address(args.address)
        else:
            # Process existing files
            output_path = colorizer.process_files(
                args.input_pc, 
                args.input_ortho,
                args.output_name
            )
        
        logger.info(f"SUCCESS: Colorized point cloud saved to {output_path}")
        return 0
        
    except Exception as e:
        logger.error(f"FAILED: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
