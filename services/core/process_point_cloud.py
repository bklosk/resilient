#!/usr/bin/env python3
"""
Advanced Point Cloud Colorization - Refactored Modular Version

This script provides a complete workflow for colorizing point clouds using orthophotos,
with automatic data fetching, coordinate transformation, and optimized processing.

The script has been refactored into a modular architecture with the following components:
- Point Cloud I/O (loading and saving)
- Orthophoto I/O (loading and validation)
- Coordinate transformation (CRS detection and conversion)
- Alignment diagnostics (visualization and analysis)
- Data fetching (automatic download of point clouds and orthophotos)
- Auto-correction (download corrected orthophotos when needed)
- Core colorization engine (main processing logic)
- Summary reporting (detailed analysis and reports)

Author: GitHub Copilot
License: MIT
"""

import sys
import argparse
import logging
import time
from pathlib import Path
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("point_cloud_processing.log"),
    ],
)

logger = logging.getLogger(__name__)

# Import our modular components
from ..processing.point_cloud_io import PointCloudIO
from ..processing.orthophoto_io import OrthophotoIO
from ..processing.coordinate_transformer import CoordinateTransformer
from ..processing.alignment_diagnostics import AlignmentDiagnostics
from ..data.data_fetcher import DataFetcher
from ..data.corrected_orthophoto_downloader import CorrectedOrthophotoDownloader
from ..processing.point_cloud_colorizer import PointCloudColorizer
from ..visualization.summary_reporter import SummaryReporter

# External dependencies
try:
    from services.core.geocode import Geocoder
    from services.data.get_point_cloud import PointCloudDatasetFinder
    from services.data.get_orthophoto import NAIPFetcher

    EXTERNAL_DEPS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"External API dependencies not available: {e}")
    logger.warning("Address-based processing will not be available")
    EXTERNAL_DEPS_AVAILABLE = False


class PointCloudProcessor:
    """
    Main processor class that orchestrates the modular colorization workflow.
    """

    def __init__(self, output_dir: str = "data", create_diagnostics: bool = True):
        """
        Initialize the processor with all required modules.

        Args:
            output_dir: Output directory for results
            create_diagnostics: Whether to create diagnostic plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize all modules
        self.point_cloud_io = PointCloudIO()
        self.orthophoto_io = OrthophotoIO()
        self.transformer = CoordinateTransformer()
        self.diagnostics = AlignmentDiagnostics(self.output_dir)
        self.data_fetcher = DataFetcher(self.output_dir)
        self.downloader = CorrectedOrthophotoDownloader(self.output_dir)
        self.colorizer = PointCloudColorizer(self.output_dir, create_diagnostics)
        self.reporter = SummaryReporter(self.output_dir)

        logger.info(f"Initialized processor with output directory: {self.output_dir}")

    def process_from_address(self, address: str) -> str:
        """
        Complete workflow: fetch data and colorize point cloud for an address.

        Args:
            address: Street address

        Returns:
            Path to colorized point cloud file
        """
        if not EXTERNAL_DEPS_AVAILABLE:
            raise RuntimeError(
                "Address-based processing requires external API dependencies. "
                "Please use --input_pc and --input_ortho instead."
            )

        logger.info(f"Processing address: {address}")
        start_time = time.time()

        # Initialize external fetchers
        geocoder = Geocoder()
        pc_fetcher = PointCloudDatasetFinder()
        ortho_fetcher = NAIPFetcher()

        try:
            # Geocode address
            lat, lon = geocoder.geocode_address(address)
            logger.info(f"Coordinates: {lat:.6f}, {lon:.6f}")

            # Fetch orthophoto first for optimal dataset selection
            logger.info("Fetching orthophoto first for optimal dataset selection...")
            ortho_path = self.data_fetcher.fetch_orthophoto_data(
                ortho_fetcher, address, lat, lon
            )
            logger.info("Orthophoto fetch completed successfully")

            # Get orthophoto bounds for dataset selection
            ortho_bounds, ortho_crs = self.orthophoto_io.get_orthophoto_bounds(
                ortho_path
            )

            # Fetch point cloud using orthophoto-aware selection
            logger.info("Fetching point cloud with orthophoto-aware selection...")
            downloaded_pc = self.data_fetcher.fetch_point_cloud_data(
                pc_fetcher, lat, lon, ortho_bounds, ortho_crs
            )
            logger.info("Point cloud fetch completed successfully")

            logger.info("Both datasets fetched successfully, starting processing...")

            # Process the data
            result_path = self.process_files(str(downloaded_pc), str(ortho_path))

            # Log processing time
            processing_time = time.time() - start_time
            logger.info(f"Total processing time: {processing_time:.1f} seconds")

            return result_path

        except Exception as e:
            logger.error(f"Processing failed: {e}")
            raise

    def process_files(
        self,
        point_cloud_path: str,
        orthophoto_path: str,
        output_name: Optional[str] = None,
        create_summary: bool = True,
        source_crs: Optional[str] = None,
    ) -> str:
        """
        Process existing point cloud and orthophoto files.

        Args:
            point_cloud_path: Path to point cloud file
            orthophoto_path: Path to orthophoto file
            output_name: Output filename (optional)
            create_summary: Whether to create summary report
            source_crs: Override source CRS for point cloud

        Returns:
            Path to colorized point cloud file
        """
        logger.info("Starting point cloud colorization...")
        start_time = time.time()

        try:
            # Load data using modular I/O
            logger.info("Loading point cloud data...")
            las_data = self.point_cloud_io.load_point_cloud(point_cloud_path)

            logger.info("Loading orthophoto data...")
            ortho_dataset = self.orthophoto_io.load_orthophoto(orthophoto_path)

            # Colorize using the core engine
            logger.info("Starting colorization process...")
            colors, valid_mask = self.colorizer.colorize_point_cloud(
                las_data, ortho_dataset, source_crs
            )

            # Generate output filename
            if output_name is None:
                pc_name = Path(point_cloud_path).stem
                output_name = f"{pc_name}_colorized.laz"

            output_path = self.output_dir / output_name

            # Save result (now trimmed to orthophoto intersection)
            logger.info("Saving colorized point cloud...")
            self.point_cloud_io.save_colorized_point_cloud(
                las_data, colors, valid_mask, str(output_path)
            )

            # Create summary report if requested
            if create_summary:
                logger.info("Generating summary report...")
                self.reporter.create_summary_report(
                    point_cloud_path,
                    orthophoto_path,
                    str(output_path),
                    colors,
                    valid_mask,
                )

            # Log processing time
            processing_time = time.time() - start_time
            logger.info(f"Processing completed in {processing_time:.1f} seconds")

            return str(output_path)

        except Exception as e:
            logger.error(f"Processing failed: {e}")
            raise
        finally:
            # Ensure orthophoto dataset is closed
            if "ortho_dataset" in locals():
                ortho_dataset.close()

    def validate_environment(self) -> bool:
        """
        Validate that all required dependencies are available.

        Returns:
            True if all dependencies are available, False otherwise
        """
        missing_deps = []

        try:
            import laspy
        except ImportError:
            missing_deps.append("laspy")

        try:
            import rasterio
        except ImportError:
            missing_deps.append("rasterio")

        try:
            import numpy
        except ImportError:
            missing_deps.append("numpy")

        try:
            import pyproj
        except ImportError:
            missing_deps.append("pyproj")

        if missing_deps:
            logger.error(f"Missing required dependencies: {', '.join(missing_deps)}")
            logger.error(
                "Please install missing packages with: pip install "
                + " ".join(missing_deps)
            )
            return False

        return True


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Advanced Point Cloud Colorization - Modular Version",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process by address (downloads data automatically)
  python process_point_cloud.py --address "1250 Wildwood Road, Boulder, CO"
  
  # Process existing files
  python process_point_cloud.py --input_pc data/point_cloud.laz --input_ortho data/orthophoto.tif
  
  # Fast processing mode (no diagnostics or summary reports)
  python process_point_cloud.py --address "123 Main St" --fast
  
  # Custom performance options
  python process_point_cloud.py --input_pc data/pc.laz --input_ortho data/ortho.tif --no-diagnostics
  
  # Specify output directory
  python process_point_cloud.py --address "123 Main St" --output_dir results/
        """,
    )

    # Input options
    parser.add_argument(
        "--address", type=str, help="Address to process (automatically downloads data)"
    )

    # File input arguments (both required together if using files)
    parser.add_argument(
        "--input_pc", type=str, help="Path to input point cloud file (LAZ/LAS)"
    )
    parser.add_argument("--input_ortho", type=str, help="Path to input orthophoto file")

    # Output options
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data",
        help="Output directory (default: data)",
    )
    parser.add_argument(
        "--output_name", type=str, help="Output filename (default: auto-generated)"
    )

    # Processing options
    parser.add_argument(
        "--source_crs",
        type=str,
        help="Override source CRS for point cloud (e.g., EPSG:2232)",
    )

    # Performance options
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Enable fast processing mode (disables diagnostics and summary reports)",
    )
    parser.add_argument(
        "--no-diagnostics",
        action="store_true",
        help="Disable diagnostic plot creation for faster processing",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Disable summary report creation for faster processing",
    )

    # Validation options
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate environment and dependencies without processing",
    )

    args = parser.parse_args()

    # Validate environment if requested
    if args.validate:
        processor = PointCloudProcessor()
        is_valid = processor.validate_environment()
        if is_valid:
            logger.info("Environment validation passed - all dependencies available")
            if EXTERNAL_DEPS_AVAILABLE:
                logger.info(
                    "External API dependencies available - address processing supported"
                )
            else:
                logger.info(
                    "External API dependencies not available - only file processing supported"
                )
        return 0 if is_valid else 1

    # Validate input arguments
    if args.address and (args.input_pc or args.input_ortho):
        parser.error("Cannot use --address with --input_pc or --input_ortho")
    elif not args.address and (not args.input_pc or not args.input_ortho):
        parser.error(
            "Must provide either --address OR both --input_pc and --input_ortho"
        )

    # Check address processing availability
    if args.address and not EXTERNAL_DEPS_AVAILABLE:
        parser.error(
            "Address processing requires external API dependencies. "
            "Please use --input_pc and --input_ortho instead, or install required packages."
        )

    try:
        # Determine performance settings
        create_diagnostics = not (args.fast or args.no_diagnostics)
        create_summary = not (args.fast or args.no_summary)

        if args.fast:
            logger.info(
                "Fast processing mode enabled - diagnostics and summary reports disabled"
            )

        # Initialize processor with performance options
        processor = PointCloudProcessor(
            args.output_dir, create_diagnostics=create_diagnostics
        )

        # Validate environment before processing
        if not processor.validate_environment():
            return 1

        if args.address:
            # Process by address
            output_path = processor.process_from_address(args.address)
        else:
            # Process existing files
            output_path = processor.process_files(
                args.input_pc,
                args.input_ortho,
                args.output_name,
                create_summary=create_summary,
                source_crs=args.source_crs,
            )

        logger.info(f"SUCCESS: Colorized point cloud saved to {output_path}")
        return 0

    except Exception as e:
        logger.error(f"FAILED: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
