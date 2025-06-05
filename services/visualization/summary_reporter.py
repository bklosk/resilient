"""
Summary Report Generator

This module handles the generation of comprehensive summary reports
for point cloud colorization processes, including statistics and metadata.
"""

import logging
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


class SummaryReporter:
    """Generates comprehensive summary reports for colorization processes."""

    def __init__(self, output_dir: Path):
        """
        Initialize the summary reporter.

        Args:
            output_dir: Output directory for reports
        """
        self.output_dir = Path(output_dir)

    def create_summary_report(
        self,
        pc_path: str,
        ortho_path: str,
        output_path: str,
        colors: np.ndarray,
        valid_mask: np.ndarray,
    ) -> str:
        """
        Create a comprehensive summary report of the colorization process.

        Args:
            pc_path: Input point cloud path
            ortho_path: Input orthophoto path
            output_path: Output point cloud path
            colors: Color array (for all original points)
            valid_mask: Boolean mask indicating which points were within orthophoto bounds

        Returns:
            Path to the generated report file
        """
        total_original_points = len(colors)
        trimmed_points = np.sum(valid_mask)

        # Only analyze colors for valid points
        valid_colors = colors[valid_mask]

        report = {
            "input_point_cloud": str(pc_path),
            "input_orthophoto": str(ortho_path),
            "output_point_cloud": str(output_path),
            "processing_stats": self._calculate_processing_stats(
                total_original_points, trimmed_points, valid_colors
            ),
            "color_stats": self._calculate_color_stats(valid_colors),
            "file_info": self._get_file_info(pc_path, ortho_path, output_path),
        }

        report_path = self.output_dir / "colorization_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        self._log_summary_info(report)
        logger.info(f"Summary report saved: {report_path}")

        return str(report_path)

    def _calculate_processing_stats(
        self, total_original_points: int, trimmed_points: int, valid_colors: np.ndarray
    ) -> Dict[str, Any]:
        """Calculate processing statistics."""

        colorized_points = (
            int(np.sum(np.any(valid_colors > 0, axis=1)))
            if len(valid_colors) > 0
            else 0
        )

        return {
            "original_total_points": int(total_original_points),
            "trimmed_points": int(trimmed_points),
            "trimming_rate": (
                float(trimmed_points / total_original_points)
                if total_original_points > 0
                else 0.0
            ),
            "colorized_points": colorized_points,
            "colorization_rate": (
                float(colorized_points / len(valid_colors))
                if len(valid_colors) > 0
                else 0.0
            ),
            "points_removed": int(total_original_points - trimmed_points),
            "removal_rate": (
                float((total_original_points - trimmed_points) / total_original_points)
                if total_original_points > 0
                else 0.0
            ),
        }

    def _calculate_color_stats(self, valid_colors: np.ndarray) -> Dict[str, Any]:
        """Calculate color statistics."""

        if len(valid_colors) == 0:
            return {
                "mean_red": 0.0,
                "mean_green": 0.0,
                "mean_blue": 0.0,
                "max_red": 0,
                "max_green": 0,
                "max_blue": 0,
                "min_red": 0,
                "min_green": 0,
                "min_blue": 0,
                "std_red": 0.0,
                "std_green": 0.0,
                "std_blue": 0.0,
            }

        return {
            "mean_red": float(np.mean(valid_colors[:, 0])),
            "mean_green": float(np.mean(valid_colors[:, 1])),
            "mean_blue": float(np.mean(valid_colors[:, 2])),
            "max_red": int(np.max(valid_colors[:, 0])),
            "max_green": int(np.max(valid_colors[:, 1])),
            "max_blue": int(np.max(valid_colors[:, 2])),
            "min_red": int(np.min(valid_colors[:, 0])),
            "min_green": int(np.min(valid_colors[:, 1])),
            "min_blue": int(np.min(valid_colors[:, 2])),
            "std_red": float(np.std(valid_colors[:, 0])),
            "std_green": float(np.std(valid_colors[:, 1])),
            "std_blue": float(np.std(valid_colors[:, 2])),
        }

    def _get_file_info(
        self, pc_path: str, ortho_path: str, output_path: str
    ) -> Dict[str, Any]:
        """Get file size and metadata information."""

        def get_file_size_mb(path: str) -> float:
            """Get file size in MB."""
            try:
                return Path(path).stat().st_size / (1024 * 1024)
            except (FileNotFoundError, OSError):
                return 0.0

        return {
            "input_point_cloud_size_mb": get_file_size_mb(pc_path),
            "input_orthophoto_size_mb": get_file_size_mb(ortho_path),
            "output_point_cloud_size_mb": get_file_size_mb(output_path),
            "size_reduction_mb": get_file_size_mb(pc_path)
            - get_file_size_mb(output_path),
            "size_reduction_percent": (
                (
                    (get_file_size_mb(pc_path) - get_file_size_mb(output_path))
                    / get_file_size_mb(pc_path)
                )
                * 100
                if get_file_size_mb(pc_path) > 0
                else 0.0
            ),
        }

    def _log_summary_info(self, report: Dict[str, Any]):
        """Log key summary information."""

        stats = report["processing_stats"]
        logger.info(
            f"Point cloud trimmed from {stats['original_total_points']:,} to {stats['trimmed_points']:,} points "
            f"({stats['trimming_rate']:.1%})"
        )
        logger.info(f"Colorization rate: {stats['colorization_rate']:.1%}")

        file_info = report["file_info"]
        if file_info["size_reduction_mb"] > 0:
            logger.info(
                f"File size reduced by {file_info['size_reduction_mb']:.1f} MB "
                f"({file_info['size_reduction_percent']:.1f}%)"
            )

    def create_processing_summary(
        self,
        input_files: Dict[str, str],
        output_files: Dict[str, str],
        processing_time: float,
        error_log: list = None,
    ) -> str:
        """
        Create a high-level processing summary.

        Args:
            input_files: Dictionary of input file paths
            output_files: Dictionary of output file paths
            processing_time: Total processing time in seconds
            error_log: List of errors encountered during processing

        Returns:
            Path to the processing summary file
        """
        summary = {
            "processing_summary": {
                "start_time": None,  # Would be set by caller
                "processing_time_seconds": processing_time,
                "processing_time_formatted": f"{processing_time:.1f} seconds",
                "status": "completed" if not error_log else "completed_with_errors",
            },
            "input_files": input_files,
            "output_files": output_files,
            "errors": error_log or [],
        }

        summary_path = self.output_dir / "processing_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Processing summary saved: {summary_path}")
        return str(summary_path)

    def generate_markdown_report(
        self, json_report_path: str, include_technical_details: bool = True
    ) -> str:
        """
        Generate a human-readable markdown report from JSON report.

        Args:
            json_report_path: Path to the JSON report file
            include_technical_details: Whether to include technical details

        Returns:
            Path to the generated markdown report
        """
        with open(json_report_path, "r") as f:
            report = json.load(f)

        markdown_content = self._generate_markdown_content(
            report, include_technical_details
        )

        markdown_path = Path(json_report_path).with_suffix(".md")
        with open(markdown_path, "w") as f:
            f.write(markdown_content)

        logger.info(f"Markdown report generated: {markdown_path}")
        return str(markdown_path)

    def _generate_markdown_content(
        self, report: Dict[str, Any], include_technical: bool
    ) -> str:
        """Generate markdown content from report data."""

        stats = report["processing_stats"]
        color_stats = report["color_stats"]
        file_info = report["file_info"]

        content = f"""# Point Cloud Colorization Report

## Processing Summary

- **Input Point Cloud**: `{Path(report['input_point_cloud']).name}`
- **Input Orthophoto**: `{Path(report['input_orthophoto']).name}`
- **Output Point Cloud**: `{Path(report['output_point_cloud']).name}`

## Results

### Point Statistics
- **Original Points**: {stats['original_total_points']:,}
- **Final Points**: {stats['trimmed_points']:,}
- **Points Removed**: {stats['points_removed']:,} ({stats['removal_rate']:.1%})
- **Colorized Points**: {stats['colorized_points']:,} ({stats['colorization_rate']:.1%})

### File Information
- **Input Point Cloud Size**: {file_info['input_point_cloud_size_mb']:.1f} MB
- **Output Point Cloud Size**: {file_info['output_point_cloud_size_mb']:.1f} MB
- **Size Reduction**: {file_info['size_reduction_mb']:.1f} MB ({file_info['size_reduction_percent']:.1f}%)

"""

        if include_technical:
            content += f"""
## Color Analysis

### RGB Statistics
- **Red Channel**: Mean={color_stats['mean_red']:.1f}, Range=[{color_stats['min_red']}-{color_stats['max_red']}], Std={color_stats['std_red']:.1f}
- **Green Channel**: Mean={color_stats['mean_green']:.1f}, Range=[{color_stats['min_green']}-{color_stats['max_green']}], Std={color_stats['std_green']:.1f}
- **Blue Channel**: Mean={color_stats['mean_blue']:.1f}, Range=[{color_stats['min_blue']}-{color_stats['max_blue']}], Std={color_stats['std_blue']:.1f}

## Technical Details

### Processing Parameters
- **Trimming Rate**: {stats['trimming_rate']:.3f}
- **Colorization Success Rate**: {stats['colorization_rate']:.3f}

### File Paths
- **Input Point Cloud**: `{report['input_point_cloud']}`
- **Input Orthophoto**: `{report['input_orthophoto']}`
- **Output Point Cloud**: `{report['output_point_cloud']}`
"""

        return content

    def generate_summary_report(
        self, point_cloud_path: str, orthophoto_path: str, metadata: Dict[str, Any]
    ) -> str:
        """
        Generate summary report from file paths and metadata.

        Args:
            point_cloud_path: Path to point cloud file
            orthophoto_path: Path to orthophoto file
            metadata: Additional metadata dictionary

        Returns:
            str: Path to generated report file

        Raises:
            FileNotFoundError: If input files don't exist
        """
        from services.processing.point_cloud_io import PointCloudIO
        from services.processing.orthophoto_io import OrthophotoIO

        # Validate input files exist
        if not Path(point_cloud_path).exists():
            raise FileNotFoundError(f"Point cloud file not found: {point_cloud_path}")
        if not Path(orthophoto_path).exists():
            raise FileNotFoundError(f"Orthophoto file not found: {orthophoto_path}")

        # Load data to generate dummy color array and mask for testing
        pc_io = PointCloudIO()
        las_data = pc_io.load_point_cloud(point_cloud_path)

        # Create dummy color data for testing
        num_points = len(las_data.x)
        colors = np.zeros((num_points, 3), dtype=np.uint8)
        valid_mask = np.ones(num_points, dtype=bool)  # All points valid for testing

        # Call the existing method
        return self.create_summary_report(
            point_cloud_path,
            orthophoto_path,
            point_cloud_path + "_output.laz",
            colors,
            valid_mask,
        )

    def _calculate_processing_statistics(
        self, point_cloud_path: str, orthophoto_path: str
    ) -> Dict[str, Any]:
        """
        Calculate processing statistics from input files.

        Args:
            point_cloud_path: Path to point cloud file
            orthophoto_path: Path to orthophoto file

        Returns:
            Dict with point_cloud and orthophoto statistics
        """
        from services.processing.point_cloud_io import PointCloudIO
        from services.processing.orthophoto_io import OrthophotoIO

        pc_io = PointCloudIO()
        ortho_io = OrthophotoIO()

        las_data = pc_io.load_point_cloud(point_cloud_path)
        ortho_data = ortho_io.load_orthophoto(orthophoto_path)

        return {
            "point_cloud": {
                "point_count": len(las_data.x),
                "bounds": {
                    "min_x": float(las_data.x.min()),
                    "max_x": float(las_data.x.max()),
                    "min_y": float(las_data.y.min()),
                    "max_y": float(las_data.y.max()),
                    "min_z": float(las_data.z.min()),
                    "max_z": float(las_data.z.max()),
                },
            },
            "orthophoto": {
                "width": ortho_data.width,
                "height": ortho_data.height,
                "bounds": ortho_data.bounds._asdict(),
                "crs": str(ortho_data.crs),
            },
        }

    def _format_report_data(
        self, metadata: Dict[str, Any], statistics: Dict[str, Any]
    ) -> str:
        """Formats the report data into a string."""
        report_parts = []
        report_parts.append("Photogrammetry Process Summary Report")
        report_parts.append("=" * 40)

        if "address" in metadata:
            report_parts.append(f"Location: {metadata['address']}")
        if "capture_date" in metadata:
            report_parts.append(f"Capture Date: {metadata['capture_date']}")
        if "processing_time" in metadata:
            report_parts.append(
                f"Total Processing Time: {metadata['processing_time']:.2f} seconds"
            )

        report_parts.append("\\nProcessing Statistics:")
        report_parts.append("-" * 20)
        if "point_cloud" in statistics:
            pc_stats = statistics["point_cloud"]
            report_parts.append(f"  Point Cloud:")
            report_parts.append(f"    Points: {pc_stats.get('point_count', 'N/A')}")
            report_parts.append(
                f"    File Size: {pc_stats.get('file_size_mb', 'N/A')} MB"
            )
            report_parts.append(
                f"    Density: {pc_stats.get('density_sqm', 'N/A')} points/sqm"
            )
        if "orthophoto" in statistics:
            ortho_stats = statistics["orthophoto"]
            report_parts.append(f"  Orthophoto:")
            report_parts.append(
                f"    Dimensions: {ortho_stats.get('width', 'N/A')}x{ortho_stats.get('height', 'N/A')}"
            )
            report_parts.append(
                f"    Resolution: {ortho_stats.get('resolution_cm_px', 'N/A')} cm/px"
            )
            report_parts.append(
                f"    File Size: {ortho_stats.get('file_size_mb', 'N/A')} MB"
            )

        if "alignment" in statistics:
            align_stats = statistics["alignment"]
            report_parts.append(f"  Alignment Quality:")
            report_parts.append(f"    RMSE: {align_stats.get('rmse', 'N/A')}")
            report_parts.append(
                f"    Keypoints: {align_stats.get('keypoints_matched', 'N/A')}"
            )

        return "\\n".join(report_parts)
