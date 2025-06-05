#!/usr/bin/env python3
"""
Alignment Diagnostics Module

Creates diagnostic plots and analyses for point cloud and orthophoto alignment.
"""

import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import laspy

logger = logging.getLogger(__name__)


class AlignmentDiagnostics:
    """Creates diagnostic plots and alignment analysis."""

    def __init__(self, output_dir: Path):
        """
        Initialize diagnostics with output directory.

        Args:
            output_dir: Directory for diagnostic outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def create_alignment_diagnostic(
        self,
        las_data: laspy.LasData,
        ortho_dataset: rasterio.DatasetReader,
        transformed_x: np.ndarray,
        transformed_y: np.ndarray,
        output_name: str = "alignment_diagnostic.png",
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
        sample_indices = np.random.choice(
            len(transformed_x), sample_size, replace=False
        )

        sample_x = transformed_x[sample_indices]
        sample_y = transformed_y[sample_indices]

        # Orthophoto bounds
        rect = plt.Rectangle(
            (ortho_bounds.left, ortho_bounds.bottom),
            ortho_bounds.right - ortho_bounds.left,
            ortho_bounds.top - ortho_bounds.bottom,
            linewidth=2,
            edgecolor="red",
            facecolor="none",
            label="Orthophoto Bounds",
        )
        ax1.add_patch(rect)

        # Point cloud
        ax1.scatter(
            sample_x,
            sample_y,
            s=1,
            c="blue",
            alpha=0.6,
            label=f"Point Cloud ({sample_size:,} points)",
        )

        ax1.set_title("Coordinate Alignment Overview")
        ax1.set_xlabel(f"X ({ortho_dataset.crs})")
        ax1.set_ylabel(f"Y ({ortho_dataset.crs})")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Intersection analysis
        # Find points within orthophoto bounds
        within_bounds = (
            (sample_x >= ortho_bounds.left)
            & (sample_x <= ortho_bounds.right)
            & (sample_y >= ortho_bounds.bottom)
            & (sample_y <= ortho_bounds.top)
        )

        points_in_bounds = np.sum(within_bounds)

        ax2.scatter(
            sample_x[~within_bounds],
            sample_y[~within_bounds],
            s=1,
            c="red",
            alpha=0.6,
            label=f"Outside bounds ({np.sum(~within_bounds):,})",
        )
        ax2.scatter(
            sample_x[within_bounds],
            sample_y[within_bounds],
            s=1,
            c="green",
            alpha=0.6,
            label=f"Within bounds ({points_in_bounds:,})",
        )

        ax2.add_patch(
            plt.Rectangle(
                (ortho_bounds.left, ortho_bounds.bottom),
                ortho_bounds.right - ortho_bounds.left,
                ortho_bounds.top - ortho_bounds.bottom,
                linewidth=2,
                edgecolor="black",
                facecolor="none",
            )
        )

        ax2.set_title("Point Distribution Analysis")
        ax2.set_xlabel(f"X ({ortho_dataset.crs})")
        ax2.set_ylabel(f"Y ({ortho_dataset.crs})")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = self.output_dir / output_name
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Diagnostic plot saved: {output_path}")
        logger.info(
            f"Points within orthophoto bounds: {points_in_bounds:,}/{sample_size:,} "
            f"({100*points_in_bounds/sample_size:.1f}%)"
        )

    def generate_alignment_diagnostic(
        self,
        point_cloud_path: str,
        orthophoto_path: str,
        output_name: str = "alignment_diagnostic.png",
    ):
        """
        Generate alignment diagnostic from file paths.

        Args:
            point_cloud_path: Path to point cloud file
            orthophoto_path: Path to orthophoto file
            output_name: Name for output diagnostic file

        Returns:
            str: Path to generated diagnostic file

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

        # Load data
        pc_io = PointCloudIO()
        ortho_io = OrthophotoIO()

        las_data = pc_io.load_point_cloud(point_cloud_path)
        ortho_data = ortho_io.load_orthophoto(orthophoto_path)

        # Use point cloud coordinates directly (simplified for testing)
        x_coords = las_data.x
        y_coords = las_data.y

        # Generate diagnostic
        self.create_alignment_diagnostic(
            las_data, ortho_data, x_coords, y_coords, output_name
        )

        return str(self.output_dir / output_name)
