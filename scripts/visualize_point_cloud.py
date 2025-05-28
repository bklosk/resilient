#!/usr/bin/env python3
"""
Simple Point Cloud Visualization Script

Creates images of colorized point clouds for testing and validation.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging

try:
    import laspy
except ImportError:
    print("ERROR: laspy not found. Install with: pip install laspy lazrs")
    exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_and_visualize_point_cloud(file_path: str, output_dir: str = "visualizations"):
    """
    Load and create visualizations of a colorized point cloud.

    Args:
        file_path: Path to LAZ/LAS file
        output_dir: Directory for output images
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load point cloud
    logger.info(f"Loading point cloud: {file_path}")
    las_data = laspy.read(file_path)

    # Extract coordinates as numpy arrays
    x = np.array(las_data.x)
    y = np.array(las_data.y)
    z = np.array(las_data.z)

    logger.info(f"Loaded {len(x):,} points")
    logger.info(f"Bounds - X: {x.min():.2f} to {x.max():.2f}")
    logger.info(f"Bounds - Y: {y.min():.2f} to {y.max():.2f}")
    logger.info(f"Bounds - Z: {z.min():.2f} to {z.max():.2f}")

    # Check for colors
    has_colors = (
        hasattr(las_data, "red")
        and hasattr(las_data, "green")
        and hasattr(las_data, "blue")
    )

    if has_colors:
        # Extract and normalize colors
        red = np.array(las_data.red).astype(np.float32) / 65535.0
        green = np.array(las_data.green).astype(np.float32) / 65535.0
        blue = np.array(las_data.blue).astype(np.float32) / 65535.0
        colors = np.column_stack((red, green, blue))

        # Check how many points are actually colorized
        colorized_mask = np.any(colors > 0, axis=1)
        num_colorized = np.sum(colorized_mask)
        logger.info(
            f"Found RGB colors: {num_colorized:,}/{len(colors):,} points colorized ({100*num_colorized/len(colors):.1f}%)"
        )
    else:
        logger.warning("No color information found")
        colors = None

    # Sample points for visualization (to avoid overwhelming the plot)
    max_points = 50000
    if len(x) > max_points:
        indices = np.random.choice(len(x), max_points, replace=False)
        x_sample = x[indices]
        y_sample = y[indices]
        z_sample = z[indices]
        colors_sample = colors[indices] if colors is not None else None
        logger.info(f"Sampling {max_points:,} points for visualization")
    else:
        x_sample = x
        y_sample = y
        z_sample = z
        colors_sample = colors

    # Create visualizations
    fig = plt.figure(figsize=(20, 12))

    # 1. Top view with colors
    ax1 = plt.subplot(2, 3, 1)
    if colors_sample is not None:
        scatter = ax1.scatter(x_sample, y_sample, c=colors_sample, s=0.5, alpha=0.7)
        ax1.set_title("Top View - RGB Colors")
    else:
        scatter = ax1.scatter(
            x_sample, y_sample, c=z_sample, s=0.5, alpha=0.7, cmap="viridis"
        )
        ax1.set_title("Top View - Height Colored")
        plt.colorbar(scatter, ax=ax1, label="Height")

    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.3)

    # 2. Side view (X-Z)
    ax2 = plt.subplot(2, 3, 2)
    if colors_sample is not None:
        ax2.scatter(x_sample, z_sample, c=colors_sample, s=0.5, alpha=0.7)
        ax2.set_title("Side View (X-Z) - RGB Colors")
    else:
        scatter = ax2.scatter(
            x_sample, z_sample, c=y_sample, s=0.5, alpha=0.7, cmap="viridis"
        )
        ax2.set_title("Side View (X-Z) - Y Colored")
        plt.colorbar(scatter, ax=ax2, label="Y")

    ax2.set_xlabel("X")
    ax2.set_ylabel("Z (Height)")
    ax2.grid(True, alpha=0.3)

    # 3. Front view (Y-Z)
    ax3 = plt.subplot(2, 3, 3)
    if colors_sample is not None:
        ax3.scatter(y_sample, z_sample, c=colors_sample, s=0.5, alpha=0.7)
        ax3.set_title("Front View (Y-Z) - RGB Colors")
    else:
        scatter = ax3.scatter(
            y_sample, z_sample, c=x_sample, s=0.5, alpha=0.7, cmap="viridis"
        )
        ax3.set_title("Front View (Y-Z) - X Colored")
        plt.colorbar(scatter, ax=ax3, label="X")

    ax3.set_xlabel("Y")
    ax3.set_ylabel("Z (Height)")
    ax3.grid(True, alpha=0.3)

    # 4. Height distribution
    ax4 = plt.subplot(2, 3, 4)
    ax4.hist(z, bins=50, alpha=0.7, color="steelblue", edgecolor="black")
    ax4.set_title("Height Distribution")
    ax4.set_xlabel("Height (Z)")
    ax4.set_ylabel("Number of Points")
    ax4.grid(True, alpha=0.3)

    # 5. Color analysis (if available)
    ax5 = plt.subplot(2, 3, 5)
    if colors is not None:
        ax5.hist(
            colors[:, 0], bins=50, alpha=0.7, color="red", label="Red", density=True
        )
        ax5.hist(
            colors[:, 1], bins=50, alpha=0.7, color="green", label="Green", density=True
        )
        ax5.hist(
            colors[:, 2], bins=50, alpha=0.7, color="blue", label="Blue", density=True
        )
        ax5.set_title("RGB Color Distribution")
        ax5.set_xlabel("Color Value (0-1)")
        ax5.set_ylabel("Density")
        ax5.legend()
    else:
        ax5.text(
            0.5,
            0.5,
            "No Color Information\nAvailable",
            ha="center",
            va="center",
            transform=ax5.transAxes,
            fontsize=14,
        )
        ax5.set_title("Color Analysis")
    ax5.grid(True, alpha=0.3)

    # 6. Summary statistics
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis("off")

    stats_text = f"""Point Cloud Statistics

Total Points: {len(x):,}
Visualized: {len(x_sample):,}

Coordinate Ranges:
X: {x.min():.1f} to {x.max():.1f}
Y: {y.min():.1f} to {y.max():.1f}
Z: {z.min():.1f} to {z.max():.1f}

Height Stats:
Mean: {z.mean():.1f}
Std: {z.std():.1f}
Range: {z.max() - z.min():.1f}
"""

    if colors is not None:
        stats_text += f"\nColor Information:"
        stats_text += f"\nColorized Points: {num_colorized:,}"
        stats_text += f"\nColorization Rate: {100*num_colorized/len(colors):.1f}%"
        if num_colorized > 0:
            stats_text += f"\nMean RGB: ({colors[colorized_mask, 0].mean():.3f}, "
            stats_text += f"{colors[colorized_mask, 1].mean():.3f}, "
            stats_text += f"{colors[colorized_mask, 2].mean():.3f})"
    else:
        stats_text += f"\nColors: Not Available"

    ax6.text(
        0.1,
        0.9,
        stats_text,
        transform=ax6.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8),
    )

    plt.tight_layout()

    # Save the visualization
    output_path = output_dir / f"point_cloud_visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Visualization saved: {output_path}")

    # Create a focused colorized view if colors are available
    if colors is not None and num_colorized > 0:
        create_colorized_focus_view(x, y, z, colors, colorized_mask, output_dir)


def create_colorized_focus_view(x, y, z, colors, colorized_mask, output_dir):
    """Create a focused view of only the colorized points."""

    # Extract only colorized points
    x_colored = x[colorized_mask]
    y_colored = y[colorized_mask]
    z_colored = z[colorized_mask]
    colors_colored = colors[colorized_mask]

    logger.info(f"Creating focused view of {len(x_colored):,} colorized points")

    # Sample if too many points
    max_points = 30000
    if len(x_colored) > max_points:
        indices = np.random.choice(len(x_colored), max_points, replace=False)
        x_colored = x_colored[indices]
        y_colored = y_colored[indices]
        z_colored = z_colored[indices]
        colors_colored = colors_colored[indices]

    # Create focused visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Top view - large points for better color visibility
    ax1.scatter(x_colored, y_colored, c=colors_colored, s=2, alpha=0.8)
    ax1.set_title("Colorized Points - Top View")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.3)

    # Side view
    ax2.scatter(x_colored, z_colored, c=colors_colored, s=2, alpha=0.8)
    ax2.set_title("Colorized Points - Side View")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Z (Height)")
    ax2.grid(True, alpha=0.3)

    # 3D-like view (isometric projection)
    x_iso = x_colored - 0.5 * y_colored
    y_iso = z_colored + 0.25 * y_colored
    ax3.scatter(x_iso, y_iso, c=colors_colored, s=1, alpha=0.8)
    ax3.set_title("Colorized Points - Isometric View")
    ax3.set_xlabel("X - 0.5*Y")
    ax3.set_ylabel("Z + 0.25*Y")
    ax3.grid(True, alpha=0.3)

    # Color intensity analysis
    brightness = np.mean(colors_colored, axis=1)
    ax4.scatter(x_colored, y_colored, c=brightness, s=1, alpha=0.8, cmap="viridis")
    ax4.set_title("Brightness Distribution")
    ax4.set_xlabel("X")
    ax4.set_ylabel("Y")
    ax4.set_aspect("equal")
    cbar = plt.colorbar(ax4.collections[0], ax=ax4)
    cbar.set_label("Brightness")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = output_dir / "colorized_points_focus.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Focused colorized view saved: {output_path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python visualize_point_cloud.py <point_cloud_file.laz>")
        print("Example: python visualize_point_cloud.py data/colorized_point_cloud.laz")
        sys.exit(1)

    point_cloud_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "visualizations"

    try:
        load_and_visualize_point_cloud(point_cloud_file, output_dir)
        print(f"Visualization complete! Check the {output_dir} directory for images.")
    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        sys.exit(1)
