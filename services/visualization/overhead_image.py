"""Render colored overhead flood depth imagery."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
from rasterio.plot import reshape_as_image
import matplotlib.cm as cm
from PIL import Image
from scipy.ndimage import zoom


def render(tiff_path: str, output_dir: Optional[str] = None, target_size: int = 8192) -> str:
    """Create a high-resolution PNG preview for a flood depth GeoTIFF with perceptually ordered colormap.
    
    Args:
        tiff_path: Path to the input GeoTIFF file
        output_dir: Output directory for the PNG file (defaults to same directory as TIFF)
        target_size: Target pixel size for the output PNG (default 8192x8192 for ultra-high resolution)
    
    Returns:
        str: Path to the generated PNG file
    """
    src_path = Path(tiff_path)
    if output_dir is None:
        output_dir = src_path.parent
    out_path = Path(output_dir) / f"{src_path.stem}.png"

    with rasterio.open(src_path) as src:
        data = src.read(1)
        nodata = src.nodata
        original_height, original_width = data.shape

    print(f"Original TIFF resolution: {original_width}x{original_height} pixels")
    
    # Calculate upscaling factor
    upscale_factor = target_size / max(original_width, original_height)
    
    if upscale_factor > 1.0:
        # Upscale using bicubic interpolation for smooth results
        print(f"Upscaling by factor of {upscale_factor:.2f} to target size {target_size}x{target_size}")
        
        # Use scipy.ndimage.zoom for high-quality bicubic interpolation
        upscaled_data = zoom(data, upscale_factor, order=3, mode='nearest', prefilter=False)
        
        # Handle nodata values - zoom might modify them, so we need to reapply the mask
        if nodata is not None:
            # Create original mask
            original_mask = (data == nodata) | np.isnan(data)
            # Upscale the mask using nearest neighbor to preserve exact boundaries
            upscaled_mask = zoom(original_mask.astype(float), upscale_factor, order=0) > 0.5
            # Apply nodata to upscaled data
            upscaled_data[upscaled_mask] = np.nan
        
        data = upscaled_data
        print(f"Upscaled resolution: {data.shape[1]}x{data.shape[0]} pixels")
    else:
        print(f"No upscaling needed, original resolution {original_width}x{original_height} is already >= {target_size}")

    # Create mask for nodata values
    mask = np.isnan(data) if nodata is None else (data == nodata) | np.isnan(data)
    valid = np.ma.array(data, mask=mask)

    if valid.count() == 0:
        # No valid data - create transparent image
        scaled = np.zeros((*data.shape, 4), dtype=np.uint8)
    else:
        # Get min/max from valid (non-masked) data only
        valid_data = data[~mask]
        mn = float(valid_data.min())
        mx = float(valid_data.max())

        print(f"Depth range: {mn:.3f} to {mx:.3f}m ({len(valid_data)} valid pixels)")

        # Handle edge case where min == max
        if mx - mn < 1e-6:
            norm = np.zeros_like(data)
        else:
            # Normalize to 0-1, ensuring we use the full colormap range
            norm = np.clip((data - mn) / (mx - mn), 0, 1)

        # Apply viridis colormap (perceptually ordered: dark blue -> green -> yellow)
        cmap = cm.get_cmap("viridis")

        # Apply colormap to normalized data
        rgba = np.zeros((*data.shape, 4), dtype=np.uint8)

        # Only apply colormap to valid (non-masked) pixels
        valid_mask = ~mask
        if np.any(valid_mask):
            # Get colors for valid pixels
            valid_colors = cmap(norm[valid_mask])

            # Convert to 0-255 range and assign to RGBA array
            rgba[valid_mask] = (valid_colors * 255).astype(np.uint8)

            # Set alpha: opaque for valid data, transparent for nodata
            rgba[..., 3] = np.where(valid_mask, 255, 0)

        scaled = rgba

    # Create and save PNG image with high quality settings
    img = Image.fromarray(scaled, mode="RGBA")
    # Save with maximum quality and no compression for best results
    img.save(out_path, optimize=False, compress_level=0)
    
    # Print final statistics
    print(f"Generated PNG: {out_path}")
    print(f"Final PNG resolution: {img.size[0]}x{img.size[1]} pixels")
    print(f"File size: {Path(out_path).stat().st_size} bytes")
    
    return str(out_path)


class OverheadImageGenerator:
    """Generate overhead images from point clouds."""

    def __init__(self):
        """Initialize the overhead image generator."""
        pass

    def generate_overhead_view(
        self, point_cloud_path: str, output_dir: str, colormap: str = "viridis", target_resolution: int = 4096
    ) -> str:
        """
        Generate overhead view image from point cloud.

        Args:
            point_cloud_path: Path to point cloud file
            output_dir: Directory for output image
            colormap: Matplotlib colormap name
            target_resolution: Target resolution for the output image in pixels (default 4096)

        Returns:
            str: Path to generated image file

        Raises:
            FileNotFoundError: If point cloud file doesn't exist
        """
        from services.processing.point_cloud_io import PointCloudIO
        from pathlib import Path
        import matplotlib.pyplot as plt
        import numpy as np

        # Validate input file exists
        if not Path(point_cloud_path).exists():
            raise FileNotFoundError(f"Point cloud file not found: {point_cloud_path}")

        # Load point cloud data
        pc_io = PointCloudIO()
        las_data = pc_io.load_point_cloud(point_cloud_path)

        # Extract coordinates
        x = las_data.x
        y = las_data.y
        z = las_data.z

        # Create overhead view plot with adaptive figure size and high DPI for better resolution
        # Calculate figure size based on target resolution
        fig_size = max(12, target_resolution / 300)  # Scale figure size with target resolution
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
        
        # Adjust point size based on density and target resolution
        point_density = self._calculate_point_density(point_cloud_path)
        # Smaller points for higher density and higher resolution
        point_size = max(0.001, 0.5 / np.sqrt(point_density * target_resolution / 1000))
        
        scatter = ax.scatter(x, y, c=z, s=point_size, alpha=0.8, cmap=colormap, rasterized=True)
        ax.set_title("Overhead View", fontsize=16)
        ax.set_xlabel("X", fontsize=14)
        ax.set_ylabel("Y", fontsize=14)
        ax.set_aspect("equal")
        plt.colorbar(scatter, ax=ax, label="Height (Z)")

        # Save image with very high DPI for ultra-high resolution
        output_path = Path(output_dir) / "overhead_view.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Calculate DPI to achieve target resolution
        target_dpi = target_resolution / fig_size
        plt.savefig(output_path, dpi=target_dpi, bbox_inches="tight", facecolor='white')
        plt.close()

        return str(output_path)

    def _calculate_point_density(self, point_cloud_path: str) -> float:
        """
        Calculate point density of the point cloud.

        Args:
            point_cloud_path: Path to point cloud file

        Returns:
            float: Point density (points per square meter)
        """
        from services.processing.point_cloud_io import PointCloudIO

        pc_io = PointCloudIO()
        las_data = pc_io.load_point_cloud(point_cloud_path)

        # Calculate bounding box area
        x_range = las_data.x.max() - las_data.x.min()
        y_range = las_data.y.max() - las_data.y.min()
        area = x_range * y_range

        # Calculate density
        if area > 0:
            density = len(las_data.x) / area
        else:
            density = 0.0

        return density
