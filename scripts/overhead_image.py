"""Render colored overhead flood depth imagery."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
from rasterio.plot import reshape_as_image
import matplotlib.cm as cm
from PIL import Image


def render(tiff_path: str, output_dir: Optional[str] = None) -> str:
    """Create a PNG preview for a flood depth GeoTIFF with perceptually ordered colormap."""
    src_path = Path(tiff_path)
    if output_dir is None:
        output_dir = src_path.parent
    out_path = Path(output_dir) / f"{src_path.stem}.png"

    with rasterio.open(src_path) as src:
        data = src.read(1)
        nodata = src.nodata

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

    # Create and save PNG image
    img = Image.fromarray(scaled, mode="RGBA")
    img.save(out_path, optimize=True)
    return str(out_path)
