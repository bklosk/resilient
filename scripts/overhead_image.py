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
    """Create a PNG preview for a flood depth GeoTIFF."""
    src_path = Path(tiff_path)
    if output_dir is None:
        output_dir = src_path.parent
    out_path = Path(output_dir) / f"{src_path.stem}.png"

    with rasterio.open(src_path) as src:
        data = src.read(1)
        nodata = src.nodata

    mask = np.isnan(data) if nodata is None else data == nodata
    valid = np.ma.array(data, mask=mask)
    if valid.count() == 0:
        scaled = np.zeros((*data.shape, 4), dtype=np.uint8)
    else:
        mn = float(valid.min())
        mx = float(valid.max())
        norm = (data - mn) / (mx - mn + 1e-6)
        cmap = cm.get_cmap("viridis")
        rgba = (cmap(norm) * 255).astype(np.uint8)
        rgba[..., 3] = np.where(mask, 0, 255)
        scaled = reshape_as_image(rgba)

    img = Image.fromarray(scaled, mode="RGBA")
    img.save(out_path)
    return str(out_path)
