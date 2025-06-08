"""
Image generation and retrieval endpoints.
"""

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from .shared import OrthophotoRequest

logger = logging.getLogger("photogrammetry-api")

router = APIRouter()


@router.post("/orthophoto")
async def download_orthophoto(request: OrthophotoRequest):
    """Fetch and return an orthophoto for the given address."""
    try:
        from services.data.get_orthophoto import NAIPFetcher
        
        fetcher = NAIPFetcher()
        data_dir = Path(__file__).parent.parent / "data" / "orthophotos"
        data_dir.mkdir(parents=True, exist_ok=True)
        output_path, _ = fetcher.get_orthophoto_for_address(
            request.address, str(data_dir), request.image_size
        )
        file_path = Path(output_path)
        if not file_path.exists() or not file_path.is_file():
            logger.error(f"Orthophoto download failed for {request.address}")
            raise HTTPException(status_code=500, detail="Orthophoto download failed")
        return FileResponse(
            path=str(file_path),
            filename=file_path.name,
            media_type="image/tiff",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error downloading orthophoto for {request.address}: {e}", exc_info=True
        )
        raise HTTPException(status_code=500, detail="Failed to fetch orthophoto")


@router.get("/flood-overhead")
async def flood_overhead(address: str, bbox_m: float = 64.0, resolution: int = 8192):
    """Return a colored PNG visualization of 100-year flood depth for an address.

    This endpoint generates flood depth data by:
    1. Geocoding the provided address to lat/lon coordinates
    2. Creating a bounding box of the specified size around the location
    3. Attempting to fetch real FEMA National Flood Hazard Layer (NFHL) data including:
       - Flood zones (AE/VE) from FEMA shapefiles
       - Base Flood Elevation (BFE) lines from FEMA data
    4. Downloading 1-meter USGS 3DEP DEM tiles for the area
    5. Rasterizing BFE lines onto the DEM grid
    6. Calculating flood depths by subtracting DEM from BFE surface
    7. Clipping results to flood zones (areas outside get nodata)
    8. Applying a perceptually ordered colormap (viridis) to create PNG
    9. Inverting the colors of the PNG

    If FEMA data is unavailable, falls back to synthetic flood modeling.

    Args:
        address: Street address to analyze (e.g., "1250 Wildwood Road, Boulder, CO")
        bbox_m: Bounding box size in meters (default 64m ≈ 1 acre square)
        resolution: Output image resolution in pixels (default 8192x8192 for ultra-high quality)

    Returns:
        PNG image (8192x8192 pixels by default, upscaled with bicubic interpolation) showing flood depths with an inverted viridis colormap:
        - Yellow: Shallow depths
        - Green: Medium depths
        - Dark blue: Deep depths
        - Transparent: Areas outside flood zones
    """
    try:
        # Validate resolution parameter
        if resolution < 512:
            raise HTTPException(status_code=400, detail="Resolution must be at least 512 pixels")
        if resolution > 16384:
            raise HTTPException(status_code=400, detail="Resolution cannot exceed 16384 pixels (memory constraints)")
        
        from services.utils.flood_depth import generate
        from services.visualization.overhead_image import render
        from services.visualization.invert_image import invert_image_colors

        # Generate flood depth GeoTIFF
        tiff = generate(address, bbox_m)

        # Convert to colored PNG visualization
        png = render(tiff, target_size=resolution)

        # Invert the colors of the PNG
        inverted_png = invert_image_colors(png)

        file_path = Path(inverted_png)
        return FileResponse(
            path=str(file_path),
            filename=f"flood_depth_{address.replace(' ', '_').replace(',', '')}.png",
            media_type="image/png",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error generating flood overhead for {address}: {e}", exc_info=True
        )
        raise HTTPException(status_code=500, detail="Failed to generate flood image")
