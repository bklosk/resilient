#!/usr/bin/env python3
"""
FastAPI Application for Photogrammetry Point Cloud Processing

This API provides endpoints to generate colorized point clouds from addresses.
It orchestrates the existing helper scripts to download LiDAR data and orthophotos,
then colorizes the point cloud data.

Endpoints:
- GET /health - Health check endpoint
- POST /process - Generate colorized point cloud from an address
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any
import tempfile
import shutil

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# Add the scripts directory to the Python path so we can import the modules
scripts_dir = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(scripts_dir))

# Import our helper modules
try:
    from process_point_cloud import PointCloudColorizer
    from geocode import Geocoder
    from get_point_cloud import PointCloudFetcher
    from get_orthophoto import NAIPFetcher
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print(
        "Make sure you're running from the correct directory and all dependencies are installed"
    )
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Photogrammetry Point Cloud API",
    description="API for generating colorized point clouds from address inputs",
    version="1.0.0",
)


class ProcessRequest(BaseModel):
    """Request model for point cloud processing."""

    address: str = Field(
        ...,
        description="Street address to process",
        example="1250 Wildwood Road, Boulder, CO",
    )
    buffer_km: float = Field(
        default=1.0,
        description="Buffer distance in kilometers for LiDAR search",
        ge=0.1,
        le=5.0,
    )


class ProcessResponse(BaseModel):
    """Response model for point cloud processing."""

    success: bool
    message: str
    output_file: str = None
    metadata: Dict[str, Any] = None


@app.get("/")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "photogrammetry-api", "version": "1.0.0"}


@app.post("/process", response_model=ProcessResponse)
async def process_point_cloud(
    request: ProcessRequest, background_tasks: BackgroundTasks
):
    """
    Generate a colorized point cloud from an address.

    This endpoint:
    1. Geocodes the provided address
    2. Searches for LiDAR point cloud data
    3. Downloads orthophoto imagery
    4. Colorizes the point cloud using the orthophoto
    5. Returns the path to the colorized point cloud file

    Args:
        request: ProcessRequest containing the address and optional parameters
        background_tasks: FastAPI background tasks for cleanup

    Returns:
        ProcessResponse with success status and file information
    """
    logger.info(f"Processing request for address: {request.address}")

    # Create a temporary working directory for this request
    temp_dir = Path(tempfile.mkdtemp(prefix="photogrammetry_"))

    try:
        # Initialize the colorizer with the temporary directory
        colorizer = PointCloudColorizer(output_dir=str(temp_dir))

        # Validate that the address can be geocoded first
        try:
            geocoder = Geocoder()
            lat, lon = geocoder.geocode_address(request.address)
            logger.info(f"Address geocoded to: {lat:.6f}, {lon:.6f}")
        except Exception as e:
            logger.error(f"Failed to geocode address: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Failed to geocode address '{request.address}': {str(e)}",
            )

        # Check for LiDAR data availability before proceeding
        try:
            pc_fetcher = PointCloudFetcher()
            bbox = pc_fetcher.generate_bounding_box(lat, lon, request.buffer_km)
            products = pc_fetcher.search_lidar_products(bbox)

            if not products:
                raise HTTPException(
                    status_code=404,
                    detail=f"No LiDAR data found for location: {request.address}",
                )

            laz_products = pc_fetcher.filter_laz_products(products)
            if not laz_products:
                raise HTTPException(
                    status_code=404,
                    detail=f"No LAZ format LiDAR data found for location: {request.address}",
                )

            logger.info(f"Found {len(laz_products)} LAZ products for processing")

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error checking LiDAR data availability: {e}")
            raise HTTPException(
                status_code=500, detail=f"Error checking data availability: {str(e)}"
            )

        # Process the point cloud
        try:
            output_path = colorizer.process_from_address(request.address)

            # Move the output file to a permanent location
            output_dir = Path(__file__).parent.parent / "data" / "outputs"
            output_dir.mkdir(parents=True, exist_ok=True)

            output_filename = f"colorized_{hash(request.address) % 10000:04d}.laz"
            final_output_path = output_dir / output_filename

            shutil.copy2(output_path, final_output_path)

            # Prepare metadata
            metadata = {
                "address": request.address,
                "coordinates": {"latitude": lat, "longitude": lon},
                "buffer_km": request.buffer_km,
                "output_file_size_mb": final_output_path.stat().st_size / (1024 * 1024),
                "lidar_products_found": len(laz_products),
            }

            # Schedule cleanup of temporary directory
            background_tasks.add_task(cleanup_temp_dir, temp_dir)

            logger.info(f"Successfully processed point cloud: {final_output_path}")

            return ProcessResponse(
                success=True,
                message=f"Successfully generated colorized point cloud for {request.address}",
                output_file=str(final_output_path),
                metadata=metadata,
            )

        except Exception as e:
            logger.error(f"Error processing point cloud: {e}")
            raise HTTPException(
                status_code=500, detail=f"Error processing point cloud: {str(e)}"
            )

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        background_tasks.add_task(cleanup_temp_dir, temp_dir)
        raise
    except Exception as e:
        # Handle unexpected errors
        logger.error(f"Unexpected error: {e}")
        background_tasks.add_task(cleanup_temp_dir, temp_dir)
        raise HTTPException(
            status_code=500, detail=f"Unexpected error occurred: {str(e)}"
        )


@app.get("/download/{filename}")
async def download_file(filename: str):
    """
    Download a processed point cloud file.

    Args:
        filename: Name of the file to download

    Returns:
        FileResponse with the requested file
    """
    output_dir = Path(__file__).parent.parent / "data" / "outputs"
    file_path = output_dir / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        path=str(file_path), filename=filename, media_type="application/octet-stream"
    )


def cleanup_temp_dir(temp_dir: Path):
    """Clean up temporary directory."""
    try:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temporary directory: {temp_dir}")
    except Exception as e:
        logger.warning(f"Failed to clean up temporary directory {temp_dir}: {e}")


@app.on_event("startup")
async def startup_event():
    """Startup event handler."""
    logger.info("Photogrammetry API starting up...")

    # Create necessary directories
    data_dir = Path(__file__).parent.parent / "data"
    output_dir = data_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Photogrammetry API startup complete")


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler."""
    logger.info("Photogrammetry API shutting down...")


if __name__ == "__main__":
    import uvicorn

    # Run the server
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
