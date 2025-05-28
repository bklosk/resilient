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
from typing import Dict, Any, Optional
import tempfile
import shutil
import uuid
import asyncio
from datetime import datetime
from enum import Enum

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


class JobStatus(str, Enum):
    """Job status enumeration."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Job(BaseModel):
    """Job tracking model."""

    job_id: str
    address: str
    status: JobStatus
    created_at: datetime
    completed_at: Optional[datetime] = None
    output_file: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = {}


# In-memory job storage (in production, use Redis or a database)
jobs: Dict[str, Job] = {}


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
    job_id: str
    status: JobStatus
    metadata: Dict[str, Any] = {}


class JobStatusResponse(BaseModel):
    """Response model for job status queries."""

    job_id: str
    status: JobStatus
    address: str
    created_at: datetime
    completed_at: Optional[datetime] = None
    output_file: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = {}


@app.get("/")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "photogrammetry-api", "version": "1.0.0"}


async def process_point_cloud_background(job_id: str, address: str, buffer_km: float):
    """
    Background task to process point cloud.

    Args:
        job_id: Unique job identifier
        address: Address to process
        buffer_km: Buffer distance in kilometers
    """
    try:
        # Update job status to processing
        if job_id in jobs:
            jobs[job_id].status = JobStatus.PROCESSING

        logger.info(f"Starting background processing for job {job_id}")

        # Create a temporary working directory for this request
        temp_dir = Path(tempfile.mkdtemp(prefix=f"photogrammetry_{job_id}_"))

        # Initialize the colorizer with the temporary directory
        colorizer = PointCloudColorizer(output_dir=str(temp_dir))

        # Process the point cloud (this is the time-consuming part)
        output_path = colorizer.process_from_address(address)

        # Move the output file to a permanent location
        output_dir = Path(__file__).parent.parent / "data" / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)

        output_filename = f"colorized_{job_id}.laz"
        final_output_path = output_dir / output_filename

        shutil.copy2(output_path, final_output_path)

        # Update job with completion info
        if job_id in jobs:
            jobs[job_id].status = JobStatus.COMPLETED
            jobs[job_id].completed_at = datetime.now()
            jobs[job_id].output_file = str(final_output_path)
            jobs[job_id].metadata.update(
                {
                    "output_file_size_mb": final_output_path.stat().st_size
                    / (1024 * 1024),
                    "processing_completed_at": datetime.now().isoformat(),
                }
            )

        # Clean up temporary directory
        cleanup_temp_dir(temp_dir)

        logger.info(f"Successfully completed background processing for job {job_id}")

    except Exception as e:
        logger.error(f"Error in background processing for job {job_id}: {e}")

        # Update job with error info
        if job_id in jobs:
            jobs[job_id].status = JobStatus.FAILED
            jobs[job_id].error_message = str(e)
            jobs[job_id].completed_at = datetime.now()


@app.post("/process", response_model=ProcessResponse)
async def process_point_cloud(
    request: ProcessRequest, background_tasks: BackgroundTasks
):
    """
    Initiate colorized point cloud generation from an address.

    This endpoint:
    1. Geocodes the provided address
    2. Searches for LiDAR point cloud data availability
    3. Returns immediately with a job ID for tracking
    4. Processes the point cloud in the background

    Args:
        request: ProcessRequest containing the address and optional parameters
        background_tasks: FastAPI background tasks for processing

    Returns:
        ProcessResponse with job ID and initial status
    """
    logger.info(f"Processing request for address: {request.address}")

    # Generate unique job ID
    job_id = str(uuid.uuid4())

    try:
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

        # Create job entry
        job = Job(
            job_id=job_id,
            address=request.address,
            status=JobStatus.PENDING,
            created_at=datetime.now(),
            metadata={
                "coordinates": {"latitude": lat, "longitude": lon},
                "buffer_km": request.buffer_km,
                "lidar_products_found": len(laz_products),
            },
        )
        jobs[job_id] = job

        # Start background processing
        background_tasks.add_task(
            process_point_cloud_background, job_id, request.address, request.buffer_km
        )

        logger.info(f"Started background processing for job {job_id}")

        # Return immediately with job information
        return ProcessResponse(
            success=True,
            message=f"Processing started for {request.address}. Use job ID to check status.",
            job_id=job_id,
            status=JobStatus.PENDING,
            metadata=job.metadata,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Unexpected error occurred: {str(e)}"
        )


@app.get("/job/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """
    Get the status of a processing job.

    Args:
        job_id: The job ID returned from the /process endpoint

    Returns:
        JobStatusResponse with current job status and details
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]

    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        address=job.address,
        created_at=job.created_at,
        completed_at=job.completed_at,
        output_file=job.output_file,
        error_message=job.error_message,
        metadata=job.metadata,
    )


@app.get("/jobs")
async def list_jobs():
    """
    List all jobs (for debugging/monitoring).

    Returns:
        List of all jobs with their current status
    """
    return [
        {
            "job_id": job.job_id,
            "address": job.address,
            "status": job.status,
            "created_at": job.created_at,
            "completed_at": job.completed_at,
        }
        for job in jobs.values()
    ]


@app.get("/download/{job_id}")
async def download_file(job_id: str):
    """
    Download a processed point cloud file by job ID.

    Args:
        job_id: Job ID of the completed processing task

    Returns:
        FileResponse with the requested file
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]

    if job.status != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Job is not completed. Current status: {job.status}",
        )

    if not job.output_file:
        raise HTTPException(status_code=404, detail="Output file not found")

    file_path = Path(job.output_file)

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found on disk")

    return FileResponse(
        path=str(file_path),
        filename=f"colorized_{job_id}.laz",
        media_type="application/octet-stream",
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
