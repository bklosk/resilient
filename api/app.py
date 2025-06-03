#!/usr/bin/env python3
"""
FastAPI Application for Photogrammetry Point Cloud Processing
"""

import os
import sys
import logging
import threading
import time
from pathlib import Path
from typing import Dict, Any, Optional
import tempfile
import shutil
import uuid
from datetime import datetime
from enum import Enum

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, validator
from uvicorn.logging import DefaultFormatter

# Add the scripts directory to the Python path so we can import the modules
scripts_dir = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(scripts_dir))

# Configure logging with uvicorn-style colors first (before any logging usage)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(
    DefaultFormatter(
        fmt="%(levelprefix)s %(message)s",  # identical to uvicorn default
        use_colors=True,  # force colours
    )
)

logger = logging.getLogger("photogrammetry-api")
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Remove any default handlers to avoid duplicate logs
logger.propagate = False

# Import our helper modules
try:
    from process_point_cloud import PointCloudColorizer
    from geocode import Geocoder
    from get_point_cloud import PointCloudDatasetFinder
    from get_orthophoto import NAIPFetcher
except ImportError as e:
    logger.error(f"Error importing required modules: {e}")
    logger.error(
        "Make sure you're running from the correct directory and all dependencies are installed"
    )
    sys.exit(1)

app = FastAPI(
    title="Photogrammetry Point Cloud API",
    description="API for generating colorized point clouds from address inputs",
    version="1.0.0",
)

# Configure CORS
# Allow origins can be overridden with the ``CORS_ORIGINS`` environment
# variable (comma separated). This helps when the frontend is served from
# a different host/port (e.g. Docker or Codespaces).
default_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://0.0.0.0:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3001",
]
cors_origins_env = os.environ.get("CORS_ORIGINS")
allow_origins = (
    [o.strip() for o in cors_origins_env.split(",") if o.strip()]
    if cors_origins_env
    else default_origins
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger.info(f"CORS allowed origins: {allow_origins}")


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


# Thread-safe job storage with lock
jobs: Dict[str, Job] = {}
jobs_lock = threading.Lock()


def update_job_status(job_id: str, **updates) -> bool:
    """Thread-safe job status update."""
    with jobs_lock:
        if job_id not in jobs:
            logger.error(f"Job {job_id} not found in jobs dict")
            return False

        for key, value in updates.items():
            if key == "metadata" and isinstance(value, dict):
                jobs[job_id].metadata.update(value)
            else:
                setattr(jobs[job_id], key, value)
        return True


def get_job_safe(job_id: str) -> Optional[Job]:
    """Thread-safe job retrieval."""
    with jobs_lock:
        return jobs.get(job_id)


class ProcessRequest(BaseModel):
    """Request model for point cloud processing."""

    address: str = Field(
        ...,
        description="Street address to process",
        example="1250 Wildwood Road, Boulder, CO",
        min_length=5,
        max_length=200,
    )
    buffer_km: float = Field(
        default=1.0,
        description="Buffer distance in kilometers for LiDAR search",
        ge=0.1,
        le=5.0,
    )

    @validator("address")
    def validate_address(cls, v):
        if not v or not v.strip():
            raise ValueError("Address cannot be empty")
        return v.strip()


class ProcessResponse(BaseModel):
    """Response model for point cloud processing."""

    success: bool
    message: str
    job_id: str
    status: JobStatus
    metadata: Dict[str, Any] = {}


class OrthophotoRequest(BaseModel):
    """Request model for orthophoto download."""

    address: str = Field(
        ...,
        description="Street address to fetch orthophoto for",
        example="1250 Wildwood Road, Boulder, CO",
        min_length=5,
        max_length=200,
    )
    image_size: Optional[str] = Field(
        default=None,
        description="Image size as 'width,height'. Use 'auto' or omit for native resolution",
    )

    @validator("address")
    def validate_address(cls, v):
        if not v or not v.strip():
            raise ValueError("Address cannot be empty")
        return v.strip()


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


@app.get("/health")
async def health_check():
    """Health check endpoint with dependency verification."""
    try:
        # Test imports
        from process_point_cloud import PointCloudColorizer
        from geocode import Geocoder

        return {
            "status": "healthy",
            "service": "photogrammetry-api",
            "version": "1.0.0",
            "dependencies": "ok",
            "active_jobs": len(jobs),
        }
    except ImportError as e:
        return {
            "status": "unhealthy",
            "service": "photogrammetry-api",
            "version": "1.0.0",
            "error": f"Missing dependencies: {str(e)}",
        }


def process_point_cloud_background(job_id: str, address: str, buffer_km: float):
    """Background task with enhanced error handling and thread safety."""
    temp_dir = None

    try:
        # Update job status to processing
        if not update_job_status(
            job_id,
            status=JobStatus.PROCESSING,
            metadata={"processing_step": "starting"},
        ):
            return

        logger.info(f"Starting background processing for job {job_id}")

        # Step 1: Input validation
        address = address.strip() if address else ""
        if not address:
            raise ValueError("Address cannot be empty")

        if not (0.1 <= buffer_km <= 5.0):
            raise ValueError(
                f"Buffer distance must be between 0.1 and 5.0 km, got {buffer_km}"
            )

        # Step 2: Geocoding with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                update_job_status(
                    job_id,
                    metadata={"processing_step": "geocoding", "attempt": attempt + 1},
                )

                geocoder = Geocoder()
                lat, lon = geocoder.geocode_address(address)

                if lat is None or lon is None:
                    raise ValueError("Failed to get valid coordinates")

                # Validate coordinates are reasonable
                if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
                    raise ValueError(f"Invalid coordinates: {lat}, {lon}")

                logger.info(f"Address geocoded to: {lat:.6f}, {lon:.6f}")

                update_job_status(
                    job_id,
                    metadata={
                        "coordinates": {
                            "latitude": float(lat),
                            "longitude": float(lon),
                        },
                        "processing_step": "geocoded",
                    },
                )
                break

            except Exception as e:
                if attempt == max_retries - 1:  # Last attempt
                    error_msg = f"Failed to geocode address '{address}' after {max_retries} attempts: {str(e)}"
                    logger.error(error_msg)
                    update_job_status(
                        job_id,
                        status=JobStatus.FAILED,
                        error_message=error_msg,
                        completed_at=datetime.now(),
                    )
                    return
                else:
                    logger.warning(
                        f"Geocoding attempt {attempt + 1} failed: {e}, retrying..."
                    )
                    time.sleep(2**attempt)  # Exponential backoff

        # Step 3: LiDAR data search with validation
        try:
            update_job_status(job_id, metadata={"processing_step": "searching_lidar"})

            pc_fetcher = PointCloudDatasetFinder()
            bbox = pc_fetcher.generate_bounding_box(lat, lon, buffer_km)

            # Validate bounding box - handle both string and list formats
            if not bbox:
                raise ValueError("No bounding box generated")

            # If bbox is a string, validate it has 4 comma-separated values
            if isinstance(bbox, str):
                bbox_parts = bbox.split(",")
                if len(bbox_parts) != 4:
                    raise ValueError(
                        f"Invalid bounding box format - expected 4 values, got {len(bbox_parts)}: {bbox}"
                    )
                try:
                    # Validate all parts are numeric
                    [float(part.strip()) for part in bbox_parts]
                except ValueError:
                    raise ValueError(
                        f"Invalid bounding box values - non-numeric data: {bbox}"
                    )
            # If bbox is a list, validate it has 4 values
            elif isinstance(bbox, (list, tuple)):
                if len(bbox) != 4:
                    raise ValueError(
                        f"Invalid bounding box format - expected 4 values, got {len(bbox)}: {bbox}"
                    )
            else:
                raise ValueError(
                    f"Invalid bounding box type - expected string or list, got {type(bbox)}: {bbox}"
                )

            logger.info(f"Generated valid bounding box: {bbox}")

            products = pc_fetcher.search_lidar_products(bbox)

            if not products:
                error_msg = f"No LiDAR data found for location: {address} (lat: {lat:.6f}, lon: {lon:.6f})"
                logger.error(error_msg)
                update_job_status(
                    job_id,
                    status=JobStatus.FAILED,
                    error_message=error_msg,
                    completed_at=datetime.now(),
                )
                return

            laz_products = pc_fetcher.filter_laz_products(products)
            if not laz_products:
                error_msg = f"No LAZ format LiDAR data found for location: {address}"
                logger.error(error_msg)
                update_job_status(
                    job_id,
                    status=JobStatus.FAILED,
                    error_message=error_msg,
                    completed_at=datetime.now(),
                )
                return

            logger.info(f"Found {len(laz_products)} LAZ products for processing")

            update_job_status(
                job_id,
                metadata={
                    "lidar_products_found": len(laz_products),
                    "processing_step": "lidar_data_found",
                    "bbox": bbox,
                },
            )

        except Exception as e:
            error_msg = f"Error checking LiDAR data availability: {str(e)}"
            logger.error(error_msg, exc_info=True)
            update_job_status(
                job_id,
                status=JobStatus.FAILED,
                error_message=error_msg,
                completed_at=datetime.now(),
            )
            return

        # Step 4: Processing with temporary directory
        try:
            temp_dir = Path(tempfile.mkdtemp(prefix=f"photogrammetry_{job_id}_"))
            logger.info(f"Created temporary directory: {temp_dir}")

            update_job_status(
                job_id,
                metadata={
                    "processing_step": "downloading_and_processing",
                    "temp_dir": str(temp_dir),
                },
            )

            # Initialize colorizer and process with detailed error handling
            try:
                colorizer = PointCloudColorizer(output_dir=str(temp_dir))
                logger.info(f"Initialized PointCloudColorizer for job {job_id}")

                update_job_status(
                    job_id,
                    metadata={"processing_step": "starting_point_cloud_processing"},
                )

                output_path = colorizer.process_from_address(address)

                if not output_path:
                    raise ValueError("Point cloud processing returned no output path")

                logger.info(f"Point cloud processing completed, output: {output_path}")

            except RuntimeError as e:
                if "Failed to download point cloud" in str(e):
                    error_msg = f"Point cloud download failed for location: {address}. This could be due to: 1) No LiDAR data available for this area, 2) Network connectivity issues, 3) USGS service unavailable. Original error: {str(e)}"
                    logger.error(error_msg)
                    update_job_status(
                        job_id,
                        status=JobStatus.FAILED,
                        error_message=error_msg,
                        completed_at=datetime.now(),
                    )
                    return
                else:
                    raise  # Re-raise other RuntimeErrors
            except Exception as e:
                error_msg = f"Point cloud processing failed: {str(e)}"
                logger.error(error_msg, exc_info=True)
                update_job_status(
                    job_id,
                    status=JobStatus.FAILED,
                    error_message=error_msg,
                    completed_at=datetime.now(),
                )
                return

            output_file = Path(output_path)
            if not output_file.exists():
                raise ValueError(
                    f"Point cloud processing failed - output file not found: {output_path}"
                )

            if output_file.stat().st_size == 0:
                raise ValueError("Point cloud processing produced empty file")

            # Step 5: Move to permanent location with validation
            output_dir = Path(__file__).parent.parent / "data" / "outputs"
            output_dir.mkdir(parents=True, exist_ok=True)

            output_filename = f"colorized_{job_id}.laz"
            final_output_path = output_dir / output_filename

            # Ensure we don't overwrite existing files
            counter = 1
            while final_output_path.exists():
                output_filename = f"colorized_{job_id}_{counter}.laz"
                final_output_path = output_dir / output_filename
                counter += 1

            shutil.copy2(output_path, final_output_path)

            if not final_output_path.exists():
                raise ValueError("Failed to copy output file to final location")

            # Verify file integrity
            file_size_bytes = final_output_path.stat().st_size
            if file_size_bytes == 0:
                raise ValueError("Output file is empty after copy")

            file_size_mb = file_size_bytes / (1024 * 1024)

            # Update job completion
            update_job_status(
                job_id,
                status=JobStatus.COMPLETED,
                completed_at=datetime.now(),
                output_file=str(final_output_path),
                metadata={
                    "output_file_size_mb": round(file_size_mb, 2),
                    "output_file_size_bytes": file_size_bytes,
                    "processing_completed_at": datetime.now().isoformat(),
                    "processing_step": "completed",
                    "final_filename": output_filename,
                },
            )

            logger.info(
                f"Successfully completed processing for job {job_id}, file size: {file_size_mb:.2f} MB"
            )

        except Exception as e:
            error_msg = f"Processing error: {str(e)}"
            logger.error(error_msg, exc_info=True)
            update_job_status(
                job_id,
                status=JobStatus.FAILED,
                error_message=error_msg,
                completed_at=datetime.now(),
            )

    except Exception as e:
        logger.error(
            f"Unexpected error in background processing for job {job_id}: {e}",
            exc_info=True,
        )
        update_job_status(
            job_id,
            status=JobStatus.FAILED,
            error_message=f"Unexpected processing error: {str(e)}",
            completed_at=datetime.now(),
        )
    finally:
        # Always clean up temporary directory
        if temp_dir:
            cleanup_temp_dir(temp_dir)


@app.post("/process", response_model=ProcessResponse)
async def process_point_cloud(
    request: ProcessRequest, background_tasks: BackgroundTasks
):
    """Initiate colorized point cloud generation from an address."""
    try:
        # Generate unique job ID
        job_id = str(uuid.uuid4())

        logger.info(
            f"Received processing request for address: '{request.address}', job_id: {job_id}"
        )

        # Additional validation
        address = request.address.strip()
        if len(address) < 5:
            raise HTTPException(
                status_code=400, detail="Address too short (minimum 5 characters)"
            )

        if len(address) > 200:
            raise HTTPException(
                status_code=400, detail="Address too long (maximum 200 characters)"
            )

        # Create job entry with thread safety
        job = Job(
            job_id=job_id,
            address=address,
            status=JobStatus.PENDING,
            created_at=datetime.now(),
            metadata={
                "buffer_km": request.buffer_km,
                "processing_step": "pending",
                "created_by": "api",
                "request_timestamp": datetime.now().isoformat(),
            },
        )

        with jobs_lock:
            jobs[job_id] = job

        # Start background processing
        background_tasks.add_task(
            process_point_cloud_background, job_id, address, request.buffer_km
        )

        logger.info(f"Job {job_id} created and background task started")

        return ProcessResponse(
            success=True,
            message=f"Processing started for '{address}'. Use job ID to check status.",
            job_id=job_id,
            status=JobStatus.PENDING,
            metadata=job.metadata,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating processing job: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to start processing: {str(e)}"
        )


@app.get("/job/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get job status with validation."""
    if not job_id or not job_id.strip():
        raise HTTPException(status_code=400, detail="Job ID cannot be empty")

    # Validate UUID format
    try:
        uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid job ID format")

    job = get_job_safe(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

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
    """Download file with enhanced validation."""
    try:
        if not job_id or not job_id.strip():
            raise HTTPException(status_code=400, detail="Job ID cannot be empty")

        # Validate UUID format
        try:
            uuid.UUID(job_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid job ID format")

        job = get_job_safe(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        if job.status == JobStatus.PENDING:
            raise HTTPException(status_code=202, detail="Job is still pending")
        elif job.status == JobStatus.PROCESSING:
            raise HTTPException(status_code=202, detail="Job is still processing")
        elif job.status == JobStatus.FAILED:
            raise HTTPException(
                status_code=400,
                detail=f"Job failed: {job.error_message or 'Unknown error'}",
            )
        elif job.status != JobStatus.COMPLETED:
            raise HTTPException(
                status_code=400, detail=f"Job is in unexpected state: {job.status}"
            )

        if not job.output_file:
            raise HTTPException(status_code=404, detail="Output file path not found")

        file_path = Path(job.output_file)

        if not file_path.exists():
            logger.error(f"Output file missing from disk: {file_path}")
            raise HTTPException(status_code=404, detail="Output file not found on disk")

        if not file_path.is_file():
            logger.error(f"Output path is not a file: {file_path}")
            raise HTTPException(
                status_code=400, detail="Output path is not a valid file"
            )

        # Security check - ensure file is within expected directory
        expected_dir = Path(__file__).parent.parent / "data" / "outputs"
        try:
            file_path.resolve().relative_to(expected_dir.resolve())
        except ValueError:
            logger.error(
                f"Security violation: file outside expected directory: {file_path}"
            )
            raise HTTPException(status_code=403, detail="Access denied")

        filename = job.metadata.get("final_filename", f"colorized_{job_id}.laz")

        return FileResponse(
            path=str(file_path),
            filename=filename,
            media_type="application/octet-stream",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading file for job {job_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/orthophoto")
async def download_orthophoto(request: OrthophotoRequest):
    """Fetch and return an orthophoto for the given address."""
    try:
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
        logger.error(f"Error downloading orthophoto for {request.address}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch orthophoto")


def cleanup_temp_dir(temp_dir: Path):
    """Clean up temporary directory with error handling."""
    try:
        if temp_dir and temp_dir.exists():
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
    ortho_dir = data_dir / "orthophotos"
    ortho_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Photogrammetry API startup complete")


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler."""
    logger.info("Photogrammetry API shutting down...")


if __name__ == "__main__":
    import uvicorn

    # Run the server
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
