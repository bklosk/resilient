"""
Job management endpoints for point cloud processing.
"""

import logging
import threading
import time
import tempfile
import shutil
import uuid
from pathlib import Path
from datetime import datetime
from typing import List

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from uvicorn.logging import DefaultFormatter

from .shared import (
    Job, JobStatus, ProcessRequest, ProcessResponse, JobStatusResponse,
    jobs, jobs_lock, update_job_status, get_job_safe
)

logger = logging.getLogger("photogrammetry-api")

router = APIRouter()


def cleanup_temp_dir(temp_dir: Path):
    """Clean up temporary directory with error handling."""
    try:
        if temp_dir and temp_dir.exists():
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temporary directory: {temp_dir}")
    except Exception as e:
        logger.warning(f"Failed to clean up temporary directory {temp_dir}: {e}")


def process_point_cloud_background(job_id: str, address: str, buffer_km: float):
    """Background task with enhanced error handling and thread safety."""
    temp_dir = None
    # Setup per-job logging to capture progress
    logs_dir = Path(__file__).parent.parent / "data" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / f"{job_id}.log"
    thread_id = threading.get_ident()

    class ThreadFilter(logging.Filter):
        def __init__(self, tid):
            super().__init__()
            self.tid = tid

        def filter(self, record: logging.LogRecord) -> bool:
            return record.thread == self.tid

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        DefaultFormatter(fmt="%(levelprefix)s %(message)s", use_colors=False)
    )
    file_handler.addFilter(ThreadFilter(thread_id))
    logger.addHandler(file_handler)

    # Store log file path in job metadata for later retrieval
    update_job_status(job_id, metadata={"log_file": str(log_file)})

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

                from services.core.geocode import Geocoder
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

            from services.data.get_point_cloud import PointCloudDatasetFinder
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

            datasets = pc_fetcher.find_datasets_for_location(lat, lon)

            if not datasets:
                error_msg = f"No LiDAR data found for location: {address} (lat: {lat:.6f}, lon: {lon:.6f})"
                logger.error(error_msg)
                update_job_status(
                    job_id,
                    status=JobStatus.FAILED,
                    error_message=error_msg,
                    completed_at=datetime.now(),
                )
                return

            logger.info(f"Found {len(datasets)} LiDAR datasets for processing")

            update_job_status(
                job_id,
                metadata={
                    "lidar_datasets_found": len(datasets),
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

            # Initialize processor and process with detailed error handling
            try:
                from services.core.process_point_cloud import PointCloudProcessor
                processor = PointCloudProcessor(output_dir=str(temp_dir))
                logger.info(f"Initialized PointCloudProcessor for job {job_id}")

                update_job_status(
                    job_id,
                    metadata={"processing_step": "starting_point_cloud_processing"},
                )

                output_path = processor.process_from_address(address)

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
        logger.removeHandler(file_handler)
        file_handler.close()


@router.post("/process", response_model=ProcessResponse)
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


@router.get("/job/{job_id}", response_model=JobStatusResponse)
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

    # Read last 20 log lines if log file exists
    log_tail: List[str] = []
    log_file = job.metadata.get("log_file")
    if log_file:
        try:
            with open(log_file, "r") as fh:
                lines = fh.readlines()[-20:]
                log_tail = [ln.strip() for ln in lines if ln.strip()]
        except Exception as e:
            logger.warning(f"Could not read log file for job {job_id}: {e}")

    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        address=job.address,
        created_at=job.created_at,
        completed_at=job.completed_at,
        output_file=job.output_file,
        error_message=job.error_message,
        metadata=job.metadata,
        log_tail=log_tail,
    )


@router.get("/jobs")
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


@router.get("/download/{job_id}")
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
