"""
Shared models and utilities for routers.
"""

import threading
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List

from pydantic import BaseModel, Field, validator


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


# Thread-safe job storage with lock - shared across all routers
jobs: Dict[str, Job] = {}
jobs_lock = threading.Lock()


def update_job_status(job_id: str, **updates) -> bool:
    """Thread-safe job status update."""
    with jobs_lock:
        if job_id not in jobs:
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


class FloodAnalysisRequest(BaseModel):
    """Request model for OpenAI flood analysis."""

    address: str = Field(
        ...,
        description="Street address to analyze",
        example="1250 Wildwood Road, Boulder, CO",
        min_length=5,
        max_length=200,
    )
    bbox_m: float = Field(
        default=64.0,
        description="Bounding box size in meters for imagery",
        ge=32.0,
        le=500.0,
    )

    @validator("address")
    def validate_address(cls, v):
        if not v or not v.strip():
            raise ValueError("Address cannot be empty")
        return v.strip()


class FloodAnalysisResponse(BaseModel):
    """Response model for OpenAI flood analysis."""

    success: bool
    message: str
    analysis: Optional[str] = None
    model: Optional[str] = None
    tokens_used: Optional[int] = None
    timestamp: datetime
    flood_image_path: Optional[str] = None
    satellite_image_path: Optional[str] = None
    error: Optional[str] = None


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
    log_tail: List[str] = []
