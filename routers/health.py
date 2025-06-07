"""
Health and system status endpoints.
"""

import logging
from fastapi import APIRouter
from fastapi.responses import RedirectResponse

from .shared import jobs

logger = logging.getLogger("photogrammetry-api")

router = APIRouter()


@router.get("/")
async def root():
    """Root endpoint that redirects to the OpenAPI specification."""
    return RedirectResponse("/openapi.json")


@router.get("/health")
async def health_check():
    """Health check endpoint with dependency verification."""
    try:
        # Test imports - use new modular processor
        from services.core.process_point_cloud import PointCloudProcessor
        from services.core.geocode import Geocoder

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
