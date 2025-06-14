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
    """Health check endpoint with minimal dependency verification."""
    import time
    import os
    
    # Basic health check without heavy imports
    health_data = {
        "status": "healthy",
        "service": "photogrammetry-api",
        "version": "1.0.0",
        "active_jobs": len(jobs),
        "timestamp": time.time(),
    }
    
    # Get process info safely
    try:
        import psutil
        health_data["uptime"] = time.time() - psutil.Process(os.getpid()).create_time()
    except Exception:
        health_data["uptime"] = "unknown"
    
    # Optional: Test critical imports only if needed
    try:
        # Light weight test - just check if modules exist without instantiating
        import services.core.process_point_cloud
        import services.core.geocode
        health_data["dependencies"] = "ok"
    except ImportError as e:
        health_data["dependencies"] = f"warning: {str(e)}"
        # Still return healthy status - dependencies might load lazily
    except Exception as e:
        health_data["dependencies"] = f"loading: {str(e)}"
        # Still return healthy status during startup
    
    return health_data


@router.get("/health/ready")
async def readiness_check():
    """Lightweight readiness check for container startup."""
    return {"status": "ready", "timestamp": __import__("time").time()}


@router.get("/health/deep")
async def deep_health_check():
    """Comprehensive health check with full dependency verification."""
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


@router.get("/ready")
async def readiness_check():
    """Readiness check endpoint that ensures all services are fully loaded."""
    try:
        # Test full initialization of critical services
        from services.core.process_point_cloud import PointCloudProcessor
        from services.core.geocode import Geocoder
        
        # Try to instantiate key objects to verify they're working
        processor = PointCloudProcessor()
        geocoder = Geocoder()
        
        return {
            "status": "ready",
            "service": "photogrammetry-api",
            "version": "1.0.0",
            "all_services_loaded": True,
            "active_jobs": len(jobs),
        }
    except Exception as e:
        return {
            "status": "not_ready", 
            "service": "photogrammetry-api",
            "version": "1.0.0",
            "all_services_loaded": False,
            "error": str(e),
        }
