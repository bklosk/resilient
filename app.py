#!/usr/bin/env python3
"""
FastAPI Application for Photogrammetry Point Cloud Processing
"""

import os
import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from uvicorn.logging import DefaultFormatter
from contextlib import asynccontextmanager

# Configure logging with uvicorn-style colors first (before any logging usage)
handler = logging.StreamHandler()
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

# Add the services directory to the Python path so we can import the modules
import sys
services_dir = Path(__file__).parent / "services"
sys.path.insert(0, str(services_dir))

# Import routers
from routers.health import router as health_router
from routers.jobs import router as jobs_router
from routers.images import router as images_router
from routers.analysis import router as analysis_router
from services.data.get_wrtc_tif import router as cog_router # Added import

@asynccontextmanager
async def lifespan(app : FastAPI):
    """Startup Events & Shutdown Events. Shutdown events occur after this event is yield'd."""
    logger.info("Photogrammetry API starting up...")

    # Create necessary directories
    data_dir = Path(__file__).parent / "data"
    output_dir = data_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    ortho_dir = data_dir / "orthophotos"
    ortho_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Photogrammetry API startup complete")

    yield
    """Shutdown Events."""
    logger.info("Photogrammetry API shutting down...")

app = FastAPI(
    title="Photogrammetry Point Cloud API",
    description="API for generating colorized point clouds from address inputs",
    version="1.0.0",
    lifespan=lifespan,
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

# Include routers
app.include_router(health_router, tags=["health"])
app.include_router(jobs_router, tags=["jobs"])
app.include_router(images_router, tags=["images"])
app.include_router(analysis_router, tags=["analysis"])
app.include_router(cog_router) # Added router

if __name__ == "__main__":
    import uvicorn

    # Run the server with reload only watching specific directories
    # This prevents infinite reload loops caused by log files, cache files, and output files
    uvicorn.run(
        "app:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True, 
        log_level="info",
        reload_dirs=["./services", "./routers"]  # Watch both services and routers directories
    )
