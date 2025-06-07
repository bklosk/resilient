"""
AI analysis endpoints.
"""

import logging
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

from fastapi import APIRouter, HTTPException

from .shared import FloodAnalysisRequest, FloodAnalysisResponse

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


@router.post("/analyze-flood", response_model=FloodAnalysisResponse)
async def analyze_flood_with_openai(request: FloodAnalysisRequest):
    """Analyze flood damage using OpenAI by comparing flood overhead and satellite images.
    
    This endpoint:
    1. Generates a flood overhead image for the specified address using existing flood-overhead functionality
    2. Fetches a satellite image (NAIP orthophoto) for the same bounding box
    3. Sends both images to OpenAI GPT-4 Vision for comprehensive flood damage analysis
    4. Returns the AI analysis along with metadata about the process
    
    Args:
        request: FloodAnalysisRequest containing address, bounding box size, and optional custom prompt
        
    Returns:
        FloodAnalysisResponse with AI analysis, success status, and metadata
    """
    try:
        logger.info(f"Starting flood analysis for address: {request.address}")
        
        # Create temporary directory for image storage
        temp_dir = Path(tempfile.mkdtemp(prefix="flood_analysis_"))
        
        try:
            # Step 1: Generate flood overhead image
            logger.info("Generating flood overhead image...")
            from services.utils.flood_depth import generate
            from services.visualization.overhead_image import render
            
            # Generate flood depth GeoTIFF
            flood_tiff = generate(request.address, request.bbox_m)
            
            # Convert to colored PNG visualization (high resolution for better AI analysis)
            flood_png = render(flood_tiff, target_size=2048)
            
            # Copy flood image to temp directory with consistent naming
            flood_image_path = temp_dir / "flood_overhead.png"
            shutil.copy2(flood_png, flood_image_path)
            logger.info(f"Flood overhead image saved to: {flood_image_path}")
            
            # Step 2: Get satellite image using NAIP fetcher
            logger.info("Fetching satellite image...")
            from services.core.geocode import Geocoder
            geocoder = Geocoder()
            lat, lon = geocoder.geocode_address(request.address)
            
            # Calculate bounding box for satellite image (same as flood image)
            # Convert meters to degrees (rough approximation)
            meter_to_deg = request.bbox_m / 111000  # 1 degree ≈ 111km
            bbox = [
                lon - meter_to_deg / 2,  # min_x (west)
                lat - meter_to_deg / 2,  # min_y (south)  
                lon + meter_to_deg / 2,  # max_x (east)
                lat + meter_to_deg / 2   # max_y (north)
            ]
            
            # Fetch NAIP satellite image
            from services.data.get_orthophoto import NAIPFetcher
            naip_fetcher = NAIPFetcher()
            satellite_output_path = str(temp_dir / "satellite_image.tif")
            satellite_metadata = naip_fetcher.export_image(
                min_lon=bbox[0],
                min_lat=bbox[1], 
                max_lon=bbox[2],
                max_lat=bbox[3],
                output_path=satellite_output_path
            )
            satellite_image_path = satellite_output_path
            
            if not satellite_image_path or not Path(satellite_image_path).exists():
                raise HTTPException(
                    status_code=500,
                    detail="Failed to fetch satellite image for the specified location"
                )
            
            logger.info(f"Satellite image saved to: {satellite_image_path}")
            
            # Step 3: Analyze both images with OpenAI
            logger.info("Sending images to OpenAI for analysis...")
            from services.ai.openai_analysis import OpenAIAnalyzer
            analyzer = OpenAIAnalyzer()
            
            # Always use the default prompt - no custom prompts allowed
            analysis_result = analyzer.analyze_flood_images(
                flood_image_path=str(flood_image_path),
                satellite_image_path=str(satellite_image_path)
            )
            
            if not analysis_result["success"]:
                return FloodAnalysisResponse(
                    success=False,
                    message="OpenAI analysis failed",
                    error=analysis_result.get("error", "Unknown error during AI analysis"),
                    timestamp=datetime.now(),
                    flood_image_path=str(flood_image_path),
                    satellite_image_path=str(satellite_image_path)
                )
            
            logger.info("OpenAI analysis completed successfully")
            
            return FloodAnalysisResponse(
                success=True,
                message="Flood analysis completed successfully",
                analysis=analysis_result["analysis"],
                model=analysis_result.get("model", "gpt-4-vision-preview"),
                tokens_used=analysis_result.get("tokens_used"),
                timestamp=datetime.now(),
                flood_image_path=str(flood_image_path),
                satellite_image_path=str(satellite_image_path)
            )
            
        finally:
            # Clean up temporary directory
            cleanup_temp_dir(temp_dir)
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in flood analysis for {request.address}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to complete flood analysis: {str(e)}"
        )
