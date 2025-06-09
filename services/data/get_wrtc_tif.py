from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse # Added
from rio_tiler.io import COGReader
from typing import Optional, Dict, Any
import io # Added

router = APIRouter(prefix="/cog", tags=["cog"])

# Mapping from state abbreviation to full name for URL construction
STATES_DATA: Dict[str, str] = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
    "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "DC": "District of Columbia",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois",
    "IN": "Indiana", "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana",
    "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota",
    "MS": "Mississippi", "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon",
    "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota",
    "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia",
    "WA": "Washington", "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming"
}

# Layer definitions: key is the short name used in API, value is the filename prefix
# Based on data.txt and image:
# BP: Burn Probability
# CFL: Conditional Flame Length
# CRPS: Conditional Risk to Potential Structures
# Exposure: Exposure Type
# FLEP4: 4ft Flame Length Exceedance Probability
# FLEP8: 8ft Flame Length Exceedance Probability
# RPS: Risk to Potential Structures
# WHP: Wildfire Hazard Potential
LAYERS_PREFIX: Dict[str, str] = {
    "burn_probability": "BP",
    "conditional_flame_length": "CFL",
    "conditional_risk_to_potential_structures": "CRPS",
    "exposure_type": "Exposure",
    "flame_length_exceedance_probability_4ft": "FLEP4",
    "flame_length_exceedance_probability_8ft": "FLEP8",
    "risk_to_potential_structures": "RPS",
    "wildfire_hazard_potential": "WHP"
}

COG_INDEX: Dict[str, Dict[str, str]] = {}
BASE_URL = "https://wrtc.nyc3.cdn.digitaloceanspaces.com"

for state_abbr, state_full_name in STATES_DATA.items():
    COG_INDEX[state_abbr] = {}
    for layer_api_name, layer_prefix in LAYERS_PREFIX.items():
        # The URL path uses the full state name (e.g., "Alabama", "New York", "District of Columbia")
        # The COG filename uses the state abbreviation (e.g., BP_AL_cog.tif)
        cog_filename = f"{layer_prefix}_{state_abbr}_cog.tif"
        COG_INDEX[state_abbr][layer_api_name] = f"{BASE_URL}/{state_full_name}/{cog_filename}"


@router.get("/point", summary="Get pixel value at lon/lat")
async def point_lookup(
    state: str = Query(..., description="State abbreviation, e.g. 'AL'", enum=list(STATES_DATA.keys())),
    layer: str = Query(..., description="Layer name", enum=list(LAYERS_PREFIX.keys())),
    lon: float = Query(..., description="Longitude"),
    lat: float = Query(..., description="Latitude"),
    band: Optional[int] = Query(1, description="Band index (1-based)", ge=1)
) -> Dict[str, Any]:
    """
    Retrieves a pixel value from a Cloud Optimized GeoTIFF (COG) for a given state, layer,
    longitude, latitude, and band.
    """
    # 1) Resolve URL
    state_layers = COG_INDEX.get(state)
    if not state_layers:
        # This should not happen if enum validation works, but as a safeguard:
        raise HTTPException(status_code=404, detail=f"State '{state}' not found.")

    url = state_layers.get(layer)
    if not url:
        # This should not happen if enum validation works, but as a safeguard:
        raise HTTPException(status_code=404, detail=f"Layer '{layer}' not found for state '{state}'.")

    # 2) Read the pixel
    try:
        with COGReader(url) as cog:
            # rio-tiler expects 'indexes' to be a tuple or list of band numbers
            point_data = cog.point(lon, lat, indexes=(band,))
    except Exception as e:
        # Catching generic Exception, but more specific exceptions from rio-tiler could be caught
        # e.g., TileOutsideBounds, InvalidFormat, etc.
        raise HTTPException(status_code=400, detail=f"Error reading COG for {state}/{layer} at ({lon},{lat}): {str(e)}")

    if not point_data or point_data.data is None or point_data.data.size == 0:
        # This can happen if the point is outside the COG's bounds or in a nodata area not caught by an exception
        value = None # Or some other indicator of no data
        mask_value = 0 # No data
    else:
        value = point_data.data[0].item() # .item() to convert numpy type to Python native type
        # rio-tiler's mask: 0 means masked (no data), 255 means valid data.
        # We want 0 for no-data, 1 for data.
        mask_value = 1 if point_data.mask.item() == 255 else 0


    return {
        "state": state,
        "layer": layer,
        "lon": lon,
        "lat": lat,
        "band": band,
        "value": value,
        "mask": mask_value  # 0=no-data, 1=data
    }

@router.get("/raster", summary="Get raster image for a bounding box", response_class=StreamingResponse)
async def raster_lookup(
    state: str = Query(..., description="State abbreviation, e.g. 'AL'", enum=list(STATES_DATA.keys())),
    layer: str = Query(..., description="Layer name", enum=list(LAYERS_PREFIX.keys())),
    min_lon: float = Query(..., description="Minimum longitude of the bounding box"),
    min_lat: float = Query(..., description="Minimum latitude of the bounding box"),
    max_lon: float = Query(..., description="Maximum longitude of the bounding box"),
    max_lat: float = Query(..., description="Maximum latitude of the bounding box"),
    band: Optional[int] = Query(1, description="Band index (1-based)", ge=1),
    width: Optional[int] = Query(256, description="Output width in pixels", gt=0),
    height: Optional[int] = Query(256, description="Output height in pixels", gt=0)
) -> StreamingResponse:
    """
    Retrieves a raster image (PNG) from a Cloud Optimized GeoTIFF (COG)
    for a given state, layer, bounding box, and output dimensions.
    """
    # 1) Resolve URL
    state_layers = COG_INDEX.get(state)
    if not state_layers:
        raise HTTPException(status_code=404, detail=f"State '{state}' not found.")

    url = state_layers.get(layer)
    if not url:
        raise HTTPException(status_code=404, detail=f"Layer '{layer}' not found for state '{state}'.")

    # Define bounding box
    bbox = (min_lon, min_lat, max_lon, max_lat)

    # 2) Read the raster part
    try:
        with COGReader(url) as cog:
            # rio-tiler expects 'indexes' to be a tuple or list of band numbers
            # Use part() to get data for a specific bounding box, resampled to width/height
            # Assuming input bbox coordinates are in WGS84 (EPSG:4326), which is default for rio-tiler.part
            img_data = cog.part(bbox, indexes=(band,), width=width, height=height)
            
            if img_data.data is None or img_data.data.size == 0:
                 raise HTTPException(status_code=404, detail=f"No data found for the given bounding box and band for {state}/{layer}.")

            # Render the image data to PNG bytes
            img_bytes = img_data.render(img_format="PNG")

    except HTTPException: # Re-raise HTTPException if it was raised above
        raise
    except Exception as e:
        # Catching generic Exception, but more specific exceptions from rio-tiler could be caught
        raise HTTPException(status_code=400, detail=f"Error reading or processing COG for {state}/{layer} with bbox {bbox}: {str(e)}")

    if not img_bytes:
        raise HTTPException(status_code=500, detail="Failed to render image from COG data.")

    return StreamingResponse(io.BytesIO(img_bytes), media_type="image/png")

# To integrate this router into your main FastAPI application:
# In your main app.py or similar:
# from services.data.get_wrtc_tif import router as cog_router
# app.include_router(cog_router)
