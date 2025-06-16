from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse  # Added
from rio_tiler.io import COGReader
from typing import Optional, Dict, Any
import io  # Added
import os
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
import logging
import numpy as np
import matplotlib.cm as cm
from PIL import Image

router = APIRouter(prefix="/cog", tags=["cog"])

logger = logging.getLogger(__name__)

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

# Get DigitalOcean Spaces configuration from environment
DO_SPACES_ENDPOINT = os.getenv(
    'DO_SPACES_ENDPOINT', 'https://nyc3.digitaloceanspaces.com')
DO_SPACES_REGION = os.getenv('DO_SPACES_REGION', 'nyc3')
DO_SPACES_KEY = os.getenv('DO_SPACES_KEY')
DO_SPACES_SECRET = os.getenv('DO_SPACES_SECRET')
DO_SPACES_BUCKET = os.getenv('DO_SPACES_BUCKET', 'wrtc')


def get_s3_client():
    """Create and return a DigitalOcean Spaces S3 client"""
    if not DO_SPACES_KEY or not DO_SPACES_SECRET:
        raise HTTPException(
            status_code=500,
            detail="DigitalOcean Spaces credentials not configured"
        )

    return boto3.client(
        's3',
        endpoint_url=DO_SPACES_ENDPOINT,
        region_name=DO_SPACES_REGION,
        aws_access_key_id=DO_SPACES_KEY,
        aws_secret_access_key=DO_SPACES_SECRET
    )


def get_signed_url(s3_key: str, expiration: int = 3600) -> str:
    """Generate a signed URL for accessing the S3 object"""
    try:
        s3_client = get_s3_client()
        url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': DO_SPACES_BUCKET, 'Key': s3_key},
            ExpiresIn=expiration
        )
        return url
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating signed URL for {s3_key}: {str(e)}"
        )


for state_abbr, state_full_name in STATES_DATA.items():
    COG_INDEX[state_abbr] = {}
    for layer_api_name, layer_prefix in LAYERS_PREFIX.items():
        # Store the S3 key instead of a full URL
        # The S3 key uses the full state name as folder (e.g., "Colorado/BP_CO_cog.tif")
        cog_filename = f"{layer_prefix}_{state_abbr}_cog.tif"
        COG_INDEX[state_abbr][layer_api_name] = f"{state_full_name}/{cog_filename}"


@router.get("/point", summary="Get pixel value at lon/lat")
async def point_lookup(
    state: str = Query(..., description="State abbreviation, e.g. 'AL'", enum=list(
        STATES_DATA.keys())),
    layer: str = Query(..., description="Layer name",
                       enum=list(LAYERS_PREFIX.keys())),
    lon: float = Query(..., description="Longitude"),
    lat: float = Query(..., description="Latitude"),
    band: Optional[int] = Query(1, description="Band index (1-based)", ge=1)
) -> Dict[str, Any]:
    """
    Retrieves a pixel value from a Cloud Optimized GeoTIFF (COG) for a given state, layer,
    longitude, latitude, and band.
    """
    # 1) Resolve S3 key
    state_layers = COG_INDEX.get(state)
    if not state_layers:
        # This should not happen if enum validation works, but as a safeguard:
        raise HTTPException(
            status_code=404, detail=f"State '{state}' not found.")

    s3_key = state_layers.get(layer)
    if not s3_key:
        # This should not happen if enum validation works, but as a safeguard:
        raise HTTPException(
            status_code=404, detail=f"Layer '{layer}' not found for state '{state}'.")

    # 2) Generate signed URL for authenticated access
    url = get_signed_url(s3_key)

    # 3) Read the pixel
    try:
        with COGReader(url) as cog:
            # rio-tiler expects 'indexes' to be a tuple or list of band numbers
            point_data = cog.point(lon, lat, indexes=(band,))
    except Exception as e:
        # Catching generic Exception, but more specific exceptions from rio-tiler could be caught
        # e.g., TileOutsideBounds, InvalidFormat, etc.
        raise HTTPException(
            status_code=400, detail=f"Error reading COG for {state}/{layer} at ({lon},{lat}): {str(e)}")

    if not point_data or point_data.data is None or point_data.data.size == 0:
        # This can happen if the point is outside the COG's bounds or in a nodata area not caught by an exception
        value = None  # Or some other indicator of no data
        mask_value = 0  # No data
    else:
        # .item() to convert numpy type to Python native type
        value = point_data.data[0].item()
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
    state: str = Query(..., description="State abbreviation, e.g. 'AL'", enum=list(
        STATES_DATA.keys())),
    layer: str = Query(..., description="Layer name",
                       enum=list(LAYERS_PREFIX.keys())),
    min_lon: float = Query(...,
                           description="Minimum longitude of the bounding box"),
    min_lat: float = Query(...,
                           description="Minimum latitude of the bounding box"),
    max_lon: float = Query(...,
                           description="Maximum longitude of the bounding box"),
    max_lat: float = Query(...,
                           description="Maximum latitude of the bounding box"),
    band: Optional[int] = Query(1, description="Band index (1-based)", ge=1),
    width: Optional[int] = Query(
        256, description="Output width in pixels", gt=0),
    height: Optional[int] = Query(
        256, description="Output height in pixels", gt=0),
    colormap: Optional[str] = Query(
        "Reds",
        description="Colormap to apply",
        enum=["Reds", "YlOrRd", "viridis",
              "plasma", "inferno", "magma", "none"]
    )
) -> StreamingResponse:
    """
    Retrieves a raster image (PNG) from a Cloud Optimized GeoTIFF (COG)
    for a given state, layer, bounding box, and output dimensions.
    """
    # 1) Resolve S3 key
    state_layers = COG_INDEX.get(state)
    if not state_layers:
        raise HTTPException(
            status_code=404, detail=f"State '{state}' not found.")

    s3_key = state_layers.get(layer)
    if not s3_key:
        raise HTTPException(
            status_code=404, detail=f"Layer '{layer}' not found for state '{state}'.")

    # 2) Generate signed URL for authenticated access
    url = get_signed_url(s3_key)

    # Define bounding box
    bbox = (min_lon, min_lat, max_lon, max_lat)

    # 3) Read the raster part
    try:
        print(f"DEBUG: Reading COG data for bbox: {bbox}")
        with COGReader(url) as cog:
            print(
                f"DEBUG: COG info - bands: {cog.info().count}, dtype: {cog.info().dtype}")

            # rio-tiler expects 'indexes' to be a tuple or list of band numbers
            # Use part() to get data for a specific bounding box,
            # resampled to width/height
            # Assuming input bbox coordinates are in WGS84 (EPSG:4326),
            # which is default for rio-tiler.part
            img_data = cog.part(bbox, indexes=(band,),
                                width=width, height=height)

            print(f"DEBUG: Extracted img_data type: {type(img_data)}")
            print(f"DEBUG: img_data.data shape: {img_data.data.shape}")
            print(f"DEBUG: img_data.data dtype: {img_data.data.dtype}")
            print(
                f"DEBUG: img_data has mask: {hasattr(img_data, 'mask') and img_data.mask is not None}")

            if img_data.data is None or img_data.data.size == 0:
                raise HTTPException(
                    status_code=404,
                    detail=f"No data found for the given bounding box "
                    f"and band for {state}/{layer}."
                )

            # Apply colormap if requested
            if colormap != "none":
                img_bytes = apply_colormap_to_raster(img_data, colormap)
            else:
                # For raw data without colormap, we need to handle data type conversion
                # Convert int32 to uint8 for PNG compatibility
                img_bytes = convert_raw_to_png(img_data)

    except HTTPException:  # Re-raise HTTPException if it was raised above
        raise
    except Exception as e:
        # Catching generic Exception, but more specific exceptions
        # from rio-tiler could be caught
        raise HTTPException(
            status_code=400,
            detail=f"Error reading or processing COG for {state}/{layer} "
            f"with bbox {bbox}: {str(e)}"
        )

    if not img_bytes:
        raise HTTPException(
            status_code=500,
            detail="Failed to render image from COG data."
        )

    return StreamingResponse(io.BytesIO(img_bytes), media_type="image/png")


def convert_raw_to_png(img_data) -> bytes:
    """Convert raw raster data to PNG bytes with proper data type handling."""
    # Get the data array (first band)
    data = img_data.data[0]  # Shape: (height, width)

    # Handle nodata values - rio-tiler uses different mask format
    if hasattr(img_data, 'mask') and img_data.mask is not None:
        # rio-tiler mask: 255 = valid data, 0 = masked (invalid)
        # We need to invert this for np.ma.array (True = masked, False = valid)
        if len(img_data.mask.shape) > 2:
            mask = img_data.mask[0]  # First band mask
        else:
            mask = img_data.mask
        # Convert rio-tiler mask (255=valid, 0=invalid) to numpy mask (True=invalid, False=valid)
        mask = mask == 0  # True where data is invalid (mask value 0)
        data = np.ma.array(data, mask=mask)
    else:
        # Create mask for NaN values
        mask = np.isnan(data)
        data = np.ma.array(data, mask=mask)

    # Get valid data for normalization
    valid_data = data.compressed()  # Gets non-masked values

    if len(valid_data) == 0:
        # No valid data - create transparent image
        height, width = data.shape
        rgba = np.zeros((height, width, 4), dtype=np.uint8)
    else:
        # Normalize data to 0-255 range for grayscale
        vmin = float(valid_data.min())
        vmax = float(valid_data.max())

        if vmax - vmin < 1e-6:
            # All values are the same - use middle gray
            norm_data = np.full_like(data, 128, dtype=np.uint8)
        else:
            norm_data = np.clip(
                ((data - vmin) / (vmax - vmin)) * 255, 0, 255
            ).astype(np.uint8)

        # Create RGBA array
        height, width = data.shape
        rgba = np.zeros((height, width, 4), dtype=np.uint8)

        # Set grayscale values for RGB channels
        valid_mask = ~data.mask
        rgba[valid_mask, 0] = norm_data[valid_mask]  # R
        rgba[valid_mask, 1] = norm_data[valid_mask]  # G
        rgba[valid_mask, 2] = norm_data[valid_mask]  # B

        # Set alpha: opaque for valid data, transparent for nodata
        rgba[..., 3] = np.where(valid_mask, 255, 0)

    # Convert to PIL image and save to bytes
    img = Image.fromarray(rgba, mode="RGBA")
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    return img_bytes.getvalue()


def apply_colormap_to_raster(img_data, colormap_name: str) -> bytes:
    """Apply a colormap to raster data and return PNG bytes."""
    # Get the data array (first band)
    data = img_data.data[0]  # Shape: (height, width)

    print(f"DEBUG: Data shape: {data.shape}")
    print(f"DEBUG: Data dtype: {data.dtype}")
    print(f"DEBUG: Data min: {data.min()}, max: {data.max()}")
    print(
        f"DEBUG: Data unique values (first 10): {np.unique(data.flatten())[:10]}")

    # Handle nodata values - rio-tiler uses different mask format
    if hasattr(img_data, 'mask') and img_data.mask is not None:
        print(f"DEBUG: Mask shape: {img_data.mask.shape}")
        print(f"DEBUG: Mask dtype: {img_data.mask.dtype}")
        print(f"DEBUG: Mask unique values: {np.unique(img_data.mask)}")

        # rio-tiler mask: 255 = valid data, 0 = masked (invalid)
        # We need to invert this for np.ma.array (True = masked, False = valid)
        if len(img_data.mask.shape) > 2:
            mask = img_data.mask[0]  # First band mask
        else:
            mask = img_data.mask

        # Convert rio-tiler mask (255=valid, 0=invalid) to numpy mask (True=invalid, False=valid)
        mask = mask == 0  # True where data is invalid (mask value 0)
        data = np.ma.array(data, mask=mask)
        print(
            f"DEBUG: Valid pixels count after mask inversion: {(~mask).sum()}")
        print(f"DEBUG: Masked pixels count after mask inversion: {mask.sum()}")
    else:
        # Create mask for NaN values
        mask = np.isnan(data)
        data = np.ma.array(data, mask=mask)
        print(f"DEBUG: NaN pixels count: {mask.sum()}")

    # Get valid data for normalization
    valid_data = data.compressed()  # Gets non-masked values
    print(f"DEBUG: Valid data length: {len(valid_data)}")

    if len(valid_data) == 0:
        # No valid data - create transparent image
        print("DEBUG: No valid data found, creating transparent image")
        height, width = data.shape
        rgba = np.zeros((height, width, 4), dtype=np.uint8)
    else:
        # Normalize data to 0-1 range
        vmin = float(valid_data.min())
        vmax = float(valid_data.max())
        print(f"DEBUG: Valid data range: {vmin} to {vmax}")

        if vmax - vmin < 1e-6:
            # All values are the same
            print("DEBUG: All values are the same, using zeros")
            norm_data = np.zeros_like(data)
        else:
            norm_data = np.clip((data - vmin) / (vmax - vmin), 0, 1)
            print(
                f"DEBUG: Normalized data range: {norm_data.min()} to {norm_data.max()}")

        # Apply colormap
        cmap = cm.get_cmap(colormap_name)
        print(f"DEBUG: Using colormap: {colormap_name}")

        # Apply colormap to normalized data
        rgba = np.zeros((*data.shape, 4), dtype=np.uint8)

        # Only apply colormap to valid (non-masked) pixels
        valid_mask = ~data.mask
        if np.any(valid_mask):
            # Get colors for valid pixels
            valid_colors = cmap(norm_data[valid_mask])
            print(f"DEBUG: Applied colormap to {valid_mask.sum()} pixels")

            # Convert to 0-255 range and assign to RGBA array
            rgba[valid_mask] = (valid_colors * 255).astype(np.uint8)

            # Set alpha: opaque for valid data, transparent for nodata
            rgba[..., 3] = np.where(valid_mask, 255, 0)
            print(
                f"DEBUG: RGBA array stats - R: {rgba[..., 0].max()}, G: {rgba[..., 1].max()}, B: {rgba[..., 2].max()}, A: {rgba[..., 3].max()}")

    # Convert to PIL image and save to bytes
    img = Image.fromarray(rgba, mode="RGBA")
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    print(f"DEBUG: Generated PNG with {len(img_bytes.getvalue())} bytes")
    return img_bytes.getvalue()


# To integrate this router into your main FastAPI application:
# In your main app.py or similar:
# from services.data.get_wrtc_tif import router as cog_router
# app.include_router(cog_router)
