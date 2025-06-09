import os
import sys
import time
from datetime import datetime
from pathlib import Path
import boto3
import pystac
import rasterio
from shapely.geometry import box, mapping
from botocore.exceptions import ClientError, NoCredentialsError
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 1️⃣ Validate environment variables and setup S3 client
required_env_vars = ['DO_SPACES_ENDPOINT', 'DO_SPACES_KEY', 'DO_SPACES_SECRET']
missing_vars = [var for var in required_env_vars if not os.environ.get(var)]
if missing_vars:
    logging.error(f"Missing required environment variables: {missing_vars}")
    sys.exit(1)

try:
    session = boto3.session.Session()
    s3 = session.client(
        's3',
        endpoint_url=os.environ['DO_SPACES_ENDPOINT'],
        aws_access_key_id=os.environ['DO_SPACES_KEY'],
        aws_secret_access_key=os.environ['DO_SPACES_SECRET']
    )
    
    # Test S3 connection
    logging.info("Testing S3 connection...")
    s3.head_bucket(Bucket='wrtc')
    logging.info("S3 connection successful")
    
except NoCredentialsError:
    logging.error("Invalid S3 credentials")
    sys.exit(1)
except ClientError as e:
    error_code = e.response['Error']['Code']
    if error_code == '404':
        logging.error("Bucket 'wrtc' not found")
    else:
        logging.error(f"S3 connection failed: {e}")
    sys.exit(1)
except Exception as e:
    logging.error(f"Failed to setup S3 client: {e}")
    sys.exit(1)

# Configure GDAL environment variables for rasterio to access DO Spaces
# Extract endpoint without https:// prefix for GDAL
endpoint_url = os.environ['DO_SPACES_ENDPOINT']
if endpoint_url.startswith('https://'):
    gdal_endpoint = endpoint_url[8:]  # Remove 'https://'
elif endpoint_url.startswith('http://'):
    gdal_endpoint = endpoint_url[7:]   # Remove 'http://'
else:
    gdal_endpoint = endpoint_url

# Set GDAL environment variables for S3 access
os.environ['AWS_S3_ENDPOINT'] = gdal_endpoint
os.environ['AWS_VIRTUAL_HOSTING'] = 'FALSE'  # Required for DigitalOcean Spaces
os.environ['AWS_ACCESS_KEY_ID'] = os.environ['DO_SPACES_KEY']
os.environ['AWS_SECRET_ACCESS_KEY'] = os.environ['DO_SPACES_SECRET']

BUCKET = 'wrtc'  # Your actual bucket name
PREFIX = ''       # Changed from 'cogs/' to empty string to search from bucket root

# Ensure output directory exists
output_dir = Path('./stac_catalog')
output_dir.mkdir(exist_ok=True)
logging.info(f"Output directory: {output_dir.absolute()}")

def create_safe_item_id(key):
    """Create a safe STAC item ID from a file key"""
    # Replace problematic characters and ensure uniqueness
    safe_id = key.replace('/', '_').replace('.tif', '').replace(' ', '_')
    # Remove any other problematic characters
    safe_id = ''.join(c for c in safe_id if c.isalnum() or c in ['_', '-'])
    return safe_id

def retry_rasterio_operation(func, max_retries=3, delay=1):
    """Retry rasterio operations with exponential backoff"""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            logging.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
            time.sleep(delay)
            delay *= 2
    return None

# 2️⃣ Create root catalog
catalog = pystac.Catalog(id='us-cogs', description='STAC catalog of all COGs')

# 3️⃣ Walk your bucket
paginator = s3.get_paginator('list_objects_v2')
item_count = 0
error_count = 0
processed_ids = set()  # Track IDs to prevent duplicates

try:
    logging.info(f"Starting to process files in bucket '{BUCKET}' with prefix '{PREFIX}'")
    
    for page_num, page in enumerate(paginator.paginate(Bucket=BUCKET, Prefix=PREFIX)):
        contents = page.get('Contents', [])
        if not contents:
            if page_num == 0:  # Only log on first page if empty
                logging.warning(f"No objects found in bucket '{BUCKET}' with prefix '{PREFIX}'")
            continue
            
        logging.info(f"Processing page {page_num + 1} with {len(contents)} objects")
        
        for obj in contents:
            key = obj['Key']
            
            # Skip directories (keys ending with /)
            if key.endswith('/'):
                continue
                
            if not key.lower().endswith('.tif'):
                continue

            # Create safe item ID and check for duplicates
            item_id = create_safe_item_id(key)
            if item_id in processed_ids:
                logging.warning(f"Duplicate item ID '{item_id}' for key '{key}'. Skipping.")
                continue
            processed_ids.add(item_id)

            asset_s3_uri = f"s3://{BUCKET}/{key}" # URI for rasterio to open
            # Use the correct public URL format
            public_href = f"https://{BUCKET}.nyc3.cdn.digitaloceanspaces.com/{key}"

            item_bbox = None
            item_geometry = None
            # Default datetime, can be overridden if extracted from COG
            item_datetime = datetime.utcnow()
            item_properties = {}

            try:
                logging.info(f"Processing ({item_count + 1}): {key}")
                
                def read_raster_metadata():
                    with rasterio.open(asset_s3_uri) as src:
                        bounds = src.bounds
                        # Validate bounds
                        if not all(isinstance(b, (int, float)) and not (b != b) for b in bounds):  # Check for NaN
                            raise ValueError(f"Invalid bounds: {bounds}")
                        
                        bbox = list(bounds)
                        geometry = mapping(box(*bounds))
                        
                        # Validate geometry
                        if not geometry or 'coordinates' not in geometry:
                            raise ValueError("Invalid geometry generated")
                        
                        properties = {
                            'crs': str(src.crs) if src.crs else 'UNKNOWN',
                            'width': src.width,
                            'height': src.height,
                            'dtype': str(src.dtypes[0]) if src.dtypes else 'UNKNOWN'
                        }
                        
                        # Add resolution if available
                        if hasattr(src, 'res') and src.res:
                            properties['resolution_x'] = src.res[0]
                            properties['resolution_y'] = src.res[1]
                        
                        return bbox, geometry, properties
                
                item_bbox, item_geometry, raster_props = retry_rasterio_operation(read_raster_metadata)
                item_properties.update(raster_props)
                
                # Try to extract state from the key path
                key_parts = key.split('/')
                if len(key_parts) > 1:
                    item_properties['state'] = key_parts[0]
                    
                # Extract file info
                item_properties['file_size'] = obj.get('Size', 0)
                item_properties['last_modified'] = obj.get('LastModified', '').isoformat() if obj.get('LastModified') else ''
                        
            except Exception as e:
                logging.error(f"Could not read metadata from {asset_s3_uri}. Error: {e}")
                error_count += 1
                
                # Create minimal properties for failed reads
                key_parts = key.split('/')
                if len(key_parts) > 1:
                    item_properties['state'] = key_parts[0]
                item_properties['metadata_error'] = str(e)
                item_properties['file_size'] = obj.get('Size', 0)

            # Validate required STAC item fields
            if item_bbox and len(item_bbox) != 4:
                logging.warning(f"Invalid bbox for {key}: {item_bbox}. Setting to None.")
                item_bbox = None
                item_geometry = None

            # 4️⃣ Build a STAC Item
            try:
                item = pystac.Item(
                    id=item_id,
                    geometry=item_geometry, # Use extracted geometry
                    bbox=item_bbox, # Use extracted bbox
                    datetime=item_datetime,
                    properties=item_properties # Add any other extracted properties
                )
                
                asset = pystac.Asset(
                    href=public_href,  # Use the already constructed public_href
                    media_type=pystac.MediaType.COG,
                    roles=["data"]
                )
                item.add_asset('cog', asset)
                
                # Validate item before adding to catalog
                item.validate()
                catalog.add_item(item)
                item_count += 1
                
                if item_count % 10 == 0:  # Progress update every 10 items
                    logging.info(f"Processed {item_count} items so far...")
                    
            except Exception as e:
                logging.error(f"Failed to create STAC item for {key}: {e}")
                error_count += 1

except KeyboardInterrupt:
    logging.info("Process interrupted by user")
    sys.exit(0)
except Exception as e:
    logging.error(f"Error accessing bucket '{BUCKET}': {e}")
    raise

# 5️⃣ Persist the catalog to ./stac_catalog/
try:
    total_items = len(list(catalog.get_items()))
    logging.info(f"Processing complete. Successfully created {item_count} items, {error_count} errors")
    
    if total_items == 0:
        logging.warning("No items were added to the catalog. Catalog will still be saved but will be empty.")
    
    logging.info(f"Saving catalog with {total_items} items to {output_dir}")
    catalog.normalize_and_save(root_href=str(output_dir), catalog_type=pystac.CatalogType.SELF_CONTAINED)
    
    # Validate the saved catalog
    saved_catalog = pystac.Catalog.from_file(output_dir / 'catalog.json')
    saved_items = len(list(saved_catalog.get_items()))
    
    if saved_items != total_items:
        logging.error(f"Catalog validation failed: Expected {total_items} items, found {saved_items}")
        sys.exit(1)
    
    logging.info("STAC catalog saved and validated successfully!")
    logging.info(f"Summary: {item_count} items processed, {error_count} errors, {saved_items} items in final catalog")
    
except Exception as e:
    logging.error(f"Failed to save catalog: {e}")
    sys.exit(1)
