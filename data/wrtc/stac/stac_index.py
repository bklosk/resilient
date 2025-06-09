import os
import sys
import time
from datetime import datetime, timezone
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

# Dictionary to store collections by state
collections = {}

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
            item_datetime = datetime.now(timezone.utc)
            item_properties = {}

            try:
                logging.info(f"Processing ({item_count + 1}): {key}")
                
                def read_raster_metadata():
                    try:
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
                                'dtype': str(src.dtypes[0]) if src.dtypes and len(src.dtypes) > 0 else 'UNKNOWN'
                            }
                            
                            # Add resolution if available
                            if hasattr(src, 'res') and src.res:
                                properties['resolution_x'] = src.res[0]
                                properties['resolution_y'] = src.res[1]
                            
                            return bbox, geometry, properties
                    except Exception as e:
                        logging.error(f"Error in read_raster_metadata for {key}: {e}")
                        raise
                
                item_bbox, item_geometry, raster_props = retry_rasterio_operation(read_raster_metadata)
                item_properties.update(raster_props)
                
                # Try to extract state from the key path
                key_parts = key.split('/')
                state = None
                if len(key_parts) > 1:
                    state = key_parts[0]
                    item_properties['state'] = state
                    
                # Extract file info
                item_properties['file_size'] = obj.get('Size', 0)
                item_properties['last_modified'] = obj.get('LastModified', '').isoformat() if obj.get('LastModified') else ''
                        
            except Exception as e:
                logging.error(f"Could not read metadata from {asset_s3_uri}. Error: {e}")
                error_count += 1
                
                # Create minimal properties for failed reads
                key_parts = key.split('/')
                state = None
                if len(key_parts) > 1:
                    state = key_parts[0]
                    item_properties['state'] = state
                item_properties['metadata_error'] = str(e)
                item_properties['file_size'] = obj.get('Size', 0)

            # Validate required STAC item fields
            if item_bbox and len(item_bbox) != 4:
                logging.warning(f"Invalid bbox for {key}: {item_bbox}. Setting to None.")
                item_bbox = None
                item_geometry = None

            # 4️⃣ Build a STAC Item
            try:
                logging.debug(f"Creating STAC item with bbox: {item_bbox}, geometry: {item_geometry is not None}")
                
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
                
                # Determine which collection this item belongs to
                collection_id = state if state else 'unknown'
                
                # Create collection if it doesn't exist
                if collection_id not in collections:
                    collection_title = f"{state.upper()} COGs" if state else "Unknown State COGs"
                    collection_description = f"Cloud Optimized GeoTIFFs for {state.upper()}" if state else "Cloud Optimized GeoTIFFs for unknown state"
                    
                    collection = pystac.Collection(
                        id=collection_id,
                        description=collection_description,
                        extent=pystac.Extent(
                            spatial=pystac.SpatialExtent(bboxes=[[-180, -90, 180, 90]]),  # Use world extent as placeholder
                            temporal=pystac.TemporalExtent(intervals=[[item_datetime, item_datetime]])  # Use current datetime as placeholder
                        ),
                        title=collection_title
                    )
                    collections[collection_id] = collection
                    catalog.add_child(collection)
                    logging.info(f"Created collection: {collection_id}")
                
                # Add item to the appropriate collection
                collections[collection_id].add_item(item)
                item_count += 1
                
                if item_count % 10 == 0:  # Progress update every 10 items
                    logging.info(f"Processed {item_count} items so far...")
                    
            except Exception as e:
                logging.error(f"Failed to create STAC item for {key}: {e}")
                import traceback
                logging.error(f"Full traceback: {traceback.format_exc()}")
                error_count += 1

except KeyboardInterrupt:
    logging.info("Process interrupted by user")
    sys.exit(0)
except Exception as e:
    logging.error(f"Error accessing bucket '{BUCKET}': {e}")
    raise

# 5️⃣ Update collection extents and persist the catalog to ./stac_catalog/
try:
    # Update extents for each collection
    for collection_id, collection in collections.items():
        items = list(collection.get_items())
        if items:
            # Calculate spatial extent
            all_bboxes = []
            all_datetimes = []
            
            for item in items:
                if item.bbox:
                    all_bboxes.append(item.bbox)
                if item.datetime:
                    all_datetimes.append(item.datetime)
            
            if all_bboxes:
                # Calculate overall bbox
                min_x = min(bbox[0] for bbox in all_bboxes)
                min_y = min(bbox[1] for bbox in all_bboxes)
                max_x = max(bbox[2] for bbox in all_bboxes)
                max_y = max(bbox[3] for bbox in all_bboxes)
                overall_bbox = [min_x, min_y, max_x, max_y]
                
                collection.extent.spatial = pystac.SpatialExtent(bboxes=[overall_bbox])
            
            if all_datetimes:
                # Calculate temporal extent
                min_datetime = min(all_datetimes)
                max_datetime = max(all_datetimes)
                collection.extent.temporal = pystac.TemporalExtent(intervals=[[min_datetime, max_datetime]])
        
        logging.info(f"Collection '{collection_id}' contains {len(items)} items")
    
    total_items = sum(len(list(collection.get_items())) for collection in collections.values())
    logging.info(f"Processing complete. Successfully created {item_count} items across {len(collections)} collections, {error_count} errors")
    
    if total_items == 0:
        logging.warning("No items were added to the catalog. Catalog will still be saved but will be empty.")
    
    logging.info(f"Saving catalog with {total_items} items to {output_dir}")
    catalog.normalize_and_save(root_href=str(output_dir), catalog_type=pystac.CatalogType.SELF_CONTAINED)
    
    # Validate the saved catalog
    saved_catalog = pystac.Catalog.from_file(output_dir / 'catalog.json')
    saved_collections = list(saved_catalog.get_collections())
    saved_items = sum(len(list(collection.get_items())) for collection in saved_collections)
    
    if saved_items != total_items:
        logging.error(f"Catalog validation failed: Expected {total_items} items, found {saved_items}")
        sys.exit(1)
    
    logging.info("STAC catalog saved and validated successfully!")
    logging.info(f"Summary: {item_count} items processed, {error_count} errors, {len(saved_collections)} collections, {saved_items} items in final catalog")
    
    # Log the structure created
    logging.info("Catalog structure:")
    logging.info(f"  📁 catalog.json (root)")
    for collection in saved_collections:
        collection_items = len(list(collection.get_items()))
        logging.info(f"  📁 {collection.id}/")
        logging.info(f"    📄 collection.json ({collection_items} items)")
        if collection_items <= 5:  # Show individual items for small collections
            for item in collection.get_items():
                logging.info(f"    📄 {item.id}/item.json")
        else:
            logging.info(f"    📄 ... {collection_items} item.json files")
    
except Exception as e:
    logging.error(f"Failed to save catalog: {e}")
    sys.exit(1)
