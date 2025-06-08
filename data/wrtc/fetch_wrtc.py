#!/usr/bin/env python3
"""
Script to download USDA RDS-2020-0016-2 state zip files, extract TIFFs,
convert them to Cloud-Optimized GeoTIFFs (COGs), and upload to DigitalOcean Spaces.
Requirements:
    - Python 3.x
    - requests
    - boto3
    - GDAL command-line tools (gdal_translate, gdaladdo)
    - Fill in environment variables: DO_SPACES_KEY, DO_SPACES_SECRET, DO_SPACES_REGION, DO_SPACES_ENDPOINT, DO_SPACES_BUCKET
"""
import os
import requests
import zipfile
import subprocess
import boto3
import datetime
import json
import shutil

# List of state identifiers matching USDA zip filenames
state_names = [
    'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado',
    'Connecticut', 'Delaware', 'DistrictOfColumbia', 'Florida', 'Georgia',
    'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky',
    'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan',
    'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska',
    'Nevada', 'NewHampshire', 'NewJersey', 'NewMexico', 'NewYork',
    'NorthCarolina', 'NorthDakota', 'Ohio', 'Oklahoma', 'Oregon',
    'Pennsylvania', 'RhodeIsland', 'SouthCarolina', 'SouthDakota',
    'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington',
    'WestVirginia', 'Wisconsin', 'Wyoming'
]

DOWNLOAD_DIR = "downloads"
COG_DIR = "cogs"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(COG_DIR, exist_ok=True)

# Global variable to track total data uploaded
total_bytes_uploaded = 0
upload_log = []  # Track individual uploads for detailed logging

def get_s3_client():
    """Initialize and return DigitalOcean Spaces client"""
    return boto3.session.Session().client(
        's3',
        region_name=os.environ['DO_SPACES_REGION'],
        endpoint_url=os.environ['DO_SPACES_ENDPOINT'],
        aws_access_key_id=os.environ['DO_SPACES_KEY'],
        aws_secret_access_key=os.environ['DO_SPACES_SECRET']
    )

def get_bucket_name():
    """Get the bucket name from environment variables"""
    return os.environ['DO_SPACES_BUCKET']

def format_bytes(bytes):
    """Convert bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes < 1024.0:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.2f} PB"

def check_disk_space():
    """Check available disk space and warn if low"""
    try:
        statvfs = os.statvfs('/workspaces')
        free_bytes = statvfs.f_frsize * statvfs.f_bavail
        total_bytes_disk = statvfs.f_frsize * statvfs.f_blocks
        used_bytes = total_bytes_disk - free_bytes
        
        print(f"💾 Disk space: {format_bytes(free_bytes)} free, {format_bytes(used_bytes)} used, {format_bytes(total_bytes_disk)} total")
        
        # Warn if less than 1GB free
        if free_bytes < 1024**3:
            print(f"⚠️  WARNING: Low disk space! Only {format_bytes(free_bytes)} available")
            return False
        return True
    except Exception as e:
        print(f"❌ Error checking disk space: {e}")
        return False

def cleanup_directories():
    """Clean up download and cog directories to free space"""
    print("🧹 Cleaning up temporary directories...")
    
    for dir_path in [DOWNLOAD_DIR, COG_DIR]:
        if os.path.exists(dir_path):
            try:
                shutil.rmtree(dir_path)
                print(f"   Removed {dir_path}")
            except Exception as e:
                print(f"   Error removing {dir_path}: {e}")
    
    # Recreate directories
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    os.makedirs(COG_DIR, exist_ok=True)

def save_upload_log():
    """Save upload statistics to a log file"""
    log_file = "upload_log.txt"
    with open(log_file, 'w') as f:
        f.write(f"WRTC Data Upload Log - {datetime.datetime.now()}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total data uploaded: {format_bytes(total_bytes_uploaded)}\n")
        f.write(f"Number of files uploaded: {len(upload_log)}\n\n")
        f.write("Individual file uploads:\n")
        f.write("-" * 40 + "\n")
        for entry in upload_log:
            f.write(f"{entry['timestamp']}: {entry['file']} ({entry['size_formatted']}) -> {entry['key']}\n")
    print(f"Upload log saved to {log_file}")

def save_progress_log():
    """Save progress to JSON file"""
    progress = {
        'timestamp': datetime.datetime.now().isoformat(),
        'total_bytes_uploaded': total_bytes_uploaded,
        'total_bytes_human': format_bytes(total_bytes_uploaded),
        'upload_log': upload_log
    }
    
    with open('progress_log.json', 'w') as f:
        json.dump(progress, f, indent=2)
    
    print(f"💾 Progress saved to progress_log.json")

def download_zip(state):
    # Dictionary mapping state names to their Box.com download URLs
    state_urls = {
        'Alabama': 'https://usfs-public.box.com/shared/static/xsk49ho8gtr18jqiiqidapw09bjxiiya.zip',
        'Alaska': 'https://usfs-public.box.com/shared/static/jh6l2x2blct82hbtmu4n6dvoe9bz25ap.zip',
        'Arizona': 'https://usfs-public.box.com/shared/static/er1j9w81buzpi6spu2ode6krj57frxts.zip',
        'Arkansas': 'https://usfs-public.box.com/shared/static/pys3hd29a2jdfh9v1yg1f0d2t1iz32ya.zip',
        'California': 'https://usfs-public.box.com/shared/static/et4mghz8sq0kxsk2ag6fpey5uec4f8fn.zip',
        'Colorado': 'https://usfs-public.box.com/shared/static/g3lr7dgb6jivvk2267ure83dgjmd0kha.zip',
        'Connecticut': 'https://usfs-public.box.com/shared/static/j0izm7hbhsgchxc5uq2qsdvtrz1slux3.zip',
        'Delaware': 'https://usfs-public.box.com/shared/static/xqnwmh30f860qvw0behnnycm3ufvaxrz.zip',
        'DistrictOfColumbia': 'https://usfs-public.box.com/shared/static/n1i3mhptsebdff0zlstofbm9ovkh3sk5.zip',
        'Florida': 'https://usfs-public.box.com/shared/static/16e0sqzzf3kbe2zbhvcsyvhpv66izjcv.zip',
        'Georgia': 'https://usfs-public.box.com/shared/static/cdvcvixizr1r4qthuo4pd88x4lx913ki.zip',
        'Hawaii': 'https://usfs-public.box.com/shared/static/fyv3pecykr26juimm2h1th4own1ehdj7.zip',
        'Idaho': 'https://usfs-public.box.com/shared/static/jgmjss1rgjc7pnguc7480544ktvk5ild.zip',
        'Illinois': 'https://usfs-public.box.com/shared/static/b77db50aa9t0lahoqp3tql8zcnbjxu5y.zip',
        'Indiana': 'https://usfs-public.box.com/shared/static/vxeutwkvl6i5etr42valervu45xq0ncq.zip',
        'Iowa': 'https://usfs-public.box.com/shared/static/hup68wj334lfjv2e3opvyxkousn6udvd.zip',
        'Kansas': 'https://usfs-public.box.com/shared/static/pwqp9cjgb8yq3didhzs8kzuypt5p2h86.zip',
        'Kentucky': 'https://usfs-public.box.com/shared/static/lcqp2c5rf3pd2r66sd3irdq14cmvmpps.zip',
        'Louisiana': 'https://usfs-public.box.com/shared/static/tek1bx4vu3mbggxm3oad0242gizpqtbt.zip',
        'Maine': 'https://usfs-public.box.com/shared/static/hfqd6nynsm92xv258mowlc7d9cct0srt.zip',
        'Maryland': 'https://usfs-public.box.com/shared/static/nc9xszu487yxj6zw1u99uo6p2ayae5rf.zip',
        'Massachusetts': 'https://usfs-public.box.com/shared/static/21qnyn2gi0ibavlwbbd8ec88vtx2gvud.zip',
        'Michigan': 'https://usfs-public.box.com/shared/static/zhzeoeeitiw9ik76wl6ec7st5gbizwro.zip',
        'Minnesota': 'https://usfs-public.box.com/shared/static/kyhlaskf5dtev6mtlhk58d64u4xx3sgj.zip',
        'Mississippi': 'https://usfs-public.box.com/shared/static/uls1fjtacudgxndp4oqmp1mzynhhmts1.zip',
        'Missouri': 'https://usfs-public.box.com/shared/static/ayzc2ge33ateswfuy6w9h0lgpwbsy14u.zip',
        'Montana': 'https://usfs-public.box.com/shared/static/w8k4837yxpxbjyhoa2xqd28wa05laa47.zip',
        'Nebraska': 'https://usfs-public.box.com/shared/static/n6ra3sg9trrv6iivm9evtna191dtfu78.zip',
        'Nevada': 'https://usfs-public.box.com/shared/static/qdhjh8u6m91nytawj1r737dbghhnj1ft.zip',
        'NewHampshire': 'https://usfs-public.box.com/shared/static/8y0mrc364fhv3hhq4nh5r54fccsiaamy.zip',
        'NewJersey': 'https://usfs-public.box.com/shared/static/j2q2e01iy87iltullvgwn8pl5s63dcww.zip',
        'NewMexico': 'https://usfs-public.box.com/shared/static/p258b37c0dvyvuwyl7068ckaqvto6gqe.zip',
        'NewYork': 'https://usfs-public.box.com/shared/static/s21gq5g8siisakon8xgu5tb5djesf4el.zip',
        'NorthCarolina': 'https://usfs-public.box.com/shared/static/d80581t1zjol4a1nu3ndq51tg6s0xtwg.zip',
        'NorthDakota': 'https://usfs-public.box.com/shared/static/6xjs8nz2icjyo2dhm93v6nw3nk2nc3q8.zip',
        'Ohio': 'https://usfs-public.box.com/shared/static/u9chfbnaw0ycj0nf4jkxzvh0bqgjkdrw.zip',
        'Oklahoma': 'https://usfs-public.box.com/shared/static/c9skxlqnejzc671m1fk33816puk3up64.zip',
        'Oregon': 'https://usfs-public.box.com/shared/static/bqotzr1k53et252mf6k82gwz1904kbhl.zip',
        'Pennsylvania': 'https://usfs-public.box.com/shared/static/9c8634qc635s7mxkyhu7umlugnpptaqd.zip',
        'RhodeIsland': 'https://usfs-public.box.com/shared/static/k1xq9xxv3zxrerrybybmnoz8e9mcjwr7.zip',
        'SouthCarolina': 'https://usfs-public.box.com/shared/static/anbtryj2ikhm5cvdltejz0qrn85s5w05.zip',
        'SouthDakota': 'https://usfs-public.box.com/shared/static/ehroppi8luaoorqa27fmcur45djhhtcj.zip',
        'Tennessee': 'https://usfs-public.box.com/shared/static/ods2qiz9ab8tqeh1matb58aqpeelan9m.zip',
        'Texas': 'https://usfs-public.box.com/shared/static/8qztn7vbnyguzgtnmbsf367uzc6at4jn.zip',
        'Utah': 'https://usfs-public.box.com/shared/static/vgd5pm9bggnyu5muj9l86vx74srah1s5.zip',
        'Vermont': 'https://usfs-public.box.com/shared/static/i8ccl7czwqxmcr9cb52xme4l1cmvrdx8.zip',
        'Virginia': 'https://usfs-public.box.com/shared/static/e6rfe4pdmdyz3yhccf2m8rzjk2kwonhs.zip',
        'Washington': 'https://usfs-public.box.com/shared/static/39osvpm8f3lpk3vderz7308lnpepiic3.zip',
        'WestVirginia': 'https://usfs-public.box.com/shared/static/tytm9gfqyc7z3epd7do6scrfngcwczi6.zip',
        'Wisconsin': 'https://usfs-public.box.com/shared/static/og3kpewggcko6o4qlhfd46ejqolhviur.zip',
        'Wyoming': 'https://usfs-public.box.com/shared/static/fi6khm73akzg7finmqv2wmyjog8t5nxk.zip'
    }
    
    if state not in state_urls:
        raise ValueError(f"Unknown state: {state}")
    
    zip_name = f"RDS-2020-0016-2_{state}.zip"
    url = state_urls[state]
    dest = os.path.join(DOWNLOAD_DIR, zip_name)
    if os.path.exists(dest):
        print(f"{zip_name} already exists, skipping download")
        return dest
    print(f"Downloading {zip_name}...")
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(dest, 'wb') as f:
        for chunk in r.iter_content(8 * 1024):
            f.write(chunk)
    return dest


def extract_tifs(zip_path):
    print(f"Extracting TIFFs from {zip_path}...")
    extracted = []
    with zipfile.ZipFile(zip_path, 'r') as z:
        for member in z.namelist():
            if member.lower().endswith('.tif'):
                out_path = z.extract(member, DOWNLOAD_DIR)
                extracted.append(out_path)
    return extracted


def convert_to_cog(input_tif):
    fname = os.path.basename(input_tif)
    cog_name = os.path.splitext(fname)[0] + "_cog.tif"
    cog_path = os.path.join(COG_DIR, cog_name)
    if os.path.exists(cog_path):
        print(f"{cog_name} exists, skipping COG conversion")
        return cog_path
    print(f"Converting {fname} to COG...")
    
    # Convert to COG with internal overviews
    subprocess.run([
        "gdal_translate", "-of", "COG",
        "-co", "COMPRESS=DEFLATE", "-co", "BIGTIFF=YES",
        "-co", "OVERVIEW_RESAMPLING=AVERAGE",
        input_tif, cog_path
    ], check=True)
    
    # COG format already includes internal overviews, so we don't need gdaladdo
    return cog_path


def upload_to_spaces(cog_path, state):
    global total_bytes_uploaded, upload_log
    
    # Get S3 client and bucket
    s3 = get_s3_client()
    bucket = get_bucket_name()
    
    # Get file size before upload
    file_size = os.path.getsize(cog_path)
    
    key = f"{state}/{os.path.basename(cog_path)}"
    
    # Check if file already exists in Spaces
    try:
        s3.head_object(Bucket=bucket, Key=key)
        print(f"File {key} already exists in Spaces, skipping upload")
        return
    except Exception as e:
        # Handle both ClientError and other S3 exceptions
        if hasattr(e, 'response') and e.response.get('Error', {}).get('Code') == '404':
            pass  # File doesn't exist, continue with upload
        else:
            print(f"Error checking if file exists: {e}")
            # Continue with upload anyway - better to try than fail completely
    
    print(f"Uploading {cog_path} ({format_bytes(file_size)}) to {bucket}/{key}...")
    
    try:
        s3.upload_file(cog_path, bucket, key)
        
        # Update total bytes uploaded and log
        total_bytes_uploaded += file_size
        upload_log.append({
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'file': os.path.basename(cog_path),
            'size_bytes': file_size,
            'size_formatted': format_bytes(file_size),
            'key': key,
            'state': state
        })
        
        print(f"Upload complete. Total uploaded so far: {format_bytes(total_bytes_uploaded)}")
        
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to upload {cog_path} to {bucket}/{key}: {e}")
        print(f"This could indicate bucket permissions issues or network problems.")
        print(f"Continuing with next file...")
        # Don't raise - continue processing other files

def process_state(state):
    """Process a single state with storage-efficient cleanup"""
    print(f"\n{'='*60}")
    print(f"🏛️  Processing {state}")
    print(f"{'='*60}")
    
    try:
        # Clean up before processing this state
        cleanup_directories()
        
        # Check disk space before starting
        if not check_disk_space():
            print(f"⚠️  Skipping {state} due to insufficient disk space")
            return
        
        # Download the ZIP file
        zip_path = download_zip(state)
        if not zip_path:
            print(f"❌ Failed to download {state}")
            return
            
        # Extract TIFF files
        tifs = extract_tifs(zip_path)
        print(f"📊 Found {len(tifs)} TIFF files to process")
        
        if not tifs:
            print(f"⚠️  No TIFF files found in {state}, skipping...")
            return
        
        # Remove ZIP file immediately after extraction to save space
        os.remove(zip_path)
        print(f"   🗑️  Removed ZIP file to save space")
        
        # Process each TIFF file individually
        success_count = 0
        for i, tif in enumerate(tifs, 1):
            print(f"   Processing TIFF {i}/{len(tifs)}: {os.path.basename(tif)}")
            
            try:
                # Convert to COG
                cog = convert_to_cog(tif)
                if cog:
                    # Upload to Spaces
                    upload_to_spaces(cog, state)
                    success_count += 1
                    
                    # Remove COG file immediately after upload
                    os.remove(cog)
                
                # Remove original TIFF file
                os.remove(tif)
                
            except Exception as file_error:
                print(f"❌ Error processing {os.path.basename(tif)}: {file_error}")
                print(f"   Continuing with next file...")
                continue
                
        print(f"✅ Completed processing {state}: {success_count}/{len(tifs)} files uploaded")
        
        # Save progress after each state
        save_progress_log()
        
    except Exception as e:
        print(f"❌ Error processing {state}: {e}")
        print(f"   Continuing with next state...")
    
    finally:
        # Clean up after this state
        cleanup_directories()

def validate_spaces_setup():
    """Validate DigitalOcean Spaces connection and bucket access before starting"""
    try:
        s3 = get_s3_client()
        bucket = get_bucket_name()
        
        print(f"Validating DigitalOcean Spaces setup...")
        print(f"  Bucket: {bucket}")
        print(f"  Endpoint: {os.environ['DO_SPACES_ENDPOINT']}")
        print(f"  Region: {os.environ['DO_SPACES_REGION']}")
        
        # Test bucket access
        try:
            s3.head_bucket(Bucket=bucket)
            print("✓ Bucket exists and is accessible")
            return True
        except Exception as e:
            print(f"✗ CRITICAL: Bucket access failed: {e}")
            print(f"  Make sure the bucket '{bucket}' exists in your DigitalOcean Spaces")
            print(f"  and your credentials have proper permissions.")
            return False
            
    except Exception as e:
        print(f"✗ CRITICAL: Failed to connect to DigitalOcean Spaces: {e}")
        print("  Check your environment variables and credentials.")
        return False


def main():
    global total_bytes_uploaded
    
    start_time = datetime.datetime.now()
    print("🚀 Starting storage-efficient WRTC data processing and upload to DigitalOcean Spaces...")
    print(f"📅 Start time: {start_time}")
    print(f"📊 Total states to process: {len(state_names)}")
    
    # Validate Spaces setup before starting
    if not validate_spaces_setup():
        print("\n❌ SETUP VALIDATION FAILED!")
        print("Please fix the DigitalOcean Spaces configuration before running.")
        print("The script will NOT process terabytes of data without valid upload destination.")
        return
    
    print(f"\n✅ Setup validation passed! Processing {len(state_names)} states")
    
    # Initial cleanup to start with fresh directories
    cleanup_directories()
    check_disk_space()
    
    print("-" * 60)
    
    # Process each state
    for i, s in enumerate(state_names, 1):
        print(f"\n📍 State {i}/{len(state_names)}: {s}")
        process_state(s)
        
        # Check disk space after each state
        if not check_disk_space():
            print("⚠️  Low disk space detected - but continuing with cleanup between states")
    
    end_time = datetime.datetime.now()
    duration = end_time - start_time
    
    print("-" * 60)
    print(f"🏁 Processing complete!")
    print(f"📅 Start time: {start_time}")
    print(f"📅 End time: {end_time}")
    print(f"⏱️  Total duration: {duration}")
    print(f"📁 Total files uploaded: {len(upload_log)}")
    print(f"📊 Total data uploaded to DigitalOcean Spaces: {format_bytes(total_bytes_uploaded)}")
    
    # Save detailed logs
    save_upload_log()
    save_progress_log()

if __name__ == "__main__":
    main()
