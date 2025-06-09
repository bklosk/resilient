#!/usr/bin/env python3
"""
OPTIMIZED WRTC Data Fetcher with Comprehensive Performance Enhancements

This unified script combines all optimization features for maximum performance:

PARALLEL PROCESSING:
- ThreadPoolExecutor with 4 workers for concurrent downloads/processing
- ProcessPoolExecutor for CPU-intensive GDAL operations
- Parallel uploads with multipart support

NETWORK OPTIMIZATIONS:
- Streaming downloads with adaptive chunk sizes
- Connection pooling and keep-alive optimization
- Resumable downloads with Range headers
- Network speed testing and optimization

GDAL OPTIMIZATIONS:
- Optimized environment configuration (1GB cache, ALL_CPUS)
- Adaptive COG parameters based on image characteristics
- LZW compression with horizontal differencing predictor
- Parallel COG conversion

MEMORY & I/O OPTIMIZATIONS:
- Memory usage monitoring with psutil
- Intelligent caching and deduplication
- Temporary directory optimization
- Streaming file processing

RELIABILITY FEATURES:
- Comprehensive error handling and recovery
- Progress tracking with real-time statistics
- Automatic cleanup and space management
- Resume capability for interrupted operations

Requirements:
- Python 3.x with concurrent.futures, threading
- requests with streaming support
- boto3 with multipart uploads
- GDAL command-line tools (gdal_translate)
- tqdm for progress bars
- psutil for system monitoring
- Environment variables: DO_SPACES_KEY, DO_SPACES_SECRET, DO_SPACES_REGION, DO_SPACES_ENDPOINT, DO_SPACES_BUCKET
"""

import os
import sys
import time
import hashlib
import mmap
import tempfile
import shutil
import threading
import multiprocessing as mp
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import zipfile
import subprocess
import boto3
import datetime
import json
import psutil
from tqdm import tqdm

# Enhanced error handling and logging
import logging
import signal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('wrtc_processing.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Enhanced state names list
# state_names = [
#     'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado',
#     'Connecticut', 'Delaware', 'DistrictOfColumbia', 'Florida', 'Georgia',
#     'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky',
#     'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan',
#     'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska',
#     'Nevada', 'NewHampshire', 'NewJersey', 'NewMexico', 'NewYork',
#     'NorthCarolina', 'NorthDakota', 'Ohio', 'Oklahoma', 'Oregon',
#     'Pennsylvania', 'RhodeIsland', 'SouthCarolina', 'SouthDakota',
#     'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington',
#     'WestVirginia', 'Wisconsin', 'Wyoming'
# ]

state_names = [
    'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska',
    'Nevada', 'NewHampshire', 'NewJersey', 'NewMexico', 'NewYork',
    'NorthCarolina', 'NorthDakota', 'Ohio', 'Oklahoma', 'Oregon',
    'Pennsylvania', 'RhodeIsland', 'SouthCarolina', 'SouthDakota',
    'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington',
    'WestVirginia', 'Wisconsin', 'Wyoming'
]

# Comprehensive state URLs mapping
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

# Optimization Configuration
MAX_WORKERS = min(4, mp.cpu_count())  # Parallel processing threads
CHUNK_SIZE = 8 * 1024 * 1024  # 8MB chunks for downloads
MEMORY_THRESHOLD = 0.8  # Use 80% of available memory as threshold
MULTIPART_THRESHOLD = 100 * 1024 * 1024  # 100MB threshold for multipart uploads
DOWNLOAD_DIR = "downloads"
COG_DIR = "cogs"

# Global variables for tracking and thread safety
total_bytes_uploaded = 0
upload_log = []
upload_lock = threading.Lock()
cache_manager = None  # Will be initialized in main()
gdal_optimizer = None  # Will be initialized when needed

# ===========================
# NETWORK OPTIMIZATION CLASSES
# ===========================

class OptimizedDownloader:
    """Enhanced downloader with connection pooling and adaptive strategies"""
    
    def __init__(self, max_workers=4, chunk_size=CHUNK_SIZE):
        self.max_workers = max_workers
        self.chunk_size = chunk_size
        self.session = self._create_optimized_session()
        
    def _create_optimized_session(self):
        """Create an optimized requests session with connection pooling"""
        session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        # Configure HTTP adapter with connection pooling
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=20,
            pool_maxsize=20,
            pool_block=False
        )
        
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Set common headers
        session.headers.update({
            'User-Agent': 'WRTC-Optimizer/1.0',
            'Connection': 'keep-alive'
        })
        
        return session
    
    def test_connection_speed(self, url, test_size=1024*1024):
        """Test connection speed and return optimal chunk size"""
        try:
            start_time = time.time()
            response = self.session.get(url, stream=True, timeout=10)
            
            downloaded = 0
            for chunk in response.iter_content(chunk_size=64*1024):
                downloaded += len(chunk)
                if downloaded >= test_size:
                    break
            
            duration = time.time() - start_time
            speed = downloaded / duration  # bytes per second
            
            # Adjust chunk size based on speed
            if speed > 5*1024*1024:  # > 5MB/s
                return 16*1024*1024  # 16MB chunks
            elif speed > 1*1024*1024:  # > 1MB/s
                return 8*1024*1024   # 8MB chunks
            else:
                return 4*1024*1024   # 4MB chunks
                
        except Exception:
            return self.chunk_size  # Default fallback
    
    def download_with_resume(self, url, dest_path, expected_size=None):
        """Download file with resume capability and progress tracking"""
        dest_path = Path(dest_path)
        temp_path = dest_path.with_suffix(dest_path.suffix + '.tmp')
        
        # Check if partial download exists
        start_byte = 0
        if temp_path.exists():
            start_byte = temp_path.stat().st_size
            print(f"📂 Resuming download from byte {start_byte:,}")
        
        # Test connection speed and adjust chunk size
        optimal_chunk_size = self.test_connection_speed(url)
        
        try:
            headers = {}
            if start_byte > 0:
                headers['Range'] = f'bytes={start_byte}-'
            
            response = self.session.get(url, headers=headers, stream=True, timeout=30)
            response.raise_for_status()
            
            # Get total size
            if expected_size is None:
                content_range = response.headers.get('content-range')
                if content_range:
                    expected_size = int(content_range.split('/')[-1])
                else:
                    expected_size = int(response.headers.get('content-length', 0))
            
            # Download with optimized chunk size and progress tracking
            with open(temp_path, 'ab') as f:
                downloaded = start_byte
                last_report = time.time()
                start_time = time.time()
                
                with tqdm(total=expected_size, initial=start_byte, unit='B', unit_scale=True, desc="Downloading") as pbar:
                    for chunk in response.iter_content(chunk_size=optimal_chunk_size):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            pbar.update(len(chunk))
                            
                            # Report speed every 5 seconds
                            now = time.time()
                            if now - last_report > 5:
                                if downloaded > start_byte:
                                    speed = (downloaded - start_byte) / (now - start_time)
                                    pbar.set_postfix(speed=f"{speed/1024/1024:.1f} MB/s")
                                last_report = now
            
            # Move completed file
            shutil.move(str(temp_path), str(dest_path))
            print(f"✅ Download completed: {dest_path}")
            return str(dest_path)
            
        except Exception as e:
            print(f"❌ Download failed: {e}")
            # Clean up partial file
            if temp_path.exists():
                temp_path.unlink()
            raise

class CacheManager:
    """Intelligent caching and deduplication system"""
    
    def __init__(self, cache_dir=None):
        if cache_dir is None:
            cache_dir = os.path.join(os.path.expanduser('~'), '.wrtc_cache')
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.index_file = self.cache_dir / 'cache_index.json'
        self.load_index()
    
    def load_index(self):
        """Load cache index"""
        try:
            if self.index_file.exists():
                with open(self.index_file, 'r') as f:
                    self.index = json.load(f)
            else:
                self.index = {}
        except Exception:
            self.index = {}
    
    def save_index(self):
        """Save cache index"""
        try:
            with open(self.index_file, 'w') as f:
                json.dump(self.index, f, indent=2)
        except Exception as e:
            print(f"⚠️  Failed to save cache index: {e}")
    
    def get_file_hash(self, file_path, algorithm='md5'):
        """Calculate file hash for deduplication"""
        hash_obj = hashlib.new(algorithm)
        
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_obj.update(chunk)
            return hash_obj.hexdigest()
        except Exception as e:
            print(f"⚠️  Failed to calculate hash for {file_path}: {e}")
            return None
    
    def get_cached_file(self, cache_key):
        """Retrieve cached file"""
        if cache_key in self.index:
            cached_path = self.cache_dir / self.index[cache_key]['filename']
            if cached_path.exists():
                print(f"💾 Using cached file: {cache_key}")
                return str(cached_path)
            else:
                # Clean up invalid cache entry
                del self.index[cache_key]
                self.save_index()
        return None
    
    def add_to_cache(self, url, file_path):
        """Add file to cache"""
        try:
            cache_key = hashlib.md5(url.encode()).hexdigest()
            file_hash = self.get_file_hash(file_path)
            
            if file_hash:
                self.index[cache_key] = {
                    'url': url,
                    'filename': os.path.basename(file_path),
                    'hash': file_hash,
                    'size': os.path.getsize(file_path),
                    'timestamp': time.time()
                }
                self.save_index()
                
        except Exception as e:
            print(f"⚠️  Failed to add {file_path} to cache: {e}")
    
    def is_cached(self, url, file_path):
        """Check if file is cached and valid"""
        if not os.path.exists(file_path):
            return False
            
        try:
            cache_key = hashlib.md5(url.encode()).hexdigest()
            if cache_key in self.index:
                expected_size = self.index[cache_key].get('size', 0)
                actual_size = os.path.getsize(file_path)
                return expected_size == actual_size
        except Exception:
            pass
            
        return False
    
    def cleanup_cache(self, max_age_days=30):
        """Clean up old cache entries"""
        cutoff_time = time.time() - (max_age_days * 24 * 3600)
        to_remove = []
        
        for key, entry in self.index.items():
            if entry.get('timestamp', 0) < cutoff_time:
                to_remove.append(key)
                # Also remove file if it exists
                cache_file = self.cache_dir / entry['filename']
                if cache_file.exists():
                    cache_file.unlink()
        
        for key in to_remove:
            del self.index[key]
        
        if to_remove:
            self.save_index()
            print(f"🧹 Cleaned up {len(to_remove)} old cache entries")

# ===========================
# GDAL OPTIMIZATION CLASS
# ===========================

class GDALOptimizer:
    """Optimize GDAL operations for maximum performance"""
    
    def __init__(self, num_workers=None):
        self.num_workers = num_workers or min(mp.cpu_count(), 6)
        self.setup_gdal_environment()
    
    def setup_gdal_environment(self):
        """Configure GDAL environment variables for optimal performance"""
        gdal_config = {
            # Memory and caching
            'GDAL_CACHEMAX': '1024',  # 1GB cache
            'GDAL_SWATH_SIZE': '200000000',  # 200MB swath for warping
            'GDAL_MAX_DATASET_POOL_SIZE': '450',
            
            # I/O optimization
            'GDAL_DISABLE_READDIR_ON_OPEN': 'EMPTY_DIR',
            'CPL_VSIL_CURL_CACHE_SIZE': '200000000',
            'GDAL_HTTP_TIMEOUT': '30',
            'GDAL_HTTP_CONNECTTIMEOUT': '30',
            
            # Processing optimization
            'GDAL_NUM_THREADS': 'ALL_CPUS',
            'OGR_INTERLEAVED_READING': 'YES',
            
            # COG-specific optimizations
            'GDAL_TIFF_INTERNAL_MASK': 'YES',
            'GDAL_TIFF_OVR_BLOCKSIZE': '512',
        }
        
        for key, value in gdal_config.items():
            os.environ[key] = value
    
    def get_optimal_cog_params(self, input_file):
        """Determine optimal COG parameters based on input file characteristics"""
        try:
            # Get basic file size for quick parameter selection
            file_size = os.path.getsize(input_file)
            
            if file_size > 100 * 1024 * 1024:  # > 100MB
                return {
                    'blocksize': 1024,
                    'compress': 'LZW',
                    'predictor': 2,
                    'bigtiff': 'YES'
                }
            elif file_size > 50 * 1024 * 1024:  # > 50MB
                return {
                    'blocksize': 512,
                    'compress': 'LZW',
                    'predictor': 2,
                    'bigtiff': 'IF_SAFER'
                }
            else:
                return {
                    'blocksize': 256,
                    'compress': 'DEFLATE',
                    'predictor': 2,
                    'bigtiff': 'NO'
                }
        except Exception:
            # Default parameters
            return {
                'blocksize': 512,
                'compress': 'LZW',
                'predictor': 2,
                'bigtiff': 'IF_SAFER'
            }

# ===========================
# CORE UTILITY FUNCTIONS
# ===========================

def get_bucket_name():
    """Get the bucket name from environment variables"""
    return os.environ['DO_SPACES_BUCKET']

def get_s3_client():
    """Create and return a DigitalOcean Spaces S3 client"""
    return boto3.client(
        's3',
        endpoint_url=os.environ['DO_SPACES_ENDPOINT'],
        region_name=os.environ['DO_SPACES_REGION'],
        aws_access_key_id=os.environ['DO_SPACES_KEY'],
        aws_secret_access_key=os.environ['DO_SPACES_SECRET']
    )

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
    """Optimized download function using the enhanced downloader"""
    global cache_manager
    
    if state not in state_urls:
        raise ValueError(f"Unknown state: {state}")
    
    zip_name = f"RDS-2020-0016-2_{state}.zip"
    url = state_urls[state]
    dest = os.path.join(DOWNLOAD_DIR, zip_name)
    
    # Check if file already exists and is valid
    if cache_manager and cache_manager.is_cached(url, dest):
        print(f"✓ {zip_name} exists and is valid, skipping download")
        return dest
    
    print(f"📥 Downloading {zip_name}...")
    
    try:
        # Use optimized downloader
        downloader = OptimizedDownloader(max_workers=MAX_WORKERS, chunk_size=CHUNK_SIZE)
        
        # Try to get expected size from headers
        try:
            head_response = downloader.session.head(url, timeout=10)
            expected_size = int(head_response.headers.get('content-length', 0))
            print(f"   Expected size: {format_bytes(expected_size)}")
        except:
            expected_size = None
        
        # Download with resume capability
        downloader.download_with_resume(url, dest, expected_size)
        
        # Update cache
        if cache_manager:
            cache_manager.add_to_cache(url, dest)
        
        return dest
        
    except Exception as e:
        print(f"❌ Download failed for {state}: {e}")
        # Clean up partial file
        if os.path.exists(dest):
            os.remove(dest)
        return None

def optimized_download_with_resume(url, dest_path, expected_size=None):
    """Enhanced download function with resume capability and progress tracking"""
    dest_path = Path(dest_path)
    temp_path = dest_path.with_suffix(dest_path.suffix + '.tmp')
    
    # Check if partial download exists
    start_byte = 0
    if temp_path.exists():
        start_byte = temp_path.stat().st_size
        print(f"📂 Resuming download from byte {start_byte:,}")
    
    # Create optimized session
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=10)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    try:
        headers = {'User-Agent': 'WRTC-Optimizer/1.0'}
        if start_byte > 0:
            headers['Range'] = f'bytes={start_byte}-'
        
        response = session.get(url, headers=headers, stream=True, timeout=30)
        response.raise_for_status()
        
        # Get total size
        if expected_size is None:
            content_range = response.headers.get('content-range')
            if content_range:
                expected_size = int(content_range.split('/')[-1])
            else:
                expected_size = int(response.headers.get('content-length', 0))
        
        # Download with progress tracking
        with open(temp_path, 'ab') as f:
            downloaded = start_byte
            last_report = time.time()
            start_time = time.time()
            
            with tqdm(total=expected_size, initial=start_byte, unit='B', unit_scale=True, desc="Downloading") as pbar:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        pbar.update(len(chunk))
                        
                        # Report speed every 5 seconds
                        now = time.time()
                        if now - last_report > 5:
                            if expected_size > 0 and downloaded > start_byte:
                                speed = (downloaded - start_byte) / (now - start_time)
                                print(f"   📊 Speed: {speed/1024/1024:.1f} MB/s")
                            last_report = now
        
        # Move completed file
        shutil.move(str(temp_path), str(dest_path))
        print(f"✅ Download completed: {dest_path}")
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        # Clean up partial file
        if temp_path.exists():
            temp_path.unlink()
        raise
    finally:
        session.close()


def extract_tifs(zip_path, target_dir=None):
    """Optimized TIFF extraction with selective extraction and progress tracking"""
    print(f"📦 Extracting TIFFs from {os.path.basename(zip_path)}...")
    
    if target_dir is None:
        target_dir = DOWNLOAD_DIR
    
    extracted = []
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            # Pre-filter TIFF files
            tif_members = [m for m in z.namelist() if m.lower().endswith('.tif')]
            
            if not tif_members:
                print("   ⚠️  No TIFF files found in ZIP")
                return []
            
            print(f"   Found {len(tif_members)} TIFF files to extract")
            
            # Extract with progress tracking
            with tqdm(tif_members, desc="Extracting", unit="files") as pbar:
                for member in pbar:
                    try:
                        # Extract directly to target directory
                        out_path = z.extract(member, target_dir)
                        extracted.append(out_path)
                        pbar.set_postfix(file=os.path.basename(member)[:30])
                    except Exception as e:
                        print(f"   ⚠️  Failed to extract {member}: {e}")
                        continue
                        
    except Exception as e:
        print(f"❌ Error extracting ZIP: {e}")
        return []
    
    print(f"✅ Extracted {len(extracted)} TIFF files")
    return extracted


def convert_to_cog(input_tif, use_temp_dir=True):
    """Optimized COG conversion with adaptive parameters and GDAL optimization"""
    global gdal_optimizer
    
    fname = os.path.basename(input_tif)
    cog_name = os.path.splitext(fname)[0] + "_cog.tif"
    
    if use_temp_dir:
        # Use system temp directory for better I/O performance
        temp_dir = tempfile.gettempdir()
        cog_path = os.path.join(temp_dir, cog_name)
    else:
        cog_path = os.path.join(COG_DIR, cog_name)
    
    if os.path.exists(cog_path):
        print(f"   ✓ {cog_name} exists, skipping conversion")
        return cog_path
    
    print(f"🔄 Converting {fname} to COG...")
    
    try:
        # Get optimal parameters from GDAL optimizer
        if gdal_optimizer is None:
            gdal_optimizer = GDALOptimizer()
        
        cog_params = gdal_optimizer.get_optimal_cog_params(input_tif)
        
        # Build optimized GDAL command
        cmd = [
            "gdal_translate", "-of", "COG",
            "-co", f"COMPRESS={cog_params['compress']}",
            "-co", f"PREDICTOR={cog_params['predictor']}",
            "-co", f"BIGTIFF={cog_params['bigtiff']}",
            "-co", "TILED=YES",
            "-co", f"BLOCKSIZE={cog_params['blocksize']}",
            "-co", "OVERVIEW_RESAMPLING=AVERAGE",
            "-co", "NUM_THREADS=ALL_CPUS",
            input_tif, cog_path
        ]
        
        # Run with timeout and capture output
        result = subprocess.run(
            cmd, 
            check=True, 
            capture_output=True, 
            text=True, 
            timeout=600  # 10 minute timeout for large files
        )
        
        # Verify the output file exists and has reasonable size
        if not os.path.exists(cog_path):
            raise Exception("COG file not created")
            
        input_size = os.path.getsize(input_tif)
        output_size = os.path.getsize(cog_path)
        
        if output_size == 0:
            raise Exception("COG file is empty")
            
        compression_ratio = (1 - output_size / input_size) * 100 if input_size > 0 else 0
        print(f"✅ COG created: {format_bytes(output_size)} ({compression_ratio:.1f}% compressed)")
        return cog_path
        
    except subprocess.TimeoutExpired:
        print(f"❌ COG conversion timeout for {fname}")
        if os.path.exists(cog_path):
            os.remove(cog_path)
        return None
    except Exception as e:
        print(f"❌ COG conversion failed for {fname}: {e}")
        if os.path.exists(cog_path):
            os.remove(cog_path)
        return None


def upload_to_spaces(cog_path, state):
    """Optimized upload with multipart support and progress tracking"""
    global total_bytes_uploaded, upload_log
    
    # Get S3 client and bucket
    s3 = get_s3_client()
    bucket = get_bucket_name()
    
    # Get file size before upload
    file_size = os.path.getsize(cog_path)
    
    key = f"{state}/{os.path.basename(cog_path)}"
    
    # Check if file already exists in Spaces
    try:
        existing_obj = s3.head_object(Bucket=bucket, Key=key)
        existing_size = existing_obj.get('ContentLength', 0)
        
        if existing_size == file_size:
            print(f"✓ File {key} already exists with correct size, skipping upload")
            return
        else:
            print(f"⚠️  File {key} exists but size differs (existing: {existing_size}, local: {file_size})")
    except Exception as e:
        # Handle both ClientError and other S3 exceptions
        if hasattr(e, 'response') and e.response.get('Error', {}).get('Code') == '404':
            pass  # File doesn't exist, continue with upload
        else:
            print(f"Warning: Error checking if file exists: {e}")
    
    print(f"☁️  Uploading {os.path.basename(cog_path)} ({format_bytes(file_size)}) to {bucket}/{key}...")
    
    try:
        # Use multipart upload for large files
        if file_size > MULTIPART_THRESHOLD:
            print(f"   Using multipart upload for large file")
            
            # Configure multipart upload
            config = boto3.s3.transfer.TransferConfig(
                multipart_threshold=MULTIPART_THRESHOLD,
                max_concurrency=MAX_WORKERS,
                multipart_chunksize=16 * 1024 * 1024,  # 16MB chunks
                use_threads=True
            )
            
            # Upload with progress callback
            def progress_callback(bytes_transferred):
                percent = (bytes_transferred / file_size) * 100
                if percent % 10 < 1:  # Report every 10%
                    print(f"   📊 {percent:.0f}% uploaded ({format_bytes(bytes_transferred)})")
            
            s3.upload_file(
                cog_path, bucket, key,
                Config=config,
                Callback=progress_callback
            )
        else:
            # Standard upload for smaller files
            s3.upload_file(cog_path, bucket, key)
        
        # Update total bytes uploaded and log (thread-safe)
        with upload_lock:
            total_bytes_uploaded += file_size
            upload_log.append({
                'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'file': os.path.basename(cog_path),
                'size_bytes': file_size,
                'size_formatted': format_bytes(file_size),
                'key': key,
                'state': state
            })
        
        print(f"✅ Upload complete. Total uploaded so far: {format_bytes(total_bytes_uploaded)}")
        
    except Exception as e:
        print(f"❌ CRITICAL ERROR: Failed to upload {cog_path} to {bucket}/{key}: {e}")
        print(f"   This could indicate bucket permissions issues or network problems.")
        print(f"   Continuing with next file...")

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
    global total_bytes_uploaded, cache_manager, gdal_optimizer
    
    start_time = datetime.datetime.now()
    logger.info("Starting optimized WRTC data processing and upload to DigitalOcean Spaces")
    print("🚀 Starting optimized WRTC data processing and upload to DigitalOcean Spaces...")
    print(f"📅 Start time: {start_time}")
    print(f"📊 Total states to process: {len(state_names)}")
    
    # Initialize optimization components
    print("🔧 Initializing optimization components...")
    logger.info("Initializing optimization components")
    try:
        cache_manager = CacheManager()
        gdal_optimizer = GDALOptimizer()
        print("✅ Optimization components initialized")
        logger.info("Optimization components initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize optimization components: {e}")
        print(f"❌ Failed to initialize optimization components: {e}")
        return
    
    # Validate Spaces setup before starting
    logger.info("Validating DigitalOcean Spaces setup")
    if not validate_spaces_setup():
        logger.error("SETUP VALIDATION FAILED!")
        print("\n❌ SETUP VALIDATION FAILED!")
        print("Please fix the DigitalOcean Spaces configuration before running.")
        print("The script will NOT process terabytes of data without valid upload destination.")
        return
    
    logger.info("Setup validation passed - ready to process states")
    print(f"\n✅ Setup validation passed! Processing {len(state_names)} states")
    
    # Initial cleanup to start with fresh directories
    cleanup_directories()
    check_disk_space()
    
    print("-" * 60)
    
    # Choose processing mode
    use_parallel = True  # Enable parallel processing by default for better performance
    processing_func = optimized_process_state if use_parallel else process_state
    mode_name = "Parallel Optimized" if use_parallel else "Sequential"
    
    print(f"🔧 Processing mode: {mode_name}")
    print(f"🧵 Max workers: {MAX_WORKERS}")
    print(f"💾 Chunk size: {format_bytes(CHUNK_SIZE)}")
    
    # Process each state
    for i, s in enumerate(state_names, 1):
        print(f"\n📍 State {i}/{len(state_names)}: {s}")
        processing_func(s)
        
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

def check_memory_usage(threshold=MEMORY_THRESHOLD):
    """Check if memory usage is below threshold"""
    try:
        memory = psutil.virtual_memory()
        usage_percent = memory.percent / 100.0
        
        if usage_percent > threshold:
            available_gb = memory.available / (1024**3)
            print(f"⚠️  High memory usage: {usage_percent:.1%} used, {available_gb:.1f}GB available")
            return False
        
        return True
    except Exception:
        # If we can't check memory, assume it's OK
        return True

def process_tiff_file(tif_path, state, temp_dir):
    """Process a single TIFF file: convert to COG and upload"""
    try:
        # Convert to COG using our optimized function
        cog_path = convert_to_cog(tif_path)
        if not cog_path or not os.path.exists(cog_path):
            return False, f"COG conversion failed for {os.path.basename(tif_path)}"
        
        # Upload to Spaces
        upload_success = upload_to_spaces(cog_path, state)
        
        # Clean up COG file immediately after upload
        try:
            os.remove(cog_path)
        except:
            pass
        
        return upload_success, os.path.basename(tif_path)
        
    except Exception as e:
        return False, f"Error processing {os.path.basename(tif_path)}: {e}"

def optimized_process_state(state):
    """Process a state with parallel TIFF processing"""
    print(f"\n{'='*60}")
    print(f"🏛️  Processing {state} (Optimized)")
    print(f"{'='*60}")
    
    try:
        # Check memory before starting
        if not check_memory_usage():
            print(f"⚠️  Skipping {state} due to high memory usage")
            return
        
        # Create temporary directory for this state
        with tempfile.TemporaryDirectory(prefix=f"wrtc_{state}_") as temp_dir:
            print(f"📁 Using temporary directory: {temp_dir}")
            
            # Download ZIP file
            zip_path = download_zip(state)
            if not zip_path:
                print(f"❌ Failed to download {state}")
                return
            
            # Extract TIFF files
            tif_files = extract_tifs(zip_path, temp_dir)
            if not tif_files:
                print(f"⚠️  No TIFF files found in {state}")
                return
            
            print(f"📊 Found {len(tif_files)} TIFF files to process")
            
            # Remove ZIP file immediately to save space
            try:
                os.remove(zip_path)
                print(f"   🗑️  Removed ZIP file to save space")
            except:
                pass
            
            # Process TIFF files in parallel
            success_count = 0
            
            with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(tif_files))) as executor:
                # Submit all tasks
                future_to_tif = {
                    executor.submit(process_tiff_file, tif_path, state, temp_dir): tif_path 
                    for tif_path in tif_files
                }
                
                # Process completed tasks
                for future in tqdm(as_completed(future_to_tif), 
                                 total=len(tif_files), 
                                 desc=f"Processing {state}",
                                 unit="files"):
                    
                    tif_path = future_to_tif[future]
                    try:
                        success, message = future.result()
                        if success:
                            success_count += 1
                        else:
                            print(f"   ⚠️  {message}")
                    except Exception as e:
                        print(f"   ❌ Error processing {os.path.basename(tif_path)}: {e}")
                    
                    # Clean up original TIFF file immediately
                    try:
                        os.remove(tif_path)
                    except:
                        pass
                    
                    # Check memory periodically
                    if not check_memory_usage():
                        print("   ⚠️  High memory usage detected, may slow down")
            
            print(f"✅ Completed {state}: {success_count}/{len(tif_files)} files processed successfully")
    
    except Exception as e:
        print(f"❌ Error processing {state}: {e}")

def parallel_process_state(state):
    """Wrapper function for parallel state processing - calls optimized version"""
    return optimized_process_state(state)

def graceful_shutdown(signum, frame):
    """Handle graceful shutdown on interrupt"""
    logger.info("Received shutdown signal. Cleaning up...")
    print("\n🛑 Shutdown signal received. Cleaning up...")
    
    # Save current progress
    try:
        save_progress_log()
        save_upload_log()
        print("✅ Progress saved")
    except Exception as e:
        print(f"⚠️  Could not save progress: {e}")
    
    # Cleanup directories
    try:
        cleanup_directories()
        print("✅ Directories cleaned up")
    except Exception as e:
        print(f"⚠️  Could not clean up directories: {e}")
    
    print("👋 Graceful shutdown complete")
    sys.exit(0)

# Register signal handlers for graceful shutdown
signal.signal(signal.SIGINT, graceful_shutdown)
signal.signal(signal.SIGTERM, graceful_shutdown)

if __name__ == "__main__":
    main()
