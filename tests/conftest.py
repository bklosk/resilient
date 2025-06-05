"""
Pytest configuration and fixtures for the photogrammetry test suite.
"""

import pytest
import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import types

# Add the services directory to the Python path
REPO_ROOT = Path(__file__).resolve().parents[1]
services_dir = REPO_ROOT / "services"
if str(services_dir) not in sys.path:
    sys.path.insert(0, str(services_dir))


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def mock_point_cloud_data():
    """Mock point cloud data for testing."""
    mock_las = Mock()
    mock_las.x = [100.0, 101.0, 102.0]
    mock_las.y = [200.0, 201.0, 202.0]
    mock_las.z = [300.0, 301.0, 302.0]
    mock_las.red = [255, 128, 64]
    mock_las.green = [255, 128, 64]
    mock_las.blue = [255, 128, 64]

    # Mock header
    mock_header = Mock()
    mock_header.epsg = 26913
    mock_header.parse_crs = Mock(
        return_value=Mock(to_string=Mock(return_value="EPSG:26913"))
    )
    mock_las.header = mock_header

    return mock_las


@pytest.fixture
def mock_orthophoto_data():
    """Mock orthophoto data for testing."""
    mock_raster = Mock()
    mock_raster.read = Mock(
        return_value=[[255, 128, 64], [64, 128, 255], [128, 255, 64]]
    )
    mock_raster.width = 100
    mock_raster.height = 100
    mock_raster.crs = "EPSG:4326"
    mock_raster.bounds = Mock(left=-105.1, bottom=39.9, right=-104.9, top=40.1)
    mock_raster.transform = Mock()
    return mock_raster


@pytest.fixture
def mock_geocoding_result():
    """Mock geocoding result."""
    return 40.0274, -105.2519


@pytest.fixture
def sample_bbox():
    """Sample bounding box for testing."""
    return "-105.1,39.9,-104.9,40.1"


@pytest.fixture
def sample_address():
    """Sample address for testing."""
    return "1250 Wildwood Road, Boulder, CO"


@pytest.fixture
def mock_dependencies():
    """Mock heavy dependencies for fast testing."""
    with patch.dict(
        "sys.modules",
        {
            "laspy": Mock(),
            "rasterio": Mock(),
            "pyproj": Mock(),
            "pyvista": Mock(),
            "open3d": Mock(),
        },
    ):
        yield
