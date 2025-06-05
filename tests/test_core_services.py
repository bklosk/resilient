"""
Tests for core services (geocoding and point cloud processing).
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import types

# Ensure services can be imported
REPO_ROOT = Path(__file__).resolve().parents[1]
services_dir = REPO_ROOT / "services"
if str(services_dir) not in sys.path:
    sys.path.insert(0, str(services_dir))


class TestGeocoder:
    """Test the geocoding service."""

    def test_geocoder_initialization(self):
        """Test geocoder initializes with multiple services."""
        from services.core.geocode import Geocoder

        geocoder = Geocoder()
        assert len(geocoder.geocoders) >= 2
        assert geocoder.user_agent == "photogrammetry_geocoder"

    def test_geocoder_custom_user_agent(self):
        """Test geocoder with custom user agent."""
        from services.core.geocode import Geocoder

        custom_agent = "test_agent"
        geocoder = Geocoder(user_agent=custom_agent)
        assert geocoder.user_agent == custom_agent

    @patch("geopy.geocoders.Photon.geocode")
    def test_successful_geocoding(
        self, mock_geocode, sample_address, mock_geocoding_result
    ):
        """Test successful geocoding with first service."""
        from services.core.geocode import Geocoder

        # Mock successful geocoding result
        mock_location = Mock()
        mock_location.latitude = mock_geocoding_result[0]
        mock_location.longitude = mock_geocoding_result[1]
        mock_geocode.return_value = mock_location

        geocoder = Geocoder()
        lat, lon = geocoder.geocode_address(sample_address)

        assert lat == mock_geocoding_result[0]
        assert lon == mock_geocoding_result[1]
        mock_geocode.assert_called_once_with(sample_address)

    @patch("geopy.geocoders.ArcGIS.geocode")
    @patch("geopy.geocoders.Photon.geocode")
    def test_fallback_geocoding(
        self, mock_photon, mock_arcgis, sample_address, mock_geocoding_result
    ):
        """Test geocoding fallback when first service fails."""
        from services.core.geocode import Geocoder
        from geopy.exc import GeopyError

        # First service fails
        mock_photon.side_effect = GeopyError("Service unavailable")

        # Second service succeeds
        mock_location = Mock()
        mock_location.latitude = mock_geocoding_result[0]
        mock_location.longitude = mock_geocoding_result[1]
        mock_arcgis.return_value = mock_location

        geocoder = Geocoder()
        lat, lon = geocoder.geocode_address(sample_address, max_retries=1)

        assert lat == mock_geocoding_result[0]
        assert lon == mock_geocoding_result[1]
        mock_photon.assert_called_once()
        mock_arcgis.assert_called_once_with(sample_address)

    @patch("geopy.geocoders.ArcGIS.geocode")
    @patch("geopy.geocoders.Photon.geocode")
    def test_geocoding_all_services_fail(
        self, mock_photon, mock_arcgis, sample_address
    ):
        """Test when all geocoding services fail."""
        from services.core.geocode import Geocoder
        from geopy.exc import GeopyError

        # All services fail
        mock_photon.side_effect = GeopyError("Service unavailable")
        mock_arcgis.side_effect = GeopyError("Service unavailable")

        geocoder = Geocoder()

        with pytest.raises(Exception, match="All geocoding services failed"):
            geocoder.geocode_address(sample_address, max_retries=1)

    @patch("geopy.geocoders.ArcGIS.geocode")
    @patch("geopy.geocoders.Photon.geocode")
    def test_geocoding_no_result(self, mock_photon, mock_arcgis, sample_address):
        """Test when geocoding returns no result."""
        from services.core.geocode import Geocoder

        mock_photon.return_value = None
        mock_arcgis.return_value = None

        geocoder = Geocoder()

        with pytest.raises(Exception, match="All geocoding services failed"):
            geocoder.geocode_address(sample_address, max_retries=1)

    @patch("geopy.geocoders.ArcGIS.geocode")
    @patch("geopy.geocoders.Photon.geocode")
    def test_geocoding_empty_address(self, mock_photon, mock_arcgis):
        """Test geocoding with empty address."""
        from services.core.geocode import Geocoder

        # Mock geocoders to return None for empty address
        mock_photon.return_value = None
        mock_arcgis.return_value = None

        geocoder = Geocoder()

        with pytest.raises(Exception, match="All geocoding services failed"):
            geocoder.geocode_address("")

    @patch("geopy.geocoders.ArcGIS.geocode")
    @patch("geopy.geocoders.Photon.geocode")
    def test_geocoding_whitespace_address(self, mock_photon, mock_arcgis):
        """Test geocoding with whitespace-only address."""
        from services.core.geocode import Geocoder

        # Mock geocoders to return None for whitespace address
        mock_photon.return_value = None
        mock_arcgis.return_value = None

        geocoder = Geocoder()

        with pytest.raises(Exception, match="All geocoding services failed"):
            geocoder.geocode_address("   ")


class TestPointCloudProcessor:
    """Test the point cloud processor."""

    def test_processor_initialization(self, temp_dir):
        """Test processor initializes correctly."""
        from services.core.process_point_cloud import PointCloudProcessor

        processor = PointCloudProcessor(output_dir=str(temp_dir))
        assert processor.output_dir == Path(temp_dir)
        assert temp_dir.exists()

    def test_processor_creates_output_dir(self, temp_dir):
        """Test processor creates output directory if it doesn't exist."""
        from services.core.process_point_cloud import PointCloudProcessor

        new_dir = temp_dir / "new_output"
        processor = PointCloudProcessor(output_dir=str(new_dir))
        assert new_dir.exists()

    @patch("services.core.process_point_cloud.EXTERNAL_DEPS_AVAILABLE", True)
    @patch("services.core.geocode.Geocoder.geocode_address")
    @patch("services.data.data_fetcher.DataFetcher.fetch_point_cloud_data")
    @patch("services.data.data_fetcher.DataFetcher.fetch_orthophoto_data")
    @patch("services.processing.orthophoto_io.OrthophotoIO.get_orthophoto_bounds")
    @patch(
        "services.processing.point_cloud_colorizer.PointCloudColorizer.colorize_point_cloud"
    )
    @patch("services.processing.point_cloud_io.PointCloudIO.load_point_cloud")
    @patch("services.processing.orthophoto_io.OrthophotoIO.load_orthophoto")
    @patch("services.processing.point_cloud_io.PointCloudIO.save_colorized_point_cloud")
    def test_process_from_address_success(
        self,
        mock_save_colorized_pc,
        mock_load_ortho,
        mock_load_pc,
        mock_colorize,
        mock_get_bounds,
        mock_fetch_ortho,
        mock_fetch_pc,
        mock_geocode,
        temp_dir,
        sample_address,
        mock_geocoding_result,
    ):
        """Test successful processing from address."""
        from services.core.process_point_cloud import PointCloudProcessor

        # Create a mock LAS data object
        from unittest.mock import Mock

        mock_las_data = Mock()
        mock_las_data.points = []  # Empty points for testing

        # Create a mock orthophoto dataset
        mock_ortho_dataset = Mock()

        # Setup mocks
        mock_geocode.return_value = mock_geocoding_result
        mock_fetch_pc.return_value = str(temp_dir / "test.laz")
        mock_fetch_ortho.return_value = str(temp_dir / "test.tif")
        mock_get_bounds.return_value = (
            {"left": -105.3, "right": -105.2, "bottom": 39.9, "top": 40.1},
            "EPSG:4326",
        )
        mock_load_pc.return_value = mock_las_data
        mock_load_ortho.return_value = mock_ortho_dataset

        # Mock colorize_point_cloud to return (colors, valid_mask) tuple
        import numpy as np

        mock_colors = np.array(
            [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
        )  # Mock RGB colors
        mock_valid_mask = np.array([True, True, True])  # Mock valid mask
        mock_colorize.return_value = (mock_colors, mock_valid_mask)

        # Mock save_colorized_point_cloud - it doesn't return a value, just saves the file
        mock_save_colorized_pc.return_value = None

        # Create dummy files (just need them to exist, methods are mocked)
        (temp_dir / "test.laz").write_text("dummy")
        (temp_dir / "test.tif").write_text("dummy")
        (temp_dir / "colorized.laz").write_text("dummy")

        processor = PointCloudProcessor(output_dir=str(temp_dir))
        result = processor.process_from_address(sample_address)

        assert result == str(temp_dir / "test_colorized.laz")
        mock_geocode.assert_called_once_with(sample_address)
        mock_fetch_pc.assert_called_once()
        mock_fetch_ortho.assert_called_once()
        mock_load_pc.assert_called_once()
        mock_load_ortho.assert_called_once()
        mock_colorize.assert_called_once()
        mock_save_colorized_pc.assert_called_once()

    @patch("services.core.process_point_cloud.EXTERNAL_DEPS_AVAILABLE", True)
    def test_process_from_address_empty_address(self, temp_dir):
        """Test processing with empty address."""
        from services.core.process_point_cloud import PointCloudProcessor

        processor = PointCloudProcessor(output_dir=str(temp_dir))

        # Empty address will cause geocoding to fail, not input validation
        with pytest.raises(Exception, match="All geocoding services failed"):
            processor.process_from_address("")

    @patch("services.core.process_point_cloud.EXTERNAL_DEPS_AVAILABLE", True)
    @patch("services.core.geocode.Geocoder.geocode_address")
    def test_process_from_address_geocoding_fails(
        self, mock_geocode, temp_dir, sample_address
    ):
        """Test processing when geocoding fails."""
        from services.core.process_point_cloud import PointCloudProcessor

        mock_geocode.side_effect = RuntimeError("Geocoding failed")

        processor = PointCloudProcessor(output_dir=str(temp_dir))

        with pytest.raises(RuntimeError, match="Geocoding failed"):
            processor.process_from_address(sample_address)

    @patch("services.core.process_point_cloud.EXTERNAL_DEPS_AVAILABLE", True)
    @patch("services.core.geocode.Geocoder.geocode_address")
    @patch("services.data.data_fetcher.DataFetcher.fetch_point_cloud_data")
    def test_process_from_address_point_cloud_fetch_fails(
        self,
        mock_fetch_pc,
        mock_geocode,
        temp_dir,
        sample_address,
        mock_geocoding_result,
    ):
        """Test processing when point cloud fetch fails."""
        from services.core.process_point_cloud import PointCloudProcessor

        mock_geocode.return_value = mock_geocoding_result
        mock_fetch_pc.side_effect = RuntimeError("Point cloud fetch failed")

        processor = PointCloudProcessor(output_dir=str(temp_dir))

        with pytest.raises(RuntimeError, match="Point cloud fetch failed"):
            processor.process_from_address(sample_address)
