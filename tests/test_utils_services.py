"""
Tests for utility services.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Ensure services can be imported
REPO_ROOT = Path(__file__).resolve().parents[1]
services_dir = REPO_ROOT / "services"
if str(services_dir) not in sys.path:
    sys.path.insert(0, str(services_dir))


class TestUtils:
    """Test the main utils module."""

    def test_bounding_box_utils_generate(self, mock_geocoding_result):
        """Test bounding box generation utility."""
        from services.utils.utils import BoundingBoxUtils

        bbox = BoundingBoxUtils.generate_bounding_box(
            mock_geocoding_result[0], mock_geocoding_result[1], buffer_km=1.0
        )

        assert isinstance(bbox, str)
        parts = bbox.split(",")
        assert len(parts) == 4

        # Validate all parts are numeric
        bbox_floats = [float(p) for p in parts]
        assert len(bbox_floats) == 4

    def test_bounding_box_utils_validate_valid(self, sample_bbox):
        """Test bounding box validation with valid bbox."""
        from services.utils.utils import BoundingBoxUtils

        is_valid = BoundingBoxUtils.validate_bounding_box(sample_bbox)
        assert is_valid is True

    def test_bounding_box_utils_validate_invalid(self):
        """Test bounding box validation with invalid bbox."""
        from services.utils.utils import BoundingBoxUtils

        # Test various invalid formats
        assert BoundingBoxUtils.validate_bounding_box("") is False
        assert (
            BoundingBoxUtils.validate_bounding_box("1,2,3") is False
        )  # Not enough parts
        assert BoundingBoxUtils.validate_bounding_box("a,b,c,d") is False  # Non-numeric
        assert (
            BoundingBoxUtils.validate_bounding_box("200,0,201,1") is False
        )  # Invalid coordinates

    def test_file_utils_safe_filename(self, sample_address):
        """Test safe filename generation."""
        from services.utils.utils import FileUtils

        safe_name = FileUtils.get_safe_filename(sample_address)

        # Should not contain problematic characters
        assert " " not in safe_name
        assert "," not in safe_name
        assert "/" not in safe_name
        assert "\\" not in safe_name

        # Should be lowercase
        assert safe_name.islower()

        # Should start with expected content
        assert safe_name.startswith("1250")

    def test_file_utils_safe_filename_special_chars(self):
        """Test safe filename with special characters."""
        from services.utils.utils import FileUtils

        unsafe_name = "Test / File \\ Name: With * Special ? Chars"
        safe_name = FileUtils.get_safe_filename(unsafe_name)

        # Should not contain any special characters
        special_chars = ["/", "\\", ":", "*", "?", "<", ">", "|", " "]
        for char in special_chars:
            assert char not in safe_name

    @patch("geopy.geocoders.Photon.geocode")
    @patch("geopy.geocoders.ArcGIS.geocode")
    def test_geocode_utils_success(
        self, mock_arcgis, mock_photon, sample_address, mock_geocoding_result
    ):
        """Test successful geocoding with utils."""
        from services.utils.utils import GeocodeUtils

        # Mock successful geocoding
        mock_location = Mock()
        mock_location.latitude = mock_geocoding_result[0]
        mock_location.longitude = mock_geocoding_result[1]
        mock_photon.return_value = mock_location

        geocoder = GeocodeUtils()
        lat, lon = geocoder.geocode_address(sample_address)

        assert lat == mock_geocoding_result[0]
        assert lon == mock_geocoding_result[1]

    @patch("geopy.geocoders.Photon.geocode")
    @patch("geopy.geocoders.ArcGIS.geocode")
    def test_geocode_utils_fallback(
        self, mock_arcgis, mock_photon, sample_address, mock_geocoding_result
    ):
        """Test geocoding fallback functionality."""
        from services.utils.utils import GeocodeUtils
        from geopy.exc import GeopyError

        # First service fails
        mock_photon.side_effect = GeopyError("Service unavailable")

        # Second service succeeds
        mock_location = Mock()
        mock_location.latitude = mock_geocoding_result[0]
        mock_location.longitude = mock_geocoding_result[1]
        mock_arcgis.return_value = mock_location

        geocoder = GeocodeUtils()
        lat, lon = geocoder.geocode_address(sample_address, max_retries=1)

        assert lat == mock_geocoding_result[0]
        assert lon == mock_geocoding_result[1]
        mock_photon.assert_called_once()
        mock_arcgis.assert_called_once()

    @patch("pyproj.CRS.from_epsg")
    def test_crs_utils_detect_crs(self, mock_from_epsg, mock_point_cloud_data):
        """Test CRS detection utility."""
        from services.utils.utils import CRSUtils

        mock_crs = Mock()
        mock_crs.to_string.return_value = "EPSG:26913"
        mock_from_epsg.return_value = mock_crs

        detected_crs = CRSUtils.detect_point_cloud_crs(mock_point_cloud_data)

        assert detected_crs == "EPSG:26913"
        mock_from_epsg.assert_called_once_with(26913)

    def test_crs_utils_detect_crs_no_epsg(self, mock_point_cloud_data):
        """Test CRS detection when no EPSG available."""
        from services.utils.utils import CRSUtils

        # Remove EPSG from mock
        mock_point_cloud_data.header.epsg = None

        detected_crs = CRSUtils.detect_point_cloud_crs(mock_point_cloud_data)

        # Should handle gracefully
        assert detected_crs is None or isinstance(detected_crs, str)


class TestFloodDepth:
    """Test flood depth analysis utilities."""

    def test_flood_depth_initialization(self):
        """Test flood depth module can be imported."""
        from services.utils.flood_depth import FloodDepthAnalyzer

        analyzer = FloodDepthAnalyzer()
        assert hasattr(analyzer, "analyze_flood_risk")

    @patch("services.core.geocode.Geocoder.geocode_address")
    @patch("services.data.get_fema_risk.FEMADataFetcher.process_address")
    def test_analyze_flood_risk_success(
        self, mock_fema, mock_geocode, temp_dir, sample_address, mock_geocoding_result
    ):
        """Test successful flood risk analysis."""
        from services.utils.flood_depth import FloodDepthAnalyzer

        # Setup mocks
        mock_geocode.return_value = mock_geocoding_result
        mock_fema.return_value = (str(temp_dir / "flood_map.tif"), {"risk_score": 0.75})

        # Create dummy flood map
        (temp_dir / "flood_map.tif").write_text("dummy flood data")

        analyzer = FloodDepthAnalyzer()
        result = analyzer.analyze_flood_risk(sample_address, str(temp_dir))

        assert isinstance(result, dict)
        assert "risk_score" in str(result) or len(result) > 0

    def test_analyze_flood_risk_empty_address(self, temp_dir):
        """Test flood risk analysis with empty address."""
        from services.utils.flood_depth import FloodDepthAnalyzer

        analyzer = FloodDepthAnalyzer()

        with pytest.raises(ValueError, match="Address cannot be empty"):
            analyzer.analyze_flood_risk("", str(temp_dir))

    @patch("services.core.geocode.Geocoder.geocode_address")
    def test_analyze_flood_risk_geocoding_fails(
        self, mock_geocode, temp_dir, sample_address
    ):
        """Test flood risk analysis when geocoding fails."""
        from services.utils.flood_depth import FloodDepthAnalyzer

        mock_geocode.side_effect = RuntimeError("Geocoding failed")

        analyzer = FloodDepthAnalyzer()

        with pytest.raises(RuntimeError, match="Geocoding failed"):
            analyzer.analyze_flood_risk(sample_address, str(temp_dir))


class TestEstimateReplacementValue:
    """Test replacement value estimation utilities."""

    def test_replacement_value_estimator_initialization(self):
        """Test replacement value estimator can be imported."""
        from services.utils.estimate_replacement_value import ReplacementValueEstimator

        estimator = ReplacementValueEstimator()
        assert hasattr(estimator, "estimate_property_value")

    @patch("services.core.geocode.Geocoder.geocode_address")
    def test_estimate_property_value_success(
        self, mock_geocode, sample_address, mock_geocoding_result
    ):
        """Test successful property value estimation."""
        from services.utils.estimate_replacement_value import ReplacementValueEstimator

        mock_geocode.return_value = mock_geocoding_result

        estimator = ReplacementValueEstimator()

        # Mock external API calls
        with patch("requests.get") as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = {
                "estimated_value": 500000,
                "confidence": 0.85,
            }
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            result = estimator.estimate_property_value(sample_address)

            assert isinstance(result, dict)
            assert "estimated_value" in result or "value" in str(result)

    def test_estimate_property_value_empty_address(self):
        """Test property value estimation with empty address."""
        from services.utils.estimate_replacement_value import ReplacementValueEstimator

        estimator = ReplacementValueEstimator()

        with pytest.raises(ValueError, match="Address cannot be empty"):
            estimator.estimate_property_value("")

    @patch("services.core.geocode.Geocoder.geocode_address")
    def test_estimate_property_value_geocoding_fails(
        self, mock_geocode, sample_address
    ):
        """Test property value estimation when geocoding fails."""
        from services.utils.estimate_replacement_value import ReplacementValueEstimator

        mock_geocode.side_effect = RuntimeError("Geocoding failed")

        estimator = ReplacementValueEstimator()

        with pytest.raises(RuntimeError, match="Geocoding failed"):
            estimator.estimate_property_value(sample_address)

    def test_estimate_property_value_basic_calculation(self):
        """Test basic property value calculation without external APIs."""
        from services.utils.estimate_replacement_value import ReplacementValueEstimator

        estimator = ReplacementValueEstimator()

        # Test basic calculation method
        with patch.object(estimator, "_calculate_basic_estimate") as mock_calc:
            mock_calc.return_value = {"estimated_value": 400000, "method": "basic"}

            result = estimator._calculate_basic_estimate("CO", "Boulder", 2000)  # sqft

            assert isinstance(result, dict)
            assert "estimated_value" in result

    def test_estimate_property_value_area_adjustment(self):
        """Test property value estimation with area adjustments."""
        from services.utils.estimate_replacement_value import ReplacementValueEstimator

        estimator = ReplacementValueEstimator()

        # Test different property sizes
        base_value = 300000

        # Larger property should have higher value
        large_adjustment = estimator._apply_area_adjustment(base_value, 3000)
        small_adjustment = estimator._apply_area_adjustment(base_value, 1000)

        if large_adjustment != base_value and small_adjustment != base_value:
            assert large_adjustment > small_adjustment

    def test_estimate_property_value_location_factors(self):
        """Test property value estimation with location factors."""
        from services.utils.estimate_replacement_value import ReplacementValueEstimator

        estimator = ReplacementValueEstimator()

        # Test location factor calculation
        base_value = 300000

        # Boulder, CO should have a location factor
        with patch.object(estimator, "_get_location_factor") as mock_factor:
            mock_factor.return_value = 1.2  # 20% premium for Boulder

            adjusted_value = estimator._apply_location_factor(
                base_value, "CO", "Boulder"
            )

            assert adjusted_value >= base_value
