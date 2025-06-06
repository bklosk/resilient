"""
Tests for data services (fetchers and data providers).
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json

# Ensure services can be imported
REPO_ROOT = Path(__file__).resolve().parents[1]
services_dir = REPO_ROOT / "services"
if str(services_dir) not in sys.path:
    sys.path.insert(0, str(services_dir))


class TestDataFetcher:
    """Test the main data fetcher service."""

    def test_data_fetcher_initialization(self, temp_dir):
        """Test data fetcher initializes correctly."""
        from services.data.data_fetcher import DataFetcher

        fetcher = DataFetcher(str(temp_dir))
        assert fetcher.output_dir == temp_dir
        assert temp_dir.exists()

    def test_data_fetcher_creates_output_dir(self, temp_dir):
        """Test data fetcher creates output directory."""
        from services.data.data_fetcher import DataFetcher

        new_dir = temp_dir / "new_data"
        fetcher = DataFetcher(str(new_dir))
        assert new_dir.exists()

    @patch(
        "services.data.get_point_cloud.PointCloudDatasetFinder.find_datasets_for_location"
    )
    @patch(
        "services.data.get_point_cloud.PointCloudDatasetFinder.select_best_dataset_for_location"
    )
    @patch("services.data.get_point_cloud.PointCloudDatasetFinder.download_point_cloud")
    def test_fetch_point_cloud_data_success(
        self,
        mock_download,
        mock_select,
        mock_find_datasets,
        temp_dir,
        sample_bbox,
        mock_geocoding_result,
    ):
        """Test successful point cloud data fetch."""
        from services.data.data_fetcher import DataFetcher

        # Setup mocks
        datasets = [{"name": "test_dataset", "title": "Test Dataset"}]
        best_dataset = {"name": "best_dataset", "title": "Best Dataset"}

        mock_find_datasets.return_value = datasets
        mock_select.return_value = best_dataset
        mock_download.return_value = str(temp_dir / "test.laz")

        # Create dummy file
        (temp_dir / "test.laz").write_text("dummy")

        # Create mock fetcher
        mock_pc_fetcher = Mock()
        mock_pc_fetcher.find_datasets_for_location = mock_find_datasets
        mock_pc_fetcher.select_best_dataset_for_location = mock_select
        mock_pc_fetcher.download_point_cloud = mock_download

        fetcher = DataFetcher(str(temp_dir))
        result = fetcher.fetch_point_cloud_data(
            mock_pc_fetcher, mock_geocoding_result[0], mock_geocoding_result[1]
        )

        assert result == str(temp_dir / "test.laz")
        mock_find_datasets.assert_called_once_with(mock_geocoding_result[0], mock_geocoding_result[1])
        mock_select.assert_called_once()
        mock_download.assert_called_once()

    def test_fetch_point_cloud_data_no_products(self, temp_dir, mock_geocoding_result):
        """Test point cloud fetch when no datasets found."""
        from services.data.data_fetcher import DataFetcher

        mock_pc_fetcher = Mock()
        mock_pc_fetcher.find_datasets_for_location.return_value = []

        fetcher = DataFetcher(str(temp_dir))

        with pytest.raises(RuntimeError, match="No LiDAR data found"):
            fetcher.fetch_point_cloud_data(
                mock_pc_fetcher, mock_geocoding_result[0], mock_geocoding_result[1]
            )

    @patch("services.data.get_orthophoto.NAIPFetcher.get_orthophoto_for_address")
    @patch("rasterio.open")
    def test_fetch_orthophoto_data_success(
        self,
        mock_rasterio_open,
        mock_get_orthophoto,
        temp_dir,
        sample_address,
        mock_geocoding_result,
        mock_orthophoto_data,
    ):
        """Test successful orthophoto data fetch."""
        from services.data.data_fetcher import DataFetcher

        # Setup mocks
        ortho_path = str(temp_dir / "test.tif")
        ortho_metadata = {"crs": "EPSG:4326"}
        mock_get_orthophoto.return_value = (ortho_path, ortho_metadata)
        mock_rasterio_open.return_value.__enter__.return_value = mock_orthophoto_data

        # Create dummy file
        (temp_dir / "test.tif").write_text("dummy")

        mock_ortho_fetcher = Mock()
        mock_ortho_fetcher.get_orthophoto_for_address = mock_get_orthophoto

        fetcher = DataFetcher(str(temp_dir))
        result = fetcher.fetch_orthophoto_data(
            mock_ortho_fetcher,
            sample_address,
            mock_geocoding_result[0],
            mock_geocoding_result[1],
        )

        assert result == ortho_path
        mock_get_orthophoto.assert_called_once_with(sample_address, str(temp_dir))

    def test_fetch_orthophoto_data_file_not_found(
        self, temp_dir, sample_address, mock_geocoding_result
    ):
        """Test orthophoto fetch when file is not created."""
        from services.data.data_fetcher import DataFetcher

        mock_ortho_fetcher = Mock()
        mock_ortho_fetcher.get_orthophoto_for_address.return_value = (
            str(temp_dir / "nonexistent.tif"),
            {},
        )

        fetcher = DataFetcher(str(temp_dir))

        with pytest.raises(RuntimeError, match="No valid orthophoto file found"):
            fetcher.fetch_orthophoto_data(
                mock_ortho_fetcher,
                sample_address,
                mock_geocoding_result[0],
                mock_geocoding_result[1],
            )


class TestPointCloudDatasetFinder:
    """Test the point cloud dataset finder."""

    def test_bounding_box_generation(self, mock_geocoding_result):
        """Test bounding box generation from coordinates."""
        from services.data.get_point_cloud import PointCloudDatasetFinder

        finder = PointCloudDatasetFinder()
        bbox = finder.generate_bounding_box(
            mock_geocoding_result[0], mock_geocoding_result[1], buffer_km=1.0
        )

        assert isinstance(bbox, str)
        parts = bbox.split(",")
        assert len(parts) == 4

        # Verify all parts are numeric
        bbox_floats = [float(p) for p in parts]
        assert len(bbox_floats) == 4

    def test_invalid_coordinates(self):
        """Test bounding box generation with invalid coordinates."""
        from services.data.get_point_cloud import PointCloudDatasetFinder

        finder = PointCloudDatasetFinder()

        with pytest.raises(ValueError):
            finder.generate_bounding_box(91.0, -105.0, buffer_km=1.0)  # Invalid lat

        with pytest.raises(ValueError):
            finder.generate_bounding_box(40.0, 181.0, buffer_km=1.0)  # Invalid lon

    def test_find_datasets_for_location_success(self, mock_geocoding_result):
        """Test successful dataset search using spatial index."""
        from services.data.get_point_cloud import PointCloudDatasetFinder
        from pyproj import Transformer

        # Transform test coordinates to Web Mercator (same as what the class does internally)
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        test_x, test_y = transformer.transform(mock_geocoding_result[1], mock_geocoding_result[0])
        
        # Create bounds that will contain the transformed coordinates (with buffer)
        buffer = 50000  # 50km buffer in Web Mercator
        bounds_containing = [test_x - buffer, test_y - buffer, test_x + buffer, test_y + buffer]
        bounds_not_containing = [test_x + buffer*2, test_y + buffer*2, test_x + buffer*3, test_y + buffer*3]

        # Mock the spatial index with test data
        with patch.object(PointCloudDatasetFinder, '_load_spatial_index') as mock_load:
            mock_load.return_value = [
                {
                    "name": "test_dataset_1",
                    "bounds": bounds_containing,  # Contains test point
                    "points": 1000000
                },
                {
                    "name": "test_dataset_2", 
                    "bounds": bounds_not_containing,  # Does not contain test point
                    "points": 500000
                }
            ]
            
            finder = PointCloudDatasetFinder()
            datasets = finder.find_datasets_for_location(
                mock_geocoding_result[0], mock_geocoding_result[1]
            )

            # Should find the dataset that contains the test coordinates
            assert len(datasets) >= 1
            assert any("test_dataset_1" in d["name"] for d in datasets)

    def test_filter_laz_products(self):
        """Test filtering for LAZ format products."""
        from services.data.get_point_cloud import PointCloudDatasetFinder

        products = [
            {"title": "Product 1", "format": "LAZ"},
            {"title": "Product 2", "format": "TIF"},
            {"title": "Product 3", "format": "laz"},  # lowercase
            {"title": "Product 4", "format": "XYZ"},
        ]

        finder = PointCloudDatasetFinder()
        laz_products = finder.filter_laz_products(products)

        assert len(laz_products) == 2
        assert all(p["format"].upper() == "LAZ" for p in laz_products)

    def test_select_best_dataset_for_location(self, mock_geocoding_result):
        """Test selecting best dataset for location."""
        from services.data.get_point_cloud import PointCloudDatasetFinder

        products = [
            {
                "title": "Old Product",
                "format": "LAZ",
                "spatial_bounds": {
                    "coordinates": [
                        [
                            [-106.0, 39.0],
                            [-104.0, 39.0],
                            [-104.0, 41.0],
                            [-106.0, 41.0],
                            [-106.0, 39.0],
                        ]
                    ]
                },
                "created": "2020-01-01",
            },
            {
                "title": "New Product",
                "format": "LAZ",
                "spatial_bounds": {
                    "coordinates": [
                        [
                            [-105.5, 39.5],
                            [-104.5, 39.5],
                            [-104.5, 40.5],
                            [-105.5, 40.5],
                            [-105.5, 39.5],
                        ]
                    ]
                },
                "created": "2022-01-01",
            },
        ]

        finder = PointCloudDatasetFinder()
        best = finder.select_best_dataset_for_location(
            products, mock_geocoding_result[0], mock_geocoding_result[1]
        )

        # Should select the newer product that contains the location
        assert best["title"] == "New Product"


class TestNAIPFetcher:
    """Test the NAIP orthophoto fetcher."""

    def test_naip_fetcher_initialization(self):
        """Test NAIP fetcher initializes correctly."""
        from services.data.get_orthophoto import NAIPFetcher

        fetcher = NAIPFetcher()
        assert hasattr(fetcher, "get_orthophoto_for_address")

    def test_get_orthophoto_for_address_success(self, temp_dir, sample_address):
        """Test successful orthophoto fetch."""
        from services.data.get_orthophoto import NAIPFetcher

        fetcher = NAIPFetcher()

        # Mock the export_image method instead of trying to mock all HTTP calls
        mock_metadata = {
            "bbox": "-105.267176,39.975936,-105.265681,39.977082",
            "image_size": "127,127",
            "export_time": "2024-01-01T00:00:00Z",
        }

        # Create a fake output file
        fake_output_path = str(temp_dir / "naip_orthophoto_test.tif")
        with open(fake_output_path, "wb") as f:
            f.write(b"fake_image_data")

        with patch.object(fetcher, "export_image") as mock_export:
            mock_export.return_value = mock_metadata

            with patch.object(fetcher, "_get_orthophoto_url") as mock_url:
                mock_url.return_value = "http://example.com/image.tif"

                # Also mock the output path generation
                with patch("os.path.join", return_value=fake_output_path):
                    result_path, metadata = fetcher.get_orthophoto_for_address(
                        sample_address, str(temp_dir)
                    )

                    assert Path(result_path).exists()
                    assert isinstance(metadata, dict)

    def test_get_orthophoto_for_address_empty_address(self, temp_dir):
        """Test orthophoto fetch with empty address."""
        from services.data.get_orthophoto import NAIPFetcher

        fetcher = NAIPFetcher()

        with pytest.raises(ValueError, match="Address cannot be empty"):
            fetcher.get_orthophoto_for_address("", str(temp_dir))


class TestFEMADataFetcher:
    """Test the FEMA risk data fetcher."""

    def test_fema_fetcher_initialization(self):
        """Test FEMA fetcher initializes correctly."""
        from services.data.get_fema_risk import FEMADataFetcher

        fetcher = FEMADataFetcher()
        assert hasattr(fetcher, "process_address")

    @patch("services.core.geocode.Geocoder.geocode_address")
    @patch("requests.get")
    def test_process_address_success(
        self, mock_get, mock_geocode, temp_dir, sample_address, mock_geocoding_result
    ):
        """Test successful FEMA risk processing."""
        from services.data.get_fema_risk import FEMADataFetcher

        # Setup mocks
        mock_geocode.return_value = mock_geocoding_result

        # Mock FEMA API response
        mock_response = Mock()
        mock_response.content = b"fake_flood_map_data"
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "features": [{"attributes": {"risk_score": 0.75}}]
        }
        mock_get.return_value = mock_response

        fetcher = FEMADataFetcher()
        flood_map, risk_data = fetcher.process_address(
            sample_address, output_dir=temp_dir
        )

        assert Path(flood_map).exists()
        assert isinstance(risk_data, dict)
        assert "risk_score" in str(risk_data) or len(risk_data) > 0

    def test_process_address_empty_address(self, temp_dir):
        """Test FEMA processing with empty address."""
        from services.data.get_fema_risk import FEMADataFetcher

        fetcher = FEMADataFetcher()

        with pytest.raises(ValueError, match="Address cannot be empty"):
            fetcher.process_address("", output_dir=temp_dir)
