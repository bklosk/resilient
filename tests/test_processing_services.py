"""
Tests for processing services (I/O, transformations, colorization, etc.).
"""

import pytest
import sys
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Ensure services can be imported
REPO_ROOT = Path(__file__).resolve().parents[1]
services_dir = REPO_ROOT / "services"
if str(services_dir) not in sys.path:
    sys.path.insert(0, str(services_dir))


class TestPointCloudIO:
    """Test point cloud I/O operations."""

    def test_point_cloud_io_initialization(self):
        """Test PointCloudIO can be imported and used."""
        from services.processing.point_cloud_io import PointCloudIO

        # Test static methods exist
        assert hasattr(PointCloudIO, "load_point_cloud")
        assert hasattr(PointCloudIO, "save_point_cloud")

    @patch("laspy.read")
    def test_load_point_cloud_success(self, mock_read, temp_dir, mock_point_cloud_data):
        """Test successful point cloud loading."""
        from services.processing.point_cloud_io import PointCloudIO

        # Create dummy file
        test_file = temp_dir / "test.laz"
        test_file.write_text("dummy")

        mock_read.return_value = mock_point_cloud_data

        result = PointCloudIO.load_point_cloud(str(test_file))

        assert result == mock_point_cloud_data
        mock_read.assert_called_once_with(str(test_file))

    def test_load_point_cloud_file_not_found(self, temp_dir):
        """Test loading non-existent point cloud file."""
        from services.processing.point_cloud_io import PointCloudIO

        non_existent = temp_dir / "nonexistent.laz"

        with pytest.raises(FileNotFoundError):
            PointCloudIO.load_point_cloud(str(non_existent))

    def test_save_point_cloud_success(self, temp_dir, mock_point_cloud_data):
        """Test successful point cloud saving."""
        from services.processing.point_cloud_io import PointCloudIO

        output_file = temp_dir / "output.laz"

        # Mock the write method on the point cloud data object
        mock_point_cloud_data.write = Mock()

        # Create the file to simulate successful write
        def mock_write(path):
            Path(path).touch()

        mock_point_cloud_data.write.side_effect = mock_write

        PointCloudIO.save_point_cloud(mock_point_cloud_data, str(output_file))

        mock_point_cloud_data.write.assert_called_once_with(str(output_file))

    def test_save_point_cloud_invalid_path(self, mock_point_cloud_data):
        """Test saving to invalid path."""
        from services.processing.point_cloud_io import PointCloudIO

        with pytest.raises(RuntimeError):
            PointCloudIO.save_point_cloud(
                mock_point_cloud_data, "/invalid/path/output.laz"
            )


class TestOrthophotoIO:
    """Test orthophoto I/O operations."""

    def test_orthophoto_io_initialization(self):
        """Test OrthophotoIO can be imported and used."""
        from services.processing.orthophoto_io import OrthophotoIO

        assert hasattr(OrthophotoIO, "load_orthophoto")
        assert hasattr(OrthophotoIO, "validate_orthophoto")

    @patch("rasterio.open")
    def test_load_orthophoto_success(self, mock_open, temp_dir, mock_orthophoto_data):
        """Test successful orthophoto loading."""
        from services.processing.orthophoto_io import OrthophotoIO

        # Create dummy file
        test_file = temp_dir / "test.tif"
        test_file.write_text("dummy")

        mock_open.return_value = mock_orthophoto_data

        result = OrthophotoIO.load_orthophoto(str(test_file))

        assert result == mock_orthophoto_data
        mock_open.assert_called_once_with(str(test_file))

    def test_load_orthophoto_file_not_found(self, temp_dir):
        """Test loading non-existent orthophoto file."""
        from services.processing.orthophoto_io import OrthophotoIO

        non_existent = temp_dir / "nonexistent.tif"

        with pytest.raises(FileNotFoundError):
            OrthophotoIO.load_orthophoto(str(non_existent))

    @patch("rasterio.open")
    def test_validate_orthophoto_success(
        self, mock_open, temp_dir, mock_orthophoto_data
    ):
        """Test successful orthophoto validation."""
        from services.processing.orthophoto_io import OrthophotoIO

        test_file = temp_dir / "test.tif"
        test_file.write_text("dummy")

        mock_orthophoto_data.width = 100
        mock_orthophoto_data.height = 100
        mock_orthophoto_data.count = 3
        mock_open.return_value.__enter__.return_value = mock_orthophoto_data

        is_valid = OrthophotoIO.validate_orthophoto(str(test_file))

        assert is_valid is True

    @patch("rasterio.open")
    def test_validate_orthophoto_invalid_dimensions(
        self, mock_open, temp_dir, mock_orthophoto_data
    ):
        """Test validation with invalid dimensions."""
        from services.processing.orthophoto_io import OrthophotoIO

        test_file = temp_dir / "test.tif"
        test_file.write_text("dummy")

        mock_orthophoto_data.width = 0
        mock_orthophoto_data.height = 0
        mock_open.return_value.__enter__.return_value = mock_orthophoto_data

        is_valid = OrthophotoIO.validate_orthophoto(str(test_file))

        assert is_valid is False


class TestCoordinateTransformer:
    """Test coordinate transformation operations."""

    def test_coordinate_transformer_initialization(self):
        """Test CoordinateTransformer can be imported."""
        from services.processing.coordinate_transformer import CoordinateTransformer

        transformer = CoordinateTransformer()
        assert hasattr(transformer, "detect_point_cloud_crs")
        assert hasattr(transformer, "transform_coordinates")

    @patch("pyproj.CRS.from_epsg")
    def test_detect_point_cloud_crs_with_epsg(
        self, mock_from_epsg, mock_point_cloud_data
    ):
        """Test CRS detection from EPSG code."""
        from services.processing.coordinate_transformer import CoordinateTransformer

        mock_crs = Mock()
        mock_crs.to_string.return_value = "EPSG:26913"
        mock_from_epsg.return_value = mock_crs

        transformer = CoordinateTransformer()
        crs = transformer.detect_point_cloud_crs(mock_point_cloud_data)

        assert crs == "EPSG:26913"
        mock_from_epsg.assert_called_once_with(26913)

    def test_detect_point_cloud_crs_no_epsg(self, mock_point_cloud_data):
        """Test CRS detection when no EPSG code available."""
        from services.processing.coordinate_transformer import CoordinateTransformer

        # Remove EPSG from mock
        mock_point_cloud_data.header.epsg = None

        transformer = CoordinateTransformer()
        crs = transformer.detect_point_cloud_crs(mock_point_cloud_data)

        # Should return a default or None
        assert crs is None or isinstance(crs, str)

    @patch("pyproj.Transformer.from_crs")
    def test_transform_coordinates_success(self, mock_from_crs):
        """Test successful coordinate transformation."""
        from services.processing.coordinate_transformer import CoordinateTransformer

        # Mock transformer
        mock_transformer = Mock()
        mock_transformer.transform.return_value = ([1001.0, 1002.0], [2001.0, 2002.0])
        mock_from_crs.return_value = mock_transformer

        transformer = CoordinateTransformer()
        new_x, new_y = transformer.transform_coordinates(
            [100.0, 101.0], [200.0, 201.0], "EPSG:4326", "EPSG:26913"
        )

        assert new_x == [1001.0, 1002.0]
        assert new_y == [2001.0, 2002.0]
        mock_from_crs.assert_called_once_with("EPSG:4326", "EPSG:26913", always_xy=True)

    def test_transform_coordinates_same_crs(self):
        """Test transformation when source and target CRS are the same."""
        from services.processing.coordinate_transformer import CoordinateTransformer

        transformer = CoordinateTransformer()
        x_coords = [100.0, 101.0]
        y_coords = [200.0, 201.0]

        new_x, new_y = transformer.transform_coordinates(
            x_coords, y_coords, "EPSG:4326", "EPSG:4326"
        )

        assert new_x == x_coords
        assert new_y == y_coords


class TestPointCloudColorizer:
    """Test point cloud colorization operations."""

    def test_point_cloud_colorizer_initialization(self, temp_dir):
        """Test PointCloudColorizer initialization."""
        from services.processing.point_cloud_colorizer import PointCloudColorizer

        colorizer = PointCloudColorizer(str(temp_dir))
        assert colorizer.output_dir == Path(temp_dir)

    @patch("services.processing.point_cloud_io.PointCloudIO.load_point_cloud")
    @patch("services.processing.orthophoto_io.OrthophotoIO.load_orthophoto")
    @patch("services.processing.point_cloud_io.PointCloudIO.save_colorized_point_cloud")
    def test_colorize_success(
        self,
        mock_save,
        mock_load_ortho,
        mock_load_pc,
        temp_dir,
        mock_point_cloud_data,
        mock_orthophoto_data,
    ):
        """Test successful point cloud colorization."""
        from services.processing.point_cloud_colorizer import PointCloudColorizer
        import numpy as np

        # Setup input files
        pc_file = temp_dir / "input.laz"
        ortho_file = temp_dir / "input.tif"
        pc_file.write_text("dummy")
        ortho_file.write_text("dummy")

        # Setup mocks
        mock_load_pc.return_value = mock_point_cloud_data
        mock_load_ortho.return_value = mock_orthophoto_data

        colorizer = PointCloudColorizer(str(temp_dir))

        # Mock the colorize_point_cloud method to return valid colors and mask
        colorizer.colorize_point_cloud = Mock(
            return_value=(np.array([[255, 128, 64]]), np.array([True]))
        )

        result = colorizer.colorize(str(pc_file), str(ortho_file))

        assert Path(result).name.endswith(".laz")
        mock_load_pc.assert_called_once_with(str(pc_file))
        mock_load_ortho.assert_called_once_with(str(ortho_file))
        mock_save.assert_called_once()

    def test_colorize_file_not_found(self, temp_dir):
        """Test colorization with non-existent files."""
        from services.processing.point_cloud_colorizer import PointCloudColorizer

        colorizer = PointCloudColorizer(str(temp_dir))

        with pytest.raises(FileNotFoundError):
            colorizer.colorize("nonexistent.laz", "nonexistent.tif")

    @patch("services.processing.point_cloud_io.PointCloudIO.load_point_cloud")
    def test_colorize_orthophoto_not_found(
        self, mock_load_pc, temp_dir, mock_point_cloud_data
    ):
        """Test colorization when orthophoto file doesn't exist."""
        from services.processing.point_cloud_colorizer import PointCloudColorizer

        pc_file = temp_dir / "input.laz"
        pc_file.write_text("dummy")

        mock_load_pc.return_value = mock_point_cloud_data

        colorizer = PointCloudColorizer(str(temp_dir))

        with pytest.raises(FileNotFoundError):
            colorizer.colorize(str(pc_file), "nonexistent.tif")


class TestAlignmentDiagnostics:
    """Test alignment diagnostics functionality."""

    def test_alignment_diagnostics_initialization(self, temp_dir):
        """Test AlignmentDiagnostics initialization."""
        from services.processing.alignment_diagnostics import AlignmentDiagnostics

        diagnostics = AlignmentDiagnostics(str(temp_dir))
        assert diagnostics.output_dir == Path(temp_dir)

    @patch("services.processing.point_cloud_io.PointCloudIO.load_point_cloud")
    @patch("services.processing.orthophoto_io.OrthophotoIO.load_orthophoto")
    @patch("matplotlib.pyplot.savefig")
    def test_generate_alignment_diagnostic(
        self,
        mock_savefig,
        mock_load_ortho,
        mock_load_pc,
        temp_dir,
        mock_point_cloud_data,
        mock_orthophoto_data,
    ):
        """Test alignment diagnostic generation."""
        from services.processing.alignment_diagnostics import AlignmentDiagnostics

        # Setup input files
        pc_file = temp_dir / "input.laz"
        ortho_file = temp_dir / "input.tif"
        pc_file.write_text("dummy")
        ortho_file.write_text("dummy")

        mock_load_pc.return_value = mock_point_cloud_data
        mock_load_ortho.return_value = mock_orthophoto_data

        diagnostics = AlignmentDiagnostics(str(temp_dir))
        result = diagnostics.generate_alignment_diagnostic(
            str(pc_file), str(ortho_file)
        )

        assert Path(result).suffix == ".png"
        mock_load_pc.assert_called_once()
        mock_load_ortho.assert_called_once()
        mock_savefig.assert_called_once()

    def test_generate_alignment_diagnostic_file_not_found(self, temp_dir):
        """Test diagnostic generation with non-existent files."""
        from services.processing.alignment_diagnostics import AlignmentDiagnostics

        diagnostics = AlignmentDiagnostics(str(temp_dir))

        with pytest.raises(FileNotFoundError):
            diagnostics.generate_alignment_diagnostic(
                "nonexistent.laz", "nonexistent.tif"
            )


class TestBuildSpatialIndex:
    """Test spatial index building functionality."""

    def test_spatial_index_initialization(self):
        """Test spatial index builder can be imported."""
        from services.processing.build_spatial_index import SpatialIndexBuilder

        builder = SpatialIndexBuilder()
        assert hasattr(builder, "build_index")

    @patch("services.processing.point_cloud_io.PointCloudIO.load_point_cloud")
    def test_build_spatial_index_success(
        self, mock_load_pc, temp_dir, mock_point_cloud_data
    ):
        """Test successful spatial index building."""
        from services.processing.build_spatial_index import SpatialIndexBuilder

        pc_file = temp_dir / "input.laz"
        pc_file.write_text("dummy")

        mock_load_pc.return_value = mock_point_cloud_data

        builder = SpatialIndexBuilder()
        result = builder.build_index(str(pc_file), str(temp_dir))

        assert Path(result).suffix == ".json"
        mock_load_pc.assert_called_once()

    def test_build_spatial_index_file_not_found(self, temp_dir):
        """Test spatial index building with non-existent file."""
        from services.processing.build_spatial_index import SpatialIndexBuilder

        builder = SpatialIndexBuilder()

        with pytest.raises(FileNotFoundError):
            builder.build_index("nonexistent.laz", str(temp_dir))
