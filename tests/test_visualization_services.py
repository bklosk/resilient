"""
Tests for visualization services.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Ensure services can be imported
REPO_ROOT = Path(__file__).resolve().parents[1]
services_dir = REPO_ROOT / "services"
if str(services_dir) not in sys.path:
    sys.path.insert(0, str(services_dir))


class TestSummaryReporter:
    """Test the summary reporting functionality."""

    def test_summary_reporter_initialization(self, temp_dir):
        """Test SummaryReporter initialization."""
        from services.visualization.summary_reporter import SummaryReporter

        reporter = SummaryReporter(str(temp_dir))
        assert reporter.output_dir == Path(temp_dir)

    @patch("services.processing.point_cloud_io.PointCloudIO.load_point_cloud")
    @patch("services.processing.orthophoto_io.OrthophotoIO.load_orthophoto")
    def test_generate_summary_report_success(
        self,
        mock_load_ortho,
        mock_load_pc,
        temp_dir,
        mock_point_cloud_data,
        mock_orthophoto_data,
        sample_address,
    ):
        """Test successful summary report generation."""
        from services.visualization.summary_reporter import SummaryReporter

        # Setup input files
        pc_file = temp_dir / "input.laz"
        ortho_file = temp_dir / "input.tif"
        pc_file.write_text("dummy")
        ortho_file.write_text("dummy")

        # Setup mocks
        mock_load_pc.return_value = mock_point_cloud_data
        mock_load_ortho.return_value = mock_orthophoto_data

        reporter = SummaryReporter(str(temp_dir))

        metadata = {
            "address": sample_address,
            "processing_time": 120.5,
            "point_count": 1000000,
        }

        result = reporter.generate_summary_report(
            str(pc_file), str(ortho_file), metadata
        )

        assert Path(result).suffix == ".json"
        assert Path(result).exists()

    def test_generate_summary_report_file_not_found(self, temp_dir, sample_address):
        """Test summary report generation with non-existent files."""
        from services.visualization.summary_reporter import SummaryReporter

        reporter = SummaryReporter(str(temp_dir))

        with pytest.raises(FileNotFoundError):
            reporter.generate_summary_report(
                "nonexistent.laz", "nonexistent.tif", {"address": sample_address}
            )

    @patch("services.processing.point_cloud_io.PointCloudIO.load_point_cloud")
    @patch("services.processing.orthophoto_io.OrthophotoIO.load_orthophoto")
    def test_calculate_processing_statistics(
        self,
        mock_load_ortho,
        mock_load_pc,
        temp_dir,
        mock_point_cloud_data,
        mock_orthophoto_data,
    ):
        """Test processing statistics calculation."""
        from services.visualization.summary_reporter import SummaryReporter

        pc_file = temp_dir / "input.laz"
        ortho_file = temp_dir / "input.tif"
        pc_file.write_text("dummy")
        ortho_file.write_text("dummy")

        mock_load_pc.return_value = mock_point_cloud_data
        mock_load_ortho.return_value = mock_orthophoto_data

        reporter = SummaryReporter(str(temp_dir))
        stats = reporter._calculate_processing_statistics(str(pc_file), str(ortho_file))

        assert isinstance(stats, dict)
        assert "point_cloud" in stats
        assert "orthophoto" in stats

    def test_format_report_data(self, temp_dir, sample_address):
        """Test report data formatting."""
        from services.visualization.summary_reporter import SummaryReporter

        reporter = SummaryReporter(str(temp_dir))

        metadata = {
            "address": sample_address,
            "processing_time": 120.5,
            "coordinates": {"latitude": 40.0274, "longitude": -105.2519},
        }

        stats = {
            "point_cloud": {"point_count": 1000000, "file_size_mb": 45.2},
            "orthophoto": {"width": 1024, "height": 1024, "bands": 3},
        }

        formatted = reporter._format_report_data(metadata, stats)

        assert isinstance(formatted, dict)
        assert "metadata" in formatted
        assert "statistics" in formatted
        assert formatted["metadata"]["address"] == sample_address


class TestOverheadImage:
    """Test overhead image generation functionality."""

    def test_overhead_image_initialization(self):
        """Test OverheadImage can be imported."""
        from services.visualization.overhead_image import OverheadImageGenerator

        generator = OverheadImageGenerator()
        assert hasattr(generator, "generate_overhead_view")

    @patch("services.processing.point_cloud_io.PointCloudIO.load_point_cloud")
    @patch("matplotlib.pyplot.savefig")
    def test_generate_overhead_view_success(
        self, mock_savefig, mock_load_pc, temp_dir, mock_point_cloud_data
    ):
        """Test successful overhead view generation."""
        from services.visualization.overhead_image import OverheadImageGenerator

        pc_file = temp_dir / "input.laz"
        pc_file.write_text("dummy")

        mock_load_pc.return_value = mock_point_cloud_data

        generator = OverheadImageGenerator()
        result = generator.generate_overhead_view(str(pc_file), str(temp_dir))

        assert Path(result).suffix == ".png"
        # Point cloud is loaded twice: once for density calculation, once for visualization
        assert mock_load_pc.call_count == 2
        mock_savefig.assert_called_once()

    def test_generate_overhead_view_file_not_found(self, temp_dir):
        """Test overhead view generation with non-existent file."""
        from services.visualization.overhead_image import OverheadImageGenerator

        generator = OverheadImageGenerator()

        with pytest.raises(FileNotFoundError):
            generator.generate_overhead_view("nonexistent.laz", str(temp_dir))

    @patch("services.processing.point_cloud_io.PointCloudIO.load_point_cloud")
    @patch("matplotlib.pyplot.savefig")
    def test_generate_overhead_view_with_colormap(
        self, mock_savefig, mock_load_pc, temp_dir, mock_point_cloud_data
    ):
        """Test overhead view generation with custom colormap."""
        from services.visualization.overhead_image import OverheadImageGenerator

        pc_file = temp_dir / "input.laz"
        pc_file.write_text("dummy")

        mock_load_pc.return_value = mock_point_cloud_data

        generator = OverheadImageGenerator()
        result = generator.generate_overhead_view(
            str(pc_file), str(temp_dir), colormap="viridis"
        )

        assert Path(result).suffix == ".png"
        mock_savefig.assert_called_once()

    @patch("services.processing.point_cloud_io.PointCloudIO.load_point_cloud")
    def test_calculate_point_density(
        self, mock_load_pc, temp_dir, mock_point_cloud_data
    ):
        """Test point density calculation."""
        from services.visualization.overhead_image import OverheadImageGenerator

        pc_file = temp_dir / "input.laz"
        pc_file.write_text("dummy")

        mock_load_pc.return_value = mock_point_cloud_data

        generator = OverheadImageGenerator()
        density = generator._calculate_point_density(str(pc_file))

        assert isinstance(density, (int, float))
        assert density > 0


class TestVisualizePointCloud:
    """Test point cloud visualization functionality."""

    def test_point_cloud_visualizer_initialization(self):
        """Test PointCloudVisualizer can be imported."""
        from services.visualization.visualize_point_cloud import PointCloudVisualizer

        visualizer = PointCloudVisualizer()
        assert hasattr(visualizer, "create_3d_visualization")

    @patch("services.processing.point_cloud_io.PointCloudIO.load_point_cloud")
    @patch("pyvista.save_meshio")
    def test_create_3d_visualization_success(
        self, mock_save, mock_load_pc, temp_dir, mock_point_cloud_data
    ):
        """Test successful 3D visualization creation."""
        from services.visualization.visualize_point_cloud import PointCloudVisualizer

        pc_file = temp_dir / "input.laz"
        pc_file.write_text("dummy")

        mock_load_pc.return_value = mock_point_cloud_data

        visualizer = PointCloudVisualizer()
        result = visualizer.create_3d_visualization(str(pc_file), str(temp_dir))

        assert Path(result).suffix in [".ply", ".obj", ".html"]
        mock_load_pc.assert_called_once()

    def test_create_3d_visualization_file_not_found(self, temp_dir):
        """Test 3D visualization with non-existent file."""
        from services.visualization.visualize_point_cloud import PointCloudVisualizer

        visualizer = PointCloudVisualizer()

        with pytest.raises(FileNotFoundError):
            visualizer.create_3d_visualization("nonexistent.laz", str(temp_dir))

    @patch("services.processing.point_cloud_io.PointCloudIO.load_point_cloud")
    @patch("open3d.io.write_point_cloud")
    def test_export_to_open3d_format(
        self, mock_write, mock_load_pc, temp_dir, mock_point_cloud_data
    ):
        """Test export to Open3D format."""
        from services.visualization.visualize_point_cloud import PointCloudVisualizer

        pc_file = temp_dir / "input.laz"
        pc_file.write_text("dummy")

        mock_load_pc.return_value = mock_point_cloud_data
        mock_write.return_value = True

        visualizer = PointCloudVisualizer()
        result = visualizer.export_to_open3d_format(str(pc_file), str(temp_dir))

        assert Path(result).suffix == ".ply"
        mock_load_pc.assert_called_once()
        mock_write.assert_called_once()

    @patch("services.processing.point_cloud_io.PointCloudIO.load_point_cloud")
    def test_generate_interactive_plot(
        self, mock_load_pc, temp_dir, mock_point_cloud_data
    ):
        """Test interactive plot generation."""
        from services.visualization.visualize_point_cloud import PointCloudVisualizer

        pc_file = temp_dir / "input.laz"
        pc_file.write_text("dummy")

        mock_load_pc.return_value = mock_point_cloud_data

        visualizer = PointCloudVisualizer()

        with patch("plotly.graph_objects.Scatter3d") as mock_scatter:
            with patch("plotly.offline.plot") as mock_plot:
                mock_plot.return_value = str(temp_dir / "plot.html")

                result = visualizer.generate_interactive_plot(
                    str(pc_file), str(temp_dir)
                )

                assert Path(result).suffix == ".html"
                mock_load_pc.assert_called_once()

    @patch("services.processing.point_cloud_io.PointCloudIO.load_point_cloud")
    def test_create_cross_section_view(
        self, mock_load_pc, temp_dir, mock_point_cloud_data
    ):
        """Test cross-section view creation."""
        from services.visualization.visualize_point_cloud import PointCloudVisualizer

        pc_file = temp_dir / "input.laz"
        pc_file.write_text("dummy")

        mock_load_pc.return_value = mock_point_cloud_data

        visualizer = PointCloudVisualizer()

        with patch("matplotlib.pyplot.savefig") as mock_savefig:
            result = visualizer.create_cross_section_view(
                str(pc_file), str(temp_dir), axis="z"
            )

            assert Path(result).suffix == ".png"
            mock_load_pc.assert_called_once()
            mock_savefig.assert_called_once()

    def test_create_cross_section_view_invalid_axis(self, temp_dir):
        """Test cross-section view with invalid axis."""
        from services.visualization.visualize_point_cloud import PointCloudVisualizer

        visualizer = PointCloudVisualizer()

        # Create a dummy file for the test
        dummy_file_path = temp_dir / "dummy.laz"
        dummy_file_path.touch()

        with pytest.raises(ValueError, match="Invalid axis"):
            visualizer.create_cross_section_view(
                str(dummy_file_path), str(temp_dir), axis="invalid"
            )

    @patch("services.processing.point_cloud_io.PointCloudIO.load_point_cloud")
    def test_generate_point_cloud_metrics(
        self, mock_load_pc, temp_dir, mock_point_cloud_data
    ):
        """Test point cloud metrics generation."""
        from services.visualization.visualize_point_cloud import PointCloudVisualizer

        pc_file = temp_dir / "input.laz"
        pc_file.write_text("dummy")

        mock_load_pc.return_value = mock_point_cloud_data

        visualizer = PointCloudVisualizer()
        metrics = visualizer.generate_point_cloud_metrics(str(pc_file))

        assert isinstance(metrics, dict)
        assert "point_count" in metrics
        assert "bounds" in metrics
        assert "density" in metrics
        mock_load_pc.assert_called_once()
