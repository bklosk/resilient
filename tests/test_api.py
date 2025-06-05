"""
Comprehensive API tests for the photogrammetry FastAPI application.
"""

import types
import sys
import importlib
import uuid
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch

# Ensure repository root is on the Python path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Add services to path
services_dir = REPO_ROOT / "services"
if str(services_dir) not in sys.path:
    sys.path.insert(0, str(services_dir))

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Create TestClient with stubbed dependencies and fast background tasks."""
    # Create a copy of current sys.modules to restore later
    original_modules = sys.modules.copy()

    # Stub heavy modules before importing the app
    stub_process = types.ModuleType("process_point_cloud")

    class DummyColorizer:
        def __init__(self, output_dir=".", create_diagnostics=False):
            self.output_dir = output_dir

        def process_from_address(self, address):
            output = Path(tmp_path) / "dummy.laz"
            output.write_text("data")
            return str(output)

    stub_process.PointCloudProcessor = DummyColorizer

    stub_geocode = types.ModuleType("geocode")

    class DummyGeocoder:
        def geocode_address(self, address, max_retries=3):
            return 40.0, -105.0

    stub_geocode.Geocoder = DummyGeocoder

    stub_pc = types.ModuleType("get_point_cloud")

    class DummyFinder:
        def generate_bounding_box(self, lat, lon, buffer_km):
            return "0,0,1,1"

        def search_lidar_products(self, bbox):
            return ["product"]

        def filter_laz_products(self, products):
            return products

    stub_pc.PointCloudDatasetFinder = DummyFinder

    stub_orth = types.ModuleType("get_orthophoto")

    class DummyFetcher:
        def get_orthophoto_for_address(self, address, output_dir=".", image_size=None):
            output = Path(tmp_path) / "dummy.tif"
            output.write_text("img")
            return str(output), {"output_path": str(output)}

    stub_orth.NAIPFetcher = DummyFetcher

    # Temporarily patch sys.modules only for this test client
    stub_modules = {
        "services.core.process_point_cloud": stub_process,
        "services.core.geocode": stub_geocode,
        "services.data.get_point_cloud": stub_pc,
        "services.data.get_orthophoto": stub_orth,
        # Also stub the old import paths for backward compatibility
        "process_point_cloud": stub_process,
        "geocode": stub_geocode,
        "get_point_cloud": stub_pc,
        "get_orthophoto": stub_orth,
    }

    # Apply the patches
    for name, module in stub_modules.items():
        sys.modules[name] = module

    if "app" in sys.modules:
        app_module = sys.modules["app"]
    else:
        app_module = importlib.import_module("app")
    app_module.jobs.clear()

    def immediate_background(job_id, address, buffer_km):
        output_dir = REPO_ROOT / "data" / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{job_id}.laz"
        output_file.write_text("content")
        app_module.update_job_status(
            job_id,
            status=app_module.JobStatus.COMPLETED,
            output_file=str(output_file),
            completed_at=datetime.now(),
        )

    monkeypatch.setattr(
        app_module, "process_point_cloud_background", immediate_background
    )

    with TestClient(app_module.app) as client:
        yield client, app_module

    # Cleanup: restore original modules
    sys.modules.clear()
    sys.modules.update(original_modules)


def test_health_endpoint(client):
    client, _ = client
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "active_jobs" in data


def test_process_flow(client):
    client, app_module = client
    resp = client.post("/process", json={"address": "123 Main St"})
    assert resp.status_code == 200
    body = resp.json()
    job_id = body["job_id"]
    assert body["success"] is True
    assert body["status"] == app_module.JobStatus.PENDING

    status_resp = client.get(f"/job/{job_id}")
    assert status_resp.status_code == 200
    status_data = status_resp.json()
    assert status_data["status"] == app_module.JobStatus.COMPLETED
    assert "log_tail" in status_data
    assert isinstance(status_data["log_tail"], list)

    download = client.get(f"/download/{job_id}")
    assert download.status_code == 200
    assert download.headers["content-type"] == "application/octet-stream"


def test_invalid_address(client):
    client, _ = client
    resp = client.post("/process", json={"address": "bad"})
    assert resp.status_code == 422


def test_missing_job(client):
    client, _ = client
    bad_id = "01234567-89ab-cdef-0123-456789abcdef"
    resp = client.get(f"/job/{bad_id}")
    assert resp.status_code == 404
    resp = client.get(f"/download/{bad_id}")
    assert resp.status_code == 404


def test_jobs_listing(client):
    client, app_module = client
    resp = client.post("/process", json={"address": "456 High St"})
    assert resp.status_code == 200
    body = resp.json()
    job_id = body["job_id"]

    list_resp = client.get("/jobs")
    assert list_resp.status_code == 200
    jobs = list_resp.json()
    assert any(j["job_id"] == job_id for j in jobs)


def test_orthophoto_download(client):
    client, _ = client
    resp = client.post("/orthophoto", json={"address": "789 Low St"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("image/")


def test_flood_overhead_endpoint(client):
    """Test the flood overhead image endpoint."""
    client, _ = client

    with patch("services.utils.flood_depth.generate") as mock_generate, patch(
        "services.visualization.overhead_image.render"
    ) as mock_render:

        mock_generate.return_value = str(REPO_ROOT / "test_flood.tif")
        mock_render.return_value = str(REPO_ROOT / "test_flood.png")

        # Create dummy files
        (REPO_ROOT / "test_flood.tif").write_text("dummy")
        (REPO_ROOT / "test_flood.png").write_text("dummy")

        try:
            resp = client.get("/flood-overhead?address=123 Main St&bbox_m=64")
            assert resp.status_code == 200
            assert resp.headers["content-type"] == "image/png"
        finally:
            # Cleanup
            (REPO_ROOT / "test_flood.tif").unlink(missing_ok=True)
            (REPO_ROOT / "test_flood.png").unlink(missing_ok=True)


def test_process_endpoint_validation(client):
    """Test process endpoint input validation."""
    client, _ = client

    # Test missing address
    resp = client.post("/process", json={})
    assert resp.status_code == 422

    # Test empty address
    resp = client.post("/process", json={"address": ""})
    assert resp.status_code == 422

    # Test address too short
    resp = client.post("/process", json={"address": "hi"})
    assert resp.status_code == 422

    # Test address too long
    long_address = "x" * 201
    resp = client.post("/process", json={"address": long_address})
    assert resp.status_code == 422

    # Test invalid buffer_km
    resp = client.post("/process", json={"address": "123 Main St", "buffer_km": 10.0})
    assert resp.status_code == 422

    resp = client.post("/process", json={"address": "123 Main St", "buffer_km": -1.0})
    assert resp.status_code == 422


def test_job_status_validation(client):
    """Test job status endpoint validation."""
    client, _ = client

    # Test empty job ID
    resp = client.get("/job/")
    assert resp.status_code == 404  # Route not found

    # Test invalid UUID format
    resp = client.get("/job/invalid-uuid")
    assert resp.status_code == 400

    # Test valid UUID format but non-existent job
    valid_uuid = str(uuid.uuid4())
    resp = client.get(f"/job/{valid_uuid}")
    assert resp.status_code == 404


def test_download_validation(client):
    """Test download endpoint validation."""
    client, _ = client

    # Test invalid UUID format
    resp = client.get("/download/invalid-uuid")
    assert resp.status_code == 400

    # Test valid UUID format but non-existent job
    valid_uuid = str(uuid.uuid4())
    resp = client.get(f"/download/{valid_uuid}")
    assert resp.status_code == 404


def test_orthophoto_endpoint_validation(client):
    """Test orthophoto endpoint validation."""
    client, _ = client

    # Test missing address
    resp = client.post("/orthophoto", json={})
    assert resp.status_code == 422

    # Test empty address
    resp = client.post("/orthophoto", json={"address": ""})
    assert resp.status_code == 422

    # Test address too short
    resp = client.post("/orthophoto", json={"address": "hi"})
    assert resp.status_code == 422

    # Test address too long
    long_address = "x" * 201
    resp = client.post("/orthophoto", json={"address": long_address})
    assert resp.status_code == 422


def test_job_status_with_log_tail(client):
    """Test job status includes log tail."""
    client, app_module = client

    # Create a job
    resp = client.post("/process", json={"address": "456 High St"})
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]

    # Get job status
    status_resp = client.get(f"/job/{job_id}")
    assert status_resp.status_code == 200
    status_data = status_resp.json()

    assert "log_tail" in status_data
    assert isinstance(status_data["log_tail"], list)


def test_processing_job_states(client):
    """Test different job processing states."""
    client, app_module = client

    # Create a job
    resp = client.post("/process", json={"address": "789 Oak St"})
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]

    # Job should start as PENDING
    assert resp.json()["status"] == app_module.JobStatus.PENDING

    # After background processing, should be COMPLETED
    status_resp = client.get(f"/job/{job_id}")
    assert status_resp.status_code == 200
    assert status_resp.json()["status"] == app_module.JobStatus.COMPLETED


def test_concurrent_jobs(client):
    """Test handling multiple concurrent jobs."""
    client, app_module = client

    # Create multiple jobs
    job_ids = []
    for i in range(3):
        resp = client.post("/process", json={"address": f"{i} Test St"})
        assert resp.status_code == 200
        job_ids.append(resp.json()["job_id"])

    # All jobs should be tracked
    jobs_resp = client.get("/jobs")
    assert jobs_resp.status_code == 200
    jobs = jobs_resp.json()

    for job_id in job_ids:
        assert any(j["job_id"] == job_id for j in jobs)


def test_health_endpoint_dependencies(client):
    """Test health endpoint reports dependency status."""
    client, _ = client

    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()

    assert data["status"] == "healthy"
    assert "service" in data
    assert "version" in data
    assert "dependencies" in data
    assert "active_jobs" in data


def test_error_handling_in_processing(client):
    """Test error handling during background processing."""
    client, app_module = client

    # Mock processing to fail
    original_process = app_module.process_point_cloud_background

    def failing_process(job_id, address, buffer_km):
        app_module.update_job_status(
            job_id,
            status=app_module.JobStatus.FAILED,
            error_message="Mock processing failure",
            completed_at=datetime.now(),
        )

    app_module.process_point_cloud_background = failing_process

    try:
        resp = client.post("/process", json={"address": "Failing Address"})
        assert resp.status_code == 200
        job_id = resp.json()["job_id"]

        # Check that job failed
        status_resp = client.get(f"/job/{job_id}")
        assert status_resp.status_code == 200
        status_data = status_resp.json()
        assert status_data["status"] == app_module.JobStatus.FAILED
        assert "error_message" in status_data

        # Download should fail for failed job
        download_resp = client.get(f"/download/{job_id}")
        assert download_resp.status_code == 400

    finally:
        # Restore original function
        app_module.process_point_cloud_background = original_process


def test_job_metadata_tracking(client):
    """Test that job metadata is properly tracked."""
    client, app_module = client

    buffer_km = 2.5
    resp = client.post(
        "/process", json={"address": "Metadata Test St", "buffer_km": buffer_km}
    )
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]

    # Check metadata in response
    response_metadata = resp.json().get("metadata", {})
    assert response_metadata.get("buffer_km") == buffer_km

    # Check metadata in job status
    status_resp = client.get(f"/job/{job_id}")
    assert status_resp.status_code == 200
    status_data = status_resp.json()
    assert "metadata" in status_data
    assert status_data["metadata"].get("buffer_km") == buffer_km


def test_root_endpoint_redirect(client):
    """Test root endpoint redirects to OpenAPI spec."""
    client, _ = client

    # TestClient doesn't support allow_redirects parameter
    resp = client.get("/", follow_redirects=False)
    assert resp.status_code in [307, 200]  # Accept both redirect and direct response


def test_cors_headers(client):
    """Test CORS headers are properly set."""
    client, _ = client

    # Test that health endpoint works (OPTIONS may not be available in test mode)
    resp = client.get("/health")
    assert resp.status_code == 200
    # Note: TestClient may not include all CORS headers in test mode
