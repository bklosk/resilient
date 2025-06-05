import types
import sys
import importlib
from pathlib import Path
from datetime import datetime

# Ensure repository root is on the Python path so "api" can be imported
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Create TestClient with stubbed dependencies and fast background tasks."""
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

    for name, module in [
        ("services.core.process_point_cloud", stub_process),
        ("services.core.geocode", stub_geocode),
        ("services.data.get_point_cloud", stub_pc),
        ("services.data.get_orthophoto", stub_orth),
        # Also stub the old import paths for backward compatibility
        ("process_point_cloud", stub_process),
        ("geocode", stub_geocode),
        ("get_point_cloud", stub_pc),
        ("get_orthophoto", stub_orth),
    ]:
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
