import sys
import importlib
import threading
import time
from datetime import datetime
from pathlib import Path
import types

import pytest
import requests
import uvicorn

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

services_dir = REPO_ROOT / "services"
if str(services_dir) not in sys.path:
    sys.path.insert(0, str(services_dir))


@pytest.fixture(scope="module")
def dev_server(tmp_path_factory):
    """Start the FastAPI dev server in a background thread."""
    original_modules = sys.modules.copy()

    stub_process = types.ModuleType("process_point_cloud")

    class DummyColorizer:
        def __init__(self, output_dir=".", create_diagnostics=False):
            self.output_dir = output_dir

        def process_from_address(self, address):
            output = tmp_path_factory.mktemp("out") / "dummy.laz"
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
            output = tmp_path_factory.mktemp("out") / "dummy.tif"
            output.write_text("img")
            return str(output), {"output_path": str(output)}

    stub_orth.NAIPFetcher = DummyFetcher

    stub_modules = {
        "services.core.process_point_cloud": stub_process,
        "services.core.geocode": stub_geocode,
        "services.data.get_point_cloud": stub_pc,
        "services.data.get_orthophoto": stub_orth,
        "process_point_cloud": stub_process,
        "geocode": stub_geocode,
        "get_point_cloud": stub_pc,
        "get_orthophoto": stub_orth,
    }

    for name, module in stub_modules.items():
        sys.modules[name] = module

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

    app_module.process_point_cloud_background = immediate_background

    config = uvicorn.Config(app_module.app, host="127.0.0.1", port=8001, log_level="info")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    timeout = time.time() + 10
    while not server.started and time.time() < timeout:
        time.sleep(0.1)
    if not server.started:
        server.should_exit = True
        thread.join()
        raise RuntimeError("Server failed to start")

    yield "http://127.0.0.1:8001"

    server.should_exit = True
    thread.join()
    sys.modules.clear()
    sys.modules.update(original_modules)


def test_dev_server_health(dev_server):
    resp = requests.get(f"{dev_server}/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


def test_dev_server_process_flow(dev_server):
    resp = requests.post(f"{dev_server}/process", json={"address": "123 Main St"})
    assert resp.status_code == 200
    body = resp.json()
    job_id = body["job_id"]
    # Poll until the background job completes
    for _ in range(20):
        status_resp = requests.get(f"{dev_server}/job/{job_id}")
        assert status_resp.status_code == 200
        if status_resp.json()["status"] == "completed":
            break
        time.sleep(0.1)
    else:
        pytest.fail("Job did not complete in time")

    download = requests.get(f"{dev_server}/download/{job_id}")
    assert download.status_code == 200
    assert download.headers["content-type"] == "application/octet-stream"


def test_dev_server_orthophoto(dev_server):
    resp = requests.post(f"{dev_server}/orthophoto", json={"address": "789 Low St"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("image/")
