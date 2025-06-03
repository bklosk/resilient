import sys
from pathlib import Path
from unittest import mock
from geopy.exc import GeopyError

# Ensure repository root is on the Python path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import utils


def test_generate_and_validate_bbox():
    buffer_km = 1.0
    bbox = utils.BoundingBoxUtils.generate_bounding_box(40.0, -105.0, buffer_km=buffer_km)
    assert utils.BoundingBoxUtils.validate_bounding_box(bbox)
    parts = list(map(float, bbox.split(',')))
    assert len(parts) == 4
    buffer_deg = buffer_km / 111.0
    expected_min_lon = -105.0 - buffer_deg
    expected_max_lon = -105.0 + buffer_deg
    assert abs(parts[0] - expected_min_lon) < 1e-6
    assert abs(parts[2] - expected_max_lon) < 1e-6


def test_validate_bbox_invalid():
    assert not utils.BoundingBoxUtils.validate_bounding_box('')
    assert not utils.BoundingBoxUtils.validate_bounding_box('1,2,3')
    assert not utils.BoundingBoxUtils.validate_bounding_box('200,0,201,1')


def test_safe_filename():
    name = utils.FileUtils.get_safe_filename('123 Main St., Boulder, CO')
    assert ' ' not in name
    assert ',' not in name
    assert name.startswith('123_main')


def test_geocode_fallback(monkeypatch):
    g = utils.GeocodeUtils()
    # Simulate failure of the first geocoder
    monkeypatch.setattr(
        g.geocoders[0],
        "geocode",
        mock.Mock(side_effect=GeopyError("fail")),
    )

    # Provide a successful response from the fallback geocoder
    mock_location = mock.Mock(latitude=40.0274, longitude=-105.2519)
    monkeypatch.setattr(
        g.geocoders[1],
        "geocode",
        mock.Mock(return_value=mock_location),
    )

    lat, lon = g.geocode_address("1250 Wildwood Road, Boulder, CO", max_retries=1)
    assert abs(lat - 40.0274) < 1e-4
    assert abs(lon - (-105.2519)) < 1e-4


def test_job_helpers():
    from api import app as app_module

    job_id = '1234'
    app_module.jobs[job_id] = app_module.Job(
        job_id=job_id,
        address='test',
        status=app_module.JobStatus.PENDING,
        created_at=app_module.datetime.now(),
    )
    updated = app_module.update_job_status(job_id, status=app_module.JobStatus.COMPLETED)
    assert updated
    job = app_module.get_job_safe(job_id)
    assert job.status == app_module.JobStatus.COMPLETED
