import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import numpy as np

# Ensure repository root on path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import utils


def test_detect_point_cloud_crs_wgs84():
    las = SimpleNamespace(x=[-105.0, -104.9], y=[40.0, 40.1])
    assert utils.CRSUtils.detect_point_cloud_crs(las) == "EPSG:4326"


def test_transform_coordinates_identity():
    x = np.array([-105.0, -104.9])
    y = np.array([40.0, 40.1])
    tx, ty = utils.CRSUtils.transform_coordinates(x, y, "EPSG:4326", "EPSG:4326")
    assert np.allclose(tx, x)
    assert np.allclose(ty, y)


def test_file_utils_validation(monkeypatch, tmp_path):
    f = tmp_path / "cloud.las"
    f.write_bytes(b"data")
    dummy_las = SimpleNamespace(points=[1, 2, 3])
    monkeypatch.setattr(utils.laspy, "read", mock.Mock(return_value=dummy_las))
    assert utils.FileUtils.validate_point_cloud_file(str(f))

    invalid = tmp_path / "cloud.txt"
    invalid.write_text("text")
    assert not utils.FileUtils.validate_point_cloud_file(str(invalid))


def test_json_utils_roundtrip(tmp_path):
    data = {"a": 1}
    path = utils.JSONUtils.save_metadata(data, str(tmp_path), "meta.json")
    loaded = utils.JSONUtils.load_json(path)
    assert loaded == data


def test_http_utils_post_and_get(monkeypatch):
    response_mock = mock.Mock()
    response_mock.json.return_value = {"ok": True}
    response_mock.raise_for_status.return_value = None

    monkeypatch.setattr(utils.requests, "post", mock.Mock(return_value=response_mock))
    monkeypatch.setattr(utils.requests, "get", mock.Mock(return_value=response_mock))

    assert utils.HTTPUtils.post_json("http://x", {}) == {"ok": True}
    assert utils.HTTPUtils.get_json("http://x") == {"ok": True}

