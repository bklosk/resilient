from pathlib import Path
import json
import types

import pytest


@pytest.fixture
def fetcher(monkeypatch):
    from scripts.get_fema_risk import FEMADataFetcher

    fetcher = FEMADataFetcher()

    class DummyNAIP:
        def calculate_acre_bbox(self, lat, lon):
            assert lat == 40.0 and lon == -105.0
            return -105.1, 39.9, -104.9, 40.1

    class DummyGeo:
        def geocode_address(self, address):
            return 40.0, -105.0

    fetcher.naip = DummyNAIP()
    fetcher.geocoder = DummyGeo()

    class DummyResp:
        def __init__(self, content):
            self.content = content
        def raise_for_status(self):
            pass
        def json(self):
            return json.loads(self.content.decode())

    def dummy_get(url, params=None, timeout=0):
        if "export" in url:
            return DummyResp(b"data")
        return DummyResp(b'{"features": [{"attributes": {"score": 1}}]}')

    monkeypatch.setattr("requests.get", dummy_get)

    return fetcher


def test_process_address(fetcher, tmp_path):
    flood_map, risk = fetcher.process_address("test", output_dir=tmp_path)
    assert Path(flood_map).exists()
    assert risk == {"score": 1}
