#!/usr/bin/env python3
"""Fetch FEMA flood map and climate disaster risk for an address.

This module downloads a FEMA flood hazard map image for the same
bounding box used when fetching orthophotos and queries the FEMA
National Risk Index for risk attributes.
"""

import os
import json
from typing import Tuple, Dict

import requests

from .get_orthophoto import NAIPFetcher
from ..utils.utils import GeocodeUtils


class FEMADataFetcher:
    """Retrieve FEMA flood map image and National Risk Index data."""

    FLOOD_SERVICE_URL = "https://msc.fema.gov/arcgis/rest/services/NFHL/DFIRM_Flood_Hazard/MapServer/export"
    NRI_SERVICE_URL = (
        "https://hazards.fema.gov/gis/nrimap/rest/services/NRI/MapServer/0/query"
    )

    def __init__(self):
        self.naip = NAIPFetcher()
        self.geocoder = GeocodeUtils()

    def _bbox_string(
        self, lat: float, lon: float
    ) -> Tuple[str, Tuple[float, float, float, float]]:
        min_lon, min_lat, max_lon, max_lat = self.naip.calculate_acre_bbox(lat, lon)
        bbox = f"{min_lon},{min_lat},{max_lon},{max_lat}"
        return bbox, (min_lon, min_lat, max_lon, max_lat)

    def fetch_flood_map(self, lat: float, lon: float, output_dir: str) -> str:
        bbox_str, _ = self._bbox_string(lat, lon)
        params = {
            "bbox": bbox_str,
            "bboxSR": 4326,
            "size": "1024,1024",
            "imageSR": 4326,
            "format": "png",
            "transparent": "true",
            "f": "image",
        }
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, "fema_flood_map.png")
        resp = requests.get(self.FLOOD_SERVICE_URL, params=params, timeout=60)
        resp.raise_for_status()
        with open(out_path, "wb") as f:
            f.write(resp.content)
        return out_path

    def fetch_risk_data(self, lat: float, lon: float) -> Dict:
        params = {
            "geometry": f"{lon},{lat}",
            "geometryType": "esriGeometryPoint",
            "inSR": 4326,
            "outFields": "*",
            "f": "json",
        }
        resp = requests.get(self.NRI_SERVICE_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if data.get("features"):
            return data["features"][0].get("attributes", {})
        return {}

    def process_address(
        self, address: str, output_dir: str = "../data"
    ) -> Tuple[str, Dict]:
        # Validate address is not empty
        if not address or not address.strip():
            raise ValueError("Address cannot be empty")

        lat, lon = self.geocoder.geocode_address(address)
        flood_map = self.fetch_flood_map(lat, lon, output_dir)
        risk = self.fetch_risk_data(lat, lon)
        meta = {
            "address": address,
            "coordinates": {"latitude": lat, "longitude": lon},
            "flood_map": flood_map,
            "risk_data": risk,
        }
        meta_path = os.path.join(output_dir, "fema_risk_metadata.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        return flood_map, risk


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Download FEMA flood map and risk data for an address"
    )
    parser.add_argument("address", help="Street address")
    parser.add_argument(
        "--output-dir", default="../data", help="Directory for downloaded files"
    )

    args = parser.parse_args()
    fetcher = FEMADataFetcher()
    flood_map, risk = fetcher.process_address(args.address, args.output_dir)
    print(f"Flood map saved to: {flood_map}")
    if risk:
        print("Risk data:\n" + json.dumps(risk, indent=2))
    else:
        print("No risk data returned")


if __name__ == "__main__":
    main()
