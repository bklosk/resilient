#!/usr/bin/env python3
import os, sys, json, requests
from geopy.geocoders import Photon
from geopy.distance import geodesic


def sqft_from_osm(lat, lon):
    q = f"[out:json];way['building'](around:50,{lat},{lon});out geom;"
    r = requests.get("https://overpass-api.de/api/interpreter", params={"data": q})
    if r.ok and r.json().get("elements"):
        g = r.json()["elements"][0]["geometry"]
        l = [p["lat"] for p in g]
        w = [p["lon"] for p in g]
        h = geodesic((min(l), min(w)), (max(l), min(w))).meters
        d = geodesic((min(l), min(w)), (min(l), max(w))).meters
        return h * d * 10.7639


class ReplacementValueEstimator:
    """Estimates property replacement value based on location and property characteristics."""

    def __init__(self):
        """Initialize the replacement value estimator."""
        from utils import GeocodeUtils

        self.geocoder = GeocodeUtils()
        self.default_cost_per_sqft = 200.0  # Default cost per square foot

    def estimate_property_value(
        self, address: str, sqft: float = None, **kwargs
    ) -> dict:
        """
        Estimate property replacement value for a given address.

        Args:
            address: Street address to analyze
            sqft: Property square footage (if None, will estimate from OSM)
            **kwargs: Additional parameters like cost_per_sqft, location_factor, etc.

        Returns:
            Dictionary containing property value estimation

        Raises:
            ValueError: If address is empty
        """
        if not address or not address.strip():
            raise ValueError("Address cannot be empty")

        try:
            # Get coordinates for the address
            lat, lon = self.geocoder.geocode_address(address)

            # Use provided square footage or estimate from OSM
            if sqft is None:
                sqft = sqft_from_osm(lat, lon) or 2300  # Default fallback

            # Get cost parameters
            cost_per_sqft = kwargs.get("cost_per_sqft", self.default_cost_per_sqft)
            location_factor = kwargs.get("location_factor", 1.0)
            area_adjustment = kwargs.get("area_adjustment", 1.0)

            # Calculate replacement value
            base_value = sqft * cost_per_sqft
            adjusted_value = base_value * location_factor * area_adjustment

            return {
                "address": address,
                "coordinates": {"latitude": lat, "longitude": lon},
                "square_footage": sqft,
                "cost_per_sqft": cost_per_sqft,
                "base_value": base_value,
                "location_factor": location_factor,
                "area_adjustment": area_adjustment,
                "replacement_value": adjusted_value,
                "calculation_date": "2024-01-01",
            }
        except Exception as e:
            if "Address cannot be empty" in str(e):
                raise
            raise RuntimeError(f"Failed to estimate property value: {e}")


def main():
    addr = " ".join(sys.argv[1:]) or input("Address: ")
    loc = Photon(user_agent="rv").geocode(addr)
    sqft = sqft_from_osm(loc.latitude, loc.longitude) if loc else None
    sqft = sqft or 2300
    cost = float(os.getenv("COST_PER_SQFT", 200))
    print(
        json.dumps(
            {"address": addr, "sqft": sqft, "replacement_value": sqft * cost}, indent=2
        )
    )


if __name__ == "__main__":
    main()
