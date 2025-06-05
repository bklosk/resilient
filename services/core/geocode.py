#!/usr/bin/env python3
"""
Geocoding Module

This module provides reliable geocoding functionality using multiple geocoding services
with fallback support for maximum reliability.

Usage:
    from geocode import Geocoder

    geocoder = Geocoder()
    lat, lon = geocoder.geocode_address("1250 Wildwood Road, Boulder, CO")
"""

import sys
import time
from typing import Tuple, Optional
from geopy.geocoders import Photon, ArcGIS, GoogleV3
from geopy.exc import GeopyError


class Geocoder:
    """Reliable geocoding using multiple services with fallback support."""

    def __init__(self, user_agent: str = "photogrammetry_geocoder"):
        """Initialize geocoder with multiple fallback services."""
        self.user_agent = user_agent
        # Multiple geocoding services in order of preference
        self.geocoders = [
            Photon(user_agent=self.user_agent, timeout=10),  # Free, no API key needed
            ArcGIS(timeout=10),  # Free tier, no API key needed
            # GoogleV3 would require API key, so commented out
            # GoogleV3(api_key="your_api_key", timeout=10),
        ]

    def geocode_address(
        self, address: str, max_retries: int = 3
    ) -> Tuple[float, float]:
        """Convert address to coordinates using multiple geocoding services."""
        last_error: Optional[str] = None

        for geocoder in self.geocoders:
            for attempt in range(max_retries):
                try:
                    print(f"Attempting geocoding with {geocoder.__class__.__name__}...")

                    # Use appropriate parameters for each geocoder
                    if geocoder.__class__.__name__ == "Photon":
                        location = geocoder.geocode(address)
                    elif geocoder.__class__.__name__ == "ArcGIS":
                        location = geocoder.geocode(address)
                    else:
                        # Default case for other geocoders
                        location = geocoder.geocode(address)

                    if location:
                        lat, lon = float(location.latitude), float(location.longitude)
                        print(f"Successfully geocoded to: {lat:.6f}, {lon:.6f}")
                        return lat, lon
                    else:
                        last_error = f"{geocoder.__class__.__name__}: No results found"

                except GeopyError as e:
                    last_error = f"{geocoder.__class__.__name__}: {str(e)}"
                    print(f"Geocoding attempt {attempt + 1} failed: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(1)  # Brief delay between retries
                except Exception as e:
                    last_error = f"{geocoder.__class__.__name__}: {str(e)}"
                    print(f"Unexpected error: {e}")
                    break  # Don't retry on unexpected errors

        raise Exception(f"All geocoding services failed. Last error: {last_error}")

    def _extract_state(self, addr: str) -> str:
        """Extract state abbreviation or name."""
        for state_key in self.state_centers.keys():
            if state_key in addr:
                return state_key
        return ""

    def _extract_city(self, addr: str) -> str:
        """Extract city name from address."""
        # Common patterns: "city, state" or "city state"
        parts = re.split(r"[,\s]+", addr)
        for i, part in enumerate(parts):
            if part in self.state_centers and i > 0:
                return parts[i - 1]

        # If no state found, try common city names
        cities = ["boulder", "denver", "aurora", "lakewood", "thornton"]
        for city in cities:
            if city in addr:
                return city
        return ""

    def _extract_street_number(self, addr: str) -> int:
        """Extract street number from address."""
        numbers = re.findall(r"\b\d+\b", addr)
        return int(numbers[0]) if numbers else 0

    def _extract_street_name(self, addr: str) -> str:
        """Extract street name from address."""
        # Remove numbers and common suffixes, get remaining text
        cleaned = re.sub(r"\b\d+\b", "", addr)
        cleaned = re.sub(
            r"\b(st|street|rd|road|ave|avenue|blvd|boulevard|dr|drive|ln|lane|way|ct|court|pl|place)\b",
            "",
            cleaned,
        )
        cleaned = re.sub(
            r"\b(co|colorado|ca|california|tx|texas|ny|new york|fl|florida)\b",
            "",
            cleaned,
        )
        words = [w for w in cleaned.split() if len(w) > 2]
        return " ".join(words[:2]) if words else ""


def geocode_address(
    address: str, user_agent: str = "photogrammetry_geocoder"
) -> Tuple[float, float]:
    """
    Convenience function to geocode an address.

    Args:
        address: The address string to geocode
        user_agent: User agent string for the geocoding service

    Returns:
        Tuple of (latitude, longitude) in decimal degrees
    """
    geocoder = Geocoder(user_agent=user_agent)
    return geocoder.geocode_address(address)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python geocode.py <address>")
        sys.exit(1)

    address = sys.argv[1]
    try:
        lat, lon = geocode_address(address)
        print(f"Address: {address}")
        print(f"Coordinates: {lat:.6f}, {lon:.6f}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
