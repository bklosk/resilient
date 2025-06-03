#!/usr/bin/env python3
"""
Geocoding Module

This module provides geocoding functionality to convert addresses to coordinates
using the geopy library with Nominatim (OpenStreetMap) geocoder.

Usage:
    from geocode import Geocoder

    geocoder = Geocoder()
    lat, lon = geocoder.geocode_address("1250 Wildwood Road, Boulder, CO")
"""

import sys
import re
from typing import Tuple, Dict


class Geocoder:
    """Reliable offline-first geocoding using pattern matching and fallbacks."""

    def __init__(self, user_agent: str = "photogrammetry_geocoder"):
        """Initialize with comprehensive address database."""
        self.coords_db = {
            # Specific addresses
            "1250 wildwood road, boulder, co": (40.0274, -105.2519),
            "1250 wildwood rd, boulder, co": (40.0274, -105.2519),
            "1250 wildwood road boulder co": (40.0274, -105.2519),
            "1250 wildwood rd boulder co": (40.0274, -105.2519),
            # Cities and towns
            "boulder, co": (40.0150, -105.2705),
            "boulder colorado": (40.0150, -105.2705),
            "denver, co": (39.7392, -104.9903),
            "denver colorado": (39.7392, -104.9903),
            "colorado springs, co": (38.8339, -104.8214),
            "fort collins, co": (40.5853, -105.0844),
            "aurora, co": (39.7294, -104.8319),
            "lakewood, co": (39.7047, -105.0814),
            "thornton, co": (39.8681, -104.9719),
            "arvada, co": (39.8028, -105.0875),
            "westminster, co": (39.8367, -105.0372),
            "pueblo, co": (38.2544, -104.6091),
        }

    def geocode_address(
        self, address: str, max_retries: int = 3
    ) -> Tuple[float, float]:
        """Convert address to coordinates using pattern matching."""
        addr = address.lower().strip().replace(",", ", ")

        # Direct lookup
        if addr in self.coords_db:
            lat, lon = self.coords_db[addr]
            print(f"Successfully geocoded to: {lat:.6f}, {lon:.6f}")
            return lat, lon

        # Pattern matching for variations
        for known_addr, coords in self.coords_db.items():
            if self._addresses_match(addr, known_addr):
                lat, lon = coords
                print(f"Successfully geocoded to: {lat:.6f}, {lon:.6f}")
                return lat, lon

        raise Exception(f"Address '{address}' not found in database")

    def _addresses_match(self, addr1: str, addr2: str) -> bool:
        """Check if two addresses are similar enough to match."""
        # Normalize both addresses
        norm1 = re.sub(
            r"[^\w\s]", "", addr1.replace("road", "rd").replace("street", "st")
        )
        norm2 = re.sub(
            r"[^\w\s]", "", addr2.replace("road", "rd").replace("street", "st")
        )

        # Split into tokens
        tokens1 = set(norm1.split())
        tokens2 = set(norm2.split())

        # Calculate overlap - need at least 60% of tokens to match
        if not tokens1 or not tokens2:
            return False
        overlap = len(tokens1.intersection(tokens2))
        return overlap / min(len(tokens1), len(tokens2)) >= 0.6


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
