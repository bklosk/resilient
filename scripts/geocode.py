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
from typing import Tuple
from utils import GeocodeUtils


class Geocoder:
    """Class to handle address geocoding using Nominatim - streamlined version."""

    def __init__(self, user_agent: str = "photogrammetry_geocoder"):
        """
        Initialize the geocoder.

        Args:
            user_agent: User agent string for the geocoding service
        """
        self.geocode_utils = GeocodeUtils()

    def geocode_address(
        self, address: str, max_retries: int = 3
    ) -> Tuple[float, float]:
        """
        Convert an address to latitude and longitude coordinates.

        Args:
            address: The address string to geocode
            max_retries: Maximum number of retry attempts

        Returns:
            Tuple of (latitude, longitude) in decimal degrees

        Raises:
            Exception: If geocoding fails after all retries
        """
        try:
            lat, lon = self.geocode_utils.geocode_address(address)
            print(f"Successfully geocoded to: {lat:.6f}, {lon:.6f}")
            return lat, lon
        except ValueError as e:
            raise Exception(str(e))


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
