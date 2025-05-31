#!/usr/bin/env python3
"""
Spatial Index Builder for USGS LiDAR Datasets

This script builds a spatial index of all EPT datasets in the USGS AWS bucket
by downloading all ept.json metadata files and storing them in a format that
allows for fast spatial queries.

The index is saved as a JSON file that can be quickly loaded for spatial lookups.

Usage:
    python build_spatial_index.py [--output spatial_index.json] [--update]
"""

import json
import os
import sys
import argparse
import time
from typing import Dict, List, Optional, Tuple
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


class SpatialIndexBuilder:
    """Build a spatial index of all EPT datasets in the USGS bucket."""

    def __init__(self, max_workers: int = 20):
        """Initialize the spatial index builder."""
        self.bucket_name = "usgs-lidar-public"
        self.max_workers = max_workers

        # Create S3 client with no credentials required for public bucket
        self.s3_client = boto3.client(
            "s3", region_name="us-west-2", config=Config(signature_version=UNSIGNED)
        )

        # Thread-safe counter for progress tracking
        self.processed_count = 0
        self.lock = threading.Lock()

    def list_all_datasets(self) -> List[str]:
        """List all dataset prefixes in the S3 bucket."""
        try:
            print("Listing all datasets in USGS LiDAR bucket...")
            paginator = self.s3_client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=self.bucket_name, Delimiter="/")

            datasets = []
            for page in pages:
                if "CommonPrefixes" in page:
                    for prefix in page["CommonPrefixes"]:
                        dataset_name = prefix["Prefix"].rstrip("/")
                        datasets.append(dataset_name)

            print(f"Found {len(datasets)} datasets in AWS bucket")
            return datasets

        except Exception as e:
            print(f"Error listing datasets: {e}")
            return []

    def get_ept_metadata(self, dataset_name: str) -> Optional[Dict]:
        """Get EPT metadata for a single dataset."""
        try:
            ept_json_key = f"{dataset_name}/ept.json"
            response = self.s3_client.get_object(
                Bucket=self.bucket_name, Key=ept_json_key
            )
            metadata = json.loads(response["Body"].read().decode("utf-8"))

            # Add dataset name to metadata
            metadata["dataset_name"] = dataset_name

            return metadata

        except self.s3_client.exceptions.NoSuchKey:
            # Silently skip datasets without ept.json
            return None
        except Exception as e:
            print(f"Error getting EPT metadata for {dataset_name}: {e}")
            return None

    def process_dataset_batch(self, dataset_names: List[str]) -> List[Dict]:
        """Process a batch of datasets and return their metadata."""
        results = []

        for dataset_name in dataset_names:
            metadata = self.get_ept_metadata(dataset_name)
            if metadata and "bounds" in metadata:
                # Extract key information for spatial indexing
                dataset_info = {
                    "name": dataset_name,
                    "bounds": metadata["bounds"],
                    "points": metadata.get("points", 0),
                    "srs": metadata.get("srs", {}),
                    "dataType": metadata.get("dataType", "unknown"),
                    "schema": metadata.get("schema", []),
                }
                results.append(dataset_info)

            # Update progress counter
            with self.lock:
                self.processed_count += 1
                if self.processed_count % 100 == 0:
                    print(f"Processed {self.processed_count} datasets...")

        return results

    def build_spatial_index(self, existing_index: Optional[Dict] = None) -> Dict:
        """Build the complete spatial index."""
        print("Building spatial index for all EPT datasets...")

        # Get list of all datasets
        all_datasets = self.list_all_datasets()

        if not all_datasets:
            return {}

        # Filter out datasets we already have if updating
        datasets_to_process = all_datasets
        processed_datasets = {}

        if existing_index:
            processed_datasets = {
                item["name"]: item for item in existing_index.get("datasets", [])
            }
            datasets_to_process = [
                d for d in all_datasets if d not in processed_datasets
            ]
            print(f"Updating index: {len(datasets_to_process)} new datasets to process")

        if not datasets_to_process:
            print("No new datasets to process")
            return existing_index or {
                "datasets": [],
                "total_datasets": 0,
                "created_at": time.time(),
            }

        # Split datasets into batches for parallel processing
        batch_size = max(1, len(datasets_to_process) // self.max_workers)
        batches = [
            datasets_to_process[i : i + batch_size]
            for i in range(0, len(datasets_to_process), batch_size)
        ]

        print(
            f"Processing {len(datasets_to_process)} datasets in {len(batches)} batches using {self.max_workers} workers..."
        )

        # Process batches in parallel
        all_results = []
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_batch = {
                executor.submit(self.process_dataset_batch, batch): batch
                for batch in batches
            }

            for future in as_completed(future_to_batch):
                try:
                    batch_results = future.result()
                    all_results.extend(batch_results)
                except Exception as e:
                    print(f"Error processing batch: {e}")

        # Combine with existing data if updating
        if existing_index:
            all_results.extend(processed_datasets.values())

        end_time = time.time()
        processing_time = end_time - start_time

        print(
            f"Processed {len(datasets_to_process)} datasets in {processing_time:.2f} seconds"
        )
        print(f"Found {len(all_results)} datasets with valid EPT metadata")

        # Create spatial index structure
        spatial_index = {
            "datasets": all_results,
            "total_datasets": len(all_results),
            "created_at": time.time(),
            "processing_time_seconds": processing_time,
            "version": "1.0",
        }

        return spatial_index

    def create_geographic_grid_index(
        self, spatial_index: Dict, grid_size: float = 1.0
    ) -> Dict:
        """Create a geographic grid index for faster spatial queries."""
        print(f"Creating geographic grid index with {grid_size} degree cells...")

        grid_index = {}
        datasets = spatial_index.get("datasets", [])

        for dataset in datasets:
            bounds = dataset.get("bounds", [])
            if len(bounds) < 6:
                continue

            min_x, min_y, _, max_x, max_y, _ = bounds

            # Skip datasets with obviously projected coordinates for geographic grid
            if not (-180 <= min_x <= 180 and -90 <= min_y <= 90):
                continue

            # Calculate grid cells that this dataset intersects
            min_grid_x = int(min_x // grid_size)
            max_grid_x = int(max_x // grid_size)
            min_grid_y = int(min_y // grid_size)
            max_grid_y = int(max_y // grid_size)

            # Add dataset to all intersecting grid cells
            for grid_x in range(min_grid_x, max_grid_x + 1):
                for grid_y in range(min_grid_y, max_grid_y + 1):
                    grid_key = f"{grid_x},{grid_y}"
                    if grid_key not in grid_index:
                        grid_index[grid_key] = []
                    grid_index[grid_key].append(dataset["name"])

        spatial_index["geographic_grid"] = {
            "grid_size": grid_size,
            "cells": grid_index,
            "total_cells": len(grid_index),
        }

        print(f"Created geographic grid with {len(grid_index)} cells")
        return spatial_index


def load_existing_index(filepath: str) -> Optional[Dict]:
    """Load existing spatial index if it exists."""
    try:
        if os.path.exists(filepath):
            print(f"Loading existing index from {filepath}")
            with open(filepath, "r") as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading existing index: {e}")
    return None


def save_spatial_index(spatial_index: Dict, filepath: str) -> bool:
    """Save spatial index to file."""
    try:
        print(f"Saving spatial index to {filepath}")
        with open(filepath, "w") as f:
            json.dump(spatial_index, f, indent=2)

        # Print summary
        datasets = spatial_index.get("datasets", [])
        print(f"Saved index with {len(datasets)} datasets")

        # Print some statistics
        total_points = sum(d.get("points", 0) for d in datasets)
        print(f"Total points across all datasets: {total_points:,}")

        # Count by data type
        data_types = {}
        for dataset in datasets:
            dt = dataset.get("dataType", "unknown")
            data_types[dt] = data_types.get(dt, 0) + 1

        print("Datasets by data type:")
        for dt, count in sorted(data_types.items()):
            print(f"  {dt}: {count}")

        return True

    except Exception as e:
        print(f"Error saving spatial index: {e}")
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Build spatial index for USGS LiDAR datasets"
    )
    parser.add_argument(
        "--output",
        default="spatial_index.json",
        help="Output file path for spatial index (default: spatial_index.json)",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Update existing index instead of rebuilding from scratch",
    )
    parser.add_argument(
        "--workers", type=int, default=20, help="Number of worker threads (default: 20)"
    )
    parser.add_argument(
        "--grid-size",
        type=float,
        default=1.0,
        help="Grid cell size in degrees for geographic index (default: 1.0)",
    )

    args = parser.parse_args()

    try:
        # Load existing index if updating
        existing_index = None
        if args.update:
            existing_index = load_existing_index(args.output)

        # Build spatial index
        builder = SpatialIndexBuilder(max_workers=args.workers)
        spatial_index = builder.build_spatial_index(existing_index)

        if not spatial_index.get("datasets"):
            print("No datasets found or processed")
            return

        # Create geographic grid index
        spatial_index = builder.create_geographic_grid_index(
            spatial_index, args.grid_size
        )

        # Save to file
        if save_spatial_index(spatial_index, args.output):
            print(f"Spatial index successfully saved to {args.output}")
        else:
            print("Failed to save spatial index")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
