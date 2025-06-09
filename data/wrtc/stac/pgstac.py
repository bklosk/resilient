from pystac import Catalog
import json
import glob

# Load the root catalog
catalog = Catalog.from_file("/workspaces/photogrammetry/data/wrtc/stac_catalog/catalog.json")

# Extract collections
collections = list(catalog.get_all_collections())
with open("collections.ndjson", "w") as f:
    for col in collections:
        f.write(json.dumps(col.to_dict()) + "\n")

# Extract items
items = []
for col in collections:
    items.extend(col.get_all_items())
with open("items.ndjson", "w") as f:
    for item in items:
        f.write(json.dumps(item.to_dict()) + "\n")
