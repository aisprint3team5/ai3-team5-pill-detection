import re

def extract_category_ids_from_filename(filename: str) -> list[int]:
    """Extract category IDs from a filename like K-aaaa-bbbb-cccc-dddd_x_y_z.png"""
    base = filename.split("_")[0]  # Remove trailing cam angle, size etc.
    parts = base.split("-")
    if len(parts) < 5:
        return []
    try:
        return [int(p) for p in parts[1:5]]
    except ValueError:
        return []
import os
import json
from collections import defaultdict

def find_missing_annotations(ann_dir, img_dir) -> dict:
    missing_annotations = defaultdict(list)

    for root, _, files in os.walk(ann_dir):
        for file in files:
            if not file.endswith(".json"):
                continue

            json_path = os.path.join(root, file)
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                print(f"[ERROR] Could not load {json_path}: {e}")
                continue

            # Extract actual category_ids from filename
            if not data.get("images") or not data.get("annotations"):
                continue

            imgfile = data["images"][0].get("imgfile")
            expected_cat_ids = extract_category_ids_from_filename(imgfile)

            # Extract actual category_ids from annotation
            actual_cat_ids = set()
            for ann in data["annotations"]:
                cat_id = ann.get("category_id")
                if cat_id is not None:
                    actual_cat_ids.add(cat_id)

            # Find missing category_ids
            for cat_id in expected_cat_ids:
                if cat_id not in actual_cat_ids:
                    missing_annotations[imgfile].append(cat_id)

    return missing_annotations
ann_dir = "/path/to/jsons"
img_dir = "/path/to/images"

missing = find_missing_annotations(ann_dir, img_dir)

print("===== Missing Annotations Summary =====")
for imgfile, missing_ids in missing.items():
    print(f"{imgfile}: Missing {missing_ids}")
print(f"\nTotal images with missing annotations: {len(missing)}")
