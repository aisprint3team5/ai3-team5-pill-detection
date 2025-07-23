import os
import json
import pickle
from collections import defaultdict
from utils.Dataset.dataloader import is_valid_annotation
class PillDatasetParser:
    def __init__(self, img_dir, ann_dir):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.dataset = []
        self.image_id_map = {}
        self.image_counter = 0
        self.annotation_counter = 0
        self.not_found = 0
    #@TODO
    #Convert label to json format

    def parse(self):
        imgfile_to_jsons = defaultdict(list)

        for root, _, files in os.walk(self.ann_dir):
            for file in files:
                if not file.endswith(".json"):
                    continue
                json_path = os.path.join(root, file)
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception as e:
                    print(f"Error loading {json_path}: {e}")
                    continue
                if not is_valid_annotation(data.get("annotations")):
                    self.not_found += 1
                if not data.get("images") or not data.get("annotations") or not data.get("categories"):
                    #self.not_found += 1
                    continue

                imgfile = data["images"][0].get("imgfile")
                if imgfile:
                    imgfile_to_jsons[imgfile].append(data)

        image_files = [f for f in os.listdir(self.img_dir) if f.endswith(".png")]

        for imgfile in image_files:
            if imgfile not in imgfile_to_jsons:
                continue

            # if imgfile not in self.image_id_map:
            #     self.image_id_map[imgfile] = self.image_counter
            #     self.image_counter += 1
            data_samples = imgfile_to_jsons[imgfile]

            image_info = data_samples[0]["images"][0]
            image_id = image_info.get("id")

            if image_id is None:
                print(f"No image_id found for {imgfile}, skipping...")
                continue

            record = {
                "image_file": imgfile,
                "image_id": image_id,
                "categories": []
            }

            category_map = {}
            for data in imgfile_to_jsons[imgfile]:
                image_info = data["images"][0]
                category_info = data["categories"][0]
                category_id = category_info["id"]
                category_name = category_info["name"]

                if category_id not in category_map:
                    category_map[category_id] = {
                        "category_id": category_id,
                        "category_name": category_name,
                        "annotations": []
                    }

                for ann in data["annotations"]:
                    bbox = ann.get("bbox")
                    if not bbox or not isinstance(bbox, list) or len(bbox) != 4:
                        print(f"incorrect bbox: {bbox}")
                        continue

                    annotation = {
                        "annotation_id": ann.get("id",None), #self.annotation_counter,
                        "bbox": bbox,
                        "iscrowd": ann.get("iscrowd", 0),
                        "area": ann.get("area", 0),
                        "ignore": ann.get("ignore", 0),
                        "image_id": ann.get("image_id", image_id)
                    }

                    category_map[category_id]["annotations"].append(annotation)
                    self.annotation_counter += 1

            record["categories"] = list(category_map.values())
            self.dataset.append(record)

        print(f"[INFO] Completed parsing. {self.not_found} image(s) had no annotations.")
        return self.dataset

    def save_to_pickle(self, path="parsed_dataset.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self.dataset, f)
        print(f"[INFO] Saved parsed dataset to: {path}")

    def load_from_pickle(self, path="parsed_dataset.pkl"):
        with open(path, "rb") as f:
            self.dataset = pickle.load(f)
        print(f"[INFO] Loaded parsed dataset from: {path}")
        return self.dataset
