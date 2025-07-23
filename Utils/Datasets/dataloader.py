import os
import json
from PIL import Image, ImageDraw
from collections import defaultdict
#from sklearn.preprocessing import LabelEncoder
import sqlite3
import matplotlib.pyplot as plt
#import script.db.insert_pills as db
import zipfile
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
IMAGE_DIR = ROOT_DIR / "YOLO_dataset/images"
LABEL_DIR = ROOT_DIR / "YOLO_dataset/labels"
CLASS_FILE = ROOT_DIR / "YOLO_dataset/class_names.txt"
OUTPUT_DIR = ROOT_DIR/"outputs/with_bounding_boxes"

TRAIN_IMG_DIR = ROOT_DIR / "data/raw/train_images"
TRAIN_ANN_DIR = ROOT_DIR / "data/raw/train_annotations"

YOLO_IMG_DIR = "YOLO_dataset/images"
YOLO_LABEL_DIR = "YOLO_dataset/labels"

def convert_to_yolo(dataset, image_root_dir, output_img_dir, output_label_dir):
    """
       Converts a dataset to YOLO format.

       Args:
           dataset (list): List of parsed annotation dictionaries.
           image_root_dir (str or Path): Directory where original images are stored.
           output_img_dir (str or Path): Directory to save YOLO images.
           output_label_dir (str or Path): Directory to save YOLO-format label files.
       """
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    # Build class_id map
    all_class_names = set()
    for data in dataset:
        for category in data["categories"]:
            all_class_names.add(category["category_name"]) # category_name
    class_name_to_id = {name: idx for idx, name in enumerate(sorted(all_class_names))}

    # Convert each image & label
    for data in dataset:
        image_filename = data["image_file"]
        image_path = os.path.join(image_root_dir, image_filename)

        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            print(f"[Warning] Image not found: {image_path}")
            continue

        width, height = image.size
        label_lines = []

        for category in data["categories"]:
            class_name = category["category_name"]
            class_id = class_name_to_id[class_name]

            for annotation in category["annotations"]:
                x, y, w, h = annotation["bbox"]

                # YOLO format: center_x, center_y, width, height (normalized)
                x_center = (x + w / 2) / width
                y_center = (y + h / 2) / height
                w_norm = w / width
                h_norm = h / height

                line = f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
                label_lines.append(line)

        # Save image
        output_img_path = os.path.join(output_img_dir, image_filename)
        image.save(output_img_path)

        #  label
        txt_filename = os.path.splitext(image_filename)[0] + ".txt"
        label_path = os.path.join(output_label_dir, txt_filename)
        with open(label_path, "w") as f:
            f.write("\n".join(label_lines))

        print(f"[Saved] {image_filename} -> YOLO format with {len(label_lines)} boxes")


    with open("YOLO_dataset/class_names.txt", "w") as f:
        for name, idx in sorted(class_name_to_id.items(), key=lambda x: x[1]):
            f.write(f"{idx}: {name}\n")


def convert_to_json(db_path):

   # os.makedirs(OUTPUT_DIR, exist_ok=True)

    dataset = []

    image_id_map = {}
    image_counter = 0
    annotation_counter = 0
    not_found = 0
    # Map image file name to JSON data
    imgfile_to_jsons = defaultdict(list)
    #conn  = sqlite3.connect(db_path)

    for root, _, files in os.walk(TRAIN_ANN_DIR):
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

            if not data.get("images") or not data.get("annotations") or not data.get("categories"):
                not_found +=1
                continue

            imgfile = data["images"][0].get("imgfile")
            if imgfile:
                imgfile_to_jsons[imgfile].append(data)


    image_files = [f for f in os.listdir(TRAIN_IMG_DIR) if f.endswith(".png")]

    # Match image files to JSONs
    for imgfile in image_files:
        if imgfile not in imgfile_to_jsons:
            continue

        if imgfile not in image_id_map:
            image_id_map[imgfile] = image_counter
            image_counter += 1

        image_id = image_id_map[imgfile]
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
            #db.insert_category_to_db(conn, category_id, category_name)
            if category_id not in category_map:
                category_map[category_id] = {
                    "category_id": category_id,
                    "category_name": category_name,
                    "annotations": []
                }

            for ann in data["annotations"]:
                bbox = ann.get("bbox")
                if not bbox or not isinstance(bbox, list) or len(bbox) != 4:
                    continue  # skip invalid

                annotation = {
                    "annotation_id": annotation_counter,
                    "bbox": bbox,
                    "iscrowd": ann.get("iscrowd", 0),
                    "area": ann.get("area", 0),
                    "ignore": ann.get("ignore", 0),
                    "images": image_info
                }

                category_map[category_id]["annotations"].append(annotation)
                annotation_counter += 1


        record["categories"] = list(category_map.values())
        dataset.append(record)
    print(f'No annotation {not_found} images')
    # Save to file
    # with open("pill_dataset_structured.json", "w", encoding="utf-8") as f:
    #     json.dump(dataset, f, ensure_ascii=False, indent=2)


    # # Load image
    # for idx, data in enumerate(dataset):
    #     image_filename = data["image_file"]
    #     image_path = os.path.join(TRAIN_IMG_DIR, image_filename)
    #
    #     # Load image
    #     try:
    #         image = Image.open(image_path).convert("RGB")
    #     except FileNotFoundError:
    #         print(f"[Warning] Image not found: {image_path}")
    #         continue
    #
    # # Create figure and axis
    #     fig, ax = plt.subplots(1, figsize=(10, 12))
    #     ax.imshow(image)
    #
    #     # Draw bounding boxes
    #     for category in data["categories"]:
    #         name = category["category_name"]
    #         for annotation in category["annotations"]:
    #             x, y, w, h = annotation["bbox"]
    #             rect = patches.Rectangle((x, y), w, h, linewidth=2,
    #                                      edgecolor='red', facecolor='none')
    #             ax.add_patch(rect)
    #             ax.text(x, y - 5, name, fontsize=10, color='white',
    #                     bbox=dict(facecolor='red', alpha=0.5))
    #
    #     ax.axis('off')
    #     output_path = os.path.join(OUTPUT_DIR, f"annotated_{idx}_{image_filename}")
    #     plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    #     plt.close(fig)  # Close the figure to free memory
    #
    #     print(f"[Saved] {output_path}")
    conn.close()
    return dataset

def download_data():
    # Set permission for kaggle.json
    kaggle_json_path = os.path.expanduser("~/.kaggle/kaggle.json")
    if not os.path.exists(kaggle_json_path):
        raise FileNotFoundError("kaggle.json not found in ~/.kaggle/")
    os.chmod(kaggle_json_path, 0o600)


    api = KaggleApi()
    api.authenticate()


    COMPETITION = "ai03-level1-project"
    DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
    ZIP_PATH = os.path.join(DATA_DIR, f"{COMPETITION}.zip")

    os.makedirs(DATA_DIR, exist_ok=True)

    print("Downloading dataset from Kaggle")
    api.competition_download_files(
        competition=COMPETITION,
        path=DATA_DIR,
        quiet=False
    )


    print("Extracting dataset")
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(DATA_DIR)

    print(f"{DATA_DIR}")

def get_all_annotation_files(root_folder):
    json_files = []
    for root, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith(".json"):
                json_files.append(os.path.join(root, file))
    return json_files

# def analyze_pill_annotations(json_paths):
#     records = []
#     for path in tqdm(json_paths, desc="Parsing JSON files"):
#         with open(path, 'r', encoding='utf-8') as f:
#             data = json.load(f)
#
#         image_info = {img["id"]: img for img in data.get("images", [])}
#         categories = {cat["id"]: cat["name"] for cat in data.get("categories", [])}
#
#         for ann in data.get("annotations", []):
#             img = image_info.get(ann["image_id"], {})
#             if not img:
#                 continue
#
#             record = {
#                 "file_name": img.get("file_name"),
#                 "width": img.get("width"),
#                 "height": img.get("height"),
#                 "camera_la": img.get("camera_la"),
#                 "camera_lo": img.get("camera_lo"),
#                 "size": img.get("size"),
#                 "drug_S": img.get("drug_S"),
#                 "dl_name": img.get("dl_name"),
#                 "dl_name_en": img.get("dl_name_en"),
#                 "di_class_no": img.get("di_class_no"),
#                 "drug_shape": img.get("drug_shape", None),
#                 "thick": img.get("thick", None),
#                 "leng_short":  img.get("leng_short", None),
#                 "leng_long":  img.get("leng_long", None),
#                 "back_color": img.get("back_color"),
#                 "drug_dir": img.get("drug_dir"),
#                 "light_color": img.get("light_color"),
#                 "bbox_x": ann["bbox"][0],
#                 "bbox_y": ann["bbox"][1],
#                 "bbox_w": ann["bbox"][2],
#                 "bbox_h": ann["bbox"][3],
#                 "bbox_area": ann["bbox"][2] * ann["bbox"][3],
#                 "category_id": ann["category_id"],
#                 "category_name": categories.get(ann["category_id"], "Unknown")
#             }
#             records.append(record)
#
#     df = pd.DataFrame(records)
#
#     return df

def load_training_data(image_dir, annotation_root):
    """
    Loads image paths and matching annotation JSONs from nested structure.
    - image_dir: path to train_images/
    - annotation_root: path to train_annotations/
    """
    data = []

    for image_filename in os.listdir(image_dir):
        if not image_filename.endswith(".png"):
            continue

        image_path = os.path.join(image_dir, image_filename)
        base_name = image_filename.replace(".png", "")

        #'K-003351-016688-029543-044199_0_2_0_2_75_000_200' → 'K-003351-016688-029543-044199'
        base_id = base_name.split('_')[0]
        annotation_base_dir = os.path.join(annotation_root, base_id)

        annotation_base_dir = "{}_json".format(annotation_base_dir)
        found = False

        if os.path.exists(annotation_base_dir):
            for subdir in os.listdir(annotation_base_dir):
                json_path = os.path.join(annotation_base_dir, subdir, f"{base_name}.json")
                print(json_path)
                if os.path.exists(json_path):
                    with open(json_path, "r") as f:
                        annotation = json.load(f)
                    data.append({
                        "image_path": image_path,
                        "annotation": annotation
                    })
                    found = True
                    break

        if not found:
            print(f"[WARNING] No annotation found for: {image_filename}")
    print(f"length : {len(data)}")

    return data




def is_valid_annotation(annotation):
    if isinstance(annotation, dict):
        print('xx')
        return  "bbox" in annotation and isinstance(annotation["bbox"], list) and len(annotation["bbox"]) == 4
    elif isinstance(annotation, list):
        print('yy')
        return any(
            "bbox" in ann and isinstance(ann["bbox"], list) and len(ann["bbox"]) == 4
            for ann in annotation
        )
    return False

def explore_dataset(data, limit=5, draw_bbox=True):
    count = 0
    for entry in data:
        image_path = entry["image_path"]
        annotation = entry["annotation"]

        if not is_valid_annotation(annotation):
            continue  # Skip invalid annotations

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Could not open image {image_path}: {e}")
            continue

        if draw_bbox:
            draw = ImageDraw.Draw(image)
            x, y, w, h = annotation["bbox"]
            draw.rectangle([x, y, x + w, y + h], outline="red", width=2)

        # Show image
        plt.figure(figsize=(6, 6))
        plt.imshow(image)
        plt.title(f"{image_path.split('/')[-1]}")
        plt.axis("off")
        plt.show()

        count += 1
        if count >= limit:
            break
