import zipfile
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

import os
import random
import shutil
from typing import Tuple
import sqlite3
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

import json
from collections import Counter
from PIL import Image, ImageDraw, ImageFont

import uuid

COMPETITION = "ai03-level1-project"

def is_data_downloaded(data_dir):
    data_dir = os.path.abspath(data_dir)
    print(data_dir)
    zip_path = os.path.join(data_dir, f"{COMPETITION}.zip")
    return os.path.exists(zip_path)


def download_data(data_dir):

    data_dir = os.path.abspath(data_dir)

    # Check kaggle.json exists
    kaggle_json_path = os.path.expanduser("~/.kaggle/kaggle.json")
    if not os.path.exists(kaggle_json_path):
        raise FileNotFoundError("kaggle.json not found in ~/.kaggle/")
    os.chmod(kaggle_json_path, 0o600)


    api = KaggleApi()
    api.authenticate()


    COMPETITION = "ai03-level1-project"

    ZIP_PATH = os.path.join(data_dir, f"{COMPETITION}.zip")

    os.makedirs(data_dir, exist_ok=True)

    print("Downloading dataset from Kaggle")
    api.competition_download_files(
        competition=COMPETITION,
        path=data_dir,
        quiet=False
    )


    print("Extracting dataset")
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(data_dir)

    print(f"{data_dir}")

def split_yolo_dataset(
        images_dir: str,
        labels_dir: str,
        output_dir: str,
        val_ratio: float = 0.1,
        seed: int = 42
) :

    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    # Prepare destination folders
    for split in ['train', 'val']:
        os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)

    # List all images
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(image_files)

    val_count = int(len(image_files) * val_ratio)
    val_set = set(image_files[:val_count])

    train_count = 0
    val_count_actual = 0

    for img_file in image_files:
        label_file = os.path.splitext(img_file)[0] + ".txt"

        src_img = os.path.join(images_dir, img_file)
        src_lbl = os.path.join(labels_dir, label_file)

        if not os.path.exists(src_lbl):
            print(f"[WARNING] Label file not found for {img_file}, skipping.")
            continue

        split = "val" if img_file in val_set else "train"
        dst_img = os.path.join(output_dir, split, 'images', img_file)
        dst_lbl = os.path.join(output_dir, split, 'labels', label_file)

        shutil.copy(src_img, dst_img)
        shutil.copy(src_lbl, dst_lbl)

        if split == "train":
            train_count += 1
        else:
            val_count_actual += 1

    print(f"[INFO] Split complete: {train_count} training, {val_count_actual} validation samples.")

def load_class_mapping(class_file):
    class_name_to_idx = {}
    idx_to_class_name = {}
    with open(class_file) as f:
        for line in f:
            idx, name = line.strip().split(": ")
            class_name_to_idx[name] = int(idx)
            idx_to_class_name[int(idx)] = name
    return class_name_to_idx, idx_to_class_name

def extract_category_ids_from_filename(filename):
    #  K-003483-016232-027777-031885_0_2_0_2_70_000_200.txt
    base = filename.split("_")[0]
    ids = base.split("-")[1:]  # Skip 'K'
    new_ids = [str(int(i) - 1).zfill(len(i)) for i in ids]

    return new_ids

def get_label_ids_from_file(filepath):
    label_ids = set()
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    label_ids.add(int(parts[0]))
    return label_ids

def get_category_name_from_db(category_id, conn):
    cursor = conn.cursor()
    #category_id = int(category_id)

    cursor.execute("SELECT name FROM pill_metadata WHERE category_id = ?", (category_id,))
    row = cursor.fetchone()
    return row[0] if row else None

def load_class_name_map(txt_path):
    class_map = {}
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            if ':' not in line:
                continue
            idx_str, name = line.strip().split(':', 1)
            idx = int(idx_str.strip())
            class_map[idx] = name.strip()
    return class_map
def draw_bboxes_on_images(image_dir, label_dir, output_dir, class_name_map=None):
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for image_file in image_dir.iterdir():
        if image_file.suffix.lower() not in [".jpg", ".png"]:
            continue

        label_file = label_dir / (image_file.stem + ".txt")
        if not label_file.exists():
            continue  # Skip if no corresponding label

        # Open image
        image = Image.open(image_file).convert("RGB")
        draw = ImageDraw.Draw(image)

        # Load label data
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue  # skip malformed lines
                cls_id, x_center, y_center, width, height = map(float, parts)
                cls_id = int(cls_id)
                label = class_name_map[cls_id] if class_name_map else str(cls_id)

                img_w, img_h = image.size
                x_center *= img_w
                y_center *= img_h
                width *= img_w
                height *= img_h

                x0 = int(x_center - width / 2)
                y0 = int(y_center - height / 2)
                x1 = int(x_center + width / 2)
                y1 = int(y_center + height / 2)

                draw.rectangle([x0, y0, x1, y1], outline="red", width=2)
                draw.text((x0, y0 - 10), label, fill="red")

        # Save only if label file existed
        save_path = output_dir / image_file.name
        image.save(save_path)



def merge_labels_with_db(
    existed_label_dir,
    new_label_dir,
    output_label_dir,
    class_name_to_idx,
    db_path
):
    os.makedirs(output_label_dir, exist_ok=True)

    conn = sqlite3.connect(db_path)

    for filename in os.listdir(new_label_dir):
        if not filename.endswith(".txt"):
            continue

        existed_file = os.path.join(existed_label_dir, filename)
        new_file = os.path.join(new_label_dir, filename)
        output_file = os.path.join(output_label_dir, filename)

        if not os.path.exists(existed_file):
            continue

        existing_label_ids = get_label_ids_from_file(existed_file)

        # Step 1: parse filename → category_ids
        category_ids = extract_category_ids_from_filename(filename)

        # Step 2: Use DB to get category_name → label_id
        all_label_ids = []
        for cid in category_ids:
            cname = get_category_name_from_db(cid, conn)
            if not cname:
                print(f"[Warning] category_id {cid} not found in DB")
                continue
            label_id = class_name_to_idx.get(cname)
            if label_id is not None:
                all_label_ids.append(label_id)
            else:
                print(f"[Warning] category_name '{cname}' not in class_names.txt")

        # Step 3: find missing
        missing_label_ids = set(all_label_ids) - existing_label_ids
        if not missing_label_ids:
            continue

     
        new_bbox_lines = []
        with open(new_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                _, cx, cy, w, h = parts
                for mid in missing_label_ids:
                    new_bbox_lines.append(f"{mid} {cx} {cy} {w} {h}")

        # Step 5: Merge and write to output
        with open(existed_file, "r", encoding="utf-8") as f:
            existing_lines = f.readlines()

        with open(output_file, "w", encoding="utf-8") as f:
            for line in existing_lines:
                f.write(line.strip() + "\n")
            for line in new_bbox_lines:
                f.write(line + "\n")

        print(f" Merged: {filename}")

    conn.close()

def move_files(split_data, split_name,INPUT_IMAGE_DIR,INPUT_LABEL_DIR,OUTPUT_DIR):
    for item in split_data:
        img_name = item["image_file"]
        image_path = os.path.join(INPUT_IMAGE_DIR, img_name)
        label_path = os.path.join(INPUT_LABEL_DIR, os.path.splitext(img_name)[0] + ".txt")

        out_img_path = os.path.join(OUTPUT_DIR, split_name, "images", img_name)
        out_lbl_path = os.path.join(OUTPUT_DIR, split_name, "labels", os.path.splitext(img_name)[0] + ".txt")

        os.makedirs(os.path.dirname(out_img_path), exist_ok=True)
        os.makedirs(os.path.dirname(out_lbl_path), exist_ok=True)

        shutil.copy(image_path, out_img_path)
        shutil.copy(label_path, out_lbl_path)


def augment_dataset(image_dir, label_dir, output_image_dir, output_label_dir, class_map_path=None, augmentations_per_image=3, rare_boost_factor=2):
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    class_counter = Counter()
    label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
    for label_file in label_files:
        with open(os.path.join(label_dir, label_file)) as f:
            for line in f:
                if line.strip():
                    class_id = int(line.strip().split()[0])
                    class_counter[class_id] += 1

    # Define rare classes
    avg_count = sum(class_counter.values()) / len(class_counter)
    rare_classes = [cls for cls, count in class_counter.items() if count < avg_count * 0.4]
    print(f"[INFO] Rare classes (boosted): {rare_classes}")

    transform = A.Compose([
        A.RandomResizedCrop(size=(640, 640), scale=(0.85, 1.0), ratio=(0.75, 1.33), p=0.5),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=10, p=0.5),
       # A.RandomBrightnessContrast(p=0.2),
       # A.HueSaturationValue(p=0.3),
       # A.RGBShift(p=0.3),
       # A.CLAHE(p=0.2),
        A.Blur(p=0.1),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    for fname in os.listdir(image_dir):
        if not fname.endswith(".png"):
            continue

        base_name = os.path.splitext(fname)[0]
        image_path = os.path.join(image_dir, fname)
        label_path = os.path.join(label_dir, base_name + ".txt")

        if not os.path.exists(label_path):
            print(f"[!] Missing label: {label_path}")
            continue

        # Read image and labels
        image = cv2.imread(image_path)
        if image is None:
            print(f"[!] Could not read image: {image_path}")
            continue
        height, width = image.shape[:2]

        with open(label_path, "r") as f:
            lines = f.read().strip().splitlines()

        if not lines:
            continue  # skip if empty label

        bboxes = []
        class_labels = []
        for line in lines:
            parts = line.strip().split()
            class_id = int(parts[0])
            bbox = list(map(float, parts[1:5]))
            class_labels.append(class_id)
            bboxes.append(bbox)

        # rare class
        contains_rare = any(c in rare_classes for c in class_labels)
        n_aug = augmentations_per_image * rare_boost_factor if contains_rare else augmentations_per_image

    
        for i in range(n_aug):
            try:
                transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
                transformed_image = transformed['image']
                transformed_bboxes = transformed['bboxes']
                transformed_labels = transformed['class_labels']

                # Save new image with new name 
                out_img_name = f"{base_name}_aug{i}.png"
                out_lbl_name = f"{base_name}_aug{i}.txt"

                out_img_path = os.path.join(output_image_dir, out_img_name)
                out_lbl_path = os.path.join(output_label_dir, out_lbl_name)

                cv2.imwrite(out_img_path, transformed_image)

                with open(out_lbl_path, "w") as f:
                    for cid, box in zip(transformed_labels, transformed_bboxes):
                        cid = int(cid)  
                        x, y, w, h = [str(np.clip(coord, 0.0, 1.0)) for coord in box]
                        f.write(f"{cid} {x} {y} {w} {h}\n")

                print(f"Augmented: {out_img_name} (Rare={contains_rare})")

            except Exception as e:
                print(f"Failed on {fname} (i={i}): {e}")
                continue

def check_image_label_consistency(image_dir, label_dir, class_map_path, report_path="mismatched_files.txt"):
    with open(class_map_path, 'r') as f:
        class_map = json.load(f)

    class_id_to_label_id = {}
    for label_id_str, entry in class_map.items():
        class_id = str(entry["class_id"]).zfill(6)
        class_id_to_label_id[class_id] = label_id_str  # YOLO label IDs are strings

    mismatches = []
    for fname in os.listdir(label_dir):
        if not fname.endswith(".txt"):
            continue

        label_path = os.path.join(label_dir, fname)
        base_name = os.path.splitext(fname)[0]
        image_path = os.path.join(image_dir, base_name + '.png')

        # 1. Extract class_ids from filename
        raw_ids = extract_category_ids_from_filename(fname)

        # 2. Convert class_ids to YOLO label IDs
        try:
            expected_label_ids = [class_id_to_label_id[cat_id] for cat_id in raw_ids]
        except KeyError as e:
            print(f"[!] Class ID {e} not found in class_map.json for file {fname}")
            continue

        # 3. Read actual YOLO label file
        with open(label_path, 'r') as f:
            label_lines = [line.strip() for line in f if line.strip()]
            actual_label_ids = [line.split()[0] for line in label_lines]

        # 4. Compare sorted sets (order doesn't matter)
        if sorted(actual_label_ids) != sorted(expected_label_ids):
            mismatches.append((fname, expected_label_ids, actual_label_ids))

             # Delete mismatched files
            try:
                os.remove(image_path)
                os.remove(label_path)
                print(f"[✓] Deleted mismatched files: {fname}")
            except Exception as e:
                print(f"[!] Error deleting files for {fname}: {e}")

    # Save mismatch report
    if mismatches:
        with open(report_path, "w", encoding="utf-8") as f:
            for fname, expected, actual in mismatches:
                f.write(f"{fname} | expected: {expected} | actual: {actual}\n")
        print(f"\n[✓] Mismatch report saved to: {report_path}")
    else:
        print("\n[✓] No mismatches found.")

def merge_into_folder_a(folder_a, folder_b):
    """
    Merge folder_b into folder_a.
    Files from folder_b will overwrite any conflicting files in folder_a.
    """
    if not os.path.exists(folder_a):
        raise FileNotFoundError(f"folder_a '{folder_a}' does not exist.")
    if not os.path.exists(folder_b):
        raise FileNotFoundError(f"folder_b '{folder_b}' does not exist.")

    for fname in os.listdir(folder_b):
        src_path = os.path.join(folder_b, fname)
        dest_path = os.path.join(folder_a, fname)
        if os.path.isfile(src_path):
            shutil.copy2(src_path, dest_path)  # Overwrites if exists