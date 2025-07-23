import zipfile
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
import os
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


# utils/common.py

import os
import random
import shutil
from typing import Tuple
import sqlite3

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
    # Assumes filename like: K-003483-016232-027777-031885_0_2_0_2_70_000_200.txt
    base = filename.split("_")[0]
    ids = base.split("-")[1:]  # Skip 'K'
    return ids

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
    category_id = int(category_id) - 1  # → 3482

    cursor.execute("SELECT name FROM pill_metadata WHERE category_id = ?", (category_id,))
    row = cursor.fetchone()
    return row[0] if row else None


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

        # Step 4: Load bbox from new_label
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
