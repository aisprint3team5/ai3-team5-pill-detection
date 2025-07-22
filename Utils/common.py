# utils/common.py

import os
import shutil

import os
import random
import shutil
from typing import Tuple


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
