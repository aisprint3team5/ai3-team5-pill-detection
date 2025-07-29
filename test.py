import os
import torch

from utils.Dataset.Dataset import PillYoloDataset
#from utils.Dataset.transformer import PillImageTransform
from script.db.database import create_db
from utils.Dataset import dataset_parser
from utils.Dataset.dataloader import *
from config.config import CONFIG
from utils.common import *
import random
import yaml
import pandas as pd

import albumentations as A
import cv2
import json
import os
from tqdm import tqdm
import uuid
import copy
import argparse


DB_PATH = ROOT_DIR / "script/db/pill_metadata.db"

# Config paths

CONFIG_TRAIN           = CONFIG["paths"]["yolo_train_dir"]
CONFIG_TRAIN_LABEL     = CONFIG["paths"]["yolo_train_label"]
CONFIG_VAL             = CONFIG["paths"]["yolo_val_dir"]
CONFIG_VAL_LABEL       = CONFIG["paths"]["yolo_val_label"]
CONFIG_ORIGINAL_DATASET = CONFIG["paths"]["original_dataset"]
CONFIG_TRAIN_IMAGE_DIR = CONFIG["paths"]["train_image_dir"]
CONFIG_TRAIN_ANNO_DIR = CONFIG["paths"]["train_annot_dir"]
#CONFIG_TRAIN_LABEL_DIR = CONFIG["paths"]["train_label_dir"]
CONFIG_SPLIT_DATA_DIR   = CONFIG["paths"]["yolo_data_dir"]
CONFIG_DATA_YAML_PATH = CONFIG["paths"]["data_yaml_path"]
CONFIG_DB_PATH = CONFIG["paths"]["db_path"]
CONFIG_CACHE_DIR = CONFIG["paths"]["pickle_path"]
CONFIG_CLASS_FILE = CONFIG["paths"]["class_names_path"]
#Test
CONFIG_MISSING_ANOT_DIR = CONFIG["paths"]["missing_anot_dir"]
CONFIG_TRAIN_NEW_LABEL = CONFIG["paths"]["yolo_train_new_label"]

CONFIG_INPUT_IMAGE_DIR = CONFIG["paths"]["yolo_all_dir"]
CONFIG_INPUT_LABEL_DIR = CONFIG["paths"]["yolo_all_label"]
CONFIG_OUTPUT_DIR = CONFIG["paths"]["yolo_all_new_label"]

VAL_SPLIT = 0.1

# if not is_data_downloaded(CONFIG_ORIGINAL_DATASET):
#     print("Data is not downloaded. Downloading...")
#download_data(CONFIG_ORIGINAL_DATASET)
# else:
#     print("Data is already downloaded.")

def main():

    parser = argparse.ArgumentParser(description="YOLO Training Utility")

    parser.add_argument('--augment', action='store_true', help="Run data augmentation pipeline")
    parser.add_argument('--consistency', action='store_true', help="Run annotation consistency check")

    parser.add_argument('--annotation', action='store_true', help="Run merge new annotation into folder")
    args = parser.parse_args()

    create_db(CONFIG_DB_PATH)
    print("Running test script...")
    dataset = convert_to_json(CONFIG_DB_PATH)


    # Parse json file
    parser = dataset_parser.PillDatasetParser(CONFIG_TRAIN_IMAGE_DIR, CONFIG_TRAIN_ANNO_DIR)
    if os.path.exists(CONFIG_CACHE_DIR):
        dataset = parser.load_from_pickle(CONFIG_CACHE_DIR)
    else:
        # Parse and save
        dataset = parser.parse()
        parser.save_to_pickle(CONFIG_CACHE_DIR)


    all_class_names = set()
    for data in dataset:
        for cat in data["categories"]:
            all_class_names.add(cat["category_name"])
    class_names = sorted(list(all_class_names))
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    # #  디렉토리 생성
    for split in ["train", "val"]:
        for t in ["images", "labels"]:
            os.makedirs(os.path.join(CONFIG_SPLIT_DATA_DIR, split, t), exist_ok=True)

    # # # split dataset
    # random.shuffle(dataset)
    # val_size = int(len(dataset) * VAL_SPLIT)
    # val_data = dataset[:val_size]
    # train_data = dataset[val_size:]
    convert_to_yolo(dataset, class_to_idx, CONFIG_TRAIN_IMAGE_DIR, CONFIG_INPUT_IMAGE_DIR, CONFIG_INPUT_LABEL_DIR)
   # convert_to_yolo(train_data, class_to_idx, CONFIG_TRAIN_IMAGE_DIR, CONFIG_TRAIN, CONFIG_TRAIN_LABEL)
   # convert_to_yolo(val_data,class_to_idx, CONFIG_TRAIN_IMAGE_DIR, CONFIG_VAL, CONFIG_VAL_LABEL)
    # # #
    
    if args.annotation:
        merge_into_folder(CONFIG_INPUT_LABEL_DIR, CONFIG_OUTPUT_DIR)

    if args.consistency:
        print("Checking annotation consistency...")
        check_image_label_consistency(
            CONFIG_TRAIN_IMAGE_DIR,
            CONFIG_TRAIN_ANNO_DIR,
            CONFIG_CLASS_FILE,
            CONFIG_MISSING_ANOT_DIR
        )
    if args.augment:
        print("Running augmentation...")
        # Augment train and val datasets
        augment_dataset(
        image_dir='data/yolo_split/all/images',
        label_dir='data/yolo_split/all/labels',
        output_image_dir='data/yolo_split/train_aug/images',
        output_label_dir='data/yolo_split/train_aug/labels',
        class_map_path='class_map.json',
        augmentations_per_image=2,
        rare_boost_factor=4  # Images with rare classes get more augmentations
        )
    # split after dataset is converted to YOLO format 
    split_yolo_dataset( CONFIG_INPUT_IMAGE_DIR,
                 CONFIG_INPUT_LABEL_DIR,
                 CONFIG_SPLIT_DATA_DIR,
                 VAL_SPLIT)

    data_yaml = {
        "yolo_train_dir": CONFIG_TRAIN,
        "yolo_train_label": CONFIG_TRAIN_LABEL,
        "yolo_val_dir": CONFIG_VAL,
        "yolo_val_label": CONFIG_VAL_LABEL,
        "train": CONFIG_TRAIN,
        "val": CONFIG_VAL,
        "path": CONFIG_SPLIT_DATA_DIR,
        "nc": len(class_names),
        "names": class_names
    }
    #Save data.yaml
    with open(CONFIG_DATA_YAML_PATH, "w") as f:
        yaml.dump(data_yaml, f, allow_unicode=True)
    class_name_to_idx, _ = load_class_mapping(CONFIG_CLASS_FILE)
    # merge_labels_with_db(
    #     CONFIG_TRAIN_LABEL,
    #     CONFIG_MISSING_ANOT_DIR,
    #     CONFIG_TRAIN_NEW_LABEL,
    #     class_name_to_idx,
    #     DB_PATH
    # )

    print("Finished test script...")

# parser = dataset_parser.PillDatasetParser(CONFIG_TRAIN_IMAGE_DIR, CONFIG_TRAIN_ANNO_DIR)
# if os.path.exists(CONFIG_CACHE_DIR):
#     dataset = parser.load_from_pickle(CONFIG_CACHE_DIR)
# else:
#     # Parse and save
#     dataset = parser.parse()
#     parser.save_to_pickle(CONFIG_CACHE_DIR)

def merge_yolo_annotations(folder_a, folder_b, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Get all .txt files from folder_a
    txt_files_a = [f for f in os.listdir(folder_a) if f.endswith(".txt")]
    merged_count = 0
    skipped_count = 0

    for file_name in txt_files_a:
        path_a = os.path.join(folder_a, file_name)
        path_b = os.path.join(folder_b, file_name)

        # Only merge if the same file exists in both folders
        if os.path.exists(path_b):
            with open(path_a, "r") as fa:
                lines_a = fa.read().strip().splitlines()

            with open(path_b, "r") as fb:
                lines_b = fb.read().strip().splitlines()

            # Combine both sets of labels
            merged_lines = lines_a + lines_b

            # Write merged result to output_dir
            output_path = os.path.join(output_dir, file_name)
            with open(output_path, "w") as fout:
                fout.write("\n".join(merged_lines) + "\n")

            print(f"✅ Merged: {file_name}")
            merged_count += 1
        else:
            #print(f"⚠️ Skipped (no match in folder_b): {file_name}")
            skipped_count += 1
    print(f" Merged {merged_count} files")
    print(f"Skipped {skipped_count} files")
# merge_yolo_annotations(
#     folder_a="data/yolo_split/train/labels",
#     folder_b="data/updated_labels_2",
#     output_dir="data/yolo_split/train/merged_labels"  # Optional. If not given, it overwrites in folder_a
# )




if __name__ == "__main__":
    main()


