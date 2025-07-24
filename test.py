#Library
import os
import random
import shutil
import yaml
from pathlib import Path

import torch
from torchvision import transforms

import Utils.common
# Custom modules
from Utils.Datasets.Dataset import PillYoloDataset
from Utils.Datasets.dataloader import *
from Utils.preprocessing import *
from Utils.common import split_yolo_dataset
from Utils.Datasets.dataset_parser import PillDatasetParser
from Utils.Datasets.transformer import PillImageTransform
from config.config import CONFIG



# Constants & Paths

ROOT_DIR = Path(__file__).resolve().parent

# Config paths
CONFIG_INPUT_IMAGE_DIR = CONFIG["paths"]["input_image_dir"]
CONFIG_INPUT_LABEL_DIR = CONFIG["paths"]["input_label_dir"]
CONFIG_OUTPUT_DIR      = CONFIG["paths"]["output_dir"]
CONFIG_TRAIN           = CONFIG["paths"]["yolo_train_dir"]
CONFIG_TRAIN_LABEL     = CONFIG["paths"]["yolo_train_label"]
CONFIG_VAL             = CONFIG["paths"]["yolo_val_dir"]
CONFIG_VAL_LABEL       = CONFIG["paths"]["yolo_val_label"]

# Static paths
INPUT_IMAGE_DIR = ROOT_DIR / "YOLO_dataset/images"
INPUT_LABEL_DIR = ROOT_DIR / "YOLO_dataset/labels"
CLASS_FILE      = ROOT_DIR / "YOLO_dataset/class_names.txt"
TRAIN_IMAGE_DIR = ROOT_DIR / "data/raw/train_images"
TRAIN_ANN_DIR   = ROOT_DIR / "data/raw/train_annotations"
OUTPUT_IMG_DIR  = ROOT_DIR / "output/contour_detection"
OUTPUT_DIR      = ROOT_DIR / "output"
CONFIG_DIR      = ROOT_DIR / "config"
PICKLE_PATH     = ROOT_DIR / "cache/parsed_dataset.pkl"
DB_PATH = ROOT_DIR / "script/db/pill_metadata.db"

VAL_SPLIT = 0.1
# Load and explore
#download_data()

dataset = convert_to_json(DB_PATH)

#
# #
# # # Step 1 : Parse json file
parser = PillDatasetParser(TRAIN_IMAGE_DIR, TRAIN_ANN_DIR)
if os.path.exists(PICKLE_PATH):
    dataset = parser.load_from_pickle(PICKLE_PATH)
else:
    # Parse and save
    dataset = parser.parse()
    parser.save_to_pickle(PICKLE_PATH)
# #
# # #print(dataset[0])
# #
# # # Step 2 :
# #
# # # 1. 클래스 이름 추출 (dataset 구조 기반)
all_class_names = set()
for data in dataset:
    for cat in data["categories"]:
        all_class_names.add(cat["category_name"])
class_names = sorted(list(all_class_names))
class_to_idx = {name: idx for idx, name in enumerate(class_names)}
# #
# # # 2. 디렉토리 생성
for split in ["train", "val"]:
    for t in ["images", "labels"]:
        os.makedirs(os.path.join(CONFIG_OUTPUT_DIR, split, t), exist_ok=True)

os.makedirs(CONFIG_DIR, exist_ok=True)
# #
# # # split dataset
random.shuffle(dataset)
val_size = int(len(dataset) * VAL_SPLIT)
val_data = dataset[:val_size]
train_data = dataset[val_size:]
# #
#
convert_to_yolo(train_data, TRAIN_IMAGE_DIR, CONFIG_TRAIN, CONFIG_TRAIN_LABEL)
convert_to_yolo(val_data, TRAIN_IMAGE_DIR, CONFIG_VAL, CONFIG_VAL_LABEL)
# # #
# # # # 실제 파일 이동
split_yolo_dataset( CONFIG_INPUT_IMAGE_DIR,
            CONFIG_INPUT_LABEL_DIR,
            CONFIG_OUTPUT_DIR)
# #
data_yaml = {
    "yolo_train_dir": str(CONFIG_OUTPUT_DIR / "train" / "images"),
    "yolo_train_label": str(CONFIG_OUTPUT_DIR / "train" / "labels"),
    "yolo_val_dir": str(CONFIG_OUTPUT_DIR / "val" / "images"),
    "yolo_val_label": str(CONFIG_OUTPUT_DIR / "val" / "labels"),
    "nc": len(class_names),
    "names": class_names
}
#
with open(CONFIG_DIR / "data.yaml", "w") as f:
    yaml.dump(data_yaml, f, allow_unicode=True)
#
# # print(f"YOLO 데이터셋 생성 완료\n 클래스 수: {len(class_names)}")
#
class_name_to_idx, _ = Utils.common.load_class_mapping(CLASS_FILE)
#
# # Define transformation
transform = PillImageTransform(resize=(640, 640))
#
# # Load train and val datasets
train_dataset = PillYoloDataset(
    image_dir=CONFIG_TRAIN,
    label_dir=CONFIG_TRAIN_LABEL,
    class_to_idx=class_name_to_idx,
    S=7, B=2, C=len(class_name_to_idx),
    transform=transform
)
val_dataset = PillYoloDataset(
    image_dir=CONFIG_VAL,
    label_dir=CONFIG_VAL_LABEL,
    class_to_idx=class_name_to_idx,
    transform=transform
)




# class_txt_path = "YOLO_dataset/class_names.txt"
# existed_label_dir = "existed_label"
# new_label_dir = "data/new_anno"
# output_label_dir = "output_label"


#class_name_to_idx, _ = Utils.common.load_class_mapping(class_txt_path)

# === Run ===
# Utils.common.merge_labels_with_db(
#     CONFIG_TRAIN_LABEL,
#     new_label_dir,
#     output_label_dir,
#     class_name_to_idx,
#     DB_PATH
# )
#json_files = get_all_annotation_files(TRAIN_ANN_DIR)
#print(f"총 {len(json_files)}개의 annotation 파일을 찾았습니다.")
#df_annotations = analyze_pill_annotations(json_files)


#class_counts = plot_class_distribution(data, top_n=30)
#analyze_image_sizes(data)
#metadata_and_annotation_analysis(data)

# detect_missing_pills(
#     dataset=data,
#     train_img_dir=TRAIN_IMG_DIR,
#     output_dir=OUTPUT_IMG_DIR
# )
