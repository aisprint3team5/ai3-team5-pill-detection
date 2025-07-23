import os
import torch

from utils.Dataset.Dataset import PillYoloDataset
from utils.Dataset.transformer import PillImageTransform
from utils.Dataset import dataset_parser
from utils.Dataset.dataloader import *
from config.config import CONFIG
from utils.common import *
import random
import yaml


DB_PATH = ROOT_DIR / "script/db/pill_metadata.db"

# Config paths
CONFIG_INPUT_IMAGE_DIR = CONFIG["paths"]["input_image_dir"]
CONFIG_INPUT_LABEL_DIR = CONFIG["paths"]["input_label_dir"]

CONFIG_TRAIN           = CONFIG["paths"]["yolo_train_dir"]
CONFIG_TRAIN_LABEL     = CONFIG["paths"]["yolo_train_label"]
CONFIG_VAL             = CONFIG["paths"]["yolo_val_dir"]
CONFIG_VAL_LABEL       = CONFIG["paths"]["yolo_val_label"]
CONFIG_ORIGINAL_DATASET = CONFIG["paths"]["original_dataset"]
CONFIG_TRAIN_IMAGE_DIR = CONFIG["paths"]["train_image_dir"]
#CONFIG_TRAIN_LABEL_DIR = CONFIG["paths"]["train_label_dir"]
CONFIG_SPLIT_DATA_DIR   = CONFIG["paths"]["yolo_data_dir"]
# Static paths
INPUT_IMAGE_DIR = ROOT_DIR / "YOLO_dataset/images"
INPUT_LABEL_DIR = ROOT_DIR / "YOLO_dataset/labels"
CLASS_FILE      = ROOT_DIR / "Data/class_names.txt"
TRAIN_IMAGE_DIR = ROOT_DIR / "data/raw/train_images"
TRAIN_ANN_DIR   = ROOT_DIR / "data/raw/train_annotations"
OUTPUT_IMG_DIR  = ROOT_DIR / "output/contour_detection"
OUTPUT_DIR      = ROOT_DIR / "output"
CONFIG_DIR      = ROOT_DIR / "config"
PICKLE_PATH     = ROOT_DIR / "cache/parsed_dataset.pkl"
DB_PATH = ROOT_DIR / "script/db/pill_metadata.db"

VAL_SPLIT = 0.1

# if not is_data_downloaded(CONFIG_ORIGINAL_DATASET):
#     print("Data is not downloaded. Downloading...")
download_data(CONFIG_ORIGINAL_DATASET)
# else:
#     print("Data is already downloaded.")


dataset = convert_to_json(DB_PATH)

print(CONFIG.get("paths", {}).keys())

 # Parse json file
parser = dataset_parser.PillDatasetParser(TRAIN_IMAGE_DIR, TRAIN_ANN_DIR)
if os.path.exists(PICKLE_PATH):
    dataset = parser.load_from_pickle(PICKLE_PATH)
else:
    # Parse and save
    dataset = parser.parse()
    parser.save_to_pickle(PICKLE_PATH)


all_class_names = set()
for data in dataset:
    for cat in data["categories"]:
        all_class_names.add(cat["category_name"])
class_names = sorted(list(all_class_names))
class_to_idx = {name: idx for idx, name in enumerate(class_names)}

# #  디렉토리 생성
for split in ["train", "val"]:
    for t in ["images", "labels"]:
        os.makedirs(os.path.join("Data/yolo_split", split, t), exist_ok=True)

# # split dataset
random.shuffle(dataset)
val_size = int(len(dataset) * VAL_SPLIT)
val_data = dataset[:val_size]
train_data = dataset[val_size:]

convert_to_yolo(train_data, CONFIG_TRAIN_IMAGE_DIR, CONFIG_TRAIN, CONFIG_TRAIN_LABEL)
convert_to_yolo(val_data, CONFIG_TRAIN_IMAGE_DIR, CONFIG_VAL, CONFIG_VAL_LABEL)
# # #
# split after dataset is converted to YOLO format 
# split_yolo_dataset( CONFIG_INPUT_IMAGE_DIR,
#             CONFIG_INPUT_LABEL_DIR,
#             CONFIG_OUTPUT_DIR)

data_yaml = {
    "yolo_train_dir": CONFIG_TRAIN,
    "yolo_train_label": CONFIG_TRAIN_LABEL,
    "yolo_val_dir": CONFIG_VAL,
    "yolo_val_label": CONFIG_VAL_LABEL,
    "nc": len(class_names),
    "names": class_names
}
#for db 
with open(CONFIG_DIR / "data.yaml", "w") as f:
    yaml.dump(data_yaml, f, allow_unicode=True)

    class_name_to_idx, _ = load_class_mapping(CLASS_FILE)
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
