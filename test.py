import os
import random
import shutil

from utils.Datasets.Dataset import SampleDataset
from utils.Datasets.dataloader import *
from pathlib import Path
import torch
from torchvision import transforms

# Paths

ROOT_DIR = Path(__file__).resolve().parent
IMAGE_DIR = ROOT_DIR / "YOLO_dataset/images"
LABEL_DIR = ROOT_DIR / "YOLO_dataset/labels"
CLASS_FILE = ROOT_DIR / "YOLO_dataset/class_names.txt"

TRAIN_IMG_DIR = ROOT_DIR / "data/raw/train_images"
TRAIN_ANN_DIR = ROOT_DIR / "data/raw/train_annotations"

# Load and explore
#download_data()

data = convert_to_json()
convert_to_yolo(data)

#data = load_training_data(TRAIN_IMG_DIR, TRAIN_ANN_DIR)
#explore_dataset(data,limit=5, draw_bbox=True)

#json_files = get_all_annotation_files(TRAIN_ANN_DIR)
#print(f"총 {len(json_files)}개의 annotation 파일을 찾았습니다.")
#df_annotations = analyze_pill_annotations(json_files)

#class_counts = plot_class_distribution(data, top_n=30)
#analyze_image_sizes(data)
#metadata_and_annotation_analysis(data)
# if __name__ == "__main__":
#     annotation_root = "train_annotations"
#     db_path = "pill_metadata.db"
#
#     build_pill_database(annotation_root, db_path)





transform = transforms.Compose([
    #transforms.Resize((448, 448)),
    transforms.ToTensor(),
])

class_name_to_idx = {}
with open(CLASS_FILE) as f:
    for line in f:
        idx, name = line.strip().split(": ")
        class_name_to_idx[name] = int(idx)

dataset = SampleDataset(
    image_dir=IMAGE_DIR,
    label_dir=LABEL_DIR,
    class_to_idx=class_name_to_idx,
    S=7, B=2, C=len(class_name_to_idx),
    transform=transform
)
