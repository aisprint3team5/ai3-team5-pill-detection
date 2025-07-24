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


VAL_SPLIT = 0.1

# if not is_data_downloaded(CONFIG_ORIGINAL_DATASET):
#     print("Data is not downloaded. Downloading...")
#download_data(CONFIG_ORIGINAL_DATASET)
# else:
#     print("Data is already downloaded.")

def main():
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

    # # split dataset
    random.shuffle(dataset)
    val_size = int(len(dataset) * VAL_SPLIT)
    val_data = dataset[:val_size]
    train_data = dataset[val_size:]

    convert_to_yolo(train_data, class_to_idx, CONFIG_TRAIN_IMAGE_DIR, CONFIG_TRAIN, CONFIG_TRAIN_LABEL)
    convert_to_yolo(val_data,class_to_idx, CONFIG_TRAIN_IMAGE_DIR, CONFIG_VAL, CONFIG_VAL_LABEL)
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

    # merge_labels_with_db(
    #     CONFIG_VAL_LABEL,
    #     CONFIG_MISSING_ANOT_DIR,
    #     CONFIG_VAL_NEW_LABEL,
    #     class_name_to_idx,
    #     DB_PATH
    # )




    # Define transformation
    # transform = PillImageTransform(resize=(640, 640))

    # # # Load train and val datasets
    # train_dataset = PillYoloDataset(
    #     image_dir=CONFIG_TRAIN,
    #     label_dir=CONFIG_TRAIN_LABEL,
    #     class_to_idx=class_name_to_idx,
    #     S=7, B=2, C=len(class_name_to_idx),
    #     transform=transform
    # )
    # val_dataset = PillYoloDataset(
    #     image_dir=CONFIG_VAL,
    #     label_dir=CONFIG_VAL_LABEL,
    #     class_to_idx=class_name_to_idx,
    #     transform=transform
    # )
    print("Finished test script...")

if __name__ == "__main__":

   # class_name_map = load_class_name_map(CONFIG_CLASS_FILE)
   # draw_bboxes_on_images(CONFIG_TRAIN, CONFIG_TRAIN_LABEL, CONFIG_MINE, class_name_map)
   # draw_bboxes_on_images(CONFIG_MINE, CONFIG_MISSING_ANOT_DIR, CONFIG_TEST, class_name_map)

   main()