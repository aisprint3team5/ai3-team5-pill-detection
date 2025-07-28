from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
RUN_DIR = Path(__file__).resolve().parent
YOLO8_YAML_PATH = ROOT_DIR / "config" / "yolo8"
TEST_IMAGE_DIR = ROOT_DIR / "data" / "raw" / "test_images"
TRAIN_IMAGE_DIR = ROOT_DIR / "data" / "temp" / "images"
TRAIN_LABEL_DIR = ROOT_DIR / "data" / "temp" / "labels"
TEMP_SPLIT_DIR = ROOT_DIR / "data" / "temp" / "yolo_split"
TRAIN_ANNOTATION_DIR = ROOT_DIR / "data" / "raw" / "train_annotations"
TRAIN_LOG_PATH = ROOT_DIR / "outputs" / "logs" / "yolo8"
TEST_LOG_PATH = ROOT_DIR / "outputs" / "logs" / "yolo8" / "test"
CSV_SAVE_PATH = ROOT_DIR / "outputs" / "submissions" / "yolo8"
VISUALIZATION_SAVE_PATH = ROOT_DIR / "outputs" / "predictions" / "yolo8"
YOLO_DATA_PATH = ROOT_DIR / "data.yaml"

MODEL_PATH = RUN_DIR / "weight"  # 기본 모델 경로, 필요시 변경 가능


# 임시
TEMP_IMAGES = ROOT_DIR / "data" / "temp" / "yolo_split" / "train" / "images"
TEMP_LABELS = ROOT_DIR / "data" / "temp" / "yolo_split" / "train" / "labels"