from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
YOLO8_YAML_PATH = ROOT_DIR / "config" / "yolo_8.yaml"
TEST_IMAGE_DIR = ROOT_DIR / "data" / "raw" / "test_images"
TRAIN_ANNOTATION_DIR = ROOT_DIR / "data" / "raw" / "train_annotations"
TRAIN_LOG_PATH = ROOT_DIR / "outputs" / "logs" / "yolo8" / "train"
TEST_LOG_PATH = ROOT_DIR / "outputs" / "logs" / "yolo8" / "test"
CSV_SAVE_PATH = ROOT_DIR / "outputs" / "submissions" / "yolo8"
VISUALIZATION_SAVE_PATH = ROOT_DIR / "outputs" / "predictions" / "yolo8"