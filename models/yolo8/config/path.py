from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
RUN_DIR = Path(__file__).resolve().parent
YOLO8_YAML_PATH = ROOT_DIR / "config" / "yolo8"
TEST_IMAGE_DIR = ROOT_DIR / "data" / "raw" / "test_images"
TRAIN_ANNOTATION_DIR = ROOT_DIR / "data" / "raw" / "train_annotations"
TRAIN_LOG_PATH = ROOT_DIR / "outputs" / "logs" / "yolo8" / "train"
TEST_LOG_PATH = ROOT_DIR / "outputs" / "logs" / "yolo8" / "test"
CSV_SAVE_PATH = ROOT_DIR / "outputs" / "submissions" / "yolo8"
VISUALIZATION_SAVE_PATH = ROOT_DIR / "outputs" / "predictions" / "yolo8"

MODEL_PATH = RUN_DIR / "weight"  # 기본 모델 경로, 필요시 변경 가능