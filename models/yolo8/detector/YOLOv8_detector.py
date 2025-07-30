from ultralytics import YOLO
from enums.yolo8_weight import Yolo8Weight
import config.path as PATH
from pathlib import Path
import torch
import datetime

from detector.hyp import hyperparameters


class YOLOV8Detector:
    def __init__(self, model_file_name: str, conf_threshold: float):
        model_path = PATH.MODEL_PATH

        self.model = YOLO(model_path / model_file_name)
        self.conf_threshold = conf_threshold
        self.run_time = datetime.datetime.now().strftime('%y%m%d_%H-%M-%S')

        print(f"[DEV-INFO] Loaded model: {model_file_name}")
        print(f"[DEV-INFO] Classes: {self.model.names}")

    def train(self, epochs=10, patience=5, imgsz=416, batch_size=16):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        print(f"[DEV-INFO] Training on {PATH.YOLO_DATA_PATH}, device: {device}")

        self.model.train(
            data=PATH.ROOT_DIR / "data.yaml",
            epochs=epochs,
            # patience=patience,
            imgsz=imgsz,
            batch=batch_size,
            project=PATH.TRAIN_LOG_PATH,
            name=f"train_results_{self.run_time}",
            device=device,
            exist_ok=True,
            **hyperparameters  
        )

        print("[DEV-INFO] Training finished.")

    def predict(self, source_path: str, save: bool = True):
        print(f"[DEV-INFO] Running prediction on: {source_path}")

        results = self.model(source=source_path, save=save)

        return results

    def test(self):
        print(f"[DEV-INFO] Running test on: {PATH.TEST_IMAGE_DIR}")

        results = self.model.predict(
            source=PATH.TEST_IMAGE_DIR,
            conf=self.conf_threshold,
            save=True,
            save_txt=True,
            save_conf=True,
            project=PATH.TEST_LOG_PATH,
            name=f"test_results_{self.run_time}",
            exist_ok=True
        )

        return results, self.run_time
