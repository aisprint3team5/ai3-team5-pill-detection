# YOLO 탐지 로직
from ultralytics import YOLO
import cv2
import numpy as np

class YOLOV8Detector():
    def __init__(self, model_path: str, conf_threshold: float):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        
        print(f"[DEV - INFO] Loaded model: {model_path}")
        print(f"[DEV - INFO] Classes: {self.model.names}")
    
    def train(self, data_yaml_path: str, epochs: int = 10, patience: int = 5, imgsz: int = 416):
        print(f"[DEV - INFO] Training on {data_yaml_path}")
        
        self.model.train(
            data=data_yaml_path,
            epochs=epochs,
            patience=patience,
            imgsz=imgsz
        )
        
        print("[DEV - INFO] Training finished.")
    
    def predict(self, source_path: str, save: bool = True):
        print(f"[DEV - INFO] Running prediction on: {source_path}")
        
        results = self.model(source=source_path, save=save)
        
        return results
    
