# YOLO 탐지 로직
from ultralytics import YOLO
import cv2
import numpy as np
import torch
import datetime

class YOLOV8Detector():
    def __init__(self, model_path: str, conf_threshold: float):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.run_time = f"{datetime.datetime.now().strftime('%y%m%d_%H-%M-%S')}"
        
        print(f"[DEV - INFO] Loaded model: {model_path}")
        print(f"[DEV - INFO] Classes: {self.model.names}")
    
    def train(self, data_yaml_path: str, epochs: int = 10, patience: int = 5, imgsz: int = 416, batch_size: int = 16):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[DEV - INFO] Training on {data_yaml_path}, device: {device}")
        
        self.model.train(
            data=data_yaml_path,
            epochs=epochs,
            patience=patience,
            imgsz=imgsz,
            batch=batch_size,
            project="runs/detect/train",
            name=f"train_results_{self.run_time}",
            device=device,
            exist_ok=True
        )
        
        print("[DEV - INFO] Training finished.")
    
    def predict(self, source_path: str, save: bool = True):
        print(f"[DEV - INFO] Running prediction on: {source_path}")
        
        results = self.model(source=source_path, save=save)
        
        return results
    
    def test(self, source_path: str):
        print(f"[DEV - INFO] Running test on: {source_path}")
        
        results = self.model.predict(
            source=source_path,
            conf=self.conf_threshold,
            save=True,           
            save_txt=True,       
            save_conf=True,      
            project="runs/detect/test",
            name=f"test_results_{self.run_time}",  # 원하는 이름 지정
            exist_ok=True
        )
        
        return results