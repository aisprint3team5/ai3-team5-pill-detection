from config.config import load_config
import config.path as PATH
from detector.YOLOv8_detector import YOLOV8Detector
from pipeline.inference import run_inference
from utils.visualize import visualize_detection
from utils.split_dataset import split_dataset
from utils.to_submission_format import to_submission_format
from utils.build_class_id_map import build_class_id_map
import asyncio
from pathlib import Path
import os

if __name__ == "__main__":
    config = load_config(PATH.YOLO8_YAML_PATH)
    print(f"🙏🙏🙏🙏🙏🙏🙏: {config}")
    model_path = config["model"]
    conf_threshold = config["conf_threshold"]
    iou_threshold = config["iou_threshold"]
    
    train_data_yaml = Path(__file__).resolve().parent.parent.parent / config["train_data_yaml"]

    train_epoch = config["epochs"]
    train_patience = config["patience"]
    image_size = config["image_size"]
    batch_size = config["batch_size"]

    test_image_dir = config["test_image_dir"]

    print(f'asdasdasd: {train_data_yaml}')

    # YOLOv8 Detector 객체 생성
    detector = YOLOV8Detector(
        model_path=model_path,
        conf_threshold=conf_threshold
    )

    # 모델 학습
    detector.train(
        data_yaml_path= train_data_yaml,
        epochs=train_epoch,
        patience=train_patience,
        imgsz=image_size,
        batch_size=batch_size
    )
    
    # 예측 수행
    results = run_inference(detector, test_image_dir)
    
    # 결과 요약
    print(f"[DEV - INFO] Detection completed. Total results: {len(results)}")
    
    # 시각화 결과 저장
    for idx, result in enumerate(results):
        save_path = f"{PATH.VISUALIZATION_SAVE_PATH}/result_{idx}.jpg"
        visualize_detection([result], save_path=save_path, show=False)
        
    # 테스트 결과 저장
    test_result, timestamp = detector.test(source_path=test_image_dir)

    to_submission_format(test_result, timestamp)