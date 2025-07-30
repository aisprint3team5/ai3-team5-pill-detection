import config.path as PATH
from detector.YOLOv8_detector import YOLOV8Detector
from pipeline.inference import run_inference
from utils.visualize import visualize_detection
from utils.split_dataset import split_dataset
from utils.to_submission_format import to_submission_format
from utils.build_class_id_map import build_class_id_map
from utils.select_model import SelectModel
import asyncio
from pathlib import Path
import os

if __name__ == "__main__":
    selector = SelectModel()
    model_enum, model_basename, config = selector.build()

    # YOLOv8 Detector 객체 생성
    detector = YOLOV8Detector(
        model_file_name = model_enum.model_filename(selector.model_size, selector.use_p6),
        conf_threshold = config["conf_threshold"]
    )

    # 모델 학습
    detector.train(
        epochs=config["epochs"],
        # patience=config["patience"],
        imgsz=config["image_size"],
        batch_size=config["batch_size"]
    )
    
    # 예측 수행
    results = run_inference(detector)
    
    # 결과 요약
    print(f"[DEV - INFO] Detection completed. Total results: {len(results)}")
    
    # 시각화 결과 저장
    for idx, result in enumerate(results):
        save_path = f"{PATH.VISUALIZATION_SAVE_PATH}/result_{idx}.jpg"
        visualize_detection([result], save_path=save_path, show=False)
        
    # 테스트 결과 저장
    test_result, timestamp = detector.test()
    to_submission_format(test_result, timestamp)