from config.config import load_config
from detector.YOLOv8_detector import YOLOV8Detector
from pipeline.inference import run_inference
from utils.visualize import visualize_detection

if __name__ == "__main__":
    # 설정 로드
    # conf_threshold = config.get("conf_threshold", 0.25)
    config = load_config()
    model_path = config["model"]
    conf_threshold = config["conf_threshold"]
    iou_threshold = config["iou_threshold"]
    
    train_data_yaml = config["train_data_yaml"]
    train_epoch = config["epochs"]
    train_patience = config["patience"]
    
    image_size = config["image_size"]
    test_image_dir = config["test_image_dir"]
    save_path = config["run_save_path"]

    # YOLOv8 Detector 객체 생성
    # detector = YoloV8Detector(model_path=model_path, conf_threshold=conf_threshold)
    detector = YOLOV8Detector(
        model_path=model_path,
        conf_threshold=conf_threshold
    )

    # 모델 학습
    detector.train(
        data_yaml_path=train_data_yaml,
        epochs=train_epoch,
        patience=train_patience,
        imgsz=image_size
    )
    
    # 예측 수행
    # results = detector.predict(source_path=test_image_dir, save=True)
    results = run_inference(detector, test_image_dir)
    
    # 결과 요약
    print(f"[DEV - INFO] Detection completed. Total results: {len(results)}")
    
    # 시각화 결과 저장
    for idx, result in enumerate(results):
        save_path = f"outputs/result_{idx}.jpg"
        visualize_detection([result], save_path=save_path, show=False)