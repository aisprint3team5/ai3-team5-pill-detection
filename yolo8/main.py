







# yolo8/main.py
from pathlib import Path
import yaml

# 상대 import로 변경 (yolo8 패키지 내부)
from .config.config              import load_config
from .detector.YOLOv8_detector   import YOLOV8Detector
from .pipeline.inference         import run_inference
from .utils.visualize            import visualize_detection
from .utils.split_dataset        import split_dataset
from .utils.build_class_id_map   import build_class_id_map
from .utils.to_submission_format import to_submission_format

if __name__ == "__main__":
    # 설정 불러오기
    config = load_config()  

    # 프로젝트 루트 기준으로 data.yaml 위치 설정
    from pathlib import Path
    PROJECT_ROOT = Path(__file__).resolve().parents[1]  # …/yolo8 → level1 = 프로젝트 루트
    config['train_data_yaml'] = str(PROJECT_ROOT / 'data.yaml')


    model_path = config["model"]
    conf_threshold = config["conf_threshold"]
    iou_threshold = config["iou_threshold"]
    
    train_data_yaml = config["train_data_yaml"]

    train_epoch = config["epochs"]
    train_patience = config["patience"]
    image_size = config["image_size"]
    batch_size = config["batch_size"]

    test_image_dir = config["test_image_dir"]
    save_path = config["run_save_path"]
    
    # TODO: 혜준님 코드와 병햡 필요한 부분
    # split_dataset(
    #     images_dir=config["train_images_dir"],
    #     labels_dir=config["train_labels_dir"],
    #     output_dir=config["output_dir"]
    # )

    print(f'asdasdasd: {train_data_yaml}')

    # YOLOv8 Detector 객체 생성
    detector = YOLOV8Detector(
        model_path=model_path,
        conf_threshold=conf_threshold
    )

    # 모델 학습
    detector.train(
        data_yaml_path=train_data_yaml,
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
        save_path = f"outputs/result_{idx}.jpg"
        visualize_detection([result], save_path=save_path, show=False)
        
    # 테스트 결과 저장
    
    test_result = detector.test(source_path=config["test_image_dir"])
    anno_dir    = config["annotation_dir"]          # data.yaml에 정의
    to_submission_format(test_result, anno_dir)
