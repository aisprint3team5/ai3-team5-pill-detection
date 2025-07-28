from enums.yolo8_weight import Yolo8Weight
from config.config import load_config
import config.path as PATH
from detector.YOLOv8_detector import YOLOV8Detector
from pipeline.inference import run_inference
from utils.visualize import visualize_detection
from utils.to_submission_format import to_submission_format
import os
import sys

if __name__ == "__main__":
    # key=value 형태로 인자 파싱
    cli_args = dict(arg.split("=") for arg in sys.argv[1:])

    model_type = int(cli_args["model_type"])
    model_size = cli_args["model_size"]
    use_p6 = cli_args.get("use_p6", "False").lower() == "true"

    model_enum = {
        1: Yolo8Weight.DETECT,
        2: Yolo8Weight.SEGMENT,
        3: Yolo8Weight.POSE,
        4: Yolo8Weight.CLASSIFY,
        5: Yolo8Weight.ORIENT
    }[model_type]

    model_file_name = model_enum.model_filename(model_size, use_p6)
    config_filename = model_enum.config_filename(model_size)

    print(f"선택된 모델: {model_file_name}")
    print(f"config/yolo8/{config_filename} 구성파일을 로드합니다")

    config_path = PATH.YOLO8_YAML_PATH / config_filename
    if not config_path.exists():
        raise FileNotFoundError(f"구성 파일이 존재하지 않습니다: {config_path}")

    config = load_config(config_path)

    # 구성값 추출
    conf_threshold = config["conf_threshold"]
    train_data_yaml = PATH.ROOT_DIR / config["train_data_yaml"]
    train_epoch = config["epochs"]
    train_patience = config["patience"]
    image_size = config["image_size"]
    batch_size = config["batch_size"]
    test_image_dir = config["test_image_dir"]

    model_path = PATH.MODEL_PATH / model_file_name
    detector = YOLOV8Detector(model_file_name=model_file_name, conf_threshold=conf_threshold)


    detector.train(
        data_yaml_path=train_data_yaml,
        epochs=train_epoch,
        patience=train_patience,
        imgsz=image_size,
        batch_size=batch_size
    )

    results = run_inference(detector)
    print(f"[INFO] Detection 완료. 총 결과 수: {len(results)}")

    for idx, result in enumerate(results):
        save_path = f"{PATH.VISUALIZATION_SAVE_PATH}/result_{idx}.jpg"
        visualize_detection([result], save_path=save_path, show=False)

    test_result, timestamp = detector.test()
    to_submission_format(test_result, timestamp)