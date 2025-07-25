from enums.yolo8_weight import Yolo8Weight
from config.config import load_config
import config.path as PATH
from pathlib import Path

class SelectModel:
    def build(self):
        model_type = self.__choose_model_type()
        model_size = self.__choose_model_size()

        use_p6 = False
        if model_type == Yolo8Weight.POSE:
            use_p6_input = input("pose 모델에서 p6 버전을 사용할까요? (y/n): ").strip().lower()
            use_p6 = use_p6_input == "y"

        model_filename = model_type.model_filename(model_size, use_p6)
        config_filename = model_type.config_filename(model_size)

        print(f"선택된 모델: {model_filename}")
        print(f"config/yolo8/{config_filename} 구성파일을 로드합니다")

        config_path = PATH.YOLO8_YAML_PATH / config_filename
        if not config_path.exists():
            raise FileNotFoundError(f"구성 파일이 존재하지 않습니다: {config_path}")

        config = load_config(config_path)
        
        return model_filename, config 


    def __choose_model_type(self):
        print("사용할 YOLOv8 모델 종류를 선택하세요:")
        for i, weight in enumerate(Yolo8Weight, start=1):
            print(f"{i}. {weight.name} ({weight.value})")

        try:
            choice = int(input("번호를 입력하세요: "))
            return list(Yolo8Weight)[choice - 1]
        
        except (ValueError, IndexError):
            print("잘못된 선택입니다. 직접 입력을 시도합니다.")
            family = input("family 입력 (detect/seg/pose/cls/obb): ").strip().lower()
            return Yolo8Weight(family)


    def __choose_model_size(self):
        size = input("모델 크기를 입력하세요 (n/s/m/l/x): ").strip().lower()

        if size not in ["n", "s", "m", "l", "x"]:
            raise ValueError("올바른 모델 크기(n/s/m/l/x)가 아닙니다.")
        
        return size