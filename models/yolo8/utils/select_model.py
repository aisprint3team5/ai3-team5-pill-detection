from enums.yolo8_weight import Yolo8Weight
from config.config import load_config
import config.path as PATH
from pathlib import Path

class SelectModel:
    def __init__(self):
        self.model_type_enum: Yolo8Weight | None = None  # 선택된 모델 타입 (enum)
        self.model_size: str | None = None               # 선택된 모델 크기
        self.use_p6: bool = False                        # pose 모델의 p6 여부

    def build(self):
        self.model_type_enum = self._choose_model_type_enum()
        self.model_size = self._choose_model_size()

        # P6 여부 결정
        if self.model_type_enum == Yolo8Weight.POSE:
            use_p6_input = input("pose 모델에서 p6 버전을 사용할까요? (y/n): ").strip().lower()
            self.use_p6 = use_p6_input == "y"

        # 파일 이름 계산
        model_filename = self.model_type_enum.model_filename(self.model_size, self.use_p6)
        config_filename = self.model_type_enum.config_filename(self.model_size)
        model_basename = self.model_type_enum.base_name(self.model_size, self.use_p6)

        print(f"선택된 모델: {model_filename}")
        print(f"config/yolo8/{config_filename} 구성파일을 로드합니다")

        config_path = PATH.YOLO8_YAML_PATH / config_filename
        if not config_path.exists():
            raise FileNotFoundError(f"구성 파일이 존재하지 않습니다: {config_path}")

        config = load_config(config_path)

        # 선택 결과 반환
        return self.model_type_enum, model_basename, config

    def _choose_model_type_enum(self):
        print("사용할 YOLOv8 모델 종류를 선택하세요:")
        for i, weight_type in enumerate(Yolo8Weight, start=1):
            print(f"{i}. {weight_type.name.replace('_', ' ').title()} ({weight_type.value})")

        while True:
            try:
                choice_input = input("번호를 입력하세요: ").strip()
                choice = int(choice_input)
                if 1 <= choice <= len(Yolo8Weight):
                    return list(Yolo8Weight)[choice - 1]
                else:
                    print("잘못된 번호입니다. 다시 입력해주세요.")
            except ValueError:
                print("숫자를 입력해주세요. 또는 직접 모델 타입(detect/seg/pose/cls/obb)을 입력하세요.")
                family_str = input("모델 타입 입력: ").strip().lower()
                try:
                    return Yolo8Weight(family_str)
                except ValueError:
                    print(f"'{family_str}'는 유효한 모델 타입이 아닙니다. 다시 시도해주세요.")

    def _choose_model_size(self):
        while True:
            size = input("모델 크기를 입력하세요 (n/s/m/l/x): ").strip().lower()
            if size in ["n", "s", "m", "l", "x"]:
                return size
            else:
                print("올바른 모델 크기(n/s/m/l/x)가 아닙니다. 다시 입력해주세요.")
