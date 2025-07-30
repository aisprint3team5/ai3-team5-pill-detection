from enum import Enum, auto

class Yolo8Weight(Enum):
    DETECT = "detect"   # detect 
    SEGMENT = "seg"     # seg 
    POSE = "pose"       # pose (p6 가능성 존재)
    CLASSIFY = "cls"    # cls 
    ORIENT = "obb"      # obb 

    def _model_base_name(self, size: str, use_p6: bool = False) -> str:
        if self == Yolo8Weight.DETECT:
            return f"yolov8{size}"
        elif self == Yolo8Weight.POSE and use_p6:
            return f"yolov8{size}-{self.value}-p6"
        else:
            return f"yolov8{size}-{self.value}"

    def _config_base_name(self, size: str) -> str:
        if self == Yolo8Weight.DETECT:
            return f"yolo_8{size}"
        else:
            return f"yolo_v8{size}_{self.value}"

    # .pt 확장자를 포함한 모델 파일 이름을 반환
    def model_filename(self, size: str, use_p6: bool = False) -> str:
        return f"{self._model_base_name(size, use_p6)}.pt"

    # .yaml 확장자를 포함한 설정 파일 이름을 반환
    def config_filename(self, size: str) -> str:
        return f"{self._config_base_name(size)}.yaml"

    # 모델 파일의 이름만 반환
    def base_name(self, size: str, use_p6: bool = False) -> str:
        return self._model_base_name(size, use_p6)