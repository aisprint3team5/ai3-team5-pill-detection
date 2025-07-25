from enum import Enum, auto

class Yolo8Weight(Enum):
    DETECT = "detect"   # detect
    SEGMENT = "seg"     # seg
    POSE = "pose"       # pose
    CLASSIFY = "cls"    # cls
    ORIENT = "obb"      # obb

    def model_filename(self, size: str, use_p6: bool = False) -> str:
        if self == Yolo8Weight.DETECT:
            return f"yolov8{size}.pt"
        elif self == Yolo8Weight.POSE and use_p6:
            return f"yolov8{size}-{self.value}-p6.pt"
        else:
            return f"yolov8{size}-{self.value}.pt"

    def config_filename(self, size: str) -> str:
        if self == Yolo8Weight.DETECT:
            return f"yolo_8{size}.yaml"
        else:
            return f"yolo_v8{size}_{self.value}.yaml"