import yaml
import torch
import argparse


class Config:
    # YAML 키 → 클래스 속성명 매핑
    KEY_MAP = {
        's':               'S',
        'b':               'B',
        'c':               'C',
        'imgsz':           'IMAGE_SIZE',
        'imgchsz':         'IMAGE_CH_SIZE',
        'batch':           'BATCH_SIZE',
        'epochs':          'EPOCHS',
        'lr0':             'LR',
        'lrf':             'LRF',
        'weight_decay':    'WD',
        'conf_thresh':     'CONF_THRESHOLD',
        'nms_iou_thresh':  'NMS_IOU_THRESH',
        # optimizer, momentum, betas, name 은 이름 그대로 사용
    }

    @classmethod
    def load(cls, path_yaml: str, overrides: dict):
        # 1) YAML 기본값 로드
        with open(path_yaml, 'r', encoding='utf-8') as f:
            defaults = yaml.safe_load(f)

        # 2) 터미널 인자로 넘어온 값(None이 아닌 것)으로 덮어쓰기
        for k, v in overrides.items():
            if v is not None:
                defaults[k] = v

        # 3) 클래스 속성으로 동적 등록 (static 변수)
        for key, val in defaults.items():
            attr = cls.KEY_MAP.get(key, key).upper()
            if attr == 'DEVICE':
                val = 'cuda' if torch.cuda.is_available() else 'cpu'
            setattr(cls, attr, val)


def parse_args(defaults_path="defaults.yaml"):
    # 1) YAML 읽어서 기본값 구조 파악
    with open(defaults_path, "r", encoding="utf-8") as f:
        defaults = yaml.safe_load(f)

    parser = argparse.ArgumentParser(
        description="YOLO Training with defaults.yaml + CLI overrides"
    )

    # 2) config/yolo_1.yaml 각 키마다 --key 인자 추가
    for key, val in defaults.items():
        arg = f"--{key}"
        kwargs = {"help": f"(override) default={val}", "default": None}

        # name만 필수(required)로
        if key == "name":
            kwargs["required"] = True

        if isinstance(val, bool):
            # bool 타입은 store_true/store_false으로 처리
            kwargs["action"] = "store_false" if val else "store_true"
            # 'default'와 action 충돌 방지
            kwargs.pop("default", None)
        elif isinstance(val, list):
            # list 예: betas
            kwargs["type"] = lambda s: list(map(float, s.split(",")))
        else:
            kwargs["type"] = type(val)

        parser.add_argument(arg, **kwargs)

    return parser.parse_args()

# class Yolo1Config:
#     S = 7               # Grid size
#     B = 2               # Bounding boxes per cell
#     C = 20              # Classes (Pascal VOC has 20)
#     IMAGE_SIZE = 448    # Input 이미지 크기
#     IMAGE_CH_SIZE = 3
#     BATCH_SIZE = 16
#     LR = 1e-5  # 1e-5 #1e-5 #1e-4 #3e-5 #1e-4 #1e-5 # 1e-4
#     WD = 1e-6  # 5e-4
#     EPOCHS = 40
#     CONF_THRESH = 0.2
#     NMS_IOU_THRESH = 0.4

#     DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
