import yaml
import torch
import argparse


class Config:
    S = None
    B = None
    C = None
    IMAGE_SIZE = None
    IMAGE_CH_SIZE = None
    BATCH_SIZE = None
    LR = None
    LRF = None
    COS_LR = None
    MOMENTUM = None
    BETA = None
    OPTIMIZER = None
    TRAIN_IMAGES_DIR = None
    TRAIN_LABELS_DIR = None
    VAL_IMAGES_DIR = None
    VAL_LABELS_DIR = None
    WD = None
    EPOCHS = None
    CONF_THRESHOLD = None
    NMS_IOU_THRESH = None
    PROJECT = None
    NAME = None
    DEVICE = None
    CLASS_NAMES = None

    # ─── 1) YAML 키 → 클래스 속성명 매핑 ────────────────────────────────
    KEY_MAP = {
        's':              'S',
        'b':              'B',
        'c':              'C',
        'imgsz':          'IMAGE_SIZE',
        'imgchsz':        'IMAGE_CH_SIZE',
        'batch':          'BATCH_SIZE',
        'epochs':         'EPOCHS',
        'lr0':            'LR',
        'lrf':            'LRF',
        'cos_lr':         'COS_LR',
        'momentum':       'MOMENTUM',
        'betas':          'BETAS',
        'optimizer':      'OPTIMIZER',
        'train_images_dir':    'TRAIN_IMAGES_DIR',
        'train_labels_dir':    'TRAIN_LABELS_DIR',
        'val_images_dir':      'VAL_IMAGES_DIR',
        'val_labels_dir':      'VAL_LABELS_DIR',
        'weight_decay':   'WD',
        'conf_thresh':    'CONF_THRESH',
        'nms_iou_thresh': 'NMS_IOU_THRESH',
        'augment':        'AUGMENT',
        'project':        'PROJECT',
        'name':           'NAME',
        'device':         'DEVICE',   # defaults.yaml에 넣어도, load 마지막에 자동 세팅해도 OK
    }

    @classmethod
    def load(cls, yaml_path: str, overrides: dict):
        # 1) yaml_path 로드
        with open(yaml_path, 'r', encoding='utf-8') as f:
            defaults = yaml.safe_load(f)

        # 2) CLI 인자(overrides)로 덮어쓰기 (None 아닌 값만)
        for k, v in overrides.items():
            if v is not None:
                defaults[k] = v

        # 3) 모든 키를 클래스 속성으로 등록
        for key, val in defaults.items():
            attr = cls.KEY_MAP.get(key, key).upper()
            # DEVICE만 자동 분기 처리
            if attr == 'DEVICE':
                # overrides에 device가 없다면 auto, 항상 여기서 덮어써도 OK
                if isinstance(val, str) and val.lower() == 'auto':
                    val = 'cuda' if torch.cuda.is_available() else 'cpu'
            setattr(cls, attr, val)

        # 4) Class Name 정보 로드
        with open('data.yaml', 'r', encoding='utf-8') as f:
            class_names = yaml.safe_load(f)['names']
            setattr(cls, 'CLASS_NAMES', class_names)


def parse_args(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        defaults = yaml.safe_load(f)

    parser = argparse.ArgumentParser()
    for key, val in defaults.items():
        arg = f'--{key}'
        kwargs = {'default': None, 'help': f'default={val}'}
        if key == 'name':
            kwargs['required'] = True
        if isinstance(val, bool):
            # bool → store_true/false
            kwargs['action'] = 'store_false' if val else 'store_true'
            kwargs.pop('default', None)
        elif isinstance(val, list):
            # list(e.g. betas)
            kwargs['type'] = lambda s: list(map(float, s.split(',')))
        else:
            kwargs['type'] = type(val)
        parser.add_argument(arg, **kwargs)

    return parser.parse_args()
