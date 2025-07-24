# yolo8/config/config.py

import yaml
from pathlib import Path

def load_config(path: str = None):
    # path 인자가 없으면 yolo8/config/yolo_8.yaml 사용
    if path is None:
        path = Path(__file__).resolve().parent / "yolo_8.yaml"
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)
