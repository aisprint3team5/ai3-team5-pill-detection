# yolo8/config/config.py

import yaml
from pathlib import Path

def load_config(path: str = None):
    """
    path 인자가 없으면
    └─ 프로젝트/yolo8/data.yaml 을 기본으로 읽어 옵니다.
    """
    if path is None:
        # __file__ → .../yolo8/config/config.py
        # parents[1] → .../yolo8
        path = Path(__file__).resolve().parents[1] / "data.yaml"
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

