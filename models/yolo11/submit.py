import os
import sys
import argparse
import torch
import yaml                                  # pip install pyyaml
from ultralytics import YOLO
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt


# ─── 1) 외부 YAML 읽어 DEFAULTS 생성 ────────────────────────────
with open("config/yolo_11.yaml", "r", encoding="utf-8") as f:
    DEFAULTS = yaml.safe_load(f)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Ultralytics YOLOv11 Training Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # 프로젝트/이름
    parser.add_argument('--project', type=str, default=DEFAULTS['project'],
                        help='save results to project/name')
    parser.add_argument('--name',    type=str, required=True, default=DEFAULTS['name'],
                        help='experiment name (subfolder)')
    return parser


def predict_yolo11(project, name):
    model_path: str = os.path.join(project, name, 'weights', 'best.pt')
    model = YOLO(model_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"[INFO] Running test on: {name}")

    results = model.predict(
        source=DEFAULTS['test_images_dir'],  # 테스트할 이미지 또는 폴더 경로
        conf=0.25,        # 필요시 인자 추가 가능
        save=True,
        save_txt=True,
        save_conf=True,
        project=project,
        name=name,
        exist_ok=True,
        device=device,
    )
    return results


def main():
    parser = build_parser()
    # 아무 인자 없이 실행 시 도움말 출력
    if len(sys.argv) == 1:
        parser.print_help()
    args = parser.parse_args()
    results = predict_yolo11(args.project, arg.name)
    print(results)


if __name__ == '__main__':
    main()