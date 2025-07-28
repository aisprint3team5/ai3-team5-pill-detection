import os
import csv
import sys
import argparse
import torch
import yaml                                  # pip install pyyaml
import pandas as pd
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


def to_submission_format(model_results, csv_save_path="runs/predictions/test_predictions.csv"):
    # 결과 수집용 리스트
    data = []
    annotation_id = 1  # annotation_id는 1부터 시작

    # 이미지별로 결과 저장
    for result in model_results:
        image_path = result.path
        image_name = os.path.basename(image_path)
        image_id = int(os.path.splitext(image_name)[0])  # '123.png' → 123 (숫자만 추출)

        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls[0])                 # category_id
            conf = float(box.conf[0])                # confidence score
            xyxy = box.xyxy[0].tolist()              # [x1, y1, x2, y2]

            cx, cy, w, h = box.xywh[0].tolist()
            x1 = cx - w / 2
            y1 = cy - h / 2
            bbox_w = w
            bbox_h = h

            data.append({
                'annotation_id': annotation_id,
                'image_id': image_id,
                'category_id': cls_id,
                'bbox_x': int(x1),
                'bbox_y': int(y1),
                'bbox_w': int(bbox_w),
                'bbox_h': int(bbox_h),
                'score': round(conf, 4)
            })
            annotation_id += 1

    # 디렉토리가 없으면 생성
    os.makedirs(os.path.dirname(csv_save_path), exist_ok=True)

    # DataFrame으로 저장
    df = pd.DataFrame(data)
    df.to_csv(csv_save_path, index=False)

    print(f"제출 포맷으로 결과 저장 완료: {csv_save_path}")


def main():
    parser = build_parser()
    # 아무 인자 없이 실행 시 도움말 출력
    if len(sys.argv) == 1:
        parser.print_help()
    args = parser.parse_args()
    results = predict_yolo11(args.project, args.name)
    print(results)
    csv_path: str = os.path.join(args.project, args.name, 'submission.csv')
    to_submission_format(results, csv_path)


if __name__ == '__main__':
    main()
