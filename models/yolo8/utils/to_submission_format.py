import os
import csv
import pandas as pd

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