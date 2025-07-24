import os
import pandas as pd
import asyncio
from utils.build_class_id_map import build_class_id_map

def to_submission_format(model_results, csv_save_path="runs/predictions/test_predictions.csv"):
    data = []
    annotation_dir = r'C:\Users\USER\Desktop\dev\Deep_Learning\codeit_project\ai3-team5-pill-detection\data\raw\train_annotations'

    # category_map, yolo_class_names 생성
    category_map, yolo_class_names = asyncio.run(build_class_id_map(annotation_dir))
    name_to_category_id = {v: k for k, v in category_map.items()}
    yolo_cls_to_category_id = {
        i: name_to_category_id.get(name, -1)
        for i, name in enumerate(yolo_class_names)
    }

    # YOLO 결과 → 리스트 저장
    for result in model_results:
        image_path = result.path
        image_name = os.path.basename(image_path)
        image_id = int(os.path.splitext(image_name)[0])  # 예: '123.png' → 123

        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            category_id = yolo_cls_to_category_id.get(cls_id, -1)
            if category_id == -1:
                print(f"cls_id {cls_id} 의 category_id 매핑 없음 → 건너뜀")
                continue

            cx, cy, w, h = box.xywh[0].tolist()
            x1 = cx - w / 2
            y1 = cy - h / 2

            data.append({
                'image_id': image_id,
                'category_id': category_id,
                'bbox_x': int(x1),
                'bbox_y': int(y1),
                'bbox_w': int(w),
                'bbox_h': int(h),
                'score': round(conf, 2)
            })

    # image_id 기준 정렬
    df = pd.DataFrame(data)
    df = df.sort_values(by="image_id").reset_index(drop=True)

    # annotation_id 재부여 (1부터 시작)
    df.insert(0, 'annotation_id', range(1, len(df) + 1))

    # 디렉토리 생성 및 저장
    os.makedirs(os.path.dirname(csv_save_path), exist_ok=True)
    df.to_csv(csv_save_path, index=False, encoding="utf-8")

    print(f"제출 포맷으로 결과 저장 완료: {csv_save_path}")
