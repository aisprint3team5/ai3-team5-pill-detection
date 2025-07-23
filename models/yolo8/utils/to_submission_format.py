import os
import csv

def to_submission_format(model_result, submission_file):
    with open(submission_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["annotation_id", "image_id", "category_id", "bbox_x", "bbox_y", "bbox_w", "bbox_h", "score"])

        for result in model_result:
            image_path = result.path  # 원본 이미지 경로
            image_id = os.path.splitext(os.path.basename(image_path))[0]  # 숫자만 추출할 수도 있음

            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                w, h = x2 - x1, y2 - y1
                cls = int(box.cls[0].item())  # class index
                conf = float(box.conf[0].item())  # confidence score

                writer.writerow([
                    annotation_id,
                    image_id,
                    cls,
                    int(x1), int(y1), int(w), int(h),
                    round(conf, 4)
                ])
                annotation_id += 1

    print(f"[INFO] 제출용 CSV 생성 완료 → {submission_file}")