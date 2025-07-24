# 시각화, 이미지 저장 등 유틸
import cv2

def visualize_detection(results, save_path=None, show=False):
    """
    YOLOv8 결과 객체를 받아 바운딩 박스를 시각화합니다.
    Args:
        results (list or Results): YOLO 객체에서 반환되는 탐지 결과 객체
        save_path (str or None): 시각화한 이미지를 저장할 경로
        show (bool): OpenCV imshow()로 화면에 결과 표시 여부
    """
    for r in results:
        img = r.orig_img.copy()
        boxes = r.boxes

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = f"{r.names[cls_id]} {conf:.2f}"
            color = (0, 255, 0)

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if save_path:
            cv2.imwrite(save_path, img)
        if show:
            cv2.imshow("YOLOv8 Detection", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()