import torch
from metrics import compute_metrics

# 샘플 detections: shape (B, N_obj, 6) → [x1, y1, x2, y2, score, cls]
detections = torch.tensor([
    # image 0: 3개 예측
    [
        [ 50,  50, 150, 150, 0.90, 0],  # TP(class0)
        [140, 140, 240, 240, 0.60, 0],  # IoU 낮아 FP
        [300, 300, 400, 400, 0.70, 1],  # TP(class1)
    ],
    # image 1: 3개 예측
    [
        [100, 100, 200, 200, 0.85, 1],  # TP(class1)
        [100, 100, 200, 200, 0.50, 0],  # class mismatch → FP
        [500, 500, 600, 600, 0.40, 0],  # no GT overlap → FP
    ]
], dtype=torch.float32)

# 샘플 GT: shape (B, N_gt, 5) → [x1, y1, x2, y2, cls]
gts = torch.tensor([
    # image 0: 2개 GT
    [
        [ 45,  45, 155, 155, 0],  # class0
        [295, 295, 405, 405, 1],  # class1
    ],
    # image 1: 2개 GT
    [
        [105, 105, 195, 195, 1],  # class1
        [495, 495, 605, 605, 1],  # class1 (second GT)
    ]
], dtype=torch.float32)

# 계산 함수 호출
metrics = compute_metrics(
    detections,            # torch.Tensor (B, N_obj, 6)
    gts,                   # torch.Tensor (B, N_gt, 5)
    iou_thresh=0.5,        # TP/FP 판정용 IoU
    map_iou_range=(0.5, 0.95),
    map_steps=10
)

'''
1. mAP@0.5 계산
클래스별로 AP@0.5를 구해서 평균을 내는데,
class0: GT=1, Preds 순위별 TP/FP = [TP, FP, FP, FP] → Recall curve: [1/1, 1/1, 1/1, 1/1] → Precision curve: [1.0, 0.5, 0.33, 0.25] → AP@0.5 = 1.0
class1: GT=3, Preds 순위별 TP/FP = [TP, TP] → Recall curve: [1/3≈0.3333, 2/3≈0.6667] → Precision curve: [1.0, 1.0] → AP@0.5 = (0.3333–0)1 + (0.6667–0.3333)1 = 0.6667
따라서 mAP50 = (1.0 + 0.6667) / 2 ≈ 0.8333

2. mAP@[0.5:0.95] 계산
IoU 임계치를 0.5, 0.55, …, 0.95 총 10단계로 바꿔서 각 단계마다 AP를 계산한 뒤 평균을 냅니다.
class0은 모든 단계에서 완벽 검출 → AP=1.0
class1은 IoU 요구조건이 높아질수록(예: 0.75, 0.85) TP가 하나만 남거나 전부 사라지는 단계가 있어 AP가 0.x 이하로 떨어집니다. 예를 들어
- IoU >= 0.5,0.55…0.7까지는 TP=2 → AP≈0.6667
- IoU >= 0.75부터는 TP=1 → AP≈0.3333
- IoU >= 0.9에서는 TP=0 → AP=0 이런 값들을 모두 더하고 10으로 나누면 class1 AP@[0.5:0.95]≈0.1667이 되고 두 클래스를 평균내면 mAP50-95≈(1.0 + 0.1667)/2 ≈ 0.5833 이 되는 겁니다.
'''

assert metrics['precision'] >= 0.49999 and metrics['precision'] <= 0.5 # 전체 예측 6개 중 TP는 3개, FP는 3개 → Precision = 3/(3+3) = 0.5
assert metrics['recall'] >= 0.749999 and metrics['recall'] <= 0.75     # 전체 GT 4개 중 TP는 3개, FN은 1개 → Recall = 3/(3+1) = 0.75
assert metrics['mAP50'] >= 0.83333 and metrics['mAP50'] <= 83334
assert metrics['mAP5095'] >= 0.58333 and metrics['mAP5095'] <= 0.58334

print(metrics)
# 예상 출력: {'precision': 0.49999991666668053, 'recall': 0.7499998125000469, 'mAP50': 0.8333320555572314, 'mAP5095': 0.583332438890062}
