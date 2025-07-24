import torch
from yolo1_loss import Yolo1Loss


device = 'cuda' if torch.cuda.is_available() else 'cpu'
S, B, C = 7, 2, 20

# 배치=1, target: (1,7,7,30)
target = torch.zeros(1, S, S, C + B*5, device=device)

# 한 셀(3,4)에만 물체
i, j, cls_idx = 3, 4, 5
dx, dy, w, h = 0.3, 0.7, 0.2, 0.1

# 클래스 원-핫
target[0, i, j, cls_idx] = 1.0
# 첫 번째 박스 conf + coords
target[0, i, j, C] = 1.0
target[0, i, j, C+1:C+5] = torch.tensor([dx, dy, w, h], device=device)
# 두 번째 박스 슬롯은 모두 0

# predictions: perfect prediction → target과 동일한 값을 플래튼
predictions = target.clone().view(1, -1)
loss_fn = Yolo1Loss(S, B, C)
loss = loss_fn(predictions, target)

assert loss['total_loss'].item() == 0.0
assert loss['box_loss'].item() == 0.0
assert loss['cls_loss'].item() == 0.0
assert loss['dfl_loss'].item() == 0.0

print("▶ Minimal test loss:", loss['total_loss'].item())  # → 0.0