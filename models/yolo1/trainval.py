import torch
from tqdm import tqdm
from config import Config
from utils import Utils
from metrics import compute_metrics, metas_to_gts


# train_one_epoch: box/cls/dfl 누적 → 평균 리턴
def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    sum_total, sum_box, sum_cls, sum_dfl = 0.0, 0.0, 0.0, 0.0
    n = len(loader)

    for imgs, targets, metas in loader: #tqdm(loader, desc='Train batches'):
        imgs, targets = imgs.to(device), targets.to(device)
        preds = model(imgs)
        losses = loss_fn(preds, targets)
        # losses: { 'total_loss', 'box_loss', 'cls_loss', 'dfl_loss' }

        optimizer.zero_grad()
        losses['total_loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=20)  # Yolov1에는 포함이 안되나 gradient 발산을 막기 위해서 추가함
        optimizer.step()

        sum_total += losses['total_loss'].item()
        sum_box += losses['box_loss'].item()
        sum_cls += losses['cls_loss'].item()
        sum_dfl += losses['dfl_loss'].item()

    return {
        'total_loss': sum_total / n,
        'box_loss': sum_box / n,
        'cls_loss': sum_cls / n,
        'dfl_loss': sum_dfl / n
    }


# validate: train과 유사하되, metric 계산 추가
def validate(model, loader, loss_fn, device):
    model.eval()
    with torch.no_grad():
        sum_total, sum_box, sum_cls, sum_dfl = 0.0, 0.0, 0.0, 0.0
        n = len(loader)

        all_preds, all_metas = [], []
        for imgs, targets, metas in loader: #tqdm(loader, desc='Val batches'):
            imgs, targets = imgs.to(device), targets.to(device)

            preds = model(imgs)
            losses = loss_fn(preds, targets)

            sum_total += losses['total_loss'].item()
            sum_box += losses['box_loss'].item()
            sum_cls += losses['cls_loss'].item()
            sum_dfl += losses['dfl_loss'].item()

            # NMS나 후처리된 preds 형식에 맞춰 append
            all_preds.append(preds.detach().cpu())
            all_metas.extend(metas)

        avg_losses = {
            'total_loss': sum_total / n,
            'box_loss': sum_box / n,
            'cls_loss': sum_cls / n,
            'dfl_loss': sum_dfl / n
        }

        # 2) (batch, S, S, D) → (N_images, S, S, D)
        preds_tensor = torch.cat(all_preds, dim=0)   # shape=(N,S,S,D)
        preds_detections = Utils.postprocess(preds_tensor, Config.CONF_THRESH,
                                             Config.NMS_IOU_THRESH, Config.S, Config.B, Config.C)
        metrics = compute_metrics(preds_detections, metas_to_gts(all_metas))
        return avg_losses, metrics
