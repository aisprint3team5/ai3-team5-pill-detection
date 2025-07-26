import torch
import torch.nn.functional as F


# Yolo1 손실 함수 (채널 매핑: 0–4 box1,4 conf1,5–9 box2,9 conf2,10–29 class)
def Yolo1Loss(S, B, C, lambda_coord=5.0, lambda_noobj=0.5):
    def iou_xyxy(boxes1, boxes2):
        x11, y11, x12, y12 = boxes1.unbind(-1)
        x21, y21, x22, y22 = boxes2.unbind(-1)

        inter_x1 = torch.max(x11, x21)
        inter_y1 = torch.max(y11, y21)
        inter_x2 = torch.min(x12, x22)
        inter_y2 = torch.min(y12, y22)

        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        inter_area = inter_w * inter_h

        area1 = (x12 - x11) * (y12 - y11)
        area2 = (x22 - x21) * (y22 - y21)
        union = area1 + area2 - inter_area
        return inter_area / union.clamp(min=1e-6)

    def xywh_to_xyxy(box):
        cx, cy, w, h = box.unbind(-1)
        x1 = cx - w/2
        y1 = cy - h/2
        x2 = cx + w/2
        y2 = cy + h/2
        return torch.stack([x1, y1, x2, y2], dim=-1)

    def yolo_loss(predictions, target):
        '''
        predictions: (batch, S*S*(5*B + C))
        target:      (batch, S, S, 5*B + C)
        '''
        device = predictions.device

        # 1) reshape to (batch, S, S, 5*B + C)
        preds = predictions.view(-1, S, S, 5*B + C)

        # 2) 그리드 오프셋 생성
        grid_y, grid_x = torch.meshgrid(
            torch.arange(S, device=device),
            torch.arange(S, device=device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).view(1, S, S, 1, 2)

        # 3) 두 박스 슬롯 (cx,cy,w,h) 추출
        rel1 = preds[...,  0:4].view(-1, S, S, 1, 4)
        rel2 = preds[...,  5:9].view(-1, S, S, 1, 4)

        # 4) 셀 내 상대 → 절대좌표 [0..1] 디코딩
        abs1 = torch.cat([
            (grid + rel1[..., :2]) / S,    # center
            rel1[..., 2:4]                 # w,h
        ], dim=-1)  # (batch, S, S, 1, 4)

        abs2 = torch.cat([
            (grid + rel2[..., :2]) / S,
            rel2[..., 2:4]
        ], dim=-1)

        # 5) GT 박스도 동일하게 디코딩 (target[...,0:4])
        gt_rel = target[..., 0:4].view(-1, S, S, 1, 4)
        gt_abs = torch.cat([
            (grid + gt_rel[..., :2]) / S,
            gt_rel[..., 2:4]
        ], dim=-1)

        bb1 = xywh_to_xyxy(abs1).squeeze(3)  # (batch, S, S, 4)
        bb2 = xywh_to_xyxy(abs2).squeeze(3)
        gt = xywh_to_xyxy(gt_abs).squeeze(3)

        # 6) IoU 계산 & 책임 박스 선정
        iou_b1 = iou_xyxy(bb1, gt)
        iou_b2 = iou_xyxy(bb2, gt)
        ious = torch.stack([iou_b1, iou_b2], dim=0)       # (2, batch, S, S)
        iou_maxes, bestbox = torch.max(ious, dim=0)       # (batch,S,S)
        bestbox = bestbox.unsqueeze(-1).float()           # (batch,S,S,1)

        # 7) object mask (conf1 자리: channel 4)
        exists_box = target[..., 4].unsqueeze(-1)         # (batch,S,S,1)

        # 8) Box 좌표 손실
        box_pred = exists_box * (
            bestbox * preds[..., 5:9] +      # slot2 coords
            (1-bestbox) * preds[..., 0:4]    # slot1 coords
        )
        box_tgt = exists_box * target[...,  0:4]

        # sqrt(w), sqrt(h)
        p_wh = torch.sqrt(box_pred[..., 2:4].clamp(min=1e-6))
        t_wh = torch.sqrt(box_tgt[...,  2:4].clamp(min=1e-6))
        box_predxy = torch.cat([box_pred[..., :2], p_wh], dim=-1)
        box_target = torch.cat([box_tgt[..., :2], t_wh], dim=-1)

        box_loss = torch.sum((box_predxy - box_target) ** 2)

        # 9) Object confidence loss
        pred_conf = bestbox * preds[..., 9:10] + (1-bestbox) * preds[..., 4:5]
        conf_target = exists_box * iou_maxes.unsqueeze(-1)
        obj_conf_loss = torch.sum((exists_box * (pred_conf - conf_target)) ** 2)

        # 10) No-object confidence loss
        noobj_mask = 1 - exists_box
        noobj_loss = torch.sum((noobj_mask * preds[..., 4:5]) ** 2)
        noobj_loss += torch.sum((noobj_mask * preds[..., 9:10]) ** 2)

        # 11) Class probability loss (channels 10~10+C): from MSE -> CrossEntropy (11-1 ~ 11-4)
        cls_pred = preds[..., 10:10+C]
        cls_target = target[..., 10:10+C]
        # class_loss = torch.sum((exists_box * (cls_pred - cls_target)) ** 2) # YOLOv1 오리지널 구현(MSE)

        # 11-1) object가 있는 셀만 골라내기
        # exists_box: (batch, S, S, 1) → mask: (batch, S, S)
        obj_mask = exists_box.squeeze(-1).bool()

        # 11-2) 예측과 정답을 2D로 펼치기
        cls_pred_flat = cls_pred[obj_mask]
        cls_target_flat = cls_target[obj_mask]

        # 11-3) one-hot → class index 변환
        #   예: [0,0,1,0] → 2
        target_indices = cls_target_flat.argmax(dim=-1)  # (N_obj,)

        # 11-4) CrossEntropyLoss 계산 (reduction='sum' 으로 MSE와 비슷한 스케일 유지)
        class_loss = F.cross_entropy(cls_pred_flat, target_indices, reduction='sum')

        # 12) 총합
        # total_loss = ( # YOLOv1 논문대로
        #     lambda_coord * box_loss +
        #     obj_conf_loss +
        #     lambda_noobj * noobj_loss +
        #     class_loss
        # )

        total_loss = (
            lambda_coord * box_loss +
            10.0 * obj_conf_loss +
            0.2 * noobj_loss +
            5.0 * class_loss
        )

        batch_size = predictions.size(0)
        # return total_loss / batch_size  # 배치 사이즈로 나눠준다.

        if torch.isnan(total_loss):
            print(f'box_loss={box_loss.item():.6f}, obj_conf={obj_conf_loss.item():.6f}, noobj={noobj_loss.item():.6f}, class={class_loss.item():.6f}')
            print('total before /batch', total_loss)
            print('batch_size:', batch_size)

        return {
            'total_loss': total_loss / batch_size,
            'box_loss':   box_loss / batch_size,
            'cls_loss':   class_loss / batch_size,
            # DFL 대신 confidence losses 합을 dfl_loss로 간주
            'dfl_loss':   (obj_conf_loss + noobj_loss) / batch_size
        }
    return yolo_loss
