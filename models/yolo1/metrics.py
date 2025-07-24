import numpy as np
import torch


def compute_metrics(
    detections,      # torch.Tensor, (B, N_obj, 6) → [x1,y1,x2,y2,score,cls]
    gts,             # torch.Tensor, (B, N_gt, 5) →  [x1,y1,x2,y2,cls]
    iou_thresh=0.5,          # PR 계산 TP/FP 판정용 IoU
    map_iou_range=(0.5, 0.95),  # mAP 계산용 IoU 범위
    map_steps=10             # mAP50-95 계산 시 IoU 샘플 수
):
    B = detections.shape[0]

    # 1) per-image → flatten into one big dets, gts array with img_id
    all_dets, all_gts = [], []
    for img_id in range(B):
        det = detections[img_id].cpu().numpy()  # (N_obj,6)
        gt = gts[img_id].cpu().numpy()         # (N_gt,5)

        # dets: [img_id,x1,y1,x2,y2,score,cls]
        for x1, y1, x2, y2, score, cls in det:
            all_dets.append([img_id, x1, y1, x2, y2, score, int(cls)])
        # gts: [img_id,x1,y1,x2,y2,cls]
        for x1, y1, x2, y2, cls in gt:
            all_gts.append([img_id, x1, y1, x2, y2, int(cls)])

    if not all_dets or not all_gts:
        return {'precision': 0., 'recall': 0., 'mAP50': 0., 'mAP5095': 0.}

    dets = np.array(all_dets)  # (M,7)
    gts = np.array(all_gts)   # (N,6)

    # 2) 클래스별 AP 계산
    classes = np.unique(gts[:,5]).astype(int)
    ap50_list, ap_list5095 = [], []

    for cls in classes:
        # 이 클래스의 GT, det 분리
        cls_gts = gts[gts[:, 5] == cls]
        cls_dets = dets[dets[:, 6] == cls]

        n_gt = cls_gts.shape[0]
        if n_gt == 0 or cls_dets.shape[0] == 0:
            continue

        # score 내림차순
        order = np.argsort(-cls_dets[:, 5])
        cls_dets = cls_dets[order]

        # TP/FP per det @ IoU=0.5
        tp = np.zeros(len(cls_dets))
        fp = np.zeros(len(cls_dets))
        detected = set()
        for i, det in enumerate(cls_dets):
            img_id = int(det[0])
            box = det[1:5]
            # 해당 이미지 GT만
            gt_img = cls_gts[cls_gts[:, 0] == img_id][:, 1:5]
            if gt_img.size == 0:
                fp[i] = 1
                continue
            ious = bbox_iou_np(box, gt_img)
            best = np.argmax(ious)
            if ious[best] >= iou_thresh and (img_id, best) not in detected:
                tp[i] = 1
                detected.add((img_id, best))
            else:
                fp[i] = 1

        # Precision/Recall curve @ IoU=0.5
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recall = tp_cum / (n_gt + 1e-6)
        precision = tp_cum / (tp_cum + fp_cum + 1e-6)

        # AP@0.5
        ap50 = compute_ap(recall, precision)
        ap50_list.append(ap50)

        # AP@[0.5:0.95]
        ap_iou = []
        for t in np.linspace(map_iou_range[0], map_iou_range[1], map_steps):
            # det-GT 매칭을 IoU 문턱만 바꿔 재계산
            tp_t = np.zeros(len(cls_dets))
            fp_t = np.zeros(len(cls_dets))
            detected_t = set()
            for i, det in enumerate(cls_dets):
                img_id = int(det[0])
                box = det[1:5]
                gt_img = cls_gts[cls_gts[:, 0] == img_id][:, 1:5]
                if gt_img.size == 0:
                    fp_t[i] = 1
                    continue
                ious = bbox_iou_np(box, gt_img)
                best = np.argmax(ious)
                if ious[best] >= t and (img_id, best) not in detected_t:
                    tp_t[i] = 1
                    detected_t.add((img_id, best))
                else:
                    fp_t[i] = 1
            tp_c = np.cumsum(tp_t)
            fp_c = np.cumsum(fp_t)
            rec_t = tp_c / (n_gt + 1e-6)
            prec_t = tp_c / (tp_c + fp_c + 1e-6)
            ap_iou.append(compute_ap(rec_t, prec_t))
        ap_list5095.append(np.mean(ap_iou))

    # 3) 전체 Precision/Recall @ IoU=0.5
    # dets와 gts를 전부 합쳐 같은 방식으로 계산
    tp_all = []
    fp_all = []
    for cls in classes:
        cls_gts = gts[gts[:, 5] == cls]
        cls_dets = dets[dets[:, 6] == cls]
        if cls_dets.shape[0] == 0:
            continue
        order = np.argsort(-cls_dets[:, 5])
        cls_dets = cls_dets[order]
        tp = np.zeros(len(cls_dets))
        fp = np.zeros(len(cls_dets))
        detected = set()
        for i, det in enumerate(cls_dets):
            img_id = int(det[0])
            box = det[1:5]
            gt_img = cls_gts[cls_gts[:, 0] == img_id][:, 1:5]
            if gt_img.size == 0:
                fp[i] = 1
                continue
            ious = bbox_iou_np(box, gt_img)
            best = np.argmax(ious)
            if ious[best] >= iou_thresh and (img_id, best) not in detected:
                tp[i] = 1
                detected.add((img_id, best))
            else:
                fp[i] = 1
        tp_all.append(tp)
        fp_all.append(fp)

    tp_all = np.concatenate(tp_all) if tp_all else np.array([])
    fp_all = np.concatenate(fp_all) if fp_all else np.array([])
    tp_cum_all = np.cumsum(tp_all)
    fp_cum_all = np.cumsum(fp_all)
    total_gt = gts.shape[0]
    recall_all = tp_cum_all / (total_gt + 1e-6)
    precision_all = tp_cum_all / (tp_cum_all + fp_cum_all + 1e-6)

    return {
        'precision': float(precision_all[-1]) if precision_all.size else 0.0,
        'recall':    float(recall_all[-1]) if recall_all.size else 0.0,
        'mAP50':     float(np.mean(ap50_list)) if ap50_list else 0.0,
        'mAP5095':  float(np.mean(ap_list5095)) if ap_list5095 else 0.0
    }


def bbox_iou_np(box, boxes):
    ix1 = np.maximum(box[0], boxes[:, 0])
    iy1 = np.maximum(box[1], boxes[:, 1])
    ix2 = np.minimum(box[2], boxes[:, 2])
    iy2 = np.minimum(box[3], boxes[:, 3])
    iw = np.clip(ix2 - ix1, a_min=0, a_max=None)
    ih = np.clip(iy2 - iy1, a_min=0, a_max=None)
    inter = iw * ih
    area1 = (box[2]-box[0])*(box[3]-box[1])
    area2 = (boxes[:, 2]-boxes[:, 0])*(boxes[:, 3]-boxes[:, 1])
    union = area1 + area2 - inter + 1e-6
    return inter / union


def compute_ap(recall, precision):
    # VOC 방식 AP 계산
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))
    for i in range(mpre.size-1, 0, -1):
        mpre[i-1] = np.maximum(mpre[i-1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return np.sum((mrec[idx+1] - mrec[idx]) * mpre[idx+1])


def metas_to_gts(metas, pad_value=0.0):
    '''
    metas: list of length B, each item is a dict with keys
      - 'boxes': list of [x1, y1, x2, y2]
      - 'labels': list of class indices
    pad_value: padding할 때 쓸 값 (boxes와 labels 모두에 적용)
    returns:
      tensor of shape (B, M, 5), where
        B = len(metas)
        M = max number of objects in any image
      각 [i, j] 자리에 (x1, y1, x2, y2, label) 또는 pad_value
    '''
    B = len(metas)
    # 이미지마다 object 개수
    num_objs = [len(m['boxes']) for m in metas]
    M = max(num_objs)  # 배치에서 가장 객체가 많은 수

    # 결과 텐서 초기화 (float) → [x1, y1, x2, y2, label]
    out = torch.full((B, M, 5), pad_value, dtype=torch.float)

    for i, m in enumerate(metas):
        n = num_objs[i]
        if n == 0:
            continue  # 해당 이미지에 객체가 하나도 없으면 skip

        # 1) boxes → (n,4) tensor
        boxes = torch.tensor(m['boxes'], dtype=torch.float)        # (n,4)

        # 2) labels → (n,1) tensor
        labels = torch.tensor(m['labels'], dtype=torch.float)     # (n,)
        labels = labels.view(-1, 1)                               # (n,1)

        # 3) concat → (n,5)
        objs = torch.cat([boxes, labels], dim=1)                  # (n,5)

        # 4) 결과 텐서에 복사
        out[i, :n] = objs

    return out
