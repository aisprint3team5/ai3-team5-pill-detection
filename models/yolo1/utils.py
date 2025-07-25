import os
import torch
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from config import Config
from torchvision.ops import nms
import yaml


class Utils:
    # ------------------------------------------------------------------------------
    # Post-processing: Decode + NMS
    # ------------------------------------------------------------------------------
    @staticmethod
    def postprocess(output, conf_thresh, iou_thresh, S, B, C, flatten=False):
        boxes, scores, classes = Utils.decode_predictions(output, conf_thresh, S, B, C)
        # print('boxes: ', len(boxes), boxes[0].numel())
        # print('scores: ', len(scores), scores[0].numel())
        # print('classes: ', len(classes), classes[0].numel())
        detections = Utils.apply_nms(boxes, scores, classes, iou_thresh)
        # print('detections: ', len(detections), detections[0].numel())
        if flatten:
            return torch.cat(detections, dim=0)  # if flatten=True : 모든 이미지를 하나의 텐서 (∑K_i, 6)로 합침
        return torch.stack(detections, dim=0)  # return: if flatten=False: 배치별 리스트 of 텐서 (N, K_i, 6) 반환

    # 1) Decode 단계: 모델 출력 → 바운딩박스, 점수, 클래스 리스트로 변환
    @staticmethod
    def decode_predictions(output, conf_thresh, S, B, C):
        '''
        output: [N, S, S, 5B + C]
        returns:
        batch_boxes   : list of N tensors [M_i, 4]  (x1, y1, x2, y2)
        batch_scores  : list of N tensors [M_i]     (score)
        batch_classes : list of N tensors [M_i]     (class_idx)
        '''
        N, device = output.size(0), output.device

        # 1) 셀 오프셋 계산 (한번만)
        grid_y, grid_x = torch.meshgrid(
            torch.arange(S, device=device),
            torch.arange(S, device=device),
            indexing='ij'
        )
        cell_offsets = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(2)
        # shape = (S, S, 1, 2)

        batch_boxes, batch_scores, batch_classes = [], [], []
        for b in range(N):
            single = output[b]  # (S, S, 5*B + C)

            # Model.forward()에서 이미 softmax 함 → 그대로 사용
            cls_probs = single[..., 5*B:]   # (S, S, C)
            # raw_boxes 에는 이미 x,y sigmoid, conf sigmoid
            raw_boxes = single[..., :5*B].view(S, S, B, 5)  # (S, S, B, [x,y,w,h,conf])
            # x,y,conf는 이미 활성화 끝 → 바로 쓰고
            xy = raw_boxes[..., :2]        # (S, S, B, 2)
            conf = raw_boxes[..., 4]         # (S, S, B)
            # w,h는 raw → 논문대로 square
            wh = raw_boxes[..., 2:4].pow(2)  # (S, S, B, 2)       
            # 두 박스 중 objectness(conf) 기준 책임박스 하나만
            best_conf, best_idx = conf.max(dim=-1)  # (S, S)

            boxes, scores, classes = [], [], []
            for i in range(S):
                for j in range(S):
                    score_obj = best_conf[i, j].item()
                    if score_obj < conf_thresh:
                        continue

                    bi = best_idx[i, j].item()  # 0 or 1

                    # 2-1) 셀 오프셋 + 상대→절대 [0..1]
                    x_rel, y_rel = xy[i, j, bi]
                    x_center = (cell_offsets[i, j, 0, 0] + x_rel) / S
                    y_center = (cell_offsets[i, j, 0, 1] + y_rel) / S

                    # 2-2) square→w,h
                    w_rel, h_rel = wh[i, j, bi]

                    # 3) (cx,cy,w,h) → (x1,y1,x2,y2)
                    x1 = x_center - w_rel/2
                    y1 = y_center - h_rel/2
                    x2 = x_center + w_rel/2
                    y2 = y_center + h_rel/2
                    boxes.append([x1, y1, x2, y2]) # 이미지 전체 기준의 normalized 좌표(0…1)입니다.

                    # 4) 클래스 점수 계산 (conf * cls_prob)
                    prob = cls_probs[i,j]           # (C,)
                    cls_prob, cls_idx = prob.max(dim=-1)
                    scores.append((score_obj * cls_prob).item())
                    classes.append(cls_idx.item())                    
            if boxes:
                batch_boxes.append(torch.tensor(boxes, dtype=torch.float32).to(device))  # boxes는 List[List[Tensor]]
                batch_scores.append(torch.tensor(scores, dtype=torch.float32).to(device))  # scores는 List[Tensor]
                batch_classes.append(torch.tensor(classes, dtype=torch.long).to(device))  # classes도 List[Tensor]
            else:
                batch_boxes.append(torch.zeros((0, 4)).to(device))
                batch_scores.append(torch.zeros((0,)).to(device))
                batch_classes.append(torch.zeros((0,), dtype=torch.long).to(device))
        return batch_boxes, batch_scores, batch_classes

    # 2) NMS 단계: 클래스별로 Non-Maximum Suppression 적용
    @staticmethod
    def apply_nms(batch_boxes, batch_scores, batch_classes, iou_thresh):
        '''
        batch_boxes   : list of N tensors [M_i, 4]  (x1, y1, x2, y2)
        batch_scores  : list of N tensors [M_i]     (score)
        batch_classes : list of N tensors [M_i]     (class_idx)
        iou_thresh    : IoU 임계값 (e.g. 0.4)
        returns       : list of N tensors [K_i, 6]  (x1, y1, x2, y2, score, cls)
        '''
        batch_detections = []

        for boxes, scores, classes in zip(batch_boxes, batch_scores, batch_classes):
            if boxes.numel() == 0:  # numel: 텐서가 담고 있는 전체 원소 개수를 반환. (M, 4)면 M*4이고, (M, 0) or (0, 4)면 0이다.
                batch_detections.append(torch.zeros((0,6), dtype=boxes.dtype, device=boxes.device)) # [x1, y1, x2, y2, score, class_idx]
                continue

            kept = []  # 한 이미지 내에서 클래스별 NMS를 거쳐 최종적으로 남은 박스 텐서들을 임시로 담아두는 파이썬 리스트
            for cls_id in classes.unique(): # 해당 이미지에서 예측된 클래스들(중복 제거)을 리스트로 얻음
                mask = (classes == cls_id) # (M,) 클래스별 박스만 선택
                cls_boxes = boxes[mask]   # (m,4): (M, 4)는 아직 어떤 클래스 기준으로도 걸러내지 않은 상태. (m, 4)는 '지금 보고 있는 클래스'에 속하는 박스만 남긴 결과
                cls_scores = scores[mask]  # (m,): 클래스별 박스 수

                # 예시
                # boxes = torch.tensor([
                #    [0,0,10,10],    # idx 0
                #    [1,1,11,11],    # idx 1
                #    [50,50,60,60],  # idx 2
                #    [70,70,80,80]   # idx 3
                # ])
                # scores = torch.tensor([0.9, 0.8, 0.7, 0.3])
                # keep = nms(boxes, scores, iou_thresh=0.4)
                # print(keep)  # tensor([0, 2, 3])
                # idx의 score가 0.8인데도 제거한 이유는 점수가 높은 1보다도 점수가 맞기 때문임.
                # idx 0와 idx 1의 IoU는 0.68로 iou_thresh 0.4 보다 높으므로 1이 제거됨
                # idx 3은 score가 0.3이지만 iou의 대상자체가 아니기에 출력에 포함. IoU는 겹칠때 중복된 상자를 삭제할때 쓰는 로직임
                keep_idxs = nms(cls_boxes, cls_scores, iou_thresh)  # 박스들의 index 리스트

                if keep_idxs.numel() > 0:
                    cls_id_col = torch.full((keep_idxs.numel(), 1), cls_id, dtype=boxes.dtype, device=boxes.device)
                    selected = torch.cat([
                        cls_boxes[keep_idxs],  # (k, 4) -> k <= m
                        cls_scores[keep_idxs].unsqueeze(1),  # (k, 1) -> k <= m
                        cls_id_col  # (k, 1) -> k <= m
                    ], dim=1)  # dim이 1이므로 열방향으로 concat한다. (k,6)
                    kept.append(selected)

            if kept:
                # if kept가 [torch.Size([2,6]), torch.Size([1,6]), torch.Size([4, 6])] 면, batch_detections는 torch.Size([7,6])
                batch_detections.append(torch.vstack(kept))  # kept 안의 [k,6] 텐서를 이어 붙여 [K,6] 생성
            else:
                batch_detections.append(torch.zeros((0, 6), dtype=boxes.dtype, device=boxes.device))
        return batch_detections

    def load_model(model, name):
        folder: str = os.path.join(Config.PROJECT, Config.NAME)
        model_path: str = os.path.join(folder, f'{name}.pth')
        ckpt_file = Path(model_path)
        if ckpt_file.is_file():
            state = torch.load(ckpt_file, map_location=torch.device(Config.DEVICE))
            model.load_state_dict(state)
            print(f'Loaded checkpoint from {ckpt_file}')
        else:
            print(f'No checkpoint at {ckpt_file}, skipping load')
            input_file = Path(f'/kaggle/input/modelpth/{name}.pth')
            if input_file.is_file():
                state = torch.load(input_file, map_location=torch.device(Config.DEVICE))
                model.load_state_dict(state)
                print(f'Loaded input from {input_file} from kaggle')
            else:
                print(f'No loaded input at {ckpt_file}, skipping load')

        def save_model():
            os.makedirs(folder, exist_ok=True)
            torch.save(model.state_dict(), ckpt_file)
        return save_model

    def save_args():
        _, args_arr = argparse.ArgumentParser().parse_known_args()
        folder: str = os.path.join(Config.PROJECT, Config.NAME)
        args_path: str = os.path.join(folder, 'args.yaml')
        args = {}
        key = None
        for token in args_arr:  # ex) ['--name', 'exp7', '--epochs', '2']
            if token.startswith("--"):
                key = token[2:]            # '--name' -> 'name'
                args[key] = None
            else:
                if key is not None:
                    args[key] = token      # 'exp7'
                    key = None

        os.makedirs(folder, exist_ok=True)
        with open(args_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(args, f, default_flow_style=False, allow_unicode=True)
        print(f"[INFO] saved args -> {args_path}")

    def load_args():
        folder: str = os.path.join(Config.PROJECT, Config.NAME)
        args_path: str = os.path.join(folder, 'args.yaml')
        with open(args_path, 'r', encoding='utf-8') as f:
            args = yaml.safe_load(f)
        return args

    def draw_detections_pil(image_tensor, detections, class_names, output_path=None):
        '''
        image_tensor : torch.Tensor (3, H, W), float [0,1]
        detections   : torch.Tensor or array (K,6) [x1,y1,x2,y2,score,cls_idx], normalized
        class_names  : 클래스 이름 리스트
        '''
        # 1) Tensor → H×W×3 uint8 → PIL
        img_np = (image_tensor.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy())
        pil_img = Image.fromarray(img_np)
        draw = ImageDraw.Draw(pil_img)

        # 2) 폰트 준비
        try:
            font = ImageFont.truetype('arial.ttf', size=16)
        except IOError:
            font = ImageFont.load_default()

        W, H = pil_img.size
        dets = detections.detach().cpu().tolist()

        for x1, y1, x2, y2, score, cls_idx in dets:
            # 픽셀 좌표로 변환
            x1p, y1p = int(x1*W), int(y1*H)
            x2p, y2p = int(x2*W), int(y2*H)

            # 3) 바운딩 박스
            draw.rectangle([x1p, y1p, x2p, y2p], outline='lime', width=2)

            # 4) 라벨 문자열
            label = f'{class_names[int(cls_idx)]}:{score:.2f}'

            # mask.size 로 텍스트 크기 구하기
            mask = font.getmask(label)
            tw, th = mask.size

            # 5) 라벨 배경 그리기
            bg_xy = [x1p, y1p - th - 4, x1p + tw + 4, y1p]
            draw.rectangle(bg_xy, fill='black')

            # 6) 흰색 텍스트
            draw.text((x1p+2, y1p-th-2), label, font=font, fill='white')

        # 7) 저장 또는 Matplotlib 표시
        if output_path:
            pil_img.save(output_path)
        else:
            plt.figure(figsize=(8, 6))
            plt.imshow(pil_img)
            plt.axis('off')
            plt.show()

        return pil_img
