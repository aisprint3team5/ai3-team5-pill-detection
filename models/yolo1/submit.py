import os
import csv
import time
import yaml
import torch
from torch.nn import Module
from tqdm import tqdm
from config import Config, parse_args
from yolo1 import Yolo1
from utils import Utils
from datasetloader import load_test_loaders

import torch
from config import Config
from utils import Utils


def main():
    print('Start generating csv submission file..')
    start_time: float = time.time()
    # 1) 인자 파싱
    config_file: str = 'config/yolo_1.yaml'
    args = parse_args(config_file)
    # 2) Config 로드 + 덮어쓰기
    Config.load(config_file, vars(args))
    # 3) Train을 시작했을 때 파라미터를 가져오기
    Config.load(config_file, Utils.load_args())

    # 4) Train 때 사용했던 모델과 weight 불러오기
    model: Module = Yolo1(Config.IMAGE_CH_SIZE, Config.S, Config.B, Config.C,
                          Config.CONF_THRESH, Config.NMS_IOU_THRESH).to(Config.DEVICE)
    Utils.load_model(model, 'model')
    print(f'[INFO] Moded is loaded! ({time.time() - start_time:.1f}ms)')

    # 5) Category IDs 가져오기 (index가 class 임)
    with open('category_id.yaml', 'r', encoding='utf-8') as f:
        category_ids = torch.tensor(yaml.safe_load(f).get('category_ids', []), dtype=torch.long)

    # 6) Test Dataloader 가져오기
    test_loader = load_test_loaders()
    print(f'[INFO] Test dataloader is loaded! ({time.time() - start_time:.1f}ms)')

    # 7) Submission csv file 만들기
    submission_csv_path: str = os.path.join(Config.PROJECT, Config.NAME, 'submission.csv')
    write_predictions_to_csv(model, test_loader, category_ids, submission_csv_path)
    print(f'[INFO] CSV generation is completed! ({time.time() - start_time:.1f}ms)')


def write_predictions_to_csv(model, test_loader, category_ids, output_csv):
    '''
    Args:
        model       : 학습된 모델 (eval 모드로 세팅되어 있어야 함)
        test_loader : DataLoader, 반환값 (img, _, metas)
        output_csv  : 저장할 CSV 경로
    '''
    model.eval()
    ann_id = 1  # annotation_id 카운터

    # CSV 헤더
    header = [
        'annotation_id', 'image_id', 'category_id',
        'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h', 'score'
    ]

    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        # 2) 테스트 루프
        for imgs, _, metas in tqdm(test_loader, desc='[INFO] Inference'):
            imgs = imgs.to(Config.DEVICE)

            # 3) 모델 추론 -> preds: (N, S, S, 5B+C)
            preds = model(imgs)

            # 4) 후처리: normalized coords + class_idx → list of Tensor (N,6)
            #    [x1_norm, y1_norm, x2_norm, y2_norm, score, cls_idx]
            detections = Utils.postprocess(preds, Config.CONF_THRESH, Config.NMS_IOU_THRESH, Config.S, Config.B, Config.C)

            # 5) 배치별로 순회하며 CSV에 기록
            #    metas: 리스트 of dicts, each {'image_id':str, 'width':W, 'height':H}
            for det_per_img, meta in zip(detections, metas):
                W, H = meta['width'], meta['height']

                if det_per_img is None or det_per_img.numel() == 0:
                    continue  # detection이 하나도 없으면 스킵

                # det_per_img: Tensor (N,6)
                # 마지막 컬럼은 원래 클래스 인덱스 -> 실제 category_id 로 매핑
                cls_idxs = det_per_img[:, 5].long()
                cat_ids = category_ids[cls_idxs]

                # 6) 픽셀 좌표로 변환: normalize → pixel
                #    out_boxes = det_per_img[:, :4] * torch.tensor([W, H, W, H])
                scales = torch.tensor([W, H, W, H], dtype=det_per_img.dtype, device=det_per_img.device)
                boxes_px = det_per_img[:, :4] * scales  # FloatTensor (N,4)
                scores = det_per_img[:, 4]              # (N,)

                # 7) 한 이미지 안의 모든 박스에 대해 한 줄씩 기록
                for bbox, score, cid in zip(boxes_px, scores, cat_ids):
                    x1, y1, x2, y2 = bbox.tolist()
                    w_box = x2 - x1
                    h_box = y2 - y1

                    writer.writerow([
                        ann_id,
                        meta['image_id'],
                        int(cid.item()),
                        int(x1), int(y1),
                        int(w_box), int(h_box),
                        round(float(score), 4)
                    ])
                    ann_id += 1

    print(f'[INFO] Saved predictions to {output_csv}')


if __name__ == '__main__':
    main()
