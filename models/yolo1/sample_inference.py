import torch
from config import Config
from utils import Utils
from metrics import metas_to_gts, compute_metrics


def run_sample_inference(model, loader):
    # 추론 예시
    model.eval()
    with torch.no_grad():
        for test_img, targets, metas in loader:  # range(4): # len(val_ds)
            # test_img, _, metas = val_ds[i] # train_ds[i] #val_ds[i]
            # print(test_img.size()) # torch.Size([4, 3, 448, 448])
            preds = model(test_img.to(Config.DEVICE))
            print('target.size()', targets.size())
            print('preds.size()', preds.size())
            detections = Utils.postprocess(preds, Config.CONF_THRESH, Config.NMS_IOU_THRESH, Config.S, Config.B, Config.C)
            Utils.draw_detections_pil(test_img[0], detections[0], Config.CLASS_NAMES)
            Utils.draw_detections_pil(test_img[1], detections[0], Config.CLASS_NAMES)
            print('Sample detections:', detections[0])
            print('Target GTS:', metas_to_gts(metas))
            print('metas: ', metas)
            print('detections.size(): ', detections.size())
            print('Target GTS.size(): ', metas_to_gts(metas).size())
            results = compute_metrics(detections, metas_to_gts(metas))
            print(results)
