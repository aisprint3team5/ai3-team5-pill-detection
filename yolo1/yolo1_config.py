import torch

class Yolo1Config:
    S = 7               # Grid size
    B = 2               # Bounding boxes per cell
    C = 20              # Classes (Pascal VOC has 20)
    IMAGE_SIZE = 448    # Input 이미지 크기
    IMAGE_CH_SIZE = 3
    BATCH_SIZE = 16
    LR = 1e-5  # 1e-5 #1e-5 #1e-4 #3e-5 #1e-4 #1e-5 # 1e-4
    WD = 1e-6  # 5e-4
    EPOCHS = 40
    CONF_THRESH = 0.2
    NMS_IOU_THRESH = 0.4

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
