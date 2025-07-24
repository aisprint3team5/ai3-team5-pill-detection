import os
import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils.data.dataloader import default_collate
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import ToTensor
from config import Config

# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# import os
# import numpy as np
# from PIL import Image

# TODO: 추후에 삭제 할 것
IS_DEV = False
DEV_TOTAL_SIZE = 8
DEV_BATCH_SIZE = 4

def my_collate(batch):
    # batch = [(img1, tgt1, meta1), (img2, tgt2, meta2), ...]
    imgs, tgts, metas = zip(*batch)

    # img, tgt는 모두 같은 shape이므로 default_collate 사용
    imgs = default_collate(imgs)    # → tensor (B, C, H, W)
    tgts = default_collate(tgts)    # → tensor (B, S, S, 5+C)

    # metas는 dict마다 길이가 다르니 그냥 리스트로 넘겨줌
    metas = list(metas)             # → [meta1, meta2, ..., metaB]

    return imgs, tgts, metas

def preprocess_pill_image(img: np.ndarray) -> np.ndarray:
    '''
    Preprocesses the pill image using grayscale, CLAHE, and optional adaptive thresholding.

    Args:
        img (np.ndarray): Input BGR image from cv2.imread.

    Returns:
        np.ndarray: Preprocessed grayscale image (still 3-channel for compatibility).
    '''
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    enhanced_3ch = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    return enhanced_3ch


class PillImageTransform:
    def __init__(self, resize=(640, 640)):
        self.resize = resize
        self.to_tensor = ToTensor()

    def __call__(self, img_pil: Image.Image):
        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        processed_cv = preprocess_pill_image(img_cv)
        processed_cv = cv2.resize(processed_cv, self.resize)
        # Convert back to PIL for torchvision
        processed_pil = Image.fromarray(cv2.cvtColor(processed_cv, cv2.COLOR_BGR2RGB))
        return self.to_tensor(processed_pil)


class PillYoloDataset(Dataset):
    def __init__(self, image_dir, label_dir, S=7, B=2, C=20, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
        self.S = S
        self.B = B
        self.C = C
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        image_filename = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_filename)
        image = Image.open(image_path).convert('RGB')
        w, h = image.size

        # Load label
        label_path = os.path.join(self.label_dir, os.path.splitext(image_filename)[0] + '.txt')
        boxes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    cls_id, x_center, y_center, box_w, box_h = map(float, line.strip().split())
                    boxes.append([int(cls_id), x_center, y_center, box_w, box_h])

        boxes = torch.tensor(boxes) if boxes else torch.zeros((0, 5))  # Shape: (num_boxes, 5)

        if self.transform:
            image = self.transform(image)

        print(image.shape)
        print(boxes)
        print(boxes.shape)

        return image, boxes


def load_loaders():
    print(Config.TRAIN_IMAGES_DIR, Config.TRAIN_LABELS_DIR)
    print(Config.VAL_IMAGES_DIR, Config.VAL_LABELS_DIR)
    transform = PillImageTransform(resize=(Config.IMAGE_SIZE, Config.IMAGE_SIZE))
    train_ds = PillYoloDataset(image_dir=Config.TRAIN_IMAGES_DIR, label_dir=Config.TRAIN_LABELS_DIR,
                               S=Config.S, B=Config.B, C=Config.C, transform=transform)
    val_ds = PillYoloDataset(image_dir=Config.VAL_IMAGES_DIR, label_dir=Config.VAL_LABELS_DIR,
                             S=Config.S, B=Config.B, C=Config.C, transform=transform)
    if IS_DEV:
        # train_ds, _ = random_split(train_ds, [DEV_TOTAL_SIZE, len(train_ds)-DEV_TOTAL_SIZE])
        val_ds, _ = random_split(val_ds,   [DEV_TOTAL_SIZE, len(val_ds)-DEV_TOTAL_SIZE])

        # 한개 이미지로 32개를 만들어서 오버핏 테스트
        class OverfitDataset(Dataset):
            '''
            동일한 (img, tgt, meta) 한 쌍을 batch_size만큼 복제해서
            __getitem__에서 (img, tgt, meta) 튜플을 반환하도록 합니다.
            '''
            def __init__(self, img0, tgt0, meta0, batch_size):
                # img0: (C,H,W) tensor, tgt0: (S,S,5+C) tensor, meta0: any picklable object
                self.imgs = img0.unsqueeze(0).repeat(batch_size, 1, 1, 1)
                self.tgts = tgt0.unsqueeze(0).repeat(batch_size, 1, 1, 1)
                # meta0가 튜플(예: (image_id, boxes, labels, label_names))이라면
                # 같은 객체를 batch_size번 참조해도 무방합니다.
                self.metas = [meta0 for _ in range(batch_size)]

            def __len__(self):
                return len(self.imgs)

            def __getitem__(self, idx):
                # idx번째에 대응하는 (img, tgt, meta) 튜플을 반환
                return self.imgs[idx], self.tgts[idx], self.metas[idx]    
        img0, tgt0, meta0 = train_ds[1]
        train_ds = OverfitDataset(img0, tgt0, meta0, DEV_TOTAL_SIZE)
        # 배치 차원으로 repeat
        # imgs_batch   = img0.unsqueeze(0).repeat(DEV_TOTAL_SIZE, 1, 1, 1)      # (32, C, H, W)
        # tgts_batch   = tgt0.unsqueeze(0).repeat(DEV_TOTAL_SIZE, 1, 1, 1)      # (32, S, S, 5+C)

        # overfit_ldr  = DataLoader(overfit_ds, batch_size=32, shuffle=True)
    if IS_DEV:
        train_loader = DataLoader(train_ds, batch_size=DEV_BATCH_SIZE, shuffle=False, collate_fn=my_collate)
    else:
        train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, collate_fn=my_collate)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, collate_fn=my_collate)
    return train_loader, val_loader
