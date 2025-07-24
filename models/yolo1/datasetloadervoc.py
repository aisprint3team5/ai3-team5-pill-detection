import torch
import torchvision.transforms as T
from torch.utils.data.dataloader import default_collate
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.datasets import VOCDetection
from config import Config


IS_DEV=True  # True
DEV_TOTAL_SIZE = 4
DEV_BATCH_SIZE = 2

VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]
class_to_idx = {cls_name: i for i, cls_name in enumerate(VOC_CLASSES)}


# ------------------------------------------------------------------------------
# 0. Dataset & Dataloader 예시 (Pascal VOC)
# ------------------------------------------------------------------------------
def my_collate(batch):
    # batch = [(img1, tgt1, meta1), (img2, tgt2, meta2), ...]
    imgs, tgts, metas = zip(*batch)

    # img, tgt는 모두 같은 shape이므로 default_collate 사용
    imgs = default_collate(imgs)    # → tensor (B, C, H, W)
    tgts = default_collate(tgts)    # → tensor (B, S, S, 5+C)

    # metas는 dict마다 길이가 다르니 그냥 리스트로 넘겨줌
    metas = list(metas)             # → [meta1, meta2, ..., metaB]

    return imgs, tgts, metas


class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, root, year='2007', image_set='train', S=7, B=2, C=20, transform=None):
        self.dataset = VOCDetection(root, year=year, image_set=image_set, download=True)
        self.S, self.B, self.C = S, B, C
        self.transform = transform
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, target = self.dataset[idx]
        boxes = []
        labels = []
        label_names = []
        for obj in target['annotation']['object']:
            bbox = obj['bndbox']
            # 원본 좌표 [1..W/H] → normalized [0..1]
            x1 = float(bbox['xmin']) / img.width
            y1 = float(bbox['ymin']) / img.height
            x2 = float(bbox['xmax']) / img.width
            y2 = float(bbox['ymax']) / img.height
            boxes.append([x1, y1, x2, y2])
            cls_name = obj['name']
            labels.append(self.class_to_idx[cls_name])
            label_names.append(cls_name)

        if self.transform:
            img = self.transform(img)

        # target tensor: [S, S, 5B + C], 초기 0
        target_tensor = torch.zeros((self.S, self.S, 5*self.B + self.C))
        cell_size = 1.0 / self.S

        for box, cls in zip(boxes, labels):
            x1, y1, x2, y2 = box
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1

            i = int(y_center / cell_size)
            j = int(x_center / cell_size)
            # cell 내 상대 좌표
            dx = (x_center - j*cell_size) / cell_size
            dy = (y_center - i*cell_size) / cell_size

            # 첫 번째 박스 책임 할당
            target_tensor[i, j, 0:4] = torch.tensor([dx, dy, w, h])
            target_tensor[i, j, 4] = 1
            target_tensor[i, j, 5 * self.B + cls] = 1  # target_tensor[i,j,5 + cls] = 1

        meta = {
            'image_id': idx,
            'boxes': boxes,
            'labels': labels,
            'label_names': label_names
        }

        # print('target_tensor: ', target_tensor.shape)

        return img, target_tensor, meta


def load_loaders():

    # transforms
    transform = T.Compose([
        T.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        # T.RandomHorizontalFlip(0.5),
        # T.ColorJitter(0.2, 0.2, 0.2, 0.1),
        T.ToTensor(),
        # T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    # dataset & loader
    train_ds = VOCDataset(root='./data', image_set='train', transform=transform,
                          S=Config.S, B=Config.B, C=Config.C)
    val_ds = VOCDataset(root='./data', image_set='val', transform=transform,
                        S=Config.S, B=Config.B, C=Config.C)
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
        train_loader = DataLoader(train_ds, batch_size=DEV_BATCH_SIZE, shuffle=True, collate_fn=my_collate)
    else:
        train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, collate_fn=my_collate)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, collate_fn=my_collate)
    return train_loader, val_loader
