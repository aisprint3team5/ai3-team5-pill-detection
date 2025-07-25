import os
import torch
from PIL import Image
from torch.utils.data.dataloader import default_collate
from torch.utils.data import Dataset, DataLoader, random_split
from config import Config
from transforms import PillImageTransform


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


class PillYoloDataset(Dataset):
    def __init__(self, image_dir: str, label_dir: str | None = None,
                 S: int = 7, B: int = 2, C: int = 73, transform=None):
        """
        Args:
            image_dir:   이미지 파일(.jpg/.png)들이 있는 폴더
            label_dir:   YOLO 포맷(.txt) 라벨들이 있는 폴더
            S, B, C:     YOLO 타일 크기/박스 수/클래스 수
            class_names: 인덱스→클래스명 맵(예: ['pillA','pillB',...])
            transform:   이미지 전처리/augment 함수 (PIL→Tensor 등)
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.test = (label_dir is None)
        self.S, self.B, self.C = S, B, C
        self.class_names = Config.CLASS_NAMES
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(image_dir)
                                   if f.lower().endswith(('.jpg', '.png'))])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 1) Load image
        fname = self.image_files[idx]
        img_path = os.path.join(self.image_dir, fname)
        img = Image.open(img_path).convert('RGB')
        image_id = os.path.splitext(fname)[0]

        # 2) Load YOLO 라벨: [cls, x_c, y_c, w, h] (normalized) -> 0 ~ 1 사이
        boxes, labels = [], []
        if not self.test:
            label_path = os.path.join(self.label_dir, image_id + '.txt')
            if os.path.exists(label_path):
                with open(label_path) as f:
                    for line in f:
                        cls_id, x_c, y_c, bw, bh = map(float, line.split())
                        boxes.append([x_c, y_c, bw, bh])
                        labels.append(int(cls_id))

        # 3) Image-only transform (boxes는 변형되지 않음)
        if self.transform:
            img = self.transform(img)

        # 4) 타깃 텐서 생성: [S, S, 5*B + C]
        if not self.test:
            target = torch.zeros((self.S, self.S, 5*self.B + self.C), dtype=torch.float32)
            cell_size = 1.0 / self.S

            for (x_c, y_c, bw, bh), cls in zip(boxes, labels):

                # 0~1 범위로 클램프
                x_c = min(max(x_c, 0.0), 1.0)  # x_center 클램프
                y_c = min(max(y_c, 0.0), 1.0)  # y_center 클램프
                bw = min(max(bw, 0.0), 1.0)    # width 클램프
                bh = min(max(bh, 0.0), 1.0)    # height 클램프

                # grid cell 인덱스 계산 (i,j가 0~6 안에 머뭄)
                i = int(y_c / cell_size)  # row
                j = int(x_c / cell_size)  # col

                # 안전하게 범위 고정: x_c 1.0인 경우 i=int(y_c / (1/7))로 7이 될 수 있기 때문에 범위고정 처리를 해줌. y_c의 경우도 마찬가지
                i = min(i, self.S-1)
                j = min(j, self.S-1)

                # 셀 내 상대 좌표
                dx = (x_c - j*cell_size) / cell_size
                dy = (y_c - i*cell_size) / cell_size

                # print('[dx, dy, bw, bh]', target.shape, i, j, torch.tensor([dx, dy, bw, bh]).shape, dx, dy, bw, bh)
                # 첫 번째 예측 박스 slot에 값 할당
                target[i, j, 0:4] = torch.tensor([dx, dy, bw, bh])
                target[i, j, 4] = 1.0               # objectness
                target[i, j, 5*self.B + cls] = 1.0  # class one-hot
        else:
            target = None

        # 5) 메타 구성
        meta = {"image_id": image_id}
        #    bounding box를 VOC 스타일(normalized [x1,y1,x2,y2])로 저장
        if not self.test:
            yolo_boxes = []
            for (x_c, y_c, bw, bh) in boxes:
                x1 = x_c - bw/2
                y1 = y_c - bh/2
                x2 = x_c + bw/2
                y2 = y_c + bh/2
                yolo_boxes.append([x1, y1, x2, y2])
            meta.update({
                'boxes': yolo_boxes,            # normalized YOLO boxes
                'labels': labels,               # class indices
                'label_names': [self.class_names[c] for c in labels]
            })

        return img, target, meta


def load_loaders():
    print(Config.TRAIN_IMAGES_DIR, Config.TRAIN_LABELS_DIR)
    print(Config.VAL_IMAGES_DIR, Config.VAL_LABELS_DIR)
    transform = PillImageTransform(resize=(Config.IMAGE_SIZE, Config.IMAGE_SIZE)) # PillImageTransform AlbumentationTransform
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
