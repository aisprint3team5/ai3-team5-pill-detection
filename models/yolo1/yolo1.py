import torch
import torch.nn as nn
import torch.nn.functional as F


class Yolo1(nn.Module):
    # 논문에 제시된 아키텍처 구성 (Yolo1)
    architecture_config = [
        (7, 64, 2, 3),       # (kernel_size, filters, stride, padding)
        "M",                 # maxpool
        (3, 192, 1, 1),
        "M",
        (1, 128, 1, 0),
        (3, 256, 1, 1),
        (1, 256, 1, 0),
        (3, 512, 1, 1),
        "M",
        [(1, 256, 1, 0), (3, 512, 1, 1), 4],  # 해당 블록을 4번 반복
        (1, 512, 1, 0),
        (3, 1024, 1, 1),
        "M",
        [(1, 512, 1, 0), (3, 1024, 1, 1), 2],  # 해당 블록을 2번 반복
        (3, 1024, 1, 1),
        (3, 1024, 2, 1),
        (3, 1024, 1, 1),
        (3, 1024, 1, 1)
    ]

    def __init__(self, in_channels=3, S=7, B=2, C=20, conf_thresh=0.2, iou_thresh=0.4):  # split_size=7, num_boxes=2, num_classes=20
        super(Yolo1, self).__init__()
        self.S, self.B, self.C = S, B, C
        print(in_channels, self.S, self.B, self.C)
        self.conf_thresh, self.iou_thresh = conf_thresh, iou_thresh
        self.features = Yolo1.create_conv_layers(self.architecture_config, in_channels)
        # 입력 이미지가 448x448인 경우, 마지막 컨볼루션 feature map은 7x7 (논문 기준)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * 7 * 7, 4096),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),  # 논문에서 사용한 dropout
            nn.Linear(4096, S * S * (C + B * 5))
        )

    def forward(self, x):
        # x: [B,3,H,W] → features → [B,1024,S,S]
        x = self.features(x)
        # classifier → [B, S*S*(5B + C)]
        x = self.classifier(x)
        # reshape → [B, S, S, 5B + C]
        # YOLOv1 forward torch.Size([16, 7, 7, 30])
        x = x.view(-1, self.S, self.S, 5*self.B + self.C)  # 계산 복잡도를 낮추기 위해 (N, S, S, 5*B+C) 형태로 반환한다.
        preds = x

        # — 활성화 적용 —
        for b in range(self.B):
            off = 5*b
            # t_x, t_y → sigmoid
            preds[..., off:off+2] = torch.sigmoid(preds[..., off:off+2])
            # confidence → sigmoid
            preds[..., off+4:off+5] = torch.sigmoid(preds[..., off+4:off+5])

        # class logits → softmax
        preds[..., 5*self.B:] = F.softmax(preds[..., 5*self.B:], dim=-1)
        print('preds.shape: ', preds.shape)
        return preds

    @staticmethod
    def create_conv_layers(config, in_channels):
        layers = []
        for module in config:
            if type(module) == tuple:
                # 튜플 형태: (kernel_size, filters, stride, padding)
                kernel_size, filters, stride, padding = module
                layers.append(nn.Conv2d(in_channels, filters, kernel_size, stride, padding))
                layers.append(nn.LeakyReLU(0.1))
                in_channels = filters
            elif module == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            elif type(module) == list:
                # 리스트 형태: [ conv1 튜플, conv2 튜플, 반복 횟수 ]
                conv1, conv2, num_repeats = module
                for _ in range(num_repeats):
                    # 첫 번째 컨볼루션
                    k, f, s, p = conv1
                    layers.append(nn.Conv2d(in_channels, f, k, s, p))
                    layers.append(nn.LeakyReLU(0.1))
                    in_channels = f
                    # 두 번째 컨볼루션
                    k, f, s, p = conv2
                    layers.append(nn.Conv2d(in_channels, f, k, s, p))
                    layers.append(nn.LeakyReLU(0.1))
                    in_channels = f
        return nn.Sequential(*layers)
