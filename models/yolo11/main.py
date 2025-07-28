#!/usr/bin/env python3
import os
import sys
import argparse
import torch
import yaml                                  # pip install pyyaml
from ultralytics import YOLO
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import torch.optim as optim


# ─── 1) 외부 YAML 읽어 DEFAULTS 생성 ────────────────────────────
with open("config/yolo_11.yaml", "r", encoding="utf-8") as f:
    DEFAULTS = yaml.safe_load(f)

# ─── 2) argparse 인자 설정 정보 정의 ───────────────────────────
ARGUMENTS: list[dict[str, object]] = [
    # flags, type, default, help…
    {'flags': ['--weights'],       'type': str,   'default': DEFAULTS['weights'],
                                     'help': 'pretrained weights or model .pt path'},
    {'flags': ['--data'],          'type': str,   'default': DEFAULTS['data'],
                                     'help': 'dataset yaml file (train/val paths, nc, names)'},
    {'flags': ['--epochs'],        'type': int,   'default': DEFAULTS['epochs'],
                                     'help': 'number of training epochs'},
    {'flags': ['--batch'],         'type': int,   'default': DEFAULTS['batch'],
                                     'help': 'batch size'},
    {'flags': ['--imgsz'],         'type': int,   'default': DEFAULTS['imgsz'],
                                     'help': 'input image size (HxW)'},
    {'flags': ['--optimizer'],     'type': str,   'choices': ['SGD','Adam','AdamW'],
                                     'default': DEFAULTS['optimizer'],
                                     'help': 'optimizer type'},
    {'flags': ['--lr0'],           'type': float, 'default': DEFAULTS['lr0'],
                                     'help': 'initial learning rate'},
    {'flags': ['--lrf'],           'type': float, 'default': DEFAULTS['lrf'],
                                     'help': 'final LR ratio (cosine scheduler)'},
    {'flags': ['--momentum'],      'type': float, 'default': DEFAULTS['momentum'],
                                     'help': 'SGD momentum (only if optimizer=SGD)'},
    {'flags': ['--betas'],         'type': lambda s: tuple(map(float, s.split(','))),
                                     'default': tuple(DEFAULTS['betas']),
                                     'help': 'Adam/AdamW betas as "beta1,beta2"'},
    {'flags': ['--weight_decay'],  'type': float, 'default': DEFAULTS['weight_decay'],
                                     'help': 'weight decay (L2)'},
    {'flags': ['--warmup_epochs'], 'type': int,   'default': DEFAULTS['warmup_epochs'],
                                     'help': 'number of warmup epochs'},
]


def build_parser():
    parser = argparse.ArgumentParser(
        description="Ultralytics YOLOv11 Training Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 공통 인자 추가
    for arg in ARGUMENTS:
        flags = arg.pop('flags')
        parser.add_argument(*flags, **arg)

    # 증강 옵션 (mutually exclusive)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--augment',    dest='augment', action='store_true',  help='enable augmentations')
    group.add_argument('--no-augment', dest='augment', action='store_false', help='disable augmentations')
    parser.set_defaults(augment=DEFAULTS['augment'])

    # 프로젝트/이름
    parser.add_argument('--project', type=str, default=DEFAULTS['project'],
                        help='save results to project/name')
    parser.add_argument('--name',    type=str, required=True, default=DEFAULTS['name'],
                        help='experiment name (subfolder)')

    return parser


def train_yolo11(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO(args.weights)

    # optimizer별 옵션 분기
    opt_kwargs = {'lr0': args.lr0, 'lrf': args.lrf, 'weight_decay': args.weight_decay}
    if args.optimizer.lower() == 'sgd':
        opt_kwargs['momentum'] = args.momentum
    elif args.optimizer.lower() == 'adamw':
        opt = optim.AdamW(
            model.model.parameters(),
            lr=args.lr0,
            betas=args.betas,
            weight_decay=args.weight_decay
        )
        model.trainer.optimizer = opt
    else:
        opt_kwargs['betas'] = args.betas

    return model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        optimizer=args.optimizer,
        device=device,
        warmup_epochs=args.warmup_epochs,
        augment=args.augment,
        project=args.project,
        name=args.name,
        **opt_kwargs
    )


def main():
    parser = build_parser()

    # 아무 인자 없이 실행 시 도움말 출력
    if len(sys.argv) == 1:
        parser.print_help()
        # sys.exit(0)

    args = parser.parse_args()
    results = train_yolo11(args)
    print(results)


if __name__ == '__main__':
    font_path = os.path.join(os.getcwd(), 'utils', 'font', 'KoPubWorld Batang Medium.ttf')
    fm.fontManager.addfont(font_path)
    font_name = fm.FontProperties(fname=font_path).get_name()
    plt.rcParams["font.family"] = font_name
    plt.rcParams["axes.unicode_minus"] = False
    main()

'''
해당 폴더에 해당형식으로 label이 존재해야 함.
# 예)
# source/train/labels/K-001900-010224-016551-031705_0_2_0_2_70_000_200.txt
0 0.512 0.348 0.235 0.157   # 클래스 0, 중심(0.512,0.348), 크기(0.235×0.157)
3 0.678 0.441 0.120 0.240   # 클래스 3, 중심(0.678,0.441), 크기(0.120×0.240)
'''
