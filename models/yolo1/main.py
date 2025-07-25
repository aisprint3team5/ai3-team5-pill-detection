import os
import time
import torch
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from torch.optim import Optimizer
from torch.nn import Module
from yolo1 import Yolo1
from utils import Utils
from config import Config, parse_args
from yolo1_loss import Yolo1Loss
from optimizer import build_optimizer
from scheduler import build_scheduler
from trainval import train_one_epoch, validate
# from datasetloadervoc import load_loaders
from datasetloader import load_loaders
from csvwriter import add_csv_log
from sample_inference import run_sample_inference


def main():
    start_time = time.time()
    config_file: str = 'config/yolo_1.yaml'
    # 1) 인자 파싱
    args = parse_args(config_file)
    # 2) Config 로드 + 덮어쓰기
    Config.load(config_file, vars(args))
    # 3) 인자 저장
    Utils.save_args()

    model: Module = Yolo1(Config.IMAGE_CH_SIZE, Config.S, Config.B, Config.C,
                          Config.CONF_THRESH, Config.NMS_IOU_THRESH).to(Config.DEVICE)
    save_model = Utils.load_model(model, 'model')
    elapsed = time.time() - start_time
    print(f'Model is defined! ({elapsed:.1f}s)')
    loss_fn = Yolo1Loss(Config.S, Config.B, Config.C)
    opt: Optimizer = build_optimizer(model)
    scheduler = build_scheduler(opt)
    train_loader, val_loader = load_loaders()
    elapsed = time.time() - start_time
    print(f'Data loader is defined! ({elapsed:.1f}s)')

    best_val_loss = float('inf')
    for epoch in range(1, Config.EPOCHS + 1):
        epoch_start = time.time()

        # 1) Train
        train_loss = train_one_epoch(model, train_loader, loss_fn, opt, Config.DEVICE)

        # 2) Step Scheduler (epoch 단위)
        scheduler.step()

        # 3) Validate
        val_loss, metrics = validate(model, val_loader, loss_fn, Config.DEVICE)

        epoch_time = time.time() - epoch_start

        # 4) Logging
        lrs = scheduler.get_last_lr()
        print(f"Epoch {epoch:02d} | LR {lrs[0]:.6f} | Train Loss: {train_loss['total_loss']:.4f} | Val Loss: {val_loss['total_loss']:.4f}")

        csv_path = os.path.join(Config.PROJECT, Config.NAME, 'results.csv')
        add_csv_log(csv_path, epoch, epoch_time, train_loss, metrics, val_loss, lrs)

        save_model()

        # 5) Checkpoint
        if val_loss['total_loss'] < best_val_loss:
            path: str = os.path.join(Config.PROJECT, Config.NAME, 'best_model.pth')
            best_val_loss = val_loss['total_loss']
            torch.save(model.state_dict(), path)
            print(f' New best model saved (val_loss={best_val_loss:.4f})')

    elapsed = time.time() - start_time
    print(f'Train is completed! ({elapsed:.1f}s)')

    run_sample_inference(model, val_loader)


if __name__ == '__main__':
    font_path = os.path.join(os.getcwd(), 'utils', 'font', 'KoPubWorld Batang Medium.ttf')
    fm.fontManager.addfont(font_path)
    font_name = fm.FontProperties(fname=font_path).get_name()
    plt.rcParams["font.family"] = font_name
    plt.rcParams["axes.unicode_minus"] = False
    main()
