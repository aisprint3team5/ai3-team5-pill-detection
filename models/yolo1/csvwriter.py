import os
import csv


FIELDNAMES = [
    'epoch', 'time',
    'train/box_loss', 'train/cls_loss', 'train/dfl_loss',
    'metrics/precision(B)', 'metrics/recall(B)',
    'metrics/mAP50(B)', 'metrics/mAP50-95(B)',
    'val/box_loss', 'val/cls_loss', 'val/dfl_loss',
    'lr/pg0', 'lr/pg1', 'lr/pg2'
]


def get_info(epoch, epoch_time, train_loss, metrics, val_loss, lrs):
    return {
        'epoch': epoch,
        'time': f'{epoch_time:.4f}',

        # train losses
        'train/box_loss': train_loss['box_loss'],
        'train/cls_loss': train_loss['cls_loss'],
        'train/dfl_loss': train_loss['dfl_loss'],

        # val metrics
        'metrics/precision(B)': metrics['precision'],
        'metrics/recall(B)':    metrics['recall'],
        'metrics/mAP50(B)':     metrics['mAP50'],
        'metrics/mAP50-95(B)':  metrics['mAP5095'],

        # val losses
        'val/box_loss': val_loss['box_loss'],
        'val/cls_loss': val_loss['cls_loss'],
        'val/dfl_loss': val_loss['dfl_loss'],

        # learning rates
        'lr/pg0': lrs[0] if len(lrs) > 0 else None,
        'lr/pg1': lrs[1] if len(lrs) > 1 else None,
        'lr/pg2': lrs[2] if len(lrs) > 2 else None,
    }


def add_csv_log(csv_path, epoch, epoch_time, train_loss, metrics, val_loss, lrs):
    # 1) 폴더가 없으면 생성
    dir_path = os.path.dirname(csv_path)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

    # 2) 한 행 생성
    row = get_info(epoch, epoch_time, train_loss, metrics, val_loss, lrs)

    # 3) 파일 신규 생성 여부 체크
    is_new_file = not os.path.isfile(csv_path)

    # 4) append 모드로 열고, FIELDNAMES 순서대로 기록
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if is_new_file:
            writer.writeheader()
        writer.writerow(row)
