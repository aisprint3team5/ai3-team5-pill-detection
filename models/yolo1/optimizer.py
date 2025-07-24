import torch.nn as nn
import torch.optim as optim
from config import Config


def build_optimizer(model: nn.Module) -> optim.Optimizer:

    # Yolo1Config의 OPTIMIZER 설정에 따라 SGD / Adam / AdamW 중 하나를 생성해서 반환합니다.
    opt_name = Config.OPTIMIZER.lower()
    params = model.parameters()
    lr = Config.LR
    wd = Config.WD

    if opt_name == 'sgd':
        return optim.SGD(params, lr=lr, momentum=Config.MOMENTUM, weight_decay=wd)
    elif opt_name == 'adam':
        return optim.Adam(params, lr=lr, betas=tuple(Config.BETAS), weight_decay=wd)
    elif opt_name == 'adamw':
        return optim.AdamW(params, lr=lr, betas=tuple(Config.BETAS), weight_decay=wd)
    else:
        raise ValueError(f'Unsupported optimizer: {Config.OPTIMIZER}')
