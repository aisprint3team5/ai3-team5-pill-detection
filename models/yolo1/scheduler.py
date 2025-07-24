from typing import Callable
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from config import Config


def build_scheduler(optimizer: Optimizer):
    # 2가지 스케줄러 분기 추가됨 ↓
    if Config.COS_LR:
        # 코사인 애니일링 스케줄러
        return CosineAnnealingLR(optimizer, T_max=Config.EPOCHS, eta_min=Config.LR * Config.LRF)
    else:
        # 선형(linear) 디케이 스케줄러
        lr_lambda: Callable[[int], float] = lambda e: 1 - (1 - Config.LRF) * (e / float(Config.EPOCHS))
        return LambdaLR(optimizer, lr_lambda=lr_lambda)
