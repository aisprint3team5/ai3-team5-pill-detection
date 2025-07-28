hyperparameters = {
    "optimizer": "SGD",               
    "lr0": 0.01,                      
    "lrf": 0.01,                      
    "momentum": 0.937,            
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,

    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "translate": 0.1,
    "scale": 0.5,
    "fliplr": 0.5,
    "degrees": 0.0,

    "box": 0.05,
    "cls": 0.5,
}


# [ SGD   ]:	Stochastic Gradient Descent: 일반적으로 더 안정적인 수렴.
# [ Adam  ]:	adaptive한 학습률을 가진 옵티마이저. 빠른 수렴 가능.
# [ AdamW ]:	Adam의 가중치 감쇠(weight decay) 버전. 과적합 방지에 유리.

# 옵티마이저
# optimizer:      "SGD"        # ["SGD","Adam","AdamW"]
# lr0:            0.01         # 초기 학습률
# lrf:            0.01         # 최종 학습률 = lr0 * lrf
# momentum:       0.937        # SGD 계열
# betas:          [0.9,0.999]  # Adam/AdamW 전용
# weight_decay:   0.0005       # 가중치 감쇠
# warmup_epochs:  3.0          # 워밍업 에폭

# 증강
# hsv_h:          0.015         # 이미지의 색조(hue)를 무작위로 변화시키는 정도
# hsv_s:          0.7           # 이미지의 채도(saturation)를 무작위로 변화시키는 정도
# hsv_v:          0.4           # 이미지의 명도(value)를 무작위로 변화시키는 정도
# translate:      0.1           # 이미지의 좌우 또는 상하로 무작위 이동하는 범위
# scale:          0.5           # 이미지 크기를 확대/축소하는 범위
# fliplr:         0.5           # 좌우 반전 확률
# degrees:        0.0           # 회전 각도

# Loss 가중치
# box:            0.05          # 박스 회귀 손실 가중치
# cls:            0.5           # 클래스 손실 가중치