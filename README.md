# ai3-team5-pill-detection

이미지 기반 알약 인식 프로젝트 (AI 3기 팀 5)

---

## 🧪 프로젝트 개요
- YOLO 또는 Faster R-CNN 기반으로, 이미지 내 알약을 탐지하고 정보를 추론하는 딥러닝 프로젝트입니다.
- 최대 4개의 알약 객체를 검출하여 이름(클래스)과 위치(바운딩 박스)를 반환하는 모델을 목표로 합니다.

### 서비스 목적

- 사용자가 알약 사진을 찍으면, 해당 알약의 **이름(class)과 위치(bbox)**를 정확히 알려주는 서비스 제공
- 추후 증상 기반 알약 추천 시스템으로 확장 가능

### 실험 목표
- **정확한 탐지** : 회전/부분 노출/ 저해상도 등의 조건에서도 강력하게 알약 위치 탐지 및 분류
- **사용자 신뢰 확보** : false positive(잘못된 약 이름) 최소화 → 잘못된 정보 제공 방지
  
---

## 📁 프로젝트 디렉토리 구조
```
ai3-team5-pill-detection/    
├── data/ # 원본 및 전처리 데이터    
│  ├── raw/ # 원본 이미지, 라벨 등    
│  ├── processed/ # 전처리된 데이터    
│  ├── yolo_dataset/     
│  │ └─ train/     
│  │  └─ images/      
│  │  └─ labels/      
│  │ └─ val/     
│  │  └─ images/      
│  │  └─ labels/      
│  └── stats/ # 클래스 분포 등 분석 결과    
│
├── notebooks/ # 실험 및 탐색용 주피터 노트북    
│
├── models/ # 모델 학습 코드    
│
├── outputs/ # 실험 결과    
│  ├── logs/    
│  ├── predictions/    
│  └── submissions/    
│
├── scripts/ # 실행 스크립트    
│
├── utils/ # 공용 유틸 함수    
│  ├── font/  # 공통 서체
│  ├── preprocessing.py    
│  ├── augmentation.py    
│  └── evaluation.py    
│
├── experiments/ # 실험 기록 단위 폴더    
│
├── submission/ # 제출용 결과 파일    
├── config/ # 모델별 설정파일    
├──data.yaml  # train/val/test 경로, class 이름, 개수가 들어있는 파일    
├── env.yml    
└── README.md    
```
---

## Yolo1 실행방법
### 훈련
```
python models/yolo1/main.py --name exp1

usage: main.py [-h] [--epochs EPOCHS] [--batch BATCH] [--imgsz IMGSZ] [--imgchsz IMGCHSZ] [--optimizer OPTIMIZER]
                 [--lr0 LR0] [--lrf LRF] [--cos_lr] [--momentum MOMENTUM] [--betas BETAS]
                 [--train_images_dir TRAIN_IMAGES_DIR] [--train_labels_dir TRAIN_LABELS_DIR]
                 [--val_images_dir VAL_IMAGES_DIR] [--val_labels_dir VAL_LABELS_DIR]
                 [--test_images_dir TEST_IMAGES_DIR] [--device DEVICE] [--weight_decay WEIGHT_DECAY] [--s S] [--b B]
                 [--c C] [--conf_thresh CONF_THRESH] [--nms_iou_thresh NMS_IOU_THRESH] [--project PROJECT] --name NAME
```

### 제출용 csv 파일생성
파라미터는 훈련할 때 이미 넘겼으므로 name 만 넘긴다.
```
python models/yolo1/submit.py --name exp1
```

## Yolo11 실행방법
```
python models/yolo11/main.py -- name exp1

Ultralytics YOLOv11 Training Script

options:
  -h, --help            show this help message and exit
  --weights WEIGHTS     pretrained weights or model .pt path (default: yolo11n.pt)
  --data DATA           dataset yaml file (train/val paths, nc, names) (default: data.yaml)
  --epochs EPOCHS       number of training epochs (default: 50)
  --batch BATCH         batch size (default: 16)
  --imgsz IMGSZ         input image size (HxW) (default: 640)
  --optimizer {SGD,Adam,AdamW}
                        optimizer type (default: SGD)
  --lr0 LR0             initial learning rate (default: 0.01)
  --lrf LRF             final LR ratio (cosine scheduler) (default: 0.01)
  --momentum MOMENTUM   SGD momentum (only if optimizer=SGD) (default: 0.937)
  --betas BETAS         Adam/AdamW betas as "beta1,beta2" (default: (0.9, 0.999))
  --weight_decay WEIGHT_DECAY
                        weight decay (L2) (default: 0.0005)
  --warmup_epochs WARMUP_EPOCHS
                        number of warmup epochs (default: 3)
  --augment             enable augmentations (default: True)
  --no-augment          disable augmentations (default: True)
  --project PROJECT     save results to project/name (default: runs/train)
  --name NAME           experiment name (subfolder) (default: exp)
usage: yolo11.py [-h] [--weights WEIGHTS] [--data DATA] [--epochs EPOCHS] [--batch BATCH] [--imgsz IMGSZ] [--optimizer {SGD,Adam,AdamW}] [--lr0 LR0] [--lrf LRF] [--momentum MOMENTUM] [--betas BETAS] [--weight_decay WEIGHT_DECAY] [--warmup_epochs WARMUP_EPOCHS] [--augment | --no-augment] [--project PROJECT] --name NAME
```

## Yolo11 라벨 형식
```
해당 폴더에 해당형식으로 label이 존재해야 함.
# 예)
# source/train/labels/K-001900-010224-016551-031705_0_2_0_2_70_000_200.txt
0 0.512 0.348 0.235 0.157   # 클래스 0, 중심(0.512,0.348), 크기(0.235×0.157)
3 0.678 0.441 0.120 0.240   # 클래스 3, 중심(0.678,0.441), 크기(0.120×0.240)
```

## ⚙️ Conda 환경 설정
### How to export conda environment

```
# Do not install modules with pip which makes unexpected dependencies
conda activate venv
conda install jupyter
conda install ipykernel
conda install torch
conda install matplotlib
conda env export --from-history --no-builds | findstr /V "^prefix:" > env.yml
```

### How to create conda environment by running env.yml file
```
conda env create -f env.yml -n venv
conda env list
conda activate venv
```

### Update conda environment after creation
```
conda env update --file env.yml --name venv --prune
```
