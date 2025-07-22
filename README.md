# ai3-team5-pill-detection

이미지 기반 알약 인식 프로젝트 (AI 3기 팀 5)

---

## 🧪 프로젝트 개요
- YOLO 또는 Faster R-CNN 기반으로, 이미지 내 알약을 탐지하고 정보를 추론하는 딥러닝 프로젝트입니다.
- 최대 4개의 알약 객체를 검출하여 이름(클래스)과 위치(바운딩 박스)를 반환하는 모델을 목표로 합니다.

---

## 📁 프로젝트 디렉토리 구조

ai3-team5-pill-detection/    
├── data/                 # 원본 및 전처리 데이터    
│ ├── raw/                # 원본 이미지, 라벨 등    
│ ├── processed/          # 전처리된 데이터    
│ └── stats/              # 클래스 분포 등 분석 결과    
│    
├── notebooks/            # 실험 및 탐색용 주피터 노트북    
│    
├── models/               # 모델 학습 코드    
│    
├── outputs/              # 실험 결과    
│ ├── logs/    
│ ├── predictions/    
│ └── submissions/    
│    
├── scripts/              # 실행 스크립트    
│    
├── utils/                # 공용 유틸 함수    
│ ├── preprocessing.py    
│ ├── augmentation.py    
│ └── evaluation.py    
│    
├── experiments/          # 실험 기록 단위 폴더    
│    
├── submission/           # 제출용 결과 파일    
├── README.md    
└── env.yml    

---

## Yolo11 실행방법
```
cd yolo11.py
python yolo11.py -- name exp1
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
