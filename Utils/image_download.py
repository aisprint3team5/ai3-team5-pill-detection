import os
import zipfile

# 데이터셋 다운로드
os.system("kaggle competitions download -c ai03-level1-project -p ./data")

zip_path = "./data/ai03-level1-project.zip"
if os.path.exists(zip_path):
  with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall("./data/raw")
  print("압축 해제 완료")
else:
  print("압축 파일 없음")