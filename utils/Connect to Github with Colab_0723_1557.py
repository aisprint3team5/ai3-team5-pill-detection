
"""# 깃헙과 연결하기"""

# 1) 환경 변수로 토큰 설정 (안전)
import os
os.environ['GITHUB_TOKEN'] = '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'

# 2) 토큰을 이용해 브랜치 전체 clone
!git clone https://$GITHUB_TOKEN@github.com/aisprint3team5/ai3-team5-pill-detection.git

!git branch -a




###연결된 원격 브랜치 확인
 %cd /content/ai3-team5-pill-detection
!git fetch
!git branch -r
print("==========")
# 또는
!git ls-remote --heads origin


#%cd /content/ai3-team5-pill-detection

#현재 브랜치 확인하기
!git branch


branch = "el/ETL"   # ← change this

##특정 브랜치에 체크아웃(연결)&최신 커밋받아오기
!git fetch origin {branch}
!git checkout {branch}
!git pull --rebase origin {branch}

!git checkout {branch}


# 1) 리포지토리 루트로 이동
# %cd /content/ai3-team5-pill-detection

# 2) Git 사용자 정보 설정 (필요하다면)
!git config user.email "=@.com"
!git config user.name  "="

# 3) 다른 브랜치에 체크아웃
!git checkout el/ETL

# 4) 변경된 파일 스테이징
!git add experiments/ # 또는 원하는 경로 전체

# 5) 커밋
!git commit -m "Your commit message here""

# 6) 원격에 브랜치 푸시 (upstream 설정)
!git push origin el/ETL
