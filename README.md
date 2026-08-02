# 🌱 OrganicAI
### An AI Project for Organic Intelligence

## 👥 역할

| 역할 | 이름 |
|---|---|
| 팀장 | 이예영 |
| 팀원 | 김유민, 김유리 |

## 🛠 사용 언어 및 툴

| 항목 | 내용 |
|---|---|
| 실행 환경 | Google Colab |
| 언어 | Python 3.x |
| 데이터 관리 | Google Drive |
| 코드 수정 | VS Code |
| 코드 관리 | Github |
| (선택) GUI | Gitkraken |

## 📌 레포지토리 목적

- 에이전트 단독/2-agent 실행 → 로그 수집
- 대화 로그 자동 태깅
- 태깅 결과를 판정 프레임에 넣어 협력 층위(L0~L4) 자동 판정
- 판정 결과를 인간과 대조하여 신뢰도(Cohen's Kappa) 검증

## 📁 폴더 구조

```
OrganicAI/
├── src/coop_pipeline/    # 핵심 로직 (태깅, 판정, 검증)
├── notebooks/            # Colab 실행용 노트북
├── scenarios/            # 시나리오 지문
├── configs/              # threshold 등 설정값
├── tests/                # 자동 테스트
├── data/                 # 실제 로그는 Drive에 저장
├── requirements.txt      # 라이브러리 목록
└── README.md
```

## 🔗 링크

- 팀 노션: https://app.notion.com/p/39eba2af2b00804da927ea9edacdb3ed?source=copy_link
- Github: https://github.com/uri-git23/OrganicAI.git
- Google Drive(데이터용): `유기농지능 > data`
  - 점 세 개 클릭 → 정리 → 바로가기 추가 → "내 드라이브"에 등록

## ⚠️ 참고사항

- VS Code로 코드 수정하고 싶다 → 세팅 방법 (1)
- Colab에서 실행만 하고 싶다 → 세팅 방법 (2)
- 데이터(로그 파일)는 Git에 직접 올리지 않기 (Google Drive 공유 폴더 사용)
- 코드 작업 전에 `git pull` 하고 시작하기
- API Key는 코드에 직접 쓰지 않기 (Colab Secrets 사용)

## 🚀 세팅 방법

### 1. VS Code로 코드 수정

클론:

```
git clone https://github.com/uri-git23/OrganicAI.git
cd OrganicAI
python -m venv venv
```

가상환경 활성화:

```
# macOS/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

라이브러리 설치:

```
pip install -r requirements.txt
```

VS Code에서:
- OrganicAI 폴더 열기
- Python: Select Interpreter → `venv` 선택

이후 작업 순서:
Pull → 수정 → Stage → Commit → Push

### 2. Colab에서 실행만

Google Colab 새 노트북 생성 후 첫 번째 셀 실행:

```python
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/uri-git23/OrganicAI.git
%cd OrganicAI
!pip install -r requirements.txt -q

import sys
sys.path.append('src')

DATA_DIR = "/content/drive/MyDrive/data"
```

Google Drive 연결:
Google Drive에 연결 → 계정 선택 → 허용

API Key는 Colab Secrets에 등록:
GEMINI_API_KEY
GROQ_API_KEY
ANTHROPIC_API_KEY

이후에는 `git clone` 대신 `git pull` 사용

## 💾 데이터 저장 규칙

- 로그는 Git 대신 Drive 공유 폴더(`유기농지능 > data`)에 저장
- 파일명: {scenario_id}_{condition}_{rep번호}.json

## 🌿 브랜치 규칙

- 평소에는 `main`에서 바로 작업
- 커밋 메시지:
```
Type: 내용
```

| Add | 파일, 기능, 문서 등 추가 |
| Fix | 버그 수정 |
| Chore | 설정, 문서, 리팩토링, 의존성, 폴더 정리 |

예시:
```
Add: 프로젝트 구조 세팅
```

- 판정 로직 / threshold처럼 큰 변경사항은 dev 브랜치 생성 → 테스트 → merge