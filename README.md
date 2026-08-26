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
├── src/coop_pipeline/    # 핵심 로직
│   ├── runner.py         #   ★ 진입점: 시나리오 넣으면 L0~L4가 나온다
│   ├── agents.py         #   에이전트 실행 (gemini / llama)
│   ├── tagging.py        #   발화 태깅 (claude judge)
│   ├── scoring.py        #   산출물 채점
│   ├── features.py       #   정량 지표 추출
│   ├── classify.py       #   L0~L4 결정 트리
│   ├── validate_log.py   #   로그 형식 검증
│   ├── thresholds.py     #   임계값 로더 (configs/를 읽는다)
│   └── llm.py            #   모델 ID / 재시도 / JSON 파싱 공통 래퍼
├── docs/                 # 실행 가이드와 채점 기준 ← 먼저 읽을 것
├── scenarios/            # 시나리오 지문
├── configs/              # threshold 등 설정값
├── tests/                # 자동 테스트 (API 키 없이 돌아간다)
├── data/                 # 실제 로그는 Drive에 저장 (git에 올리지 않음)
├── requirements.txt      # 라이브러리 목록
└── README.md
```

## 📖 문서

| 문서 | 언제 읽는가 |
|---|---|
| [docs/PIPELINE.md](docs/PIPELINE.md) | **실행 방법.** 처음이면 여기부터 |
| [docs/RUBRIC.md](docs/RUBRIC.md) | 채점 기준, 층위 정의, 임계값 캘리브레이션 절차 |
| [docs/CODING_MANUAL.md](docs/CODING_MANUAL.md) | 발화 태깅 기준 (인간 코더용 + LLM 프롬프트 원본) |
| [docs/MODEL_ASSIGNMENT.md](docs/MODEL_ASSIGNMENT.md) | alpha/beta/judge를 어떻게 골랐나. 모델이 퇴역해 다시 배정할 때 |
| [scenarios/README.md](scenarios/README.md) | 시나리오 작성법 |

## ⚡ 가장 짧은 실행

```python
import sys; sys.path.append('src')
from coop_pipeline.runner import check_scenario_dir, run_scenario

check_scenario_dir("scenarios")          # 형식 점검 (API 호출 없음)

result = run_scenario(
    "scenarios/A1/A1_simple_meeting.json",
    condition="명시",
    api_keys={"gemini": ..., "groq": ..., "anthropic": ...},
    out_dir=DATA_DIR,
)
print(result["level"])                   # "L0" ~ "L4"
```

자세한 내용과 결과 읽는 법은 [docs/PIPELINE.md](docs/PIPELINE.md) 참고.

## 🔗 링크

- 팀 노션: https://app.notion.com/p/39eba2af2b00804da927ea9edacdb3ed?source=copy_link
- Github: https://github.com/ewha-oi/OrganicAI.git
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
git clone https://github.com/ewha-oi/OrganicAI.git
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

!git clone https://github.com/ewha-oi/OrganicAI.git
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
- 파일명에 사람 구분이 없다. **rep 번호를 사람마다 나눠서 돌린다** — 같은 번호를 두 사람이 돌리면 한쪽 데이터가 조용히 덮어써진다

| 사람 | rep |
|---|---|
| 김유리 | 1~5 |
| 이예영 | 6~10 |
| 김유민 | 11~15 |

- 수집 파라미터(모델 · `max_turns` · `n_solo` · 임계값)는 팀 노션의 **동결표**에 고정한다. 수집 도중에 바꾸면 앞뒤 데이터를 함께 분석할 수 없다

## 🌿 브랜치 규칙

- 평소에는 `main`에서 바로 작업
- 커밋 메시지:
```
Type: 내용
```

| Type | 용도 |
|---|---|
| Add | 파일, 기능, 문서 등 추가 |
| Fix | 버그 수정 |
| Chore | 설정, 문서, 리팩토링, 의존성, 폴더 정리 |

예시:
```
Add: 프로젝트 구조 세팅
```

- 판정 로직 / threshold처럼 큰 변경사항은 dev 브랜치 생성 → 테스트 → merge