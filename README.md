# OrganicAI
An AI Project for Organic Intelligence


## 역할
- 팀장 : 이예영
- 팀원 : 김유민, 김유리

## 사용 언어 및 툴
- 실행 환경 : Google Colab
- 언어 : Python 3.x
- 데이터(로그 파일) 관리 : Google Drive
- 코드 수정 : VS Code
- 코드 원본 관리 : Github
- (선택) GUI : Gitkraken

## 레포지토리 목적
- 에이전트 단독/2-agent 실행 -> 로그 수집
- 대화 로그에 자동 태깅
- 태깅 결과를 판정 프레임에 넣어 협력 층위(L0~L4) 자동 판정
- 판정 결과를 인간과 대조해 신뢰도(이하 Kappa) 검증

## 참고사항
- VS Code로 코드 수정하고 싶다 : 세팅 방법 (1)
- Colab에서 실행만 하고 싶다 : 세팅 방법 (2)
- 데이터(로그 파일)는 git에 직접 올리지 않기 (Google Drive 공유 폴더 사용)
- 코드 작업 전에 git pull 하고 시작하기
- API 키는 코드에 직접 쓰지 않기 (Colab Secrets 사용)

## 링크
- 팀 노션 : https://app.notion.com/p/39eba2af2b00804da927ea9edacdb3ed?source=copy_link
- Github : https://github.com/uri-git23/OrganicAI.git
- 구글 드라이브(데이터용) : 유기농지능 > data (점 세 개짜리 클릭 > 정리 > 바로가기 추가 해서 '내 드라이브'에 등록해두기)

## 폴더 구조
- OrganicAI/
    - src/coop_pipeline/    # 핵심 로직 (태깅, 판정, 검증)
    - notebooks/            # Colab 실행용 노트북
    - scenarios/            # 시나리오 지문
    - configs/              # threshold 등 설정값
    - tests/                # 자동 테스트
    - data/                 # 실제 로그는 Drive에 저장
    - requirements.txt      # 라이브러리 목록
    - README.md             

## 첫 세팅
- VS Code 설치
- Git 설치
- Python 3.x로 버전 맞추기

## 세팅 방법
1. VS Code로 코드 수정

    git clone https://github.com/uri-git23/OrganicAI.git
    cd OrganicAI
    python -m venv venv

    # 가상환경 활성화
    # macOS/Linux
    source venv/bin/activate
    # Windows
    venv\Scripts\activate

    pip install -r requirements.txt

VS Code에서 OrganicAI 폴더를 클론해서 열고, Python: Select Interpreter에서 venv를 선택하기

이후 작업 순서:
Pull → 수정 → Stage → Commit → Push


2. Colab에서 실행만
구글 Colab 접속 후 새 노트북 생성하고 첫 번째 셀에 아래 코드 붙여넣기

    from google.colab import drive
    drive.mount('/content/drive')

    !git clone https://github.com/uri-git23/OrganicAI.git
    %cd OrganicAI
    !pip install -r requirements.txt -q

    import sys
    sys.path.append('src')

    DATA_DIR = "/content/drive/MyDrive/data"

셀 재생 누르고 Google 계정 접근 권한 묻는 창이 뜨면 'Google Drive에 연결' 클릭 → 본인 계정 선택 → 허용 클릭하기. API 키는 왼쪽 열쇠 아이콘(Secrets)에 GEMINI_API_KEY, GROQ_API_KEY, ANTHROPIC_API_KEY로 등록. 다음부턴 git clone 대신 git pull만 실행.

## 데이터 저장 규칙
- 로그는 git 대신 Drive 공유 폴더(유기농지능 > data)에 저장
- 파일명 : {scenario_id}_{condition}_{rep번호}.json

## 브랜치 규칙
- 평소에는 main 에 바로 작업
- 커밋 메시지 : Add(파일, 기능, 문서 등 추가) / Fix(버그 수정) / Chore(설정, 문서, 리팩토링, 의존성, 폴더 정리 등) 3개 중 하나 쓰고 뭐 했는지 쓰기 (e.g. Add: 프로젝트 구조 세팅)
- 판정 로직 / threshold 처럼 큰 변경사항인 경우에만 dev 브랜치로 분리 후 테스트해보고 merge