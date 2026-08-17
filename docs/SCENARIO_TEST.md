# 새 시나리오 테스트 가이드

시나리오를 새로 쓴 뒤, 그 시나리오가 파이프라인에서 실제로 끝까지 도는지 확인하는 절차다.

**두 단계로 나뉜다.**

| 단계 | 어디서 | 비용 | 시간 |
|---|---|---|---|
| 1. 형식 검사 | 내 컴퓨터 | 무료 | 30초 |
| 2. 실기동 | Google Colab | API 호출 | 10분 내외 |

1단계에서 걸리는 문제는 2단계에서도 똑같이 걸린다. 다만 2단계는 API를 쓰므로,
**반드시 1단계를 먼저 통과시킨 뒤 커밋·푸시할 것.**

---

## 0. 준비 (최초 1회만)

### 0-1. Groq API 키 발급 — 무료

1. https://console.groq.com 접속 → Google 계정으로 로그인
2. 좌측 메뉴 **API Keys** → **Create API Key** → 이름은 아무거나 (예: `organicai`)
3. 생성되면 `gsk_...`로 시작하는 문자열이 뜬다. **그 화면에서 바로 복사할 것.**
   창을 닫으면 다시 볼 수 없고, 새로 발급받아야 한다.

### 0-2. Colab에 키 등록

1. Colab 노트북을 연다
2. 좌측 사이드바에서 **🔑 열쇠 아이콘(보안 비밀)** 클릭
3. **+ 새 보안 비밀 추가**
4. 이름에 `GROQ_API_KEY`, 값에 아까 복사한 `gsk_...` 붙여넣기
5. **왼쪽의 "노트북 액세스" 토글을 켠다** ← 이걸 빼먹는 실수가 가장 흔하다

> **`GEMINI_API_KEY`는 지금 필요 없다.**
> Google이 Gemini API 접근을 차단해서(전 모델 403/404) 생성 2개와 채점 1개가
> 모두 Groq로 돌아간다. 실행 시 `gemini !! 없음`이 떠도 정상이다.

### 0-3. 저장소 받기 (내 컴퓨터)

```bash
git clone -b yr https://github.com/ewha-oi/OrganicAI.git
cd OrganicAI
pip install -r requirements.txt
```

---

## 1. 시나리오 작성

> **작성 규칙과 설계 원칙은 [`scenarios/README.md`](../scenarios/README.md)에 있다. 먼저 그것을 읽을 것.**
> 이 문서는 다 쓴 뒤 **어떻게 검사하고 실행하는지**만 다룬다.

파일 위치: `scenarios/A1/` 또는 `scenarios/A2/` 또는 `scenarios/A4/`
파일명 형식: `{과제유형}_{복잡도}_{시나리오명}.json` (예: `A4_simple_festival.json`)

아래 표는 **검사에서 걸리는 최소 요건**만 추린 것이다. 설계 규칙은 위 문서를 볼 것.

| 항목 | 최소 요건 |
|---|---|
| 파일명 | `scenario_id`와 **정확히 일치** (`A4_simple_club.json` → `"scenario_id": "A4_simple_club"`) |
| `task_type` | `A1` / `A2` / `A4` 중 하나 |
| `complexity` | `simple` / `complex` |
| `task_variants.solo` | **항상 필수** |
| `task_variants` 나머지 | **A1·A4** → `alpha` + `beta` (비대칭) / **A2** → `shared` |
| `scoring.checklist` | **A1만 필수.** 없으면 자동 채점 불가 |

과제 유형에 따라 채점 방식과 판정 기준이 자동으로 갈린다:

- **A1** — 체크리스트 기반 결정론 채점 (LLM 호출 없음). 판정은 단독 분포의 90퍼센타일과 비교
- **A2 / A4** — LLM 채점자가 1~5점 등급. 판정은 단독 중앙값과의 등급차로 비교
  (A2는 의견 조율, A4는 아이디어 생성 — 척도가 서로 다르다)

> ⚠️ 형식 검사(`check_scenario_dir`)는 `shared`와 `alpha`+`beta`를 **둘 다 통과시킨다.**
> A4를 `shared`로 써도 에러는 안 나지만 설계 규칙 위반이다. 검사에 의존하지 말고
> `scenarios/README.md`의 작성 후 체크리스트를 직접 확인할 것.

---

## 2. 내 컴퓨터에서 형식 검사 — 30초, 무료

```bash
cd OrganicAI
git pull
python -c "import sys; sys.path.append('src'); from coop_pipeline.runner import check_scenario_dir; check_scenario_dir('scenarios')"
```

출력 예시:

```
OK   scenarios/A2/A2_simple_club.json
FAIL scenarios/A2/A2_simple_cafe.json
     A2_simple_cafe.json 형식 오류:
       - task_variants.solo 없음 (단독 조건을 실행할 수 없음)

결과: 7개 통과 / 1개 오류
```

**내 파일이 `OK`로 뜰 때까지 고친다.** 그 뒤에 커밋·푸시한다.

---

## 3. Colab에서 실기동 — 10분 내외

### 3-1. 노트북 셀을 위에서부터 순서대로 실행

**순서가 중요하다.** 특히 `import os` 셀은 파이프라인을 임포트하는 어떤 셀보다 먼저
실행되어야 한다. 채점자 설정을 임포트 시점에 한 번만 읽기 때문이다.

| 순서 | 셀 (첫 줄) | 하는 일 |
|---|---|---|
| 1 | `!git clone -b yr https://...` | 최초 1회만. 이미 받았다면 건너뛰고 맨 아래 `!git pull` 셀 사용 |
| 2 | `# 재시작 후 여기서 시작` | 작업 폴더 이동 + 모듈 경로 설정 |
| 3 | `from google.colab import userdata` | 키 로드. **`groq   OK`** 가 뜨는지 확인 |
| 4 | `import os` | 채점자를 Groq로 지정. ⚠️ **아래 셀들보다 반드시 먼저** |
| 5 | `!python -m pytest tests/ -q` | `17 passed` 확인 |
| 6 | `import re, types` | alpha를 Groq로 임시 대체 (Gemini 차단 대응) |

### 3-2. 내 시나리오를 돌릴 새 셀 만들기

**위치:** 6번 셀(`import re, types`)보다 **아래**면 어디든 된다.
`import re, types` 셀보다 위에 두면 Gemini를 호출해서 죽는다.

**만드는 법:**

1. `import re, types` 로 시작하는 셀을 **클릭**한다
2. 셀 위쪽에 뜨는 **`+ 코드`** 버튼을 누른다 → 바로 아래에 빈 셀이 생긴다
3. 아래 코드를 붙여넣고 **경로만** 본인 파일로 바꾼다

```python
from coop_pipeline.runner import run_scenario

res = run_scenario(
    "scenarios/A2/A2_simple_club.json",   # ← 본인 파일 경로로 바꿀 것
    condition="명시",
    api_keys=API_KEYS,
    n_solo=2,        # 시험용. 정식 실행은 5
    max_turns=6,     # 시험용. 정식 실행은 10
    out_dir="runs_pilot",
)
```

4. `Shift + Enter` 로 실행

### 3-3. 통과 기준

**에러 없이 `L0` ~ `L4` 중 하나가 나오면 통과다. 그게 전부다.**

출력 예시:

```
==============================================================
시나리오 : A2_simple_club (A2)
조건     : 명시 / rep 1
판정     : L2 — 상호 참조 — 주고받았으나 단독 조건보다 나은 산출물을 내지 못함
이유     : 쌍방향 참조는 있으나 집단 우위 미충족 (등급차=+0.5 (기준 2))
--------------------------------------------------------------
[PASS] Q1_참조존재
[PASS] Q2_양방향성
[FAIL] Q3_집단우위

병목: Q3 에서 판정이 멈춤
==============================================================
```

> ### ⚠️ 어느 층위가 나오든 상관없다
>
> `L0`이나 `L2`가 나와도 **시나리오 실패가 아니다.** 그건 실험 결과이지 형식 오류가 아니다.
> `[FAIL] Q3` 도 마찬가지다 — 판정이 어디서 멈췄는지 알려주는 정상 출력이다.
>
> 이 단계에서 확인하는 것은 오직 하나: **파이프라인이 이 시나리오를 끝까지 처리하는가.**

---

## 4. 에러 대처

| 메시지 | 원인 / 조치 |
|---|---|
| `형식 오류: ...` | 2단계를 건너뜀. 내 컴퓨터에서 먼저 확인할 것 |
| `task_variants.solo 없음` | `solo` 지문 누락 |
| `A1인데 scoring.checklist가 없음` | A1 채점 항목 누락 |
| `scenario_id가 파일명과 불일치` | 둘 중 하나를 고쳐 맞출 것 |
| `judge 제공사가 'groq'인데 api_keys['groq']가 비어 있음` | Colab 보안 비밀의 **노트북 액세스 토글**을 켤 것 (0-2의 5번) |
| `NameError: API_KEYS` | `from google.colab import userdata` 셀을 실행하지 않음 |
| `413 ... Request too large` | 분당 토큰 한도 초과. `max_turns`를 5로 낮출 것 |
| `LLMCallError ... 3회 모두 실패` | 분당 호출 한도. 1~2분 기다렸다 재실행 |
| `설정 경고: max_turns=...` | `max_turns`를 5 이상으로. 그 아래면 판정이 내용과 무관하게 L0/L1이 된다 |
| 코드를 고쳤는데 옛날 대로 동작 | `!git pull` 후 **런타임 → 세션 다시 시작** 필수 |

이 표에 없는 에러는 로그 전문과 함께 공유할 것.

---

## 5. 참고

- 저장된 로그는 `runs_pilot/` 에 쌓인다. `runs_pilot/raw/` 는 태깅 전 원본 체크포인트다
- 여러 로그를 한 번에 다시 판정하려면 (API 호출 없음, 무료):

  ```python
  from coop_pipeline.runner import classify_saved_dir
  classify_saved_dir("runs_pilot")
  ```

- 판정 기준값의 정의와 근거는 `docs/RUBRIC.md`, 파이프라인 단계별 설명은 `docs/PIPELINE.md` 참고
