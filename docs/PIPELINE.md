# 파이프라인 실행 가이드

시나리오 JSON 하나를 넣으면 협력 층위(L0~L4) 판정 결과가 나온다.
이 문서만 읽고 그대로 따라 하면 실행된다.

- **처음 돌려보는 거라면 → [DRYRUN.md](DRYRUN.md)** (무비용 점검 → 소규모 실행 순서)
- 채점 기준이 궁금하면 → [RUBRIC.md](RUBRIC.md)
- 발화 태깅 기준이 궁금하면 → [CODING_MANUAL.md](CODING_MANUAL.md)
- 시나리오 작성법이 궁금하면 → [../scenarios/README.md](../scenarios/README.md)

---

## 0. 파이프라인이 하는 일

```
scenarios/A1_simple_meeting.json
        │
        ├─ (1) 단독 조건 실행      gemini x5, llama x5     → solo 산출물 10개
        ├─ (2) 2-agent 대화 실행   alpha=gemini, beta=llama → turns + final_output
        ├─ (3) 발화 태깅           claude judge            → 각 turn에 codes, ref
        ├─ (4) 채점                A1=체크리스트 / A2·A4=judge 등급
        ├─ (5) 점수 부착           solo_scores, group_score, new_idea_flag
        ├─ (6) 형식 검증           validate_log()
        └─ (7) 판정                classify()  →  L0 ~ L4
```

각 단계는 앞 단계의 출력을 그대로 받는다. 중간 단계를 건너뛰면 (6)에서 에러가 난다.
**에러가 나는 것이 정상이다** — 태깅을 건너뛴 로그는 조용히 L0으로 판정되면 안 되기 때문이다.

---

## 1. Colab 세팅

새 노트북 첫 셀:

```python
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/uri-git23/OrganicAI.git
%cd OrganicAI
!pip install -r requirements.txt -q

import sys
sys.path.append('src')

DATA_DIR = "/content/drive/MyDrive/data"     # 로그 저장 위치 (Git 아님)
```

> 두 번째 실행부터는 `git clone` 대신 `%cd OrganicAI` 후 `!git pull`.

API 키는 Colab **Secrets**(왼쪽 열쇠 아이콘)에 등록한다. 코드에 직접 쓰지 않는다.

```python
from google.colab import userdata

API_KEYS = {
    "gemini":    userdata.get('GEMINI_API_KEY'),
    "groq":      userdata.get('GROQ_API_KEY'),
    "anthropic": userdata.get('ANTHROPIC_API_KEY'),
}
```

---

## 2. 실행 전 점검 (API 호출 없음, 비용 0)

### 2-1. 시나리오 형식 점검

새 시나리오를 추가했으면 **반드시 먼저** 이것부터 돌린다.

```python
from coop_pipeline.runner import check_scenario_dir
check_scenario_dir("scenarios")
```

```
OK   scenarios\A1\A1_simple_meeting.json
...
결과: 7개 통과 / 0개 오류
```

`FAIL`이 뜨면 무엇이 틀렸는지 함께 출력된다. 전부 `OK`가 될 때까지 고친 뒤 진행한다.

### 2-2. 모델 ID 확인 ⚠️ 중요

**제공사가 구형 모델을 예고 없이 퇴역시킨다.** 실험을 시작하기 전에 한 번 확인한다.
(실제로 이 레포에 원래 적혀 있던 `gemini-1.5-flash`, `llama-3.1-70b-versatile`,
`claude-sonnet-4-6`은 전부 지금 쓸 수 없는 ID였다.)

```python
# Gemini에서 지금 쓸 수 있는 모델
import google.generativeai as genai
genai.configure(api_key=API_KEYS["gemini"])
print([m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods])

# Groq에서 지금 쓸 수 있는 모델
from groq import Groq
print([m.id for m in Groq(api_key=API_KEYS["groq"]).models.list().data])
```

목록에 있어도 **실제로 호출되는지는 별개다** (`list_models`는 접근 권한을 반영하지
않는다). 확인 절차와 에러 코드별 대응은 `docs/DRYRUN.md` §3-4에 정리해 두었다.

쓸 수 없는 ID면 아래처럼 교체한다. **코드는 고치지 않는다.**

```python
import os
os.environ["COOP_ALPHA_MODEL"] = "gemini-2.5-flash"
os.environ["COOP_BETA_MODEL"]  = "llama-3.3-70b-versatile"
os.environ["COOP_JUDGE_MODEL"] = "claude-sonnet-5"
# 환경변수는 임포트 전에 설정해야 반영된다. 이미 임포트했다면 런타임 재시작.
```

현재 기본값은 `src/coop_pipeline/llm.py`의 `MODELS`에서 확인할 수 있다.

```python
from coop_pipeline.llm import MODELS
print(MODELS)
```

---

## 3. 실행

### 3-1. 가장 짧은 형태 — 시나리오 1개, 조건 1개

```python
from coop_pipeline.runner import run_scenario

result = run_scenario(
    "scenarios/A1/A1_simple_meeting.json",
    condition="명시",              # "명시" 또는 "묵시"
    api_keys=API_KEYS,
    replicate=1,
    n_solo=5,                      # 에이전트당 단독 조건 반복 수 (총 10개)
    max_turns=10,
    out_dir=DATA_DIR,              # 로그 저장. 생략하면 저장 안 함
)
```

실행하면 진행 상황과 판정 리포트가 바로 출력된다.

### 3-2. 명시/묵시 두 조건을 한 번에

단독 산출물을 한 번만 만들어 두 조건이 공유한다. **비용이 거의 절반이 되고,
두 조건이 같은 기준선을 쓰므로 비교가 정당해진다.**

```python
from coop_pipeline.runner import run_scenario_both_conditions

results = run_scenario_both_conditions(
    "scenarios/A1/A1_simple_meeting.json",
    api_keys=API_KEYS, n_solo=5, out_dir=DATA_DIR,
)
print(results["명시"]["level"], results["묵시"]["level"])
```

### 3-3. 시나리오 폴더 전체 × 두 조건 × 반복

```python
from pathlib import Path
from coop_pipeline.runner import run_scenario_both_conditions

REPS = 3
all_results = []

for path in sorted(Path("scenarios").rglob("*.json")):
    for rep in range(1, REPS + 1):
        try:
            r = run_scenario_both_conditions(
                path, api_keys=API_KEYS, replicate=rep,
                n_solo=5, out_dir=DATA_DIR, verbose=False,
            )
            all_results.extend(r.values())
        except Exception as e:
            print(f"SKIP {path.name} rep{rep}: {e}")

print(f"완료: {len(all_results)}건")
```

> 시나리오 7개 × 조건 2개 × rep 3회 = **42 dyad**. 이것이 파일럿 목표 규모다.

### 3-4. 중간에 실패했을 때 — 이어서 돌리기

`out_dir`을 주면 **태깅 전에 원본 대화 로그를 `{out_dir}/raw/`에 먼저 저장한다.**
대화 실행이 가장 비싼 단계이므로, 뒤에서 실패해도 그 결과를 잃지 않게 하기 위해서다.

```
DATA_DIR/
├── raw/                                    ← 대화만 끝난 상태 (codes/ref 없음)
│   └── A1_simple_meeting_명시_1.json
└── A1_simple_meeting_명시_1.json           ← 태깅·채점까지 끝난 완성본
```

태깅부터 이어서 돌리려면:

```python
import json
from coop_pipeline.runner import load_scenario, run_solo_batch, score_outputs, save_log
from coop_pipeline.agents import group_output_text
from coop_pipeline.tagging import tag_log
from coop_pipeline.scoring import attach_scores
from coop_pipeline import classify_log, format_result

log = json.load(open(f"{DATA_DIR}/raw/A1_simple_meeting_명시_1.json", encoding="utf-8"))
scenario = load_scenario("scenarios/A1/A1_simple_meeting.json")

log = tag_log(log, api_key=API_KEYS["anthropic"])
solo = run_solo_batch(scenario, API_KEYS, n_reps=5)
group_text = group_output_text(log)
s = score_outputs(scenario, solo, group_text, API_KEYS["anthropic"])
log["group_output_text"] = group_text
log = attach_scores(log, s["solo_values"], s["group_value"], s["new_idea_flag"])
save_log(log, DATA_DIR)
print(format_result(classify_log(log)))
```

---

## 4. 결과 읽는 법

`run_scenario()`가 돌려주는 딕셔너리의 구조:

| 키 | 뜻 |
|---|---|
| `level` | `"L0"` ~ `"L4"` — **협력이 어느 층위였는가** |
| `meaning` | 그 층위의 한 줄 설명 |
| `reason` | 그 층위로 판정된 직접적 이유 |
| `stopped_at` | 판정이 멈춘 질문(`"Q1"`~`"Q4"`). `None`이면 끝까지 통과(L4) |
| `checks` | Q1~Q4 각각의 통과 여부 + **실측값 + 기준값** |
| `features` | 계산된 정량 지표 전체 |
| `log` | 태깅·채점이 끝난 로그 전체 |
| `log_path` | 저장 경로 (`out_dir`을 준 경우) |

### 층위의 뜻

| 층위 | 뜻 |
|---|---|
| **L0** | 상호작용 없음 — 서로를 참조하지 않고 각자 말함 |
| **L1** | 편측 참조 — 한쪽만 상대를 반영 (협력이라기보다 청취) |
| **L2** | 상호 참조 — 주고받았으나 단독 조건보다 나은 산출물은 못 냄 |
| **L3** | 협력적 우위 — 상호 참조 + 단독 대비 산출 품질 우위 |
| **L4** | 창발적 협력 — L3 + 어느 쪽도 단독으로는 못 낸 새 해결책 생성 |

### 리포트 출력

```python
from coop_pipeline import format_result
print(format_result(result))
```

```
==============================================================
시나리오 : A1_simple_meeting (A1)
조건     : 명시 / rep 1
판정     : L2 — 상호 참조 — 주고받았으나 단독 조건보다 나은 산출물을 내지 못함
이유     : 쌍방향 참조는 있으나 집단 우위 미충족 (그룹=1.000, 단독 p90=1.000 (단독 n=10))
--------------------------------------------------------------
[PASS] Q1_참조존재
       실측: A→B=2회, B→A=3회
       기준: 둘 중 하나라도 1회 이상
[PASS] Q2_양방향성
       실측: A→B=2회, B→A=3회
       기준: 양방향 모두 2회 이상
[FAIL] Q3_집단우위
       실측: 그룹=1.000, 단독 p90=1.000 (단독 n=10)
       기준: 그룹 점수 > 단독 상위 90퍼센타일

병목: Q3 에서 판정이 멈춤
==============================================================
```

**`stopped_at`(병목)이 가장 중요한 정보다.** 결과가 이상할 때
"L2가 나왔다"보다 "Q3에서 그룹 1.000 vs 단독 p90 1.000으로 막혔다"가 원인을 알려준다.

위 예시는 `단독 p90 = 1.000`이므로 **단독으로도 만점이 나오는 시나리오라 협력의 여지가 없다**는
뜻이다. 시나리오가 너무 쉬운 것이지 코드 문제가 아니다.

### 요약 표

```python
import pandas as pd

df = pd.DataFrame([{
    "scenario": r["scenario_id"], "condition": r["condition"], "rep": r["replicate"],
    "level": r["level"], "stopped_at": r["stopped_at"],
    "dir_AB": r["features"]["dir_AB"], "dir_BA": r["features"]["dir_BA"],
    "comp_ratio": round(r["features"]["comp_ratio"], 3),
    "new_idea": r["features"]["new_idea_flag"],
} for r in all_results])

display(df)
print(pd.crosstab(df["condition"], df["level"]))    # 명시 vs 묵시 층위 분포
print(df["stopped_at"].value_counts())              # 어디서 가장 많이 막히는가
```

---

## 5. 저장된 로그로 다시 판정하기 (API 호출 0회, 비용 0)

임계값을 바꿔가며 결과가 어떻게 달라지는지 보는 것이 **캘리브레이션**이다.
이미 저장된 로그를 쓰므로 돈이 들지 않는다.

```python
from coop_pipeline.runner import classify_saved_log, classify_saved_dir
from coop_pipeline import load_thresholds

# 로그 1개 다시 판정
classify_saved_log(f"{DATA_DIR}/A1_simple_meeting_명시_1.json")

# 폴더 전체 요약
classify_saved_dir(DATA_DIR)

# 임계값을 바꿔서 재판정
harsh = dict(load_thresholds("v1"), bidirectional_min=3, comp_ratio_min=0.25)
classify_saved_dir(DATA_DIR, thresholds=harsh)
```

임계값을 확정하는 정식 절차는 [RUBRIC.md의 "임계값 캘리브레이션"](RUBRIC.md#6-임계값-캘리브레이션-절차) 참고.

---

## 6. 단계별로 따로 돌리기

`run_scenario()`는 전 단계를 한 번에 돈다. 중간에서 확인하고 싶으면 이렇게 나눠 쓴다.

```python
from coop_pipeline.runner import load_scenario, run_solo_batch, score_outputs
from coop_pipeline.agents import run_dyad, group_output_text
from coop_pipeline.tagging import tag_log
from coop_pipeline.scoring import attach_scores
from coop_pipeline import classify_log, format_result

scenario = load_scenario("scenarios/A1/A1_simple_meeting.json")

# (1) 단독
solo = run_solo_batch(scenario, API_KEYS, n_reps=5)

# (2) 대화 — 여기서 turns를 눈으로 확인
log = run_dyad(scenario, "명시", API_KEYS["gemini"], API_KEYS["groq"], replicate=1)
for t in log["turns"]:
    print(f"[{t['turn']}] {t['speaker']}: {t['text'][:80]}")
print("최종 산출물:", log["final_output"])

# (3) 태깅 — codes/ref가 잘 붙었는지 확인
log = tag_log(log, api_key=API_KEYS["anthropic"])
for t in log["turns"]:
    print(t["turn"], t["speaker"], t["codes"], "ref:", t["ref"], t.get("evidence"))

# (4~5) 채점
group_text = group_output_text(log)
s = score_outputs(scenario, solo, group_text, API_KEYS["anthropic"])
print(s)
log = attach_scores(log, s["solo_values"], s["group_value"], s["new_idea_flag"])

# (6~7) 판정
print(format_result(classify_log(log)))
```

---

## 7. 새 시나리오를 추가할 때

1. [scenarios/README.md](../scenarios/README.md)의 규칙대로 JSON을 만든다.
2. `check_scenario_dir("scenarios")` → 전부 `OK`인지 확인.
3. `run_scenario(..., n_solo=2, max_turns=4)`로 **작게 한 번** 돌려본다 (비용 절약).
4. 결과의 `stopped_at`을 본다.
   - 계속 `Q1`/`Q2`에서 막힌다 → 시나리오가 정보 교환을 요구하지 않는다. 지문을 다시 본다.
   - 계속 `Q3`에서 막힌다 → 단독으로도 풀리는 과제다. 난이도를 올린다.
5. 문제가 없으면 정식 규모(`n_solo=5, max_turns=10`)로 돌린다.

**코드는 고칠 필요가 없다.** 시나리오가 `task_variants`에 `shared` 또는 `(alpha, beta)`를
가지고 있기만 하면 파이프라인이 알아서 처리한다.

---

## 8. 테스트

로직을 고쳤으면 반드시 돌린다. **API 키 없이 돌아간다.**

```bash
pytest tests/ -q
```

Colab에서는:

```python
!pip install pytest -q
!python -m pytest tests/ -q
```

---

## 9. 자주 나는 에러

| 에러 메시지 | 원인 | 해결 |
|---|---|---|
| `turns[0]에 'codes' 없음 — 태깅(tag_log)을 먼저 실행할 것` | 태깅 단계를 건너뜀 | `tag_log()`를 먼저 실행 |
| `task_type='A1' 로그에 'solo_scores'가 없음` | 채점 단계를 건너뜀 | `attach_scores()`를 먼저 실행 |
| `필수 필드 누락: 'replicate'` | 옛날에 저장한 로그 | `log["replicate"] = 1`을 넣고 다시 저장 |
| `judge 호출 3회 모두 실패: ... not_found_error` | 모델 ID가 퇴역함 | §2-2 참고 |
| `응답을 JSON으로 해석할 수 없음: ...` | judge가 형식을 안 지킴 | 3회 재시도 후에도 실패한 것. 해당 발화 텍스트가 비정상적으로 길거나 비어 있는지 확인 |
| `단독 조건 점수가 비어 있음` | `n_solo=0`이거나 단독 실행이 전부 실패 | `n_solo`를 확인, 단독 실행부터 따로 돌려볼 것 |
| `'...'의 task_variants에 (alpha, beta) 또는 shared가 필요함` | 시나리오 형식 오류 | `check_scenario_dir()`로 확인 |
| `429` / rate limit | API 호출이 너무 잦음 | `agents.RATE_LIMIT_SLEEP`을 2~3초로 올린다 |

---

## 10. 비용 감각

시나리오 1개 × 조건 1개를 `n_solo=5, max_turns=10`으로 돌릴 때 대략:

| 단계 | 호출 수 |
|---|---|
| 단독 조건 | 10회 (gemini 5 + llama 5) |
| 2-agent 대화 | 10회 + 마무리 1회 |
| 태깅 | **10회** (턴당 1회, claude) |
| 채점 (A1) | 0회 — 체크리스트는 코드 계산 |
| 채점 (A2/A4) | 11회 (단독 10 + 그룹 1, claude) |
| 신규 아이디어 판정 | 1회 (claude) |

A1은 claude 호출 11회, A2/A4는 22회.
**단독 산출물은 조건 간에 재사용된다** — 반드시 `run_scenario_both_conditions()`를 쓸 것.
