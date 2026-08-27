# 판정 프레임 시험하기

돈을 쓰기 전에 판정 프레임이 제대로 도는지 확인하고, 그다음 파이프라인 전체를
한 번 돌려보는 순서를 적는다. 예시는 `A4_simple_energy_save`을 쓴다.

- 전체 실행 옵션과 결과 해석 → [PIPELINE.md](PIPELINE.md)
- 채점 기준 → [RUBRIC.md](RUBRIC.md)
- 태깅 기준 → [CODING_MANUAL.md](CODING_MANUAL.md)

> **시나리오 파일은 읽기만 한다.** 아래 어느 절차도 `scenarios/*.json`을 고치지 않는다.

---

## 0. 핵심 아이디어

파이프라인은 두 부분으로 나뉜다.

| | 하는 일 | API | 비용 |
|---|---|---|---|
| **생성부** `agents` `tagging` `scoring` | 대화를 만들고 태깅·채점한다 | 필요 | 든다 |
| **판정부** `validate_log` `features` `classify` | 완성된 로그를 L0~L4로 판정한다 | **불필요** | **0원** |

판정부는 **순수 함수**다. 로그 딕셔너리 하나만 받고 네트워크를 쓰지 않는다.
그래서 대화를 실제로 돌리지 않고 **모의 로그를 손으로 만들어 넣으면**
판정 프레임만 따로 시험할 수 있다. 이것이 §1이다.

---

## 1. 판정 프레임만 시험하기 (API 호출 0회)

```bash
python tools/dryrun_frame.py --scenario scenarios/A4/A4_simple_energy_save.json
```

인자를 생략하면 A4_simple_energy_save이 기본값이다. 네 가지를 순서대로 한다.

| 단계 | 확인하는 것 |
|---|---|
| `[1]` 시나리오 형식 점검 | JSON 필드, 파일명-scenario_id 일치, solo/shared 유무 |
| `[2]` 결정 트리 반응 시험 | 입력을 조작해 만든 모의 로그 5개가 L0~L4로 갈리는가 |
| `[3]` 임계값 민감도 | 어느 기준을 한 칸 올리면 판정이 뒤집히는가 |
| `[4]` 실행 전 환경 점검 | SDK 설치 여부, 모델 ID, 키 존재 여부 (호출은 안 함) |

`[2]`의 출력이 이렇게 나오면 판정부는 정상이다.

```
  L0   서로 참조 0회                    -> L0   OK
  L1   alpha만 상대를 참조 (편측)         -> L1   OK
  L2   쌍방향이지만 집단 우위 없음          -> L2   OK
  L3   집단 우위 있으나 신규성 미달         -> L3   OK
  L4   집단 우위 + Q4 세 조건 모두 충족     -> L4   OK
```

**여기서 쓰는 점수는 전부 가짜다.** 프레임이 그 숫자에 어떻게 반응하는지를 보는
것이지, A4 시나리오의 실제 성능을 보는 것이 아니다.

### 1-1. 직접 손으로 만져보기

스크립트를 읽지 않고 바로 실험하려면 `make_log()`의 손잡이만 돌리면 된다.

```python
import sys
sys.path.insert(0, "src"); sys.path.insert(0, "tools")   # 레포 루트에서 실행

from dryrun_frame import make_log
from coop_pipeline import classify_log, format_result

log = make_log("A4_simple_energy_save", "A4",
               n_turns=10,
               ref_ab=3, ref_ba=3,      # 참조 방향별 횟수  -> Q1, Q2
               comp=2,                  # comp 코드 턴 수    -> Q4a
               agree=2,                 # agree+참조 턴 수   -> Q4c
               solo=[3, 3, 3, 3, 3],    # 단독 등급          -> Q3
               group=5,                 # 그룹 등급          -> Q3
               new_idea=True)           # 신규 해결책        -> Q4b

print(format_result(classify_log(log)))
```

하나씩 바꿔가며 "이 값이면 몇 층위가 나와야 하지?"를 먼저 예측하고 돌려보는 것이
프레임을 검증하는 방법이다. 예측과 다르면 프레임이 틀렸거나 기준의 뜻을 잘못
이해한 것이고, 둘 다 실제 실행 전에 알아야 할 문제다.

### 1-2. A4에서 특히 볼 것 — Q3가 얼마나 빡센가

A4는 LLM-judge 1~5 등급을 쓰고, Q3 통과 조건이 `grade_gap_min = 2`다.

> 단독 중앙값이 3점이면 **그룹이 5점(만점)이어야** Q3를 통과한다.

`A4_simple_energy_save`은 제약이 뚜렷하다 (슬로건 12자, 아이디어 정확히 3개,
무비용). judge 루브릭은 제약 위반 시 최대 3점으로 자른다. 단독·그룹이 나란히
3점을 받으면 등급차 0 → **항상 L2**가 된다.

이건 코드 버그가 아니라 **임계값이 아직 파일럿 근거 없이 정해진 가설값**이라서다
(`configs/thresholds_v1.json`의 주석 참고). 파일럿에서 `stopped_at`이 Q3에 몰리면
그때 캘리브레이션 대상이 된다. 지금 미리 고치지 말 것 — 근거 없이 기준을 낮추면
"협력 효과가 있었다"가 아니라 "기준을 낮췄다"가 된다.

### 1-3. 자동 테스트

```bash
python -m pytest tests/ -q      # 17 passed
```

`tests/test_classify.py`도 같은 원리(모의 로그)로 판정부를 검증한다.
판정 로직을 고쳤으면 반드시 돌린다. **API 키 없이 돌아간다.**

---

## 2. 실제 실행 환경 — 어디서 돌릴 것인가

**결론: 전체 파이프라인은 Colab에서 돌린다. 이 PC에서는 안 된다.**

이 PC의 Python은 **3.7**인데, `requirements.txt`의 SDK는 그보다 높은 버전을 요구한다.

| 패키지 | 요구 Python | 이 PC(3.7) |
|---|---|---|
| `google-generativeai>=0.8` | 3.9+ | 설치 불가 |
| `anthropic>=0.40` | 3.8+ | 설치 불가 |
| `groq>=0.11` | 3.8+ | 설치 불가 |

`tools/dryrun_frame.py`와 `pytest tests/`는 **표준 라이브러리만 쓰므로 이 PC에서도
돈다**. 즉 역할이 이렇게 나뉜다.

- **로컬 (이 PC)** — 판정 프레임 시험, 임계값 실험, 저장된 로그 재판정. 비용 0.
- **Colab** — 실제 대화 생성 + 태깅 + 채점. API 키 필요.

로컬에서 전체를 돌리고 싶으면 Python 3.11을 따로 설치하고 가상환경을 만들어야 한다.
파일럿 규모에서는 굳이 그럴 이유가 없다 (Colab이 이미 3.11이고 GPU도 필요 없다).

---

## 3. 파이프라인 전체 한 번 돌려보기 (Colab)

### 3-1. 준비물

| | 어디서 | 비고 |
|---|---|---|
| `GEMINI_API_KEY` | [aistudio.google.com](https://aistudio.google.com/apikey) | alpha 생성. 무료 티어 있음 |
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com/keys) | beta 생성. 무료 티어 있음 |
| `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com) | 태깅·채점. **유료** (아래 참고) |

키는 Colab **Secrets**(왼쪽 열쇠 아이콘)에 등록한다. 노트북 셀에 직접 붙여넣지 않는다.
붙여넣으면 노트북을 공유하는 순간 키가 같이 나간다.

### judge 제공사 바꾸기 (비용 0으로 돌리기)

태깅·채점(judge)만 유료다. `COOP_JUDGE_PROVIDER`로 무료 제공사에 넘길 수 있다.

```python
import os
os.environ["COOP_JUDGE_PROVIDER"] = "groq"
os.environ["COOP_JUDGE_MODEL"]    = "openai/gpt-oss-120b"   # 셀 4에서 실제 ID 확인
# 임포트 전에 설정할 것. 이미 임포트했으면 런타임 재시작.
```

| 제공사 | 비용 | 필요한 키 |
|---|---|---|
| `anthropic` (기본) | 유료 | `ANTHROPIC_API_KEY` |
| `groq` | 무료 티어 | `GROQ_API_KEY` |
| `gemini` | 무료 티어 | `GEMINI_API_KEY` |

> ⚠️ **judge는 생성 모델과 다른 계열이어야 한다.** alpha=Gemini가 최종 산출물을
> 작성하므로(`agents.FINALIZE_PROMPT`) judge까지 Gemini면 자기 계열 글을 자기가
> 채점하게 되고(self-preference), `group_grade`가 부풀어 L3/L4가 과대 추정된다.
> beta=Llama이므로 Llama 계열도 피해야 한다.
>
> 무료로 갈 때 안전한 조합은 **judge = Groq의 Llama가 아닌 계열**이다.
> `tools/dryrun_frame.py`의 `[4]` 블록이 계열이 겹치면 경고를 띄운다.

### 3-2. 셀 1 — 세팅

```python
!git clone https://github.com/ewha-oi/OrganicAI.git
%cd OrganicAI
!pip install -r requirements.txt -q

import sys; sys.path.append('src')

from google.colab import userdata
API_KEYS = {
    "gemini":    userdata.get('GEMINI_API_KEY'),
    "groq":      userdata.get('GROQ_API_KEY'),
    "anthropic": userdata.get('ANTHROPIC_API_KEY'),
}
```

> 이 브랜치(`yr`)를 쓰려면 `!git clone -b yr ...`. 두 번째부터는 `!git pull`.

### 3-3. 셀 2 — 무비용 점검부터

키를 쓰기 전에 §1을 Colab에서도 한 번 돌린다. 여기서 걸리면 돈이 안 든다.

```python
!python tools/dryrun_frame.py --scenario scenarios/A4/A4_simple_energy_save.json
```

### 3-4. 셀 3 — 모델 ID 확인 ⚠️

**가장 흔한 실패 원인이다.** 제공사가 구형 모델을 예고 없이 퇴역시킨다.

```python
import google.generativeai as genai
genai.configure(api_key=API_KEYS["gemini"])
print([m.name for m in genai.list_models()
       if 'generateContent' in m.supported_generation_methods])

from groq import Groq
print([m.id for m in Groq(api_key=API_KEYS["groq"]).models.list().data])
```

**목록에 있다고 쓸 수 있는 것이 아니다.** `list_models`는 계정의 접근 권한을 반영하지
않는다. 실제로 목록에 `gemini-2.5-flash`가 보이는데도 호출하면
`404 no longer available to new users`가 오는 경우가 있다 (새로 만든 프로젝트).
그러니 **한 번 호출해 본다.**

```python
for cand in ["gemini-2.5-flash", "gemini-flash-latest"]:
    try:
        print("OK  ", cand, genai.GenerativeModel(cand).generate_content("ping").text[:30])
    except Exception as e:
        print("FAIL", cand, type(e).__name__, str(e)[:300])
```

에러 코드로 원인이 갈린다.

| 코드 | 뜻 | 대응 |
|---|---|---|
| 404 `no longer available to new users` | 프로젝트가 컷오프 이후 생성됨 | 컷오프 이전 프로젝트에서 키를 발급받거나 신형 모델로 |
| 403 `project has been denied access` | 무료·무결제 프로젝트에 안 열린 모델 | 결제 등록, 또는 다른 모델 |
| 429 `quota exceeded` | 그 모델의 무료 쿼터가 0 | 다른 모델 |

`OK`가 뜬 ID로 **코드를 고치지 말고** 환경변수로 교체한다 (`llm.py`가 환경변수를 먼저 읽는다).

```python
import os
os.environ["COOP_ALPHA_MODEL"] = "실제 되는 ID"
os.environ["COOP_BETA_MODEL"]  = "실제 되는 ID"
# 임포트 전에 설정해야 반영된다. 이미 임포트했으면 런타임 재시작.
# 재시작이 곤란하면 딕셔너리를 직접 고친다 (agents.py가 같은 객체를 참조한다):
#   from coop_pipeline import llm; llm.MODELS["alpha"] = "실제 되는 ID"
```

### 3-5. 셀 4 — 작게 한 번 (권장 첫 실행)

**처음부터 정식 규모로 돌리지 않는다.** 배관이 새는지부터 본다.

```python
from coop_pipeline.runner import run_scenario

result = run_scenario(
    "scenarios/A4/A4_simple_energy_save.json",
    condition="명시",
    api_keys=API_KEYS,
    n_solo=2,          # 정식은 5
    max_turns=4,       # 정식은 10
    out_dir="runs",    # 로그 저장 (raw 체크포인트 포함)
)
```

이 설정의 Claude 호출은 **약 10회**다 (태깅 4 + 채점 5 + 신규성 1).
Gemini/Groq는 무료 티어로 충분하다.

단계별 진행이 출력되므로 **어디서 멈췄는지가 바로 보인다.**

```
[A4_simple_energy_save/명시] 단독 조건 실행 (2 에이전트 x 2회)
[A4_simple_energy_save/명시] 2-agent 대화 실행 (4턴)
[A4_simple_energy_save/명시] 발화 태깅
[A4_simple_energy_save/명시] 채점
[A4_simple_energy_save/명시] 판정
```

### 3-6. 셀 5 — 중간 산출물을 눈으로 볼 것

숫자만 보면 파이프라인이 "돌았는지"는 알아도 "제대로 돌았는지"는 모른다.

```python
log = result["log"]

# 대화가 실제로 오갔는가? (혼잣말만 하고 있지 않은가)
for t in log["turns"]:
    print("[%d] %s: %s" % (t["turn"], t["speaker"], t["text"][:120]))

# 태깅이 붙었는가? codes가 전부 비었거나 ref가 전부 None이면 태깅 실패다
for t in log["turns"]:
    print(t["turn"], t["speaker"], t["codes"], "ref:", t["ref"], t.get("evidence"))

# 무엇을 채점했는가? 이게 이상하면 등급도 이상하다
print(log["group_output_text"])
print("단독 등급:", log["solo_grades"], "/ 그룹 등급:", log["group_grade"])
```

특히 볼 것:

- `ref`가 **전부 None**이면 → 무조건 L0이 나온다. 대화가 정말 겉돌았는지,
  태깅이 실패했는지 원문을 보고 판단한다. 이 둘은 결과가 같아도 의미가 정반대다.
- `group_output_text`가 `"alpha: ..."` 같은 대화체면 → finalize 단계가 실패한 것이다.
- `solo_grades`가 전부 같은 값이면 → 시나리오가 변별력이 없다는 신호다.

### 3-7. 셀 6 — 결과 읽기

```python
from coop_pipeline import format_result
print(format_result(result))
print("병목:", result["stopped_at"])
```

**`stopped_at`이 가장 중요하다.** "L2가 나왔다"보다 "Q3에서 등급차 0으로 막혔다"가
다음에 뭘 해야 할지를 알려준다.

| 병목 | 뜻 | 할 일 |
|---|---|---|
| `Q1`/`Q2` | 서로를 참조하지 않음 | 지문이 정보 교환을 요구하는지, 태깅이 됐는지 확인 |
| `Q3` | 단독보다 나을 게 없음 | 과제가 혼자서도 풀린다. §1-2 참고 |
| `Q4` | 우위는 있으나 신규성 미달 | 정상. L3도 유효한 결과다 |
| `None` | 끝까지 통과 = L4 | 로그 원문을 반드시 눈으로 검증할 것 |

### 3-8. 셀 7 — 두 조건 비교 (정식 실행)

배관이 다 확인되면 정식 규모로 간다. 명시/묵시가 **단독 산출물을 공유**하므로
비용이 거의 절반이고, 두 조건이 같은 기준선을 쓰게 되어 비교가 정당해진다.

```python
from coop_pipeline.runner import run_scenario_both_conditions

results = run_scenario_both_conditions(
    "scenarios/A4/A4_simple_energy_save.json",
    api_keys=API_KEYS, n_solo=5, max_turns=10, out_dir="runs",
)
print(results["명시"]["level"], results["묵시"]["level"])
```

### 3-9. 셀 8 — 저장된 로그로 재판정 (비용 0, 로컬에서도 가능)

`out_dir`에 저장된 로그는 API 없이 몇 번이든 다시 판정할 수 있다.
임계값 캘리브레이션은 전부 여기서 한다.

```python
from coop_pipeline.runner import classify_saved_dir
from coop_pipeline import load_thresholds

classify_saved_dir("runs")

harsh = dict(load_thresholds("v1"), grade_gap_min=1)
classify_saved_dir("runs", thresholds=harsh)
```

`runs/`를 로컬로 내려받으면 이 PC(Python 3.7)에서도 그대로 돈다.

---

## 4. 자주 나는 실패

| 증상 | 원인 | 대응 |
|---|---|---|
| `judge 호출 3회 모두 실패: not_found_error` | 모델 ID 퇴역 | §3-4 |
| `turns[0]에 'codes' 없음` | 태깅 건너뜀 | `tag_log()` 먼저 |
| `task_type='A4' 로그에 'solo_grades'가 없음` | 채점 건너뜀 | `attach_scores()` 먼저 |
| `429` / rate limit | 호출이 너무 잦음 | `agents.RATE_LIMIT_SLEEP`을 2~3초로 |
| 태깅 후 `ref`가 전부 None | 대화가 겉돌았거나 태깅 실패 | 원문부터 볼 것 (§3-6) |
| 결과가 늘 L2 | Q3 병목 | §1-2 |

대화까지는 됐는데 뒤에서 죽었다면 **다시 처음부터 돌리지 않는다.**
`out_dir`을 줬으면 `{out_dir}/raw/`에 대화 로그가 이미 저장돼 있다.
이어서 돌리는 코드는 [PIPELINE.md §3-4](PIPELINE.md) 참고.

---

## 5. 설계 경계 — 오케스트레이터를 두지 않는다

회의록의 "오케스트레이터 없이" 결정은 지금 코드에서 지켜지고 있다.
`runner.py`는 이름이 한때 "오케스트레이터"였지만 실제로 하는 일은 다르다.

| | 오케스트레이터 (안 씀) | 실행 드라이버 (`runner.py`) |
|---|---|---|
| 다음 화자 결정 | 세 번째 LLM이 판단 | `turn % 2` 고정 교대 |
| 종료 시점 | LLM이 판단 | `max_turns` 도달 |
| 메시지 중계 | 요약·재작성해서 전달 | 전달하지 않음 (각자 history를 받음) |
| 제어 흐름의 LLM | 있음 | **없음** |

차이는 "LLM이 제어 흐름에 개입하는가"다. 중재자가 끼면 측정 대상이
*alpha와 beta의 협력*이 아니라 *중재자의 조율 능력*이 되어 실험이 무효가 된다.

유일한 개입은 대화 종료 후 1회 실행되는 finalize 호출
(`agents.FINALIZE_PROMPT`)이다. 고정 문자열 템플릿이고 LLM의 판단이 아니지만,
대화에 대한 개입인 것은 맞다. "무엇을 채점할 것인가"를 정의하려고 의도적으로 넣었고,
현재 설계에서는 **alpha가 최종 산출물을 작성한다** — 즉 그룹 산출물의 문장력에
alpha 모델의 특성이 섞인다. 논문에 쓸 때 명시해야 할 한계다.

앞으로 이 선을 넘지 말 것:

> `runner.py`나 `agents.run_dyad()`에서 LLM을 불러 다음 화자·종료 시점·논의 주제를
> 정하게 만들면 그 순간 오케스트레이터가 되고 회의록의 설계 결정이 깨진다.
> 새 단계가 필요하면 LLM 판단이 아니라 결정론적 분기로 추가한다.

같은 내용이 `src/coop_pipeline/runner.py` 파일 상단 주석에도 적혀 있다.
