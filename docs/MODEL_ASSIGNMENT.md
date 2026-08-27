# 모델 배정 기록 (alpha / beta / judge)

파일럿에 쓸 세 모델을 **어떻게 골랐고, 왜 그렇게 골랐는지** 남긴 문서다.
배정은 이미 끝났고, 확정값은 노트북 1절 「모델 배정」 셀 한 곳에만 적혀 있다.

> 이 문서는 **읽는 문서**다. 평소 실행에는 필요 없다.
> 모델이 퇴역했거나 Groq 키/티어가 바뀌어 **다시 배정해야 할 때만** §4의 측정 코드를 꺼내 쓴다.

## 확정 배정

| 역할 | 모델 | 계열 | 근거 |
|---|---|---|---|
| judge | `qwen/qwen3.6-27b` (provider `groq`) | qwen | 태깅 4/5 — 통과 기준 충족 |
| alpha | `openai/gpt-oss-120b` | gpt | 태깅 4/5로 셋 중 가장 나았으나 judge 자리를 qwen에 내주고 생성으로 |
| beta | `openai/gpt-oss-20b` | gpt | `_call_llama` 경로 실측 OK (346자 / 1.8초) |

노트북에서 이 값을 넣는 곳:

```python
os.environ["COOP_JUDGE_PROVIDER"] = "groq"
os.environ["COOP_JUDGE_MODEL"]    = "qwen/qwen3.6-27b"
os.environ["COOP_ALPHA_MODEL"]    = "openai/gpt-oss-120b"
os.environ["COOP_BETA_MODEL"]     = "openai/gpt-oss-20b"
```

`llm.py`가 **임포트 시점에 한 번만** 환경변수를 읽는다(`llm.py:92`). 그래서 이 셀은
`coop_pipeline`을 임포트하는 어떤 셀보다 먼저 실행해야 하고, 값을 바꿨으면 런타임을 재시작해야 한다.

---

## 1. 왜 배정을 다시 해야 했나

원래 설계는 alpha=Gemini, beta=Llama, judge=Claude로 **계열이 셋 다 달랐다**.
파일럿 직전에 두 개가 동시에 무너졌다.

| 무너진 것 | 증상 | 결과 |
|---|---|---|
| Gemini | 전 모델 403/404 (계정 차단) | alpha를 Groq 모델로 대체해야 함 |
| `llama-3.3-70b-versatile` | 이 계정 모델 목록에 없음 | `MODELS["beta"]` 기본값 그대로 두면 beta 턴마다 404 |

그래서 alpha·beta·judge 셋을 **전부 Groq 무료 티어 안에서** 다시 고르게 됐다.
Anthropic 키를 쓰지 않게 된 것도 이 때문이다(`API_KEYS`에서 `anthropic`이 빠진 이유).

## 2. 무엇을 봤나 (요건)

역할마다 요건이 다르다. 프로브 셀(§4 `[0]`)이 이 다섯 가지를 한 번에 잰다.

| 항목 | 왜 보는가 | 어느 역할에 치명적인가 |
|---|---|---|
| JSON 모드 (`response_format`) | 지원하지 않으면 `llm.py`가 400을 받는다 | **judge** — 없으면 탈락 |
| 한글 비율 | 낮으면 한국어 지시를 무시한다는 뜻 | alpha / beta |
| `<think>` 누출 | 추론 과정이 그대로 산출물이 되어 채점 대상에 섞인다 | alpha / beta — 생성 역할 탈락 |
| context window | alpha는 대화 전체 + 마무리 프롬프트를 받는다 | alpha |
| TPM (분당 토큰) | `max_turns` 상한을 사실상 이것이 정한다 | 전부 |

고르는 순서는 **judge → alpha → beta**다. 채점이 무너지면 실험 자체가 성립하지 않으므로
가장 까다로운 자리를 먼저 채운다.

## 3. 측정 결과와 판단

### 3-1. 프로브 결과에서 걸러낸 것

| 모델 | 왜 뺐나 |
|---|---|
| `groq/compound` | RPD 250 + 내부 도구 호출 — 대화에 외부 정보가 섞일 수 있어 실험 조건이 오염된다 |
| `allam-2-7b` | ctx 4096 < `llm.py`의 `_MIN_OUTPUT_TOKENS` 4096 — 구조적으로 사용 불가 |
| `llama-3.3-70b-versatile` | 이 키로 호출 불가 (목록에 없음) |

남은 것은 사실상 **gpt 계열 2개 + qwen 계열 1개**뿐이었다.

### 3-2. judge 태깅 정확도 (§4 `[1]`)

통과 기준은 노트북 4절과 같은 5개 사례에서 **4/5 이상**이다.

| 모델 | 계열 | 태깅 | 관찰 |
|---|---|---|---|
| `qwen/qwen3.6-27b` | qwen | **4/5** | 통과 |
| `openai/gpt-oss-120b` | gpt | **4/5** | `comp`를 놓쳤다 |
| `openai/gpt-oss-20b` | gpt | 3/5 | 과잉 태깅 (`phatic`을 `lead`로) — 탈락 |

여기서 **측정 자체가 한 번 막혔던 것**을 기록해 둔다. `llm.py:167`의
`_GROQ_REASONING_MODELS`에 `"qwen3"`이 들어 있어 qwen에 `reasoning_effort="low"`를 보내는데,
`qwen3.6`은 `none`/`default`만 받는다 → 400. 그대로 재면 qwen은 **능력과 무관하게 0/5**가 나온다.
그래서 `[1]`은 `llm.py`를 우회해 같은 조건(system=`CODING_MANUAL`, JSON 모드, `temperature=0`)을
직접 재현하되 모델마다 맞는 `reasoning_effort`를 보내 '태깅 능력'만 분리해 쟀다.
(이 400 자체는 이후 `llm.py`에서 고쳤다 — 커밋 `59bfe87`.)

### 3-3. 계열이 겹치는 문제를 어디에 둘 것인가

쓸 수 있는 계열이 gpt와 qwen 둘뿐이라 **어딘가는 반드시 겹친다.** 겹치는 자리를 미리 정해 뒀다.

| `[1]` 결과 | judge | alpha | beta | 겹치는 곳 | 판정에 미치는 영향 |
|---|---|---|---|---|---|
| qwen ≥ 4/5 ← **실제** | `qwen3.6-27b` | `gpt-oss-120b` | `gpt-oss-20b` | alpha ≡ beta | 채점자가 어느 쪽 편도 아니다 — **편향 없음** |
| qwen < 4/5 | `gpt-oss-120b` | `qwen3.6-27b` | `gpt-oss-20b` | judge ≡ beta | beta 단독 점수만 부풀어 Q3(집단 우위)이 **어려워지는** 보수적 방향 |

qwen이 4/5로 통과했으므로 위쪽이 됐다. 요지는 **겹침을 '측정(채점)'이 아니라 '설정(생성)' 쪽에 몰아넣는 것**이다.
alpha와 beta가 같은 계열인 것은 두 에이전트를 서로 비교하지 않으므로 판정에 영향을 주지 않는다.

### 3-4. beta 호출 경로 점검 (§4 `[2]`)

`agents._call_llama`(`agents.py:83`)는 `max_tokens`도 `reasoning_effort`도 보내지 않는다.
`gpt-oss`는 추론 모델이라 alpha에서 실제로 겪은 것처럼 **빈 응답**이 올 수 있어서, 추측하지 않고
실제 호출 경로에 그대로 태워 확인했다 → **OK (346자 / 1.8초)**. 그래서 `src` 수정 없이 beta로 썼다.

반대로 alpha는 같은 문제에 걸렸다. 그래서 노트북 3절의 shim이 `max_tokens=4096`과
`reasoning_effort`를 붙이고, `<think>` 블록을 지운 뒤 텍스트를 넘긴다. 어떤 추론 옵션을 받는지는
모델마다 다르고 틀리면 400이므로 shim이 **런타임에 하나씩 시도해서** 통과하는 조합을 고른다.

### 3-5. 남은 것 하나

`gpt-oss-120b`(alpha)가 `[1]`에서 `comp`를 놓쳤다. `comp_ratio_min`이 판정 Q4a에 걸려 있으므로
**임계값을 v2로 확정하기 전에 다시 확인해야 한다.** 지금 judge는 qwen이라 판정에 직접 쓰이지는 않지만,
judge를 gpt 계열로 되돌리는 상황이 오면 이 항목이 먼저다.

## 4. 다시 배정해야 할 때 쓰는 코드

아래 세 셀을 노트북에 붙여 넣고 순서대로 돌린다. `API_KEYS`는 1절에서 만들어져 있어야 한다.
`[1]`·`[2]`는 `coop_pipeline`을 임포트하므로, 결과를 보고 배정 셀을 고쳤다면 **런타임을 재시작**해야 반영된다.

### `[0]` 쓸 수 있는 모델의 특성 조사

```python
# 내 Groq 키로 지금 쓸 수 있는 모델의 특성을 조사한다. 모델당 1~2회, 짧은 호출.
import re, time
from groq import Groq

client   = Groq(api_key=API_KEYS["groq"])
NOT_CHAT = ("whisper", "tts", "embed", "guard", "moderation", "ocr")
FAMILIES = ("gemini","gemma","claude","llama","gpt","qwen","deepseek",
            "mistral","kimi","moonshot","grok","compound")
PROBE = ('한국어로만 답하라. 교내 에너지 절약 캠페인 기획안을 3문장으로 요약해 '
         '아래 JSON 형식으로만 출력하라. {"제목": "...", "요약": "..."}')

rows, bad = [], []
for m in sorted(client.models.list().data, key=lambda x: x.id):
    if any(t in m.id.lower() for t in NOT_CHAT):
        continue
    low = m.id.lower()
    r = {"id": m.id, "ctx": getattr(m, "context_window", 0) or 0,
         "fam": next((f for f in FAMILIES if f in low), low.split("/")[-1].split("-")[0])}
    t0 = time.time()
    for json_mode in (True, False):      # JSON 모드 지원 여부 = judge 자격 요건
        kw = {"response_format": {"type": "json_object"}} if json_mode else {}
        try:
            resp = client.chat.completions.with_raw_response.create(
                model=m.id, messages=[{"role": "user", "content": PROBE}],
                max_tokens=2048, temperature=0, **kw)
            text  = resp.parse().choices[0].message.content or ""
            dense = re.sub(r"\s", "", text)
            r.update(json_mode=json_mode, sec=round(time.time()-t0, 1),
                     ko=round(len(re.findall(r"[가-힣]", text))/max(len(dense),1), 2),
                     think="<think>" in text, empty=not dense,
                     tpm=resp.headers.get("x-ratelimit-limit-tokens", "?"),
                     rpd=resp.headers.get("x-ratelimit-limit-requests", "?"),
                     head=re.sub(r"\s+", " ", text)[:60])
            rows.append(r); break
        except Exception as e:
            err = str(e)[:70]
    else:
        bad.append({"id": m.id, "err": err})
    print(".", end="")

print(f"\n\n{'모델':42s}{'계열':9s}{'ctx':>8s}{'JSON':>6s}{'한글':>6s}"
      f"{'초':>6s}{'TPM':>8s}{'RPD':>7s}  비고")
print("-" * 100)
for r in rows:
    note = ("<think>누출 " if r["think"] else "") + ("빈응답(토큰부족)" if r["empty"] else "")
    print(f"{r['id'][:41]:42s}{r['fam'][:8]:9s}{r['ctx']:>8d}"
          f"{('O' if r['json_mode'] else 'X'):>6s}{r['ko']:>6.2f}{r['sec']:>6.1f}"
          f"{str(r['tpm']):>8s}{str(r['rpd']):>7s}  {note}")

print("\n[이 키로 호출 불가 — 목록에는 있으나 티어/퇴역]")
for r in bad:
    print(f"  {r['id'][:41]:42s}{r['err']}")

print("\n[샘플 출력 — 한국어와 형식을 눈으로 볼 것]")
for r in rows:
    print(f"  {r['id'][:41]:42s}{r['head']}")
```

읽는 법은 §2의 표와 같다. JSON이 `O`가 아닌 모델은 judge 후보에서 바로 빠지고,
한글 비율이 낮거나 `<think>`가 누출되는 모델은 alpha/beta 후보에서 빠진다.

### `[1]` judge 후보의 태깅 정확도 — 4/5 이상이 통과

```python
# llm.py 를 거치지 않고 같은 조건(system=CODING_MANUAL, JSON 모드, temperature=0)을
# 직접 재현한다. 모델마다 맞는 reasoning_effort 를 보내 '태깅 능력'만 분리해서 잰다.
from coop_pipeline.tagging import CODING_MANUAL
from coop_pipeline.llm import parse_json_strict
from groq import Groq

CANDIDATES = {
    "qwen/qwen3.6-27b":      "none",   # 'low' 를 안 받는다
    # "openai/gpt-oss-120b": "low",    # 이미 4/5. 다시 재려면 주석 해제
    # "openai/gpt-oss-20b":  "low",    # 이미 3/5
}

CASES = [
    ("수요일 B실로 하자.",     "좋아, 그렇게 하자.",                                 "phatic"),
    ("수요일에 하는 게 어때?",  "맞네, 나는 A실을 생각했는데 수요일이면 B실이 맞겠다.",  "agree"),
    ("수요일 B실로 하자.",     "좋아. 그런데 예산 확인도 필요해 보여.",                "comp"),
    ("예산은 200이야.",       "응, 200이지.",                                     "phatic"),
    ("회의 준비 시작하자.",    "지금 정할 건 요일이야. 시간은 나중에.",                "lead"),
]

client = Groq(api_key=API_KEYS["groq"])
for model_id, effort in CANDIDATES.items():
    print(f"=== {model_id}  (reasoning_effort={effort})")
    hit = 0
    for prev, cur, want in CASES:
        try:
            r = client.chat.completions.create(
                model=model_id, temperature=0, max_tokens=4096,
                reasoning_effort=effort,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": CODING_MANUAL},
                          {"role": "user", "content":
                           f"대화 맥락:\nalpha: {prev}\n\n분류할 발화 (beta): {cur}"}])
            text = r.choices[0].message.content or ""
            if not text:
                print(f"  X 기대={want:6s} 빈 응답 — 추론 토큰 소진")
                continue
            codes = parse_json_strict(text).get("codes", [])
        except Exception as e:
            print(f"  X 기대={want:6s} {str(e)[:80]}")
            continue
        ok = (codes == ["phatic"]) if want == "phatic" else (want in codes)
        hit += ok
        print(f"  {'O' if ok else 'X'} 기대={want:6s} 실제={codes}")
    print(f"  -> {hit}/5\n")
```

### `[2]` beta 호출 경로 점검

```python
# agents._call_llama 는 max_tokens 도 reasoning_effort 도 보내지 않는다.
# 추측하지 말고 실제 호출 경로를 그대로 태워서 확인한다.
#   OK 가 뜨면  -> beta 는 코드 수정 없이 쓸 수 있다.
#   실패하면    -> _call_llama 에 추론 옵션을 넣는 src 수정이 필요하다.
import time
from groq import Groq
from coop_pipeline import agents, llm
from coop_pipeline.runner import load_scenario

# 하드코딩하지 말 것. 1절의 COOP_BETA_MODEL 이 유일한 출처다.
BETA_CANDIDATE = llm.MODELS["beta"]

v = load_scenario("scenarios/A4/A4_simple_energy_save.json")["task_variants"]
prompt = agents.SYSTEM_PROMPT_TEMPLATES["명시"].format(
    name="beta", partner="alpha", task=v.get("beta") or v["shared"]
) + "\n\n지금까지의 대화:\n(없음)\n\n너의 다음 발언:"

print(f"beta = {BETA_CANDIDATE}")
t0 = time.time()
try:
    out = agents._call_llama(Groq(api_key=API_KEYS["groq"]), prompt)
    print(f"OK — {len(out)}자 / {time.time() - t0:.1f}초\n")
    print(out[:400])
except Exception as e:
    # with_retry 가 3회 재시도하므로 실패 시 10초 안팎 걸린다.
    print(f"실패 ({time.time() - t0:.1f}초):", str(e)[:250])
```

### 다시 배정할 때 지킬 것

1. **judge부터** 정한다. `[1]`에서 4/5 미만이면 그 모델은 judge가 될 수 없다.
2. judge는 alpha·beta 어느 쪽과도 **다른 계열**이어야 한다 (self-preference 편향).
   불가피하게 겹치면 §3-3의 표대로 겹침을 생성 쪽에 둔다.
3. 새로 배정했다면 **파일럿 데이터를 이어서 모으지 않는다.** 모델은 동결 대상이다
   (팀 노션의 동결표). 배정이 바뀌면 그 전 데이터와 섞을 수 없다.

---

## 기록에 대한 주의

노트북은 출력을 지우고 커밋하므로, 위 수치(4/5, 3/5, 346자/1.8초 등)는 **측정 당시 셀 출력에서
옮겨 적은 값**이다. 셀을 다시 돌리면 그대로 재현되는 값이 아니다 (모델 업데이트·티어 변경).
지금 배정이 맞는지 확인하려면 §4를 다시 돌린다.
