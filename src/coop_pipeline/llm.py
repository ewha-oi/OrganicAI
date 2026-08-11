# LLM 호출 공통 래퍼

"""
LLM 호출 공통 래퍼
====================
모델 ID와 JSON 파싱, 재시도 로직을 이 파일 한 곳에서만 관리한다.

이 파일이 필요한 이유:
1. 모델 ID가 tagging.py / scoring.py / agents.py에 흩어져 있으면
   모델이 퇴역했을 때 한 곳만 고치고 나머지를 놓친다.
2. LLM이 JSON을 요구해도 ```json 펜스나 "네, 알겠습니다." 같은 접두문을
   붙이는 일이 흔하다. json.loads()를 직접 부르면 배치 전체가 죽는다.
3. API rate limit / 일시적 5xx는 재시도하면 대부분 통과한다.

모델 ID는 환경변수로 덮어쓸 수 있다 (Colab에서 코드 수정 없이 교체 가능):
    import os
    os.environ["COOP_JUDGE_MODEL"] = "claude-haiku-4-5-20251001"
"""

import json
import os
import re
import time

# ---------------------------------------------------------------------------
# 모델 ID (여기 한 곳에서만 관리)
# ---------------------------------------------------------------------------
# 주의: 제공사가 구형 모델을 예고 없이 퇴역시킨다. 실험 시작 전에 반드시
#       docs/PIPELINE.md의 "모델 ID 확인" 절차를 한 번 돌려볼 것.
MODELS = {
    # 판정자(judge): 태깅 + 채점. 생성 모델과 다른 계열이어야 self-preference 편향이 없다.
    "judge": os.environ.get("COOP_JUDGE_MODEL", "claude-sonnet-5"),
    # alpha 역할 생성 모델 (Google)
    "alpha": os.environ.get("COOP_ALPHA_MODEL", "gemini-2.5-flash"),
    # beta 역할 생성 모델 (Groq)
    "beta": os.environ.get("COOP_BETA_MODEL", "llama-3.3-70b-versatile"),
}

MAX_RETRIES = 3
RETRY_BASE_SLEEP = 2.0  # 초. 재시도마다 배수로 늘어난다.


class LLMCallError(RuntimeError):
    """LLM 호출이 재시도 후에도 실패했거나, 응답을 JSON으로 해석할 수 없을 때."""


# ---------------------------------------------------------------------------
# JSON 파싱
# ---------------------------------------------------------------------------
_FENCE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")
_JSON_OBJECT = re.compile(r"\{.*\}", re.S)


def parse_json_strict(raw: str) -> dict:
    """
    LLM 응답 문자열에서 JSON 객체를 뽑아낸다.
    ```json 펜스, 앞뒤 설명 문장이 붙어 있어도 복구를 시도한다.
    끝내 실패하면 LLMCallError를 던진다 (조용히 기본값을 반환하지 않는다 —
    채점 결과가 조용히 0점이 되는 것이 가장 위험한 실패 모드이기 때문).
    """
    text = _FENCE.sub("", str(raw).strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = _JSON_OBJECT.search(str(raw))
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    raise LLMCallError(f"응답을 JSON으로 해석할 수 없음: {str(raw)[:300]!r}")


# ---------------------------------------------------------------------------
# 재시도
# ---------------------------------------------------------------------------
def with_retry(fn, what: str = "LLM 호출"):
    """fn()을 최대 MAX_RETRIES회 시도한다. 전부 실패하면 LLMCallError."""
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn()
        except Exception as exc:  # API 예외 종류가 SDK마다 달라 광범위하게 잡는다
            last_error = exc
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BASE_SLEEP * attempt)
    raise LLMCallError(f"{what} {MAX_RETRIES}회 모두 실패: {last_error}") from last_error


# ---------------------------------------------------------------------------
# judge 호출 (Anthropic)
# ---------------------------------------------------------------------------
def call_judge_json(client, system: str, user: str, max_tokens: int = 400,
                    model: str = None) -> dict:
    """
    judge 모델을 호출해 JSON 딕셔너리를 받는다.
    temperature=0 고정 — 채점은 재현 가능해야 한다.
    """

    def _once():
        response = client.messages.create(
            model=model or MODELS["judge"],
            max_tokens=max_tokens,
            temperature=0,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return parse_json_strict(response.content[0].text)

    return with_retry(_once, what="judge 호출")
