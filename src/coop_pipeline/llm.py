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


class LLMCallError(RuntimeError):
    """LLM 호출이 재시도 후에도 실패했거나, 응답을 JSON으로 해석할 수 없을 때."""


# ---------------------------------------------------------------------------
# judge 제공사 선택
# ---------------------------------------------------------------------------
# judge(태깅 + 채점)는 세 제공사 중 하나로 돌릴 수 있다.
#     anthropic : 유료. 기본값.
#     groq      : 무료 티어. 비용 0으로 파일럿을 돌릴 때.
#     gemini    : 무료 티어. **주의 — 아래 편향 경고를 읽을 것.**
#
# 환경변수로 바꾼다 (임포트 전에 설정해야 반영된다):
#     os.environ["COOP_JUDGE_PROVIDER"] = "groq"
#     os.environ["COOP_JUDGE_MODEL"]    = "openai/gpt-oss-120b"
#
# ── self-preference 편향 경고 ──────────────────────────────────────────────
# judge는 **생성 모델과 다른 계열**이어야 한다. 지금 설계에서 alpha=Gemini가
# 최종 그룹 산출물을 작성하므로(agents.FINALIZE_PROMPT), judge까지 Gemini면
# 자기 계열의 글을 자기가 채점하게 된다. group_grade가 부풀고 -> Q3 통과율이
# 오르고 -> L3/L4가 실제보다 많이 나온다. 그게 이 연구의 핵심 결과라서
# 편향이 결론을 그대로 오염시킨다.
#
# 무료로 가려면 judge를 Groq의 **Llama가 아닌** 계열에 두는 것이 안전하다
# (beta=Llama이므로 Llama도 피해야 한다). 즉 judge != alpha 계열 != beta 계열.
# ---------------------------------------------------------------------------
VALID_JUDGE_PROVIDERS = ("anthropic", "groq", "gemini")

JUDGE_PROVIDER = os.environ.get("COOP_JUDGE_PROVIDER", "anthropic").strip().lower()

# 제공사별 judge 기본 모델. COOP_JUDGE_MODEL이 있으면 그것이 우선한다.
DEFAULT_JUDGE_MODELS = {
    "anthropic": "claude-sonnet-5",
    "groq": "openai/gpt-oss-120b",      # Llama가 아닌 계열 (beta와 겹치지 않게)
    "gemini": "gemini-2.5-flash",       # alpha와 같은 계열 - 편향 주의
}

# api_keys 딕셔너리에서 judge가 꺼내 쓸 키 이름
JUDGE_KEY_NAME = {"anthropic": "anthropic", "groq": "groq", "gemini": "gemini"}


def judge_provider(provider: str = None) -> str:
    """judge 제공사 이름을 확정한다. 알 수 없는 값이면 바로 에러."""
    name = (provider or JUDGE_PROVIDER).strip().lower()
    if name not in VALID_JUDGE_PROVIDERS:
        raise LLMCallError(
            f"COOP_JUDGE_PROVIDER는 {VALID_JUDGE_PROVIDERS} 중 하나여야 함: '{name}'"
        )
    return name


def judge_model(provider: str = None) -> str:
    """해당 제공사에서 쓸 judge 모델 ID."""
    return os.environ.get("COOP_JUDGE_MODEL") or DEFAULT_JUDGE_MODELS[judge_provider(provider)]


def judge_key_name(provider: str = None) -> str:
    """api_keys 딕셔너리에서 judge가 꺼내 쓸 키 이름 ('anthropic'/'groq'/'gemini')."""
    return JUDGE_KEY_NAME[judge_provider(provider)]


# ---------------------------------------------------------------------------
# 모델 ID (여기 한 곳에서만 관리)
# ---------------------------------------------------------------------------
# 주의: 제공사가 구형 모델을 예고 없이 퇴역시킨다. 실험 시작 전에 반드시
#       docs/PIPELINE.md의 "모델 ID 확인" 절차를 한 번 돌려볼 것.
MODELS = {
    # 판정자(judge): 태깅 + 채점. 생성 모델과 다른 계열이어야 self-preference 편향이 없다.
    "judge": judge_model(),
    # alpha 역할 생성 모델 (Google)
    "alpha": os.environ.get("COOP_ALPHA_MODEL", "gemini-2.5-flash"),
    # beta 역할 생성 모델 (Groq)
    "beta": os.environ.get("COOP_BETA_MODEL", "llama-3.3-70b-versatile"),
}

MAX_RETRIES = 3
RETRY_BASE_SLEEP = 2.0  # 초. 재시도마다 배수로 늘어난다.


def _is_request_too_large(exc: Exception) -> bool:
    """제공사 SDK마다 다른 413 표기를 한 곳에서 판별한다."""
    current = exc
    # 호출부가 원인을 설명하기 위해 LLMCallError로 감싼 경우에도 413을 놓치지
    # 않아야 같은 큰 요청을 다시 보내지 않는다.
    while current is not None:
        if getattr(current, "status_code", None) == 413:
            return True
        current = current.__cause__ or current.__context__
    return False


def _is_daily_token_limit(exc: Exception) -> bool:
    """Groq의 TPD 429은 짧은 재시도로 회복되지 않는다."""
    return "tokens per day" in str(exc).lower()


def _is_json_mode_failure(exc: Exception) -> bool:
    """
    Groq의 response_format={"type":"json_object"}가 유효한 JSON을 받지 못했을 때의 400.

    다른 제공사라면 잘린 문자열이 그대로 와서 parse_json_strict가 "응답을 JSON으로
    해석할 수 없음"을 냈을 상황인데, Groq는 그 전에 400으로 막는다. 그래서
    _MIN_OUTPUT_TOKENS 주석이 적어 둔 '되돌림 신호'가 이 얼굴로 나타난다.

    413과 같은 이유로 원인 체인을 따라간다 — _call_groq이 실패한 출력 원문을 붙여
    LLMCallError로 감싸는데, 겉면만 보면 with_retry가 못 알아보고 3회 재시도한다.
    """
    current = exc
    while current is not None:
        if "failed to generate json" in str(current).lower():
            return True
        current = current.__cause__ or current.__context__
    return False


def _failed_generation(exc: Exception) -> str:
    """
    Groq가 400 본문에 함께 주는 실패한 출력 원문. 원인 판별에 이것만 있으면 된다.
    JSON이 중간에서 끊겼으면 절단, 앞에 설명이나 <think>가 붙었으면 군말이다.
    """
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            return str(error.get("failed_generation", ""))
    return ""


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
    """fn()을 최대 MAX_RETRIES회 시도한다. 413은 같은 요청을 반복하지 않는다."""
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn()
        except Exception as exc:  # API 예외 종류가 SDK마다 달라 광범위하게 잡는다
            last_error = exc
            # 413은 현재 요청 자체가 한도를 넘었다는 뜻이다. 대기하거나 그대로
            # 재시도해도 성공하지 않으므로, 호출부가 예산을 낮추지 않는 한 즉시 멈춘다.
            if _is_request_too_large(exc):
                raise LLMCallError(f"{what}: 요청이 모델 한도를 넘음: {exc}") from exc
            if _is_daily_token_limit(exc):
                raise LLMCallError(f"{what}: 일일 토큰 한도(TPD)에 도달: {exc}") from exc
            # temperature=0이라 같은 입력은 같은 출력을 낸다. _call_groq이 예산을
            # 올리고 JSON 모드까지 껐는데도 여기 왔다면 재시도는 실패를 3배로
            # 늘릴 뿐이다 (그만큼 TPD도 나간다).
            if _is_json_mode_failure(exc):
                raise LLMCallError(f"{what}: {exc}") from exc
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BASE_SLEEP * attempt)
    raise LLMCallError(f"{what} {MAX_RETRIES}회 모두 실패: {last_error}") from last_error


# ---------------------------------------------------------------------------
# judge 호출 (제공사 무관)
# ---------------------------------------------------------------------------
# tagging.py / scoring.py는 아래 JudgeClient만 쓴다. 제공사별 차이(SDK 인터페이스,
# system 프롬프트 전달 방식, JSON 강제 옵션)는 전부 이 파일 안에 가둔다.
# temperature=0 고정 — 채점은 재현 가능해야 한다.

# 추론(reasoning) 모델은 추론 토큰이 출력 한도를 같이 소모한다. 태깅 프롬프트 기준
# max_tokens=400~500을 그대로 주면 추론에 전부 쓰고 본문이 비거나 JSON이 중간에
# 잘려 파싱이 실패한다. 제공사를 가리지 않는 함정이므로 하한을 공통으로 둔다
# (Gemini 2.5 계열, Groq의 gpt-oss/qwen3 계열에서 모두 확인됨).
#
# 값을 4096 -> 512로 낮췄다. 제공사는 max_tokens를 '예약량'으로 일일 한도(TPD)에
# 미리 청구하므로, 실제로는 수십 토큰짜리 JSON을 받으면서 4096을 매번 다 낸다.
# judge 호출 1회가 5,200토큰씩 나가 TPD 200,000이 시나리오 한 개에 소진됐다.
# 지금 judge는 reasoning_effort를 none/low로 보내므로 추론 토큰이 거의 없고,
# qwen은 reasoning_effort=none으로 호출하므로 태깅/채점의 짧은 JSON에는 512면 충분하다.
#
# 되돌려야 하는 신호: "응답을 JSON으로 해석할 수 없음" 또는 빈 응답 에러가 반복되면
# 출력이 중간에 잘린 것이다. 그때는 이 값을 올린다 (조용히 틀리지 않고 에러로 드러난다).
_MIN_OUTPUT_TOKENS = int(os.environ.get("COOP_JUDGE_MAX_TOKENS", 512))

# JSON 태깅/채점은 매우 짧은 응답만 필요하다. 요청 전체가 TPM 한도를 넘을 때만
# 512 -> 256으로 한 번 낮춘다. 정상 요청의 출력 품질과 기존 실험 설정은 유지한다.
_MIN_GROQ_JUDGE_TOKENS = 256

# JSON 모드가 실패했을 때만 한 번 올려 보는 예산. 하한 자체를 올리지 않는 이유는
# 태깅 호출이 시나리오당 20회씩 나가서, 전부 두 배로 예약하면 TPD가 다시 빠듯해지기
# 때문이다. 긴 발화 몇 개만 이 비용을 내면 된다.
_GROQ_JSON_RETRY_TOKENS = 1024

# 호출 종류별 출력 하한.
#
# 예약량은 실제 사용량과 무관하게 TPD에 전액 청구된다. 그래서 '이 응답이 실제로
# 얼마나 긴가'에 맞춰 하한을 나눠야 한다. 재는 대상은 그대로 두고 낭비만 줄인다.
#
#   태깅  : {"codes":[...], "reference":..., "evidence":{"코드":"원문 인용"}}
#           evidence가 원문을 그대로 인용하므로 길다. 512 유지.
#   채점  : {"grade": 4, "reason": "..."}                    실측 60~120토큰
#   신규성: {"new_idea": true, "reason": "..."}               실측 60~120토큰
#
# 뒤 둘을 256으로 낮추면 건당 judge 호출 14회 x 256토큰이 절약된다.
# 낮춰서 잘리면 조용히 틀리지 않고 "응답을 JSON으로 해석할 수 없음"으로 드러나므로,
# 그 에러가 반복되면 이 값을 되돌린다.
JUDGE_TAGGING_MIN_TOKENS = _MIN_OUTPUT_TOKENS
JUDGE_SHORT_JSON_MIN_TOKENS = int(os.environ.get("COOP_JUDGE_SHORT_MAX_TOKENS", 256))

# reasoning_effort를 받는 Groq 모델과, 그 모델이 받아들이는 값.
# **값이 계열마다 다르다.** 틀린 값을 보내면 400이므로 하나로 뭉뚱그릴 수 없다.
#     gpt-oss : low / medium / high
#     qwen3   : none / default      ("low"를 보내면 400 — 2026-08 확인)
# 목록에 없는 모델에는 아예 보내지 않는다. 안 보내도 위의 토큰 하한만으로 동작은 한다 —
# 이건 추론량을 줄여 지연시간과 토큰을 아끼는 최적화다.
_GROQ_REASONING_EFFORT = {"gpt-oss": "low", "qwen3": "none"}


class JudgeClient:
    """
    judge 제공사 하나를 감싼 얇은 어댑터.

        judge = make_judge(api_key)          # 제공사는 COOP_JUDGE_PROVIDER를 따름
        parsed = judge.json(system, user)    # dict를 돌려준다
    """

    def __init__(self, provider: str, client, model: str):
        self.provider = provider
        self.client = client
        self.model = model

    def __repr__(self):
        return f"<JudgeClient {self.provider}:{self.model}>"

    def json(self, system: str, user: str, max_tokens: int = 400,
             model: str = None, min_output_tokens: int = None) -> dict:
        """
        min_output_tokens: 이 호출 종류의 출력 하한. 짧은 JSON만 받는 호출
            (채점 · 신규성)은 256을 주면 예약 낭비를 줄인다. 생략하면 512.
        """
        model_id = model or self.model
        floor = _MIN_OUTPUT_TOKENS if min_output_tokens is None else min_output_tokens
        caller = {
            "anthropic": self._call_anthropic,
            "groq": self._call_groq,
            "gemini": self._call_gemini,
        }[self.provider]

        def _once():
            return parse_json_strict(
                caller(system, user, max_tokens, model_id, floor))

        output_budget = max(max_tokens, floor)
        details = (
            f"judge 호출({self.provider}:{model_id}; system={len(system)}자, "
            f"user={len(user)}자, output={output_budget})"
        )
        return with_retry(_once, what=details)

    # -- 제공사별 구현 -------------------------------------------------------
    def _call_anthropic(self, system, user, max_tokens, model_id, floor):
        response = self.client.messages.create(
            model=model_id, max_tokens=max(max_tokens, floor), temperature=0,
            system=system, messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text

    def _call_groq(self, system, user, max_tokens, model_id, floor):
        # OpenAI 호환 인터페이스. response_format으로 JSON을 강제한다
        # (프롬프트에 'JSON'이라는 단어가 있어야 이 옵션이 동작한다 —
        #  CODING_MANUAL / JUDGE_RUBRIC 모두 조건을 만족한다).
        budget = max(max_tokens, floor)
        min_budget = min(budget, _MIN_GROQ_JUDGE_TOKENS)
        json_mode = True
        # 예산 상향은 한 번뿐이다. 올린 예산이 413을 맞으면 위 분기가 다시 예산을
        # 낮추는데, 그때 또 올리면 두 분기가 서로를 되살려 루프가 끝나지 않는다.
        budget_raised = False
        while True:
            kwargs = dict(
                model=model_id, max_tokens=budget,
                temperature=0,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
            )
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            for tag, effort in _GROQ_REASONING_EFFORT.items():
                if tag in model_id:
                    kwargs["reasoning_effort"] = effort
                    break
            try:
                response = self.client.chat.completions.create(**kwargs)
                break
            except Exception as exc:
                if _is_request_too_large(exc) and budget > min_budget:
                    budget = min_budget
                    continue
                if _is_request_too_large(exc):
                    raise LLMCallError(
                        f"groq judge({model_id}) 요청이 여전히 너무 큼 "
                        f"(system={len(system)}자, user={len(user)}자, "
                        f"max_tokens={budget})"
                    ) from exc
                # JSON 모드 400. 원인이 둘인데 이 자리에서는 구별할 수 없으므로
                # 둘 다에 순서대로 대응한다.
                #   (1) 출력이 예산에서 끊긴 경우  -> 예산을 한 번 올린다
                #   (2) 모델이 JSON 앞뒤에 군말을 붙인 경우 -> JSON 모드를 끄면
                #       그 문자열이 그대로 오고, parse_json_strict가 펜스·설명이
                #       붙은 응답에서 객체를 뽑아낸다
                if _is_json_mode_failure(exc):
                    if not budget_raised and budget < _GROQ_JSON_RETRY_TOKENS:
                        budget = _GROQ_JSON_RETRY_TOKENS
                        budget_raised = True
                        continue
                    if json_mode:
                        json_mode = False
                        continue
                    # 여기까지 왔으면 예산 문제도 형식 문제도 아니다. 실패한 출력
                    # 원문을 메시지 앞쪽에 실어 둔다 — 노트북 수집 루프가 에러를
                    # 잘라서 찍으므로 뒤에 붙이면 잘려 나간다.
                    raise LLMCallError(
                        f"groq judge({model_id}) JSON 생성 실패 "
                        f"[{_failed_generation(exc)[:200]}] "
                        f"(max_tokens={budget}, json_mode 해제 후에도 실패)"
                    ) from exc
                raise

        content = response.choices[0].message.content
        if not content:
            # 추론만 하고 본문을 못 낸 경우. 조용히 넘기면 파싱 단계에서
            # 엉뚱한 에러가 나므로 여기서 원인을 밝혀 둔다.
            raise LLMCallError(
                f"groq judge({model_id})가 빈 응답을 반환했다 — "
                f"max_tokens={kwargs['max_tokens']}가 추론 토큰에 모두 소모된 것으로 보인다"
            )
        return content

    def _call_gemini(self, system, user, max_tokens, model_id, floor):
        model = self.client.GenerativeModel(model_id, system_instruction=system)
        response = model.generate_content(
            user,
            generation_config={
                "temperature": 0,
                "max_output_tokens": max(max_tokens, floor),
                "response_mime_type": "application/json",
            },
        )
        return response.text


def make_judge(api_key: str, provider: str = None, model: str = None) -> JudgeClient:
    """
    judge 클라이언트를 만든다. SDK는 실제로 쓸 제공사의 것만 임포트한다
    (anthropic을 안 깔아도 groq judge로 돌릴 수 있어야 한다).
    """
    name = judge_provider(provider)
    model_id = model or judge_model(name)

    if not api_key:
        raise LLMCallError(
            f"judge 제공사가 '{name}'인데 해당 API 키가 비어 있음 — "
            f"api_keys['{judge_key_name(name)}']를 확인할 것"
        )

    if name == "anthropic":
        import anthropic  # noqa: PLC0415
        return JudgeClient(name, anthropic.Anthropic(api_key=api_key), model_id)

    if name == "groq":
        from groq import Groq  # noqa: PLC0415
        return JudgeClient(name, Groq(api_key=api_key), model_id)

    import google.generativeai as genai  # noqa: PLC0415
    genai.configure(api_key=api_key)
    return JudgeClient(name, genai, model_id)


def call_judge_json(judge: JudgeClient, system: str, user: str,
                    max_tokens: int = 400, model: str = None,
                    min_output_tokens: int = None) -> dict:
    """JudgeClient.json()의 함수형 별칭 (기존 호출부 호환용)."""
    return judge.json(system, user, max_tokens=max_tokens, model=model,
                      min_output_tokens=min_output_tokens)
