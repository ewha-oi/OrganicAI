# gemini/llama API 호출 wrapper

"""
에이전트 실행 wrapper
======================
scenarios/*.json 을 받아 gemini(alpha 역할)/llama(beta 역할)를 실행하고
대화 로그(turns)를 만든다. 이 시점의 로그는 아직 codes/ref, group_score 등이
채워지지 않은 "원본(raw)" 상태 -> 이후 tagging.py, scoring.py를 거쳐야
validate_log()를 통과할 수 있는 완전한 로그가 된다.

모델 ID는 llm.MODELS에서 관리한다. 이 파일에 하드코딩하지 말 것.
"""

import os
import time

from .llm import MODELS, LLMCallError, with_retry

# google-generativeai / groq SDK는 실제로 API를 호출할 때만 필요하다.
# 시나리오 형식 점검(runner.check_scenario_dir)처럼 SDK 없이 돌려야 하는 작업이
# 있으므로, 모듈 최상단이 아니라 아래 헬퍼에서 지연 임포트한다.


def _gemini_model():
    import google.generativeai as genai  # noqa: PLC0415
    return genai


def _groq_client(api_key: str):
    from groq import Groq  # noqa: PLC0415
    return Groq(api_key=api_key)

# 발언 길이 규칙. 명시/묵시 양쪽에 **똑같이** 붙는다 — 조건 간 차이는 협력 지시
# 유무 하나뿐이어야 하므로, 여기에 조건별 차이를 두면 조작이 오염된다.
#
# 이것이 필요한 이유: run_dyad는 매 턴 '지금까지의 대화 전체'를 프롬프트에 넣는다.
# 에이전트가 턴마다 표까지 그린 장문 보고서를 쓰면 이력이 턴당 900토큰 넘게 불어나고,
# 10턴째에는 프롬프트만으로 제공사의 분당 토큰 한도(TPM)를 넘겨 413이 난다.
# 발언을 짧게 하면 턴 수를 줄이지 않고도 한도 안에 들어온다.
#
# 마무리(FINALIZE_PROMPT)에는 적용되지 않는다. final_output이 채점 대상이므로
# 거기서는 형식을 갖춘 긴 산출물이 나와야 한다.
_TURN_STYLE = (
    "\n\n대화 중 각 발언은 5문장 이내의 줄글로 하라. "
    "표·제목·번호 목록 같은 문서 형식은 쓰지 말고, 핵심만 말하라. "
    "(최종 결론을 작성할 때는 이 길이 제한을 적용하지 않는다.)"
)

SYSTEM_PROMPT_TEMPLATES = {
    "명시": "너는 {name}다. {partner}와 협력하여 함께 다음의 과제를 해결해야 해.\n\n{task}"
            + _TURN_STYLE,
    "묵시": "너는 {name}다. {partner}와 다음의 과제를 해결해야 해.\n\n{task}"
            + _TURN_STYLE,
}

# A1은 양쪽이 서로 다른 사실을 가진 정보통합 과제다. 아래 순서를 명시하지 않으면
# '자료를 달라'는 요청만 서로 반복해도 ref가 붙어 협력처럼 보이는 조용한 실패가 난다.
# A2/A4에는 적용하지 않는다. 이 유형들은 같은 지문을 공유하거나, 정보 교환보다
# 의견 조율·공동 생성 자체가 과제이기 때문이다.
_A1_EXCHANGE_PROTOCOL = (
    "\n\n[A1 정보 교환 규칙] 너는 네 지문에 적힌 사실과 ID만 알고 있다. "
    "첫 발화에서 네가 가진 핵심 사실·ID·후보를 구체적으로 공유하라. "
    "상대가 공유한 사실·ID를 다음 발화에서 명시적으로 확인하고 반영한 뒤에만 "
    "최종 결론을 확정하라. 지문 밖의 자료를 반복해서 요구하거나 추측으로 채우지 말고, "
    "두 사람이 이미 공유한 정보만으로 판단하라."
)


def _apply_task_protocol(task: str, task_type: str) -> str:
    """과제 유형별로 필요한 대화 규칙만 덧붙인다."""
    if task_type == "A1":
        return task + _A1_EXCHANGE_PROTOCOL
    return task

# 대화 후 최종 산출물을 뽑아내는 마무리 프롬프트.
# 이것이 없으면 "무엇을 채점할 것인가"가 정의되지 않는다.
FINALIZE_PROMPT = (
    "{system}\n\n지금까지의 대화:\n{history}\n\n"
    "이제 논의를 마치고, 두 사람의 대화를 반영한 최종 결론을 작성하라.\n"
    "상대에 대한 언급이나 대화체 없이, 최종 산출물만 제시하라."
)

MAX_TURNS = 10
TEMPERATURE = 0.7          # 생성 다양성. replicate 간 변동을 만드는 값.
RATE_LIMIT_SLEEP = 1.0     # 턴 사이 대기 (초)

# ---------------------------------------------------------------------------
# 출력 예산 — **용도별로 분리한다. 하나로 묶으면 안 된다.**
# ---------------------------------------------------------------------------
# 대화 턴과 산출물(단독 · 최종 결론)은 성격이 정반대다.
#
#   대화 턴   : 매 턴 이력이 누적돼 프롬프트가 커진다. 예산을 키우면 뒤쪽 턴에서
#               반드시 413이 난다. 게다가 _TURN_STYLE이 5문장으로 묶어 두었으므로
#               1024면 5배 여유다. -> 작게 유지한다.
#   산출물    : **채점 대상**이다. 여기서 잘리면 그 손실이 그대로 점수가 된다.
#               단독은 프롬프트가 작고(과제 지문뿐) 최종 결론은 1회뿐이라
#               예산을 키워도 413 위험이 낮다. -> 넉넉히 준다.
#
# rep=1 수집에서 이 둘을 1024로 묶어 둔 것이 실제로 데이터를 망가뜨렸다:
#   - A4_complex_career_bootcamp 단독 10개가 10개 모두 표 중간에서 잘렸고,
#     judge가 "요구 항목을 빠뜨림"을 이유로 7개에 1점을 줬다. 빠뜨린 것은
#     모델이 아니라 max_tokens였다.
#   - A1_complex_power_outage 명시본이 "7. 기각된 가설" 제목에서 잘려
#     체크리스트 3항목(D14 · 해킹 · 번개)을 통째로 잃고 0.857 -> 0.429가 됐다.
#   - 단독 산출물이 그룹 산출물보다 길어서 **단독만 골라 깎였다.** 즉 오차가
#     "협력 이득이 있다" 쪽으로 편향된다 — 이 연구가 검증하려는 그 방향이다.
#
# _TURN_STYLE의 "(최종 결론을 작성할 때는 이 길이 제한을 적용하지 않는다)"는
# 지금부터 예산으로도 실제로 성립한다.
TURN_MAX_TOKENS = int(os.environ.get("COOP_TURN_MAX_TOKENS", 1024))

# beta도 대화 이력과 태깅 프롬프트에 그대로 들어가므로, SDK의 큰 기본 출력
# 한도에 맡기지 않는다. gpt-oss-20b의 low 추론 + 1024는 기존 실측을 보존하면서
# 장문 한 발화가 judge의 TPM을 넘기는 경로를 막는다.
BETA_MAX_TOKENS = int(os.environ.get("COOP_BETA_MAX_TOKENS", TURN_MAX_TOKENS))

# 산출물 예산. (1차, 잘렸을 때 올려서 재시도할 2차) 순서다.
# 실측 기준 한글 1024토큰 ~= 1,750자였고, 잘린 단독 산출물이 1,700~2,000자에
# 몰려 있었다. 3072면 ~5,200자까지 담긴다 (관측된 최장 산출물의 2.6배).
SOLO_MAX_TOKENS = (int(os.environ.get("COOP_SOLO_MAX_TOKENS", 3072)),
                   int(os.environ.get("COOP_SOLO_RETRY_TOKENS", 4096)))

# 마무리는 대화 10턴 전체가 프롬프트에 들어간다(~2,500토큰). 예산까지 더한 값이
# TPM 8,000 안에 있어야 하므로 단독보다 보수적으로 잡는다.
#   2560 -> 프롬프트 2,500 + 2,560 = 5,060   여유 있음
#   3584 -> 프롬프트 2,500 + 3,584 = 6,084   재시도 1회분까지 한도 안
FINALIZE_MAX_TOKENS = (int(os.environ.get("COOP_FINALIZE_MAX_TOKENS", 2560)),
                       int(os.environ.get("COOP_FINALIZE_RETRY_TOKENS", 3584)))

VALID_CONDITIONS = ("명시", "묵시")


class ScenarioError(ValueError):
    """시나리오 JSON에 실행에 필요한 필드가 없을 때."""


class TruncatedOutputError(LLMCallError):
    """채점 대상 산출물이 max_tokens에 걸려 중간에서 끊겼을 때.

    조용히 넘기면 안 되는 이유: 잘린 산출물은 '내용이 부실한 산출물'과 겉모습이
    같다. A1은 뒤쪽 체크리스트 항목을 잃고, A2/A4는 judge에게 '요구 항목 누락'
    판정을 받는다. 둘 다 실험 결과처럼 보이지만 실제로는 설정 오류다.
    """


# ---------------------------------------------------------------------------
# 저수준 호출
# ---------------------------------------------------------------------------
def _require_text(raw, what: str) -> str:
    """
    생성 결과가 비어 있으면 즉시 실패시킨다.

    빈 발화를 그대로 로그에 남기면 '한쪽이 한 마디도 안 했다'가 되는데,
    이는 '대화가 겉돌았다'는 **실험 결과**와 구별되지 않는다 (그대로 L0/L1이 된다).
    추론 모델이 max_tokens를 추론 토큰에 다 쓰면 실제로 이런 응답이 나온다.
    """
    text = (raw or "").strip()
    if not text:
        raise LLMCallError(
            f"{what}이 빈 응답을 반환했다 — 추론 모델이면 max_tokens가 추론에 "
            f"모두 소모됐을 수 있다. 빈 발화는 로그에 남기지 않는다"
        )
    return text


# 제공사마다 '출력 한도에 걸려 끊겼다'를 다르게 표기한다. 한 곳에서만 판별한다.
#   groq/OpenAI 호환 : finish_reason == "length"
#   google-genai     : candidates[0].finish_reason == FinishReason.MAX_TOKENS
_TRUNCATION_MARKS = {"length", "max_tokens", "maxtokens"}


def _finish_reason(response):
    """응답 객체에서 종료 사유를 꺼낸다. 없으면 None (판정 불가)."""
    reason = getattr(response, "finish_reason", None)
    if reason is None:
        choices = getattr(response, "choices", None) or ()
        if choices:
            reason = getattr(choices[0], "finish_reason", None)
    if reason is None:
        candidates = getattr(response, "candidates", None) or ()
        if candidates:
            reason = getattr(candidates[0], "finish_reason", None)
    return reason


def _is_truncated(response) -> bool:
    """출력 한도에 걸려 끊긴 응답인가. 사유를 알 수 없으면 False(=끊기지 않음)."""
    reason = _finish_reason(response)
    if reason is None:
        return False
    # Enum이면 이름을, 문자열이면 그대로 본다.
    name = getattr(reason, "name", reason)
    return str(name).strip().lower().replace("-", "_") in _TRUNCATION_MARKS


class Reply:
    """생성 결과 한 건. text와 '끊겼는가'를 함께 들고 다닌다.

    두 값을 붙여 두는 이유: 호출부가 text만 받으면 잘린 문자열과 짧은 문자열을
    구별할 방법이 없다. rep=1 수집이 정확히 그렇게 망가졌다.
    """

    __slots__ = ("text", "truncated")

    def __init__(self, text: str, truncated: bool = False):
        self.text = text
        self.truncated = truncated

    def __bool__(self):
        return bool(self.text)

    def __str__(self):
        return self.text


def _call_gemini(model, prompt: str, max_tokens: int = None) -> Reply:
    """alpha 호출. max_tokens를 주면 그 예산으로 요청한다.

    노트북의 alpha 대체 어댑터(shim)도 generation_config를 받아야 한다.
    받지 못하면 예산이 무시되고 조용히 잘리므로, 여기서 즉시 실패시킨다.
    """
    budget = TURN_MAX_TOKENS if max_tokens is None else max_tokens

    def _once():
        try:
            response = model.generate_content(
                prompt,
                generation_config={"temperature": TEMPERATURE,
                                   "max_output_tokens": budget},
            )
        except TypeError as exc:
            raise LLMCallError(
                "alpha 어댑터가 generation_config를 받지 않는다 — 노트북의 "
                "'alpha 대체 어댑터' 셀이 옛 버전이다.\n"
                "  git pull -> 런타임 재시작 -> 1절부터 다시 실행할 것.\n"
                "  (이 셀이 낡으면 출력 예산이 무시돼 산출물이 조용히 잘린다)"
            ) from exc
        text = _require_text(getattr(response, "text", None),
                             f"alpha({MODELS['alpha']})")
        return Reply(text, _is_truncated(response))

    return with_retry(_once, what="gemini 호출")


def _call_llama(client, prompt: str, model_id: str = None,
                max_tokens: int = None) -> Reply:
    selected_model = model_id or MODELS["beta"]
    # 모듈 전역을 호출 시점에 읽는다 (테스트가 monkeypatch로 바꿀 수 있어야 한다).
    budget = BETA_MAX_TOKENS if max_tokens is None else max_tokens

    def _once():
        kwargs = dict(
            model=selected_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
            max_tokens=budget,
        )
        if "gpt-oss" in selected_model:
            kwargs["reasoning_effort"] = "low"
        response = client.chat.completions.create(**kwargs)
        text = _require_text(response.choices[0].message.content,
                             f"beta({selected_model})")
        return Reply(text, _is_truncated(response))
    return with_retry(_once, what="llama 호출")


def _generate_scoring_target(call, budgets, what: str) -> str:
    """채점 대상(단독 산출물 · 최종 결론)을 생성한다. 잘리면 예산을 올려 재생성한다.

    call    : budget(int)을 받아 Reply를 돌려주는 함수
    budgets : 시도할 예산 순서. 마지막 예산에서도 잘리면 예외를 던진다.

    끝까지 잘리면 **조용히 통과시키지 않는다.** 잘린 산출물이 채점되면 그 결과는
    '협력 이득'과 구별되지 않는 가짜 신호가 되고, 저장된 뒤에는 되돌릴 수 없다.
    수집 루프는 이 예외를 실패 1건으로 세고 다음 시나리오로 넘어간다.
    """
    for index, budget in enumerate(budgets):
        reply = call(budget)
        if not reply.truncated:
            return reply.text
        if index + 1 < len(budgets):
            print(f"  ! {what}이 잘렸다 ({len(reply.text)}자, max_tokens={budget}) "
                  f"— {budgets[index + 1]}로 올려 재생성")
        else:
            raise TruncatedOutputError(
                f"{what}이 max_tokens={budget}에서도 잘렸다 ({len(reply.text)}자).\n"
                f"  이 텍스트는 채점 대상이므로 잘린 채로 저장하지 않는다.\n"
                f"  - 예산을 더 올리려면: COOP_SOLO_MAX_TOKENS / "
                f"COOP_FINALIZE_MAX_TOKENS\n"
                f"  - 413(TPM 초과)이 함께 떴다면 예산을 올릴 수 없는 상태다. "
                f"alpha를 TPM이 큰 모델로 재배정할 것 (docs/MODEL_ASSIGNMENT.md §4)"
            )


def _resolve_tasks(scenario: dict) -> tuple:
    """
    시나리오에서 alpha/beta에게 줄 지문을 꺼낸다.

    - alpha/beta가 따로 있으면 그것을 쓴다 (A1의 정보 비대칭, A4의 제약 비대칭).
    - 없으면 shared 하나를 양쪽에 똑같이 준다 (현재 A2/A4 시나리오 방식).
    둘 다 없으면 실행할 수 없으므로 명확한 에러를 낸다.
    """
    variants = scenario.get("task_variants")
    if not isinstance(variants, dict):
        raise ScenarioError(f"'{scenario.get('scenario_id')}'에 task_variants가 없음")

    alpha_task = variants.get("alpha") or variants.get("shared")
    beta_task = variants.get("beta") or variants.get("shared")
    if not alpha_task or not beta_task:
        raise ScenarioError(
            f"'{scenario.get('scenario_id')}'의 task_variants에 "
            f"(alpha, beta) 또는 shared가 필요함. 현재 키: {sorted(variants)}"
        )
    return alpha_task, beta_task


# ---------------------------------------------------------------------------
# 단독 조건
# ---------------------------------------------------------------------------
def run_solo(scenario: dict, agent: str, replicate: int = 1,
             gemini_api_key: str = None, groq_api_key: str = None) -> dict:
    """
    단독 조건 실행. agent는 'gemini' 또는 'llama'.
    task_variants['solo']를 사용한다 (A1은 정보를 합친 버전이어야 함 -> 시나리오 작성 시 주의).
    반환값은 아직 채점되지 않은 원본 산출물이다.
    """
    variants = scenario.get("task_variants", {})
    if "solo" not in variants:
        raise ScenarioError(f"'{scenario.get('scenario_id')}'에 task_variants.solo가 없음")

    task_text = variants["solo"]
    prompt = f"다음 과제를 혼자 해결하라.\n\n{task_text}\n\n최종 답을 명확하게 제시하라."

    # 단독 산출물은 채점 대상이다. 대화 턴 예산(1024)이 아니라 산출물 예산을 쓰고,
    # 잘리면 올려서 다시 만든다. 프롬프트가 과제 지문뿐이라 413 위험이 낮다.
    what = f"단독 산출물({scenario['scenario_id']}/{agent})"

    if agent == "gemini":
        genai = _gemini_model()
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel(MODELS["alpha"])
        output = _generate_scoring_target(
            lambda budget: _call_gemini(model, prompt, max_tokens=budget),
            SOLO_MAX_TOKENS, what)
        model_id = MODELS["alpha"]
    elif agent == "llama":
        client = _groq_client(groq_api_key)
        output = _generate_scoring_target(
            lambda budget: _call_llama(client, prompt, max_tokens=budget),
            SOLO_MAX_TOKENS, what)
        model_id = MODELS["beta"]
    else:
        raise ValueError(f"알 수 없는 agent: {agent} ('gemini' 또는 'llama')")

    return {
        "scenario_id": scenario["scenario_id"],
        "task_type": scenario["task_type"],
        "condition": "solo",
        "agent": agent,
        "model": model_id,
        "replicate": replicate,
        "output": output,
    }


# ---------------------------------------------------------------------------
# 2-agent 조건
# ---------------------------------------------------------------------------
def run_dyad(scenario: dict, condition: str, gemini_api_key: str, groq_api_key: str,
             replicate: int = 1, max_turns: int = MAX_TURNS,
             finalize: bool = True) -> dict:
    """
    2-agent 대화 실행. alpha=gemini, beta=llama로 고정.
    condition은 '명시' 또는 '묵시'.

    A1이면 task_variants에 'alpha'/'beta' 키로 서로 다른 정보가 들어있어야 하고,
    A2/A4면 'shared' 키 하나로 동일한 과제를 준다.
    (A4를 비대칭 제약으로 재설계하면 alpha/beta 키를 넣기만 하면 되고 코드는 그대로다.)

    finalize=True면 대화 후 alpha에게 최종 결론을 한 번 더 요청해
    log['final_output']에 넣는다. 채점(scoring)은 이 필드를 대상으로 한다.
    """
    if condition not in VALID_CONDITIONS:
        raise ValueError(f"condition은 {VALID_CONDITIONS} 중 하나여야 함: '{condition}'")

    genai = _gemini_model()
    genai.configure(api_key=gemini_api_key)
    gemini_model = genai.GenerativeModel(MODELS["alpha"])
    groq_client = _groq_client(groq_api_key)

    alpha_task, beta_task = _resolve_tasks(scenario)
    alpha_task = _apply_task_protocol(alpha_task, scenario["task_type"])
    beta_task = _apply_task_protocol(beta_task, scenario["task_type"])
    alpha_system = SYSTEM_PROMPT_TEMPLATES[condition].format(
        name="alpha", partner="beta", task=alpha_task)
    beta_system = SYSTEM_PROMPT_TEMPLATES[condition].format(
        name="beta", partner="alpha", task=beta_task)

    turns = []
    history = ""
    speaker_order = ["alpha", "beta"]

    for turn_num in range(1, max_turns + 1):
        speaker = speaker_order[(turn_num - 1) % 2]
        system = alpha_system if speaker == "alpha" else beta_system
        prompt = f"{system}\n\n지금까지의 대화:\n{history}\n\n너의 다음 발언:"

        if speaker == "alpha":
            reply = _call_gemini(gemini_model, prompt, max_tokens=TURN_MAX_TOKENS)
        else:
            reply = _call_llama(groq_client, prompt, max_tokens=BETA_MAX_TOKENS)

        # 턴이 잘린 것은 실패로 세지 않는다 — _TURN_STYLE이 5문장으로 묶어 두었으므로
        # 1024는 5배 여유이고, 여기서 잘렸다면 그 규칙이 안 먹었다는 신호다.
        # 다만 조용히 넘기면 원인을 못 찾으므로 로그와 화면 양쪽에 남긴다.
        turn = {"turn": turn_num, "speaker": speaker, "text": reply.text}
        if reply.truncated:
            turn["truncated"] = True
            print(f"  ! {turn_num}턴({speaker}) 발화가 잘렸다 ({len(reply.text)}자) "
                  f"— 발언 길이 규칙이 안 먹은 것이다 (SYSTEM_PROMPT_TEMPLATES 확인)")

        turns.append(turn)
        history += f"{speaker}: {reply.text}\n"
        time.sleep(RATE_LIMIT_SLEEP)  # API rate limit 여유

    log = {
        "scenario_id": scenario["scenario_id"],
        "task_type": scenario["task_type"],
        "condition": condition,
        "replicate": replicate,
        "max_turns": max_turns,
        "models": {"alpha": MODELS["alpha"], "beta": MODELS["beta"]},
        "turns": turns,
    }

    if finalize:
        # 그룹 산출물 = 채점 대상. 턴 예산이 아니라 산출물 예산을 쓴다.
        finalize_prompt = FINALIZE_PROMPT.format(system=alpha_system, history=history)
        log["final_output"] = _generate_scoring_target(
            lambda budget: _call_gemini(gemini_model, finalize_prompt,
                                        max_tokens=budget),
            FINALIZE_MAX_TOKENS,
            f"최종 결론({scenario['scenario_id']}/{condition})")

    return log


def group_output_text(log: dict) -> str:
    """
    채점 대상이 되는 '그룹 산출물' 텍스트를 로그에서 꺼낸다.

    규칙 (docs/RUBRIC.md와 동일):
      1) final_output이 있으면 그것을 쓴다.
      2) 없으면 마지막 두 턴(양쪽 화자 각 1턴)을 이어 붙인 것을 쓴다.
    무엇을 채점했는지가 결과 해석을 좌우하므로 이 규칙을 바꾸면 문서도 함께 고칠 것.
    """
    final = log.get("final_output")
    if final:
        return final
    turns = log.get("turns", [])
    return "\n".join(t["text"] for t in turns[-2:])
