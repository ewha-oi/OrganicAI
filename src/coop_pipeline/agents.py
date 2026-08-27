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

VALID_CONDITIONS = ("명시", "묵시")


class ScenarioError(ValueError):
    """시나리오 JSON에 실행에 필요한 필드가 없을 때."""


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


def _call_gemini(model, prompt: str) -> str:
    def _once():
        response = model.generate_content(prompt)
        return _require_text(response.text, f"alpha({MODELS['alpha']})")
    return with_retry(_once, what="gemini 호출")


def _call_llama(client, prompt: str, model_id: str = None) -> str:
    def _once():
        response = client.chat.completions.create(
            model=model_id or MODELS["beta"],
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
        )
        return _require_text(response.choices[0].message.content,
                             f"beta({model_id or MODELS['beta']})")
    return with_retry(_once, what="llama 호출")


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

    if agent == "gemini":
        genai = _gemini_model()
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel(MODELS["alpha"])
        output = _call_gemini(model, prompt)
        model_id = MODELS["alpha"]
    elif agent == "llama":
        client = _groq_client(groq_api_key)
        output = _call_llama(client, prompt)
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
            text = _call_gemini(gemini_model, prompt)
        else:
            text = _call_llama(groq_client, prompt)

        turns.append({"turn": turn_num, "speaker": speaker, "text": text})
        history += f"{speaker}: {text}\n"
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
        log["final_output"] = _call_gemini(
            gemini_model,
            FINALIZE_PROMPT.format(system=alpha_system, history=history),
        )

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
