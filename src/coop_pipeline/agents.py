# gemini/llama API 호출 wrapper
# 시나리오 포맷이 확정돼야 인터페이스를 맞출 수 있으므로 
# 추후에 알맞게 교체할 것

"""
에이전트 실행 wrapper
======================
scenarios/*.json 을 받아 gemini(alpha 역할)/llama(beta 역할)를 실행하고
대화 로그(turns)를 만든다. 이 시점의 로그는 아직 codes/ref, group_score 등이
채워지지 않은 "원본(raw)" 상태 -> 이후 scoring.py, tagging.py를 거쳐야
validate_log()를 통과할 수 있는 완전한 로그가 된다.
"""

import time

import google.generativeai as genai
from groq import Groq

SYSTEM_PROMPT_TEMPLATES = {
    "명시": "너는 {name}다. {partner}와 협력하여 함께 다음의 과제를 해결해야 해.\n\n{task}",
    "묵시": "너는 {name}다. {partner}와 다음의 과제를 해결해야 해.\n\n{task}",
}

MAX_TURNS = 10
GEMINI_MODEL = "gemini-1.5-flash"
LLAMA_MODEL = "llama-3.1-70b-versatile"


def _call_gemini(model, prompt: str) -> str:
    response = model.generate_content(prompt)
    return response.text.strip()


def _call_llama(client, prompt: str) -> str:
    response = client.chat.completions.create(
        model=LLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()


def run_solo(scenario: dict, agent: str, gemini_api_key: str = None, groq_api_key: str = None) -> dict:
    """
    단독 조건 실행. agent는 'gemini' 또는 'llama'.
    task_variants['solo']를 사용한다 (A1은 정보를 합친 버전이어야 함 -> 시나리오 작성 시 주의).
    반환값은 아직 채점되지 않은 원본 산출물이다.
    """
    task_text = scenario["task_variants"]["solo"]
    prompt = f"다음 과제를 혼자 해결하라.\n\n{task_text}\n\n최종 답을 명확하게 제시하라."

    if agent == "gemini":
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel(GEMINI_MODEL)
        output = _call_gemini(model, prompt)
    elif agent == "llama":
        client = Groq(api_key=groq_api_key)
        output = _call_llama(client, prompt)
    else:
        raise ValueError(f"알 수 없는 agent: {agent}")

    return {
        "scenario_id": scenario["scenario_id"],
        "task_type": scenario["task_type"],
        "agent": agent,
        "output": output,
    }


def run_dyad(scenario: dict, condition: str, gemini_api_key: str, groq_api_key: str,
             max_turns: int = MAX_TURNS) -> dict:
    """
    2-agent 대화 실행. alpha=gemini, beta=llama로 고정.
    condition은 '명시' 또는 '묵시'.
    A1이면 task_variants에 'alpha'/'beta' 키로 서로 다른 정보가 들어있어야 하고,
    A2/A4면 'shared' 키 하나로 동일한 과제를 준다.
    """
    genai.configure(api_key=gemini_api_key)
    gemini_model = genai.GenerativeModel(GEMINI_MODEL)
    groq_client = Groq(api_key=groq_api_key)

    variants = scenario["task_variants"]
    alpha_task = variants.get("alpha", variants.get("shared"))
    beta_task = variants.get("beta", variants.get("shared"))

    alpha_system = SYSTEM_PROMPT_TEMPLATES[condition].format(name="alpha", partner="beta", task=alpha_task)
    beta_system = SYSTEM_PROMPT_TEMPLATES[condition].format(name="beta", partner="alpha", task=beta_task)

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
        time.sleep(1)  # API rate limit 여유

    return {
        "scenario_id": scenario["scenario_id"],
        "task_type": scenario["task_type"],
        "condition": condition,
        "turns": turns,
    }