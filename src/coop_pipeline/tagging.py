# LLM-as-judge 발화 태깅

"""
LLM 발화 태깅
==============
로그의 각 발화(turn)에 협력 발화 코드(phatic/meta/lead/arch/agree/comp)와
참조 방향(reference)을 LLM-as-judge로 자동 태깅한다.

주의: 실제 실행에는 ANTHROPIC_API_KEY가 필요하다.
      Colab에서는 Secrets 기능으로 등록해서 불러올 것.
"""

import json

import anthropic

from .validate_log import VALID_CODES

CODING_MANUAL = """당신은 대화 로그의 발화 기능을 분류하는 코더입니다.
다음 발화를 보고 아래 코드 중 해당하는 것을 모두 표시하세요:
phatic, meta, lead, arch, agree, comp (해당 없으면 빈 배열)

- phatic: 사교적 발화, 대화 유지용 (예: "알겠어", "그렇구나")
- meta: 협력 과정 자체를 언급 (예: "우리 방향 다시 잡자")
- lead: 방향 제시/정리 (예: "지금 집중할 건 X야")
- arch: 새 정보/주장/아이디어 제시
- agree: 상대 발화를 받아들여 자기 계획 수정
- comp: 상대가 놓친 것을 자발적으로 보완

또한 이 발화가 상대방의 직전 발화를 직접 참조/반영하는지도 표시하세요.
참조가 있으면 상대방 이름을, 없으면 null을 넣으세요.

반드시 아래 JSON 형식으로만 답하세요. 다른 설명은 절대 추가하지 마세요.
{"codes": [...], "reference": "상대방 이름 또는 null"}
"""


def tag_turn(client: anthropic.Anthropic, prev_turns: list, current_turn: dict) -> dict:
    """
    발화 하나를 태깅한다.
    prev_turns: 직전 문맥 (최근 2턴 정도를 리스트로 전달)
    current_turn: 태깅할 대상 turn (dict, 'speaker'와 'text' 포함)
    반환: {"codes": [...], "ref": "상대방 이름 또는 None"}
    """
    context = "\n".join(f"{t['speaker']}: {t['text']}" for t in prev_turns)
    prompt = (
        f"대화 맥락:\n{context}\n\n"
        f"분류할 발화 ({current_turn['speaker']}): {current_turn['text']}"
    )

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=200,
        temperature=0,
        system=CODING_MANUAL,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip()
    parsed = json.loads(raw)

    codes = [c for c in parsed.get("codes", []) if c in VALID_CODES]
    ref = parsed.get("reference")
    if ref in (None, "null", ""):
        ref = None

    return {"codes": codes, "ref": ref}


def tag_log(log: dict, api_key: str, context_window: int = 2) -> dict:
    """
    로그 전체(turns)를 순회하며 각 turn에 codes, ref를 채워넣는다.
    log를 복사하지 않고 그 자리에서 수정 후 반환한다.
    """
    client = anthropic.Anthropic(api_key=api_key)
    turns = log["turns"]

    for i, turn in enumerate(turns):
        prev_turns = turns[max(0, i - context_window):i]
        tag_result = tag_turn(client, prev_turns, turn)
        turn["codes"] = tag_result["codes"]
        turn["ref"] = tag_result["ref"]

    return log


if __name__ == "__main__":
    import os

    with open("example_log.json", encoding="utf-8") as f:
        log = json.load(f)

    tagged = tag_log(log, api_key=os.environ["ANTHROPIC_API_KEY"])
    print(json.dumps(tagged, ensure_ascii=False, indent=2))