# LLM-as-judge 발화 태깅

"""
LLM 발화 태깅
==============
로그의 각 발화(turn)에 협력 발화 코드(phatic/meta/lead/arch/agree/comp)와
참조 방향(ref)을 LLM-as-judge로 자동 태깅한다.

코딩 매뉴얼의 전체 정의와 경계 사례는 docs/CODING_MANUAL.md에 있다.
아래 CODING_MANUAL 문자열은 그 문서를 LLM 코더용으로 압축한 것이므로,
기준을 바꿀 때는 문서와 이 문자열을 **함께** 고쳐야 한다.
(둘이 어긋나면 인간 코더와 LLM 코더의 Kappa가 낮게 나오는데 원인을 못 찾는다.)

주의: 실제 실행에는 ANTHROPIC_API_KEY가 필요하다.
      Colab에서는 Secrets 기능으로 등록해서 불러올 것.
"""

from .llm import call_judge_json, make_judge
from .validate_log import VALID_CODES

CODING_MANUAL = """당신은 두 참여자(alpha, beta)의 대화 로그에서 각 발화의 기능을 분류하는 코더입니다.

# 코드 정의
- phatic: 사교적/대화 유지용 발화. 과제 내용이 없음. (예: "알겠어", "좋아", "고마워")
- meta : 협력 과정 자체를 언급. 진행 방식·역할·순서에 대한 발화. (예: "우리 방향 다시 잡자")
- lead : 논의의 초점이나 다음 할 일을 지정. (예: "지금 정할 건 요일이야")
- arch : 대화에 아직 등장하지 않은 정보·주장·아이디어를 새로 제시.
- agree: 상대 발화를 받아들여 **자신의 이전 입장이나 계획을 바꿈**.
- comp : 상대가 요청하지 않았는데 상대 발화의 **빠진 부분을 자발적으로 메움**.

# 판정 규칙 (반드시 지킬 것)
1. phatic은 잔여 범주다. 다른 코드가 하나라도 붙으면 phatic을 붙이지 않는다.
2. agree는 두 조건을 모두 만족해야 한다.
   (a) 상대 의견을 수용하는 표현이 있고,
   (b) 자신이 앞서 말한 내용을 철회·수정·보강하는 진술이 있다.
   수용 표현만 있고 자기 입장 변경이 없으면 agree가 아니라 phatic이다.
3. comp는 두 조건을 모두 만족해야 한다.
   (a) 상대가 명시적으로 요청하지 않았고,
   (b) 상대 발화에 빠져 있던 요소를 채운다.
   상대의 질문에 대한 단순 답변은 comp가 아니다.
4. arch는 대화에 처음 등장하는 내용이어야 한다. 앞서 나온 내용의 반복·재진술은 arch가 아니다.
5. 한 발화에 여러 코드를 붙일 수 있다. 해당 없으면 빈 배열을 준다.

# 참조(reference) 판정
이 발화가 상대방 발화의 **내용**을 직접 받아 말하고 있으면 상대방 이름을, 아니면 null을 넣는다.
- 상대 이름을 부르기만 한 것("beta야, 내 생각엔")은 참조가 아니다.
- 상대가 말한 사실·제안·질문을 언급하거나 답할 때만 참조다.
- 자기 자신은 절대 참조 대상이 될 수 없다.

# 근거
붙인 코드마다 그 판단의 근거가 된 발화 속 원문 구절을 그대로 인용한다.
(인간 코더와 결과가 갈렸을 때 원인을 찾기 위한 것이다.)

반드시 아래 JSON 형식으로만 답하세요. 다른 설명은 절대 추가하지 마세요.
{"codes": ["..."], "reference": "상대방 이름 또는 null", "evidence": {"코드명": "원문 구절"}}
"""


def tag_turn(client, prev_turns: list, current_turn: dict,
             model: str = None) -> dict:  # client: llm.JudgeClient
    """
    발화 하나를 태깅한다.
    prev_turns : 직전 문맥 (기본 2턴 = 자기 직전 발화 + 상대 직전 발화)
    current_turn: 태깅할 대상 turn (dict, 'speaker'와 'text' 포함)
    반환: {"codes": [...], "ref": "상대방 이름 또는 None", "evidence": {...}}
    """
    context = "\n".join(f"{t['speaker']}: {t['text']}" for t in prev_turns)
    prompt = (
        f"대화 맥락:\n{context or '(없음 - 이 발화가 대화의 첫 발화입니다)'}\n\n"
        f"분류할 발화 ({current_turn['speaker']}): {current_turn['text']}"
    )

    parsed = call_judge_json(
        client,
        system=CODING_MANUAL,
        user=prompt,
        max_tokens=500,
        model=model,
    )

    codes = [c for c in parsed.get("codes", []) if c in VALID_CODES]

    ref = parsed.get("reference")
    if ref in (None, "null", "None", ""):
        ref = None
    # 자기 자신 참조는 정의상 불가능하므로 버린다 (validate_log에서도 막힌다).
    if ref == current_turn["speaker"]:
        ref = None

    evidence = parsed.get("evidence") or {}
    if not isinstance(evidence, dict):
        evidence = {}

    return {"codes": codes, "ref": ref, "evidence": evidence}


def tag_log(log: dict, api_key: str, context_window: int = 2,
            model: str = None, provider: str = None) -> dict:
    """
    로그 전체(turns)를 순회하며 각 turn에 codes, ref, evidence를 채워넣는다.
    log를 복사하지 않고 그 자리에서 수정 후 반환한다.

    api_key  : judge 제공사의 키. 어느 제공사인지는 COOP_JUDGE_PROVIDER가 정한다
               (기본 anthropic). provider 인자로 직접 지정할 수도 있다.

    한계: context_window=2는 '자기 직전 발화 + 상대 직전 발화'만 본다.
          여러 턴 전에 나온 제안을 뒤늦게 반영하는 경우는 참조로 잡히지 않는다.
          이 값을 바꾸면 dir_AB / dir_BA가 달라지므로, 파일럿과 본실험에서
          같은 값을 써야 한다 (달라지면 임계값 캘리브레이션이 무효가 된다).
    """
    client = make_judge(api_key, provider=provider, model=model)
    turns = log["turns"]

    for i, turn in enumerate(turns):
        prev_turns = turns[max(0, i - context_window):i]
        tag_result = tag_turn(client, prev_turns, turn, model=model)
        turn["codes"] = tag_result["codes"]
        turn["ref"] = tag_result["ref"]
        turn["evidence"] = tag_result["evidence"]

    # 어느 제공사·모델로 태깅했는지 로그에 남긴다.
    # 파일럿과 본실험의 judge가 달라지면 Kappa 비교가 무효가 되므로 기록이 필요하다.
    log["tagging"] = {
        "provider": client.provider,
        "model": client.model,
        "context_window": context_window,
        "manual_version": "v2",
    }
    return log
