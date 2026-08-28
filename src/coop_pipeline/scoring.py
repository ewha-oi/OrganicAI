# 채점

"""
채점 (단독/집단 산출 품질 평가)
================================
A1(정보통합): 체크리스트 항목 매칭으로 정답률(0~1) 계산
A2/A4(의견수렴/창의생성): LLM-judge 등급 평가(1~5점, 블라인드)
+ L4 판정에 쓰이는 new_idea_flag도 여기서 LLM-judge로 산출한다.

채점 기준의 전체 정의와 근거는 docs/RUBRIC.md에 있다.
이 파일은 그 문서를 코드로 옮긴 것이므로, 기준을 바꿀 때는 문서를 먼저 고칠 것.
"""

import os
import re
import unicodedata

from .llm import call_judge_json, make_judge

# judge SDK(anthropic/groq/google-generativeai)는 실제로 호출할 때만 필요하다.
# A1 채점(score_a1)과 정규화는 API 호출이 없으므로, SDK가 설치되지 않은
# 환경에서도 이 모듈을 임포트하고 테스트할 수 있어야 한다.
# 어느 제공사를 쓸지는 llm.JUDGE_PROVIDER(=COOP_JUDGE_PROVIDER)가 정한다.

# ---------------------------------------------------------------------------
# A1: 체크리스트 기반 결정론 채점
# ---------------------------------------------------------------------------
# 정규화에서 제거할 문자: 공백, 구두점, 기호.
# "19:00" -> "1900", "panel-3" -> "panel3", "B 실" -> "b실" 처럼 만들어
# 표기 흔들림 때문에 정답을 놓치는 위양성(false negative)을 줄인다.
_STRIP_CHARS = re.compile(r"[\s​·,./\\|~!@#$%^&*()\[\]{}<>\"'`:;?_+=\-–—]+")


def normalize(text) -> str:
    """대소문자/전각/공백/구두점 차이를 없앤 비교용 문자열로 바꾼다."""
    return _STRIP_CHARS.sub("", unicodedata.normalize("NFKC", str(text)).lower())


def _item_tokens(item) -> list:
    """
    체크리스트 항목 하나를 '공백 기준 토큰' 목록으로 쪼갠다.
    - "수요일"                     -> ["수요일"]                (1토큰)
    - "C동 전기실 panel-3 과부하 차단" -> ["c동","전기실","panel3","과부하","차단"] (5토큰)
    """
    return [tok for tok in (normalize(t) for t in str(item).split()) if tok]


def score_item(normalized_output: str, item) -> float:
    """
    체크리스트 항목 하나의 충족도를 0~1로 반환한다.
    항목의 토큰 중 몇 개가 산출물에 등장하는지의 비율(부분 점수).

    부분 점수를 주는 이유: 시나리오에 따라 체크리스트 항목이
    "C동 전기실 panel-3 과부하 차단"처럼 문장형으로 작성돼 있는데,
    LLM이 이 문장을 토씨 하나 안 틀리고 재현할 확률은 사실상 0이다.
    항목 전체 일치만 인정하면 그런 시나리오의 점수가 구조적으로 0에 수렴하고,
    결과적으로 판정이 항상 L2로 고정돼 실험 자체가 무의미해진다.
    """
    tokens = _item_tokens(item)
    if not tokens:
        return 0.0
    hits = sum(1 for tok in tokens if tok in normalized_output)
    return hits / len(tokens)


def score_a1(output_text: str, checklist: list, partial: bool = True) -> float:
    """
    output_text가 checklist를 얼마나 반영했는지 0~1 점수로 반환.

    partial=True (기본): 항목별 토큰 일치 비율의 평균.
    partial=False       : 항목의 모든 토큰이 있어야 1점 (엄격 채점, 민감도 분석용).

    한계 (docs/RUBRIC.md에 동일 내용 기재):
    - 의미가 같아도 표기가 다르면 못 잡는다. "19:00"은 잡지만 "오후 7시"는 못 잡는다.
    - 오답을 함께 나열해도 감점하지 않는다. 양다리 답변이 만점을 받을 수 있다.
    두 한계 모두 파일럿에서 빈도를 측정한 뒤 v2에서 보완한다.
    """
    if not checklist:
        return 0.0
    normalized_output = normalize(output_text)
    scores = [score_item(normalized_output, item) for item in checklist]
    if not partial:
        scores = [1.0 if s == 1.0 else 0.0 for s in scores]
    return sum(scores) / len(scores)


def score_a1_detail(output_text: str, checklist: list) -> dict:
    """
    score_a1과 같은 계산을 하되, 항목별 점수를 함께 돌려준다.
    어떤 항목에서 점수를 잃었는지 확인해 체크리스트를 개선할 때 쓴다.
    """
    normalized_output = normalize(output_text)
    per_item = {str(item): round(score_item(normalized_output, item), 3)
                for item in checklist}
    total = sum(per_item.values()) / len(per_item) if per_item else 0.0
    return {"score": total, "per_item": per_item}


# ---------------------------------------------------------------------------
# 블라인드 처리
# ---------------------------------------------------------------------------
_SPEAKER_MARK = re.compile(r"(?m)^\s*(alpha|beta|알파|베타)\s*[::]\s*")
_SPEAKER_MENTION = re.compile(r"\b(alpha|beta)\b", re.I)


def anonymize_output(text: str) -> str:
    """
    judge에게 넘기기 전 화자 표기를 지운다.

    이 처리가 없으면 그룹 산출물에 "alpha:", "beta:" 표기가 남아 judge가
    한 줄만 보고도 그룹 조건임을 알아차린다. 즉 '블라인드 채점'이 이름뿐이 된다.
    """
    text = _SPEAKER_MARK.sub("", str(text))
    return _SPEAKER_MENTION.sub("참여자", text)


# ---------------------------------------------------------------------------
# A2 / A4: LLM-judge 등급 (1~5점)
# ---------------------------------------------------------------------------
_HEADER = """당신은 과제 산출물의 품질을 평가하는 채점자입니다.
아래 산출물을 읽고 1~5점으로 평가하세요. 이 산출물이 단독 조건인지 그룹 조건인지는
알려주지 않습니다 - 오직 내용만 보고 평가하세요.
"""

_FOOTER = """
반드시 아래 JSON 형식으로만 답하세요. 다른 설명은 절대 추가하지 마세요.
{"grade": 1~5 중 하나, "reason": "..."}
"""

# A2(의견수렴): 대립하는 입장을 어떻게 조율했는가가 평가 대상이다.
JUDGE_RUBRIC_A2 = _HEADER + """
1점: 한쪽 입장/관점만 반영
2점: 절충 시도했으나 피상적
3점: 핵심 논거를 균형있게 반영
4점: 논거 간 트레이드오프까지 조율
5점: 새로운 대안까지 제시

채점 규칙:
- 산출물의 길이나 형식이 아니라 내용만 보고 판단하세요.
- 지문의 제약(예산, 개수, 기한 등)을 위반한 산출물은 최대 3점까지만 줄 수 있습니다.
- 결론을 하나로 확정하지 않고 여러 안을 나열만 한 산출물은 최대 2점까지만 줄 수 있습니다.
- 확신이 서지 않으면 낮은 쪽 점수를 주세요.
""" + _FOOTER

# A4(창의공동생성): 대립하는 입장 자체가 없는 과제다.
#
# A2 척도를 A4에 그대로 쓰면 안 된다. 파일럿에서 확인된 실패:
# A4 산출물 5개(단독 4 + 그룹 1)가 전부 1~2점을 받았고, 감점 사유가 모두
# "다른 관점·트레이드오프·대안이 없다"였다. 캠페인 기획 과제에는 조율할 대립이
# 없으니 당연한 결과다. 그 결과 grade_gap_min=2를 구조적으로 넘을 수 없어
# A4는 판정이 Q3에 고정됐다. 또 "여러 안을 나열만 하면 최대 2점" 규칙은
# '아이디어를 정확히 3개 제시하라'는 A4 지문과 정면으로 충돌한다.
JUDGE_RUBRIC_A4 = _HEADER + """
1점: 과제가 요구한 항목을 빠뜨렸거나 내용이 이름뿐임
2점: 요구 항목은 채웠으나 누구나 떠올릴 수준이고 항목들이 서로 비슷함
3점: 요구를 충족하고 각 항목이 구체적이며 실행 가능함
4점: 3점에 더해 항목들이 서로 다른 상황·원인을 겨냥해 중복이 없음
5점: 4점에 더해 통상적으로 나오기 어려운 접근이 포함됨

채점 규칙:
- 산출물의 길이나 형식이 아니라 내용만 보고 판단하세요.
- 지문의 제약(예산, 개수, 기한 등)을 위반한 산출물은 최대 3점까지만 줄 수 있습니다.
- 이 과제는 여러 아이디어를 나열하는 것이 요구사항입니다. 나열했다는 이유로 감점하지 마세요.
- 찬반 대립이나 트레이드오프 조율은 이 과제의 평가 대상이 아닙니다.
- 확신이 서지 않으면 낮은 쪽 점수를 주세요.
""" + _FOOTER

JUDGE_RUBRICS = {"A2": JUDGE_RUBRIC_A2, "A4": JUDGE_RUBRIC_A4}


def judge_rubric(task_type: str) -> str:
    """과제 유형에 맞는 채점 척도를 고른다. 모르는 유형이면 조용히 넘기지 않는다."""
    if task_type not in JUDGE_RUBRICS:
        raise ValueError(
            f"LLM-judge 등급 채점은 {sorted(JUDGE_RUBRICS)}만 지원한다: '{task_type}' "
            f"(A1은 score_a1의 체크리스트 채점을 쓴다)"
        )
    return JUDGE_RUBRICS[task_type]


def score_a2_a4(output_text: str, api_key: str, task_type: str,
                model: str = None, provider: str = None) -> int:
    """
    LLM-judge로 1~5점 등급 산출.
    블라인드: 단독/그룹 여부를 프롬프트에 노출하지 않고, 화자 표기도 제거한다.

    task_type은 기본값을 두지 않는다. 과제와 안 맞는 척도를 쓰면 모든 산출물이
    같은 이유로 바닥 등급을 받는데, 그것이 '협력 이득 없음'과 구별되지 않는다.
    """
    return score_a2_a4_detail(output_text, api_key, task_type,
                              model=model, provider=provider)["grade"]


def score_a2_a4_detail(output_text: str, api_key: str, task_type: str,
                       model: str = None, provider: str = None) -> dict:
    """score_a2_a4와 동일하되 judge가 쓴 근거(reason)까지 반환한다."""
    client = make_judge(api_key, provider=provider, model=model)
    parsed = call_judge_json(
        client,
        system=judge_rubric(task_type),
        user=f"산출물:\n{anonymize_output(output_text)}",
        max_tokens=300,
    )
    grade = int(parsed["grade"])
    if not 1 <= grade <= 5:
        raise ValueError(f"judge가 범위를 벗어난 등급을 반환: {grade}")
    return {"grade": grade, "reason": parsed.get("reason", "")}


# ---------------------------------------------------------------------------
# L4 보조 판정: 신규 해결책 존재 여부
# ---------------------------------------------------------------------------
NEW_IDEA_PROMPT = """당신은 두 산출물을 비교하는 평가자입니다.
아래는 (A) 단독 참여자들의 산출물 목록과 (B) 협의를 거친 산출물입니다.
B에 A 중 어느 것에도 없던 해결책 요소가 포함되어 있습니까?

판정 규칙:
- '해결책 요소'란 실행 가능한 제안 한 개를 말합니다. 표현이나 문체의 차이,
  같은 내용을 더 자세히 쓴 것, 순서만 바꾼 것은 새로운 요소가 아닙니다.
- A에 있는 두 요소를 단순히 나란히 붙인 것도 새로운 요소가 아닙니다.
- A에는 없던 판단 기준, 절차, 대안이 새로 등장했을 때만 true입니다.
- 확신이 서지 않으면 false를 주세요.

반드시 아래 JSON 형식으로만 답하세요. 다른 설명은 절대 추가하지 마세요.
{"new_idea": true 또는 false, "reason": "..."}
"""

# new_idea 판정은 단독 산출물 10개와 그룹 산출물을 한 요청에 넣는다. 생성 모델이
# 장문으로 답하면 이 단계만 2만 자 이상이 되어 judge의 단일 요청 TPM을 넘는다.
# 각 산출물의 시작(제안)과 끝(결론)을 보존한 결정론적 발췌로 요청을 제한한다.
# 모든 담당자가 같은 값을 써야 하므로 환경변수는 임포트 전에만 바꿀 것.
NEW_IDEA_SOLO_CHARS = int(os.environ.get("COOP_NEW_IDEA_SOLO_CHARS", 550))
NEW_IDEA_GROUP_CHARS = int(os.environ.get("COOP_NEW_IDEA_GROUP_CHARS", 1500))
_OMISSION_MARK = "\n[중략]\n"


def _new_idea_excerpt(text: str, limit: int) -> str:
    """신규성 비교용으로 앞부분과 결론을 남긴 제한 길이 발췌를 만든다."""
    text = str(text)
    if len(text) <= limit:
        return text
    if limit <= len(_OMISSION_MARK):
        return text[:limit]

    available = limit - len(_OMISSION_MARK)
    head = (available * 2) // 3
    tail = available - head
    return text[:head] + _OMISSION_MARK + text[-tail:]


def detect_new_idea(solo_outputs: list, group_output: str, api_key: str,
                    model: str = None, provider: str = None) -> bool:
    """단독 산출물들에 없던 새로운 해결책이 그룹 산출물에 있는지 LLM-judge로 판정."""
    return detect_new_idea_detail(solo_outputs, group_output, api_key,
                                  model=model, provider=provider)["new_idea"]


def detect_new_idea_detail(solo_outputs: list, group_output: str, api_key: str,
                           model: str = None, provider: str = None) -> dict:
    """detect_new_idea와 동일하되 judge가 쓴 근거까지 반환한다.

    이 호출은 10개 단독 산출물을 한 번에 비교하므로, 원문 로그는 보존하되 judge에는
    고정 길이 발췌만 보낸다. 기본값 기준 사용자 프롬프트는 약 7,100자 이하가 된다.
    """
    client = make_judge(api_key, provider=provider, model=model)
    solo_text = "\n---\n".join(
        _new_idea_excerpt(anonymize_output(output), NEW_IDEA_SOLO_CHARS)
        for output in solo_outputs
    )
    prompt = (
        f"(A) 단독 산출물들:\n{solo_text}\n\n"
        f"(B) 협의 산출물:\n"
        f"{_new_idea_excerpt(anonymize_output(group_output), NEW_IDEA_GROUP_CHARS)}"
    )
    parsed = call_judge_json(
        client,
        system=NEW_IDEA_PROMPT,
        user=prompt,
        max_tokens=300,
    )
    return {"new_idea": bool(parsed["new_idea"]), "reason": parsed.get("reason", "")}


# ---------------------------------------------------------------------------
# 로그에 채점 결과 붙이기
# ---------------------------------------------------------------------------
def attach_scores(log: dict, solo_scores_or_grades: list, group_score_or_grade,
                  new_idea_flag: bool) -> dict:
    """
    채점 결과를 로그 딕셔너리에 붙여서 반환한다.
    validate_log() 스펙에 맞춰 task_type에 따라 필드명을 다르게 넣는다.
    """
    if not solo_scores_or_grades:
        raise ValueError(
            "단독 조건 점수가 비어 있음. A1의 solo_p90 / A2·A4의 중앙값을 계산할 수 없다."
        )

    if log["task_type"] == "A1":
        log["solo_scores"] = list(solo_scores_or_grades)
        log["group_score"] = group_score_or_grade
    else:
        log["solo_grades"] = list(solo_scores_or_grades)
        log["group_grade"] = group_score_or_grade
    log["new_idea_flag"] = bool(new_idea_flag)
    return log
