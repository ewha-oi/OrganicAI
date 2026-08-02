# 채점

"""
채점 (단독/집단 산출 품질 평가)
================================
A1(정보통합): 체크리스트 항목 매칭으로 정답률(0~1) 계산
A2/A4(의견수렴/창의생성): LLM-judge 등급 평가(1~5점, 블라인드)
+ L4 판정에 쓰이는 new_idea_flag도 여기서 LLM-judge로 산출한다.
"""

import json

import anthropic

JUDGE_MODEL = "claude-sonnet-4-6"


def score_a1(output_text: str, checklist: list) -> float:
    """
    output_text 안에 checklist 항목이 몇 개나 반영됐는지 세서 0~1 점수로 반환.
    지금은 단순 키워드 포함 여부로 근사 채점한다 (필요하면 LLM 매칭으로 교체 가능).
    """
    if not checklist:
        return 0.0
    hits = sum(1 for item in checklist if item in output_text)
    return hits / len(checklist)


JUDGE_RUBRIC = """당신은 과제 산출물의 품질을 평가하는 채점자입니다.
아래 산출물을 읽고 1~5점으로 평가하세요. 이 산출물이 단독 조건인지 그룹 조건인지는
알려주지 않습니다 - 오직 내용만 보고 평가하세요.

1점: 한쪽 입장/관점만 반영
2점: 절충 시도했으나 피상적
3점: 핵심 논거를 균형있게 반영
4점: 논거 간 트레이드오프까지 조율
5점: 새로운 대안까지 제시

반드시 아래 JSON 형식으로만 답하세요. 다른 설명은 절대 추가하지 마세요.
{"grade": 1~5 중 하나, "reason": "..."}
"""


def score_a2_a4(output_text: str, api_key: str) -> int:
    """LLM-judge로 1~5점 등급 산출 (블라인드: 단독/그룹 여부는 프롬프트에 노출 안 함)."""
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=200,
        temperature=0,
        system=JUDGE_RUBRIC,
        messages=[{"role": "user", "content": f"산출물:\n{output_text}"}],
    )
    parsed = json.loads(response.content[0].text.strip())
    return int(parsed["grade"])


NEW_IDEA_PROMPT = """당신은 두 산출물을 비교하는 평가자입니다.
아래는 (A) 단독 에이전트들의 산출물 목록과 (B) 그룹(2-agent) 산출물입니다.
B에 A 중 어느 것에도 없던 해결책 요소가 포함되어 있습니까?

반드시 아래 JSON 형식으로만 답하세요. 다른 설명은 절대 추가하지 마세요.
{"new_idea": true 또는 false, "reason": "..."}
"""


def detect_new_idea(solo_outputs: list, group_output: str, api_key: str) -> bool:
    """단독 산출물들에 없던 새로운 해결책이 그룹 산출물에 있는지 LLM-judge로 판정."""
    client = anthropic.Anthropic(api_key=api_key)
    solo_text = "\n---\n".join(solo_outputs)
    prompt = f"(A) 단독 산출물들:\n{solo_text}\n\n(B) 그룹 산출물:\n{group_output}"

    response = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=200,
        temperature=0,
        system=NEW_IDEA_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    parsed = json.loads(response.content[0].text.strip())
    return bool(parsed["new_idea"])


def attach_scores(log: dict, solo_scores_or_grades: list, group_score_or_grade,
                   new_idea_flag: bool) -> dict:
    """
    채점 결과를 로그 딕셔너리에 붙여서 반환한다.
    validate_log() 스펙에 맞춰 task_type에 따라 필드명을 다르게 넣는다.
    """
    if log["task_type"] == "A1":
        log["solo_scores"] = solo_scores_or_grades
        log["group_score"] = group_score_or_grade
    else:
        log["solo_grades"] = solo_scores_or_grades
        log["group_grade"] = group_score_or_grade
    log["new_idea_flag"] = new_idea_flag
    return log