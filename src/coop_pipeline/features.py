# 정량 지표 추출

"""
특징 추출
==========
태깅(codes, ref 필드가 채워진) 완료된 로그를 받아,
판정 프레임(classify.py)이 사용할 정량 지표를 계산한다.

각 지표의 정의는 docs/RUBRIC.md의 "정량 지표" 절과 1:1로 대응한다.
"""

import math
from statistics import median

from .thresholds import THRESHOLDS


class FeatureError(ValueError):
    """특징 추출에 필요한 필드가 없거나 값이 이상할 때."""


def percentile(values: list, q: float) -> float:
    """
    numpy.percentile(values, q)의 기본 방식('linear' 보간)과 동일한 값을 계산한다.

    numpy를 쓰지 않는 이유: 이 함수 하나 때문에 판정 로직 전체가 numpy에 묶이면
    가벼운 환경(로컬 테스트, CI)에서 판정 결과를 재확인할 수 없다.

    주의: 표본이 작으면 p90은 사실상 최댓값이 된다.
          예) n=5 -> 인덱스 3.6 지점이므로 상위 두 값 사이. 그룹이 이기기 매우 어렵다.
          단독 조건 표본 수(n)를 충분히 확보할 것. (docs/RUBRIC.md 참고)
    """
    if not values:
        raise FeatureError("퍼센타일을 계산할 값이 없음")
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]

    index = (len(ordered) - 1) * (q / 100.0)
    low, high = math.floor(index), math.ceil(index)
    if low == high:
        return ordered[int(index)]
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def extract_features(log: dict, thresholds: dict = THRESHOLDS) -> dict:
    """태깅이 완료된 로그를 받아 판정에 필요한 정량 지표를 계산한다."""
    turns = log["turns"]
    speakers = sorted({t["speaker"] for t in turns})
    if len(speakers) != 2:
        raise FeatureError(f"2-agent 로그가 아닙니다. speakers={speakers}")
    a, b = speakers

    # dir_AB: a가 b의 발화 내용을 받아 말한 턴 수 (방향성 있는 참조 횟수)
    dir_ab = sum(1 for t in turns if t["speaker"] == a and t.get("ref") == b)
    dir_ba = sum(1 for t in turns if t["speaker"] == b and t.get("ref") == a)

    total_turns = len(turns)
    comp_count = sum(1 for t in turns if "comp" in t.get("codes", []))
    comp_ratio = comp_count / total_turns if total_turns else 0.0

    revision_seq = _count_revision_turns(turns)

    features = {
        "speaker_a": a,
        "speaker_b": b,
        "dir_AB": dir_ab,
        "dir_BA": dir_ba,
        "comp_count": comp_count,
        "comp_ratio": comp_ratio,
        "revision_seq": revision_seq,
        "total_turns": total_turns,
    }

    task_type = log["task_type"]
    if task_type == "A1":
        solo_scores = _require(log, "solo_scores", task_type)
        features["solo_n"] = len(solo_scores)
        features["solo_p90"] = percentile(
            solo_scores, thresholds["solo_percentile_A1"]
        )
        features["group_score"] = float(_require(log, "group_score", task_type))
    else:
        solo_grades = _require(log, "solo_grades", task_type)
        features["solo_n"] = len(solo_grades)
        features["solo_median_grade"] = float(median(solo_grades))
        features["group_grade"] = float(_require(log, "group_grade", task_type))

    # L4 판정 보조 플래그: scoring.detect_new_idea()가 미리 채워 둔 값
    features["new_idea_flag"] = bool(log.get("new_idea_flag", False))

    return features


def _require(log: dict, field: str, task_type: str):
    if field not in log:
        raise FeatureError(
            f"task_type='{task_type}' 로그에 '{field}'가 없음 — "
            f"scoring.attach_scores()를 먼저 실행할 것"
        )
    value = log[field]
    if isinstance(value, list) and not value:
        raise FeatureError(f"'{field}'가 빈 리스트임 — 단독 조건 채점 결과가 필요함")
    return value


def _count_revision_turns(turns: list) -> int:
    """
    'agree' 코드가 붙고 참조 대상이 있는 발화 수를 센다.

    이름이 revision_seq이지만 실제로 세는 것은 '연속 시퀀스'가 아니라 '발화 개수'다.
    코딩 매뉴얼 v2에서 agree는 '수용 표현 + 자기 입장 변경'을 모두 요구하므로
    이 카운트는 계획 수정이 일어난 횟수의 근사값이다.
    (docs/RUBRIC.md의 "알려진 한계" 참고)
    """
    return sum(
        1 for t in turns
        if "agree" in t.get("codes", []) and t.get("ref") is not None
    )
