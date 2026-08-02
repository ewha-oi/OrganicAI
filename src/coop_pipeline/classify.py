# L0~L4 결정 트리

"""
협력 층위(L0~L4) 판정
========================
결정 트리:
  Q1 참조 존재?        -> 없으면 L0
  Q2 양방향 참조?       -> 편측이면 L1
  Q3 집단 산출 우위?    -> 아니면 L2
  Q4 신규성 조건 충족?  -> 예면 L4, 아니면 L3
"""

from .features import extract_features
from .thresholds import THRESHOLDS
from .validate_log import validate_log


def classify(features: dict, task_type: str, thresholds: dict = THRESHOLDS) -> dict:
    """features 딕셔너리를 결정 트리에 통과시켜 L0~L4 라벨과 근거를 반환한다."""

    # Q1: 참조 존재 여부
    if features["dir_AB"] == 0 and features["dir_BA"] == 0:
        return {"level": "L0", "reason": "양방향 참조 모두 0회"}

    # Q2: 쌍방향성
    if (features["dir_AB"] < thresholds["bidirectional_min"]
            or features["dir_BA"] < thresholds["bidirectional_min"]):
        return {
            "level": "L1",
            "reason": f"편측 참조 (A→B={features['dir_AB']}, B→A={features['dir_BA']})",
        }

    # Q3: 집단 산출 우위 (과제유형별 분기)
    if task_type == "A1":
        group_wins = features["group_score"] > features["solo_p90"]
        detail = f"group={features['group_score']:.2f}, solo_p90={features['solo_p90']:.2f}"
    else:
        gap = features["group_grade"] - features["solo_median_grade"]
        group_wins = gap >= thresholds["grade_gap_min"]
        detail = f"등급차={gap:.1f} (기준 {thresholds['grade_gap_min']})"

    if not group_wins:
        return {"level": "L2", "reason": f"쌍방향 참조는 있으나 집단 우위 미충족 ({detail})"}

    # Q4: L3 / L4 경계
    is_l4 = (
        features["comp_ratio"] >= thresholds["comp_ratio_min"]
        and features["new_idea_flag"]
        and features["revision_seq"] >= thresholds["revision_seq_min"]
    )
    if is_l4:
        return {
            "level": "L4",
            "reason": f"comp비율={features['comp_ratio']:.2f}, 신규해결책=True, 집단우위({detail})",
        }

    return {"level": "L3", "reason": f"집단 우위 확인({detail}), 신규성 기준 미충족"}


def classify_log(log: dict, thresholds: dict = THRESHOLDS) -> dict:
    """편의 함수: 로그 하나를 넣으면 검증 + 특징 추출 + 판정을 한 번에 수행."""
    validate_log(log)  # 형식이 틀리면 여기서 바로 에러가 나고 멈춘다
    features = extract_features(log, thresholds)
    result = classify(features, task_type=log["task_type"], thresholds=thresholds)
    return {"scenario_id": log.get("scenario_id"), "features": features, **result}