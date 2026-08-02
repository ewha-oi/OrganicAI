# 정량 지표 추출

"""
특징 추출
==========
태깅(codes, ref 필드가 채워진) 완료된 로그를 받아,
판정 프레임(classify.py)이 사용할 정량 지표를 계산한다.
"""

from statistics import median

import numpy as np

from .thresholds import THRESHOLDS


def extract_features(log: dict, thresholds: dict = THRESHOLDS) -> dict:
    """태깅이 완료된 로그를 받아 판정에 필요한 정량 지표를 계산한다."""
    turns = log["turns"]
    speakers = sorted({t["speaker"] for t in turns})
    if len(speakers) != 2:
        raise ValueError(f"2-agent 로그가 아닙니다. speakers={speakers}")
    a, b = speakers

    dir_ab = sum(1 for t in turns if t["speaker"] == a and t.get("ref") == b)
    dir_ba = sum(1 for t in turns if t["speaker"] == b and t.get("ref") == a)

    total_turns = len(turns)
    comp_count = sum(1 for t in turns if "comp" in t.get("codes", []))
    comp_ratio = comp_count / total_turns if total_turns else 0.0

    revision_seq = _count_revision_sequences(turns)

    features = {
        "dir_AB": dir_ab,
        "dir_BA": dir_ba,
        "comp_ratio": comp_ratio,
        "revision_seq": revision_seq,
        "total_turns": total_turns,
    }

    task_type = log["task_type"]
    if task_type == "A1":
        solo_scores = log["solo_scores"]
        features["solo_p90"] = float(
            np.percentile(solo_scores, thresholds["solo_percentile_A1"])
        )
        features["group_score"] = log["group_score"]
    else:
        solo_grades = log["solo_grades"]
        features["solo_median_grade"] = median(solo_grades)
        features["group_grade"] = log["group_grade"]

    # L4 판정 보조 플래그: LLM-judge가 별도로 채점해 로그에 미리 심어둔 값
    features["new_idea_flag"] = log.get("new_idea_flag", False)

    return features


def _count_revision_sequences(turns: list) -> int:
    """
    'agree' 코드가 붙고 참조 대상이 있는 발화 수를 근사적으로 카운트한다.
    실제 판정 시에는 LLM-judge에게 "이 agree가 실제 계획 수정을 동반하는가"를
    별도로 물어 new_idea_flag / revision 여부를 보강하는 것을 권장한다.
    """
    return sum(
        1 for t in turns
        if "agree" in t.get("codes", []) and t.get("ref") is not None
    )