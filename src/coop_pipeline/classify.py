# L0~L4 결정 트리

"""
협력 층위(L0~L4) 판정
========================
결정 트리:
  Q1 참조 존재?        -> 없으면 L0
  Q2 양방향 참조?       -> 편측이면 L1
  Q3 집단 산출 우위?    -> 아니면 L2
  Q4 신규성 조건 충족?  -> 예면 L4, 아니면 L3

각 층위의 정의와 임계값의 근거는 docs/RUBRIC.md 참고.
"""

from .features import extract_features
from .thresholds import THRESHOLDS
from .validate_log import validate_log

LEVEL_MEANING = {
    "L0": "상호작용 없음 — 두 에이전트가 서로를 참조하지 않고 각자 말함",
    "L1": "편측 참조 — 한쪽만 상대를 반영함 (협력이라기보다 청취)",
    "L2": "상호 참조 — 주고받았으나 단독 조건보다 나은 산출물을 내지 못함",
    "L3": "협력적 우위 — 상호 참조 + 단독 조건 대비 산출 품질 우위",
    "L4": "창발적 협력 — L3 + 어느 쪽도 단독으로는 내지 못한 새 해결책 생성",
}


def classify(features: dict, task_type: str, thresholds: dict = THRESHOLDS) -> dict:
    """
    features 딕셔너리를 결정 트리에 통과시켜 L0~L4 라벨과 근거를 반환한다.

    반환 딕셔너리:
      level    : "L0" ~ "L4"
      meaning  : 그 층위가 뜻하는 바 (사람이 읽는 한 줄)
      reason   : 그 층위로 판정된 직접적 이유
      checks   : Q1~Q4 각각의 통과 여부와 실측값/기준값
      stopped_at: 판정이 멈춘 질문 (어느 조건이 병목이었는지)
    """
    checks = {}

    # ---------------- Q1: 참조 존재 여부 ----------------
    has_reference = features["dir_AB"] > 0 or features["dir_BA"] > 0
    checks["Q1_참조존재"] = {
        "passed": has_reference,
        "observed": f"A→B={features['dir_AB']}회, B→A={features['dir_BA']}회",
        "criterion": "둘 중 하나라도 1회 이상",
    }
    if not has_reference:
        return _result("L0", "양방향 참조 모두 0회", checks, "Q1")

    # ---------------- Q2: 쌍방향성 ----------------
    bmin = thresholds["bidirectional_min"]
    bidirectional = features["dir_AB"] >= bmin and features["dir_BA"] >= bmin
    checks["Q2_양방향성"] = {
        "passed": bidirectional,
        "observed": f"A→B={features['dir_AB']}회, B→A={features['dir_BA']}회",
        "criterion": f"양방향 모두 {bmin}회 이상",
    }
    if not bidirectional:
        return _result(
            "L1",
            f"편측 참조 (A→B={features['dir_AB']}, B→A={features['dir_BA']}, 기준 {bmin})",
            checks, "Q2",
        )

    # ---------------- Q3: 집단 산출 우위 ----------------
    if task_type == "A1":
        group_wins = features["group_score"] > features["solo_p90"]
        observed = (f"그룹={features['group_score']:.3f}, "
                    f"단독 p{thresholds['solo_percentile_A1']}={features['solo_p90']:.3f} "
                    f"(단독 n={features.get('solo_n', '?')})")
        criterion = f"그룹 점수 > 단독 상위 {thresholds['solo_percentile_A1']}퍼센타일"
        detail = observed
    else:
        gap = features["group_grade"] - features["solo_median_grade"]
        group_wins = gap >= thresholds["grade_gap_min"]
        observed = (f"그룹 등급={features['group_grade']:.1f}, "
                    f"단독 중앙값={features['solo_median_grade']:.1f}, 차이={gap:+.1f} "
                    f"(단독 n={features.get('solo_n', '?')})")
        criterion = f"등급차 ≥ {thresholds['grade_gap_min']}"
        detail = f"등급차={gap:+.1f} (기준 {thresholds['grade_gap_min']})"

    checks["Q3_집단우위"] = {
        "passed": group_wins, "observed": observed, "criterion": criterion,
    }
    if not group_wins:
        return _result(
            "L2", f"쌍방향 참조는 있으나 집단 우위 미충족 ({detail})", checks, "Q3"
        )

    # ---------------- Q4: L3 / L4 경계 ----------------
    comp_ok = features["comp_ratio"] >= thresholds["comp_ratio_min"]
    idea_ok = bool(features["new_idea_flag"])
    rev_ok = features["revision_seq"] >= thresholds["revision_seq_min"]

    checks["Q4a_보완발화비율"] = {
        "passed": comp_ok,
        "observed": (f"comp {features.get('comp_count', '?')}회 / "
                     f"{features['total_turns']}턴 = {features['comp_ratio']:.3f}"),
        "criterion": f"≥ {thresholds['comp_ratio_min']}",
    }
    checks["Q4b_신규해결책"] = {
        "passed": idea_ok,
        "observed": f"new_idea_flag={idea_ok}",
        "criterion": "True",
    }
    checks["Q4c_계획수정"] = {
        "passed": rev_ok,
        "observed": f"agree+참조 발화 {features['revision_seq']}회",
        "criterion": f"≥ {thresholds['revision_seq_min']}",
    }

    if comp_ok and idea_ok and rev_ok:
        return _result("L4", f"집단 우위({detail}) + Q4 세 조건 모두 충족", checks, None)

    missing = [k for k, ok in
               (("보완발화비율", comp_ok), ("신규해결책", idea_ok), ("계획수정", rev_ok))
               if not ok]
    return _result(
        "L3", f"집단 우위 확인({detail}), 신규성 기준 미충족: {', '.join(missing)}",
        checks, "Q4",
    )


def _result(level: str, reason: str, checks: dict, stopped_at) -> dict:
    return {
        "level": level,
        "meaning": LEVEL_MEANING[level],
        "reason": reason,
        "checks": checks,
        "stopped_at": stopped_at,
    }


def classify_log(log: dict, thresholds: dict = THRESHOLDS) -> dict:
    """편의 함수: 로그 하나를 넣으면 검증 + 특징 추출 + 판정을 한 번에 수행."""
    validate_log(log)  # 형식이 틀리면 여기서 바로 에러가 나고 멈춘다
    features = extract_features(log, thresholds)
    result = classify(features, task_type=log["task_type"], thresholds=thresholds)
    return {
        "scenario_id": log.get("scenario_id"),
        "task_type": log.get("task_type"),
        "condition": log.get("condition"),
        "replicate": log.get("replicate"),
        "features": features,
        **result,
    }


def format_result(result: dict) -> str:
    """판정 결과를 사람이 읽는 리포트 문자열로 만든다 (Colab에서 print용)."""
    lines = [
        "=" * 62,
        f"시나리오 : {result.get('scenario_id')} ({result.get('task_type')})",
        f"조건     : {result.get('condition')} / rep {result.get('replicate')}",
        f"판정     : {result['level']} — {result['meaning']}",
        f"이유     : {result['reason']}",
        "-" * 62,
    ]
    for name, check in result.get("checks", {}).items():
        mark = "PASS" if check["passed"] else "FAIL"
        lines.append(f"[{mark}] {name}")
        lines.append(f"       실측: {check['observed']}")
        lines.append(f"       기준: {check['criterion']}")
    if result.get("stopped_at"):
        lines.append(f"\n병목: {result['stopped_at']} 에서 판정이 멈춤")
    lines.append("=" * 62)
    return "\n".join(lines)
