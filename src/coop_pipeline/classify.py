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


def min_turns_for_bidirectional(thresholds: dict = THRESHOLDS) -> int:
    """
    Q2가 통과 '가능'해지는 최소 max_turns.

    화자는 번갈아 말하므로 한 화자의 발화는 전체의 절반이고, 그중 대화의 첫 발화는
    참조할 대상이 없다(tagging에서 맥락이 '(없음)'이 된다). 따라서 한 화자가 낼 수
    있는 참조는 최대 ceil(max_turns/2) - 1회다. 이것이 bidirectional_min보다 작으면
    Q2는 협력의 정도와 무관하게 **반드시** 실패한다.
    """
    return 2 * thresholds["bidirectional_min"] + 1


def _reference_ceiling(turns: list, speaker: str) -> int:
    """이 화자가 이 로그에서 낼 수 있었던 참조의 최대 횟수 (첫 발화는 제외)."""
    return sum(1 for i, t in enumerate(turns) if t["speaker"] == speaker and i > 0)


def config_warnings(log: dict, features: dict, thresholds: dict = THRESHOLDS) -> list:
    """
    판정 결과를 그대로 해석하면 안 되는 설정상의 문제를 찾는다.

    턴 수가 모자라면 Q2는 대화 내용과 무관하게 실패하는데, 그 결과가 L0/L1이라
    '협력이 없었다'는 실험 결과와 겉모습이 완전히 같다. 조용히 지나가면
    파일럿 전체를 잘못 해석하게 되므로 리포트 맨 앞에 띄운다.
    """
    found = []
    bmin = thresholds["bidirectional_min"]
    for speaker in (features["speaker_a"], features["speaker_b"]):
        ceiling = _reference_ceiling(log["turns"], speaker)
        if ceiling < bmin:
            found.append(
                f"{speaker}의 참조 가능 횟수는 최대 {ceiling}회인데 "
                f"bidirectional_min={bmin}이다 — 이 설정에서 Q2는 통과할 수 없다. "
                f"L0/L1이 나와도 협력의 부재가 아니라 턴 수 부족의 산물이다. "
                f"max_turns를 {min_turns_for_bidirectional(thresholds)} 이상으로 올릴 것"
            )
    return found


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
        # 저장된 로그를 다시 판정할 때(classify_saved_dir)도 매번 다시 계산되므로
        # 이 경고는 유실되지 않는다.
        "config_warnings": config_warnings(log, features, thresholds),
        **result,
    }


def format_result(result: dict) -> str:
    """판정 결과를 사람이 읽는 리포트 문자열로 만든다 (Colab에서 print용)."""
    warnings = result.get("config_warnings") or []

    # 경고는 리포트의 맨 앞과 맨 뒤에 모두 찍는다. 판정 줄만 보고 넘어가거나
    # 출력이 길어 위쪽이 잘려도 눈에 걸리게 하기 위한 것이다.
    banner = []
    if warnings:
        banner.append("!!! 설정 경고 — 이 판정을 그대로 해석하지 말 것 !!!")
        banner.extend(f"  - {w}" for w in warnings)

    lines = [
        "=" * 62,
        *banner,
        *(["-" * 62] if banner else []),
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
    if banner:
        lines.append("-" * 62)
        lines.extend(banner)
    lines.append("=" * 62)
    return "\n".join(lines)
