# 로그 형식 검증

"""
로그 JSON 스키마 검증기
=========================
포맷 문서(필드명/타입/설명 표)는 팀원이 읽는 스펙이고
이 파일은 그 스펙을 코드로 강제하는 역할을 한다.
판정 프레임에 로그를 넣기 전, 반드시 validate_log()를 먼저 통과시킬 것.
"""

REQUIRED_TOP_FIELDS = {
    "scenario_id": str,
    "task_type": str,
    "condition": str,
    "replicate": int,
    "turns": list,
}

VALID_TASK_TYPES = {"A1", "A2", "A4"}
VALID_CONDITIONS = {"명시", "묵시"}
VALID_CODES = {"phatic", "meta", "lead", "arch", "agree", "comp"}

REQUIRED_TURN_FIELDS = {
    "turn": int,
    "speaker": str,
    "text": str,
}


class LogValidationError(Exception):
    """로그가 포맷 스펙을 위반했을 때 발생시키는 예외."""


def validate_log(log: dict, require_tags: bool = True) -> None:
    """
    로그가 포맷 스펙을 따르는지 검사한다.
    문제가 있으면 LogValidationError를 발생시키고, 통과하면 아무것도 반환하지 않는다.

    require_tags=True (기본):
        모든 turn에 codes와 ref 키가 실제로 존재해야 한다.
        이 검사가 없으면 태깅을 건너뛴 로그가 에러 없이 전부 L0으로 판정되는데,
        이는 "협력이 전혀 없었다"는 실험 결과와 구별되지 않아 가장 위험하다.
        태깅 전 원본(raw) 로그를 형식만 확인하고 싶을 때만 False를 준다.
    """
    _check_top_fields(log)
    _check_task_type_fields(log)
    _check_turns(log["turns"], require_tags=require_tags)


def _check_top_fields(log: dict) -> None:
    for field, expected_type in REQUIRED_TOP_FIELDS.items():
        if field not in log:
            raise LogValidationError(f"필수 필드 누락: '{field}'")
        if not isinstance(log[field], expected_type):
            raise LogValidationError(
                f"'{field}' 타입 오류: {expected_type.__name__} 이어야 하는데 "
                f"{type(log[field]).__name__} 임"
            )

    if log["task_type"] not in VALID_TASK_TYPES:
        raise LogValidationError(
            f"task_type은 {VALID_TASK_TYPES} 중 하나여야 함: '{log['task_type']}'"
        )

    if log["condition"] not in VALID_CONDITIONS:
        raise LogValidationError(
            f"condition은 {VALID_CONDITIONS} 중 하나여야 함: '{log['condition']}'"
        )


def _check_task_type_fields(log: dict) -> None:
    task_type = log["task_type"]
    if task_type == "A1":
        if "solo_scores" not in log or not isinstance(log["solo_scores"], list):
            raise LogValidationError("task_type='A1'인 로그는 solo_scores(list)가 필요함")
        if "group_score" not in log:
            raise LogValidationError("task_type='A1'인 로그는 group_score가 필요함")
    else:
        if "solo_grades" not in log or not isinstance(log["solo_grades"], list):
            raise LogValidationError(
                f"task_type='{task_type}'인 로그는 solo_grades(list)가 필요함"
            )
        if "group_grade" not in log:
            raise LogValidationError(f"task_type='{task_type}'인 로그는 group_grade가 필요함")


def _check_turns(turns: list, require_tags: bool = True) -> None:
    if len(turns) == 0:
        raise LogValidationError("turns가 비어 있음")

    speakers = {t["speaker"] for t in turns if isinstance(t, dict) and "speaker" in t}

    for i, t in enumerate(turns):
        for field, expected_type in REQUIRED_TURN_FIELDS.items():
            if field not in t:
                raise LogValidationError(f"turns[{i}]에 필수 필드 '{field}' 없음")
            if not isinstance(t[field], expected_type):
                raise LogValidationError(f"turns[{i}]의 '{field}' 타입 오류")

        # 빈 발화는 생성 실패다. 그대로 통과시키면 '한쪽이 한 마디도 안 했다'가
        # '대화가 겉돌았다'는 실험 결과로 둔갑해 L0/L1이 조용히 나온다.
        if not t["text"].strip():
            raise LogValidationError(
                f"turns[{i}]({t['speaker']})의 text가 비어 있음 — 생성이 실패한 "
                f"발화다. 이 로그로는 판정할 수 없다"
            )

        if require_tags:
            if "codes" not in t:
                raise LogValidationError(
                    f"turns[{i}]에 'codes' 없음 — 태깅(tag_log)을 먼저 실행할 것"
                )
            if "ref" not in t:
                raise LogValidationError(
                    f"turns[{i}]에 'ref' 없음 — 태깅(tag_log)을 먼저 실행할 것"
                )

        if not isinstance(t.get("codes", []), list):
            raise LogValidationError(f"turns[{i}]의 'codes'는 리스트여야 함")

        for c in t.get("codes", []):
            if c not in VALID_CODES:
                raise LogValidationError(
                    f"turns[{i}]에 알 수 없는 코드 '{c}' (허용: {VALID_CODES})"
                )

        ref = t.get("ref")
        if ref is not None:
            if ref == t["speaker"]:
                raise LogValidationError(
                    f"turns[{i}]의 ref가 자기 자신('{ref}') — ref는 상대 화자여야 함"
                )
            if ref not in speakers:
                raise LogValidationError(
                    f"turns[{i}]의 ref '{ref}'가 화자 목록 {sorted(speakers)}에 없음"
                )

    if len(speakers) != 2:
        raise LogValidationError(f"화자는 정확히 2명이어야 함. 현재: {speakers}")