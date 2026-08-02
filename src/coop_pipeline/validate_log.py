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


def validate_log(log: dict) -> None:
    """
    로그가 포맷 스펙을 따르는지 검사한다.
    문제가 있으면 LogValidationError를 발생시키고, 통과하면 아무것도 반환하지 않는다.
    """
    _check_top_fields(log)
    _check_task_type_fields(log)
    _check_turns(log["turns"])


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


def _check_turns(turns: list) -> None:
    if len(turns) == 0:
        raise LogValidationError("turns가 비어 있음")

    speakers = set()
    for i, t in enumerate(turns):
        for field, expected_type in REQUIRED_TURN_FIELDS.items():
            if field not in t:
                raise LogValidationError(f"turns[{i}]에 필수 필드 '{field}' 없음")
            if not isinstance(t[field], expected_type):
                raise LogValidationError(f"turns[{i}]의 '{field}' 타입 오류")
        speakers.add(t["speaker"])

        for c in t.get("codes", []):
            if c not in VALID_CODES:
                raise LogValidationError(
                    f"turns[{i}]에 알 수 없는 코드 '{c}' (허용: {VALID_CODES})"
                )

    if len(speakers) != 2:
        raise LogValidationError(f"화자는 정확히 2명이어야 함. 현재: {speakers}")