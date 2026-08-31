"""
coop_pipeline 자동 테스트
==============================
로직을 고칠 때마다 `pytest tests/`로 돌려서 기존 케이스가 안 깨졌는지 확인.

이 파일은 외부 API를 호출하지 않는다 (API 키 없이 돌아가야 한다).
"""

import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from coop_pipeline import classify_log, load_thresholds
from coop_pipeline.scoring import normalize, score_a1
from coop_pipeline import scoring
from coop_pipeline.validate_log import LogValidationError, validate_log
from coop_pipeline import agents, llm


def make_base_log(**overrides):
    """테스트용 최소 로그를 만들고, overrides로 필요한 부분만 덮어쓴다."""
    log = {
        "scenario_id": "TEST",
        "task_type": "A1",
        "condition": "명시",
        "replicate": 1,
        "turns": [
            {"turn": 1, "speaker": "alpha", "text": "...", "codes": ["arch"], "ref": None},
            {"turn": 2, "speaker": "beta", "text": "...", "codes": ["agree"], "ref": "alpha"},
        ],
        "group_score": 0.9,
        "solo_scores": [0.5, 0.6, 0.55, 0.7, 0.6],
        "new_idea_flag": False,
    }
    log.update(overrides)
    return log


# ---------------------------------------------------------------------------
# 결정 트리
# ---------------------------------------------------------------------------
def test_l0_no_reference():
    log = make_base_log(turns=[
        {"turn": 1, "speaker": "alpha", "text": "...", "codes": [], "ref": None},
        {"turn": 2, "speaker": "beta", "text": "...", "codes": [], "ref": None},
    ])
    result = classify_log(log)
    assert result["level"] == "L0"
    assert result["stopped_at"] == "Q1"


def test_l1_one_directional():
    log = make_base_log(turns=[
        {"turn": 1, "speaker": "alpha", "text": "...", "codes": ["arch"], "ref": None},
        {"turn": 2, "speaker": "beta", "text": "...", "codes": ["agree"], "ref": "alpha"},
    ])
    result = classify_log(log)
    assert result["level"] == "L1"
    assert result["stopped_at"] == "Q2"


def test_l2_group_does_not_beat_solo():
    """쌍방향 참조는 충분하지만 그룹 점수가 단독 p90을 못 넘는 경우."""
    turns = [
        {"turn": 1, "speaker": "alpha", "text": "...", "codes": ["arch"], "ref": None},
        {"turn": 2, "speaker": "beta", "text": "...", "codes": ["agree"], "ref": "alpha"},
        {"turn": 3, "speaker": "alpha", "text": "...", "codes": ["agree"], "ref": "beta"},
        {"turn": 4, "speaker": "beta", "text": "...", "codes": ["comp"], "ref": "alpha"},
        {"turn": 5, "speaker": "alpha", "text": "...", "codes": ["agree"], "ref": "beta"},
    ]
    log = make_base_log(turns=turns, group_score=0.5)
    result = classify_log(log)
    assert result["level"] == "L2"
    assert result["stopped_at"] == "Q3"


def test_l3_when_new_idea_missing():
    """집단 우위는 있으나 신규 해결책이 없으면 L4가 아니라 L3."""
    turns = [
        {"turn": 1, "speaker": "alpha", "text": "...", "codes": ["arch"], "ref": None},
        {"turn": 2, "speaker": "beta", "text": "...", "codes": ["agree"], "ref": "alpha"},
        {"turn": 3, "speaker": "alpha", "text": "...", "codes": ["agree"], "ref": "beta"},
        {"turn": 4, "speaker": "beta", "text": "...", "codes": ["comp"], "ref": "alpha"},
        {"turn": 5, "speaker": "alpha", "text": "...", "codes": ["agree", "comp"], "ref": "beta"},
    ]
    log = make_base_log(turns=turns, group_score=0.9, new_idea_flag=False)
    result = classify_log(log)
    assert result["level"] == "L3"
    assert result["checks"]["Q4b_신규해결책"]["passed"] is False


def test_l4_full_synergy():
    turns = [
        {"turn": 1, "speaker": "alpha", "text": "...", "codes": ["arch"], "ref": None},
        {"turn": 2, "speaker": "beta", "text": "...", "codes": ["agree", "arch"], "ref": "alpha"},
        {"turn": 3, "speaker": "alpha", "text": "...", "codes": ["agree"], "ref": "beta"},
        {"turn": 4, "speaker": "beta", "text": "...", "codes": ["comp"], "ref": "alpha"},
        {"turn": 5, "speaker": "alpha", "text": "...", "codes": ["agree", "comp"], "ref": "beta"},
    ]
    log = make_base_log(turns=turns, group_score=0.9, new_idea_flag=True)
    result = classify_log(log)
    assert result["level"] == "L4"
    assert result["stopped_at"] is None


def test_a2_uses_grade_gap_not_score():
    log = make_base_log(
        task_type="A2",
        turns=[
            {"turn": 1, "speaker": "alpha", "text": "...", "codes": ["arch"], "ref": None},
            {"turn": 2, "speaker": "beta", "text": "...", "codes": ["agree", "arch"], "ref": "alpha"},
            {"turn": 3, "speaker": "alpha", "text": "...", "codes": ["agree"], "ref": "beta"},
            {"turn": 4, "speaker": "beta", "text": "...", "codes": ["comp"], "ref": "alpha"},
            {"turn": 5, "speaker": "alpha", "text": "...", "codes": ["agree"], "ref": "beta"},
        ],
    )
    del log["group_score"]
    del log["solo_scores"]
    log["group_grade"] = 4
    log["solo_grades"] = [2, 2, 3, 2, 3]

    result = classify_log(log)
    assert result["level"] in {"L2", "L3", "L4"}


# ---------------------------------------------------------------------------
# 로그 검증
# ---------------------------------------------------------------------------
def test_validation_error_on_missing_field():
    log = make_base_log()
    del log["group_score"]
    with pytest.raises(LogValidationError):
        classify_log(log)


def test_untagged_log_is_rejected_not_silently_l0():
    """
    태깅을 건너뛴 로그가 조용히 L0으로 판정되면 '협력이 없었다'와 구별할 수 없다.
    반드시 에러로 막혀야 한다.
    """
    log = make_base_log(turns=[
        {"turn": 1, "speaker": "alpha", "text": "..."},
        {"turn": 2, "speaker": "beta", "text": "..."},
    ])
    with pytest.raises(LogValidationError):
        classify_log(log)


def test_raw_log_passes_when_tags_not_required():
    log = make_base_log(turns=[
        {"turn": 1, "speaker": "alpha", "text": "..."},
        {"turn": 2, "speaker": "beta", "text": "..."},
    ])
    validate_log(log, require_tags=False)  # 예외가 나지 않아야 한다


def test_a1_exchange_protocol_is_applied_only_to_a1():
    task = "개별 과제 지문"
    a1 = agents._apply_task_protocol(task, "A1")

    assert task in a1
    assert "첫 발화에서" in a1
    assert "상대가 공유한 사실·ID" in a1
    assert agents._apply_task_protocol(task, "A2") == task
    assert agents._apply_task_protocol(task, "A4") == task


def test_self_reference_is_rejected():
    log = make_base_log(turns=[
        {"turn": 1, "speaker": "alpha", "text": "...", "codes": [], "ref": "alpha"},
        {"turn": 2, "speaker": "beta", "text": "...", "codes": [], "ref": None},
    ])
    with pytest.raises(LogValidationError):
        validate_log(log)


# ---------------------------------------------------------------------------
# A1 채점
# ---------------------------------------------------------------------------
def test_normalize_absorbs_formatting_noise():
    assert normalize("19:00") == normalize("19 : 00") == "1900"
    assert normalize("panel-3") == "panel3"
    assert normalize("B 실") == "b실"


def test_score_a1_short_keywords_exact():
    output = "수요일 19:00에 B실을 예약하는 것이 좋겠습니다."
    assert score_a1(output, ["수요일", "19:00", "B실"]) == 1.0


def test_score_a1_gives_zero_when_nothing_matches():
    assert score_a1("전혀 다른 이야기", ["수요일", "19:00", "B실"]) == 0.0


def test_score_a1_partial_credit_for_sentence_checklist():
    """
    문장형 체크리스트(A1_complex_power_outage)가 0점으로 고정되지 않아야 한다.
    항목 전체 일치만 인정하면 이 케이스는 구조적으로 0점이 되고,
    판정이 항상 L2에 고정돼 실험이 무의미해진다.
    """
    checklist = ["C동 전기실 panel-3 과부하 차단", "대형 장비 동시 기동", "D10", "D14"]
    output = (
        "최종 원인은 C동 전기실의 panel-3 차단기가 과부하로 내려간 것입니다. "
        "3층 실험실에서 대형 장비를 동시에 기동한 것이 방아쇠였습니다. "
        "근거 단서: D10, D14."
    )
    score = score_a1(output, checklist)
    assert score > 0.9, f"문장형 체크리스트가 제대로 채점되지 않음 (score={score})"

    # 엄격 모드에서도 토큰이 전부 등장하면 만점이어야 한다
    assert score_a1(output, checklist, partial=False) == 1.0


def test_score_a1_empty_checklist():
    assert score_a1("아무 말", []) == 0.0


# ---------------------------------------------------------------------------
# 임계값 설정
# ---------------------------------------------------------------------------
def test_thresholds_are_loaded_from_configs():
    """configs/thresholds_v1.json이 실제로 읽히는지 (死파일이 아닌지) 확인."""
    values = load_thresholds("v1")
    assert set(values) == {
        "bidirectional_min", "solo_percentile_A1",
        "grade_gap_min", "comp_ratio_min", "revision_seq_min",
    }
    assert "version" not in values and "note" not in values


def test_threshold_change_can_flip_the_verdict():
    """임계값만 바꿔도 판정이 달라져야 캘리브레이션이 의미를 갖는다."""
    # alpha -> beta 2회, beta -> alpha 2회
    turns = [
        {"turn": 1, "speaker": "alpha", "text": "...", "codes": ["arch"], "ref": None},
        {"turn": 2, "speaker": "beta", "text": "...", "codes": ["agree"], "ref": "alpha"},
        {"turn": 3, "speaker": "alpha", "text": "...", "codes": ["agree"], "ref": "beta"},
        {"turn": 4, "speaker": "beta", "text": "...", "codes": ["agree"], "ref": "alpha"},
        {"turn": 5, "speaker": "alpha", "text": "...", "codes": ["agree"], "ref": "beta"},
    ]
    log = make_base_log(turns=turns)

    # 기준을 3회로 올리면 편측(L1)으로 떨어지고
    strict = dict(load_thresholds("v1"), bidirectional_min=3)
    assert classify_log(log, thresholds=strict)["level"] == "L1"

    # 2회로 내리면 Q2를 통과해 더 높은 층위로 올라간다
    loose = dict(load_thresholds("v1"), bidirectional_min=2)
    assert classify_log(log, thresholds=loose)["level"] != "L1"


# ---------------------------------------------------------------------------
# 요청 예산 / 413 처리
# ---------------------------------------------------------------------------
class _TooLarge(Exception):
    status_code = 413


class _Message:
    content = '{"codes": [], "reference": null, "evidence": {}}'


class _Completion:
    choices = [type("Choice", (), {"message": _Message()})()]


class _JudgeCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            raise _TooLarge("too large")
        return _Completion()


def test_groq_judge_lowers_output_budget_once_on_413(monkeypatch):
    monkeypatch.setattr(llm, "_MIN_OUTPUT_TOKENS", 1024)
    client = type("Client", (), {"chat": type("Chat", (), {"completions": _JudgeCompletions()})()})()
    judge = llm.JudgeClient("groq", client, "qwen/qwen3.6-27b")

    assert judge.json("system", "user") == {"codes": [], "reference": None, "evidence": {}}
    assert [call["max_tokens"] for call in client.chat.completions.calls] == [1024, 256]


def test_413_is_not_retried():
    calls = 0

    def always_too_large():
        nonlocal calls
        calls += 1
        raise _TooLarge("too large")

    with pytest.raises(llm.LLMCallError, match="모델 한도를 넘음"):
        llm.with_retry(always_too_large, what="test")
    assert calls == 1


def test_tpd_429_is_not_retried():
    calls = 0

    def daily_limit():
        nonlocal calls
        calls += 1
        raise RuntimeError("tokens per day (TPD): Limit 200000")

    with pytest.raises(llm.LLMCallError, match="일일 토큰 한도"):
        llm.with_retry(daily_limit, what="test")
    assert calls == 1


def test_judge_stops_after_its_single_budget_fallback():
    class Completions:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            raise _TooLarge("too large")

    completions = Completions()
    client = type("Client", (), {"chat": type("Chat", (), {"completions": completions})()})()
    judge = llm.JudgeClient("groq", client, "qwen/qwen3.6-27b")

    with pytest.raises(llm.LLMCallError, match="user=4자"):
        judge.json("system", "user")
    assert [call["max_tokens"] for call in completions.calls] == [512, 256]


class _JsonModeFailure(Exception):
    """Groq가 json_object 모드에서 유효한 JSON을 못 받았을 때의 400."""

    status_code = 400
    body = {"error": {"message": "Failed to generate JSON. Please adjust your prompt.",
                      "failed_generation": '{"codes": ["arch"], "evidence": {"arch": "D14'}}

    def __str__(self):
        return "Error code: 400 - Failed to generate JSON. Please adjust your prompt."


def test_groq_judge_raises_budget_then_drops_json_mode(monkeypatch):
    """잘림(예산)과 군말(형식) 중 어느 쪽인지 모르므로 순서대로 둘 다 시도한다."""
    calls = []

    class Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            if "response_format" in kwargs:
                raise _JsonModeFailure()
            return _Completion()   # JSON 모드를 끄니 문자열이 그대로 온다

    client = type("Client", (), {"chat": type("Chat", (), {"completions": Completions()})()})()
    judge = llm.JudgeClient("groq", client, "qwen/qwen3.6-27b")

    assert judge.json("system", "user") == {"codes": [], "reference": None, "evidence": {}}
    assert [c["max_tokens"] for c in calls] == [512, 1024, 1024]
    assert [("response_format" in c) for c in calls] == [True, True, False]


def test_json_mode_failure_is_not_retried_and_keeps_failed_generation():
    """확정적 실패다. 3회 재시도하면 TPD만 3배로 나간다."""
    calls = []

    class Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            raise _JsonModeFailure()

    client = type("Client", (), {"chat": type("Chat", (), {"completions": Completions()})()})()
    judge = llm.JudgeClient("groq", client, "qwen/qwen3.6-27b")

    with pytest.raises(llm.LLMCallError, match="D14"):
        judge.json("system", "user")
    # 512 -> 1024 -> json_mode 해제. 그 뒤로는 재시도하지 않는다.
    assert len(calls) == 3


def test_beta_call_has_bounded_output_and_gpt_reasoning(monkeypatch):
    calls = []

    class Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return _Completion()

    client = type("Client", (), {"chat": type("Chat", (), {"completions": Completions()})()})()
    monkeypatch.setattr(agents, "BETA_MAX_TOKENS", 1024)

    assert agents._call_llama(client, "prompt", model_id="openai/gpt-oss-20b")
    assert calls[0]["max_tokens"] == 1024
    assert calls[0]["reasoning_effort"] == "low"


def test_new_idea_prompt_is_bounded_but_keeps_each_output_ends(monkeypatch):
    captured = {}

    def fake_call(_client, system, user, max_tokens, min_output_tokens=None):
        captured.update(system=system, user=user, max_tokens=max_tokens,
                        min_output_tokens=min_output_tokens)
        return {"new_idea": False, "reason": ""}

    monkeypatch.setattr(scoring, "make_judge", lambda *args, **kwargs: object())
    monkeypatch.setattr(scoring, "call_judge_json", fake_call)
    monkeypatch.setattr(scoring, "NEW_IDEA_SOLO_CHARS", 50)
    monkeypatch.setattr(scoring, "NEW_IDEA_GROUP_CHARS", 100)

    solos = [f"start-{i}-" + "x" * 200 + f"-end-{i}" for i in range(10)]
    group = "group-start-" + "y" * 300 + "-group-end"
    result = scoring.detect_new_idea_detail(solos, group, "key")

    assert result["new_idea"] is False
    assert len(captured["user"]) <= 10 * 50 + 100 + 100
    assert "start-0-" in captured["user"] and "-end-0" in captured["user"]
    assert "group-start-" in captured["user"] and "-group-end" in captured["user"]
