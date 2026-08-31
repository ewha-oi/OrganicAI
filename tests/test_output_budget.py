"""
산출물 예산 · 절단 감지 · 근거 저장 · 단독 재사용 테스트
=========================================================
전부 rep=1 수집에서 **조용히** 데이터를 망가뜨린 경로다. 조용했다는 것이
핵심이라, 다시 조용해지지 않도록 여기서 못을 박는다. 외부 API를 호출하지 않는다.
"""

import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from coop_pipeline import agents, runner, scoring


# ---------------------------------------------------------------------------
# 1. 대시 정규화 — A1 체크리스트가 U+2011 때문에 영원히 빗나가던 문제
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("written", [
    "panel-3",   # U+002D  HYPHEN-MINUS      (체크리스트 표기)
    "panel‑3",  # U+2011 NON-BREAKING HYPHEN   (alpha가 실제로 쓴 것)
    "panel‐3",  # U+2010 HYPHEN         (NFKC가 U+2011을 옮기는 곳)
    "panel–3",  # U+2013 EN DASH
    "panel—3",  # U+2014 EM DASH
    "panel−3",  # U+2212 MINUS SIGN
    "panel－3",  # U+FF0D FULLWIDTH HYPHEN-MINUS
    "panel­3",  # U+00AD SOFT HYPHEN
])
def test_all_unicode_dashes_normalize_to_the_same_token(written):
    assert scoring.normalize(written) == "panel3"


def test_checklist_matches_output_written_with_non_breaking_hyphen():
    # 실제 로그(A1_complex_power_outage)에서 그대로 가져온 형태.
    checklist = ["C동 전기실 panel-3 과부하 차단"]
    output = "최종 원인은 C동 전기실의 panel‑3 차단기가 과부하로 내려간 것입니다."
    assert scoring.score_a1(output, checklist) == 1.0


# ---------------------------------------------------------------------------
# 2. 절단 감지 — 잘린 산출물이 채점 대상이 되던 문제
# ---------------------------------------------------------------------------
class _Choice:
    def __init__(self, text, finish_reason):
        self.message = type("Msg", (), {"content": text})()
        self.finish_reason = finish_reason


class _Completion:
    def __init__(self, text="응답", finish_reason="stop"):
        self.choices = [_Choice(text, finish_reason)]


def _client(responses):
    """responses를 순서대로 돌려주는 가짜 Groq 클라이언트."""
    calls = []

    class Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return responses[min(len(calls) - 1, len(responses) - 1)]

    client = type("Client", (), {
        "chat": type("Chat", (), {"completions": Completions()})()})()
    return client, calls


def test_finish_reason_length_is_detected_as_truncated():
    assert agents._is_truncated(_Completion(finish_reason="length"))
    assert not agents._is_truncated(_Completion(finish_reason="stop"))


def test_gemini_style_enum_finish_reason_is_detected():
    """google-genai는 candidates[0].finish_reason에 Enum을 넣는다."""
    reason = type("FinishReason", (), {"name": "MAX_TOKENS"})()
    response = type("R", (), {"candidates": [type("C", (), {
        "finish_reason": reason})()]})()
    assert agents._is_truncated(response)


def test_unknown_finish_reason_is_not_treated_as_truncated():
    """사유를 알 수 없으면 끊기지 않은 것으로 본다 (없는 절단을 만들지 않는다)."""
    assert not agents._is_truncated(object())


def test_scoring_target_raises_budget_then_succeeds(capsys):
    responses = [_Completion("잘린 글", "length"), _Completion("온전한 글", "stop")]
    client, calls = _client(responses)

    text = agents._generate_scoring_target(
        lambda budget: agents._call_llama(client, "prompt", max_tokens=budget),
        (1024, 3072), "단독 산출물")

    assert text == "온전한 글"
    assert [c["max_tokens"] for c in calls] == [1024, 3072]
    assert "잘렸다" in capsys.readouterr().out


def test_scoring_target_never_returns_a_truncated_text():
    """마지막 예산에서도 잘리면 저장하지 않고 실패시킨다.

    이것이 이 파일 전체에서 가장 중요한 단언이다. 잘린 산출물을 통과시키면
    A1은 뒤쪽 체크리스트를, A2/A4는 judge의 '요구 항목 누락' 판정을 받는데,
    둘 다 실험 결과와 겉모습이 같아서 사후에 구별할 방법이 없다.
    """
    client, _ = _client([_Completion("끝까지 잘린 글", "length")])

    with pytest.raises(agents.TruncatedOutputError, match="채점 대상"):
        agents._generate_scoring_target(
            lambda budget: agents._call_llama(client, "prompt", max_tokens=budget),
            (1024, 3072), "단독 산출물")


def test_turn_budget_and_output_budget_are_not_the_same_value():
    """둘을 한 값으로 묶으면 rep=1의 절단이 그대로 재발한다."""
    assert agents.TURN_MAX_TOKENS < min(agents.SOLO_MAX_TOKENS)
    assert agents.TURN_MAX_TOKENS < min(agents.FINALIZE_MAX_TOKENS)


def test_solo_uses_output_budget_not_turn_budget(monkeypatch):
    client, calls = _client([_Completion("단독 산출물", "stop")])
    monkeypatch.setattr(agents, "_groq_client", lambda key: client)

    scenario = {"scenario_id": "T", "task_type": "A1",
                "task_variants": {"solo": "과제 지문", "shared": "과제 지문"}}
    agents.run_solo(scenario, "llama", groq_api_key="key")

    assert calls[0]["max_tokens"] == agents.SOLO_MAX_TOKENS[0]


# ---------------------------------------------------------------------------
# 3. 근거 저장 — judge가 써 보낸 reason을 버리던 문제
# ---------------------------------------------------------------------------
def test_attach_scores_keeps_judge_reasons():
    log = {"task_type": "A2"}
    scoring.attach_scores(log, [2, 3], 4, True,
                          reasons={"group": "트레이드오프까지 조율함",
                                   "solo": ["피상적", "한쪽만"],
                                   "new_idea": "새 판단 기준이 등장"})

    assert log["group_grade_reason"] == "트레이드오프까지 조율함"
    assert log["solo_grade_reasons"] == ["피상적", "한쪽만"]
    assert log["new_idea_reason"] == "새 판단 기준이 등장"


def test_a1_gets_new_idea_reason_but_no_grade_reason():
    """A1은 코드로 채점한다. 빈 grade_reason을 넣으면 'judge가 근거를 안 줬다'와
    '애초에 judge를 안 썼다'가 구별되지 않는다."""
    log = {"task_type": "A1"}
    scoring.attach_scores(log, [0.5], 0.7, False, reasons={"new_idea": "없음"})

    assert log["new_idea_reason"] == "없음"
    assert "group_grade_reason" not in log
    assert "solo_grade_reasons" not in log


# ---------------------------------------------------------------------------
# 4. 단독 재사용 — rep마다 새로 뽑고 버리던 문제
# ---------------------------------------------------------------------------
SCENARIO = {"scenario_id": "T_scenario", "task_type": "A2",
            "task_variants": {"solo": "단독 지문", "shared": "공유 지문"}}


def _outputs(n=4):
    return [{"scenario_id": "T_scenario", "task_type": "A2", "condition": "solo",
             "agent": "gemini", "model": "m", "replicate": 1,
             "output": f"산출물 {i}"} for i in range(n)]


def test_reuses_solo_across_reps(tmp_path):
    runner.save_solo_outputs(_outputs(), tmp_path, replicate=1,
                             solo_values=[2, 2, 3, 3],
                             solo_reasons=["a", "b", "c", "d"],
                             solo_task=SCENARIO["task_variants"]["solo"])

    found = runner.find_reusable_solo(tmp_path, SCENARIO, n_solo=2)
    assert found is not None
    assert found["solo_values"] == [2, 2, 3, 3]
    assert found["solo_reasons"] == ["a", "b", "c", "d"]
    assert found["replicate"] == 1     # rep 2·3도 rep 1의 파일에 덮어써야 한다


def test_does_not_reuse_when_solo_task_changed(tmp_path):
    """시나리오의 단독 지문이 바뀌면 옛 기준선을 쓰면 안 된다."""
    runner.save_solo_outputs(_outputs(), tmp_path, replicate=1,
                             solo_values=[2, 2, 3, 3],
                             solo_task="옛 지문")

    assert runner.find_reusable_solo(tmp_path, SCENARIO, n_solo=2) is None


def test_does_not_reuse_checkpoint_without_scores(tmp_path):
    """채점 전 체크포인트는 기준선이 될 수 없다."""
    runner.save_solo_outputs(_outputs(), tmp_path, replicate=1,
                             solo_task=SCENARIO["task_variants"]["solo"])

    assert runner.find_reusable_solo(tmp_path, SCENARIO, n_solo=2) is None


def test_does_not_reuse_when_n_solo_differs(tmp_path):
    runner.save_solo_outputs(_outputs(), tmp_path, replicate=1,
                             solo_values=[2, 2, 3, 3],
                             solo_task=SCENARIO["task_variants"]["solo"])

    assert runner.find_reusable_solo(tmp_path, SCENARIO, n_solo=5) is None


def test_does_not_reuse_pre_fingerprint_files(tmp_path):
    """지문 해시가 없는 파일 = 출력 예산 1024 시절의 산출물. 잘렸을 수 있다."""
    import json

    solo_dir = tmp_path / "solo"
    solo_dir.mkdir()
    (solo_dir / "T_scenario_solo_1.json").write_text(json.dumps({
        "scenario_id": "T_scenario", "condition": "solo", "replicate": 1,
        "outputs": _outputs(), "solo_values": [2, 2, 3, 3],
    }, ensure_ascii=False), encoding="utf-8")

    assert runner.find_reusable_solo(tmp_path, SCENARIO, n_solo=2) is None


def test_checkpoint_and_final_save_share_one_file_per_scenario(tmp_path):
    """run_scenario는 대화 전(체크포인트)과 채점 후 두 번 저장한다.

    두 저장이 서로 다른 rep 번호를 쓰면 재사용할 때마다 단독 파일이 하나씩
    늘어난다. 인간 채점자는 그중 어느 것이 Q3의 기준선인지 알 수 없게 된다.
    """
    # 체크포인트: 값 없이, 재사용된 rep(1)로 저장
    runner.save_solo_outputs(_outputs(), tmp_path, 1,
                             solo_task=SCENARIO["task_variants"]["solo"])
    # 채점 후: 같은 rep으로 덮어쓰기
    runner.save_solo_outputs(_outputs(), tmp_path, 1, solo_values=[2, 2, 3, 3],
                             solo_task=SCENARIO["task_variants"]["solo"])

    files = sorted(p.name for p in (tmp_path / "solo").glob("*.json"))
    assert files == ["T_scenario_solo_1.json"]
    assert runner.find_reusable_solo(tmp_path, SCENARIO, n_solo=2)["solo_values"]


def test_reuse_prefers_lowest_rep_numerically(tmp_path):
    for rep in (2, 10, 1):
        runner.save_solo_outputs(_outputs(), tmp_path, replicate=rep,
                                 solo_values=[rep, rep, rep, rep],
                                 solo_task=SCENARIO["task_variants"]["solo"])

    found = runner.find_reusable_solo(tmp_path, SCENARIO, n_solo=2)
    assert found["replicate"] == 1     # 문자열 정렬이면 10이 잡힌다
