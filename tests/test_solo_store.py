"""
단독 산출물 저장/로드 테스트
==============================
save_solo_outputs / load_solo_outputs 왕복이 원문과 채점값을 보존하는지 확인.
외부 API를 호출하지 않는다 (API 키 없이 돌아가야 한다).
"""

import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from coop_pipeline.runner import (
    PipelineError,
    load_solo_outputs,
    save_solo_outputs,
)


def make_solo_outputs():
    """run_solo()가 반환하는 형태의 최소 산출물 목록."""
    return [
        {
            "scenario_id": "TEST_scenario",
            "task_type": "A1",
            "condition": "solo",
            "agent": agent,
            "model": f"{agent}-model",
            "replicate": rep,
            "output": f"{agent} 산출물 {rep}",
        }
        for agent in ("gemini", "llama")
        for rep in (1, 2)
    ]


def test_save_then_load_roundtrip(tmp_path):
    outputs = make_solo_outputs()
    path = save_solo_outputs(outputs, tmp_path, replicate=3,
                             solo_values=[0.5, 0.6, 0.7, 0.8])

    assert Path(path) == tmp_path / "solo" / "TEST_scenario_solo_3.json"

    saved = load_solo_outputs(tmp_path, "TEST_scenario", 3)
    assert saved["outputs"] == outputs
    assert saved["solo_values"] == [0.5, 0.6, 0.7, 0.8]
    assert saved["condition"] == "solo"
    assert saved["replicate"] == 3


def test_save_without_values_then_overwrite_with_values(tmp_path):
    # run_scenario는 대화 전에 값 없이 저장하고(체크포인트), 채점 후 값을 채워
    # 같은 파일에 다시 저장한다. 두 번째 저장이 첫 번째를 완전히 대체해야 한다.
    outputs = make_solo_outputs()
    save_solo_outputs(outputs, tmp_path, replicate=1)
    assert load_solo_outputs(tmp_path, "TEST_scenario", 1)["solo_values"] is None

    save_solo_outputs(outputs, tmp_path, replicate=1, solo_values=[1.0] * 4)
    assert load_solo_outputs(tmp_path, "TEST_scenario", 1)["solo_values"] == [1.0] * 4


def test_save_empty_outputs_fails(tmp_path):
    with pytest.raises(PipelineError):
        save_solo_outputs([], tmp_path, replicate=1)


def test_load_missing_file_fails_with_hint(tmp_path):
    with pytest.raises(PipelineError, match="run_solo_batch"):
        load_solo_outputs(tmp_path, "NO_SUCH", 1)
