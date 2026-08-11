# 파이프라인 실행 드라이버

"""
파이프라인 실행 드라이버 (batch runner)
========================================
시나리오 JSON 하나를 넣으면 협력 층위(L0~L4) 판정 결과가 나오는 진입점.

**이 파일은 오케스트레이터가 아니다.**
연구 설계상 dyad에는 중재자(orchestrator)를 두지 않기로 했고, 이 파일은 그 결정을
어기지 않는다. 둘의 차이는 "LLM이 제어 흐름에 개입하는가"이다.

    오케스트레이터 (우리가 쓰지 않는 것)
      - 세 번째 LLM이 누가 말할지, 언제 끝낼지, 무엇을 논의할지 결정한다
      - 에이전트 사이에서 메시지를 요약·중계·재작성한다
      - => 측정 대상이 'alpha와 beta의 협력'이 아니라 '중재자의 조율 능력'이 된다

    실행 드라이버 (이 파일)
      - 아래 (1)~(7)을 **고정된 순서로** 호출하기만 한다
      - 분기 조건이 전부 결정론적이다: task_type, condition, n_solo, max_turns
      - 제어 흐름에 LLM이 단 한 번도 개입하지 않는다
      - 대화 내용을 읽지도, 바꾸지도, 중계하지도 않는다

발화 순서 결정도 마찬가지다. agents.run_dyad()는 alpha/beta를 고정 교대시키고
(`turn % 2`), max_turns에 도달하면 멈춘다. 누가 말할지를 모델에게 묻지 않는다.

유일한 예외는 대화 종료 후 1회 실행되는 finalize 호출(agents.FINALIZE_PROMPT)이다.
이것은 고정 문자열 템플릿이며 LLM의 판단이 아니지만, 대화에 대한 개입인 것은 맞다.
"무엇을 채점할 것인가"를 정의하기 위해 의도적으로 넣은 것이고,
현재 설계에서는 alpha가 최종 산출물을 작성한다 (docs/RUBRIC.md의 채점 대상 정의 참고).

이 파일을 고칠 때 지켜야 할 선:
  - 여기(또는 agents.run_dyad)에서 LLM을 불러 다음 화자·종료 시점·논의 주제를
    정하게 만들면, 그 순간 오케스트레이터가 되고 회의록의 설계 결정이 깨진다.
  - 새 단계가 필요하면 LLM 판단이 아니라 결정론적 분기로 추가할 것.

    scenario.json
      1) run_solo x N       단독 조건 산출물
      2) run_dyad           2-agent 대화 로그 (raw) + final_output
      3) tag_log            발화 코드 + 참조 태깅
      4) score              A1: 체크리스트 / A2·A4: LLM judge 등급
      5) attach_scores      점수를 로그에 부착
      6) validate_log       형식 검증 (여기서 걸리면 위 단계가 빠진 것)
      7) classify           L0~L4 판정

사용법은 docs/PIPELINE.md 참고. 가장 짧은 형태:

    from coop_pipeline.runner import run_scenario
    result = run_scenario("scenarios/A1/A1_simple_meeting.json",
                          condition="명시",
                          api_keys={"gemini": ..., "groq": ..., "anthropic": ...})
    print(result["level"])
"""

import json
from pathlib import Path

from .agents import group_output_text, run_dyad, run_solo
from .classify import classify_log, format_result
from .scoring import attach_scores, detect_new_idea, score_a1, score_a2_a4
from .tagging import tag_log
from .thresholds import THRESHOLDS

SOLO_AGENTS = ("gemini", "llama")

# 단독 조건 표본 수 기본값.
# A1의 solo_p90은 표본이 작으면 사실상 최댓값이 되어 그룹이 이길 수 없다.
# 5는 최소선이고, 파일럿에서는 에이전트당 5회(=총 10개)를 권장한다.
DEFAULT_SOLO_REPS = 5


class PipelineError(RuntimeError):
    """파이프라인 단계에서 복구 불가능한 문제가 생겼을 때."""


# ---------------------------------------------------------------------------
# 시나리오 로드
# ---------------------------------------------------------------------------
def load_scenario(path) -> dict:
    """
    시나리오 JSON을 읽고, 실행 전에 확인할 수 있는 형식 오류를 미리 잡는다.
    (scenarios/README.md의 체크리스트를 코드로 옮긴 것)
    """
    path = Path(path)
    scenario = json.loads(path.read_text(encoding="utf-8"))

    problems = []
    for field in ("scenario_id", "task_type", "complexity", "task_variants"):
        if field not in scenario:
            problems.append(f"'{field}' 필드 없음")

    expected_id = path.stem
    if scenario.get("scenario_id") != expected_id:
        problems.append(
            f"scenario_id('{scenario.get('scenario_id')}')가 파일명('{expected_id}')과 불일치"
        )

    if scenario.get("task_type") not in ("A1", "A2", "A4"):
        problems.append(f"task_type이 A1/A2/A4가 아님: {scenario.get('task_type')!r}")

    variants = scenario.get("task_variants", {})
    if isinstance(variants, dict):
        if "solo" not in variants:
            problems.append("task_variants.solo 없음 (단독 조건을 실행할 수 없음)")
        if not (variants.get("shared") or (variants.get("alpha") and variants.get("beta"))):
            problems.append("task_variants에 shared 또는 (alpha, beta) 쌍이 없음")

    if scenario.get("task_type") == "A1":
        checklist = (scenario.get("scoring") or {}).get("checklist")
        if not checklist:
            problems.append("A1인데 scoring.checklist가 없음 (자동 채점 불가)")

    if problems:
        raise PipelineError(f"{path.name} 형식 오류:\n  - " + "\n  - ".join(problems))

    return scenario


def check_scenario_dir(scenario_dir="scenarios") -> dict:
    """
    시나리오 폴더 전체를 형식 검사만 한다 (API 호출 없음, 비용 0).
    새 시나리오를 추가한 뒤 가장 먼저 돌려볼 것.
    """
    ok, errors = [], {}
    for path in sorted(Path(scenario_dir).rglob("*.json")):
        try:
            load_scenario(path)
            ok.append(str(path))
        except PipelineError as exc:
            errors[str(path)] = str(exc)

    for path in ok:
        print(f"OK   {path}")
    for path, msg in errors.items():
        print(f"FAIL {path}\n     {msg}")
    print(f"\n결과: {len(ok)}개 통과 / {len(errors)}개 오류")
    return {"ok": ok, "errors": errors}


# ---------------------------------------------------------------------------
# 단계별 실행
# ---------------------------------------------------------------------------
def run_solo_batch(scenario: dict, api_keys: dict, n_reps: int = DEFAULT_SOLO_REPS,
                   agents=SOLO_AGENTS) -> list:
    """단독 조건을 agents x n_reps 회 실행해 산출물 목록을 만든다."""
    outputs = []
    for agent in agents:
        for rep in range(1, n_reps + 1):
            outputs.append(run_solo(
                scenario, agent, replicate=rep,
                gemini_api_key=api_keys.get("gemini"),
                groq_api_key=api_keys.get("groq"),
            ))
    return outputs


def score_outputs(scenario: dict, solo_outputs: list, group_text: str,
                  anthropic_key: str) -> dict:
    """
    단독/그룹 산출물을 채점한다.
    A1  : scoring.checklist 기반 결정론 채점 (API 호출 없음)
    A2/A4: LLM-judge 1~5점 (산출물 1개당 1회 호출)
    """
    task_type = scenario["task_type"]
    solo_texts = [o["output"] for o in solo_outputs]

    if task_type == "A1":
        checklist = scenario["scoring"]["checklist"]
        solo_values = [score_a1(text, checklist) for text in solo_texts]
        group_value = score_a1(group_text, checklist)
    else:
        solo_values = [score_a2_a4(text, anthropic_key) for text in solo_texts]
        group_value = score_a2_a4(group_text, anthropic_key)

    new_idea = detect_new_idea(solo_texts, group_text, anthropic_key)
    return {"solo_values": solo_values, "group_value": group_value,
            "new_idea_flag": new_idea}


# ---------------------------------------------------------------------------
# 전체 실행
# ---------------------------------------------------------------------------
def run_scenario(scenario_path, condition: str, api_keys: dict,
                 replicate: int = 1, n_solo: int = DEFAULT_SOLO_REPS,
                 max_turns: int = 10, thresholds: dict = THRESHOLDS,
                 out_dir=None, solo_outputs: list = None,
                 verbose: bool = True) -> dict:
    """
    시나리오 하나를 조건 하나로 끝까지 돌리고 판정 결과를 반환한다.

    api_keys : {"gemini": ..., "groq": ..., "anthropic": ...}
    solo_outputs: 이미 돌려둔 단독 산출물이 있으면 넘겨서 재사용한다
                  (같은 시나리오를 명시/묵시 두 조건으로 돌릴 때 비용이 절반이 된다)
    out_dir  : 지정하면 태깅·채점이 끝난 로그를 JSON으로 저장한다

    반환값은 classify_log()의 결과에 'log'와 'solo_outputs'가 추가된 딕셔너리.
    """
    scenario = load_scenario(scenario_path)

    def step(msg):
        if verbose:
            print(f"[{scenario['scenario_id']}/{condition}] {msg}")

    # 1) 단독 조건
    if solo_outputs is None:
        step(f"단독 조건 실행 ({len(SOLO_AGENTS)} 에이전트 x {n_solo}회)")
        solo_outputs = run_solo_batch(scenario, api_keys, n_reps=n_solo)

    # 2) 2-agent 대화
    step(f"2-agent 대화 실행 ({max_turns}턴)")
    log = run_dyad(
        scenario, condition,
        gemini_api_key=api_keys["gemini"], groq_api_key=api_keys["groq"],
        replicate=replicate, max_turns=max_turns,
    )

    # 2-1) 원본 저장 (체크포인트)
    # 여기까지가 가장 비싼 단계다. 뒤 단계에서 실패해도 대화 로그를 잃지 않도록
    # 태깅 전에 먼저 저장해 둔다. 나중에 tag_log()부터 이어서 돌릴 수 있다.
    if out_dir:
        save_log(log, Path(out_dir) / "raw")

    # 3) 태깅
    step("발화 태깅")
    log = tag_log(log, api_key=api_keys["anthropic"])

    # 4~5) 채점 + 부착
    step("채점")
    group_text = group_output_text(log)
    scores = score_outputs(scenario, solo_outputs, group_text, api_keys["anthropic"])
    log["group_output_text"] = group_text
    log = attach_scores(log, scores["solo_values"], scores["group_value"],
                        scores["new_idea_flag"])

    # 6~7) 검증 + 판정
    step("판정")
    result = classify_log(log, thresholds=thresholds)
    result["log"] = log
    result["solo_outputs"] = solo_outputs

    if out_dir:
        result["log_path"] = save_log(log, out_dir)

    if verbose:
        print(format_result(result))

    return result


def run_scenario_both_conditions(scenario_path, api_keys: dict, **kwargs) -> dict:
    """
    같은 시나리오를 명시/묵시 두 조건으로 돌린다.
    단독 산출물은 한 번만 생성해 두 조건이 공유한다 (조건 간 비교의 기준선을 통일).
    """
    scenario = load_scenario(scenario_path)
    n_solo = kwargs.pop("n_solo", DEFAULT_SOLO_REPS)
    solo_outputs = run_solo_batch(scenario, api_keys, n_reps=n_solo)

    return {
        cond: run_scenario(scenario_path, cond, api_keys,
                           solo_outputs=solo_outputs, **kwargs)
        for cond in ("명시", "묵시")
    }


# ---------------------------------------------------------------------------
# 저장 / 재분석
# ---------------------------------------------------------------------------
def save_log(log: dict, out_dir) -> str:
    """
    README의 파일명 규칙 {scenario_id}_{condition}_{rep}.json 으로 저장한다.
    로그는 Git이 아니라 Drive 공유 폴더에 저장할 것.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{log['scenario_id']}_{log['condition']}_{log.get('replicate', 1)}.json"
    path = out_dir / name
    path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def classify_saved_log(path, thresholds: dict = THRESHOLDS, verbose: bool = True) -> dict:
    """
    저장된 로그를 다시 판정한다. API 호출이 전혀 없으므로 비용이 0이다.
    임계값을 바꿔가며 결과가 어떻게 달라지는지 볼 때(캘리브레이션) 쓴다.
    """
    log = json.loads(Path(path).read_text(encoding="utf-8"))
    result = classify_log(log, thresholds=thresholds)
    if verbose:
        print(format_result(result))
    return result


def classify_saved_dir(log_dir, thresholds: dict = THRESHOLDS) -> list:
    """저장된 로그 폴더 전체를 재판정하고 요약 표를 출력한다."""
    results = []
    for path in sorted(Path(log_dir).glob("*.json")):
        try:
            results.append(classify_saved_log(path, thresholds, verbose=False))
        except Exception as exc:
            print(f"SKIP {path.name}: {exc}")

    print(f"{'시나리오':<34}{'조건':<8}{'rep':<5}{'층위':<5}병목")
    print("-" * 70)
    for r in results:
        print(f"{str(r['scenario_id']):<34}{str(r['condition']):<8}"
              f"{str(r['replicate']):<5}{r['level']:<5}{r['stopped_at'] or '-'}")
    return results
