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
from .classify import classify_log, format_result, min_turns_for_bidirectional
from .llm import judge_key_name, judge_model, judge_provider
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


def judge_key(api_keys: dict) -> str:
    """
    현재 judge 제공사(COOP_JUDGE_PROVIDER)에 맞는 API 키를 api_keys에서 꺼낸다.

    judge를 anthropic -> groq로 바꾸면 필요한 키도 바뀐다. 호출부가
    api_keys["anthropic"]을 직접 참조하면 제공사를 바꿀 때마다 코드를 고쳐야 하므로
    여기 한 곳에서만 결정한다.
    """
    name = judge_key_name()
    key = api_keys.get(name)
    if not key:
        raise PipelineError(
            f"judge 제공사가 '{judge_provider()}'인데 api_keys['{name}']가 비어 있음.\n"
            f"  - 키를 넣거나\n"
            f"  - 다른 제공사를 쓰려면 임포트 전에 "
            f"os.environ['COOP_JUDGE_PROVIDER']를 바꾸고 런타임을 재시작할 것"
        )
    return key


def score_outputs(scenario: dict, solo_outputs: list, group_text: str,
                  judge_api_key: str, solo_values: list = None) -> dict:
    """
    단독/그룹 산출물을 채점한다.
    A1  : scoring.checklist 기반 결정론 채점 (API 호출 없음)
    A2/A4: LLM-judge 1~5점 (산출물 1개당 1회 호출)

    judge_api_key는 judge_key(api_keys)로 얻는다 (제공사에 따라 어느 키인지 달라진다).

    solo_values: 이미 채점해 둔 단독 점수가 있으면 넘겨서 재사용한다.
        명시/묵시는 같은 단독 산출물을 공유하는데, 조건마다 다시 채점하면
        (1) judge 호출을 10회씩 두 번 내고 (2) 같은 글에 다른 등급이 나와
        두 조건의 기준선이 어긋난다. 한 번 채점해 양쪽이 그대로 쓰는 것이 맞다.
    """
    task_type = scenario["task_type"]
    solo_texts = [o["output"] for o in solo_outputs]

    if task_type == "A1":
        checklist = scenario["scoring"]["checklist"]
        if solo_values is None:
            solo_values = [score_a1(text, checklist) for text in solo_texts]
        group_value = score_a1(group_text, checklist)
    else:
        if solo_values is None:
            solo_values = [score_a2_a4(text, judge_api_key, task_type)
                           for text in solo_texts]
        group_value = score_a2_a4(group_text, judge_api_key, task_type)

    new_idea = detect_new_idea(solo_texts, group_text, judge_api_key)
    return {"solo_values": solo_values, "group_value": group_value,
            "new_idea_flag": new_idea}


# ---------------------------------------------------------------------------
# 전체 실행
# ---------------------------------------------------------------------------
def run_scenario(scenario_path, condition: str, api_keys: dict,
                 replicate: int = 1, n_solo: int = DEFAULT_SOLO_REPS,
                 max_turns: int = 10, thresholds: dict = THRESHOLDS,
                 out_dir=None, solo_outputs: list = None,
                 solo_values: list = None, verbose: bool = True) -> dict:
    """
    시나리오 하나를 조건 하나로 끝까지 돌리고 판정 결과를 반환한다.

    api_keys : {"gemini": ..., "groq": ..., "anthropic": ...}
    solo_outputs: 이미 돌려둔 단독 산출물이 있으면 넘겨서 재사용한다
                  (같은 시나리오를 명시/묵시 두 조건으로 돌릴 때 비용이 절반이 된다)
    solo_values : 이미 채점해 둔 단독 점수. 넘기면 judge 재호출을 건너뛴다.
                  반환값의 'solo_values'로 나오므로 다음 조건에 그대로 넘기면 된다.
    out_dir  : 지정하면 태깅·채점이 끝난 로그를 JSON으로 저장한다

    반환값은 classify_log()의 결과에 'log'와 'solo_outputs'가 추가된 딕셔너리.
    """
    scenario = load_scenario(scenario_path)
    jkey = judge_key(api_keys)   # 키가 없으면 대화를 돌리기 전에 여기서 멈춘다

    def step(msg):
        if verbose:
            print(f"[{scenario['scenario_id']}/{condition}] {msg}")

    if verbose:
        step(f"judge = {judge_provider()}:{judge_model()}")

    # 0) 설정 점검 — 비싼 호출을 시작하기 전에 알려야 의미가 있다.
    #    턴 수가 모자라면 Q2는 대화 내용과 무관하게 실패하고, 그 L0/L1은
    #    '협력이 없었다'는 결과와 겉모습이 같아 구별되지 않는다.
    min_turns = min_turns_for_bidirectional(thresholds)
    if max_turns < min_turns:
        print(
            f"\n{'!' * 62}\n"
            f"설정 경고: max_turns={max_turns}에서는 Q2가 통과할 수 없다\n"
            f"  한 화자의 참조 가능 횟수 최대 {max(0, -(-max_turns // 2) - 1)}회 "
            f"< bidirectional_min={thresholds['bidirectional_min']}\n"
            f"  판정은 반드시 L0 또는 L1이 된다 (협력의 부재가 아니라 턴 수 부족).\n"
            f"  파일럿이라도 max_turns를 {min_turns} 이상으로 둘 것.\n"
            f"{'!' * 62}\n"
        )

    # 1) 단독 조건
    if solo_outputs is None:
        step(f"단독 조건 실행 ({len(SOLO_AGENTS)} 에이전트 x {n_solo}회)")
        solo_outputs = run_solo_batch(scenario, api_keys, n_reps=n_solo)

    # 1-1) 단독 산출물 저장 (체크포인트)
    # 로그에는 점수(solo_scores/solo_grades)만 남고 원문은 남지 않는다. 원문이
    # 사라지면 인간 채점자가 나중에 Q3(집단 우위)·Q4b(신규성)를 검증할 비교
    # 대상이 없다. 대화를 돌리기 전에 원문부터 저장하고, 채점이 끝나면
    # solo_values를 채워 같은 파일에 다시 저장한다.
    if out_dir:
        save_solo_outputs(solo_outputs, out_dir, replicate,
                          solo_values=solo_values)

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
    log = tag_log(log, api_key=jkey)

    # 4~5) 채점 + 부착
    step("채점")
    group_text = group_output_text(log)
    scores = score_outputs(scenario, solo_outputs, group_text, jkey,
                           solo_values=solo_values)
    log["group_output_text"] = group_text
    log = attach_scores(log, scores["solo_values"], scores["group_value"],
                        scores["new_idea_flag"])

    # 6~7) 검증 + 판정
    step("판정")
    result = classify_log(log, thresholds=thresholds)
    result["log"] = log
    result["solo_outputs"] = solo_outputs
    # 다음 조건이 그대로 재사용한다 (judge 호출 10회 절약 + 기준선 일치)
    result["solo_values"] = scores["solo_values"]

    if out_dir:
        result["log_path"] = save_log(log, out_dir)
        # 채점값을 포함해 다시 저장한다 (solo_values[i]는 outputs[i]의 점수)
        result["solo_path"] = save_solo_outputs(
            solo_outputs, out_dir, replicate,
            solo_values=scores["solo_values"])

    if verbose:
        print(format_result(result))

    return result


def run_scenario_both_conditions(scenario_path, api_keys: dict, **kwargs) -> dict:
    """
    같은 시나리오를 명시/묵시 두 조건으로 돌린다.

    단독 산출물은 한 번만 생성하고, **채점 결과도 한 번만 내서** 두 조건이 공유한다.
    조건마다 다시 채점하면 judge 호출을 10회씩 두 번 내는 데다, 같은 글에 다른
    등급이 나와 "두 조건이 같은 기준선을 쓴다"는 전제가 실제로는 깨진다.
    """
    scenario = load_scenario(scenario_path)
    n_solo = kwargs.pop("n_solo", DEFAULT_SOLO_REPS)
    solo_outputs = run_solo_batch(scenario, api_keys, n_reps=n_solo)

    results, solo_values = {}, kwargs.pop("solo_values", None)
    for cond in ("명시", "묵시"):
        results[cond] = run_scenario(scenario_path, cond, api_keys,
                                     solo_outputs=solo_outputs,
                                     solo_values=solo_values, **kwargs)
        solo_values = results[cond]["solo_values"]   # 뒤 조건은 재채점하지 않는다
    return results


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


def save_solo_outputs(solo_outputs: list, out_dir, replicate: int = 1,
                      solo_values: list = None) -> str:
    """
    단독 산출물 원문을 {out_dir}/solo/{scenario_id}_solo_{rep}.json 으로 저장한다.

    로그(save_log)에는 단독 조건의 점수만 남고 원문은 남지 않는다. 인간 채점자가
    Q3(집단 우위)·Q4b(신규성)를 검증하려면 그룹 산출물과 비교할 단독 원문이
    필요하므로 별도 파일로 남긴다. solo/ 하위에 두는 이유: 완성 로그와 파일명
    규칙이 달라서, 현황 셀과 classify_saved_dir()가 루트의 *.json을 훑을 때
    걸리지 않게 하기 위해서다.

    rep는 dyad의 replicate 번호다 (명시/묵시가 같은 단독 산출물을 공유하므로
    조건 구분은 없다). solo_values를 주면 함께 저장한다 — solo_values[i]는
    outputs[i]의 점수다 (score_outputs가 순서를 보존한다).
    """
    if not solo_outputs:
        raise PipelineError("단독 산출물이 비어 있음 — 저장할 것이 없다")

    out_dir = Path(out_dir) / "solo"
    out_dir.mkdir(parents=True, exist_ok=True)
    first = solo_outputs[0]
    record = {
        "scenario_id": first["scenario_id"],
        "task_type": first["task_type"],
        "condition": "solo",
        "replicate": replicate,
        "outputs": solo_outputs,
        "solo_values": list(solo_values) if solo_values is not None else None,
    }
    path = out_dir / f"{first['scenario_id']}_solo_{replicate}.json"
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    return str(path)


def load_solo_outputs(out_dir, scenario_id: str, replicate: int = 1) -> dict:
    """
    save_solo_outputs()로 저장한 단독 산출물을 읽는다.

    반환 딕셔너리의 'outputs'와 'solo_values'를 run_scenario(solo_outputs=...,
    solo_values=...)에 그대로 넘기면 단독 실행과 채점을 재사용할 수 있다
    (중간에 실패한 조건을 이어서 돌릴 때 단독 10회 + judge 10회가 절약된다).
    """
    path = Path(out_dir) / "solo" / f"{scenario_id}_solo_{replicate}.json"
    if not path.exists():
        raise PipelineError(
            f"저장된 단독 산출물이 없음: {path}\n"
            f"  이 파일은 저장 기능 추가 이후의 실행에서만 만들어진다 — "
            f"없으면 run_solo_batch()로 다시 생성할 것"
        )
    return json.loads(path.read_text(encoding="utf-8"))


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
