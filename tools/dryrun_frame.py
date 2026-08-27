# 판정 프레임 무비용 시험 도구

"""
판정 프레임 드라이런 (API 호출 0회, 비용 0원)
================================================
판정 프레임(validate_log -> extract_features -> classify)은 **순수 함수**다.
로그 딕셔너리 하나만 받고, 모델도 네트워크도 쓰지 않는다.
따라서 대화를 실제로 돌리지 않고 **모의 로그를 손으로 만들어 넣으면**
판정 프레임만 따로 시험할 수 있다. 시나리오 파일은 읽기만 하고 건드리지 않는다.

    python tools/dryrun_frame.py
    python tools/dryrun_frame.py --scenario scenarios/A1/A1_simple_meeting_room.json

하는 일:
    [1] 시나리오 형식 점검      실행 전에 잡을 수 있는 JSON 오류
    [2] 결정 트리 반응 시험      L0~L4가 의도대로 갈리는지
    [3] 임계값 민감도            어느 기준이 실제로 판정을 바꾸는지
    [4] 실행 전 환경 점검        SDK/키/모델 ID (호출은 하지 않음)

[2]와 [3]에서 쓰는 점수는 **가짜다**. 프레임이 그 숫자에 어떻게 반응하는지를
보는 것이지, 시나리오의 실제 성능을 보는 것이 아니다.
"""

from __future__ import print_function

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Windows 콘솔 기본 코드페이지(cp949)에서 한글/기호가 깨지는 것을 막는다.
# Colab(utf-8)에서는 아무 영향이 없다.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from coop_pipeline import classify_log, format_result, load_thresholds  # noqa: E402
from coop_pipeline.thresholds import THRESHOLDS, THRESHOLDS_SOURCE  # noqa: E402

DEFAULT_SCENARIO = "scenarios/A4/A4_simple_energy_save.json"

# 모델 ID에서 계열을 알아내기 위한 키워드. self-preference 편향 점검에 쓴다.
MODEL_FAMILIES = ("gemini", "gemma", "claude", "llama", "gpt", "qwen",
                  "deepseek", "mistral", "kimi", "moonshot", "grok")


def model_family(model_id: str) -> str:
    """
    'gemini-2.5-flash' -> 'gemini', 'openai/gpt-oss-120b' -> 'gpt'
    제공사가 아니라 모델 계열을 본다 (Groq는 llama 외 계열도 서빙하기 때문).
    """
    text = str(model_id).lower()
    for family in MODEL_FAMILIES:
        if family in text:
            return family
    return text.split("/")[-1].split("-")[0] or "unknown"


# ---------------------------------------------------------------------------
# 모의 로그 생성
# ---------------------------------------------------------------------------
def make_log(scenario_id, task_type, n_turns=10, ref_ab=0, ref_ba=0,
             comp=0, agree=0, solo=None, group=None, new_idea=False,
             condition="명시"):
    """
    판정 프레임에 넣을 모의 로그를 만든다.

    ref_ab / ref_ba : alpha->beta, beta->alpha 참조 턴 수   (Q1, Q2를 움직인다)
    comp            : 'comp' 코드가 붙은 턴 수              (Q4a)
    agree           : 'agree' 코드 + 참조가 함께 있는 턴 수  (Q4c)
    solo / group    : 단독 점수 목록과 그룹 점수            (Q3)
    new_idea        : 신규 해결책 플래그                    (Q4b)

    text는 판정에 쓰이지 않으므로(태깅은 이미 끝난 것으로 가정) 자리표시자를 넣는다.
    """
    turns = []
    used_ab = used_ba = used_comp = used_agree = 0

    for i in range(1, n_turns + 1):
        speaker = "alpha" if i % 2 == 1 else "beta"
        ref = None
        codes = []

        # 참조 방향 채우기
        if speaker == "alpha" and used_ab < ref_ab:
            ref, used_ab = "beta", used_ab + 1
        elif speaker == "beta" and used_ba < ref_ba:
            ref, used_ba = "alpha", used_ba + 1

        # agree는 참조가 있는 턴에만 붙인다 (revision_seq의 정의가 그렇다)
        if ref is not None and used_agree < agree:
            codes.append("agree")
            used_agree += 1
        if used_comp < comp:
            codes.append("comp")
            used_comp += 1
        if not codes:
            codes.append("arch")

        turns.append({"turn": i, "speaker": speaker, "text": "(모의 발화)",
                      "codes": codes, "ref": ref})

    log = {
        "scenario_id": scenario_id,
        "task_type": task_type,
        "condition": condition,
        "replicate": 1,
        "turns": turns,
        "new_idea_flag": bool(new_idea),
    }

    if task_type == "A1":
        log["solo_scores"] = list(solo if solo is not None else [0.5] * 5)
        log["group_score"] = group if group is not None else 0.5
    else:
        log["solo_grades"] = list(solo if solo is not None else [3, 3, 3, 3, 3])
        log["group_grade"] = group if group is not None else 3

    return log


def _win(task_type):
    """해당 task_type에서 Q3(집단 우위)를 통과하는 (solo, group) 값 한 쌍."""
    if task_type == "A1":
        return [0.3, 0.4, 0.4, 0.5, 0.5], 0.9
    return [2, 2, 3, 2, 3], 5      # 등급차 +2 (grade_gap_min 기본값)


def _lose(task_type):
    """Q3에서 막히는 (solo, group) 값 한 쌍."""
    if task_type == "A1":
        return [0.8, 0.9, 0.9, 1.0, 1.0], 0.5
    return [3, 3, 4, 3, 4], 3      # 등급차 -0


def _boundary(task_type):
    """
    Q3를 '간신히' 통과하는 (solo, group) 값 한 쌍.
    임계값을 한 칸만 올려도 뒤집히므로 민감도 관찰에 쓴다.
    """
    if task_type == "A1":
        return [0.3, 0.4, 0.5, 0.6, 0.9], 0.80   # p90=0.78 < 0.80, p100=0.9 > 0.80
    return [3, 3, 3, 3, 3], 5                    # 등급차 정확히 +2


# ---------------------------------------------------------------------------
# [1] 시나리오 형식 점검
# ---------------------------------------------------------------------------
def step_scenario_check(scenario_path):
    from coop_pipeline.runner import load_scenario

    print("=" * 68)
    print("[1] 시나리오 형식 점검 (읽기만 한다)")
    print("=" * 68)

    scenario = load_scenario(scenario_path)
    variants = scenario["task_variants"]
    shape = "alpha/beta 비대칭" if variants.get("alpha") else "shared 대칭"

    print("파일       : %s" % scenario_path)
    print("scenario_id: %s" % scenario["scenario_id"])
    print("task_type  : %s  (%s)" % (
        scenario["task_type"],
        "체크리스트 채점" if scenario["task_type"] == "A1" else "LLM-judge 등급 채점"))
    print("complexity : %s" % scenario.get("complexity"))
    print("지문 구성  : %s / solo %s" % (shape, "있음" if "solo" in variants else "없음"))
    print("형식 점검  : 통과\n")
    return scenario


# ---------------------------------------------------------------------------
# [2] 결정 트리 반응 시험
# ---------------------------------------------------------------------------
def step_decision_tree(scenario):
    sid, ttype = scenario["scenario_id"], scenario["task_type"]
    lose_solo, lose_group = _lose(ttype)
    win_solo, win_group = _win(ttype)

    cases = [
        ("L0", "서로 참조 0회",
         dict(ref_ab=0, ref_ba=0, solo=lose_solo, group=lose_group)),
        ("L1", "alpha만 상대를 참조 (편측)",
         dict(ref_ab=3, ref_ba=0, solo=lose_solo, group=lose_group)),
        ("L2", "쌍방향이지만 집단 우위 없음",
         dict(ref_ab=3, ref_ba=3, solo=lose_solo, group=lose_group)),
        ("L3", "집단 우위 있으나 신규성 미달",
         dict(ref_ab=3, ref_ba=3, solo=win_solo, group=win_group,
              comp=0, agree=0, new_idea=False)),
        ("L4", "집단 우위 + Q4 세 조건 모두 충족",
         dict(ref_ab=3, ref_ba=3, solo=win_solo, group=win_group,
              comp=2, agree=2, new_idea=True)),
    ]

    print("=" * 68)
    print("[2] 결정 트리 반응 시험 — 모의 로그 %d개" % len(cases))
    print("=" * 68)
    print("입력을 의도적으로 조작해 L0~L4가 실제로 갈리는지 본다.")
    print("기대와 다르면 판정 프레임 쪽 문제다 (시나리오 문제가 아니다).\n")

    failures = []
    for expected, description, kwargs in cases:
        result = classify_log(make_log(sid, ttype, **kwargs))
        got = result["level"]
        ok = got == expected
        if not ok:
            failures.append((expected, got, description, result["reason"]))
        print("  %-4s %-32s -> %-4s %s" % (
            expected, description, got, "OK" if ok else "*** 불일치 ***"))
        print("       %s" % result["reason"])

    print()
    if failures:
        print("!! %d개 불일치 — 판정 프레임을 확인할 것" % len(failures))
    else:
        print("전부 기대대로 갈렸다. 결정 트리는 정상 동작한다.")
    print()
    return failures


# ---------------------------------------------------------------------------
# [3] 임계값 민감도
# ---------------------------------------------------------------------------
def step_threshold_sweep(scenario):
    sid, ttype = scenario["scenario_id"], scenario["task_type"]
    base = dict(THRESHOLDS)

    print("=" * 68)
    print("[3] 임계값 민감도")
    print("=" * 68)
    print("임계값 출처: %s" % THRESHOLDS_SOURCE)
    print("현재 값    : %s\n" % base)

    # 모든 기준을 '간신히' 통과하는 로그를 만든 뒤, 기준을 하나씩 올려본다.
    bsolo, bgroup = _boundary(ttype)
    log = make_log(sid, ttype, n_turns=10, ref_ab=2, ref_ba=2,
                   comp=2, agree=1, new_idea=True, solo=bsolo, group=bgroup)

    print("경계 로그: 참조 2/2회, comp 2/10턴(=0.20), agree+ref 1회, 신규성 있음")
    print("           단독 %s vs 그룹 %s\n" % (bsolo, bgroup))
    print("  %-36s %-6s %s" % ("임계값 설정", "판정", "병목"))
    print("  " + "-" * 62)

    sweeps = [("기본값 그대로", {})]
    sweeps += [("bidirectional_min %d -> 3" % base["bidirectional_min"],
                {"bidirectional_min": 3}),
               ("comp_ratio_min %.2f -> 0.30" % base["comp_ratio_min"],
                {"comp_ratio_min": 0.30}),
               ("revision_seq_min %d -> 2" % base["revision_seq_min"],
                {"revision_seq_min": 2})]
    if ttype == "A1":
        sweeps.append(("solo_percentile_A1 %d -> 100" % base["solo_percentile_A1"],
                       {"solo_percentile_A1": 100}))
    else:
        sweeps.append(("grade_gap_min %d -> 3" % base["grade_gap_min"],
                       {"grade_gap_min": 3}))

    for label, override in sweeps:
        result = classify_log(log, thresholds=dict(base, **override))
        print("  %-36s %-6s %s" % (label, result["level"],
                                   result["stopped_at"] or "-"))

    print("\n기준을 한 칸 올릴 때마다 판정이 뒤집힌다 = 이 로그가 모든 기준의 경계에 있다는 뜻.")
    print("실제 로그로 이 표를 만들면 어느 기준이 결과를 좌우하는지 바로 보인다.")

    if ttype != "A1":
        print("\n[주의] %s는 1~5 등급 척도에서 등급차 %d 이상을 요구한다."
              % (ttype, base["grade_gap_min"]))
        print("       단독이 3점이면 그룹이 5점(만점)이어야 Q3를 통과한다.")
        print("       파일럿에서 Q3 병목이 몰리면 임계값 캘리브레이션 대상이다.")
    print()


# ---------------------------------------------------------------------------
# [4] 실행 전 환경 점검
# ---------------------------------------------------------------------------
def step_preflight():
    print("=" * 68)
    print("[4] 실제 실행 전 환경 점검 (API를 호출하지는 않는다)")
    print("=" * 68)

    print("SDK 설치 여부:")
    for module, used_by in (("google.generativeai", "alpha 생성"),
                            ("groq", "beta 생성"),
                            ("anthropic", "judge 태깅/채점")):
        try:
            __import__(module)
            mark = "설치됨"
        except ImportError:
            mark = "없음  -> pip install -r requirements.txt"
        print("  %-22s %-10s (%s)" % (module, mark, used_by))

    from coop_pipeline.llm import MODELS, judge_key_name, judge_model, judge_provider
    print("\n모델 ID (llm.MODELS):")
    for role in ("alpha", "beta", "judge"):
        print("  %-6s %s" % (role, MODELS[role]))
    print("  * 퇴역한 ID면 실행이 통째로 실패한다. docs/PIPELINE.md §2-2로 확인할 것.")

    provider = judge_provider()
    print("\njudge 제공사: %s (%s)" % (provider, judge_model()))
    print("  필요한 키   : api_keys['%s']" % judge_key_name())
    print("  비용        : %s" % (
        "유료" if provider == "anthropic" else "무료 티어"))

    # self-preference 편향 점검.
    # 제공사가 아니라 **모델 ID**로 계열을 본다 (Groq는 llama 외 계열도 서빙한다).
    judge_family = model_family(MODELS["judge"])
    clash = [role for role in ("alpha", "beta")
             if model_family(MODELS[role]) == judge_family]
    print("  계열        : judge=%s / alpha=%s / beta=%s" % (
        judge_family, model_family(MODELS["alpha"]), model_family(MODELS["beta"])))
    if clash:
        print("  !! 경고: judge가 %s와 같은 계열이다 (self-preference 편향)."
              % ", ".join(clash))
        print("     자기 계열 산출물을 자기가 채점하면 group 점수가 부풀 수 있다.")
        print("     특히 alpha는 최종 산출물 작성자이므로 alpha와 겹치면 치명적이다.")
    else:
        print("  OK: judge가 alpha/beta 어느 쪽과도 다른 계열이다.")

    print("\nAPI 키 환경변수 (값은 출력하지 않는다):")
    for name in ("GEMINI_API_KEY", "GROQ_API_KEY", "ANTHROPIC_API_KEY"):
        print("  %-20s %s" % (name, "있음" if os.environ.get(name) else "없음"))
    print("  * Colab에서는 환경변수 대신 Secrets(userdata.get)를 쓴다. '없음'이어도 정상.")
    print()


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="판정 프레임을 API 호출 없이 시험한다.")
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO,
                        help="대상 시나리오 JSON (기본: %s)" % DEFAULT_SCENARIO)
    parser.add_argument("--skip-preflight", action="store_true",
                        help="[4] 환경 점검 생략")
    args = parser.parse_args()

    scenario = step_scenario_check(args.scenario)
    failures = step_decision_tree(scenario)
    step_threshold_sweep(scenario)
    if not args.skip_preflight:
        step_preflight()

    print("=" * 68)
    if failures:
        print("결과: 판정 프레임에 불일치 %d건. 실제 실행 전에 먼저 고칠 것." % len(failures))
        return 1
    print("결과: 판정 프레임 정상. 실제 실행 절차는 docs/DRYRUN.md §3 참고.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
