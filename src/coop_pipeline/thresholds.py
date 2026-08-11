# 판정 기준값

"""
판정 기준값
========================
값의 정본(single source of truth)은 `configs/thresholds_{version}.json` 파일이다.
이 모듈은 그 파일을 읽어오는 로더이고, 아래 DEFAULT_THRESHOLDS는
configs를 못 읽을 때만 쓰이는 비상용 사본이다.

파일럿 데이터를 수집한 뒤에는 configs/thresholds_v2.json을 새로 만들고
    THRESHOLDS = load_thresholds("v2")
로 바꾸기만 하면 된다. classify.py는 손댈 필요 없다.

임계값을 "어떻게" 정하는지(캘리브레이션 절차)는 docs/RUBRIC.md 참고.

버전 기록:
- v1 : 파일럿 전 초기 가설값. 근거 없음. 절대 본실험 결과 보고에 쓰지 말 것.
"""

import json
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"

# configs/*.json에 들어있지만 임계값이 아닌 키 (로드 시 제외한다)
_META_KEYS = {"version", "note"}

# 반드시 존재해야 하는 임계값 키와, 파일을 못 읽을 때 쓰는 비상 기본값
DEFAULT_THRESHOLDS = {
    "bidirectional_min": 2,     # Q2: 양방향 판정에 필요한 방향별 최소 참조 턴 수
    "solo_percentile_A1": 90,   # Q3(A1): 단독 분포의 몇 퍼센타일을 넘어야 우위인지
    "grade_gap_min": 2,         # Q3(A2/A4): 그룹등급 - 단독등급 중앙값의 최소 차이
    "comp_ratio_min": 0.15,     # Q4: L3/L4 경계, comp 발화 비율 최소값
    "revision_seq_min": 1,      # Q4: 계획 수정(agree+참조) 발화 최소 횟수
}


class ThresholdConfigError(ValueError):
    """configs/thresholds_*.json이 없거나 키가 빠졌을 때."""


def load_thresholds(version: str = "v1") -> dict:
    """
    configs/thresholds_{version}.json을 읽어 임계값 딕셔너리를 반환한다.
    필수 키가 하나라도 없으면 ThresholdConfigError를 던진다
    (조용히 기본값으로 넘어가면 어떤 값으로 판정했는지 알 수 없게 된다).
    """
    path = CONFIG_DIR / f"thresholds_{version}.json"
    if not path.exists():
        raise ThresholdConfigError(f"임계값 파일이 없음: {path}")

    raw = json.loads(path.read_text(encoding="utf-8"))
    values = {k: v for k, v in raw.items() if k not in _META_KEYS}

    missing = set(DEFAULT_THRESHOLDS) - set(values)
    if missing:
        raise ThresholdConfigError(f"{path.name}에 필수 키 누락: {sorted(missing)}")

    unknown = set(values) - set(DEFAULT_THRESHOLDS)
    if unknown:
        raise ThresholdConfigError(
            f"{path.name}에 알 수 없는 키: {sorted(unknown)} "
            f"(오타이거나 DEFAULT_THRESHOLDS에 추가해야 함)"
        )
    return values


try:
    THRESHOLDS = load_thresholds("v1")
    THRESHOLDS_SOURCE = "configs/thresholds_v1.json"
except (ThresholdConfigError, OSError, json.JSONDecodeError):
    # configs를 못 읽는 환경(예: 패키지만 복사한 경우)에서도 임포트는 되게 한다.
    THRESHOLDS = dict(DEFAULT_THRESHOLDS)
    THRESHOLDS_SOURCE = "DEFAULT_THRESHOLDS (내장 비상값)"
