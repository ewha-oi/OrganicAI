# 패키지 진입점

"""
협력 층위 판정 파이프라인.

여기서는 외부 SDK(google-generativeai, groq, anthropic)가 없어도 임포트되는
모듈만 노출한다. 실제 실행(agents, tagging, scoring, runner)은 SDK가 필요하므로
    from coop_pipeline.runner import run_scenario
처럼 필요할 때 직접 임포트한다.
"""

from .classify import classify, classify_log, format_result, LEVEL_MEANING
from .features import extract_features
from .validate_log import validate_log, LogValidationError
from .thresholds import THRESHOLDS, THRESHOLDS_SOURCE, load_thresholds

__all__ = [
    "classify",
    "classify_log",
    "format_result",
    "LEVEL_MEANING",
    "extract_features",
    "validate_log",
    "LogValidationError",
    "THRESHOLDS",
    "THRESHOLDS_SOURCE",
    "load_thresholds",
]
