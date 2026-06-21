"""
qwen_quality_policy.py
역할: Qwen self-check 결과의 fail-closed 통과/차단 정책 정의
입력: self-check 결과 dict
출력: fallback 필요 여부와 검사 실패 표준 결과
"""

from typing import Any


def self_check_requires_fallback(result: dict[str, Any] | None) -> bool:
    """
    역할: self-check가 명시적으로 OK인 경우만 원문 응답 통과
    입력: self-check 결과 dict 또는 None
    출력: 안전 fallback 교체 필요 여부
    """
    if not isinstance(result, dict):
        return True
    verdict = str(result.get("verdict") or "").strip().upper()
    return verdict != "OK"


def build_self_check_error_result() -> dict[str, str]:
    """
    역할: self-check 실행 예외를 감사 가능한 fail-closed 결과로 변환
    입력: 없음
    출력: ERROR verdict 표준 dict
    """
    return {
        "verdict": "ERROR",
        "category": "self_check_error",
        "raw": "",
    }
