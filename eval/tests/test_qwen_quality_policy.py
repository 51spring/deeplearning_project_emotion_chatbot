"""
test_qwen_quality_policy.py
역할: Qwen self-check fail-closed 정책 회귀 테스트
입력: pytest 실행
출력: assertion 성공/실패
"""

from backend.qwen_quality_policy import (
    build_self_check_error_result,
    self_check_requires_fallback,
)


def test_only_explicit_ok_passes_self_check() -> None:
    """
    역할: OK 외 verdict와 손상된 결과가 모두 fallback을 요구하는지 검증
    입력: 없음
    출력: 없음
    """
    assert self_check_requires_fallback({"verdict": "OK"}) is False
    assert self_check_requires_fallback({"verdict": "BAD"}) is True
    assert self_check_requires_fallback({"verdict": "ERROR"}) is True
    assert self_check_requires_fallback({"verdict": "UNKNOWN"}) is True
    assert self_check_requires_fallback(None) is True


def test_error_result_is_auditable_and_blocking() -> None:
    """
    역할: 검사 예외 표준 결과가 감사 가능하고 차단되는지 검증
    입력: 없음
    출력: 없음
    """
    result = build_self_check_error_result()
    assert result == {
        "verdict": "ERROR",
        "category": "self_check_error",
        "raw": "",
    }
    assert self_check_requires_fallback(result) is True
