"""
test_crisis_policy.py
역할: 관계 후회 문맥의 NLI hard 오탐 방어 회귀 테스트
입력: pytest 실행
출력: assertion 성공/실패
"""

from backend.crisis_handler import should_hard_interrupt
from pipeline.utterance_type import is_interpersonal_remorse_text


def test_interpersonal_remorse_is_not_hard_crisis() -> None:
    """
    역할: 상대에게 정서적 상처를 줬을까 걱정하는 문맥은 하드 위기가 아님을 검증
    입력: 없음
    출력: 없음
    """
    cases = [
        "말실수를 한 것 같은데 그 친구한테 상처를 줬을 것 같아",
        "내 말 때문에 동료 마음을 아프게 한 것 같아서 미안해",
        "괜히 말해서 친구를 실망시킨 것 같아 걱정돼",
    ]
    for text in cases:
        assert is_interpersonal_remorse_text(text), text
        assert not should_hard_interrupt(text, True, 0.999), text


def test_harm_intent_and_direct_crisis_are_not_demoted() -> None:
    """
    역할: 타인 가해 의도와 직접 자해·자살 표현은 관계 후회 예외에서 제외되는지 검증
    입력: 없음
    출력: 없음
    """
    assert not is_interpersonal_remorse_text("그 친구한테 상처를 주고 싶어")
    assert not is_interpersonal_remorse_text("그 사람을 해치고 싶어")
    assert should_hard_interrupt("죽고 싶다는 생각이 계속 들어", False, 0.10)
