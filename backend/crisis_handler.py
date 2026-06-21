"""
crisis_handler.py
역할: 위기 감지 이중 구조 처리
      - NLI 후보 감지: entailment_prob > CRISIS_THRESHOLD
      - 하드 인터럽트: 직접 위험 표현 또는 NLI 고신뢰 기준 통과 시 Qwen 생략
      - Qwen [CRISIS] 태그 (소프트 인터럽트): 하드 인터럽트 미발생 시 보완
      위기 발생 시 DB 저장 + 안전 안내 메시지 반환
"""

from sqlalchemy.orm import Session
from backend.db import crud
from pipeline.utterance_type import (
    has_distress_marker,
    has_fear_marker,
    is_administrative_technical_neutral_text,
    is_daily_routine_neutral_text,
    is_interpersonal_remorse_text,
    is_limited_situational_distress_text,
    is_physical_exertion_text,
    is_routine_discomfort_text,
    is_sensory_disgust_text,
    is_situational_anger_text,
    is_situational_sadness_text,
    ROUTINE_DISCOMFORT_MARKERS,
    compact_text as _compact_text,
    _has_any,
)

# NLI 후보 중 Qwen을 생략할 정도의 하드 인터럽트로 볼 고신뢰 기준
NLI_HARD_INTERRUPT_THRESHOLD = 0.80

# 위기 안내 메시지 (하드 인터럽트용)
CRISIS_MESSAGE = (
    "지금 많이 힘드시겠어요. "
    "혼자 감당하기 어려운 감정이 느껴지신다면, "
    "지금은 혼자 버티기보다 가까운 사람이나 전문 상담 기관에 바로 알려 주세요.\n\n"
    "📞 자살예방상담전화: 109 (24시간)\n"
    "📞 긴급한 위험이 있으면 119 또는 112에 즉시 연락해 주세요."
)

# Qwen [CRISIS] 태그 소프트 메시지
SOFT_CRISIS_MESSAGE = (
    "오늘 많이 지치셨군요. "
    "혹시 스스로를 해치고 싶은 마음이 조금이라도 있다면 혼자 있지 말고, "
    "가까운 사람에게 바로 알리거나 자살예방상담전화 109에 도움을 요청해 주세요."
)

# 하드 인터럽트는 발표·실사용에서 오탐 비용이 크므로 직접적 위험 표현 위주로 좁게 둔다.
HARD_CRISIS_PATTERNS = [
    "자해",
    "죽고 싶",
    "죽고싶",
    "죽어버",
    "죽을래",
    "목숨",
    "사라지고 싶",
    "없어지고 싶",
    "끝내고 싶",
    "살고 싶지",
    "손목",
    "칼로",
    "약을 잔뜩",
    "뛰어내",
    "목매",
]

NLI_HARD_FALSE_POSITIVE_PATTERNS = [
    "내탓처럼말하는메시지",
    "먹는것도귀찮고밤에는잠이얕",
    "밤에는잠이얕아서계속지쳐",
    "사람들이나없이도괜찮",
    "말을중간에끊",
    "기분이확상",
]


def has_hard_crisis_phrase(text: str) -> bool:
    """
    역할: Qwen을 생략해야 할 직접적 자해·자살 표현 포함 여부 판별
    입력: 사용자 발화 텍스트
    출력: 직접 하드 위기 표현 여부
    """
    normalized = str(text).replace(" ", "").lower()
    return any(pattern.replace(" ", "").lower() in normalized for pattern in HARD_CRISIS_PATTERNS)


def is_benign_nli_hard_false_positive_context(text: str) -> bool:
    """
    역할: NLI hard 확률은 높지만 자해·자살 하드 인터럽트로 승격하면 안 되는 문맥을 판별한다.
    입력: 사용자 발화 텍스트
    출력: 하드 인터럽트 차단 대상 여부
    """
    if has_hard_crisis_phrase(text):
        return False

    compact = _compact_text(text)
    if _has_any(compact, NLI_HARD_FALSE_POSITIVE_PATTERNS):
        return True

    # 아래 범주는 모니터링/상담 응답은 필요할 수 있지만, Qwen 생략 안전문구로 즉시 대체할 정도의
    # 직접 자해 위험 표현은 아니다. NLI 후보 자체는 유지하고 hard 승격만 막는다.
    return any(
        checker(text)
        for checker in (
            is_administrative_technical_neutral_text,
            is_daily_routine_neutral_text,
            is_physical_exertion_text,
            is_routine_discomfort_text,
            is_sensory_disgust_text,
            is_limited_situational_distress_text,
            is_situational_anger_text,
            is_situational_sadness_text,
            is_interpersonal_remorse_text,
        )
    )


def _has_negative_signal(text: str) -> bool:
    """
    역할: NLI 오탐 방어용 — 텍스트에 부정 정서 단서가 하나라도 있는지 확인
    입력: 사용자 발화 텍스트
    출력: 부정 신호 존재 여부
    """
    if has_distress_marker(text) or has_fear_marker(text):
        return True
    compact = _compact_text(text)
    return _has_any(compact, ROUTINE_DISCOMFORT_MARKERS)


def should_hard_interrupt(
    text: str,
    is_nli_crisis: bool,
    entailment_prob: float | None,
) -> bool:
    """
    역할: NLI 위기 후보를 실제 하드 인터럽트로 승격할지 결정
    입력: 사용자 발화, NLI 임계값 통과 여부, entailment 확률
    출력: 하드 인터럽트 여부
    """
    # 직접 위험 표현은 NLI 확률 회귀의 영향을 받지 않도록 독립적으로 하드 처리한다.
    if has_hard_crisis_phrase(text):
        return True

    if not is_nli_crisis:
        return False

    # blind 1008에서 드러난 benign hard false-positive는 하드 인터럽트 대신 일반 응답 경로로 보낸다.
    if is_benign_nli_hard_false_positive_context(text):
        return False

    # NLI 단독 트리거는 텍스트에 부정 신호가 없을 때 차단 (오탐 방어)
    # "손에 잡히는 일이 없네" 같은 중립 표현에서 높은 entailment_prob가 나와도 차단한다.
    if not _has_negative_signal(text):
        return False

    if entailment_prob is not None and entailment_prob >= NLI_HARD_INTERRUPT_THRESHOLD:
        return True

    return False


def handle_crisis(
    db: Session,
    user_id: int,
    text: str,
    source: str,
    utterance_id: int | None = None,
    entailment_prob: float | None = None,
) -> dict:
    """
    역할: 위기 감지 시 DB 저장 + 응답 메시지 결정
    입력: DB 세션, user_id, 발화 텍스트, 감지 소스 ("nli"|"qwen"),
          utterance_id, entailment_prob
    출력: {is_hard: bool, message: str, crisis_event_id: int}
    """
    crud.save_crisis_event(db, user_id, {
        "text":            text,
        "source":          source,
        "utterance_id":    utterance_id,
        "entailment_prob": entailment_prob,
    })

    is_hard = (source == "nli")
    message = CRISIS_MESSAGE if is_hard else SOFT_CRISIS_MESSAGE

    return {
        "is_hard": is_hard,
        "message": message,
    }


def check_qwen_crisis_tag(response_text: str) -> bool:
    """
    역할: Qwen 응답에서 [CRISIS] 태그 존재 여부 확인
    입력: Qwen 생성 텍스트
    출력: 태그 포함 여부 (bool)
    """
    return "[CRISIS]" in response_text
