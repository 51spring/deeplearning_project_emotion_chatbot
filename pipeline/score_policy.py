"""
score_policy.py
역할: 발화 타입별 웰니스 점수 반영 정책을 한곳에서 관리한다.
입력: RoBERTa 발화 타입 / 위기 여부 / 발화별 depression_score
출력: 당일 EWMA 반영 여부와 실제 반영 점수
"""

from pipeline.wellness_score import NO_SIGNAL_DEPRESSION_SCORE


FULL_IMPACT_UTTERANCE_TYPES = frozenset({
    "emotional_distress",
    "routine_discomfort",
    "crisis_candidate",
})

LOW_IMPACT_UTTERANCE_TYPES = frozenset({
    "casual_neutral",
    "casual_share",
    "positive_share",
})

SCORE_AFFECTING_UTTERANCE_TYPES = (
    FULL_IMPACT_UTTERANCE_TYPES | LOW_IMPACT_UTTERANCE_TYPES
)

LOW_IMPACT_SCORE_DELTA = 0.08
LOW_IMPACT_SCORE_MIN = NO_SIGNAL_DEPRESSION_SCORE - LOW_IMPACT_SCORE_DELTA
LOW_IMPACT_SCORE_MAX = NO_SIGNAL_DEPRESSION_SCORE + LOW_IMPACT_SCORE_DELTA
LOW_POSITIVE_SCORE_MAX = NO_SIGNAL_DEPRESSION_SCORE
# 명확한 긍정/회복 발화 전용 하한 — baseline(0.30, wellness 70) 아래 양호권까지 내려가도록 허용.
# floor 0.12 = wellness 88 천장. 단일 긍정 발화가 100점으로 튀지 않게 88을 상한으로 둔다.
LOW_POSITIVE_SCORE_MIN = 0.12
# 긍정 기여에서 raw 점수(cbt 누출 등 잡음 포함) 반영 비중 — 나머지는 floor로 모은다.
LOW_POSITIVE_RAW_WEIGHT = 0.15
LOW_SENSORY_DISGUST_SCORE_MIN = NO_SIGNAL_DEPRESSION_SCORE + 0.05
LOW_SENSORY_DISGUST_SCORE_MAX = NO_SIGNAL_DEPRESSION_SCORE + 0.15


def _safe_float(value: object, default: float) -> float:
    """
    역할: 숫자 변환 실패 시 기본값을 반환한다.
    입력: 임의 값, 기본 float 값
    출력: float 값
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _clamp_score(value: float, low: float, high: float) -> float:
    """
    역할: 웰니스 반영 점수를 지정 범위 안으로 제한한다.
    입력: 원본 점수, 하한, 상한
    출력: 제한된 점수
    """
    return max(low, min(high, float(value)))


def _is_positive_low_impact(roberta_out: dict) -> bool:
    """
    역할: 저강도 반영 대상 중 긍정/회복 신호인지 판별한다.
    입력: RoBERTa/Qwen 전처리 후 발화 점수 dict
    출력: 긍정 저강도 신호 여부
    """
    return (
        roberta_out.get("utterance_type") == "positive_share"
        or roberta_out.get("top_emotion") == "행복"
        or roberta_out.get("emotion_guard") == "positive_affect_emotion_preserve"
        or roberta_out.get("nli_guard") == "positive_affect_safety_cap"
    )


def _is_sensory_disgust_low_impact(roberta_out: dict) -> bool:
    """
    역할: 감각 혐오가 혐오 라벨은 보존하되 웰니스 full-impact에서는 제외되어야 하는지 판별한다.
    입력: RoBERTa/Qwen 전처리 후 발화 점수 dict
    출력: 저위험 감각 혐오 신호 여부
    """
    reason = roberta_out.get("utterance_type_reason") or roberta_out.get("type_reason")
    return (
        reason in {
            "sensory_disgust_low_impact_marker",
            "sensory_disgust_low_impact_override",
        }
        or roberta_out.get("score_policy_hint") == "sensory_disgust_low_impact"
    )


def score_affects_wellness(
    utterance_type: str | None,
    is_crisis: bool = False,
) -> bool:
    """
    역할: 발화가 웰니스 점수 변화에 반영되어야 하는지 판정
    입력: 발화 타입 문자열, 최종 위기 여부
    출력: 점수 반영 여부
    """
    if is_crisis:
        return True
    return utterance_type in SCORE_AFFECTING_UTTERANCE_TYPES


def compute_wellness_contribution(
    roberta_out: dict,
    is_crisis: bool = False,
) -> dict:
    """
    역할: 발화별 원점수에서 실제 웰니스 EWMA에 넣을 기여 점수를 산출한다.
    입력: RoBERTa 결과 dict, 최종 위기 여부
    출력: {score_affects_wellness, wellness_contribution_score, wellness_impact_type, score_policy}
    """
    utterance_type = roberta_out.get("utterance_type")
    raw_score = _safe_float(
        roberta_out.get("depression_score"),
        NO_SIGNAL_DEPRESSION_SCORE,
    )

    if not is_crisis and _is_sensory_disgust_low_impact(roberta_out):
        contribution_score = _clamp_score(
            raw_score,
            LOW_SENSORY_DISGUST_SCORE_MIN,
            LOW_SENSORY_DISGUST_SCORE_MAX,
        )
        return {
            "score_affects_wellness": True,
            "wellness_contribution_score": contribution_score,
            "wellness_impact_type": "low",
            "score_policy": "low_sensory_disgust_clamped_plus_5_15",
        }

    if is_crisis or utterance_type in FULL_IMPACT_UTTERANCE_TYPES:
        return {
            "score_affects_wellness": True,
            "wellness_contribution_score": raw_score,
            "wellness_impact_type": "full",
            "score_policy": "full_affecting_type_or_crisis",
        }

    if utterance_type in LOW_IMPACT_UTTERANCE_TYPES:
        if _is_sensory_disgust_low_impact(roberta_out):
            contribution_score = _clamp_score(
                raw_score,
                LOW_SENSORY_DISGUST_SCORE_MIN,
                LOW_SENSORY_DISGUST_SCORE_MAX,
            )
            policy = "low_sensory_disgust_clamped_plus_5_15"
        elif _is_positive_low_impact(roberta_out):
            # 명확한 긍정/회복 발화는 baseline(0.30) 위가 아니라 아래(양호권)로 내려가도록
            # floor를 LOW_POSITIVE_SCORE_MIN(0.12, wellness 88)까지 낮추고,
            # raw 점수는 소폭(LOW_POSITIVE_RAW_WEIGHT)만 반영해 floor 근처(양호 ~85)로 모은다.
            contribution_score = _clamp_score(
                LOW_POSITIVE_SCORE_MIN
                + (raw_score - LOW_POSITIVE_SCORE_MIN) * LOW_POSITIVE_RAW_WEIGHT,
                LOW_POSITIVE_SCORE_MIN,
                LOW_POSITIVE_SCORE_MAX,
            )
            policy = "low_positive_floor_lift_to_yangho"
        else:
            contribution_score = _clamp_score(
                raw_score,
                LOW_IMPACT_SCORE_MIN,
                LOW_IMPACT_SCORE_MAX,
            )
            policy = "low_neutral_clamped_plus_minus_8"
        return {
            "score_affects_wellness": True,
            "wellness_contribution_score": contribution_score,
            "wellness_impact_type": "low",
            "score_policy": policy,
        }

    return {
        "score_affects_wellness": False,
        "wellness_contribution_score": None,
        "wellness_impact_type": "none",
        "score_policy": "non_emotional_question_or_no_signal",
    }
