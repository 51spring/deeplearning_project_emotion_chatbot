"""
cbt_reliability.py
역할: CBT anchor 점수가 실제 웰니스/인지왜곡 판단에 반영될 만큼 신뢰 가능한지 판정한다.
입력: CBT anchor 결과, RoBERTa 발화 타입/위기/강도, CBT class head, 우울 경향 힌트
출력: CBT 점수 반영 방식(full/low)과 cap 적용 메타데이터
"""

from __future__ import annotations

from typing import Any

from pipeline.cbt_similarity import CBT_THRESHOLD


CBT_RELIABILITY_LOW_CAP = 0.45
CBT_RELIABILITY_AMBIGUOUS_CAP = 0.55
CBT_HEAD_CONFIDENCE_FLOOR = 0.50
CBT_HEAD_STRONG_CONFIDENCE = 0.65

LOW_SIGNAL_UTTERANCE_TYPES = frozenset({
    "casual_neutral",
    "casual_share",
    "positive_share",
    "preference_question",
    "practical_question",
})

FULL_SIGNAL_UTTERANCE_TYPES = frozenset({
    "emotional_distress",
    "routine_discomfort",
    "crisis_candidate",
})

DISTRESS_TOP_EMOTIONS = frozenset({"슬픔", "공포", "분노", "혐오"})
LOW_SIGNAL_TOP_EMOTIONS = frozenset({"중립", "행복"})


def _safe_float(value: Any, default: float = 0.0) -> float:
    """
    역할: 임의 값을 float로 변환하고 실패 시 기본값을 반환한다.
    입력: 변환 대상 값, 기본값
    출력: float 값
    """
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _bool(value: Any) -> bool:
    """
    역할: None/문자열/숫자 값을 보수적인 bool 값으로 해석한다.
    입력: 임의 값
    출력: bool 값
    """
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _cbt_margin(cbt_result: dict[str, Any] | None, cbt_score: float) -> float:
    """
    역할: 왜곡 anchor와 contrast anchor 간 margin을 계산한다.
    입력: CBT 결과 dict, 현재 CBT 점수
    출력: raw_cbt - raw_contrast 추정 margin
    """
    if cbt_result:
        raw_cbt = cbt_result.get("raw_cbt")
        raw_contrast = cbt_result.get("raw_contrast")
        if raw_cbt is not None and raw_contrast is not None:
            return _safe_float(raw_cbt) - _safe_float(raw_contrast)
    # contrastive 점수는 margin + 0.5 형태이므로 fallback에서도 대략적인 margin을 복원한다.
    return float(cbt_score) - 0.5


def _score_context(
    *,
    roberta_out: dict[str, Any],
    cbt_head_pred: dict[str, Any] | None,
    depression_tendency_score: float | None,
    distress_severity: float | None,
    margin: float,
) -> tuple[float, float, list[str]]:
    """
    역할: CBT anchor 신뢰도 판단에 쓸 위험 근거와 benign 근거를 점수화한다.
    입력: RoBERTa/CBT head/우울 경향/강도/margin 정보
    출력: (risk_points, benign_points, reason 리스트)
    """
    reasons: list[str] = []
    risk_points = 0.0
    benign_points = 0.0

    utterance_type = roberta_out.get("utterance_type")
    top_emotion = roberta_out.get("top_emotion")
    roberta_score = _safe_float(roberta_out.get("roberta_score"))
    entailment_prob = _safe_float(roberta_out.get("entailment_prob"))
    tendency_score = _safe_float(depression_tendency_score)
    severity = _safe_float(distress_severity)

    if utterance_type in FULL_SIGNAL_UTTERANCE_TYPES:
        risk_points += 1.0
        reasons.append(f"full_signal_type:{utterance_type}")
    if utterance_type in LOW_SIGNAL_UTTERANCE_TYPES:
        benign_points += 2.0
        reasons.append(f"low_signal_type:{utterance_type}")

    if top_emotion in DISTRESS_TOP_EMOTIONS:
        risk_points += 0.75
        reasons.append(f"distress_emotion:{top_emotion}")
    if top_emotion in LOW_SIGNAL_TOP_EMOTIONS:
        benign_points += 1.0
        reasons.append(f"low_signal_emotion:{top_emotion}")

    if roberta_score >= 0.75:
        risk_points += 3.0
        reasons.append("roberta_very_high")
    elif roberta_score >= 0.60:
        risk_points += 2.0
        reasons.append("roberta_high")
    elif roberta_score >= 0.45:
        risk_points += 1.0
        reasons.append("roberta_mid")
    elif roberta_score <= 0.40:
        benign_points += 1.0
        reasons.append("roberta_low")

    if severity >= 0.65:
        risk_points += 3.0
        reasons.append("distress_severity_high")
    elif severity >= 0.45:
        risk_points += 1.5
        reasons.append("distress_severity_mid")
    elif severity <= 0.30:
        benign_points += 1.0
        reasons.append("distress_severity_low")

    if entailment_prob >= 0.70:
        risk_points += 2.0
        reasons.append("nli_high")
    elif entailment_prob >= 0.35:
        risk_points += 1.0
        reasons.append("nli_candidate")
    elif entailment_prob <= 0.20:
        benign_points += 1.0
        reasons.append("nli_low")

    if tendency_score >= 0.60:
        risk_points += 2.0
        reasons.append("tendency_high")
    elif tendency_score >= 0.40:
        risk_points += 1.25
        reasons.append("tendency_mid")
    elif tendency_score <= 0.20:
        benign_points += 1.0
        reasons.append("tendency_low")

    if cbt_head_pred:
        top_conf = _safe_float(cbt_head_pred.get("top_distortion_confidence"))
        if _bool(cbt_head_pred.get("is_non_distortion")):
            benign_points += 2.0
            reasons.append("cbt_head_non_distortion")
        if top_conf >= CBT_HEAD_STRONG_CONFIDENCE:
            risk_points += 2.0
            reasons.append("cbt_head_strong_distortion")
        elif top_conf < CBT_HEAD_CONFIDENCE_FLOOR:
            benign_points += 1.5
            reasons.append("cbt_head_low_distortion_conf")

    if margin >= 0.28:
        risk_points += 1.0
        reasons.append("anchor_margin_high")
    elif margin <= 0.15:
        benign_points += 1.0
        reasons.append("anchor_margin_low")

    return risk_points, benign_points, reasons


def evaluate_cbt_reliability(
    *,
    cbt_score: float | None,
    cbt_result: dict[str, Any] | None,
    roberta_out: dict[str, Any],
    cbt_head_pred: dict[str, Any] | None = None,
    depression_tendency_score: float | None = None,
    distress_severity: float | None = None,
    cbt_threshold: float = CBT_THRESHOLD,
) -> dict[str, Any]:
    """
    역할: CBT anchor 점수를 full로 반영할지, 낮은 cap으로만 반영할지 결정한다.
    입력:
      - cbt_score: 현재까지 개별 cap이 반영된 CBT 점수
      - cbt_result: raw_cbt/raw_contrast/top_category 등 anchor 결과
      - roberta_out: RoBERTa 발화 타입/감정/NLI/위기 결과
      - cbt_head_pred: CBT class head 결과(있으면)
      - depression_tendency_score: 우울 경향 힌트(있으면)
      - distress_severity: distress head 또는 emotion proxy 강도
      - cbt_threshold: CBT 후보 임계값
    출력: CBT 반영 정책 dict
    """
    if cbt_score is None:
        return {
            "cbt_effect": "none",
            "cbt_reliability_policy": "no_cbt_score",
            "cbt_reliability_applied": False,
            "cbt_score_after_reliability": None,
            "cbt_reliability_cap": None,
            "cbt_reliability_risk_points": 0.0,
            "cbt_reliability_benign_points": 0.0,
            "cbt_reliability_reasons": [],
        }

    score = float(cbt_score)
    if score < cbt_threshold:
        return {
            "cbt_effect": "below_threshold",
            "cbt_reliability_policy": "below_threshold_no_gate",
            "cbt_reliability_applied": False,
            "cbt_score_after_reliability": score,
            "cbt_reliability_cap": None,
            "cbt_reliability_risk_points": 0.0,
            "cbt_reliability_benign_points": 0.0,
            "cbt_reliability_reasons": [],
        }

    if _bool(roberta_out.get("is_crisis")) or roberta_out.get("utterance_type") == "crisis_candidate":
        return {
            "cbt_effect": "full",
            "cbt_reliability_policy": "safety_signal_preserve",
            "cbt_reliability_applied": False,
            "cbt_score_after_reliability": score,
            "cbt_reliability_cap": None,
            "cbt_reliability_risk_points": 9.0,
            "cbt_reliability_benign_points": 0.0,
            "cbt_reliability_reasons": ["safety_signal"],
        }

    margin = _cbt_margin(cbt_result, score)
    risk_points, benign_points, reasons = _score_context(
        roberta_out=roberta_out,
        cbt_head_pred=cbt_head_pred,
        depression_tendency_score=depression_tendency_score,
        distress_severity=distress_severity,
        margin=margin,
    )

    # RoBERTa/강도/NLI가 매우 강한 경우는 head 신뢰도가 낮아도 안전 쪽으로 보존한다.
    if (
        _safe_float(roberta_out.get("roberta_score")) >= 0.75
        or _safe_float(distress_severity) >= 0.65
        or _safe_float(roberta_out.get("entailment_prob")) >= 0.70
    ):
        return {
            "cbt_effect": "full",
            "cbt_reliability_policy": "strong_distress_preserve",
            "cbt_reliability_applied": False,
            "cbt_score_after_reliability": score,
            "cbt_reliability_cap": None,
            "cbt_reliability_risk_points": round(risk_points, 3),
            "cbt_reliability_benign_points": round(benign_points, 3),
            "cbt_reliability_reasons": reasons,
        }

    head_conf = _safe_float((cbt_head_pred or {}).get("top_distortion_confidence"))
    head_strong = head_conf >= CBT_HEAD_STRONG_CONFIDENCE
    head_low = cbt_head_pred is not None and head_conf < CBT_HEAD_CONFIDENCE_FLOOR
    head_non_distortion = _bool((cbt_head_pred or {}).get("is_non_distortion"))

    if head_strong and risk_points >= 2.5:
        return {
            "cbt_effect": "full",
            "cbt_reliability_policy": "head_strong_with_risk_preserve",
            "cbt_reliability_applied": False,
            "cbt_score_after_reliability": score,
            "cbt_reliability_cap": None,
            "cbt_reliability_risk_points": round(risk_points, 3),
            "cbt_reliability_benign_points": round(benign_points, 3),
            "cbt_reliability_reasons": reasons,
        }

    cap_value = None
    policy = None
    effect = "full"

    tendency_score = _safe_float(depression_tendency_score)

    if head_non_distortion and risk_points <= 3.0:
        cap_value = CBT_RELIABILITY_LOW_CAP
        policy = "head_non_distortion_low_cap"
        effect = "low"
    elif head_low and benign_points >= 4.0 and risk_points <= 3.25 and tendency_score < 0.40:
        cap_value = CBT_RELIABILITY_LOW_CAP
        policy = "low_head_conf_benign_context_cap"
        effect = "low"
    elif head_low and risk_points <= 2.75:
        cap_value = CBT_RELIABILITY_AMBIGUOUS_CAP
        policy = "low_head_conf_ambiguous_cap"
        effect = "low"
    elif benign_points >= 5.0 and risk_points <= 2.5:
        cap_value = CBT_RELIABILITY_LOW_CAP
        policy = "benign_context_anchor_cap"
        effect = "low"

    if cap_value is None:
        return {
            "cbt_effect": effect,
            "cbt_reliability_policy": "full_reliability_preserve",
            "cbt_reliability_applied": False,
            "cbt_score_after_reliability": score,
            "cbt_reliability_cap": None,
            "cbt_reliability_risk_points": round(risk_points, 3),
            "cbt_reliability_benign_points": round(benign_points, 3),
            "cbt_reliability_reasons": reasons,
        }

    capped_score = min(score, cap_value)
    return {
        "cbt_effect": effect,
        "cbt_reliability_policy": policy,
        "cbt_reliability_applied": capped_score < score,
        "cbt_score_after_reliability": capped_score,
        "cbt_reliability_cap": cap_value,
        "cbt_reliability_risk_points": round(risk_points, 3),
        "cbt_reliability_benign_points": round(benign_points, 3),
        "cbt_reliability_reasons": reasons,
    }
