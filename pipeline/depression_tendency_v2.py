"""
depression_tendency_v2.py
역할: 우울 경향 전용 점수 v2 — evidence span / distress severity / persistence 3축 분리.
      v1.5와 달리 top_emotion 의존성을 완전히 제거하고, 매칭된 근거 문장과
      distress severity 스칼라(감정분류 비의존)를 fusion으로 합쳐 산출한다.

설계 문서: eval/report/depression_tendency_v2_design.md

입력 신호 (모두 옵션, 가용한 만큼 사용):
    - text: 사용자 발화 원문
    - distress_severity: 0.0~1.0 스칼라 (Phase 4-1 distress vector_T 기반 산출)
        * P(crisis_candidate)는 NLI/하드 인터럽트가 처리하므로 dts에는 가산하지 않는다.
    - utterance_type / type_reason: utterance_type.classify_utterance_type 결과 (cap 판정용)
    - cbt_score / cbt_non_distortion: 보조 (현재는 사용하지 않음, 호환성용)
    - is_crisis: 위기 후보 여부 (메타데이터로만 보존)
    - entailment_prob: NLI entailment 확률 (메타데이터로만 보존)

출력 dict:
    - depression_tendency_score: 0.0~0.95
    - version: "v2"
    - evidence: [{text, category, weight, start, end}, ...]
    - severity_band: "calm" | "mild" | "moderate" | "high"
    - severity_scalar: 입력 그대로 보존
    - persistence_band: "single_event" | "recurrent" | "persistent"
    - persistence_evidence: 매칭된 시간 단서 문자열 리스트
    - caps_applied: 적용된 cap 사유 리스트
    - raw_score_before_cap: cap 적용 전 점수
    - components: {evidence_score, severity_mult, persistence_mult}
    - is_crisis: 위기 여부 (메타데이터)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from pipeline.depression_tendency import (
    CATEGORY_KEYWORDS,
    CATEGORY_WEIGHTS,
    PERSISTENCE_MARKERS,
    SOFT_CLIP_MAX,
    CAP_DAILY_ROUTINE,
    CAP_POSITIVE_RECOVERY,
    CAP_SITUATIONAL_ANGER,
    CAP_PHYSICAL_EXERTION,
    CAP_ACADEMIC_ANXIETY,
    CAP_LIMITED_SITUATIONAL,
    CAP_SITUATIONAL_SADNESS,
    CAP_SINGLE_EVENT_DISTRESS,
    CAP_TRANSIENT_SINGLE_CAT,
    SINGLE_EVENT_DISTRESS_CONTEXTS,
    SINGLE_EVENT_DISTRESS_OUTCOMES,
    SINGLE_EVENT_SADNESS_OUTCOMES,
    TRANSIENT_MARKERS,
    MILD_SADNESS_PHRASES,
    PHYSICAL_V15_CONTEXTS,
    PHYSICAL_V15_OUTCOME_FRAGMENTS,
    PHYSICAL_V15_BLOCKERS,
)
from pipeline.utterance_type import (
    compact_text,
    has_crisis_marker,
    is_academic_anxiety_text,
    is_daily_routine_neutral_text,
    is_limited_situational_distress_text,
    is_physical_exertion_text,
    is_positive_affect_text,
    is_situational_anger_text,
    is_situational_sadness_text,
)


# ---------------------------------------------------------------------------
# v2 전용 상수
# ---------------------------------------------------------------------------

# 명시적 시간 단서 — persistence_band 판정용 (v1.5 PERSISTENCE_MARKERS 중 강한 신호)
EXPLICIT_DURATION_MARKERS = (
    "며칠째", "몇주째", "몇달째", "한달째", "한참째",
    "한달동안", "한참전부터", "예전부터", "예전엔", "예전에는",
    "오랫동안", "오래야",
)
# 반복 신호 (recurrent) — explicit_duration보다 약함
RECURRENT_MARKERS = (
    "요즘계속", "요즘자꾸", "요즘", "매일",
    "전부터", "자꾸", "쭉그래", "계속그래",
    "계속우울", "계속힘들", "마음이계속",
    "더심해", "더악화",
)

# severity multiplier 임계 (P_high+P_crisis 기반 스칼라 적용)
# severity_scalar = 0.0*P_calm + 0.25*P_mild + 0.50*P_moderate + 0.85*P_high + 0.0*P_crisis
SEVERITY_BANDS = (
    (0.65, "high", 1.00),
    (0.40, "moderate", 0.85),
    (0.20, "mild", 0.65),
    (0.0, "calm", 0.40),
)

# severity-only baseline (evidence 없음 + severity moderate+ 시) 활성 임계
SEVERITY_ONLY_BASELINE_THRESHOLD = 0.50
SEVERITY_ONLY_BASELINE_VALUE = 0.20


# ---------------------------------------------------------------------------
# 보조 함수
# ---------------------------------------------------------------------------

def _collect_evidence_spans(
    text: str,
    compact: str,
) -> Tuple[List[dict], dict]:
    """
    역할: v1.5 카테고리 키워드 매칭 + mild_sadness_phrase를 evidence span 형태로 수집.
    입력: 원문 text, compact_text 결과
    출력:
        - evidence_list: [{text, category, weight, start, end}, ...]
        - hits: {category: count}
    설명:
        - v1.5의 _count_hits를 evidence span 형태로 확장.
        - mild_sadness_phrase("마음이 안 좋아", "기분이 가라앉아" 등)도 별도
          'mild_sadness' 카테고리로 evidence 가산(weight 0.18) — v2 fusion에서
          v1.5의 sadness baseline(top_emotion 의존)을 대체하는 역할.
    """
    evidence: List[dict] = []
    hits: dict[str, int] = {}
    lower = text.lower()

    for category, keywords in CATEGORY_KEYWORDS.items():
        cat_count = 0
        base_weight, extra_weight = CATEGORY_WEIGHTS[category]
        for kw in keywords:
            if kw not in compact:
                continue
            cat_count += 1
            search_target = kw
            start = lower.find(search_target)
            end = start + len(search_target) if start >= 0 else -1
            evidence.append({
                "text": kw,
                "category": category,
                "weight": float(base_weight if cat_count == 1 else extra_weight),
                "start": int(start),
                "end": int(end),
            })
        if cat_count > 0:
            hits[category] = cat_count

    # mild_sadness_phrase — 별도 카테고리(weight 0.18)로 evidence 가산
    # v1.5의 sadness baseline(top_emotion=='슬픔'+emotional_distress 시 max(base, 0.20))
    # 을 top_emotion 의존 없이 대체.
    # v2 확장: v1.5 sadness baseline이 잡던 "서운/쓸쓸/허전" 계열도 mild_sadness로 직접 포함.
    extended_mild_phrases = tuple(MILD_SADNESS_PHRASES) + (
        "서운", "쓸쓸", "허전",
        "외로워", "외로움",
        "아쉬워", "아쉬움",
        "눈물날", "눈물이날", "눈물이나",
        "마음상함", "마음상해", "맘상함",
    )
    mild_count = 0
    seen_mild: set[str] = set()
    for phrase in extended_mild_phrases:
        if phrase in compact and phrase not in seen_mild:
            seen_mild.add(phrase)
            mild_count += 1
            start = lower.find(phrase)
            end = start + len(phrase) if start >= 0 else -1
            evidence.append({
                "text": phrase,
                "category": "mild_sadness",
                "weight": 0.18,
                "start": int(start),
                "end": int(end),
            })
    if mild_count > 0:
        hits["mild_sadness"] = mild_count

    return evidence, hits


def _probabilistic_or(scores: List[float]) -> float:
    """확률 OR 합산."""
    if not scores:
        return 0.0
    if len(scores) == 1:
        return float(scores[0])
    product = 1.0
    for s in scores:
        product *= max(0.0, 1.0 - s)
    return float(1.0 - product)


def _evidence_score(evidence: List[dict], hits: dict) -> float:
    """
    역할: evidence span을 카테고리별 합산 후 확률 OR로 결합.
    입력: evidence 리스트, hits dict
    출력: 0.0~1.0 evidence_score
    설명:
        - CATEGORY_WEIGHTS에 정의된 7개 핵심 카테고리는 base+extra 가중치 사용.
        - 'mild_sadness'는 별도 카테고리로 weight 0.18 고정 (sadness baseline 대체).
    """
    if not evidence:
        return 0.0
    cat_scores: dict[str, float] = {}
    for cat, count in hits.items():
        if cat in CATEGORY_WEIGHTS:
            base, extra = CATEGORY_WEIGHTS[cat]
            bonus = extra if count >= 2 else 0.0
            cat_scores[cat] = base + bonus
        elif cat == "mild_sadness":
            cat_scores[cat] = 0.18
    return _probabilistic_or(list(cat_scores.values()))


def _resolve_severity_band(severity_scalar: float) -> Tuple[str, float]:
    """
    역할: severity 스칼라 → band + multiplier.
    입력: 0.0~1.0
    출력: (band_name, multiplier)
    """
    for threshold, band, mult in SEVERITY_BANDS:
        if severity_scalar >= threshold:
            return band, mult
    return "calm", SEVERITY_BANDS[-1][2]


def _resolve_persistence(
    text: str,
    compact: str,
    has_evidence: bool,
) -> Tuple[str, List[str]]:
    """
    역할: persistence_band 판정 + 매칭된 시간 단서 리스트.
    입력: 원문, compact_text, evidence 보유 여부
    출력: (band, persistence_evidence_list)
    설명:
        - explicit_duration 키워드 1개 이상 + evidence 1+ → persistent
        - recurrent 키워드 1개 이상 + evidence 1+ → recurrent
        - 그 외 → single_event
    """
    explicit_hits = [m for m in EXPLICIT_DURATION_MARKERS if m in compact]
    if explicit_hits and has_evidence:
        return "persistent", explicit_hits
    recurrent_hits = [m for m in RECURRENT_MARKERS if m in compact]
    if recurrent_hits and has_evidence:
        return "recurrent", recurrent_hits
    return "single_event", []


def _persistence_multiplier(band: str) -> float:
    """persistence band → multiplier."""
    return {"persistent": 1.50, "recurrent": 1.30, "single_event": 1.00}.get(band, 1.00)


def _collect_caps(
    text: str,
    compact: str,
    hits: dict,
    persistence_band: str,
    has_crisis: bool,
) -> List[Tuple[float, str]]:
    """
    역할: v1.5 cap 우선순위를 그대로 재사용해 cap 후보 리스트 반환.
    입력: text, compact, hits, persistence_band, crisis 여부
    출력: [(cap_value, reason), ...] (정렬 안 됨)
    """
    cap_candidates: List[Tuple[float, str]] = []
    has_mild_sadness_phrase = any(p in compact for p in MILD_SADNESS_PHRASES)

    if not hits and not has_mild_sadness_phrase:
        if is_daily_routine_neutral_text(text):
            cap_candidates.append((CAP_DAILY_ROUTINE, "daily_routine_neutral_cap"))
        if is_positive_affect_text(text) and not has_crisis:
            cap_candidates.append((CAP_POSITIVE_RECOVERY, "positive_affect_safety_cap"))

    if is_situational_anger_text(text):
        cap_candidates.append((CAP_SITUATIONAL_ANGER, "situational_anger_cap"))

    if is_physical_exertion_text(text):
        cap_candidates.append((CAP_PHYSICAL_EXERTION, "physical_exertion_context"))
    elif _has_v15_physical(compact):
        cap_candidates.append((CAP_PHYSICAL_EXERTION, "physical_exertion_v15"))

    if is_academic_anxiety_text(text):
        cap_candidates.append((CAP_ACADEMIC_ANXIETY, "academic_anxiety_cbt_cap"))

    if is_limited_situational_distress_text(text) and persistence_band == "single_event":
        cap_candidates.append((CAP_LIMITED_SITUATIONAL, "limited_situational_distress_cap"))

    if is_situational_sadness_text(text) and persistence_band == "single_event" and len(hits) <= 1:
        cap_candidates.append((CAP_SITUATIONAL_SADNESS, "situational_sadness_cap"))

    has_single_event_ctx = any(c in compact for c in SINGLE_EVENT_DISTRESS_CONTEXTS)
    has_single_event_outcome = any(o in compact for o in SINGLE_EVENT_DISTRESS_OUTCOMES)
    if (
        has_single_event_ctx
        and has_single_event_outcome
        and persistence_band == "single_event"
        and len(hits) <= 1
    ):
        has_sadness_outcome = any(o in compact for o in SINGLE_EVENT_SADNESS_OUTCOMES)
        is_sit_sadness = is_situational_sadness_text(text)
        if has_mild_sadness_phrase or has_sadness_outcome or hits or is_sit_sadness:
            cap_candidates.append((CAP_SINGLE_EVENT_DISTRESS, "single_event_distress_cap"))
        else:
            cap_candidates.append((0.15, "single_event_distress_no_sadness_cap"))

    transient_hit = any(m in compact for m in TRANSIENT_MARKERS)
    if transient_hit and len(hits) <= 1 and persistence_band == "single_event" and not has_crisis:
        cap_candidates.append((CAP_TRANSIENT_SINGLE_CAT, "transient_single_category_cap"))

    return cap_candidates


def _has_v15_physical(compact: str) -> bool:
    """v1.5 physical_v15 보조 감지 (depression_tendency.py와 동일 로직)."""
    has_ctx = any(c in compact for c in PHYSICAL_V15_CONTEXTS)
    has_out = any(o in compact for o in PHYSICAL_V15_OUTCOME_FRAGMENTS)
    has_blocker = any(b in compact for b in PHYSICAL_V15_BLOCKERS)
    return has_ctx and has_out and not has_blocker


# ---------------------------------------------------------------------------
# 메인 함수
# ---------------------------------------------------------------------------

def compute_depression_tendency_v2(
    text: str,
    *,
    distress_severity: Optional[float] = None,
    utterance_type: Optional[str] = None,
    type_reason: Optional[str] = None,
    cbt_score: Optional[float] = None,
    cbt_non_distortion: Optional[bool] = None,
    is_crisis: bool = False,
    entailment_prob: Optional[float] = None,
    # 호환성: 기존 호출자가 top_emotion / roberta_score 인자를 넘겨도 무시
    top_emotion: Optional[str] = None,
    roberta_score: Optional[float] = None,
) -> dict:
    """
    역할: 우울 경향 점수 v2 산출 (감정분류 비의존, evidence + severity + persistence 3축 fusion).
    입력:
        text: 사용자 발화 원문
        distress_severity: 0.0~1.0 (감정분류 비의존), None이면 0.0
        utterance_type, type_reason: cap 판정 보조
        cbt_score, cbt_non_distortion: 호환성 (현재 미사용)
        is_crisis, entailment_prob: 메타데이터
        top_emotion, roberta_score: 호환성 (v2는 의도적으로 무시)
    출력: 설계 문서 §4 audit 스키마 dict
    """
    compact = compact_text(text)
    has_crisis = bool(is_crisis) or has_crisis_marker(text)
    severity = max(0.0, min(1.0, float(distress_severity or 0.0)))

    # 1. evidence span 수집
    evidence, hits = _collect_evidence_spans(text, compact)

    # 2. evidence_score
    evidence_score = _evidence_score(evidence, hits)

    # 3. severity band/multiplier
    severity_band, severity_mult = _resolve_severity_band(severity)

    # 4. persistence band
    persistence_band, persistence_evidence = _resolve_persistence(
        text, compact, has_evidence=bool(evidence)
    )
    persistence_mult = _persistence_multiplier(persistence_band)

    # 5. base = evidence × persistence (severity는 보조 부스트로만 사용)
    #    설계 변경: severity multiplier가 evidence 신호를 깎지 않도록,
    #    evidence가 있을 때는 1.0 사용. evidence 없을 때만 severity_mult 적용 (baseline용).
    if evidence:
        base = float(evidence_score * persistence_mult)
        # evidence 강한데 severity가 high라면 약간의 보너스 (multiplier 1.10)
        if severity >= 0.65:
            base = float(min(SOFT_CLIP_MAX, base * 1.10))
    else:
        base = 0.0

    # 7. cap 후보 수집 (severity-only baseline 발동 결정에 사용)
    cap_candidates = _collect_caps(text, compact, hits, persistence_band, has_crisis)
    cap_candidates.sort(key=lambda x: x[0])
    smallest_cap = cap_candidates[0][0] if cap_candidates else None

    # 6. severity-only baseline (evidence 0 + severity moderate+ + cap이 baseline 미만이면 보류)
    #    cap이 baseline 값 이하라면 baseline 발동 시 그 cap에 막힐 뿐, low band가 mid로 잘못
    #    분류될 수 있어 보류한다. (예: academic_anxiety_cap=0.20과 baseline 0.20이 일치 시 mid 경계)
    severity_only_baseline_used = False
    if (
        not evidence
        and severity >= SEVERITY_ONLY_BASELINE_THRESHOLD
        and not has_crisis
        and (smallest_cap is None or smallest_cap >= SEVERITY_ONLY_BASELINE_VALUE + 0.05)
    ):
        base = max(base, SEVERITY_ONLY_BASELINE_VALUE)
        severity_only_baseline_used = True

    # 8. cap 적용
    caps_applied: List[str] = []
    raw_before_cap = base
    if cap_candidates:
        cap_value, cap_reason = cap_candidates[0]
        if base > cap_value:
            base = cap_value
            caps_applied.append(cap_reason)

    # 8. soft clip
    final_score = max(0.0, min(SOFT_CLIP_MAX, base))

    return {
        "depression_tendency_score": round(final_score, 4),
        "version": "v2",
        "evidence": evidence,
        "hit_categories": sorted(hits.keys()),
        "category_hit_counts": dict(hits),
        "severity_band": severity_band,
        "severity_scalar": round(severity, 4),
        "persistence_band": persistence_band,
        "persistence_evidence": persistence_evidence,
        "caps_applied": caps_applied,
        "raw_score_before_cap": round(raw_before_cap, 4),
        "components": {
            "evidence_score": round(evidence_score, 4),
            "severity_mult": round(severity_mult, 4),
            "persistence_mult": round(persistence_mult, 4),
        },
        "severity_only_baseline_used": bool(severity_only_baseline_used),
        "is_crisis": bool(has_crisis),
    }


def severity_scalar_from_distress_probs(probs: List[float]) -> float:
    """
    역할: distress severity head softmax 출력 → 0~1 스칼라 변환.
    입력: 5클래스 확률 [calm, mild, moderate, high, crisis]
    출력: 0.0~1.0 severity_scalar
    설명:
        crisis_candidate은 NLI/하드 인터럽트가 처리하므로 dts 가산에서 0으로 둔다.
        이중 카운팅 방지.
    """
    if not probs or len(probs) < 5:
        return 0.0
    p_calm, p_mild, p_moderate, p_high, _p_crisis = probs[:5]
    # crisis weight = 0.0 (NLI가 별도 처리)
    return float(0.0 * p_calm + 0.25 * p_mild + 0.50 * p_moderate + 0.85 * p_high + 0.0 * _p_crisis)
