"""
ensemble.py
역할: roberta_score + cbt_score 앙상블 → depression_score 산출
      두 점수 불일치(차이 > 0.3) 시 보수적 판단(max 값 채택)
      Phase 5 fusion: distress_severity 신호와 합의하지 않는 CBT 단독 폭주를 cap.
입력: roberta_score, cbt_score, 위기 여부, (옵션) distress_severity
출력: depression_score (float, 0~1) + fusion 메타
"""

# 앙상블 가중치 (cbt_score는 4주차 threshold 확정 전까지 가중치 낮게 유지)
W_ROBERTA = 0.7
W_CBT     = 0.3

# 두 점수 불일치 판단 기준 (절대 차이)
DISAGREEMENT_THRESHOLD = 0.3

# Phase 5 — CBT 단독 폭주 감지 임계
#   cbt가 roberta보다 크게 높고 + roberta가 낮음 + distress_severity가 낮음 →
#   CBT anchor가 단독으로 위험도를 올리는 false positive로 본다.
CBT_SOLO_BURST_DIFF = 0.30        # cbt - roberta >= 0.30
CBT_SOLO_BURST_ROBERTA_MAX = 0.40 # roberta < 0.40
CBT_SOLO_BURST_SEVERITY_MAX = 0.30 # distress_severity < 0.30
CBT_SOLO_BURST_CAP_HEADROOM = 0.10 # cap = roberta + 0.10


def ensemble_scores(
    roberta_score: float,
    cbt_score: float | None = None,
    *,
    distress_severity: float | None = None,
) -> dict:
    """
    역할: roberta_score와 cbt_score를 앙상블해 depression_score 산출
          cbt_score가 None이면 roberta_score만 사용.
          distress_severity가 주어지면 CBT 단독 폭주 케이스에서 CBT를 cap.
    입력:
        roberta_score (0~1)
        cbt_score (0~1 또는 None)
        distress_severity (0~1, 옵션) — 감정분류 비의존 distress 강도 스칼라.
            현재 운영에서는 emotion 7-class prob 기반 proxy 사용.
    출력: {
        depression_score: float (0~1),
        method: str ("weighted" | "max_conservative" | "roberta_only"),
        fusion_caps: list[dict] (적용된 fusion cap 사유와 변환값),
        cbt_score_effective: float | None (cap 적용 후 실제 사용된 cbt)
    }
    """
    if cbt_score is None:
        return {
            "depression_score": float(roberta_score),
            "method": "roberta_only",
            "fusion_caps": [],
            "cbt_score_effective": None,
        }

    fusion_caps: list[dict] = []
    cbt_effective = float(cbt_score)

    # Phase 5 — CBT 단독 폭주 cap
    # cbt가 roberta보다 크게 높고, roberta는 낮은 위험, distress signal도 약하면
    # CBT anchor 단독으로 점수를 끌어올린 케이스로 보고 cbt를 roberta + headroom으로 cap.
    # distress_severity가 None(legacy)이면 이 cap은 적용하지 않는다(backward-compat).
    if (
        distress_severity is not None
        and cbt_effective - roberta_score >= CBT_SOLO_BURST_DIFF
        and roberta_score < CBT_SOLO_BURST_ROBERTA_MAX
        and distress_severity < CBT_SOLO_BURST_SEVERITY_MAX
    ):
        cap_value = float(roberta_score + CBT_SOLO_BURST_CAP_HEADROOM)
        if cbt_effective > cap_value:
            fusion_caps.append({
                "reason": "cbt_solo_burst_cap",
                "cbt_before": float(cbt_score),
                "cbt_after": cap_value,
                "roberta_score": float(roberta_score),
                "distress_severity": float(distress_severity),
            })
            cbt_effective = cap_value

    diff = abs(roberta_score - cbt_effective)

    if diff > DISAGREEMENT_THRESHOLD:
        # 불일치 시 더 높은(보수적) 값 채택 — 위기 방향으로 오류를 허용
        depression_score = max(roberta_score, cbt_effective)
        method = "max_conservative"
    else:
        depression_score = W_ROBERTA * roberta_score + W_CBT * cbt_effective
        method = "weighted"

    return {
        "depression_score": float(depression_score),
        "method": method,
        "fusion_caps": fusion_caps,
        "cbt_score_effective": float(cbt_effective),
    }
