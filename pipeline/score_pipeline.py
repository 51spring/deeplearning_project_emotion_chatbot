"""
score_pipeline.py
역할: 발화 단위 전체 점수 산출 파이프라인 진입점
      RoBERTa 추론 → CBT 유사도 → 앙상블 → EWMA → wellness_score 순서로 실행
      ⚠️ RoBERTa와 Qwen을 동시에 메모리에 올리지 말 것 (RTX 3060Ti 8GB 제한)
입력: 발화 텍스트, 사전 로드된 모델 컴포넌트
출력: {roberta_score, cbt_score, depression_score, wellness_score, label, is_crisis, ...}
"""

import os
from pipeline.roberta_score  import (
    infer_single,
    load_temperature,
    load_emotion_vector_T,
    load_emotion_logit_bias,
    ROBERTA_SCORE_P95,
)
from pipeline.cbt_similarity import compute_cbt_score
from pipeline.cbt_reliability import evaluate_cbt_reliability
from pipeline.depression_tendency import compute_depression_tendency
from pipeline.ensemble       import ensemble_scores
from pipeline.ewma           import utterance_to_daily, daily_to_smoothed
from pipeline.score_policy   import compute_wellness_contribution, LOW_POSITIVE_SCORE_MIN
from pipeline.wellness_score import (
    NO_SIGNAL_DEPRESSION_SCORE,
    apply_no_signal_floor,
    compute_wellness,
    depression_to_display_wellness,
    display_wellness_label,
)


def _floor_no_signal_daily(score: float) -> float:
    """
    역할: 2단계 EWMA 평활 입력 보정 — 무신호/레거시(daily_score=0.0 등) 일별 점수만
          baseline(0.30, wellness 70)으로 올리고, 명확한 긍정 day(긍정 floor 0.12 이상)는
          양호권 점수를 그대로 통과시킨다.
          기존 apply_no_signal_floor(max(score, 0.30))는 긍정 day(예: 0.154)까지 0.30으로 눌러
          마감/캘린더 웰니스가 70 천장이었다. 실측상 [0.12, 0.22) 구간은 긍정 day만 만들고
          중립 day는 ~0.38이라 값만으로 안전하게 분리된다.
    입력: 일별 점수(daily_score)
    출력: 평활 입력용 보정 점수
    """
    if float(score) < LOW_POSITIVE_SCORE_MIN:
        return NO_SIGNAL_DEPRESSION_SCORE
    return float(score)


class ScorePipeline:
    """
    역할: 발화 점수 산출 파이프라인 — 상태(일별 히스토리)를 보존하며 연속 호출 지원
    """

    def __init__(
        self,
        model,
        tokenizer,
        device,
        anchor_embs: dict,
        p95: float = ROBERTA_SCORE_P95,
        use_cbt: bool = True,
    ):
        """
        역할: 파이프라인 초기화
        입력: RoBERTa 모델/토크나이저/디바이스, CBT 앵커 임베딩 dict,
              P95 기준값, CBT 사용 여부
        """
        self.model       = model
        self.tokenizer   = tokenizer
        self.device      = device
        self.anchor_embs = anchor_embs
        self.p95         = p95
        self.use_cbt     = use_cbt

        self.T_emotion, self.T_nli = load_temperature()
        # Vector Scaling 채택 시 7-dim list, 미설정이면 None → 단일 T 폴백.
        self.vector_T_emotion = load_emotion_vector_T()
        # 선택 채택된 additive logit bias. 기본 미설정(None)이면 기존 운영 동작을 유지한다.
        self.emotion_logit_bias = load_emotion_logit_bias()

        # 발화 단위 점수 버퍼 (당일 EWMA 계산용)
        self._today_utterances: list[float] = []
        # 날짜별 daily_score 히스토리 (EWMA 2단계용)
        self._daily_scores: list[float] = []
        # 날짜별 wellness_score 히스토리 (레이블 결정용)
        self._daily_wellness: list[float] = []

        # ── Depression Tendency v1.5 ──
        # 우울 경향 전용 발화 버퍼/일별 히스토리 — 기존 종합 distress 축과 병렬 운영
        self._today_tendency: list[float] = []
        self._daily_tendency: list[float] = []

    def score_utterance(self, text: str) -> dict:
        """
        역할: 단일 발화 점수 산출 (발화 버퍼에 추가)
        입력: 발화 텍스트
        출력: {
            roberta_score, cbt_score, depression_score,
            is_crisis, top_emotion, entailment_prob,
            method (앙상블 방식)
        }
        """
        # 1. RoBERTa 추론
        roberta_result = infer_single(
            text, self.model, self.tokenizer, self.device,
            T_emotion=self.T_emotion,
            T_nli=self.T_nli,
            p95=self.p95,
            vector_T_emotion=self.vector_T_emotion,
            emotion_logit_bias=getattr(self, "emotion_logit_bias", None),
        )

        # 2. CBT 유사도
        cbt_score = None
        cbt_detail = {}
        if self.use_cbt and self.anchor_embs:
            cbt_result = compute_cbt_score(
                text, self.model, self.tokenizer, self.device, self.anchor_embs
            )
            cbt_score  = cbt_result["cbt_score"]
            cbt_detail = cbt_result

        # 3. 앙상블 → depression_score(종합 distress / wellness risk)
        distress_severity = roberta_result.get("distress_severity_scalar")
        if distress_severity is None:
            emotion_probs = roberta_result.get("emotion_probs") or []
            if len(emotion_probs) >= 7:
                # distress head가 없던 legacy 호출에서도 scheduler와 같은 proxy를 사용한다.
                distress_severity = min(
                    1.0,
                    (
                        float(emotion_probs[2])
                        + float(emotion_probs[3])
                        + float(emotion_probs[5])
                    ) * 0.85,
                )

        cbt_reliability = evaluate_cbt_reliability(
            cbt_score=cbt_score,
            cbt_result=cbt_detail,
            roberta_out=roberta_result,
            cbt_head_pred=None,
            depression_tendency_score=None,
            distress_severity=distress_severity,
        )
        if cbt_reliability["cbt_reliability_applied"]:
            cbt_score = cbt_reliability["cbt_score_after_reliability"]

        ens = ensemble_scores(
            roberta_result["roberta_score"],
            cbt_score,
            distress_severity=distress_severity,
        )

        # 4. 우울 경향 전용 축(v1.5.2)과 웰니스 반영 정책을 운영 경로와 맞춘다.
        analysis_text = roberta_result.get("analysis_text") or text
        dep_tendency = compute_depression_tendency(
            analysis_text,
            top_emotion=roberta_result.get("top_emotion"),
            roberta_score=roberta_result.get("roberta_score"),
            cbt_score=cbt_score,
            cbt_non_distortion=cbt_detail.get("cbt_head_non_distortion"),
            utterance_type=roberta_result.get("utterance_type"),
            type_reason=roberta_result.get("utterance_type_reason"),
            is_crisis=roberta_result.get("is_crisis", False),
            entailment_prob=roberta_result.get("entailment_prob"),
        )

        result = {
            **roberta_result,
            "cbt_score":       cbt_score,
            "cbt_effect": cbt_reliability["cbt_effect"],
            "cbt_reliability_policy": cbt_reliability["cbt_reliability_policy"],
            "cbt_reliability_applied": cbt_reliability["cbt_reliability_applied"],
            "cbt_reliability_cap": cbt_reliability["cbt_reliability_cap"],
            "cbt_reliability_risk_points": cbt_reliability["cbt_reliability_risk_points"],
            "cbt_reliability_benign_points": cbt_reliability["cbt_reliability_benign_points"],
            "cbt_reliability_reasons": cbt_reliability["cbt_reliability_reasons"],
            "depression_score": ens["depression_score"],
            "method":          ens["method"],
            "ensemble_method":  ens["method"],
            "ensemble_fusion_caps": ens.get("fusion_caps", []),
            "cbt_score_effective": ens.get("cbt_score_effective", cbt_score),
            "depression_tendency_score": dep_tendency["depression_tendency_score"],
            "depression_tendency_categories": dep_tendency["hit_categories"],
            "depression_tendency_caps": dep_tendency["caps_applied"],
            "depression_tendency_persistence": dep_tendency["persistence_marker_hit"],
            "depression_tendency_version": dep_tendency["version"],
            **({k: v for k, v in cbt_detail.items() if k != "cbt_score"} if cbt_detail else {}),
        }
        contribution = compute_wellness_contribution(
            result,
            is_crisis=result.get("is_crisis", False),
        )
        result.update(contribution)

        # 당일 EWMA에는 raw depression_score가 아니라 score_policy 통과 기여 점수만 넣는다.
        if result["score_affects_wellness"]:
            self._today_utterances.append(float(result["wellness_contribution_score"]))
            self._today_tendency.append(float(result["depression_tendency_score"]))

        return result

    def close_day(self) -> dict:
        """
        역할: 하루 종료 시 호출 — 발화 버퍼를 EWMA로 집계해 daily_score + wellness_score 산출
              우울 경향 전용 일별/평활 점수도 같은 단계 구조로 함께 계산해 응답에 포함한다.
        출력: {daily_score, smoothed_score, daily_wellness_score,
               daily_wellness_label, cumulative_wellness_score,
               cumulative_wellness_label, wellness_score, label, n_days,
               depression_tendency_daily, depression_tendency_smoothed}
        """
        if not self._today_utterances:
            previous_score = (
                self._daily_scores[-1]
                if self._daily_scores
                else NO_SIGNAL_DEPRESSION_SCORE
            )
            daily_score = apply_no_signal_floor(previous_score)
        else:
            daily_score = utterance_to_daily(self._today_utterances)

        self._daily_scores.append(daily_score)
        self._today_utterances = []  # 버퍼 초기화

        # 2단계 EWMA: 날짜 시퀀스 전체 재계산
        # 무신호/레거시(0.0 등)만 baseline(0.30)으로 올리고, 명확한 긍정 day는 양호권으로 통과시킨다.
        smoothed = daily_to_smoothed([
            _floor_no_signal_daily(score) for score in self._daily_scores
        ])
        smoothed_score = smoothed[-1]

        # wellness + 레이블
        wellness_result = compute_wellness(
            depression_score=smoothed_score,
            history_wellness=self._daily_wellness,
            n_days=len(self._daily_scores),
        )
        self._daily_wellness.append(wellness_result["wellness_score"])

        # ── Depression Tendency v1.5 일별/평활 ──
        # 종합 depression_score와 동일한 EWMA 구조를 우울 경향 전용 축에도 적용한다.
        if not self._today_tendency:
            tendency_daily = self._daily_tendency[-1] if self._daily_tendency else 0.0
        else:
            tendency_daily = utterance_to_daily(self._today_tendency)
        self._daily_tendency.append(tendency_daily)
        self._today_tendency = []
        tendency_smoothed = daily_to_smoothed(self._daily_tendency)[-1]

        daily_wellness_score = depression_to_display_wellness(daily_score)

        return {
            "daily_score":     round(daily_score, 4),
            "smoothed_score":  round(smoothed_score, 4),
            "daily_wellness_score": daily_wellness_score,
            "daily_wellness_label": display_wellness_label(daily_wellness_score),
            "cumulative_wellness_score": wellness_result["wellness_score"],
            "cumulative_wellness_label": wellness_result["label"],
            "wellness_score":  wellness_result["wellness_score"],
            "label":           wellness_result["label"],
            "n_days":          len(self._daily_scores),
            "depression_tendency_daily":    round(tendency_daily, 4),
            "depression_tendency_smoothed": round(tendency_smoothed, 4),
        }

    def reset(self):
        """역할: 파이프라인 상태 전체 초기화 (새 사용자 세션용)"""
        self._today_utterances = []
        self._daily_scores     = []
        self._daily_wellness   = []
        self._today_tendency   = []
        self._daily_tendency   = []
