"""
cbt_reliability_guard.py
역할: CBT reliability gate v1이 문장별 guard가 아니라 feature 조합으로 동작하는지 검증한다.
입력: 없음(내장 케이스)
출력: 콘솔 검증 결과 및 실패 시 AssertionError
실행: C:/Users/WD/anaconda3/envs/dl_study/python.exe eval/cbt_reliability_guard.py
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from pipeline.cbt_reliability import evaluate_cbt_reliability


def make_roberta_out(
    *,
    utterance_type: str,
    top_emotion: str,
    roberta_score: float,
    entailment_prob: float = 0.0,
    is_crisis: bool = False,
) -> dict:
    """
    역할: reliability gate 테스트용 RoBERTa 출력 dict를 만든다.
    입력: 발화 타입, top 감정, roberta_score, NLI 확률, 위기 여부
    출력: RoBERTa 출력 형태 dict
    """
    return {
        "utterance_type": utterance_type,
        "top_emotion": top_emotion,
        "roberta_score": roberta_score,
        "entailment_prob": entailment_prob,
        "is_crisis": is_crisis,
    }


def make_cbt_result(raw_cbt: float, raw_contrast: float, top_category: str = "파국화") -> dict:
    """
    역할: reliability gate 테스트용 CBT anchor 결과 dict를 만든다.
    입력: 왜곡 anchor raw 유사도, contrast raw 유사도, top category
    출력: CBT 결과 형태 dict
    """
    return {
        "raw_cbt": raw_cbt,
        "raw_contrast": raw_contrast,
        "top_category": top_category,
    }


def make_head(confidence: float, *, non_distortion: bool = False) -> dict:
    """
    역할: reliability gate 테스트용 CBT class head 결과 dict를 만든다.
    입력: 왜곡 top confidence, 비왜곡 여부
    출력: CBT head 결과 형태 dict
    """
    return {
        "top_distortion_label": "파국화",
        "top_distortion_confidence": confidence,
        "is_non_distortion": non_distortion,
    }


def assert_benign_low_head_conf_is_capped() -> None:
    """
    역할: 저신호/저우울/low-head-conf anchor가 넓은 feature 조합으로 cap 되는지 확인한다.
    입력: 없음
    출력: 없음
    """
    out = evaluate_cbt_reliability(
        cbt_score=0.66,
        cbt_result=make_cbt_result(0.70, 0.54),
        roberta_out=make_roberta_out(
            utterance_type="casual_share",
            top_emotion="중립",
            roberta_score=0.35,
            entailment_prob=0.0,
        ),
        cbt_head_pred=make_head(0.41),
        depression_tendency_score=0.0,
        distress_severity=0.28,
    )
    assert out["cbt_reliability_applied"], out
    assert out["cbt_score_after_reliability"] == 0.45, out
    assert out["cbt_reliability_policy"] == "low_head_conf_benign_context_cap", out
    print("[CBT gate] 저신호 low-head-conf anchor → 0.45 cap")


def assert_ambiguous_low_head_conf_is_soft_capped() -> None:
    """
    역할: 일부 우울 경향 힌트가 있는 low-head-conf anchor는 제거 대신 0.55로 완화되는지 확인한다.
    입력: 없음
    출력: 없음
    """
    out = evaluate_cbt_reliability(
        cbt_score=0.68,
        cbt_result=make_cbt_result(0.71, 0.53),
        roberta_out=make_roberta_out(
            utterance_type="emotional_distress",
            top_emotion="중립",
            roberta_score=0.35,
            entailment_prob=0.0,
        ),
        cbt_head_pred=make_head(0.42),
        depression_tendency_score=0.40,
        distress_severity=0.29,
    )
    assert out["cbt_reliability_applied"], out
    assert out["cbt_score_after_reliability"] == 0.55, out
    assert out["cbt_reliability_policy"] == "low_head_conf_ambiguous_cap", out
    print("[CBT gate] 애매한 low-head-conf anchor → 0.55 soft cap")


def assert_safety_signal_is_preserved() -> None:
    """
    역할: 위기/안전 신호가 있으면 low-head-conf라도 CBT를 낮추지 않는지 확인한다.
    입력: 없음
    출력: 없음
    """
    out = evaluate_cbt_reliability(
        cbt_score=0.89,
        cbt_result=make_cbt_result(0.86, 0.47),
        roberta_out=make_roberta_out(
            utterance_type="emotional_distress",
            top_emotion="슬픔",
            roberta_score=1.0,
            entailment_prob=0.55,
            is_crisis=True,
        ),
        cbt_head_pred=make_head(0.40),
        depression_tendency_score=0.0,
        distress_severity=0.90,
    )
    assert not out["cbt_reliability_applied"], out
    assert out["cbt_effect"] == "full", out
    assert out["cbt_reliability_policy"] == "safety_signal_preserve", out
    print("[CBT gate] 위기/안전 신호 → CBT full 보존")


def assert_strong_head_with_distress_is_preserved() -> None:
    """
    역할: CBT head가 강하게 왜곡을 지지하고 distress 근거가 있으면 full 보존되는지 확인한다.
    입력: 없음
    출력: 없음
    """
    out = evaluate_cbt_reliability(
        cbt_score=0.72,
        cbt_result=make_cbt_result(0.77, 0.55),
        roberta_out=make_roberta_out(
            utterance_type="emotional_distress",
            top_emotion="슬픔",
            roberta_score=0.61,
            entailment_prob=0.1,
        ),
        cbt_head_pred=make_head(0.72),
        depression_tendency_score=0.25,
        distress_severity=0.50,
    )
    assert not out["cbt_reliability_applied"], out
    assert out["cbt_effect"] == "full", out
    assert out["cbt_reliability_policy"] == "head_strong_with_risk_preserve", out
    print("[CBT gate] 강한 CBT head + distress 근거 → full 보존")


def assert_non_distortion_head_is_capped() -> None:
    """
    역할: CBT head가 비왜곡으로 보는 anchor 고점은 낮은 cap으로만 반영되는지 확인한다.
    입력: 없음
    출력: 없음
    """
    out = evaluate_cbt_reliability(
        cbt_score=0.64,
        cbt_result=make_cbt_result(0.68, 0.54),
        roberta_out=make_roberta_out(
            utterance_type="casual_neutral",
            top_emotion="중립",
            roberta_score=0.33,
            entailment_prob=0.0,
        ),
        cbt_head_pred=make_head(0.48, non_distortion=True),
        depression_tendency_score=0.0,
        distress_severity=0.20,
    )
    assert out["cbt_reliability_applied"], out
    assert out["cbt_score_after_reliability"] == 0.45, out
    assert out["cbt_reliability_policy"] == "head_non_distortion_low_cap", out
    print("[CBT gate] CBT head 비왜곡 anchor → 0.45 cap")


def assert_below_threshold_is_not_touched() -> None:
    """
    역할: 임계 미만 CBT 점수는 reliability gate가 추가로 건드리지 않는지 확인한다.
    입력: 없음
    출력: 없음
    """
    out = evaluate_cbt_reliability(
        cbt_score=0.52,
        cbt_result=make_cbt_result(0.61, 0.59),
        roberta_out=make_roberta_out(
            utterance_type="casual_share",
            top_emotion="중립",
            roberta_score=0.34,
            entailment_prob=0.0,
        ),
        cbt_head_pred=make_head(0.20),
        depression_tendency_score=0.0,
        distress_severity=0.20,
    )
    assert not out["cbt_reliability_applied"], out
    assert out["cbt_score_after_reliability"] == 0.52, out
    assert out["cbt_reliability_policy"] == "below_threshold_no_gate", out
    print("[CBT gate] 임계 미만 CBT → 추가 변경 없음")


def main() -> None:
    """
    역할: CBT reliability gate v1 회귀 검증을 실행한다.
    입력: 없음
    출력: 없음
    """
    assert_benign_low_head_conf_is_capped()
    assert_ambiguous_low_head_conf_is_soft_capped()
    assert_safety_signal_is_preserved()
    assert_strong_head_with_distress_is_preserved()
    assert_non_distortion_head_is_capped()
    assert_below_threshold_is_not_touched()
    print("[완료] CBT reliability gate v1 검증 통과")


if __name__ == "__main__":
    main()
