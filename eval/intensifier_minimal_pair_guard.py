"""
intensifier_minimal_pair_guard.py
역할: 강조어 추가 전후 최소쌍을 실제 RoBERTa+CBT 경로로 재추론해 점수 안정성을 검증한다.
입력: 없음 (내장 최소쌍)
출력: 콘솔 검증 결과 및 실패 시 AssertionError
실행: C:/Users/WD/anaconda3/envs/dl_study/python.exe -u eval/intensifier_minimal_pair_guard.py
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from backend.scheduler import ModelScheduler


def _assert_low_risk_pair(
    scheduler: ModelScheduler,
    base_text: str,
    emphasized_text: str,
    max_delta: float,
) -> None:
    """
    역할: 저위험 최소쌍의 depression_score 차이가 허용 범위 안인지 확인한다.
    입력: 스케줄러, 원문, 강조어 포함 문장, 허용 최대 차이
    출력: 없음
    """
    base = scheduler.run_roberta(base_text)
    emphasized = scheduler.run_roberta(emphasized_text)
    delta = abs(float(emphasized["depression_score"]) - float(base["depression_score"]))
    assert delta <= max_delta + 1e-6, {
        "base_text": base_text,
        "emphasized_text": emphasized_text,
        "base_score": base["depression_score"],
        "emphasized_score": emphasized["depression_score"],
        "delta": delta,
        "max_delta": max_delta,
        "base_type": base.get("utterance_type"),
        "emphasized_type": emphasized.get("utterance_type"),
        "emphasized_guard": emphasized.get("intensifier_delta_guard"),
        "emphasized_cbt_guard": emphasized.get("intensifier_cbt_delta_guard"),
    }
    print(
        "[최소쌍 확인] "
        f"{base_text} / {emphasized_text} → "
        f"{base['depression_score']:.4f} / {emphasized['depression_score']:.4f} "
        f"(delta={delta:.4f})"
    )


def _assert_distress_not_flattened(
    scheduler: ModelScheduler,
    text: str,
    min_depression_score: float,
    min_roberta_score: float = 0.55,
    expected_top_emotion: str | None = None,
) -> None:
    """
    역할: 정서 distress 신호가 강조어 guard나 CBT reliability gate 때문에 저신호로 평탄화되지 않는지 확인한다.
    입력: 스케줄러, 검증 문장, 최소 종합 점수, 최소 RoBERTa 점수, 기대 대표 감정
    출력: 없음
    """
    result = scheduler.run_roberta(text)
    assert result.get("utterance_type") in {"emotional_distress", "crisis_candidate"}, result
    assert result.get("intensifier_delta_guard") is None, result
    assert float(result["roberta_score"]) >= min_roberta_score, result
    assert float(result["depression_score"]) >= min_depression_score, result
    if expected_top_emotion is not None:
        assert result.get("top_emotion") == expected_top_emotion, result
    print(
        "[보존 확인] "
        f"{text} → type={result.get('utterance_type')}, "
        f"roberta={result['roberta_score']:.4f}, "
        f"score={result['depression_score']:.4f}"
    )


def main() -> None:
    """
    역할: 실제 모델 기반 강조어 최소쌍 회귀 검증을 실행한다.
    입력: 없음
    출력: 없음
    """
    scheduler = ModelScheduler(use_cbt=True)
    _assert_low_risk_pair(
        scheduler,
        "오늘 코딩 공부하느라 힘들었어",
        "오늘 코딩 공부하느라 너무 힘들었어",
        0.08,
    )
    _assert_low_risk_pair(
        scheduler,
        "출근하기 싫다",
        "출근하기 진짜 싫다",
        0.08,
    )
    _assert_low_risk_pair(
        scheduler,
        "친구가 선물 줘서 감동이었어",
        "친구가 선물 줘서 진짜 감동이었어",
        0.08,
    )
    _assert_distress_not_flattened(
        scheduler,
        "너무 지쳐서 출근하기 싫다",
        min_depression_score=0.50,
        expected_top_emotion="슬픔",
    )
    _assert_distress_not_flattened(
        scheduler,
        "요즘 너무 힘들고 더는 못 버티겠어",
        min_depression_score=0.55,
    )
    print("[완료] 강조어 최소쌍 실모델 회귀 검증 통과")


if __name__ == "__main__":
    main()
