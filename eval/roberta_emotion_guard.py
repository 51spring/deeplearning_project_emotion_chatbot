"""
roberta_emotion_guard.py
역할: RoBERTa 감정 후처리 보정 규칙을 모델 로딩 없이 검증한다.
입력: 없음 (내장 케이스)
출력: 콘솔 검증 결과 및 실패 시 AssertionError
실행: C:/Users/WD/anaconda3/envs/dl_study/python.exe -u eval/roberta_emotion_guard.py
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from pipeline.roberta_score import apply_emotion_sanity_guard
from pipeline.roberta_score import apply_intensifier_delta_cap
from pipeline.roberta_score import apply_positive_affect_guard
from pipeline.roberta_score import apply_utterance_type_adjustment
from pipeline.roberta_score import attenuate_intensifiers
from pipeline.roberta_score import is_intensifier_delta_cap_candidate
from pipeline.ensemble import ensemble_scores
from pipeline.ewma import daily_to_smoothed, utterance_to_daily
from pipeline.score_policy import compute_wellness_contribution, score_affects_wellness
from pipeline.wellness_score import apply_no_signal_floor, compute_wellness, depression_to_wellness
from pipeline.depression_tendency import compute_depression_tendency
from backend.scheduler import (
    ROUTINE_DISCOMFORT_CBT_SCORE_CAP,
    _cap_routine_discomfort_cbt_score,
)
from backend.crisis_handler import should_hard_interrupt
from pipeline.utterance_type import (
    classify_utterance_type,
    is_administrative_technical_neutral_text,
    is_daily_routine_neutral_text,
    is_laughter_only_text,
    is_low_risk_sensory_disgust_text,
    is_physical_exertion_text,
    is_positive_affect_text,
    is_sensory_disgust_text,
    is_situational_anxiety_surprise_text,
    is_situational_anger_text,
    is_situational_sadness_text,
    normalize_emotion_analysis_text,
)


def assert_routine_avoidance_is_demoted() -> None:
    """
    역할: 일상 회피 표현이 공포·고점수로 유지되지 않는지 확인한다.
    입력: 없음
    출력: 없음
    """
    top_emotion, score, reason = apply_emotion_sanity_guard(
        "출근하기 싫다",
        "공포",
        1.0,
    )
    assert top_emotion == "중립"
    assert score == 0.45
    assert reason == "routine_discomfort_cap"
    print(f"[보정 확인] 출근하기 싫다 → {top_emotion}, score={score}, reason={reason}")


def assert_distress_avoidance_is_kept_negative() -> None:
    """
    역할: 피로·우울 맥락이 있는 회피 표현은 슬픔 쪽으로 보정되는지 확인한다.
    입력: 없음
    출력: 없음
    """
    top_emotion, score, reason = apply_emotion_sanity_guard(
        "너무 지쳐서 출근하기 싫다",
        "공포",
        1.0,
    )
    assert top_emotion == "슬픔"
    assert score == 0.70
    assert reason == "emotional_distress_false_fear_cap"
    print(f"[보정 확인] 너무 지쳐서 출근하기 싫다 → {top_emotion}, score={score}, reason={reason}")


def assert_real_fear_is_preserved() -> None:
    """
    역할: 실제 공포 단서가 있는 문장은 보정하지 않는지 확인한다.
    입력: 없음
    출력: 없음
    """
    top_emotion, score, reason = apply_emotion_sanity_guard(
        "출근길이 너무 무서워서 가기 싫다",
        "공포",
        0.91,
    )
    assert top_emotion == "공포"
    assert score == 0.91
    assert reason is None
    top_emotion, score, reason, utterance_info = apply_utterance_type_adjustment(
        "운동하다가 숨이 안 쉬어지고 쓰러질 것 같아",
        "중립",
        0.20,
        utterance_info={
            "utterance_type": "casual_share",
            "type_confidence": 0.60,
            "type_reason": "test_head_misfire",
        },
    )
    assert utterance_info["utterance_type"] == "emotional_distress"
    assert top_emotion == "공포"
    assert score == 0.72
    assert reason == "high_intensity_fear_override"
    print("[보존 확인] 실제 공포 단서 문장 유지")


def assert_practical_surprise_is_demoted() -> None:
    """
    역할: 불안 완화 질문 같은 실용 발화가 놀람 감정으로 노출되지 않는지 확인한다.
    입력: 없음
    출력: 없음
    """
    top_emotion, score, reason, _ = apply_utterance_type_adjustment(
        "불안감 해소?",
        "놀람",
        1.0,
        utterance_info={
            "utterance_type": "practical_question",
            "type_confidence": 0.90,
            "type_reason": "test",
        },
    )
    assert top_emotion == "중립"
    assert score == 0.35
    assert reason == "casual_neutral_surprise_cap"
    print(f"[보정 확인] 불안감 해소? → {top_emotion}, score={score}, reason={reason}")


def assert_academic_anxiety_question_stays_practical() -> None:
    """
    역할: 시험 긴장 완화 방법을 묻는 질문은 정서 호소 보정보다 실용 질문으로 유지한다.
    입력: 없음
    출력: 없음
    """
    top_emotion, score, reason, utterance_info = apply_utterance_type_adjustment(
        "시험 긴장 완화 방법?",
        "놀람",
        1.0,
        utterance_info={
            "utterance_type": "casual_share",
            "type_confidence": 0.32,
            "type_reason": "test_head_misfire",
        },
    )
    assert utterance_info["utterance_type"] == "practical_question"
    assert top_emotion == "중립"
    assert score == 0.35
    assert reason == "casual_neutral_surprise_cap"
    print(f"[보정 확인] 시험 긴장 완화 방법? → {utterance_info['utterance_type']}, {top_emotion}, score={score}")


def assert_mild_unease_is_not_crisis_or_fear() -> None:
    """
    역할: 막연한 이상한 기분 표현이 위기·공포로 과대 노출되지 않는지 확인한다.
    입력: 없음
    출력: 없음
    """
    top_emotion, score, reason, utterance_info = apply_utterance_type_adjustment(
        "이상한 기분이 들어",
        "공포",
        1.0,
        utterance_info={
            "utterance_type": "crisis_candidate",
            "type_confidence": 0.59,
            "type_reason": "test_head_misfire",
        },
    )
    assert utterance_info["utterance_type"] == "emotional_distress"
    assert top_emotion == "중립"
    assert score == 0.45
    assert reason == "mild_unease_cap"
    print(f"[보정 확인] 이상한 기분이 들어 → {utterance_info['utterance_type']}, {top_emotion}, score={score}")


def assert_floating_mind_is_mild_unease() -> None:
    """
    역할: 붕 뜬 느낌·생각 과다 표현이 약한 불편감으로 라우팅되는지 확인한다.
    입력: 없음
    출력: 없음
    """
    top_emotion, score, reason, utterance_info = apply_utterance_type_adjustment(
        "오늘은 뭔가 마음이 붕 떠 있는 느낌이었어.",
        "슬픔",
        1.0,
        utterance_info={
            "utterance_type": "casual_share",
            "type_confidence": 0.38,
            "type_reason": "test_head_misfire",
        },
    )
    assert utterance_info["utterance_type"] == "emotional_distress"
    assert top_emotion == "중립"
    assert score == 0.45
    assert reason == "mild_unease_cap"

    rumination_type = classify_utterance_type("맞아 생각이 너무 많은거같아")
    assert rumination_type["utterance_type"] == "emotional_distress", rumination_type
    print("[보정 확인] 붕 뜬 느낌/생각 과다 → 약한 불편감으로 라우팅")


def assert_mild_low_mood_is_not_disgust() -> None:
    """
    역할: 가벼운 기분 저조 표현이 혐오로 노출되지 않는지 확인한다.
    입력: 없음
    출력: 없음
    """
    top_emotion, score, reason, utterance_info = apply_utterance_type_adjustment(
        "기분이 별로야",
        "혐오",
        1.0,
        utterance_info={
            "utterance_type": "casual_share",
            "type_confidence": 0.60,
            "type_reason": "test_head_misfire",
        },
    )
    assert utterance_info["utterance_type"] == "emotional_distress"
    assert top_emotion == "슬픔"
    assert score == 0.55
    assert reason == "mild_low_mood_cap"
    print(f"[보정 확인] 기분이 별로야 → {utterance_info['utterance_type']}, {top_emotion}, score={score}")


def assert_situational_low_mood_is_moderated() -> None:
    """
    역할: 5/4 사례처럼 단일 상황성 속상함이 위험권 고점수로 과대 반영되지 않는지 확인한다.
    입력: 없음
    출력: 없음
    """
    low_mood_emotion, low_mood_score, low_mood_reason, low_mood_info = apply_utterance_type_adjustment(
        "오늘은 괜히 마음이 가라앉아",
        "슬픔",
        1.0,
        utterance_info={
            "utterance_type": "casual_share",
            "type_confidence": 0.40,
            "type_reason": "test_head_misfire",
        },
    )
    assert low_mood_info["utterance_type"] == "emotional_distress"
    assert low_mood_emotion == "슬픔"
    assert low_mood_score == 0.55
    assert low_mood_reason == "mild_low_mood_cap"

    task_emotion, task_score, task_reason, task_info = apply_utterance_type_adjustment(
        "오늘 과제 하는데 주제가 생각이 안나서 되게 어려웠어",
        "슬픔",
        1.0,
        utterance_info={
            "utterance_type": "casual_share",
            "type_confidence": 0.42,
            "type_reason": "test_head_misfire",
        },
    )
    assert task_info["utterance_type"] == "emotional_distress"
    assert task_emotion == "슬픔"
    assert task_score == 0.60
    assert task_reason == "limited_situational_distress_cap"

    study_emotion, study_score, study_reason, study_info = apply_utterance_type_adjustment(
        "오늘 코딩 공부하느라 힘들었어",
        "중립",
        0.70,
        utterance_info={
            "utterance_type": "casual_share",
            "type_confidence": 0.43,
            "type_reason": "test_head_misfire",
        },
    )
    assert study_info["utterance_type"] == "routine_discomfort"
    assert study_emotion == "중립"
    assert study_score == 0.45
    assert study_reason == "routine_discomfort_cap"
    assert score_affects_wellness(study_info["utterance_type"])

    exam_emotion, exam_score, exam_reason, exam_info = apply_utterance_type_adjustment(
        "시험을 못봐서 힘들어",
        "슬픔",
        1.0,
        utterance_info={
            "utterance_type": "casual_share",
            "type_confidence": 0.42,
            "type_reason": "test_head_misfire",
        },
    )
    assert exam_info["utterance_type"] == "emotional_distress"
    assert exam_emotion == "슬픔"
    assert exam_score == 0.60
    assert exam_reason == "limited_situational_distress_cap"

    exam_fail_emotion, exam_fail_score, exam_fail_reason, _ = apply_utterance_type_adjustment(
        "시험을 망친거 같아",
        "슬픔",
        1.0,
        utterance_info={
            "utterance_type": "casual_share",
            "type_confidence": 0.42,
            "type_reason": "test_head_misfire",
        },
    )
    assert exam_fail_emotion == "슬픔"
    assert exam_fail_score == 0.60
    assert exam_fail_reason == "limited_situational_distress_cap"

    disappointment_emotion, disappointment_score, disappointment_reason, _ = apply_utterance_type_adjustment(
        "기대했던 일이 안 돼서 좀 속상해",
        "슬픔",
        1.0,
        utterance_info={
            "utterance_type": "casual_share",
            "type_confidence": 0.44,
            "type_reason": "test_head_misfire",
        },
    )
    assert disappointment_emotion == "슬픔"
    assert disappointment_score == 0.60
    assert disappointment_reason == "limited_situational_distress_cap"
    print("[보정 확인] 5/4 상황성 저조감/속상함 → mild cap 적용")


def assert_positive_affect_does_not_raise_crisis_or_score() -> None:
    """
    역할: 긍정·회복 발화가 NLI 오탐이나 감정 점수로 위험권에 올라가지 않는지 확인한다.
    입력: 없음
    출력: 없음
    """
    score, entailment, is_crisis, reason = apply_positive_affect_guard(
        "그래서 맛있는거 먹었더니 행복해",
        "행복",
        0.82,
        0.83,
        True,
        {
            "utterance_type": "positive_share",
            "type_confidence": 0.93,
            "type_reason": "test_positive_head",
        },
    )
    assert score == 0.30
    assert entailment == 0.20
    assert is_crisis is False
    assert reason == "positive_affect_safety_cap"

    assert is_positive_affect_text("친구가 깜짝 선물 줬는데 진짜 감동이었어")
    assert is_positive_affect_text("친구가 웃긴 얘기를 해줘서 많이 웃었어")
    assert is_positive_affect_text("오랜만에 일이 잘 풀려서 마음이 가벼워졌어")
    assert is_positive_affect_text("작게 정한 목표를 지켜서 나 자신이 조금 대견했어")
    assert is_positive_affect_text("생각보다 여유가 생겨서 오늘은 편안하게 보냈어")
    assert is_positive_affect_text("오래 헤맨 버그를 해결해서 속이 시원했어")
    assert is_positive_affect_text("작은 화분에 새잎이 올라온 걸 보고 괜히 반가웠어")
    assert is_positive_affect_text("미뤄둔 신청서를 제출하고 나니 한결 홀가분했어")
    assert is_positive_affect_text("동생이 응원해줘서 다시 해볼 힘이 조금 생겼어")
    assert is_positive_affect_text("오늘 할 일을 하나씩 지워가니 나름 잘 버틴 느낌이었어")
    assert not is_positive_affect_text("기대했던 만큼 반응이 없어서 조금 서운했어")
    assert not is_positive_affect_text("기대했던 연락이 안 와서 조금 쓸쓸했어")
    assert not is_positive_affect_text("좋아하던 일이 예전만큼 즐겁지 않아서 아쉬웠어")
    assert not is_positive_affect_text("요즘은 좋아하던 일도 손이 안 가고 하루가 그냥 지나가")
    assert not is_positive_affect_text("하나도 웃기지 않아서 마음이 허전했어")

    negative_score, negative_entailment, negative_crisis, negative_reason = apply_positive_affect_guard(
        "요즘은 좋아하던 일도 손이 안 가고 하루가 그냥 지나가",
        "행복",
        0.82,
        0.10,
        False,
        {
            "utterance_type": "positive_share",
            "type_confidence": 0.93,
            "type_reason": "test_positive_head_misfire",
        },
    )
    assert negative_score == 0.82
    assert negative_entailment == 0.10
    assert negative_crisis is False
    assert negative_reason is None

    anhedonia = compute_depression_tendency(
        "요즘은 좋아하던 일도 손이 안 가고 하루가 그냥 지나가",
        top_emotion="슬픔",
        utterance_type="emotional_distress",
    )
    assert anhedonia["depression_tendency_score"] >= 0.40
    assert "anhedonia" in anhedonia["hit_categories"]
    print("[보정 확인] 긍정/회복 발화 → NLI 후보 및 고우울 점수 cap 적용")


def assert_positive_affect_preserves_happiness_label() -> None:
    """
    역할: 명시 긍정 발화가 저위험 cap 이후에도 표시 감정은 행복으로 남는지 확인한다.
    입력: 없음
    출력: 없음
    """
    for text in [
        "오랜만에 일이 잘 풀려서 마음이 가벼워졌어",
        "오래 헤맨 버그를 해결해서 속이 시원했어",
        "작은 화분에 새잎이 올라온 걸 보고 괜히 반가웠어",
        "미뤄둔 신청서를 제출하고 나니 한결 홀가분했어",
        "맛있는 저녁을 먹고 기분이 꽤 좋아졌어",
        "작게 정한 목표를 지켜서 나 자신이 조금 대견했어",
        "생각보다 여유가 생겨서 오늘은 편안하게 보냈어",
        "동생이 응원해줘서 다시 해볼 힘이 조금 생겼어",
        "오늘 할 일을 하나씩 지워가니 나름 잘 버틴 느낌이었어",
    ]:
        top_emotion, score, reason, utterance_info = apply_utterance_type_adjustment(
            text,
            "중립",
            0.82,
            utterance_info={
                "utterance_type": "casual_share",
                "type_confidence": 0.55,
                "type_reason": "test_head_misfire",
            },
        )
        assert utterance_info["utterance_type"] in {"casual_share", "casual_neutral"}
        assert top_emotion == "행복"
        assert score == 0.30
        assert reason == "positive_affect_emotion_preserve"

    assert not is_daily_routine_neutral_text("맛있는 저녁을 먹고 기분이 꽤 좋아졌어")
    print("[보정 확인] 명시 긍정 발화 → 행복 라벨 보존 + 저위험 cap")


def assert_laughter_only_text_is_positive_low_signal() -> None:
    """
    역할: `ㅋㅋ/ㅎㅎ` 단독 발화가 raw 모델 점수 때문에 wellness를 낮추지 않도록 확인한다.
    입력: 없음
    출력: 없음
    """
    for text in ["ㅋ", "ㅋㅋ", "ㅎㅎㅎ", "ㅋㅋㅋㅋ~"]:
        assert is_laughter_only_text(text), text
    assert not is_laughter_only_text("죽고 싶다 ㅋㅋ")
    assert not is_laughter_only_text("하나도 웃기지 않아서 마음이 허전했어 ㅋㅋ")

    utterance_info = classify_utterance_type("ㅋㅋ")
    assert utterance_info["utterance_type"] == "positive_share"
    assert utterance_info["type_reason"] == "laughter_only_positive_low_signal_marker"

    top_emotion, score, reason, adjusted_info = apply_utterance_type_adjustment(
        "ㅋㅋ",
        "슬픔",
        0.90,
        utterance_info={
            "utterance_type": "emotional_distress",
            "type_confidence": 0.80,
            "type_reason": "test_head_misfire",
        },
    )
    assert adjusted_info["utterance_type"] == "positive_share"
    assert top_emotion == "행복"
    assert score == 0.30
    assert reason == "laughter_only_positive_cap"

    adjusted_score, adjusted_entailment, adjusted_crisis, nli_reason = apply_positive_affect_guard(
        "ㅋㅋ",
        top_emotion,
        0.90,
        0.90,
        True,
        adjusted_info,
    )
    assert adjusted_score == 0.30
    assert adjusted_entailment == 0.20
    assert not adjusted_crisis
    assert nli_reason == "laughter_only_safety_cap"

    contribution = compute_wellness_contribution({
        "utterance_type": "positive_share",
        "depression_score": 0.90,
        "top_emotion": "행복",
    })
    assert contribution["score_affects_wellness"]
    assert contribution["wellness_contribution_score"] == 0.30
    assert compute_wellness(contribution["wellness_contribution_score"], [], 1)["wellness_score"] == 70.0
    print("[보정 확인] 웃음 단독 발화 → 긍정 저신호로 처리해 70점 아래로 내리지 않음")


def assert_low_signal_chat_does_not_raise_nli_candidate() -> None:
    """
    역할: 실용·일상 질문이 NLI 오탐만으로 위기 후보가 되지 않는지 확인한다.
    입력: 없음
    출력: 없음
    """
    score, entailment, is_crisis, reason = apply_positive_affect_guard(
        "AI/ 백엔드 / 네트워크/ DB / 보안 너는 뭐가 제일 나은거같아?",
        "중립",
        0.42,
        0.74,
        True,
        {
            "utterance_type": "casual_share",
            "type_confidence": 0.80,
            "type_reason": "test_low_signal_chat",
        },
    )
    assert score == 0.42
    assert entailment == 0.20
    assert is_crisis is False
    assert reason == "low_signal_chat_nli_cap"

    admin_score, admin_entailment, admin_is_crisis, admin_reason = apply_positive_affect_guard(
        "택배 조회 번호를 확인해서 배송 상태를 기록했어",
        "중립",
        0.45,
        0.96,
        True,
        {
            "utterance_type": "routine_discomfort",
            "type_confidence": 0.56,
            "type_reason": "test_head_misfire",
        },
    )
    assert admin_score == 0.45
    assert admin_entailment == 0.20
    assert admin_is_crisis is False
    assert admin_reason == "low_signal_chat_nli_cap"
    print("[보정 확인] 저신호 일상/실용 발화 → NLI 후보 cap 적용")


def assert_contextual_nli_caps_do_not_raise_crisis() -> None:
    """
    역할: 직접 위기 문구가 없는 상황성 발화가 NLI 오탐만으로 위기 후보가 되지 않는지 확인한다.
    입력: 없음
    출력: 없음
    """
    cases = [
        (
            "철봉 매달리기 했는데 버티기 힘들더라",
            "중립",
            0.38,
            "casual_neutral",
            "physical_exertion_nli_cap",
        ),
        (
            "불안감 해소방법 알려줘",
            "중립",
            0.35,
            "practical_question",
            "practical_question_nli_cap",
        ),
        (
            "오늘 누가 무심하게 말해서 괜히 상처받았어.",
            "슬픔",
            0.60,
            "emotional_distress",
            "situational_sadness_nli_cap",
        ),
        (
            "시험을 못봐서 힘들어",
            "슬픔",
            0.60,
            "emotional_distress",
            "limited_situational_distress_nli_cap",
        ),
        (
            "아직 일어나지도 않은 일인데 자꾸 걱정돼서 마음이 불안했어.",
            "공포",
            0.55,
            "emotional_distress",
            "situational_anxiety_surprise_nli_cap",
        ),
        (
            "냉장고 구석 반찬통에서 시큼한 냄새가 올라와서 바로 닫았어",
            "혐오",
            0.50,
            "casual_neutral",
            "sensory_disgust_nli_cap",
        ),
    ]
    for text, emotion, score, utterance_type, expected_reason in cases:
        adjusted_score, entailment, is_crisis, reason = apply_positive_affect_guard(
            text,
            emotion,
            score,
            0.96,
            True,
            {
                "utterance_type": utterance_type,
                "type_confidence": 0.90,
                "type_reason": "test_contextual_nli_cap",
            },
        )
        assert adjusted_score == score
        assert entailment == 0.20
        assert is_crisis is False
        assert reason == expected_reason
    print("[NLI 확인] 신체 활동/실용 질문/상황성 슬픔·불안 → NLI 후보 cap 적용")


def assert_administrative_technical_text_is_low_signal() -> None:
    """
    역할: 번호·버전·문서 처리 발화가 NLI/CBT 후보성 저신호 문맥으로 라우팅되는지 확인한다.
    입력: 없음
    출력: 없음
    """
    for text in [
        "택배 조회 번호를 확인해서 배송 상태를 기록했어",
        "버전 번호를 맞추고 변경 내역을 표로 옮겼어",
        "문서 제목 형식을 맞추고 오래된 파일을 분류했어",
        "노트북 충전 케이블을 서랍에 넣어뒀어",
    ]:
        assert is_administrative_technical_neutral_text(text), text
        top_emotion, score, reason, utterance_info = apply_utterance_type_adjustment(
            text,
            "공포",
            0.98,
            utterance_info={
                "utterance_type": "routine_discomfort",
                "type_confidence": 0.56,
                "type_reason": "test_head_misfire",
            },
        )
        assert utterance_info["utterance_type"] == "casual_neutral"
        assert top_emotion == "중립"
        assert score == 0.35
        assert reason == "administrative_technical_neutral_cap"

    assert not is_administrative_technical_neutral_text("파일이 사라진 줄 알고 순간 가슴이 내려앉았어")
    late_night_cleaned = normalize_emotion_analysis_text(
        "밤에 다시 떠올려보니 노트북 충전 케이블을 서랍에 넣어뒀어."
    )
    assert late_night_cleaned == "노트북 충전 케이블을 서랍에 넣어뒀어"
    assert is_administrative_technical_neutral_text(late_night_cleaned)
    print("[보정 확인] 행정/기술 처리 발화 → 저신호 중립 cap 적용")


def assert_physical_exertion_is_not_crisis_or_fear() -> None:
    """
    역할: 운동·근무·생활 신체 활동의 힘듦 표현이 위기/공포 고점수로 흐르지 않는지 확인한다.
    입력: 없음
    출력: 없음
    """
    cases = [
        "철봉 매달리기 했는데 버티기 힘들더라",
        "오늘 가게에서 하루종일 일하다 왔더니 온몸이 쑤셔",
        "알바 끝나고 집에 오니까 다리가 너무 뻐근해",
        "서서 일했더니 발바닥이 아파",
        "이사 짐 옮기느라 허리가 아파",
        "창고에서 박스 나르다 어깨가 결려",
        "가게에서 일하고 피곤해",
        "청소하다 힘들었어",
        "마트에서 상품 진열하느라 팔이 너무 뻐근해",
        "퇴근하고 오니까 허리가 너무 아파",
        "계단을 계속 오르내렸더니 종아리가 땡겨",
        "편의점 카운터에 오래 서 있었더니 무릎이 아파",
        "택배 상하차하고 손목이 저려",
        "오늘 학교에서 바닥 닦았더니 몸이 힘드네",
        "오늘 학교에서 바닥 청소를 했더니 고되",
    ]
    for text in cases:
        assert is_physical_exertion_text(text), text
        top_emotion, score, reason, utterance_info = apply_utterance_type_adjustment(
            text,
            "공포",
            1.0,
            utterance_info={
                "utterance_type": "crisis_candidate",
                "type_confidence": 0.62,
                "type_reason": "test_head_misfire",
            },
        )
        assert utterance_info["utterance_type"] == "casual_neutral"
        assert top_emotion == "중립"
        assert score == 0.35
        assert reason == "physical_exertion_cap"

    assert not is_physical_exertion_text("요즘 계속 몸이 무겁고 의욕이 없어")
    assert not is_physical_exertion_text("회사에서 하루종일 무시당해서 마음이 너무 아파")
    assert not is_physical_exertion_text("일하면서 마음이 무거워")
    assert not is_physical_exertion_text("운동하다가 숨이 안 쉬어지고 쓰러질 것 같아")
    print("[보정 확인] 운동·근무·생활 신체 피로 표현 → 일상 활동으로 cap 적용")


def assert_daily_routine_is_low_signal() -> None:
    """
    역할: 음식·휴식 루틴 발화가 정서 위험 발화로 과대 라우팅되지 않는지 확인한다.
    입력: 없음
    출력: 없음
    """
    assert is_daily_routine_neutral_text("저녁에는 간단히 라면 끓여 먹었어")
    top_emotion, score, reason, utterance_info = apply_utterance_type_adjustment(
        "저녁에는 간단히 라면 끓여 먹었어",
        "슬픔",
        1.0,
        utterance_info={
            "utterance_type": "emotional_distress",
            "type_confidence": 0.58,
            "type_reason": "test_head_misfire",
        },
    )
    assert utterance_info["utterance_type"] == "casual_neutral"
    assert top_emotion == "중립"
    assert score == 0.35
    assert reason == "daily_routine_neutral_cap"
    assert not is_daily_routine_neutral_text("잠을 자도 피곤하고 마음이 무거워")
    print("[보정 확인] 음식·휴식 루틴 발화 → 저신호 일상으로 cap 적용")


def assert_sensory_disgust_preserves_disgust_label() -> None:
    """
    역할: 감각/청결/사회 혐오 표현이 중립으로 눌리지 않고 혐오 라벨로 보존되는지 확인한다.
    입력: 없음
    출력: 없음
    """
    low_risk_cases = [
        "상한 냄새가 올라와서 속이 울렁거렸어",
        "책상 위 끈적한 자국을 보고 불쾌했어",
        "음식에서 이상한 식감이 느껴져서 비위가 상했어",
        "손에 묻은 기름기가 계속 남아서 찝찝했어",
        "하수구 냄새가 올라와서 얼굴을 찡그렸어",
        "젖은 행주를 만졌는데 축축한 촉감이 너무 싫었어",
        "상한 우유 냄새를 맡고 바로 컵을 내려놨어",
        "냉장고 구석 반찬통에서 시큼한 냄새가 올라와서 바로 닫았어",
        "컵 바닥에 남은 찌꺼기 때문에 마실 생각이 사라졌어",
    ]
    social_cases = [
        "남의 약점을 재미로 말하는 걸 듣고 정이 떨어졌어",
        "누군가를 은근히 따돌리는 분위기가 정말 불편했어",
        "거짓말을 웃으면서 넘기는 태도가 역하게 느껴졌어",
        "남을 깎아내리며 웃는 말을 듣고 마음이 확 식었어",
    ]
    for text in low_risk_cases + social_cases:
        assert is_sensory_disgust_text(text), text
        top_emotion, score, reason, utterance_info = apply_utterance_type_adjustment(
            text,
            "중립",
            0.96,
            utterance_info={
                "utterance_type": "routine_discomfort",
                "type_confidence": 0.58,
                "type_reason": "test_head_misfire",
            },
        )
        if text in low_risk_cases:
            assert is_low_risk_sensory_disgust_text(text), text
            assert utterance_info["utterance_type"] == "casual_neutral"
        else:
            assert not is_low_risk_sensory_disgust_text(text), text
            assert utterance_info["utterance_type"] == "emotional_distress"
        assert top_emotion == "혐오"
        assert score == 0.50
        assert reason == "sensory_disgust_cap"

    assert not is_sensory_disgust_text("무례한 농담을 계속해서 화가 났어")
    print("[보정 확인] 감각/사회 혐오 발화 → 혐오 라벨 보존 + 중등도 cap")


def assert_disappointment_words_route_to_sadness() -> None:
    """
    역할: 서운함·쓸쓸함·허전함 계열 후속 발화가 중립으로 남지 않는지 확인한다.
    입력: 없음
    출력: 없음
    """
    for text in [
        "사실 그 서운함이 아직 조금 남아 있어.",
        "기대했던 연락이 안 와서 조금 쓸쓸했어.",
        "하루가 끝났는데 마음이 허전했어.",
        "작은 말 한마디에 마음이 좀 상했어.",
        "좋아하던 일이 예전만큼 즐겁지 않아서 아쉬웠어.",
    ]:
        top_emotion, score, reason, utterance_info = apply_utterance_type_adjustment(
            text,
            "중립",
            0.30,
            utterance_info={
                "utterance_type": "casual_share",
                "type_confidence": 0.55,
                "type_reason": "test_head_misfire",
            },
        )
        assert utterance_info["utterance_type"] == "emotional_distress"
        assert top_emotion == "슬픔"
        assert score == 0.55
        assert reason in {"distress_marker_emotion_override", "situational_sadness_cap"}
    print("[보정 확인] 서운함/쓸쓸함/허전함 계열 → 슬픔 라우팅")


def assert_situational_sadness_is_moderated() -> None:
    """
    역할: 단일 사건성 슬픔이 슬픔으로 보존되되 위험권 고점수로 고정되지 않는지 확인한다.
    입력: 없음
    출력: 없음
    """
    for text in [
        "오늘 친구랑 약속이 취소돼서 좀 서운했어.",
        "기대했던 연락이 안 와서 조금 쓸쓸했어.",
        "예전 생각이 나서 눈물이 날 것 같았어.",
        "오늘은 혼자 있는 시간이 유난히 길게 느껴졌어.",
        "좋아하던 일이 예전만큼 즐겁지 않아서 아쉬웠어.",
        "작은 말 한마디에 마음이 좀 상했어.",
        "오늘 누가 무심하게 말해서 괜히 상처받았어.",
        "오늘 사람들 사이에 있었는데도 이상하게 좀 외로웠어.",
        "집에 오는데 괜히 외로운 느낌이 들었어.",
        "하루가 끝났는데 마음이 허전했어.",
    ]:
        assert is_situational_sadness_text(text), text
        top_emotion, score, reason, utterance_info = apply_utterance_type_adjustment(
            text,
            "슬픔",
            0.96,
            utterance_info={
                "utterance_type": "casual_share",
                "type_confidence": 0.50,
                "type_reason": "test_head_misfire",
            },
        )
        assert utterance_info["utterance_type"] == "emotional_distress"
        assert top_emotion == "슬픔"
        assert score == 0.60
        assert reason == "situational_sadness_cap"

    assert not is_situational_sadness_text("요즘 계속 우울해")
    assert not is_situational_sadness_text("잠을 자도 피곤하고 마음이 무거워")
    assert not is_situational_sadness_text("아무것도 하기 싫고 의욕이 없어")
    print("[보정 확인] 단일 사건성 슬픔 → 주의권 cap, 지속 우울 → 보존")


def assert_academic_anxiety_is_moderated() -> None:
    """
    역할: 시험 전 긴장 표현이 일상 중립이나 위험 고점수로 흐르지 않는지 확인한다.
    입력: 없음
    출력: 없음
    """
    top_emotion, score, reason, utterance_info = apply_utterance_type_adjustment(
        "내일 시험이 있어서 긴장돼",
        "중립",
        1.0,
        utterance_info={
            "utterance_type": "casual_share",
            "type_confidence": 0.32,
            "type_reason": "test_head_misfire",
        },
    )
    assert utterance_info["utterance_type"] == "emotional_distress"
    assert top_emotion == "공포"
    assert score == 0.55
    assert reason == "academic_anxiety_cap"
    print(f"[보정 확인] 시험 전 긴장 → {utterance_info['utterance_type']}, {top_emotion}, score={score}")


def assert_situational_anxiety_surprise_is_moderated() -> None:
    """
    역할: holdout 공포·놀람 문맥이 우울 경향이 아닌 중등도 상황 반응으로 제한되는지 확인한다.
    입력: 없음
    출력: 없음
    """
    cases = [
        ("심사 결과를 기다리는데 혹시 떨어질까 봐 손이 차가워져", "공포"),
        ("갑자기 일정이 앞당겨졌다는 말을 듣고 당황했어", "놀람"),
        ("파일이 사라진 줄 알고 순간 가슴이 내려앉았어", "놀람"),
        ("예상보다 회의가 빨리 끝나서 어리둥절했어", "놀람"),
        ("예정에 없던 칭찬을 듣고 바로 말이 안 나왔어", "놀람"),
        ("아직 일어나지도 않은 일인데 자꾸 걱정돼서 마음이 불안했어.", "공포"),
    ]
    for text, expected_emotion in cases:
        assert is_situational_anxiety_surprise_text(text), text
        top_emotion, score, reason, utterance_info = apply_utterance_type_adjustment(
            text,
            expected_emotion,
            0.99,
            utterance_info={
                "utterance_type": "casual_share",
                "type_confidence": 0.42,
                "type_reason": "test_head_misfire",
            },
        )
        assert utterance_info["utterance_type"] == "emotional_distress"
        assert top_emotion == expected_emotion
        assert score == 0.55
        assert reason == "situational_anxiety_surprise_cap"

    assert not is_situational_anxiety_surprise_text("공황이 와서 숨이 안 쉬어지고 통제가 안 돼")
    print("[보정 확인] 경도 상황 불안/놀람 → 중등도 cap, 고강도 공포 → 보존")


def assert_assistant_quote_context_is_removed() -> None:
    """
    역할: Qwen 응답 인용형 후속 발화에서 실제 사용자 감정 절이 남는지 확인한다.
    입력: 없음
    출력: 없음
    """
    cleaned = normalize_emotion_analysis_text(
        "네가 '지금 마음이 꽤 버거웠던 것 같아요.'라고 말해주니까 조금 말할 수 있을 것 같아. "
        "그래도 아직 몸이 무겁고 의욕이 없어."
    )
    assert "버거웠던 것 같아요" not in cleaned
    assert "몸이 무겁고 의욕이 없어" in cleaned
    quote_emotion, quote_score, quote_reason, quote_info = apply_utterance_type_adjustment(
        cleaned,
        "중립",
        0.30,
        utterance_info={
            "utterance_type": "casual_share",
            "type_confidence": 0.55,
            "type_reason": "test_quote_head_misfire",
        },
    )
    assert quote_info["utterance_type"] == "emotional_distress"
    assert quote_emotion == "슬픔"
    assert quote_score == 0.55
    assert quote_reason == "distress_marker_emotion_override"

    long_context_cleaned = normalize_emotion_analysis_text(
        "아침부터 자잘한 일정이 이어졌고 중간에 신경 쓸 일도 조금 있었어. "
        "그 흐름 속에서 친구가 작은 선물을 챙겨줘서 하루가 따뜻하게 느껴졌어. "
        "그래서 오늘 감정을 기록해두고 싶었어."
    )
    assert long_context_cleaned == "친구가 작은 선물을 챙겨줘서 하루가 따뜻하게 느껴졌어"

    memo_cleaned = normalize_emotion_analysis_text(
        "[오늘 기록] 감정=혐오; 내용=상한 냄새가 올라와서 속이 울렁거렸어; 메모=나중에 다시 확인"
    )
    assert memo_cleaned == "상한 냄새가 올라와서 속이 울렁거렸어"

    calendar_cleaned = normalize_emotion_analysis_text(
        '캘린더 여백에 쓰면 "먹는 것도 귀찮고 밤에는 잠이 얕아서 계속 지쳐" 정도야.'
    )
    assert calendar_cleaned == "먹는 것도 귀찮고 밤에는 잠이 얕아서 계속 지쳐"

    weather_cleaned = normalize_emotion_analysis_text(
        "창밖을 보다가 갑자기 선명해졌는데, 버스에서 자리를 양보받아 하루가 부드럽게 시작됐어."
    )
    assert weather_cleaned == "버스에서 자리를 양보받아 하루가 부드럽게 시작됐어"

    assert normalize_emotion_analysis_text(
        "오늘 상태를 말씀드리면, 동생이 응원해줘서 다시 해볼 힘이 조금 생겼어요."
    ) == "동생이 응원해줘서 다시 해볼 힘이 조금 생겼어요"
    assert normalize_emotion_analysis_text(
        "아침에 눈을 떠도 몸이 무겁고 시작할 힘이 안 나. 그냥 짧게 남겨두면 이래."
    ) == "아침에 눈을 떠도 몸이 무겁고 시작할 힘이 안 나"
    assert normalize_emotion_analysis_text(
        "몸으로 먼저 느껴진 건 어깨에 힘이 들어가는 거였고, 이어서 버스 시간표를 확인하고 출발 시간을 계산했어."
    ) == "버스 시간표를 확인하고 출발 시간을 계산했어"
    assert normalize_emotion_analysis_text(
        "겉으로는 별일 아닌 척했지만, 안쪽에서는 남의 약점을 재미로 말하는 걸 듣고 정이 떨어졌어."
    ) == "남의 약점을 재미로 말하는 걸 듣고 정이 떨어졌어"
    assert normalize_emotion_analysis_text(
        "그때는 그냥 넘겼는데 시간이 지나고 보니 미뤄둔 신청서를 제출하고 나니 한결 홀가분했어."
    ) == "미뤄둔 신청서를 제출하고 나니 한결 홀가분했어"
    assert normalize_emotion_analysis_text(
        "상대가 약속 시간을 가볍게 넘겨서 기분이상했어... ㅠ"
    ) == "상대가 약속 시간을 가볍게 넘겨서 기분이상했어"
    print(f"[전처리 확인] 인용형 후속 발화 → {cleaned}")


def assert_anger_marker_routes_to_anger() -> None:
    """
    역할: 화남·억울함 표현이 중립/슬픔으로 남지 않고 분노로 라우팅되는지 확인한다.
    입력: 없음
    출력: 없음
    """
    top_emotion, score, reason, utterance_info = apply_utterance_type_adjustment(
        "상대가 말을 너무 무례하게 해서 화가 났어.",
        "중립",
        0.42,
        utterance_info={
            "utterance_type": "casual_share",
            "type_confidence": 0.50,
            "type_reason": "test_head_misfire",
        },
    )
    assert utterance_info["utterance_type"] == "emotional_distress"
    assert top_emotion == "분노"
    assert score == 0.60
    assert reason == "situational_anger_cap"

    anger_type = classify_utterance_type("억울한 말을 들어서 계속 분이 안 풀려.")
    assert anger_type["utterance_type"] == "emotional_distress", anger_type
    print("[보정 확인] 화남/억울함 발화 → 분노 라우팅")


def assert_situational_anger_is_moderated() -> None:
    """
    역할: 일회성 짜증·무례·억울함이 분노로 보존되되 위험권 고점수로 고정되지 않는지 확인한다.
    입력: 없음
    출력: 없음
    """
    for text in [
        "오늘 팀원이 약속을 또 안 지켜서 화가 났어.",
        "괜히 나한테만 일이 몰리는 것 같아서 짜증났어.",
        "내 의견을 계속 무시해서 기분이 나빴어.",
        "기다리게 해놓고 사과도 안 해서 짜증났어.",
        "상대가 약속 시간을 가볍게 넘겨서 기분이 상했어.",
        "문의 답변을 계속 미뤄서 계획이 다 꼬였어.",
    ]:
        assert is_situational_anger_text(text)
        top_emotion, score, reason, utterance_info = apply_utterance_type_adjustment(
            text,
            "분노",
            0.95,
            utterance_info={
                "utterance_type": "casual_share",
                "type_confidence": 0.50,
                "type_reason": "test_head_misfire",
            },
        )
        assert utterance_info["utterance_type"] == "emotional_distress"
        assert top_emotion == "분노"
        assert score == 0.60
        assert reason == "situational_anger_cap"

    top_emotion, score, reason, _ = apply_utterance_type_adjustment(
        "너무 화가 나서 참을 수 없고 통제가 안 돼.",
        "분노",
        0.50,
        utterance_info={
            "utterance_type": "casual_share",
            "type_confidence": 0.50,
            "type_reason": "test_head_misfire",
        },
    )
    assert top_emotion == "분노"
    assert score == 0.72
    assert reason == "high_intensity_anger_override"
    print("[보정 확인] 상황성 분노 → 주의권 cap, 강한 분노 → 고강도 보존")


def assert_utterance_types() -> None:
    """
    역할: 대표 발화가 주요 발화 타입으로 올바르게 라우팅되는지 확인한다.
    입력: 없음
    출력: 없음
    """
    cases = [
        ("밥 먹었어", "casual_neutral"),
        ("저녁에는 간단히 라면 끓여 먹었어", "casual_neutral"),
        ("내일이 기대돼", "casual_neutral"),
        ("철봉 매달리기 했는데 버티기 힘들더라", "casual_neutral"),
        ("오늘 가게에서 하루종일 일하다 왔더니 온몸이 쑤셔", "casual_neutral"),
        ("서서 일했더니 발바닥이 아파", "casual_neutral"),
        ("가게에서 일하고 피곤해", "casual_neutral"),
        ("청소하다 힘들었어", "casual_neutral"),
        ("마트에서 상품 진열하느라 팔이 너무 뻐근해", "casual_neutral"),
        ("퇴근하고 오니까 허리가 너무 아파", "casual_neutral"),
        ("계단을 계속 오르내렸더니 종아리가 땡겨", "casual_neutral"),
        ("편의점 카운터에 오래 서 있었더니 무릎이 아파", "casual_neutral"),
        ("택배 상하차하고 손목이 저려", "casual_neutral"),
        ("오늘 학교에서 바닥 닦았더니 몸이 힘드네", "casual_neutral"),
        ("오늘 학교에서 바닥 청소를 했더니 고되", "casual_neutral"),
        ("택배 조회 번호를 확인해서 배송 상태를 기록했어", "casual_neutral"),
        ("친구가 깜짝 선물 줬는데 진짜 감동이었어", "casual_neutral"),
        ("상한 냄새가 올라와서 속이 울렁거렸어", "casual_neutral"),
        ("책상 위 끈적한 자국을 보고 불쾌했어", "casual_neutral"),
        ("냉장고 구석 반찬통에서 시큼한 냄새가 올라와서 바로 닫았어", "casual_neutral"),
        ("컵 바닥에 남은 찌꺼기 때문에 마실 생각이 사라졌어", "casual_neutral"),
        ("출근하기 싫다", "routine_discomfort"),
        ("오늘 코딩 공부하느라 힘들었어", "routine_discomfort"),
        ("불안감 해소?", "practical_question"),
        ("시험 긴장 완화 방법?", "practical_question"),
        ("내일 시험이 있어서 긴장돼", "emotional_distress"),
        ("심사 결과를 기다리는데 혹시 떨어질까 봐 손이 차가워져", "emotional_distress"),
        ("갑자기 일정이 앞당겨졌다는 말을 듣고 당황했어", "emotional_distress"),
        ("이상한 기분이 들어", "emotional_distress"),
        ("오늘은 뭔가 마음이 붕 떠 있는 느낌이었어.", "emotional_distress"),
        ("맞아 생각이 너무 많은거같아", "emotional_distress"),
        ("기분이 별로야", "emotional_distress"),
        ("일하면서 마음이 무거워", "emotional_distress"),
        ("오늘은 괜히 마음이 가라앉아", "emotional_distress"),
        ("기대했던 일이 안 돼서 좀 속상해", "emotional_distress"),
        ("사실 그 서운함이 아직 조금 남아 있어.", "emotional_distress"),
        ("하루가 끝났는데 마음이 허전했어.", "emotional_distress"),
        ("상대가 말을 너무 무례하게 해서 화가 났어.", "emotional_distress"),
        ("억울한 말을 들어서 계속 분이 안 풀려.", "emotional_distress"),
        ("요즘 계속 우울해", "emotional_distress"),
        ("죽고 싶다", "crisis_candidate"),
    ]
    for text, expected_type in cases:
        result = classify_utterance_type(text)
        assert result["utterance_type"] == expected_type, result
        print(f"[타입 확인] {text} → {result['utterance_type']}")


def assert_score_policy() -> None:
    """
    역할: 발화 타입별 웰니스 반영 정책이 full/low/none으로 분리되는지 확인한다.
    입력: 없음
    출력: 없음
    """
    assert score_affects_wellness("emotional_distress")
    assert score_affects_wellness("routine_discomfort")
    assert score_affects_wellness("crisis_candidate")
    assert score_affects_wellness("casual_neutral")
    assert score_affects_wellness("casual_share")
    assert score_affects_wellness("positive_share")
    assert not score_affects_wellness("practical_question")
    assert score_affects_wellness("casual_share", is_crisis=True)

    neutral = compute_wellness_contribution({
        "utterance_type": "casual_neutral",
        "depression_score": 0.80,
        "top_emotion": "중립",
    })
    assert neutral["score_affects_wellness"]
    assert neutral["wellness_impact_type"] == "low"
    assert neutral["wellness_contribution_score"] == 0.38

    positive = compute_wellness_contribution({
        "utterance_type": "positive_share",
        "depression_score": 0.45,
        "top_emotion": "행복",
    })
    assert positive["score_affects_wellness"]
    assert positive["wellness_impact_type"] == "low"
    assert positive["wellness_contribution_score"] == 0.30

    sensory = compute_wellness_contribution({
        "utterance_type": "casual_neutral",
        "utterance_type_reason": "sensory_disgust_low_impact_override",
        "depression_score": 0.99,
        "top_emotion": "혐오",
        "emotion_guard": "sensory_disgust_cap",
    })
    assert sensory["score_affects_wellness"]
    assert sensory["wellness_impact_type"] == "low"
    assert round(sensory["wellness_contribution_score"], 2) == 0.45

    practical = compute_wellness_contribution({
        "utterance_type": "practical_question",
        "depression_score": 0.70,
        "top_emotion": "중립",
    })
    assert not practical["score_affects_wellness"]
    assert practical["wellness_contribution_score"] is None

    full = compute_wellness_contribution({
        "utterance_type": "routine_discomfort",
        "depression_score": 0.4503,
        "top_emotion": "중립",
    })
    assert full["score_affects_wellness"]
    assert full["wellness_impact_type"] == "full"
    assert full["wellness_contribution_score"] == 0.4503
    print("[정책 확인] 중립/긍정/감각 혐오 저신호 발화는 좁은 연속 범위로 반영")


def assert_legacy_score_pipeline_uses_score_policy() -> None:
    """
    역할: legacy ScorePipeline도 raw depression_score 대신 score_policy 기여 점수를 버퍼에 넣는지 확인한다.
    입력: 없음
    출력: 없음
    """
    import pipeline.score_pipeline as score_pipeline_module
    from pipeline.score_pipeline import ScorePipeline

    original_infer_single = score_pipeline_module.infer_single

    def fake_infer_single(text, *args, **kwargs):
        """
        역할: 모델 로딩 없이 ScorePipeline 경로의 정책 반영만 검증하는 가짜 RoBERTa 출력 생성
        입력: 발화 텍스트와 무시되는 모델 인자
        출력: infer_single 호환 dict
        """
        utterance_type = "practical_question" if "방법" in text else "casual_neutral"
        return {
            "roberta_score": 0.80,
            "is_crisis": False,
            "top_emotion": "중립",
            "entailment_prob": 0.0,
            "emotion_probs": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "utterance_type": utterance_type,
            "utterance_type_reason": "test_stub",
            "analysis_text": text,
        }

    try:
        score_pipeline_module.infer_single = fake_infer_single
        pipeline = ScorePipeline.__new__(ScorePipeline)
        pipeline.model = None
        pipeline.tokenizer = None
        pipeline.device = None
        pipeline.anchor_embs = {}
        pipeline.p95 = 1.0
        pipeline.use_cbt = False
        pipeline.T_emotion = 1.0
        pipeline.T_nli = 1.0
        pipeline.vector_T_emotion = None
        pipeline._today_utterances = []
        pipeline._today_tendency = []

        low_impact = pipeline.score_utterance("그냥 이런저런 얘기야")
        assert low_impact["depression_score"] == 0.80
        assert low_impact["wellness_contribution_score"] == 0.38
        assert low_impact["score_policy"] == "low_neutral_clamped_plus_minus_8"
        assert pipeline._today_utterances == [0.38]
        assert pipeline._today_tendency == [low_impact["depression_tendency_score"]]

        practical = pipeline.score_utterance("불안감 해소방법 알려줘")
        assert practical["depression_score"] == 0.80
        assert not practical["score_affects_wellness"]
        assert practical["wellness_contribution_score"] is None
        assert pipeline._today_utterances == [0.38]
        assert pipeline._today_tendency == [low_impact["depression_tendency_score"]]
    finally:
        score_pipeline_module.infer_single = original_infer_single

    print("[정책 확인] legacy ScorePipeline도 score_policy 기여 점수만 EWMA에 반영")


def assert_routine_discomfort_cbt_burst_is_capped() -> None:
    """
    역할: 일상 과업 피로에서 CBT anchor 단독 고점이 위험권 점수를 만들지 않는지 확인한다.
    입력: 없음
    출력: 없음
    """
    capped_cbt, capped = _cap_routine_discomfort_cbt_score(
        "오늘 코딩 공부하느라 너무 힘들었어",
        "routine_discomfort",
        0.8867,
    )
    assert capped
    assert capped_cbt == ROUTINE_DISCOMFORT_CBT_SCORE_CAP
    ensemble = ensemble_scores(
        0.45,
        capped_cbt,
        distress_severity=0.49,
    )
    assert ensemble["depression_score"] == 0.45

    uncapped_cbt, uncapped = _cap_routine_discomfort_cbt_score(
        "요즘 계속 우울해",
        "emotional_distress",
        0.8867,
    )
    assert not uncapped
    assert uncapped_cbt == 0.8867
    print("[보정 확인] routine_discomfort CBT 단독 과상승 cap 정상")


def assert_task_overload_is_routine_discomfort() -> None:
    """
    역할: 할 일 과부하 표현이 위기/지속 우울이 아니라 일상 과업 불편으로 완화되는지 확인한다.
    입력: 없음
    출력: 없음
    """
    text = "오늘 아침에 일어나서 할일이 너무 많아서 힘들었어"
    routed = classify_utterance_type(text)
    assert routed["utterance_type"] == "routine_discomfort", routed

    top_emotion, score, reason, info = apply_utterance_type_adjustment(
        text,
        "슬픔",
        1.0,
        utterance_info={
            "utterance_type": "emotional_distress",
            "type_confidence": 0.97,
            "type_reason": "test_head_misfire",
        },
    )
    assert info["utterance_type"] == "routine_discomfort"
    assert top_emotion == "중립"
    assert score == 0.45
    assert reason == "routine_discomfort_cap"

    adjusted_score, adjusted_entailment, adjusted_crisis, nli_reason = apply_positive_affect_guard(
        text,
        top_emotion,
        score,
        entailment_prob=0.4646,
        is_crisis=True,
        utterance_info=info,
    )
    assert adjusted_score == 0.45
    assert adjusted_entailment == 0.20
    assert not adjusted_crisis
    assert nli_reason == "routine_discomfort_nli_cap"
    print("[보정 확인] 할 일 과부하 표현 → routine_discomfort + NLI cap")


def assert_intensifier_delta_cap_is_low_risk_only() -> None:
    """
    역할: 강조어 한 단어가 저위험 발화 점수를 크게 튀기지 않도록 delta cap이 작동하는지 확인한다.
    입력: 없음
    출력: 없음
    """
    text = "오늘 코딩 공부하느라 너무 힘들었어"
    attenuated = attenuate_intensifiers(text)
    assert attenuated == "오늘 코딩 공부하느라 힘들었어"
    assert is_intensifier_delta_cap_candidate(text, "routine_discomfort", False)

    capped_score, reason, meta = apply_intensifier_delta_cap(
        text,
        current_score=0.80,
        attenuated_score=0.45,
        utterance_type="routine_discomfort",
    )
    assert reason == "intensifier_delta_cap"
    assert round(capped_score, 2) == 0.53
    assert meta["intensifier_allowed_delta"] == 0.08

    crisis_score, crisis_reason, crisis_meta = apply_intensifier_delta_cap(
        "요즘 너무 힘들고 더는 못 버티겠어",
        current_score=0.95,
        attenuated_score=0.60,
        utterance_type="crisis_candidate",
        is_crisis=True,
    )
    assert crisis_score == 0.95
    assert crisis_reason is None
    assert crisis_meta is None
    assert not is_intensifier_delta_cap_candidate("너무 지쳐서 출근하기 싫다", "emotional_distress", False)
    print("[보정 확인] 강조어 delta cap은 저위험 라우트에만 적용")


def assert_no_signal_baseline_is_not_perfect() -> None:
    """
    역할: 정서 점수 반영 발화가 없는 날이 wellness 100으로 저장되지 않는지 확인한다.
    입력: 없음
    출력: 없음
    """
    baseline_score = apply_no_signal_floor(0.0)
    assert baseline_score == 0.30
    assert depression_to_wellness(baseline_score) == 70.0
    print("[정책 확인] 정서 신호 없음 → wellness 70 기준 유지")


def assert_live_wellness_uses_today_only() -> None:
    """
    역할: 채팅 화면의 실시간 오늘 점수는 과거 일별 EWMA와 분리되는지 확인한다.
    입력: 없음
    출력: 없음
    """
    previous_daily_scores = [0.30, 0.4248, 0.5707, 0.6727]
    today_positive_scores = [0.2796150454878807, 0.2744261512160301]

    today_daily = utterance_to_daily(today_positive_scores)
    live_wellness = compute_wellness(today_daily, [], 1)

    smoothed_with_history = daily_to_smoothed(
        [apply_no_signal_floor(score) for score in previous_daily_scores] + [today_daily]
    )[-1]
    rolling_wellness = compute_wellness(
        smoothed_with_history,
        [70.0, 66.25, 59.26, 51.30],
        len(previous_daily_scores) + 1,
    )

    assert round(today_daily, 4) == 0.2781
    assert live_wellness["wellness_score"] == 72.19
    assert rolling_wellness["wellness_score"] == 57.57
    print("[정책 확인] 실시간 오늘 점수는 과거 EWMA가 아니라 오늘 발화 기준으로 계산")


def assert_benign_nli_hard_candidates_are_not_interrupted() -> None:
    """
    역할: blind 1008에서 나온 비자해 NLI hard 후보가 하드 인터럽트로 승격되지 않는지 확인한다.
    입력: 없음
    출력: 없음
    """
    benign_cases = [
        "음성으로 남기면, 내 탓처럼 말하는 메시지를 보고 손이 굳었어.",
        "음성으로 남기면, 먹는 것도 귀찮고 밤에는 잠이 얕아서 계속 지쳐.",
        "바닥에 눌어붙은 껌을 밟아서 계속 찝찝했어.",
        "밤에 다시 떠올려보니 이어폰 배터리가 얼마 남았는지 확인했어.",
        "밤에 다시 떠올려보니 회의에서 내 말을 중간에 끊어서 기분이 확 상했어.",
    ]
    for text in benign_cases:
        assert not should_hard_interrupt(text, True, 0.99), text

    assert should_hard_interrupt("요즘 너무 힘들고 더는 못 버티겠어", True, 0.99)
    assert should_hard_interrupt("죽고 싶다는 생각이 계속 들어", False, 0.10)
    print("[NLI 확인] 비자해 hard 오탐 후보는 차단하고 직접/강한 위험 신호는 보존")


def main() -> None:
    """
    역할: 감정 보정 규칙 회귀 테스트를 실행한다.
    입력: 없음
    출력: 없음
    """
    assert_routine_avoidance_is_demoted()
    assert_distress_avoidance_is_kept_negative()
    assert_real_fear_is_preserved()
    assert_practical_surprise_is_demoted()
    assert_academic_anxiety_question_stays_practical()
    assert_mild_unease_is_not_crisis_or_fear()
    assert_floating_mind_is_mild_unease()
    assert_mild_low_mood_is_not_disgust()
    assert_situational_low_mood_is_moderated()
    assert_positive_affect_does_not_raise_crisis_or_score()
    assert_positive_affect_preserves_happiness_label()
    assert_laughter_only_text_is_positive_low_signal()
    assert_low_signal_chat_does_not_raise_nli_candidate()
    assert_contextual_nli_caps_do_not_raise_crisis()
    assert_administrative_technical_text_is_low_signal()
    assert_physical_exertion_is_not_crisis_or_fear()
    assert_daily_routine_is_low_signal()
    assert_sensory_disgust_preserves_disgust_label()
    assert_disappointment_words_route_to_sadness()
    assert_situational_sadness_is_moderated()
    assert_academic_anxiety_is_moderated()
    assert_situational_anxiety_surprise_is_moderated()
    assert_assistant_quote_context_is_removed()
    assert_anger_marker_routes_to_anger()
    assert_situational_anger_is_moderated()
    assert_utterance_types()
    assert_score_policy()
    assert_legacy_score_pipeline_uses_score_policy()
    assert_routine_discomfort_cbt_burst_is_capped()
    assert_task_overload_is_routine_discomfort()
    assert_intensifier_delta_cap_is_low_risk_only()
    assert_no_signal_baseline_is_not_perfect()
    assert_live_wellness_uses_today_only()
    assert_benign_nli_hard_candidates_are_not_interrupted()
    print("[완료] RoBERTa 감정 보정 규칙 검증 통과")


if __name__ == "__main__":
    main()
