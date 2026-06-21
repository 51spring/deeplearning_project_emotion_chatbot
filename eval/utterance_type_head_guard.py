"""
utterance_type_head_guard.py
역할: RoBERTa 발화 타입 head가 런타임에서 로드되고 대표 문장을 분류하는지 검증한다.
입력: 없음 (내장 케이스)
출력: 콘솔 검증 결과 및 실패 시 AssertionError
실행: C:/Users/WD/anaconda3/envs/dl_study/python.exe -u eval/utterance_type_head_guard.py
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from pipeline.roberta_score import infer_single, load_roberta_model, load_temperature  # noqa: E402


CASES = [
    ("밥 먹었어", {"casual_share", "positive_share"}),
    ("내일이 기대돼", {"positive_share"}),
    ("출근하기 싫다", {"routine_discomfort"}),
    ("요즘 계속 우울해", {"emotional_distress"}),
    ("멜론이랑 스포티파이중에 뭐가 더 좋을까?", {"preference_question"}),
    ("오늘 뭐 먹지?", {"preference_question", "practical_question"}),
    ("죽고 싶다", {"crisis_candidate"}),
]


def main() -> None:
    """
    역할: 대표 발화의 발화 타입 head 추론 결과를 검증한다.
    입력: 없음
    출력: 없음
    """
    model, tokenizer, device = load_roberta_model()
    T_emotion, T_nli = load_temperature()

    for text, expected_types in CASES:
        result = infer_single(text, model, tokenizer, device, T_emotion=T_emotion, T_nli=T_nli)
        assert result["utterance_type_reason"] in {
            "roberta_utterance_intent_head",
            "practical_question_override",
        }, result
        assert result["utterance_type"] in expected_types, result
        assert result["utterance_type_confidence"] >= 0.45, result
        print(
            f"[head 확인] {text} → {result['utterance_type']} "
            f"conf={result['utterance_type_confidence']:.3f}"
        )

    print("[완료] RoBERTa 발화 타입 head 검증 통과")


if __name__ == "__main__":
    main()
