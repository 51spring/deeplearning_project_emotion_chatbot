"""
qwen_response_style_data_guard.py
역할: Qwen 응답 스타일 SFT 데이터가 목표 분포와 기본 품질 규칙을 만족하는지 검증한다.
입력:
  - data/processed/qwen_response_style_sft.jsonl
  - data/processed/qwen_finetune_cleaned_strict.jsonl
  - data/processed/qwen_finetune_response_style_mix.jsonl
  - data/processed/qwen_response_style_report.json
출력: 콘솔 검증 결과 및 실패 시 AssertionError
실행: C:/Users/WD/anaconda3/envs/dl_study/python.exe -u eval/qwen_response_style_data_guard.py
"""

import json
import os
from collections import Counter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

STYLE_PATH = os.path.join(PROCESSED_DIR, "qwen_response_style_sft.jsonl")
STRICT_PATH = os.path.join(PROCESSED_DIR, "qwen_finetune_cleaned_strict.jsonl")
MIX_PATH = os.path.join(PROCESSED_DIR, "qwen_finetune_response_style_mix.jsonl")
REPORT_PATH = os.path.join(PROCESSED_DIR, "qwen_response_style_report.json")

EXPECTED_COUNTS = {
    "casual_share": 420,
    "positive_share": 420,
    "routine_discomfort": 420,
    "emotional_distress": 600,
    "relationship": 420,
    "sleep_fatigue": 360,
    "preference_question": 420,
    "practical_question": 420,
    "crisis_candidate": 300,
}

BANNED_TEXTS = [
    "네 좋습니다",
    "그다음에는",
    "자세히 좀 이야기",
    "목소리 잘 들리",
    "상담사입니다",
    "커피 섭취",
    "새벽 세 시부터 근무",
    "선생님들이",
    "어린 연령층",
    "넣으라고 하더라고요",
    "유비",
    "관우",
    "장비",
    "홈트이",
    "저녁 운동가",
    "위주으로",
]


def load_jsonl(path: str) -> list[dict]:
    """
    역할: JSONL 파일을 읽어 샘플 리스트로 반환한다.
    입력: JSONL 경로
    출력: 샘플 dict 리스트
    """
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def extract_turn(sample: dict) -> tuple[str, str]:
    """
    역할: 샘플에서 user와 assistant 발화를 추출한다.
    입력: SFT 샘플 dict
    출력: (사용자 발화, assistant 응답)
    """
    messages = sample.get("messages", [])
    user_text = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")
    assistant_text = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "assistant"), "")
    return user_text, assistant_text


def count_categories(samples: list[dict]) -> Counter:
    """
    역할: meta.category 기준 분포를 계산한다.
    입력: SFT 샘플 리스트
    출력: 카테고리별 Counter
    """
    counter = Counter()
    for sample in samples:
        counter[sample.get("meta", {}).get("category", "unknown")] += 1
    return counter


def assert_distribution(style_samples: list[dict], mix_samples: list[dict], strict_samples: list[dict]) -> None:
    """
    역할: 스타일/혼합/엄격 정제 데이터의 개수 조건을 검증한다.
    입력: 스타일 샘플, 혼합 샘플, 엄격 정제 샘플
    출력: 없음
    """
    style_counts = count_categories(style_samples)
    assert dict(style_counts) == EXPECTED_COUNTS, style_counts
    assert len(style_samples) == sum(EXPECTED_COUNTS.values()), len(style_samples)
    assert len(strict_samples) >= 9000, len(strict_samples)
    assert len(mix_samples) == len(style_samples) + 1400, len(mix_samples)
    print("[분포 확인] 스타일/혼합/엄격 정제 개수 통과")


def assert_text_quality(style_samples: list[dict]) -> None:
    """
    역할: 스타일 데이터에 금지 표현과 위기 태그 오염이 없는지 확인한다.
    입력: 스타일 SFT 샘플 리스트
    출력: 없음
    """
    seen = set()
    for sample in style_samples:
        user_text, assistant_text = extract_turn(sample)
        category = sample.get("meta", {}).get("category", "")
        key = (user_text, assistant_text)
        assert key not in seen, key
        seen.add(key)
        assert user_text and assistant_text, sample
        assert len(assistant_text) <= 230, assistant_text
        for banned in BANNED_TEXTS:
            assert banned not in assistant_text and banned not in user_text, (banned, user_text, assistant_text)
        if category == "crisis_candidate":
            assert "[CRISIS]" in assistant_text, assistant_text
        else:
            assert "[CRISIS]" not in assistant_text, assistant_text
    print("[품질 확인] 금지 표현/중복/위기 태그 조건 통과")


def assert_representative_cases(style_samples: list[dict]) -> None:
    """
    역할: 과거 오답을 유발한 대표 발화들이 의도에 맞는 응답으로 포함됐는지 확인한다.
    입력: 스타일 SFT 샘플 리스트
    출력: 없음
    """
    by_user = {}
    for sample in style_samples:
        user_text, assistant_text = extract_turn(sample)
        by_user.setdefault(user_text, []).append((sample.get("meta", {}).get("category", ""), assistant_text))

    expected_users = [
        "밥 먹었어",
        "내일이 기대돼",
        "출근하기 싫다",
        "요즘 계속 우울해",
        "친구랑 싸웠어",
        "요즘 잠이 잘 안 와",
        "멜론이랑 스포티파이중에 뭐가 더 좋을까?",
        "오늘 뭐 먹지?",
        "죽고 싶다는 생각이 들어",
    ]
    for user_text in expected_users:
        assert user_text in by_user, user_text

    assert any("기대" in text or "설렘" in text or "기다려지는" in text for _, text in by_user["내일이 기대돼"])
    assert any("시작" in text or "다그치지" in text for _, text in by_user["출근하기 싫다"])
    assert any("멜론" in text and "스포티파이" in text for _, text in by_user["멜론이랑 스포티파이중에 뭐가 더 좋을까?"])
    assert any("109" in text and "[CRISIS]" in text for _, text in by_user["죽고 싶다는 생각이 들어"])
    print("[대표 케이스 확인] 과거 오답 유발 발화 포함 통과")


def assert_report_consistency() -> None:
    """
    역할: JSON 리포트와 실제 파일의 핵심 수치가 일치하는지 확인한다.
    입력: 없음
    출력: 없음
    """
    with open(REPORT_PATH, encoding="utf-8") as f:
        report = json.load(f)
    assert report["style_total"] == sum(EXPECTED_COUNTS.values()), report
    assert report["style_categories"] == EXPECTED_COUNTS, report["style_categories"]
    assert report["mixed_total"] == 5180, report["mixed_total"]
    print("[리포트 확인] 생성 리포트 수치 통과")


def main() -> None:
    """
    역할: Qwen 응답 스타일 데이터 검증을 전체 실행한다.
    입력: 없음
    출력: 없음
    """
    style_samples = load_jsonl(STYLE_PATH)
    strict_samples = load_jsonl(STRICT_PATH)
    mix_samples = load_jsonl(MIX_PATH)
    assert_distribution(style_samples, mix_samples, strict_samples)
    assert_text_quality(style_samples)
    assert_representative_cases(style_samples)
    assert_report_consistency()
    print("[완료] Qwen 응답 스타일 데이터 검증 통과")


if __name__ == "__main__":
    main()
