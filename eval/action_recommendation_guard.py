"""
action_recommendation_guard.py
역할: 행동 추천 v1(pipeline.action_recommendation.build_chat_recommendations)의
      라우팅 규칙 회귀 가드. 모델/서버 없이 순수 함수만 검증한다(빠름).
실행: C:\\Users\\WD\\anaconda3\\envs\\dl_study\\python.exe eval\\action_recommendation_guard.py
"""
import os
import sys

# 저장소 루트를 import 경로에 추가한다.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pipeline.action_recommendation import build_chat_recommendations  # noqa: E402

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    """단일 케이스 통과/실패를 기록한다."""
    if condition:
        PASSED.append(name)
        print(f"[PASS] {name}")
    else:
        FAILED.append(name)
        print(f"[FAIL] {name} :: {detail}")


def _ids(recs: list[dict]) -> list[str]:
    return [r.get("id") for r in recs]


def _cats(recs: list[dict]) -> list[str]:
    return [r.get("category") for r in recs]


def main() -> int:
    """행동 추천 v1 규칙 케이스를 검증하고 통과/실패 수를 반환한다."""
    # 1. 위기 발화 결과 → safety 추천만 반환
    recs = build_chat_recommendations(
        {"top_emotion": "슬픔", "depression_tendency_score": 0.9,
         "utterance_type": "crisis_candidate"},
        {"wellness_score": 5.0, "label": "위험"},
        is_crisis=True,
    )
    check(
        "위기 → safety 추천만",
        len(recs) == 1 and recs[0]["category"] == "safety"
        and recs[0]["id"] == "safety_check",
        f"recs={_ids(recs)} cats={_cats(recs)}",
    )

    # 2. positive_share/행복 → 긍정 유지 추천
    recs = build_chat_recommendations(
        {"top_emotion": "행복", "depression_tendency_score": 0.02,
         "utterance_type": "positive_share"},
        {"wellness_score": 85.0, "label": "양호"},
        is_crisis=False,
    )
    check(
        "행복/positive_share → 긍정 추천",
        len(recs) >= 1 and all(c == "positive" for c in _cats(recs))
        and "positive_save" in _ids(recs),
        f"recs={_ids(recs)}",
    )

    # 3. routine_discomfort → 할 일 줄이기 추천
    recs = build_chat_recommendations(
        {"top_emotion": "중립", "depression_tendency_score": 0.05,
         "utterance_type": "routine_discomfort"},
        {"wellness_score": 55.0, "label": "주의"},
        is_crisis=False,
    )
    check(
        "routine_discomfort → 할 일 줄이기 추천",
        "routine_break" in _ids(recs) and all(c == "routine" for c in _cats(recs)),
        f"recs={_ids(recs)}",
    )

    # 4. depression_tendency_score >= 0.4 → 우울 경향 지원 추천
    recs = build_chat_recommendations(
        {"top_emotion": "슬픔", "depression_tendency_score": 0.55,
         "utterance_type": "emotional_distress"},
        {"wellness_score": 35.0, "label": "주의"},
        is_crisis=False,
    )
    check(
        "우울 경향 >= 0.4 → 우울 경향 지원 추천",
        len(recs) >= 1 and recs[0]["category"] == "tendency"
        and "tendency_support" in _ids(recs),
        f"recs={_ids(recs)}",
    )

    # 5. 중립/저신호 → 짧은 체크인 유지 추천
    recs = build_chat_recommendations(
        {"top_emotion": "중립", "depression_tendency_score": 0.05,
         "utterance_type": "casual_neutral"},
        {"wellness_score": 70.0, "label": "보통"},
        is_crisis=False,
    )
    check(
        "중립/저신호 → 짧은 체크인 유지",
        len(recs) == 1 and recs[0]["id"] == "checkin"
        and recs[0]["category"] == "checkin",
        f"recs={_ids(recs)}",
    )

    # 추가 가드 a) 우울 경향 우선순위가 일상 과부하/감정보다 앞선다
    recs = build_chat_recommendations(
        {"top_emotion": "슬픔", "depression_tendency_score": 0.45,
         "utterance_type": "routine_discomfort"},
        {"wellness_score": 40.0, "label": "주의"},
        is_crisis=False,
    )
    check(
        "우울 경향 우선순위 > 일상 과부하",
        recs and recs[0]["category"] == "tendency",
        f"recs={_ids(recs)}",
    )

    # 추가 가드 b) 항상 최대 2개까지만 반환
    over = []
    for emo in ["행복", "슬픔", "분노", "공포", "놀람", "중립", "혐오"]:
        for ut in ["casual_neutral", "casual_share", "positive_share",
                   "routine_discomfort", "emotional_distress"]:
            for dts in [0.0, 0.25, 0.5]:
                r = build_chat_recommendations(
                    {"top_emotion": emo, "depression_tendency_score": dts,
                     "utterance_type": ut},
                    {"wellness_score": 50.0, "label": "주의"},
                    is_crisis=False,
                )
                if len(r) > 2:
                    over.append((emo, ut, dts, len(r)))
    check("항상 최대 2개 이하", not over, f"over={over[:3]}")

    # 추가 가드 c) 단정적/진단 표현을 쓰지 않는다(부드러운 표현 정책)
    banned = ["우울합니다", "치료가 필요", "진단", "장애입니다", "병원에 가야"]
    texts: list[str] = []
    for emo in ["행복", "슬픔", "분노", "공포", "놀람", "중립", "혐오"]:
        for ut in ["casual_neutral", "casual_share", "positive_share",
                   "routine_discomfort", "emotional_distress", "practical_question"]:
            for dts in [0.0, 0.25, 0.5]:
                for r in build_chat_recommendations(
                    {"top_emotion": emo, "depression_tendency_score": dts,
                     "utterance_type": ut},
                    {"wellness_score": 50.0, "label": "주의"},
                    is_crisis=False,
                ):
                    texts.append(f"{r['title']} {r['message']}")
    for r in build_chat_recommendations({}, {}, is_crisis=True):
        texts.append(f"{r['title']} {r['message']}")
    found = [w for w in banned if any(w in t for t in texts)]
    check("단정적/진단 표현 미사용", not found, f"found={found}")

    # 추가 가드 d) 빈/누락 입력에도 안전하게 기본 추천을 반환
    recs = build_chat_recommendations({}, {}, is_crisis=False)
    check(
        "빈 입력에도 기본 체크인 반환",
        len(recs) >= 1 and recs[0]["id"] == "checkin",
        f"recs={_ids(recs)}",
    )

    # 추가 가드 e) 추천 dict 필수 키 형식 검증
    sample = build_chat_recommendations(
        {"top_emotion": "공포", "depression_tendency_score": 0.0,
         "utterance_type": "emotional_distress"},
        {"wellness_score": 45.0, "label": "주의"},
        is_crisis=False,
    )
    required_keys = {"id", "title", "message", "reason", "priority", "category"}
    ok_schema = all(required_keys.issubset(r.keys()) for r in sample) and all(
        r["priority"] in {"high", "medium", "low"} for r in sample
    )
    check("추천 dict 필수 키/우선순위 형식", ok_schema, f"sample={sample}")

    print("\n" + "=" * 50)
    print(f"통과 {len(PASSED)} / 실패 {len(FAILED)}")
    if FAILED:
        print("실패 목록:")
        for name in FAILED:
            print(f"  - {name}")
        return 1
    print("행동 추천 가드 전체 통과")
    return 0


if __name__ == "__main__":
    sys.exit(main())
