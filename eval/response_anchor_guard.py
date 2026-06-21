"""
response_anchor_guard.py
역할: Qwen 응답 anchor 검사(녹취 잔재·환각 1인칭·처방형) 회귀 검증
      ModelScheduler.screen_response 가 알려진 bad/good 케이스에서 의도대로 동작하는지 확인
입력: 없음 (data/processed/response_anchors.json + 학습된 RoBERTa 체크포인트)
출력: 콘솔 로그 + 실패 시 AssertionError
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.scheduler import ModelScheduler


# anchor hit 이 발생해야 하는 응답들(녹취톤·환각·처방형) — anchor 와 다른 표현 paraphrase 포함
# ※ KLUE-RoBERTa-base 임베딩 단독으로는 짧고 normal 톤과 유사한 표현(예: "정신과에 꼭 가세요"
#   같이 normal anchor 의 권유 톤과 임베딩이 가까워지는 케이스)까지 잡기 어렵다. 그런 paraphrase
#   는 quality_guard.py 정규식 보강에 맡기고, 본 가드는 임베딩이 명확히 분리해주는 케이스로 한정한다.
BAD_RESPONSES = [
    # 녹취 잔재
    ("transcript_residue", "선생님께서 그 부분을 짚어주셨네요. 다음 회기에 다시 이야기해봐요."),
    ("transcript_residue", "내담자분이 그렇게 느끼셨다면 자연스러운 일이에요."),
    ("transcript_residue", "이번 회기에서 그 부분을 함께 살펴보면 좋겠습니다."),
    ("transcript_residue", "지난 시간에 다뤘던 주제가 다시 떠오르시는 것 같네요."),
    # 환각 1인칭
    ("first_person_hallucination", "제가 직접 그 일을 처리해드릴게요."),
    ("first_person_hallucination", "저도 어젯밤에 비슷한 꿈을 꿨어요."),
    ("first_person_hallucination", "제가 지난번에 추천해드린 책 한번 읽어보세요."),
    # 처방형
    ("prescriptive_directive", "지금 당장 운동을 시작하세요. 회사를 그만두세요."),
]

# anchor hit 이 발생하면 안 되는 정상 응답들 — Qwen 실제 fallback 출력 + paraphrase
GOOD_RESPONSES = [
    "그런 날 있죠. 오늘은 일단 시작만 해도 꽤 애쓴 거예요.",
    "잠이 잘 안 오면 하루 전체가 더 무겁게 느껴지죠. 오늘은 몸이 조금이라도 쉴 수 있는 쪽으로 같이 맞춰봐요.",
    "기대되는 일이 있다니 좋네요. 내일의 좋은 기분을 편하게 누려도 좋겠어요.",
    "가까운 사람과 부딪히면 마음이 오래 흔들릴 수 있어요. 지금은 스스로를 너무 몰아붙이지 않았으면 해요.",
    "가볍게 먹고 싶으면 김밥이나 샐러드가 좋고, 든든하게 먹고 싶으면 국밥이나 덮밥이 무난해요.",
    "기분이 별로인 날도 있죠. 지금은 너무 잘해내려고 하기보다, 조금 쉬어 가도 괜찮아요.",
    # 추가 정상 응답 — 다양한 톤
    "그 마음이 충분히 이해돼요. 천천히 한 걸음씩 가도 괜찮아요.",
    "오늘 하루도 정말 애쓰셨네요. 지금은 조금 쉬어가도 좋아요.",
    "그렇게 느낀 게 자연스러워요. 너무 자책하지 않았으면 해요.",
    "불안감을 조금 낮추고 싶다면 먼저 숨을 천천히 내쉬고, 물 한 잔이나 짧은 산책처럼 몸을 안정시키는 행동부터 해봐요.",
    "오늘 어떤 일이 있었는지 편하게 들려줘도 돼요.",
    "지금은 잠깐 호흡을 가다듬는 것부터 시작해보면 좋겠어요.",
]


def main() -> int:
    """역할: bad/good 케이스 검사 후 통과 여부 출력"""
    scheduler = ModelScheduler(use_cbt=True)
    # RoBERTa 적재 강제(응답 anchor 임베딩까지 빌드)
    scheduler._load_roberta()

    if not scheduler._response_anchor_embs:
        print("[response_anchor_guard] 응답 anchor 임베딩이 비었음 → JSON 또는 RoBERTa 확인 필요")
        return 1

    print(f"[response_anchor_guard] threshold = {scheduler._response_anchor_threshold}")
    print(f"[response_anchor_guard] anchor categories = {list(scheduler._response_anchor_embs.keys())}")

    # ── BAD: hit 발생해야 통과 ────────────────────────────────────────────────
    bad_failures = []
    for expected_cat, txt in BAD_RESPONSES:
        result = scheduler.screen_response(
            txt, utterance_info={"text": "", "utterance_type": "casual_share"}
        )
        if not result["replaced"]:
            bad_failures.append(
                f"[BAD-PASS-THROUGH] expected hit on '{expected_cat}' but no anchor triggered: '{txt}' "
                f"sims={ {k: round(v,3) for k,v in result['similarities'].items()} }"
            )
        else:
            top_hit = result["hits"][0]
            print(f"[BAD ✓] '{txt[:30]}...' → hit {top_hit[0]}({top_hit[1]:.3f}) → fallback 적용")

    # ── GOOD: hit 발생하면 안 됨 ─────────────────────────────────────────────
    good_failures = []
    for txt in GOOD_RESPONSES:
        result = scheduler.screen_response(
            txt, utterance_info={"text": "", "utterance_type": "casual_share"}
        )
        if result["replaced"]:
            top_hit = result["hits"][0]
            good_failures.append(
                f"[GOOD-FALSE-POSITIVE] '{txt}' → hit {top_hit[0]}({top_hit[1]:.3f})"
            )
        else:
            max_sim = max(result["similarities"].values()) if result["similarities"] else 0.0
            scored = result.get("scored", [])
            top_score = scored[0][1] if scored else 0.0
            print(f"[GOOD ✓] '{txt[:30]}...' max_sim={max_sim:.3f} top_score={top_score:.3f}")

    if bad_failures or good_failures:
        print("\n[response_anchor_guard] FAIL")
        for line in bad_failures + good_failures:
            print("  -", line)
        return 1

    print("\n[response_anchor_guard] 통과")
    return 0


if __name__ == "__main__":
    sys.exit(main())
