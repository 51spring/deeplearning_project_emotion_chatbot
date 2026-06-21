"""
clean_qwen_jsonl.py
역할: Qwen LoRA 학습용 JSONL에서 응답 품질을 해치는 샘플을 제거한다.
      - 에코 응답 (사용자 발화 4글자+ 복사)
      - 상담 전사 필러 ("네 좋습니다", "그다음에", "자세히 좀 이야기 좀 해봐" 등)
      - 의심 인물명 포함 응답 (유비/관우/장비 등 유출)
      - 응답 길이 극단값 (30자 미만 필러 톤, 250자 초과 장문)
      - 짧은 사용자 발화(<20자)에 의문문 종결 응답 (일상 잡담 되묻기 패턴)
      위기 태그([CRISIS]) 포함 샘플은 길이 필터만 완화 적용해 보존.
입력: data/processed/qwen_finetune_crisis_weighted.jsonl
출력: data/processed/qwen_finetune_cleaned.jsonl (+ 콘솔 리포트)
실행: C:/Users/WD/anaconda3/envs/dl_study/python.exe -u data/preprocess/clean_qwen_jsonl.py
"""

import os
import re
import json
import sys
from collections import Counter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(BASE_DIR, "data", "processed", "qwen_finetune_crisis_weighted.jsonl")
DST = os.path.join(BASE_DIR, "data", "processed", "qwen_finetune_cleaned.jsonl")

MIN_LEN, MAX_LEN = 30, 250
CRISIS_MIN_LEN = 20  # 위기 응답은 짧아도 허용(안전 안내 짧은 경우)
ECHO_CHUNK = 4
SHORT_USER_THRESHOLD = 20

FILLER_SUBSTRS = [
    "네 좋습니다", "네. 좋습니다",
    "그다음에는",
    "자세히 좀 이야기 좀 해봐",
    "자세히 이야기 좀 해봐",
    "목소리 잘 들리시나요",
    "상담사입니다",
    "오늘 상담",
    "회기를 시작",
]

SUSPICIOUS_NAMES = [
    "유비", "관우", "장비", "조조", "제갈량",
    "영희", "철수", "민수", "지영",
]


def normalize(text: str) -> str:
    """공백/구두점 제거 후 소문자화 (에코·필러 비교용)"""
    return re.sub(r"[\s\.,!?~·…'\"“”‘’]+", "", text).lower()


def has_echo(user: str, assistant: str, min_chunk: int = ECHO_CHUNK) -> bool:
    """사용자 발화의 min_chunk 글자 이상 연속 구간이 응답에 그대로 나타나는지"""
    u = normalize(user)
    a = normalize(assistant)
    if len(u) < min_chunk:
        return False
    for i in range(len(u) - min_chunk + 1):
        if u[i:i + min_chunk] in a:
            return True
    return False


def has_filler(assistant: str) -> bool:
    """상담 전사 필러 표현 포함 여부"""
    norm = normalize(assistant)
    return any(normalize(p) in norm for p in FILLER_SUBSTRS)


def has_suspicious_name(assistant: str) -> bool:
    """의심 인물명 포함 여부"""
    return any(name in assistant for name in SUSPICIOUS_NAMES)


def ends_with_question(assistant: str) -> bool:
    """응답이 물음표로 종결되는지 (일상 잡담 되묻기 탐지)"""
    return assistant.rstrip().endswith(("?", "요?", "까?", "나요?"))


def extract_turn(msgs: list[dict]) -> tuple[str, str]:
    """messages에서 마지막 user/assistant 턴 추출"""
    user = next((m["content"] for m in msgs if m["role"] == "user"), "")
    asst = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
    return user, asst


def classify(sample: dict) -> tuple[bool, str]:
    """
    역할: 샘플을 유지할지 결정하고 제외 사유를 반환한다.
    입력: {"messages": [...]}
    출력: (keep_bool, reason)
    """
    user, asst = extract_turn(sample.get("messages", []))
    if not user or not asst:
        return False, "empty_turn"

    is_crisis = "[CRISIS]" in asst

    # 길이 필터 — 위기 응답은 최소 길이만 완화
    min_len = CRISIS_MIN_LEN if is_crisis else MIN_LEN
    if len(asst) < min_len:
        return False, "too_short"
    if len(asst) > MAX_LEN:
        return False, "too_long"

    # 인물명·필러는 위기 여부와 무관하게 제거
    if has_suspicious_name(asst):
        return False, "suspicious_name"
    if has_filler(asst):
        return False, "filler"

    # 에코는 위기 샘플이 아닐 때만 차단 (위기 샘플은 사용자 문구 반복이 안전 안내에 흔함)
    if not is_crisis and has_echo(user, asst):
        return False, "echo"

    # 짧은 사용자 발화 + 의문문 응답 = 되묻기 루프 패턴 → 위기 아닌 경우만 차단
    if not is_crisis and len(user) < SHORT_USER_THRESHOLD and ends_with_question(asst):
        # 단, 일정 길이 이상이고 공감 단어가 포함되면 보존(공감+탐색 질문 샘플)
        empathic_markers = ["힘드셨", "속상", "마음", "느끼", "공감", "이해", "지치", "버거"]
        if not any(k in asst for k in empathic_markers):
            return False, "short_user_question_end"

    return True, "keep"


def main() -> None:
    drop_counter = Counter()
    kept = []
    total = 0

    with open(SRC, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            obj = json.loads(line)
            keep, reason = classify(obj)
            if keep:
                kept.append(obj)
            else:
                drop_counter[reason] += 1

    with open(DST, "w", encoding="utf-8") as f:
        for obj in kept:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"[입력]  {SRC}")
    print(f"[출력]  {DST}")
    print(f"[총 샘플] {total}")
    print(f"[유지]    {len(kept)} ({len(kept)/total*100:.1f}%)")
    print(f"[제거]    {sum(drop_counter.values())}")
    for reason, cnt in drop_counter.most_common():
        print(f"  - {reason:>26}: {cnt}")

    # 위기 샘플 보존 확인
    crisis_kept = sum(1 for s in kept if "[CRISIS]" in extract_turn(s["messages"])[1])
    print(f"\n[위기 샘플 보존] {crisis_kept}건")

    # 정제 후 분포 재확인
    lens = [len(extract_turn(s["messages"])[1]) for s in kept]
    if lens:
        lens_sorted = sorted(lens)
        n = len(lens_sorted)
        print("\n[정제 후 응답 길이]")
        print(f"  min={lens_sorted[0]}, p25={lens_sorted[n//4]}, "
              f"median={lens_sorted[n//2]}, p75={lens_sorted[3*n//4]}, "
              f"p95={lens_sorted[int(0.95*n)]}, max={lens_sorted[-1]}, "
              f"mean={sum(lens)/n:.1f}")

    # 응답 패턴 재확인
    q_end = sum(
        1 for s in kept
        if ends_with_question(extract_turn(s["messages"])[1])
    )
    echo_left = sum(
        1 for s in kept
        if has_echo(*extract_turn(s["messages"]))
    )
    print(f"  물음표 종결 비율: {q_end/len(kept)*100:.2f}%")
    print(f"  에코 잔존 비율  : {echo_left/len(kept)*100:.2f}% (위기 샘플만 허용)")


if __name__ == "__main__":
    main()
