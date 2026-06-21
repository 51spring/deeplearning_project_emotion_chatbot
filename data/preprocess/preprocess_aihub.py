"""
preprocess_aihub.py
역할: AI Hub 심리상담 데이터 전처리
      ① 내담자 발화 파싱 → 위기 발화 추출 → NLI entailment 쌍 보완
      ② 비위기 발화 추출 → NLI contradiction 쌍 보완
      ③ 기존 data/nli/nli_pairs.csv 에 병합 → data/processed/nli_pairs.csv 에도 복사
실행: python data/preprocess/preprocess_aihub.py
"""

import os
import re
import json
import random
import pandas as pd

# ── 경로 ────────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR       = os.path.join(BASE_DIR, "data", "raw", "16.심리상담 데이터", "1.데이터")
NLI_OUT       = os.path.join(BASE_DIR, "data", "nli", "nli_pairs.csv")
PROCESSED_OUT = os.path.join(BASE_DIR, "data", "processed", "nli_pairs.csv")

# NLI hypothesis 고정
HYPOTHESIS = "이 사람은 스스로를 해치려 한다"

# 레이블: 0=entailment(위기), 1=neutral, 2=contradiction(비위기)
LABEL_ENTAILMENT    = 0
LABEL_NEUTRAL       = 1
LABEL_CONTRADICTION = 2

# ── 위기 키워드 ─────────────────────────────────────────────────────────────────
CRISIS_KEYWORDS = [
    "죽고 싶", "죽고싶", "자살", "스스로 목숨", "목숨을 끊",
    "삶을 끝내", "살기 싫", "살고 싶지 않", "살아있기 싫",
    "사라지고 싶", "없어지고 싶", "그냥 다 끝내", "다 끝내버리",
    "죽어버리", "죽어야", "안 살고 싶", "안살고 싶",
    "스스로를 해치", "해치고 싶", "자해", "손목을 그",
]

# 비위기 오탐 방지 키워드 — 이 패턴이 있으면 contradiction 후보
NON_CRISIS_PATTERNS = [
    "배고파 죽겠", "더워 죽겠", "웃겨 죽겠", "졸려 죽겠",
    "재밌어 죽겠", "뿌듯해 죽겠", "행복해 죽겠", "맛있어 죽겠",
    "신나 죽겠", "감사해서 죽겠",
]

# 익명화 태그 제거 패턴
ANON_PATTERN = re.compile(r"@\S+")


# ────────────────────────────────────────────────────────────────────────────────
# 텍스트 파일 파싱
# ────────────────────────────────────────────────────────────────────────────────
def parse_session_file(filepath: str) -> list:
    """
    역할: AI Hub 심리상담 txt 파일 하나를 파싱, 발화 리스트 반환
    입력: 파일 경로
    출력: [{"speaker": str, "text": str}] 리스트
    """
    turns = []
    try:
        with open(filepath, encoding="utf-8") as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        try:
            with open(filepath, encoding="cp949") as f:
                lines = f.readlines()
        except Exception:
            return []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("상담사"):
            text = re.sub(r"^상담사\s*:\s*", "", line).strip()
            text = ANON_PATTERN.sub("", text).strip()
            if text:
                turns.append({"speaker": "상담사", "text": text})
        elif line.startswith("내담자"):
            text = re.sub(r"^내담자\s*:\s*", "", line).strip()
            text = ANON_PATTERN.sub("", text).strip()
            if text:
                turns.append({"speaker": "내담자", "text": text})
    return turns


def collect_all_client_utterances(raw_dir: str) -> list:
    """
    역할: Training/Validation 전체 txt 파일에서 내담자 발화 수집
    입력: raw_dir (16.심리상담 데이터/1.데이터 경로)
    출력: 내담자 발화 문자열 리스트
    """
    all_texts = []
    for split in ["Training", "Validation"]:
        session_dir = os.path.join(raw_dir, split, "01.원천데이터")
        if not os.path.isdir(session_dir):
            continue
        for session_folder in os.listdir(session_dir):
            folder_path = os.path.join(session_dir, session_folder)
            if not os.path.isdir(folder_path):
                continue
            for fname in os.listdir(folder_path):
                if not fname.endswith(".txt"):
                    continue
                fpath = os.path.join(folder_path, fname)
                turns = parse_session_file(fpath)
                for t in turns:
                    if t["speaker"] == "내담자":
                        all_texts.append(t["text"])
    return all_texts


# ────────────────────────────────────────────────────────────────────────────────
# 위기 / 비위기 / 중립 분류
# ────────────────────────────────────────────────────────────────────────────────
def classify_utterance(text: str) -> str:
    """
    역할: 발화 텍스트를 crisis / non_crisis / neutral 로 분류
    입력: 발화 문자열
    출력: "crisis" | "non_crisis" | "neutral"
    """
    for kw in NON_CRISIS_PATTERNS:
        if kw in text:
            return "non_crisis"
    for kw in CRISIS_KEYWORDS:
        if kw in text:
            return "crisis"
    return "neutral"


def filter_utterances(texts: list) -> dict:
    """
    역할: 전체 발화를 분류 + 길이 필터 적용
    입력: 발화 리스트
    출력: {"crisis": [...], "non_crisis": [...], "neutral": [...]}
    """
    result = {"crisis": [], "non_crisis": [], "neutral": []}
    for t in texts:
        if len(t) < 5 or len(t) > 300:
            continue
        cls = classify_utterance(t)
        result[cls].append(t)
    return result


def dedup_texts(texts: list) -> list:
    """역할: 문자열 리스트 중복 제거 (순서 유지)"""
    seen = set()
    result = []
    for t in texts:
        if t not in seen:
            seen.add(t)
            result.append(t)
    return result


# ────────────────────────────────────────────────────────────────────────────────
# NLI 쌍 생성
# ────────────────────────────────────────────────────────────────────────────────
def build_nli_pairs_from_aihub(
    crisis_texts: list,
    non_crisis_texts: list,
    neutral_texts: list,
    max_crisis: int = 80,
    max_non_crisis: int = 60,
    max_neutral: int = 40,
    seed: int = 42,
) -> list:
    """
    역할: AI Hub 발화에서 NLI 쌍 생성
    입력:
        crisis_texts     - 위기 발화 (label=entailment)
        non_crisis_texts - 비위기 발화 (label=contradiction)
        neutral_texts    - 중립 발화 (label=neutral)
        max_*            - 각 레이블 최대 샘플 수
    출력: NLI 쌍 딕셔너리 리스트
    """
    random.seed(seed)
    pairs = []

    def sample_texts(texts, max_n):
        return random.sample(texts, min(len(texts), max_n))

    for t in sample_texts(crisis_texts, max_crisis):
        pairs.append({"premise": t, "hypothesis": HYPOTHESIS, "label": LABEL_ENTAILMENT})

    for t in sample_texts(non_crisis_texts, max_non_crisis):
        pairs.append({"premise": t, "hypothesis": HYPOTHESIS, "label": LABEL_CONTRADICTION})

    for t in sample_texts(neutral_texts, max_neutral):
        pairs.append({"premise": t, "hypothesis": HYPOTHESIS, "label": LABEL_NEUTRAL})

    return pairs


# ────────────────────────────────────────────────────────────────────────────────
# 기존 NLI CSV 병합
# ────────────────────────────────────────────────────────────────────────────────
def merge_nli_pairs(existing_path: str, new_pairs: list) -> pd.DataFrame:
    """
    역할: 기존 nli_pairs.csv + AI Hub 추출 쌍 병합 (premise 중복 제거)
    입력: 기존 CSV 경로, 신규 쌍 리스트
    출력: 병합된 DataFrame
    """
    if os.path.exists(existing_path) and os.path.getsize(existing_path) > 0:
        existing_df = pd.read_csv(existing_path, encoding="utf-8-sig")
    else:
        existing_df = pd.DataFrame(columns=["premise", "hypothesis", "label"])

    new_df = pd.DataFrame(new_pairs)
    merged = pd.concat([existing_df, new_df], ignore_index=True)
    # 기존 데이터 우선, premise 기준 중복 제거
    merged = merged.drop_duplicates(subset=["premise"], keep="first").reset_index(drop=True)
    return merged


# ────────────────────────────────────────────────────────────────────────────────
# 통계 출력
# ────────────────────────────────────────────────────────────────────────────────
def print_stats(df: pd.DataFrame, title: str = "NLI 쌍 통계"):
    """역할: NLI DataFrame 레이블 분포 출력"""
    label_map = {0: "entailment(위기)", 1: "neutral", 2: "contradiction(비위기)"}
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(f"  총 쌍 수: {len(df)}")
    for lbl, name in label_map.items():
        cnt = (df["label"] == lbl).sum()
        print(f"  {name}({lbl}): {cnt}")


# ────────────────────────────────────────────────────────────────────────────────
# 메인
# ────────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("AI Hub 심리상담 데이터 전처리 시작")
    print("=" * 60)

    # ── 1. 내담자 발화 수집 ──────────────────────────────────────────────────────
    print("\n[1] 내담자 발화 수집 중...")
    all_client_texts = collect_all_client_utterances(RAW_DIR)
    print(f"  수집된 내담자 발화 수: {len(all_client_texts)}")

    # ── 2. 위기/비위기/중립 분류 ─────────────────────────────────────────────────
    print("\n[2] 위기/비위기/중립 분류 중...")
    classified    = filter_utterances(all_client_texts)
    crisis_texts     = dedup_texts(classified["crisis"])
    non_crisis_texts = dedup_texts(classified["non_crisis"])
    neutral_texts    = dedup_texts(classified["neutral"])

    print(f"  위기 발화:    {len(crisis_texts)}")
    print(f"  비위기 발화:  {len(non_crisis_texts)}")
    print(f"  중립 발화:    {len(neutral_texts)}")

    if crisis_texts:
        print("\n  [위기 발화 샘플 5개]")
        for t in crisis_texts[:5]:
            print(f"    - {t[:80]}")
    if non_crisis_texts:
        print("\n  [비위기 발화 샘플 3개]")
        for t in non_crisis_texts[:3]:
            print(f"    - {t[:80]}")

    # ── 3. NLI 쌍 생성 ──────────────────────────────────────────────────────────
    print("\n[3] AI Hub 기반 NLI 쌍 생성 중...")
    new_pairs = build_nli_pairs_from_aihub(
        crisis_texts, non_crisis_texts, neutral_texts,
        max_crisis=80, max_non_crisis=60, max_neutral=40,
    )
    print(f"  생성된 NLI 쌍: {len(new_pairs)}")

    # ── 4. 기존 NLI CSV 병합 ─────────────────────────────────────────────────────
    print("\n[4] 기존 nli_pairs.csv 병합 중...")
    merged_df = merge_nli_pairs(NLI_OUT, new_pairs)
    print_stats(merged_df, "병합 후 NLI 쌍 통계")

    # ── 5. 저장 ─────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(NLI_OUT), exist_ok=True)
    os.makedirs(os.path.dirname(PROCESSED_OUT), exist_ok=True)

    merged_df.to_csv(NLI_OUT, index=False, encoding="utf-8-sig")
    merged_df.to_csv(PROCESSED_OUT, index=False, encoding="utf-8-sig")

    print(f"\n  저장 완료:")
    print(f"    data/nli/nli_pairs.csv       → {len(merged_df)}행")
    print(f"    data/processed/nli_pairs.csv → {len(merged_df)}행 (동일)")

    # ── 6. 위기 발화 전체 목록 JSON 저장 (참고용) ────────────────────────────────
    crisis_out = os.path.join(BASE_DIR, "data", "nli", "crisis_utterances_aihub.json")
    with open(crisis_out, "w", encoding="utf-8") as f:
        json.dump(crisis_texts, f, ensure_ascii=False, indent=2)
    print(f"    data/nli/crisis_utterances_aihub.json → {len(crisis_texts)}개 위기 발화")

    print("\n" + "=" * 60)
    print("전처리 완료")
    print("=" * 60)


if __name__ == "__main__":
    main()
