"""
preprocess_emotion.py
역할: 한국어 연속적 대화 데이터셋 전처리
입력: data/raw/한국어_연속적_대화_데이터셋.xlsx
출력: data/processed/emotion_train.csv, emotion_val.csv, emotion_calib.csv

변경 이력:
  v2 — dialog_id 단위 분리로 교체 (동일 대화가 여러 split에 섞이는 누수 방지)
       다운샘플링·증강은 train에만 적용, val/calib은 원본 그대로 유지
       split 내 텍스트 중복 제거 (짧은 반복 발화 오염 방지)
"""

import os
import random
import time
import pandas as pd
from sklearn.model_selection import train_test_split

# 재현성 고정
SEED = 42
random.seed(SEED)

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_PATH    = os.path.join(BASE_DIR, "data", "raw", "한국어_연속적_대화_데이터셋.xlsx")
OUT_DIR     = os.path.join(BASE_DIR, "data", "processed")
BT_CACHE    = os.path.join(OUT_DIR, "bt_cache.csv")   # Back-translation 결과 캐시
os.makedirs(OUT_DIR, exist_ok=True)

# ── 유효 감정 레이블 (7클래스) ────────────────────────────────────────────────
VALID_EMOTIONS = {"행복", "중립", "슬픔", "공포", "혐오", "분노", "놀람"}
LABEL2ID = {"행복": 0, "중립": 1, "슬픔": 2, "공포": 3, "혐오": 4, "분노": 5, "놀람": 6}
RARE_EMOTIONS = ["공포", "혐오"]
SPLIT_TRIALS = 30


# ────────────────────────────────────────────────────────────────────────────────
# 1단계: 원본 데이터 로드 및 정제
# ────────────────────────────────────────────────────────────────────────────────
def load_and_clean(path: str) -> pd.DataFrame:
    """
    역할: xlsx 로드 후 노이즈 레이블·NaN 텍스트 제거. dialog_id 보존.
    입력: xlsx 파일 경로
    출력: dialog_id, text, emotion, label 컬럼 DataFrame
    """
    # 상단 2행은 메타 정보(클래스 수, 컬럼명) → skip
    raw = pd.read_excel(path, header=None, skiprows=2)
    raw.columns = ["dialog_id", "text", "emotion"] + [f"_c{i}" for i in range(9)]

    # 첫 열은 고유 dialog_id가 아니라 대화 시작 마커이므로
    # 비어 있지 않은 행을 기준으로 누적 번호를 새 dialog_id로 생성함
    df = raw[["dialog_id", "text", "emotion"]].copy()
    df["dialog_id"] = raw["dialog_id"].notna().cumsum()

    # NaN 텍스트 제거
    before = len(df)
    df = df.dropna(subset=["text"])

    # 노이즈 레이블 제거 (유효 7클래스 외 모두 제거)
    df = df[df["emotion"].isin(VALID_EMOTIONS)]
    after = len(df)
    print(f"[정제] {before} → {after} 행 (제거: {before - after})")

    df["label"] = df["emotion"].map(LABEL2ID)
    df = df.reset_index(drop=True)
    return df


def remove_ambiguous_texts(df: pd.DataFrame) -> pd.DataFrame:
    """
    역할: 동일 텍스트에 서로 다른 감정 레이블이 붙은 모호 샘플을 제거
    입력: dialog_id, text, emotion, label 컬럼 DataFrame
    출력: 모호 텍스트가 제거된 DataFrame
    """
    label_counts = df.groupby("text")["label"].nunique()
    ambiguous_texts = label_counts[label_counts > 1].index

    before = len(df)
    df = df[~df["text"].isin(ambiguous_texts)].reset_index(drop=True)
    after = len(df)
    print(
        f"[모호 텍스트 제거] {len(ambiguous_texts)}종 / "
        f"{before} → {after} 행 (제거: {before - after})"
    )
    return df


# ────────────────────────────────────────────────────────────────────────────────
# 2단계: dialog_id 단위로 train / val / calib 분리
# ────────────────────────────────────────────────────────────────────────────────
def _summarize_eval_split(df: pd.DataFrame) -> dict:
    """
    역할: 평가용 split의 클래스 분포 요약
    입력: split DataFrame
    출력: 클래스별 개수 dict
    """
    counts = df["emotion"].value_counts().to_dict()
    return {emotion: int(counts.get(emotion, 0)) for emotion in VALID_EMOTIONS}


def split_by_dialog(df: pd.DataFrame):
    """
    역할: dialog_id 단위 분리 — 같은 대화가 여러 split에 섞이는 누수 방지
          각 dialog의 대표 감정(최빈값)을 stratify 기준으로 사용
    입력: dialog_id, text, emotion, label 컬럼 DataFrame
    출력: (train_df, val_df, calib_df) — dialog_id 컬럼 포함
    """
    # dialog_id별 대표 감정 레이블 (최빈값) 계산
    dialog_dominant = (
        df.groupby("dialog_id")["label"]
        .agg(lambda x: x.value_counts().index[0])
        .reset_index()
        .rename(columns={"label": "dominant_label"})
    )

    best_score = None
    best_split = None

    for trial in range(SPLIT_TRIALS):
        trial_seed = SEED + trial

        # 여러 시드를 시도해 희소 클래스가 val/calib에 최대한 남는 split을 선택함
        train_ids, temp_ids = train_test_split(
            dialog_dominant["dialog_id"],
            test_size=0.30,
            random_state=trial_seed,
            stratify=dialog_dominant["dominant_label"],
        )

        temp_dominant = dialog_dominant[dialog_dominant["dialog_id"].isin(temp_ids)]
        val_ids, calib_ids = train_test_split(
            temp_dominant["dialog_id"],
            test_size=0.50,
            random_state=trial_seed,
            stratify=temp_dominant["dominant_label"],
        )

        train_df = df[df["dialog_id"].isin(train_ids)].copy().reset_index(drop=True)
        val_df = df[df["dialog_id"].isin(val_ids)].copy().reset_index(drop=True)
        calib_df = df[df["dialog_id"].isin(calib_ids)].copy().reset_index(drop=True)

        # 평가 split 품질만 미리 점수화하고, 실제 train 증강 전의 상태에서 최적 조합을 선택함
        val_counts = _summarize_eval_split(val_df)
        calib_counts = _summarize_eval_split(calib_df)
        rare_floor = min(
            min(val_counts[emotion], calib_counts[emotion]) for emotion in RARE_EMOTIONS
        )
        rare_total = sum(val_counts[emotion] + calib_counts[emotion] for emotion in RARE_EMOTIONS)
        score = (rare_floor, rare_total)

        if best_score is None or score > best_score:
            best_score = score
            best_split = (train_df, val_df, calib_df, trial_seed, val_counts, calib_counts)

    train_df, val_df, calib_df, best_seed, val_counts, calib_counts = best_split
    print(f"[split 선택] trial {SPLIT_TRIALS}회 중 seed={best_seed} 선택")
    print(f"[split 선택] val 희소 클래스 분포: 공포={val_counts['공포']}, 혐오={val_counts['혐오']}")
    print(
        f"[split 선택] calib 희소 클래스 분포: 공포={calib_counts['공포']}, 혐오={calib_counts['혐오']}"
    )
    return train_df, val_df, calib_df


# ────────────────────────────────────────────────────────────────────────────────
# 3단계: 중립 다운샘플링 (train에만 적용)
# ────────────────────────────────────────────────────────────────────────────────
def downsample_neutral(df: pd.DataFrame, n: int = 6500) -> pd.DataFrame:
    """
    역할: 중립 클래스를 n개로 다운샘플링
    입력: DataFrame, 목표 샘플 수
    출력: 다운샘플링된 DataFrame
    """
    neutral_count = len(df[df["emotion"] == "중립"])
    target_n = min(n, neutral_count)
    neutral = df[df["emotion"] == "중립"].sample(n=target_n, random_state=SEED)
    others  = df[df["emotion"] != "중립"]
    result  = pd.concat([neutral, others], ignore_index=True)
    print(f"[다운샘플링] 중립: {neutral_count} → {target_n}")
    return result


# ────────────────────────────────────────────────────────────────────────────────
# 4단계: 소수 클래스 증강 (train에만 적용 / KoEDA 실패 시 랜덤 삭제 폴백)
# ────────────────────────────────────────────────────────────────────────────────
def _augment_text_koeda(text: str, n_aug: int) -> list:
    """
    역할: KoEDA로 텍스트 증강 (AEDA: 구두점 삽입 방식)
    입력: 원본 텍스트, 생성할 변형 수
    출력: 증강된 텍스트 리스트
    """
    try:
        from koeda import AEDA
        aeda = AEDA(morpheme_analyzer="Okt", punc_ratio=0.3)
        results = []
        for _ in range(n_aug):
            augmented = aeda(text, p=0.3)
            if isinstance(augmented, list):
                results.append(augmented[0])
            else:
                results.append(augmented)
        return results
    except Exception:
        pass

    # 폴백: 랜덤 단어 삭제 (KoEDA 실패 시)
    tokens = text.split()
    results = []
    for _ in range(n_aug):
        if len(tokens) > 2:
            drop_idx = random.randint(0, len(tokens) - 1)
            new_tokens = tokens[:drop_idx] + tokens[drop_idx + 1:]
            results.append(" ".join(new_tokens))
        else:
            results.append(text)
    return results


def augment_minority(df: pd.DataFrame, target_count: int = 500) -> pd.DataFrame:
    """
    역할: 혐오·공포 클래스를 target_count 이상으로 증강 (train에만 적용)
          증강 행은 dialog_id=None으로 추가됨
    입력: DataFrame, 목표 샘플 수
    출력: 증강된 DataFrame
    """
    minority_classes = ["혐오", "공포"]
    aug_rows = []

    for cls in minority_classes:
        cls_df  = df[df["emotion"] == cls]
        current = len(cls_df)
        needed  = target_count - current
        if needed <= 0:
            print(f"[증강 불필요] {cls}: {current}개 (목표 {target_count})")
            continue

        print(f"[증강 시작] {cls}: {current} → 목표 {target_count} (추가 필요: {needed})")

        source_texts = cls_df["text"].tolist()
        generated = 0
        attempt   = 0
        while generated < needed and attempt < needed * 3:
            src = random.choice(source_texts)
            augmented = _augment_text_koeda(src, n_aug=1)
            for aug_text in augmented:
                if aug_text and aug_text != src:
                    aug_rows.append({
                        "dialog_id": None,          # 증강 행은 dialog_id 없음
                        "text":      aug_text,
                        "emotion":   cls,
                        "label":     LABEL2ID[cls],
                    })
                    generated += 1
                    if generated >= needed:
                        break
            attempt += 1

        print(f"[증강 완료] {cls}: {current} + {generated} = {current + generated}")

    if aug_rows:
        aug_df = pd.DataFrame(aug_rows)
        df = pd.concat([df, aug_df], ignore_index=True)

    return df


# ────────────────────────────────────────────────────────────────────────────────
# 5단계: Back-translation 증강 (train에만 적용 / ko→en→ko)
# ────────────────────────────────────────────────────────────────────────────────
def _backtranslate_batch(texts: list, batch_size: int = 15, sleep_sec: float = 0.6) -> list:
    """
    역할: 텍스트 리스트를 ko→en→ko 번역 (deep_translator GoogleTranslator)
    입력: 텍스트 리스트, 배치 크기, 배치 간 대기 시간
    출력: 번역된 한국어 텍스트 리스트 (실패 항목은 빈 문자열)
    """
    try:
        from deep_translator import GoogleTranslator
    except ImportError:
        print("  [경고] deep_translator 미설치 — pip install deep-translator")
        return [""] * len(texts)

    results = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        try:
            en = GoogleTranslator(source="ko", target="en").translate_batch(chunk)
            ko = GoogleTranslator(source="en", target="ko").translate_batch(en)
            results.extend(ko)
        except Exception as e:
            print(f"  [번역 실패 배치 {i//batch_size}] {e}")
            results.extend([""] * len(chunk))
        if i + batch_size < len(texts):
            time.sleep(sleep_sec)

    return results


def augment_backtranslation(df: pd.DataFrame, target_count: int = 800) -> pd.DataFrame:
    """
    역할: 공포·혐오 클래스를 Back-translation(ko→en→ko)으로 target_count까지 증강
          bt_cache.csv가 있으면 API 없이 캐시에서 즉시 로드,
          없으면 번역 후 캐시에 저장 (train에만 적용)
    입력: DataFrame (KoEDA 증강 포함), 목표 샘플 수
    출력: Back-translation 행이 추가된 DataFrame
    """
    minority_classes = ["혐오", "공포"]

    # ── 캐시 히트: 저장된 BT 결과 재사용 ────────────────────────────────────────
    if os.path.exists(BT_CACHE):
        print(f"[BT 캐시 로드] {BT_CACHE}")
        cached = pd.read_csv(BT_CACHE, encoding="utf-8-sig")
        required_cols = {"text", "emotion", "label"}

        # 캐시 파일 형식이 깨졌거나 비어 있으면 잘못된 증강을 막기 위해 무시함
        if cached.empty:
            print("[BT 캐시 무시] 빈 캐시 파일이라 재사용하지 않음")
        elif not required_cols.issubset(set(cached.columns)):
            print("[BT 캐시 무시] 필수 컬럼(text, emotion, label)이 없어 재사용하지 않음")
        else:
            cached = cached[cached["emotion"].isin(minority_classes)].copy().reset_index(drop=True)
            if cached.empty:
                print("[BT 캐시 무시] 공포/혐오 증강 행이 없어 재사용하지 않음")
            else:
                cached["dialog_id"] = None  # 증강 행 표시
                df = pd.concat([df, cached], ignore_index=True)
                for cls in minority_classes:
                    cnt = len(df[df["emotion"] == cls])
                    print(f"  {cls}: 캐시 로드 후 {cnt}개")
                return df

    # ── 캐시 미스: Google Translate API 호출 후 저장 ────────────────────────────
    aug_rows = []

    for cls in minority_classes:
        cls_df  = df[df["emotion"] == cls]
        current = len(cls_df)
        needed  = target_count - current
        if needed <= 0:
            print(f"[BT 증강 불필요] {cls}: {current}개 (목표 {target_count})")
            continue

        print(f"[BT 증강 시작] {cls}: {current} → 목표 {target_count} (추가 필요: {needed})")

        # 원본 텍스트에서만 Back-translation (dialog_id가 있는 행 = 원본)
        source_texts = cls_df[cls_df["dialog_id"].notna()]["text"].tolist()
        if not source_texts:
            source_texts = cls_df["text"].tolist()

        # 필요한 수만큼 소스 샘플링 (중복 허용)
        random.seed(SEED)
        sample_texts = [random.choice(source_texts) for _ in range(needed)]

        translated = _backtranslate_batch(sample_texts)

        added = 0
        original_set = set(cls_df["text"].tolist())
        for orig, trans in zip(sample_texts, translated):
            if trans and trans != orig and trans not in original_set:
                aug_rows.append({
                    "text":    trans,
                    "emotion": cls,
                    "label":   LABEL2ID[cls],
                })
                original_set.add(trans)
                added += 1

        print(f"[BT 증강 완료] {cls}: {current} + {added} = {current + added}")

    if aug_rows:
        cache_df = pd.DataFrame(aug_rows)
        cache_df.to_csv(BT_CACHE, index=False, encoding="utf-8-sig")
        print(f"[BT 캐시 저장] {BT_CACHE} ({len(cache_df)}행)")

        aug_df = cache_df.copy()
        aug_df["dialog_id"] = None
        df = pd.concat([df, aug_df], ignore_index=True)
    else:
        print("[BT 캐시 저장 생략] 새 Back-translation 결과가 없어 캐시를 만들지 않음")

    return df


# ────────────────────────────────────────────────────────────────────────────────
# 6단계: split 내 텍스트 중복 제거
# ────────────────────────────────────────────────────────────────────────────────
def dedup_within_split(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """
    역할: 동일 텍스트 중복 행 제거 (짧은 반복 발화 평가 오염 방지)
    입력: split DataFrame, split 이름 (로그용)
    출력: 중복 제거된 DataFrame
    """
    before = len(df)
    # 동일 문장·동일 감정만 중복 제거해, 남아 있는 합법적 다중 라벨 문장을 임의로 지우지 않음
    df = df.drop_duplicates(subset=["text", "emotion"]).reset_index(drop=True)
    after = len(df)
    print(f"[중복 제거 {split_name}] {before} → {after} 행 (제거: {before - after})")
    return df


def remove_cross_split_overlap(
    train_df: pd.DataFrame, val_df: pd.DataFrame, calib_df: pd.DataFrame
):
    """
    역할: split 간 동일 텍스트 누수 제거
          train 우선 유지, val은 train과 겹치지 않게 정리,
          calib은 train/val과 모두 겹치지 않게 정리
    입력: train, val, calib DataFrame
    출력: 누수 제거 후 (train_df, val_df, calib_df)
    """
    train_texts = set(train_df["text"])

    val_before = len(val_df)
    val_df = val_df[~val_df["text"].isin(train_texts)].reset_index(drop=True)
    val_after = len(val_df)

    blocked_texts = train_texts | set(val_df["text"])
    calib_before = len(calib_df)
    calib_df = calib_df[~calib_df["text"].isin(blocked_texts)].reset_index(drop=True)
    calib_after = len(calib_df)

    print(f"[split 간 누수 제거] val: {val_before} → {val_after} 행 (제거: {val_before - val_after})")
    print(f"[split 간 누수 제거] calib: {calib_before} → {calib_after} 행 (제거: {calib_before - calib_after})")

    return train_df, val_df, calib_df


# ────────────────────────────────────────────────────────────────────────────────
# 메인 실행
# ────────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("감정 분류 데이터셋 전처리 시작")
    print("=" * 60)

    # 1. 로드 + 정제 (dialog_id 보존)
    df = load_and_clean(RAW_PATH)

    # 1-1. 동일 텍스트-상이 라벨 모호 샘플 제거
    df = remove_ambiguous_texts(df)

    print("\n[정제 후 클래스 분포]")
    print(df["emotion"].value_counts().to_string())
    print(f"\n[고유 dialog_id 수] {df['dialog_id'].nunique():,}개")

    # 2. dialog_id 단위로 split 먼저 분리
    print("\n[dialog_id 단위 split 분리]")
    train_df, val_df, calib_df = split_by_dialog(df)
    print(f"  split 전 행 수  — train: {len(train_df):,} / val: {len(val_df):,} / calib: {len(calib_df):,}")

    # 3. train에만 중립 다운샘플링
    print("\n[train 중립 다운샘플링]")
    # 2026-04-27 [P1-A] train ↔ val 분포 갭 축소: 3500 → 6500
    # val/calib는 자연 분포 그대로(중립 80%)인데 train만 중립 29%로 다운샘플링되어
    # 모델이 "중립을 보수적으로 예측" 학습 → 평가 시 중립 누설 누적(2,606건).
    # 6500은 train 중립 비율 약 45%로 자연 분포에 한 발 가깝게 조정한 값.
    train_df = downsample_neutral(train_df, n=6500)

    # 4. train에만 소수 클래스 증강 — KoEDA (500개까지)
    print("\n[train KoEDA 증강]")
    train_df = augment_minority(train_df, target_count=500)

    # 5. Back-translation 증강 — KoEDA 이후 추가 (800개까지)
    print("\n[train Back-translation 증강 (ko→en→ko)]")
    train_df = augment_backtranslation(train_df, target_count=800)

    # 6. 각 split 내 텍스트 중복 제거
    print("\n[split 내 중복 제거]")
    train_df = dedup_within_split(train_df, "train")
    val_df   = dedup_within_split(val_df,   "val")
    calib_df = dedup_within_split(calib_df, "calib")

    # 7. split 간 동일 텍스트 누수 제거
    print("\n[split 간 누수 제거]")
    train_df, val_df, calib_df = remove_cross_split_overlap(train_df, val_df, calib_df)

    # 8. 최종 분포 출력
    print("\n[증강 후 train 클래스 분포]")
    print(train_df["emotion"].value_counts().to_string())

    print(f"\n[최종 분리 결과]")
    print(f"  train : {len(train_df):,}행")
    print(f"  val   : {len(val_df):,}행")
    print(f"  calib : {len(calib_df):,}행")
    print(f"  합계  : {len(train_df) + len(val_df) + len(calib_df):,}행")

    # split 간 텍스트 누수 검증
    train_texts = set(train_df["text"])
    val_texts = set(val_df["text"])
    calib_texts = set(calib_df["text"])
    val_overlap   = len(val_texts & train_texts)
    calib_overlap = len(calib_texts & train_texts)
    val_calib_overlap = len(val_texts & calib_texts)
    print(
        f"\n[누수 검증] train↔val 중복: {val_overlap}건 / "
        f"train↔calib 중복: {calib_overlap}건 / val↔calib 중복: {val_calib_overlap}건"
    )

    # 9. 저장 (dialog_id 제외, text·emotion·label만 저장)
    save_cols = ["text", "emotion", "label"]
    train_df[save_cols].to_csv(
        os.path.join(OUT_DIR, "emotion_train.csv"), index=False, encoding="utf-8-sig"
    )
    val_df[save_cols].to_csv(
        os.path.join(OUT_DIR, "emotion_val.csv"),   index=False, encoding="utf-8-sig"
    )
    calib_df[save_cols].to_csv(
        os.path.join(OUT_DIR, "emotion_calib.csv"), index=False, encoding="utf-8-sig"
    )

    print("\n[저장 완료]")
    print(f"  → {OUT_DIR}/emotion_train.csv")
    print(f"  → {OUT_DIR}/emotion_val.csv")
    print(f"  → {OUT_DIR}/emotion_calib.csv")
    print("=" * 60)


if __name__ == "__main__":
    main()
