"""
build_semantic_emotion_dataset.py
역할: Semantic Emotion Judge Phase 2용 7감정 + distress weak-label 학습 CSV를 생성한다.
입력:
  - data/processed/emotion_train.csv
  - data/processed/emotion_val_clean.csv
  - data/processed/emotion_calib_clean.csv
  - data/raw/웰니스_대화_스크립트_데이터셋.xlsx
  - data/raw/02)웰니스_대화_스크립트_데이터셋.xlsx
  - data/raw/16.심리상담 데이터/1.데이터/*/02.라벨링데이터/*.json
출력:
  - data/processed/semantic_emotion_train.csv
  - data/processed/semantic_emotion_val.csv
  - data/processed/semantic_emotion_calib.csv
  - data/processed/semantic_emotion_label_map.json
실행:
  C:/Users/WD/anaconda3/envs/dl_study/python.exe data/preprocess/build_semantic_emotion_dataset.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RAW_DIR = BASE_DIR / "data" / "raw"
AIHUB_LABEL_ROOT = RAW_DIR / "16.심리상담 데이터" / "1.데이터"

EMOTION_SPLITS = {
    "train": PROCESSED_DIR / "emotion_train.csv",
    "val": PROCESSED_DIR / "emotion_val_clean.csv",
    "calib": PROCESSED_DIR / "emotion_calib_clean.csv",
}
WELLNESS01_PATH = RAW_DIR / "웰니스_대화_스크립트_데이터셋.xlsx"
WELLNESS02_PATH = RAW_DIR / "02)웰니스_대화_스크립트_데이터셋.xlsx"

OUT_PATHS = {
    "train": PROCESSED_DIR / "semantic_emotion_train.csv",
    "val": PROCESSED_DIR / "semantic_emotion_val.csv",
    "calib": PROCESSED_DIR / "semantic_emotion_calib.csv",
}
LABEL_MAP_OUT = PROCESSED_DIR / "semantic_emotion_label_map.json"

EMOTIONS = ["행복", "중립", "슬픔", "공포", "혐오", "분노", "놀람"]
EMOTION_TO_LABEL = {emotion: idx for idx, emotion in enumerate(EMOTIONS)}
LABEL_TO_EMOTION = {str(idx): emotion for emotion, idx in EMOTION_TO_LABEL.items()}

SEED = 42
WEAK_SPLIT_RATIOS = {"train": 0.80, "val": 0.10, "calib": 0.10}
ORIGINAL_SPLIT_PRIORITY = {"val": 0, "calib": 1, "train": 2}
ORIGINAL_DEDUP_POLICY = (
    "원본 감정 split의 동일 텍스트 키는 val→calib→train 우선순위로 1건만 보존하고, "
    "emotion 라벨이 상충하는 키는 보수적으로 전부 제외한다."
)
AIHUB_MAX_PER_CLASS = {
    "ANXIETY": 4500,
    "DEPRESSION": 4500,
    "NORMAL": 3000,
    "ADDICTION": 1500,
}

OUTPUT_COLUMNS = [
    "text",
    "emotion",
    "label",
    "primary_emotion",
    "distress_level",
    "situation_tag",
    "label_source",
    "weak_label_confidence",
    "sample_weight",
    "is_weak_label",
    "source_dataset",
    "source_detail",
    "source_split",
]

WELLNESS02_UTTERANCE_COLUMNS = [
    "utterance",
    "utterance(2차)",
    "utterance(긍정)",
    "utterance(부정)",
    "추가발화(190917)",
    "추가발화 (191031)",
]

ANON_PATTERN = re.compile(r"@\S+")
SPACE_PATTERN = re.compile(r"\s+")

CRISIS_KEYWORDS = [
    "자살",
    "자해",
    "죽고 싶",
    "죽고싶",
    "죽어버",
    "살기 싫",
    "살고 싶지",
    "사라지고 싶",
    "없어지고 싶",
    "해치고 싶",
    "목숨",
]

NON_CRISIS_IDIOMS = [
    "배고파 죽",
    "더워 죽",
    "웃겨 죽",
    "졸려 죽",
    "귀여워 죽",
    "맛있어 죽",
    "좋아 죽",
    "재밌어 죽",
    "행복해 죽",
]

POSITIVE_KEYWORDS = ["기쁘", "좋아", "행복", "안도", "고마", "감사", "뿌듯", "해냈", "괜찮아졌", "나아졌"]
ANGER_KEYWORDS = ["화", "분노", "짜증", "억울", "열받", "빡", "미워", "원망", "신경질", "욱"]
FEAR_KEYWORDS = ["불안", "걱정", "긴장", "초조", "무서", "두려", "공포", "떨", "울렁", "숨 막", "심장", "두근"]
SAD_KEYWORDS = ["우울", "슬프", "외로", "힘들", "괴로", "눈물", "절망", "무기력", "후회", "자괴", "허무", "상실"]
DISGUST_KEYWORDS = ["혐오", "역겨", "불쾌", "구역질", "징그러"]
SURPRISE_KEYWORDS = ["놀라", "충격", "당황", "깜짝", "멘붕"]

ACADEMIC_KEYWORDS = ["시험", "면접", "발표", "과제", "마감", "평가", "성적", "합격", "불합격"]
WORK_KEYWORDS = ["회사", "직장", "출근", "퇴근", "회의", "상사", "동료", "업무", "프로젝트"]
RELATIONSHIP_KEYWORDS = ["친구", "남자친구", "여자친구", "애인", "연락", "이별", "부모", "남편", "아내", "자녀"]
HEALTH_KEYWORDS = ["잠", "불면", "식욕", "두통", "어지러", "호흡", "심장", "피로", "아파", "건강"]

AIHUB_DEPRESSION_LABELS = {
    "depressive_mood",
    "worthlessness",
    "guilt",
    "anhedonia",
    "fatigue",
    "negative_self-image",
    "motivation_for_change",
    "psychomotor_changes",
    "weight_appetite",
    "sleep_disturbance",
}
AIHUB_ANXIETY_LABELS = {
    "trauma_experience",
    "stressful_event",
    "emotional_requlation",
    "loss_of_control",
    "underlying_physical_condition",
}
AIHUB_CRISIS_LABELS = {"suicidal"}


@dataclass(frozen=True)
class SemanticRecord:
    """
    역할: semantic emotion CSV 한 행을 표현한다.
    입력: 텍스트, 감정 라벨, distress, 출처 메타데이터
    출력: pandas 저장 가능한 dict로 변환 가능한 불변 레코드
    """

    text: str
    emotion: str
    distress_level: int
    situation_tag: str
    label_source: str
    weak_label_confidence: float
    source_dataset: str
    source_detail: str
    source_split: str = ""

    def to_row(self) -> dict[str, Any]:
        """
        역할: 레코드를 CSV 저장용 dict로 변환한다.
        입력: 없음
        출력: OUTPUT_COLUMNS에 맞춘 dict
        """
        label = EMOTION_TO_LABEL[self.emotion]
        confidence = round(float(self.weak_label_confidence), 4)
        is_weak = confidence < 0.999
        return {
            "text": self.text,
            "emotion": self.emotion,
            "label": label,
            "primary_emotion": self.emotion,
            "distress_level": int(self.distress_level),
            "situation_tag": self.situation_tag,
            "label_source": self.label_source,
            "weak_label_confidence": confidence,
            "sample_weight": confidence,
            "is_weak_label": bool(is_weak),
            "source_dataset": self.source_dataset,
            "source_detail": self.source_detail,
            "source_split": self.source_split,
        }


def normalize_text(value: Any) -> str:
    """
    역할: 원천 셀/JSON 값을 학습 CSV용 짧은 문자열로 정규화한다.
    입력: 임의의 값
    출력: 익명화 태그와 중복 공백이 정리된 문자열
    """
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    text = ANON_PATTERN.sub("", text)
    text = text.replace("\u3000", " ")
    text = SPACE_PATTERN.sub(" ", text).strip()
    return text


def compact_text_key(text: str) -> str:
    """
    역할: split 간 중복 제거에 사용할 텍스트 키를 만든다.
    입력: 정규화된 텍스트
    출력: 공백을 제거한 소문자 키
    """
    return SPACE_PATTERN.sub("", text).lower()


def stable_hash(value: str) -> int:
    """
    역할: 샘플링과 split 배정을 위한 재현 가능한 해시값을 계산한다.
    입력: 문자열 값
    출력: 64비트 정수 해시
    """
    payload = f"{SEED}:{value}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:16]
    return int(digest, 16)


def assign_weak_split(text: str, source: str) -> str:
    """
    역할: weak-label 샘플을 train/val/calib로 deterministic split한다.
    입력: 텍스트와 출처 문자열
    출력: split 이름
    """
    bucket = stable_hash(f"{source}:{compact_text_key(text)}") % 10000
    train_cut = int(WEAK_SPLIT_RATIOS["train"] * 10000)
    val_cut = train_cut + int(WEAK_SPLIT_RATIOS["val"] * 10000)
    if bucket < train_cut:
        return "train"
    if bucket < val_cut:
        return "val"
    return "calib"


def is_usable_text(text: str) -> bool:
    """
    역할: 문맥 없이 감정 후보로 쓰기 어려운 너무 짧거나 긴 발화를 걸러낸다.
    입력: 정규화된 텍스트
    출력: 사용 가능 여부
    """
    if len(text) < 5 or len(text) > 220:
        return False
    if text in {"네", "아니요", "응", "어", "아니", "그래", "그렇죠"}:
        return False
    return True


def contains_any(text: str, keywords: list[str] | set[str]) -> bool:
    """
    역할: 텍스트에 지정 키워드 중 하나가 포함되는지 확인한다.
    입력: 텍스트, 키워드 목록
    출력: 포함 여부
    """
    compact = compact_text_key(text)
    return any(compact_text_key(keyword) in compact for keyword in keywords)


def infer_situation_tag(text: str, source_detail: str, emotion: str, distress_level: int) -> str:
    """
    역할: 텍스트와 출처 정보를 바탕으로 상황 태그를 추정한다.
    입력: 발화 텍스트, 출처 상세 문자열, 감정 라벨, distress level
    출력: 세미콜론으로 묶은 상황 태그 문자열
    """
    context = f"{text} {source_detail}"
    tags = []
    if distress_level >= 4 or contains_any(context, CRISIS_KEYWORDS):
        tags.append("crisis")
    if contains_any(context, ACADEMIC_KEYWORDS):
        tags.append("academic")
    if contains_any(context, WORK_KEYWORDS):
        tags.append("work")
    if contains_any(context, RELATIONSHIP_KEYWORDS):
        tags.append("relationship")
    if contains_any(context, HEALTH_KEYWORDS):
        tags.append("health")
    if emotion == "공포":
        tags.append("anxiety")
    elif emotion == "슬픔":
        tags.append("depressive")
    elif emotion == "분노":
        tags.append("anger")
    elif emotion == "행복":
        tags.append("positive")
    elif emotion == "중립":
        tags.append("neutral_hard_negative")
    elif emotion == "혐오":
        tags.append("disgust")
    elif emotion == "놀람":
        tags.append("surprise")
    return ";".join(dict.fromkeys(tags)) or "routine"


def infer_emotion_from_text(text: str, fallback: str) -> tuple[str, float]:
    """
    역할: 텍스트 표면 단서로 weak 감정 라벨을 보수적으로 보정한다.
    입력: 발화 텍스트, 기본 감정 라벨
    출력: 추정 감정과 confidence 보정값
    """
    if contains_any(text, NON_CRISIS_IDIOMS):
        return "중립", 0.46
    if contains_any(text, CRISIS_KEYWORDS):
        return "슬픔", 0.58
    keyword_rules = [
        ("행복", POSITIVE_KEYWORDS, 0.58),
        ("분노", ANGER_KEYWORDS, 0.60),
        ("공포", FEAR_KEYWORDS, 0.60),
        ("슬픔", SAD_KEYWORDS, 0.60),
        ("혐오", DISGUST_KEYWORDS, 0.55),
        ("놀람", SURPRISE_KEYWORDS, 0.54),
    ]
    for emotion, keywords, confidence in keyword_rules:
        if contains_any(text, keywords):
            return emotion, confidence
    return fallback, 0.0


def infer_distress_level(text: str, emotion: str, source_detail: str, base_level: int) -> int:
    """
    역할: 텍스트/출처/감정 라벨을 이용해 distress level 0~4를 추정한다.
    입력: 발화 텍스트, 감정, 출처 상세, 기본 level
    출력: 0~4 정수 distress level
    """
    context = f"{text} {source_detail}"
    level = int(base_level)
    if contains_any(context, CRISIS_KEYWORDS):
        level = max(level, 4)
    if contains_any(context, ["절망", "죽음공포", "공황", "호흡곤란", "자해", "자살충동"]):
        level = max(level, 3)
    if contains_any(context, ["불면", "무기력", "우울", "불안", "초조", "죄책감", "분노", "괴로움"]):
        level = max(level, 2)
    if emotion in {"공포", "슬픔", "분노"}:
        level = max(level, 1)
    if emotion in {"행복", "중립"} and not contains_any(context, CRISIS_KEYWORDS):
        level = min(level, 1)
    return max(0, min(level, 4))


def make_record(
    text: str,
    emotion: str,
    distress_level: int,
    label_source: str,
    confidence: float,
    source_dataset: str,
    source_detail: str,
    source_split: str = "",
) -> SemanticRecord | None:
    """
    역할: 검증과 보정 후 SemanticRecord를 만든다.
    입력: 텍스트, 감정, distress, 출처 메타데이터
    출력: 유효하면 SemanticRecord, 아니면 None
    """
    clean_text = normalize_text(text)
    if not is_usable_text(clean_text):
        return None
    if emotion not in EMOTION_TO_LABEL:
        return None
    distress = infer_distress_level(clean_text, emotion, source_detail, distress_level)
    situation_tag = infer_situation_tag(clean_text, source_detail, emotion, distress)
    return SemanticRecord(
        text=clean_text,
        emotion=emotion,
        distress_level=distress,
        situation_tag=situation_tag,
        label_source=label_source,
        weak_label_confidence=max(0.0, min(float(confidence), 1.0)),
        source_dataset=source_dataset,
        source_detail=source_detail,
        source_split=source_split,
    )


def load_original_emotion_records() -> dict[str, list[SemanticRecord]]:
    """
    역할: 기존 emotion train/val_clean/calib_clean을 정답 라벨 레코드로 읽는다.
    입력: 없음
    출력: split별 SemanticRecord 리스트
    """
    records_by_split: dict[str, list[SemanticRecord]] = {split: [] for split in EMOTION_SPLITS}
    for split, path in EMOTION_SPLITS.items():
        df = pd.read_csv(path, encoding="utf-8-sig")
        for _, row in df.iterrows():
            text = normalize_text(row["text"])
            emotion = normalize_text(row["emotion"])
            if not text or emotion not in EMOTION_TO_LABEL:
                continue
            distress = 0 if emotion in {"중립", "행복"} else 1
            source_detail = f"original_split={split}"
            record = SemanticRecord(
                text=text,
                emotion=emotion,
                distress_level=distress,
                situation_tag=infer_situation_tag(text, source_detail, emotion, distress),
                label_source="original_emotion",
                weak_label_confidence=1.0,
                source_dataset=path.name,
                source_detail=source_detail,
                source_split=split,
            )
            records_by_split[split].append(record)
    return records_by_split


def deduplicate_original_records(
    records_by_split: dict[str, list[SemanticRecord]],
) -> tuple[dict[str, list[SemanticRecord]], dict[str, Any]]:
    """
    역할: 원본 감정 split 내부/간 동일 텍스트 중복과 상충 라벨을 정리한다.
    입력: split별 original_emotion SemanticRecord 리스트
    출력: 중복 정리된 split별 레코드와 감사용 통계 dict
    """
    grouped_records: dict[str, list[tuple[str, int, SemanticRecord]]] = defaultdict(list)
    input_rows = 0
    for split, records in records_by_split.items():
        for order, record in enumerate(records):
            grouped_records[compact_text_key(record.text)].append((split, order, record))
            input_rows += 1

    cleaned_by_split: dict[str, list[SemanticRecord]] = {split: [] for split in records_by_split}
    conflict_examples: list[dict[str, Any]] = []
    duplicate_text_keys = 0
    conflict_text_keys = 0
    conflict_rows_removed = 0
    removed_duplicate_rows = 0

    for text_key, items in grouped_records.items():
        if len(items) > 1:
            duplicate_text_keys += 1

        emotions = {record.emotion for _, _, record in items}
        if len(emotions) > 1:
            conflict_text_keys += 1
            conflict_rows_removed += len(items)
            if len(conflict_examples) < 20:
                conflict_examples.append(
                    {
                        "text_key": text_key,
                        "records": [
                            {
                                "split": split,
                                "text": record.text,
                                "emotion": record.emotion,
                                "source_dataset": record.source_dataset,
                            }
                            for split, _, record in items
                        ],
                    }
                )
            continue

        kept_split, _, kept_record = sorted(
            items,
            key=lambda item: (ORIGINAL_SPLIT_PRIORITY.get(item[0], 99), item[1]),
        )[0]
        cleaned_by_split[kept_split].append(kept_record)
        removed_duplicate_rows += len(items) - 1

    output_rows = sum(len(records) for records in cleaned_by_split.values())
    audit = {
        "policy": ORIGINAL_DEDUP_POLICY,
        "input_rows": int(input_rows),
        "output_rows": int(output_rows),
        "duplicate_text_keys": int(duplicate_text_keys),
        "same_label_duplicate_text_keys": int(duplicate_text_keys - conflict_text_keys),
        "removed_same_label_duplicate_rows": int(removed_duplicate_rows),
        "conflict_text_keys": int(conflict_text_keys),
        "conflict_rows_removed": int(conflict_rows_removed),
        "examples": conflict_examples,
    }
    return cleaned_by_split, audit


def map_wellness01_category(category: str, text: str) -> tuple[str, int, float] | None:
    """
    역할: 웰니스01 카테고리를 7감정 weak label로 매핑한다.
    입력: 웰니스01 구분값과 사용자 발화
    출력: 감정, distress level, confidence 또는 None
    """
    category = normalize_text(category)
    context = f"{category} {text}"
    text_emotion, text_conf = infer_emotion_from_text(text, "중립")

    if contains_any(context, ["자살충동", "자해", "자살시도"]):
        return "슬픔", 4, 0.58
    if contains_any(context, ["불안", "걱정", "긴장", "초조", "공황", "무서움", "죽음공포", "호흡곤란", "두근거림"]):
        return "공포", 2, max(0.62, text_conf)
    if contains_any(context, ["우울", "힘듦", "눈물", "슬픔", "외로움", "절망", "무기력", "의욕상실", "자존감", "자괴", "괴로움", "후회", "서운"]):
        return "슬픔", 2, max(0.62, text_conf)
    if contains_any(context, ["화", "분노", "짜증", "억울", "불만", "감정조절", "신경쓰임"]):
        return "분노", 2, max(0.62, text_conf)
    if contains_any(context, ["불쾌감", "혐오"]):
        return "혐오", 1, max(0.54, text_conf)
    if contains_any(context, ["충격"]):
        return "놀람", 1, max(0.52, text_conf)
    if text_emotion != "중립":
        return text_emotion, 1, max(0.50, text_conf)
    if contains_any(context, ["일반대화", "이상없음", "상태/호전", "상태/증상없음"]):
        return "중립", 0, 0.44
    return None


def load_wellness01_records() -> list[SemanticRecord]:
    """
    역할: 웰니스01 사용자 발화를 weak label 후보로 변환한다.
    입력: 없음
    출력: SemanticRecord 리스트
    """
    df = pd.read_excel(WELLNESS01_PATH)
    df.columns = [normalize_text(col) for col in df.columns]
    category_col = "구분"
    user_col = "유저"
    records = []
    for _, row in df.iterrows():
        text = normalize_text(row.get(user_col, ""))
        if not is_usable_text(text):
            continue
        category = normalize_text(row.get(category_col, ""))
        mapped = map_wellness01_category(category, text)
        if mapped is None:
            continue
        emotion, distress, confidence = mapped
        record = make_record(
            text=text,
            emotion=emotion,
            distress_level=distress,
            label_source="wellness_category",
            confidence=confidence,
            source_dataset=WELLNESS01_PATH.name,
            source_detail=f"category={category}",
        )
        if record:
            records.append(record)
    return records


def normalize_wellness02_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    역할: 웰니스02 컬럼명 공백을 정리하고 주요 계층 컬럼을 forward fill한다.
    입력: 원본 DataFrame
    출력: 정규화된 DataFrame
    """
    df = df.copy()
    df.columns = [normalize_text(col) for col in df.columns]
    fill_cols = ["핵심증상", "intent", "keyword(임상키워드)", "연관표현", "임상질문그룹(연세의료원제공)"]
    for col in fill_cols:
        if col in df.columns:
            df[col] = df[col].ffill()
    for col in df.columns:
        df[col] = df[col].map(normalize_text)
    return df


def map_wellness02_context(intent: str, keyword: str, utterance_col: str, text: str) -> tuple[str, int, float, str] | None:
    """
    역할: 웰니스02 intent/keyword/utterance를 3-tier 분리 weak label로 매핑한다.
    입력: intent, 임상 keyword, 발화 컬럼명, 발화 텍스트
    출력: 감정, distress level, confidence, label_source variant 또는 None
    설명 (Phase 3.2):
      - tier 1: 자살충동/해치다 — 안전 신호라 무조건 슬픔 distress=4 보존
      - tier 2: 텍스트 자체에 감정 키워드 존재(text_conf>0) — text_emotion 우선 채택, label_source=wellness02_text_explicit
      - tier 3: text는 무색이지만 intent가 distress 신호 — 중립 weak hard negative로 라우팅,
                emotion=중립/distress=intent 추정 1~2/conf=0.40, label_source=wellness02_context_only
        (이 분리가 academic_anxiety 등 source-aligned shortcut 제거의 핵심)
    """
    context = f"{intent} {keyword} {text}"
    text_emotion, text_conf = infer_emotion_from_text(text, "중립")
    base_conf = 0.58 if utterance_col in {"utterance", "utterance(2차)"} else 0.50

    # tier 1 — 안전 신호 보존 (text 무색이어도 유지)
    if contains_any(context, ["자살충동", "해치다", "충동"]):
        return "슬픔", 4, max(base_conf, 0.56, text_conf), "wellness02_intent"

    # tier 2 — 텍스트 자체에 감정 표현이 있으면 그것 우선
    if text_conf > 0:
        text_distress = 2 if text_emotion in {"공포", "슬픔", "분노"} else 1
        if text_emotion == "혐오":
            text_distress = 1
        return text_emotion, text_distress, max(base_conf, text_conf), "wellness02_text_explicit"

    # tier 3 — intent에 distress 신호 있지만 text 무색 → 중립 weak hard negative
    intent_has_distress_signal = contains_any(
        context,
        [
            "불안", "초조", "불면", "긴장", "두렵", "떨림", "울렁", "압박감", "수면장애",
            "우울", "슬픔", "무기력", "피로", "자존감", "죄책감", "외로움", "절망", "상실감", "허망", "한심", "활동감소",
            "분노", "욱함", "화", "짜증", "억울", "원망", "신경질", "갈등",
            "비웃음", "피해의식", "불쾌", "혐오",
            "감정기복", "종잡을수없음", "충격",
        ],
    )
    if intent_has_distress_signal:
        # distress level은 intent 신호로 약하게 1~2 카운트 (4는 tier 1만)
        intent_distress = 2 if contains_any(
            context, ["절망", "공황", "수면장애", "자존감저하", "피해의식", "원망"]
        ) else 1
        return "중립", intent_distress, 0.40, "wellness02_context_only"

    return None


def load_wellness02_records() -> list[SemanticRecord]:
    """
    역할: 웰니스02 사용자 발화를 forward fill 후 weak label 후보로 변환한다.
    입력: 없음
    출력: SemanticRecord 리스트
    """
    df = pd.read_excel(WELLNESS02_PATH, sheet_name="사용자 발화")
    df = normalize_wellness02_columns(df)
    records = []
    available_cols = [col for col in WELLNESS02_UTTERANCE_COLUMNS if col in df.columns]
    for _, row in df.iterrows():
        intent = normalize_text(row.get("intent", ""))
        keyword = normalize_text(row.get("keyword(임상키워드)", ""))
        for col in available_cols:
            text = normalize_text(row.get(col, ""))
            if not is_usable_text(text):
                continue
            mapped = map_wellness02_context(intent, keyword, col, text)
            if mapped is None:
                continue
            emotion, distress, confidence, source_variant = mapped
            source_detail = f"intent={intent};keyword={keyword};utterance_column={col}"
            record = make_record(
                text=text,
                emotion=emotion,
                distress_level=distress,
                label_source=source_variant,
                confidence=confidence,
                source_dataset=WELLNESS02_PATH.name,
                source_detail=source_detail,
            )
            if record:
                records.append(record)
    return records


def iter_aihub_label_files() -> list[Path]:
    """
    역할: AI Hub Training/Validation 라벨 JSON 경로를 수집한다.
    입력: 없음
    출력: JSON 파일 경로 리스트
    """
    files = []
    for split in ["Training", "Validation"]:
        label_dir = AIHUB_LABEL_ROOT / split / "02.라벨링데이터"
        files.extend(sorted(label_dir.rglob("*.json")))
    return files


def collect_active_paragraph_labels(paragraph: dict[str, Any]) -> set[str]:
    """
    역할: AI Hub paragraph에서 0보다 큰 증상 라벨명을 수집한다.
    입력: paragraph dict
    출력: 활성 라벨명 set
    """
    skip_keys = {
        "start_point",
        "end_point",
        "character_count",
        "cps",
        "paragraph_speaker",
        "paragraph_text",
        "index",
    }
    active = set()
    for key, value in paragraph.items():
        if key in skip_keys:
            continue
        if isinstance(value, (int, float)) and value > 0:
            active.add(str(key))
    return active


def map_aihub_paragraph(
    text: str,
    session_class: str,
    severity_prior: int,
    active_labels: set[str],
) -> tuple[str, int, float] | None:
    """
    역할: AI Hub 세션 prior와 paragraph 라벨을 발화 단위 weak label로 변환한다.
    입력: 내담자 발화, 세션 class, 세션 severity prior, paragraph 활성 라벨
    출력: 감정, distress level, confidence 또는 None
    """
    text_emotion, text_conf = infer_emotion_from_text(text, "중립")
    if active_labels & AIHUB_CRISIS_LABELS or contains_any(text, CRISIS_KEYWORDS):
        return "슬픔", 4, max(0.58, text_conf)
    if text_emotion != "중립" and text_conf >= 0.54:
        base_distress = 0 if text_emotion == "행복" else max(1, severity_prior)
        return text_emotion, base_distress, min(0.62, max(0.50, text_conf))
    if active_labels & AIHUB_DEPRESSION_LABELS:
        return "슬픔", max(2, severity_prior), 0.56
    if active_labels & AIHUB_ANXIETY_LABELS:
        return "공포", max(2, severity_prior), 0.52
    if session_class == "ANXIETY":
        return "공포", max(1, severity_prior), 0.46
    if session_class == "DEPRESSION":
        return "슬픔", max(1, severity_prior), 0.48
    if session_class == "NORMAL":
        return "중립", 0, 0.42
    return None


def should_keep_aihub_record(text: str, session_class: str, emotion: str) -> bool:
    """
    역할: AI Hub 후보 수가 과도해지지 않도록 class별 deterministic cap을 적용한다.
    입력: 텍스트, 세션 class, 감정
    출력: 보존 여부
    """
    max_count = AIHUB_MAX_PER_CLASS.get(session_class, 0)
    if max_count <= 0:
        return False
    class_bucket = stable_hash(f"aihub:{session_class}:{emotion}:{compact_text_key(text)}")
    return class_bucket % 100000 < max_count


def load_aihub_records() -> list[SemanticRecord]:
    """
    역할: AI Hub 라벨 JSON의 내담자 발화를 weak label 후보로 변환한다.
    입력: 없음
    출력: SemanticRecord 리스트
    """
    records = []
    skipped_json = 0
    for path in iter_aihub_label_files():
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            skipped_json += 1
            continue

        session_class = normalize_text(data.get("class", ""))
        severity_values = [
            int(data.get(field, 0) or 0)
            for field in ["depression", "anxiety", "addiction"]
            if isinstance(data.get(field, 0), (int, float))
        ]
        severity_prior = max(severity_values) if severity_values else 0
        split = "Training" if "\\Training\\" in str(path) or "/Training/" in str(path) else "Validation"

        for paragraph in data.get("paragraph", []):
            if normalize_text(paragraph.get("paragraph_speaker", "")) != "내담자":
                continue
            text = normalize_text(paragraph.get("paragraph_text", ""))
            if not is_usable_text(text):
                continue
            active_labels = collect_active_paragraph_labels(paragraph)
            mapped = map_aihub_paragraph(text, session_class, severity_prior, active_labels)
            if mapped is None:
                continue
            emotion, distress, confidence = mapped
            if not should_keep_aihub_record(text, session_class, emotion):
                continue
            detail_labels = ",".join(sorted(active_labels)) if active_labels else "none"
            source_detail = (
                f"class={session_class};severity_prior={severity_prior};"
                f"active_labels={detail_labels};json_split={split}"
            )
            record = make_record(
                text=text,
                emotion=emotion,
                distress_level=distress,
                label_source="aihub_session_prior",
                confidence=confidence,
                source_dataset="16.심리상담 데이터",
                source_detail=source_detail,
                source_split=split,
            )
            if record:
                records.append(record)

    if skipped_json:
        print(f"[AI Hub] 깨진 JSON {skipped_json}개 skip")
    return records


def deduplicate_weak_records(records: list[SemanticRecord]) -> list[SemanticRecord]:
    """
    역할: weak 후보끼리 중복 텍스트를 제거하고 confidence가 높은 후보를 우선 보존한다.
    입력: weak SemanticRecord 리스트
    출력: 중복 제거된 리스트
    """
    best_by_key: dict[str, SemanticRecord] = {}
    for record in records:
        key = compact_text_key(record.text)
        previous = best_by_key.get(key)
        if previous is None or record.weak_label_confidence > previous.weak_label_confidence:
            best_by_key[key] = record
    return sorted(best_by_key.values(), key=lambda item: (item.label_source, item.emotion, item.text))


def merge_records(
    original_by_split: dict[str, list[SemanticRecord]],
    weak_records: list[SemanticRecord],
) -> dict[str, pd.DataFrame]:
    """
    역할: 정리된 정답 감정 split과 weak-label 후보를 병합한다.
    입력: 기존 split별 레코드, weak 후보 레코드
    출력: split별 DataFrame
    """
    rows_by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in OUT_PATHS}
    seen_text_keys: set[str] = set()

    for split, records in original_by_split.items():
        for record in records:
            key = compact_text_key(record.text)
            seen_text_keys.add(key)
            rows_by_split[split].append(record.to_row())

    for record in deduplicate_weak_records(weak_records):
        key = compact_text_key(record.text)
        if key in seen_text_keys:
            continue
        split = assign_weak_split(record.text, record.label_source)
        seen_text_keys.add(key)
        row = record.to_row()
        row["source_split"] = row["source_split"] or split
        rows_by_split[split].append(row)

    return {
        split: pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
        for split, rows in rows_by_split.items()
    }


def build_label_map(
    frames: dict[str, pd.DataFrame],
    source_counts: Counter[str],
    original_dedup_audit: dict[str, Any],
) -> dict[str, Any]:
    """
    역할: 라벨/출처/confidence 정책과 생성 통계를 JSON으로 정리한다.
    입력: split별 DataFrame, 원천 후보 수 Counter, 원본 중복 정리 감사 통계
    출력: label map dict
    """
    split_summary = {}
    for split, df in frames.items():
        split_summary[split] = {
            "rows": int(len(df)),
            "emotion_distribution": {str(k): int(v) for k, v in df["emotion"].value_counts().items()},
            "label_source_distribution": {str(k): int(v) for k, v in df["label_source"].value_counts().items()},
            "weak_rows": int(df["is_weak_label"].sum()),
            "gold_rows": int((~df["is_weak_label"]).sum()),
        }

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_of_truth": "SEMANTIC_EMOTION_JUDGE_ROADMAP.md",
        "manual_scenario_eval_included": False,
        "manual_scenario_eval_policy": "eval/report/semantic_emotion_scenario_eval.csv는 평가 전용이며 이 빌더에서 읽지 않는다.",
        "emotion_label_map": EMOTION_TO_LABEL,
        "label_id_to_emotion": LABEL_TO_EMOTION,
        "distress_level_map": {
            "0": "calm_or_positive",
            "1": "mild_distress",
            "2": "moderate_distress",
            "3": "high_distress",
            "4": "crisis_candidate",
        },
        "columns": OUTPUT_COLUMNS,
        "split_policy": {
            "original_emotion": ORIGINAL_DEDUP_POLICY,
            "weak_label": f"텍스트 해시 기반 deterministic split {WEAK_SPLIT_RATIOS}",
            "deduplication": "원본 정답 split을 먼저 중복 정리하고, 이후 weak 후보는 텍스트 키 기준 1회만 사용",
        },
        "confidence_policy": {
            "original_emotion": 1.0,
            "wellness_category": "0.44~0.62, 웰니스 카테고리를 발화 단위 정답이 아닌 weak label로 사용",
            "wellness02_intent": "0.56~0.58, 자살충동 등 안전 신호 — 슬픔 distress=4 보존 (Phase 3.2 tier 1)",
            "wellness02_text_explicit": "0.50~0.60, 텍스트 자체에 감정 키워드가 있는 경우만 7감정 weak label로 채택 (Phase 3.2 tier 2)",
            "wellness02_context_only": "0.40, intent에는 distress 신호가 있지만 텍스트는 무색 — 중립 weak hard negative로 라우팅 (Phase 3.2 tier 3, source bias 차단)",
            "aihub_session_prior": "0.42~0.58, 세션 class/severity/paragraph 활성 라벨 기반 weak prior",
        },
        "source_files": {
            "original_train": str(EMOTION_SPLITS["train"].relative_to(BASE_DIR)),
            "original_val": str(EMOTION_SPLITS["val"].relative_to(BASE_DIR)),
            "original_calib": str(EMOTION_SPLITS["calib"].relative_to(BASE_DIR)),
            "wellness01": str(WELLNESS01_PATH.relative_to(BASE_DIR)),
            "wellness02": str(WELLNESS02_PATH.relative_to(BASE_DIR)),
            "aihub_label_root": str(AIHUB_LABEL_ROOT.relative_to(BASE_DIR)),
        },
        "candidate_counts_before_merge": {str(k): int(v) for k, v in source_counts.items()},
        "original_dedup_audit": original_dedup_audit,
        "split_summary": split_summary,
    }


def validate_outputs(frames: dict[str, pd.DataFrame]) -> None:
    """
    역할: 생성 산출물이 Phase 2 안전 조건을 만족하는지 검증한다.
    입력: split별 DataFrame
    출력: 검증 실패 시 예외, 성공 시 None
    """
    seen_text_keys: dict[str, tuple[str, int]] = {}
    for split, df in frames.items():
        missing_cols = [col for col in OUTPUT_COLUMNS if col not in df.columns]
        if missing_cols:
            raise ValueError(f"{split} split 누락 컬럼: {missing_cols}")
        if df["label_source"].isna().any() or df["weak_label_confidence"].isna().any():
            raise ValueError(f"{split} split에 label_source/weak_label_confidence 결측 존재")
        if "manual_scenario" in set(df["label_source"].astype(str)):
            raise ValueError("manual scenario eval 샘플이 학습 산출물에 포함됨")
        if not df["weak_label_confidence"].between(0.0, 1.0).all():
            raise ValueError(f"{split} split confidence 범위 오류")
        weak_sources = set(df[df["is_weak_label"]]["label_source"].astype(str))
        allowed_weak_sources = {
            "wellness_category",
            "wellness02_intent",
            "wellness02_text_explicit",
            "wellness02_context_only",
            "aihub_session_prior",
        }
        if weak_sources - allowed_weak_sources:
            raise ValueError(f"{split} split에 허용되지 않은 weak label source 존재: {weak_sources}")
        for row_index, text in enumerate(df["text"].astype(str), start=2):
            key = compact_text_key(text)
            previous = seen_text_keys.get(key)
            if previous:
                prev_split, prev_row_index = previous
                raise ValueError(
                    f"중복 텍스트 키 발견: {split}:{row_index} == {prev_split}:{prev_row_index}"
                )
            seen_text_keys[key] = (split, row_index)


def write_outputs(frames: dict[str, pd.DataFrame], label_map: dict[str, Any]) -> None:
    """
    역할: split CSV와 label map JSON을 data/processed에 저장한다.
    입력: split별 DataFrame, label map dict
    출력: 없음
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for split, df in frames.items():
        df.to_csv(OUT_PATHS[split], index=False, encoding="utf-8-sig")
        print(f"[저장 완료] {OUT_PATHS[split]} rows={len(df):,}")
    with open(LABEL_MAP_OUT, "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    print(f"[저장 완료] {LABEL_MAP_OUT}")


def print_summary(frames: dict[str, pd.DataFrame], source_counts: Counter[str]) -> None:
    """
    역할: 콘솔에 생성 결과 핵심 통계를 출력한다.
    입력: split별 DataFrame, 원천 후보 수 Counter
    출력: 없음
    """
    print("\n[후보 수]")
    for source, count in source_counts.items():
        print(f"  - {source}: {count:,}")
    print("\n[최종 split]")
    for split, df in frames.items():
        print(f"  - {split}: rows={len(df):,}, weak={int(df['is_weak_label'].sum()):,}")
        emotion_counts = {str(k): int(v) for k, v in df["emotion"].value_counts().items()}
        source_distribution = {str(k): int(v) for k, v in df["label_source"].value_counts().items()}
        print(f"    emotion={emotion_counts}")
        print(f"    sources={source_distribution}")


def build_semantic_emotion_dataset(disable_wellness02: bool = False) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """
    역할: Phase 2 semantic emotion 데이터셋 전체를 생성한다.
    입력: disable_wellness02 — True면 웰니스02 weak label을 전혀 포함하지 않는다 (Phase 3.3 ablation A 변종 1).
    출력: split별 DataFrame과 label map dict
    """
    original_by_split_raw = load_original_emotion_records()
    original_by_split, original_dedup_audit = deduplicate_original_records(original_by_split_raw)
    wellness01_records = load_wellness01_records()
    wellness02_records = [] if disable_wellness02 else load_wellness02_records()
    aihub_records = load_aihub_records()

    source_counts = Counter(
        {
            "original_emotion": sum(len(records) for records in original_by_split_raw.values()),
            "wellness_category": len(wellness01_records),
            "wellness02_intent": len(wellness02_records),
            "aihub_session_prior": len(aihub_records),
        }
    )
    weak_records = wellness01_records + wellness02_records + aihub_records
    frames = merge_records(original_by_split, weak_records)
    validate_outputs(frames)
    label_map = build_label_map(frames, source_counts, original_dedup_audit)
    label_map["disable_wellness02"] = bool(disable_wellness02)
    return frames, label_map


def main() -> None:
    """
    역할: semantic emotion 데이터셋을 생성하고 파일로 저장한다.
    입력: CLI 인자 (--disable-wellness02 옵션)
    출력: 없음
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--disable-wellness02",
        action="store_true",
        help="웰니스02 weak label을 모두 제외 (Phase 3.3 ablation A 변종 1).",
    )
    args = parser.parse_args()
    frames, label_map = build_semantic_emotion_dataset(disable_wellness02=args.disable_wellness02)
    write_outputs(frames, label_map)
    print_summary(frames, Counter(label_map["candidate_counts_before_merge"]))


if __name__ == "__main__":
    main()
