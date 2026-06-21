"""
build_cbt_anchors_from_koacd.py
역할: KoACD(한국 청소년 인지왜곡) 데이터셋 6개 xlsx에서 카테고리당 30개 anchor를
      추출하고, normal 대조군을 emotion_train 중립/행복에서 보강한 cbt_anchors.json v3을
      생성한다. 5범주 v2를 10범주로 확장하면서 mean-anchor 평탄화를 막기 위해 K-means
      클러스터링으로 의미 다양성을 확보한다.

입력:
  - data/raw/Cognitive_{Clarification,Balancing}_{Claude,Gemini,Gpt}.xlsx (6개)
  - data/processed/emotion_train.csv (normal 대조군 보강용)
  - data/processed/cbt_anchors.json (기존 v2 — normal 12개 시드 재사용)
출력:
  - data/processed/cbt_anchors_v3_keep.json    (--catastrophe-mode keep)
  - data/processed/cbt_anchors_v3_replace.json (--catastrophe-mode replace)

실행 예:
  C:/Users/WD/anaconda3/envs/dl_study/python.exe data/preprocess/build_cbt_anchors_from_koacd.py --catastrophe-mode keep
  C:/Users/WD/anaconda3/envs/dl_study/python.exe data/preprocess/build_cbt_anchors_from_koacd.py --catastrophe-mode replace

라이선스: KoACD는 CC BY 4.0 + research-only.
  Kim & Kim (2025), KoACD: The First Korean Adolescent Dataset for Cognitive
  Distortion Analysis, Findings of EMNLP 2025. https://github.com/cocoboldongle/KoACD
"""

import argparse
import glob
import json
import os
import random
import re
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from transformers import AutoTokenizer

# 프로젝트 루트 sys.path 등록 (models/roberta/train_roberta 임포트용)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)
from models.roberta.train_roberta import (  # noqa: E402
    RoBERTaMultiTask, NUM_EMOTION_CLS, NUM_NLI_CLS, DEVICE,
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")
CKPT_DIR = os.path.join(BASE_DIR, "models", "roberta", "checkpoints")
MODEL_NAME = "klue/roberta-base"
EMOTION_TRAIN_CSV = os.path.join(PROC_DIR, "emotion_train.csv")
EXISTING_ANCHOR_PATH = os.path.join(PROC_DIR, "cbt_anchors.json")

# anchor 길이 범위 (한글 문자 수 기준). 너무 짧으면 임베딩 노이즈, 너무 길면 평균 흐려짐.
MIN_LEN = 12
MAX_LEN = 60
ANCHORS_PER_CATEGORY = 30          # 카테고리당 총 anchor 수 (v2 시드/시드 + KoACD 보강)
NORMAL_ANCHOR_TARGET = 30          # normal 대조군 (기존 12 + 신규 18)
PER_CATEGORY_STORY_SAMPLE = 2000   # 카테고리당 사용할 Generated Story 행 수
EMBED_BATCH = 32
SEED_TOP_N = 400                   # 시드 mean과 유사도 상위 N개만 K-means 입력 (의미 응집도↑)


# 카테고리별 prototype anchor 풀.
#   - 기존 5범주는 v2 anchor 10개를 그대로 사용 (이미 검증된 punchy 표현, baseline +0.460 근거)
#   - 신규 5범주는 Beck 정의에 맞춘 1인칭 단언 표현 10개를 수기 작성
# AUGMENT 모드: 이 prototype 들이 그대로 최종 anchor 목록의 backbone이 되고,
#   KoACD 후보를 시드 mean 코사인 상위 → K-means 로 (ANCHORS_PER_CATEGORY - len(prototypes)) 개
#   만 추가 보강한다. v2의 punchy 분포가 유지되어 contrastive normal 차감에 밀리지 않는다.
CATEGORY_SEEDS: dict[str, list[str]] = {
    "all_or_nothing": [
        "완전히 실패한 것 같아요",
        "전부 아니면 아무것도 아닌 것 같아요",
        "제가 완벽하지 않으면 아무 의미가 없어요",
        "하나라도 잘못되면 다 망한 거잖아요",
        "100점이 아니면 0점이나 마찬가지예요",
        "조금이라도 실수하면 저는 형편없는 사람이에요",
        "성공 아니면 실패밖에 없어요",
        "조금이라도 부족하면 아예 안 하는 게 나아요",
        "완전히 잘하거나 아니면 포기해야죠",
        "완벽하지 않으면 의미 없다고 생각해요",
    ],
    "overgeneralization": [
        "항상 이런 식으로 잘못돼요",
        "저는 언제나 실패해요",
        "매번 이렇게 돼요, 달라질 게 없어요",
        "어떤 일을 해도 다 실패해요",
        "늘 제가 문제를 일으키는 것 같아요",
        "전 항상 이런 상황에 빠져요",
        "제 인생은 항상 이렇게 힘들어요",
        "저는 어디서든 미움을 받아요",
        "아무도 저를 좋아하지 않아요",
        "모든 게 항상 저 때문에 망가져요",
    ],
    "catastrophizing": [
        "이러면 다 끝나는 거 아닐까요",
        "이 일이 잘못되면 저는 완전히 망하는 거예요",
        "최악의 상황이 올 것 같아서 너무 무서워요",
        "이 실수 하나로 제 인생이 끝날 수도 있어요",
        "앞으로 어떻게 될지 생각만 해도 끔찍해요",
        "이게 실패하면 다시는 회복 못할 것 같아요",
        "한 번 망가지면 영원히 못 돌아와요",
        "이 상황이 계속되면 저는 버티지 못할 것 같아요",
        "조금만 더 이러면 진짜 무너질 것 같아요",
        "이번 일이 잘못되면 정말 다 끝이에요",
    ],
    # replace 모드는 catastrophizing 시드를 KoACD 라벨 의미(확대와 축소)에 맞게 확장
    "magnification_minimization": [
        "이 실수 하나로 제 인생이 끝날 수도 있어요",
        "최악의 상황이 올 것 같아서 너무 무서워요",
        "이번 일이 잘못되면 정말 다 끝이에요",
        "이게 실패하면 다시는 회복 못할 것 같아요",
        "앞으로 어떻게 될지 생각만 해도 끔찍해요",
        "제가 잘한 건 별로 의미 없는 일이에요",
        "남들이 잘한 건 정말 대단한 일인데",
        "제 성과는 사소하고 다른 사람 성과는 너무 커 보여요",
        "저는 작은 실수도 너무 크게 느껴져요",
        "제 좋은 점은 작아 보이고 단점만 커 보여요",
    ],
    "self_blame": [
        "다 제 잘못인 것 같아요",
        "제가 부족해서 이렇게 된 거예요",
        "제가 더 잘했더라면 이런 일은 없었을 텐데",
        "모든 게 제 탓인 것 같아서 너무 힘들어요",
        "저만 아니었으면 이런 일이 안 생겼을 거예요",
        "제가 무능해서 주변 사람들을 힘들게 하는 것 같아요",
        "저 때문에 다들 고생하는 것 같아요",
        "제가 좀 더 나은 사람이었더라면 달라졌겠죠",
        "이건 전부 제 능력이 안 되기 때문이에요",
        "제가 존재 자체가 문제인 것 같아요",
    ],
    "emotional_reasoning": [
        "느낌이 그러니까 분명히 그런 거예요",
        "불안하다는 건 뭔가 나쁜 일이 생긴다는 뜻이에요",
        "이렇게 기분이 나쁜 걸 보면 제가 정말 형편없는 거예요",
        "제가 무가치하다는 느낌이 드니까 진짜 무가치한 거겠죠",
        "두려움이 드는 건 위험하다는 신호잖아요",
        "이렇게 우울한 건 제 인생이 잘못돼 있다는 증거예요",
        "창피하다는 느낌이 드는 걸 보면 제가 정말 부끄러운 행동을 한 거겠죠",
        "아무도 안 좋아할 것 같은 느낌이 드니까 실제로 그런 거예요",
        "죄책감이 드는 건 제가 진짜 나쁜 사람이기 때문이에요",
        "이렇게 힘들다는 건 희망이 없다는 뜻이에요",
    ],
    "mental_filtering": [
        "좋은 일은 잘 기억나지 않고 안 좋은 일만 자꾸 떠올라요",
        "잘된 부분은 안 보이고 실수한 것만 계속 생각나요",
        "오늘 하루 다 괜찮았는데 그 한마디만 머릿속에 남아요",
        "남들이 칭찬한 건 다 잊고 비판만 곱씹게 돼요",
        "긍정적인 건 안 보이고 부정적인 것만 눈에 들어와요",
        "잘한 것은 안 보이고 망친 것만 계속 떠올라요",
        "기분 좋은 일은 금방 잊고 안 좋은 일만 오래 생각해요",
        "다 좋았는데 그 부정적인 한 가지만 남아요",
        "안 좋은 부분에만 자꾸 시선이 가요",
        "괜찮은 점은 안 보이고 부족한 점만 자꾸 보여요",
    ],
    "labeling": [
        "저는 그냥 실패자예요",
        "저는 무가치한 사람이에요",
        "저는 한심한 인간이에요",
        "저는 어차피 안 되는 사람이에요",
        "나는 쓸모없는 존재인 것 같아요",
        "나는 멍청한 사람이에요",
        "나는 그냥 못난 사람이에요",
        "저는 패배자라고 느껴져요",
        "저는 형편없는 인간이라는 생각이 들어요",
        "저는 정말 한심하고 별 볼 일 없는 사람이에요",
    ],
    "discounting_positive": [
        "그건 그냥 운이 좋았을 뿐이에요",
        "별거 아닌 일에 칭찬받은 것 같아요",
        "제가 잘해서가 아니라 그냥 쉬워서였어요",
        "그 정도는 누구나 할 수 있는 일이에요",
        "사람들이 그냥 좋게 말해준 거예요",
        "제 성과는 별거 아니에요",
        "운이 좋아서 된 거지 제 실력이 아니에요",
        "그건 그냥 우연이에요",
        "쉬운 일이라서 한 거지 잘한 게 아니에요",
        "저한테 잘했다고 해주신 건 의례적인 말이에요",
    ],
    "should_statements": [
        "저는 반드시 잘해야만 해요",
        "이런 식으로 살면 안 되는데",
        "꼭 완벽하게 해내야만 해요",
        "다른 사람을 실망시키면 안 돼요",
        "이 정도는 당연히 해야 하는 거잖아요",
        "이 나이에 이 정도는 해야 마땅해요",
        "저는 반드시 더 노력해야만 해요",
        "이런 감정을 느끼면 안 되는데",
        "약한 모습을 보여서는 안 돼요",
        "꼭 잘해내야 한다는 생각에 너무 힘들어요",
    ],
    "jumping_to_conclusions": [
        "분명 저를 싫어하는 거예요",
        "어차피 잘 안 될 게 뻔해요",
        "말 안 해도 저를 어떻게 생각하는지 알아요",
        "분명 나쁜 결과가 나올 거예요",
        "이건 분명 안 좋은 신호예요",
        "굳이 물어보지 않아도 답은 뻔해요",
        "분명히 저를 비웃었을 거예요",
        "결과를 안 봐도 망했을 게 분명해요",
        "어차피 안 좋은 일만 생길 거예요",
        "굳이 보지 않아도 결과는 안 봐도 알아요",
    ],
}

# KoACD 한글 라벨 정규화 (오타 통합)
KOACD_LABEL_CANONICAL = {
    "흑백 사고": "흑백 사고",
    "과잉 일반화": "과잉 일반화",
    "확대와 축소": "확대와 축소",
    "부정적 편향": "부정적 편향",
    "개인화": "개인화",
    "감정적 추론": "감정적 추론",
    "낙인찍기": "낙인찍기",
    "긍정 축소화": "긍정 축소화",
    "'해야 한다' 진술": "'해야 한다' 진술",
    "해야 한다' 진술": "'해야 한다' 진술",   # 오타 2건 정규화
    "성급한 판단": "성급한 판단",
}

# 우리 프로젝트 카테고리 정의. catastrophe_mode 에 따라 "확대와 축소" 매핑이 달라진다.
def get_category_specs(catastrophe_mode: str) -> list[dict]:
    """
    역할: KoACD 한글 라벨 → 우리 cbt_anchors.json 카테고리(id, name, description) 매핑 정의.
    입력: catastrophe_mode ("keep" | "replace")
          - keep: "확대와 축소"를 기존 "파국화" 카테고리로 흡수 (DB row 호환 유지)
          - replace: "확대와 축소"를 KoACD 표준 라벨 그대로 사용 (의미 일관성 우선)
    출력: 카테고리 dict 리스트. 각 dict 는 anchors 비어있는 상태로 반환된다.
    """
    if catastrophe_mode not in {"keep", "replace"}:
        raise ValueError(f"catastrophe_mode 는 keep|replace 여야 함: {catastrophe_mode}")

    specs = [
        {
            "id": "all_or_nothing", "name": "이분법적 사고",
            "koacd_label": "흑백 사고",
            "description": "중간 지점 없이 극단적으로 평가 (완전한 성공 아니면 완전한 실패)",
        },
        {
            "id": "overgeneralization", "name": "과잉일반화",
            "koacd_label": "과잉 일반화",
            "description": "한 번의 부정적 사건을 모든 상황에 적용 (항상, 언제나, 절대로 등)",
        },
        {
            # catastrophe_mode 에 따라 라벨/이름 분기
            "id": "catastrophizing" if catastrophe_mode == "keep" else "magnification_minimization",
            "name": "파국화" if catastrophe_mode == "keep" else "확대와 축소",
            "koacd_label": "확대와 축소",
            "description": (
                "작은 문제를 최악의 결과로 확대 해석 (KoACD 확대와 축소 흡수)"
                if catastrophe_mode == "keep"
                else "부정적인 일은 크게, 긍정적인 일은 작게 평가 (확대와 축소)"
            ),
        },
        {
            "id": "self_blame", "name": "자기비난·개인화",
            "koacd_label": "개인화",
            "description": "부정적 사건의 원인을 전적으로 자신에게 귀속",
        },
        {
            "id": "emotional_reasoning", "name": "감정적 추론",
            "koacd_label": "감정적 추론",
            "description": "감정적 느낌을 객관적 사실로 간주",
        },
        {
            "id": "mental_filtering", "name": "부정적 편향",
            "koacd_label": "부정적 편향",
            "description": "긍정 정보를 무시하고 부정 정보만 선택적으로 주목",
        },
        {
            "id": "labeling", "name": "낙인찍기",
            "koacd_label": "낙인찍기",
            "description": "자신이나 타인을 단정적인 부정 라벨로 규정 (예: 나는 실패자다)",
        },
        {
            "id": "discounting_positive", "name": "긍정 축소화",
            "koacd_label": "긍정 축소화",
            "description": "긍정적 경험을 별것 아닌 것으로 깎아내림",
        },
        {
            "id": "should_statements", "name": "당위 진술",
            "koacd_label": "'해야 한다' 진술",
            "description": "비현실적 당위(해야 한다/하면 안 된다)로 자신이나 상황을 평가",
        },
        {
            "id": "jumping_to_conclusions", "name": "성급한 판단",
            "koacd_label": "성급한 판단",
            "description": "근거 없이 부정적 결론으로 비약 (독심술/예언자적 사고)",
        },
    ]
    return specs


# ─────────────────────────────────────────────────────────────────────────────
# KoACD 데이터 로딩 / 품질 필터
# ─────────────────────────────────────────────────────────────────────────────
def load_koacd_unified() -> pd.DataFrame:
    """
    역할: KoACD 6개 xlsx 를 하나의 DataFrame 으로 통합한다. 파일마다 컬럼 순서가
          달라 한글 라벨/Generated Story/품질 점수만 추출해 표준화한다.
    입력: 없음
    출력: columns=[label_kr, story, quality_mean, fluency_min, source_file, generative_model]
          quality_mean: 두 cross-evaluator 의 6개 점수(consistency*2, accuracy*2, fluency*2) 평균
          fluency_min: 두 evaluator fluency 의 최소값
    """
    score_pairs = [
        ("Gpt Consistency", "Gpt Accuracy", "Gpt Fluency"),
        ("Gemini Consistency", "Gemini Accuracy", "Gemini Fluency"),
        ("Claude Consistency", "Claude Accuracy", "Claude Fluency"),
    ]
    rows = []
    files = sorted(glob.glob(os.path.join(RAW_DIR, "Cognitive_*.xlsx")))
    for fp in files:
        df = pd.read_excel(fp)
        for _, r in df.iterrows():
            label_raw = str(r.get("Cognitive Distortion (Korean)", "")).strip()
            label = KOACD_LABEL_CANONICAL.get(label_raw)
            story = r.get("Generated Story")
            if label is None or pd.isna(story):
                continue

            # 두 cross-evaluator 점수만 추려 평균/최소 산출
            scores = []
            fluencies = []
            for cons_col, acc_col, flu_col in score_pairs:
                if cons_col not in df.columns:
                    continue
                cons = r.get(cons_col)
                acc = r.get(acc_col)
                flu = r.get(flu_col)
                if pd.isna(cons) or pd.isna(acc) or pd.isna(flu):
                    continue
                scores.extend([float(cons), float(acc), float(flu)])
                fluencies.append(float(flu))
            if not scores:
                continue

            rows.append({
                "label_kr": label,
                "story": str(story),
                "quality_mean": float(np.mean(scores)),
                "fluency_min": float(min(fluencies)) if fluencies else 0.0,
                "source_file": os.path.basename(fp),
                "generative_model": str(r.get("Generative Model", "")).upper(),
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 텍스트 정제 / 문장 분리 / anchor 후보 추출
# ─────────────────────────────────────────────────────────────────────────────
_HEADER_RE = re.compile(r"\[[^\]]+\]\s*-+\s*", re.MULTILINE)  # "[남자/16세]\n---\n" 헤더 제거
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_HANGUL_RE = re.compile(r"[가-힣]")
_BAD_PATTERN_RE = re.compile(
    r"(ㅋ{2,}|ㅎ{2,}|ㅠ{2,}|ㅜ{2,}|;{2,}|"   # 청소년 의성어/반복 기호
    r"\d{3,}|"                              # 긴 숫자
    r"[!?]{3,})"                             # 감탄 과다
)


def strip_header(story: str) -> str:
    """
    역할: Generated Story 앞단의 "[남자/16세]\n---\n" 헤더를 제거한다.
    입력: 원본 story 문자열
    출력: 헤더가 제거된 본문
    """
    cleaned = _HEADER_RE.sub("", story)
    cleaned = cleaned.replace("---", " ")
    return cleaned.strip()


def split_sentences(text: str) -> list[str]:
    """
    역할: 한국어 텍스트를 문장 단위로 분리. 종결부호 + 개행 기준.
    입력: 본문 문자열
    출력: 문장 리스트 (앞뒤 공백 제거됨)
    """
    parts = _SENT_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p and p.strip()]


def is_good_anchor(s: str) -> bool:
    """
    역할: 문장이 anchor 후보로 적합한지 판정. 길이/한글 비율/노이즈 패턴 검사.
    입력: 문장 문자열
    출력: True/False
    """
    if not (MIN_LEN <= len(s) <= MAX_LEN):
        return False
    hangul = len(_HANGUL_RE.findall(s))
    if hangul / max(len(s), 1) < 0.6:   # 한글 비율 60% 미만은 제외
        return False
    if _BAD_PATTERN_RE.search(s):
        return False
    return True


def collect_candidates_per_category(
    df: pd.DataFrame, koacd_label: str, sample_n: int
) -> list[str]:
    """
    역할: 한 KoACD 라벨에 해당하는 Generated Story 들을 다운샘플 후 문장 분해해
          anchor 후보 풀을 만든다. 품질 필터(quality_mean ≥ 2.3, fluency_min ≥ 2)
          를 적용하고 중복 제거한다.
    입력: 통합 DataFrame, KoACD 한글 라벨, 라벨당 사용할 story 개수
    출력: 문장 후보 리스트 (중복 제거)
    """
    sub = df[(df["label_kr"] == koacd_label) &
             (df["quality_mean"] >= 2.3) &
             (df["fluency_min"] >= 2.0)]
    if len(sub) > sample_n:
        sub = sub.sample(n=sample_n, random_state=SEED)
    cands = []
    seen = set()
    for story in sub["story"].tolist():
        body = strip_header(story)
        for sent in split_sentences(body):
            if not is_good_anchor(sent):
                continue
            if sent in seen:
                continue
            seen.add(sent)
            cands.append(sent)
    return cands


# ─────────────────────────────────────────────────────────────────────────────
# 임베딩 + 클러스터링 기반 다양성 샘플링
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def encode_batch(texts: list[str], model, tokenizer) -> np.ndarray:
    """
    역할: 텍스트 리스트를 KLUE-RoBERTa [CLS] 임베딩으로 변환 (배치 처리).
    입력: 텍스트 리스트, 모델, 토크나이저
    출력: (N, hidden_dim) numpy 배열
    """
    embs = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i:i + EMBED_BATCH]
        enc = tokenizer(batch, return_tensors="pt", truncation=True,
                        max_length=64, padding=True)
        out = model.encoder(input_ids=enc["input_ids"].to(DEVICE),
                            attention_mask=enc["attention_mask"].to(DEVICE))
        cls = out.last_hidden_state[:, 0, :].cpu().numpy()
        embs.append(cls)
    return np.vstack(embs)


def filter_by_seed(
    texts: list[str], embs: np.ndarray, seed_emb_mean: np.ndarray, top_n: int
) -> tuple[list[str], np.ndarray]:
    """
    역할: 카테고리 시드 임베딩 평균과 코사인 유사도가 높은 상위 top_n 후보만 남겨
          K-means 입력 단계에서 카테고리 시그니처에서 멀어진 노이즈 표현을 제거한다.
    입력: 후보 텍스트 리스트, 후보 임베딩 (N, dim), 시드 mean 임베딩 (dim,), 보존할 N
    출력: (필터링된 텍스트 리스트, 필터링된 임베딩 (top_n, dim))
    """
    if len(texts) <= top_n:
        return texts, embs
    text_norms = np.linalg.norm(embs, axis=1) + 1e-8
    seed_norm = np.linalg.norm(seed_emb_mean) + 1e-8
    sims = embs @ seed_emb_mean / (text_norms * seed_norm)
    top_idx = np.argsort(-sims)[:top_n]
    return [texts[i] for i in top_idx], embs[top_idx]


def select_diverse(texts: list[str], embs: np.ndarray, k: int) -> list[str]:
    """
    역할: K-means 로 k개 클러스터를 만들고, 각 클러스터의 centroid 와 가장 가까운
          문장을 1개씩 선택해 의미 다양성을 확보한 anchor 표현을 반환한다.
    입력: 후보 문장 리스트, 같은 순서의 임베딩 (N, dim), 목표 anchor 수 k
    출력: 길이 k 의 대표 문장 리스트 (텍스트 길이 오름차순 정렬)
    """
    if len(texts) <= k:
        return sorted(set(texts), key=len)
    km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
    labels = km.fit_predict(embs)
    selected = []
    for c in range(k):
        idxs = np.where(labels == c)[0]
        if len(idxs) == 0:
            continue
        centroid = km.cluster_centers_[c]
        d = np.linalg.norm(embs[idxs] - centroid, axis=1)
        best = idxs[int(np.argmin(d))]
        selected.append(texts[best])
    # 같은 텍스트가 다른 클러스터에서 동시 선택될 가능성은 거의 없지만 dedup 추가
    selected = list(dict.fromkeys(selected))
    return sorted(selected, key=len)


# ─────────────────────────────────────────────────────────────────────────────
# normal 대조군 구성
# ─────────────────────────────────────────────────────────────────────────────
def build_normal_anchors(seed_anchors: list[str], target: int) -> list[str]:
    """
    역할: 기존 normal 대조군 12개 + emotion_train.csv 중립/행복 짧은 발화에서
          target 개 만큼 보강한다. 인지 왜곡 표현과 임베딩 거리를 확보하기 위해
          질문/감탄/이모티콘 노이즈는 제외한다.
    입력: 기존 v2 normal anchor 시드 리스트, 목표 개수
    출력: 보강된 normal anchor 리스트 (target 개 이하면 가능한 만큼)
    """
    used = list(dict.fromkeys(seed_anchors))   # 순서 유지 dedup
    if len(used) >= target:
        return used[:target]

    df = pd.read_csv(EMOTION_TRAIN_CSV)
    df = df[df["emotion"].isin(["중립", "행복"])]
    df = df[df["text"].astype(str).str.len().between(MIN_LEN, MAX_LEN)]
    bad = df["text"].astype(str).str.contains(r"[?!]{2,}|ㅋ|ㅎ|ㅠ|ㅜ", regex=True)
    df = df[~bad]
    pool = df["text"].astype(str).drop_duplicates().tolist()
    random.Random(SEED).shuffle(pool)

    for t in pool:
        if len(used) >= target:
            break
        if t not in used and is_good_anchor(t):
            used.append(t)
    return used


# ─────────────────────────────────────────────────────────────────────────────
# 메인 파이프라인
# ─────────────────────────────────────────────────────────────────────────────
def load_seed_normal() -> list[str]:
    """
    역할: 기존 cbt_anchors.json v2 의 normal 대조군 12개를 시드로 읽어온다.
    입력: 없음
    출력: normal 시드 anchor 리스트
    """
    if not os.path.exists(EXISTING_ANCHOR_PATH):
        return []
    with open(EXISTING_ANCHOR_PATH, encoding="utf-8") as f:
        data = json.load(f)
    for cat in data.get("categories", []):
        if cat.get("is_contrast"):
            return [str(a) for a in cat.get("anchors", [])]
    return []


def build_anchors(catastrophe_mode: str) -> dict:
    """
    역할: KoACD 데이터 → 카테고리당 anchor 추출 → normal 대조군 보강 →
          cbt_anchors.json v3 dict 구성. 두 모드(keep/replace) 공용.
    입력: catastrophe_mode
    출력: anchors json dict (categories + meta)
    """
    print(f"[1/5] KoACD xlsx 로드 중...")
    df = load_koacd_unified()
    print(f"  → 총 {len(df)} 행 (label dist):\n    "
          + ", ".join(f"{k}={v}" for k, v in df['label_kr'].value_counts().to_dict().items()))

    print(f"[2/5] RoBERTa 인코더 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = RoBERTaMultiTask(MODEL_NAME, NUM_EMOTION_CLS, NUM_NLI_CLS).to(DEVICE)
    state = torch.load(os.path.join(CKPT_DIR, "roberta_final.pt"),
                       map_location=DEVICE, weights_only=True)
    model.load_state_dict(state, strict=False)   # utterance_type_head 별도 저장이라 strict=False
    model.eval()

    specs = get_category_specs(catastrophe_mode)
    print(f"[3/5] 카테고리 {len(specs)}개 anchor 추출 (mode={catastrophe_mode}, AUGMENT, seed top-N={SEED_TOP_N})")
    categories = []
    for spec in specs:
        prototypes = CATEGORY_SEEDS.get(spec["id"])
        if not prototypes:
            raise ValueError(f"카테고리 '{spec['id']}'의 prototype 시드가 정의되지 않음")
        # KoACD 보강 목표 = 총 anchor 수 - prototype 개수 (예: 30 - 10 = 20)
        n_supplement = max(0, ANCHORS_PER_CATEGORY - len(prototypes))

        prototype_embs = encode_batch(prototypes, model, tokenizer)
        seed_mean = prototype_embs.mean(axis=0)

        cands = collect_candidates_per_category(
            df, spec["koacd_label"], PER_CATEGORY_STORY_SAMPLE
        )
        # prototype 과 동일/유사한 표현이 KoACD 후보에서 다시 뽑히는 것 방지
        cand_set_init = set(cands)
        for p in prototypes:
            cand_set_init.discard(p)
        cands = [c for c in cands if c in cand_set_init]

        supplements: list[str] = []
        if cands and n_supplement > 0:
            cand_embs = encode_batch(cands, model, tokenizer)
            cands_f, embs_f = filter_by_seed(cands, cand_embs, seed_mean, SEED_TOP_N)
            supplements = select_diverse(cands_f, embs_f, n_supplement)

        # 최종 anchor: prototype backbone + KoACD 보강 (중복 제거)
        merged = list(dict.fromkeys(prototypes + supplements))[:ANCHORS_PER_CATEGORY]
        print(f"  - {spec['name']:12s} (KoACD '{spec['koacd_label']}'): "
              f"prototype {len(prototypes)} + KoACD {len(supplements)} = anchor {len(merged)}")
        categories.append({
            "id": spec["id"],
            "name": spec["name"],
            "description": spec["description"],
            "anchors": merged,
            "source": (
                f"prototype {len(prototypes)} (v2/handwritten) + KoACD '{spec['koacd_label']}' "
                f"보강 {len(supplements)} (seed-filter top-{SEED_TOP_N} → K-means K={n_supplement})"
            ),
        })

    print(f"[4/5] normal 대조군 보강 (목표 {NORMAL_ANCHOR_TARGET}개)")
    seed = load_seed_normal()
    normals = build_normal_anchors(seed, NORMAL_ANCHOR_TARGET)
    print(f"  - 시드 {len(seed)} → 최종 {len(normals)}")
    categories.append({
        "id": "non_distortion_neutral",
        "name": "비왜곡_중립대조",
        "description": "인지 왜곡이 없는 평범한 일상·감정 진술. contrastive 차감용 대조군.",
        "anchors": normals,
        "is_contrast": True,
        "source": "v2 시드 12 + emotion_train.csv 중립/행복 보강",
    })

    print(f"[5/5] meta 구성 + 반환")
    total_anchors = sum(len(c["anchors"]) for c in categories)
    distortion_n = len([c for c in categories if not c.get("is_contrast")])
    return {
        "categories": categories,
        "meta": {
            "version": "3.0",
            "catastrophe_mode": catastrophe_mode,
            "total_categories": len(categories),
            "total_distortion_categories": distortion_n,
            "total_anchors": total_anchors,
            "anchors_per_category": ANCHORS_PER_CATEGORY,
            "normal_anchors": NORMAL_ANCHOR_TARGET,
            "usage": "CBT 인지 왜곡 탐지 — KLUE-RoBERTa [CLS] mean-anchor + contrastive 차감",
            "source": (
                "KoACD (Kim & Kim 2025, EMNLP Findings, CC BY 4.0 research-only) "
                "+ emotion_train.csv 중립/행복 보강"
            ),
            "build_method": (
                "Generated Story → 헤더 제거 → 문장 분리 → 길이/한글비율/노이즈 필터 "
                "→ KLUE-RoBERTa [CLS] 임베딩 → 시드 mean 코사인 상위 "
                f"{SEED_TOP_N}개 → K-means(K={ANCHORS_PER_CATEGORY}) 클러스터 대표 선택"
            ),
            "built_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "threshold_note": "v3 anchors 기준 CBT_THRESHOLD 는 eval_cbt.py 재측정 후 확정",
        },
    }


def main():
    """
    역할: CLI 진입점. catastrophe_mode 별로 v3 JSON 파일을 생성한다.
    입력: --catastrophe-mode {keep, replace}
    출력: data/processed/cbt_anchors_v3_{mode}.json
    """
    parser = argparse.ArgumentParser(description="KoACD 기반 CBT v3 anchor 빌드")
    parser.add_argument(
        "--catastrophe-mode", choices=["keep", "replace"], default="keep",
        help="확대와 축소 처리 방식 (keep=기존 파국화에 흡수 / replace=KoACD 라벨로 교체)",
    )
    args = parser.parse_args()

    out = build_anchors(args.catastrophe_mode)
    out_path = os.path.join(PROC_DIR, f"cbt_anchors_v3_{args.catastrophe_mode}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n[완료] 저장: {out_path}")
    print(f"  카테고리 {out['meta']['total_categories']} "
          f"(왜곡 {out['meta']['total_distortion_categories']} + 대조 1), "
          f"총 anchor {out['meta']['total_anchors']}")


if __name__ == "__main__":
    main()
