"""
cbt_similarity.py
역할: RoBERTa [CLS] 임베딩과 CBT 앵커 표현 간 코사인 유사도로 인지 왜곡 점수(cbt_score) 산출
      5가지 CBT 카테고리(이분법적 사고/과잉일반화/파국화/자기비난/감정적 추론) 기준
입력: 발화 텍스트, 로드된 모델/토크나이저, CBT 앵커 dict
출력: cbt_score (float, 0~1)
"""

import os
import json
import torch
import numpy as np

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANCHOR_PATH = os.path.join(BASE_DIR, "data", "processed", "cbt_anchors.json")

# CBT threshold — v3 prototype-only anchors(10범주 + max-over-anchors) 기준.
# v3 초기(2026-04-26)에는 0.70 채택했으나 명백한 prototype 표현 8건이 0.5~0.6대로
# 임계 미만 누락되는 문제 발견(예: "나는 항상 모든 일에 실패해" 0.529, "이 실수 하나로
# 인생 끝났어" 0.568). v3.1(2026-04-27)에서 0.60으로 낮춰 회수율 증가 + utterance_type
# 게이트(scheduler) 추가로 false positive 차단 안전망 확보.
# eval/eval_cbt.py (val 300샘플 random_state=42):
#   threshold 0.60: CBT관련 감지율 0.640 / 비관련 오탐 0.167 / margin +0.473
#   threshold 0.70: 감지율 0.540 / 오탐 0.080 / margin +0.460
# → 0.60 채택 (utterance_type 게이트로 false positive 추가 방어).
# 점수 방식: max-over-anchors per category, contrastive 차감 (왜곡 max - 대조 max + 0.5).
CBT_THRESHOLD = 0.60


# 대조군 카테고리 식별용 플래그 (v2 anchors 스키마). 런타임에서 모듈 수준으로 캐시.
_CONTRAST_CATEGORIES: set[str] = set()
_ANCHOR_CACHE: dict[str, tuple[dict[str, list[str]], set[str]]] = {}


def _normalize_anchor_payload(raw_data: dict) -> dict[str, list[str]]:
    """
    역할: CBT 앵커 JSON을 런타임이 쓰기 쉬운 {이름: [표현]} 구조로 정규화
          또한 `is_contrast=True` 플래그가 있는 카테고리를 모듈 수준 set에 기록해
          compute_cbt_score에서 대조군으로 사용되게 한다.
    입력: JSON에서 읽은 원본 dict
    출력: {카테고리명: 앵커 문장 리스트}
    """
    _CONTRAST_CATEGORIES.clear()

    if "categories" in raw_data:
        normalized: dict[str, list[str]] = {}
        for item in raw_data.get("categories", []):
            if not isinstance(item, dict):
                continue
            category_name = str(item.get("name", "")).strip()
            phrases = item.get("anchors", [])
            if not category_name:
                continue
            if not isinstance(phrases, list):
                raise ValueError("CBT 앵커의 anchors 필드는 리스트여야 합니다.")
            normalized[category_name] = [str(phrase) for phrase in phrases if str(phrase).strip()]
            if bool(item.get("is_contrast", False)):
                _CONTRAST_CATEGORIES.add(category_name)
        return normalized

    # 과거 포맷과의 호환을 위해 기존 {카테고리명: [표현]} 구조도 그대로 허용한다.
    normalized = {}
    for category_name, phrases in raw_data.items():
        if category_name == "meta":
            continue
        if not isinstance(phrases, list):
            raise ValueError("CBT 앵커 항목은 리스트여야 합니다.")
        normalized[str(category_name)] = [str(phrase) for phrase in phrases if str(phrase).strip()]
    return normalized


def load_anchors(path: str = ANCHOR_PATH) -> dict:
    """
    역할: CBT 앵커 표현 json 로드 및 경로별 캐시 재사용
    입력: json 파일 경로
    출력: {카테고리명: [표현 리스트]} dict
    """
    cache_key = os.path.abspath(path)
    if cache_key in _ANCHOR_CACHE:
        anchors, contrast_categories = _ANCHOR_CACHE[cache_key]
        _CONTRAST_CATEGORIES.clear()
        _CONTRAST_CATEGORIES.update(contrast_categories)
        return anchors

    if not os.path.exists(path):
        raise FileNotFoundError(f"CBT 앵커 파일 없음: {path}")
    with open(path, encoding="utf-8") as f:
        raw_data = json.load(f)
    anchors = _normalize_anchor_payload(raw_data)
    if not anchors:
        raise ValueError(f"CBT 앵커 정규화 결과가 비었습니다: {path}")
    _ANCHOR_CACHE[cache_key] = (anchors, set(_CONTRAST_CATEGORIES))
    return anchors


@torch.no_grad()
def _encode(texts: list, model, tokenizer, device) -> np.ndarray:
    """
    역할: 텍스트 리스트를 [CLS] 임베딩으로 변환
    입력: 텍스트 리스트, 모델, 토크나이저, 디바이스
    출력: (N, hidden_dim) numpy 배열
    """
    enc = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=True,
    )
    input_ids = enc["input_ids"].to(device)
    attn_mask  = enc["attention_mask"].to(device)

    # RoBERTaMultiTask는 공유 인코더를 encoder 속성으로 보관한다.
    outputs = model.encoder(input_ids=input_ids, attention_mask=attn_mask)
    cls_emb = outputs.last_hidden_state[:, 0, :]  # (N, hidden_dim)
    return cls_emb.cpu().numpy()


def build_anchor_embeddings(anchors: dict, model, tokenizer, device) -> dict:
    """
    역할: 카테고리별 앵커 표현을 임베딩 행렬로 변환한다 (max-over-anchors 방식).
          v2까지는 카테고리당 mean 벡터 1개를 사용했으나, 카테고리 수가 늘면
          mean이 평탄화되어 prototype 표현이 흐려지는 문제가 있어 v3부터 카테고리 내
          모든 앵커 임베딩을 보존하고 compute_cbt_score 에서 max 유사도를 취한다.
          response_anchors 와 동일한 점수 방식 (max bad - max normal contrastive).
    입력: 앵커 dict, 모델, 토크나이저, 디바이스
    출력: {카테고리명: anchor_embeddings (2D numpy [K, dim])} dict
    """
    anchor_embs = {}
    for category, phrases in anchors.items():
        embs = _encode(phrases, model, tokenizer, device)  # (K, dim)
        anchor_embs[category] = embs                       # (K, dim) 보존
    return anchor_embs


def _max_cosine(text_emb: np.ndarray, anchor_matrix: np.ndarray) -> float:
    """
    역할: 단일 텍스트 임베딩 vs 카테고리 앵커 행렬의 코사인 유사도 max를 계산한다.
    입력: text_emb (dim,), anchor_matrix (K, dim)
    출력: 카테고리 내 최대 코사인 유사도 ([-1, 1])
    """
    text_norm = np.linalg.norm(text_emb) + 1e-8
    anchor_norms = np.linalg.norm(anchor_matrix, axis=1) + 1e-8
    sims = anchor_matrix @ text_emb / (anchor_norms * text_norm)
    return float(sims.max())


def compute_cbt_score(
    text: str,
    model,
    tokenizer,
    device,
    anchor_embs: dict,
) -> dict:
    """
    역할: 발화 텍스트의 CBT 인지 왜곡 점수 산출 (v3 max-over-anchors 방식).
          카테고리 내 anchor 별 코사인 유사도의 최대값을 카테고리 점수로 사용.
          v3 anchors(대조군 존재): contrastive 방식으로
              raw_cbt = max_cat( max_anchor( cosine(text, anchor) ) )  # 왜곡 카테고리들
              raw_ctr = max_cat( max_anchor( cosine(text, anchor) ) )  # 대조 카테고리들
              cbt_score = clip(raw_cbt - raw_ctr + 0.5, 0, 1)
          → mean-anchor 방식 대비 prototype 표현이 살아있을 때 카테고리 점수가
            평탄화되지 않음. 단 단일 noisy anchor 가 점수를 끌어올릴 수 있어
            anchor 큐레이션 품질이 더 중요해진다.
    입력: 발화 텍스트, 모델, 토크나이저, 디바이스, 사전 계산된 앵커 임베딩 dict
          (각 카테고리 값은 (K, dim) 형태의 anchor 행렬)
    출력: {
        cbt_score:    float (0~1),
        raw_cbt:      float (왜곡 max-of-max 유사도, 0~1),
        raw_contrast: float (대조 max-of-max 유사도, 0~1),
        top_category: str,
        similarities: {카테고리명: float},
    }
    """
    text_emb = _encode([text], model, tokenizer, device)[0]  # (dim,)

    similarities = {}
    for category, anchor_matrix in anchor_embs.items():
        max_sim = _max_cosine(text_emb, anchor_matrix)
        similarities[category] = (max_sim + 1.0) / 2.0  # [-1,1] → [0,1]

    distortion_sims = {c: v for c, v in similarities.items() if c not in _CONTRAST_CATEGORIES}
    contrast_sims   = {c: v for c, v in similarities.items() if c in _CONTRAST_CATEGORIES}

    if not distortion_sims:
        raise ValueError("왜곡 카테고리 앵커가 없습니다.")

    raw_cbt      = float(max(distortion_sims.values()))
    top_category = max(distortion_sims, key=distortion_sims.get)

    if contrast_sims:
        # contrastive 모드: 대조군 차감 후 0.5 오프셋(중립값 유지) + [0,1] 클립
        raw_contrast = float(max(contrast_sims.values()))
        cbt_score    = float(np.clip(raw_cbt - raw_contrast + 0.5, 0.0, 1.0))
    else:
        # v1 호환: 대조군 없으면 왜곡 max 그대로
        raw_contrast = 0.0
        cbt_score    = raw_cbt

    return {
        "cbt_score":    cbt_score,
        "raw_cbt":      raw_cbt,
        "raw_contrast": raw_contrast,
        "top_category": top_category,
        "similarities": similarities,
    }
