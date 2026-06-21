"""
roberta_score.py
역할: RoBERTa 추론 → 감정 logits + NLI logits → roberta_score / 위기 판별
      Temperature Scaling 적용 후 95퍼센타일 정규화로 0~1 점수 산출
입력: 발화 텍스트 (str), 로드된 모델/토크나이저
출력: roberta_score (float, 0~1), is_crisis (bool), emotion_probs (list)
"""

import os
import json
import re
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer
from pipeline.utterance_type import (
    UTTERANCE_TYPES,
    classify_utterance_type,
    normalize_emotion_analysis_text,
    has_fear_marker,
    has_high_intensity_fear_marker,
    has_anger_marker,
    has_high_intensity_anger_marker,
    has_distress_marker,
    is_academic_anxiety_text,
    is_practical_anxiety_relief_text,
    is_practical_question_text,
    is_mild_unease_text,
    is_mild_low_mood_text,
    is_limited_situational_distress_text,
    is_interpersonal_remorse_text,
    is_situational_sadness_text,
    is_physical_exertion_text,
    is_routine_discomfort_text,
    is_daily_routine_neutral_text,
    is_administrative_technical_neutral_text,
    is_situational_anxiety_surprise_text,
    has_situational_anxiety_marker,
    is_situational_anger_text,
    is_positive_affect_text,
    is_sensory_disgust_text,
    is_low_risk_sensory_disgust_text,
    is_laughter_only_text,
    has_negative_safety_signal,
    has_crisis_marker,
)

# ── 경로 ────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_DIR = os.path.join(BASE_DIR, "models", "roberta", "checkpoints")
T_RESULT = os.path.join(CKPT_DIR, "temperature_result.json")
RUNTIME_CONFIG = os.path.join(CKPT_DIR, "runtime_config.json")
EMOTION_LOGIT_BIAS_ENV = "EMOTION_CHATBOT_EMOTION_LOGIT_BIAS_PATH"
UTTERANCE_TYPE_HEAD_CKPT = os.path.join(CKPT_DIR, "roberta_utterance_intent_head.pt")
CBT_CLASS_HEAD_CKPT = os.path.join(CKPT_DIR, "roberta_cbt_class_head.pt")
DISTRESS_HEAD_CKPT = os.path.join(CKPT_DIR, "roberta_distress_head.pt")
DISTRESS_LABEL_MAP_DEFAULT = {
    0: "calm_or_positive",
    1: "mild_distress",
    2: "moderate_distress",
    3: "high_distress",
    4: "crisis_candidate",
}

MODEL_NAME = "klue/roberta-base"

# 기본값은 체크포인트 설정 파일이 없을 때만 사용한다.
DEFAULT_CRISIS_THRESHOLD = 0.40
# 2026-04-27 emotion_train.csv 11,831행 전체 기준 P95 측정값으로 갱신.
# 이전 0.8152는 rare_aug 이전 데이터셋에서 측정된 값.
DEFAULT_ROBERTA_SCORE_P95 = 0.8460
UTTERANCE_TYPE_CONFIDENCE_FLOOR = 0.45

ID_TO_UTTERANCE_TYPE = {
    0: UTTERANCE_TYPES["CASUAL_SHARE"],
    1: UTTERANCE_TYPES["POSITIVE_SHARE"],
    2: UTTERANCE_TYPES["ROUTINE_DISCOMFORT"],
    3: UTTERANCE_TYPES["EMOTIONAL_DISTRESS"],
    4: UTTERANCE_TYPES["PREFERENCE_QUESTION"],
    5: UTTERANCE_TYPES["PRACTICAL_QUESTION"],
    6: UTTERANCE_TYPES["CRISIS_CANDIDATE"],
}

# 감정별 우울 위험 가중치 (높을수록 우울 경향)
EMOTION_WEIGHT = {
    "행복": 0.0,
    "중립": 0.2,
    "놀람": 0.3,
    "분노": 0.6,
    "혐오": 0.7,
    "슬픔": 0.8,
    "공포": 0.9,
}
LABEL2EMOTION = ["행복", "중립", "슬픔", "공포", "혐오", "분노", "놀람"]

CASUAL_NEUTRAL_SCORE_CAP = 0.35
POSITIVE_AFFECT_SCORE_CAP = 0.30
POSITIVE_AFFECT_NLI_CAP = 0.20
ROUTINE_DISCOMFORT_SCORE_CAP = 0.45
SENSORY_DISGUST_SCORE_FLOOR = 0.45
SENSORY_DISGUST_SCORE_CAP = 0.50
ACADEMIC_ANXIETY_SCORE_CAP = 0.55
LIMITED_SITUATIONAL_DISTRESS_SCORE_CAP = 0.60
SITUATIONAL_SADNESS_SCORE_CAP = 0.60
SITUATIONAL_ANXIETY_SURPRISE_SCORE_CAP = 0.55
EMOTIONAL_DISTRESS_FALSE_FEAR_SCORE_CAP = 0.70
ANGER_MARKER_SCORE_FLOOR = 0.60
SITUATIONAL_ANGER_SCORE_CAP = 0.60
HIGH_INTENSITY_ANGER_SCORE_FLOOR = 0.72
HIGH_INTENSITY_FEAR_SCORE_FLOOR = 0.72
INTENSIFIER_DELTA_GUARD_REASON = "intensifier_delta_cap"

INTENSIFIER_DELTA_ALLOWED_BY_TYPE = {
    UTTERANCE_TYPES["ROUTINE_DISCOMFORT"]: 0.08,
    UTTERANCE_TYPES["CASUAL_NEUTRAL"]: 0.06,
    UTTERANCE_TYPES["CASUAL_SHARE"]: 0.06,
    UTTERANCE_TYPES["POSITIVE_SHARE"]: 0.05,
    UTTERANCE_TYPES["PREFERENCE_QUESTION"]: 0.04,
    UTTERANCE_TYPES["PRACTICAL_QUESTION"]: 0.04,
}

_INTENSIFIER_TOKEN_PATTERN = re.compile(
    r"(?<![0-9A-Za-z가-힣])"
    r"(?:너무너무|너무|진짜로|진짜|정말로|정말|완전히|완전|엄청|되게|굉장히|매우|무척|몹시|아주|꽤)"
    r"(?:\s+|(?=[,.;:!?]))"
)

# RoBERTa 모델은 VRAM 제약 때문에 요청마다 언로드하지만, tokenizer는 CPU 객체라 재사용한다.
_TOKENIZER_CACHE = None


def attenuate_intensifiers(text: str) -> str:
    """
    역할: 저위험 발화의 의미 핵심은 유지하고 독립 강조 부사만 제거한다.
    입력: 사용자 발화 텍스트
    출력: 강조 부사가 약화된 텍스트
    """
    cleaned = str(text or "").strip()
    attenuated = _INTENSIFIER_TOKEN_PATTERN.sub("", cleaned)
    return " ".join(attenuated.split())


def has_intensifier_delta_candidate(text: str) -> bool:
    """
    역할: 강조어 약화 비교를 수행할 만한 독립 강조 부사가 있는지 확인한다.
    입력: 사용자 발화 텍스트
    출력: 강조어 후보 포함 여부
    """
    cleaned = str(text or "").strip()
    return bool(cleaned and attenuate_intensifiers(cleaned) != cleaned)


def is_intensifier_delta_cap_candidate(
    text: str,
    utterance_type: str | None,
    is_crisis: bool = False,
) -> bool:
    """
    역할: 강조어 하나로 점수가 튀면 안 되는 저위험 라우트인지 판별한다.
    입력: 분석 텍스트, 발화 타입, 위기 후보 여부
    출력: delta cap 비교 대상 여부
    """
    if is_crisis or utterance_type not in INTENSIFIER_DELTA_ALLOWED_BY_TYPE:
        return False
    if not has_intensifier_delta_candidate(text):
        return False
    if has_crisis_marker(text):
        return False
    if has_high_intensity_fear_marker(text) or has_high_intensity_anger_marker(text):
        return False
    if (
        is_academic_anxiety_text(text)
        or is_limited_situational_distress_text(text)
        or is_situational_sadness_text(text)
        or is_situational_anxiety_surprise_text(text)
        or is_situational_anger_text(text)
    ):
        return False
    if utterance_type == UTTERANCE_TYPES["ROUTINE_DISCOMFORT"]:
        return is_routine_discomfort_text(text)
    return True


def get_intensifier_allowed_delta(utterance_type: str | None) -> float:
    """
    역할: 발화 타입별 원문-강조 약화판 허용 점수 차이를 반환한다.
    입력: 발화 타입
    출력: 허용 delta
    """
    return float(INTENSIFIER_DELTA_ALLOWED_BY_TYPE.get(utterance_type, 0.0))


def apply_intensifier_delta_cap(
    text: str,
    current_score: float | None,
    attenuated_score: float | None,
    utterance_type: str | None,
    is_crisis: bool = False,
    attenuated_is_crisis: bool = False,
) -> tuple[float | None, str | None, dict | None]:
    """
    역할: 저위험 발화에서 원문 점수가 강조 약화판보다 과도하게 높으면 허용 delta로 제한한다.
    입력: 분석 텍스트, 현재 점수, 강조 약화판 점수, 발화 타입, 위기 후보 여부
    출력: (보정 점수, 보정 사유, 비교 메타데이터)
    """
    if current_score is None or attenuated_score is None:
        return current_score, None, None
    if not is_intensifier_delta_cap_candidate(text, utterance_type, is_crisis):
        return current_score, None, None
    if attenuated_is_crisis:
        return current_score, None, None

    attenuated_text = attenuate_intensifiers(text)
    allowed_delta = get_intensifier_allowed_delta(utterance_type)
    score_cap = min(1.0, float(attenuated_score) + allowed_delta)
    meta = {
        "intensifier_attenuated_text": attenuated_text,
        "intensifier_attenuated_score": float(attenuated_score),
        "intensifier_allowed_delta": allowed_delta,
        "intensifier_original_score": float(current_score),
        "intensifier_score_cap": score_cap,
    }
    if float(current_score) > score_cap:
        return score_cap, INTENSIFIER_DELTA_GUARD_REASON, meta
    return float(current_score), None, meta


def _resolve_ckpt_path(path_or_name: str | None, default_path: str) -> str:
    """
    역할: 선택 체크포인트 파일명을 CKPT_DIR 기준 경로로 해석한다.
    입력: 파일명/경로 또는 None, 기본 경로
    출력: 실제 파일 경로
    """
    if not path_or_name:
        return default_path
    if os.path.isabs(path_or_name):
        return path_or_name
    return os.path.join(CKPT_DIR, path_or_name)

def load_runtime_config(
    config_path: str = RUNTIME_CONFIG,
    t_path: str = T_RESULT,
) -> dict:
    """
    역할: RoBERTa 런타임 설정(T 값, 위기 threshold, P95, vector_T)을 파일에서 로드
    입력: 런타임 설정 파일 경로, temperature 결과 파일 경로
    출력: 설정 dict (vector_T_emotion 은 list[float] 또는 None)
    """
    config: dict = {
        "T_emotion": 1.0,
        "T_nli": 1.0,
        "vector_T_emotion": None,  # 7-dim list 또는 None (Vector Scaling 채택 시 사용)
        "emotion_logit_bias": None,  # 7-dim list 또는 None (선택 채택 시 logits에 더함)
        "crisis_threshold": DEFAULT_CRISIS_THRESHOLD,
        "roberta_score_p95": DEFAULT_ROBERTA_SCORE_P95,
    }

    if os.path.exists(t_path):
        with open(t_path, encoding="utf-8") as f:
            temp_data = json.load(f)
        config["T_emotion"] = float(temp_data.get("T_emotion", config["T_emotion"]))
        config["T_nli"] = float(temp_data.get("T_nli", config["T_nli"]))
        config["roberta_score_p95"] = float(
            temp_data.get("roberta_score_p95", config["roberta_score_p95"])
        )
        # Vector Scaling: 클래스별 T 벡터 (LABEL2EMOTION 순서) — 길이 7 list 만 채택.
        vec_T = temp_data.get("vector_T_emotion")
        if isinstance(vec_T, list) and len(vec_T) == len(LABEL2EMOTION):
            try:
                config["vector_T_emotion"] = [float(x) for x in vec_T]
            except (TypeError, ValueError):
                config["vector_T_emotion"] = None
        logit_bias = temp_data.get("emotion_logit_bias")
        if isinstance(logit_bias, list) and len(logit_bias) == len(LABEL2EMOTION):
            try:
                config["emotion_logit_bias"] = [float(x) for x in logit_bias]
            except (TypeError, ValueError):
                config["emotion_logit_bias"] = None

    if os.path.exists(config_path):
        with open(config_path, encoding="utf-8") as f:
            runtime_data = json.load(f)
        config["crisis_threshold"] = float(
            runtime_data.get("crisis_threshold", config["crisis_threshold"])
        )
        config["roberta_score_p95"] = float(
            runtime_data.get("roberta_score_p95", config["roberta_score_p95"])
        )
        logit_bias = runtime_data.get("emotion_logit_bias")
        if isinstance(logit_bias, list) and len(logit_bias) == len(LABEL2EMOTION):
            try:
                config["emotion_logit_bias"] = [float(x) for x in logit_bias]
            except (TypeError, ValueError):
                config["emotion_logit_bias"] = None

    return config


_RUNTIME_CONFIG = load_runtime_config()
CRISIS_THRESHOLD = float(_RUNTIME_CONFIG["crisis_threshold"])
ROBERTA_SCORE_P95 = float(_RUNTIME_CONFIG["roberta_score_p95"])
VECTOR_T_EMOTION = _RUNTIME_CONFIG.get("vector_T_emotion")  # list[float] | None
EMOTION_LOGIT_BIAS = _RUNTIME_CONFIG.get("emotion_logit_bias")  # list[float] | None


def get_crisis_threshold() -> float:
    """
    역할: runtime_config.json에서 crisis_threshold를 매 추론마다 동적으로 읽음
          (서버 재시작 없이 임계값 변경 반영 가능)
    출력: crisis_threshold (float)
    """
    try:
        with open(RUNTIME_CONFIG, encoding="utf-8") as f:
            data = json.load(f)
        return float(data.get("crisis_threshold", DEFAULT_CRISIS_THRESHOLD))
    except Exception:
        return DEFAULT_CRISIS_THRESHOLD


def load_temperature(t_path: str = T_RESULT) -> tuple:
    """
    역할: temperature_result.json에서 T_emotion, T_nli 로드 (단일 스칼라 호환 함수)
    입력: json 파일 경로
    출력: (T_emotion, T_nli) float 튜플
    참고: Vector Scaling 적용 여부와 무관하게 단일 T 값을 그대로 반환한다.
          벡터 보정이 필요한 호출부는 load_emotion_vector_T() 를 함께 사용한다.
    """
    config = load_runtime_config(t_path=t_path)
    if not os.path.exists(t_path):
        print("[경고] temperature_result.json 없음 — T=1.0 기본값 사용")
    return float(config["T_emotion"]), float(config["T_nli"])


def load_emotion_vector_T(t_path: str = T_RESULT) -> list | None:
    """
    역할: Vector Scaling 클래스별 T 벡터(7-dim, LABEL2EMOTION 순서)를 로드
    입력: json 파일 경로
    출력: list[float] 또는 미설정 시 None — 호출부는 None 인 경우 단일 T_emotion 으로 폴백
    """
    config = load_runtime_config(t_path=t_path)
    return config.get("vector_T_emotion")


def load_emotion_logit_bias(
    config_path: str = RUNTIME_CONFIG,
    t_path: str = T_RESULT,
) -> list | None:
    """
    역할: 선택 채택된 감정 logits additive bias를 로드한다.
    입력: runtime/temperature 설정 경로
    출력: 7-dim bias list 또는 미설정 시 None
    """
    env_path = os.environ.get(EMOTION_LOGIT_BIAS_ENV, "").strip()
    if env_path:
        try:
            with open(env_path, encoding="utf-8") as f:
                payload = json.load(f)
            bias = payload.get("emotion_logit_bias")
            if bias is None and isinstance(payload.get("selected"), dict):
                bias = payload["selected"].get("bias")
            if isinstance(bias, list) and len(bias) == len(LABEL2EMOTION):
                return [float(x) for x in bias]
        except Exception as exc:
            print(f"[경고] emotion logit bias env 로드 실패: {exc}")

    config = load_runtime_config(config_path=config_path, t_path=t_path)
    bias = config.get("emotion_logit_bias")
    if isinstance(bias, list) and len(bias) == len(LABEL2EMOTION):
        return [float(x) for x in bias]
    return None


def get_roberta_tokenizer(model_name: str = MODEL_NAME):
    """
    역할: RoBERTa 토크나이저를 프로세스 내에서 1회만 생성해 재사용
    입력: Hugging Face 모델 이름
    출력: 캐시된 AutoTokenizer 인스턴스
    """
    global _TOKENIZER_CACHE
    if _TOKENIZER_CACHE is None:
        _TOKENIZER_CACHE = AutoTokenizer.from_pretrained(model_name)
    return _TOKENIZER_CACHE


def load_roberta_model(
    ckpt_name: str = "roberta_final.pt",
    utterance_head_name: str | None = None,
    cbt_class_head_name: str | None = None,
    distress_head_name: str | None = None,
):
    """
    역할: RoBERTa 멀티태스크 모델과 토크나이저 로드
    입력: 체크포인트 파일명, 선택 head 체크포인트 파일명들
    출력: (model, tokenizer, device)
    """
    import sys
    sys.path.insert(0, os.path.join(BASE_DIR, "models", "roberta"))
    from train_roberta import (
        NUM_EMOTION_CLS,
        NUM_NLI_CLS,
        NUM_UTTERANCE_TYPE_CLS,
        RoBERTaMultiTask,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = get_roberta_tokenizer(MODEL_NAME)

    # 학습 코드와 동일한 생성자 시그니처를 맞춰 로드하되, 신규 utterance_type_head는 별도 체크포인트에서 채운다.
    model = RoBERTaMultiTask(
        MODEL_NAME,
        NUM_EMOTION_CLS,
        NUM_NLI_CLS,
        NUM_UTTERANCE_TYPE_CLS,
    ).to(device)
    ckpt_path = _resolve_ckpt_path(ckpt_name, os.path.join(CKPT_DIR, "roberta_final.pt"))
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=False)
    model._utterance_type_head_loaded = _load_utterance_type_head(model, device, utterance_head_name)
    model._cbt_class_head_loaded = _load_cbt_class_head(model, device, cbt_class_head_name)
    model._distress_head_loaded = _load_distress_head(model, device, distress_head_name)
    model.eval()
    print(f"[RoBERTa 로드] {ckpt_path} → {device}")
    return model, tokenizer, device


def _load_utterance_type_head(model, device, head_name: str | None = None) -> bool:
    """
    역할: 별도 저장된 발화 의도/타입 head 체크포인트가 있으면 모델에 로드한다.
    입력: RoBERTa 모델, torch device
    출력: head 로드 성공 여부
    """
    head_path = _resolve_ckpt_path(head_name, UTTERANCE_TYPE_HEAD_CKPT)
    if not os.path.exists(head_path):
        return False

    payload = torch.load(head_path, map_location=device, weights_only=True)
    head_state = payload.get("utterance_type_head")
    if not head_state:
        return False

    model.utterance_type_head.load_state_dict(head_state)
    model._utterance_type_label_map = payload.get("id_to_label", ID_TO_UTTERANCE_TYPE)
    print(f"[발화 의도 head 로드] {head_path}")
    return True


def _load_cbt_class_head(model, device, head_name: str | None = None) -> bool:
    """
    역할: 별도 저장된 CBT 10범주 분류 head 체크포인트(KoACD 학습) 가 있으면 로드한다.
          없으면 anchor argmax 폴백을 그대로 쓴다.
    입력: RoBERTa 모델, torch device
    출력: head 로드 성공 여부
    """
    head_path = _resolve_ckpt_path(head_name, CBT_CLASS_HEAD_CKPT)
    if not os.path.exists(head_path):
        return False

    payload = torch.load(head_path, map_location=device, weights_only=True)
    head_state = payload.get("cbt_class_head")
    if not head_state:
        return False

    model.cbt_class_head.load_state_dict(head_state)
    # id_to_label 은 라벨 매핑(int → 카테고리 이름) 형태로 저장됨
    raw_map = payload.get("id_to_label", {})
    # JSON 라운드트립 시 int 키가 str 로 바뀔 수 있으므로 정규화
    model._cbt_class_label_map = {int(k): v for k, v in raw_map.items()}
    model._cbt_class_best_f1 = float(payload.get("best_macro_f1", 0.0))
    print(f"[CBT 분류 head 로드] {head_path} "
          f"(best_macro_f1={model._cbt_class_best_f1:.4f})")
    return True


def _load_distress_head(model, device, head_name: str | None = None) -> bool:
    """
    역할: 별도 저장된 distress severity head(5클래스: calm/mild/moderate/high/crisis)를 로드해
          legacy 모델에 동적으로 attach한다 (Phase 5b — emotion proxy 대체용).
    입력: RoBERTa 모델, torch device
    출력: head 로드 성공 여부
    """
    head_path = _resolve_ckpt_path(head_name, DISTRESS_HEAD_CKPT)
    if not os.path.exists(head_path):
        return False

    payload = torch.load(head_path, map_location=device, weights_only=True)
    head_state = payload.get("distress_head")
    if not head_state:
        return False

    import torch.nn as nn
    encoder_module = getattr(model, "encoder", None) or getattr(model, "roberta", None)
    hidden_size = int(encoder_module.config.hidden_size) if encoder_module is not None else 768
    head = nn.Sequential(nn.Dropout(0.1), nn.Linear(hidden_size, 5)).to(device)
    head.load_state_dict(head_state)
    head.eval()
    model.distress_head = head

    raw_map = payload.get("id_to_label", DISTRESS_LABEL_MAP_DEFAULT)
    model._distress_label_map = {int(k): v for k, v in raw_map.items()}
    vt = payload.get("vector_T_distress")
    if vt is not None:
        model._distress_vector_T = torch.tensor(vt, dtype=torch.float32, device=device)
    else:
        model._distress_vector_T = None
    print(f"[distress head 로드] {head_path} "
          f"(vector_T={[round(float(x), 3) for x in vt] if vt else 'None'})")
    return True


@torch.no_grad()
def predict_distress_with_head(model, cls_emb) -> dict | None:
    """
    역할: 학습된 distress_head로 5클래스 distress severity 분포 + 0~1 스칼라를 산출.
    입력: RoBERTa 모델(distress_head attach 후), [batch=1, hidden] [CLS] 임베딩
    출력: {
        distress_probs: list[float] (5클래스, vector_T 적용 후 softmax)
        distress_top_label: str (argmax 라벨)
        distress_severity_scalar: float (0~1, crisis=0 이중카운팅 방지)
    } 또는 head 미로드 시 None
    """
    if not getattr(model, "_distress_head_loaded", False):
        return None
    logits = model.distress_head(cls_emb)
    vector_T = getattr(model, "_distress_vector_T", None)
    if vector_T is not None:
        probs = F.softmax(logits / vector_T, dim=-1).squeeze(0)
    else:
        probs = F.softmax(logits, dim=-1).squeeze(0)
    plist = [float(x) for x in probs.cpu().tolist()]
    label_map = getattr(model, "_distress_label_map", DISTRESS_LABEL_MAP_DEFAULT)
    top_idx = int(probs.argmax().item())
    top_label = label_map.get(top_idx, str(top_idx))
    # severity_scalar: crisis는 NLI/하드 인터럽트가 처리하므로 0
    severity = float(0.0 * plist[0] + 0.25 * plist[1] + 0.50 * plist[2]
                     + 0.85 * plist[3] + 0.0 * plist[4])
    return {
        "distress_probs": plist,
        "distress_top_label": top_label,
        "distress_severity_scalar": severity,
    }


@torch.no_grad()
def predict_cbt_class_with_head(text: str, model, tokenizer, device) -> dict | None:
    """
    역할: 학습된 cbt_class_head 로 발화의 CBT 분류를 예측한다 (KoACD 학습).
          v2(11클래스, 2026-04-27): 10범주 왜곡 + 1범주 "비왜곡" 동시 예측.
          - is_non_distortion=True 면 head 가 비왜곡으로 판정 → 카테고리 보존 안 함
          - distortion_max_prob: 왜곡 카테고리 10개 중 max softmax 확률
            (이진 감지용 — anchor cbt_score 와 앙상블 가능)
          - top_distortion_label/confidence: 왜곡 중 argmax (비왜곡이 더 높아도 별도 보존)
    입력: 발화 텍스트, 모델, 토크나이저, device
    출력: {
        label,                # 전체 클래스(0~10) 중 argmax 라벨 이름 (비왜곡 포함)
        confidence,           # label 의 softmax 확률
        is_non_distortion,    # label == "비왜곡" 여부
        top_distortion_label, # 왜곡 카테고리 0~9 중 argmax 라벨 이름
        top_distortion_confidence,  # 위의 softmax 확률
        distortion_max_prob,  # max(p_0..p_9) — 이진 감지 신호
        all_probs,            # 11개 raw 확률 리스트
    } 또는 head 미로드 시 None
    """
    if not getattr(model, "_cbt_class_head_loaded", False):
        return None

    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding="max_length",
    )
    input_ids = enc["input_ids"].to(device)
    attn_mask = enc["attention_mask"].to(device)
    logits = model.forward_cbt_class(input_ids, attn_mask)
    probs = F.softmax(logits, dim=-1)[0]
    pred_id = int(torch.argmax(probs).item())
    confidence = float(probs[pred_id].item())

    label_map = getattr(model, "_cbt_class_label_map", {})
    label = label_map.get(pred_id) or label_map.get(str(pred_id))
    if label is None:
        return None

    # v2 (11클래스): 비왜곡 클래스 처리. id_to_label 에 "비왜곡" 이 있는 schema.
    is_non_distortion = (label == "비왜곡")
    # 왜곡 카테고리(비왜곡 제외) 중 argmax 와 max prob 계산.
    distortion_ids = [i for i in label_map if (label_map[i] != "비왜곡")]
    if distortion_ids:
        distortion_probs = [(i, float(probs[i].item())) for i in distortion_ids]
        top_d_id, top_d_prob = max(distortion_probs, key=lambda x: x[1])
        top_distortion_label = label_map.get(top_d_id) or label_map.get(str(top_d_id))
        distortion_max_prob = top_d_prob
        top_distortion_confidence = top_d_prob
    else:
        # 구버전(10클래스 only) 호환 — 모든 클래스가 왜곡
        top_distortion_label = label
        top_distortion_confidence = confidence
        distortion_max_prob = confidence

    return {
        "label": label,
        "confidence": confidence,
        "is_non_distortion": is_non_distortion,
        "top_distortion_label": top_distortion_label,
        "top_distortion_confidence": top_distortion_confidence,
        "distortion_max_prob": distortion_max_prob,
        "all_probs": probs.cpu().tolist(),
    }


@torch.no_grad()
def predict_utterance_type_with_head(text: str, model, tokenizer, device) -> dict | None:
    """
    역할: RoBERTa 발화 의도/타입 head로 사용자 발화를 7클래스 분류한다.
    입력: 사용자 발화, 모델, tokenizer, device
    출력: 발화 타입 dict 또는 head 미로드 시 None
    """
    if not getattr(model, "_utterance_type_head_loaded", False):
        return None

    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding="max_length",
    )
    input_ids = enc["input_ids"].to(device)
    attn_mask = enc["attention_mask"].to(device)
    logits = model.forward_utterance_type(input_ids, attn_mask)
    probs = F.softmax(logits, dim=-1)[0]
    pred_id = int(torch.argmax(probs).item())
    confidence = float(probs[pred_id].item())

    id_to_label = getattr(model, "_utterance_type_label_map", ID_TO_UTTERANCE_TYPE)
    label = id_to_label.get(pred_id) or id_to_label.get(str(pred_id)) or ID_TO_UTTERANCE_TYPE[pred_id]
    if confidence < UTTERANCE_TYPE_CONFIDENCE_FLOOR:
        label = UTTERANCE_TYPES["CASUAL_SHARE"]

    return {
        "utterance_type": label,
        "type_confidence": confidence,
        "type_reason": "roberta_utterance_intent_head",
    }


def apply_utterance_type_adjustment(
    text: str,
    top_emotion: str,
    roberta_score: float,
    utterance_info: dict | None = None,
) -> tuple[str, float, str | None, dict]:
    """
    역할: 발화 타입에 따라 감정 표시와 RoBERTa 점수 과대평가를 보정한다.
    입력: 사용자 발화, 모델 1순위 감정, 정규화된 점수, 선택적 발화 타입 정보
    출력: (보정 감정, 보정 점수, 보정 사유, 발화 타입 정보)
    """
    utterance_info = utterance_info or classify_utterance_type(text)

    # 헤드가 preference_question으로 분류했지만 비교 맥락 단서가 없으면 규칙 기반으로 재분류
    # (예: "오늘 뭐할까" 같은 일상 질문이 preference_question으로 오탐되는 케이스 방어)
    _PREF_SIGNALS = ["중에", "아니면", "둘 중", "뭐가 더", "뭐가 나아", "뭐가 좋아", "고르면", "골라", "중 뭐"]
    if (
        utterance_info.get("utterance_type") == UTTERANCE_TYPES["PREFERENCE_QUESTION"]
        and utterance_info.get("type_reason") == "roberta_utterance_intent_head"
        and not any(s in text for s in _PREF_SIGNALS)
    ):
        utterance_info = classify_utterance_type(text)

    practical_request = is_practical_anxiety_relief_text(text) or is_practical_question_text(text)
    if (
        utterance_info["utterance_type"] != UTTERANCE_TYPES["CRISIS_CANDIDATE"]
        and practical_request
    ):
        utterance_info = {
            "utterance_type": UTTERANCE_TYPES["PRACTICAL_QUESTION"],
            "type_confidence": max(float(utterance_info.get("type_confidence", 0.0)), 0.80),
            "type_reason": "practical_question_override",
        }
    if is_laughter_only_text(text):
        utterance_info = {
            "utterance_type": UTTERANCE_TYPES["POSITIVE_SHARE"],
            "type_confidence": max(float(utterance_info.get("type_confidence", 0.0)), 0.86),
            "type_reason": "laughter_only_positive_low_signal_override",
        }
    if is_academic_anxiety_text(text) and not practical_request:
        utterance_info = {
            "utterance_type": UTTERANCE_TYPES["EMOTIONAL_DISTRESS"],
            "type_confidence": max(float(utterance_info.get("type_confidence", 0.0)), 0.84),
            "type_reason": "academic_anxiety_override",
        }
    if has_high_intensity_fear_marker(text):
        utterance_info = {
            "utterance_type": UTTERANCE_TYPES["EMOTIONAL_DISTRESS"],
            "type_confidence": max(float(utterance_info.get("type_confidence", 0.0)), 0.88),
            "type_reason": "high_intensity_fear_override",
        }
    if is_low_risk_sensory_disgust_text(text):
        utterance_info = {
            "utterance_type": UTTERANCE_TYPES["CASUAL_NEUTRAL"],
            "type_confidence": max(float(utterance_info.get("type_confidence", 0.0)), 0.80),
            "type_reason": "sensory_disgust_low_impact_override",
        }
    elif is_sensory_disgust_text(text):
        utterance_info = {
            "utterance_type": UTTERANCE_TYPES["EMOTIONAL_DISTRESS"],
            "type_confidence": max(float(utterance_info.get("type_confidence", 0.0)), 0.80),
            "type_reason": "sensory_disgust_override",
        }
    if is_mild_unease_text(text):
        utterance_info = {
            "utterance_type": UTTERANCE_TYPES["EMOTIONAL_DISTRESS"],
            "type_confidence": max(float(utterance_info.get("type_confidence", 0.0)), 0.74),
            "type_reason": "mild_unease_override",
        }
    if is_mild_low_mood_text(text):
        utterance_info = {
            "utterance_type": UTTERANCE_TYPES["EMOTIONAL_DISTRESS"],
            "type_confidence": max(float(utterance_info.get("type_confidence", 0.0)), 0.76),
            "type_reason": "mild_low_mood_override",
        }
    if is_limited_situational_distress_text(text):
        utterance_info = {
            "utterance_type": UTTERANCE_TYPES["EMOTIONAL_DISTRESS"],
            "type_confidence": max(float(utterance_info.get("type_confidence", 0.0)), 0.76),
            "type_reason": "limited_situational_distress_override",
        }
    if is_interpersonal_remorse_text(text):
        utterance_info = {
            "utterance_type": UTTERANCE_TYPES["EMOTIONAL_DISTRESS"],
            "type_confidence": max(float(utterance_info.get("type_confidence", 0.0)), 0.80),
            "type_reason": "interpersonal_remorse_override",
        }
    if is_situational_sadness_text(text):
        utterance_info = {
            "utterance_type": UTTERANCE_TYPES["EMOTIONAL_DISTRESS"],
            "type_confidence": max(float(utterance_info.get("type_confidence", 0.0)), 0.78),
            "type_reason": "situational_sadness_override",
        }
    if has_anger_marker(text):
        utterance_info = {
            "utterance_type": UTTERANCE_TYPES["EMOTIONAL_DISTRESS"],
            "type_confidence": max(float(utterance_info.get("type_confidence", 0.0)), 0.84),
            "type_reason": "anger_marker_override",
        }
    if has_distress_marker(text):
        utterance_info = {
            "utterance_type": UTTERANCE_TYPES["EMOTIONAL_DISTRESS"],
            "type_confidence": max(float(utterance_info.get("type_confidence", 0.0)), 0.78),
            "type_reason": "distress_marker_override",
        }
    if is_physical_exertion_text(text):
        utterance_info = {
            "utterance_type": UTTERANCE_TYPES["CASUAL_NEUTRAL"],
            "type_confidence": max(float(utterance_info.get("type_confidence", 0.0)), 0.80),
            "type_reason": "physical_exertion_override",
        }
    if (
        is_routine_discomfort_text(text)
        and not is_physical_exertion_text(text)
        and not is_limited_situational_distress_text(text)
        and not is_academic_anxiety_text(text)
    ):
        # 공부·출근 같은 일상 과업 피로는 낮은 신뢰도의 head 오분류보다 규칙 신호를 우선한다.
        utterance_info = {
            "utterance_type": UTTERANCE_TYPES["ROUTINE_DISCOMFORT"],
            "type_confidence": max(float(utterance_info.get("type_confidence", 0.0)), 0.82),
            "type_reason": "routine_discomfort_override",
        }
    if is_daily_routine_neutral_text(text):
        utterance_info = {
            "utterance_type": UTTERANCE_TYPES["CASUAL_NEUTRAL"],
            "type_confidence": max(float(utterance_info.get("type_confidence", 0.0)), 0.78),
            "type_reason": "daily_routine_neutral_override",
        }
    if is_administrative_technical_neutral_text(text):
        utterance_info = {
            "utterance_type": UTTERANCE_TYPES["CASUAL_NEUTRAL"],
            "type_confidence": max(float(utterance_info.get("type_confidence", 0.0)), 0.78),
            "type_reason": "administrative_technical_neutral_override",
        }
    if is_situational_anxiety_surprise_text(text):
        utterance_info = {
            "utterance_type": UTTERANCE_TYPES["EMOTIONAL_DISTRESS"],
            "type_confidence": max(float(utterance_info.get("type_confidence", 0.0)), 0.78),
            "type_reason": "situational_anxiety_surprise_override",
        }
    if is_low_risk_sensory_disgust_text(text):
        # 감각 혐오는 "찝찝/비위" 같은 단어가 broad distress 규칙에 걸려도 저영향 라우팅을 유지한다.
        utterance_info = {
            "utterance_type": UTTERANCE_TYPES["CASUAL_NEUTRAL"],
            "type_confidence": max(float(utterance_info.get("type_confidence", 0.0)), 0.80),
            "type_reason": "sensory_disgust_low_impact_override",
        }
    if (
        is_positive_affect_text(text)
        and utterance_info["utterance_type"] != UTTERANCE_TYPES["CRISIS_CANDIDATE"]
    ):
        # 명시 긍정/회복 발화(부정·위기 안전신호 없음)는 head가 routine_discomfort 등
        # full-impact 타입으로 오분류해도 positive_share로 보정한다.
        # 예: "길고양이가 귀여웠어" → head는 routine_discomfort로 오분류하지만
        #     감정은 행복(아래 802)으로 보존하고, 웰니스도 full이 아니라 low로 반영한다.
        utterance_info = {
            "utterance_type": UTTERANCE_TYPES["POSITIVE_SHARE"],
            "type_confidence": max(float(utterance_info.get("type_confidence", 0.0)), 0.80),
            "type_reason": "positive_affect_share_override",
        }
    utterance_type = utterance_info["utterance_type"]

    if utterance_type == UTTERANCE_TYPES["CRISIS_CANDIDATE"]:
        return top_emotion, roberta_score, None, utterance_info

    if is_laughter_only_text(text):
        return (
            "행복",
            min(roberta_score, POSITIVE_AFFECT_SCORE_CAP),
            "laughter_only_positive_cap",
            utterance_info,
        )

    if is_positive_affect_text(text):
        return (
            "행복",
            min(roberta_score, POSITIVE_AFFECT_SCORE_CAP),
            "positive_affect_emotion_preserve",
            utterance_info,
        )

    if is_sensory_disgust_text(text):
        return (
            "혐오",
            min(max(roberta_score, SENSORY_DISGUST_SCORE_FLOOR), SENSORY_DISGUST_SCORE_CAP),
            "sensory_disgust_cap",
            utterance_info,
        )

    if is_situational_anxiety_surprise_text(text) and not has_anger_marker(text):
        if has_fear_marker(text) or has_situational_anxiety_marker(text):
            adjusted_emotion = "공포"
        else:
            adjusted_emotion = "놀람"
        return (
            adjusted_emotion,
            min(roberta_score, SITUATIONAL_ANXIETY_SURPRISE_SCORE_CAP),
            "situational_anxiety_surprise_cap",
            utterance_info,
        )

    if is_physical_exertion_text(text):
        return (
            "중립",
            min(roberta_score, CASUAL_NEUTRAL_SCORE_CAP),
            "physical_exertion_cap",
            utterance_info,
        )

    if utterance_type == UTTERANCE_TYPES["ROUTINE_DISCOMFORT"]:
        return (
            "중립",
            min(roberta_score, ROUTINE_DISCOMFORT_SCORE_CAP),
            "routine_discomfort_cap",
            utterance_info,
        )

    if is_daily_routine_neutral_text(text):
        return (
            "중립",
            min(roberta_score, CASUAL_NEUTRAL_SCORE_CAP),
            "daily_routine_neutral_cap",
            utterance_info,
        )

    if is_administrative_technical_neutral_text(text):
        return (
            "중립",
            min(roberta_score, CASUAL_NEUTRAL_SCORE_CAP),
            "administrative_technical_neutral_cap",
            utterance_info,
        )

    if utterance_type in {
        UTTERANCE_TYPES["CASUAL_NEUTRAL"],
        UTTERANCE_TYPES["CASUAL_SHARE"],
        UTTERANCE_TYPES["POSITIVE_SHARE"],
        UTTERANCE_TYPES["PREFERENCE_QUESTION"],
        UTTERANCE_TYPES["PRACTICAL_QUESTION"],
    }:
        if top_emotion in {"공포", "혐오", "분노", "슬픔", "놀람"}:
            reason = (
                "casual_neutral_surprise_cap"
                if top_emotion == "놀람"
                else "casual_neutral_negative_cap"
            )
            return (
                "중립",
                min(roberta_score, CASUAL_NEUTRAL_SCORE_CAP),
                reason,
                utterance_info,
            )
        return top_emotion, roberta_score, None, utterance_info

    if utterance_type == UTTERANCE_TYPES["EMOTIONAL_DISTRESS"]:
        if has_anger_marker(text):
            if is_situational_anger_text(text):
                return (
                    "분노",
                    min(max(roberta_score, ANGER_MARKER_SCORE_FLOOR), SITUATIONAL_ANGER_SCORE_CAP),
                    "situational_anger_cap",
                    utterance_info,
                )
            if has_high_intensity_anger_marker(text):
                return (
                    "분노",
                    max(roberta_score, HIGH_INTENSITY_ANGER_SCORE_FLOOR),
                    "high_intensity_anger_override",
                    utterance_info,
                )
            return (
                "분노",
                max(roberta_score, ANGER_MARKER_SCORE_FLOOR),
                "anger_marker_override",
                utterance_info,
            )
        if is_situational_sadness_text(text):
            return (
                "슬픔",
                min(max(roberta_score, 0.55), SITUATIONAL_SADNESS_SCORE_CAP),
                "situational_sadness_cap",
                utterance_info,
            )
        if is_interpersonal_remorse_text(text):
            return (
                "슬픔",
                min(max(roberta_score, 0.55), SITUATIONAL_SADNESS_SCORE_CAP),
                "interpersonal_remorse_cap",
                utterance_info,
            )
        if has_distress_marker(text) and top_emotion in {"중립", "행복", "놀람", "혐오"}:
            return (
                "슬픔",
                max(roberta_score, 0.55),
                "distress_marker_emotion_override",
                utterance_info,
            )
        if is_academic_anxiety_text(text):
            return (
                "공포" if top_emotion in {"중립", "행복", "놀람"} else top_emotion,
                min(roberta_score, ACADEMIC_ANXIETY_SCORE_CAP),
                "academic_anxiety_cap",
                utterance_info,
            )
        if is_limited_situational_distress_text(text):
            return (
                "슬픔" if top_emotion in {"중립", "행복", "놀람"} else top_emotion,
                min(roberta_score, LIMITED_SITUATIONAL_DISTRESS_SCORE_CAP),
                "limited_situational_distress_cap",
                utterance_info,
            )
        if is_situational_anxiety_surprise_text(text):
            if has_fear_marker(text) or has_situational_anxiety_marker(text):
                adjusted_emotion = "공포"
            else:
                adjusted_emotion = "놀람"
            return (
                adjusted_emotion,
                min(roberta_score, SITUATIONAL_ANXIETY_SURPRISE_SCORE_CAP),
                "situational_anxiety_surprise_cap",
                utterance_info,
            )
        if has_high_intensity_fear_marker(text):
            return (
                "공포" if top_emotion in {"중립", "행복", "놀람", "슬픔"} else top_emotion,
                max(roberta_score, HIGH_INTENSITY_FEAR_SCORE_FLOOR),
                "high_intensity_fear_override",
                utterance_info,
            )
        # EMOTIONAL_DISTRESS인데 RoBERTa가 행복으로 오분류한 경우 슬픔으로 보정
        if top_emotion == "행복":
            return (
                "슬픔",
                min(roberta_score, 0.55),
                "emotional_distress_false_happy_cap",
                utterance_info,
            )
        if is_mild_unease_text(text) and top_emotion in {"공포", "혐오", "분노", "슬픔", "놀람"}:
            return (
                "중립",
                min(roberta_score, ROUTINE_DISCOMFORT_SCORE_CAP),
                "mild_unease_cap",
                utterance_info,
            )
        if is_mild_low_mood_text(text) and top_emotion in {"공포", "혐오", "분노", "놀람", "슬픔"}:
            return (
                "슬픔",
                min(roberta_score, 0.55),
                "mild_low_mood_cap",
                utterance_info,
            )
        if (
            top_emotion == "공포"
            and not has_fear_marker(text)
            and not has_high_intensity_fear_marker(text)
            and not has_situational_anxiety_marker(text)
        ):
            return (
                "슬픔",
                min(roberta_score, EMOTIONAL_DISTRESS_FALSE_FEAR_SCORE_CAP),
                "emotional_distress_false_fear_cap",
                utterance_info,
            )
        return top_emotion, roberta_score, None, utterance_info

    return top_emotion, roberta_score, None, utterance_info


def apply_positive_affect_guard(
    text: str,
    top_emotion: str,
    roberta_score: float,
    entailment_prob: float,
    is_crisis: bool,
    utterance_info: dict,
) -> tuple[float, float, bool, str | None]:
    """
    역할: 긍정·회복 발화가 NLI/감정 점수 오탐으로 위험권에 올라가지 않도록 제한한다.
    입력: 사용자 발화, 1순위 감정, RoBERTa 점수, NLI entailment 확률, 위기 후보 여부, 발화 타입 정보
    출력: (보정 RoBERTa 점수, 보정 entailment 확률, 보정 위기 후보 여부, 보정 사유)
    """
    utterance_type = utterance_info.get("utterance_type")
    type_conf = float(utterance_info.get("type_confidence", 0.0) or 0.0)
    has_negative_signal = has_negative_safety_signal(text)
    is_positive_type = (
        utterance_type == UTTERANCE_TYPES["POSITIVE_SHARE"]
        and type_conf >= 0.70
        and not has_negative_signal
    )
    is_laughter_only = is_laughter_only_text(text)
    is_positive_text = is_positive_affect_text(text)
    is_positive_emotion = top_emotion == "행복" and is_positive_text
    low_signal_chat_types = {
        UTTERANCE_TYPES["CASUAL_NEUTRAL"],
        UTTERANCE_TYPES["CASUAL_SHARE"],
        UTTERANCE_TYPES["POSITIVE_SHARE"],
        UTTERANCE_TYPES["ROUTINE_DISCOMFORT"],
        UTTERANCE_TYPES["PREFERENCE_QUESTION"],
        UTTERANCE_TYPES["PRACTICAL_QUESTION"],
    }
    is_low_signal_chat = (
        utterance_type in low_signal_chat_types
        and (
            is_administrative_technical_neutral_text(text)
            or not has_negative_signal
        )
    )

    # 직접 위기 문구가 있으면 긍정 단어가 섞여 있어도 안전 판정을 절대 낮추지 않는다.
    if has_crisis_marker(text):
        return roberta_score, entailment_prob, is_crisis, None

    if is_laughter_only:
        adjusted_score = min(roberta_score, POSITIVE_AFFECT_SCORE_CAP)
        adjusted_entailment = min(entailment_prob, POSITIVE_AFFECT_NLI_CAP)
        return adjusted_score, adjusted_entailment, False, "laughter_only_safety_cap"

    if is_physical_exertion_text(text) and is_crisis:
        adjusted_entailment = min(entailment_prob, POSITIVE_AFFECT_NLI_CAP)
        return roberta_score, adjusted_entailment, False, "physical_exertion_nli_cap"

    if (
        is_routine_discomfort_text(text)
        and not is_physical_exertion_text(text)
        and not is_limited_situational_distress_text(text)
        and not is_academic_anxiety_text(text)
        and not is_situational_sadness_text(text)
        and not is_situational_anxiety_surprise_text(text)
        and not is_situational_anger_text(text)
        and is_crisis
    ):
        adjusted_entailment = min(entailment_prob, POSITIVE_AFFECT_NLI_CAP)
        return roberta_score, adjusted_entailment, False, "routine_discomfort_nli_cap"

    if (is_practical_anxiety_relief_text(text) or is_practical_question_text(text)) and is_crisis:
        adjusted_entailment = min(entailment_prob, POSITIVE_AFFECT_NLI_CAP)
        return roberta_score, adjusted_entailment, False, "practical_question_nli_cap"

    if is_low_risk_sensory_disgust_text(text) and is_crisis:
        adjusted_entailment = min(entailment_prob, POSITIVE_AFFECT_NLI_CAP)
        return roberta_score, adjusted_entailment, False, "sensory_disgust_nli_cap"

    if is_situational_sadness_text(text) and is_crisis:
        adjusted_entailment = min(entailment_prob, POSITIVE_AFFECT_NLI_CAP)
        return roberta_score, adjusted_entailment, False, "situational_sadness_nli_cap"

    if is_interpersonal_remorse_text(text) and is_crisis:
        adjusted_entailment = min(entailment_prob, POSITIVE_AFFECT_NLI_CAP)
        return roberta_score, adjusted_entailment, False, "interpersonal_remorse_nli_cap"

    if is_limited_situational_distress_text(text) and is_crisis:
        adjusted_entailment = min(entailment_prob, POSITIVE_AFFECT_NLI_CAP)
        return roberta_score, adjusted_entailment, False, "limited_situational_distress_nli_cap"

    if is_situational_anxiety_surprise_text(text) and is_crisis:
        adjusted_entailment = min(entailment_prob, POSITIVE_AFFECT_NLI_CAP)
        return roberta_score, adjusted_entailment, False, "situational_anxiety_surprise_nli_cap"

    if is_positive_type or is_positive_emotion or is_positive_text:
        adjusted_score = min(roberta_score, POSITIVE_AFFECT_SCORE_CAP)
        adjusted_entailment = min(entailment_prob, POSITIVE_AFFECT_NLI_CAP)
        return adjusted_score, adjusted_entailment, False, "positive_affect_safety_cap"

    if is_low_signal_chat and is_crisis:
        adjusted_entailment = min(entailment_prob, POSITIVE_AFFECT_NLI_CAP)
        return roberta_score, adjusted_entailment, False, "low_signal_chat_nli_cap"

    return roberta_score, entailment_prob, is_crisis, None


def apply_emotion_sanity_guard(
    text: str,
    top_emotion: str,
    roberta_score: float,
) -> tuple[str, float, str | None]:
    """
    역할: 기존 테스트·호출부 호환을 위해 발화 타입 보정 결과 중 감정/점수/사유만 반환한다.
    입력: 사용자 발화, 모델 1순위 감정, 정규화된 점수
    출력: (보정 감정, 보정 점수, 보정 사유)
    """
    adjusted_emotion, adjusted_score, reason, _ = apply_utterance_type_adjustment(
        text,
        top_emotion,
        roberta_score,
    )
    return adjusted_emotion, adjusted_score, reason


@torch.no_grad()
def _infer_roberta_core_outputs(
    analysis_text: str,
    model,
    tokenizer,
    device,
    T_emotion: float = 1.0,
    T_nli: float = 1.0,
    p95: float = ROBERTA_SCORE_P95,
    vector_T_emotion: list | None = None,
    emotion_logit_bias: list | None = None,
) -> dict:
    """
    역할: 후처리 전 RoBERTa 감정/NLI/distress 핵심 출력을 계산한다.
    입력: 분석 텍스트, 모델, 토크나이저, 디바이스, 온도/P95 설정, 선택 logit bias
    출력: 감정 확률, NLI 확률, 정규화 점수, 위기 후보, distress 정보 dict
    """
    enc = tokenizer(
        analysis_text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding="max_length",
    )
    input_ids = enc["input_ids"].to(device)
    attn_mask  = enc["attention_mask"].to(device)

    emotion_logits, nli_logits = model(input_ids, attn_mask)

    # Phase 5b — distress severity head (있으면 [CLS] 임베딩으로 별도 forward)
    distress_info = None
    if getattr(model, "_distress_head_loaded", False):
        try:
            # RoBERTaMultiTask는 self.encoder 사용 (legacy 호환). SemanticEmotionRoBERTa도 self.encoder.
            encoder_module = getattr(model, "encoder", None) or getattr(model, "roberta", None)
            if encoder_module is not None:
                cls_emb = encoder_module(input_ids=input_ids, attention_mask=attn_mask).last_hidden_state[:, 0, :]
                distress_info = predict_distress_with_head(model, cls_emb)
        except Exception as e:
            print(f"[distress head infer 실패] {e}")
            distress_info = None

    # Temperature Scaling 후 softmax — vector_T_emotion 가 주어지면 클래스별 보정 우선
    if vector_T_emotion is not None and len(vector_T_emotion) == len(LABEL2EMOTION):
        T_vec = torch.tensor(
            vector_T_emotion, dtype=emotion_logits.dtype, device=emotion_logits.device,
        )
        emotion_scores = emotion_logits / T_vec
    else:
        emotion_scores = emotion_logits / T_emotion
    if emotion_logit_bias is not None and len(emotion_logit_bias) == len(LABEL2EMOTION):
        bias_vec = torch.tensor(
            emotion_logit_bias, dtype=emotion_scores.dtype, device=emotion_scores.device,
        )
        emotion_scores = emotion_scores + bias_vec
    emotion_probs = F.softmax(emotion_scores, dim=-1).cpu().numpy()[0]
    nli_probs     = F.softmax(nli_logits    / T_nli,     dim=-1).cpu().numpy()[0]

    # 감정 가중치 기반 raw depression score
    raw_score = float(sum(
        emotion_probs[i] * EMOTION_WEIGHT[LABEL2EMOTION[i]]
        for i in range(len(LABEL2EMOTION))
    ))

    # P95 정규화 → 0~1 클리핑
    roberta_score = float(np.clip(raw_score / max(p95, 1e-6), 0.0, 1.0))

    # NLI entailment(인덱스 0) > 임계값 → 위기
    # 매 추론마다 파일에서 동적으로 읽어 서버 재시작 없이 임계값 변경 가능
    raw_entailment_prob = float(nli_probs[0])
    entailment_prob = raw_entailment_prob
    is_crisis = entailment_prob > get_crisis_threshold()

    top_emotion = LABEL2EMOTION[int(emotion_probs.argmax())]
    return {
        "roberta_score": roberta_score,
        "is_crisis": is_crisis,
        "emotion_probs": emotion_probs,
        "top_emotion": top_emotion,
        "nli_probs": nli_probs,
        "entailment_prob": entailment_prob,
        "raw_entailment_prob": raw_entailment_prob,
        "distress_info": distress_info,
    }


@torch.no_grad()
def infer_single(
    text: str,
    model,
    tokenizer,
    device,
    T_emotion: float = 1.0,
    T_nli: float = 1.0,
    p95: float = ROBERTA_SCORE_P95,
    vector_T_emotion: list | None = None,
    emotion_logit_bias: list | None = None,
) -> dict:
    """
    역할: 단일 발화 추론 → roberta_score, 위기 여부, 감정 확률 반환
    입력: 발화 텍스트, 모델, 토크나이저, 디바이스, 온도 값 2개, P95 기준값,
          vector_T_emotion(7-dim Vector Scaling, None 이면 단일 T_emotion 사용),
          emotion_logit_bias(7-dim additive bias, None 이면 미적용)
    출력: {
        roberta_score: float (0~1),
        is_crisis: bool,
        emotion_probs: list[float] (7클래스),
        top_emotion: str,
        nli_probs: list[float] (3클래스),
        entailment_prob: float,
    }
    """
    analysis_text = normalize_emotion_analysis_text(text)
    core = _infer_roberta_core_outputs(
        analysis_text,
        model,
        tokenizer,
        device,
        T_emotion=T_emotion,
        T_nli=T_nli,
        p95=p95,
        vector_T_emotion=vector_T_emotion,
        emotion_logit_bias=emotion_logit_bias,
    )

    roberta_score = core["roberta_score"]
    is_crisis = core["is_crisis"]
    emotion_probs = core["emotion_probs"]
    top_emotion = core["top_emotion"]
    nli_probs = core["nli_probs"]
    entailment_prob = core["entailment_prob"]
    raw_entailment_prob = core["raw_entailment_prob"]
    distress_info = core["distress_info"]
    utterance_info = predict_utterance_type_with_head(analysis_text, model, tokenizer, device)
    top_emotion, roberta_score, emotion_guard, utterance_info = apply_utterance_type_adjustment(
        analysis_text,
        top_emotion,
        roberta_score,
        utterance_info=utterance_info,
    )
    roberta_score, entailment_prob, is_crisis, nli_guard = apply_positive_affect_guard(
        analysis_text,
        top_emotion,
        roberta_score,
        entailment_prob,
        is_crisis,
        utterance_info,
    )
    intensifier_guard = None
    intensifier_meta = None
    if is_intensifier_delta_cap_candidate(
        analysis_text,
        utterance_info.get("utterance_type"),
        is_crisis,
    ):
        attenuated_text = attenuate_intensifiers(analysis_text)
        attenuated_core = _infer_roberta_core_outputs(
            attenuated_text,
            model,
            tokenizer,
            device,
            T_emotion=T_emotion,
            T_nli=T_nli,
            p95=p95,
            vector_T_emotion=vector_T_emotion,
            emotion_logit_bias=emotion_logit_bias,
        )
        attenuated_info = predict_utterance_type_with_head(attenuated_text, model, tokenizer, device)
        attenuated_top_emotion, attenuated_score, _, attenuated_info = apply_utterance_type_adjustment(
            attenuated_text,
            attenuated_core["top_emotion"],
            attenuated_core["roberta_score"],
            utterance_info=attenuated_info,
        )
        attenuated_score, _, attenuated_is_crisis, _ = apply_positive_affect_guard(
            attenuated_text,
            attenuated_top_emotion,
            attenuated_score,
            attenuated_core["entailment_prob"],
            attenuated_core["is_crisis"],
            attenuated_info,
        )
        roberta_score, intensifier_guard, intensifier_meta = apply_intensifier_delta_cap(
            analysis_text,
            roberta_score,
            attenuated_score,
            utterance_info.get("utterance_type"),
            is_crisis=is_crisis,
            attenuated_is_crisis=attenuated_is_crisis,
        )

    result = {
        "roberta_score":   roberta_score,
        "is_crisis":       is_crisis,
        "emotion_probs":   emotion_probs.tolist(),
        "top_emotion":     top_emotion,
        "emotion_guard":   emotion_guard,
        "nli_guard":       nli_guard,
        "utterance_type":   utterance_info["utterance_type"],
        "utterance_type_confidence": utterance_info["type_confidence"],
        "utterance_type_reason": utterance_info["type_reason"],
        "nli_probs":       nli_probs.tolist(),
        "entailment_prob": entailment_prob,
        "raw_entailment_prob": raw_entailment_prob,
        "analysis_text":    analysis_text,
        "analysis_text_changed": analysis_text != str(text or "").strip(),
        "intensifier_delta_guard": intensifier_guard,
    }
    if intensifier_meta is not None:
        result.update(intensifier_meta)
    # Phase 5b — distress prob 5클래스 + severity 스칼라 (head 로드 시에만)
    if distress_info is not None:
        result["distress_probs"] = distress_info["distress_probs"]
        result["distress_top_label"] = distress_info["distress_top_label"]
        result["distress_severity_scalar"] = distress_info["distress_severity_scalar"]
    return result


def measure_score_p95(
    model,
    tokenizer,
    device,
    T_emotion: float = 1.0,
    sample_size: int | None = None,
    vector_T_emotion: list | None = None,
    emotion_logit_bias: list | None = None,
) -> float:
    """
    역할: train CSV 샘플로 raw_score 95퍼센타일 산출 (4주차 실험 후 1회 실행)
    입력: 모델/토크나이저/디바이스, T_emotion, 샘플 수(None이면 전체 데이터 사용),
          vector_T_emotion(주어지면 단일 T 대신 클래스별 보정으로 raw_score 계산),
          emotion_logit_bias(주어지면 logits 보정 뒤 raw_score 계산)
    출력: p95 값 (float)
    참고: raw_score 분포가 양봉(bimodal)이라 sample_size=2000 수준에서는
          시드별로 P95 위치가 0.073 이상 흔들린다(eval/report/p95_stability.json
          참조). 운영 안정값을 위해 기본은 전체 데이터로 측정한다.
          Vector Scaling 적용 시 raw_score 분포가 단일 T 와 달라지므로
          운영 채택 직후에는 반드시 동일한 보정 방식으로 P95 를 재측정해야 한다.
    """
    import pandas as pd

    train_csv = os.path.join(BASE_DIR, "data", "processed", "emotion_train.csv")
    df = pd.read_csv(train_csv)
    if sample_size is not None:
        df = df.sample(n=min(sample_size, len(df)), random_state=42)

    use_vector = (
        vector_T_emotion is not None and len(vector_T_emotion) == len(LABEL2EMOTION)
    )
    T_vec = (
        torch.tensor(vector_T_emotion, dtype=torch.float32, device=device)
        if use_vector else None
    )
    use_bias = (
        emotion_logit_bias is not None and len(emotion_logit_bias) == len(LABEL2EMOTION)
    )
    bias_vec = (
        torch.tensor(emotion_logit_bias, dtype=torch.float32, device=device)
        if use_bias else None
    )

    raw_scores = []
    for text in df["text"].tolist():
        enc = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=128, padding="max_length"
        )
        input_ids = enc["input_ids"].to(device)
        attn_mask  = enc["attention_mask"].to(device)
        with torch.no_grad():
            emotion_logits, _ = model(input_ids, attn_mask)
        if use_vector:
            emotion_scores = emotion_logits / T_vec
        else:
            emotion_scores = emotion_logits / T_emotion
        if use_bias:
            emotion_scores = emotion_scores + bias_vec
        probs = F.softmax(emotion_scores, dim=-1).cpu().numpy()[0]
        raw = float(sum(
            probs[i] * EMOTION_WEIGHT[LABEL2EMOTION[i]] for i in range(len(LABEL2EMOTION))
        ))
        raw_scores.append(raw)

    p95 = float(np.percentile(raw_scores, 95))
    print(f"[P95 산출] 샘플 {len(raw_scores)}개 → p95={p95:.4f}")
    return p95
