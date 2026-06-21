"""
scheduler.py
역할: RoBERTa / Qwen 추론 스케줄러
      ⚠️ RTX 3060Ti 8GB VRAM — 학습 시 동시 적재 금지, 추론은 동시 적재 허용
      (RoBERTa ~0.5GB + Qwen 4bit ~2.5GB ≈ 3GB, 8GB 여유 있음)
      추론 동시 적재 목적: Qwen 응답을 RoBERTa 임베딩 anchor로 즉시 검사
      단, 여러 HTTP 요청의 모델 실행은 단일 FIFO 추론 큐로 직렬화
입력: 발화 텍스트
출력: {roberta 점수 dict, qwen_response, qwen_crisis_tag, anchor_screen}
"""

import gc
import importlib
import json
import os
import sys
from functools import wraps

import numpy as np
import torch
from backend.runtime_guards import SerializedInferenceQueue
from pipeline.roberta_score  import (
    load_roberta_model, load_temperature, load_emotion_vector_T, load_emotion_logit_bias,
    infer_single, ROBERTA_SCORE_P95,
    predict_cbt_class_with_head,
    attenuate_intensifiers,
    apply_intensifier_delta_cap,
    is_intensifier_delta_cap_candidate,
)
from pipeline.cbt_similarity import (
    load_anchors,
    build_anchor_embeddings,
    compute_cbt_score,
    CBT_THRESHOLD,
    _encode as _encode_with_roberta,
)
from pipeline.utterance_type import (
    is_academic_anxiety_text,
    is_limited_situational_distress_text,
    is_interpersonal_remorse_text,
    is_mild_low_mood_text,
    is_mild_unease_text,
    is_physical_exertion_text,
    is_positive_affect_text,
    is_routine_discomfort_text,
    is_daily_routine_neutral_text,
    is_administrative_technical_neutral_text,
    is_situational_anxiety_surprise_text,
    is_situational_anger_text,
    is_situational_sadness_text,
    is_sensory_disgust_text,
)


@torch.no_grad()
def _encode_mean_pool(texts: list, model, tokenizer, device) -> np.ndarray:
    """역할: KLUE-RoBERTa [CLS] anisotropy 회피용 mean-pooled 임베딩 (mask 가중평균)
    응답 anchor 비교 전용 — CBT 점수 산출에는 영향 없음.
    """
    enc = tokenizer(texts, return_tensors="pt", truncation=True, max_length=128, padding=True)
    input_ids = enc["input_ids"].to(device)
    attn_mask = enc["attention_mask"].to(device)
    outputs = model.encoder(input_ids=input_ids, attention_mask=attn_mask)
    last_hidden = outputs.last_hidden_state  # (B, L, H)
    mask = attn_mask.unsqueeze(-1).float()  # (B, L, 1)
    summed = (last_hidden * mask).sum(dim=1)  # (B, H)
    counts = mask.sum(dim=1).clamp(min=1.0)  # (B, 1)
    pooled = summed / counts
    return pooled.cpu().numpy()
from pipeline.ensemble       import ensemble_scores
from pipeline.cbt_reliability import evaluate_cbt_reliability
from pipeline.depression_tendency import compute_depression_tendency
from pipeline.depression_tendency_v2 import (
    compute_depression_tendency_v2,
    severity_scalar_from_distress_probs,
)


# 응답 anchor 설정 — 녹취 잔재·환각 1인칭·처방형 표현 감지용
RESPONSE_ANCHOR_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "processed", "response_anchors.json",
)
RESPONSE_ANCHOR_THRESHOLD_DEFAULT = 0.62  # contrastive 점수 기준 (CBT 와 동일 패턴)
ACADEMIC_ANXIETY_CBT_SCORE_CAP = 0.55
MILD_AFFECTIVE_CBT_SCORE_CAP = 0.55
LIMITED_SITUATIONAL_CBT_SCORE_CAP = 0.58
SITUATIONAL_SADNESS_CBT_SCORE_CAP = 0.60
POSITIVE_AFFECT_CBT_SCORE_CAP = 0.45
PHYSICAL_EXERTION_CBT_SCORE_CAP = 0.45
ROUTINE_DISCOMFORT_CBT_SCORE_CAP = 0.45
DAILY_ROUTINE_CBT_SCORE_CAP = 0.45
ADMIN_TECHNICAL_CBT_SCORE_CAP = 0.45
SENSORY_DISGUST_CBT_SCORE_CAP = 0.50
SITUATIONAL_ANXIETY_SURPRISE_CBT_SCORE_CAP = 0.55
SITUATIONAL_ANGER_CBT_SCORE_CAP = 0.60
QUESTION_CBT_SCORE_CAP = 0.45


def serialized_inference(func):
    """
    역할: ModelScheduler 추론 메서드를 단일 FIFO 큐에서 실행하도록 래핑
    입력: 인스턴스 메서드
    출력: 직렬화된 메서드
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        """
        역할: 스케줄러 FIFO 추론 slot 안에서 원본 메서드 실행
        입력: ModelScheduler 인스턴스, 원본 메서드 인자
        출력: 원본 메서드 반환값
        """
        with self._inference_queue.slot():
            return func(self, *args, **kwargs)

    return wrapper


def _cap_routine_discomfort_cbt_score(
    analysis_text: str,
    utterance_type: str | None,
    cbt_score: float | None,
) -> tuple[float | None, bool]:
    """
    역할: 공부·출근 같은 일상 과업 피로의 CBT anchor 단독 과상승을 제한한다.
    입력: 분석 텍스트, 발화 타입, 원 CBT 점수
    출력: 제한된 CBT 점수와 cap 적용 여부
    """
    if cbt_score is None:
        return None, False
    if (
        utterance_type == "routine_discomfort"
        and is_routine_discomfort_text(analysis_text)
        and not is_physical_exertion_text(analysis_text)
        and not is_limited_situational_distress_text(analysis_text)
        and not is_academic_anxiety_text(analysis_text)
    ):
        capped_score = min(float(cbt_score), ROUTINE_DISCOMFORT_CBT_SCORE_CAP)
        return capped_score, capped_score < float(cbt_score)
    return cbt_score, False


def _resolve_distress_severity(roberta_result: dict) -> float | None:
    """
    역할: distress head 직접 출력 또는 emotion probability proxy로 distress 강도 스칼라를 산출한다.
    입력: RoBERTa 추론 결과 dict
    출력: 0~1 distress severity 또는 계산 불가 시 None
    """
    distress_scalar = roberta_result.get("distress_severity_scalar")
    if distress_scalar is not None:
        return float(distress_scalar)

    emotion_probs = roberta_result.get("emotion_probs") or []
    if emotion_probs and len(emotion_probs) >= 7:
        p_sad = float(emotion_probs[2])
        p_fear = float(emotion_probs[3])
        p_anger = float(emotion_probs[5])
        return min(1.0, (p_sad + p_fear + p_anger) * 0.85)
    return None


class ModelScheduler:
    """
    역할: RoBERTa → Qwen 순차 실행 및 VRAM 관리
          lazy load — 실제 호출 전까지 모델 미적재
    """

    def __init__(
        self,
        p95: float = ROBERTA_SCORE_P95,
        use_cbt: bool = True,
        roberta_ckpt_name: str | None = None,
        utterance_head_name: str | None = None,
        t_emotion_override: float | None = None,
        t_nli_override: float | None = None,
        vector_t_emotion_override: list[float] | None = None,
        emotion_logit_bias_override: list[float] | None = None,
    ):
        """
        역할: RoBERTa/Qwen 스케줄러의 지연 로드 설정을 초기화한다.
        입력: p95 정규화 값, CBT 사용 여부, 평가용 RoBERTa 체크포인트 파일명, 평가용 발화 head 파일명,
              후보 temperature/vector scaling/logit bias override
        출력: 없음
        """
        self.p95     = p95
        self.use_cbt = use_cbt
        self.roberta_ckpt_name = roberta_ckpt_name
        self.utterance_head_name = utterance_head_name
        self.t_emotion_override = t_emotion_override
        self.t_nli_override = t_nli_override
        self.vector_t_emotion_override = vector_t_emotion_override
        self.emotion_logit_bias_override = emotion_logit_bias_override

        # 모델 객체는 lazy load
        self._roberta_model    = None
        self._roberta_tokenizer = None
        self._roberta_device   = None
        self._anchor_embs      = None
        # 응답 anchor: bad 카테고리(녹취/환각/처방형) + normal 대조군
        self._response_anchor_embs: dict[str, np.ndarray] = {}
        self._response_contrast_ids: set[str] = set()
        self._response_anchor_threshold = RESPONSE_ANCHOR_THRESHOLD_DEFAULT
        self._T_emotion        = 1.0
        self._T_nli            = 1.0
        # Vector Scaling 채택 시 7-dim list, 미설정이면 None → 단일 T 폴백.
        self._vector_T_emotion: list | None = None
        # 선택 채택된 additive logit bias. 미설정이면 None → 기존 동작 유지.
        self._emotion_logit_bias: list | None = None
        self._qwen_module      = None
        # 모델 로드와 GPU 추론은 하나의 FIFO 큐에서 실행한다.
        self._inference_queue = SerializedInferenceQueue()

    # ── RoBERTa ───────────────────────────────────────────────────────────────
    @serialized_inference
    def _load_roberta(self):
        """역할: RoBERTa 모델 로드 (미로드 시에만 실행)
        VRAM 정책 변경: 추론 시 Qwen 동시 적재 허용 → 선제 unload 호출 제거.
        학습 모드 진입 시에는 호출 측에서 명시적으로 unload 해야 한다.
        """
        if self._roberta_model is None:
            # 후보 체크포인트 평가는 운영 roberta_final.pt를 덮어쓰지 않고 별도 파일명으로 로드한다.
            self._roberta_model, self._roberta_tokenizer, self._roberta_device = (
                load_roberta_model(
                    self.roberta_ckpt_name,
                    utterance_head_name=self.utterance_head_name,
                )
                if self.roberta_ckpt_name
                else load_roberta_model(utterance_head_name=self.utterance_head_name)
            )
            self._T_emotion, self._T_nli = load_temperature()
            self._vector_T_emotion = load_emotion_vector_T()
            self._emotion_logit_bias = load_emotion_logit_bias()
            if self.t_emotion_override is not None:
                self._T_emotion = float(self.t_emotion_override)
            if self.t_nli_override is not None:
                self._T_nli = float(self.t_nli_override)
            if self.vector_t_emotion_override is not None:
                self._vector_T_emotion = [float(value) for value in self.vector_t_emotion_override]
            if self.emotion_logit_bias_override is not None:
                self._emotion_logit_bias = [float(value) for value in self.emotion_logit_bias_override]

            if self.use_cbt:
                anchors = load_anchors()
                self._anchor_embs = build_anchor_embeddings(
                    anchors, self._roberta_model, self._roberta_tokenizer, self._roberta_device
                )

            # 응답 anchor 임베딩도 같은 인코더로 한 번 계산해 캐시한다.
            self._build_response_anchors()

    def _build_response_anchors(self) -> None:
        """역할: response_anchors.json 의 카테고리별 평균 임베딩을 계산해 캐시
        cbt_similarity._encode 를 그대로 재사용해 동일한 벡터 공간을 보장한다.
        """
        if not os.path.exists(RESPONSE_ANCHOR_PATH):
            print(f"[스케줄러] 응답 anchor 파일 없음 → screen 비활성화: {RESPONSE_ANCHOR_PATH}")
            self._response_anchor_embs = {}
            return

        with open(RESPONSE_ANCHOR_PATH, encoding="utf-8") as f:
            payload = json.load(f)

        meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
        threshold = meta.get("threshold")
        if isinstance(threshold, (int, float)):
            self._response_anchor_threshold = float(threshold)

        embs: dict[str, np.ndarray] = {}
        contrast_ids: set[str] = set()
        for item in payload.get("categories", []):
            if not isinstance(item, dict):
                continue
            cat_id = str(item.get("id", "")).strip()
            phrases = item.get("anchors", []) or []
            phrases = [str(p) for p in phrases if str(p).strip()]
            if not cat_id or not phrases:
                continue
            vecs = _encode_mean_pool(
                phrases, self._roberta_model, self._roberta_tokenizer, self._roberta_device,
            )
            embs[cat_id] = vecs.mean(axis=0)
            if bool(item.get("is_contrast", False)):
                contrast_ids.add(cat_id)
        self._response_anchor_embs = embs
        self._response_contrast_ids = contrast_ids
        bad_ct = len([c for c in embs if c not in contrast_ids])
        print(
            f"[스케줄러] 응답 anchor 임베딩 캐시 완료 "
            f"(bad={bad_ct}, contrast={len(contrast_ids)}, threshold={self._response_anchor_threshold})"
        )

    @serialized_inference
    def _unload_roberta(self):
        """역할: RoBERTa 모델 메모리 해제 + CUDA 캐시 정리
        (학습 모드 전환 등 명시적 호출에서만 사용. 추론 경로는 호출하지 않는다.)
        """
        if self._roberta_model is not None:
            del self._roberta_model
            del self._roberta_tokenizer
            self._roberta_model     = None
            self._roberta_tokenizer = None
            self._anchor_embs       = None
            self._response_anchor_embs = {}
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("[스케줄러] RoBERTa 언로드 완료")

    @serialized_inference
    def _get_qwen_functions(self):
        """
        역할: Qwen 추론/언로드 함수를 동적으로 로드
        입력: 없음
        출력: (generate_response, unload_qwen)
        """
        if self._qwen_module is not None:
            return self._qwen_module.generate_response, self._qwen_module.unload_qwen

        qwen_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "models",
            "qwen",
        )
        if qwen_dir not in sys.path:
            sys.path.insert(0, qwen_dir)

        self._qwen_module = importlib.import_module("inference_qwen")
        return self._qwen_module.generate_response, self._qwen_module.unload_qwen

    @serialized_inference
    def _unload_qwen(self):
        """
        역할: Qwen 모델 메모리 해제 + CUDA 캐시 정리
        입력: 없음
        출력: 없음
        """
        if self._qwen_module is None:
            return

        _, unload_qwen = self._get_qwen_functions()
        unload_qwen()

    @serialized_inference
    def run_roberta(self, text: str) -> dict:
        """
        역할: RoBERTa 추론 → roberta_score + cbt_score + depression_score
        입력: 발화 텍스트
        출력: {roberta_score, cbt_score, depression_score, is_crisis,
               top_emotion, entailment_prob}
        """
        self._load_roberta()

        roberta_result = infer_single(
            text,
            self._roberta_model,
            self._roberta_tokenizer,
            self._roberta_device,
            T_emotion=self._T_emotion,
            T_nli=self._T_nli,
            p95=self.p95,
            vector_T_emotion=self._vector_T_emotion,
            emotion_logit_bias=self._emotion_logit_bias,
        )
        analysis_text = roberta_result.get("analysis_text") or text

        cbt_score = None
        cbt_top_category = None
        cbt_top_category_source = None
        cbt_class_confidence = None
        head_is_non_distortion = None
        academic_anxiety_cbt_capped = False
        mild_affective_cbt_capped = False
        limited_situational_cbt_capped = False
        positive_affect_cbt_capped = False
        physical_exertion_cbt_capped = False
        daily_routine_cbt_capped = False
        admin_technical_cbt_capped = False
        sensory_disgust_cbt_capped = False
        situational_anxiety_surprise_cbt_capped = False
        situational_anger_cbt_capped = False
        situational_sadness_cbt_capped = False
        routine_discomfort_cbt_capped = False
        intensifier_cbt_capped = False
        intensifier_cbt_guard = None
        intensifier_cbt_meta = None
        utterance_type_cbt_capped = False
        cbt_reliability = {
            "cbt_effect": "none",
            "cbt_reliability_policy": "no_cbt_score",
            "cbt_reliability_applied": False,
            "cbt_score_after_reliability": None,
            "cbt_reliability_cap": None,
            "cbt_reliability_risk_points": 0.0,
            "cbt_reliability_benign_points": 0.0,
            "cbt_reliability_reasons": [],
        }
        dep_tendency = None
        if self.use_cbt and self._anchor_embs:
            cbt_result = compute_cbt_score(
                analysis_text,
                self._roberta_model,
                self._roberta_tokenizer,
                self._roberta_device,
                self._anchor_embs,
            )
            cbt_score = cbt_result["cbt_score"]
            if cbt_score is not None and is_academic_anxiety_text(analysis_text):
                # 시험·면접 긴장은 인지왜곡 anchor와 가까워도 단일 발화만으로 위험 점수까지 올리지 않는다.
                raw_cbt_score = float(cbt_score)
                cbt_score = min(raw_cbt_score, ACADEMIC_ANXIETY_CBT_SCORE_CAP)
                academic_anxiety_cbt_capped = cbt_score < raw_cbt_score
            if cbt_score is not None and is_limited_situational_distress_text(analysis_text):
                # 과제·발표 등 단일 상황성 속상함은 왜곡 anchor가 높아도 중등도 이상으로 제한한다.
                raw_cbt_score = float(cbt_score)
                cbt_score = min(raw_cbt_score, LIMITED_SITUATIONAL_CBT_SCORE_CAP)
                limited_situational_cbt_capped = cbt_score < raw_cbt_score
            if cbt_score is not None and is_positive_affect_text(analysis_text):
                # 긍정·기대 발화는 CBT anchor가 우연히 높아도 max 앙상블을 빼앗지 못하게 제한한다.
                raw_cbt_score = float(cbt_score)
                cbt_score = min(raw_cbt_score, POSITIVE_AFFECT_CBT_SCORE_CAP)
                positive_affect_cbt_capped = cbt_score < raw_cbt_score
            if cbt_score is not None and is_physical_exertion_text(analysis_text):
                # 운동·근무·집안일 뒤의 신체 피로는 정서 위기 신호가 아니므로 CBT 단독 고점을 제한한다.
                raw_cbt_score = float(cbt_score)
                cbt_score = min(raw_cbt_score, PHYSICAL_EXERTION_CBT_SCORE_CAP)
                physical_exertion_cbt_capped = cbt_score < raw_cbt_score
            if cbt_score is not None:
                # 공부·출근 같은 일상 과업 피로는 full-impact로 반영하되,
                # CBT anchor 단독 고점이 위험권 점수를 만들지는 않게 RoBERTa cap과 맞춘다.
                cbt_score, routine_discomfort_cbt_capped = _cap_routine_discomfort_cbt_score(
                    analysis_text,
                    roberta_result.get("utterance_type"),
                    cbt_score,
                )
            if cbt_score is not None and is_daily_routine_neutral_text(analysis_text):
                # 음식·휴식·집안일 같은 일상 루틴은 CBT anchor 단독 고점이 당일 점수를 흔들지 않게 제한한다.
                raw_cbt_score = float(cbt_score)
                cbt_score = min(raw_cbt_score, DAILY_ROUTINE_CBT_SCORE_CAP)
                daily_routine_cbt_capped = cbt_score < raw_cbt_score
            if cbt_score is not None and is_administrative_technical_neutral_text(analysis_text):
                # 번호·버전·문서 처리 같은 행정/기술 발화는 CBT anchor 단독 고점을 저신호로 제한한다.
                raw_cbt_score = float(cbt_score)
                cbt_score = min(raw_cbt_score, ADMIN_TECHNICAL_CBT_SCORE_CAP)
                admin_technical_cbt_capped = cbt_score < raw_cbt_score
            if cbt_score is not None and is_sensory_disgust_text(analysis_text):
                # 감각 혐오는 감정 라벨은 보존하되 CBT 단독 고점으로 위험권에 고정하지 않는다.
                raw_cbt_score = float(cbt_score)
                cbt_score = min(raw_cbt_score, SENSORY_DISGUST_CBT_SCORE_CAP)
                sensory_disgust_cbt_capped = cbt_score < raw_cbt_score
            if cbt_score is not None and is_situational_anxiety_surprise_text(analysis_text):
                # 경도 상황 불안·놀람은 감정 신호는 보존하되 CBT 단독으로 위험권에 고정하지 않는다.
                raw_cbt_score = float(cbt_score)
                cbt_score = min(raw_cbt_score, SITUATIONAL_ANXIETY_SURPRISE_CBT_SCORE_CAP)
                situational_anxiety_surprise_cbt_capped = cbt_score < raw_cbt_score
            if cbt_score is not None and is_situational_anger_text(analysis_text):
                # 일회성 분노·억울함은 감정 라우팅은 보존하되 CBT 단독 고점으로 위험권에 고정하지 않는다.
                raw_cbt_score = float(cbt_score)
                cbt_score = min(raw_cbt_score, SITUATIONAL_ANGER_CBT_SCORE_CAP)
                situational_anger_cbt_capped = cbt_score < raw_cbt_score
            if cbt_score is not None and is_situational_sadness_text(analysis_text):
                # 단일 사건성 슬픔은 상담 신호로 남기되 CBT 단독 고점으로 위험권에 고정하지 않는다.
                raw_cbt_score = float(cbt_score)
                cbt_score = min(raw_cbt_score, SITUATIONAL_SADNESS_CBT_SCORE_CAP)
                situational_sadness_cbt_capped = cbt_score < raw_cbt_score
            if cbt_score is not None and is_interpersonal_remorse_text(analysis_text):
                # 상대에게 상처를 줬을까 걱정하는 관계 후회는 가해/자해 의도가 아니므로 고점을 제한한다.
                raw_cbt_score = float(cbt_score)
                cbt_score = min(raw_cbt_score, SITUATIONAL_SADNESS_CBT_SCORE_CAP)
                situational_sadness_cbt_capped = cbt_score < raw_cbt_score
            if cbt_score is not None and (
                is_mild_unease_text(analysis_text) or is_mild_low_mood_text(analysis_text)
            ):
                # 막연한 불편감·가벼운 저조감은 CBT anchor와 가까워도 파국화 점수로 바로 끌어올리지 않는다.
                raw_cbt_score = float(cbt_score)
                cbt_score = min(raw_cbt_score, MILD_AFFECTIVE_CBT_SCORE_CAP)
                mild_affective_cbt_capped = cbt_score < raw_cbt_score
            if (
                cbt_score is not None
                and float(cbt_score) > 0.45
                and is_intensifier_delta_cap_candidate(
                    analysis_text,
                    roberta_result.get("utterance_type"),
                    bool(roberta_result.get("is_crisis", False)),
                )
            ):
                # 저위험 발화에서는 원문 CBT와 강조어 약화판 CBT를 비교해 강조어 한 단어의 폭주를 제한한다.
                attenuated_text = (
                    roberta_result.get("intensifier_attenuated_text")
                    or attenuate_intensifiers(analysis_text)
                )
                if attenuated_text != analysis_text:
                    attenuated_cbt_result = compute_cbt_score(
                        attenuated_text,
                        self._roberta_model,
                        self._roberta_tokenizer,
                        self._roberta_device,
                        self._anchor_embs,
                    )
                    cbt_score, intensifier_cbt_guard, intensifier_cbt_meta = apply_intensifier_delta_cap(
                        analysis_text,
                        cbt_score,
                        attenuated_cbt_result.get("cbt_score"),
                        roberta_result.get("utterance_type"),
                        is_crisis=bool(roberta_result.get("is_crisis", False)),
                    )
                    intensifier_cbt_capped = intensifier_cbt_guard is not None

            # 발화 타입 게이트 — 비왜곡 가능성 큰 긍정/취향/실용 발화는 카테고리 미보존.
            # 이미 추론된 utterance_type(roberta_result) 을 재사용해 비용 0.
            # v3(2026-04-27 03:30) 1차 도입: 11클래스 head 의 비왜곡 클래스가 데이터 풀
            #   부족(3.4k)으로 false negative 발생 → head 의 is_non_distortion 대신
            #   utterance_type 으로 차단.
            # v3.1(2026-04-27) α+β 완화: 1차에서 64건 smoke test 결과 utterance_type
            #   head 가 명백한 왜곡 발화 8건을 casual_share 로 오분류 → 차단 → false
            #   negative 양산이 발견됨. 따라서:
            #   α) casual_share 를 게이트에서 제외 (이 타입은 anchor 임계로만 1차 방어)
            #   β) utterance_type confidence < 0.7 일 때는 게이트 자체를 무시 (head 가
            #      자기 분류를 신뢰 못 하면 차단도 보수적으로) — false negative 안전망.
            CBT_BLOCK_TYPES = {"positive_share",
                               "preference_question", "practical_question"}
            CBT_BLOCK_UTT_CONF_FLOOR = 0.7
            utt_type = roberta_result.get("utterance_type")
            utt_conf = roberta_result.get("utterance_type_confidence", 0.0) or 0.0
            type_blocked = (
                utt_type in CBT_BLOCK_TYPES
                and utt_conf >= CBT_BLOCK_UTT_CONF_FLOOR
            )
            if type_blocked and cbt_score is not None:
                # 실용 질문·취향 질문·긍정 발화는 카테고리만 막아도 raw CBT가 max 앙상블을 끌어올릴 수 있어 점수도 제한한다.
                raw_cbt_score = float(cbt_score)
                cbt_score = min(raw_cbt_score, QUESTION_CBT_SCORE_CAP)
                utterance_type_cbt_capped = cbt_score < raw_cbt_score
            if academic_anxiety_cbt_capped:
                type_blocked = True
            if limited_situational_cbt_capped:
                type_blocked = True
            if positive_affect_cbt_capped:
                type_blocked = True
            if physical_exertion_cbt_capped:
                type_blocked = True
            if daily_routine_cbt_capped:
                type_blocked = True
            if admin_technical_cbt_capped:
                type_blocked = True
            if sensory_disgust_cbt_capped:
                type_blocked = True
            if situational_anxiety_surprise_cbt_capped:
                type_blocked = True
            if situational_anger_cbt_capped:
                type_blocked = True
            if situational_sadness_cbt_capped:
                type_blocked = True
            if routine_discomfort_cbt_capped:
                type_blocked = True
            if intensifier_cbt_capped:
                type_blocked = True
            head_pred = predict_cbt_class_with_head(
                analysis_text,
                self._roberta_model,
                self._roberta_tokenizer,
                self._roberta_device,
            )
            if head_pred is not None:
                head_is_non_distortion = head_pred.get("is_non_distortion")

            _fusion_severity = _resolve_distress_severity(roberta_result)
            dep_tendency = compute_depression_tendency(
                analysis_text,
                top_emotion=roberta_result.get("top_emotion"),
                roberta_score=roberta_result.get("roberta_score"),
                cbt_score=cbt_score,
                cbt_non_distortion=head_is_non_distortion,
                utterance_type=roberta_result.get("utterance_type"),
                type_reason=roberta_result.get("utterance_type_reason"),
                is_crisis=roberta_result.get("is_crisis", False),
                entailment_prob=roberta_result.get("entailment_prob"),
            )

            cbt_reliability = evaluate_cbt_reliability(
                cbt_score=cbt_score,
                cbt_result=cbt_result,
                roberta_out=roberta_result,
                cbt_head_pred=head_pred,
                depression_tendency_score=dep_tendency["depression_tendency_score"],
                distress_severity=_fusion_severity,
                cbt_threshold=CBT_THRESHOLD,
            )
            if cbt_reliability["cbt_reliability_applied"]:
                cbt_score = cbt_reliability["cbt_score_after_reliability"]
                type_blocked = True

            anchor_above_threshold = (
                cbt_score is not None and cbt_score >= CBT_THRESHOLD
            )

            if anchor_above_threshold and not type_blocked and head_pred is not None:
                # head 신뢰도 충분하면 head 카테고리, 아니면 anchor argmax 폴백.
                # is_non_distortion 게이트는 의도적으로 무시 (false negative 위험 큼).
                top_d_label = head_pred.get("top_distortion_label")
                top_d_conf = head_pred.get("top_distortion_confidence", 0.0)
                if top_d_label and top_d_conf >= 0.50:
                    cbt_top_category = top_d_label
                    cbt_class_confidence = top_d_conf
                    cbt_top_category_source = "cbt_class_head"
                else:
                    cbt_top_category = cbt_result.get("top_category")
                    cbt_top_category_source = "anchor_argmax_low_head_conf"
                    cbt_class_confidence = top_d_conf
            elif anchor_above_threshold and not type_blocked and head_pred is None:
                # head 미로드 — 기존 anchor argmax 폴백
                cbt_top_category = cbt_result.get("top_category")
                cbt_top_category_source = "anchor_argmax"
            elif anchor_above_threshold and type_blocked:
                # anchor 임계 통과했지만 발화 타입이 일상/긍정/취향/실용 → 차단
                if academic_anxiety_cbt_capped:
                    cbt_top_category_source = "academic_anxiety_cbt_cap"
                elif limited_situational_cbt_capped:
                    cbt_top_category_source = "limited_situational_distress_cbt_cap"
                elif positive_affect_cbt_capped:
                    cbt_top_category_source = "positive_affect_cbt_cap"
                elif physical_exertion_cbt_capped:
                    cbt_top_category_source = "physical_exertion_cbt_cap"
                elif daily_routine_cbt_capped:
                    cbt_top_category_source = "daily_routine_cbt_cap"
                elif admin_technical_cbt_capped:
                    cbt_top_category_source = "administrative_technical_cbt_cap"
                elif sensory_disgust_cbt_capped:
                    cbt_top_category_source = "sensory_disgust_cbt_cap"
                elif situational_anxiety_surprise_cbt_capped:
                    cbt_top_category_source = "situational_anxiety_surprise_cbt_cap"
                elif situational_anger_cbt_capped:
                    cbt_top_category_source = "situational_anger_cbt_cap"
                elif situational_sadness_cbt_capped:
                    cbt_top_category_source = "situational_sadness_cbt_cap"
                elif routine_discomfort_cbt_capped:
                    cbt_top_category_source = "routine_discomfort_cbt_cap"
                elif intensifier_cbt_capped:
                    cbt_top_category_source = "intensifier_delta_cbt_cap"
                elif cbt_reliability["cbt_reliability_applied"]:
                    cbt_top_category_source = (
                        f"cbt_reliability_gate({cbt_reliability['cbt_reliability_policy']})"
                    )
                elif utterance_type_cbt_capped:
                    cbt_top_category_source = f"utterance_type_cbt_cap({utt_type})"
                else:
                    cbt_top_category_source = f"utterance_type_block({utt_type})"
            elif academic_anxiety_cbt_capped:
                cbt_top_category_source = "academic_anxiety_cbt_cap"
            elif limited_situational_cbt_capped:
                cbt_top_category_source = "limited_situational_distress_cbt_cap"
            elif positive_affect_cbt_capped:
                cbt_top_category_source = "positive_affect_cbt_cap"
            elif physical_exertion_cbt_capped:
                cbt_top_category_source = "physical_exertion_cbt_cap"
            elif daily_routine_cbt_capped:
                cbt_top_category_source = "daily_routine_cbt_cap"
            elif admin_technical_cbt_capped:
                cbt_top_category_source = "administrative_technical_cbt_cap"
            elif sensory_disgust_cbt_capped:
                cbt_top_category_source = "sensory_disgust_cbt_cap"
            elif situational_anxiety_surprise_cbt_capped:
                cbt_top_category_source = "situational_anxiety_surprise_cbt_cap"
            elif situational_anger_cbt_capped:
                cbt_top_category_source = "situational_anger_cbt_cap"
            elif situational_sadness_cbt_capped:
                cbt_top_category_source = "situational_sadness_cbt_cap"
            elif routine_discomfort_cbt_capped:
                cbt_top_category_source = "routine_discomfort_cbt_cap"
            elif intensifier_cbt_capped:
                cbt_top_category_source = "intensifier_delta_cbt_cap"
            elif cbt_reliability["cbt_reliability_applied"]:
                cbt_top_category_source = (
                    f"cbt_reliability_gate({cbt_reliability['cbt_reliability_policy']})"
                )
            elif mild_affective_cbt_capped:
                cbt_top_category_source = "mild_affective_cbt_cap"
            elif utterance_type_cbt_capped:
                cbt_top_category_source = f"utterance_type_cbt_cap({utt_type})"

        # Phase 5b — distress severity는 distress head 직접 출력 우선, 없으면 emotion proxy fallback.
        _fusion_severity = _resolve_distress_severity(roberta_result)
        ens = ensemble_scores(
            roberta_result["roberta_score"],
            cbt_score,
            distress_severity=_fusion_severity,
        )

        # 우울 경향 전용 점수 v1.5 (규칙 기반) — 기존 depression_score(종합 distress)와 병렬 운영.
        # 명시 우울/무기력/흥미저하/무가치감/절망/수면식욕/사회적 위축 표현만 가산하고
        # 시험불안/단일 사건성 분노·속상함/신체 피로/일상 루틴/긍정 회복은 cap으로 제한한다.
        if dep_tendency is None:
            dep_tendency = compute_depression_tendency(
                analysis_text,
                top_emotion=roberta_result.get("top_emotion"),
                roberta_score=roberta_result.get("roberta_score"),
                cbt_score=cbt_score,
                cbt_non_distortion=head_is_non_distortion,
                utterance_type=roberta_result.get("utterance_type"),
                type_reason=roberta_result.get("utterance_type_reason"),
                is_crisis=roberta_result.get("is_crisis", False),
                entailment_prob=roberta_result.get("entailment_prob"),
            )

        # v2 우울 경향 점수 — top_emotion 의존성 제거, evidence span + severity + persistence
        #   현재는 dual-output: v1.5가 운영 표시값, v2는 audit/모니터링 전용.
        #   Phase 5b — distress_severity는 distress head 직접 출력을 우선 사용,
        #   없으면 emotion 7-class prob proxy로 fallback.
        _v2_distress_scalar = roberta_result.get("distress_severity_scalar")
        if _v2_distress_scalar is not None:
            severity_proxy = float(_v2_distress_scalar)
        else:
            emotion_probs = roberta_result.get("emotion_probs") or []
            if emotion_probs:
                p_sad = float(emotion_probs[2]) if len(emotion_probs) > 2 else 0.0
                p_fear = float(emotion_probs[3]) if len(emotion_probs) > 3 else 0.0
                p_anger = float(emotion_probs[5]) if len(emotion_probs) > 5 else 0.0
                severity_proxy = min(1.0, (p_sad + p_fear + p_anger) * 0.85)
            else:
                severity_proxy = 0.0
        dep_tendency_v2 = compute_depression_tendency_v2(
            analysis_text,
            distress_severity=severity_proxy,
            utterance_type=roberta_result.get("utterance_type"),
            type_reason=roberta_result.get("utterance_type_reason"),
            cbt_score=cbt_score,
            cbt_non_distortion=head_is_non_distortion,
            is_crisis=roberta_result.get("is_crisis", False),
            entailment_prob=roberta_result.get("entailment_prob"),
            top_emotion=roberta_result.get("top_emotion"),
            roberta_score=roberta_result.get("roberta_score"),
        )

        return {
            **roberta_result,
            "cbt_score":        cbt_score,
            "cbt_top_category": cbt_top_category,
            "cbt_top_category_source": cbt_top_category_source,
            "cbt_class_confidence": cbt_class_confidence,
            "cbt_head_non_distortion": head_is_non_distortion,
            "cbt_effect": cbt_reliability["cbt_effect"],
            "cbt_reliability_policy": cbt_reliability["cbt_reliability_policy"],
            "cbt_reliability_applied": cbt_reliability["cbt_reliability_applied"],
            "cbt_reliability_cap": cbt_reliability["cbt_reliability_cap"],
            "cbt_reliability_risk_points": cbt_reliability["cbt_reliability_risk_points"],
            "cbt_reliability_benign_points": cbt_reliability["cbt_reliability_benign_points"],
            "cbt_reliability_reasons": cbt_reliability["cbt_reliability_reasons"],
            "intensifier_cbt_delta_guard": intensifier_cbt_guard,
            "intensifier_cbt_attenuated_text": (
                (intensifier_cbt_meta or {}).get("intensifier_attenuated_text")
            ),
            "intensifier_cbt_attenuated_score": (
                (intensifier_cbt_meta or {}).get("intensifier_attenuated_score")
            ),
            "intensifier_cbt_allowed_delta": (
                (intensifier_cbt_meta or {}).get("intensifier_allowed_delta")
            ),
            "intensifier_cbt_original_score": (
                (intensifier_cbt_meta or {}).get("intensifier_original_score")
            ),
            "depression_score": ens["depression_score"],
            "ensemble_method":  ens["method"],
            "ensemble_fusion_caps": ens.get("fusion_caps", []),
            "cbt_score_effective": ens.get("cbt_score_effective", cbt_score),
            "depression_tendency_score":   dep_tendency["depression_tendency_score"],
            "depression_tendency_categories": dep_tendency["hit_categories"],
            "depression_tendency_caps":   dep_tendency["caps_applied"],
            "depression_tendency_persistence": dep_tendency["persistence_marker_hit"],
            "depression_tendency_version": dep_tendency["version"],
            # v2 audit (운영 표시값 X, model_audit_events 박제용)
            "depression_tendency_v2_score": dep_tendency_v2["depression_tendency_score"],
            "depression_tendency_v2_evidence": dep_tendency_v2["evidence"],
            "depression_tendency_v2_severity_band": dep_tendency_v2["severity_band"],
            "depression_tendency_v2_severity_scalar": dep_tendency_v2["severity_scalar"],
            "depression_tendency_v2_persistence_band": dep_tendency_v2["persistence_band"],
            "depression_tendency_v2_caps": dep_tendency_v2["caps_applied"],
            "depression_tendency_v2_version": dep_tendency_v2["version"],
        }

    # ── 응답 anchor 검사 ────────────────────────────────────────────────────
    @serialized_inference
    def screen_response(
        self,
        response_text: str,
        utterance_info: dict | None = None,
    ) -> dict:
        """
        역할: Qwen 응답을 RoBERTa 임베딩 anchor 와 비교해 녹취 잔재·환각·처방형 차단
        입력: 응답 텍스트, 발화 타입 정보(이용자 케이스별 fallback 분기에 사용)
        출력: {
            replaced: bool,           # anchor hit 으로 교체했는지
            final: str,               # 최종 응답(원문 또는 fallback)
            hits: list[(cat, sim)],   # 임계 초과 카테고리 + 유사도
            similarities: dict,       # 카테고리별 유사도(디버깅용)
        }
        """
        if (
            self._roberta_model is None
            or not self._response_anchor_embs
            or not response_text
        ):
            return {"replaced": False, "final": response_text, "hits": [], "similarities": {}}

        text_emb = _encode_mean_pool(
            [response_text],
            self._roberta_model, self._roberta_tokenizer, self._roberta_device,
        )[0]

        sims: dict[str, float] = {}
        for cat_id, anchor_emb in self._response_anchor_embs.items():
            denom = float(np.linalg.norm(text_emb) * np.linalg.norm(anchor_emb)) + 1e-8
            cos = float(np.dot(text_emb, anchor_emb) / denom)
            sims[cat_id] = (cos + 1.0) / 2.0  # [-1,1] → [0,1]

        bad_sims = {c: v for c, v in sims.items() if c not in self._response_contrast_ids}
        normal_sims = {c: v for c, v in sims.items() if c in self._response_contrast_ids}
        if not bad_sims:
            return {"replaced": False, "final": response_text, "hits": [], "similarities": sims}

        # contrastive 점수: KLUE-RoBERTa anisotropy 보정 (CBT 와 동일 패턴)
        # score = clip(bad_max - normal_max + 0.5, 0, 1)
        normal_max = max(normal_sims.values()) if normal_sims else 0.0
        scored: list[tuple[str, float, float]] = []
        for cat_id, raw in bad_sims.items():
            score = float(np.clip(raw - normal_max + 0.5, 0.0, 1.0))
            scored.append((cat_id, score, raw))

        scored.sort(key=lambda x: -x[1])
        hits = [(c, score) for c, score, _ in scored if score >= self._response_anchor_threshold]

        if not hits:
            return {
                "replaced":     False,
                "final":        response_text,
                "hits":         [],
                "similarities": sims,
                "scored":       scored,
                "normal_max":   normal_max,
            }

        # anchor hit → 발화 타입에 맞는 fallback 으로 교체
        replacement = self._anchor_fallback(utterance_info, hits[0][0])
        return {
            "replaced":     True,
            "final":        replacement,
            "hits":         hits,
            "similarities": sims,
            "scored":       scored,
            "normal_max":   normal_max,
        }

    def _anchor_fallback(self, utterance_info: dict | None, hit_category: str) -> str:
        """역할: anchor hit 시 발화 타입에 맞는 안전 fallback 문장 반환
        inference_qwen._fallback_response 를 재사용해 기존 카테고리 분기와 일관 유지.
        Phase 5c — utterance_info에 distress_top_label / distress_severity_scalar이 박제돼 있으면
        fallback 함수가 high distress 시 가장 보수적 톤을 우선 선택한다.
        """
        try:
            self._get_qwen_functions()  # 모듈 import 보장
            user_text = (utterance_info or {}).get("text", "") if isinstance(utterance_info, dict) else ""
            avoid_texts = (utterance_info or {}).get("avoid_responses") if isinstance(utterance_info, dict) else None
            distress_top = (utterance_info or {}).get("distress_top_label") if isinstance(utterance_info, dict) else None
            distress_sev = (utterance_info or {}).get("distress_severity_scalar") if isinstance(utterance_info, dict) else None
            return self._qwen_module._fallback_response(
                user_text,
                utterance_info=utterance_info,
                avoid_texts=avoid_texts,
                distress_top_label=distress_top,
                distress_severity_scalar=distress_sev,
            )
        except Exception as exc:
            # 모듈 fallback 사용 불가 시 카테고리별 최소 안전 문구
            print(f"[스케줄러] anchor fallback import 실패({hit_category}): {exc}")
            return "잠시 정리하고 다시 이야기해볼까요. 지금 마음에 가장 걸리는 부분을 한 줄로 알려줘요."

    @staticmethod
    def _recent_assistant_responses(history: list[dict] | None, limit: int = 6) -> list[str]:
        """
        역할: 최근 대화 히스토리에서 반복 회피용 assistant 응답만 추출
        입력: 대화 히스토리, 최대 개수
        출력: 최근 assistant 응답 목록
        """
        if not history:
            return []
        responses: list[str] = []
        for item in history:
            if not isinstance(item, dict):
                continue
            if item.get("role") == "assistant" and item.get("content"):
                responses.append(str(item["content"]))
        return responses[-limit:]

    # ── Qwen ──────────────────────────────────────────────────────────────────
    @serialized_inference
    def run_qwen(
        self,
        text: str,
        history: list[dict] | None = None,
        utterance_info: dict | None = None,
    ) -> dict:
        """
        역할: Qwen 상담 응답 생성 + RoBERTa anchor 기반 응답 검사
        입력: 발화 텍스트, 대화 히스토리 [{"role": ..., "content": ...}],
              발화 타입 정보
        출력: {response: str, has_crisis_tag: bool, anchor_screen: dict}
        """
        # 추론 시 동시 적재 허용 — RoBERTa 는 anchor 검사용으로 유지한다.
        # (학습 모드 진입 시에만 외부에서 _unload_roberta 호출)
        generate_response, _ = self._get_qwen_functions()
        from backend.crisis_handler import check_qwen_crisis_tag

        avoid_responses = self._recent_assistant_responses(history)
        screen_utterance_info = {
            **(utterance_info or {}),
            "text": text,
            "avoid_responses": avoid_responses,
        }

        response = generate_response(text, history or [], utterance_info=screen_utterance_info)

        # 1차: 응답 anchor 임베딩 검사 (KLUE-RoBERTa contrastive)
        screen = self.screen_response(
            response,
            utterance_info=screen_utterance_info,
        )

        # 2차: 임베딩 검사 통과 시 Qwen 자기검토(two-pass self-check)
        # — 임베딩이 못 잡는 paraphrase(짧은 처방형, normal 톤과 가까운 환각) 보강
        from backend.qwen_quality_policy import self_check_requires_fallback

        self_check_result = {"verdict": "SKIPPED", "category": None, "raw": ""}
        if not screen["replaced"]:
            self_check_result = self._run_self_check(text, screen["final"])
            if self_check_requires_fallback(self_check_result):
                fallback = self._anchor_fallback(
                    screen_utterance_info,
                    f"self_check:{self_check_result['category']}",
                )
                screen = {
                    **screen,
                    "replaced":     True,
                    "final":        fallback,
                    "self_check":   self_check_result,
                }

        final_response = screen["final"]
        dedupe_replaced = False
        try:
            final_response, dedupe_replaced = self._qwen_module.diversify_repeated_response(
                text,
                final_response,
                utterance_info=screen_utterance_info,
                avoid_texts=avoid_responses,
            )
        except Exception as exc:
            print(f"[scheduler] 반복 응답 치환 실패 → 원문 유지: {exc}")
        has_crisis_tag = check_qwen_crisis_tag(final_response)

        return {
            "response":       final_response,
            "has_crisis_tag": has_crisis_tag,
            "anchor_screen":  {
                "replaced":     screen["replaced"],
                "hits":         screen["hits"],
                "similarities": screen["similarities"],
                "self_check":   self_check_result,
                "dedupe_replaced": dedupe_replaced,
                "dedupe_avoid_count": len(avoid_responses),
                # raw Qwen 출력(anchor/self_check 교체 전)을 audit에 보존해
                # 운영 모니터링에서 어떤 응답이 BAD로 잡혔는지 사후 리뷰 가능하게 함.
                "raw_response": response,
            },
        }

    @serialized_inference
    def generate_summary(self, texts: list[str]) -> str:
        """
        역할: Qwen 서사 요약을 단일 추론 큐에서 생성
        입력: 요약할 사용자 발화 목록
        출력: 생성된 요약 문자열
        """
        self._get_qwen_functions()
        return self._qwen_module.generate_summary(texts)

    @serialized_inference
    def _run_self_check(self, user_text: str, response_text: str) -> dict:
        """
        역할: inference_qwen.self_check_response 위임 호출
        입력: 사용자 발화, Qwen 생성 응답
        출력: self-check 결과 dict, 검사 실패 시 ERROR verdict
        """
        from backend.qwen_quality_policy import build_self_check_error_result

        try:
            self._get_qwen_functions()
            return self._qwen_module.self_check_response(user_text, response_text)
        except Exception as exc:
            print(f"[scheduler] self_check 호출 실패 → 안전 fallback 처리: {exc}")
            return build_self_check_error_result()

    @serialized_inference
    def run_full(self, text: str, history: list[dict] | None = None) -> dict:
        """
        역할: RoBERTa → Qwen 순차 실행 (VRAM 안전 관리 포함)
        입력: 발화 텍스트, 대화 히스토리
        출력: roberta 결과 + qwen 결과 통합 dict
        """
        roberta_out = self.run_roberta(text)
        qwen_out    = self.run_qwen(text, history, utterance_info=roberta_out)
        return {**roberta_out, **qwen_out}
