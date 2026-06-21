"""
eval_roberta.py
역할: RoBERTa 감정 분류 성능 평가
      val/calib CSV로 Accuracy, Macro F1, 클래스별 F1, ECE 측정
실행: C:/Users/WD/anaconda3/envs/dl_study/python.exe eval/eval_roberta.py
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.roberta.train_roberta import (
    RoBERTaMultiTask, EmotionDataset, NUM_EMOTION_CLS, NUM_NLI_CLS,
    MAX_LEN, BATCH_SIZE, DEVICE,
)
from models.roberta.temperature_scaling import compute_ece
from pipeline.roberta_score import (
    load_temperature,
    load_emotion_vector_T,
    load_emotion_logit_bias,
    LABEL2EMOTION,
)

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(BASE_DIR, "data", "processed")
CKPT_DIR  = os.path.join(BASE_DIR, "models", "roberta", "checkpoints")

from transformers import AutoTokenizer
MODEL_NAME = "klue/roberta-base"


def apply_emotion_calibration(
    logits_t: torch.Tensor,
    T_emotion: float,
) -> tuple[np.ndarray, str, list[float] | None, list[float] | None]:
    """
    역할: 운영 추론과 동일하게 감정 logits에 보정을 적용한다.
    입력: 감정 logits tensor, scalar temperature fallback 값
    출력: (보정 확률, 보정 방식 이름, vector temperature 또는 None, additive bias 또는 None)
    """
    import torch.nn.functional as F

    vector_T = load_emotion_vector_T()
    emotion_logit_bias = load_emotion_logit_bias()
    if vector_T is not None:
        # 운영 경로는 클래스별 vector temperature를 우선 사용하고, 없을 때만 scalar T로 되돌아간다.
        vector_t = torch.tensor(vector_T, dtype=logits_t.dtype, device=logits_t.device)
        scores = logits_t / vector_t
        method = "vector_scaling"
        vector_value = [float(v) for v in vector_T]
    else:
        scores = logits_t / float(T_emotion)
        method = "scalar_temperature"
        vector_value = None

    bias_value = None
    if emotion_logit_bias is not None and len(emotion_logit_bias) == len(LABEL2EMOTION):
        # 채택 후보 bias는 환경변수/설정에 있을 때만 적용해 기본 평가는 기존과 동일하게 둔다.
        bias_t = torch.tensor(emotion_logit_bias, dtype=scores.dtype, device=scores.device)
        scores = scores + bias_t
        method = f"{method}+logit_bias"
        bias_value = [float(v) for v in emotion_logit_bias]

    probs = F.softmax(scores, dim=-1).numpy()
    return probs, method, vector_value, bias_value


def evaluate(split: str = "val"):
    """
    역할: 지정 split(val 또는 calib)으로 RoBERTa 감정 분류 평가
    입력: split 이름 ("val" | "calib")
    출력: 평가 결과 dict (콘솔 출력 포함)
    """
    csv_path  = os.path.join(DATA_DIR, f"emotion_{split}.csv")
    ckpt_path = os.path.join(CKPT_DIR, "roberta_final.pt")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset   = EmotionDataset(csv_path, tokenizer, MAX_LEN)
    loader    = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = RoBERTaMultiTask(MODEL_NAME, NUM_EMOTION_CLS, NUM_NLI_CLS).to(DEVICE)
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    # utterance_type_head는 별도 체크포인트에 저장되며 본 평가는 감정/NLI만 사용하므로 strict=False
    model.load_state_dict(state, strict=False)
    model.eval()

    T_emotion, _ = load_temperature()

    all_logits, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attn_mask  = batch["attention_mask"].to(DEVICE)
            logits, _  = model(input_ids, attn_mask)
            all_logits.append(logits.cpu())
            all_labels.append(batch["label"])

    logits_t = torch.cat(all_logits)
    labels   = torch.cat(all_labels).numpy()

    # Temperature Scaling 적용
    probs_t, calibration_method, vector_T, emotion_logit_bias = apply_emotion_calibration(logits_t, T_emotion)
    preds     = probs_t.argmax(axis=1)

    acc      = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    ece      = compute_ece(probs_t, labels)

    print(f"\n{'='*50}")
    print(f"[eval_roberta] split={split}  calibration={calibration_method}  T_emotion={T_emotion:.4f}")
    if vector_T is not None:
        print(f"  vector_T  : {[round(v, 4) for v in vector_T]}")
    if emotion_logit_bias is not None:
        print(f"  logit_bias: {[round(v, 4) for v in emotion_logit_bias]}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Macro F1  : {macro_f1:.4f}")
    print(f"  ECE       : {ece:.4f}")
    print(f"\n{classification_report(labels, preds, target_names=LABEL2EMOTION, zero_division=0)}")

    return {
        "split":    split,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "ece":      ece,
        "calibration_method": calibration_method,
        "vector_T_emotion": vector_T,
        "emotion_logit_bias": emotion_logit_bias,
    }


if __name__ == "__main__":
    evaluate("val")
    evaluate("calib")
