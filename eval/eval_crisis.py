"""
eval_crisis.py
역할: NLI 위기 감지 성능 평가
      nli_pairs.csv로 Precision / Recall / F1 + 오탐/미탐 사례 출력
실행: C:/Users/WD/anaconda3/envs/dl_study/python.exe eval/eval_crisis.py
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.roberta.train_roberta import (
    RoBERTaMultiTask, NUM_EMOTION_CLS, NUM_NLI_CLS, DEVICE,
)
from pipeline.roberta_score import load_temperature, CRISIS_THRESHOLD
from transformers import AutoTokenizer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NLI_CSV  = os.path.join(BASE_DIR, "data", "nli", "nli_pairs.csv")
CKPT_DIR = os.path.join(BASE_DIR, "models", "roberta", "checkpoints")
MODEL_NAME = "klue/roberta-base"

# NLI 레이블: 0=entailment(위기), 1=neutral, 2=contradiction(비위기)
NLI_LABELS = ["entailment(위기)", "neutral", "contradiction(비위기)"]

# 위기 판별 이진 분류: entailment → 위기(1), 그 외 → 비위기(0)
CRISIS_LABEL = 0


def evaluate_nli():
    """
    역할: NLI 3클래스 성능 + 이진 위기 감지 성능 측정
    출력: 콘솔 리포트 + 오탐/미탐 샘플 5개씩 출력
    """
    df = pd.read_csv(NLI_CSV)
    texts  = df["premise"].tolist()
    hypotheses = df["hypothesis"].tolist()
    labels = df["label"].tolist()   # 0=entailment, 1=neutral, 2=contradiction

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = RoBERTaMultiTask(MODEL_NAME, NUM_EMOTION_CLS, NUM_NLI_CLS).to(DEVICE)
    ckpt_path = os.path.join(CKPT_DIR, "roberta_final.pt")
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    # utterance_type_head는 별도 체크포인트에 저장되며 본 평가는 NLI만 사용하므로 strict=False
    model.load_state_dict(state, strict=False)
    model.eval()

    _, T_nli = load_temperature()

    all_probs = []

    with torch.no_grad():
        for text, hypothesis in zip(texts, hypotheses):
            enc = tokenizer(
                text, hypothesis, return_tensors="pt", truncation=True,
                max_length=128, padding="max_length",
            )
            input_ids = enc["input_ids"].to(DEVICE)
            attn_mask  = enc["attention_mask"].to(DEVICE)
            _, nli_logits = model(input_ids, attn_mask)
            probs = F.softmax(nli_logits / T_nli, dim=-1).cpu().numpy()[0]
            all_probs.append(probs)

    all_probs = np.array(all_probs)
    preds_3cls = all_probs.argmax(axis=1)
    entail_probs = all_probs[:, CRISIS_LABEL]

    # ── 3클래스 리포트 ──────────────────────────────────────────────────────────
    print("\n" + "="*55)
    print(f"[eval_crisis] NLI 3클래스  T_nli={T_nli:.4f}")
    print(classification_report(labels, preds_3cls, target_names=NLI_LABELS, zero_division=0))

    # ── 이진 위기 감지 리포트 ───────────────────────────────────────────────────
    true_binary = [1 if l == CRISIS_LABEL else 0 for l in labels]
    pred_binary = [1 if p > CRISIS_THRESHOLD else 0 for p in entail_probs]

    print(f"[이진 위기 감지] threshold={CRISIS_THRESHOLD}")
    print(classification_report(true_binary, pred_binary,
                                target_names=["비위기", "위기"], zero_division=0))

    cm = confusion_matrix(true_binary, pred_binary)
    print(f"혼동 행렬 (비위기/위기):\n{cm}")

    # ── 오탐/미탐 샘플 출력 ────────────────────────────────────────────────────
    false_pos = [i for i, (t, p) in enumerate(zip(true_binary, pred_binary)) if t == 0 and p == 1]
    false_neg = [i for i, (t, p) in enumerate(zip(true_binary, pred_binary)) if t == 1 and p == 0]

    print(f"\n[오탐(비위기→위기 감지)] {len(false_pos)}건 — 상위 5개:")
    for i in false_pos[:5]:
        print(f"  prob={entail_probs[i]:.3f}  {texts[i][:60]}")

    print(f"\n[미탐(위기→미감지)] {len(false_neg)}건 — 상위 5개:")
    for i in false_neg[:5]:
        print(f"  prob={entail_probs[i]:.3f}  {texts[i][:60]}")

    return {
        "n_samples":    len(labels),
        "false_pos":    len(false_pos),
        "false_neg":    len(false_neg),
    }


if __name__ == "__main__":
    evaluate_nli()
