"""
temperature_scaling.py
역할: 멀티태스크 RoBERTa 모델의 Temperature Scaling 보정
      calib 셋으로 T_emotion / T_nli 각각 독립 최적화 후 ECE 측정
실행: python models/roberta/temperature_scaling.py
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import f1_score, classification_report

# train_roberta.py 와 동일 환경 import
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_roberta import (
    RoBERTaMultiTask, EmotionDataset, NLIDataset, build_nli_subsets,
    NUM_EMOTION_CLS, NUM_NLI_CLS, MAX_LEN, BATCH_SIZE, DEVICE,
)
from transformers import AutoTokenizer

# ── 경로 ────────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR  = os.path.join(BASE_DIR, "data", "processed")
NLI_DIR   = os.path.join(BASE_DIR, "data", "nli")
CKPT_DIR  = os.path.join(BASE_DIR, "models", "roberta", "checkpoints")
MODEL_NAME = "klue/roberta-base"

EMOTION_LABELS = ["행복", "중립", "슬픔", "공포", "혐오", "분노", "놀람"]


# ────────────────────────────────────────────────────────────────────────────────
# ECE 계산
# ────────────────────────────────────────────────────────────────────────────────
def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """
    역할: Expected Calibration Error 계산 (등빈도 구간)
    입력: softmax 확률 (N, C), 정답 레이블 (N,)
    출력: ECE 값
    """
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct     = (predictions == labels).astype(float)

    ece   = 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)

    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        acc  = correct[mask].mean()
        conf = confidences[mask].mean()
        ece += (mask.sum() / len(labels)) * abs(acc - conf)

    return float(ece)


# ────────────────────────────────────────────────────────────────────────────────
# logit 수집 (모델 forward pass)
# ────────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def collect_logits_emotion(model, loader):
    """역할: calib 셋 전체 감정 logit + 정답 레이블 수집"""
    model.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attn_mask = batch["attention_mask"].to(DEVICE)
        emotion_logits, _ = model(input_ids, attn_mask)
        all_logits.append(emotion_logits.cpu())
        all_labels.append(batch["label"])
    return torch.cat(all_logits), torch.cat(all_labels)


@torch.no_grad()
def collect_logits_nli(model, loader):
    """역할: NLI calib logit + 정답 레이블 수집"""
    model.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attn_mask = batch["attention_mask"].to(DEVICE)
        _, nli_logits = model(input_ids, attn_mask)
        all_logits.append(nli_logits.cpu())
        all_labels.append(batch["label"])
    return torch.cat(all_logits), torch.cat(all_labels)


# ────────────────────────────────────────────────────────────────────────────────
# Temperature 최적화 (NLL 최소화)
# ────────────────────────────────────────────────────────────────────────────────
def optimize_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    역할: NLL 손실 최소화로 최적 Temperature 탐색 (스칼라 T)
    입력: logits (N, C), labels (N,)
    출력: 최적 T 값
    """
    log_T = nn.Parameter(torch.log(torch.ones(1) * 1.5))
    optimizer = torch.optim.LBFGS([log_T], lr=0.01, max_iter=200)
    criterion = nn.CrossEntropyLoss()

    def eval_step():
        optimizer.zero_grad()
        T = torch.exp(log_T).clamp_min(1e-4)
        loss = criterion(logits / T, labels)
        loss.backward()
        return loss

    optimizer.step(eval_step)
    return float(torch.exp(log_T).clamp_min(1e-4).item())


def optimize_vector_temperature(logits: torch.Tensor, labels: torch.Tensor) -> list:
    """
    역할: Vector Scaling — 클래스마다 별도 양수 스칼라 학습 (z_c / T_c)
    입력: logits (N, C), labels (N,)
    출력: 클래스별 Temperature list (길이 C, 양수)
    참고: 단일 T 와 달리 logit 의 클래스별 평탄화 정도가 달라 ECE 가
          크게 떨어지는 경향이 있다. 단조 변환은 아니므로 argmax 도 일부
          뒤집혀 F1·Accuracy 도 변할 수 있다(보통 개선).
    """
    n_cls = logits.shape[1]
    log_T = nn.Parameter(torch.zeros(n_cls))  # T = exp(0) = 1.0 시작
    optimizer = torch.optim.LBFGS([log_T], lr=0.01, max_iter=300)
    criterion = nn.CrossEntropyLoss()

    def eval_step():
        optimizer.zero_grad()
        T = torch.exp(log_T)  # 항상 양수
        loss = criterion(logits / T, labels)
        loss.backward()
        return loss

    optimizer.step(eval_step)
    return [float(x) for x in torch.exp(log_T).detach().tolist()]


def load_nli_calibration_loader(tokenizer):
    """
    역할: stage2 학습에서 저장한 NLI 검증 인덱스를 재사용해 held-out calibration 로더 생성
    입력: 토크나이저
    출력: NLI calibration DataLoader
    """
    nli_csv = os.path.join(NLI_DIR, "nli_pairs.csv")
    split_meta_path = os.path.join(CKPT_DIR, "nli_split.json")

    if os.path.exists(split_meta_path):
        with open(split_meta_path, "r", encoding="utf-8") as f:
            split_meta = json.load(f)
        val_indices = split_meta["val_indices"]
        full_ds = NLIDataset(nli_csv, tokenizer)
        calib_ds = Subset(full_ds, val_indices)
        print(f"NLI calibration split 로드 완료: {split_meta_path}")
    else:
        # 기존 체크포인트와의 호환을 위해 split 정보가 없으면 동일 규칙으로 재현
        _, calib_ds, _ = build_nli_subsets(nli_csv, tokenizer, val_ratio=0.2)
        print("NLI split 정보가 없어 동일 stratified 규칙으로 calibration split을 재구성합니다.")

    return DataLoader(calib_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


# ────────────────────────────────────────────────────────────────────────────────
# 상세 평가 (per-class F1)
# ────────────────────────────────────────────────────────────────────────────────
def detailed_eval(logits: torch.Tensor, labels: torch.Tensor,
                  T: float, label_names: list, title: str):
    """역할: Temperature 보정 전/후 per-class F1 + ECE 출력"""
    # 보정 전
    probs_raw  = torch.softmax(logits, dim=-1).numpy()
    preds_raw  = probs_raw.argmax(axis=1)
    labels_np  = labels.numpy()

    # 보정 후
    probs_cal  = torch.softmax(logits / T, dim=-1).numpy()
    preds_cal  = probs_cal.argmax(axis=1)

    ece_before = compute_ece(probs_raw, labels_np)
    ece_after  = compute_ece(probs_cal, labels_np)
    f1_before  = f1_score(labels_np, preds_raw, average="macro", zero_division=0)
    f1_after   = f1_score(labels_np, preds_cal, average="macro", zero_division=0)

    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(f"  T = {T:.4f}")
    print(f"  ECE: {ece_before:.4f} → {ece_after:.4f}  (목표 ≤ 0.05)")
    print(f"  Macro F1: {f1_before:.4f} → {f1_after:.4f}")
    print(f"\n[보정 후 per-class F1]")
    print(classification_report(labels_np, preds_cal, target_names=label_names,
                                 zero_division=0))
    return T, ece_after, f1_after


# ────────────────────────────────────────────────────────────────────────────────
# 균형 샘플 평가 (val 분포 불일치 진단용)
# ────────────────────────────────────────────────────────────────────────────────
def balanced_eval(model, tokenizer):
    """
    역할: val 셋에서 클래스당 최대 200개 균형 샘플링 후 감정 분류 성능 측정
          train 분포와 val 분포 불일치가 F1에 미치는 영향 진단
    """
    print("\n" + "="*60)
    print("val 균형 샘플 평가 (분포 불일치 진단)")
    print("="*60)

    val_df = pd.read_csv(os.path.join(DATA_DIR, "emotion_val.csv"), encoding="utf-8-sig")
    print(f"val 원본 분포:\n{val_df['emotion'].value_counts().to_string()}\n")

    # 클래스당 최대 200개 균형 샘플링
    balanced = val_df.groupby("label", group_keys=False).apply(
        lambda x: x.sample(min(len(x), 200), random_state=42)
    ).reset_index(drop=True)
    print(f"균형 샘플 분포:\n{balanced['emotion'].value_counts().to_string()}\n")

    # 임시 CSV 저장 후 Dataset 생성
    tmp_path = os.path.join(DATA_DIR, "_tmp_balanced_val.csv")
    balanced.to_csv(tmp_path, index=False, encoding="utf-8-sig")

    ds = EmotionDataset(tmp_path, tokenizer)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    logits, labels = collect_logits_emotion(model, dl)
    probs = torch.softmax(logits, dim=-1).numpy()
    preds = probs.argmax(axis=1)
    macro_f1 = f1_score(labels.numpy(), preds, average="macro", zero_division=0)

    print(f"균형 샘플 Macro F1 = {macro_f1:.4f}  (전체 val F1과 비교)")
    print(classification_report(labels.numpy(), preds,
                                  target_names=EMOTION_LABELS, zero_division=0))

    os.remove(tmp_path)
    return macro_f1


# ────────────────────────────────────────────────────────────────────────────────
# 메인
# ────────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Temperature Scaling 보정 시작")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = RoBERTaMultiTask(MODEL_NAME, NUM_EMOTION_CLS, NUM_NLI_CLS).to(DEVICE)
    model.load_state_dict(
        torch.load(os.path.join(CKPT_DIR, "roberta_final.pt"), weights_only=True)
    )
    model.eval()
    print("체크포인트 로드 완료: roberta_final.pt")

    # ── 감정 분류 Temperature Scaling ──────────────────────────────────────────
    calib_ds = EmotionDataset(os.path.join(DATA_DIR, "emotion_calib.csv"), tokenizer)
    calib_dl = DataLoader(calib_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    logits_e, labels_e = collect_logits_emotion(model, calib_dl)
    T_emotion = optimize_temperature(logits_e.clone(), labels_e.clone())

    T_emotion, ece_emotion, f1_emotion = detailed_eval(
        logits_e, labels_e, T_emotion, EMOTION_LABELS, "감정 분류 Temperature Scaling"
    )

    # ── Vector Scaling (클래스별 T) — 운영 채택 보정 방식 ────────────────────────
    vector_T_emotion = optimize_vector_temperature(logits_e.clone(), labels_e.clone())
    T_vec = torch.tensor(vector_T_emotion, dtype=logits_e.dtype)
    probs_vec = torch.softmax(logits_e / T_vec, dim=-1).numpy()
    preds_vec = probs_vec.argmax(axis=1)
    labels_e_np = labels_e.numpy()
    ece_emotion_vector = compute_ece(probs_vec, labels_e_np)
    f1_emotion_vector = f1_score(labels_e_np, preds_vec, average="macro", zero_division=0)
    print(f"\n{'='*60}")
    print("감정 분류 Vector Scaling (운영 채택)")
    print(f"{'='*60}")
    print(f"  ECE(calib): {ece_emotion:.4f} → {ece_emotion_vector:.4f}  (목표 ≤ 0.05)")
    print(f"  Macro F1(calib): {f1_emotion:.4f} → {f1_emotion_vector:.4f}")
    print("  vector_T_emotion = " + ", ".join(
        f"{name}={t:.3f}" for name, t in zip(EMOTION_LABELS, vector_T_emotion)
    ))

    # ── NLI Temperature Scaling ─────────────────────────────────────────────────
    nli_dl = load_nli_calibration_loader(tokenizer)

    logits_n, labels_n = collect_logits_nli(model, nli_dl)
    T_nli = optimize_temperature(logits_n.clone(), labels_n.clone())

    NLI_LABELS = ["entailment(위기)", "neutral", "contradiction(비위기)"]
    T_nli, ece_nli, f1_nli = detailed_eval(
        logits_n, labels_n, T_nli, NLI_LABELS, "NLI Temperature Scaling"
    )

    # ── val 균형 샘플 진단 ──────────────────────────────────────────────────────
    balanced_f1 = balanced_eval(model, tokenizer)

    # ── 결과 저장 ───────────────────────────────────────────────────────────────
    result = {
        "T_emotion":      round(T_emotion, 4),
        "T_nli":          round(T_nli, 4),
        "vector_T_emotion": [round(t, 6) for t in vector_T_emotion],
        "ece_emotion":    round(ece_emotion, 4),
        "ece_emotion_vector": round(ece_emotion_vector, 4),
        "ece_nli":        round(ece_nli, 4),
        "f1_emotion_calib": round(f1_emotion, 4),
        "f1_emotion_calib_vector": round(f1_emotion_vector, 4),
        "f1_nli_calib":     round(f1_nli, 4),
        "f1_emotion_balanced_val": round(balanced_f1, 4),
    }
    result_path = os.path.join(CKPT_DIR, "temperature_result.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("\n" + "="*60)
    print("최종 요약")
    print("="*60)
    for k, v in result.items():
        print(f"  {k}: {v}")
    print(f"\n결과 저장: {result_path}")
    print("="*60)


if __name__ == "__main__":
    main()
