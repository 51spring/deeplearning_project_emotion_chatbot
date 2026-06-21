"""
train_cbt_classification_head.py
역할: KoACD(한국 청소년 인지왜곡) 데이터셋으로 RoBERTa 본체 freeze + CBT 10범주
      분류 head 학습. cbt_anchors 임베딩 기반 이진 감지(cbt_score)와 보완 관계로,
      카테고리 식별 신호를 학습 기반으로 보강하기 위함.
입력:
  - data/raw/Cognitive_*.xlsx (KoACD 6개 파일)
  - models/roberta/checkpoints/roberta_final.pt (본체 freeze 베이스)
출력:
  - models/roberta/checkpoints/roberta_cbt_class_head.pt (head + 라벨 매핑)
  - models/roberta/checkpoints/cbt_class_report.json (학습 메타 + classification report)

실행: C:/Users/WD/anaconda3/envs/dl_study/python.exe models/roberta/train_cbt_classification_head.py

라이선스: KoACD CC BY 4.0 + research-only
  Kim & Kim (2025), KoACD: Findings of EMNLP 2025. https://github.com/cocoboldongle/KoACD
"""

import argparse
import glob
import json
import os
import random
import re
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from models.roberta.train_roberta import (  # noqa: E402
    MODEL_NAME, NUM_EMOTION_CLS, NUM_NLI_CLS, NUM_UTTERANCE_TYPE_CLS,
    NUM_CBT_CLS, RoBERTaMultiTask,
)

SEED = 42
MAX_LEN = 128
BATCH_SIZE = 32     # pre-tokenized 후 forward only — RTX 3060Ti 8GB 여유 충분
LR = 1e-3
EPOCHS = int(os.getenv("CBT_HEAD_EPOCHS", "10"))
PATIENCE = int(os.getenv("CBT_HEAD_PATIENCE", "4"))

DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")
CKPT_DIR = os.path.join(BASE_DIR, "models", "roberta", "checkpoints")
# worktree에 emotion_train.csv가 없으면 메인 리포로 fallback
_FALLBACK_PROC_DIR = r"C:\Users\WD\emotion_chatbot\data\processed"
def _resolve_proc(name: str) -> str:
    """processed CSV를 worktree 우선, 없으면 메인 리포로 해석."""
    local = os.path.join(PROC_DIR, name)
    return local if os.path.exists(local) else os.path.join(_FALLBACK_PROC_DIR, name)
BASE_CKPT = os.path.join(CKPT_DIR, "roberta_final.pt")
OUTPUT_CKPT = os.path.join(CKPT_DIR, "roberta_cbt_class_head.pt")
REPORT_PATH = os.path.join(CKPT_DIR, "cbt_class_report.json")
EMOTION_TRAIN_CSV = _resolve_proc("emotion_train.csv")

# KoACD 한글 라벨 → 우리 cbt_anchors 카테고리 → label id (0~10).
# v2(2026-04-27): 11번째 클래스 "비왜곡" 추가 — 일상/긍정 발화를 학습해 head 가
# 비왜곡을 명시적으로 감지하게 한다. 기존 10클래스 head 의 한계(KoACD 100% 왜곡 학습)
# 를 보완.
ID_TO_LABEL = {
    0: "이분법적 사고",
    1: "과잉일반화",
    2: "파국화",
    3: "자기비난·개인화",
    4: "감정적 추론",
    5: "부정적 편향",
    6: "낙인찍기",
    7: "긍정 축소화",
    8: "당위 진술",
    9: "성급한 판단",
    10: "비왜곡",
}
LABEL_TO_ID = {v: k for k, v in ID_TO_LABEL.items()}
NON_DISTORTION_LABEL_ID = 10
DISTORTION_LABEL_IDS = list(range(10))

KOACD_TO_OUR_LABEL = {
    "흑백 사고": "이분법적 사고",
    "과잉 일반화": "과잉일반화",
    "확대와 축소": "파국화",
    "개인화": "자기비난·개인화",
    "감정적 추론": "감정적 추론",
    "부정적 편향": "부정적 편향",
    "낙인찍기": "낙인찍기",
    "긍정 축소화": "긍정 축소화",
    "'해야 한다' 진술": "당위 진술",
    "해야 한다' 진술": "당위 진술",   # 오타 정규화
    "성급한 판단": "성급한 판단",
}

# 카테고리당 train/val 샘플 수 (stratified). 학습 시간 vs 신호 품질 트레이드오프.
TRAIN_PER_CAT = 3000
VAL_PER_CAT = 500
# 비왜곡 클래스는 단일 클래스라 다른 10개 카테고리 합산 만큼 데이터 보강 (5000/750)
NON_DISTORTION_TRAIN = 5000
NON_DISTORTION_VAL = 750

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_HEADER_RE = re.compile(r"\[[^\]]+\]\s*-+\s*", re.MULTILINE)


def set_seed(seed: int = SEED) -> None:
    """
    역할: 학습 재현성 시드 고정.
    입력: seed 정수
    출력: 없음
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def strip_header(text: str) -> str:
    """
    역할: KoACD Generated Story 의 청소년 헤더("[남자/16세]\n---\n")를 제거한다.
    입력: 원본 story 문자열
    출력: 헤더 제거된 본문
    """
    cleaned = _HEADER_RE.sub("", text)
    cleaned = cleaned.replace("---", " ")
    return cleaned.strip()


def load_non_distortion_data(train_n: int, val_n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    역할: emotion_train.csv 의 중립/행복 발화에서 비왜곡 샘플을 추출한다.
          노이즈 패턴(의성어/감탄 과다/극단 짧은 문장)을 제외하고 길이 적정선만 사용.
          11번째 클래스(NON_DISTORTION_LABEL_ID=10) 학습용.
    입력: 원하는 train/val 샘플 개수
    출력: (train_df, val_df) — columns=[text, label]
    """
    df = pd.read_csv(EMOTION_TRAIN_CSV)
    df = df[df["emotion"].isin(["중립", "행복"])]
    df["text"] = df["text"].astype(str)
    df["len"] = df["text"].str.len()
    df = df[(df["len"] >= 10) & (df["len"] <= 100)]
    bad = df["text"].str.contains(r"[?!]{2,}|ㅋ{2,}|ㅎ{2,}|ㅠ{2,}|ㅜ{2,}|ㅡ{2,}", regex=True)
    df = df[~bad].drop_duplicates(subset=["text"]).reset_index(drop=True)
    print(f"[load_non_distortion] 정제 후 가용 {len(df)} 건")

    # 행복:중립 비율을 1:2 정도로 맞춰 일상톤 다양성 확보
    happy = df[df["emotion"] == "행복"]
    neutral = df[df["emotion"] == "중립"]
    target_h = min(len(happy), (train_n + val_n) // 3)
    target_n = (train_n + val_n) - target_h
    happy_s = happy.sample(n=min(target_h, len(happy)), random_state=SEED)
    neut_s = neutral.sample(n=min(target_n, len(neutral)), random_state=SEED)
    pool = pd.concat([happy_s, neut_s]).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    pool["label"] = NON_DISTORTION_LABEL_ID
    pool["label_name"] = "비왜곡"
    pool = pool[["text", "label", "label_name"]]

    train_df = pool.iloc[:train_n].reset_index(drop=True)
    val_df = pool.iloc[train_n:train_n + val_n].reset_index(drop=True)
    print(f"[load_non_distortion] train={len(train_df)}, val={len(val_df)} (행복+중립)")
    return train_df, val_df


def load_koacd_dataset(train_per_cat: int, val_per_cat: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    역할: KoACD 6개 xlsx 를 통합 로드하고, 품질 필터(quality_mean ≥ 2.3, fluency_min ≥ 2)
          적용 후 카테고리당 train_per_cat + val_per_cat 만큼 stratified 샘플링한다.
    입력: 카테고리당 train/val 샘플 수
    출력: (train_df, val_df) — columns=[text, label]
          text: Generated Story 헤더 제거본
          label: 0~9 label id
    """
    score_pairs = [
        ("Gpt Consistency", "Gpt Accuracy", "Gpt Fluency"),
        ("Gemini Consistency", "Gemini Accuracy", "Gemini Fluency"),
        ("Claude Consistency", "Claude Accuracy", "Claude Fluency"),
    ]
    rows = []
    files = sorted(glob.glob(os.path.join(DATA_DIR, "Cognitive_*.xlsx")))
    print(f"[load] KoACD {len(files)}개 파일 로드 중...")
    for fp in files:
        df = pd.read_excel(fp)
        for _, r in df.iterrows():
            label_raw = str(r.get("Cognitive Distortion (Korean)", "")).strip()
            our_label = KOACD_TO_OUR_LABEL.get(label_raw)
            story = r.get("Generated Story")
            if our_label is None or pd.isna(story):
                continue
            scores, fluencies = [], []
            for cons_col, acc_col, flu_col in score_pairs:
                if cons_col not in df.columns:
                    continue
                cons, acc, flu = r.get(cons_col), r.get(acc_col), r.get(flu_col)
                if pd.isna(cons) or pd.isna(acc) or pd.isna(flu):
                    continue
                scores.extend([float(cons), float(acc), float(flu)])
                fluencies.append(float(flu))
            if not scores:
                continue
            quality_mean = float(np.mean(scores))
            fluency_min = float(min(fluencies)) if fluencies else 0.0
            if quality_mean < 2.3 or fluency_min < 2.0:
                continue
            text = strip_header(str(story))
            if len(text) < 20:
                continue
            rows.append({
                "text": text,
                "label": LABEL_TO_ID[our_label],
                "label_name": our_label,
            })
    df = pd.DataFrame(rows).drop_duplicates(subset=["text"]).reset_index(drop=True)
    print(f"[load] 품질 필터 + dedup 후 총 {len(df)} 건")
    print(f"[load] 카테고리별:")
    for n, lab in sorted([(c, ID_TO_LABEL[i]) for i, c in df["label"].value_counts().items()]):
        print(f"  {lab:14s}: {n}")

    # train/val stratified 샘플링: 왜곡 카테고리 0~9 만 (비왜곡은 별도 소스에서 로드)
    target_per_cat = train_per_cat + val_per_cat
    parts_train = []
    parts_val = []
    for label_id in DISTORTION_LABEL_IDS:
        sub = df[df["label"] == label_id]
        if len(sub) < target_per_cat:
            print(f"  [경고] label {label_id}({ID_TO_LABEL[label_id]}) "
                  f"가용 {len(sub)} < 목표 {target_per_cat} — 가용분만 사용")
            sub = sub.sample(frac=1.0, random_state=SEED)
        else:
            sub = sub.sample(n=target_per_cat, random_state=SEED)
        # 셔플 후 앞쪽 train_per_cat 은 train, 나머지 val
        sub = sub.sample(frac=1.0, random_state=SEED + label_id).reset_index(drop=True)
        parts_train.append(sub.iloc[:train_per_cat])
        parts_val.append(sub.iloc[train_per_cat:train_per_cat + val_per_cat])
    train_df = pd.concat(parts_train).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    val_df = pd.concat(parts_val).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    print(f"[split] train={len(train_df)}, val={len(val_df)}")
    return train_df, val_df


class CBTDataset(Dataset):
    """
    역할: KoACD + 비왜곡 CBT 분류 학습용 PyTorch Dataset (pre-tokenized 캐시).
          기존 v1 은 __getitem__ 안에서 tokenizer 호출하여 CPU 단일 스레드 병목으로
          epoch 당 7~8분 걸렸음(바람직한 1분 대비 6~8배). v2 부터 생성자에서 일괄
          토큰화하여 메모리에 캐싱하면 epoch 당 30~60초 수준으로 단축.
    입력: DataFrame(text, label), tokenizer, max_len
    출력: __getitem__ 시 input_ids, attention_mask, label tensor (이미 토큰화됨)
    """

    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int = MAX_LEN):
        """
        역할: 모든 텍스트를 일괄 토큰화하여 메모리에 캐싱한다.
        입력: df, tokenizer, max_len
        출력: 없음 (self.input_ids/attention_mask/labels 보관)
        """
        texts = df["text"].astype(str).tolist()
        self.labels = torch.tensor(df["label"].astype(int).tolist(), dtype=torch.long)

        # 일괄 토큰화 — 한 번만 실행되어 메모리에 (N, max_len) 으로 캐싱.
        # 60k 샘플 × 128 토큰 × 4B(int32) ≈ 30MB → RAM 여유 충분.
        print(f"[pre-tokenize] {len(texts)} 건 토큰화 중...")
        enc = tokenizer(
            texts,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        print(f"[pre-tokenize] 완료 — input_ids shape={tuple(self.input_ids.shape)}")

    def __len__(self) -> int:
        """역할: 데이터셋 크기. 입력: 없음. 출력: 샘플 개수."""
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        """
        역할: 캐시된 토큰을 인덱싱해 반환 (CPU 연산 없음).
        입력: 인덱스
        출력: input_ids, attention_mask, label dict
        """
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "label": self.labels[idx],
        }


def load_model() -> RoBERTaMultiTask:
    """
    역할: 기존 RoBERTa 본체 + 기존 head 들을 로드하고 신규 cbt_class_head 만 학습 가능하게
          설정한다 (본체와 다른 head 는 모두 freeze).
    입력: 없음
    출력: 학습 준비된 RoBERTaMultiTask 모델
    """
    model = RoBERTaMultiTask(
        MODEL_NAME, NUM_EMOTION_CLS, NUM_NLI_CLS,
        NUM_UTTERANCE_TYPE_CLS, NUM_CBT_CLS,
    ).to(DEVICE)
    state = torch.load(BASE_CKPT, map_location=DEVICE, weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[기존 체크포인트 로드] {BASE_CKPT}")
    if missing:
        print(f"[초기화된 신규 파라미터] {missing}")
    if unexpected:
        print(f"[사용하지 않은 파라미터] {unexpected}")

    for param in model.parameters():
        param.requires_grad = False
    for param in model.cbt_class_head.parameters():
        param.requires_grad = True
    return model


@torch.no_grad()
def evaluate(model: RoBERTaMultiTask, loader: DataLoader, criterion) -> tuple[float, float, dict]:
    """
    역할: val 셋 loss, Macro F1, classification report 계산.
    입력: 모델, val loader, loss 함수
    출력: (avg_loss, macro_f1, classification_report dict)
    """
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["label"].to(DEVICE)
        logits = model.forward_cbt_class(input_ids, attention_mask)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        preds = logits.argmax(dim=-1).cpu().numpy().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy().tolist())
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    report = classification_report(
        all_labels, all_preds,
        labels=list(ID_TO_LABEL.keys()),
        target_names=[ID_TO_LABEL[i] for i in ID_TO_LABEL],
        zero_division=0,
        output_dict=True,
    )
    return total_loss / max(len(loader), 1), macro_f1, report


def save_checkpoint(model: RoBERTaMultiTask, best_f1: float, epoch: int,
                    report: dict, train_size: int, val_size: int) -> None:
    """
    역할: cbt_class_head state_dict + 라벨 매핑 + 학습 메타를 저장한다.
    입력: 모델, best_f1, best_epoch, classification report, train/val 개수
    출력: 없음
    """
    payload = {
        "cbt_class_head": model.cbt_class_head.state_dict(),
        "label_to_id": LABEL_TO_ID,
        "id_to_label": ID_TO_LABEL,
        "best_macro_f1": best_f1,
        "best_epoch": epoch,
        "model_name": MODEL_NAME,
        "max_len": MAX_LEN,
        "train_size": train_size,
        "val_size": val_size,
    }
    torch.save(payload, OUTPUT_CKPT)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "best_macro_f1": best_f1,
            "best_epoch": epoch,
            "train_size": train_size,
            "val_size": val_size,
            "label_to_id": LABEL_TO_ID,
            "report": report,
            "data_source": "KoACD (Kim & Kim 2025, EMNLP Findings, CC BY 4.0 research-only)",
        }, f, ensure_ascii=False, indent=2)


def train(args) -> None:
    """
    역할: 학습 전체 흐름 실행 — 데이터 로드 → 모델 로드 → train/val 루프 → 저장.
    입력: argparse Namespace
    출력: 없음
    """
    set_seed()
    os.makedirs(CKPT_DIR, exist_ok=True)

    print("[1/5] KoACD 왜곡 10범주 로드")
    koacd_train, koacd_val = load_koacd_dataset(args.train_per_cat, args.val_per_cat)
    print("\n[2/5] 비왜곡(emotion_train 중립/행복) 로드")
    nondist_train, nondist_val = load_non_distortion_data(
        args.non_distortion_train, args.non_distortion_val,
    )
    train_df = pd.concat([koacd_train, nondist_train]).sample(
        frac=1.0, random_state=SEED).reset_index(drop=True)
    val_df = pd.concat([koacd_val, nondist_val]).sample(
        frac=1.0, random_state=SEED).reset_index(drop=True)
    print(f"\n[3/5] 통합 데이터셋: train={len(train_df)}, val={len(val_df)}")
    print(f"  train 라벨 분포:")
    for lid, n in sorted(train_df["label"].value_counts().items()):
        print(f"    {lid:2d} {ID_TO_LABEL[lid]:14s}: {n}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = CBTDataset(train_df, tokenizer)
    val_ds = CBTDataset(val_df, tokenizer)
    # num_workers 는 pre-tokenize 후 GPU 데이터 전송만 하면 되므로 0~2 충분
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = load_model()
    # 카테고리 균등 샘플링이라 별도 class weights 불필요 — 균등 가중
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.cbt_class_head.parameters(), lr=LR)

    best_f1 = 0.0
    best_epoch = 0
    no_improve = 0

    print(f"[학습 시작] device={DEVICE}, train={len(train_ds)}, val={len(val_ds)}, "
          f"batch={BATCH_SIZE}, lr={LR}, epochs={args.epochs}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_dl:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            optimizer.zero_grad()
            logits = model.forward_cbt_class(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_loss, val_f1, report = evaluate(model, val_dl, criterion)
        train_loss = total_loss / max(len(train_dl), 1)
        print(f"[epoch {epoch:02d}] train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} val_macro_f1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            no_improve = 0
            save_checkpoint(model, best_f1, best_epoch, report,
                            len(train_ds), len(val_ds))
            print(f"  [저장] best head checkpoint → {OUTPUT_CKPT}")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"[early stop] {args.patience} epoch 동안 개선 없음")
                break

    print(f"\n[완료] best_epoch={best_epoch}, best_macro_f1={best_f1:.4f}")


def main():
    """
    역할: CLI 진입점. 학습 인자 파싱 후 train() 호출.
    입력: --train-per-cat, --val-per-cat, --epochs, --patience
    출력: 없음
    """
    parser = argparse.ArgumentParser(description="CBT 11범주(10왜곡+비왜곡) 분류 head 학습")
    parser.add_argument("--train-per-cat", type=int, default=TRAIN_PER_CAT)
    parser.add_argument("--val-per-cat", type=int, default=VAL_PER_CAT)
    parser.add_argument("--non-distortion-train", type=int, default=NON_DISTORTION_TRAIN)
    parser.add_argument("--non-distortion-val", type=int, default=NON_DISTORTION_VAL)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
