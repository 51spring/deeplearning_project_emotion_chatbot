"""
train_roberta.py
역할: KLUE-RoBERTa-base 멀티태스크 학습 (감정 분류 + NLI 위기 감지)
2단계 분리 학습:
  1단계: RoBERTa 전체 + 감정 분류 헤드 (epoch=5, best val Macro F1 체크포인트)
  2단계: RoBERTa freeze + NLI 헤드만 (epoch=10~20, early stopping patience=5)
실행: conda activate dl_study → python models/roberta/train_roberta.py
"""

import os
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.amp import autocast, GradScaler

# ── 재현성 고정 ────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR    = os.path.join(BASE_DIR, "data", "processed")
NLI_DIR     = os.path.join(BASE_DIR, "data", "nli")
CKPT_DIR    = os.path.join(BASE_DIR, "models", "roberta", "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

# ── 하이퍼파라미터 ────────────────────────────────────────────────────────────
MODEL_NAME       = "klue/roberta-base"
MAX_LEN          = 128
BATCH_SIZE       = 8          # RTX 3060Ti 8GB 기준
STAGE1_LR        = 2e-5
STAGE1_EPOCHS    = 15         # early stopping으로 제어
STAGE1_PATIENCE  = 3          # val_loss 기반 early stopping (감정)
STAGE2_LR        = 1e-3
STAGE2_EPOCHS    = 30
PATIENCE         = 5          # NLI early stopping
NUM_EMOTION_CLS  = 7
NUM_NLI_CLS      = 3
NUM_UTTERANCE_TYPE_CLS = 7
NUM_CBT_CLS      = 11   # KoACD 10범주 인지왜곡 + 비왜곡(정상) 1클래스 = 11. v2(2026-04-27 03:15)
                        # 비왜곡 클래스 추가 이유: 기존 10클래스 head 가 비왜곡 발화를 모름 →
                        # 일상/긍정 발화에서도 균등 임의 카테고리 강제 → false positive.
                        # 11번째 클래스(비왜곡, label_id=10)로 이진 감지 + 카테고리 식별 동시 수행.
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 멀티태스크 손실 가중치
EMOTION_LOSS_W   = 0.7
NLI_LOSS_W       = 0.3


# ────────────────────────────────────────────────────────────────────────────────
# 데이터셋
# ────────────────────────────────────────────────────────────────────────────────
class EmotionDataset(Dataset):
    """역할: 감정 분류용 텍스트 데이터셋"""

    def __init__(self, csv_path: str, tokenizer, max_len: int = MAX_LEN):
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        self.texts  = df["text"].tolist()
        self.labels = df["label"].tolist()
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
        }


class NLIDataset(Dataset):
    """역할: NLI 위기 감지용 premise-hypothesis 데이터셋"""

    def __init__(self, csv_path: str, tokenizer, max_len: int = MAX_LEN):
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        self.premises    = df["premise"].tolist()
        self.hypotheses  = df["hypothesis"].tolist()
        self.labels      = df["label"].tolist()
        self.tokenizer   = tokenizer
        self.max_len     = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.premises[idx]),
            str(self.hypotheses[idx]),
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ────────────────────────────────────────────────────────────────────────────────
# 멀티태스크 모델
# ────────────────────────────────────────────────────────────────────────────────
class RoBERTaMultiTask(nn.Module):
    """
    역할: 감정 분류 헤드 + NLI 위기 감지 헤드 + 발화 타입 헤드를 공유 RoBERTa 인코더 위에 구성
    입력: input_ids, attention_mask
    출력: emotion_logits (7cls), nli_logits (3cls)
    """

    def __init__(
        self,
        model_name: str,
        num_emotion: int,
        num_nli: int,
        num_utterance_type: int = NUM_UTTERANCE_TYPE_CLS,
        num_cbt: int = NUM_CBT_CLS,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.encoder       = AutoModel.from_pretrained(model_name)
        hidden_size        = self.encoder.config.hidden_size  # 768

        self.emotion_head  = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_emotion),
        )
        self.nli_head      = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_nli),
        )
        self.utterance_type_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_utterance_type),
        )
        # KoACD 기반 CBT 10범주 분류 head. 본체 freeze 후 별도 학습되며, 임베딩 anchor
        # 기반 cbt_score(이진 감지)와 보완 관계 — 카테고리 식별 신호로 사용.
        self.cbt_class_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_cbt),
        )

    def forward(self, input_ids, attention_mask):
        """
        역할: 기존 감정/NLI 추론 호환을 위해 두 head의 logits만 반환한다.
        입력: token id, attention mask
        출력: emotion_logits, nli_logits
        """
        outputs   = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb   = outputs.last_hidden_state[:, 0, :]  # [CLS] 토큰

        emotion_logits = self.emotion_head(cls_emb)
        nli_logits     = self.nli_head(cls_emb)
        return emotion_logits, nli_logits

    def forward_utterance_type(self, input_ids, attention_mask):
        """
        역할: 발화 타입 분류 head의 logits를 반환한다.
        입력: token id, attention mask
        출력: utterance_type_logits (4cls)
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0, :]
        return self.utterance_type_head(cls_emb)

    def forward_cbt_class(self, input_ids, attention_mask):
        """
        역할: CBT 10범주 분류 head의 logits를 반환한다 (KoACD 기반 학습).
        입력: token id, attention mask
        출력: cbt_class_logits (10cls)
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0, :]
        return self.cbt_class_head(cls_emb)

    def get_cls_embedding(self, input_ids, attention_mask):
        """역할: CBT 유사도 계산용 [CLS] 임베딩 반환"""
        with torch.no_grad():
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]


# ────────────────────────────────────────────────────────────────────────────────
# 클래스 가중치 계산
# ────────────────────────────────────────────────────────────────────────────────
def compute_class_weights(csv_path: str, num_classes: int) -> torch.Tensor:
    """역할: 불균형 클래스에 대한 역빈도 가중치 계산"""
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    counts = df["label"].value_counts().sort_index()
    # 누락 클래스 0으로 채우기
    counts = counts.reindex(range(num_classes), fill_value=1)
    weights = 1.0 / counts.values.astype(float)
    weights = weights / weights.sum() * num_classes  # 정규화
    return torch.tensor(weights, dtype=torch.float32)


def build_nli_subsets(csv_path: str, tokenizer, val_ratio: float = 0.2):
    """
    역할: NLI CSV를 레이블 기준 stratified split으로 train/val 서브셋으로 분리
    입력: NLI CSV 경로, 토크나이저, 검증 비율
    출력: (train_subset, val_subset, split_meta)
    """
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    indices = np.arange(len(df))

    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_ratio,
        random_state=SEED,
        stratify=df["label"],
    )

    full_ds = NLIDataset(csv_path, tokenizer)
    split_meta = {
        "seed": SEED,
        "val_ratio": val_ratio,
        "train_indices": train_idx.tolist(),
        "val_indices": val_idx.tolist(),
    }
    return Subset(full_ds, train_idx.tolist()), Subset(full_ds, val_idx.tolist()), split_meta


# ────────────────────────────────────────────────────────────────────────────────
# 평가 함수
# ────────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_emotion(model, loader, criterion):
    """역할: 감정 분류 val loss + Macro F1 계산"""
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    for batch in loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attn_mask = batch["attention_mask"].to(DEVICE)
        labels    = batch["label"].to(DEVICE)

        emotion_logits, _ = model(input_ids, attn_mask)
        loss = criterion(emotion_logits, labels)
        total_loss += loss.item()

        preds = emotion_logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return total_loss / len(loader), macro_f1


@torch.no_grad()
def evaluate_nli(model, loader, criterion):
    """역할: NLI val loss + Macro F1 계산"""
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    for batch in loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attn_mask = batch["attention_mask"].to(DEVICE)
        labels    = batch["label"].to(DEVICE)

        _, nli_logits = model(input_ids, attn_mask)
        loss = criterion(nli_logits, labels)
        total_loss += loss.item()

        preds = nli_logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return total_loss / len(loader), macro_f1


# ────────────────────────────────────────────────────────────────────────────────
# 1단계 학습: 감정 분류
# ────────────────────────────────────────────────────────────────────────────────
def stage1_train(model, tokenizer):
    """역할: RoBERTa 전체 파라미터 + 감정 헤드 학습"""
    print("\n" + "=" * 60)
    print("1단계: 감정 분류 학습")
    print("=" * 60)

    train_ds = EmotionDataset(os.path.join(DATA_DIR, "emotion_train.csv"), tokenizer)
    val_ds   = EmotionDataset(os.path.join(DATA_DIR, "emotion_val.csv"),   tokenizer)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 클래스 가중치 + label smoothing으로 과적합 방지
    class_weights = compute_class_weights(
        os.path.join(DATA_DIR, "emotion_train.csv"), NUM_EMOTION_CLS
    ).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=STAGE1_LR, weight_decay=0.05)
    use_amp   = DEVICE.type == "cuda"
    scaler    = GradScaler(device="cuda" if use_amp else "cpu", enabled=use_amp)

    # linear warmup (1 epoch) + linear decay
    total_steps  = STAGE1_EPOCHS * len(train_dl)
    warmup_steps = len(train_dl)  # 1 에폭 warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    best_f1, best_epoch = 0.0, 0
    best_val_loss = float("inf")
    no_improve = 0
    history = []

    for epoch in range(1, STAGE1_EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for batch in train_dl:
            input_ids = batch["input_ids"].to(DEVICE)
            attn_mask = batch["attention_mask"].to(DEVICE)
            labels    = batch["label"].to(DEVICE)

            optimizer.zero_grad()
            with autocast(device_type=DEVICE.type, enabled=use_amp):
                emotion_logits, _ = model(input_ids, attn_mask)
                loss = criterion(emotion_logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item()

        val_loss, val_f1 = evaluate_emotion(model, val_dl, criterion)
        train_loss = total_loss / len(train_dl)

        print(f"  Epoch {epoch}/{STAGE1_EPOCHS} | "
              f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
              f"val_Macro_F1={val_f1:.4f}")

        history.append({"epoch": epoch, "train_loss": train_loss,
                         "val_loss": val_loss, "val_f1": val_f1})

        # best F1 체크포인트 저장
        if val_f1 > best_f1:
            best_f1, best_epoch = val_f1, epoch
            ckpt_path = os.path.join(CKPT_DIR, "stage1_best.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ✓ best 체크포인트 저장 (val_F1={best_f1:.4f})")

        # val_loss 기반 early stopping (과적합 제어)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= STAGE1_PATIENCE:
                print(f"  Early stopping (val_loss patience={STAGE1_PATIENCE})")
                break

    print(f"\n[1단계 완료] best epoch={best_epoch}, best val_Macro_F1={best_f1:.4f}")
    print(f"  체크포인트: {os.path.join(CKPT_DIR, 'stage1_best.pt')}")

    # 학습 이력 저장
    with open(os.path.join(CKPT_DIR, "stage1_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # best 가중치 로드
    model.load_state_dict(torch.load(os.path.join(CKPT_DIR, "stage1_best.pt"), weights_only=True))
    return model


# ────────────────────────────────────────────────────────────────────────────────
# 2단계 학습: NLI 위기 감지
# ────────────────────────────────────────────────────────────────────────────────
def stage2_train(model, tokenizer):
    """역할: RoBERTa 인코더 freeze 후 NLI 헤드만 학습 (early stopping)"""
    print("\n" + "=" * 60)
    print("2단계: NLI 위기 감지 학습 (RoBERTa freeze)")
    print("=" * 60)

    # 인코더 파라미터 고정
    for param in model.encoder.parameters():
        param.requires_grad = False

    nli_csv = os.path.join(NLI_DIR, "nli_pairs.csv")
    # NLI 데이터를 레이블 기준 stratified split으로 분리해 검증 분포를 안정화
    train_ds, val_ds, split_meta = build_nli_subsets(nli_csv, tokenizer, val_ratio=0.2)
    split_meta_path = os.path.join(CKPT_DIR, "nli_split.json")
    with open(split_meta_path, "w", encoding="utf-8") as f:
        json.dump(split_meta, f, indent=2, ensure_ascii=False)
    print(f"  NLI split 정보 저장: {split_meta_path}")

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    criterion = nn.CrossEntropyLoss()
    # NLI 헤드 파라미터만 업데이트
    optimizer = torch.optim.Adam(
        [p for p in model.nli_head.parameters() if p.requires_grad],
        lr=STAGE2_LR
    )

    best_f1, best_epoch, no_improve = 0.0, 0, 0
    history = []

    for epoch in range(1, STAGE2_EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for batch in train_dl:
            input_ids = batch["input_ids"].to(DEVICE)
            attn_mask = batch["attention_mask"].to(DEVICE)
            labels    = batch["label"].to(DEVICE)

            optimizer.zero_grad()
            _, nli_logits = model(input_ids, attn_mask)
            loss = criterion(nli_logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_loss, val_f1 = evaluate_nli(model, val_dl, criterion)
        train_loss = total_loss / len(train_dl)

        print(f"  Epoch {epoch}/{STAGE2_EPOCHS} | "
              f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
              f"val_Macro_F1={val_f1:.4f}")

        history.append({"epoch": epoch, "train_loss": train_loss,
                         "val_loss": val_loss, "val_f1": val_f1})

        if val_f1 > best_f1:
            best_f1, best_epoch, no_improve = val_f1, epoch, 0
            ckpt_path = os.path.join(CKPT_DIR, "stage2_best.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ✓ best 체크포인트 저장 (val_F1={best_f1:.4f})")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  Early stopping (patience={PATIENCE})")
                break

    print(f"\n[2단계 완료] best epoch={best_epoch}, best val_Macro_F1={best_f1:.4f}")
    print(f"  체크포인트: {os.path.join(CKPT_DIR, 'stage2_best.pt')}")

    with open(os.path.join(CKPT_DIR, "stage2_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # 인코더 파라미터 해제 (추론 시 필요)
    for param in model.encoder.parameters():
        param.requires_grad = True

    model.load_state_dict(torch.load(os.path.join(CKPT_DIR, "stage2_best.pt"), weights_only=True))
    return model


# ────────────────────────────────────────────────────────────────────────────────
# 메인
# ────────────────────────────────────────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)} | "
              f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print(f"\n모델 로딩: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = RoBERTaMultiTask(MODEL_NAME, NUM_EMOTION_CLS, NUM_NLI_CLS).to(DEVICE)

    # gradient checkpointing으로 VRAM 절약
    model.encoder.gradient_checkpointing_enable()

    # 1단계: 감정 분류
    model = stage1_train(model, tokenizer)

    # 2단계: NLI 위기 감지
    model = stage2_train(model, tokenizer)

    # 최종 체크포인트 저장
    final_path = os.path.join(CKPT_DIR, "roberta_final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"\n[최종 모델 저장] → {final_path}")


if __name__ == "__main__":
    main()
