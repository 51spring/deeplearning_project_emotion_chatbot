"""
train_semantic_emotion.py
역할: Semantic Emotion Judge Phase 3 multi-head 학습을 실행한다.
입력:
  - data/processed/semantic_emotion_train.csv
  - data/processed/semantic_emotion_val.csv
  - data/nli/nli_pairs.csv
출력:
  - models/roberta/checkpoints/{run_name}_best.pt
  - models/roberta/checkpoints/{run_name}_final.pt
  - models/roberta/checkpoints/{run_name}_history.json
실행:
  C:/Users/WD/anaconda3/envs/dl_study/python.exe models/roberta/train_semantic_emotion.py --epochs 3
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.roberta.semantic_emotion_model import (  # noqa: E402
    NUM_DISTRESS_CLS,
    NUM_NLI_CLS,
    NUM_SEMANTIC_EMOTION_CLS,
    SemanticEmotionRoBERTa,
)


SEED = 42
MODEL_NAME = "klue/roberta-base"
MAX_LEN = 128
DEFAULT_BATCH_SIZE = 8
DATA_DIR = PROJECT_ROOT / "data" / "processed"
NLI_DIR = PROJECT_ROOT / "data" / "nli"
NLI_PATH = NLI_DIR / "nli_pairs.csv"
CKPT_DIR = PROJECT_ROOT / "models" / "roberta" / "checkpoints"
LABEL_MAP_PATH = DATA_DIR / "semantic_emotion_label_map.json"

EMOTION_NAMES = ["행복", "중립", "슬픔", "공포", "혐오", "분노", "놀람"]
DISTRESS_NAMES = {
    0: "calm_or_positive",
    1: "mild_distress",
    2: "moderate_distress",
    3: "high_distress",
    4: "crisis_candidate",
}


def set_seed(seed: int = SEED) -> None:
    """
    역할: 학습 재현성을 위해 주요 random seed를 고정한다.
    입력: seed 정수
    출력: 없음
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class TrainConfig:
    """
    역할: Phase 3 학습 설정을 체크포인트에 함께 저장한다.
    입력: argparse에서 받은 하이퍼파라미터
    출력: asdict로 직렬화 가능한 설정 객체
    """

    run_name: str
    epochs: int
    batch_size: int
    max_len: int
    lr: float
    nli_lr: float
    weight_decay: float
    warmup_ratio: float
    semantic_loss_weight: float
    distress_loss_weight: float
    nli_loss_weight: float
    weak_weight_cap: float
    label_smoothing: float
    distress_label_smoothing: float
    max_train_samples: int | None
    max_val_samples: int | None
    max_nli_samples: int | None
    init_ckpt: str
    use_nli_aux: bool
    patience: int
    log_every: int
    weak_source_fraction: float
    weak_sample_seed: int
    nli_train_csv: str
    nli_holdout_csv: str | None
    train_csv: str
    val_csv: str


class SemanticEmotionDataset(Dataset):
    """
    역할: semantic emotion CSV를 PyTorch Dataset으로 변환한다.
    입력: CSV 경로, tokenizer, max_len, 선택적 샘플 수 제한
    출력: tokenized batch item
    """

    def __init__(
        self,
        csv_path: Path,
        tokenizer,
        max_len: int,
        weak_weight_cap: float,
        max_samples: int | None = None,
        weak_source_fraction: float = 1.0,
        weak_sample_seed: int = SEED,
    ) -> None:
        """
        역할: CSV를 읽고 학습에 필요한 컬럼을 tensor 준비 형태로 보관한다.
        입력: CSV 경로, tokenizer, 최대 길이, weak label loss 상한, 샘플 제한,
              weak source별 sampling fraction, 샘플링 시드
        출력: Dataset 인스턴스
        """
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        # weak label source별 fraction sampling — original_emotion(=gold)은 그대로 둔다.
        if 0.0 < weak_source_fraction < 1.0 and "label_source" in df.columns:
            rng = np.random.default_rng(weak_sample_seed)
            keep_indices: list[int] = []
            self.weak_sample_audit: dict[str, dict[str, int]] = {}
            for source, sub in df.groupby("label_source"):
                source_str = str(source)
                if source_str == "original_emotion":
                    keep_indices.extend(int(idx) for idx in sub.index)
                    self.weak_sample_audit[source_str] = {
                        "before": int(len(sub)),
                        "after": int(len(sub)),
                    }
                    continue
                sample_n = max(1, int(round(len(sub) * weak_source_fraction)))
                chosen = rng.choice(sub.index.to_numpy(), size=sample_n, replace=False)
                keep_indices.extend(int(idx) for idx in chosen)
                self.weak_sample_audit[source_str] = {
                    "before": int(len(sub)),
                    "after": int(sample_n),
                }
            df = df.loc[sorted(keep_indices)].reset_index(drop=True)
        else:
            self.weak_sample_audit = {}

        if max_samples is not None:
            df = df.head(int(max_samples)).copy()
        self.texts = df["text"].astype(str).tolist()
        self.emotion_labels = df["label"].astype(int).tolist()
        self.distress_labels = df["distress_level"].astype(int).tolist()
        self.is_weak = df["is_weak_label"].astype(str).str.lower().isin(["true", "1"]).tolist()
        raw_weights = df["sample_weight"].astype(float).clip(lower=0.05, upper=1.0).tolist()
        self.sample_weights = [
            min(float(weight), weak_weight_cap) if weak else 1.0
            for weight, weak in zip(raw_weights, self.is_weak)
        ]
        raw_distress_weights = df.get(
            "distress_sample_weight",
            pd.Series(self.sample_weights),
        ).astype(float).clip(lower=0.0, upper=1.0).tolist()
        self.distress_sample_weights = [
            min(float(weight), weak_weight_cap) if weak else float(weight)
            for weight, weak in zip(raw_distress_weights, self.is_weak)
        ]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        """
        역할: Dataset 크기를 반환한다.
        입력: 없음
        출력: 행 수
        """
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        역할: 한 발화를 RoBERTa 입력 tensor와 label tensor로 변환한다.
        입력: row index
        출력: input_ids, attention_mask, emotion/distress label, sample_weight, is_weak
        """
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "emotion_label": torch.tensor(self.emotion_labels[idx], dtype=torch.long),
            "distress_label": torch.tensor(self.distress_labels[idx], dtype=torch.long),
            "sample_weight": torch.tensor(self.sample_weights[idx], dtype=torch.float32),
            "distress_sample_weight": torch.tensor(
                self.distress_sample_weights[idx],
                dtype=torch.float32,
            ),
            "is_weak": torch.tensor(bool(self.is_weak[idx]), dtype=torch.bool),
        }


class NLIPairDataset(Dataset):
    """
    역할: 기존 위기 감지 NLI pair CSV를 보조 학습 Dataset으로 변환한다.
    입력: nli_pairs.csv, tokenizer
    출력: pair tokenized batch item
    """

    def __init__(self, csv_path: Path, tokenizer, max_len: int, max_samples: int | None = None) -> None:
        """
        역할: NLI CSV를 읽고 필요 시 앞부분 일부만 사용한다.
        입력: CSV 경로, tokenizer, 최대 길이, 샘플 제한
        출력: Dataset 인스턴스
        """
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        if max_samples is not None:
            df = df.head(int(max_samples)).copy()
        self.premises = df["premise"].astype(str).tolist()
        self.hypotheses = df["hypothesis"].astype(str).tolist()
        self.labels = df["label"].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        """
        역할: Dataset 크기를 반환한다.
        입력: 없음
        출력: 행 수
        """
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        역할: premise-hypothesis 한 쌍을 RoBERTa pair 입력으로 변환한다.
        입력: row index
        출력: input_ids, attention_mask, NLI label
        """
        enc = self.tokenizer(
            self.premises[idx],
            self.hypotheses[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def weighted_ce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    sample_weights: torch.Tensor,
    criterion: nn.Module,
) -> torch.Tensor:
    """
    역할: 샘플별 weak-label weight를 반영한 cross entropy loss를 계산한다.
    입력: logits, labels, sample_weights, reduction='none' criterion
    출력: batch 평균 weighted loss
    """
    per_item_loss = criterion(logits, labels)
    return (per_item_loss * sample_weights).mean()


def compute_class_weights(csv_path: Path, label_col: str, num_classes: int) -> torch.Tensor:
    """
    역할: sample_weight를 반영한 역빈도 클래스 가중치를 계산한다.
    입력: CSV 경로, 라벨 컬럼명, 클래스 수
    출력: torch float tensor
    """
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    weights = df.get("sample_weight", pd.Series(np.ones(len(df)))).astype(float)
    # pandas groupby.apply 경고를 피하기 위해 임시 weight 컬럼을 명시적으로 집계한다.
    sums = df.assign(_effective_weight=weights).groupby(label_col)["_effective_weight"].sum()
    counts = sums.reindex(range(num_classes), fill_value=1.0).clip(lower=1.0)
    inv = 1.0 / counts.to_numpy(dtype=float)
    inv = inv / inv.sum() * num_classes
    return torch.tensor(inv, dtype=torch.float32)


def build_nli_loaders(tokenizer, config: TrainConfig) -> tuple[DataLoader, DataLoader, dict[str, Any]]:
    """
    역할: NLI 보조 학습용 train/val loader와 split meta를 만든다.
    입력: tokenizer, 학습 설정
    출력: train loader, val loader, split 메타데이터
    설명:
      - `nli_train_csv`만 주어지면 기존 방식대로 80/20 random split을 만든다.
      - `nli_holdout_csv`가 같이 주어지면 train CSV 전체를 학습에, holdout CSV를 val에 사용한다 (overlap 검증 포함).
    """
    train_path = Path(config.nli_train_csv)
    if not train_path.is_absolute():
        train_path = PROJECT_ROOT / train_path
    if not train_path.exists():
        raise FileNotFoundError(f"NLI train CSV 없음: {train_path}")

    raw_train_df = pd.read_csv(train_path, encoding="utf-8-sig")
    if config.max_nli_samples is not None:
        raw_train_df = raw_train_df.head(int(config.max_nli_samples)).reset_index(drop=True)

    train_dataset = NLIPairDataset(train_path, tokenizer, config.max_len, config.max_nli_samples)

    if config.nli_holdout_csv:
        holdout_path = Path(config.nli_holdout_csv)
        if not holdout_path.is_absolute():
            holdout_path = PROJECT_ROOT / holdout_path
        if not holdout_path.exists():
            raise FileNotFoundError(f"NLI holdout CSV 없음: {holdout_path}")
        holdout_df = pd.read_csv(holdout_path, encoding="utf-8-sig")
        train_keys = set(zip(raw_train_df["premise"].astype(str), raw_train_df["hypothesis"].astype(str)))
        holdout_keys = set(zip(holdout_df["premise"].astype(str), holdout_df["hypothesis"].astype(str)))
        overlap = train_keys & holdout_keys
        if overlap:
            raise RuntimeError(
                f"NLI train과 holdout 사이에 {len(overlap)}쌍 중복 발견 — split_nli_holdout.py 재실행 필요"
            )
        holdout_dataset = NLIPairDataset(holdout_path, tokenizer, config.max_len, None)
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(holdout_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
        meta = {
            "seed": SEED,
            "source_train": str(train_path.relative_to(PROJECT_ROOT)),
            "source_holdout": str(holdout_path.relative_to(PROJECT_ROOT)),
            "max_nli_samples": config.max_nli_samples,
            "train_size": int(len(raw_train_df)),
            "val_size": int(len(holdout_df)),
            "split_mode": "explicit_holdout",
            "overlap_pairs": 0,
        }
        return train_loader, val_loader, meta

    # 기존 동작 호환: train CSV 안에서 80/20 random split
    indices = np.arange(len(raw_train_df))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=SEED,
        stratify=raw_train_df["label"],
    )
    train_loader = DataLoader(Subset(train_dataset, train_idx.tolist()), batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(Subset(train_dataset, val_idx.tolist()), batch_size=config.batch_size, shuffle=False, num_workers=0)
    meta = {
        "seed": SEED,
        "source_train": str(train_path.relative_to(PROJECT_ROOT)),
        "max_nli_samples": config.max_nli_samples,
        "train_size": int(len(train_idx)),
        "val_size": int(len(val_idx)),
        "train_indices": train_idx.tolist(),
        "val_indices": val_idx.tolist(),
        "split_mode": "internal_random",
    }
    return train_loader, val_loader, meta


def evaluate_semantic(
    model: SemanticEmotionRoBERTa,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    """
    역할: semantic emotion/distress 검증 지표를 계산한다.
    입력: 모델, DataLoader, device
    출력: macro F1과 gold/weak subset 지표 dict
    """
    model.eval()
    emotion_preds: list[int] = []
    emotion_labels: list[int] = []
    distress_preds: list[int] = []
    distress_labels: list[int] = []
    weak_flags: list[bool] = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            result = model(input_ids, attention_mask, include_nli=False)
            emotion_preds.extend(result["semantic_emotion_logits"].argmax(dim=-1).cpu().tolist())
            distress_preds.extend(result["distress_logits"].argmax(dim=-1).cpu().tolist())
            emotion_labels.extend(batch["emotion_label"].cpu().tolist())
            distress_labels.extend(batch["distress_label"].cpu().tolist())
            weak_flags.extend(batch["is_weak"].cpu().tolist())

    def _macro(labels: list[int], preds: list[int]) -> float:
        """
        역할: 빈 subset에도 안전한 macro F1을 계산한다.
        입력: label/pred 리스트
        출력: macro F1 float
        """
        if not labels:
            return 0.0
        return float(f1_score(labels, preds, average="macro", zero_division=0))

    gold_mask = [not flag for flag in weak_flags]
    weak_mask = [bool(flag) for flag in weak_flags]
    metrics = {
        "semantic_macro_f1": _macro(emotion_labels, emotion_preds),
        "distress_macro_f1": _macro(distress_labels, distress_preds),
        "semantic_gold_macro_f1": _macro(
            [label for label, keep in zip(emotion_labels, gold_mask) if keep],
            [pred for pred, keep in zip(emotion_preds, gold_mask) if keep],
        ),
        "semantic_weak_macro_f1": _macro(
            [label for label, keep in zip(emotion_labels, weak_mask) if keep],
            [pred for pred, keep in zip(emotion_preds, weak_mask) if keep],
        ),
        "distress_gold_macro_f1": _macro(
            [label for label, keep in zip(distress_labels, gold_mask) if keep],
            [pred for pred, keep in zip(distress_preds, gold_mask) if keep],
        ),
        "distress_weak_macro_f1": _macro(
            [label for label, keep in zip(distress_labels, weak_mask) if keep],
            [pred for pred, keep in zip(distress_preds, weak_mask) if keep],
        ),
        "rows": len(emotion_labels),
        "gold_rows": int(sum(gold_mask)),
        "weak_rows": int(sum(weak_mask)),
    }
    return metrics


def evaluate_nli(model: SemanticEmotionRoBERTa, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    """
    역할: 보조 NLI head의 검증 macro F1을 계산한다.
    입력: 모델, NLI DataLoader, device
    출력: NLI macro F1 및 클래스별 F1
    """
    model.eval()
    preds: list[int] = []
    labels: list[int] = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model.forward_nli(input_ids, attention_mask)
            preds.extend(logits.argmax(dim=-1).cpu().tolist())
            labels.extend(batch["label"].cpu().tolist())

    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, labels=list(range(NUM_NLI_CLS)), zero_division=0
    )
    return {
        "nli_macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "per_class": {
            str(idx): {
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1[idx]),
                "support": int(support[idx]),
            }
            for idx in range(NUM_NLI_CLS)
        },
    }


def save_checkpoint(
    path: Path,
    model: SemanticEmotionRoBERTa,
    config: TrainConfig,
    epoch: int,
    metrics: dict[str, Any],
    legacy_load_summary: dict[str, list[str]],
) -> None:
    """
    역할: 새 이름의 Phase 3 체크포인트를 저장한다.
    입력: 저장 경로, 모델, 설정, epoch, 지표, legacy 로드 요약
    출력: 없음
    """
    payload = {
        "model_state_dict": model.state_dict(),
        "model_name": MODEL_NAME,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "epoch": epoch,
        "metrics": metrics,
        "config": asdict(config),
        "emotion_names": EMOTION_NAMES,
        "distress_names": DISTRESS_NAMES,
        "legacy_load_summary": {
            "copied_count": len(legacy_load_summary.get("copied", [])),
            "skipped_count": len(legacy_load_summary.get("skipped", [])),
        },
    }
    torch.save(payload, path)


def train(config: TrainConfig) -> dict[str, Any]:
    """
    역할: Phase 3 semantic emotion/distress/NLI 보조 학습 루프를 실행한다.
    입력: TrainConfig
    출력: 최종 학습 이력과 best checkpoint 정보
    """
    if config.batch_size > 8:
        raise ValueError("batch_size는 RTX 3060Ti 8GB 기준 8 이하만 허용한다.")

    set_seed(SEED)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Device: {device}")
    if use_amp:
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # train_csv / val_csv가 절대경로면 그대로, 상대경로면 PROJECT_ROOT 기준 해석
    train_csv_path = Path(config.train_csv)
    if not train_csv_path.is_absolute():
        train_csv_path = PROJECT_ROOT / train_csv_path
    val_csv_path = Path(config.val_csv)
    if not val_csv_path.is_absolute():
        val_csv_path = PROJECT_ROOT / val_csv_path

    train_ds = SemanticEmotionDataset(
        train_csv_path,
        tokenizer,
        config.max_len,
        config.weak_weight_cap,
        config.max_train_samples,
        weak_source_fraction=config.weak_source_fraction,
        weak_sample_seed=config.weak_sample_seed,
    )
    val_ds = SemanticEmotionDataset(
        val_csv_path,
        tokenizer,
        config.max_len,
        config.weak_weight_cap,
        config.max_val_samples,
        weak_source_fraction=1.0,  # val에서는 weak source sampling을 적용하지 않는다
        weak_sample_seed=config.weak_sample_seed,
    )
    if train_ds.weak_sample_audit:
        print(f"[weak source sampling fraction={config.weak_source_fraction}] {train_ds.weak_sample_audit}")
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=0)
    nli_train_loader, nli_val_loader, nli_split_meta = build_nli_loaders(tokenizer, config)

    model = SemanticEmotionRoBERTa(MODEL_NAME).to(device)
    model.encoder.gradient_checkpointing_enable()

    legacy_path = CKPT_DIR / config.init_ckpt
    if not legacy_path.exists():
        raise FileNotFoundError(f"초기화 체크포인트 없음: {legacy_path}")
    legacy_state = torch.load(legacy_path, map_location=device, weights_only=True)
    if "model_state_dict" in legacy_state:
        legacy_state = legacy_state["model_state_dict"]
    legacy_load_summary = model.load_legacy_roberta_state(legacy_state)
    print(
        f"[초기화] {legacy_path.name}에서 {len(legacy_load_summary['copied'])}개 tensor 로드 "
        f"(skip {len(legacy_load_summary['skipped'])})"
    )

    semantic_class_weights = compute_class_weights(
        train_csv_path, "label", NUM_SEMANTIC_EMOTION_CLS
    ).to(device)
    distress_class_weights = compute_class_weights(
        train_csv_path, "distress_level", NUM_DISTRESS_CLS
    ).to(device)
    semantic_criterion = nn.CrossEntropyLoss(
        weight=semantic_class_weights,
        label_smoothing=config.label_smoothing,
        reduction="none",
    )
    distress_criterion = nn.CrossEntropyLoss(
        weight=distress_class_weights,
        label_smoothing=config.distress_label_smoothing,
        reduction="none",
    )
    nli_criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": config.lr},
            {"params": model.semantic_emotion_head.parameters(), "lr": config.lr},
            {"params": model.distress_head.parameters(), "lr": config.lr},
            {"params": model.nli_head.parameters(), "lr": config.nli_lr},
        ],
        weight_decay=config.weight_decay,
    )
    total_steps = max(1, len(train_loader) * config.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config.warmup_ratio),
        num_training_steps=total_steps,
    )
    scaler = GradScaler(device="cuda" if use_amp else "cpu", enabled=use_amp)

    nli_iter = iter(nli_train_loader)
    best_score = -1.0
    best_gold_score = -1.0
    best_path = CKPT_DIR / f"{config.run_name}_best.pt"
    gold_best_path = CKPT_DIR / f"{config.run_name}_gold_best.pt"
    final_path = CKPT_DIR / f"{config.run_name}_final.pt"
    history_path = CKPT_DIR / f"{config.run_name}_history.json"
    split_path = CKPT_DIR / f"{config.run_name}_nli_split.json"
    no_improve = 0
    best_gold_epoch = 0
    history: list[dict[str, Any]] = []

    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(nli_split_meta, f, ensure_ascii=False, indent=2)

    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        total_semantic = 0.0
        total_distress = 0.0
        total_nli = 0.0

        for step, batch in enumerate(train_loader, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            emotion_labels = batch["emotion_label"].to(device)
            distress_labels = batch["distress_label"].to(device)
            sample_weights = batch["sample_weight"].to(device)
            distress_sample_weights = batch["distress_sample_weight"].to(device)

            nli_batch = None
            if config.use_nli_aux:
                try:
                    nli_batch = next(nli_iter)
                except StopIteration:
                    nli_iter = iter(nli_train_loader)
                    nli_batch = next(nli_iter)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=use_amp):
                result = model(input_ids, attention_mask, include_nli=False)
                semantic_loss = weighted_ce_loss(
                    result["semantic_emotion_logits"],
                    emotion_labels,
                    sample_weights,
                    semantic_criterion,
                )
                distress_loss = weighted_ce_loss(
                    result["distress_logits"],
                    distress_labels,
                    distress_sample_weights,
                    distress_criterion,
                )
                nli_loss = torch.tensor(0.0, device=device)
                if nli_batch is not None:
                    nli_ids = nli_batch["input_ids"].to(device)
                    nli_mask = nli_batch["attention_mask"].to(device)
                    nli_labels = nli_batch["label"].to(device)
                    nli_logits = model.forward_nli(nli_ids, nli_mask)
                    nli_loss = nli_criterion(nli_logits, nli_labels)
                loss = (
                    config.semantic_loss_weight * semantic_loss
                    + config.distress_loss_weight * distress_loss
                    + config.nli_loss_weight * nli_loss
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += float(loss.detach().cpu())
            total_semantic += float(semantic_loss.detach().cpu())
            total_distress += float(distress_loss.detach().cpu())
            total_nli += float(nli_loss.detach().cpu())

            if step % config.log_every == 0:
                print(
                    f"  epoch {epoch}/{config.epochs} step {step}/{len(train_loader)} "
                    f"loss={total_loss / step:.4f} sem={total_semantic / step:.4f} "
                    f"dist={total_distress / step:.4f} nli={total_nli / step:.4f}"
                )

        semantic_metrics = evaluate_semantic(model, val_loader, device)
        nli_metrics = evaluate_nli(model, nli_val_loader, device) if config.use_nli_aux else {"nli_macro_f1": 0.0}
        train_loss = total_loss / max(1, len(train_loader))
        combined_score = (
            semantic_metrics["semantic_macro_f1"]
            + 0.5 * semantic_metrics["distress_macro_f1"]
            + 0.25 * nli_metrics["nli_macro_f1"]
        )
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "semantic_train_loss": total_semantic / max(1, len(train_loader)),
            "distress_train_loss": total_distress / max(1, len(train_loader)),
            "nli_train_loss": total_nli / max(1, len(train_loader)),
            "semantic_val": semantic_metrics,
            "nli_val": nli_metrics,
            "combined_score": combined_score,
        }
        history.append(epoch_record)
        print(
            f"[epoch {epoch}] loss={train_loss:.4f} "
            f"semantic_f1={semantic_metrics['semantic_macro_f1']:.4f} "
            f"distress_f1={semantic_metrics['distress_macro_f1']:.4f} "
            f"nli_f1={nli_metrics['nli_macro_f1']:.4f} score={combined_score:.4f}"
        )

        # combined best (기존 운영 기준 — 현행 multi-head 학습 진행 보장)
        if combined_score > best_score:
            best_score = combined_score
            no_improve = 0
            save_checkpoint(best_path, model, config, epoch, epoch_record, legacy_load_summary)
            print(f"  [저장] combined best checkpoint: {best_path}")
        else:
            no_improve += 1

        # gold-only best (semantic_gold_macro_f1 기준 — weak label/NLI bias 영향 배제)
        gold_score = float(semantic_metrics.get("semantic_gold_macro_f1", 0.0))
        if gold_score > best_gold_score:
            best_gold_score = gold_score
            best_gold_epoch = epoch
            save_checkpoint(gold_best_path, model, config, epoch, epoch_record, legacy_load_summary)
            print(
                f"  [저장] gold best checkpoint (semantic_gold_macro_f1={gold_score:.4f}): {gold_best_path}"
            )

        # combined patience 기준은 기존과 동일하게 적용
        if no_improve >= config.patience:
            print(f"  Early stopping (combined patience={config.patience})")
            break

        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

    final_metrics = history[-1] if history else {}
    save_checkpoint(final_path, model, config, int(final_metrics.get("epoch", 0)), final_metrics, legacy_load_summary)
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    return {
        "best_path": str(best_path),
        "gold_best_path": str(gold_best_path),
        "final_path": str(final_path),
        "history_path": str(history_path),
        "nli_split_path": str(split_path),
        "best_score": best_score,
        "best_gold_score": best_gold_score,
        "best_gold_epoch": best_gold_epoch,
        "history": history,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    """
    역할: CLI 인자 파서를 구성한다.
    입력: 없음
    출력: ArgumentParser
    """
    parser = argparse.ArgumentParser()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser.add_argument("--run-name", default=f"semantic_emotion_phase3_{timestamp}")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-len", type=int, default=MAX_LEN)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--nli-lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-ratio", type=float, default=0.08)
    parser.add_argument("--semantic-loss-weight", type=float, default=0.65)
    parser.add_argument("--distress-loss-weight", type=float, default=0.35)
    parser.add_argument("--nli-loss-weight", type=float, default=0.15)
    parser.add_argument("--weak-weight-cap", type=float, default=0.60)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--distress-label-smoothing", type=float, default=0.03)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-nli-samples", type=int, default=None)
    parser.add_argument("--init-ckpt", default="roberta_final.pt")
    parser.add_argument("--skip-nli-aux", action="store_true")
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--weak-source-fraction", type=float, default=1.0,
                        help="weak label_source별로 사용할 비율 (0<frac<=1.0). original_emotion은 항상 1.0 유지.")
    parser.add_argument("--weak-sample-seed", type=int, default=SEED)
    parser.add_argument("--nli-train-csv", default="data/nli/nli_pairs.csv",
                        help="NLI 보조 학습용 train CSV. holdout이 같이 주어지면 이 전체가 학습에 쓰인다.")
    parser.add_argument("--nli-holdout-csv", default=None,
                        help="명시적 NLI holdout CSV. 주어지면 train/val을 random split하지 않고 holdout을 val로 쓴다.")
    parser.add_argument("--train-csv", default="data/processed/semantic_emotion_train.csv",
                        help="semantic emotion 학습용 train CSV 경로. v2 사이클에서 boost 통합본으로 교체 가능.")
    parser.add_argument("--val-csv", default="data/processed/semantic_emotion_val.csv",
                        help="semantic emotion 학습용 val CSV 경로. 평가셋 누수 방지를 위해 v2 사이클에서도 기본값 유지 권장.")
    return parser


def main() -> None:
    """
    역할: CLI 인자를 읽어 Phase 3 학습을 실행하고 결과 경로를 출력한다.
    입력: 명령행 인자
    출력: 콘솔 로그와 체크포인트 파일
    """
    args = build_arg_parser().parse_args()
    config = TrainConfig(
        run_name=args.run_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_len=args.max_len,
        lr=args.lr,
        nli_lr=args.nli_lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        semantic_loss_weight=args.semantic_loss_weight,
        distress_loss_weight=args.distress_loss_weight,
        nli_loss_weight=0.0 if args.skip_nli_aux else args.nli_loss_weight,
        weak_weight_cap=args.weak_weight_cap,
        label_smoothing=args.label_smoothing,
        distress_label_smoothing=args.distress_label_smoothing,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        max_nli_samples=args.max_nli_samples,
        init_ckpt=args.init_ckpt,
        use_nli_aux=not args.skip_nli_aux,
        patience=args.patience,
        log_every=args.log_every,
        weak_source_fraction=args.weak_source_fraction,
        weak_sample_seed=args.weak_sample_seed,
        nli_train_csv=args.nli_train_csv,
        nli_holdout_csv=args.nli_holdout_csv,
        train_csv=args.train_csv,
        val_csv=args.val_csv,
    )
    result = train(config)
    print(json.dumps({k: v for k, v in result.items() if k != "history"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
