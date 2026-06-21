"""
train_utterance_type_head.py
역할: 기존 KLUE-RoBERTa 멀티태스크 체크포인트 위에 발화 의도/타입 7클래스 head만 추가 학습한다.
입력: data/processed/utterance_intent_train.csv, utterance_intent_val.csv
출력: models/roberta/checkpoints/roberta_utterance_intent_head.pt
실행: C:/Users/WD/anaconda3/envs/dl_study/python.exe models/roberta/train_utterance_type_head.py
"""

import json
import os
import random
import sys
import argparse

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
    MODEL_NAME,
    NUM_EMOTION_CLS,
    NUM_NLI_CLS,
    NUM_UTTERANCE_TYPE_CLS,
    RoBERTaMultiTask,
)


SEED = 42
MAX_LEN = 128
BATCH_SIZE = 8
LR = 1e-3
EPOCHS = int(os.getenv("UTTERANCE_TYPE_EPOCHS", "12"))
PATIENCE = int(os.getenv("UTTERANCE_TYPE_PATIENCE", "4"))

DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
CKPT_DIR = os.path.join(BASE_DIR, "models", "roberta", "checkpoints")
REPORT_DIR = os.path.join(BASE_DIR, "eval", "report")
# worktree에 utterance_intent CSV가 없으면 메인 리포로 fallback (gitignored 산출물 공유 패턴)
_FALLBACK_DATA_DIR = r"C:\Users\WD\emotion_chatbot\data\processed"
def _resolve_utt(name: str) -> str:
    """utterance intent CSV를 worktree 우선, 없으면 메인 리포로 해석."""
    local = os.path.join(DATA_DIR, name)
    return local if os.path.exists(local) else os.path.join(_FALLBACK_DATA_DIR, name)
TRAIN_CSV = _resolve_utt("utterance_intent_train.csv")
VAL_CSV = _resolve_utt("utterance_intent_val.csv")
BASE_CKPT = os.path.join(CKPT_DIR, "roberta_final.pt")
OUTPUT_CKPT = os.path.join(CKPT_DIR, "roberta_utterance_intent_head.pt")
REPORT_PATH = os.path.join(CKPT_DIR, "utterance_intent_report.json")

ID_TO_LABEL = {
    0: "casual_share",
    1: "positive_share",
    2: "routine_discomfort",
    3: "emotional_distress",
    4: "preference_question",
    5: "practical_question",
    6: "crisis_candidate",
}
LABEL_TO_ID = {value: key for key, value in ID_TO_LABEL.items()}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_ckpt_path(path_or_name: str) -> str:
    """
    역할: 체크포인트 파일명을 CKPT_DIR 기준 절대경로로 해석한다.
    입력: 파일명 또는 절대/상대 경로
    출력: 절대 경로 문자열
    """
    if os.path.isabs(path_or_name):
        return path_or_name
    return os.path.join(CKPT_DIR, path_or_name)


def resolve_data_path(path_or_name: str) -> str:
    """
    역할: 데이터 CSV 파일명을 DATA_DIR 기준 경로로 해석한다.
    입력: 파일명 또는 절대/상대 경로
    출력: 실제 CSV 경로
    """
    if os.path.isabs(path_or_name):
        return path_or_name
    local = os.path.join(DATA_DIR, path_or_name)
    if os.path.exists(local):
        return local
    return path_or_name


def resolve_report_path(path_or_name: str) -> str:
    """
    역할: 리포트 파일명을 eval/report 기준 경로로 해석한다.
    입력: 파일명 또는 절대/상대 경로
    출력: 실제 리포트 저장 경로
    """
    if os.path.isabs(path_or_name):
        return path_or_name
    if os.path.dirname(path_or_name):
        return os.path.abspath(path_or_name)
    return os.path.join(REPORT_DIR, path_or_name)


def build_arg_parser() -> argparse.ArgumentParser:
    """
    역할: 발화 타입 head 학습 CLI 인자를 정의한다.
    입력: 없음
    출력: ArgumentParser
    """
    parser = argparse.ArgumentParser(description="RoBERTa utterance type head 학습")
    parser.add_argument(
        "--base-ckpt",
        default=BASE_CKPT,
        help="head를 학습할 encoder 체크포인트 파일명 또는 경로",
    )
    parser.add_argument(
        "--output-ckpt",
        default=OUTPUT_CKPT,
        help="저장할 utterance head 체크포인트 파일명 또는 경로",
    )
    parser.add_argument(
        "--init-head-ckpt",
        default="",
        help="초기값으로 불러올 기존 utterance head 체크포인트 파일명 또는 경로",
    )
    parser.add_argument(
        "--report-path",
        default=REPORT_PATH,
        help="학습 리포트 JSON 파일명 또는 경로",
    )
    parser.add_argument(
        "--train-csv",
        default=TRAIN_CSV,
        help="학습 CSV 파일명 또는 경로",
    )
    parser.add_argument(
        "--val-csv",
        default=VAL_CSV,
        help="검증 CSV 파일명 또는 경로",
    )
    return parser


def set_seed(seed: int = SEED) -> None:
    """
    역할: 학습 재현성을 위해 난수 시드를 고정한다.
    입력: seed 정수
    출력: 없음
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class UtteranceTypeDataset(Dataset):
    """
    역할: 발화 의도/타입 분류용 CSV를 PyTorch Dataset으로 제공한다.
    입력: CSV 경로, tokenizer, 최대 토큰 길이
    출력: tokenized tensor와 label
    """

    def __init__(self, csv_path: str, tokenizer, max_len: int = MAX_LEN):
        """
        역할: CSV를 읽어 텍스트와 라벨을 메모리에 올린다.
        입력: csv_path, tokenizer, max_len
        출력: 없음
        """
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        self.texts = df["text"].astype(str).tolist()
        self.labels = df["label"].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        """
        역할: 데이터셋 크기를 반환한다.
        입력: 없음
        출력: 샘플 개수
        """
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        """
        역할: 단일 발화를 토큰화하고 라벨과 함께 반환한다.
        입력: 샘플 인덱스
        출력: input_ids, attention_mask, label dict
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
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def compute_class_weights(csv_path: str) -> torch.Tensor:
    """
    역할: 클래스 불균형 보정을 위한 역빈도 가중치를 계산한다.
    입력: 학습 CSV 경로
    출력: CrossEntropyLoss에 넣을 class weight 텐서
    """
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    counts = df["label"].value_counts().sort_index().reindex(
        range(NUM_UTTERANCE_TYPE_CLS),
        fill_value=1,
    )
    weights = 1.0 / counts.values.astype(float)
    weights = weights / weights.sum() * NUM_UTTERANCE_TYPE_CLS
    return torch.tensor(weights, dtype=torch.float32)


def load_model(base_ckpt: str, init_head_ckpt: str | None = None) -> RoBERTaMultiTask:
    """
    역할: 기존 RoBERTa 감정/NLI 체크포인트를 로드하고 발화 타입 head를 초기화한다.
    입력: base_ckpt 체크포인트 경로, 선택 초기 head 체크포인트 경로
    출력: 학습 준비된 RoBERTaMultiTask 모델
    """
    model = RoBERTaMultiTask(
        MODEL_NAME,
        NUM_EMOTION_CLS,
        NUM_NLI_CLS,
        NUM_UTTERANCE_TYPE_CLS,
    ).to(DEVICE)

    # 기존 체크포인트에는 utterance_type_head가 없을 수 있으므로 strict=False로 호환 로드한다.
    state = torch.load(base_ckpt, map_location=DEVICE, weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[기존 체크포인트 로드] {base_ckpt}")
    if missing:
        print(f"[초기화된 신규 파라미터] {missing}")
    if unexpected:
        print(f"[사용하지 않은 파라미터] {unexpected}")

    if init_head_ckpt:
        head_payload = torch.load(init_head_ckpt, map_location=DEVICE, weights_only=True)
        head_state = head_payload.get("utterance_type_head")
        if not head_state:
            raise ValueError(f"utterance_type_head가 없는 초기 head 체크포인트: {init_head_ckpt}")
        model.utterance_type_head.load_state_dict(head_state)
        print(f"[초기 head 로드] {init_head_ckpt}")

    # 인코더와 기존 head는 고정하고, 새 발화 타입 head만 학습한다.
    for param in model.parameters():
        param.requires_grad = False
    for param in model.utterance_type_head.parameters():
        param.requires_grad = True

    return model


@torch.no_grad()
def evaluate(model: RoBERTaMultiTask, loader: DataLoader, criterion) -> tuple[float, float, dict]:
    """
    역할: 발화 의도/타입 검증 loss, Macro F1, 상세 리포트를 계산한다.
    입력: 모델, 검증 DataLoader, 손실 함수
    출력: 평균 loss, Macro F1, classification report dict
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        logits = model.forward_utterance_type(input_ids, attention_mask)
        loss = criterion(logits, labels)
        total_loss += loss.item()

        preds = logits.argmax(dim=-1).cpu().numpy().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy().tolist())

    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    report = classification_report(
        all_labels,
        all_preds,
        labels=list(ID_TO_LABEL.keys()),
        target_names=[ID_TO_LABEL[idx] for idx in ID_TO_LABEL],
        zero_division=0,
        output_dict=True,
    )
    return total_loss / max(len(loader), 1), macro_f1, report


def save_head_checkpoint(
    model: RoBERTaMultiTask,
    best_f1: float,
    epoch: int,
    report: dict,
    output_ckpt: str,
    report_path: str,
    base_ckpt: str,
) -> None:
    """
    역할: 학습된 발화 의도/타입 head와 메타데이터를 저장한다.
    입력: 모델, 최고 F1, epoch, 평가 리포트, 출력 경로, 리포트 경로, base 체크포인트 경로
    출력: 없음
    """
    payload = {
        "utterance_type_head": model.utterance_type_head.state_dict(),
        "label_to_id": LABEL_TO_ID,
        "id_to_label": ID_TO_LABEL,
        "best_macro_f1": best_f1,
        "best_epoch": epoch,
        "model_name": MODEL_NAME,
        "max_len": MAX_LEN,
        "base_ckpt": base_ckpt,
    }
    torch.save(payload, output_ckpt)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"best_macro_f1": best_f1, "best_epoch": epoch, "report": report}, f, ensure_ascii=False, indent=2)


def train(args: argparse.Namespace | None = None) -> None:
    """
    역할: 발화 의도/타입 head 학습 전체 흐름을 실행한다.
    입력: CLI 인자 namespace
    출력: 없음
    """
    args = args or build_arg_parser().parse_args()
    base_ckpt = resolve_ckpt_path(args.base_ckpt)
    init_head_ckpt = resolve_ckpt_path(args.init_head_ckpt) if args.init_head_ckpt else None
    output_ckpt = resolve_ckpt_path(args.output_ckpt)
    report_path = resolve_report_path(args.report_path)
    train_csv = resolve_data_path(args.train_csv)
    val_csv = resolve_data_path(args.val_csv)
    if os.path.exists(output_ckpt):
        raise FileExistsError(f"기존 utterance head를 덮어쓰지 않도록 중단: {output_ckpt}")

    set_seed()
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = UtteranceTypeDataset(train_csv, tokenizer)
    val_ds = UtteranceTypeDataset(val_csv, tokenizer)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = load_model(base_ckpt, init_head_ckpt=init_head_ckpt)
    class_weights = compute_class_weights(train_csv).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.utterance_type_head.parameters(), lr=LR)

    best_f1 = 0.0
    best_epoch = 0
    no_improve = 0

    print(f"[학습 시작] device={DEVICE}, train={len(train_ds)}, val={len(val_ds)}")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for batch in train_dl:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            optimizer.zero_grad()
            logits = model.forward_utterance_type(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_loss, val_f1, report = evaluate(model, val_dl, criterion)
        train_loss = total_loss / max(len(train_dl), 1)
        print(
            f"[epoch {epoch:02d}] train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_macro_f1={val_f1:.4f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            no_improve = 0
            save_head_checkpoint(model, best_f1, best_epoch, report, output_ckpt, report_path, base_ckpt)
            print(f"  [저장] best head checkpoint → {output_ckpt}")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"[early stop] {PATIENCE} epoch 동안 개선 없음")
                break

    print(f"[완료] best_epoch={best_epoch}, best_macro_f1={best_f1:.4f}")


if __name__ == "__main__":
    train()
