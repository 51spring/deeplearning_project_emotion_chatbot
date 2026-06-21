"""
train_qwen_lora.py
역할: Qwen2.5-3B-Instruct QLoRA 파인튜닝
      DoT 시스템 프롬프트 내장 JSONL 데이터로 상담 응답 생성 학습
실행: C:/Users/WD/anaconda3/envs/dl_study/python.exe -u models/qwen/train_qwen_lora.py
"""

import os
import json
import math
import torch
from torch.utils.data import Dataset, DataLoader

# Windows CUDA 환경에서 OOM 방지 — expandable_segments 미지원이므로 주석 처리
# os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, PeftModel, get_peft_model, TaskType, prepare_model_for_kbit_training

# ── 설정 ────────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CKPT_DIR   = os.path.join(BASE_DIR, "models", "qwen", "checkpoints")
TRAIN_LOG_DIR = os.path.join(BASE_DIR, "train_log")
DEFAULT_LOG_PATH = os.path.join(TRAIN_LOG_DIR, "qwen_train.log")
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(TRAIN_LOG_DIR, exist_ok=True)

DEFAULT_AUG_JSONL_PATH = os.path.join(BASE_DIR, "data", "processed", "qwen_finetune_crisis_augmented.jsonl")
DEFAULT_WEIGHTED_JSONL_PATH = os.path.join(BASE_DIR, "data", "processed", "qwen_finetune_crisis_weighted.jsonl")
DEFAULT_DAILY_MIX_JSONL_PATH = os.path.join(BASE_DIR, "data", "processed", "qwen_finetune_daily_mix.jsonl")
DEFAULT_BASE_JSONL_PATH = os.path.join(BASE_DIR, "data", "processed", "qwen_finetune.jsonl")
LORA_CKPT_DIR = os.path.join(CKPT_DIR, "qwen_lora_best")
DEFAULT_OUTPUT_DIR = os.path.join(CKPT_DIR, "qwen_lora_best")
JSONL_PATH = os.getenv(
    "QWEN_JSONL_PATH",
    DEFAULT_WEIGHTED_JSONL_PATH
    if os.path.exists(DEFAULT_WEIGHTED_JSONL_PATH)
    else DEFAULT_AUG_JSONL_PATH
    if os.path.exists(DEFAULT_AUG_JSONL_PATH)
    else DEFAULT_BASE_JSONL_PATH,
)
TRAIN_LOG_PATH = os.getenv("QWEN_TRAIN_LOG_PATH", DEFAULT_LOG_PATH)
OUTPUT_DIR = os.getenv("QWEN_LORA_OUTPUT_DIR", DEFAULT_OUTPUT_DIR)
RESUME_LORA_PATH = os.getenv("QWEN_RESUME_LORA_PATH", "")

MODEL_NAME  = "Qwen/Qwen2.5-3B-Instruct"
MAX_LEN     = int(os.getenv("QWEN_MAX_LEN", "256"))
BATCH_SIZE  = 2          # RTX 3060Ti 8GB — OOM 방지: 4→2 (VRAM ~6GB 목표)
EPOCHS      = int(os.getenv("QWEN_EPOCHS", "1"))
LR          = 2e-4
WARMUP_RATIO = 0.05
GRAD_ACCUM  = 8          # 유효 배치 16 유지 (2×8=16)
TRAIN_SUBSET = int(os.getenv("QWEN_TRAIN_SUBSET", "5000"))
LOG_INTERVAL = int(os.getenv("QWEN_LOG_INTERVAL", "50"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log_message(message: str) -> None:
    """
    역할: 학습 로그를 콘솔과 train_log 파일에 동시에 즉시 기록한다.
    입력: 기록할 로그 문자열
    출력: 없음
    """
    print(message, flush=True)
    with open(TRAIN_LOG_PATH, "a", encoding="utf-8") as log_file:
        log_file.write(message + "\n")
        log_file.flush()


def reset_train_log() -> None:
    """
    역할: 새 학습 시작 시 로그 파일 헤더를 초기화해 현재 실행 로그를 분리한다.
    입력: 없음
    출력: 없음
    """
    with open(TRAIN_LOG_PATH, "w", encoding="utf-8") as log_file:
        log_file.write("[Qwen Train Log]\n")
        log_file.write(f"JSONL_PATH={JSONL_PATH}\n")
        log_file.write(f"MODEL_NAME={MODEL_NAME}\n")
        log_file.write(f"OUTPUT_DIR={OUTPUT_DIR}\n")
        log_file.write(f"RESUME_LORA_PATH={RESUME_LORA_PATH or '(none)'}\n")
        log_file.write("=" * 60 + "\n")

# ── LoRA 설정 ─────────────────────────────────────────────────────────────────
LORA_CONFIG = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
)

# ── 4bit 양자화 설정 ──────────────────────────────────────────────────────────
BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)


# ── 데이터셋 ──────────────────────────────────────────────────────────────────
class QwenDataset(Dataset):
    """
    역할: JSONL 파일에서 chat template 적용 후 토크나이징
    입력: JSONL 경로, tokenizer, 최대 길이
    출력: input_ids, attention_mask, labels (패딩 위치 -100)
    """

    def __init__(self, jsonl_path: str, tokenizer, max_len: int = MAX_LEN, subset_size: int = 0):
        """
        역할: JSONL 학습 샘플을 읽고 supervision 가능한 샘플만 필터링한다.
        입력: jsonl 경로, tokenizer, 최대 길이, 사용할 샘플 수 제한
        출력: 초기화된 데이터셋 인스턴스
        """
        self.tokenizer = tokenizer
        self.max_len   = max_len
        self.samples   = []
        self.valid_indices = []
        skipped_samples = 0

        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

        if subset_size > 0:
            self.samples = self.samples[:subset_size]

        for idx, sample in enumerate(self.samples):
            if self._has_supervised_tokens(sample["messages"]):
                self.valid_indices.append(idx)
            else:
                skipped_samples += 1

        log_message(
            f"[QwenDataset] 원본 {len(self.samples)}개 로드 / "
            f"유효 {len(self.valid_indices)}개 / 제외 {skipped_samples}개"
        )

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        sample = self.samples[self.valid_indices[idx]]
        messages = sample["messages"]
        assistant_message = messages[-1]
        if assistant_message["role"] != "assistant":
            raise ValueError("학습 샘플의 마지막 메시지는 assistant여야 합니다.")

        # chat template 적용 — add_generation_prompt=False (학습용)
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids      = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        # 마지막 assistant 응답 시작 이전 토큰은 전부 마스킹해
        # SFT가 사용자/시스템 프롬프트 복원 대신 응답 생성에 집중되도록 한다.
        prompt_text = self.tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_enc = self.tokenizer(
            prompt_text,
            truncation=True,
            max_length=self.max_len,
            padding=False,
            return_tensors="pt",
        )
        prompt_len = prompt_enc["input_ids"].shape[1]
        valid_len = int(attention_mask.sum().item())

        if prompt_len >= valid_len:
            raise ValueError("supervision이 남지 않는 샘플이 데이터셋에 포함되었습니다.")

        # labels: assistant 응답 span만 loss에 포함하고, 패딩은 -100으로 마스킹
        labels = input_ids.clone()
        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
        }

    def _has_supervised_tokens(self, messages: list[dict]) -> bool:
        """
        역할: truncation 후 assistant supervision 토큰이 1개 이상 남는지 검사
        입력: chat template용 messages 리스트
        출력: supervision 존재 여부 (bool)
        """
        if not messages or messages[-1]["role"] != "assistant":
            return False

        full_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        prompt_text = self.tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True,
        )

        full_enc = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_len,
            padding=False,
            return_tensors="pt",
        )
        prompt_enc = self.tokenizer(
            prompt_text,
            truncation=True,
            max_length=self.max_len,
            padding=False,
            return_tensors="pt",
        )

        full_len = full_enc["input_ids"].shape[1]
        prompt_len = prompt_enc["input_ids"].shape[1]
        return prompt_len < full_len


# ── 모델 로드 ─────────────────────────────────────────────────────────────────
def load_model_and_tokenizer():
    """
    역할: Qwen2.5-3B-Instruct 4bit 양자화 로드 + LoRA 어댑터 적용
    출력: (model, tokenizer)
    """
    # 오프라인 환경에서도 학습이 재현되도록, 이어학습 LoRA 또는 저장된 tokenizer가 있으면 우선 사용한다.
    tokenizer_source = (
        RESUME_LORA_PATH
        if RESUME_LORA_PATH and os.path.isdir(RESUME_LORA_PATH)
        else LORA_CKPT_DIR
        if os.path.isdir(LORA_CKPT_DIR)
        else MODEL_NAME
    )

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=BNB_CONFIG,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()

    if RESUME_LORA_PATH:
        if not os.path.isdir(RESUME_LORA_PATH):
            raise FileNotFoundError(f"이어학습 LoRA 경로가 없습니다: {RESUME_LORA_PATH}")
        # 기존 LoRA 어댑터를 학습 가능 상태로 불러와 문장 스타일만 추가로 맞춘다.
        model = PeftModel.from_pretrained(model, RESUME_LORA_PATH, is_trainable=True)
        log_message(f"[LoRA 이어학습] {RESUME_LORA_PATH}")
    else:
        model = get_peft_model(model, LORA_CONFIG)
    model.print_trainable_parameters()

    return model, tokenizer


def log_training_config():
    """
    역할: 현재 학습 설정을 한 번에 출력해 재실행 조건을 명확히 남긴다.
    입력: 없음
    출력: 없음
    """
    log_message("[학습 설정]")
    log_message(f"  - MODEL_NAME={MODEL_NAME}")
    log_message(f"  - JSONL_PATH={JSONL_PATH}")
    log_message(f"  - TRAIN_LOG_PATH={TRAIN_LOG_PATH}")
    log_message(f"  - OUTPUT_DIR={OUTPUT_DIR}")
    log_message(f"  - RESUME_LORA_PATH={RESUME_LORA_PATH or '(none)'}")
    log_message(f"  - MAX_LEN={MAX_LEN}")
    log_message(f"  - BATCH_SIZE={BATCH_SIZE}")
    log_message(f"  - GRAD_ACCUM={GRAD_ACCUM}")
    log_message(f"  - EPOCHS={EPOCHS}")
    log_message(f"  - TRAIN_SUBSET={TRAIN_SUBSET}")
    log_message(f"  - LOG_INTERVAL={LOG_INTERVAL}")
    log_message(f"  - DEVICE={DEVICE}")


# ── 학습 ─────────────────────────────────────────────────────────────────────
def train():
    """
    역할: 파일럿 기준 QLoRA 파인튜닝 메인 루프를 실행한다.
    입력: 없음
    출력: 없음
    """
    reset_train_log()
    log_training_config()
    if not os.path.exists(JSONL_PATH):
        raise FileNotFoundError(f"학습 데이터 파일이 없습니다: {JSONL_PATH}")

    model, tokenizer = load_model_and_tokenizer()

    dataset = QwenDataset(
        JSONL_PATH,
        tokenizer,
        max_len=MAX_LEN,
        subset_size=TRAIN_SUBSET,
    )

    # train / val 9:1 분리
    val_size   = max(1, int(len(dataset) * 0.1))
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    steps_per_epoch = math.ceil(len(train_dl) / GRAD_ACCUM)
    total_steps   = steps_per_epoch * EPOCHS
    warmup_steps  = int(total_steps * WARMUP_RATIO)
    scheduler     = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_val_loss = float("inf")
    completed_optimizer_steps = 0

    log_message(
        f"[학습 시작] train_batches={len(train_dl)}  val_batches={len(val_dl)}  "
        f"steps_per_epoch={steps_per_epoch}  total_optimizer_steps={total_steps}"
    )

    for epoch in range(1, EPOCHS + 1):
        # ── train ──────────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        optimizer_steps = 0

        for step, batch in enumerate(train_dl, 1):
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["labels"].to(DEVICE)

            out  = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = out.loss / GRAD_ACCUM
            loss.backward()
            train_loss += out.loss.item()

            if step % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                optimizer_steps += 1
                completed_optimizer_steps += 1

            if step % LOG_INTERVAL == 0 or step == len(train_dl):
                epoch_progress = step / len(train_dl) * 100.0
                overall_progress = completed_optimizer_steps / total_steps * 100.0 if total_steps else 0.0
                log_message(
                    f"[epoch {epoch}/{EPOCHS}] "
                    f"step {step}/{len(train_dl)} "
                    f"epoch_progress={epoch_progress:.1f}% "
                    f"overall_progress={overall_progress:.1f}% "
                    f"raw_loss={out.loss.item():.4f}"
                )

        # 마지막 미완료 gradient accumulation도 반영해 학습 스텝 누락을 막는다.
        if len(train_dl) % GRAD_ACCUM != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            optimizer_steps += 1
            completed_optimizer_steps += 1

        avg_train = train_loss / len(train_dl)

        # ── val ────────────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_dl:
                input_ids      = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels         = batch["labels"].to(DEVICE)
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += out.loss.item()

        avg_val = val_loss / len(val_dl)
        epoch_progress = epoch / EPOCHS * 100.0
        overall_progress = completed_optimizer_steps / total_steps * 100.0 if total_steps else 0.0
        log_message(
            f"[epoch {epoch}/{EPOCHS}] "
            f"epoch_progress={epoch_progress:.1f}%  "
            f"overall_progress={overall_progress:.1f}%  "
            f"train_loss={avg_train:.4f}  val_loss={avg_val:.4f}  "
            f"optimizer_steps={optimizer_steps}"
        )

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            model.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)
            log_message(f"  → 체크포인트 저장: {OUTPUT_DIR}")

    log_message(f"[학습 완료] best val_loss={best_val_loss:.4f}")


if __name__ == "__main__":
    train()
