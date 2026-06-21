"""
retrain_nli_only.py
역할: 기존 stage1_best.pt(감정 학습 완료) 위에 NLI 헤드만 새 nli_pairs.csv(474쌍)로 재학습
      stage2_best.pt와 roberta_final.pt 갱신, nli_split.json 재생성
실행: C:/Users/WD/anaconda3/envs/dl_study/python.exe models/roberta/retrain_nli_only.py

안전 동작:
  - 실행 시작 시점에 stage2_best.pt / stage2_history.json / nli_split.json /
    roberta_final.pt 가 존재하면 BACKUP_SUFFIX 로 백업한다.
  - 백업 접미사는 환경변수 NLI_RETRAIN_BACKUP_SUFFIX 로 지정 (기본: pre_nli_retrain).
  - 같은 접미사의 백업 파일이 이미 존재하면 NLI_RETRAIN_OVERWRITE_BACKUP=1 이 아닐 때
    중단해 이전 백업을 덮어쓰지 않는다.
"""

import os
import sys
import shutil
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from transformers import AutoTokenizer

from models.roberta.train_roberta import (
    CKPT_DIR, DEVICE, MODEL_NAME, NUM_EMOTION_CLS, NUM_NLI_CLS,
    RoBERTaMultiTask, stage2_train,
)


# 자동 백업 대상 — stage2 재학습으로 영향받는 산출물 일체
BACKUP_TARGETS = [
    "stage2_best.pt",
    "stage2_history.json",
    "nli_split.json",
    "roberta_final.pt",
]


def _backup_existing(suffix: str, overwrite: bool) -> list[str]:
    """역할: 실행 전 영향받는 체크포인트/메타 파일을 일괄 백업
    입력: 백업 접미사, 기존 동일 접미사 백업 덮어쓰기 허용 여부
    출력: 실제 백업한 경로 리스트
    """
    backed_up: list[str] = []
    for fname in BACKUP_TARGETS:
        src = os.path.join(CKPT_DIR, fname)
        if not os.path.exists(src):
            continue
        stem, ext = os.path.splitext(fname)
        dst_name = f"{stem}.{suffix}{ext}"
        dst = os.path.join(CKPT_DIR, dst_name)
        if os.path.exists(dst) and not overwrite:
            raise RuntimeError(
                f"백업 대상이 이미 존재: {dst}. 다른 NLI_RETRAIN_BACKUP_SUFFIX 를 쓰거나 "
                f"NLI_RETRAIN_OVERWRITE_BACKUP=1 환경변수로 덮어쓰기를 명시해라."
            )
        shutil.copy2(src, dst)
        backed_up.append(dst)
        print(f"  [백업] {src} → {dst}")
    return backed_up


def main():
    """역할: 안전 백업 후 stage1 위에 NLI head 만 재학습"""
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)} | "
              f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    suffix = os.environ.get("NLI_RETRAIN_BACKUP_SUFFIX", "pre_nli_retrain").strip()
    if not suffix:
        suffix = "pre_nli_retrain"
    overwrite = os.environ.get("NLI_RETRAIN_OVERWRITE_BACKUP", "0") in ("1", "true", "True")

    print(f"\n[자동 백업] suffix={suffix}, overwrite={overwrite}")
    backed = _backup_existing(suffix, overwrite)
    if not backed:
        print("  (백업 대상 파일 없음 — 새 학습으로 간주)")

    print(f"\n[모델 초기화] {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = RoBERTaMultiTask(MODEL_NAME, NUM_EMOTION_CLS, NUM_NLI_CLS).to(DEVICE)
    model.encoder.gradient_checkpointing_enable()

    stage1_ckpt = os.path.join(CKPT_DIR, "stage1_best.pt")
    print(f"\n[stage1 체크포인트 로드] {stage1_ckpt}")
    model.load_state_dict(torch.load(stage1_ckpt, weights_only=True))

    print("\n[stage2 NLI 재학습 시작 — 새 nli_pairs.csv 474쌍]")
    model = stage2_train(model, tokenizer)

    # 통합 체크포인트 갱신 (roberta_final.pt = stage2 결과)
    final_path = os.path.join(CKPT_DIR, "roberta_final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"\n[저장] {final_path}")

    print("\n[완료] stage2_best.pt + roberta_final.pt + nli_split.json 갱신")
    if backed:
        print("  이전 상태는 다음 백업으로 복구 가능:")
        for path in backed:
            print(f"    - {path}")


if __name__ == "__main__":
    main()
