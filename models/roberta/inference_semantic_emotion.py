"""
inference_semantic_emotion.py
역할: Phase 3 Semantic Emotion Judge 체크포인트로 단일 발화를 추론한다.
입력: 사용자 발화 텍스트
출력: semantic emotion, distress severity, NLI crisis 확률 JSON
실행:
  C:/Users/WD/anaconda3/envs/dl_study/python.exe models/roberta/inference_semantic_emotion.py --text "내일 발표라 잠이 안 와"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.roberta.semantic_emotion_model import SemanticEmotionRoBERTa  # noqa: E402


MODEL_NAME = "klue/roberta-base"
CKPT_DIR = PROJECT_ROOT / "models" / "roberta" / "checkpoints"
DEFAULT_HYPOTHESIS = "이 사람은 스스로를 해치려 한다"
EMOTION_NAMES = ["행복", "중립", "슬픔", "공포", "혐오", "분노", "놀람"]
DISTRESS_NAMES = {
    0: "calm_or_positive",
    1: "mild_distress",
    2: "moderate_distress",
    3: "high_distress",
    4: "crisis_candidate",
}
NLI_NAMES = {0: "entailment_crisis", 1: "neutral", 2: "contradiction"}


def find_latest_checkpoint() -> Path:
    """
    역할: 가장 최근 생성된 semantic emotion best 체크포인트를 찾는다.
    입력: 없음
    출력: 체크포인트 경로
    """
    candidates = sorted(CKPT_DIR.glob("semantic_emotion_phase3_*_best.pt"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError("semantic_emotion_phase3_*_best.pt 체크포인트가 없습니다.")
    return candidates[-1]


def load_model(checkpoint_path: Path) -> tuple[SemanticEmotionRoBERTa, Any, torch.device, dict[str, Any]]:
    """
    역할: Phase 3 체크포인트와 tokenizer를 로드한다.
    입력: 체크포인트 경로
    출력: 모델, tokenizer, device, payload metadata
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = SemanticEmotionRoBERTa(MODEL_NAME).to(device)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state = payload.get("model_state_dict", payload)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, tokenizer, device, payload


@torch.no_grad()
def predict(
    text: str,
    model: SemanticEmotionRoBERTa,
    tokenizer,
    device: torch.device,
    max_len: int = 128,
) -> dict[str, Any]:
    """
    역할: 단일 텍스트의 semantic emotion/distress/NLI 확률을 계산한다.
    입력: 텍스트, 모델, tokenizer, device, max_len
    출력: 추론 결과 dict
    """
    enc = tokenizer(
        text,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    outputs = model(input_ids, attention_mask, include_nli=False)
    emotion_probs = F.softmax(outputs["semantic_emotion_logits"], dim=-1).squeeze(0).cpu().tolist()
    distress_probs = F.softmax(outputs["distress_logits"], dim=-1).squeeze(0).cpu().tolist()

    nli_enc = tokenizer(
        text,
        DEFAULT_HYPOTHESIS,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    nli_logits = model.forward_nli(
        nli_enc["input_ids"].to(device),
        nli_enc["attention_mask"].to(device),
    )
    nli_probs = F.softmax(nli_logits, dim=-1).squeeze(0).cpu().tolist()

    emotion_idx = int(max(range(len(emotion_probs)), key=lambda idx: emotion_probs[idx]))
    distress_idx = int(max(range(len(distress_probs)), key=lambda idx: distress_probs[idx]))
    nli_idx = int(max(range(len(nli_probs)), key=lambda idx: nli_probs[idx]))
    return {
        "text": text,
        "top_emotion": EMOTION_NAMES[emotion_idx],
        "top_emotion_confidence": float(emotion_probs[emotion_idx]),
        "emotion_probs": {EMOTION_NAMES[idx]: float(prob) for idx, prob in enumerate(emotion_probs)},
        "distress_level": distress_idx,
        "distress_label": DISTRESS_NAMES[distress_idx],
        "distress_confidence": float(distress_probs[distress_idx]),
        "distress_probs": {DISTRESS_NAMES[idx]: float(prob) for idx, prob in enumerate(distress_probs)},
        "nli_top_label": NLI_NAMES[nli_idx],
        "nli_crisis_prob": float(nli_probs[0]),
        "nli_probs": {NLI_NAMES[idx]: float(prob) for idx, prob in enumerate(nli_probs)},
    }


def main() -> None:
    """
    역할: CLI에서 체크포인트와 텍스트를 받아 JSON 추론 결과를 출력한다.
    입력: 명령행 인자
    출력: 콘솔 JSON
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--max-len", type=int, default=128)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint) if args.checkpoint else find_latest_checkpoint()
    if not checkpoint.is_absolute():
        checkpoint = CKPT_DIR / checkpoint
    model, tokenizer, device, payload = load_model(checkpoint)
    result = predict(args.text, model, tokenizer, device, args.max_len)
    result["checkpoint"] = str(checkpoint)
    result["checkpoint_epoch"] = payload.get("epoch")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
