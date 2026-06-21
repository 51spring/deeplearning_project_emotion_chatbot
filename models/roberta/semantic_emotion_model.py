"""
semantic_emotion_model.py
역할: Phase 3 Semantic Emotion Judge용 공유 KLUE-RoBERTa encoder와 multi-head 구조를 정의한다.
입력: 단일 발화 또는 NLI premise-hypothesis 토큰
출력: semantic emotion logits, distress severity logits, NLI crisis logits
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from transformers import AutoModel


NUM_SEMANTIC_EMOTION_CLS = 7
NUM_DISTRESS_CLS = 5
NUM_NLI_CLS = 3


class SemanticEmotionRoBERTa(nn.Module):
    """
    역할: 하나의 KLUE-RoBERTa encoder를 semantic emotion, distress, NLI head가 공유한다.
    입력: Hugging Face 모델명과 클래스 수
    출력: multi-head logits를 반환하는 PyTorch 모듈
    """

    def __init__(
        self,
        model_name: str,
        num_semantic_emotion: int = NUM_SEMANTIC_EMOTION_CLS,
        num_distress: int = NUM_DISTRESS_CLS,
        num_nli: int = NUM_NLI_CLS,
        dropout: float = 0.2,
    ) -> None:
        """
        역할: encoder와 세 개의 task head를 초기화한다.
        입력: 모델명, semantic emotion 클래스 수, distress 클래스 수, NLI 클래스 수, dropout
        출력: 초기화된 모듈
        """
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = int(self.encoder.config.hidden_size)

        self.semantic_emotion_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_semantic_emotion),
        )
        self.distress_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_distress),
        )
        self.nli_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_nli),
        )

    def encode_cls(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        역할: 입력 토큰에서 [CLS] 임베딩을 추출한다.
        입력: input_ids, attention_mask
        출력: [batch, hidden] 형태의 CLS 임베딩
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        include_nli: bool = True,
    ) -> dict[str, torch.Tensor]:
        """
        역할: 단일 토큰 입력에 대해 semantic emotion/distress/NLI logits를 계산한다.
        입력: input_ids, attention_mask, NLI head 계산 여부
        출력: logits dict
        """
        cls_emb = self.encode_cls(input_ids, attention_mask)
        result = {
            "semantic_emotion_logits": self.semantic_emotion_head(cls_emb),
            "distress_logits": self.distress_head(cls_emb),
        }
        if include_nli:
            result["nli_logits"] = self.nli_head(cls_emb)
        return result

    def forward_nli(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        역할: premise-hypothesis pair 입력에 대해 NLI 위기 감지 logits를 계산한다.
        입력: pair token input_ids, attention_mask
        출력: NLI logits
        """
        cls_emb = self.encode_cls(input_ids, attention_mask)
        return self.nli_head(cls_emb)

    def get_cls_embedding(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        역할: 후속 CBT/anchor 정합 점검용 [CLS] 임베딩을 반환한다.
        입력: input_ids, attention_mask
        출력: [CLS] 임베딩 tensor
        """
        with torch.no_grad():
            return self.encode_cls(input_ids, attention_mask)

    def load_legacy_roberta_state(self, state_dict: dict[str, Any]) -> dict[str, list[str]]:
        """
        역할: 기존 RoBERTaMultiTask 체크포인트에서 encoder, emotion head, NLI head를 가져온다.
        입력: 기존 roberta_final.pt state_dict
        출력: 로드/스킵된 키 요약
        """
        own_state = self.state_dict()
        copied: list[str] = []
        skipped: list[str] = []

        # 기존 emotion_head는 새 semantic_emotion_head의 warm start로만 사용한다.
        rename_prefixes = {
            "emotion_head.": "semantic_emotion_head.",
            "nli_head.": "nli_head.",
            "encoder.": "encoder.",
        }

        with torch.no_grad():
            for old_key, tensor in state_dict.items():
                new_key = None
                for old_prefix, new_prefix in rename_prefixes.items():
                    if old_key.startswith(old_prefix):
                        new_key = f"{new_prefix}{old_key[len(old_prefix):]}"
                        break
                if new_key is None or new_key not in own_state:
                    skipped.append(old_key)
                    continue
                if tuple(own_state[new_key].shape) != tuple(tensor.shape):
                    skipped.append(old_key)
                    continue
                own_state[new_key].copy_(tensor)
                copied.append(f"{old_key}->{new_key}")

        return {"copied": copied, "skipped": skipped}
