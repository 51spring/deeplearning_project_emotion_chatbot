"""
inference_roberta.py
역할: 학습 완료된 RoBERTa 멀티태스크 체크포인트 단독 추론 진입점
      필요 시 CBT 앵커까지 함께 계산해 CLI/운영 점검에서 바로 재사용한다.
실행: C:/Users/WD/anaconda3/envs/dl_study/python.exe models/roberta/inference_roberta.py --text "요즘 많이 지쳐요"
"""

import argparse
import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from pipeline.cbt_similarity import load_anchors, build_anchor_embeddings, compute_cbt_score
from pipeline.ensemble import ensemble_scores
from pipeline.roberta_score import (
    CRISIS_THRESHOLD,
    ROBERTA_SCORE_P95,
    infer_single,
    load_roberta_model,
    load_temperature,
)


def run_inference(text: str, use_cbt: bool = True) -> dict:
    """
    역할: 단일 발화에 대해 RoBERTa 감정/NLI/CBT 점수를 계산한다.
    입력: 사용자 발화 문자열, CBT 사용 여부
    출력: 추론 결과 dict
    """
    model, tokenizer, device = load_roberta_model()
    T_emotion, T_nli = load_temperature()

    roberta_result = infer_single(
        text=text,
        model=model,
        tokenizer=tokenizer,
        device=device,
        T_emotion=T_emotion,
        T_nli=T_nli,
        p95=ROBERTA_SCORE_P95,
    )

    cbt_result = None
    cbt_score = None
    if use_cbt:
        anchors = load_anchors()
        anchor_embs = build_anchor_embeddings(anchors, model, tokenizer, device)
        cbt_result = compute_cbt_score(text, model, tokenizer, device, anchor_embs)
        cbt_score = cbt_result["cbt_score"]

    ensemble_result = ensemble_scores(roberta_result["roberta_score"], cbt_score)

    return {
        **roberta_result,
        "cbt_score": cbt_score,
        "cbt_detail": cbt_result,
        "depression_score": ensemble_result["depression_score"],
        "ensemble_method": ensemble_result["method"],
        "runtime_config": {
            "crisis_threshold": CRISIS_THRESHOLD,
            "roberta_score_p95": ROBERTA_SCORE_P95,
            "T_emotion": T_emotion,
            "T_nli": T_nli,
        },
    }


def main():
    """
    역할: CLI에서 텍스트를 받아 RoBERTa 추론 결과를 JSON으로 출력한다.
    입력: argparse 인자
    출력: 콘솔 JSON
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True, help="추론할 사용자 발화")
    parser.add_argument(
        "--no-cbt",
        action="store_true",
        help="CBT 앵커 유사도 계산을 생략",
    )
    args = parser.parse_args()

    result = run_inference(text=args.text, use_cbt=not args.no_cbt)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
