"""
eval_qwen.py
역할: Qwen 상담 응답 품질 평가
      - 파인튜닝 전/후 응답 비교 (BLEU / ROUGE-L)
      - [CRISIS] 태그 감지율 (위기 발화 대상)
      - 베이스라인 비교: LinearSVC 감정 분류
실행: C:/Users/WD/anaconda3/envs/dl_study/python.exe eval/eval_qwen.py
"""

import os
import sys
import json
import random
import torch
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "data", "processed")
NLI_DIR    = os.path.join(BASE_DIR, "data", "nli")
CRISIS_JSON = os.path.join(NLI_DIR, "crisis_utterances_aihub.json")

SEED = 42
random.seed(SEED)


# ── BLEU / ROUGE-L 헬퍼 ───────────────────────────────────────────────────────
def _bleu_1(ref: str, hyp: str) -> float:
    """단어 단위 BLEU-1 (unigram precision)"""
    ref_tokens = ref.split()
    hyp_tokens = hyp.split()
    if not hyp_tokens:
        return 0.0
    matches = sum(1 for t in hyp_tokens if t in ref_tokens)
    return matches / len(hyp_tokens)


def _rouge_l(ref: str, hyp: str) -> float:
    """LCS 기반 ROUGE-L F1"""
    ref_t = ref.split()
    hyp_t = hyp.split()
    if not ref_t or not hyp_t:
        return 0.0

    m, n = len(ref_t), len(hyp_t)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i-1][j-1] + 1 if ref_t[i-1] == hyp_t[j-1] else max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]

    prec   = lcs / n
    recall = lcs / m
    if prec + recall == 0:
        return 0.0
    return 2 * prec * recall / (prec + recall)


# ── Qwen 응답 품질 평가 ───────────────────────────────────────────────────────
def evaluate_qwen_quality(n_samples: int = 50):
    """
    역할: JSONL에서 샘플 추출 → Qwen 응답 생성 → BLEU/ROUGE-L 측정
    입력: 평가 샘플 수
    출력: 콘솔 리포트
    """
    from models.qwen.inference_qwen import generate_response, unload_qwen

    jsonl_path = os.path.join(DATA_DIR, "qwen_finetune.jsonl")
    samples = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    eval_samples = random.sample(samples, min(n_samples, len(samples)))

    bleu_scores, rouge_scores = [], []

    for s in eval_samples:
        messages  = s["messages"]
        # 마지막 assistant 응답이 정답 레퍼런스
        ref = messages[-1]["content"]
        # user 발화와 그 이전 히스토리로 생성
        user_text = messages[-2]["content"]
        history   = messages[1:-2]  # system 제외, 마지막 user+assistant 제외

        hyp = generate_response(user_text, history)
        bleu_scores.append(_bleu_1(ref, hyp))
        rouge_scores.append(_rouge_l(ref, hyp))

    unload_qwen()

    print("\n" + "="*55)
    print(f"[eval_qwen] 응답 품질  n={len(eval_samples)}")
    print(f"  BLEU-1  : {np.mean(bleu_scores):.4f} ± {np.std(bleu_scores):.4f}")
    print(f"  ROUGE-L : {np.mean(rouge_scores):.4f} ± {np.std(rouge_scores):.4f}")

    return {
        "bleu_1":  float(np.mean(bleu_scores)),
        "rouge_l": float(np.mean(rouge_scores)),
    }


# ── [CRISIS] 태그 감지율 ──────────────────────────────────────────────────────
def evaluate_crisis_tag(n_samples: int = 30):
    """
    역할: AI Hub 위기 발화에서 Qwen이 [CRISIS] 태그를 출력하는 비율 측정
    입력: 평가 샘플 수
    출력: 콘솔 리포트
    """
    from models.qwen.inference_qwen import generate_response, unload_qwen
    from backend.crisis_handler import check_qwen_crisis_tag

    if not os.path.exists(CRISIS_JSON):
        print(f"[경고] {CRISIS_JSON} 없음 — eval_crisis_tag 건너뜀")
        return {}

    with open(CRISIS_JSON, encoding="utf-8") as f:
        crisis_utts = json.load(f)

    samples  = random.sample(crisis_utts, min(n_samples, len(crisis_utts)))
    detected = 0

    for utt in samples:
        text = utt if isinstance(utt, str) else utt.get("text", "")
        response = generate_response(text)
        if check_qwen_crisis_tag(response):
            detected += 1

    unload_qwen()
    rate = detected / len(samples)

    print(f"\n[CRISIS 태그 감지율] {detected}/{len(samples)} = {rate:.3f}")
    return {"crisis_tag_rate": rate, "n_samples": len(samples)}


# ── LinearSVC 베이스라인 ──────────────────────────────────────────────────────
def evaluate_linearsvc_baseline():
    """
    역할: TF-IDF + LinearSVC 감정 분류 베이스라인 성능 측정
          파인튜닝 모델과 비교용
    출력: 콘솔 리포트
    """
    train_df = pd.read_csv(os.path.join(DATA_DIR, "emotion_train.csv"))
    val_df   = pd.read_csv(os.path.join(DATA_DIR, "emotion_val.csv"))

    vec = TfidfVectorizer(max_features=30000, ngram_range=(1, 2))
    X_train = vec.fit_transform(train_df["text"].astype(str))
    X_val   = vec.transform(val_df["text"].astype(str))

    clf = LinearSVC(max_iter=2000, C=1.0)
    clf.fit(X_train, train_df["label"])
    preds = clf.predict(X_val)

    from pipeline.roberta_score import LABEL2EMOTION
    print("\n" + "="*55)
    print("[eval_qwen] LinearSVC 베이스라인")
    print(classification_report(
        val_df["label"], preds, target_names=LABEL2EMOTION, zero_division=0
    ))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quality",  action="store_true", help="Qwen 응답 품질 평가")
    parser.add_argument("--crisis",   action="store_true", help="CRISIS 태그 감지율")
    parser.add_argument("--baseline", action="store_true", help="LinearSVC 베이스라인")
    parser.add_argument("--all",      action="store_true", help="전체 실행")
    parser.add_argument("--quality-samples", type=int, default=50, help="응답 품질 평가 샘플 수")
    parser.add_argument("--crisis-samples", type=int, default=30, help="위기 태그 평가 샘플 수")
    args = parser.parse_args()

    if args.all or args.baseline:
        evaluate_linearsvc_baseline()
    if args.all or args.quality:
        evaluate_qwen_quality(n_samples=args.quality_samples)
    if args.all or args.crisis:
        evaluate_crisis_tag(n_samples=args.crisis_samples)

    if not any([args.quality, args.crisis, args.baseline, args.all]):
        print("사용법: python eval/eval_qwen.py --baseline | --quality | --crisis | --all")
