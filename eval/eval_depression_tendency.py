"""
eval_depression_tendency.py
역할: depression_tendency_score v1.5.2 규칙 기반 계산기 평가
      `eval/report/depression_tendency_eval.csv`의 40문장에 대해
      RoBERTa 모델 출력(top_emotion, utterance_type, roberta_score, is_crisis)을 얻어
      compute_depression_tendency()를 호출하고 band/in_range/false_high/false_low 메트릭을 출력한다.
실행: C:/Users/WD/anaconda3/envs/dl_study/python.exe eval/eval_depression_tendency.py
출력: eval/report/depression_tendency_eval_result.{csv,json,txt}
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from pipeline.depression_tendency import compute_depression_tendency
from pipeline.roberta_score import (
    ROBERTA_SCORE_P95,
    infer_single,
    load_emotion_vector_T,
    load_roberta_model,
    load_temperature,
)

DEFAULT_EVAL_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "report",
    "depression_tendency_eval.csv",
)
RESULT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "report",
)


def predict_band(score: float) -> str:
    """
    역할: 점수 → band 분류 (spec section 2.1 기반)
    입력: depression_tendency_score (0~1)
    출력: 'high' / 'mid' / 'low'
    """
    if score >= 0.40:
        return "high"
    if score >= 0.20:
        return "mid"
    return "low"


def main(use_model: bool = True, eval_csv: str = DEFAULT_EVAL_CSV, suffix_tag: str = "") -> dict:
    """
    역할: 평가셋 전체 실행 + 메트릭 출력
    입력: use_model (False면 텍스트 전용 — RoBERTa 미로딩),
          eval_csv (평가 CSV 경로),
          suffix_tag (출력 파일명 접미사)
    출력: 결과 dict (콘솔 + 파일 동시 저장)
    """
    df = pd.read_csv(eval_csv)
    rows: list[dict] = []

    if use_model:
        print("[load] RoBERTa 체크포인트 로딩 중...")
        model, tokenizer, device = load_roberta_model()
        T_emotion, T_nli = load_temperature()
        vector_T = load_emotion_vector_T()
        print(f"[load] device={device} T_emotion={T_emotion} T_nli={T_nli} vector_T={'on' if vector_T else 'off'}")
    else:
        model = tokenizer = device = None
        T_emotion = T_nli = 1.0
        vector_T = None

    for _, row in df.iterrows():
        text = str(row["text"])
        kwargs = {}

        if use_model:
            res = infer_single(
                text, model, tokenizer, device,
                T_emotion=T_emotion, T_nli=T_nli,
                p95=ROBERTA_SCORE_P95, vector_T_emotion=vector_T,
            )
            kwargs.update({
                "top_emotion": res["top_emotion"],
                "roberta_score": res["roberta_score"],
                "utterance_type": res["utterance_type"],
                "type_reason": res["utterance_type_reason"],
                "is_crisis": res["is_crisis"],
                "entailment_prob": res["entailment_prob"],
            })

        out = compute_depression_tendency(text, **kwargs)
        s = out["depression_tendency_score"]
        bp = predict_band(s)
        in_range = bool(row["expected_score_min"] <= s <= row["expected_score_max"])

        rows.append({
            "id":           row["id"],
            "text":         text,
            "category":     row["category"],
            "exp_band":     row["expected_band"],
            "exp_min":      row["expected_score_min"],
            "exp_max":      row["expected_score_max"],
            "pred_score":   s,
            "pred_band":    bp,
            "band_match":   bp == row["expected_band"],
            "in_range":     in_range,
            "hit_categories": ",".join(out["hit_categories"]),
            "caps_applied": ",".join(out["caps_applied"]),
            "persistence":  out["persistence_marker_hit"],
            "transient":    out["transient_marker_hit"],
            "raw_before_cap": out["raw_score_before_cap"],
            "top_emotion":  kwargs.get("top_emotion", ""),
            "roberta_score": kwargs.get("roberta_score", float("nan")),
            "utterance_type": kwargs.get("utterance_type", ""),
            "is_crisis":    kwargs.get("is_crisis", False),
            "sadness_baseline_used": out["sadness_baseline_used"],
        })

    res = pd.DataFrame(rows)

    # 메트릭
    n_high   = int((res["exp_band"] == "high").sum())
    n_mid    = int((res["exp_band"] == "mid").sum())
    n_low    = int((res["exp_band"] == "low").sum())
    n_lowmid = n_low + n_mid

    band_acc       = float(res["band_match"].mean())
    in_range_rate  = float(res["in_range"].mean())
    false_low      = float(((res["exp_band"] == "high") & (res["pred_score"] < 0.40)).sum() / max(1, n_high))
    false_high     = float(((res["exp_band"].isin(["low", "mid"])) & (res["pred_score"] >= 0.60)).sum() / max(1, n_lowmid))
    mae_midpoint   = float(res.apply(lambda r: abs(r["pred_score"] - 0.5 * (r["exp_min"] + r["exp_max"])), axis=1).mean())

    summary = {
        "version": "v1.5.2",
        "use_model": bool(use_model),
        "n_total": int(len(res)),
        "n_high": n_high, "n_mid": n_mid, "n_low": n_low,
        "band_accuracy": band_acc,
        "in_range_rate": in_range_rate,
        "false_low_rate_high_below_0.40": false_low,
        "false_high_rate_lowmid_above_0.60": false_high,
        "mean_abs_error_to_midpoint": mae_midpoint,
        "gates_passed": {
            "band_accuracy_ge_0.85":   band_acc >= 0.85,
            "mean_abs_error_le_0.15":  mae_midpoint <= 0.15,
            "false_high_rate_le_0.05": false_high <= 0.05,
            "false_low_rate_le_0.10":  false_low  <= 0.10,
        },
    }

    # 결과 출력
    print()
    print("=" * 100)
    print("Depression Tendency v1.5.2 평가 결과")
    print("=" * 100)
    for _, r in res.iterrows():
        flag = "OK   " if r["in_range"] and r["band_match"] else ("BAND " if not r["band_match"] else "OOR  ")
        print(
            f"{flag} {r['id']} {r['exp_band']:4s}->{r['pred_band']:4s} "
            f"score={r['pred_score']:.3f} (exp {r['exp_min']:.2f}~{r['exp_max']:.2f}) "
            f"emo={str(r['top_emotion']):4s} ut={str(r['utterance_type']):20s} "
            f"cats={r['hit_categories']:25s} caps={r['caps_applied']:35s}"
        )
    print()
    print("=== Summary ===")
    for k, v in summary.items():
        if k == "gates_passed":
            continue
        print(f"  {k}: {v}")
    print("  gates_passed:")
    for k, v in summary["gates_passed"].items():
        print(f"    - {k}: {v}")

    # 저장
    suffix = ("_with_model" if use_model else "_text_only") + (f"_{suffix_tag}" if suffix_tag else "")
    out_csv  = os.path.join(RESULT_DIR, f"depression_tendency_eval_result{suffix}.csv")
    out_json = os.path.join(RESULT_DIR, f"depression_tendency_eval_result{suffix}.json")
    out_txt  = os.path.join(RESULT_DIR, f"depression_tendency_eval_result{suffix}.txt")

    res.to_csv(out_csv, index=False, encoding="utf-8-sig")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": rows}, f, ensure_ascii=False, indent=2)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False, indent=2))

    print(f"\n[saved] {out_csv}")
    print(f"[saved] {out_json}")
    print(f"[saved] {out_txt}")

    return summary


if __name__ == "__main__":
    use_model_arg = "--no-model" not in sys.argv
    eval_csv_arg = DEFAULT_EVAL_CSV
    suffix_arg = ""
    args = [a for a in sys.argv[1:] if a != "--no-model"]
    if args:
        eval_csv_arg = args[0]
        if len(args) > 1:
            suffix_arg = args[1]
    main(use_model=use_model_arg, eval_csv=eval_csv_arg, suffix_tag=suffix_arg)
