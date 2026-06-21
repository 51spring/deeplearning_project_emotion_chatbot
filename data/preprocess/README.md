# data/preprocess — 전처리 스크립트 안내

전처리 스크립트를 **현재 운영(재현)에 쓰는 것**과 **과거 실험(미채택)** 으로 나눠 둔다.

- **이 폴더(루트)** = 현재 운영 런타임(Phase 3.8 **v2.2 swap** + Vector Scaling + emotion logit bias)을 재현하는 데 쓰는 스크립트.
- **`_legacy_experiments/`** = 재학습 사이클에서 만들었지만 **운영 swap에 채택되지 않은** semantic v2.1 / v2.3 / v2.4 후보. 보관/추적용. (자세한 내용은 해당 폴더의 README)

> 채택/미채택 근거: `../../MODEL_DECISIONS.md`, 시간순 기록: `../../devlog.md`

## 현재 운영 모델(v2.2) 감정 데이터 재현 핵심 체인
```
build_semantic_emotion_dataset.py            # semantic 감정 베이스셋
  → build_emotion_boost_v2_synthetic.py      # v2 boost (합성)
    build_emotion_boost_v2_aihub.py          # v2 boost (AI Hub 단락)
    build_emotion_boost_v2_candidates.py     # v2 boost 후보
  → merge_emotion_boost_v2.py / filter_emotion_train_for_v2.py
  → combine_semantic_train_v2.py             # v2 combined
  → build_emotion_boost_v2_2_stylized.py     # v2.2 stylize boost (+ v2_2_base_sentences.csv)
  → combine_semantic_train_v2_2.py           # ★ 운영 v2.2 학습셋 (semantic_emotion_v2_2_combined_train.csv)
```

## 그 외 현재 사용 스크립트(축별)
- 감정 분류 기본: `preprocess_emotion.py`, `augment_rare_emotions.py`, `augment_happy_train.py`
- AI Hub 심리상담: `preprocess_aihub.py`
- Qwen 상담 응답: `preprocess_qwen.py`, `build_qwen_daily_style_sft.py`, `build_qwen_response_style_sft.py`, `clean_qwen_jsonl.py`
- NLI 위기: `build_nli_pairs.py`, `augment_nli_pairs.py`, `expand_nli_pairs.py`, `split_nli_holdout.py`
- CBT anchor: `build_cbt_anchors_from_koacd.py`
- 발화 타입/의도: `build_utterance_type_dataset.py`, `build_utterance_intent_v2_4.py` (출력 `utterance_intent_*_v2_4.csv` = 현재 사용)
- 회귀 평가셋: `build_scenario_eval_v3_1008.py` (v3-1008 상시 회귀 게이트; v2.2 boost가 import)
- 사용자 정정: `build_user_emotion_correction_trainset.py`
- 과거 단계(P2/P3) 일회성 산출 스크립트: `augment_p3_rare_eval.py`, `extract_p3_fear_disgust_candidates.py`, `restore_p2_clean_after_p3.py` — 현재 emotion 데이터 lineage를 만든 과거 단계 스크립트라 루트에 함께 둔다(상시 재실행 대상은 아님).
