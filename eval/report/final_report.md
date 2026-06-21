# 상담 챗봇 최종 성능 보고서

> 딥러닝 수업 프로젝트 / 1인 개발 / 2026 봄
> 한국어 일상 대화 기반 감정 모니터링 + 우울 경향 인식 보조 도구
> 작성일: 2026-04-29 (Vector Scaling, NLI v2 보강, hard negative clean 라벨 정제 반영본)

---

## 2026-05-15 제출/시연 정합화 메모

이 보고서 본문은 2026-04-29 기준 최종 보고서이며, 제출/시연에서는 아래 최신 운영 기준을 우선 적용한다.

| 항목 | 제출/시연 기준 |
|---|---|
| 운영 모델 | `v2.2 + rulepatch` 유지 |
| 보류 후보 | `v2.4`는 blind-fear/head 재학습 후보이나, `v2.2 + rulepatch` 대비 개선 폭이 +0.0109로 작고 행복 회귀가 있어 swap 보류 |
| 런타임 설정 | `roberta_score_p95=0.671781`, `T_emotion=1.511487`, `T_nli=0.6244` |
| `vector_T_emotion` | `[7.107131, 0.915337, 2.08268, 1.310611, 1.775522, 2.239045, 4.943984]` |
| 최신 posttrain | val Macro F1=0.4207, calib Macro F1=0.4012, NLI 684쌍 FP=0/FN=0, smoke 5/5, roundtrip OK |
| 운영 리뷰 큐 재추론 | 80건 중 58건 해소/22건 잔여, 평균 `depression_score` 0.6023→0.4206, 평균 `entailment_prob` 0.3108→0.063 |
| 시연용 보조 문서 | `eval/report/submission_demo_readiness_20260515.md` |

결론은 “의료 진단 시스템”이 아니라 “개인 정서 모니터링과 위기 안전 안내를 보조하는 수업 프로젝트”로 설명한다. Qwen 3B raw 응답 품질과 CBT anchor 잔여 후보는 한계로 명시하되, 최종 노출 응답은 안전 게이트와 fallback으로 통제한다.

---

## 1. 개요

| 항목 | 값 |
|---|---|
| 도메인 | 한국어 일상 대화 기반 감정 분류, 위기 감지, 상담 응답 |
| 모델 1 | KLUE-RoBERTa-base 멀티태스크 (감정 7클래스 + NLI 3클래스) |
| 모델 2 | Qwen2.5-3B-Instruct + QLoRA (4bit, LoRA r=16) |
| 백엔드/프론트엔드 | FastAPI + SQLite / React |
| 하드웨어 | RTX 3060Ti 8GB |
| 목적 | 의료 진단이 아닌 개인 정서 모니터링 참고용 보조 도구 |

핵심 구현은 RoBERTa가 감정, 위기 후보, CBT 관련 점수를 계산하고, Qwen이 상담 응답을 생성하며, 백엔드가 안전 게이트와 일별 요약/캘린더 저장을 담당하는 구조다.

---

## 2. 데이터 구성

### 2.1 감정 분류

원본은 한국어 연속적 대화 데이터셋 55,629행이다. 원본 중립 비율이 78.7%로 매우 높아, 학습셋 중립 다운샘플링과 희소 클래스 보강을 함께 적용했다.

| 항목 | 현재 처리 |
|---|---|
| 노이즈 라벨 | 22개 제거 |
| 중립 | train 중립 다운샘플링 한도 6,500 적용 |
| 공포/혐오 | val/calib support를 각 60건 이상 확보하도록 P3 보강 |
| clean 평가셋 | P2 보수 라벨 정제 + hard negative clean 21건 추가 반영 |
| split | train / val / calib, calib은 보정 전용 |

최신 clean 평가셋 기준:

| split | rows | 비고 |
|---|---:|---|
| val_clean | 7,812 | hard negative clean 12건 추가 반영 |
| calib_clean | 7,660 | hard negative clean 9건 추가 반영 |

### 2.2 NLI 위기 데이터

hypothesis는 `"이 사람은 스스로를 해치려 한다"`로 고정했다.

| 버전 | 총 쌍 | 위기 | 중립 | 비위기 | 목적 |
|---|---:|---:|---:|---:|---|
| baseline | 474 | 204 | 124 | 146 | 초기 unbiased 평가 기준 |
| v1 보강 | 634 | 284 | 124 | 226 | FN 회복 |
| v2 보강 | 684 | 284 | 124 | 276 | FP 축소 + FN 추가 개선 |

v2에서는 일상 과장 표현, 직장/학업 스트레스, 신체 피로, 단순 강조 표현을 비위기 합성으로 보강해 `"죽겠어"` 계열 오탐을 줄였다.

---

## 3. RoBERTa 감정 분류 결과

### 3.1 최종 성능

| 지표 | orig | clean |
|---|---:|---:|
| val Macro F1 | 0.4239 | **0.4267** |
| calib Macro F1 | 0.4100 | **0.4156** |
| balanced val Macro F1 | **0.5444** | 별도 균형 샘플 평가 |

baseline 대비 누적 개선:

| split | baseline | 최신 clean | 개선 |
|---|---:|---:|---:|
| val | 0.2943 | **0.4267** | +0.1324 |
| calib | 0.2919 | **0.4156** | +0.1237 |

클래스별 주요 변화:

| 클래스 | 최신 val clean F1 | 메모 |
|---|---:|---|
| 공포 | 0.5614 | P3 support 보강 후 측정 신뢰성 회복 |
| 혐오 | 0.3269 | 원본 support 한계가 여전히 큼 |
| 중립 | 0.7719 | 분포 보정과 clean 정제 후 안정 |
| 놀람 | 0.4223 | 중립 경계 hard negative 정제 후 소폭 개선 |
| 행복 | 0.2202 | 단일 클래스 보강만으로는 개선 한계 확인 |

### 3.2 라벨 정제

평가셋 라벨 정제는 보수 원칙을 유지했다. 단독 발화만 봐도 80% 이상이 다르게 라벨링할 명백한 mislabel만 수정하고, 맥락 의존 또는 모델 오류 가능성이 있는 사례는 유지했다.

| 단계 | 반영 건수 | 결과 |
|---|---:|---|
| P2 정제 | val 4 / calib 14 | calib Macro F1 +0.0047 수준 개선 |
| hard negative clean | val 12 / calib 9 | val clean 0.4254 -> 0.4267, calib clean 0.4147 -> 0.4156 |
| train 후보 | 5 | 즉시 미반영, 다음 데이터 재생성 후보로 보류 |

산출물:
- `eval/report/emotion_label_review_hard_negative_actionable.csv`
- `eval/apply_hard_negative_clean_review.py`
- `eval/report/p2_clean_compare.txt`

---

## 4. NLI 위기 감지

운영 임계값은 `crisis_threshold=0.35`로 유지한다. 이는 NLI 후보 감지 기준이며, 하드 인터럽트 기준 `NLI_HARD_INTERRUPT_THRESHOLD=0.80`과 역할이 다르다.

### 4.1 474쌍 unbiased 평가 기준

| 단계 | Precision | Recall | F1 | FP | FN |
|---|---:|---:|---:|---:|---:|
| baseline | 0.6895 | 0.8382 | 0.7566 | 77 | 33 |
| P3 직후 | 0.7304 | 0.7304 | 0.7304 | 55 | 55 |
| NLI head-only 회복 | 0.7085 | 0.7745 | 0.7400 | 65 | 46 |
| NLI v1 보강 | 0.6250 | 0.8824 | 0.7317 | 108 | 24 |
| NLI v2 보강 | **0.6740** | **0.9020** | **0.7715** | **89** | **20** |

v2 보강은 FN을 baseline보다 낮추면서 FP를 v1 대비 줄였다. 최적 sweep은 0.45에서 더 높았지만, 운영 정책은 후보 감지 recall을 우선해 0.35를 유지했다.

### 4.2 위기 감지 안전 구조

| 레이어 | 기준 | 역할 |
|---|---|---|
| 직접 표현 하드 인터럽트 | 텍스트 패턴 또는 높은 NLI 확률 | Qwen 호출 없이 즉시 안전 메시지 |
| NLI 후보 감지 | entailment > 0.35 | 위기 후보 플래그와 후속 안전 처리 |
| Qwen `[CRISIS]` 태그 | 생성 응답 후 태그 검사 | 소프트 인터럽트 보완 |

직접 위험 표현은 NLI 확률과 독립적으로 처리되므로, NLI threshold 변화가 직접 표현 안전성 전체를 좌우하지 않도록 설계했다.

---

## 5. 보정과 점수화

### 5.1 Vector Scaling 채택

P3 이후 단일 Temperature Scaling만으로는 감정 ECE가 악화되어, 클래스별 Vector Scaling을 운영 채택했다.

| 방식 | val F1 | val Acc | val ECE |
|---|---:|---:|---:|
| raw | 0.4239 | 0.6450 | 0.3136 |
| single T=1.335 | 0.4239 | 0.6450 | 0.3491 |
| Vector Scaling | **0.4428** | **0.6992** | **0.0842** |
| Isotonic | 0.3218 | 0.8021 | 0.0186 |

Vector Scaling은 ECE를 크게 줄이면서 F1과 Accuracy도 함께 올렸기 때문에 운영 채택했다. Isotonic은 ECE가 가장 낮았지만 F1 손실이 커서 미채택했다.

Vector Scaling T:

| 클래스 | T |
|---|---:|
| 행복 | 4.107 |
| 중립 | 0.611 |
| 슬픔 | 3.227 |
| 공포 | 2.828 |
| 혐오 | 3.338 |
| 분노 | 2.000 |
| 놀람 | 2.088 |

### 5.2 운영 점수화

```
감정 logits -> Vector Scaling -> roberta_score raw -> P95 normalize
NLI logits  -> T_nli Scaling -> crisis 후보
[CLS] 임베딩 -> cbt_score
           -> CBT reliability gate(full/low)
           -> depression_score(종합 distress / wellness risk)
           -> score_policy -> wellness_contribution_score
           -> EWMA(alpha=0.3)
           -> wellness_score = 100 - daily_score * 100
```

| 항목 | 운영값 |
|---|---:|
| crisis_threshold | 0.35 |
| NLI_HARD_INTERRUPT_THRESHOLD | 0.80 |
| T_emotion(single fallback) | 1.511487 |
| vector_T_emotion | [7.107131, 0.915337, 2.08268, 1.310611, 1.775522, 2.239045, 4.943984] |
| T_nli | 0.6244 |
| roberta_score_p95 | **0.671781** |
| CBT_THRESHOLD | 0.60 |

---

## 6. CBT와 발화 타입 보조 헤드

### 6.1 CBT

CBT score는 prototype-only anchor + max-over-anchors 흐름을 유지한다. CBT category 식별은 anchor 단독보다 RoBERTa CBT class head를 우선한다.

| 항목 | 값 |
|---|---:|
| CBT class head macro F1 | 0.349 |
| 비왜곡 F1 | **0.939** |
| CBT smoke 왜곡 보존 | 62% |
| CBT smoke 정답 매칭 | 38% |
| 비왜곡 FP | 29% |

비왜곡 F1은 강하지만, smoke에서 비왜곡 FP가 남아 있어 운영 게이트 적용은 신중하게 유지한다.

### 6.2 Utterance Type Head

| 항목 | 값 |
|---|---:|
| val Macro F1 | **0.8572** |
| guard test | 7/7 통과 |

P3 RoBERTa 인코더 재학습 이후 head drift가 확인되어, utterance type head와 CBT class head를 모두 재학습했다. 향후 stage1 재학습이 발생하면 이 두 head도 함께 재학습해야 한다.

---

## 7. Qwen 상담 응답

| 항목 | 값 |
|---|---|
| 모델 | Qwen2.5-3B-Instruct |
| 로딩 | 4bit quantization |
| LoRA | r=16, alpha=32, q_proj/v_proj |
| 운영 전략 | 자유 생성만 믿지 않고 RoBERTa 라우팅, 품질 게이트, fallback, self-check로 제어 |
| 품질 평가 | 최종 노출 응답 기준 60/60 통과 기록 |

작은 Qwen의 원문 생성 품질보다 최종 사용자에게 노출되는 응답의 안정성을 우선했다. 추가 SFT는 실제 실패 로그가 충분히 쌓인 뒤 판단한다.

---

## 8. 통합 검증

2026-04-29 최종 통합 smoke + UI 리허설을 완료했다.

| 항목 | 결과 |
|---|---|
| posttrain checks | 5/5 시나리오 통과 |
| 실모델 RoBERTa -> Qwen -> RoBERTa roundtrip | 통과 |
| frontend build | 성공, main bundle 122.42 kB |
| endpoint 정합 | `/chat`, `/day/close`, `/calendar/{username}`, `/health` 확인 |
| hard negative clean 적용 후 비교 | val clean 0.4267, calib clean 0.4156 |

FastAPI smoke는 일반 발화, NLI 후보, 소프트 위기, 직접 하드 위기, 하드 위기 시나리오를 모두 통과했다.

---

## 9. 미채택 실험과 판단

| 실험 | 결과 | 판단 |
|---|---|---|
| KLUE-RoBERTa-large 2 epoch probe | val F1 0.3910, balanced val 0.5978 | 자연 분포 val 손해와 비용 대비 효과 부족으로 미채택 |
| 행복 +105 단일 클래스 보강 | val F1 -0.006, 행복 F1 변화 거의 없음 | 단일 클래스 보강 한계 확인 |
| Isotonic calibration | val ECE 0.0186, val F1 0.3218 | F1 손실이 커서 미채택 |
| KoACD anchor 직접 확장 | baseline 대비 회귀 | CBT class head 학습/평가 데이터로 활용하는 쪽 채택 |

---

## 10. 프로젝트 체크리스트

| 항목 | 상태 |
|---|---|
| RoBERTa 감정 분류 | 완료 |
| NLI 위기 후보 감지 + 하드 인터럽트 | 완료 |
| Qwen 상담 응답 생성 | 완료 |
| roberta_score + Vector Scaling + P95 | 완료 |
| CBT score + CBT class head | 완료 |
| FastAPI 연동 | 완료 |
| SQLite 저장/일별 요약/캘린더 | 완료 |
| React UI build | 완료 |
| 최종 smoke/UI 리허설 | 완료 |
| hard negative clean 라벨 정제 | 완료 |

---

## 11. 한계와 후속 과제

1. 감정 분류 Macro F1은 baseline 대비 크게 개선됐지만 아직 0.42대다. 데이터셋이 드라마/자막 기반이라 단일 발화 라벨 모호성이 근본 병목이다.
2. train_candidate 5건은 즉시 반영하지 않았다. 다음 데이터 재생성 또는 corpus-fresh 보강 사이클에서 함께 판단한다.
3. stage1 재학습이 발생하면 감정/NLI뿐 아니라 utterance type head, CBT class head, temperature/vector_T, P95를 모두 다시 맞춰야 한다.
4. 운영 로그 기반 모니터링은 생성됐고, 2026-05-15 재추론에서 리뷰 큐 80건 중 58건이 최신 기준으로 해소됐다. 다만 잔여 22건은 대부분 CBT anchor 진단 후보라 발표 시 한계와 후속 리뷰 대상으로 분리한다.
5. Qwen 추가 SFT는 현재 우선순위가 낮다. 실패 케이스가 충분히 쌓이면 평가셋 기반으로 재학습 여부를 판단한다.

---

## 12. 재현 실행 명령

```bash
# 환경
conda activate dl_study

# 감정/NLI 전처리 및 보강
python data/preprocess/preprocess_emotion.py
python data/preprocess/augment_p3_rare_eval.py
python data/preprocess/augment_nli_pairs.py

# RoBERTa 학습과 NLI head 재학습
python models/roberta/train_roberta.py
python models/roberta/retrain_nli_only.py

# 보정과 P95
python models/roberta/temperature_scaling.py
python eval/run_posttrain_checks.py

# 라벨 정제/비교
python eval/mine_emotion_hard_negatives.py
python eval/prepare_hard_negative_review.py
python eval/review_hard_negative_labels.py
python eval/preview_hard_negative_clean_apply.py
python eval/apply_hard_negative_clean_review.py
python eval/compare_clean_eval.py

# 통합 검증
python eval/run_posttrain_checks.py
```

---

## 13. 주요 산출물

| 파일 | 내용 |
|---|---|
| `models/roberta/checkpoints/roberta_final.pt` | 최종 RoBERTa 멀티태스크 가중치 |
| `models/roberta/checkpoints/temperature_result.json` | single/vector temperature, ECE, P95 기록 |
| `models/roberta/checkpoints/runtime_config.json` | crisis_threshold, roberta_score_p95 등 런타임 설정 |
| `models/roberta/checkpoints/score_norm_result.json` | P95 측정 기록 |
| `models/qwen/checkpoints/qwen_lora_best/` | Qwen LoRA 어댑터 |
| `data/nli/nli_pairs.csv` | NLI v2 보강 684쌍 |
| `data/processed/emotion_val_clean.csv` | clean val 평가셋 |
| `data/processed/emotion_calib_clean.csv` | clean calib 평가셋 |
| `eval/report/p2_clean_compare.txt` | 최신 clean 평가 비교 |
| `eval/report/emotion_label_guide.md` | 감정 라벨링 가이드 |
| `eval/report/emotion_label_review_hard_negative_*.csv` | hard negative 리뷰 산출물 |
| `eval/report/submission_demo_readiness_20260515.md` | 제출/시연 기준 플로우, 핵심 수치, 한계 정리 |

---

## 14. 최종 판단

기능 완성도는 운영/시연 기준으로 충분하다. 핵심 모델, 위기 감지, 점수화, 상담 응답, 백엔드, 프론트엔드, 통합 smoke가 모두 통과했고, 마지막 데이터 정제 사이클도 clean F1을 소폭 개선했다.

남은 작업은 필수 구현보다 운영 품질 관리에 가깝다. 다음 단계는 train 후보 5건과 신규 실사용 로그를 누적한 뒤, 데이터 재생성/재학습 사이클을 열지 판단하는 것이다.
