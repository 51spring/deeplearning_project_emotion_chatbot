# 프로젝트 보고서 — 일상 대화 기반 감정 모니터링 및 우울 경향 인식 시스템

- 작성일: 2026-05-29
- 기준 런타임: Phase 3.8 v2.2 swap + Vector Scaling + emotion logit bias (2026-05-22 운영값)
- 본 보고서의 두 핵심: **(A) 공포·놀람·혐오·분노를 하나로 묶었을 때의 macro F1 비교 분석**, **(B) 초기 계획(v14) 대비 변경점·이유·추가 내용 상세 기록**

> ⚠️ 의료 진단이 아닌 개인 정서 모니터링 참고용 보조 도구다. 모든 점수/레이블은 진단이 아니라 추세 관찰용이다.

---

## 1. 시스템 개요 (최종 운영 기준)

| 구성 | 역할 | 비고 |
|---|---|---|
| KLUE-RoBERTa-base (Semantic Emotion Judge v2.2) | 7클래스 감정 분류 + NLI 위기 후보 + CBT 유사도 + 발화 타입/CBT class head + distress severity head | `roberta_final.pt` = `semantic_emotion_phase3_8_v2_2_..._gold_best.pt`(epoch 2) |
| Qwen2.5-3B-Instruct + QLoRA(4bit) | DoT 상담 응답 생성 + `[CRISIS]` 보조 태그 | 품질 게이트·fallback·self-check로 안정화 |
| 점수 파이프라인 | roberta_score / cbt_score / NLI → 라우팅·신뢰도 게이트 → depression_score → EWMA → wellness | `pipeline/*` |
| 백엔드/프론트 | FastAPI(SQLite) + React, 단일 origin 서빙 + Docker | 인증·관측성·관리자 기능 포함 |

- 8GB VRAM 제약: 학습은 순차 실행, Qwen은 4bit 양자화, 추론 시 RoBERTa+Qwen 동시 적재(≈3GB) 허용.
- 핵심 운영 상수: `crisis_threshold=0.35`(NLI 후보), `NLI_HARD_INTERRUPT_THRESHOLD=0.80`(하드 인터럽트), `CBT_THRESHOLD=0.60`, `NO_SIGNAL_DEPRESSION_SCORE=0.30`, `roberta_score_p95=0.699979`, `T_emotion=1.511487`, `T_nli=0.6244`.

---

## 2. 감정 분류 성능 — 현재 최종 런타임 (Fresh Blind 1008)

평가셋 `submission_fresh_blind_1008_20260522`은 12스타일 × 7감정 × 12케이스 = 1008행으로, 기존 모든 eval CSV와 정확/축약 중복이 0건임을 검증한 신규 blind 세트다. 보정에 사용하지 않은 문장만으로 측정한 **현재 최종 배포 런타임의 일반화 대표값**이다.

| 지표 | 값 |
|---|---:|
| 전체 행 | 1008 |
| emotion match (accuracy) | 0.5893 |
| **macro F1 (7감정)** | **0.5917** |
| 평균 depression_score | 0.4749 |
| 평균 depression_tendency_score | 0.0252 |

감정별 (gold = `expected_emotion`, pred = `top_emotion`):

| 감정 | Precision | Recall | F1 | n |
|---|---:|---:|---:|---:|
| 행복 | 0.6478 | 0.7153 | 0.6799 | 144 |
| 중립 | 0.3426 | 0.8542 | 0.4891 | 144 |
| 슬픔 | 0.6727 | 0.5139 | 0.5827 | 144 |
| 공포 | 0.7500 | 0.2500 | **0.3750** | 144 |
| 혐오 | 0.7447 | 0.7292 | 0.7368 | 144 |
| 분노 | 0.9718 | 0.4792 | 0.6419 | 144 |
| 놀람 | 0.7000 | 0.5833 | 0.6364 | 144 |

- **공포 F1 0.3750(recall 0.25)** 가 전체 평균을 가장 크게 끌어내리는 단일 병목이다. 공포 144행 중 48행이 중립으로, 36행이 놀람으로 오분류된다.
- **중립 precision 0.3426** 도 낮다. 다른 감정이 애매할 때 중립으로 흘러드는 양이 많기 때문이다(중립 recall은 0.8542로 높음).
- 반대로 **분노 precision 0.9718, 혐오 F1 0.7368** 처럼 일부 부정 감정은 정밀도가 매우 높다.

> 비교: 같은 런타임의 in-distribution val Macro F1은 약 0.44, balanced val 0.5444, scenario_eval_v2 macro F1 0.5344다. fresh blind 0.5917은 정규화·라우팅·후처리까지 포함한 실제 운영 경로 기준이므로, 단독 head 점수보다 발표 설명력이 높다.

---

## 3. ★(요청 A) 공포·놀람·혐오·분노를 하나로 묶은 macro F1 비교 분석

### 3.1 분석 동기

7감정 평가에서 점수를 깎는 오류의 상당수는 **"부정/고각성 감정을 못 잡는 것"이 아니라 "그 안에서 어느 감정인지 헷갈리는 것"** 으로 보였다(공포↔놀람, 분노↔혐오 혼동). 우울 모니터링 보조 도구의 운영 관점에서는 "이 발화가 부정·고각성 정서를 담았는가"가 1차로 중요하고, "넷 중 정확히 무엇인가"는 2차다. 이를 정량적으로 확인하기 위해 **공포·놀람·혐오·분노를 하나의 "고각성·부정군"으로 통합**한 macro F1을 같은 1008행 예측 결과로 직접 재계산했다.

- 재현 스크립트: `eval/analyze_fresh_blind_emotion_grouping.py`
- 결과 JSON: `eval/report/fresh_blind_1008_emotion_grouping_20260529.json`
- 검증: 7감정 그대로 재계산 시 macro F1 0.5917 / accuracy 0.5893으로 §2 기존 값과 정확히 일치 → 계산 신뢰성 확인.

### 3.2 세 관점 비교 결과

| 평가 관점 | 클래스 구성 | macro F1 | accuracy |
|---|---|---:|---:|
| (1) 7감정 그대로 | 행복/중립/슬픔/공포/혐오/분노/놀람 | 0.5917 | 0.5893 |
| **(2) 4감정 통합 (4-class)** | 행복 / 중립 / 슬픔 / **고각성·부정군** | **0.6340** | 0.6696 |
| **(3) 고각성·부정군 여부 (이진)** | 해당계열 / 비해당 | **0.7951** | 0.7956 |

**(2) 4-class에서 통합 클래스의 세부:**

| 클래스 | Precision | Recall | F1 | n |
|---|---:|---:|---:|---:|
| 행복 | 0.6478 | 0.7153 | 0.6799 | 144 |
| 중립 | 0.3426 | 0.8542 | 0.4891 | 144 |
| 슬픔 | 0.6727 | 0.5139 | 0.5827 | 144 |
| **고각성·부정군(공포+놀람+혐오+분노)** | **0.9868** | **0.6510** | **0.7845** | 576 |

**(3) 이진 판단의 세부:**

| 클래스 | Precision | Recall | F1 | n |
|---|---:|---:|---:|---:|
| 해당계열(4감정) | 0.9868 | 0.6510 | 0.7845 | 576 |
| 비해당(행복·중립·슬픔) | 0.6799 | 0.9884 | 0.8057 | 432 |

### 3.3 해석

1. **묶으면 macro F1이 0.5917 → 0.6340으로 +0.0423 상승**한다. 즉 7감정 오류의 의미 있는 부분이 "부정·고각성 감정 자체를 놓친 것"이 아니라 **계열 내부 세분화 혼동(공포↔놀람, 분노↔혐오)** 에서 발생함을 정량적으로 보여준다.
2. **통합 클래스의 precision 0.9868** — 모델이 이 4감정 중 하나로 판단하면 거의 항상 실제로도 부정·고각성 계열이다. 즉 **부정 정서로의 오경보(false alarm)가 매우 적다.** 우울 모니터링에서 "괜한 부정 경보"를 줄이는 것이 중요하므로 운영상 바람직한 특성이다.
3. **통합 클래스의 recall 0.6510** — 약 35%를 놓치며, 그 대부분은 **공포(recall 0.25)** 가 중립·놀람으로 새는 데서 온다. 즉 계열 누락의 책임은 분노/혐오가 아니라 공포에 집중되어 있다.
4. **이진 판단 macro F1 0.7951(accuracy 0.7956)** — "이 발화가 부정·고각성 정서인가 아닌가"는 실용 기준선 0.80에 근접한다. 운영상 가장 중요한 1차 신호(부정 정서 플래그)는 비교적 안정적으로 작동한다는 의미다.

### 3.4 결론 (요청 A)

- 본 시스템은 **"부정·고각성 정서인지 아닌지"는 잘 구분(이진 macro F1 0.7951, 통합 precision 0.9868)** 하지만, **"공포/놀람/혐오/분노 중 정확히 무엇인지" 세분화가 병목(7감정 0.5917, 특히 공포 F1 0.3750)** 이다.
- 따라서 7감정 macro F1 0.5917을 단독으로 제시하면 시스템의 부정 정서 감지 능력을 과소평가할 수 있다. 발표/보고서에서는 7감정 점수와 함께 **4감정 통합 0.6340 / 이진 0.7951** 을 보조 지표로 병기해, 능력의 위치(계열 감지 vs 내부 세분화)를 분명히 한다.
- 후속 개선의 1순위는 전체 재학습이 아니라 **공포 recall 회복**(공포↔놀람↔중립 경계 데이터 보강)으로 좁혀진다.

---

## 4. ★(요청 B) 초기 계획(v14) 대비 변경점 · 이유 · 추가 내용

초기 계획서 `상담챗봇_프로젝트_총정리_v14.md`의 설계를 기준으로, 실제 구현/운영에서 달라진 점과 그 이유, 그리고 새로 추가한 내용을 정리한다. (v14는 보호 문서로 수정하지 않았다.)

### 4.1 모델 ① RoBERTa — "멀티태스크 full FT" → "Semantic Emotion Judge 재학습 swap"

- **초기 계획:** `klue/roberta-base` 멀티태스크 full fine-tuning(감정 7클래스 head + NLI head), Loss = `0.7·emotion + 0.3·NLI`. 표준 감정 데이터로 1회 학습.
- **변경:** 같은 백본 위에 **Semantic Emotion Judge 재학습 사이클**(Phase 3.x → v2 swap → v2.2 stylize boost swap)을 추가하고, 그 가중치를 운영 `roberta_final.pt`로 swap했다. 공유 encoder + semantic emotion head + distress severity head + NLI head 구조.
- **이유:** 초기 모델은 자막형/마커 위주 표현에는 맞지만, 말투·형식이 다양한 일상 문장(마커 없는 distress, 12스타일 register)에서 일반화가 약했다. 스타일×감정 대규모 모니터링에서 행복/혐오/long_context가 중립으로 쏠리는 구조적 약점이 드러났다.
- **추가 내용:** distress severity head 직접 노출(Phase 5b, `roberta_distress_head.pt`), 발화 타입 head, CBT class head(비왜곡 게이트용), 스타일 wrapper 핵심 절 추출(`normalize_emotion_analysis_text()`).
- **효과:** scenario_eval_v2 macro F1 0.378→0.5344, markerless 0.302→0.4937. (v3-1008 historical 0.7791은 2026-05-13 v2.2+rulepatch 시점 근거로만 분리해 사용.)

### 4.2 보정(Calibration) — "단일 Temperature Scaling" → "Vector Scaling + emotion logit bias"

- **초기 계획:** 감정/NLI를 분리한 **스칼라 Temperature Scaling**(T_emotion / T_nli 각각 1개).
- **변경 1 (Vector Scaling, 2026-04-28):** 클래스별 온도 `vector_T_emotion`을 도입. val ECE 0.349→0.084로 크게 개선되고 F1/Acc도 동반 상승해 운영 채택.
- **변경 2 (emotion logit bias, 2026-05-22):** Vector Scaling 뒤 클래스별 가산 bias를 추가. clean val Macro F1 0.4210→0.4439, Accuracy 0.6886→0.7019, 공포 F1 0.4387→0.6042로 상승.
- **이유:** 단일 온도로는 클래스 불균형(중립 과다, 공포·혐오 과소)을 충분히 보정하지 못했다. 수업 제출 기준에서 F1/Accuracy 개선 이득이 더 크다고 판단.
- **트레이드오프(정직 기록):** logit bias 채택으로 ECE는 0.0745→0.0893으로 다소 악화했다(Vector Scaling 단독 기준 ECE 0.0463). 즉 **보정 정밀도 일부를 F1/Accuracy와 맞바꿨다.** P95는 bias 반영 후 0.699979로 전체 재측정했다.

### 4.3 모델 ② Qwen — "QLoRA 생성 의존" → "게이트·fallback 안정화 중심"

- **초기 계획:** Qwen2.5-3B QLoRA 파인튜닝으로 DoT 상담 응답을 생성하고 `[CRISIS]` 태그를 출력.
- **변경:** QLoRA 학습 자체는 수행했으나(2 epoch, val_loss 2.57, `[CRISIS]` 25/30), **3B raw 생성 품질의 1인칭 환각(first-person hallucination)·상담 대본 잔여가 지속**(blind 1008 기준 anchor+BAD 약 37~46%)되어, 운영은 **생성 품질을 게이트로 방어하는 구조**로 무게를 옮겼다.
- **추가 내용:** 품질 게이트(`qwen_quality_guard`, 저품질 패턴 박제), anchor 기반 self-check, 발화 타입 라우팅, fallback 응답 다양화, distress severity 기반 fallback/시스템 프롬프트 톤 힌트(Phase 5c/5d), 일본어 kana 등 다국어 유출 차단.
- **이유:** 최종 사용자 노출 품질은 게이트+fallback이 상당 부분 방어하고 있고, 3B raw 한계를 즉시 7B~9B 교체로 단정하기보다 안정화를 우선했다. 7B~9B 비교는 트리거(anchor+BAD>25%) 충족 상태로 **후속 과제**로 분리.

### 4.4 점수 파이프라인 — "2신호 앙상블" → "라우팅 + 신뢰도 게이트 다층 후처리"

- **초기 계획:** roberta_score + cbt_score 2신호 앙상블(불일치 시 max 보수 판단) → depression_score. CBT threshold는 4주차 실험으로 결정.
- **변경/추가(운영 오탐 대응으로 점진 누적):**
  - **CBT reliability gate v1**: anchor가 임계(0.60)를 넘어도 발화 타입·distress 강도·NLI·CBT head confidence·우울 경향·contrast margin을 함께 보고 full/low/below로 반영 자격을 판정. 리뷰 큐 80건 재추론 시 58해소→**76해소/4잔여**.
  - **Pipeline fusion CBT cap(Phase 5)**: `(cbt-roberta)≥0.30 & roberta<0.40 & severity<0.30` 시 CBT 단독 폭주 cap.
  - **발화 타입 라우팅 cap 다수**: routine_discomfort, physical_exertion, sensory_disgust(감각 혐오), situational_anxiety/surprise, administrative/technical neutral, limited situational distress, single-event sadness, positive_affect, daily_routine_neutral 등.
  - **Intensifier minimal-pair guard**: `너무/진짜/정말/완전/엄청` 같은 강조어 하나로 저위험 발화 점수가 튀지 않도록 원문/약화판 delta cap.
  - **Laughter-only 보정**: `ㅋㅋ/ㅎㅎ` 단독 발화를 positive_share 저신호로 처리(웰니스 70점 유지). 단 `죽고 싶다 ㅋㅋ`는 위기 보존.
- **이유:** 단순 max 앙상블은 단일 신호(특히 CBT anchor) 단독 과상승으로 일상 발화가 위험권으로 급락하는 오탐이 많았다. 대표 문장 guard는 새 표현에 일반화되지 않아, **feature 조합 기반 게이트**로 구조화했다.

### 4.5 우울 경향(Depression Tendency) — 신규 별도 축 신설

- **초기 계획:** 종합 `depression_score` 단일 축.
- **추가:** 종합 distress와 분리된 **`depression_tendency_score`** 를 신설(v1.5 운영 채택, v2 dual-output은 audit 전용). 7카테고리 마커 매칭 + 지속성 multiplier + cap 우선순위 구조. 평가 75건 band_accuracy 1.000, false_low/high 0.000, MAE 0.093~0.096으로 게이트 통과.
- **이유:** "오늘 종합적으로 힘든 정도(distress)"와 "지속적 우울 경향(tendency)"은 다른 신호다. 상황성 분노/공포로 distress가 높아도 우울 경향은 낮을 수 있어, 두 축 분리가 모니터링 목적에 맞다.

### 4.6 웰니스 반영 정책 — 정교화

- **초기 계획:** `wellness_score = 100 - depression_score×100`, 30일 기준 절대값↔퍼센타일 자동 전환.
- **변경/추가:**
  - **무신호 baseline 70점**: `NO_SIGNAL_DEPRESSION_SCORE=0.30`. "정서 신호 없음"을 100점이 아니라 70점/`보통`으로 둔다(과거 0.0 이력도 0.30 floor).
  - **full / low / none 3단계 반영**: distress·routine·crisis는 원점수 full, casual/positive는 좁은 연속 범위(중립 0.22~0.38, 긍정 0.22~0.30, 감각 혐오 0.35~0.45) low, 순수 질문은 none.
  - **실시간 "오늘의 웰니스"는 오늘 발화 버퍼만**으로 계산하고 과거 EWMA와 분리(과거 흐름은 마감·캘린더에서만).
- **이유:** 저신호·중립 발화만 있는 날이 100점으로 승격되거나, 과거 EWMA가 섞여 오늘 행복 발화가 낮게 보이는 UX 버그를 해소.

### 4.7 인증·보안·관측성 — 신규 추가

- **초기 계획:** 사실상 username 기반 로컬 편의 수준.
- **추가:** PBKDF2-SHA256 비밀번호 해시, HMAC-SHA256 Bearer access token, 보호 API 토큰/소유자 검증(401/403), CORS 제한, production 강한 secret 필수화 + 비밀번호 최소 8자 + legacy claim 차단, SQLite FK 활성화, 관리자 계정(`developer`/`root`)과 권한 게이트(`/day/advance`, `/admin/reset-db`), `model_audit_events` 관측 테이블, 원문 미노출 보안 점검(`eval/security_privacy_audit.py`).
- **이유:** 같은 네트워크/외부 터널 시연을 전제로 하면 최소한의 인증·격리·관측이 필요. 단, 공개 서비스 수준(HTTPS, rate limit, 계정 복구, 데이터 삭제/내보내기)은 후속 과제로 명시.

### 4.8 배포 — 단일 FastAPI 서버 + Docker + 터널

- **추가:** FastAPI가 `frontend/build`를 같은 origin으로 정적 서빙(`run_deploy.bat`), `eval/deploy_smoke.py` 발표 직전 점검, Docker 패키징(체크포인트/DB/processed/HF cache는 volume mount), 외부 시연은 cloudflared/ngrok 터널. Flask 전환은 하지 않고 FastAPI 모델 서빙 유지.
- **이유:** 수업 데모/시연을 개인 GPU PC 단일 서버로 단순화하고, Docker 산출물 요구를 충족.

### 4.9 성능 목표 대비 결과 (정직한 기록)

| 지표 | 초기 목표(v14) | 실제(현재 런타임) | 판정 |
|---|---:|---|---|
| Accuracy | ≥ 0.72 | val ≈ 0.70, blind 0.5893 | in-dist 근접 / blind 미달 |
| Macro F1 | ≥ 0.70 | val ≈ 0.44, balanced 0.5444, blind 0.5917 | 미달 |
| Sadness(슬픔) F1 | ≥ 0.75 | blind 0.5827 | 미달 |
| ECE | ≤ 0.05 | Vector 0.0463 / +bias 0.0893 | Vector 단독 충족, bias 후 초과 |
| NLI 위기 | (안전 보존) | 684쌍 posttrain FP=0/FN=0, 474쌍 unbiased F1 0.772 | 안전 축 양호 |

- **목표 미달의 솔직한 원인:** 초기 목표는 in-distribution 표준 감정셋 기준이었고, 실제 운영은 말투·형식이 다양한 마커리스 일반화가 핵심 난이도였다. blind 기준 7감정 0.5917은 목표 0.70에 못 미치지만, §3의 4감정 통합 0.6340 / 이진 0.7951이 보여주듯 **부정 정서 계열 감지 자체는 실용 수준**이며 병목은 세분화(특히 공포)다.
- 안전 축(NLI 위기, 하드 인터럽트, 위기 시나리오 100%)은 보존되었다.

---

## 5. 한계와 후속 과제

1. **공포 recall(0.25) 회복**이 단일 최대 레버. 공포↔놀람↔중립 경계 데이터 보강 또는 semantic judge 추가 학습.
2. **중립 precision(0.34)** 개선 — 애매 발화가 중립으로 흘러드는 양 축소.
3. **Qwen raw 생성 품질** — 7B~9B 비교/교체(현재는 게이트+fallback로 방어).
4. **Depression Tendency v2** — 마커 없는 기능저하/소속감 저하/수면·식욕 변화 표현 회수율 입증 후 표시값 채택.
5. **공개 서비스 보안** — HTTPS, rate limit, 계정 복구/삭제/내보내기, 장기 모니터링.

---

## 6. 산출물 / 재현

- 본 보고서: `eval/report/project_report_20260529.md`
- 묶음 분석 스크립트: `eval/analyze_fresh_blind_emotion_grouping.py`
  - 실행: `C:\Users\WD\anaconda3\envs\dl_study\python.exe eval/analyze_fresh_blind_emotion_grouping.py --json eval/report/fresh_blind_1008_emotion_grouping_20260529.json`
- 묶음 분석 결과: `eval/report/fresh_blind_1008_emotion_grouping_20260529.json`
- fresh blind 원천: `eval/report/submission_fresh_blind_1008_20260522_roberta_quick_current.csv`, `*_summary.json`
- 상세 근거: `MODEL_DECISIONS.md`, `DEVLOG_SUMMARY.md`, `eval/report/final_report_20260521_class_submission.md`
- 초기 계획(보호 문서, 미수정): `상담챗봇_프로젝트_총정리_v14.md`
