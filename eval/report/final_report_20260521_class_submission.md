# 일상 대화 기반 감정 모니터링 및 우울 경향 인식 상담 챗봇 최종 보고서

> 딥러닝 수업 프로젝트 / 1인 개발 / 한국어 전용
> 작성일: 2026-05-21 · 최종 제출 패키지 구성(10·15장)은 제출 시점 기준으로 갱신
> 최종 기준: `v2.2 + rulepatch + CBT reliability gate v1 + emotion logit bias + laughter-only low-signal guard`
> 배포 기준: 개인 GPU PC에서 FastAPI가 React production build와 API를 같은 origin으로 제공

---

## 1. 프로젝트 요약

본 프로젝트는 사용자의 한국어 일상 대화를 입력으로 받아 감정 상태, 우울 경향, 위기 신호를 추적하고 상담형 응답을 제공하는 웹 애플리케이션이다. 목표는 의료 진단이 아니라, 사용자가 자신의 정서 흐름을 참고하고 위험 신호가 감지될 때 안전 안내를 받을 수 있도록 돕는 수업용 딥러닝 시스템 구현이다.

최종 구현은 단일 LLM에 모든 판단을 맡기지 않았다. KLUE-RoBERTa 기반 판별 모델이 감정 분류, 위기 후보, CBT 인지 왜곡 보조 점수, 발화 타입을 계산하고, Qwen2.5-3B-Instruct가 상담 응답 생성을 담당한다. 이후 FastAPI 백엔드가 안전 게이트, 품질 검사, fallback 응답, 일별 요약, 캘린더 저장을 통합한다.

| 항목 | 최종 상태 |
|---|---|
| 프로젝트명 | 일상 대화 기반 감정 모니터링 및 우울 경향 인식 시스템 |
| 목적 | 개인 정서 모니터링 참고용 상담 챗봇 |
| 언어 | 한국어 |
| 핵심 기능 | 로그인, 채팅, 감정/웰니스 점수, 우울 경향, 위기 대응, 하루 마감, 캘린더 |
| 판별 모델 | KLUE-RoBERTa-base 기반 감정/NLI/CBT/발화 타입 파이프라인 |
| 생성 모델 | Qwen2.5-3B-Instruct + QLoRA, 4bit 추론 |
| 백엔드/DB | FastAPI + SQLite |
| 보안/개인정보 보강 | production/prod 위험 기본값 차단, SQLite FK 활성화, Qwen raw audit 기본 미저장 |
| 운영 보정 | emotion logit bias, CBT reliability gate, 저신호 웃음 발화 보정 |
| 프론트엔드 | React |
| 공식 실행환경 | `C:\Users\WD\anaconda3\envs\dl_study\python.exe` |
| 기준 GPU | RTX 3060Ti 8GB VRAM |
| 최종 배포 방식 | 개인 PC 단일 FastAPI 서버, `0.0.0.0:8000` |

---

## 2. 문제 정의와 설계 방향

일상 대화에서는 명시적인 우울 표현뿐 아니라 피로, 무기력, 불안, 분노, 사회적 위축, 수면/식욕 변화처럼 여러 간접 신호가 섞여 나타난다. 반대로 시험 긴장, 운동 후 피로, 단일 사건성 슬픔, 상황성 분노처럼 정서적 부담은 있지만 우울 경향이나 위기 신호로 과해석하면 안 되는 발화도 많다.

따라서 최종 시스템은 다음 세 가지 판단 축을 분리했다.

| 축 | 의미 | 예시 |
|---|---|---|
| 감정 분류 | 현재 발화의 주된 정서 라벨 | 행복, 중립, 슬픔, 공포, 혐오, 분노, 놀람 |
| 종합 distress / wellness risk | 발화의 정서적 부담, 위기 후보, CBT 신호를 포함한 넓은 위험도 | 시험 불안, 상황성 슬픔, 화남, 위기 후보 |
| 우울 경향 | 지속 우울, 무기력, 흥미저하 등 우울 관련 언어 신호 | 계속 우울함, 아무것도 하기 싫음, 잠/식욕 문제 |

초기에는 `depression_score`가 이름과 달리 종합 distress 축으로 동작했다. 최종 버전에서는 호환성을 위해 필드명은 유지하되, 우울 경향 전용 `depression_tendency_score`를 별도로 추가해 해석을 분리했다.

설계 원칙은 다음과 같다.

1. 위기 감지는 단일 threshold나 단일 모델에 의존하지 않는다.
2. Qwen의 자유 생성은 항상 안전 검사와 fallback으로 감싼다.
3. 우울 경향은 종합 distress와 분리해 장기 추적한다.
4. 일상/긍정/신체 피로 발화가 위험권으로 과상승하지 않도록 운영 보정을 둔다.
5. 수업 제출/시연에서는 실제 URL 접속 가능한 단일 서버 배포를 완성 기준으로 둔다.

---

## 3. 시스템 아키텍처

```text
사용자 발화
  ↓
FastAPI /chat
  ↓
RoBERTa 추론
  - 감정 7클래스
  - NLI 위기 후보
  - CBT anchor score
  - CBT class head
  - utterance type
  ↓
안전 게이트
  - 직접 위기 문구 hard interrupt
  - NLI hard interrupt
  - soft crisis 후보
  ↓
Qwen 상담 응답 생성
  - 오늘 rolling summary
  - 최근 대화 context
  - 현재 발화
  ↓
응답 품질/안전 검사
  - [CRISIS] 태그
  - self-check
  - response anchor
  - 다국어 잔여 토큰 검사
  - fallback 응답
  ↓
점수 저장
  - top_emotion
  - depression_score
  - depression_tendency_score
  - wellness_score
  - crisis flag
  - model_audit_events
  ↓
React UI
  - 채팅
  - 오늘 웰니스
  - 하루 마감
  - 캘린더
```

FastAPI는 API 라우트 등록 뒤 `frontend/build`가 존재하면 React production build를 정적으로 제공한다. 따라서 수업 시연자는 `http://<PC-IP>:8000` 또는 터널 URL 하나로 로그인, 채팅, 캘린더까지 확인할 수 있다.

---

## 4. 데이터 구성

### 4.1 감정 분류 데이터

감정 분류는 한국어 연속적 대화 데이터셋을 기반으로 구성했다. 원본은 중립 비율이 매우 높고 희소 감정 클래스가 부족해, 중립 다운샘플링과 공포/혐오 보강을 함께 수행했다.

| 항목 | 내용 |
|---|---|
| 원본 행 수 | 55,629행 |
| 감정 클래스 | 행복, 중립, 슬픔, 공포, 혐오, 분노, 놀람 |
| 원본 중립 비율 | 약 78.7% |
| 노이즈 라벨 | 22개 제거 |
| 중립 처리 | 학습셋 중립 다운샘플링 |
| 희소 클래스 | 공포/혐오 val/calib support 60건 이상 확보 |
| 평가셋 | `emotion_val_clean.csv`, `emotion_calib_clean.csv` |

라벨 정제는 보수적으로 수행했다. 단독 발화만 봐도 명백히 다른 감정으로 보이는 경우만 수정하고, 맥락 의존적이거나 모델이 틀린 것으로 볼 수 있는 사례는 평가셋에 그대로 남겼다.

### 4.2 NLI 위기 데이터

위기 감지는 NLI 구조로 만들었다. hypothesis는 `"이 사람은 스스로를 해치려 한다"`로 고정했다.

| 버전 | 총 쌍 | 위기 | 중립 | 비위기 | 목적 |
|---|---:|---:|---:|---:|---|
| baseline | 474 | 204 | 124 | 146 | 초기 기준 |
| v1 | 634 | 284 | 124 | 226 | 위기 FN 회복 |
| v2 | 684 | 284 | 124 | 276 | 일상 과장 표현 FP 축소 |

비위기 데이터에는 `"죽을 만큼 배고파"`, `"살기 싫을 정도로 덥다"`처럼 한국어에서 자주 쓰이는 과장 표현을 포함했다. 이는 위기 recall을 유지하면서 일상 표현의 오탐을 줄이기 위한 설계다.

### 4.3 상담 응답 데이터

Qwen 응답 생성을 위해 AI Hub 심리상담 데이터와 웰니스 대화 데이터를 사용했다.

| 데이터 | 활용 |
|---|---|
| AI Hub 심리상담 데이터 | Qwen 파인튜닝 메인, 상담 문체 학습 |
| 웰니스01 | 상담 응답 보조 데이터 |
| 웰니스02 | CBT anchor, NLI 위기 보완 |

`data/raw/` 원본은 수정하지 않았고, 소스코드 폴더에는 두지 않고 별도 데이터 폴더 `04_사용데이터/`로 제공한다.

---

## 5. 모델 구성과 학습

### 5.1 KLUE-RoBERTa-base

RoBERTa는 시스템의 판별 중심 모델이다.

| 역할 | 설명 |
|---|---|
| 감정 분류 | 7클래스 감정 예측 |
| NLI 위기 판별 | 위기 후보 및 hard interrupt 보조 |
| CBT score | [CLS] embedding과 anchor 간 코사인 유사도 |
| CBT class head | 왜곡 유형/비왜곡 판별 |
| utterance type head | 일상, 긍정, distress, 질문 등 발화 타입 판단 |

감정 분류는 class imbalance, 희소 클래스, 라벨 모호성 때문에 단순 학습만으로는 안정성이 부족했다. 따라서 라벨 정제, rare class 보강, vector scaling, semantic judge 계열 보정, 운영 회귀 guard를 누적했다.

### 5.2 Qwen2.5-3B-Instruct + QLoRA

Qwen은 상담 응답 생성에 사용했다. RTX 3060Ti 8GB 제약 때문에 4bit quantization과 QLoRA를 사용했다.

| 항목 | 설정 |
|---|---|
| 모델 | Qwen2.5-3B-Instruct |
| 로딩 | 4bit quantization |
| 학습 | QLoRA |
| LoRA rank | r=16 |
| LoRA alpha | 32 |
| target modules | `q_proj`, `v_proj` |
| 운영 역할 | 상담 응답 생성, `[CRISIS]` 태그 보조 |

Qwen 3B raw 응답은 항상 안정적이지 않기 때문에, 최종 사용자에게 노출되는 응답은 self-check, anchor 검사, fallback 응답으로 방어했다. 브라우저 리허설 중 일본어 kana 잔여 토큰이 발견되어 히라가나/가타카나/반각 카타카나까지 다국어 유출 감지에 포함했다.

---

## 6. 점수화와 안전 게이트

### 6.1 운영 점수 흐름

```text
감정 logits -> vector_T_emotion -> emotion_logit_bias -> roberta_score -> P95 정규화
NLI logits  -> T_nli -> 위기 후보
CBT anchor  -> CBT score
             ↓
CBT reliability gate
             ↓
depression_score = 종합 distress / wellness risk
             ↓
score_policy -> wellness_contribution_score
             ↓
EWMA(alpha=0.3)
             ↓
wellness_score = 100 - daily_score * 100
```

`/chat` 화면의 실시간 "오늘의 웰니스"는 오늘 발화만 기준으로 계산한다. 과거 날짜의 EWMA는 `/day/close` 저장값과 캘린더 추세에서만 반영한다. 이 분리를 통해 과거 우울 흐름 때문에 오늘의 긍정 발화가 즉시 낮게 보이는 UX 문제를 해결했다.

추가로 `ㅋ`, `ㅋㅋ`, `ㅎㅎ`처럼 웃음 토큰만 있는 단독 발화는 `positive_share` 저신호로 처리한다. raw RoBERTa/CBT 점수가 튀어도 웰니스가 기본 70점 아래로 내려가지 않게 하고, `죽고 싶다 ㅋㅋ`처럼 직접 위기·부정 문장에 웃음이 붙은 경우는 기존 위기 안전 흐름을 그대로 유지한다.

### 6.2 최신 운영 임계값

| 항목 | 값 |
|---|---:|
| `crisis_threshold` | 0.35 |
| `NLI_HARD_INTERRUPT_THRESHOLD` | 0.80 |
| `T_emotion` scalar fallback | 1.511487 |
| `T_nli` | 0.6244 |
| `roberta_score_p95` | 0.699979 |
| `CBT_THRESHOLD` | 0.60 |
| `vector_T_emotion` | [7.107131, 0.915337, 2.08268, 1.310611, 1.775522, 2.239045, 4.943984] |
| `emotion_logit_bias` | [0.671429, 1.121429, 1.101429, -2.478572, -1.178572, 0.871429, -0.108572] |

### 6.3 위기 대응

위기 대응은 3중 구조다.

| 레이어 | 기준 | 동작 |
|---|---|---|
| 직접 표현 hard interrupt | 자해/자살 직접 표현 | Qwen 호출 없이 안전 메시지 |
| NLI hard interrupt | NLI entailment 고확률 | Qwen 호출 없이 안전 메시지 |
| Qwen `[CRISIS]` soft interrupt | 생성 응답의 위기 태그 | 생성 원문 대신 안전 메시지 |

이 구조는 Qwen이 위기 상황에서 부적절한 자유 응답을 노출하는 위험을 줄인다.

### 6.4 CBT reliability gate v1

CBT anchor는 유사도 기반이라 특정 문장이 anchor와 가깝게 잡히면 실제 인지 왜곡이 아니어도 점수가 높게 나올 수 있다. 이를 줄이기 위해 최종 버전에서는 CBT anchor threshold 통과와 실제 점수 반영 자격을 분리했다.

CBT reliability gate v1은 다음 신호를 함께 본다.

| 신호 | 활용 |
|---|---|
| 발화 타입 | 일상/긍정/실용 질문인지, 정서 distress인지 구분 |
| RoBERTa distress 강도 | 실제 정서 부담이 충분한지 확인 |
| NLI | 위기/안전 신호 확인 |
| CBT class head | 비왜곡 또는 왜곡 confidence 확인 |
| 우울 경향 힌트 | 지속 우울 관련 신호 확인 |
| anchor contrast margin | 단순 근접인지 구분 |

운영 리뷰 큐 80건 재추론 결과, 기존 `58해소/22잔여`에서 `76해소/4잔여`로 개선됐다. 이 수치는 최신 제출/시연 기준이며, 과거 문서의 58/22와 혼동하지 않아야 한다.

---

## 7. 주요 구현 기능

### 7.1 사용자 기능

| 기능 | 구현 상태 |
|---|---|
| 회원가입/로그인 | PBKDF2-SHA256 비밀번호 해시, Bearer token |
| 채팅 | 사용자 발화, 상담 응답, 감정/웰니스 표시 |
| 위기 배너 | hard/soft 위기 시 안전 메시지 |
| 하루 마감 | 오늘 대화 요약, 점수 저장 |
| 캘린더 | 날짜별 웰니스, 위기 표시, 추세 |
| 관리자 기능 | 다음날 전환, 현재 계정 DB 초기화 |

인증은 local/demo에서는 수업 시연 편의성을 유지하지만, `EMOTION_CHATBOT_ENV=production|prod`에서는 `EMOTION_CHATBOT_AUTH_SECRET`가 강한 값으로 설정되지 않으면 서버 시작을 차단한다. 신규 비밀번호 최소 길이는 local/demo 4자, production/prod 8자이며, 기존 username-only legacy 계정의 비밀번호 claim은 기본 차단하고 `EMOTION_CHATBOT_ALLOW_LEGACY_ACCOUNT_CLAIM=1`일 때만 허용한다.

### 7.2 백엔드 API

| API | 역할 |
|---|---|
| `/health` | 서버 상태 확인 |
| `/auth/register` | 회원가입 |
| `/auth/login` | 로그인 및 토큰 발급 |
| `/chat` | 모델 추론, 응답 생성, 점수 저장 |
| `/day/close` | 하루 마감 및 요약 저장 |
| `/day/current/{username}` | 현재 날짜/상태 조회 |
| `/calendar/{username}` | 캘린더 데이터 조회 |
| `/admin/reset-db` | 관리자 계정 단위 런타임 데이터 초기화 |

### 7.3 운영 관측성

`model_audit_events`에 모델 판단 근거를 저장한다. 저장 항목은 NLI 후보, hard interrupt 여부, Qwen 호출 여부, Qwen `[CRISIS]` 태그, self-check 결과, CBT head confidence, utterance type, score policy 등이다. 이 로그를 기반으로 운영 리뷰 큐와 false-positive 개선 작업을 진행했다.

개인정보 노출을 줄이기 위해 production/prod 모드에서는 audit payload에 Qwen raw response를 기본 저장하지 않는다. 품질 리뷰 목적으로 원문성 raw 응답 보관이 꼭 필요할 때만 `EMOTION_CHATBOT_STORE_QWEN_RAW_RESPONSE=1`을 명시한다. 또한 SQLite 연결마다 `PRAGMA foreign_keys=ON`을 적용해 향후 잘못된 참조 유입을 막고, `eval/security_privacy_audit.py`로 원문 발화 없이 row count, password hash 상태, orphan 여부, env 설정 여부, 민감 파일 포함 여부를 점검한다.

---

## 8. 평가 결과

### 8.1 평가 해석 요약

본 프로젝트의 평가는 두 층으로 나누어 해석해야 한다. 첫째는 RoBERTa 감정 분류 head만 떼어 놓고 보는 단독 모델 성능이고, 둘째는 실제 사용자가 접속해 로그인, 채팅, 안전 게이트, 점수, 하루 마감, 캘린더까지 확인하는 시스템 검증이다.

수업 제출/시연에서는 두 번째가 더 중요하다. 실제 완성도는 낮은 raw Macro F1 하나가 아니라, end-to-end 기능 통과, 위기 안전성, 운영 보정 후 오탐 감소, 배포 smoke 통과를 함께 봐야 한다.

| 평가 축 | 제출/시연 기준 결과 | 해석 |
|---|---:|---|
| 배포 smoke | `/health`~캘린더 전체 통과 | URL 접속형 데모 가능 |
| 통합 smoke | 5/5 통과 | 모델-백엔드-DB 흐름 정상 |
| 실모델 roundtrip | 통과 | RoBERTa/Qwen 연동 정상 |
| NLI 684쌍 회귀 | FP=0 / FN=0 | 최신 운영 NLI 회귀셋 안전성 유지 |
| CBT reliability | 80건 중 76건 해소 / 4건 잔여 | anchor 오탐 대폭 축소 |
| 저신호 웃음 발화 guard | `ㅋ/ㅋㅋ/ㅋㅋㅋㅋ/ㅋㅋ.` wellness 71.75 내외 | 웃음 반응이 위험권/주의권으로 하락하지 않음 |
| 현재 fresh blind 1008 | macro F1 0.5917, match 0.5893 | 최종 배포 런타임 기준 새 문장 일반화 점검 |
| v3-1008 historical | macro F1 0.7791, markerless 0.7511 | 2026-05-13 `v2.2 + rulepatch` semantic 보강 근거 |
| Wellness fresh 504 | emotion match 0.6766 | 실제 `/chat` 경로 일반화 점검 |
| Qwen 최종 노출 guard | 통과 | raw 생성 한계를 fallback으로 방어 |

따라서 보고서의 핵심 주장은 "감정 분류 단독 성능이 완벽하다"가 아니라, "불완전한 모델을 안전 게이트, calibration, score policy, 운영 audit, 배포 smoke로 묶어 수업 시연 가능한 시스템까지 완성했다"이다.

### 8.2 단독 감정 분류 성능

| 지표 | baseline | 최종 clean 기준 |
|---|---:|---:|
| val Macro F1 | 0.2943 | 0.4267 |
| calib Macro F1 | 0.2919 | 0.4156 |
| balanced val Macro F1 | 0.4238 | 0.5444 |

이 수치는 낮아 보일 수 있다. 다만 이 표는 실제 서비스 전체 점수가 아니라, 중립 편중이 강한 자막형 대화 데이터에서 7감정 단독 분류 head를 평가한 값이다. 한국어 일상 발화는 단일 문장만 보면 맥락 의존성이 높고, 특히 행복/혐오/공포 같은 희소 클래스 support가 작아 Macro F1이 낮게 나온다.

이 데이터에서는 Accuracy만 보면 오히려 착시가 생긴다. `emotion_val_clean.csv`는 7,812건 중 중립이 6,162건이라, 전부 중립으로 찍는 majority baseline도 Accuracy 0.7888을 낸다. 하지만 그 경우 Macro F1은 0.1260뿐이다. 따라서 이 프로젝트의 감정 분류 평가는 Accuracy보다 Macro F1, 희소 클래스 F1, balanced/scenario 평가를 중심으로 해석하는 것이 맞다.

| 비교 기준 | Accuracy | Macro F1 | 해석 |
|---|---:|---:|---|
| majority baseline(전부 중립) | 0.7888 | 0.1260 | 중립 편중 때문에 정확도만 높음 |
| 운영 posttrain 감정 평가 | 0.7003 | 0.4407 | 희소 감정까지 맞추는 균형 성능 개선 |
| balanced val | - | 0.5444 | 클래스 균형 관점의 보완 지표 |

그래서 최종 시스템은 이 점수를 그대로 UI에 노출하지 않는다. 운영 경로에서는 vector scaling, semantic routing, 발화 타입, CBT reliability, NLI, score policy를 함께 사용해 위험권 오탐과 위기 누락을 줄인다.

최신 posttrain 회귀 검증 기준은 다음과 같다. 이 값은 모델이 새 보정 뒤에도 기본 회귀 시나리오를 유지하는지 보는 지표다.

| 항목 | 결과 |
|---|---:|
| val Macro F1 | 0.4407 |
| calib Macro F1 | 0.4526 |
| NLI 684쌍 | FP=0 / FN=0 |
| P95 | 0.699979 |
| smoke | 5/5 통과 |
| 실모델 roundtrip | 통과 |

추가로 체크포인트를 덮어쓰지 않는 post-hoc 개선 실험을 진행했다. `emotion_calib_clean.csv` 내부 train/dev split에서 클래스별 additive logit bias를 선택하고, held-out `emotion_val_clean.csv`에서만 최종 검증했다.

| 항목 | baseline | calib-selected bias | delta |
|---|---:|---:|---:|
| calib Macro F1 | 0.4080 | 0.4581 | +0.0501 |
| val Macro F1 | 0.4210 | 0.4439 | +0.0228 |
| val Accuracy(참고) | 0.6886 | 0.7019 | +0.0133 |

가장 큰 개선은 공포 클래스에서 나타났다. val 공포 F1은 0.4387에서 0.6042로 상승했다. 행복, 슬픔, 놀람은 소폭 하락했고 ECE는 0.0745에서 0.0893으로 악화했지만, 수업 제출/시연 기준에서는 Macro F1과 공포 F1 개선, NLI 안전성 유지, smoke 통과의 이득이 더 크다고 판단했다. Accuracy는 중립 편중 때문에 핵심 채택 지표로 쓰지 않고 참고값으로만 둔다. 따라서 `emotion_logit_bias`를 운영 기본값으로 채택하고, P95를 0.699979로 재측정해 runtime 설정에 반영했다. 같은 logits 위에 logistic calibration layer도 실험했지만 val Macro F1이 0.3991로 하락해 미채택했다.

채택 후 환경변수 없이 `eval/run_posttrain_checks.py`를 재실행했으며, val Macro F1 0.4407, calib Macro F1 0.4526, NLI 684쌍 FP=0/FN=0, P95 0.699979, mock smoke 5/5, 실모델 roundtrip OK를 확인했다. 추가로 `roberta_emotion_guard.py`, `intensifier_minimal_pair_guard.py`, `cbt_reliability_guard.py`, `qwen_quality_guard.py`, 실제 deploy smoke를 통과했다.

### 8.3 Semantic / blind 평가

운영 모델은 `v2.2 + rulepatch` 체크포인트를 유지하되, 2026-05-22에 `emotion_logit_bias`와 P95 재측정을 최종 런타임 기본값으로 채택했다. 따라서 2026-05-13에 측정한 `v3-1008 blind` 0.7791은 semantic 보강 단계의 historical/ablation 근거로만 사용하고, 현재 배포 런타임 대표값으로는 새 fresh 1008과 현재 재평가 결과를 따로 보고한다. `v2.4`는 후보로 보존했지만 행복 회귀와 작은 개선 폭 때문에 제출 전 swap은 보류했다.

| 평가 | 주요 결과 |
|---|---|
| scenario_eval_v2 | macro F1 0.5344, markerless 0.4937 |
| v3-1008 historical | macro F1 0.7791, markerless 0.7511, NLI binary F1 1.0000 (`2026-05-13 v2.2+rulepatch`) |
| v3-1008 current rerun | macro F1 0.5623, match 0.5565 (`2026-05-22 최종 runtime`) |
| submission fresh blind 1008 | macro F1 0.5917, match 0.5893, exact/compact overlap 0건 |
| v2.2+rulepatch quick | blind_1008 운영 quick 전체 0.6587, 공포 1.0000 |
| v2.4 후보 | quick 0.6696이나 행복 회귀로 운영 swap 보류 |

`submission_fresh_blind_1008_20260522`는 공포 클래스가 전체 평균을 가장 크게 낮추는 병목이었다. 따라서 발표에서는 전체 7감정 점수와 함께 공포 제외, 그리고 공포/혐오/분노/놀람 계열을 묶어 본 보조 해석도 함께 제시한다.

| 재해석 기준 | 결과 | 의미 |
|---|---:|---|
| 전체 7감정 | macro F1 0.5917, match 0.5893 | 최종 런타임 기본 대표값 |
| 공포 F1만 제외한 6감정 평균 | macro F1 0.6278 | 공포 병목을 제외하면 평균 성능 상승 |
| 기대 감정이 공포인 144행 제외 후 6감정 재계산 | macro F1 0.6627, match 0.6458 | 공포 문장을 평가셋에서 제외했을 때의 일반화 성능 |
| 공포/혐오/분노/놀람 4감정 세부 구분 | macro F1 0.5975 | 각 감정을 서로 정확히 나누는 능력은 약 0.60 수준 |
| 기대 감정이 4감정인 576행만 4-class 재계산 | macro F1 0.6008, match 0.5104 | 고각성/불쾌 계열 내부 세분화는 아직 흔들림 |
| 공포/혐오/분노/놀람을 하나의 계열로 통합한 이진 판단 | group F1 0.7845, binary macro F1 0.7951 | 해당 계열인지 아닌지는 비교적 잘 구분 |

이 표를 보고서에서 더 앞에 해석해야 하는 이유는, 실제 프로젝트가 단순 7감정 분류기가 아니라 상담 챗봇 운영 경로이기 때문이다. `v3-1008 blind`처럼 스타일과 표현을 다양화한 평가에서는 2026-05-13 semantic 보강 이후 macro F1 0.7791까지 상승했고, markerless 표현도 0.7511을 기록했다. 다만 최종 배포 런타임에는 post-hoc logit bias가 추가되어 같은 숫자를 현재 대표 점수로 혼용하지 않는다. 현재 최종 런타임 일반화는 새 1008문장 fresh blind의 macro F1 0.5917로 보고하며, 목표 0.70에는 아직 미달하므로 공포 recall과 중립 precision 개선은 후속 과제로 둔다.

세부적으로는 모델이 공포/혐오/분노/놀람을 모두 독립 라벨로 정확히 구분하는 데는 한계가 있지만, 이 네 감정을 하나의 "위협/불쾌/각성 계열"로 묶으면 F1이 0.7845까지 올라간다. 즉 현재 모델은 부정적 고각성 계열 자체는 상당 부분 포착하지만, 그 내부에서 공포와 놀람, 분노와 혐오를 세밀하게 가르는 능력은 다음 개선 과제로 남는다.

### 8.4 Wellness blind 504

기존 평가셋과 중복이 없는 fresh 504개 문장으로 실제 `/chat` 경로를 검증했다. 초기에는 감각 혐오나 low-impact 발화에서 점수 과하락/평평함이 관찰되었고, 이후 정책을 보정했다.

| 항목 | 보정 후 결과 |
|---|---:|
| emotion match | 0.6766 |
| false-low tendency | 24 -> 12 |
| high_distress_low_tendency | 67 -> 36 |
| wellness min / max | 22.26 / 72.74 |
| wellness mean / std | 53.65 / 10.69 |
| flat group | 11/84 -> 1/84 |
| impact 분포 | low 269 / full 230 / none 5 |

이 결과는 점수가 전혀 움직이지 않는 문제와 감각 혐오 과하락 문제를 개선했음을 보여준다. 동시에 fresh 표현에서 emotion match가 0.6766에 머물러, 감정 일반화는 아직 후속 과제임을 같이 보여준다.

### 8.5 Qwen 응답 품질

Qwen 3B raw 응답 품질은 완전히 안정적이지 않았다. 따라서 프로젝트의 운영 전략은 raw 생성 품질보다 최종 노출 응답 안정성에 초점을 두었다.

| 항목 | 상태 |
|---|---|
| 최종 노출 응답 guard | qwen_quality_guard 통과 |
| self-check/fallback | 운영 적용 |
| 일본어 kana 잔여 토큰 | 리허설 중 발견 후 차단 |
| 7B~9B 비교 | 후속 과제 |

---

## 9. 배포와 시연 준비

### 9.1 단일 서버 배포

수업 제출/시연 목표는 개인 GPU PC에서 URL로 접속 가능한 상태를 만드는 것이다. 클라우드 GPU 배포는 비용과 시간 대비 비현실적이므로 제외했다.

`run_deploy.bat`는 다음 순서로 실행된다.

```bat
npm --prefix frontend run build
C:\Users\WD\anaconda3\envs\dl_study\python.exe -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

FastAPI는 `frontend/build`가 있으면 React 화면과 API를 같은 origin에서 제공한다.

| 접속 방식 | URL |
|---|---|
| 같은 PC | `http://127.0.0.1:8000` |
| 같은 네트워크 | `http://<PC-IP>:8000` |
| 외부 접속 | ngrok/cloudflared 터널 -> `localhost:8000` |

수업의 FastAPI/모델 서빙 내용에는 현재 구조가 그대로 대응한다. `/chat` 엔드포인트는 RoBERTa/NLI/CBT/Qwen 추론을 수행하는 모델 serving API이며, Flask 전환 없이 FastAPI 기반 API 서빙으로 설명한다.

Docker 수업 산출물은 `Dockerfile`과 `.dockerignore`로 보강했다. Docker 이미지는 React build와 FastAPI 서버를 한 컨테이너에서 제공하지만, 대용량 체크포인트, 운영 DB, processed runtime 데이터, Hugging Face cache는 이미지에 넣지 않고 `models/roberta/checkpoints`, `models/qwen/checkpoints`, `backend/db`, `data/processed`, Hugging Face cache를 volume mount하는 방식으로 둔다. 실제 발표 1순위 실행 경로는 기존 개인 GPU PC + `run_deploy.bat` + HTTPS 터널이다.

### 9.2 배포 smoke

발표 직전 검증은 다음 명령으로 수행한다.

```bat
C:\Users\WD\anaconda3\envs\dl_study\python.exe eval\deploy_smoke.py --base-url http://127.0.0.1:8000
```

이 smoke는 `/health`, React 정적 루트, 가입, 로그인, 채팅, 하루 마감, 캘린더를 확인한다.

최신 smoke 결과는 2026-05-27 외부 터널 URL 기준으로 통과했다. 터널 URL은 일회성일 수 있으므로 발표 직전에는 현재 실행 중인 `localhost:8000` 또는 새 터널 URL로 같은 명령을 다시 실행한다.

| 항목 | 결과 |
|---|---|
| base URL | cloudflared 외부 URL -> `localhost:8000` |
| health | 통과 |
| React root | 통과 |
| register/login | 통과 |
| chat | 통과 |
| day close | 통과 |
| calendar | 통과 |
| smoke 계정 | `smoke20260527230657` |
| 하루 마감 label | 보통 |
| 하루 마감 wellness | 70.0 |

### 9.3 시연 시나리오

발표에서는 아래 순서로 기능을 보여주는 것이 가장 안정적이다.

1. 로그인 또는 회원가입
2. 일반 대화
3. 긍정 발화
4. 신체 피로
5. 시험 불안
6. 상황성 슬픔
7. 소프트 위기
8. 하드 위기
9. 하루 마감
10. 캘린더 확인

실제 리허설에서는 `/health`, `/auth/register`, `/auth/login`, 7개 채팅 시나리오, `/day/close`, `/calendar/{username}` API와 브라우저 로그인/채팅/캘린더 흐름을 확인했다.

추가로 2026-05-22에는 `eval/check_demo_40day_conversations.py`로 40일 x 3~5개 발화의 장기 시연 후보를 검산했다. 전체 40일 130발화 기준 레이블 분포는 양호 3일, 보통 18일, 주의 19일, 위험 0일이며, 17일차에는 hard crisis 1건이 의도대로 잡혔다. 발표에서는 1일차 평범한 하루, 8일차 과부하, 15일차 저점, 17일차 안전 개입, 24/34/40일차 회복 흐름을 대표 날짜로 사용하면 캘린더 변화가 가장 자연스럽다. 상세 입력 문장과 점수는 `eval/report/demo_40day_conversation_check_20260522.md`에 남겼다.

---

## 10. 최종 제출 패키지 구성

최종 제출물은 `감정챗봇_최종제출/` 아래 네 개 폴더로 구성한다. 이 보고서가 들어 있는 `03_소스코드/`는 코드뿐 아니라 **실행에 필요한 모델 체크포인트와 배포 설정(`docker.env`)을 함께 포함**해, 받는 사람이 추가 다운로드 없이 바로 실행할 수 있게 했다.

| 폴더 | 내용 |
|---|---|
| `01_보고서/` | 최종 보고서(`.docx`) |
| `02_발표자료/` | 발표 슬라이드(`.pptx` + `.pdf`) |
| `03_소스코드/` | 전체 소스 + 모델 체크포인트 + 배포 설정 + `README.md` |
| `04_사용데이터/` | 원본(raw)·전처리(processed)·NLI 데이터 + `DATA_README.md` |

`03_소스코드/` 포함 항목:

| 구분 | 포함 |
|---|---|
| 백엔드 | `backend/*.py`, DB 모델/CRUD/마이그레이션 소스 |
| 파이프라인 | `pipeline/*.py` |
| 모델 코드 | `models/roberta/*.py`, `models/qwen/*.py` |
| 모델 체크포인트 | `models/roberta/checkpoints/`(`roberta_final.pt` + head 3종 + runtime JSON 3종), `models/qwen/checkpoints/`(QLoRA 어댑터) |
| 프론트 | `frontend/src`, `frontend/public`, `package*.json` |
| 평가/전처리 | `eval/*.py`, `data/preprocess`, `data/nli`, `data/processed`(운영 anchor) |
| 문서 | `README.md`(실행·검증·시연 단일 안내서) |
| 배포 | `run_deploy.bat`, `Dockerfile`, `.dockerignore`, `docker.env`, `docker.env.example` |

제외/자동 항목:

| 구분 | 사유 |
|---|---|
| Qwen2.5-3B base 가중치(약 6GB) | 첫 실행 시 Hugging Face에서 자동 다운로드 |
| 운영 SQLite DB | 첫 실행 때 새로 생성 |
| `frontend/node_modules/`, `frontend/build/` | `npm ci` / build로 복원 |
| `data/raw/` 원본 | 코드 폴더에는 두지 않고 `04_사용데이터/raw/`로 별도 제공 |

`docker.env`에는 인증 secret과 관리자 비밀번호가 들어 있으므로 외부 공개를 금지한다. 체크포인트와 운영 anchor가 함께 들어 있어, 받는 사람은 `dl_study` conda 환경 또는 Docker로 추가 가중치 내려받기 없이(Qwen base 제외) 바로 시연할 수 있다. 최종 제출 폴더/패키지 점검 결과는 `eval/report/submission_package_final_check_20260612.md`에 정리했다.

---

## 11. 미채택 또는 보류한 선택

| 항목 | 판단 |
|---|---|
| 클라우드 GPU 배포 | 수업 기간/비용 대비 비현실적이라 개인 GPU PC 배포 채택 |
| CPU-only 배포 | Qwen 추론 속도 때문에 부적합 |
| Flask 전환 | FastAPI도 수업 범위에 포함되고 기존 앱이 FastAPI로 완성되어 전환하지 않음 |
| 공개 서비스 수준 배포 | production 위험 기본값 차단은 보강했지만 HTTPS, 개인정보 고지, rate limit, 계정 복구, 삭제/내보내기, 임상 검증이 없어 불가 |
| Qwen 7B~9B 교체 | raw 응답 품질 개선 후보이나 RTX 3060Ti/일정 제약으로 후속 과제 |
| v2.4 모델 swap | 일부 지표 개선은 있으나 행복 회귀와 작은 개선 폭으로 제출 전 swap 보류 |
| logistic calibration | val Macro F1 0.3991로 하락해 미채택 |
| 감정 Macro F1 0.70 달성 | 단기 제출 전에는 현실적이지 않아 운영 게이트/파이프라인 완성도를 강조 |

---

## 12. 한계

이 프로젝트는 수업 제출/시연용 시스템으로는 완성도가 높지만, 실제 정신건강 서비스로 공개하기에는 한계가 분명하다.

기술적 한계:

1. 감정 분류 Macro F1은 목표 0.70에 도달하지 못했다.
2. 한국어 대화체 라벨은 맥락 의존성이 강해 단일 발화 기준 평가에 노이즈가 있다.
3. Qwen 3B raw 응답은 self-check/fallback 없이 단독 사용하기 어렵다.
4. CBT anchor는 reliability gate로 개선했지만, 완전한 임상적 인지 왜곡 판별기가 아니다.
5. fresh blind 일반화에서는 여전히 감정별 편차가 존재한다.

서비스 한계:

1. HTTPS와 httpOnly cookie/refresh token 구조가 없다.
2. 개인정보 수집 동의/삭제/내보내기 기능이 없다.
3. rate limit과 계정 복구 기능이 없다.
4. 임상 검증을 거치지 않았다.
5. 장기 공개 운영을 위한 모니터링/백업/보안 정책이 부족하다.

따라서 발표에서는 “의료 진단 앱”이 아니라 “딥러닝 기반 정서 모니터링 수업 프로젝트”로 설명해야 한다.

---

## 13. 결론

최종 시스템은 데이터 전처리, RoBERTa 판별 모델, Qwen 상담 응답, 위기 안전 게이트, CBT 보조 점수, 우울 경향 분리, FastAPI 백엔드, React 프론트, SQLite 저장, 캘린더, 단일 서버 배포까지 하나의 흐름으로 통합했다.

가장 큰 성과는 모델 성능 숫자 하나보다도, 실제 사용자가 URL로 접속해 로그인하고 대화하며 하루 마감과 캘린더를 확인할 수 있는 end-to-end 시스템을 완성했다는 점이다. 또한 감정 분류의 한계, Qwen 3B raw 품질 한계, CBT anchor false-positive 문제를 숨기지 않고 운영 게이트와 문서화된 평가로 관리했다.

수업 제출/시연 기준으로는 다음 상태에 도달했다.

| 기준 | 상태 |
|---|---|
| 로그인/채팅/점수/하루 마감/캘린더 | 구현 완료 |
| 위기 대응 hard/soft flow | 구현 및 리허설 완료 |
| 단일 서버 배포 | 구현 완료 |
| 배포 smoke | 통과 |
| 최종 제출 패키지 구성 | 4개 폴더(보고서·발표·소스·데이터) 정리 완료 |
| 실행용 체크포인트·데이터 동봉 | 완료(추가 다운로드 없이 로컬 실행) |
| 실제 공개 서비스 수준 | production 기본값 하드닝 완료, 공개 서비스 필수 항목은 후속 과제 |

후속 과제는 감정 일반화 개선, Qwen 7B~9B 비교, 우울 경향 v2 학습형 head 검토, HTTPS/httpOnly cookie/refresh token/rate limit/계정 삭제와 내보내기/개인정보 동의 UI 같은 공개 서비스 수준 보안·개인정보·운영 기능 보강이다.

---

## 14. 재현 및 실행 명령

개발 서버:

```bat
run_backend.bat
npm --prefix frontend start
```

수업 데모 배포:

```bat
set EMOTION_CHATBOT_ENV=production
set EMOTION_CHATBOT_AUTH_SECRET=change-me-long-random
set EMOTION_CHATBOT_DEVELOPER_PASSWORD=change-me-strong
set EMOTION_CHATBOT_ROOT_PASSWORD=change-me-strong
run_deploy.bat
```

배포 smoke:

```bat
C:\Users\WD\anaconda3\envs\dl_study\python.exe eval\deploy_smoke.py --base-url http://127.0.0.1:8000
```

Docker 패키징 확인:

```bat
docker build -t emotion-chatbot:class-demo .
docker run --rm --gpus all -p 8000:8000 ^
  -e EMOTION_CHATBOT_ENV=production ^
  -e EMOTION_CHATBOT_AUTH_SECRET=change-me-long-random ^
  -e EMOTION_CHATBOT_DEVELOPER_PASSWORD=change-me-strong ^
  -e EMOTION_CHATBOT_ROOT_PASSWORD=change-me-strong ^
  -e HF_HOME=/app/.cache/huggingface ^
  -v %cd%\models\roberta\checkpoints:/app/models/roberta/checkpoints ^
  -v %cd%\models\qwen\checkpoints:/app/models/qwen/checkpoints ^
  -v %cd%\data\processed:/app/data/processed ^
  -v %cd%\backend\db:/app/backend/db ^
  -v %USERPROFILE%\.cache\huggingface:/app/.cache/huggingface ^
  emotion-chatbot:class-demo
```

핵심 회귀 검증:

```bat
C:\Users\WD\anaconda3\envs\dl_study\python.exe eval\run_posttrain_checks.py
C:\Users\WD\anaconda3\envs\dl_study\python.exe eval\roberta_emotion_guard.py
C:\Users\WD\anaconda3\envs\dl_study\python.exe eval\qwen_quality_guard.py
C:\Users\WD\anaconda3\envs\dl_study\python.exe eval\cbt_reliability_guard.py
C:\Users\WD\anaconda3\envs\dl_study\python.exe eval\intensifier_minimal_pair_guard.py
C:\Users\WD\anaconda3\envs\dl_study\python.exe eval\admin_feature_guard.py
C:\Users\WD\anaconda3\envs\dl_study\python.exe eval\security_privacy_audit.py
```

---

## 15. 참고 산출물

| 파일 | 설명 |
|---|---|
| `README.md` | 실행·검증·시연 단일 안내서 |
| `eval/report/final_report.md` | 제출 보조 보고서(운영 메모 포함) |
| `eval/report/project_report_20260529.md` | 프로젝트 종합 보고서 |
| `eval/report/submission_fresh_blind_1008_20260522_summary.txt` | 최종 감정 일반화(Fresh Blind 1008) 평가 요약 |
| `eval/report/submission_fresh_blind_1008_current_decision_20260522.md` | 위 평가 채택/판단 근거 |
| `eval/report/submission_score_risk_assessment_20260522.md` | 점수 시스템 리스크 평가 |
| `eval/report/semantic_emotion_scenario_eval_v3_1008_roberta_quick_current_20260522.md` | 시나리오 기반 감정 평가 |
| `eval/report/wellness_blind_504_analysis_after_policy_fix.md` | 웰니스 정책 보정 후 평가 |
| `eval/report/deploy_rehearsal_20260521.md` | 실제 배포 리허설 기록 |
| `eval/report/demo_40day_conversation_check_20260522.md` | 40일 시연용 대화 후보 검산 |
| `eval/report/security_privacy_followup_20260528.md` | 공개 서비스 전 보안·개인정보 후속 과제 |
| `eval/report/security_privacy_audit_latest.json` | 원문 미노출 보안/개인정보 점검 최신 결과 |
| `eval/report/submission_package_final_check_20260612.md` | 제출 폴더 최종 점검 결과 |
