# 현재 최종 런타임 Fresh Blind 1008 판단

- 작성일: 2026-05-22
- 목적: `emotion_logit_bias` 채택 이후 기존 `v3-1008 blind` 점수를 현재 최종 모델 대표값으로 제시해도 되는지 판단한다.

## 결론

`v3-1008 blind`의 과거 `macro F1 0.7791`은 현재 최종 배포 런타임 대표 점수로 제시하지 않는다.

해당 수치는 2026-05-13의 `v2.2 + rulepatch` 시점 semantic 보강 효과를 보여주는 역사적/ablation 근거로는 사용할 수 있다. 하지만 2026-05-22에는 `emotion_logit_bias`와 P95 재측정이 운영 기본값으로 들어갔고, 같은 v3-1008을 현재 런타임으로 재평가하면 성능이 달라진다.

따라서 발표/보고서에서는 다음처럼 구분한다.

| 구분 | 평가셋 | 설정 | 결과 | 보고서 사용 |
|---|---|---|---:|---|
| 과거 semantic 보강 근거 | v3-1008 blind | 2026-05-13 `v2.2 + rulepatch` | macro F1 0.7791 | historical/ablation으로만 사용 |
| 현재 런타임 재평가 | 기존 v3-1008 | 2026-05-22 최종 runtime | macro F1 0.5623, match 0.5565 | 현재 설정 변화 설명용 |
| 현재 fresh blind | 새 1008문장 | 2026-05-22 최종 runtime | macro F1 0.5917, match 0.5893 | 현재 최종 일반화 근거 |

## 새 1008문장 생성 기준

- 생성 파일: `eval/report/submission_fresh_blind_1008_20260522.csv`
- 구조: 12스타일 × 7감정 × 12케이스 = 1008행
- 감정별 분포: 각 144행
- 기존 평가 CSV와 `text`, `base_text`, `user_text`, `utterance` 기준 정확 중복 0건
- 공백·문장부호 제거 축약 중복 0건
- 생성 스크립트: `eval/build_submission_fresh_blind_1008_dataset.py`

## 현재 fresh blind 1008 결과

| 지표 | 값 |
|---|---:|
| 전체 행 수 | 1008 |
| match | 594 |
| match rate | 0.5893 |
| macro F1 | 0.5917 |
| 평균 depression_score | 0.4749 |
| 평균 depression_tendency_score | 0.0252 |
| false-low tendency | 36 |
| false-high tendency | 43 |
| high distress + low tendency | 29 |

감정별 F1:

| 감정 | Precision | Recall | F1 |
|---|---:|---:|---:|
| 행복 | 0.6478 | 0.7153 | 0.6799 |
| 중립 | 0.3426 | 0.8542 | 0.4891 |
| 슬픔 | 0.6727 | 0.5139 | 0.5827 |
| 공포 | 0.7500 | 0.2500 | 0.3750 |
| 혐오 | 0.7447 | 0.7292 | 0.7368 |
| 분노 | 0.9718 | 0.4792 | 0.6419 |
| 놀람 | 0.7000 | 0.5833 | 0.6364 |

## 해석

현재 fresh blind macro F1 0.5917은 raw val Macro F1 0.4407보다 발표용 설명력이 높다. 즉 실제 운영 경로의 정규화, 라우팅, 후처리까지 포함하면 단독 감정 head보다 낫다는 근거로 쓸 수 있다.

다만 목표선 0.70에는 아직 도달하지 못했다. 특히 공포 recall 0.2500, 중립 precision 0.3426이 병목이다. 이 fresh set을 보고 다시 룰을 추가하면 blind 성격이 깨지므로, 제출 직전에는 추가 overfit 보정보다 현재 결과를 정직하게 제시하고 후속 과제로 두는 것이 안전하다.

## 발표 문장

> 2026-05-13의 v3-1008 blind 0.7791은 semantic 보강 단계의 역사적 성능으로 제시하고, 최종 배포 런타임의 현재 일반화 점수는 새로 만든 1008문장 fresh blind에서 macro F1 0.5917로 별도 보고했습니다. 따라서 과거 점수를 현재 배포 모델 점수처럼 혼용하지 않았습니다.
