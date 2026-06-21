# 제출 평가 점수 리스크 판단

- 작성일: 2026-05-22
- 목적: 교수자가 "평가 점수가 낮으면 모델 의미가 없다"는 기준을 강하게 적용할 때 현재 제출 점수가 충분한지 판단한다.

## 결론

현재 상태로 제출은 가능하지만, **raw 감정 val Macro F1 0.4407을 대표 점수로 앞세우면 위험하다.** 또한 `v3-1008 blind`의 과거 `0.7791`은 2026-05-13 `v2.2 + rulepatch` 시점의 historical/ablation 점수이므로, 2026-05-22 최종 배포 런타임 대표값으로 혼용하면 안 된다. 대신 대표 평가는 다음 순서로 제시하는 것이 안전하다.

1. 현재 fresh blind 1008: macro F1 0.5917, match 0.5893
2. `scenario_eval_v2`: macro F1 0.5344, markerless 0.4937
3. 운영 posttrain: val Macro F1 0.4407, calib Macro F1 0.4526, NLI 684쌍 FP=0/FN=0
4. `v3-1008 blind` historical: macro F1 0.7791, markerless 0.7511(현재 대표값이 아니라 semantic 보강 근거)
5. Wellness fresh 504: emotion match 0.6766
6. CBT reliability: 운영 리뷰 큐 80건 중 76건 해소, 4건 잔여
7. 배포 smoke: `/health`부터 로그인, 채팅, 하루 마감, 캘린더까지 통과

## 왜 raw val Macro F1만 보면 약한가

`emotion_val_clean.csv`는 중립 편중이 강하다. 7,812건 중 중립이 6,162건이다.

| 기준 | Accuracy | Macro F1 |
|---|---:|---:|
| majority baseline(전부 중립) | 0.7888 | 0.1260 |
| 운영 posttrain 감정 평가 | 0.7003 | 0.4407 |
| balanced val | - | 0.5444 |

따라서 Accuracy는 대표 지표로 부적합하다. 중립만 찍어도 Accuracy는 높지만, 감정 모델로서는 의미가 없다. 이 프로젝트에서는 Macro F1, 희소 클래스 F1, scenario/blind 평가가 더 타당하다.

## 더 실험할지 판단

제출 직전 추가 모델 변경 실험은 권장하지 않는다.

- v2.4 swap은 행복 회귀와 head 정합 리스크가 있다.
- logistic calibration은 val Macro F1 0.3991로 하락해 이미 미채택했다.
- hard-negative 재학습은 시간이 더 필요하고, safety/NLI/CBT gate 재검증 비용이 크다.
- 현재 채택한 emotion logit bias는 체크포인트를 바꾸지 않고 Macro F1을 올렸고, P95 재측정과 full guard를 통과했다.
- 현재 최종 런타임 기준 새 1008문장 fresh blind는 macro F1 0.5917로 raw val보다 높지만 목표 0.70에는 못 미친다. 이 세트를 보고 추가 룰을 붙이면 blind 성격이 깨지므로 제출 직전에는 미채택한다.

즉, 지금 더 해야 할 일은 성능을 무리하게 올리는 실험이 아니라 **점수 제시 방식 정리**다.

## 발표용 한 문장

> 단순 Accuracy는 중립 편중 때문에 majority baseline도 0.7888까지 나와 대표 지표로 부적합했습니다. 그래서 Macro F1과 blind/scenario 평가를 중심으로 봤고, 최종 배포 런타임은 raw val Macro F1 0.4407, balanced val 0.5444, 새 1008문장 fresh blind macro F1 0.5917, NLI 회귀 FP/FN 0을 달성했습니다. v3-1008 0.7791은 5월 13일 semantic 보강 단계의 historical 결과로 별도 표기했습니다.

## 최종 판단

- 수업 제출/시연용으로는 현재 점수와 기능 완성도면 제출 가능하다.
- 단, "raw 감정 분류 val Macro F1 0.44"만 앞세우면 낮아 보인다.
- 대표 점수는 `fresh blind 1008 0.5917`, `scenario_eval_v2 0.5344`, `balanced val 0.5444`, `NLI FP/FN 0` 조합으로 제시한다.
- `v3-1008 blind 0.7791`은 현재 배포 런타임 대표값이 아니라 과거 semantic 보강 근거로만 제시한다.
- 추가 학습/모델 교체는 제출 전에는 미채택한다.
