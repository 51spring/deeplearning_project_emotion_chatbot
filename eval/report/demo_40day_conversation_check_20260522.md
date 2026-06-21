# 40일 시연용 대화 후보 검산

이 문서는 발표/시연용 장기 캘린더 흐름을 만들기 위한 사용자 발화 후보를 현재 RoBERTa/CBT/score_policy 경로로 검산한 결과다.
Qwen 생성은 발표 중 직접 입력할 소수 턴에서 확인하고, 40일 전체는 점수와 캘린더 흐름 검산에 초점을 둔다.

## 요약

- 일수: 40
- 사용자 발화 수: 130
- 레이블 분포: {'양호': 3, '보통': 18, '주의': 19, '위험': 0}
- wellness 범위: 47.07 ~ 66.57
- 추천 대표 날짜: [1, 8, 15, 17, 24, 34, 40]

## 발표 중 직접 입력 추천 5턴

1. 오늘은 수업 듣고 산책했더니 기분이 조금 가벼워졌어.
2. 다음 주 발표 생각하면 손이 차가워지는 느낌이 있어.
3. 요즘 잠이 얕고 아침마다 몸이 무거워서 아무것도 하기 싫어.
4. 스스로를 해칠까 봐 무서운 마음이 들어.
5. 어제보다 조금 나아져서 방 정리하고 친구에게 답장했어.

## 40일 일별 입력안

| Day | 주제 | 발화 수 | wellness | label | 위기 | 우울경향 평활 |
|---:|---|---:|---:|---|---:|---:|
| 1 | 초기 기준선 - 평범한 하루 | 3 | 65.09 | 보통 | 0 | 0.0 |
| 2 | 가벼운 긍정 | 3 | 66.57 | 보통 | 0 | 0.0 |
| 3 | 일상 루틴 | 3 | 65.74 | 보통 | 0 | 0.0 |
| 4 | 작은 성취 | 3 | 65.28 | 보통 | 0 | 0.0 |
| 5 | 가벼운 피로 | 3 | 64.95 | 보통 | 0 | 0.0 |
| 6 | 과제 부담 시작 | 4 | 65.3 | 보통 | 0 | 0.0 |
| 7 | 발표 긴장 | 3 | 60.23 | 보통 | 0 | 0.0 |
| 8 | 일상 과부하 | 4 | 58.65 | 주의 | 0 | 0.0206 |
| 9 | 관계 서운함 | 3 | 57.36 | 주의 | 0 | 0.0549 |
| 10 | 신체 피로와 휴식 | 3 | 60.19 | 보통 | 0 | 0.0384 |
| 11 | 수면 저하 | 4 | 61.33 | 보통 | 0 | 0.0269 |
| 12 | 무기력 | 4 | 58.83 | 주의 | 0 | 0.06 |
| 13 | 자기비난 | 3 | 58.98 | 주의 | 0 | 0.1331 |
| 14 | 사회적 위축 | 4 | 54.32 | 주의 | 0 | 0.154 |
| 15 | 저점 - 우울 경향 | 4 | 52.67 | 주의 | 0 | 0.1875 |
| 16 | 불안과 무기력 혼합 | 3 | 47.07 | 주의 | 0 | 0.171 |
| 17 | 안전 개입 시연용 위기 | 3 | 49.11 | 주의 | 1 | 0.1197 |
| 18 | 도움 요청 후 안정 | 4 | 53.02 | 주의 | 0 | 0.1036 |
| 19 | 느린 회복 | 3 | 47.83 | 주의 | 0 | 0.0725 |
| 20 | 상담/지원 연결 | 3 | 49.89 | 주의 | 0 | 0.0508 |
| 21 | 회복 중 흔들림 | 3 | 54.91 | 주의 | 0 | 0.0355 |
| 22 | 작은 루틴 회복 | 4 | 57.97 | 주의 | 0 | 0.0249 |
| 23 | 중립 안정 | 3 | 60.61 | 보통 | 0 | 0.0174 |
| 24 | 가벼운 긍정 회복 | 3 | 59.85 | 주의 | 0 | 0.0122 |
| 25 | 관계 회복 | 3 | 61.5 | 보통 | 0 | 0.0085 |
| 26 | 학업 재개 | 4 | 62.69 | 보통 | 0 | 0.006 |
| 27 | 경도 불안 | 3 | 55.47 | 주의 | 0 | 0.0042 |
| 28 | 감각 혐오 low-impact | 3 | 57.26 | 주의 | 0 | 0.0029 |
| 29 | 분노/짜증 | 3 | 54.18 | 주의 | 0 | 0.002 |
| 30 | 완충 구간 진입 | 3 | 56.01 | 주의 | 0 | 0.0014 |
| 31 | 개인 기준 비교 | 3 | 60.21 | 보통 | 0 | 0.0294 |
| 32 | 회복감 | 3 | 63.15 | 보통 | 0 | 0.0205 |
| 33 | 가벼운 사회 연결 | 3 | 62.39 | 보통 | 0 | 0.0144 |
| 34 | 양호 흐름 | 3 | 64.55 | 보통 | 0 | 0.0101 |
| 35 | 소폭 흔들림 | 3 | 64.01 | 보통 | 0 | 0.007 |
| 36 | 완충 종료 전 안정 | 3 | 58.16 | 주의 | 0 | 0.0343 |
| 37 | 퍼센타일 기준 전환 후 | 3 | 59.9 | 보통 | 0 | 0.024 |
| 38 | 양호한 일상 | 3 | 61.81 | 양호 | 0 | 0.0168 |
| 39 | 좋은 마무리 | 3 | 63.36 | 양호 | 0 | 0.0118 |
| 40 | 시연 마무리용 회복 | 4 | 63.64 | 양호 | 0 | 0.0082 |

## 전체 발화

### Day 1 - 초기 기준선 - 평범한 하루 (wellness 65.09, 보통)
- 오늘은 수업 듣고 점심도 챙겨 먹었어. [중립, casual_neutral, contrib=0.3605, policy=low_neutral_clamped_plus_minus_8]
- 저녁에는 산책을 조금 했고 바람이 괜찮았어. [행복, casual_share, contrib=0.2825, policy=low_positive_continuous_baseline_to_plus_8]
- 큰일은 없었고 그냥 무난한 하루였어. [중립, casual_share, contrib=0.377, policy=low_neutral_clamped_plus_minus_8]

### Day 2 - 가벼운 긍정 (wellness 66.57, 보통)
- 아침에 커피 마시면서 준비하니까 기분이 조금 괜찮았어. [행복, positive_share, contrib=0.2782, policy=low_positive_continuous_baseline_to_plus_8]
- 과제 자료를 정리해두니까 마음이 한결 가벼웠어. [중립, casual_neutral, contrib=0.3729, policy=low_neutral_clamped_plus_minus_8]
- 밤에는 좋아하는 노래를 들으면서 쉬었어. [행복, positive_share, contrib=0.2825, policy=low_positive_continuous_baseline_to_plus_8]

### Day 3 - 일상 루틴 (wellness 65.74, 보통)
- 도서관에 갔다가 집에 와서 빨래를 돌렸어. [중립, casual_neutral, contrib=0.3666, policy=low_neutral_clamped_plus_minus_8]
- 오늘은 특별한 감정은 없고 해야 할 일만 조금 했어. [중립, casual_share, contrib=0.3631, policy=low_neutral_clamped_plus_minus_8]
- 저녁에는 일찍 씻고 누웠어. [중립, casual_neutral, contrib=0.3536, policy=low_neutral_clamped_plus_minus_8]

### Day 4 - 작은 성취 (wellness 65.28, 보통)
- 미뤄둔 정리를 끝내서 방이 조금 넓어진 느낌이야. [중립, casual_neutral, contrib=0.352, policy=low_neutral_clamped_plus_minus_8]
- 수업 필기도 다시 보니까 생각보다 이해가 됐어. [중립, casual_share, contrib=0.3644, policy=low_neutral_clamped_plus_minus_8]
- 오늘은 나쁘지 않게 지나간 것 같아. [중립, casual_share, contrib=0.3633, policy=low_neutral_clamped_plus_minus_8]

### Day 5 - 가벼운 피로 (wellness 64.95, 보통)
- 오전에 이동이 많아서 몸이 조금 피곤했어. [중립, routine_discomfort, contrib=0.3526, policy=full_affecting_type_or_crisis]
- 그래도 밥은 챙겨 먹었고 과제도 조금 했어. [중립, casual_neutral, contrib=0.38, policy=low_neutral_clamped_plus_minus_8]
- 밤에는 그냥 쉬고 싶다는 생각이 컸어. [중립, casual_neutral, contrib=0.3514, policy=low_neutral_clamped_plus_minus_8]

### Day 6 - 과제 부담 시작 (wellness 65.3, 보통)
- 해야 할 일이 생각보다 많아서 조금 부담됐어. [중립, routine_discomfort, contrib=0.3563, policy=full_affecting_type_or_crisis]
- 과제 마감이 떠오르면 마음이 답답해져. [중립, emotional_distress, contrib=0.3487, policy=full_affecting_type_or_crisis]
- 그래도 오늘은 자료 찾는 것까지는 해냈어. [행복, casual_share, contrib=0.2825, policy=low_positive_continuous_baseline_to_plus_8]
- 잠깐 쉬니까 머리가 조금 정리됐어. [중립, casual_neutral, contrib=0.3536, policy=low_neutral_clamped_plus_minus_8]

### Day 7 - 발표 긴장 (wellness 60.23, 보통)
- 다음 주 발표 생각하면 손이 차가워지는 느낌이 있어. [공포, emotional_distress, contrib=0.55, policy=full_affecting_type_or_crisis]
- 실수할까 봐 계속 발표 순서를 다시 확인했어. [공포, emotional_distress, contrib=0.55, policy=full_affecting_type_or_crisis]
- 그래도 연습을 한 번 끝내니 조금 낫긴 했어. [중립, routine_discomfort, contrib=0.4362, policy=full_affecting_type_or_crisis]

### Day 8 - 일상 과부하 (wellness 58.65, 주의)
- 오늘 아침부터 할 일이 너무 많아서 버거웠어. [슬픔, emotional_distress, contrib=0.52, policy=full_affecting_type_or_crisis]
- 메시지도 밀리고 과제도 남아서 머리가 복잡했어. [중립, emotional_distress, contrib=0.4695, policy=full_affecting_type_or_crisis]
- 하나씩 처리하려고 했는데 속도가 잘 안 났어. [중립, routine_discomfort, contrib=0.4689, policy=full_affecting_type_or_crisis]
- 저녁쯤 되니까 그냥 멍해졌어. [놀람, emotional_distress, contrib=0.3484, policy=full_affecting_type_or_crisis]

### Day 9 - 관계 서운함 (wellness 57.36, 주의)
- 친구가 약속을 갑자기 미뤄서 조금 서운했어. [놀람, emotional_distress, contrib=0.55, policy=full_affecting_type_or_crisis]
- 별일 아닌데도 내가 덜 중요한 사람처럼 느껴졌어. [중립, emotional_distress, contrib=0.3515, policy=full_affecting_type_or_crisis]
- 말로 크게 표현하진 않았지만 마음이 가라앉았어. [중립, emotional_distress, contrib=0.3773, policy=full_affecting_type_or_crisis]

### Day 10 - 신체 피로와 휴식 (wellness 60.19, 보통)
- 오늘은 계단을 많이 오르내려서 다리가 무거웠어. [중립, casual_neutral, contrib=0.3551, policy=low_neutral_clamped_plus_minus_8]
- 몸이 피곤하니까 마음도 조금 예민해졌어. [중립, emotional_distress, contrib=0.3488, policy=full_affecting_type_or_crisis]
- 그래도 따뜻한 물로 씻고 쉬니까 괜찮아졌어. [행복, positive_share, contrib=0.2825, policy=low_positive_continuous_baseline_to_plus_8]

### Day 11 - 수면 저하 (wellness 61.33, 보통)
- 요즘 잠이 얕아서 아침에 일어나도 개운하지 않아. [중립, emotional_distress, contrib=0.3786, policy=full_affecting_type_or_crisis]
- 수업 중에도 집중이 자꾸 끊겼어. [중립, emotional_distress, contrib=0.3491, policy=full_affecting_type_or_crisis]
- 별일 아닌 말에도 쉽게 지치는 느낌이 있었어. [중립, emotional_distress, contrib=0.3495, policy=full_affecting_type_or_crisis]
- 오늘은 말수가 줄어든 것 같아. [중립, routine_discomfort, contrib=0.3524, policy=full_affecting_type_or_crisis]

### Day 12 - 무기력 (wellness 58.83, 주의)
- 아침부터 몸이 무겁고 아무것도 하기 싫었어. [슬픔, emotional_distress, contrib=0.5825, policy=full_affecting_type_or_crisis]
- 해야 할 걸 알면서도 계속 미루게 됐어. [분노, emotional_distress, contrib=0.6738, policy=full_affecting_type_or_crisis]
- 좋아하던 영상도 별로 보고 싶지 않았어. [행복, positive_share, contrib=0.3, policy=low_positive_continuous_baseline_to_plus_8]
- 그냥 시간이 지나가길 기다린 느낌이야. [중립, emotional_distress, contrib=0.3601, policy=full_affecting_type_or_crisis]

### Day 13 - 자기비난 (wellness 58.98, 주의)
- 오늘은 내가 너무 부족한 사람 같다는 생각이 들었어. [중립, emotional_distress, contrib=0.3677, policy=full_affecting_type_or_crisis]
- 작은 실수 하나가 계속 떠올라서 마음이 무거웠어. [슬픔, emotional_distress, contrib=0.5854, policy=full_affecting_type_or_crisis]
- 괜찮다고 생각하려 해도 마음이 쉽게 가벼워지진 않았어. [행복, emotional_distress, contrib=0.345, policy=full_affecting_type_or_crisis]

### Day 14 - 사회적 위축 (wellness 54.32, 주의)
- 사람들이랑 말하는 게 오늘은 유난히 힘들었어. [슬픔, emotional_distress, contrib=0.9719, policy=full_affecting_type_or_crisis]
- 답장도 늦게 하고 그냥 혼자 있고 싶었어. [중립, emotional_distress, contrib=0.36, policy=full_affecting_type_or_crisis]
- 괜히 내가 끼어들 자리가 없는 것처럼 느껴졌어. [중립, emotional_distress, contrib=0.3525, policy=full_affecting_type_or_crisis]
- 저녁에는 방에만 있었어. [중립, casual_neutral, contrib=0.3506, policy=low_neutral_clamped_plus_minus_8]

### Day 15 - 저점 - 우울 경향 (wellness 52.67, 주의)
- 요즘 계속 기운이 없고 뭘 해도 의미가 잘 안 느껴져. [중립, emotional_distress, contrib=0.3508, policy=full_affecting_type_or_crisis]
- 아침에 일어나는 것부터 너무 버겁게 느껴졌어. [중립, casual_neutral, contrib=0.3497, policy=low_neutral_clamped_plus_minus_8]
- 예전에는 좋아하던 일도 손이 잘 안 가. [행복, positive_share, contrib=0.2825, policy=low_positive_continuous_baseline_to_plus_8]
- 오늘은 내가 사라져도 별 차이 없을 것 같다는 생각이 스쳤어. [슬픔, emotional_distress, contrib=0.9365, policy=full_affecting_type_or_crisis]

### Day 16 - 불안과 무기력 혼합 (wellness 47.07, 주의)
- 마감이 다가오는데 몸은 안 움직이고 마음만 조급해. [중립, emotional_distress, contrib=0.4219, policy=full_affecting_type_or_crisis]
- 계속 뒤처지는 것 같아서 불안했어. [공포, emotional_distress, contrib=0.8085, policy=full_affecting_type_or_crisis]
- 괜히 주변 사람들한테 미안한 마음도 들었어. [슬픔, crisis_candidate, contrib=0.9451, policy=full_affecting_type_or_crisis]

### Day 17 - 안전 개입 시연용 위기 (wellness 49.11, 주의)
- 오늘은 혼자 있으면 위험할 것 같아서 바로 도움을 요청해야 할 것 같아. [중립, casual_share, contrib=0.3534, policy=low_neutral_clamped_plus_minus_8]
- 스스로를 해칠까 봐 무서운 마음이 들어. [슬픔, emotional_distress, contrib=0.8285, policy=full_affecting_type_or_crisis / hard crisis]
- 지금은 누군가에게 연락해서 옆에 있어 달라고 말해야 할 것 같아. [중립, casual_share, contrib=0.38, policy=low_neutral_clamped_plus_minus_8]

### Day 18 - 도움 요청 후 안정 (wellness 53.02, 주의)
- 어제는 결국 친구에게 연락했고 혼자 있지 않으려고 했어. [중립, emotional_distress, contrib=0.3507, policy=full_affecting_type_or_crisis]
- 아직 마음이 무겁지만 그래도 도움을 요청한 건 잘한 것 같아. [슬픔, emotional_distress, contrib=0.5095, policy=full_affecting_type_or_crisis]
- 오늘은 밥을 조금 먹고 잠깐 밖에 나갔어. [중립, casual_neutral, contrib=0.3539, policy=low_neutral_clamped_plus_minus_8]
- 위험한 생각은 어제보다 조금 줄었어. [중립, emotional_distress, contrib=0.3642, policy=full_affecting_type_or_crisis]

### Day 19 - 느린 회복 (wellness 47.83, 주의)
- 오늘은 아침에 일어나는 게 여전히 힘들었어. [슬픔, emotional_distress, contrib=0.967, policy=full_affecting_type_or_crisis]
- 그래도 세수하고 책상 위를 조금 치웠어. [중립, casual_share, contrib=0.3756, policy=low_neutral_clamped_plus_minus_8]
- 친구가 안부를 물어봐 줘서 마음이 조금 놓였어. [행복, positive_share, contrib=0.3, policy=low_positive_continuous_baseline_to_plus_8]

### Day 20 - 상담/지원 연결 (wellness 49.89, 주의)
- 학교 상담센터 예약을 알아봤어. [중립, casual_neutral, contrib=0.3547, policy=low_neutral_clamped_plus_minus_8]
- 막상 신청하려니 긴장됐지만 필요하다는 생각이 들어. [공포, emotional_distress, contrib=0.8015, policy=full_affecting_type_or_crisis]
- 오늘은 무리하지 않고 신청 페이지를 열어본 것만으로도 됐다고 생각했어. [중립, casual_neutral, contrib=0.3698, policy=low_neutral_clamped_plus_minus_8]

### Day 21 - 회복 중 흔들림 (wellness 54.91, 주의)
- 괜찮아지는 줄 알았는데 오후에 갑자기 마음이 꺼졌어. [행복, emotional_distress, contrib=0.345, policy=full_affecting_type_or_crisis]
- 그래도 전처럼 혼자 끌고 가지 않으려고 메모를 남겼어. [중립, casual_neutral, contrib=0.38, policy=low_neutral_clamped_plus_minus_8]
- 저녁에는 따뜻한 걸 먹으면서 조금 진정됐어. [행복, casual_share, contrib=0.2825, policy=low_positive_continuous_baseline_to_plus_8]

### Day 22 - 작은 루틴 회복 (wellness 57.97, 주의)
- 오늘은 알람을 듣고 바로 일어나진 못했지만 결국 씻었어. [중립, casual_share, contrib=0.3498, policy=low_neutral_clamped_plus_minus_8]
- 수업 하나는 집중해서 들었고 필기도 조금 했어. [중립, casual_share, contrib=0.38, policy=low_neutral_clamped_plus_minus_8]
- 아주 좋아진 건 아니지만 완전히 무너지진 않았어. [행복, positive_share, contrib=0.2825, policy=low_positive_continuous_baseline_to_plus_8]
- 밤에는 내일 할 일을 세 개만 적어뒀어. [중립, casual_neutral, contrib=0.38, policy=low_neutral_clamped_plus_minus_8]

### Day 23 - 중립 안정 (wellness 60.61, 보통)
- 오늘은 특별히 좋지도 나쁘지도 않았어. [중립, positive_share, contrib=0.3, policy=low_positive_continuous_baseline_to_plus_8]
- 점심을 챙겨 먹고 강의 자료를 정리했어. [중립, casual_neutral, contrib=0.38, policy=low_neutral_clamped_plus_minus_8]
- 저녁에는 방에서 조용히 쉬었어. [중립, casual_neutral, contrib=0.3511, policy=low_neutral_clamped_plus_minus_8]

### Day 24 - 가벼운 긍정 회복 (wellness 59.85, 주의)
- 오랜만에 산책하면서 공기가 맑다고 느꼈어. [행복, positive_share, contrib=0.3, policy=low_positive_continuous_baseline_to_plus_8]
- 해야 할 일을 조금 끝내니까 마음이 가벼워졌어. [행복, positive_share, contrib=0.275, policy=low_positive_continuous_baseline_to_plus_8]
- 오늘은 어제보다 숨이 트이는 느낌이 있었어. [중립, emotional_distress, contrib=0.7151, policy=full_affecting_type_or_crisis]

### Day 25 - 관계 회복 (wellness 61.5, 보통)
- 친구랑 짧게 통화했는데 생각보다 편했어. [중립, casual_neutral, contrib=0.363, policy=low_neutral_clamped_plus_minus_8]
- 내 상태를 조금 말했더니 이해해 줘서 고마웠어. [행복, positive_share, contrib=0.3, policy=low_positive_continuous_baseline_to_plus_8]
- 혼자만 버티는 느낌이 덜했어. [중립, emotional_distress, contrib=0.3518, policy=full_affecting_type_or_crisis]

### Day 26 - 학업 재개 (wellness 62.69, 보통)
- 오늘은 과제 목차를 다시 잡아봤어. [중립, casual_neutral, contrib=0.3575, policy=low_neutral_clamped_plus_minus_8]
- 완벽하진 않지만 시작한 것만으로도 조금 안심됐어. [중립, positive_share, contrib=0.2992, policy=low_positive_continuous_baseline_to_plus_8]
- 중간에 집중이 깨졌지만 다시 돌아오긴 했어. [중립, casual_share, contrib=0.3496, policy=low_neutral_clamped_plus_minus_8]
- 밤에는 더 욕심내지 않고 멈췄어. [중립, emotional_distress, contrib=0.351, policy=full_affecting_type_or_crisis]

### Day 27 - 경도 불안 (wellness 55.47, 주의)
- 교수님 피드백을 받기 전이라 조금 긴장돼. [공포, emotional_distress, contrib=0.8042, policy=full_affecting_type_or_crisis]
- 결과가 안 좋을까 봐 계속 메일함을 확인했어. [공포, emotional_distress, contrib=0.55, policy=full_affecting_type_or_crisis]
- 그래도 예전처럼 완전히 무너지진 않았어. [중립, casual_share, contrib=0.3478, policy=low_neutral_clamped_plus_minus_8]

### Day 28 - 감각 혐오 low-impact (wellness 57.26, 주의)
- 냉장고 안에서 이상한 냄새가 나서 좀 찝찝했어. [혐오, casual_neutral, contrib=0.45, policy=low_sensory_disgust_clamped_plus_5_15]
- 상한 반찬을 버리고 나니까 속이 조금 편해졌어. [중립, casual_share, contrib=0.3582, policy=low_neutral_clamped_plus_minus_8]
- 기분이 좋진 않았지만 큰일은 아니었어. [중립, positive_share, contrib=0.3, policy=low_positive_continuous_baseline_to_plus_8]

### Day 29 - 분노/짜증 (wellness 54.18, 주의)
- 오늘 누가 내 말을 끊어서 꽤 짜증났어. [분노, emotional_distress, contrib=0.5344, policy=full_affecting_type_or_crisis]
- 내 시간을 당연하게 여기는 태도가 거슬렸어. [혐오, emotional_distress, contrib=0.6566, policy=full_affecting_type_or_crisis]
- 그래도 바로 화내지는 않고 잠깐 자리를 피했어. [중립, routine_discomfort, contrib=0.4341, policy=full_affecting_type_or_crisis]

### Day 30 - 완충 구간 진입 (wellness 56.01, 주의)
- 벌써 한 달 가까이 기록했네. [중립, casual_neutral, contrib=0.3632, policy=low_neutral_clamped_plus_minus_8]
- 오늘은 컨디션이 중간쯤인 것 같아. [중립, casual_share, contrib=0.3751, policy=low_neutral_clamped_plus_minus_8]
- 예전보다 내 상태를 알아차리는 속도는 빨라진 것 같아. [중립, routine_discomfort, contrib=0.4678, policy=full_affecting_type_or_crisis]

### Day 31 - 개인 기준 비교 (wellness 60.21, 보통)
- 오늘은 오전에 조금 무거웠지만 오후에는 괜찮아졌어. [행복, positive_share, contrib=0.2795, policy=low_positive_continuous_baseline_to_plus_8]
- 지난주보다는 덜 가라앉은 느낌이야. [중립, emotional_distress, contrib=0.3488, policy=full_affecting_type_or_crisis]
- 해야 할 일을 두 개 끝내서 안심됐어. [행복, positive_share, contrib=0.2785, policy=low_positive_continuous_baseline_to_plus_8]

### Day 32 - 회복감 (wellness 63.15, 보통)
- 아침에 일어나서 창문을 여니까 기분이 조금 좋아졌어. [행복, positive_share, contrib=0.2796, policy=low_positive_continuous_baseline_to_plus_8]
- 수업 준비도 예상보다 빨리 끝났어. [행복, positive_share, contrib=0.2759, policy=low_positive_continuous_baseline_to_plus_8]
- 오늘은 내 속도가 아주 느리지만 괜찮다고 느꼈어. [행복, positive_share, contrib=0.2814, policy=low_positive_continuous_baseline_to_plus_8]

### Day 33 - 가벼운 사회 연결 (wellness 62.39, 보통)
- 친구랑 점심 먹으면서 근황 얘기했어. [중립, casual_neutral, contrib=0.3555, policy=low_neutral_clamped_plus_minus_8]
- 웃을 일이 조금 있어서 마음이 풀렸어. [행복, positive_share, contrib=0.2813, policy=low_positive_continuous_baseline_to_plus_8]
- 집에 와서도 그 대화가 나쁘지 않게 남아 있었어. [분노, emotional_distress, contrib=0.5355, policy=full_affecting_type_or_crisis]

### Day 34 - 양호 흐름 (wellness 64.55, 보통)
- 오늘은 과제 한 단락을 끝내고 뿌듯했어. [행복, positive_share, contrib=0.2762, policy=low_positive_continuous_baseline_to_plus_8]
- 산책하면서 음악을 들으니 기분이 꽤 가벼웠어. [중립, positive_share, contrib=0.3, policy=low_positive_continuous_baseline_to_plus_8]
- 잠들기 전에도 마음이 많이 복잡하진 않았어. [중립, emotional_distress, contrib=0.3521, policy=full_affecting_type_or_crisis]

### Day 35 - 소폭 흔들림 (wellness 64.01, 보통)
- 오후에 갑자기 지난 실수가 떠올라서 마음이 불편했어. [놀람, emotional_distress, contrib=0.3836, policy=full_affecting_type_or_crisis]
- 그래도 그 생각이 전부는 아니라고 적어봤어. [중립, casual_share, contrib=0.3764, policy=low_neutral_clamped_plus_minus_8]
- 밤에는 조금 차분해졌어. [중립, casual_share, contrib=0.3516, policy=low_neutral_clamped_plus_minus_8]

### Day 36 - 완충 종료 전 안정 (wellness 58.16, 주의)
- 오늘은 수업 듣고 바로 복습까지 조금 했어. [슬픔, emotional_distress, contrib=0.771, policy=full_affecting_type_or_crisis]
- 몸은 피곤했지만 마음은 크게 흔들리지 않았어. [행복, positive_share, contrib=0.3, policy=low_positive_continuous_baseline_to_plus_8]
- 내일은 조금 더 일찍 자보려고 해. [중립, casual_share, contrib=0.38, policy=low_neutral_clamped_plus_minus_8]

### Day 37 - 퍼센타일 기준 전환 후 (wellness 59.9, 보통)
- 기록이 쌓이니까 예전보다 내 패턴이 보이는 것 같아. [중립, casual_neutral, contrib=0.3676, policy=low_neutral_clamped_plus_minus_8]
- 오늘은 평소보다 안정적인 편이었어. [중립, routine_discomfort, contrib=0.3548, policy=full_affecting_type_or_crisis]
- 작은 일에도 덜 휘둘린 느낌이 있었어. [중립, emotional_distress, contrib=0.3533, policy=full_affecting_type_or_crisis]

### Day 38 - 양호한 일상 (wellness 61.81, 양호)
- 아침에 가볍게 스트레칭하고 수업 준비했어. [중립, casual_neutral, contrib=0.3526, policy=low_neutral_clamped_plus_minus_8]
- 과제도 조금 진행했고 점심도 챙겨 먹었어. [중립, casual_neutral, contrib=0.38, policy=low_neutral_clamped_plus_minus_8]
- 오늘은 꽤 무난하고 편안했어. [행복, casual_share, contrib=0.2825, policy=low_positive_continuous_baseline_to_plus_8]

### Day 39 - 좋은 마무리 (wellness 63.36, 양호)
- 오늘은 발표 연습을 끝내고 마음이 놓였어. [행복, positive_share, contrib=0.2812, policy=low_positive_continuous_baseline_to_plus_8]
- 예전처럼 완벽해야 한다는 생각이 덜했어. [중립, emotional_distress, contrib=0.3728, policy=full_affecting_type_or_crisis]
- 저녁에는 친구랑 웃으면서 얘기했어. [중립, casual_neutral, contrib=0.38, policy=low_neutral_clamped_plus_minus_8]

### Day 40 - 시연 마무리용 회복 (wellness 63.64, 양호)
- 40일 동안 기록해보니 내 상태가 조금씩 변하는 게 보여. [중립, casual_neutral, contrib=0.38, policy=low_neutral_clamped_plus_minus_8]
- 힘든 날도 있었지만 도움을 요청하고 다시 루틴을 잡은 게 기억나. [중립, emotional_distress, contrib=0.3925, policy=full_affecting_type_or_crisis]
- 오늘은 예전보다 나를 덜 몰아붙이게 된 것 같아. [중립, routine_discomfort, contrib=0.4044, policy=full_affecting_type_or_crisis]
- 완벽하진 않아도 지금은 조금 괜찮아. [행복, positive_share, contrib=0.2811, policy=low_positive_continuous_baseline_to_plus_8]
