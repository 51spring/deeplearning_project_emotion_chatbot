# 대화 스타일×감정 운영 모니터링 시뮬레이션

- 생성 시각: 2026-05-18T01:43:56
- 전체 발화: 504
- 독립 사용자 그룹: 84
- 전체 emotion_match_rate: 0.6766
- 전체 평균 depression_score: 0.4949
- 전체 평균 depression_tendency_score: 0.0447
- 전체 high_distress_low_tendency: 36
- 전체 high_tendency: 24
- 전체 crisis_count: 0

## 스타일별 요약

| 스타일 | n | emotion match | 평균 ds | 평균 dts | high ds + low dts | high dts | top 분포 | label 분포 |
|---|---:|---:|---:|---:|---:|---:|---|---|
| body_signal_short | 42 | 0.6667 | 0.4877 | 0.0439 | 2 | 2 | 중립:17, 행복:6, 혐오:6, 슬픔:4, 놀람:4, 분노:3, 공포:2 | 주의:28, 보통:12, 위험:2 |
| calendar_margin | 42 | 0.6905 | 0.5307 | 0.0439 | 5 | 2 | 중립:14, 혐오:9, 행복:6, 슬픔:4, 놀람:4, 분노:3, 공포:2 | 주의:24, 보통:12, 위험:6 |
| direct_observation | 42 | 0.6905 | 0.5307 | 0.0439 | 5 | 2 | 중립:14, 혐오:9, 행복:6, 슬픔:4, 놀람:4, 분노:3, 공포:2 | 주의:24, 보통:12, 위험:6 |
| errand_note | 42 | 0.6905 | 0.5134 | 0.0487 | 4 | 2 | 중립:14, 행복:6, 혐오:6, 슬픔:5, 분노:5, 놀람:4, 공포:2 | 주의:26, 보통:12, 위험:4 |
| friend_dm | 42 | 0.6905 | 0.4719 | 0.0439 | 1 | 2 | 중립:17, 행복:7, 혐오:5, 슬픔:4, 놀람:4, 분노:3, 공포:2 | 주의:30, 보통:12 |
| late_night_text | 42 | 0.6905 | 0.5307 | 0.0439 | 5 | 2 | 중립:14, 혐오:9, 행복:6, 슬픔:4, 놀람:4, 분노:3, 공포:2 | 주의:24, 보통:12, 위험:6 |
| mixed_observation | 42 | 0.6905 | 0.4746 | 0.0487 | 1 | 2 | 중립:16, 행복:7, 슬픔:5, 혐오:5, 놀람:4, 분노:3, 공포:2 | 주의:29, 보통:12, 위험:1 |
| polite_update | 42 | 0.619 | 0.4552 | 0.0439 | 1 | 2 | 중립:19, 행복:5, 혐오:5, 슬픔:4, 놀람:4, 분노:3, 공포:2 | 주의:29, 보통:12, 위험:1 |
| train_note | 42 | 0.6905 | 0.5073 | 0.0439 | 5 | 2 | 중립:14, 혐오:7, 행복:6, 분노:5, 슬픔:4, 놀람:4, 공포:2 | 주의:26, 보통:12, 위험:4 |
| voice_memo | 42 | 0.6905 | 0.4708 | 0.0439 | 2 | 2 | 중립:17, 행복:6, 혐오:5, 슬픔:4, 분노:4, 놀람:4, 공포:2 | 주의:29, 보통:12, 위험:1 |
| weather_context | 42 | 0.6905 | 0.5307 | 0.0439 | 5 | 2 | 중립:14, 혐오:9, 행복:6, 슬픔:4, 놀람:4, 분노:3, 공포:2 | 주의:24, 보통:12, 위험:6 |
| work_log | 42 | 0.619 | 0.435 | 0.0439 | 0 | 2 | 중립:20, 행복:5, 슬픔:4, 놀람:4, 혐오:4, 분노:3, 공포:2 | 주의:30, 보통:12 |

## 감정별 요약

| 기대 감정 | n | emotion match | 평균 ds | 평균 dts | high ds + low dts | high dts | top 분포 | label 분포 |
|---|---:|---:|---:|---:|---:|---:|---|---|
| 공포 | 72 | 0.3333 | 0.4939 | 0.0056 | 2 | 0 | 중립:32, 공포:24, 놀람:12, 분노:2, 슬픔:2 | 주의:70, 위험:2 |
| 놀람 | 72 | 0.5 | 0.5012 | 0.0 | 0 | 0 | 놀람:36, 중립:36 | 주의:72 |
| 분노 | 72 | 0.5417 | 0.6143 | 0.0 | 20 | 0 | 분노:39, 혐오:17, 중립:16 | 주의:47, 위험:25 |
| 슬픔 | 72 | 0.6667 | 0.529 | 0.3075 | 0 | 24 | 슬픔:48, 중립:12, 행복:12 | 주의:72 |
| 중립 | 72 | 1.0 | 0.3757 | 0.0 | 0 | 0 | 중립:72 | 보통:72 |
| 행복 | 72 | 0.8333 | 0.3708 | 0.0 | 0 | 0 | 행복:60, 중립:12 | 보통:72 |
| 혐오 | 72 | 0.8611 | 0.5793 | 0.0 | 14 | 0 | 혐오:62, 중립:10 | 주의:62, 위험:10 |

## 스타일×감정 요약

| 스타일×감정 | n | match | 평균 ds | 평균 dts | high ds + low dts | high dts |
|---|---:|---:|---:|---:|---:|---:|
| body_signal_short|공포 | 6 | 0.3333 | 0.4827 | 0.0 | 0 | 0 |
| body_signal_short|놀람 | 6 | 0.5 | 0.4878 | 0.0 | 0 | 0 |
| body_signal_short|분노 | 6 | 0.5 | 0.5532 | 0.0 | 1 | 0 |
| body_signal_short|슬픔 | 6 | 0.6667 | 0.5635 | 0.3075 | 0 | 2 |
| body_signal_short|중립 | 6 | 1.0 | 0.3791 | 0.0 | 0 | 0 |
| body_signal_short|행복 | 6 | 0.8333 | 0.3772 | 0.0 | 0 | 0 |
| body_signal_short|혐오 | 6 | 0.8333 | 0.5705 | 0.0 | 1 | 0 |
| calendar_margin|공포 | 6 | 0.3333 | 0.4989 | 0.0 | 0 | 0 |
| calendar_margin|놀람 | 6 | 0.5 | 0.527 | 0.0 | 0 | 0 |
| calendar_margin|분노 | 6 | 0.5 | 0.756 | 0.0 | 3 | 0 |
| calendar_margin|슬픔 | 6 | 0.6667 | 0.5238 | 0.3075 | 0 | 2 |
| calendar_margin|중립 | 6 | 1.0 | 0.3722 | 0.0 | 0 | 0 |
| calendar_margin|행복 | 6 | 0.8333 | 0.3751 | 0.0 | 0 | 0 |
| calendar_margin|혐오 | 6 | 1.0 | 0.662 | 0.0 | 2 | 0 |
| direct_observation|공포 | 6 | 0.3333 | 0.4989 | 0.0 | 0 | 0 |
| direct_observation|놀람 | 6 | 0.5 | 0.527 | 0.0 | 0 | 0 |
| direct_observation|분노 | 6 | 0.5 | 0.756 | 0.0 | 3 | 0 |
| direct_observation|슬픔 | 6 | 0.6667 | 0.5238 | 0.3075 | 0 | 2 |
| direct_observation|중립 | 6 | 1.0 | 0.3722 | 0.0 | 0 | 0 |
| direct_observation|행복 | 6 | 0.8333 | 0.3751 | 0.0 | 0 | 0 |
| direct_observation|혐오 | 6 | 1.0 | 0.662 | 0.0 | 2 | 0 |
| errand_note|공포 | 6 | 0.3333 | 0.5903 | 0.0333 | 1 | 0 |
| errand_note|놀람 | 6 | 0.5 | 0.5276 | 0.0 | 0 | 0 |
| errand_note|분노 | 6 | 0.6667 | 0.6213 | 0.0 | 2 | 0 |
| errand_note|슬픔 | 6 | 0.6667 | 0.5272 | 0.3075 | 0 | 2 |
| errand_note|중립 | 6 | 1.0 | 0.377 | 0.0 | 0 | 0 |
| errand_note|행복 | 6 | 0.8333 | 0.3859 | 0.0 | 0 | 0 |
| errand_note|혐오 | 6 | 0.8333 | 0.5644 | 0.0 | 1 | 0 |
| friend_dm|공포 | 6 | 0.3333 | 0.4573 | 0.0 | 0 | 0 |
| friend_dm|놀람 | 6 | 0.5 | 0.5234 | 0.0 | 0 | 0 |
| friend_dm|분노 | 6 | 0.5 | 0.4553 | 0.0 | 0 | 0 |
| friend_dm|슬픔 | 6 | 0.6667 | 0.5599 | 0.3075 | 0 | 2 |
| friend_dm|중립 | 6 | 1.0 | 0.372 | 0.0 | 0 | 0 |
| friend_dm|행복 | 6 | 1.0 | 0.3686 | 0.0 | 0 | 0 |
| friend_dm|혐오 | 6 | 0.8333 | 0.5665 | 0.0 | 1 | 0 |
| late_night_text|공포 | 6 | 0.3333 | 0.4989 | 0.0 | 0 | 0 |
| late_night_text|놀람 | 6 | 0.5 | 0.527 | 0.0 | 0 | 0 |
| late_night_text|분노 | 6 | 0.5 | 0.756 | 0.0 | 3 | 0 |
| late_night_text|슬픔 | 6 | 0.6667 | 0.5238 | 0.3075 | 0 | 2 |
| late_night_text|중립 | 6 | 1.0 | 0.3722 | 0.0 | 0 | 0 |
| late_night_text|행복 | 6 | 0.8333 | 0.3751 | 0.0 | 0 | 0 |
| late_night_text|혐오 | 6 | 1.0 | 0.662 | 0.0 | 2 | 0 |
| mixed_observation|공포 | 6 | 0.3333 | 0.5535 | 0.0333 | 0 | 0 |
| mixed_observation|놀람 | 6 | 0.5 | 0.4667 | 0.0 | 0 | 0 |
| mixed_observation|분노 | 6 | 0.5 | 0.4839 | 0.0 | 0 | 0 |
| mixed_observation|슬픔 | 6 | 0.6667 | 0.5144 | 0.3075 | 0 | 2 |
| mixed_observation|중립 | 6 | 1.0 | 0.376 | 0.0 | 0 | 0 |
| mixed_observation|행복 | 6 | 1.0 | 0.3679 | 0.0 | 0 | 0 |
| mixed_observation|혐오 | 6 | 0.8333 | 0.5598 | 0.0 | 1 | 0 |
| polite_update|공포 | 6 | 0.3333 | 0.4577 | 0.0 | 0 | 0 |
| polite_update|놀람 | 6 | 0.5 | 0.4484 | 0.0 | 0 | 0 |
| polite_update|분노 | 6 | 0.5 | 0.5546 | 0.0 | 1 | 0 |
| polite_update|슬픔 | 6 | 0.6667 | 0.547 | 0.3075 | 0 | 2 |
| polite_update|중립 | 6 | 1.0 | 0.38 | 0.0 | 0 | 0 |
| polite_update|행복 | 6 | 0.6667 | 0.3434 | 0.0 | 0 | 0 |
| polite_update|혐오 | 6 | 0.6667 | 0.455 | 0.0 | 0 | 0 |
| train_note|공포 | 6 | 0.3333 | 0.506 | 0.0 | 1 | 0 |
| train_note|놀람 | 6 | 0.5 | 0.4837 | 0.0 | 0 | 0 |
| train_note|분노 | 6 | 0.6667 | 0.7045 | 0.0 | 3 | 0 |
| train_note|슬픔 | 6 | 0.6667 | 0.5345 | 0.3075 | 0 | 2 |
| train_note|중립 | 6 | 1.0 | 0.3791 | 0.0 | 0 | 0 |
| train_note|행복 | 6 | 0.8333 | 0.3843 | 0.0 | 0 | 0 |
| train_note|혐오 | 6 | 0.8333 | 0.5593 | 0.0 | 1 | 0 |
| voice_memo|공포 | 6 | 0.3333 | 0.4319 | 0.0 | 0 | 0 |
| voice_memo|놀람 | 6 | 0.5 | 0.5089 | 0.0 | 0 | 0 |
| voice_memo|분노 | 6 | 0.6667 | 0.5268 | 0.0 | 1 | 0 |
| voice_memo|슬픔 | 6 | 0.6667 | 0.4928 | 0.3075 | 0 | 2 |
| voice_memo|중립 | 6 | 1.0 | 0.3788 | 0.0 | 0 | 0 |
| voice_memo|행복 | 6 | 0.8333 | 0.3761 | 0.0 | 0 | 0 |
| voice_memo|혐오 | 6 | 0.8333 | 0.5801 | 0.0 | 1 | 0 |
| weather_context|공포 | 6 | 0.3333 | 0.4989 | 0.0 | 0 | 0 |
| weather_context|놀람 | 6 | 0.5 | 0.527 | 0.0 | 0 | 0 |
| weather_context|분노 | 6 | 0.5 | 0.756 | 0.0 | 3 | 0 |
| weather_context|슬픔 | 6 | 0.6667 | 0.5238 | 0.3075 | 0 | 2 |
| weather_context|중립 | 6 | 1.0 | 0.3722 | 0.0 | 0 | 0 |
| weather_context|행복 | 6 | 0.8333 | 0.3751 | 0.0 | 0 | 0 |
| weather_context|혐오 | 6 | 1.0 | 0.662 | 0.0 | 2 | 0 |
| work_log|공포 | 6 | 0.3333 | 0.4515 | 0.0 | 0 | 0 |
| work_log|놀람 | 6 | 0.5 | 0.4601 | 0.0 | 0 | 0 |
| work_log|분노 | 6 | 0.5 | 0.4475 | 0.0 | 0 | 0 |
| work_log|슬픔 | 6 | 0.6667 | 0.514 | 0.3075 | 0 | 2 |
| work_log|중립 | 6 | 1.0 | 0.3778 | 0.0 | 0 | 0 |
| work_log|행복 | 6 | 0.6667 | 0.3457 | 0.0 | 0 | 0 |
| work_log|혐오 | 6 | 0.6667 | 0.4481 | 0.0 | 0 | 0 |

## 상세 결과

- WB504_direct_observation_happy_01 [direct_observation / 행복] 아침에 만든 계란말이가 모양까지 잘 나와서 괜히 뿌듯했어 => top=행복, match=1, ds=0.3283, dts=0.0000, label=보통, crisis=False
- WB504_direct_observation_happy_02 [direct_observation / 행복] 길에서 좋아하던 노래가 들려서 발걸음이 조금 빨라졌어 => top=행복, match=1, ds=0.3450, dts=0.0000, label=보통, crisis=False
- WB504_direct_observation_happy_03 [direct_observation / 행복] 친구가 내 농담에 크게 웃어줘서 마음이 환해졌어 => top=행복, match=1, ds=0.5373, dts=0.0000, label=보통, crisis=False
- WB504_direct_observation_happy_04 [direct_observation / 행복] 오래 찾던 자료를 드디어 발견해서 속이 시원했어 => top=행복, match=1, ds=0.3326, dts=0.0000, label=보통, crisis=False
- WB504_direct_observation_happy_05 [direct_observation / 행복] 퇴근길 하늘색이 예뻐서 사진을 한 장 남겼어 => top=행복, match=1, ds=0.3288, dts=0.0000, label=보통, crisis=False
- WB504_direct_observation_happy_06 [direct_observation / 행복] 오늘은 계획한 만큼만 해도 충분하다는 생각이 들어서 편했어 => top=중립, match=0, ds=0.3787, dts=0.0000, label=보통, crisis=False
- WB504_direct_observation_sad_01 [direct_observation / 슬픔] 요즘은 좋아하던 일도 손이 안 가고 하루가 그냥 지나가 => top=슬픔, match=1, ds=0.5741, dts=0.7700, label=주의, crisis=False
- WB504_direct_observation_sad_02 [direct_observation / 슬픔] 며칠째 잠을 자도 개운하지 않고 몸이 무겁게 가라앉아 => top=슬픔, match=1, ds=0.5500, dts=0.6750, label=주의, crisis=False
- WB504_direct_observation_sad_03 [direct_observation / 슬픔] 내가 있어도 없어도 별 차이 없을 것 같다는 생각이 반복돼 => top=중립, match=0, ds=0.4384, dts=0.0000, label=주의, crisis=False
- WB504_direct_observation_sad_04 [direct_observation / 슬픔] 기대했던 모임이 취소돼서 집에 오니 마음이 푹 꺼졌어 => top=행복, match=0, ds=0.3450, dts=0.0000, label=주의, crisis=False
- WB504_direct_observation_sad_05 [direct_observation / 슬픔] 괜찮다고 말했는데 막상 혼자 있으니 서운함이 남았어 => top=슬픔, match=1, ds=0.6578, dts=0.2000, label=주의, crisis=False
- WB504_direct_observation_sad_06 [direct_observation / 슬픔] 비 오는 소리를 듣다 보니 괜히 옛날 생각이 나서 먹먹했어 => top=슬픔, match=1, ds=0.5775, dts=0.2000, label=주의, crisis=False
- WB504_direct_observation_anger_01 [direct_observation / 분노] 내가 맡은 일도 아닌데 급하다고 떠넘겨서 짜증이 확 났어 => top=분노, match=1, ds=0.5310, dts=0.0000, label=주의, crisis=False
- WB504_direct_observation_anger_02 [direct_observation / 분노] 분명히 약속한 기준을 자기 마음대로 바꿔서 화가 났어 => top=분노, match=1, ds=0.5566, dts=0.0000, label=주의, crisis=False
- WB504_direct_observation_anger_03 [direct_observation / 분노] 말을 끝내기도 전에 결론부터 내리는 태도가 너무 거슬렸어 => top=혐오, match=0, ds=0.9853, dts=0.0000, label=위험, crisis=False
- WB504_direct_observation_anger_04 [direct_observation / 분노] 내 시간을 당연한 것처럼 요구하는 메시지를 보고 열이 올랐어 => top=혐오, match=0, ds=0.9450, dts=0.0000, label=위험, crisis=False
- WB504_direct_observation_anger_05 [direct_observation / 분노] 실수는 같이 했는데 설명은 나만 하라는 분위기가 억울했어 => top=분노, match=1, ds=0.5302, dts=0.0000, label=위험, crisis=False
- WB504_direct_observation_anger_06 [direct_observation / 분노] 사과 대신 농담으로 넘기려는 모습이 계속 마음에 걸렸어 => top=혐오, match=0, ds=0.9879, dts=0.0000, label=위험, crisis=False
- WB504_direct_observation_fear_01 [direct_observation / 공포] 밤에 골목 불이 꺼져 있어서 지나가는 동안 어깨가 굳었어 => top=공포, match=1, ds=0.5500, dts=0.0000, label=주의, crisis=False
- WB504_direct_observation_fear_02 [direct_observation / 공포] 휴대폰에 모르는 결제 문자가 떠서 순간 손끝이 차가워졌어 => top=공포, match=1, ds=0.5350, dts=0.0000, label=주의, crisis=False
- WB504_direct_observation_fear_03 [direct_observation / 공포] 엘리베이터 문이 한참 안 열려서 숨을 작게 쉬고 있었어 => top=중립, match=0, ds=0.3800, dts=0.0000, label=주의, crisis=False
- WB504_direct_observation_fear_04 [direct_observation / 공포] 발표 자료가 갑자기 안 열려서 심장이 빠르게 뛰었어 => top=놀람, match=0, ds=0.5200, dts=0.0000, label=주의, crisis=False
- WB504_direct_observation_fear_05 [direct_observation / 공포] 새벽에 현관 센서등이 켜져서 한동안 움직이지 못했어 => top=중립, match=0, ds=0.6532, dts=0.0000, label=주의, crisis=False
- WB504_direct_observation_fear_06 [direct_observation / 공포] 병원 전화를 받기 전부터 입안이 바짝 말랐어 => top=중립, match=0, ds=0.3554, dts=0.0000, label=주의, crisis=False
- WB504_direct_observation_surprise_01 [direct_observation / 놀람] 조용하던 프린터가 갑자기 움직여서 몸이 움찔했어 => top=놀람, match=1, ds=0.5405, dts=0.0000, label=주의, crisis=False
- WB504_direct_observation_surprise_02 [direct_observation / 놀람] 닫힌 줄 알았던 창문이 덜컥 열려서 눈이 커졌어 => top=중립, match=0, ds=0.3800, dts=0.0000, label=주의, crisis=False
- WB504_direct_observation_surprise_03 [direct_observation / 놀람] 회의 시간이 앞당겨졌다는 알림을 보고 잠깐 멍해졌어 => top=놀람, match=1, ds=0.5397, dts=0.0000, label=주의, crisis=False
- WB504_direct_observation_surprise_04 [direct_observation / 놀람] 뒤에서 누가 내 이름을 불러서 바로 돌아봤어 => top=중립, match=0, ds=0.4332, dts=0.0000, label=주의, crisis=False
- WB504_direct_observation_surprise_05 [direct_observation / 놀람] 저장해둔 파일 이름이 바뀌어 있어서 순간 당황했어 => top=놀람, match=1, ds=0.5390, dts=0.0000, label=주의, crisis=False
- WB504_direct_observation_surprise_06 [direct_observation / 놀람] 택배 도착 사진이 예상과 달라서 한참 다시 봤어 => top=중립, match=0, ds=0.7295, dts=0.0000, label=주의, crisis=False
- WB504_direct_observation_disgust_01 [direct_observation / 혐오] 냉장고 구석 반찬통에서 시큼한 냄새가 올라와서 바로 닫았어 => top=혐오, match=1, ds=0.5000, dts=0.0000, label=주의, crisis=False
- WB504_direct_observation_disgust_02 [direct_observation / 혐오] 손잡이에 끈적한 자국이 묻어 있어서 만지기가 싫었어 => top=혐오, match=1, ds=0.5000, dts=0.0000, label=주의, crisis=False
- WB504_direct_observation_disgust_03 [direct_observation / 혐오] 젖은 걸레 냄새가 방 안에 남아서 계속 인상이 찌푸려졌어 => top=혐오, match=1, ds=0.5000, dts=0.0000, label=주의, crisis=False
- WB504_direct_observation_disgust_04 [direct_observation / 혐오] 약한 사람을 일부러 몰아붙이는 장면을 보고 속이 뒤틀렸어 => top=혐오, match=1, ds=0.9873, dts=0.0000, label=위험, crisis=False
- WB504_direct_observation_disgust_05 [direct_observation / 혐오] 컵 바닥에 남은 찌꺼기를 보고 마실 생각이 사라졌어 => top=혐오, match=1, ds=0.5000, dts=0.0000, label=주의, crisis=False
- WB504_direct_observation_disgust_06 [direct_observation / 혐오] 겉으로 친한 척하면서 뒤에서 험담하는 말을 들으니 역했어 => top=혐오, match=1, ds=0.9850, dts=0.0000, label=위험, crisis=False
- WB504_direct_observation_neutral_01 [direct_observation / 중립] 노트북 충전 케이블을 책상 오른쪽으로 옮겨뒀어 => top=중립, match=1, ds=0.3719, dts=0.0000, label=보통, crisis=False
- WB504_direct_observation_neutral_02 [direct_observation / 중립] 내일 가져갈 서류를 투명 파일에 넣어뒀어 => top=중립, match=1, ds=0.3800, dts=0.0000, label=보통, crisis=False
- WB504_direct_observation_neutral_03 [direct_observation / 중립] 점심 후보를 세 곳으로 줄여서 메모장에 적었어 => top=중립, match=1, ds=0.3654, dts=0.0000, label=보통, crisis=False
- WB504_direct_observation_neutral_04 [direct_observation / 중립] 세탁 완료 알림을 보고 빨래를 건조대에 널었어 => top=중립, match=1, ds=0.3800, dts=0.0000, label=보통, crisis=False
- WB504_direct_observation_neutral_05 [direct_observation / 중립] 달력에 이번 주 분리수거 날짜를 다시 표시했어 => top=중립, match=1, ds=0.3559, dts=0.0000, label=보통, crisis=False
- WB504_direct_observation_neutral_06 [direct_observation / 중립] 이어폰 케이스 배터리를 확인하고 가방에 넣었어 => top=중립, match=1, ds=0.3800, dts=0.0000, label=보통, crisis=False
- WB504_voice_memo_happy_01 [voice_memo / 행복] 음성으로 남기면, 아침에 만든 계란말이가 모양까지 잘 나와서 괜히 뿌듯했어. => top=행복, match=1, ds=0.3277, dts=0.0000, label=보통, crisis=False
- WB504_voice_memo_happy_02 [voice_memo / 행복] 음성으로 남기면, 길에서 좋아하던 노래가 들려서 발걸음이 조금 빨라졌어. => top=행복, match=1, ds=0.3414, dts=0.0000, label=보통, crisis=False
- WB504_voice_memo_happy_03 [voice_memo / 행복] 음성으로 남기면, 친구가 내 농담에 크게 웃어줘서 마음이 환해졌어. => top=행복, match=1, ds=0.5144, dts=0.0000, label=보통, crisis=False
- WB504_voice_memo_happy_04 [voice_memo / 행복] 음성으로 남기면, 오래 찾던 자료를 드디어 발견해서 속이 시원했어. => top=행복, match=1, ds=0.3450, dts=0.0000, label=보통, crisis=False
- WB504_voice_memo_happy_05 [voice_memo / 행복] 음성으로 남기면, 퇴근길 하늘색이 예뻐서 사진을 한 장 남겼어. => top=행복, match=1, ds=0.3275, dts=0.0000, label=보통, crisis=False
- WB504_voice_memo_happy_06 [voice_memo / 행복] 음성으로 남기면, 오늘은 계획한 만큼만 해도 충분하다는 생각이 들어서 편했어. => top=중립, match=0, ds=0.4004, dts=0.0000, label=보통, crisis=False
- WB504_voice_memo_sad_01 [voice_memo / 슬픔] 음성으로 남기면, 요즘은 좋아하던 일도 손이 안 가고 하루가 그냥 지나가. => top=슬픔, match=1, ds=0.5523, dts=0.7700, label=주의, crisis=False
- WB504_voice_memo_sad_02 [voice_memo / 슬픔] 음성으로 남기면, 며칠째 잠을 자도 개운하지 않고 몸이 무겁게 가라앉아. => top=슬픔, match=1, ds=0.5446, dts=0.6750, label=주의, crisis=False
- WB504_voice_memo_sad_03 [voice_memo / 슬픔] 음성으로 남기면, 내가 있어도 없어도 별 차이 없을 것 같다는 생각이 반복돼. => top=중립, match=0, ds=0.3994, dts=0.0000, label=주의, crisis=False
- WB504_voice_memo_sad_04 [voice_memo / 슬픔] 음성으로 남기면, 기대했던 모임이 취소돼서 집에 오니 마음이 푹 꺼졌어. => top=행복, match=0, ds=0.3450, dts=0.0000, label=주의, crisis=False
- WB504_voice_memo_sad_05 [voice_memo / 슬픔] 음성으로 남기면, 괜찮다고 말했는데 막상 혼자 있으니 서운함이 남았어. => top=슬픔, match=1, ds=0.5748, dts=0.2000, label=주의, crisis=False
- WB504_voice_memo_sad_06 [voice_memo / 슬픔] 음성으로 남기면, 비 오는 소리를 듣다 보니 괜히 옛날 생각이 나서 먹먹했어. => top=슬픔, match=1, ds=0.5409, dts=0.2000, label=주의, crisis=False
- WB504_voice_memo_anger_01 [voice_memo / 분노] 음성으로 남기면, 내가 맡은 일도 아닌데 급하다고 떠넘겨서 짜증이 확 났어. => top=분노, match=1, ds=0.5400, dts=0.0000, label=주의, crisis=False
- WB504_voice_memo_anger_02 [voice_memo / 분노] 음성으로 남기면, 분명히 약속한 기준을 자기 마음대로 바꿔서 화가 났어. => top=분노, match=1, ds=0.5411, dts=0.0000, label=주의, crisis=False
- WB504_voice_memo_anger_03 [voice_memo / 분노] 음성으로 남기면, 말을 끝내기도 전에 결론부터 내리는 태도가 너무 거슬렸어. => top=분노, match=1, ds=0.8289, dts=0.0000, label=위험, crisis=False
- WB504_voice_memo_anger_04 [voice_memo / 분노] 음성으로 남기면, 내 시간을 당연한 것처럼 요구하는 메시지를 보고 열이 올랐어. => top=중립, match=0, ds=0.3564, dts=0.0000, label=주의, crisis=False
- WB504_voice_memo_anger_05 [voice_memo / 분노] 음성으로 남기면, 실수는 같이 했는데 설명은 나만 하라는 분위기가 억울했어. => top=분노, match=1, ds=0.5365, dts=0.0000, label=주의, crisis=False
- WB504_voice_memo_anger_06 [voice_memo / 분노] 음성으로 남기면, 사과 대신 농담으로 넘기려는 모습이 계속 마음에 걸렸어. => top=중립, match=0, ds=0.3580, dts=0.0000, label=주의, crisis=False
- WB504_voice_memo_fear_01 [voice_memo / 공포] 음성으로 남기면, 밤에 골목 불이 꺼져 있어서 지나가는 동안 어깨가 굳었어. => top=공포, match=1, ds=0.4029, dts=0.0000, label=주의, crisis=False
- WB504_voice_memo_fear_02 [voice_memo / 공포] 음성으로 남기면, 휴대폰에 모르는 결제 문자가 떠서 순간 손끝이 차가워졌어. => top=공포, match=1, ds=0.5243, dts=0.0000, label=주의, crisis=False
- WB504_voice_memo_fear_03 [voice_memo / 공포] 음성으로 남기면, 엘리베이터 문이 한참 안 열려서 숨을 작게 쉬고 있었어. => top=중립, match=0, ds=0.3800, dts=0.0000, label=주의, crisis=False
- WB504_voice_memo_fear_04 [voice_memo / 공포] 음성으로 남기면, 발표 자료가 갑자기 안 열려서 심장이 빠르게 뛰었어. => top=놀람, match=0, ds=0.5200, dts=0.0000, label=주의, crisis=False
- WB504_voice_memo_fear_05 [voice_memo / 공포] 음성으로 남기면, 새벽에 현관 센서등이 켜져서 한동안 움직이지 못했어. => top=중립, match=0, ds=0.3970, dts=0.0000, label=주의, crisis=False
- WB504_voice_memo_fear_06 [voice_memo / 공포] 음성으로 남기면, 병원 전화를 받기 전부터 입안이 바짝 말랐어. => top=중립, match=0, ds=0.3675, dts=0.0000, label=주의, crisis=False
- WB504_voice_memo_surprise_01 [voice_memo / 놀람] 음성으로 남기면, 조용하던 프린터가 갑자기 움직여서 몸이 움찔했어. => top=놀람, match=1, ds=0.5315, dts=0.0000, label=주의, crisis=False
- WB504_voice_memo_surprise_02 [voice_memo / 놀람] 음성으로 남기면, 닫힌 줄 알았던 창문이 덜컥 열려서 눈이 커졌어. => top=중립, match=0, ds=0.3800, dts=0.0000, label=주의, crisis=False
- WB504_voice_memo_surprise_03 [voice_memo / 놀람] 음성으로 남기면, 회의 시간이 앞당겨졌다는 알림을 보고 잠깐 멍해졌어. => top=놀람, match=1, ds=0.5500, dts=0.0000, label=주의, crisis=False
- WB504_voice_memo_surprise_04 [voice_memo / 놀람] 음성으로 남기면, 뒤에서 누가 내 이름을 불러서 바로 돌아봤어. => top=중립, match=0, ds=0.5488, dts=0.0000, label=주의, crisis=False
- WB504_voice_memo_surprise_05 [voice_memo / 놀람] 음성으로 남기면, 저장해둔 파일 이름이 바뀌어 있어서 순간 당황했어. => top=놀람, match=1, ds=0.5389, dts=0.0000, label=주의, crisis=False
- WB504_voice_memo_surprise_06 [voice_memo / 놀람] 음성으로 남기면, 택배 도착 사진이 예상과 달라서 한참 다시 봤어. => top=중립, match=0, ds=0.5040, dts=0.0000, label=주의, crisis=False
- WB504_voice_memo_disgust_01 [voice_memo / 혐오] 음성으로 남기면, 냉장고 구석 반찬통에서 시큼한 냄새가 올라와서 바로 닫았어. => top=혐오, match=1, ds=0.5000, dts=0.0000, label=주의, crisis=False
- WB504_voice_memo_disgust_02 [voice_memo / 혐오] 음성으로 남기면, 손잡이에 끈적한 자국이 묻어 있어서 만지기가 싫었어. => top=혐오, match=1, ds=0.5000, dts=0.0000, label=주의, crisis=False
- WB504_voice_memo_disgust_03 [voice_memo / 혐오] 음성으로 남기면, 젖은 걸레 냄새가 방 안에 남아서 계속 인상이 찌푸려졌어. => top=혐오, match=1, ds=0.5000, dts=0.0000, label=주의, crisis=False
- WB504_voice_memo_disgust_04 [voice_memo / 혐오] 음성으로 남기면, 약한 사람을 일부러 몰아붙이는 장면을 보고 속이 뒤틀렸어. => top=혐오, match=1, ds=0.9102, dts=0.0000, label=주의, crisis=False
- WB504_voice_memo_disgust_05 [voice_memo / 혐오] 음성으로 남기면, 컵 바닥에 남은 찌꺼기를 보고 마실 생각이 사라졌어. => top=혐오, match=1, ds=0.5000, dts=0.0000, label=주의, crisis=False
- WB504_voice_memo_disgust_06 [voice_memo / 혐오] 음성으로 남기면, 겉으로 친한 척하면서 뒤에서 험담하는 말을 들으니 역했어. => top=중립, match=0, ds=0.5705, dts=0.0000, label=주의, crisis=False
- WB504_voice_memo_neutral_01 [voice_memo / 중립] 음성으로 남기면, 노트북 충전 케이블을 책상 오른쪽으로 옮겨뒀어. => top=중립, match=1, ds=0.3800, dts=0.0000, label=보통, crisis=False
- WB504_voice_memo_neutral_02 [voice_memo / 중립] 음성으로 남기면, 내일 가져갈 서류를 투명 파일에 넣어뒀어. => top=중립, match=1, ds=0.3800, dts=0.0000, label=보통, crisis=False
- WB504_voice_memo_neutral_03 [voice_memo / 중립] 음성으로 남기면, 점심 후보를 세 곳으로 줄여서 메모장에 적었어. => top=중립, match=1, ds=0.3800, dts=0.0000, label=보통, crisis=False
- WB504_voice_memo_neutral_04 [voice_memo / 중립] 음성으로 남기면, 세탁 완료 알림을 보고 빨래를 건조대에 널었어. => top=중립, match=1, ds=0.3800, dts=0.0000, label=보통, crisis=False
- WB504_voice_memo_neutral_05 [voice_memo / 중립] 음성으로 남기면, 달력에 이번 주 분리수거 날짜를 다시 표시했어. => top=중립, match=1, ds=0.3800, dts=0.0000, label=보통, crisis=False
- WB504_voice_memo_neutral_06 [voice_memo / 중립] 음성으로 남기면, 이어폰 케이스 배터리를 확인하고 가방에 넣었어. => top=중립, match=1, ds=0.3729, dts=0.0000, label=보통, crisis=False
- WB504_train_note_happy_01 [train_note / 행복] 지하철 안에서 짧게 적자면, 아침에 만든 계란말이가 모양까지 잘 나와서 괜히 뿌듯했어. => top=행복, match=1, ds=0.3282, dts=0.0000, label=보통, crisis=False
- WB504_train_note_happy_02 [train_note / 행복] 지하철 안에서 짧게 적자면, 길에서 좋아하던 노래가 들려서 발걸음이 조금 빨라졌어. => top=행복, match=1, ds=0.3450, dts=0.0000, label=보통, crisis=False
- WB504_train_note_happy_03 [train_note / 행복] 지하철 안에서 짧게 적자면, 친구가 내 농담에 크게 웃어줘서 마음이 환해졌어. => top=행복, match=1, ds=0.5315, dts=0.0000, label=보통, crisis=False
- WB504_train_note_happy_04 [train_note / 행복] 지하철 안에서 짧게 적자면, 오래 찾던 자료를 드디어 발견해서 속이 시원했어. => top=행복, match=1, ds=0.3210, dts=0.0000, label=보통, crisis=False
- WB504_train_note_happy_05 [train_note / 행복] 지하철 안에서 짧게 적자면, 퇴근길 하늘색이 예뻐서 사진을 한 장 남겼어. => top=행복, match=1, ds=0.3370, dts=0.0000, label=보통, crisis=False
- WB504_train_note_happy_06 [train_note / 행복] 지하철 안에서 짧게 적자면, 오늘은 계획한 만큼만 해도 충분하다는 생각이 들어서 편했어. => top=중립, match=0, ds=0.4430, dts=0.0000, label=보통, crisis=False
- WB504_train_note_sad_01 [train_note / 슬픔] 지하철 안에서 짧게 적자면, 요즘은 좋아하던 일도 손이 안 가고 하루가 그냥 지나가. => top=슬픔, match=1, ds=0.5717, dts=0.7700, label=주의, crisis=False
- WB504_train_note_sad_02 [train_note / 슬픔] 지하철 안에서 짧게 적자면, 며칠째 잠을 자도 개운하지 않고 몸이 무겁게 가라앉아. => top=슬픔, match=1, ds=0.5500, dts=0.6750, label=주의, crisis=False
- WB504_train_note_sad_03 [train_note / 슬픔] 지하철 안에서 짧게 적자면, 내가 있어도 없어도 별 차이 없을 것 같다는 생각이 반복돼. => top=중립, match=0, ds=0.4433, dts=0.0000, label=주의, crisis=False
- WB504_train_note_sad_04 [train_note / 슬픔] 지하철 안에서 짧게 적자면, 기대했던 모임이 취소돼서 집에 오니 마음이 푹 꺼졌어. => top=행복, match=0, ds=0.3450, dts=0.0000, label=주의, crisis=False
- WB504_train_note_sad_05 [train_note / 슬픔] 지하철 안에서 짧게 적자면, 괜찮다고 말했는데 막상 혼자 있으니 서운함이 남았어. => top=슬픔, match=1, ds=0.6801, dts=0.2000, label=주의, crisis=False
- WB504_train_note_sad_06 [train_note / 슬픔] 지하철 안에서 짧게 적자면, 비 오는 소리를 듣다 보니 괜히 옛날 생각이 나서 먹먹했어. => top=슬픔, match=1, ds=0.6170, dts=0.2000, label=주의, crisis=False
- WB504_train_note_anger_01 [train_note / 분노] 지하철 안에서 짧게 적자면, 내가 맡은 일도 아닌데 급하다고 떠넘겨서 짜증이 확 났어. => top=분노, match=1, ds=0.5412, dts=0.0000, label=주의, crisis=False
- WB504_train_note_anger_02 [train_note / 분노] 지하철 안에서 짧게 적자면, 분명히 약속한 기준을 자기 마음대로 바꿔서 화가 났어. => top=분노, match=1, ds=0.5610, dts=0.0000, label=주의, crisis=False
- WB504_train_note_anger_03 [train_note / 분노] 지하철 안에서 짧게 적자면, 말을 끝내기도 전에 결론부터 내리는 태도가 너무 거슬렸어. => top=혐오, match=0, ds=0.8962, dts=0.0000, label=위험, crisis=False
- WB504_train_note_anger_04 [train_note / 분노] 지하철 안에서 짧게 적자면, 내 시간을 당연한 것처럼 요구하는 메시지를 보고 열이 올랐어. => top=분노, match=1, ds=0.8237, dts=0.0000, label=위험, crisis=False
- WB504_train_note_anger_05 [train_note / 분노] 지하철 안에서 짧게 적자면, 실수는 같이 했는데 설명은 나만 하라는 분위기가 억울했어. => top=분노, match=1, ds=0.5357, dts=0.0000, label=위험, crisis=False
- WB504_train_note_anger_06 [train_note / 분노] 지하철 안에서 짧게 적자면, 사과 대신 농담으로 넘기려는 모습이 계속 마음에 걸렸어. => top=혐오, match=0, ds=0.8690, dts=0.0000, label=위험, crisis=False
- WB504_train_note_fear_01 [train_note / 공포] 지하철 안에서 짧게 적자면, 밤에 골목 불이 꺼져 있어서 지나가는 동안 어깨가 굳었어. => top=공포, match=1, ds=0.4169, dts=0.0000, label=주의, crisis=False
- WB504_train_note_fear_02 [train_note / 공포] 지하철 안에서 짧게 적자면, 휴대폰에 모르는 결제 문자가 떠서 순간 손끝이 차가워졌어. => top=공포, match=1, ds=0.4301, dts=0.0000, label=주의, crisis=False
- WB504_train_note_fear_03 [train_note / 공포] 지하철 안에서 짧게 적자면, 엘리베이터 문이 한참 안 열려서 숨을 작게 쉬고 있었어. => top=중립, match=0, ds=0.3800, dts=0.0000, label=주의, crisis=False
- WB504_train_note_fear_04 [train_note / 공포] 지하철 안에서 짧게 적자면, 발표 자료가 갑자기 안 열려서 심장이 빠르게 뛰었어. => top=놀람, match=0, ds=0.5200, dts=0.0000, label=주의, crisis=False
- WB504_train_note_fear_05 [train_note / 공포] 지하철 안에서 짧게 적자면, 새벽에 현관 센서등이 켜져서 한동안 움직이지 못했어. => top=중립, match=0, ds=0.4612, dts=0.0000, label=주의, crisis=False
- WB504_train_note_fear_06 [train_note / 공포] 지하철 안에서 짧게 적자면, 병원 전화를 받기 전부터 입안이 바짝 말랐어. => top=분노, match=0, ds=0.8278, dts=0.0000, label=주의, crisis=False
- WB504_train_note_surprise_01 [train_note / 놀람] 지하철 안에서 짧게 적자면, 조용하던 프린터가 갑자기 움직여서 몸이 움찔했어. => top=놀람, match=1, ds=0.5322, dts=0.0000, label=주의, crisis=False
- WB504_train_note_surprise_02 [train_note / 놀람] 지하철 안에서 짧게 적자면, 닫힌 줄 알았던 창문이 덜컥 열려서 눈이 커졌어. => top=중립, match=0, ds=0.3800, dts=0.0000, label=주의, crisis=False
- WB504_train_note_surprise_03 [train_note / 놀람] 지하철 안에서 짧게 적자면, 회의 시간이 앞당겨졌다는 알림을 보고 잠깐 멍해졌어. => top=놀람, match=1, ds=0.5500, dts=0.0000, label=주의, crisis=False
- WB504_train_note_surprise_04 [train_note / 놀람] 지하철 안에서 짧게 적자면, 뒤에서 누가 내 이름을 불러서 바로 돌아봤어. => top=중립, match=0, ds=0.4760, dts=0.0000, label=주의, crisis=False
- WB504_train_note_surprise_05 [train_note / 놀람] 지하철 안에서 짧게 적자면, 저장해둔 파일 이름이 바뀌어 있어서 순간 당황했어. => top=놀람, match=1, ds=0.5395, dts=0.0000, label=주의, crisis=False
- WB504_train_note_surprise_06 [train_note / 놀람] 지하철 안에서 짧게 적자면, 택배 도착 사진이 예상과 달라서 한참 다시 봤어. => top=중립, match=0, ds=0.4243, dts=0.0000, label=주의, crisis=False
- WB504_train_note_disgust_01 [train_note / 혐오] 지하철 안에서 짧게 적자면, 냉장고 구석 반찬통에서 시큼한 냄새가 올라와서 바로 닫았어. => top=혐오, match=1, ds=0.5000, dts=0.0000, label=주의, crisis=False
- WB504_train_note_disgust_02 [train_note / 혐오] 지하철 안에서 짧게 적자면, 손잡이에 끈적한 자국이 묻어 있어서 만지기가 싫었어. => top=혐오, match=1, ds=0.5000, dts=0.0000, label=주의, crisis=False
- WB504_train_note_disgust_03 [train_note / 혐오] 지하철 안에서 짧게 적자면, 젖은 걸레 냄새가 방 안에 남아서 계속 인상이 찌푸려졌어. => top=혐오, match=1, ds=0.5000, dts=0.0000, label=주의, crisis=False
- WB504_train_note_disgust_04 [train_note / 혐오] 지하철 안에서 짧게 적자면, 약한 사람을 일부러 몰아붙이는 장면을 보고 속이 뒤틀렸어. => top=혐오, match=1, ds=0.9233, dts=0.0000, label=주의, crisis=False
- WB504_train_note_disgust_05 [train_note / 혐오] 지하철 안에서 짧게 적자면, 컵 바닥에 남은 찌꺼기를 보고 마실 생각이 사라졌어. => top=혐오, match=1, ds=0.5000, dts=0.0000, label=주의, crisis=False
- WB504_train_note_disgust_06 [train_note / 혐오] 지하철 안에서 짧게 적자면, 겉으로 친한 척하면서 뒤에서 험담하는 말을 들으니 역했어. => top=중립, match=0, ds=0.4325, dts=0.0000, label=주의, crisis=False
- WB504_train_note_neutral_01 [train_note / 중립] 지하철 안에서 짧게 적자면, 노트북 충전 케이블을 책상 오른쪽으로 옮겨뒀어. => top=중립, match=1, ds=0.3800, dts=0.0000, label=보통, crisis=False
- WB504_train_note_neutral_02 [train_note / 중립] 지하철 안에서 짧게 적자면, 내일 가져갈 서류를 투명 파일에 넣어뒀어. => top=중립, match=1, ds=0.3800, dts=0.0000, label=보통, crisis=False
- WB504_train_note_neutral_03 [train_note / 중립] 지하철 안에서 짧게 적자면, 점심 후보를 세 곳으로 줄여서 메모장에 적었어. => top=중립, match=1, ds=0.3747, dts=0.0000, label=보통, crisis=False
- WB504_train_note_neutral_04 [train_note / 중립] 지하철 안에서 짧게 적자면, 세탁 완료 알림을 보고 빨래를 건조대에 널었어. => top=중립, match=1, ds=0.3800, dts=0.0000, label=보통, crisis=False
- WB504_train_note_neutral_05 [train_note / 중립] 지하철 안에서 짧게 적자면, 달력에 이번 주 분리수거 날짜를 다시 표시했어. => top=중립, match=1, ds=0.3800, dts=0.0000, label=보통, crisis=False
- WB504_train_note_neutral_06 [train_note / 중립] 지하철 안에서 짧게 적자면, 이어폰 케이스 배터리를 확인하고 가방에 넣었어. => top=중립, match=1, ds=0.3800, dts=0.0000, label=보통, crisis=False
- WB504_work_log_happy_01 [work_log / 행복] 오늘 기록 한 줄로는 아침에 만든 계란말이가 모양까지 잘 나와서 괜히 뿌듯했어. => top=행복, match=1, ds=0.3261, dts=0.0000, label=보통, crisis=False
- WB504_work_log_happy_02 [work_log / 행복] 오늘 기록 한 줄로는 길에서 좋아하던 노래가 들려서 발걸음이 조금 빨라졌어. => top=행복, match=1, ds=0.3450, dts=0.0000, label=보통, crisis=False
- WB504_work_log_happy_03 [work_log / 행복] 오늘 기록 한 줄로는 친구가 내 농담에 크게 웃어줘서 마음이 환해졌어. => top=중립, match=0, ds=0.3651, dts=0.0000, label=보통, crisis=False
- WB504_work_log_happy_04 [work_log / 행복] 오늘 기록 한 줄로는 오래 찾던 자료를 드디어 발견해서 속이 시원했어. => top=행복, match=1, ds=0.3260, dts=0.0000, label=보통, crisis=False
- WB504_work_log_happy_05 [work_log / 행복] 오늘 기록 한 줄로는 퇴근길 하늘색이 예뻐서 사진을 한 장 남겼어. => top=행복, match=1, ds=0.3321, dts=0.0000, label=보통, crisis=False
- WB504_work_log_happy_06 [work_log / 행복] 오늘 기록 한 줄로는 오늘은 계획한 만큼만 해도 충분하다는 생각이 들어서 편했어. => top=중립, match=0, ds=0.3800, dts=0.0000, label=보통, crisis=False
- WB504_work_log_sad_01 [work_log / 슬픔] 오늘 기록 한 줄로는 요즘은 좋아하던 일도 손이 안 가고 하루가 그냥 지나가. => top=슬픔, match=1, ds=0.5718, dts=0.7700, label=주의, crisis=False
- WB504_work_log_sad_02 [work_log / 슬픔] 오늘 기록 한 줄로는 며칠째 잠을 자도 개운하지 않고 몸이 무겁게 가라앉아. => top=슬픔, match=1, ds=0.5500, dts=0.6750, label=주의, crisis=False
- WB504_work_log_sad_03 [work_log / 슬픔] 오늘 기록 한 줄로는 내가 있어도 없어도 별 차이 없을 것 같다는 생각이 반복돼. => top=중립, match=0, ds=0.3800, dts=0.0000, label=주의, crisis=False
- WB504_work_log_sad_04 [work_log / 슬픔] 오늘 기록 한 줄로는 기대했던 모임이 취소돼서 집에 오니 마음이 푹 꺼졌어. => top=행복, match=0, ds=0.3450, dts=0.0000, label=주의, crisis=False
- WB504_work_log_sad_05 [work_log / 슬픔] 오늘 기록 한 줄로는 괜찮다고 말했는데 막상 혼자 있으니 서운함이 남았어. => top=슬픔, match=1, ds=0.6638, dts=0.2000, label=주의, crisis=False
- WB504_work_log_sad_06 [work_log / 슬픔] 오늘 기록 한 줄로는 비 오는 소리를 듣다 보니 괜히 옛날 생각이 나서 먹먹했어. => top=슬픔, match=1, ds=0.5733, dts=0.2000, label=주의, crisis=False
- WB504_work_log_anger_01 [work_log / 분노] 오늘 기록 한 줄로는 내가 맡은 일도 아닌데 급하다고 떠넘겨서 짜증이 확 났어. => top=분노, match=1, ds=0.5411, dts=0.0000, label=주의, crisis=False
- WB504_work_log_anger_02 [work_log / 분노] 오늘 기록 한 줄로는 분명히 약속한 기준을 자기 마음대로 바꿔서 화가 났어. => top=분노, match=1, ds=0.5289, dts=0.0000, label=주의, crisis=False
- WB504_work_log_anger_03 [work_log / 분노] 오늘 기록 한 줄로는 말을 끝내기도 전에 결론부터 내리는 태도가 너무 거슬렸어. => top=중립, match=0, ds=0.3537, dts=0.0000, label=주의, crisis=False
- WB504_work_log_anger_04 [work_log / 분노] 오늘 기록 한 줄로는 내 시간을 당연한 것처럼 요구하는 메시지를 보고 열이 올랐어. => top=중립, match=0, ds=0.3654, dts=0.0000, label=주의, crisis=False
- WB504_work_log_anger_05 [work_log / 분노] 오늘 기록 한 줄로는 실수는 같이 했는데 설명은 나만 하라는 분위기가 억울했어. => top=분노, match=1, ds=0.5378, dts=0.0000, label=주의, crisis=False
- WB504_work_log_anger_06 [work_log / 분노] 오늘 기록 한 줄로는 사과 대신 농담으로 넘기려는 모습이 계속 마음에 걸렸어. => top=중립, match=0, ds=0.3578, dts=0.0000, label=주의, crisis=False
- WB504_work_log_fear_01 [work_log / 공포] 오늘 기록 한 줄로는 밤에 골목 불이 꺼져 있어서 지나가는 동안 어깨가 굳었어. => top=공포, match=1, ds=0.5500, dts=0.0000, label=주의, crisis=False
- WB504_work_log_fear_02 [work_log / 공포] 오늘 기록 한 줄로는 휴대폰에 모르는 결제 문자가 떠서 순간 손끝이 차가워졌어. => top=공포, match=1, ds=0.5147, dts=0.0000, label=주의, crisis=False
- WB504_work_log_fear_03 [work_log / 공포] 오늘 기록 한 줄로는 엘리베이터 문이 한참 안 열려서 숨을 작게 쉬고 있었어. => top=중립, match=0, ds=0.3800, dts=0.0000, label=주의, crisis=False
- WB504_work_log_fear_04 [work_log / 공포] 오늘 기록 한 줄로는 발표 자료가 갑자기 안 열려서 심장이 빠르게 뛰었어. => top=놀람, match=0, ds=0.5200, dts=0.0000, label=주의, crisis=False
- WB504_work_log_fear_05 [work_log / 공포] 오늘 기록 한 줄로는 새벽에 현관 센서등이 켜져서 한동안 움직이지 못했어. => top=중립, match=0, ds=0.3800, dts=0.0000, label=주의, crisis=False
- WB504_work_log_fear_06 [work_log / 공포] 오늘 기록 한 줄로는 병원 전화를 받기 전부터 입안이 바짝 말랐어. => top=중립, match=0, ds=0.3645, dts=0.0000, label=주의, crisis=False
- WB504_work_log_surprise_01 [work_log / 놀람] 오늘 기록 한 줄로는 조용하던 프린터가 갑자기 움직여서 몸이 움찔했어. => top=놀람, match=1, ds=0.5386, dts=0.0000, label=주의, crisis=False
- WB504_work_log_surprise_02 [work_log / 놀람] 오늘 기록 한 줄로는 닫힌 줄 알았던 창문이 덜컥 열려서 눈이 커졌어. => top=중립, match=0, ds=0.3800, dts=0.0000, label=주의, crisis=False
- WB504_work_log_surprise_03 [work_log / 놀람] 오늘 기록 한 줄로는 회의 시간이 앞당겨졌다는 알림을 보고 잠깐 멍해졌어. => top=놀람, match=1, ds=0.5429, dts=0.0000, label=주의, crisis=False
- WB504_work_log_surprise_04 [work_log / 놀람] 오늘 기록 한 줄로는 뒤에서 누가 내 이름을 불러서 바로 돌아봤어. => top=중립, match=0, ds=0.3800, dts=0.0000, label=주의, crisis=False
- WB504_work_log_surprise_05 [work_log / 놀람] 오늘 기록 한 줄로는 저장해둔 파일 이름이 바뀌어 있어서 순간 당황했어. => top=놀람, match=1, ds=0.5392, dts=0.0000, label=주의, crisis=False
- WB504_work_log_surprise_06 [work_log / 놀람] 오늘 기록 한 줄로는 택배 도착 사진이 예상과 달라서 한참 다시 봤어. => top=중립, match=0, ds=0.3800, dts=0.0000, label=주의, crisis=False
- WB504_work_log_disgust_01 [work_log / 혐오] 오늘 기록 한 줄로는 냉장고 구석 반찬통에서 시큼한 냄새가 올라와서 바로 닫았어. => top=혐오, match=1, ds=0.4850, dts=0.0000, label=주의, crisis=False
- WB504_work_log_disgust_02 [work_log / 혐오] 오늘 기록 한 줄로는 손잡이에 끈적한 자국이 묻어 있어서 만지기가 싫었어. => top=혐오, match=1, ds=0.4850, dts=0.0000, label=주의, crisis=False
- WB504_work_log_disgust_03 [work_log / 혐오] 오늘 기록 한 줄로는 젖은 걸레 냄새가 방 안에 남아서 계속 인상이 찌푸려졌어. => top=혐오, match=1, ds=0.4850, dts=0.0000, label=주의, crisis=False
- WB504_work_log_disgust_04 [work_log / 혐오] 오늘 기록 한 줄로는 약한 사람을 일부러 몰아붙이는 장면을 보고 속이 뒤틀렸어. => top=중립, match=0, ds=0.3687, dts=0.0000, label=주의, crisis=False
- WB504_work_log_disgust_05 [work_log / 혐오] 오늘 기록 한 줄로는 컵 바닥에 남은 찌꺼기를 보고 마실 생각이 사라졌어. => top=혐오, match=1, ds=0.4850, dts=0.0000, label=주의, crisis=False
- WB504_work_log_disgust_06 [work_log / 혐오] 오늘 기록 한 줄로는 겉으로 친한 척하면서 뒤에서 험담하는 말을 들으니 역했어. => top=중립, match=0, ds=0.3800, dts=0.0000, label=주의, crisis=False
- WB504_work_log_neutral_01 [work_log / 중립] 오늘 기록 한 줄로는 노트북 충전 케이블을 책상 오른쪽으로 옮겨뒀어. => top=중립, match=1, ds=0.3800, dts=0.0000, label=보통, crisis=False
- WB504_work_log_neutral_02 [work_log / 중립] 오늘 기록 한 줄로는 내일 가져갈 서류를 투명 파일에 넣어뒀어. => top=중립, match=1, ds=0.3800, dts=0.0000, label=보통, crisis=False
- WB504_work_log_neutral_03 [work_log / 중립] 오늘 기록 한 줄로는 점심 후보를 세 곳으로 줄여서 메모장에 적었어. => top=중립, match=1, ds=0.3800, dts=0.0000, label=보통, crisis=False
- WB504_work_log_neutral_04 [work_log / 중립] 오늘 기록 한 줄로는 세탁 완료 알림을 보고 빨래를 건조대에 널었어. => top=중립, match=1, ds=0.3669, dts=0.0000, label=보통, crisis=False
- WB504_work_log_neutral_05 [work_log / 중립] 오늘 기록 한 줄로는 달력에 이번 주 분리수거 날짜를 다시 표시했어. => top=중립, match=1, ds=0.3800, dts=0.0000, label=보통, crisis=False
- WB504_work_log_neutral_06 [work_log / 중립] 오늘 기록 한 줄로는 이어폰 케이스 배터리를 확인하고 가방에 넣었어. => top=중립, match=1, ds=0.3800, dts=0.0000, label=보통, crisis=False
- WB504_late_night_text_happy_01 [late_night_text / 행복] 밤에 다시 떠올려보니 아침에 만든 계란말이가 모양까지 잘 나와서 괜히 뿌듯했어. => top=행복, match=1, ds=0.3283, dts=0.0000, label=보통, crisis=False
- WB504_late_night_text_happy_02 [late_night_text / 행복] 밤에 다시 떠올려보니 길에서 좋아하던 노래가 들려서 발걸음이 조금 빨라졌어. => top=행복, match=1, ds=0.3450, dts=0.0000, label=보통, crisis=False
- WB504_late_night_text_happy_03 [late_night_text / 행복] 밤에 다시 떠올려보니 친구가 내 농담에 크게 웃어줘서 마음이 환해졌어. => top=행복, match=1, ds=0.5373, dts=0.0000, label=보통, crisis=False
- WB504_late_night_text_happy_04 [late_night_text / 행복] 밤에 다시 떠올려보니 오래 찾던 자료를 드디어 발견해서 속이 시원했어. => top=행복, match=1, ds=0.3326, dts=0.0000, label=보통, crisis=False
- WB504_late_night_text_happy_05 [late_night_text / 행복] 밤에 다시 떠올려보니 퇴근길 하늘색이 예뻐서 사진을 한 장 남겼어. => top=행복, match=1, ds=0.3288, dts=0.0000, label=보통, crisis=False
- WB504_late_night_text_happy_06 [late_night_text / 행복] 밤에 다시 떠올려보니 오늘은 계획한 만큼만 해도 충분하다는 생각이 들어서 편했어. => top=중립, match=0, ds=0.3787, dts=0.0000, label=보통, crisis=False
- WB504_late_night_text_sad_01 [late_night_text / 슬픔] 밤에 다시 떠올려보니 요즘은 좋아하던 일도 손이 안 가고 하루가 그냥 지나가. => top=슬픔, match=1, ds=0.5741, dts=0.7700, label=주의, crisis=False
- WB504_late_night_text_sad_02 [late_night_text / 슬픔] 밤에 다시 떠올려보니 며칠째 잠을 자도 개운하지 않고 몸이 무겁게 가라앉아. => top=슬픔, match=1, ds=0.5500, dts=0.6750, label=주의, crisis=False
- WB504_late_night_text_sad_03 [late_night_text / 슬픔] 밤에 다시 떠올려보니 내가 있어도 없어도 별 차이 없을 것 같다는 생각이 반복돼. => top=중립, match=0, ds=0.4384, dts=0.0000, label=주의, crisis=False
- WB504_late_night_text_sad_04 [late_night_text / 슬픔] 밤에 다시 떠올려보니 기대했던 모임이 취소돼서 집에 오니 마음이 푹 꺼졌어. => top=행복, match=0, ds=0.3450, dts=0.0000, label=주의, crisis=False
- WB504_late_night_text_sad_05 [late_night_text / 슬픔] 밤에 다시 떠올려보니 괜찮다고 말했는데 막상 혼자 있으니 서운함이 남았어. => top=슬픔, match=1, ds=0.6578, dts=0.2000, label=주의, crisis=False
- WB504_late_night_text_sad_06 [late_night_text / 슬픔] 밤에 다시 떠올려보니 비 오는 소리를 듣다 보니 괜히 옛날 생각이 나서 먹먹했어. => top=슬픔, match=1, ds=0.5775, dts=0.2000, label=주의, crisis=False
- WB504_late_night_text_anger_01 [late_night_text / 분노] 밤에 다시 떠올려보니 내가 맡은 일도 아닌데 급하다고 떠넘겨서 짜증이 확 났어. => top=분노, match=1, ds=0.5310, dts=0.0000, label=주의, crisis=False
- WB504_late_night_text_anger_02 [late_night_text / 분노] 밤에 다시 떠올려보니 분명히 약속한 기준을 자기 마음대로 바꿔서 화가 났어. => top=분노, match=1, ds=0.5566, dts=0.0000, label=주의, crisis=False
- WB504_late_night_text_anger_03 [late_night_text / 분노] 밤에 다시 떠올려보니 말을 끝내기도 전에 결론부터 내리는 태도가 너무 거슬렸어. => top=혐오, match=0, ds=0.9853, dts=0.0000, label=위험, crisis=False
- WB504_late_night_text_anger_04 [late_night_text / 분노] 밤에 다시 떠올려보니 내 시간을 당연한 것처럼 요구하는 메시지를 보고 열이 올랐어. => top=혐오, match=0, ds=0.9450, dts=0.0000, label=위험, crisis=False
- WB504_late_night_text_anger_05 [late_night_text / 분노] 밤에 다시 떠올려보니 실수는 같이 했는데 설명은 나만 하라는 분위기가 억울했어. => top=분노, match=1, ds=0.5302, dts=0.0000, label=위험, crisis=False
- WB504_late_night_text_anger_06 [late_night_text / 분노] 밤에 다시 떠올려보니 사과 대신 농담으로 넘기려는 모습이 계속 마음에 걸렸어. => top=혐오, match=0, ds=0.9879, dts=0.0000, label=위험, crisis=False
- WB504_late_night_text_fear_01 [late_night_text / 공포] 밤에 다시 떠올려보니 밤에 골목 불이 꺼져 있어서 지나가는 동안 어깨가 굳었어. => top=공포, match=1, ds=0.5500, dts=0.0000, label=주의, crisis=False
- WB504_late_night_text_fear_02 [late_night_text / 공포] 밤에 다시 떠올려보니 휴대폰에 모르는 결제 문자가 떠서 순간 손끝이 차가워졌어. => top=공포, match=1, ds=0.5350, dts=0.0000, label=주의, crisis=False
- WB504_late_night_text_fear_03 [late_night_text / 공포] 밤에 다시 떠올려보니 엘리베이터 문이 한참 안 열려서 숨을 작게 쉬고 있었어. => top=중립, match=0, ds=0.3800, dts=0.0000, label=주의, crisis=False
- WB504_late_night_text_fear_04 [late_night_text / 공포] 밤에 다시 떠올려보니 발표 자료가 갑자기 안 열려서 심장이 빠르게 뛰었어. => top=놀람, match=0, ds=0.5200, dts=0.0000, label=주의, crisis=False
- WB504_late_night_text_fear_05 [late_night_text / 공포] 밤에 다시 떠올려보니 새벽에 현관 센서등이 켜져서 한동안 움직이지 못했어. => top=중립, match=0, ds=0.6532, dts=0.0000, label=주의, crisis=False
- WB504_late_night_text_fear_06 [late_night_text / 공포] 밤에 다시 떠올려보니 병원 전화를 받기 전부터 입안이 바짝 말랐어. => top=중립, match=0, ds=0.3554, dts=0.0000, label=주의, crisis=False
- WB504_late_night_text_surprise_01 [late_night_text / 놀람] 밤에 다시 떠올려보니 조용하던 프린터가 갑자기 움직여서 몸이 움찔했어. => top=놀람, match=1, ds=0.5405, dts=0.0000, label=주의, crisis=False
- WB504_late_night_text_surprise_02 [late_night_text / 놀람] 밤에 다시 떠올려보니 닫힌 줄 알았던 창문이 덜컥 열려서 눈이 커졌어. => top=중립, match=0, ds=0.3800, dts=0.0000, label=주의, crisis=False
- WB504_late_night_text_surprise_03 [late_night_text / 놀람] 밤에 다시 떠올려보니 회의 시간이 앞당겨졌다는 알림을 보고 잠깐 멍해졌어. => top=놀람, match=1, ds=0.5397, dts=0.0000, label=주의, crisis=False
- WB504_late_night_text_surprise_04 [late_night_text / 놀람] 밤에 다시 떠올려보니 뒤에서 누가 내 이름을 불러서 바로 돌아봤어. => top=중립, match=0, ds=0.4332, dts=0.0000, label=주의, crisis=False
- WB504_late_night_text_surprise_05 [late_night_text / 놀람] 밤에 다시 떠올려보니 저장해둔 파일 이름이 바뀌어 있어서 순간 당황했어. => top=놀람, match=1, ds=0.5390, dts=0.0000, label=주의, crisis=False
- WB504_late_night_text_surprise_06 [late_night_text / 놀람] 밤에 다시 떠올려보니 택배 도착 사진이 예상과 달라서 한참 다시 봤어. => top=중립, match=0, ds=0.7295, dts=0.0000, label=주의, crisis=False
- WB504_late_night_text_disgust_01 [late_night_text / 혐오] 밤에 다시 떠올려보니 냉장고 구석 반찬통에서 시큼한 냄새가 올라와서 바로 닫았어. => top=혐오, match=1, ds=0.5000, dts=0.0000, label=주의, crisis=False
- WB504_late_night_text_disgust_02 [late_night_text / 혐오] 밤에 다시 떠올려보니 손잡이에 끈적한 자국이 묻어 있어서 만지기가 싫었어. => top=혐오, match=1, ds=0.5000, dts=0.0000, label=주의, crisis=False
- WB504_late_night_text_disgust_03 [late_night_text / 혐오] 밤에 다시 떠올려보니 젖은 걸레 냄새가 방 안에 남아서 계속 인상이 찌푸려졌어. => top=혐오, match=1, ds=0.5000, dts=0.0000, label=주의, crisis=False
- WB504_late_night_text_disgust_04 [late_night_text / 혐오] 밤에 다시 떠올려보니 약한 사람을 일부러 몰아붙이는 장면을 보고 속이 뒤틀렸어. => top=혐오, match=1, ds=0.9873, dts=0.0000, label=위험, crisis=False
- WB504_late_night_text_disgust_05 [late_night_text / 혐오] 밤에 다시 떠올려보니 컵 바닥에 남은 찌꺼기를 보고 마실 생각이 사라졌어. => top=혐오, match=1, ds=0.5000, dts=0.0000, label=주의, crisis=False
- WB504_late_night_text_disgust_06 [late_night_text / 혐오] 밤에 다시 떠올려보니 겉으로 친한 척하면서 뒤에서 험담하는 말을 들으니 역했어. => top=혐오, match=1, ds=0.9850, dts=0.0000, label=위험, crisis=False
- WB504_late_night_text_neutral_01 [late_night_text / 중립] 밤에 다시 떠올려보니 노트북 충전 케이블을 책상 오른쪽으로 옮겨뒀어. => top=중립, match=1, ds=0.3719, dts=0.0000, label=보통, crisis=False
- WB504_late_night_text_neutral_02 [late_night_text / 중립] 밤에 다시 떠올려보니 내일 가져갈 서류를 투명 파일에 넣어뒀어. => top=중립, match=1, ds=0.3800, dts=0.0000, label=보통, crisis=False
- WB504_late_night_text_neutral_03 [late_night_text / 중립] 밤에 다시 떠올려보니 점심 후보를 세 곳으로 줄여서 메모장에 적었어. => top=중립, match=1, ds=0.3654, dts=0.0000, label=보통, crisis=False
- WB504_late_night_text_neutral_04 [late_night_text / 중립] 밤에 다시 떠올려보니 세탁 완료 알림을 보고 빨래를 건조대에 널었어. => top=중립, match=1, ds=0.3800, dts=0.0000, label=보통, crisis=False
- WB504_late_night_text_neutral_05 [late_night_text / 중립] 밤에 다시 떠올려보니 달력에 이번 주 분리수거 날짜를 다시 표시했어. => top=중립, match=1, ds=0.3559, dts=0.0000, label=보통, crisis=False
- WB504_late_night_text_neutral_06 [late_night_text / 중립] 밤에 다시 떠올려보니 이어폰 케이스 배터리를 확인하고 가방에 넣었어. => top=중립, match=1, ds=0.3800, dts=0.0000, label=보통, crisis=False
- WB504_calendar_margin_happy_01 [calendar_margin / 행복] 캘린더 여백에 쓰면 "아침에 만든 계란말이가 모양까지 잘 나와서 괜히 뿌듯했어" 정도야. => top=행복, match=1, ds=0.3283, dts=0.0000, label=보통, crisis=False
- WB504_calendar_margin_happy_02 [calendar_margin / 행복] 캘린더 여백에 쓰면 "길에서 좋아하던 노래가 들려서 발걸음이 조금 빨라졌어" 정도야. => top=행복, match=1, ds=0.3450, dts=0.0000, label=보통, crisis=False
- WB504_calendar_margin_happy_03 [calendar_margin / 행복] 캘린더 여백에 쓰면 "친구가 내 농담에 크게 웃어줘서 마음이 환해졌어" 정도야. => top=행복, match=1, ds=0.5373, dts=0.0000, label=보통, crisis=False
- WB504_calendar_margin_happy_04 [calendar_margin / 행복] 캘린더 여백에 쓰면 "오래 찾던 자료를 드디어 발견해서 속이 시원했어" 정도야. => top=행복, match=1, ds=0.3326, dts=0.0000, label=보통, crisis=False
- WB504_calendar_margin_happy_05 [calendar_margin / 행복] 캘린더 여백에 쓰면 "퇴근길 하늘색이 예뻐서 사진을 한 장 남겼어" 정도야. => top=행복, match=1, ds=0.3288, dts=0.0000, label=보통, crisis=False
- WB504_calendar_margin_happy_06 [calendar_margin / 행복] 캘린더 여백에 쓰면 "오늘은 계획한 만큼만 해도 충분하다는 생각이 들어서 편했어" 정도야. => top=중립, match=0, ds=0.3787, dts=0.0000, label=보통, crisis=False
- WB504_calendar_margin_sad_01 [calendar_margin / 슬픔] 캘린더 여백에 쓰면 "요즘은 좋아하던 일도 손이 안 가고 하루가 그냥 지나가" 정도야. => top=슬픔, match=1, ds=0.5741, dts=0.7700, label=주의, crisis=False
- WB504_calendar_margin_sad_02 [calendar_margin / 슬픔] 캘린더 여백에 쓰면 "며칠째 잠을 자도 개운하지 않고 몸이 무겁게 가라앉아" 정도야. => top=슬픔, match=1, ds=0.5500, dts=0.6750, label=주의, crisis=False
- WB504_calendar_margin_sad_03 [calendar_margin / 슬픔] 캘린더 여백에 쓰면 "내가 있어도 없어도 별 차이 없을 것 같다는 생각이 반복돼" 정도야. => top=중립, match=0, ds=0.4384, dts=0.0000, label=주의, crisis=False
- WB504_calendar_margin_sad_04 [calendar_margin / 슬픔] 캘린더 여백에 쓰면 "기대했던 모임이 취소돼서 집에 오니 마음이 푹 꺼졌어" 정도야. => top=행복, match=0, ds=0.3450, dts=0.0000, label=주의, crisis=False
- WB504_calendar_margin_sad_05 [calendar_margin / 슬픔] 캘린더 여백에 쓰면 "괜찮다고 말했는데 막상 혼자 있으니 서운함이 남았어" 정도야. => top=슬픔, match=1, ds=0.6578, dts=0.2000, label=주의, crisis=False
- WB504_calendar_margin_sad_06 [calendar_margin / 슬픔] 캘린더 여백에 쓰면 "비 오는 소리를 듣다 보니 괜히 옛날 생각이 나서 먹먹했어" 정도야. => top=슬픔, match=1, ds=0.5775, dts=0.2000, label=주의, crisis=False
- WB504_calendar_margin_anger_01 [calendar_margin / 분노] 캘린더 여백에 쓰면 "내가 맡은 일도 아닌데 급하다고 떠넘겨서 짜증이 확 났어" 정도야. => top=분노, match=1, ds=0.5310, dts=0.0000, label=주의, crisis=False
- WB504_calendar_margin_anger_02 [calendar_margin / 분노] 캘린더 여백에 쓰면 "분명히 약속한 기준을 자기 마음대로 바꿔서 화가 났어" 정도야. => top=분노, match=1, ds=0.5566, dts=0.0000, label=주의, crisis=False
- WB504_calendar_margin_anger_03 [calendar_margin / 분노] 캘린더 여백에 쓰면 "말을 끝내기도 전에 결론부터 내리는 태도가 너무 거슬렸어" 정도야. => top=혐오, match=0, ds=0.9853, dts=0.0000, label=위험, crisis=False
- WB504_calendar_margin_anger_04 [calendar_margin / 분노] 캘린더 여백에 쓰면 "내 시간을 당연한 것처럼 요구하는 메시지를 보고 열이 올랐어" 정도야. => top=혐오, match=0, ds=0.9450, dts=0.0000, label=위험, crisis=False
- WB504_calendar_margin_anger_05 [calendar_margin / 분노] 캘린더 여백에 쓰면 "실수는 같이 했는데 설명은 나만 하라는 분위기가 억울했어" 정도야. => top=분노, match=1, ds=0.5302, dts=0.0000, label=위험, crisis=False
- WB504_calendar_margin_anger_06 [calendar_margin / 분노] 캘린더 여백에 쓰면 "사과 대신 농담으로 넘기려는 모습이 계속 마음에 걸렸어" 정도야. => top=혐오, match=0, ds=0.9879, dts=0.0000, label=위험, crisis=False
- WB504_calendar_margin_fear_01 [calendar_margin / 공포] 캘린더 여백에 쓰면 "밤에 골목 불이 꺼져 있어서 지나가는 동안 어깨가 굳었어" 정도야. => top=공포, match=1, ds=0.5500, dts=0.0000, label=주의, crisis=False
- WB504_calendar_margin_fear_02 [calendar_margin / 공포] 캘린더 여백에 쓰면 "휴대폰에 모르는 결제 문자가 떠서 순간 손끝이 차가워졌어" 정도야. => top=공포, match=1, ds=0.5350, dts=0.0000, label=주의, crisis=False
- WB504_calendar_margin_fear_03 [calendar_margin / 공포] 캘린더 여백에 쓰면 "엘리베이터 문이 한참 안 열려서 숨을 작게 쉬고 있었어" 정도야. => top=중립, match=0, ds=0.3800, dts=0.0000, label=주의, crisis=False
- WB504_calendar_margin_fear_04 [calendar_margin / 공포] 캘린더 여백에 쓰면 "발표 자료가 갑자기 안 열려서 심장이 빠르게 뛰었어" 정도야. => top=놀람, match=0, ds=0.5200, dts=0.0000, label=주의, crisis=False
- WB504_calendar_margin_fear_05 [calendar_margin / 공포] 캘린더 여백에 쓰면 "새벽에 현관 센서등이 켜져서 한동안 움직이지 못했어" 정도야. => top=중립, match=0, ds=0.6532, dts=0.0000, label=주의, crisis=False
- WB504_calendar_margin_fear_06 [calendar_margin / 공포] 캘린더 여백에 쓰면 "병원 전화를 받기 전부터 입안이 바짝 말랐어" 정도야. => top=중립, match=0, ds=0.3554, dts=0.0000, label=주의, crisis=False
- WB504_calendar_margin_surprise_01 [calendar_margin / 놀람] 캘린더 여백에 쓰면 "조용하던 프린터가 갑자기 움직여서 몸이 움찔했어" 정도야. => top=놀람, match=1, ds=0.5405, dts=0.0000, label=주의, crisis=False
- WB504_calendar_margin_surprise_02 [calendar_margin / 놀람] 캘린더 여백에 쓰면 "닫힌 줄 알았던 창문이 덜컥 열려서 눈이 커졌어" 정도야. => top=중립, match=0, ds=0.3800, dts=0.0000, label=주의, crisis=False
- WB504_calendar_margin_surprise_03 [calendar_margin / 놀람] 캘린더 여백에 쓰면 "회의 시간이 앞당겨졌다는 알림을 보고 잠깐 멍해졌어" 정도야. => top=놀람, match=1, ds=0.5397, dts=0.0000, label=주의, crisis=False
- WB504_calendar_margin_surprise_04 [calendar_margin / 놀람] 캘린더 여백에 쓰면 "뒤에서 누가 내 이름을 불러서 바로 돌아봤어" 정도야. => top=중립, match=0, ds=0.4332, dts=0.0000, label=주의, crisis=False
- WB504_calendar_margin_surprise_05 [calendar_margin / 놀람] 캘린더 여백에 쓰면 "저장해둔 파일 이름이 바뀌어 있어서 순간 당황했어" 정도야. => top=놀람, match=1, ds=0.5390, dts=0.0000, label=주의, crisis=False
- WB504_calendar_margin_surprise_06 [calendar_margin / 놀람] 캘린더 여백에 쓰면 "택배 도착 사진이 예상과 달라서 한참 다시 봤어" 정도야. => top=중립, match=0, ds=0.7295, dts=0.0000, label=주의, crisis=False
- WB504_calendar_margin_disgust_01 [calendar_margin / 혐오] 캘린더 여백에 쓰면 "냉장고 구석 반찬통에서 시큼한 냄새가 올라와서 바로 닫았어" 정도야. => top=혐오, match=1, ds=0.5000, dts=0.0000, label=주의, crisis=False
- WB504_calendar_margin_disgust_02 [calendar_margin / 혐오] 캘린더 여백에 쓰면 "손잡이에 끈적한 자국이 묻어 있어서 만지기가 싫었어" 정도야. => top=혐오, match=1, ds=0.5000, dts=0.0000, label=주의, crisis=False
- WB504_calendar_margin_disgust_03 [calendar_margin / 혐오] 캘린더 여백에 쓰면 "젖은 걸레 냄새가 방 안에 남아서 계속 인상이 찌푸려졌어" 정도야. => top=혐오, match=1, ds=0.5000, dts=0.0000, label=주의, crisis=False
- WB504_calendar_margin_disgust_04 [calendar_margin / 혐오] 캘린더 여백에 쓰면 "약한 사람을 일부러 몰아붙이는 장면을 보고 속이 뒤틀렸어" 정도야. => top=혐오, match=1, ds=0.9873, dts=0.0000, label=위험, crisis=False
- WB504_calendar_margin_disgust_05 [calendar_margin / 혐오] 캘린더 여백에 쓰면 "컵 바닥에 남은 찌꺼기를 보고 마실 생각이 사라졌어" 정도야. => top=혐오, match=1, ds=0.5000, dts=0.0000, label=주의, crisis=False
- WB504_calendar_margin_disgust_06 [calendar_margin / 혐오] 캘린더 여백에 쓰면 "겉으로 친한 척하면서 뒤에서 험담하는 말을 들으니 역했어" 정도야. => top=혐오, match=1, ds=0.9850, dts=0.0000, label=위험, crisis=False
- WB504_calendar_margin_neutral_01 [calendar_margin / 중립] 캘린더 여백에 쓰면 "노트북 충전 케이블을 책상 오른쪽으로 옮겨뒀어" 정도야. => top=중립, match=1, ds=0.3719, dts=0.0000, label=보통, crisis=False
- WB504_calendar_margin_neutral_02 [calendar_margin / 중립] 캘린더 여백에 쓰면 "내일 가져갈 서류를 투명 파일에 넣어뒀어" 정도야. => top=중립, match=1, ds=0.3800, dts=0.0000, label=보통, crisis=False
- WB504_calendar_margin_neutral_03 [calendar_margin / 중립] 캘린더 여백에 쓰면 "점심 후보를 세 곳으로 줄여서 메모장에 적었어" 정도야. => top=중립, match=1, ds=0.3654, dts=0.0000, label=보통, crisis=False
- WB504_calendar_margin_neutral_04 [calendar_margin / 중립] 캘린더 여백에 쓰면 "세탁 완료 알림을 보고 빨래를 건조대에 널었어" 정도야. => top=중립, match=1, ds=0.3800, dts=0.0000, label=보통, crisis=False
- WB504_calendar_margin_neutral_05 [calendar_margin / 중립] 캘린더 여백에 쓰면 "달력에 이번 주 분리수거 날짜를 다시 표시했어" 정도야. => top=중립, match=1, ds=0.3559, dts=0.0000, label=보통, crisis=False
- WB504_calendar_margin_neutral_06 [calendar_margin / 중립] 캘린더 여백에 쓰면 "이어폰 케이스 배터리를 확인하고 가방에 넣었어" 정도야. => top=중립, match=1, ds=0.3800, dts=0.0000, label=보통, crisis=False
- WB504_mixed_observation_happy_01 [mixed_observation / 행복] 겉으로는 평소처럼 움직였는데, 안쪽 느낌은 아침에 만든 계란말이가 모양까지 잘 나와서 괜히 뿌듯했어. => top=행복, match=1, ds=0.3278, dts=0.0000, label=보통, crisis=False
- WB504_mixed_observation_happy_02 [mixed_observation / 행복] 겉으로는 평소처럼 움직였는데, 안쪽 느낌은 길에서 좋아하던 노래가 들려서 발걸음이 조금 빨라졌어. => top=행복, match=1, ds=0.3450, dts=0.0000, label=보통, crisis=False
- WB504_mixed_observation_happy_03 [mixed_observation / 행복] 겉으로는 평소처럼 움직였는데, 안쪽 느낌은 친구가 내 농담에 크게 웃어줘서 마음이 환해졌어. => top=행복, match=1, ds=0.5328, dts=0.0000, label=보통, crisis=False
- WB504_mixed_observation_happy_04 [mixed_observation / 행복] 겉으로는 평소처럼 움직였는데, 안쪽 느낌은 오래 찾던 자료를 드디어 발견해서 속이 시원했어. => top=행복, match=1, ds=0.3297, dts=0.0000, label=보통, crisis=False
- WB504_mixed_observation_happy_05 [mixed_observation / 행복] 겉으로는 평소처럼 움직였는데, 안쪽 느낌은 퇴근길 하늘색이 예뻐서 사진을 한 장 남겼어. => top=행복, match=1, ds=0.3274, dts=0.0000, label=보통, crisis=False
- WB504_mixed_observation_happy_06 [mixed_observation / 행복] 겉으로는 평소처럼 움직였는데, 안쪽 느낌은 오늘은 계획한 만큼만 해도 충분하다는 생각이 들어서 편했어. => top=행복, match=1, ds=0.3450, dts=0.0000, label=보통, crisis=False
- WB504_mixed_observation_sad_01 [mixed_observation / 슬픔] 겉으로는 평소처럼 움직였는데, 안쪽 느낌은 요즘은 좋아하던 일도 손이 안 가고 하루가 그냥 지나가. => top=슬픔, match=1, ds=0.5670, dts=0.7700, label=주의, crisis=False
- WB504_mixed_observation_sad_02 [mixed_observation / 슬픔] 겉으로는 평소처럼 움직였는데, 안쪽 느낌은 며칠째 잠을 자도 개운하지 않고 몸이 무겁게 가라앉아. => top=슬픔, match=1, ds=0.5500, dts=0.6750, label=주의, crisis=False
- WB504_mixed_observation_sad_03 [mixed_observation / 슬픔] 겉으로는 평소처럼 움직였는데, 안쪽 느낌은 내가 있어도 없어도 별 차이 없을 것 같다는 생각이 반복돼. => top=중립, match=0, ds=0.4124, dts=0.0000, label=주의, crisis=False
- WB504_mixed_observation_sad_04 [mixed_observation / 슬픔] 겉으로는 평소처럼 움직였는데, 안쪽 느낌은 기대했던 모임이 취소돼서 집에 오니 마음이 푹 꺼졌어. => top=행복, match=0, ds=0.3450, dts=0.0000, label=주의, crisis=False
- WB504_mixed_observation_sad_05 [mixed_observation / 슬픔] 겉으로는 평소처럼 움직였는데, 안쪽 느낌은 괜찮다고 말했는데 막상 혼자 있으니 서운함이 남았어. => top=슬픔, match=1, ds=0.6522, dts=0.2000, label=주의, crisis=False
- WB504_mixed_observation_sad_06 [mixed_observation / 슬픔] 겉으로는 평소처럼 움직였는데, 안쪽 느낌은 비 오는 소리를 듣다 보니 괜히 옛날 생각이 나서 먹먹했어. => top=슬픔, match=1, ds=0.5600, dts=0.2000, label=주의, crisis=False
- WB504_mixed_observation_anger_01 [mixed_observation / 분노] 겉으로는 평소처럼 움직였는데, 안쪽 느낌은 내가 맡은 일도 아닌데 급하다고 떠넘겨서 짜증이 확 났어. => top=분노, match=1, ds=0.5407, dts=0.0000, label=주의, crisis=False
- WB504_mixed_observation_anger_02 [mixed_observation / 분노] 겉으로는 평소처럼 움직였는데, 안쪽 느낌은 분명히 약속한 기준을 자기 마음대로 바꿔서 화가 났어. => top=분노, match=1, ds=0.5415, dts=0.0000, label=주의, crisis=False
- WB504_mixed_observation_anger_03 [mixed_observation / 분노] 겉으로는 평소처럼 움직였는데, 안쪽 느낌은 말을 끝내기도 전에 결론부터 내리는 태도가 너무 거슬렸어. => top=중립, match=0, ds=0.4242, dts=0.0000, label=주의, crisis=False
- WB504_mixed_observation_anger_04 [mixed_observation / 분노] 겉으로는 평소처럼 움직였는데, 안쪽 느낌은 내 시간을 당연한 것처럼 요구하는 메시지를 보고 열이 올랐어. => top=중립, match=0, ds=0.4307, dts=0.0000, label=주의, crisis=False
- WB504_mixed_observation_anger_05 [mixed_observation / 분노] 겉으로는 평소처럼 움직였는데, 안쪽 느낌은 실수는 같이 했는데 설명은 나만 하라는 분위기가 억울했어. => top=분노, match=1, ds=0.5365, dts=0.0000, label=주의, crisis=False
- WB504_mixed_observation_anger_06 [mixed_observation / 분노] 겉으로는 평소처럼 움직였는데, 안쪽 느낌은 사과 대신 농담으로 넘기려는 모습이 계속 마음에 걸렸어. => top=중립, match=0, ds=0.4299, dts=0.0000, label=주의, crisis=False
- WB504_mixed_observation_fear_01 [mixed_observation / 공포] 겉으로는 평소처럼 움직였는데, 안쪽 느낌은 밤에 골목 불이 꺼져 있어서 지나가는 동안 어깨가 굳었어. => top=공포, match=1, ds=0.5032, dts=0.0000, label=주의, crisis=False
- WB504_mixed_observation_fear_02 [mixed_observation / 공포] 겉으로는 평소처럼 움직였는데, 안쪽 느낌은 휴대폰에 모르는 결제 문자가 떠서 순간 손끝이 차가워졌어. => top=공포, match=1, ds=0.5467, dts=0.0000, label=주의, crisis=False
- WB504_mixed_observation_fear_03 [mixed_observation / 공포] 겉으로는 평소처럼 움직였는데, 안쪽 느낌은 엘리베이터 문이 한참 안 열려서 숨을 작게 쉬고 있었어. => top=중립, match=0, ds=0.3800, dts=0.0000, label=주의, crisis=False
- WB504_mixed_observation_fear_04 [mixed_observation / 공포] 겉으로는 평소처럼 움직였는데, 안쪽 느낌은 발표 자료가 갑자기 안 열려서 심장이 빠르게 뛰었어. => top=놀람, match=0, ds=0.5200, dts=0.0000, label=주의, crisis=False
- WB504_mixed_observation_fear_05 [mixed_observation / 공포] 겉으로는 평소처럼 움직였는데, 안쪽 느낌은 새벽에 현관 센서등이 켜져서 한동안 움직이지 못했어. => top=중립, match=0, ds=0.4638, dts=0.0000, label=주의, crisis=False
- WB504_mixed_observation_fear_06 [mixed_observation / 공포] 겉으로는 평소처럼 움직였는데, 안쪽 느낌은 병원 전화를 받기 전부터 입안이 바짝 말랐어. => top=슬픔, match=0, ds=0.9075, dts=0.2000, label=위험, crisis=False
- WB504_mixed_observation_surprise_01 [mixed_observation / 놀람] 겉으로는 평소처럼 움직였는데, 안쪽 느낌은 조용하던 프린터가 갑자기 움직여서 몸이 움찔했어. => top=놀람, match=1, ds=0.5232, dts=0.0000, label=주의, crisis=False
- WB504_mixed_observation_surprise_02 [mixed_observation / 놀람] 겉으로는 평소처럼 움직였는데, 안쪽 느낌은 닫힌 줄 알았던 창문이 덜컥 열려서 눈이 커졌어. => top=중립, match=0, ds=0.3800, dts=0.0000, label=주의, crisis=False
- WB504_mixed_observation_surprise_03 [mixed_observation / 놀람] 겉으로는 평소처럼 움직였는데, 안쪽 느낌은 회의 시간이 앞당겨졌다는 알림을 보고 잠깐 멍해졌어. => top=놀람, match=1, ds=0.5500, dts=0.0000, label=주의, crisis=False
- WB504_mixed_observation_surprise_04 [mixed_observation / 놀람] 겉으로는 평소처럼 움직였는데, 안쪽 느낌은 뒤에서 누가 내 이름을 불러서 바로 돌아봤어. => top=중립, match=0, ds=0.3841, dts=0.0000, label=주의, crisis=False
- WB504_mixed_observation_surprise_05 [mixed_observation / 놀람] 겉으로는 평소처럼 움직였는데, 안쪽 느낌은 저장해둔 파일 이름이 바뀌어 있어서 순간 당황했어. => top=놀람, match=1, ds=0.5386, dts=0.0000, label=주의, crisis=False
- WB504_mixed_observation_surprise_06 [mixed_observation / 놀람] 겉으로는 평소처럼 움직였는데, 안쪽 느낌은 택배 도착 사진이 예상과 달라서 한참 다시 봤어. => top=중립, match=0, ds=0.4243, dts=0.0000, label=주의, crisis=False
- WB504_mixed_observation_disgust_01 [mixed_observation / 혐오] 겉으로는 평소처럼 움직였는데, 안쪽 느낌은 냉장고 구석 반찬통에서 시큼한 냄새가 올라와서 바로 닫았어. => top=혐오, match=1, ds=0.5000, dts=0.0000, label=주의, crisis=False
- WB504_mixed_observation_disgust_02 [mixed_observation / 혐오] 겉으로는 평소처럼 움직였는데, 안쪽 느낌은 손잡이에 끈적한 자국이 묻어 있어서 만지기가 싫었어. => top=혐오, match=1, ds=0.5000, dts=0.0000, label=주의, crisis=False
- WB504_mixed_observation_disgust_03 [mixed_observation / 혐오] 겉으로는 평소처럼 움직였는데, 안쪽 느낌은 젖은 걸레 냄새가 방 안에 남아서 계속 인상이 찌푸려졌어. => top=혐오, match=1, ds=0.5000, dts=0.0000, label=주의, crisis=False
- WB504_mixed_observation_disgust_04 [mixed_observation / 혐오] 겉으로는 평소처럼 움직였는데, 안쪽 느낌은 약한 사람을 일부러 몰아붙이는 장면을 보고 속이 뒤틀렸어. => top=혐오, match=1, ds=0.8926, dts=0.0000, label=주의, crisis=False
- WB504_mixed_observation_disgust_05 [mixed_observation / 혐오] 겉으로는 평소처럼 움직였는데, 안쪽 느낌은 컵 바닥에 남은 찌꺼기를 보고 마실 생각이 사라졌어. => top=혐오, match=1, ds=0.5000, dts=0.0000, label=주의, crisis=False
- WB504_mixed_observation_disgust_06 [mixed_observation / 혐오] 겉으로는 평소처럼 움직였는데, 안쪽 느낌은 겉으로 친한 척하면서 뒤에서 험담하는 말을 들으니 역했어. => top=중립, match=0, ds=0.4660, dts=0.0000, label=주의, crisis=False
- WB504_mixed_observation_neutral_01 [mixed_observation / 중립] 겉으로는 평소처럼 움직였는데, 안쪽 느낌은 노트북 충전 케이블을 책상 오른쪽으로 옮겨뒀어. => top=중립, match=1, ds=0.3800, dts=0.0000, label=보통, crisis=False
- WB504_mixed_observation_neutral_02 [mixed_observation / 중립] 겉으로는 평소처럼 움직였는데, 안쪽 느낌은 내일 가져갈 서류를 투명 파일에 넣어뒀어. => top=중립, match=1, ds=0.3737, dts=0.0000, label=보통, crisis=False
- WB504_mixed_observation_neutral_03 [mixed_observation / 중립] 겉으로는 평소처럼 움직였는데, 안쪽 느낌은 점심 후보를 세 곳으로 줄여서 메모장에 적었어. => top=중립, match=1, ds=0.3800, dts=0.0000, label=보통, crisis=False
- WB504_mixed_observation_neutral_04 [mixed_observation / 중립] 겉으로는 평소처럼 움직였는데, 안쪽 느낌은 세탁 완료 알림을 보고 빨래를 건조대에 널었어. => top=중립, match=1, ds=0.3783, dts=0.0000, label=보통, crisis=False
- WB504_mixed_observation_neutral_05 [mixed_observation / 중립] 겉으로는 평소처럼 움직였는데, 안쪽 느낌은 달력에 이번 주 분리수거 날짜를 다시 표시했어. => top=중립, match=1, ds=0.3800, dts=0.0000, label=보통, crisis=False
- WB504_mixed_observation_neutral_06 [mixed_observation / 중립] 겉으로는 평소처럼 움직였는데, 안쪽 느낌은 이어폰 케이스 배터리를 확인하고 가방에 넣었어. => top=중립, match=1, ds=0.3643, dts=0.0000, label=보통, crisis=False
- WB504_body_signal_short_happy_01 [body_signal_short / 행복] 몸이 먼저 반응했고, 뒤늦게 말로 붙이면 아침에 만든 계란말이가 모양까지 잘 나와서 괜히 뿌듯했어. => top=행복, match=1, ds=0.3285, dts=0.0000, label=보통, crisis=False
- WB504_body_signal_short_happy_02 [body_signal_short / 행복] 몸이 먼저 반응했고, 뒤늦게 말로 붙이면 길에서 좋아하던 노래가 들려서 발걸음이 조금 빨라졌어. => top=행복, match=1, ds=0.3444, dts=0.0000, label=보통, crisis=False
- WB504_body_signal_short_happy_03 [body_signal_short / 행복] 몸이 먼저 반응했고, 뒤늦게 말로 붙이면 친구가 내 농담에 크게 웃어줘서 마음이 환해졌어. => top=행복, match=1, ds=0.5276, dts=0.0000, label=보통, crisis=False
- WB504_body_signal_short_happy_04 [body_signal_short / 행복] 몸이 먼저 반응했고, 뒤늦게 말로 붙이면 오래 찾던 자료를 드디어 발견해서 속이 시원했어. => top=행복, match=1, ds=0.3316, dts=0.0000, label=보통, crisis=False
- WB504_body_signal_short_happy_05 [body_signal_short / 행복] 몸이 먼저 반응했고, 뒤늦게 말로 붙이면 퇴근길 하늘색이 예뻐서 사진을 한 장 남겼어. => top=행복, match=1, ds=0.3271, dts=0.0000, label=보통, crisis=False
- WB504_body_signal_short_happy_06 [body_signal_short / 행복] 몸이 먼저 반응했고, 뒤늦게 말로 붙이면 오늘은 계획한 만큼만 해도 충분하다는 생각이 들어서 편했어. => top=중립, match=0, ds=0.4040, dts=0.0000, label=보통, crisis=False
- WB504_body_signal_short_sad_01 [body_signal_short / 슬픔] 몸이 먼저 반응했고, 뒤늦게 말로 붙이면 요즘은 좋아하던 일도 손이 안 가고 하루가 그냥 지나가. => top=슬픔, match=1, ds=0.5691, dts=0.7700, label=주의, crisis=False
- WB504_body_signal_short_sad_02 [body_signal_short / 슬픔] 몸이 먼저 반응했고, 뒤늦게 말로 붙이면 며칠째 잠을 자도 개운하지 않고 몸이 무겁게 가라앉아. => top=슬픔, match=1, ds=0.5500, dts=0.6750, label=주의, crisis=False
- WB504_body_signal_short_sad_03 [body_signal_short / 슬픔] 몸이 먼저 반응했고, 뒤늦게 말로 붙이면 내가 있어도 없어도 별 차이 없을 것 같다는 생각이 반복돼. => top=중립, match=0, ds=0.6863, dts=0.0000, label=주의, crisis=False
- WB504_body_signal_short_sad_04 [body_signal_short / 슬픔] 몸이 먼저 반응했고, 뒤늦게 말로 붙이면 기대했던 모임이 취소돼서 집에 오니 마음이 푹 꺼졌어. => top=행복, match=0, ds=0.3450, dts=0.0000, label=주의, crisis=False
- WB504_body_signal_short_sad_05 [body_signal_short / 슬픔] 몸이 먼저 반응했고, 뒤늦게 말로 붙이면 괜찮다고 말했는데 막상 혼자 있으니 서운함이 남았어. => top=슬픔, match=1, ds=0.5735, dts=0.2000, label=주의, crisis=False
- WB504_body_signal_short_sad_06 [body_signal_short / 슬픔] 몸이 먼저 반응했고, 뒤늦게 말로 붙이면 비 오는 소리를 듣다 보니 괜히 옛날 생각이 나서 먹먹했어. => top=슬픔, match=1, ds=0.6570, dts=0.2000, label=주의, crisis=False
- WB504_body_signal_short_anger_01 [body_signal_short / 분노] 몸이 먼저 반응했고, 뒤늦게 말로 붙이면 내가 맡은 일도 아닌데 급하다고 떠넘겨서 짜증이 확 났어. => top=분노, match=1, ds=0.5382, dts=0.0000, label=주의, crisis=False
- WB504_body_signal_short_anger_02 [body_signal_short / 분노] 몸이 먼저 반응했고, 뒤늦게 말로 붙이면 분명히 약속한 기준을 자기 마음대로 바꿔서 화가 났어. => top=분노, match=1, ds=0.5346, dts=0.0000, label=주의, crisis=False
- WB504_body_signal_short_anger_03 [body_signal_short / 분노] 몸이 먼저 반응했고, 뒤늦게 말로 붙이면 말을 끝내기도 전에 결론부터 내리는 태도가 너무 거슬렸어. => top=중립, match=0, ds=0.4283, dts=0.0000, label=주의, crisis=False
- WB504_body_signal_short_anger_04 [body_signal_short / 분노] 몸이 먼저 반응했고, 뒤늦게 말로 붙이면 내 시간을 당연한 것처럼 요구하는 메시지를 보고 열이 올랐어. => top=중립, match=0, ds=0.4212, dts=0.0000, label=주의, crisis=False
- WB504_body_signal_short_anger_05 [body_signal_short / 분노] 몸이 먼저 반응했고, 뒤늦게 말로 붙이면 실수는 같이 했는데 설명은 나만 하라는 분위기가 억울했어. => top=분노, match=1, ds=0.5375, dts=0.0000, label=주의, crisis=False
- WB504_body_signal_short_anger_06 [body_signal_short / 분노] 몸이 먼저 반응했고, 뒤늦게 말로 붙이면 사과 대신 농담으로 넘기려는 모습이 계속 마음에 걸렸어. => top=혐오, match=0, ds=0.8592, dts=0.0000, label=위험, crisis=False
- WB504_body_signal_short_fear_01 [body_signal_short / 공포] 몸이 먼저 반응했고, 뒤늦게 말로 붙이면 밤에 골목 불이 꺼져 있어서 지나가는 동안 어깨가 굳었어. => top=공포, match=1, ds=0.4179, dts=0.0000, label=주의, crisis=False
- WB504_body_signal_short_fear_02 [body_signal_short / 공포] 몸이 먼저 반응했고, 뒤늦게 말로 붙이면 휴대폰에 모르는 결제 문자가 떠서 순간 손끝이 차가워졌어. => top=공포, match=1, ds=0.5284, dts=0.0000, label=주의, crisis=False
- WB504_body_signal_short_fear_03 [body_signal_short / 공포] 몸이 먼저 반응했고, 뒤늦게 말로 붙이면 엘리베이터 문이 한참 안 열려서 숨을 작게 쉬고 있었어. => top=중립, match=0, ds=0.3800, dts=0.0000, label=주의, crisis=False
- WB504_body_signal_short_fear_04 [body_signal_short / 공포] 몸이 먼저 반응했고, 뒤늦게 말로 붙이면 발표 자료가 갑자기 안 열려서 심장이 빠르게 뛰었어. => top=놀람, match=0, ds=0.5200, dts=0.0000, label=주의, crisis=False
- WB504_body_signal_short_fear_05 [body_signal_short / 공포] 몸이 먼저 반응했고, 뒤늦게 말로 붙이면 새벽에 현관 센서등이 켜져서 한동안 움직이지 못했어. => top=중립, match=0, ds=0.6084, dts=0.0000, label=주의, crisis=False
- WB504_body_signal_short_fear_06 [body_signal_short / 공포] 몸이 먼저 반응했고, 뒤늦게 말로 붙이면 병원 전화를 받기 전부터 입안이 바짝 말랐어. => top=중립, match=0, ds=0.4418, dts=0.0000, label=주의, crisis=False
- WB504_body_signal_short_surprise_01 [body_signal_short / 놀람] 몸이 먼저 반응했고, 뒤늦게 말로 붙이면 조용하던 프린터가 갑자기 움직여서 몸이 움찔했어. => top=놀람, match=1, ds=0.5500, dts=0.0000, label=주의, crisis=False
- WB504_body_signal_short_surprise_02 [body_signal_short / 놀람] 몸이 먼저 반응했고, 뒤늦게 말로 붙이면 닫힌 줄 알았던 창문이 덜컥 열려서 눈이 커졌어. => top=중립, match=0, ds=0.3800, dts=0.0000, label=주의, crisis=False
- WB504_body_signal_short_surprise_03 [body_signal_short / 놀람] 몸이 먼저 반응했고, 뒤늦게 말로 붙이면 회의 시간이 앞당겨졌다는 알림을 보고 잠깐 멍해졌어. => top=놀람, match=1, ds=0.5169, dts=0.0000, label=주의, crisis=False
- WB504_body_signal_short_surprise_04 [body_signal_short / 놀람] 몸이 먼저 반응했고, 뒤늦게 말로 붙이면 뒤에서 누가 내 이름을 불러서 바로 돌아봤어. => top=중립, match=0, ds=0.4667, dts=0.0000, label=주의, crisis=False
- WB504_body_signal_short_surprise_05 [body_signal_short / 놀람] 몸이 먼저 반응했고, 뒤늦게 말로 붙이면 저장해둔 파일 이름이 바뀌어 있어서 순간 당황했어. => top=놀람, match=1, ds=0.5390, dts=0.0000, label=주의, crisis=False
- WB504_body_signal_short_surprise_06 [body_signal_short / 놀람] 몸이 먼저 반응했고, 뒤늦게 말로 붙이면 택배 도착 사진이 예상과 달라서 한참 다시 봤어. => top=중립, match=0, ds=0.4743, dts=0.0000, label=주의, crisis=False
- WB504_body_signal_short_disgust_01 [body_signal_short / 혐오] 몸이 먼저 반응했고, 뒤늦게 말로 붙이면 냉장고 구석 반찬통에서 시큼한 냄새가 올라와서 바로 닫았어. => top=혐오, match=1, ds=0.5000, dts=0.0000, label=주의, crisis=False
- WB504_body_signal_short_disgust_02 [body_signal_short / 혐오] 몸이 먼저 반응했고, 뒤늦게 말로 붙이면 손잡이에 끈적한 자국이 묻어 있어서 만지기가 싫었어. => top=혐오, match=1, ds=0.5000, dts=0.0000, label=주의, crisis=False
- WB504_body_signal_short_disgust_03 [body_signal_short / 혐오] 몸이 먼저 반응했고, 뒤늦게 말로 붙이면 젖은 걸레 냄새가 방 안에 남아서 계속 인상이 찌푸려졌어. => top=혐오, match=1, ds=0.5000, dts=0.0000, label=주의, crisis=False
- WB504_body_signal_short_disgust_04 [body_signal_short / 혐오] 몸이 먼저 반응했고, 뒤늦게 말로 붙이면 약한 사람을 일부러 몰아붙이는 장면을 보고 속이 뒤틀렸어. => top=혐오, match=1, ds=0.9555, dts=0.0000, label=위험, crisis=False
- WB504_body_signal_short_disgust_05 [body_signal_short / 혐오] 몸이 먼저 반응했고, 뒤늦게 말로 붙이면 컵 바닥에 남은 찌꺼기를 보고 마실 생각이 사라졌어. => top=혐오, match=1, ds=0.5000, dts=0.0000, label=주의, crisis=False
- WB504_body_signal_short_disgust_06 [body_signal_short / 혐오] 몸이 먼저 반응했고, 뒤늦게 말로 붙이면 겉으로 친한 척하면서 뒤에서 험담하는 말을 들으니 역했어. => top=중립, match=0, ds=0.4677, dts=0.0000, label=주의, crisis=False
- WB504_body_signal_short_neutral_01 [body_signal_short / 중립] 몸이 먼저 반응했고, 뒤늦게 말로 붙이면 노트북 충전 케이블을 책상 오른쪽으로 옮겨뒀어. => top=중립, match=1, ds=0.3800, dts=0.0000, label=보통, crisis=False
- WB504_body_signal_short_neutral_02 [body_signal_short / 중립] 몸이 먼저 반응했고, 뒤늦게 말로 붙이면 내일 가져갈 서류를 투명 파일에 넣어뒀어. => top=중립, match=1, ds=0.3800, dts=0.0000, label=보통, crisis=False
- WB504_body_signal_short_neutral_03 [body_signal_short / 중립] 몸이 먼저 반응했고, 뒤늦게 말로 붙이면 점심 후보를 세 곳으로 줄여서 메모장에 적었어. => top=중립, match=1, ds=0.3800, dts=0.0000, label=보통, crisis=False
- WB504_body_signal_short_neutral_04 [body_signal_short / 중립] 몸이 먼저 반응했고, 뒤늦게 말로 붙이면 세탁 완료 알림을 보고 빨래를 건조대에 널었어. => top=중립, match=1, ds=0.3800, dts=0.0000, label=보통, crisis=False
- WB504_body_signal_short_neutral_05 [body_signal_short / 중립] 몸이 먼저 반응했고, 뒤늦게 말로 붙이면 달력에 이번 주 분리수거 날짜를 다시 표시했어. => top=중립, match=1, ds=0.3800, dts=0.0000, label=보통, crisis=False
- WB504_body_signal_short_neutral_06 [body_signal_short / 중립] 몸이 먼저 반응했고, 뒤늦게 말로 붙이면 이어폰 케이스 배터리를 확인하고 가방에 넣었어. => top=중립, match=1, ds=0.3744, dts=0.0000, label=보통, crisis=False
- WB504_friend_dm_happy_01 [friend_dm / 행복] 친구한테 답장하듯 말하면 아침에 만든 계란말이가 모양까지 잘 나와서 괜히 뿌듯했어. => top=행복, match=1, ds=0.3269, dts=0.0000, label=보통, crisis=False
- WB504_friend_dm_happy_02 [friend_dm / 행복] 친구한테 답장하듯 말하면 길에서 좋아하던 노래가 들려서 발걸음이 조금 빨라졌어. => top=행복, match=1, ds=0.3450, dts=0.0000, label=보통, crisis=False
- WB504_friend_dm_happy_03 [friend_dm / 행복] 친구한테 답장하듯 말하면 친구가 내 농담에 크게 웃어줘서 마음이 환해졌어. => top=행복, match=1, ds=0.5396, dts=0.0000, label=보통, crisis=False
- WB504_friend_dm_happy_04 [friend_dm / 행복] 친구한테 답장하듯 말하면 오래 찾던 자료를 드디어 발견해서 속이 시원했어. => top=행복, match=1, ds=0.3266, dts=0.0000, label=보통, crisis=False
- WB504_friend_dm_happy_05 [friend_dm / 행복] 친구한테 답장하듯 말하면 퇴근길 하늘색이 예뻐서 사진을 한 장 남겼어. => top=행복, match=1, ds=0.3289, dts=0.0000, label=보통, crisis=False
- WB504_friend_dm_happy_06 [friend_dm / 행복] 친구한테 답장하듯 말하면 오늘은 계획한 만큼만 해도 충분하다는 생각이 들어서 편했어. => top=행복, match=1, ds=0.3446, dts=0.0000, label=보통, crisis=False
- WB504_friend_dm_sad_01 [friend_dm / 슬픔] 친구한테 답장하듯 말하면 요즘은 좋아하던 일도 손이 안 가고 하루가 그냥 지나가. => top=슬픔, match=1, ds=0.5698, dts=0.7700, label=주의, crisis=False
- WB504_friend_dm_sad_02 [friend_dm / 슬픔] 친구한테 답장하듯 말하면 며칠째 잠을 자도 개운하지 않고 몸이 무겁게 가라앉아. => top=슬픔, match=1, ds=0.5500, dts=0.6750, label=주의, crisis=False
- WB504_friend_dm_sad_03 [friend_dm / 슬픔] 친구한테 답장하듯 말하면 내가 있어도 없어도 별 차이 없을 것 같다는 생각이 반복돼. => top=중립, match=0, ds=0.4354, dts=0.0000, label=주의, crisis=False
- WB504_friend_dm_sad_04 [friend_dm / 슬픔] 친구한테 답장하듯 말하면 기대했던 모임이 취소돼서 집에 오니 마음이 푹 꺼졌어. => top=행복, match=0, ds=0.3450, dts=0.0000, label=주의, crisis=False
- WB504_friend_dm_sad_05 [friend_dm / 슬픔] 친구한테 답장하듯 말하면 괜찮다고 말했는데 막상 혼자 있으니 서운함이 남았어. => top=슬픔, match=1, ds=0.8870, dts=0.2000, label=주의, crisis=False
- WB504_friend_dm_sad_06 [friend_dm / 슬픔] 친구한테 답장하듯 말하면 비 오는 소리를 듣다 보니 괜히 옛날 생각이 나서 먹먹했어. => top=슬픔, match=1, ds=0.5723, dts=0.2000, label=주의, crisis=False
- WB504_friend_dm_anger_01 [friend_dm / 분노] 친구한테 답장하듯 말하면 내가 맡은 일도 아닌데 급하다고 떠넘겨서 짜증이 확 났어. => top=분노, match=1, ds=0.5352, dts=0.0000, label=주의, crisis=False
- WB504_friend_dm_anger_02 [friend_dm / 분노] 친구한테 답장하듯 말하면 분명히 약속한 기준을 자기 마음대로 바꿔서 화가 났어. => top=분노, match=1, ds=0.5304, dts=0.0000, label=주의, crisis=False
- WB504_friend_dm_anger_03 [friend_dm / 분노] 친구한테 답장하듯 말하면 말을 끝내기도 전에 결론부터 내리는 태도가 너무 거슬렸어. => top=중립, match=0, ds=0.4257, dts=0.0000, label=주의, crisis=False
- WB504_friend_dm_anger_04 [friend_dm / 분노] 친구한테 답장하듯 말하면 내 시간을 당연한 것처럼 요구하는 메시지를 보고 열이 올랐어. => top=중립, match=0, ds=0.3512, dts=0.0000, label=주의, crisis=False
- WB504_friend_dm_anger_05 [friend_dm / 분노] 친구한테 답장하듯 말하면 실수는 같이 했는데 설명은 나만 하라는 분위기가 억울했어. => top=분노, match=1, ds=0.5337, dts=0.0000, label=주의, crisis=False
- WB504_friend_dm_anger_06 [friend_dm / 분노] 친구한테 답장하듯 말하면 사과 대신 농담으로 넘기려는 모습이 계속 마음에 걸렸어. => top=중립, match=0, ds=0.3554, dts=0.0000, label=주의, crisis=False
- WB504_friend_dm_fear_01 [friend_dm / 공포] 친구한테 답장하듯 말하면 밤에 골목 불이 꺼져 있어서 지나가는 동안 어깨가 굳었어. => top=공포, match=1, ds=0.4057, dts=0.0000, label=주의, crisis=False
- WB504_friend_dm_fear_02 [friend_dm / 공포] 친구한테 답장하듯 말하면 휴대폰에 모르는 결제 문자가 떠서 순간 손끝이 차가워졌어. => top=공포, match=1, ds=0.5218, dts=0.0000, label=주의, crisis=False
- WB504_friend_dm_fear_03 [friend_dm / 공포] 친구한테 답장하듯 말하면 엘리베이터 문이 한참 안 열려서 숨을 작게 쉬고 있었어. => top=중립, match=0, ds=0.3800, dts=0.0000, label=주의, crisis=False
- WB504_friend_dm_fear_04 [friend_dm / 공포] 친구한테 답장하듯 말하면 발표 자료가 갑자기 안 열려서 심장이 빠르게 뛰었어. => top=놀람, match=0, ds=0.5200, dts=0.0000, label=주의, crisis=False
- WB504_friend_dm_fear_05 [friend_dm / 공포] 친구한테 답장하듯 말하면 새벽에 현관 센서등이 켜져서 한동안 움직이지 못했어. => top=중립, match=0, ds=0.5619, dts=0.0000, label=주의, crisis=False
- WB504_friend_dm_fear_06 [friend_dm / 공포] 친구한테 답장하듯 말하면 병원 전화를 받기 전부터 입안이 바짝 말랐어. => top=중립, match=0, ds=0.3545, dts=0.0000, label=주의, crisis=False
- WB504_friend_dm_surprise_01 [friend_dm / 놀람] 친구한테 답장하듯 말하면 조용하던 프린터가 갑자기 움직여서 몸이 움찔했어. => top=놀람, match=1, ds=0.5500, dts=0.0000, label=주의, crisis=False
- WB504_friend_dm_surprise_02 [friend_dm / 놀람] 친구한테 답장하듯 말하면 닫힌 줄 알았던 창문이 덜컥 열려서 눈이 커졌어. => top=중립, match=0, ds=0.3800, dts=0.0000, label=주의, crisis=False
- WB504_friend_dm_surprise_03 [friend_dm / 놀람] 친구한테 답장하듯 말하면 회의 시간이 앞당겨졌다는 알림을 보고 잠깐 멍해졌어. => top=놀람, match=1, ds=0.5500, dts=0.0000, label=주의, crisis=False
- WB504_friend_dm_surprise_04 [friend_dm / 놀람] 친구한테 답장하듯 말하면 뒤에서 누가 내 이름을 불러서 바로 돌아봤어. => top=중립, match=0, ds=0.4152, dts=0.0000, label=주의, crisis=False
- WB504_friend_dm_surprise_05 [friend_dm / 놀람] 친구한테 답장하듯 말하면 저장해둔 파일 이름이 바뀌어 있어서 순간 당황했어. => top=놀람, match=1, ds=0.5380, dts=0.0000, label=주의, crisis=False
- WB504_friend_dm_surprise_06 [friend_dm / 놀람] 친구한테 답장하듯 말하면 택배 도착 사진이 예상과 달라서 한참 다시 봤어. => top=중립, match=0, ds=0.7073, dts=0.0000, label=주의, crisis=False
- WB504_friend_dm_disgust_01 [friend_dm / 혐오] 친구한테 답장하듯 말하면 냉장고 구석 반찬통에서 시큼한 냄새가 올라와서 바로 닫았어. => top=혐오, match=1, ds=0.5000, dts=0.0000, label=주의, crisis=False
- WB504_friend_dm_disgust_02 [friend_dm / 혐오] 친구한테 답장하듯 말하면 손잡이에 끈적한 자국이 묻어 있어서 만지기가 싫었어. => top=혐오, match=1, ds=0.4980, dts=0.0000, label=주의, crisis=False
- WB504_friend_dm_disgust_03 [friend_dm / 혐오] 친구한테 답장하듯 말하면 젖은 걸레 냄새가 방 안에 남아서 계속 인상이 찌푸려졌어. => top=혐오, match=1, ds=0.5000, dts=0.0000, label=주의, crisis=False
- WB504_friend_dm_disgust_04 [friend_dm / 혐오] 친구한테 답장하듯 말하면 약한 사람을 일부러 몰아붙이는 장면을 보고 속이 뒤틀렸어. => top=혐오, match=1, ds=0.9322, dts=0.0000, label=주의, crisis=False
- WB504_friend_dm_disgust_05 [friend_dm / 혐오] 친구한테 답장하듯 말하면 컵 바닥에 남은 찌꺼기를 보고 마실 생각이 사라졌어. => top=혐오, match=1, ds=0.5000, dts=0.0000, label=주의, crisis=False
- WB504_friend_dm_disgust_06 [friend_dm / 혐오] 친구한테 답장하듯 말하면 겉으로 친한 척하면서 뒤에서 험담하는 말을 들으니 역했어. => top=중립, match=0, ds=0.4690, dts=0.0000, label=주의, crisis=False
- WB504_friend_dm_neutral_01 [friend_dm / 중립] 친구한테 답장하듯 말하면 노트북 충전 케이블을 책상 오른쪽으로 옮겨뒀어. => top=중립, match=1, ds=0.3735, dts=0.0000, label=보통, crisis=False
- WB504_friend_dm_neutral_02 [friend_dm / 중립] 친구한테 답장하듯 말하면 내일 가져갈 서류를 투명 파일에 넣어뒀어. => top=중립, match=1, ds=0.3800, dts=0.0000, label=보통, crisis=False
- WB504_friend_dm_neutral_03 [friend_dm / 중립] 친구한테 답장하듯 말하면 점심 후보를 세 곳으로 줄여서 메모장에 적었어. => top=중립, match=1, ds=0.3800, dts=0.0000, label=보통, crisis=False
- WB504_friend_dm_neutral_04 [friend_dm / 중립] 친구한테 답장하듯 말하면 세탁 완료 알림을 보고 빨래를 건조대에 널었어. => top=중립, match=1, ds=0.3591, dts=0.0000, label=보통, crisis=False
- WB504_friend_dm_neutral_05 [friend_dm / 중립] 친구한테 답장하듯 말하면 달력에 이번 주 분리수거 날짜를 다시 표시했어. => top=중립, match=1, ds=0.3773, dts=0.0000, label=보통, crisis=False
- WB504_friend_dm_neutral_06 [friend_dm / 중립] 친구한테 답장하듯 말하면 이어폰 케이스 배터리를 확인하고 가방에 넣었어. => top=중립, match=1, ds=0.3622, dts=0.0000, label=보통, crisis=False
- WB504_weather_context_happy_01 [weather_context / 행복] 창밖을 보다가 갑자기 선명해졌는데, 아침에 만든 계란말이가 모양까지 잘 나와서 괜히 뿌듯했어. => top=행복, match=1, ds=0.3283, dts=0.0000, label=보통, crisis=False
- WB504_weather_context_happy_02 [weather_context / 행복] 창밖을 보다가 갑자기 선명해졌는데, 길에서 좋아하던 노래가 들려서 발걸음이 조금 빨라졌어. => top=행복, match=1, ds=0.3450, dts=0.0000, label=보통, crisis=False
- WB504_weather_context_happy_03 [weather_context / 행복] 창밖을 보다가 갑자기 선명해졌는데, 친구가 내 농담에 크게 웃어줘서 마음이 환해졌어. => top=행복, match=1, ds=0.5373, dts=0.0000, label=보통, crisis=False
- WB504_weather_context_happy_04 [weather_context / 행복] 창밖을 보다가 갑자기 선명해졌는데, 오래 찾던 자료를 드디어 발견해서 속이 시원했어. => top=행복, match=1, ds=0.3326, dts=0.0000, label=보통, crisis=False
- WB504_weather_context_happy_05 [weather_context / 행복] 창밖을 보다가 갑자기 선명해졌는데, 퇴근길 하늘색이 예뻐서 사진을 한 장 남겼어. => top=행복, match=1, ds=0.3288, dts=0.0000, label=보통, crisis=False
- WB504_weather_context_happy_06 [weather_context / 행복] 창밖을 보다가 갑자기 선명해졌는데, 오늘은 계획한 만큼만 해도 충분하다는 생각이 들어서 편했어. => top=중립, match=0, ds=0.3787, dts=0.0000, label=보통, crisis=False
- WB504_weather_context_sad_01 [weather_context / 슬픔] 창밖을 보다가 갑자기 선명해졌는데, 요즘은 좋아하던 일도 손이 안 가고 하루가 그냥 지나가. => top=슬픔, match=1, ds=0.5741, dts=0.7700, label=주의, crisis=False
- WB504_weather_context_sad_02 [weather_context / 슬픔] 창밖을 보다가 갑자기 선명해졌는데, 며칠째 잠을 자도 개운하지 않고 몸이 무겁게 가라앉아. => top=슬픔, match=1, ds=0.5500, dts=0.6750, label=주의, crisis=False
- WB504_weather_context_sad_03 [weather_context / 슬픔] 창밖을 보다가 갑자기 선명해졌는데, 내가 있어도 없어도 별 차이 없을 것 같다는 생각이 반복돼. => top=중립, match=0, ds=0.4384, dts=0.0000, label=주의, crisis=False
- WB504_weather_context_sad_04 [weather_context / 슬픔] 창밖을 보다가 갑자기 선명해졌는데, 기대했던 모임이 취소돼서 집에 오니 마음이 푹 꺼졌어. => top=행복, match=0, ds=0.3450, dts=0.0000, label=주의, crisis=False
- WB504_weather_context_sad_05 [weather_context / 슬픔] 창밖을 보다가 갑자기 선명해졌는데, 괜찮다고 말했는데 막상 혼자 있으니 서운함이 남았어. => top=슬픔, match=1, ds=0.6578, dts=0.2000, label=주의, crisis=False
- WB504_weather_context_sad_06 [weather_context / 슬픔] 창밖을 보다가 갑자기 선명해졌는데, 비 오는 소리를 듣다 보니 괜히 옛날 생각이 나서 먹먹했어. => top=슬픔, match=1, ds=0.5775, dts=0.2000, label=주의, crisis=False
- WB504_weather_context_anger_01 [weather_context / 분노] 창밖을 보다가 갑자기 선명해졌는데, 내가 맡은 일도 아닌데 급하다고 떠넘겨서 짜증이 확 났어. => top=분노, match=1, ds=0.5310, dts=0.0000, label=주의, crisis=False
- WB504_weather_context_anger_02 [weather_context / 분노] 창밖을 보다가 갑자기 선명해졌는데, 분명히 약속한 기준을 자기 마음대로 바꿔서 화가 났어. => top=분노, match=1, ds=0.5566, dts=0.0000, label=주의, crisis=False
- WB504_weather_context_anger_03 [weather_context / 분노] 창밖을 보다가 갑자기 선명해졌는데, 말을 끝내기도 전에 결론부터 내리는 태도가 너무 거슬렸어. => top=혐오, match=0, ds=0.9853, dts=0.0000, label=위험, crisis=False
- WB504_weather_context_anger_04 [weather_context / 분노] 창밖을 보다가 갑자기 선명해졌는데, 내 시간을 당연한 것처럼 요구하는 메시지를 보고 열이 올랐어. => top=혐오, match=0, ds=0.9450, dts=0.0000, label=위험, crisis=False
- WB504_weather_context_anger_05 [weather_context / 분노] 창밖을 보다가 갑자기 선명해졌는데, 실수는 같이 했는데 설명은 나만 하라는 분위기가 억울했어. => top=분노, match=1, ds=0.5302, dts=0.0000, label=위험, crisis=False
- WB504_weather_context_anger_06 [weather_context / 분노] 창밖을 보다가 갑자기 선명해졌는데, 사과 대신 농담으로 넘기려는 모습이 계속 마음에 걸렸어. => top=혐오, match=0, ds=0.9879, dts=0.0000, label=위험, crisis=False
- WB504_weather_context_fear_01 [weather_context / 공포] 창밖을 보다가 갑자기 선명해졌는데, 밤에 골목 불이 꺼져 있어서 지나가는 동안 어깨가 굳었어. => top=공포, match=1, ds=0.5500, dts=0.0000, label=주의, crisis=False
- WB504_weather_context_fear_02 [weather_context / 공포] 창밖을 보다가 갑자기 선명해졌는데, 휴대폰에 모르는 결제 문자가 떠서 순간 손끝이 차가워졌어. => top=공포, match=1, ds=0.5350, dts=0.0000, label=주의, crisis=False
- WB504_weather_context_fear_03 [weather_context / 공포] 창밖을 보다가 갑자기 선명해졌는데, 엘리베이터 문이 한참 안 열려서 숨을 작게 쉬고 있었어. => top=중립, match=0, ds=0.3800, dts=0.0000, label=주의, crisis=False
- WB504_weather_context_fear_04 [weather_context / 공포] 창밖을 보다가 갑자기 선명해졌는데, 발표 자료가 갑자기 안 열려서 심장이 빠르게 뛰었어. => top=놀람, match=0, ds=0.5200, dts=0.0000, label=주의, crisis=False
- WB504_weather_context_fear_05 [weather_context / 공포] 창밖을 보다가 갑자기 선명해졌는데, 새벽에 현관 센서등이 켜져서 한동안 움직이지 못했어. => top=중립, match=0, ds=0.6532, dts=0.0000, label=주의, crisis=False
- WB504_weather_context_fear_06 [weather_context / 공포] 창밖을 보다가 갑자기 선명해졌는데, 병원 전화를 받기 전부터 입안이 바짝 말랐어. => top=중립, match=0, ds=0.3554, dts=0.0000, label=주의, crisis=False
- WB504_weather_context_surprise_01 [weather_context / 놀람] 창밖을 보다가 갑자기 선명해졌는데, 조용하던 프린터가 갑자기 움직여서 몸이 움찔했어. => top=놀람, match=1, ds=0.5405, dts=0.0000, label=주의, crisis=False
- WB504_weather_context_surprise_02 [weather_context / 놀람] 창밖을 보다가 갑자기 선명해졌는데, 닫힌 줄 알았던 창문이 덜컥 열려서 눈이 커졌어. => top=중립, match=0, ds=0.3800, dts=0.0000, label=주의, crisis=False
- WB504_weather_context_surprise_03 [weather_context / 놀람] 창밖을 보다가 갑자기 선명해졌는데, 회의 시간이 앞당겨졌다는 알림을 보고 잠깐 멍해졌어. => top=놀람, match=1, ds=0.5397, dts=0.0000, label=주의, crisis=False
- WB504_weather_context_surprise_04 [weather_context / 놀람] 창밖을 보다가 갑자기 선명해졌는데, 뒤에서 누가 내 이름을 불러서 바로 돌아봤어. => top=중립, match=0, ds=0.4332, dts=0.0000, label=주의, crisis=False
- WB504_weather_context_surprise_05 [weather_context / 놀람] 창밖을 보다가 갑자기 선명해졌는데, 저장해둔 파일 이름이 바뀌어 있어서 순간 당황했어. => top=놀람, match=1, ds=0.5390, dts=0.0000, label=주의, crisis=False
- WB504_weather_context_surprise_06 [weather_context / 놀람] 창밖을 보다가 갑자기 선명해졌는데, 택배 도착 사진이 예상과 달라서 한참 다시 봤어. => top=중립, match=0, ds=0.7295, dts=0.0000, label=주의, crisis=False
- WB504_weather_context_disgust_01 [weather_context / 혐오] 창밖을 보다가 갑자기 선명해졌는데, 냉장고 구석 반찬통에서 시큼한 냄새가 올라와서 바로 닫았어. => top=혐오, match=1, ds=0.5000, dts=0.0000, label=주의, crisis=False
- WB504_weather_context_disgust_02 [weather_context / 혐오] 창밖을 보다가 갑자기 선명해졌는데, 손잡이에 끈적한 자국이 묻어 있어서 만지기가 싫었어. => top=혐오, match=1, ds=0.5000, dts=0.0000, label=주의, crisis=False
- WB504_weather_context_disgust_03 [weather_context / 혐오] 창밖을 보다가 갑자기 선명해졌는데, 젖은 걸레 냄새가 방 안에 남아서 계속 인상이 찌푸려졌어. => top=혐오, match=1, ds=0.5000, dts=0.0000, label=주의, crisis=False
- WB504_weather_context_disgust_04 [weather_context / 혐오] 창밖을 보다가 갑자기 선명해졌는데, 약한 사람을 일부러 몰아붙이는 장면을 보고 속이 뒤틀렸어. => top=혐오, match=1, ds=0.9873, dts=0.0000, label=위험, crisis=False
- WB504_weather_context_disgust_05 [weather_context / 혐오] 창밖을 보다가 갑자기 선명해졌는데, 컵 바닥에 남은 찌꺼기를 보고 마실 생각이 사라졌어. => top=혐오, match=1, ds=0.5000, dts=0.0000, label=주의, crisis=False
- WB504_weather_context_disgust_06 [weather_context / 혐오] 창밖을 보다가 갑자기 선명해졌는데, 겉으로 친한 척하면서 뒤에서 험담하는 말을 들으니 역했어. => top=혐오, match=1, ds=0.9850, dts=0.0000, label=위험, crisis=False
- WB504_weather_context_neutral_01 [weather_context / 중립] 창밖을 보다가 갑자기 선명해졌는데, 노트북 충전 케이블을 책상 오른쪽으로 옮겨뒀어. => top=중립, match=1, ds=0.3719, dts=0.0000, label=보통, crisis=False
- WB504_weather_context_neutral_02 [weather_context / 중립] 창밖을 보다가 갑자기 선명해졌는데, 내일 가져갈 서류를 투명 파일에 넣어뒀어. => top=중립, match=1, ds=0.3800, dts=0.0000, label=보통, crisis=False
- WB504_weather_context_neutral_03 [weather_context / 중립] 창밖을 보다가 갑자기 선명해졌는데, 점심 후보를 세 곳으로 줄여서 메모장에 적었어. => top=중립, match=1, ds=0.3654, dts=0.0000, label=보통, crisis=False
- WB504_weather_context_neutral_04 [weather_context / 중립] 창밖을 보다가 갑자기 선명해졌는데, 세탁 완료 알림을 보고 빨래를 건조대에 널었어. => top=중립, match=1, ds=0.3800, dts=0.0000, label=보통, crisis=False
- WB504_weather_context_neutral_05 [weather_context / 중립] 창밖을 보다가 갑자기 선명해졌는데, 달력에 이번 주 분리수거 날짜를 다시 표시했어. => top=중립, match=1, ds=0.3559, dts=0.0000, label=보통, crisis=False
- WB504_weather_context_neutral_06 [weather_context / 중립] 창밖을 보다가 갑자기 선명해졌는데, 이어폰 케이스 배터리를 확인하고 가방에 넣었어. => top=중립, match=1, ds=0.3800, dts=0.0000, label=보통, crisis=False
- WB504_errand_note_happy_01 [errand_note / 행복] 볼일 보러 나갔다 돌아오는 길에 아침에 만든 계란말이가 모양까지 잘 나와서 괜히 뿌듯했어. => top=행복, match=1, ds=0.3274, dts=0.0000, label=보통, crisis=False
- WB504_errand_note_happy_02 [errand_note / 행복] 볼일 보러 나갔다 돌아오는 길에 길에서 좋아하던 노래가 들려서 발걸음이 조금 빨라졌어. => top=행복, match=1, ds=0.3450, dts=0.0000, label=보통, crisis=False
- WB504_errand_note_happy_03 [errand_note / 행복] 볼일 보러 나갔다 돌아오는 길에 친구가 내 농담에 크게 웃어줘서 마음이 환해졌어. => top=행복, match=1, ds=0.5403, dts=0.0000, label=보통, crisis=False
- WB504_errand_note_happy_04 [errand_note / 행복] 볼일 보러 나갔다 돌아오는 길에 오래 찾던 자료를 드디어 발견해서 속이 시원했어. => top=행복, match=1, ds=0.3280, dts=0.0000, label=보통, crisis=False
- WB504_errand_note_happy_05 [errand_note / 행복] 볼일 보러 나갔다 돌아오는 길에 퇴근길 하늘색이 예뻐서 사진을 한 장 남겼어. => top=행복, match=1, ds=0.3294, dts=0.0000, label=보통, crisis=False
- WB504_errand_note_happy_06 [errand_note / 행복] 볼일 보러 나갔다 돌아오는 길에 오늘은 계획한 만큼만 해도 충분하다는 생각이 들어서 편했어. => top=중립, match=0, ds=0.4451, dts=0.0000, label=보통, crisis=False
- WB504_errand_note_sad_01 [errand_note / 슬픔] 볼일 보러 나갔다 돌아오는 길에 요즘은 좋아하던 일도 손이 안 가고 하루가 그냥 지나가. => top=슬픔, match=1, ds=0.5702, dts=0.7700, label=주의, crisis=False
- WB504_errand_note_sad_02 [errand_note / 슬픔] 볼일 보러 나갔다 돌아오는 길에 며칠째 잠을 자도 개운하지 않고 몸이 무겁게 가라앉아. => top=슬픔, match=1, ds=0.5500, dts=0.6750, label=주의, crisis=False
- WB504_errand_note_sad_03 [errand_note / 슬픔] 볼일 보러 나갔다 돌아오는 길에 내가 있어도 없어도 별 차이 없을 것 같다는 생각이 반복돼. => top=중립, match=0, ds=0.4410, dts=0.0000, label=주의, crisis=False
- WB504_errand_note_sad_04 [errand_note / 슬픔] 볼일 보러 나갔다 돌아오는 길에 기대했던 모임이 취소돼서 집에 오니 마음이 푹 꺼졌어. => top=행복, match=0, ds=0.3450, dts=0.0000, label=주의, crisis=False
- WB504_errand_note_sad_05 [errand_note / 슬픔] 볼일 보러 나갔다 돌아오는 길에 괜찮다고 말했는데 막상 혼자 있으니 서운함이 남았어. => top=슬픔, match=1, ds=0.6794, dts=0.2000, label=주의, crisis=False
- WB504_errand_note_sad_06 [errand_note / 슬픔] 볼일 보러 나갔다 돌아오는 길에 비 오는 소리를 듣다 보니 괜히 옛날 생각이 나서 먹먹했어. => top=슬픔, match=1, ds=0.5774, dts=0.2000, label=주의, crisis=False
- WB504_errand_note_anger_01 [errand_note / 분노] 볼일 보러 나갔다 돌아오는 길에 내가 맡은 일도 아닌데 급하다고 떠넘겨서 짜증이 확 났어. => top=분노, match=1, ds=0.5349, dts=0.0000, label=주의, crisis=False
- WB504_errand_note_anger_02 [errand_note / 분노] 볼일 보러 나갔다 돌아오는 길에 분명히 약속한 기준을 자기 마음대로 바꿔서 화가 났어. => top=분노, match=1, ds=0.5336, dts=0.0000, label=주의, crisis=False
- WB504_errand_note_anger_03 [errand_note / 분노] 볼일 보러 나갔다 돌아오는 길에 말을 끝내기도 전에 결론부터 내리는 태도가 너무 거슬렸어. => top=중립, match=0, ds=0.4229, dts=0.0000, label=주의, crisis=False
- WB504_errand_note_anger_04 [errand_note / 분노] 볼일 보러 나갔다 돌아오는 길에 내 시간을 당연한 것처럼 요구하는 메시지를 보고 열이 올랐어. => top=분노, match=1, ds=0.8484, dts=0.0000, label=위험, crisis=False
- WB504_errand_note_anger_05 [errand_note / 분노] 볼일 보러 나갔다 돌아오는 길에 실수는 같이 했는데 설명은 나만 하라는 분위기가 억울했어. => top=분노, match=1, ds=0.5332, dts=0.0000, label=주의, crisis=False
- WB504_errand_note_anger_06 [errand_note / 분노] 볼일 보러 나갔다 돌아오는 길에 사과 대신 농담으로 넘기려는 모습이 계속 마음에 걸렸어. => top=혐오, match=0, ds=0.8551, dts=0.0000, label=위험, crisis=False
- WB504_errand_note_fear_01 [errand_note / 공포] 볼일 보러 나갔다 돌아오는 길에 밤에 골목 불이 꺼져 있어서 지나가는 동안 어깨가 굳었어. => top=공포, match=1, ds=0.5460, dts=0.0000, label=주의, crisis=False
- WB504_errand_note_fear_02 [errand_note / 공포] 볼일 보러 나갔다 돌아오는 길에 휴대폰에 모르는 결제 문자가 떠서 순간 손끝이 차가워졌어. => top=공포, match=1, ds=0.5369, dts=0.0000, label=주의, crisis=False
- WB504_errand_note_fear_03 [errand_note / 공포] 볼일 보러 나갔다 돌아오는 길에 엘리베이터 문이 한참 안 열려서 숨을 작게 쉬고 있었어. => top=중립, match=0, ds=0.3800, dts=0.0000, label=주의, crisis=False
- WB504_errand_note_fear_04 [errand_note / 공포] 볼일 보러 나갔다 돌아오는 길에 발표 자료가 갑자기 안 열려서 심장이 빠르게 뛰었어. => top=놀람, match=0, ds=0.5200, dts=0.0000, label=주의, crisis=False
- WB504_errand_note_fear_05 [errand_note / 공포] 볼일 보러 나갔다 돌아오는 길에 새벽에 현관 센서등이 켜져서 한동안 움직이지 못했어. => top=슬픔, match=0, ds=0.6909, dts=0.2000, label=주의, crisis=False
- WB504_errand_note_fear_06 [errand_note / 공포] 볼일 보러 나갔다 돌아오는 길에 병원 전화를 받기 전부터 입안이 바짝 말랐어. => top=분노, match=0, ds=0.8681, dts=0.0000, label=위험, crisis=False
- WB504_errand_note_surprise_01 [errand_note / 놀람] 볼일 보러 나갔다 돌아오는 길에 조용하던 프린터가 갑자기 움직여서 몸이 움찔했어. => top=놀람, match=1, ds=0.5412, dts=0.0000, label=주의, crisis=False
- WB504_errand_note_surprise_02 [errand_note / 놀람] 볼일 보러 나갔다 돌아오는 길에 닫힌 줄 알았던 창문이 덜컥 열려서 눈이 커졌어. => top=중립, match=0, ds=0.3800, dts=0.0000, label=주의, crisis=False
- WB504_errand_note_surprise_03 [errand_note / 놀람] 볼일 보러 나갔다 돌아오는 길에 회의 시간이 앞당겨졌다는 알림을 보고 잠깐 멍해졌어. => top=놀람, match=1, ds=0.5406, dts=0.0000, label=주의, crisis=False
- WB504_errand_note_surprise_04 [errand_note / 놀람] 볼일 보러 나갔다 돌아오는 길에 뒤에서 누가 내 이름을 불러서 바로 돌아봤어. => top=중립, match=0, ds=0.4263, dts=0.0000, label=주의, crisis=False
- WB504_errand_note_surprise_05 [errand_note / 놀람] 볼일 보러 나갔다 돌아오는 길에 저장해둔 파일 이름이 바뀌어 있어서 순간 당황했어. => top=놀람, match=1, ds=0.5399, dts=0.0000, label=주의, crisis=False
- WB504_errand_note_surprise_06 [errand_note / 놀람] 볼일 보러 나갔다 돌아오는 길에 택배 도착 사진이 예상과 달라서 한참 다시 봤어. => top=중립, match=0, ds=0.7379, dts=0.0000, label=주의, crisis=False
- WB504_errand_note_disgust_01 [errand_note / 혐오] 볼일 보러 나갔다 돌아오는 길에 냉장고 구석 반찬통에서 시큼한 냄새가 올라와서 바로 닫았어. => top=혐오, match=1, ds=0.5000, dts=0.0000, label=주의, crisis=False
- WB504_errand_note_disgust_02 [errand_note / 혐오] 볼일 보러 나갔다 돌아오는 길에 손잡이에 끈적한 자국이 묻어 있어서 만지기가 싫었어. => top=혐오, match=1, ds=0.5000, dts=0.0000, label=주의, crisis=False
- WB504_errand_note_disgust_03 [errand_note / 혐오] 볼일 보러 나갔다 돌아오는 길에 젖은 걸레 냄새가 방 안에 남아서 계속 인상이 찌푸려졌어. => top=혐오, match=1, ds=0.5000, dts=0.0000, label=주의, crisis=False
- WB504_errand_note_disgust_04 [errand_note / 혐오] 볼일 보러 나갔다 돌아오는 길에 약한 사람을 일부러 몰아붙이는 장면을 보고 속이 뒤틀렸어. => top=혐오, match=1, ds=0.9534, dts=0.0000, label=위험, crisis=False
- WB504_errand_note_disgust_05 [errand_note / 혐오] 볼일 보러 나갔다 돌아오는 길에 컵 바닥에 남은 찌꺼기를 보고 마실 생각이 사라졌어. => top=혐오, match=1, ds=0.5000, dts=0.0000, label=주의, crisis=False
- WB504_errand_note_disgust_06 [errand_note / 혐오] 볼일 보러 나갔다 돌아오는 길에 겉으로 친한 척하면서 뒤에서 험담하는 말을 들으니 역했어. => top=중립, match=0, ds=0.4332, dts=0.0000, label=주의, crisis=False
- WB504_errand_note_neutral_01 [errand_note / 중립] 볼일 보러 나갔다 돌아오는 길에 노트북 충전 케이블을 책상 오른쪽으로 옮겨뒀어. => top=중립, match=1, ds=0.3800, dts=0.0000, label=보통, crisis=False
- WB504_errand_note_neutral_02 [errand_note / 중립] 볼일 보러 나갔다 돌아오는 길에 내일 가져갈 서류를 투명 파일에 넣어뒀어. => top=중립, match=1, ds=0.3800, dts=0.0000, label=보통, crisis=False
- WB504_errand_note_neutral_03 [errand_note / 중립] 볼일 보러 나갔다 돌아오는 길에 점심 후보를 세 곳으로 줄여서 메모장에 적었어. => top=중립, match=1, ds=0.3800, dts=0.0000, label=보통, crisis=False
- WB504_errand_note_neutral_04 [errand_note / 중립] 볼일 보러 나갔다 돌아오는 길에 세탁 완료 알림을 보고 빨래를 건조대에 널었어. => top=중립, match=1, ds=0.3633, dts=0.0000, label=보통, crisis=False
- WB504_errand_note_neutral_05 [errand_note / 중립] 볼일 보러 나갔다 돌아오는 길에 달력에 이번 주 분리수거 날짜를 다시 표시했어. => top=중립, match=1, ds=0.3788, dts=0.0000, label=보통, crisis=False
- WB504_errand_note_neutral_06 [errand_note / 중립] 볼일 보러 나갔다 돌아오는 길에 이어폰 케이스 배터리를 확인하고 가방에 넣었어. => top=중립, match=1, ds=0.3800, dts=0.0000, label=보통, crisis=False
- WB504_polite_update_happy_01 [polite_update / 행복] 오늘 상태를 짧게 정리하면 이런 느낌입니다: 아침에 만든 계란말이가 모양까지 잘 나와서 괜히 뿌듯했어. => top=행복, match=1, ds=0.3252, dts=0.0000, label=보통, crisis=False
- WB504_polite_update_happy_02 [polite_update / 행복] 오늘 상태를 짧게 정리하면 이런 느낌입니다: 길에서 좋아하던 노래가 들려서 발걸음이 조금 빨라졌어. => top=행복, match=1, ds=0.3303, dts=0.0000, label=보통, crisis=False
- WB504_polite_update_happy_03 [polite_update / 행복] 오늘 상태를 짧게 정리하면 이런 느낌입니다: 친구가 내 농담에 크게 웃어줘서 마음이 환해졌어. => top=중립, match=0, ds=0.3615, dts=0.0000, label=보통, crisis=False
- WB504_polite_update_happy_04 [polite_update / 행복] 오늘 상태를 짧게 정리하면 이런 느낌입니다: 오래 찾던 자료를 드디어 발견해서 속이 시원했어. => top=행복, match=1, ds=0.3347, dts=0.0000, label=보통, crisis=False
- WB504_polite_update_happy_05 [polite_update / 행복] 오늘 상태를 짧게 정리하면 이런 느낌입니다: 퇴근길 하늘색이 예뻐서 사진을 한 장 남겼어. => top=행복, match=1, ds=0.3450, dts=0.0000, label=보통, crisis=False
- WB504_polite_update_happy_06 [polite_update / 행복] 오늘 상태를 짧게 정리하면 이런 느낌입니다: 오늘은 계획한 만큼만 해도 충분하다는 생각이 들어서 편했어. => top=중립, match=0, ds=0.3635, dts=0.0000, label=보통, crisis=False
- WB504_polite_update_sad_01 [polite_update / 슬픔] 오늘 상태를 짧게 정리하면 이런 느낌입니다: 요즘은 좋아하던 일도 손이 안 가고 하루가 그냥 지나가. => top=슬픔, match=1, ds=0.5378, dts=0.7700, label=주의, crisis=False
- WB504_polite_update_sad_02 [polite_update / 슬픔] 오늘 상태를 짧게 정리하면 이런 느낌입니다: 며칠째 잠을 자도 개운하지 않고 몸이 무겁게 가라앉아. => top=슬픔, match=1, ds=0.5392, dts=0.6750, label=주의, crisis=False
- WB504_polite_update_sad_03 [polite_update / 슬픔] 오늘 상태를 짧게 정리하면 이런 느낌입니다: 내가 있어도 없어도 별 차이 없을 것 같다는 생각이 반복돼. => top=중립, match=0, ds=0.3800, dts=0.0000, label=주의, crisis=False
- WB504_polite_update_sad_04 [polite_update / 슬픔] 오늘 상태를 짧게 정리하면 이런 느낌입니다: 기대했던 모임이 취소돼서 집에 오니 마음이 푹 꺼졌어. => top=행복, match=0, ds=0.3450, dts=0.0000, label=주의, crisis=False
- WB504_polite_update_sad_05 [polite_update / 슬픔] 오늘 상태를 짧게 정리하면 이런 느낌입니다: 괜찮다고 말했는데 막상 혼자 있으니 서운함이 남았어. => top=슬픔, match=1, ds=0.8417, dts=0.2000, label=주의, crisis=False
- WB504_polite_update_sad_06 [polite_update / 슬픔] 오늘 상태를 짧게 정리하면 이런 느낌입니다: 비 오는 소리를 듣다 보니 괜히 옛날 생각이 나서 먹먹했어. => top=슬픔, match=1, ds=0.6381, dts=0.2000, label=주의, crisis=False
- WB504_polite_update_anger_01 [polite_update / 분노] 오늘 상태를 짧게 정리하면 이런 느낌입니다: 내가 맡은 일도 아닌데 급하다고 떠넘겨서 짜증이 확 났어. => top=분노, match=1, ds=0.5377, dts=0.0000, label=주의, crisis=False
- WB504_polite_update_anger_02 [polite_update / 분노] 오늘 상태를 짧게 정리하면 이런 느낌입니다: 분명히 약속한 기준을 자기 마음대로 바꿔서 화가 났어. => top=분노, match=1, ds=0.5404, dts=0.0000, label=주의, crisis=False
- WB504_polite_update_anger_03 [polite_update / 분노] 오늘 상태를 짧게 정리하면 이런 느낌입니다: 말을 끝내기도 전에 결론부터 내리는 태도가 너무 거슬렸어. => top=중립, match=0, ds=0.3746, dts=0.0000, label=주의, crisis=False
- WB504_polite_update_anger_04 [polite_update / 분노] 오늘 상태를 짧게 정리하면 이런 느낌입니다: 내 시간을 당연한 것처럼 요구하는 메시지를 보고 열이 올랐어. => top=중립, match=0, ds=0.3625, dts=0.0000, label=주의, crisis=False
- WB504_polite_update_anger_05 [polite_update / 분노] 오늘 상태를 짧게 정리하면 이런 느낌입니다: 실수는 같이 했는데 설명은 나만 하라는 분위기가 억울했어. => top=분노, match=1, ds=0.5293, dts=0.0000, label=주의, crisis=False
- WB504_polite_update_anger_06 [polite_update / 분노] 오늘 상태를 짧게 정리하면 이런 느낌입니다: 사과 대신 농담으로 넘기려는 모습이 계속 마음에 걸렸어. => top=혐오, match=0, ds=0.9832, dts=0.0000, label=위험, crisis=False
- WB504_polite_update_fear_01 [polite_update / 공포] 오늘 상태를 짧게 정리하면 이런 느낌입니다: 밤에 골목 불이 꺼져 있어서 지나가는 동안 어깨가 굳었어. => top=공포, match=1, ds=0.5500, dts=0.0000, label=주의, crisis=False
- WB504_polite_update_fear_02 [polite_update / 공포] 오늘 상태를 짧게 정리하면 이런 느낌입니다: 휴대폰에 모르는 결제 문자가 떠서 순간 손끝이 차가워졌어. => top=공포, match=1, ds=0.5360, dts=0.0000, label=주의, crisis=False
- WB504_polite_update_fear_03 [polite_update / 공포] 오늘 상태를 짧게 정리하면 이런 느낌입니다: 엘리베이터 문이 한참 안 열려서 숨을 작게 쉬고 있었어. => top=중립, match=0, ds=0.3800, dts=0.0000, label=주의, crisis=False
- WB504_polite_update_fear_04 [polite_update / 공포] 오늘 상태를 짧게 정리하면 이런 느낌입니다: 발표 자료가 갑자기 안 열려서 심장이 빠르게 뛰었어. => top=놀람, match=0, ds=0.5200, dts=0.0000, label=주의, crisis=False
- WB504_polite_update_fear_05 [polite_update / 공포] 오늘 상태를 짧게 정리하면 이런 느낌입니다: 새벽에 현관 센서등이 켜져서 한동안 움직이지 못했어. => top=중립, match=0, ds=0.3800, dts=0.0000, label=주의, crisis=False
- WB504_polite_update_fear_06 [polite_update / 공포] 오늘 상태를 짧게 정리하면 이런 느낌입니다: 병원 전화를 받기 전부터 입안이 바짝 말랐어. => top=중립, match=0, ds=0.3800, dts=0.0000, label=주의, crisis=False
- WB504_polite_update_surprise_01 [polite_update / 놀람] 오늘 상태를 짧게 정리하면 이런 느낌입니다: 조용하던 프린터가 갑자기 움직여서 몸이 움찔했어. => top=놀람, match=1, ds=0.5190, dts=0.0000, label=주의, crisis=False
- WB504_polite_update_surprise_02 [polite_update / 놀람] 오늘 상태를 짧게 정리하면 이런 느낌입니다: 닫힌 줄 알았던 창문이 덜컥 열려서 눈이 커졌어. => top=중립, match=0, ds=0.3800, dts=0.0000, label=주의, crisis=False
- WB504_polite_update_surprise_03 [polite_update / 놀람] 오늘 상태를 짧게 정리하면 이런 느낌입니다: 회의 시간이 앞당겨졌다는 알림을 보고 잠깐 멍해졌어. => top=놀람, match=1, ds=0.5116, dts=0.0000, label=주의, crisis=False
- WB504_polite_update_surprise_04 [polite_update / 놀람] 오늘 상태를 짧게 정리하면 이런 느낌입니다: 뒤에서 누가 내 이름을 불러서 바로 돌아봤어. => top=중립, match=0, ds=0.3800, dts=0.0000, label=주의, crisis=False
- WB504_polite_update_surprise_05 [polite_update / 놀람] 오늘 상태를 짧게 정리하면 이런 느낌입니다: 저장해둔 파일 이름이 바뀌어 있어서 순간 당황했어. => top=놀람, match=1, ds=0.5200, dts=0.0000, label=주의, crisis=False
- WB504_polite_update_surprise_06 [polite_update / 놀람] 오늘 상태를 짧게 정리하면 이런 느낌입니다: 택배 도착 사진이 예상과 달라서 한참 다시 봤어. => top=중립, match=0, ds=0.3800, dts=0.0000, label=주의, crisis=False
- WB504_polite_update_disgust_01 [polite_update / 혐오] 오늘 상태를 짧게 정리하면 이런 느낌입니다: 냉장고 구석 반찬통에서 시큼한 냄새가 올라와서 바로 닫았어. => top=혐오, match=1, ds=0.4850, dts=0.0000, label=주의, crisis=False
- WB504_polite_update_disgust_02 [polite_update / 혐오] 오늘 상태를 짧게 정리하면 이런 느낌입니다: 손잡이에 끈적한 자국이 묻어 있어서 만지기가 싫었어. => top=혐오, match=1, ds=0.5000, dts=0.0000, label=주의, crisis=False
- WB504_polite_update_disgust_03 [polite_update / 혐오] 오늘 상태를 짧게 정리하면 이런 느낌입니다: 젖은 걸레 냄새가 방 안에 남아서 계속 인상이 찌푸려졌어. => top=혐오, match=1, ds=0.5000, dts=0.0000, label=주의, crisis=False
- WB504_polite_update_disgust_04 [polite_update / 혐오] 오늘 상태를 짧게 정리하면 이런 느낌입니다: 약한 사람을 일부러 몰아붙이는 장면을 보고 속이 뒤틀렸어. => top=중립, match=0, ds=0.3800, dts=0.0000, label=주의, crisis=False
- WB504_polite_update_disgust_05 [polite_update / 혐오] 오늘 상태를 짧게 정리하면 이런 느낌입니다: 컵 바닥에 남은 찌꺼기를 보고 마실 생각이 사라졌어. => top=혐오, match=1, ds=0.4850, dts=0.0000, label=주의, crisis=False
- WB504_polite_update_disgust_06 [polite_update / 혐오] 오늘 상태를 짧게 정리하면 이런 느낌입니다: 겉으로 친한 척하면서 뒤에서 험담하는 말을 들으니 역했어. => top=중립, match=0, ds=0.3800, dts=0.0000, label=주의, crisis=False
- WB504_polite_update_neutral_01 [polite_update / 중립] 오늘 상태를 짧게 정리하면 이런 느낌입니다: 노트북 충전 케이블을 책상 오른쪽으로 옮겨뒀어. => top=중립, match=1, ds=0.3800, dts=0.0000, label=보통, crisis=False
- WB504_polite_update_neutral_02 [polite_update / 중립] 오늘 상태를 짧게 정리하면 이런 느낌입니다: 내일 가져갈 서류를 투명 파일에 넣어뒀어. => top=중립, match=1, ds=0.3800, dts=0.0000, label=보통, crisis=False
- WB504_polite_update_neutral_03 [polite_update / 중립] 오늘 상태를 짧게 정리하면 이런 느낌입니다: 점심 후보를 세 곳으로 줄여서 메모장에 적었어. => top=중립, match=1, ds=0.3800, dts=0.0000, label=보통, crisis=False
- WB504_polite_update_neutral_04 [polite_update / 중립] 오늘 상태를 짧게 정리하면 이런 느낌입니다: 세탁 완료 알림을 보고 빨래를 건조대에 널었어. => top=중립, match=1, ds=0.3800, dts=0.0000, label=보통, crisis=False
- WB504_polite_update_neutral_05 [polite_update / 중립] 오늘 상태를 짧게 정리하면 이런 느낌입니다: 달력에 이번 주 분리수거 날짜를 다시 표시했어. => top=중립, match=1, ds=0.3800, dts=0.0000, label=보통, crisis=False
- WB504_polite_update_neutral_06 [polite_update / 중립] 오늘 상태를 짧게 정리하면 이런 느낌입니다: 이어폰 케이스 배터리를 확인하고 가방에 넣었어. => top=중립, match=1, ds=0.3800, dts=0.0000, label=보통, crisis=False