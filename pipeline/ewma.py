"""
ewma.py
역할: 발화 단위 depression_score를 2단계 EWMA로 평활화
      1단계: 발화 → 일별 점수 (alpha=0.3, 초기값=첫 발화 점수)
      2단계: 일별 → 다일 이동 평균 (alpha=0.3)
입력: 발화 점수 시퀀스 또는 일별 점수 시퀀스
출력: 평활화된 점수 (float)
"""

ALPHA = 0.3  # EWMA 감쇠 계수


def ewma_update(prev: float, new_value: float, alpha: float = ALPHA) -> float:
    """
    역할: EWMA 단일 스텝 업데이트
    입력: 이전 평활값, 새 관측값, 감쇠 계수
    출력: 업데이트된 평활값
    """
    return alpha * new_value + (1.0 - alpha) * prev


def utterance_to_daily(utterance_scores: list[float], alpha: float = ALPHA) -> float:
    """
    역할: 1단계 — 같은 날 여러 발화 점수를 EWMA로 합산해 일별 점수 산출
          초기값은 첫 발화 점수 (warm-start)
    입력: 해당 일의 발화 점수 리스트 (시간순), 감쇠 계수
    출력: 해당 일의 daily_score (float)
    """
    if not utterance_scores:
        return 0.0

    score = utterance_scores[0]
    for s in utterance_scores[1:]:
        score = ewma_update(score, s, alpha)
    return float(score)


def daily_to_smoothed(daily_scores: list[float], alpha: float = ALPHA) -> list[float]:
    """
    역할: 2단계 — 날짜별 daily_score 시퀀스를 EWMA로 평활화
          초기값은 첫 날 daily_score
    입력: 날짜별 daily_score 리스트 (날짜 오름차순), 감쇠 계수
    출력: 평활화된 daily_score 리스트 (같은 길이)
    """
    if not daily_scores:
        return []

    smoothed = [daily_scores[0]]
    for s in daily_scores[1:]:
        smoothed.append(ewma_update(smoothed[-1], s, alpha))
    return smoothed
