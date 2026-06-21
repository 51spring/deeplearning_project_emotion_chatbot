"""
wellness_score.py
역할: depression_score → wellness_score 변환 + 위험 레이블 결정
      14일 미만: 절대 기준 레이블
      14일 이상: 개인 퍼센타일 기준 (14~20일 완충 기간 — 불일치 시 절대 기준 우선)
입력: depression_score (float, 0~1), 과거 daily_score 리스트
출력: wellness_score (float, 0~100), label (str)
"""

# 절대 기준 임계값 (wellness_score 기준)
# 2026-06-18 재보정: 평활 누적 웰니스가 중앙(~65)에 압축되는 분포에 맞춰 80/60→70/55로 좁힘.
LABEL_THRESHOLDS = {
    "위험":   40,   # wellness < 40
    "주의":   55,   # 40 ≤ wellness < 55
    "보통":   70,   # 55 ≤ wellness < 70
    "양호":  100,   # wellness ≥ 70
}

# 퍼센타일 기준 (개인 히스토리 분포 기반)
PERCENTILE_THRESHOLDS = {
    "위험":  15,    # 하위 15퍼센타일 이하
    "주의":  35,    # 15~35퍼센타일
    "보통":  65,    # 35~65퍼센타일
    "양호": 100,    # 65퍼센타일 초과
}

# 퍼센타일 기준 전환 일수 (2026-06-18: 개인 보정을 더 일찍 적용하려 30→14일로 단축)
PERCENTILE_SWITCH_DAY  = 14
PERCENTILE_BUFFER_END  = 20  # 완충 기간 종료일 (14~20일은 불일치 시 절대 기준 우선)

# 정서 모니터링 대상 발화가 없는 날의 기준점.
# "우울 신호 없음"을 완전 양호(100점)로 보지 않고 기본 상태(65점)로 둔다.
# 이 값은 score_policy의 중립/저신호 baseline으로도 파생되어 함께 65 기준으로 움직인다.
NO_SIGNAL_DEPRESSION_SCORE = 0.35


def apply_no_signal_floor(depression_score: float | None) -> float:
    """
    역할: 정서 신호가 없는 날이 wellness 100으로 튀지 않도록 최소 기준 점수 적용
    입력: depression_score 또는 None
    출력: 최소 기준을 반영한 depression_score
    """
    if depression_score is None:
        return NO_SIGNAL_DEPRESSION_SCORE
    return max(float(depression_score), NO_SIGNAL_DEPRESSION_SCORE)


def depression_to_wellness(depression_score: float) -> float:
    """
    역할: depression_score (0~1) → wellness_score (0~100) 변환
    입력: depression_score
    출력: wellness_score
    """
    return float((1.0 - depression_score) * 100.0)


def depression_to_display_wellness(depression_score: float | None) -> float | None:
    """
    역할: 저장된 종합 distress 점수를 화면 표시용 웰니스 점수(0~100)로 안전하게 변환
    입력: depression_score 또는 None
    출력: 반올림된 wellness_score, 값이 없으면 None
    """
    if depression_score is None:
        return None
    wellness = depression_to_wellness(float(depression_score))
    # 과거 백필/실험 값이 범위를 벗어나도 UI 점수는 0~100 안에 머물게 한다.
    wellness = max(0.0, min(100.0, wellness))
    return round(wellness, 2)


def display_wellness_label(wellness_score: float | None) -> str | None:
    """
    역할: 화면 표시용 웰니스 점수(0~100) 하나만 보고 절대 기준 상태 레이블 계산
    입력: wellness_score 또는 None
    출력: "양호" | "보통" | "주의" | "위험" 또는 None
    """
    if wellness_score is None:
        return None
    wellness = max(0.0, min(100.0, float(wellness_score)))
    return _label_absolute(wellness)


def depression_to_display_label(depression_score: float | None) -> str | None:
    """
    역할: 저장된 종합 distress 점수를 단일 표시 웰니스 레이블로 변환
    입력: depression_score 또는 None
    출력: 표시용 웰니스 점수 기준 레이블, 값이 없으면 None
    """
    return display_wellness_label(depression_to_display_wellness(depression_score))


def _label_absolute(wellness: float) -> str:
    """절대 기준 레이블 결정"""
    if wellness < LABEL_THRESHOLDS["위험"]:
        return "위험"
    if wellness < LABEL_THRESHOLDS["주의"]:
        return "주의"
    if wellness < LABEL_THRESHOLDS["보통"]:
        return "보통"
    return "양호"


def _label_percentile(wellness: float, history: list[float]) -> str:
    """
    역할: 개인 퍼센타일 기준 레이블 결정
    입력: 오늘의 wellness_score, 과거 wellness_score 리스트
    출력: 레이블 str
    """
    import numpy as np
    if len(history) < 2:
        return _label_absolute(wellness)

    pct = float(np.mean(np.array(history) <= wellness) * 100)

    if pct <= PERCENTILE_THRESHOLDS["위험"]:
        return "위험"
    if pct <= PERCENTILE_THRESHOLDS["주의"]:
        return "주의"
    if pct <= PERCENTILE_THRESHOLDS["보통"]:
        return "보통"
    return "양호"


def determine_label(
    wellness: float,
    history_wellness: list[float],
    n_days: int,
) -> str:
    """
    역할: 누적 데이터 일수에 따라 절대/퍼센타일 레이블 선택
          14일 미만: 절대 기준
          14~20일(완충): 두 기준 불일치 시 절대 기준 우선
          21일 이상: 퍼센타일 기준
    입력: 오늘 wellness_score, 과거 wellness_score 리스트, 누적 데이터 일수
    출력: 레이블 str
    """
    label_abs = _label_absolute(wellness)

    if n_days < PERCENTILE_SWITCH_DAY:
        return label_abs

    label_pct = _label_percentile(wellness, history_wellness)

    if n_days <= PERCENTILE_BUFFER_END:
        # 완충 기간: 불일치 시 절대 기준 우선
        return label_abs if label_abs != label_pct else label_pct

    return label_pct


def compute_wellness(
    depression_score: float,
    history_wellness: list[float] | None = None,
    n_days: int = 0,
) -> dict:
    """
    역할: depression_score를 wellness_score + 레이블로 변환하는 최종 함수
    입력: depression_score (0~1), 과거 wellness_score 리스트, 누적 일수
    출력: {wellness_score: float, label: str}
    """
    history = history_wellness or []
    wellness = depression_to_wellness(depression_score)
    label    = determine_label(wellness, history, n_days)

    return {
        "wellness_score": round(wellness, 2),
        "label":          label,
    }
