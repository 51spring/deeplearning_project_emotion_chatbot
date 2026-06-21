"""
daily_summary.py
역할: 하루 종료 시 EWMA 집계 → DailySummary DB 저장
      파이프라인 close_day() 결과를 받아 DB에 기록
입력: user_id, date_str, 파이프라인 close_day() 반환 dict, 발화 수, 위기 수
출력: DailySummary ORM 객체
"""

from sqlalchemy.orm import Session as DBSession
from backend.db import crud
from backend.db.models import DailySummary
from pipeline.wellness_score import depression_to_display_label, depression_to_display_wellness


def save_day(
    db: DBSession,
    user_id: int,
    date_str: str,
    pipeline_result: dict,
    utterance_count: int,
    crisis_count: int,
) -> DailySummary:
    """
    역할: 파이프라인 close_day() 결과를 daily_summaries 테이블에 저장
    입력: DB 세션, user_id, 날짜 문자열,
          pipeline_result (daily_score, smoothed_score, wellness_score, label),
          utterance_count (해당일 발화 수), crisis_count (해당일 위기 이벤트 수)
    출력: DailySummary ORM 객체
    """
    data = {
        "daily_score":      pipeline_result["daily_score"],
        "smoothed_score":   pipeline_result["smoothed_score"],
        "wellness_score":   pipeline_result["wellness_score"],
        "label":            pipeline_result["label"],
        "depression_tendency_daily":    pipeline_result.get("depression_tendency_daily"),
        "depression_tendency_smoothed": pipeline_result.get("depression_tendency_smoothed"),
        "utterance_count":  utterance_count,
        "crisis_count_day": crisis_count,
    }
    return crud.save_daily_summary(db, user_id, date_str, data)


def _truncate_text(text: str, max_chars: int = 90) -> str:
    """
    역할: 기록 화면 표시용 문장을 일정 길이로 축약
    입력: 원문 텍스트, 최대 글자 수
    출력: 축약된 텍스트
    """
    cleaned = " ".join(str(text).split())
    if len(cleaned) <= max_chars:
        return cleaned
    return f"{cleaned[:max_chars].rstrip()}..."


def _build_rule_based_summary(summary: DailySummary, user_utts: list) -> str:
    """
    역할: Qwen 서사 요약이 없을 때 기록 화면용 하루 요약 생성
    입력: DailySummary ORM 객체, 사용자 발화 ORM 리스트
    출력: 짧은 한국어 요약 문자열
    """
    if not user_utts:
        return "저장된 사용자 발화가 없어 점수 기록만 남아 있습니다."

    emotion_counts: dict[str, int] = {}
    scored_count = 0
    for utt in user_utts:
        if utt.top_emotion:
            emotion_counts[utt.top_emotion] = emotion_counts.get(utt.top_emotion, 0) + 1
        if utt.depression_score is not None:
            scored_count += 1

    top_emotion = max(emotion_counts, key=emotion_counts.get) if emotion_counts else None
    latest_text = next((utt.text for utt in reversed(user_utts) if utt.text), "")

    display_count = summary.utterance_count if summary.utterance_count is not None else len(user_utts)
    parts = [
        f"오늘은 {display_count}개의 대화가 기록됐고 누적 상태는 {summary.label}입니다."
    ]
    if top_emotion:
        parts.append(f"가장 자주 감지된 감정은 {top_emotion}입니다.")
    if scored_count != len(user_utts):
        parts.append(f"점수에는 정서 모니터링 대상 발화 {scored_count}개가 반영됐습니다.")
    if latest_text:
        parts.append(f"마지막으로 남긴 말: \"{_truncate_text(latest_text)}\"")
    return " ".join(parts)


def build_record_summary_text(db: DBSession, user_id: int, daily: DailySummary) -> str:
    """
    역할: 기록 화면에 표시할 하루 요약 문장 생성
    입력: DB 세션, user_id, DailySummary ORM 객체
    출력: Qwen 서사 요약 또는 규칙 기반 요약 문자열
    """
    session = crud.get_session_by_user_date(db, user_id, daily.date)
    if session is None:
        return "대화 세션을 찾지 못해 점수 기록만 표시합니다."

    if session.narrative_summary:
        return _truncate_text(session.narrative_summary, max_chars=220)

    user_utts = crud.get_user_utterances_by_session(db, session.id)
    return _build_rule_based_summary(daily, user_utts)


def get_calendar_data(db: DBSession, user_id: int, limit: int = 60) -> list[dict]:
    """
    역할: 캘린더 화면용 날짜별 일별/누적 wellness_score + 사용자 수동 감정 기록 반환
    입력: DB 세션, user_id, 조회 일수 (최근 N일)
    출력: [{date, daily_wellness_score, daily_wellness_label,
           cumulative_wellness_score, cumulative_wellness_label, wellness_score,
           label, crisis_count_day, manual_emotion_label, manual_emotion_note}, ...]
          날짜 오름차순
    """
    rows = crud.get_daily_summaries(db, user_id, limit=limit)
    notes = crud.get_daily_emotion_notes(db, user_id, limit=limit)
    summary_map = {row.date: row for row in rows}
    note_map = {row.date: row for row in notes}
    all_dates = sorted(set(summary_map) | set(note_map))

    data: list[dict] = []
    for day in all_dates:
        r = summary_map.get(day)
        note = note_map.get(day)
        daily_wellness = (
            depression_to_display_wellness(r.daily_score) if r is not None else None
        )
        data.append({
            "date":            day,
            "daily_wellness_score": daily_wellness,
            "daily_wellness_label": (
                depression_to_display_label(r.daily_score) if r is not None else None
            ),
            "cumulative_wellness_score": r.wellness_score if r is not None else None,
            "cumulative_wellness_label": r.label if r is not None else None,
            # 기존 프론트/외부 호환용 alias: 누적/평활 웰니스 점수다.
            "wellness_score":  r.wellness_score if r is not None else None,
            "label":           r.label if r is not None else None,
            "depression_tendency_daily": (
                r.depression_tendency_daily if r is not None else None
            ),
            "depression_tendency_smoothed": (
                r.depression_tendency_smoothed if r is not None else None
            ),
            "crisis_count_day": r.crisis_count_day if r is not None else 0,
            "utterance_count":  r.utterance_count if r is not None else 0,
            "summary_text": (
                build_record_summary_text(db, user_id, r)
                if r is not None
                else "아직 마감 요약은 없지만 사용자가 직접 감정을 기록했습니다."
            ),
            "manual_emotion_label": note.emotion_label if note is not None else None,
            "manual_emotion_intensity": note.intensity if note is not None else None,
            "manual_emotion_note": note.note if note is not None else None,
            "manual_emotion_updated_at": (
                note.updated_at.isoformat() if note is not None and note.updated_at else None
            ),
        })
    return data
