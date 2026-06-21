"""
test_db_integrity.py
역할: 최근 히스토리 조회와 날짜별 DB 복합 유일성 회귀 테스트
입력: pytest 실행
출력: assertion 성공/실패
"""

from datetime import date, timedelta

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from backend.db import crud
from backend.db.models import (
    Base,
    DailySummary,
    DailyEmotionNote,
    Session as ChatSession,
    User,
    _enable_sqlite_foreign_keys,
    _ensure_runtime_unique_constraints,
    _ensure_user_auth_columns,
)


def build_memory_session():
    """
    역할: 외래키가 활성화된 독립 인메모리 DB 세션 생성
    입력: 없음
    출력: SQLAlchemy 세션
    """
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
    )
    _enable_sqlite_foreign_keys(engine)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def test_recent_history_returns_latest_window_in_chronological_order() -> None:
    """
    역할: 120일 중 최신 90일만 오래된 순서로 반환하는지 검증
    입력: 없음
    출력: 없음
    """
    db = build_memory_session()
    try:
        user = User(username="history_user")
        db.add(user)
        db.commit()
        db.refresh(user)

        start = date(2026, 1, 1)
        for index in range(120):
            db.add(
                DailySummary(
                    user_id=user.id,
                    date=(start + timedelta(days=index)).isoformat(),
                    daily_score=float(index),
                    smoothed_score=float(index),
                    wellness_score=float(index),
                    label="보통",
                    depression_tendency_daily=float(index),
                )
            )
        db.commit()

        assert crud.get_wellness_history(db, user.id) == [
            float(index) for index in range(30, 120)
        ]
        assert crud.get_daily_score_history_before_date(
            db,
            user.id,
            (start + timedelta(days=120)).isoformat(),
        ) == [float(index) for index in range(30, 120)]
        assert crud.get_daily_tendency_history_before_date(
            db,
            user.id,
            (start + timedelta(days=120)).isoformat(),
        ) == [float(index) for index in range(30, 120)]
    finally:
        db.close()


def test_date_unique_constraints_reject_duplicates() -> None:
    """
    역할: 세션, 일별 요약, 수동 감정 기록의 사용자/날짜 중복이 DB에서 차단되는지 검증
    입력: 없음
    출력: 없음
    """
    db = build_memory_session()
    try:
        user = User(username="unique_user")
        db.add(user)
        db.commit()
        db.refresh(user)

        db.add_all(
            [
                ChatSession(user_id=user.id, date="2026-06-11"),
                ChatSession(user_id=user.id, date="2026-06-11"),
            ]
        )
        with pytest.raises(IntegrityError):
            db.commit()
        db.rollback()

        common = {
            "user_id": user.id,
            "date": "2026-06-11",
            "daily_score": 0.3,
            "smoothed_score": 0.3,
            "wellness_score": 70.0,
            "label": "보통",
        }
        db.add_all([DailySummary(**common), DailySummary(**common)])
        with pytest.raises(IntegrityError):
            db.commit()
        db.rollback()

        note_common = {
            "user_id": user.id,
            "date": "2026-06-11",
            "emotion_label": "행복",
            "intensity": 4,
        }
        db.add_all([DailyEmotionNote(**note_common), DailyEmotionNote(**note_common)])
        with pytest.raises(IntegrityError):
            db.commit()
    finally:
        db.rollback()
        db.close()


def test_auth_migration_adds_profile_email_and_unique_index() -> None:
    """
    역할: 기존 users 테이블에 닉네임/이메일 컬럼과 이메일 유일 인덱스가 추가되는지 검증
    입력: 없음
    출력: 없음
    """
    engine = create_engine("sqlite://")
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE users ("
                "id INTEGER PRIMARY KEY, username VARCHAR(64) UNIQUE NOT NULL)"
            )
        )

    _ensure_user_auth_columns(engine)
    with engine.begin() as conn:
        columns = {
            row[1] for row in conn.execute(text("PRAGMA table_info(users)")).fetchall()
        }
        assert {"nickname", "email", "password_hash"} <= columns
        conn.execute(
            text(
                "INSERT INTO users "
                "(username, nickname, email, password_hash) "
                "VALUES ('u1', '사용자1', 'same@example.local', 'hash')"
            )
        )
        with pytest.raises(IntegrityError):
            conn.execute(
                text(
                    "INSERT INTO users "
                    "(username, nickname, email, password_hash) "
                    "VALUES ('u2', '사용자2', 'same@example.local', 'hash')"
                )
            )


def test_legacy_duplicate_migration_merges_sessions_and_keeps_latest_summary() -> None:
    """
    역할: 기존 중복 세션의 발화를 보존하고 최신 일별 요약만 남기는지 검증
    입력: 없음
    출력: 없음
    """
    engine = create_engine("sqlite://")
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE users (id INTEGER PRIMARY KEY)"))
        conn.execute(
            text(
                "CREATE TABLE sessions ("
                "id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL, date TEXT NOT NULL, "
                "narrative_summary TEXT, narrative_until_utterance_id INTEGER)"
            )
        )
        conn.execute(
            text(
                "CREATE TABLE utterances ("
                "id INTEGER PRIMARY KEY, session_id INTEGER NOT NULL, text TEXT NOT NULL)"
            )
        )
        conn.execute(
            text(
                "CREATE TABLE daily_summaries ("
                "id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL, date TEXT NOT NULL)"
            )
        )
        conn.execute(
            text(
                "INSERT INTO sessions VALUES "
                "(1, 1, '2026-06-11', '이전', 10), "
                "(2, 1, '2026-06-11', '최신', 20)"
            )
        )
        conn.execute(text("INSERT INTO utterances VALUES (1, 2, '보존할 발화')"))
        conn.execute(
            text(
                "INSERT INTO daily_summaries VALUES "
                "(1, 1, '2026-06-11'), (2, 1, '2026-06-11')"
            )
        )

    removed = _ensure_runtime_unique_constraints(engine)
    with engine.connect() as conn:
        session_rows = conn.execute(
            text(
                "SELECT id, narrative_summary, narrative_until_utterance_id "
                "FROM sessions"
            )
        ).fetchall()
        assert session_rows == [(1, "최신", 20)]
        assert conn.execute(
            text("SELECT session_id FROM utterances WHERE id = 1")
        ).scalar_one() == 1
        assert conn.execute(
            text("SELECT id FROM daily_summaries")
        ).fetchall() == [(2,)]
    assert removed == {
        "sessions": 1,
        "daily_summaries": 1,
        "utterance_feedback": 0,
        "daily_emotion_notes": 0,
    }
