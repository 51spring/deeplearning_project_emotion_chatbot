"""
add_daily_emotion_notes.py
역할: 캘린더 날짜별 사용자 수동 감정 기록용 daily_emotion_notes 테이블을
      기존 SQLite DB에 추가하는 일회성 마이그레이션
      (신규 DB는 init_db()의 create_all이 자동 생성하므로 이 스크립트가 불필요)
입력: EMOTION_CHATBOT_DB_PATH 환경변수(없으면 기본 경로 사용)
출력: 변경 사항 콘솔 출력 (성공/스킵)
"""

import os
import sqlite3
import sys


def get_db_path() -> str:
    """
    역할: 마이그레이션 대상 SQLite 경로 결정
    입력: EMOTION_CHATBOT_DB_PATH 환경변수
    출력: SQLite DB 파일 경로
    """
    here = os.path.abspath(__file__)
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(here))))
    default_path = os.path.join(base_dir, "backend", "db", "emotion_chatbot.db")
    return os.environ.get("EMOTION_CHATBOT_DB_PATH", default_path)


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    """
    역할: 특정 테이블 존재 여부 확인
    입력: SQLite 연결, 테이블명
    출력: 존재 여부
    """
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    return row is not None


def ensure_unique_index(conn: sqlite3.Connection) -> int:
    """
    역할: 날짜별 수동 감정 기록 중복을 최신 행 하나로 정리하고 유일 인덱스를 보장
    입력: SQLite 연결
    출력: 제거한 중복 행 수
    """
    before = conn.total_changes
    conn.execute(
        """
        DELETE FROM daily_emotion_notes
        WHERE id NOT IN (
            SELECT MAX(id)
            FROM daily_emotion_notes
            GROUP BY user_id, date
        )
        """
    )
    removed = conn.total_changes - before
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS uq_daily_emotion_notes_user_date
        ON daily_emotion_notes (user_id, date)
        """
    )
    return removed


def main() -> int:
    """
    역할: daily_emotion_notes 테이블이 없으면 생성한다.
    입력: 없음
    출력: 프로세스 종료 코드
    """
    db_path = get_db_path()
    if not os.path.exists(db_path):
        print(f"[migration] DB 파일 없음 → 마이그레이션 불필요: {db_path}")
        return 0

    conn = sqlite3.connect(db_path)
    try:
        if not table_exists(conn, "daily_emotion_notes"):
            conn.execute(
                """
                CREATE TABLE daily_emotion_notes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    date VARCHAR(10) NOT NULL,
                    emotion_label VARCHAR(8) NOT NULL,
                    intensity INTEGER NOT NULL DEFAULT 3,
                    note TEXT,
                    created_at DATETIME,
                    updated_at DATETIME,
                    FOREIGN KEY(user_id) REFERENCES users (id)
                )
                """
            )
            print(f"[migration] daily_emotion_notes 테이블 생성 완료: {db_path}")

        removed = ensure_unique_index(conn)
        conn.commit()
        print(
            "[migration] daily_emotion_notes 유일 인덱스 확인 완료: "
            f"{db_path} (중복 제거 {removed}건)"
        )
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
