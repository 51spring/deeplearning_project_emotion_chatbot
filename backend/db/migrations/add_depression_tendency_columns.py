"""
add_depression_tendency_columns.py
역할: Depression Tendency v1.5 운영 적용을 위한 일회성 마이그레이션
      `utterances` 테이블에 `depression_tendency_score` 컬럼,
      `daily_summaries` 테이블에 `depression_tendency_daily`,
      `depression_tendency_smoothed` 컬럼을 추가한다.
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


def column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    """
    역할: 특정 테이블에 컬럼이 이미 있는지 확인
    입력: SQLite 연결, 테이블명, 컬럼명
    출력: 존재 여부
    """
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(r[1] == column for r in rows)


def add_column_if_missing(
    conn: sqlite3.Connection,
    table: str,
    column: str,
    column_type: str,
) -> bool:
    """
    역할: 컬럼이 없으면 ADD COLUMN 실행, 있으면 스킵
    입력: SQLite 연결, 테이블명, 컬럼명, SQL 타입 정의
    출력: 추가 여부 (True=새로 추가, False=스킵)
    """
    if column_exists(conn, table, column):
        return False
    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")
    return True


def main() -> int:
    """
    역할: utterances / daily_summaries 테이블에 우울 경향 전용 컬럼을 추가한다.
    입력: 없음
    출력: 프로세스 종료 코드
    """
    db_path = get_db_path()
    if not os.path.exists(db_path):
        print(f"[migration] DB 파일 없음 → 마이그레이션 불필요: {db_path}")
        return 0

    conn = sqlite3.connect(db_path)
    try:
        added: list[str] = []

        if add_column_if_missing(
            conn, "utterances", "depression_tendency_score", "FLOAT"
        ):
            added.append("utterances.depression_tendency_score")

        if add_column_if_missing(
            conn, "daily_summaries", "depression_tendency_daily", "FLOAT"
        ):
            added.append("daily_summaries.depression_tendency_daily")

        if add_column_if_missing(
            conn, "daily_summaries", "depression_tendency_smoothed", "FLOAT"
        ):
            added.append("daily_summaries.depression_tendency_smoothed")

        conn.commit()

        if added:
            print(f"[migration] 추가된 컬럼 {len(added)}개: {db_path}")
            for col in added:
                print(f"  - {col}")
        else:
            print(f"[migration] 추가할 컬럼 없음 (모두 존재) → 스킵: {db_path}")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
