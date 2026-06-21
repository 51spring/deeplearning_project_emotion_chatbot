"""
add_developer_user.py
역할: 개발자 전용 테스트 기능을 위한 users.is_developer 컬럼과 기본 developer 계정 보강
입력: EMOTION_CHATBOT_DB_PATH 환경변수(없으면 기본 경로 사용)
출력: 변경 사항 콘솔 출력 (성공/스킵)
"""

import os
import sqlite3
import sys


DEVELOPER_USERNAME = "developer"


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
    return any(row[1] == column for row in rows)


def ensure_developer_column(conn: sqlite3.Connection) -> bool:
    """
    역할: users.is_developer 컬럼을 보장
    입력: SQLite 연결
    출력: 새 컬럼 추가 여부
    """
    if column_exists(conn, "users", "is_developer"):
        return False
    conn.execute(
        "ALTER TABLE users ADD COLUMN is_developer BOOLEAN NOT NULL DEFAULT 0"
    )
    return True


def ensure_developer_user(conn: sqlite3.Connection) -> str:
    """
    역할: developer 계정을 생성하거나 개발자 권한으로 보정
    입력: SQLite 연결
    출력: 수행 결과 문자열
    """
    row = conn.execute(
        "SELECT id, is_developer FROM users WHERE username = ?",
        (DEVELOPER_USERNAME,),
    ).fetchone()

    if row is None:
        conn.execute(
            "INSERT INTO users (username, is_developer, created_at) "
            "VALUES (?, 1, CURRENT_TIMESTAMP)",
            (DEVELOPER_USERNAME,),
        )
        return "created"

    if not bool(row[1]):
        conn.execute(
            "UPDATE users SET is_developer = 1 WHERE username = ?",
            (DEVELOPER_USERNAME,),
        )
        return "promoted"

    return "exists"


def main() -> int:
    """
    역할: 개발자 계정 마이그레이션을 실행
    입력: 없음
    출력: 프로세스 종료 코드
    """
    db_path = get_db_path()
    if not os.path.exists(db_path):
        print(f"[migration] DB 파일 없음 → 마이그레이션 불필요: {db_path}")
        return 0

    conn = sqlite3.connect(db_path)
    try:
        column_added = ensure_developer_column(conn)
        user_result = ensure_developer_user(conn)
        conn.commit()

        if column_added:
            print("[migration] users.is_developer 컬럼 추가")
        else:
            print("[migration] users.is_developer 컬럼 이미 존재 → 스킵")

        print(f"[migration] developer 계정 상태: {user_result}")
        print(f"[migration] 대상 DB: {db_path}")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
