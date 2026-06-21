"""
add_password_auth.py
역할: 기존 SQLite DB의 users 테이블에 password_hash 컬럼과 developer 기본 비밀번호를 보강
입력: EMOTION_CHATBOT_DB_PATH, EMOTION_CHATBOT_DEVELOPER_PASSWORD 환경변수
출력: 변경 사항 콘솔 출력 (성공/스킵)
"""

import hashlib
import os
import secrets
import sqlite3
import sys


DEVELOPER_USERNAME = "developer"
DEFAULT_DEVELOPER_PASSWORD = os.environ.get("EMOTION_CHATBOT_DEVELOPER_PASSWORD", "developer")
PBKDF2_ITERATIONS = 260_000
SALT_BYTES = 16


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


def hash_password(password: str) -> str:
    """
    역할: 마이그레이션 스크립트 내부에서 기본 비밀번호를 PBKDF2 해시로 변환
    입력: 평문 비밀번호
    출력: 저장 가능한 해시 문자열
    """
    salt = secrets.token_bytes(SALT_BYTES)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PBKDF2_ITERATIONS,
    )
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${salt.hex()}${digest.hex()}"


def ensure_password_column(conn: sqlite3.Connection) -> bool:
    """
    역할: users.password_hash 컬럼을 보장
    입력: SQLite 연결
    출력: 새 컬럼 추가 여부
    """
    if column_exists(conn, "users", "password_hash"):
        return False
    conn.execute("ALTER TABLE users ADD COLUMN password_hash VARCHAR(256)")
    return True


def ensure_developer_column(conn: sqlite3.Connection) -> bool:
    """
    역할: users.is_developer 컬럼을 보장
    입력: SQLite 연결
    출력: 새 컬럼 추가 여부
    """
    if column_exists(conn, "users", "is_developer"):
        return False
    conn.execute("ALTER TABLE users ADD COLUMN is_developer BOOLEAN NOT NULL DEFAULT 0")
    return True


def ensure_developer_password(conn: sqlite3.Connection) -> str:
    """
    역할: developer 계정의 기본 비밀번호 해시를 보장
    입력: SQLite 연결
    출력: 수행 결과 문자열
    """
    row = conn.execute(
        "SELECT id, password_hash FROM users WHERE username = ?",
        (DEVELOPER_USERNAME,),
    ).fetchone()

    if row is None:
        conn.execute(
            "INSERT INTO users (username, password_hash, is_developer, created_at) "
            "VALUES (?, ?, 1, CURRENT_TIMESTAMP)",
            (DEVELOPER_USERNAME, hash_password(DEFAULT_DEVELOPER_PASSWORD)),
        )
        return "created"

    if not row[1]:
        conn.execute(
            "UPDATE users SET password_hash = ?, is_developer = 1 WHERE username = ?",
            (hash_password(DEFAULT_DEVELOPER_PASSWORD), DEVELOPER_USERNAME),
        )
        return "password_set"

    conn.execute(
        "UPDATE users SET is_developer = 1 WHERE username = ?",
        (DEVELOPER_USERNAME,),
    )
    return "exists"


def main() -> int:
    """
    역할: password_hash 컬럼과 developer 기본 비밀번호 보강을 실행
    입력: 없음
    출력: 프로세스 종료 코드
    """
    db_path = get_db_path()
    if not os.path.exists(db_path):
        print(f"[migration] DB 파일 없음 → 마이그레이션 불필요: {db_path}")
        return 0

    conn = sqlite3.connect(db_path)
    try:
        column_added = ensure_password_column(conn)
        developer_column_added = ensure_developer_column(conn)
        developer_result = ensure_developer_password(conn)
        conn.commit()

        if column_added:
            print("[migration] users.password_hash 컬럼 추가")
        else:
            print("[migration] users.password_hash 컬럼 이미 존재 → 스킵")
        if developer_column_added:
            print("[migration] users.is_developer 컬럼 추가")
        print(f"[migration] developer 비밀번호 상태: {developer_result}")
        print(f"[migration] 대상 DB: {db_path}")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
