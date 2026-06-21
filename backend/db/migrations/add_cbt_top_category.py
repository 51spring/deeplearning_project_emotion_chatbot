"""
add_cbt_top_category.py
역할: utterances 테이블에 cbt_top_category 컬럼을 추가하는 일회성 마이그레이션
      기존 SQLite 파일에서 컬럼 누락 오류를 막기 위해 ALTER TABLE 로 안전 추가한다.
입력: EMOTION_CHATBOT_DB_PATH 환경변수(없으면 기본 경로 사용)
출력: 변경 사항 콘솔 출력 (성공/스킵)
"""

import os
import sqlite3
import sys


def get_db_path() -> str:
    """역할: 마이그레이션 대상 SQLite 경로 결정 (env 우선)
    이 파일 위치: backend/db/migrations/ → 프로젝트 루트는 4 단계 위.
    """
    here = os.path.abspath(__file__)
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(here))))
    default_path = os.path.join(base_dir, "backend", "db", "emotion_chatbot.db")
    return os.environ.get("EMOTION_CHATBOT_DB_PATH", default_path)


def column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    """역할: 특정 테이블에 컬럼이 존재하는지 PRAGMA 로 확인"""
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(r[1] == column for r in rows)


def main() -> int:
    """역할: cbt_top_category 컬럼이 없으면 ALTER TABLE 로 추가"""
    db_path = get_db_path()
    if not os.path.exists(db_path):
        print(f"[migration] DB 파일 없음 → 마이그레이션 불필요: {db_path}")
        return 0

    conn = sqlite3.connect(db_path)
    try:
        if column_exists(conn, "utterances", "cbt_top_category"):
            print(f"[migration] cbt_top_category 컬럼 이미 존재 → 스킵: {db_path}")
            return 0

        conn.execute("ALTER TABLE utterances ADD COLUMN cbt_top_category VARCHAR(32)")
        conn.commit()
        print(f"[migration] utterances.cbt_top_category 컬럼 추가 완료: {db_path}")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
