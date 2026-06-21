"""
add_utterance_feedback.py
역할: 사용자 피드백(응답 평가 + 감정 셀프 정정)용 utterance_feedback 테이블을
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
    역할: 특정 테이블 존재 여부를 확인한다.
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
    역할: 피드백 upsert 키의 중복 행을 최신 행 하나로 정리하고 유일 인덱스를 보장한다.
    입력: SQLite 연결
    출력: 제거한 중복 행 수
    """
    before = conn.total_changes
    conn.execute(
        """
        DELETE FROM utterance_feedback
        WHERE id NOT IN (
            SELECT MAX(id)
            FROM utterance_feedback
            GROUP BY user_id, utterance_id, feedback_kind
        )
        """
    )
    removed = conn.total_changes - before
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS uq_utterance_feedback_owner_kind
        ON utterance_feedback (user_id, utterance_id, feedback_kind)
        """
    )
    return removed


def main() -> int:
    """
    역할: utterance_feedback 테이블이 없으면 생성한다.
    입력: 없음
    출력: 프로세스 종료 코드
    """
    db_path = get_db_path()
    if not os.path.exists(db_path):
        print(f"[migration] DB 파일 없음 → 마이그레이션 불필요: {db_path}")
        return 0

    conn = sqlite3.connect(db_path)
    try:
        if not table_exists(conn, "utterance_feedback"):
            conn.execute(
                """
                CREATE TABLE utterance_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    utterance_id INTEGER NOT NULL,
                    feedback_kind VARCHAR(32) NOT NULL,
                    feedback_value VARCHAR(16) NOT NULL,
                    model_emotion VARCHAR(8),
                    created_at DATETIME,
                    updated_at DATETIME,
                    FOREIGN KEY(user_id) REFERENCES users (id),
                    FOREIGN KEY(utterance_id) REFERENCES utterances (id)
                )
                """
            )
            print(f"[migration] utterance_feedback 테이블 생성 완료: {db_path}")

        removed = ensure_unique_index(conn)
        conn.commit()
        print(
            "[migration] utterance_feedback 유일 인덱스 확인 완료: "
            f"{db_path} (중복 제거 {removed}건)"
        )
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
