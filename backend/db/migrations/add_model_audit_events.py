"""
add_model_audit_events.py
역할: 운영 관측성용 model_audit_events 테이블을 추가하는 일회성 마이그레이션
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


def main() -> int:
    """
    역할: model_audit_events 테이블이 없으면 생성한다.
    입력: 없음
    출력: 프로세스 종료 코드
    """
    db_path = get_db_path()
    if not os.path.exists(db_path):
        print(f"[migration] DB 파일 없음 → 마이그레이션 불필요: {db_path}")
        return 0

    conn = sqlite3.connect(db_path)
    try:
        if table_exists(conn, "model_audit_events"):
            print(f"[migration] model_audit_events 테이블 이미 존재 → 스킵: {db_path}")
            return 0

        conn.execute(
            """
            CREATE TABLE model_audit_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                utterance_id INTEGER,
                hard_crisis BOOLEAN,
                final_is_crisis BOOLEAN,
                nli_candidate BOOLEAN,
                qwen_called BOOLEAN,
                qwen_crisis_tag BOOLEAN,
                qwen_anchor_replaced BOOLEAN,
                qwen_anchor_hits_json TEXT,
                qwen_anchor_similarities_json TEXT,
                qwen_self_check_verdict VARCHAR(32),
                qwen_self_check_category VARCHAR(64),
                cbt_top_category_source VARCHAR(64),
                cbt_class_confidence FLOAT,
                cbt_head_non_distortion BOOLEAN,
                utterance_type VARCHAR(32),
                utterance_type_confidence FLOAT,
                audit_payload_json TEXT,
                created_at DATETIME,
                FOREIGN KEY(user_id) REFERENCES users (id),
                FOREIGN KEY(utterance_id) REFERENCES utterances (id)
            )
            """
        )
        conn.commit()
        print(f"[migration] model_audit_events 테이블 생성 완료: {db_path}")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
