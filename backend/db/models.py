"""
db/models.py
역할: SQLite ORM 모델 정의 (SQLAlchemy)
      users / sessions / utterances / daily_summaries / crisis_events /
      model_audit_events / utterance_feedback / daily_emotion_notes 테이블
"""

from sqlalchemy import (
    Column, Integer, Float, String, Boolean, DateTime, Text, ForeignKey, create_engine,
    UniqueConstraint, event, inspect, text,
)
from sqlalchemy.orm import declarative_base, relationship
from datetime import datetime
import os

Base = declarative_base()

# DB 파일 경로
# 환경변수 EMOTION_CHATBOT_DB_PATH가 있으면 해당 경로를 우선 사용 (테스트/eval용)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DEFAULT_DB_PATH = os.path.join(BASE_DIR, "backend", "db", "emotion_chatbot.db")
DB_PATH  = os.environ.get("EMOTION_CHATBOT_DB_PATH", _DEFAULT_DB_PATH)
DB_URL   = f"sqlite:///{DB_PATH}"


class User(Base):
    """사용자 테이블 — 개인 히스토리(wellness, n_days) 관리"""
    __tablename__ = "users"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    username   = Column(String(64), unique=True, nullable=False)
    # 화면 표시용 이름 — 로그인 식별자는 username을 계속 사용한다.
    nickname   = Column(String(64), nullable=True)
    # 비밀번호 재설정 확인용 이메일 — 신규 가입부터 필수로 받되 기존 계정 호환을 위해 nullable
    email      = Column(String(254), unique=True, nullable=True)
    # PBKDF2 해시 문자열 — 기존 username-only 계정 호환을 위해 nullable
    password_hash = Column(String(256), nullable=True)
    # 개발 전용 계정 여부 — 날짜 강제 이동 같은 테스트 기능 권한에 사용
    is_developer = Column(Boolean, nullable=False, default=False)
    # 로그인 연속 실패 횟수와 잠금 만료 시각 — 재시작 뒤에도 brute force 방어를 유지한다.
    failed_login_attempts = Column(Integer, nullable=False, default=0)
    locked_until = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    sessions       = relationship("Session",      back_populates="user", cascade="all, delete-orphan")
    daily_summaries = relationship("DailySummary", back_populates="user", cascade="all, delete-orphan")
    crisis_events  = relationship("CrisisEvent",  back_populates="user", cascade="all, delete-orphan")
    daily_emotion_notes = relationship(
        "DailyEmotionNote",
        back_populates="user",
        cascade="all, delete-orphan",
    )


class Session(Base):
    """대화 세션 테이블 — 날짜 단위 대화창"""
    __tablename__ = "sessions"
    __table_args__ = (
        UniqueConstraint("user_id", "date", name="uq_sessions_user_date"),
    )

    id         = Column(Integer, primary_key=True, autoincrement=True)
    user_id    = Column(Integer, ForeignKey("users.id"), nullable=False)
    date       = Column(String(10), nullable=False)   # "YYYY-MM-DD"
    created_at = Column(DateTime, default=datetime.utcnow)
    # N발화마다 갱신되는 LLM 서사 요약 — rolling summary 의 [서사] 라인 소스
    narrative_summary           = Column(Text,    nullable=True)
    narrative_until_utterance_id = Column(Integer, nullable=True)

    user       = relationship("User",      back_populates="sessions")
    utterances = relationship("Utterance", back_populates="session", cascade="all, delete-orphan")


class Utterance(Base):
    """발화 테이블 — 발화 텍스트 + 점수 저장"""
    __tablename__ = "utterances"

    id               = Column(Integer, primary_key=True, autoincrement=True)
    session_id       = Column(Integer, ForeignKey("sessions.id"), nullable=False)
    text             = Column(Text, nullable=False)
    role             = Column(String(16), nullable=False, default="user")  # "user" | "bot"
    roberta_score    = Column(Float,   nullable=True)
    cbt_score        = Column(Float,   nullable=True)
    # CBT 앵커 임베딩 기반 top 카테고리 — rolling summary 테마 감지에 사용
    # cbt_score >= CBT_THRESHOLD(0.60) 일 때만 저장, 아니면 None
    cbt_top_category = Column(String(32), nullable=True)
    depression_score = Column(Float,   nullable=True)
    # 우울 경향 전용 점수 v1.5 (규칙 기반) — 종합 depression_score와 별도로 추적
    depression_tendency_score = Column(Float, nullable=True)
    top_emotion      = Column(String(8), nullable=True)
    entailment_prob  = Column(Float,   nullable=True)
    is_crisis        = Column(Boolean, nullable=True, default=False)
    created_at       = Column(DateTime, default=datetime.utcnow)

    session = relationship("Session", back_populates="utterances")


class DailySummary(Base):
    """일별 요약 테이블 — EWMA 평활 점수 + 레이블 저장"""
    __tablename__ = "daily_summaries"
    __table_args__ = (
        UniqueConstraint("user_id", "date", name="uq_daily_summaries_user_date"),
    )

    id               = Column(Integer, primary_key=True, autoincrement=True)
    user_id          = Column(Integer, ForeignKey("users.id"), nullable=False)
    date             = Column(String(10), nullable=False)   # "YYYY-MM-DD"
    daily_score      = Column(Float,   nullable=False)
    smoothed_score   = Column(Float,   nullable=False)
    wellness_score   = Column(Float,   nullable=False)
    label            = Column(String(8),  nullable=False)
    # 우울 경향 전용 일별/평활 점수 v1.5 — 기존 wellness 축과 병렬 운영
    depression_tendency_daily    = Column(Float, nullable=True)
    depression_tendency_smoothed = Column(Float, nullable=True)
    utterance_count  = Column(Integer, nullable=False, default=0)
    crisis_count_day = Column(Integer, nullable=False, default=0)
    created_at       = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="daily_summaries")


class CrisisEvent(Base):
    """위기 이벤트 테이블 — NLI 감지 또는 Qwen [CRISIS] 태그 발생 기록"""
    __tablename__ = "crisis_events"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    user_id         = Column(Integer, ForeignKey("users.id"), nullable=False)
    utterance_id    = Column(Integer, ForeignKey("utterances.id"), nullable=True)
    text            = Column(Text,    nullable=False)
    source          = Column(String(16), nullable=False)   # "nli" | "qwen"
    entailment_prob = Column(Float,   nullable=True)
    created_at      = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="crisis_events")


class ModelAuditEvent(Base):
    """모델 판단 감사 테이블 — 운영 관측성용 세부 판단 근거 저장"""
    __tablename__ = "model_audit_events"

    id                         = Column(Integer, primary_key=True, autoincrement=True)
    user_id                    = Column(Integer, ForeignKey("users.id"), nullable=False)
    utterance_id               = Column(Integer, ForeignKey("utterances.id"), nullable=True)
    hard_crisis                = Column(Boolean, nullable=True)
    final_is_crisis            = Column(Boolean, nullable=True)
    nli_candidate              = Column(Boolean, nullable=True)
    qwen_called                = Column(Boolean, nullable=True)
    qwen_crisis_tag            = Column(Boolean, nullable=True)
    qwen_anchor_replaced       = Column(Boolean, nullable=True)
    qwen_anchor_hits_json      = Column(Text, nullable=True)
    qwen_anchor_similarities_json = Column(Text, nullable=True)
    qwen_self_check_verdict    = Column(String(32), nullable=True)
    qwen_self_check_category   = Column(String(64), nullable=True)
    cbt_top_category_source    = Column(String(64), nullable=True)
    cbt_class_confidence       = Column(Float, nullable=True)
    cbt_head_non_distortion    = Column(Boolean, nullable=True)
    utterance_type             = Column(String(32), nullable=True)
    utterance_type_confidence  = Column(Float, nullable=True)
    audit_payload_json         = Column(Text, nullable=True)
    created_at                 = Column(DateTime, default=datetime.utcnow)


class UtteranceFeedback(Base):
    """사용자 피드백 테이블 — 챗봇 응답 평가(좋아요/별로예요) + 감정 셀프 정정 저장
    실사용 라벨 수집 목적:
      - response_rating  → Qwen 품질 리뷰의 genuine-bad / over-block 분리 보조
      - emotion_correction → 감정 분류 일반화 개선용 gold label 후보
    """
    __tablename__ = "utterance_feedback"
    __table_args__ = (
        UniqueConstraint(
            "user_id",
            "utterance_id",
            "feedback_kind",
            name="uq_utterance_feedback_owner_kind",
        ),
    )

    id            = Column(Integer, primary_key=True, autoincrement=True)
    user_id       = Column(Integer, ForeignKey("users.id"), nullable=False)
    # 사용자 발화 id 기준으로 저장 — model_audit_events와 같은 키로 묶어 리뷰에 사용
    utterance_id  = Column(Integer, ForeignKey("utterances.id"), nullable=False)
    # "response_rating" | "emotion_correction"
    feedback_kind = Column(String(32), nullable=False)
    # response_rating: "good"|"bad" / emotion_correction: 7감정 한국어 라벨
    feedback_value = Column(String(16), nullable=False)
    # 정정 시점 모델 top_emotion 스냅샷 — 모델 라벨 vs 사용자 라벨 비교용
    model_emotion = Column(String(8), nullable=True)
    created_at    = Column(DateTime, default=datetime.utcnow)
    updated_at    = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class DailyEmotionNote(Base):
    """사용자 수동 감정 기록 테이블 — 날짜별 체감 감정/강도/메모 저장
    모델이 계산한 감정·웰니스와 섞지 않고, 사용자가 직접 남긴 자기 기록 축으로만 사용한다.
    """
    __tablename__ = "daily_emotion_notes"
    __table_args__ = (
        UniqueConstraint(
            "user_id",
            "date",
            name="uq_daily_emotion_notes_user_date",
        ),
    )

    id            = Column(Integer, primary_key=True, autoincrement=True)
    user_id       = Column(Integer, ForeignKey("users.id"), nullable=False)
    date          = Column(String(10), nullable=False)
    emotion_label = Column(String(8), nullable=False)
    intensity     = Column(Integer, nullable=False, default=3)
    note          = Column(Text, nullable=True)
    created_at    = Column(DateTime, default=datetime.utcnow)
    updated_at    = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="daily_emotion_notes")


def _enable_sqlite_foreign_keys(engine) -> None:
    """
    역할: SQLite 연결마다 외래키 제약을 활성화
    입력: SQLAlchemy 엔진
    출력: 없음
    """
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, connection_record) -> None:
        """
        역할: 새 SQLite DBAPI 연결에 PRAGMA foreign_keys=ON 적용
        입력: DBAPI 연결, SQLAlchemy 연결 레코드
        출력: 없음
        """
        del connection_record
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys=ON")
        finally:
            cursor.close()


def get_engine():
    """역할: SQLAlchemy 엔진 생성 (DB 파일 자동 생성)"""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
    _enable_sqlite_foreign_keys(engine)
    return engine


def _ensure_user_auth_columns(engine) -> None:
    """
    역할: 기존 SQLite DB에 users 인증 관련 컬럼이 없으면 추가
    입력: SQLAlchemy 엔진
    출력: 없음
    """
    inspector = inspect(engine)
    if "users" not in inspector.get_table_names():
        return

    column_names = {col["name"] for col in inspector.get_columns("users")}

    with engine.begin() as conn:
        if "nickname" not in column_names:
            conn.execute(
                text("ALTER TABLE users ADD COLUMN nickname VARCHAR(64)")
            )
            print("[DB 마이그레이션] users.nickname 컬럼 추가")

        if "email" not in column_names:
            conn.execute(
                text("ALTER TABLE users ADD COLUMN email VARCHAR(254)")
            )
            print("[DB 마이그레이션] users.email 컬럼 추가")

        if "password_hash" not in column_names:
            conn.execute(
                text("ALTER TABLE users ADD COLUMN password_hash VARCHAR(256)")
            )
            print("[DB 마이그레이션] users.password_hash 컬럼 추가")

        if "is_developer" not in column_names:
            conn.execute(
                text("ALTER TABLE users ADD COLUMN is_developer BOOLEAN NOT NULL DEFAULT 0")
            )
            print("[DB 마이그레이션] users.is_developer 컬럼 추가")

        if "failed_login_attempts" not in column_names:
            conn.execute(
                text(
                    "ALTER TABLE users ADD COLUMN "
                    "failed_login_attempts INTEGER NOT NULL DEFAULT 0"
                )
            )
            print("[DB 마이그레이션] users.failed_login_attempts 컬럼 추가")

        if "locked_until" not in column_names:
            conn.execute(
                text("ALTER TABLE users ADD COLUMN locked_until DATETIME")
            )
            print("[DB 마이그레이션] users.locked_until 컬럼 추가")

        # 신규 가입 이메일 중복을 막는다. NULL은 기존/관리자 계정 호환을 위해 여러 개 허용된다.
        conn.execute(
            text(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS uq_users_email
                ON users (email)
                WHERE email IS NOT NULL
                """
            )
        )


def _deduplicate_sessions(conn) -> int:
    """
    역할: 같은 사용자/날짜 세션을 하나로 합치고 종속 발화를 대표 세션으로 이동
    입력: SQLAlchemy 트랜잭션 연결
    출력: 삭제한 중복 세션 수
    """
    duplicate_groups = conn.execute(
        text(
            """
            SELECT user_id, date
            FROM sessions
            GROUP BY user_id, date
            HAVING COUNT(*) > 1
            """
        )
    ).fetchall()
    removed = 0

    for user_id, date_value in duplicate_groups:
        rows = conn.execute(
            text(
                """
                SELECT id, narrative_summary, narrative_until_utterance_id
                FROM sessions
                WHERE user_id = :user_id AND date = :date_value
                ORDER BY id ASC
                """
            ),
            {"user_id": user_id, "date_value": date_value},
        ).mappings().all()
        canonical_id = int(rows[0]["id"])
        narrative_row = max(
            rows,
            key=lambda row: (
                row["narrative_until_utterance_id"]
                if row["narrative_until_utterance_id"] is not None
                else -1,
                int(row["id"]),
            ),
        )
        conn.execute(
            text(
                """
                UPDATE sessions
                SET narrative_summary = :summary,
                    narrative_until_utterance_id = :until_id
                WHERE id = :canonical_id
                """
            ),
            {
                "summary": narrative_row["narrative_summary"],
                "until_id": narrative_row["narrative_until_utterance_id"],
                "canonical_id": canonical_id,
            },
        )

        for row in rows[1:]:
            duplicate_id = int(row["id"])
            conn.execute(
                text(
                    "UPDATE utterances SET session_id = :canonical_id "
                    "WHERE session_id = :duplicate_id"
                ),
                {"canonical_id": canonical_id, "duplicate_id": duplicate_id},
            )
            conn.execute(
                text("DELETE FROM sessions WHERE id = :duplicate_id"),
                {"duplicate_id": duplicate_id},
            )
            removed += 1
    return removed


def _deduplicate_latest_rows(
    conn,
    *,
    table_name: str,
    group_columns: str,
) -> int:
    """
    역할: 지정 복합 키 중복에서 가장 최근 id 행만 남긴다.
    입력: SQLAlchemy 연결, 허용된 테이블명, GROUP BY 컬럼 문자열
    출력: 삭제한 행 수
    """
    before = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar_one()
    conn.execute(
        text(
            f"""
            DELETE FROM {table_name}
            WHERE id NOT IN (
                SELECT MAX(id)
                FROM {table_name}
                GROUP BY {group_columns}
            )
            """
        )
    )
    after = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar_one()
    return int(before - after)


def _ensure_runtime_unique_constraints(engine) -> dict[str, int]:
    """
    역할: 기존 SQLite DB 중복을 정리하고 런타임 복합 유일 인덱스를 보장
    입력: SQLAlchemy 엔진
    출력: 테이블별 제거한 중복 행 수
    """
    inspector = inspect(engine)
    table_names = set(inspector.get_table_names())
    removed = {
        "sessions": 0,
        "daily_summaries": 0,
        "utterance_feedback": 0,
        "daily_emotion_notes": 0,
    }

    with engine.begin() as conn:
        if "sessions" in table_names:
            removed["sessions"] = _deduplicate_sessions(conn)
            conn.execute(
                text(
                    """
                    CREATE UNIQUE INDEX IF NOT EXISTS uq_sessions_user_date
                    ON sessions (user_id, date)
                    """
                )
            )

        if "daily_summaries" in table_names:
            removed["daily_summaries"] = _deduplicate_latest_rows(
                conn,
                table_name="daily_summaries",
                group_columns="user_id, date",
            )
            conn.execute(
                text(
                    """
                    CREATE UNIQUE INDEX IF NOT EXISTS uq_daily_summaries_user_date
                    ON daily_summaries (user_id, date)
                    """
                )
            )

        if "utterance_feedback" in table_names:
            removed["utterance_feedback"] = _deduplicate_latest_rows(
                conn,
                table_name="utterance_feedback",
                group_columns="user_id, utterance_id, feedback_kind",
            )
            conn.execute(
                text(
                    """
                    CREATE UNIQUE INDEX IF NOT EXISTS uq_utterance_feedback_owner_kind
                    ON utterance_feedback (user_id, utterance_id, feedback_kind)
                    """
                )
            )

        if "daily_emotion_notes" in table_names:
            removed["daily_emotion_notes"] = _deduplicate_latest_rows(
                conn,
                table_name="daily_emotion_notes",
                group_columns="user_id, date",
            )
            conn.execute(
                text(
                    """
                    CREATE UNIQUE INDEX IF NOT EXISTS uq_daily_emotion_notes_user_date
                    ON daily_emotion_notes (user_id, date)
                    """
                )
            )

    if any(removed.values()):
        print(f"[DB 마이그레이션] 복합 키 중복 정리: {removed}")
    return removed


def init_db():
    """역할: 테이블 생성 (최초 실행 시 1회)"""
    engine = get_engine()
    Base.metadata.create_all(engine)
    _ensure_user_auth_columns(engine)
    _ensure_runtime_unique_constraints(engine)
    print(f"[DB 초기화] {DB_PATH}")
    return engine
