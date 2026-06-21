"""
db/crud.py
역할: DB CRUD 함수 모음 — 세션/발화/일별요약/수동감정/위기이벤트 저장 및 조회
입력: SQLAlchemy Session 객체 + 데이터 dict
출력: ORM 모델 인스턴스 또는 리스트
"""

import json
import math
import os
from datetime import date as date_type, datetime, timedelta
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from backend.db.models import (
    User, Session as ChatSession, Utterance, DailySummary, CrisisEvent, ModelAuditEvent,
    UtteranceFeedback, DailyEmotionNote,
)
from backend.auth_utils import hash_password, verify_password
from pipeline.score_policy import compute_wellness_contribution


DEVELOPER_USERNAME = "developer"
ROOT_USERNAME = "root"
DEVELOPER_PASSWORD_FROM_ENV = os.environ.get("EMOTION_CHATBOT_DEVELOPER_PASSWORD")
ROOT_PASSWORD_FROM_ENV = os.environ.get("EMOTION_CHATBOT_ROOT_PASSWORD")
APP_ENV = os.environ.get("EMOTION_CHATBOT_ENV", "local").strip().lower()
ADMIN_USERNAMES = {DEVELOPER_USERNAME.casefold(), ROOT_USERNAME.casefold()}
ALLOW_LEGACY_ACCOUNT_CLAIM = (
    os.environ.get("EMOTION_CHATBOT_ALLOW_LEGACY_ACCOUNT_CLAIM", "")
    .strip()
    .lower()
    in {"1", "true", "yes", "on"}
)

if APP_ENV in {"prod", "production"} and (
    not DEVELOPER_PASSWORD_FROM_ENV or not ROOT_PASSWORD_FROM_ENV
):
    raise RuntimeError(
        "production 모드에서는 EMOTION_CHATBOT_DEVELOPER_PASSWORD와 "
        "EMOTION_CHATBOT_ROOT_PASSWORD를 반드시 설정해야 합니다."
    )

MIN_PRODUCTION_ADMIN_PASSWORD_LENGTH = 12

if APP_ENV in {"prod", "production"}:
    insecure_admin_passwords = {
        "developer",
        "root",
        "password",
        "change-me",
        "developer-demo-20260521",
        "root-demo-20260521",
    }
    for account_name, password_value in (
        (DEVELOPER_USERNAME, DEVELOPER_PASSWORD_FROM_ENV),
        (ROOT_USERNAME, ROOT_PASSWORD_FROM_ENV),
    ):
        if (
            password_value is None
            or len(password_value) < MIN_PRODUCTION_ADMIN_PASSWORD_LENGTH
            or password_value.casefold() in insecure_admin_passwords
        ):
            raise RuntimeError(
                f"production 모드의 {account_name} 관리자 비밀번호는 "
                f"{MIN_PRODUCTION_ADMIN_PASSWORD_LENGTH}자 이상의 강한 값이어야 합니다."
            )
    if DEVELOPER_PASSWORD_FROM_ENV == ROOT_PASSWORD_FROM_ENV:
        raise RuntimeError(
            "production 모드에서는 developer와 root 비밀번호를 서로 다르게 설정해야 합니다."
        )

DEFAULT_DEVELOPER_PASSWORD = DEVELOPER_PASSWORD_FROM_ENV or "developer"
DEFAULT_ROOT_PASSWORD = ROOT_PASSWORD_FROM_ENV or "root"


def _normalize_username(username: str) -> str:
    """
    역할: 사용자 이름 저장 전 앞뒤 공백을 정리
    입력: 원본 사용자 이름
    출력: 정리된 사용자 이름
    """
    return str(username).strip()


def _normalize_nickname(nickname: str | None, fallback_username: str) -> str:
    """
    역할: 화면 표시용 닉네임을 정리하고 비어 있으면 아이디를 대체값으로 사용
    입력: 닉네임 원문, 대체 아이디
    출력: 저장용 닉네임
    """
    normalized = str(nickname or "").strip()
    return normalized or _normalize_username(fallback_username)


def normalize_email(email: str | None) -> str:
    """
    역할: 계정 복구용 이메일을 저장/비교 가능한 형태로 정규화
    입력: 이메일 원문
    출력: 앞뒤 공백 제거 및 소문자 변환된 이메일
    """
    return str(email or "").strip().casefold()


def is_developer_username(username: str) -> bool:
    """
    역할: 관리자 기본 계정명인지 확인
    입력: 사용자 이름
    출력: 관리자 계정명 여부
    """
    return _normalize_username(username).casefold() in ADMIN_USERNAMES


def is_developer_user(user: User) -> bool:
    """
    역할: User ORM 객체가 관리자 권한을 갖는지 확인
    입력: User ORM 객체
    출력: 관리자 권한 여부
    """
    return bool(getattr(user, "is_developer", False))


def _password_needs_update(password_hash: str | None, target_password: str) -> bool:
    """
    역할: 관리자 기본 비밀번호를 DB에 반영해야 하는지 확인
    입력: 현재 비밀번호 해시, 목표 평문 비밀번호
    출력: 해시 부재 또는 불일치 여부
    """
    if not password_hash:
        return True
    return not verify_password(target_password, password_hash)


# ── 사용자 ────────────────────────────────────────────────────────────────────
def get_user_by_username(db: Session, username: str) -> User | None:
    """
    역할: 사용자 이름으로 계정을 조회하되 새로 만들지 않음
    입력: DB 세션, 사용자 아이디
    출력: User ORM 객체 또는 None
    """
    normalized = _normalize_username(username)
    if not normalized:
        return None
    return db.query(User).filter(User.username == normalized).first()


def get_user_by_email(db: Session, email: str) -> User | None:
    """
    역할: 정규화된 이메일로 계정을 조회
    입력: DB 세션, 이메일
    출력: User ORM 객체 또는 None
    """
    normalized = normalize_email(email)
    if not normalized:
        return None
    return db.query(User).filter(User.email == normalized).first()


def get_or_create_user(db: Session, username: str) -> User:
    """
    역할: 사용자 조회 또는 신규 생성
    입력: DB 세션, 유저명
    출력: User ORM 객체
    """
    normalized = _normalize_username(username)
    should_be_developer = is_developer_username(normalized)

    user = db.query(User).filter(User.username == normalized).first()
    if not user:
        user = User(
            username=normalized,
            nickname=normalized,
            is_developer=should_be_developer,
        )
        db.add(user)
        db.commit()
        db.refresh(user)
    elif should_be_developer and not is_developer_user(user):
        # 기존 DB에 developer 계정이 일반 계정으로 만들어져 있으면 개발자 권한으로 승격한다.
        user.is_developer = True
        db.commit()
        db.refresh(user)
    return user


def create_user_with_password(
    db: Session,
    username: str,
    password: str,
    nickname: str | None = None,
    email: str | None = None,
    is_developer: bool = False,
) -> User:
    """
    역할: 닉네임/아이디/이메일/비밀번호 기반 신규 사용자 계정을 생성
    입력: DB 세션, 사용자 아이디, 평문 비밀번호, 닉네임, 이메일, 개발자 권한 여부
    출력: User ORM 객체
    """
    normalized = _normalize_username(username)
    normalized_email = normalize_email(email)
    normalized_nickname = _normalize_nickname(nickname, normalized)
    user = get_user_by_username(db, normalized)
    password_hash = hash_password(password)
    should_be_developer = bool(is_developer or is_developer_username(normalized))
    email_owner = get_user_by_email(db, normalized_email) if normalized_email else None
    if email_owner is not None and email_owner.username != normalized:
        raise ValueError("이미 사용 중인 이메일입니다.")

    if user is None:
        user = User(
            username=normalized,
            nickname=normalized_nickname,
            email=normalized_email or None,
            password_hash=password_hash,
            is_developer=should_be_developer,
        )
        db.add(user)
        try:
            db.commit()
        except IntegrityError as exc:
            db.rollback()
            raise ValueError("이미 사용 중인 아이디 또는 이메일입니다.") from exc
        db.refresh(user)
        return user

    if user.password_hash:
        raise ValueError("이미 등록된 아이디입니다.")

    # 기존 username-only 로컬 계정은 비밀번호를 설정해 새 로그인 흐름으로 승격한다.
    # 공개 접속에서는 사용자명만 아는 사람이 과거 계정을 가져갈 수 있어 기본 차단한다.
    if not ALLOW_LEGACY_ACCOUNT_CLAIM:
        raise ValueError(
            "이미 존재하는 이전 방식 계정입니다. "
            "관리자에게 계정 마이그레이션을 요청하세요."
        )

    user.nickname = normalized_nickname
    user.email = normalized_email or user.email
    user.password_hash = password_hash
    if should_be_developer and not is_developer_user(user):
        user.is_developer = True
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise ValueError("이미 사용 중인 이메일입니다.") from exc
    db.refresh(user)
    return user


def user_email_matches(user: User, email: str) -> bool:
    """
    역할: 입력 이메일이 사용자 계정의 복구 이메일과 일치하는지 확인
    입력: User ORM 객체, 이메일 원문
    출력: 이메일 일치 여부
    """
    stored_email = normalize_email(getattr(user, "email", None))
    return bool(stored_email) and stored_email == normalize_email(email)


def authenticate_user(db: Session, username: str, password: str) -> User | None:
    """
    역할: 아이디/비밀번호로 사용자를 인증
    입력: DB 세션, 사용자 아이디, 평문 비밀번호
    출력: 인증 성공 시 User ORM 객체, 실패 시 None
    """
    user = get_user_by_username(db, username)
    if user is None:
        return None
    if not verify_password(password, user.password_hash):
        return None
    return user


def get_login_lock_remaining_seconds(
    user: User,
    now: datetime | None = None,
) -> int:
    """
    역할: 사용자 계정 잠금의 남은 시간 계산
    입력: User ORM 객체, 테스트용 현재 UTC 시각
    출력: 잠금 해제까지 남은 초, 잠기지 않았으면 0
    """
    current = now or datetime.utcnow()
    locked_until = getattr(user, "locked_until", None)
    if locked_until is None or locked_until <= current:
        return 0
    return max(1, math.ceil((locked_until - current).total_seconds()))


def record_failed_login(
    db: Session,
    user: User,
    max_attempts: int,
    lock_seconds: int,
    now: datetime | None = None,
) -> int:
    """
    역할: 로그인 실패 횟수를 누적하고 임계 도달 시 계정 잠금 저장
    입력: DB 세션, 사용자, 최대 실패 횟수, 잠금 초, 테스트용 현재 UTC 시각
    출력: 잠금이 발생했으면 남은 초, 아니면 0
    """
    current = now or datetime.utcnow()
    user.failed_login_attempts = int(user.failed_login_attempts or 0) + 1
    retry_after = 0
    if user.failed_login_attempts >= max_attempts:
        user.locked_until = current + timedelta(seconds=lock_seconds)
        user.failed_login_attempts = 0
        retry_after = lock_seconds
    db.commit()
    db.refresh(user)
    return retry_after


def reset_login_failures(db: Session, user: User) -> User:
    """
    역할: 로그인 성공 또는 비밀번호 변경 후 실패 횟수와 잠금 초기화
    입력: DB 세션, User ORM 객체
    출력: 갱신된 User ORM 객체
    """
    if user.failed_login_attempts or user.locked_until is not None:
        user.failed_login_attempts = 0
        user.locked_until = None
        db.commit()
        db.refresh(user)
    return user


def update_user_password(db: Session, user: User, new_password: str) -> User:
    """
    역할: 사용자 비밀번호를 새 PBKDF2 해시로 교체
    입력: DB 세션, User ORM 객체, 새 평문 비밀번호
    출력: 갱신된 User ORM 객체
    """
    user.password_hash = hash_password(new_password)
    user.failed_login_attempts = 0
    user.locked_until = None
    db.commit()
    db.refresh(user)
    return user


def ensure_developer_user(db: Session) -> User:
    """
    역할: 기본 개발자 계정을 DB에 보장
    입력: DB 세션
    출력: 개발자 User ORM 객체
    """
    user = get_or_create_user(db, DEVELOPER_USERNAME)
    changed = False
    if _password_needs_update(user.password_hash, DEFAULT_DEVELOPER_PASSWORD):
        user.password_hash = hash_password(DEFAULT_DEVELOPER_PASSWORD)
        user.failed_login_attempts = 0
        user.locked_until = None
        changed = True
    if not is_developer_user(user):
        user.is_developer = True
        changed = True
    if changed:
        db.commit()
        db.refresh(user)
    return user


def ensure_root_user(db: Session) -> User:
    """
    역할: 기본 root 관리자 계정을 DB에 보장하고 root/root 로그인을 맞춤
    입력: DB 세션
    출력: root User ORM 객체
    """
    user = get_or_create_user(db, ROOT_USERNAME)
    changed = False
    if _password_needs_update(user.password_hash, DEFAULT_ROOT_PASSWORD):
        # root 계정은 로컬 운영용 고정 관리자 계정이므로 요청된 기본 비밀번호를 보장한다.
        user.password_hash = hash_password(DEFAULT_ROOT_PASSWORD)
        user.failed_login_attempts = 0
        user.locked_until = None
        changed = True
    if not is_developer_user(user):
        user.is_developer = True
        changed = True
    if changed:
        db.commit()
        db.refresh(user)
    return user


def ensure_admin_users(db: Session) -> list[User]:
    """
    역할: 로컬 관리자 계정들을 DB에 보장
    입력: DB 세션
    출력: 준비된 관리자 User ORM 리스트
    """
    return [
        ensure_developer_user(db),
        ensure_root_user(db),
    ]


# ── 관리자 운영 ───────────────────────────────────────────────────────────────
def _delete_runtime_data_rows(db: Session, user_id: int) -> dict[str, int]:
    """
    역할: 특정 계정의 대화/요약/수동감정/감사성 런타임 데이터를 현재 트랜잭션에서 삭제
    입력: DB 세션, 초기화 대상 user_id
    출력: 테이블별 삭제 건수 dict (commit은 호출자가 수행)
    """
    deleted: dict[str, int] = {}
    session_ids = [
        row.id
        for row in db.query(ChatSession.id).filter(ChatSession.user_id == user_id).all()
    ]

    # 외래키 참조가 있는 피드백/감사/위기 로그를 먼저 삭제한 뒤 발화와 세션을 정리한다.
    deleted["utterance_feedback"] = (
        db.query(UtteranceFeedback)
        .filter(UtteranceFeedback.user_id == user_id)
        .delete(synchronize_session=False)
    )
    deleted["daily_emotion_notes"] = (
        db.query(DailyEmotionNote)
        .filter(DailyEmotionNote.user_id == user_id)
        .delete(synchronize_session=False)
    )
    deleted["model_audit_events"] = (
        db.query(ModelAuditEvent)
        .filter(ModelAuditEvent.user_id == user_id)
        .delete(synchronize_session=False)
    )
    deleted["crisis_events"] = (
        db.query(CrisisEvent)
        .filter(CrisisEvent.user_id == user_id)
        .delete(synchronize_session=False)
    )
    deleted["utterances"] = (
        db.query(Utterance)
        .filter(Utterance.session_id.in_(session_ids))
        .delete(synchronize_session=False)
        if session_ids
        else 0
    )
    deleted["sessions"] = (
        db.query(ChatSession)
        .filter(ChatSession.user_id == user_id)
        .delete(synchronize_session=False)
    )
    deleted["daily_summaries"] = (
        db.query(DailySummary)
        .filter(DailySummary.user_id == user_id)
        .delete(synchronize_session=False)
    )
    return deleted


def reset_runtime_data(db: Session, user_id: int) -> dict[str, int]:
    """
    역할: 특정 계정의 대화/요약/수동감정/감사성 런타임 데이터만 초기화
    입력: DB 세션, 초기화 대상 user_id
    출력: 테이블별 삭제 건수 dict
    """
    try:
        deleted = _delete_runtime_data_rows(db, user_id)
        db.commit()
    except Exception:
        db.rollback()
        raise
    return deleted


def delete_user_account(db: Session, user_id: int) -> dict[str, int]:
    """
    역할: 계정 삭제 — 사용자 본인 요청으로 모든 기록과 계정 row를 완전히 삭제
          (관리자용 reset_runtime_data와 달리 users row까지 제거한다)
    입력: DB 세션, 삭제 대상 user_id
    출력: 테이블별 삭제 건수 dict
    """
    try:
        # 계정과 종속 기록을 한 트랜잭션으로 묶어 중간 실패 시 부분 삭제를 막는다.
        deleted = _delete_runtime_data_rows(db, user_id)
        deleted["users"] = (
            db.query(User)
            .filter(User.id == user_id)
            .delete(synchronize_session=False)
        )
        db.commit()
    except Exception:
        db.rollback()
        raise
    return deleted


# ── 대화 세션 ─────────────────────────────────────────────────────────────────
def get_or_create_session(db: Session, user_id: int, date_str: str) -> ChatSession:
    """
    역할: 날짜 기준 대화 세션 조회 또는 신규 생성
    입력: DB 세션, user_id, 날짜 문자열 "YYYY-MM-DD"
    출력: Session ORM 객체
    """
    session = (
        db.query(ChatSession)
        .filter(ChatSession.user_id == user_id, ChatSession.date == date_str)
        .first()
    )
    if not session:
        session = ChatSession(user_id=user_id, date=date_str)
        db.add(session)
        try:
            db.commit()
            db.refresh(session)
        except IntegrityError:
            # 동시 생성 경쟁에서 유일 제약을 먼저 획득한 행을 재사용한다.
            db.rollback()
            session = (
                db.query(ChatSession)
                .filter(
                    ChatSession.user_id == user_id,
                    ChatSession.date == date_str,
                )
                .one()
            )
    return session


def get_session_by_user_date(db: Session, user_id: int, date_str: str) -> ChatSession | None:
    """
    역할: 날짜 기준 대화 세션을 조회하되 없으면 새로 만들지 않음
    입력: DB 세션, user_id, 날짜 문자열 "YYYY-MM-DD"
    출력: Session ORM 객체 또는 None
    """
    return (
        db.query(ChatSession)
        .filter(ChatSession.user_id == user_id, ChatSession.date == date_str)
        .first()
    )


def get_unclosed_session_dates(
    db: Session,
    user_id: int,
    before_date: str,
    limit: int = 14,
) -> list[str]:
    """
    역할: 자동 하루 마감 대상 날짜 조회 — 기준 날짜 이전에 사용자 발화가 있는데
          daily_summary가 아직 없는 세션 날짜를 오래된 순으로 반환
    입력: DB 세션, user_id, 기준 날짜 문자열(이 날짜 미만만 대상), 최대 처리 일수
    출력: 마감되지 않은 날짜 문자열 리스트 (오름차순)
    """
    has_user_utterance = (
        db.query(Utterance.id)
        .filter(
            Utterance.session_id == ChatSession.id,
            Utterance.role == "user",
        )
        .exists()
    )
    already_closed = (
        db.query(DailySummary.id)
        .filter(
            DailySummary.user_id == user_id,
            DailySummary.date == ChatSession.date,
        )
        .exists()
    )
    rows = (
        db.query(ChatSession.date)
        .filter(
            ChatSession.user_id == user_id,
            ChatSession.date < before_date,
        )
        .filter(has_user_utterance)
        .filter(~already_closed)
        .order_by(ChatSession.date.asc())
        .limit(limit)
        .all()
    )
    return [row.date for row in rows]


# ── 발화 ──────────────────────────────────────────────────────────────────────
def save_utterance(db: Session, session_id: int, data: dict) -> Utterance:
    """
    역할: 발화 및 점수 저장
    입력: DB 세션, session_id, 점수 dict
          {text, role, roberta_score, cbt_score, cbt_top_category,
           depression_score, top_emotion, entailment_prob, is_crisis}
    출력: Utterance ORM 객체
    """
    utt = Utterance(
        session_id       = session_id,
        text             = data["text"],
        role             = data.get("role", "user"),
        roberta_score    = data.get("roberta_score"),
        cbt_score        = data.get("cbt_score"),
        cbt_top_category = data.get("cbt_top_category"),
        depression_score = data.get("depression_score"),
        depression_tendency_score = data.get("depression_tendency_score"),
        top_emotion      = data.get("top_emotion"),
        entailment_prob  = data.get("entailment_prob"),
        is_crisis        = data.get("is_crisis", False),
    )
    db.add(utt)
    db.commit()
    db.refresh(utt)
    return utt


def get_utterances_by_session(db: Session, session_id: int) -> list[Utterance]:
    """역할: 특정 세션의 발화 전체 조회 (시간순)"""
    return (
        db.query(Utterance)
        .filter(Utterance.session_id == session_id)
        .order_by(Utterance.created_at)
        .all()
    )


def get_user_utterances_by_session(db: Session, session_id: int) -> list[Utterance]:
    """
    역할: 특정 세션의 사용자 발화만 시간순으로 조회
    입력: DB 세션, session_id
    출력: 사용자 발화 ORM 리스트
    """
    return (
        db.query(Utterance)
        .filter(Utterance.session_id == session_id, Utterance.role == "user")
        .order_by(Utterance.created_at)
        .all()
    )


def get_user_owned_utterance(db: Session, user_id: int, utterance_id: int) -> Utterance | None:
    """
    역할: 발화 id가 해당 사용자 소유인지 세션 join으로 확인하며 조회
    입력: DB 세션, user_id, utterance_id
    출력: 소유가 확인된 Utterance ORM 객체 또는 None
    """
    return (
        db.query(Utterance)
        .join(ChatSession, Utterance.session_id == ChatSession.id)
        .filter(
            Utterance.id == utterance_id,
            ChatSession.user_id == user_id,
        )
        .first()
    )


def get_recent_utterances_by_session(
    db: Session,
    session_id: int,
    limit: int = 20,
    min_utterance_id: int | None = None,
) -> list[Utterance]:
    """
    역할: 특정 세션의 최근 발화만 조회해 시간순으로 반환
    입력: DB 세션, session_id, 조회 개수, 포함할 최소 발화 id
    출력: 최근 발화 리스트 (시간순)
    """
    query = db.query(Utterance).filter(Utterance.session_id == session_id)
    if min_utterance_id is not None:
        query = query.filter(Utterance.id >= min_utterance_id)

    rows = (
        query
        .order_by(Utterance.created_at.desc(), Utterance.id.desc())
        .limit(limit)
        .all()
    )
    return list(reversed(rows))


def get_older_utterances_for_summary(
    db: Session,
    session_id: int,
    recent_limit: int = 20,
    summary_limit: int = 40,
    min_utterance_id: int | None = None,
) -> list[Utterance]:
    """
    역할: 최근 history 창 밖의 발화를 rolling summary 생성용으로 제한 조회
    입력: DB 세션, session_id, 제외할 최근 발화 수, 요약에 사용할 최대 발화 수, 포함할 최소 발화 id
    출력: 요약 대상 발화 리스트 (시간순)
    """
    query = db.query(Utterance).filter(Utterance.session_id == session_id)
    if min_utterance_id is not None:
        query = query.filter(Utterance.id >= min_utterance_id)

    rows = (
        query
        .order_by(Utterance.created_at.desc(), Utterance.id.desc())
        .offset(recent_limit)
        .limit(summary_limit)
        .all()
    )
    return list(reversed(rows))


def update_session_narrative(
    db: Session,
    session_id: int,
    narrative_summary: str,
    until_utterance_id: int,
) -> None:
    """역할: 세션의 LLM 서사 요약 + 마지막 반영 발화 id 업데이트
    입력: DB 세션, session_id, 새 요약 문자열, 반영 마감 utterance id
    출력: 없음 (DB 갱신)
    """
    sess = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if sess is None:
        return
    sess.narrative_summary = narrative_summary
    sess.narrative_until_utterance_id = until_utterance_id
    db.commit()


def get_session_narrative(db: Session, session_id: int) -> tuple[str | None, int | None]:
    """역할: 세션의 narrative_summary 와 마지막 반영 utterance id 조회
    출력: (narrative_summary, narrative_until_utterance_id)
    """
    sess = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if sess is None:
        return None, None
    return sess.narrative_summary, sess.narrative_until_utterance_id


def count_user_utterances_by_session(db: Session, session_id: int) -> int:
    """
    역할: 특정 세션의 사용자 발화 수를 DB count 쿼리로 조회
    입력: DB 세션, session_id
    출력: 사용자 발화 개수
    """
    return (
        db.query(Utterance)
        .filter(Utterance.session_id == session_id, Utterance.role == "user")
        .count()
    )


# ── 일별 요약 ─────────────────────────────────────────────────────────────────
def save_daily_summary(db: Session, user_id: int, date_str: str, data: dict) -> DailySummary:
    """
    역할: 일별 요약 저장 (하루 종료 시 1회 호출)
    입력: DB 세션, user_id, 날짜 문자열, 요약 dict
          {daily_score, smoothed_score, wellness_score, label,
           utterance_count, crisis_count_day}
    출력: DailySummary ORM 객체
    """
    # 같은 날 기존 레코드가 있으면 업데이트
    existing = (
        db.query(DailySummary)
        .filter(DailySummary.user_id == user_id, DailySummary.date == date_str)
        .first()
    )
    if existing:
        for k, v in data.items():
            setattr(existing, k, v)
        db.commit()
        db.refresh(existing)
        return existing

    summary = DailySummary(user_id=user_id, date=date_str, **data)
    db.add(summary)
    try:
        db.commit()
        db.refresh(summary)
        return summary
    except IntegrityError:
        # 다른 요청이 같은 날짜 요약을 먼저 만들면 해당 행을 최신 값으로 갱신한다.
        db.rollback()
        existing = (
            db.query(DailySummary)
            .filter(
                DailySummary.user_id == user_id,
                DailySummary.date == date_str,
            )
            .one()
        )
        for k, v in data.items():
            setattr(existing, k, v)
        db.commit()
        db.refresh(existing)
        return existing


def get_daily_summaries(db: Session, user_id: int, limit: int = 60) -> list[DailySummary]:
    """역할: 사용자의 최근 일별 요약 조회 (날짜 내림차순)"""
    return (
        db.query(DailySummary)
        .filter(DailySummary.user_id == user_id)
        .order_by(DailySummary.date.desc())
        .limit(limit)
        .all()
    )


def get_daily_summaries_between(
    db: Session,
    user_id: int,
    start_date: str,
    end_date: str,
) -> list[DailySummary]:
    """
    역할: 주간 리포트용 — 날짜 구간(양끝 포함)의 일별 요약을 날짜 오름차순으로 조회
    입력: DB 세션, user_id, 시작 날짜, 끝 날짜 ("YYYY-MM-DD")
    출력: DailySummary ORM 리스트
    """
    return (
        db.query(DailySummary)
        .filter(
            DailySummary.user_id == user_id,
            DailySummary.date >= start_date,
            DailySummary.date <= end_date,
        )
        .order_by(DailySummary.date.asc())
        .all()
    )


def get_user_utterance_counts_by_date(
    db: Session,
    user_id: int,
    start_date: str,
    end_date: str,
) -> dict[str, int]:
    """
    역할: 주간 리포트용 — 날짜 구간의 날짜별 사용자 발화 수 집계
          (아직 마감되지 않은 오늘 발화 수도 포함하기 위해 utterances에서 직접 센다)
    입력: DB 세션, user_id, 시작 날짜, 끝 날짜
    출력: {날짜: 사용자 발화 수} dict
    """
    from sqlalchemy import func

    rows = (
        db.query(ChatSession.date, func.count(Utterance.id))
        .join(Utterance, Utterance.session_id == ChatSession.id)
        .filter(
            ChatSession.user_id == user_id,
            ChatSession.date >= start_date,
            ChatSession.date <= end_date,
            Utterance.role == "user",
        )
        .group_by(ChatSession.date)
        .all()
    )
    return {row[0]: int(row[1]) for row in rows}


def get_user_emotion_counts_between(
    db: Session,
    user_id: int,
    start_date: str,
    end_date: str,
) -> dict[str, int]:
    """
    역할: 주간 리포트용 — 날짜 구간의 사용자 발화 top_emotion 분포 집계
    입력: DB 세션, user_id, 시작 날짜, 끝 날짜
    출력: {감정 라벨: 발화 수} dict (빈도 내림차순 → 같은 빈도는 라벨 사전순)
    """
    from sqlalchemy import func

    rows = (
        db.query(Utterance.top_emotion, func.count(Utterance.id))
        .join(ChatSession, Utterance.session_id == ChatSession.id)
        .filter(
            ChatSession.user_id == user_id,
            ChatSession.date >= start_date,
            ChatSession.date <= end_date,
            Utterance.role == "user",
            Utterance.top_emotion.isnot(None),
        )
        .group_by(Utterance.top_emotion)
        .all()
    )
    counts = {row[0]: int(row[1]) for row in rows}
    return dict(sorted(counts.items(), key=lambda x: (-x[1], x[0])))


def get_user_emotion_counts_by_date_between(
    db: Session,
    user_id: int,
    start_date: str,
    end_date: str,
) -> dict[str, dict[str, int]]:
    """
    역할: 리포트 요일 패턴용 — 날짜별 사용자 발화 top_emotion 분포 집계
    입력: DB 세션, user_id, 시작 날짜, 끝 날짜
    출력: {날짜: {감정 라벨: 발화 수}} dict
    """
    from sqlalchemy import func

    rows = (
        db.query(ChatSession.date, Utterance.top_emotion, func.count(Utterance.id))
        .join(ChatSession, Utterance.session_id == ChatSession.id)
        .filter(
            ChatSession.user_id == user_id,
            ChatSession.date >= start_date,
            ChatSession.date <= end_date,
            Utterance.role == "user",
            Utterance.top_emotion.isnot(None),
        )
        .group_by(ChatSession.date, Utterance.top_emotion)
        .all()
    )

    counts_by_date: dict[str, dict[str, int]] = {}
    for day, emotion, count in rows:
        counts_by_date.setdefault(day, {})[emotion] = int(count)
    return counts_by_date


def get_crisis_counts_by_date(
    db: Session,
    user_id: int,
    start_date: str,
    end_date: str,
) -> dict[str, int]:
    """
    역할: 주간 리포트용 — 날짜 구간의 날짜별 위기 이벤트 수 집계
          (발화 → 세션 join 기준이라 아직 마감 전인 오늘 위기도 포함)
    입력: DB 세션, user_id, 시작 날짜, 끝 날짜
    출력: {날짜: 위기 이벤트 수} dict
    """
    from sqlalchemy import func

    rows = (
        db.query(ChatSession.date, func.count(CrisisEvent.id))
        .join(Utterance, CrisisEvent.utterance_id == Utterance.id)
        .join(ChatSession, Utterance.session_id == ChatSession.id)
        .filter(
            CrisisEvent.user_id == user_id,
            ChatSession.date >= start_date,
            ChatSession.date <= end_date,
        )
        .group_by(ChatSession.date)
        .all()
    )
    return {row[0]: int(row[1]) for row in rows}


def get_wellness_history(db: Session, user_id: int, limit: int = 90) -> list[float]:
    """
    역할: 퍼센타일 레이블 계산용 wellness_score 히스토리 조회 (날짜 오름차순)
    출력: wellness_score 리스트 (float)
    """
    rows = (
        db.query(DailySummary.wellness_score)
        .filter(DailySummary.user_id == user_id)
        .order_by(DailySummary.date.desc(), DailySummary.id.desc())
        .limit(limit)
        .all()
    )
    return [r.wellness_score for r in reversed(rows)]


def get_daily_score_history(db: Session, user_id: int, limit: int = 90) -> list[float]:
    """
    역할: EWMA 2단계 복원을 위한 daily_score 히스토리 조회 (날짜 오름차순)
    입력: DB 세션, user_id, 최대 조회 일수
    출력: daily_score 리스트 (float)
    """
    rows = (
        db.query(DailySummary.daily_score)
        .filter(DailySummary.user_id == user_id)
        .order_by(DailySummary.date.desc(), DailySummary.id.desc())
        .limit(limit)
        .all()
    )
    return [r.daily_score for r in reversed(rows)]


def get_wellness_history_before_date(
    db: Session,
    user_id: int,
    date_str: str,
    limit: int = 90,
) -> list[float]:
    """
    역할: 특정 날짜 이전의 wellness_score 히스토리 조회 (날짜 오름차순)
    입력: DB 세션, user_id, 기준 날짜 문자열, 최대 조회 일수
    출력: wellness_score 리스트 (float)
    """
    rows = (
        db.query(DailySummary.wellness_score)
        .filter(
            DailySummary.user_id == user_id,
            DailySummary.date < date_str,
        )
        .order_by(DailySummary.date.desc(), DailySummary.id.desc())
        .limit(limit)
        .all()
    )
    return [r.wellness_score for r in reversed(rows)]


def get_daily_score_history_before_date(
    db: Session,
    user_id: int,
    date_str: str,
    limit: int = 90,
) -> list[float]:
    """
    역할: 특정 날짜 이전의 daily_score 히스토리 조회 (날짜 오름차순)
    입력: DB 세션, user_id, 기준 날짜 문자열, 최대 조회 일수
    출력: daily_score 리스트 (float)
    """
    rows = (
        db.query(DailySummary.daily_score)
        .filter(
            DailySummary.user_id == user_id,
            DailySummary.date < date_str,
        )
        .order_by(DailySummary.date.desc(), DailySummary.id.desc())
        .limit(limit)
        .all()
    )
    return [r.daily_score for r in reversed(rows)]


def get_daily_tendency_history_before_date(
    db: Session,
    user_id: int,
    date_str: str,
    limit: int = 90,
) -> list[float]:
    """
    역할: 특정 날짜 이전의 우울 경향 일별 점수(depression_tendency_daily) 히스토리 조회
    입력: DB 세션, user_id, 기준 날짜 문자열, 최대 조회 일수
    출력: depression_tendency_daily 리스트 (None은 0.0으로 대체, 날짜 오름차순)
    """
    rows = (
        db.query(DailySummary.depression_tendency_daily)
        .filter(
            DailySummary.user_id == user_id,
            DailySummary.date < date_str,
        )
        .order_by(DailySummary.date.desc(), DailySummary.id.desc())
        .limit(limit)
        .all()
    )
    return [
        (
            r.depression_tendency_daily
            if r.depression_tendency_daily is not None
            else 0.0
        )
        for r in reversed(rows)
    ]


def _loads_audit_payload(payload_json: str | None) -> dict:
    """
    역할: audit_payload_json 문자열을 안전하게 dict로 변환
    입력: JSON 문자열 또는 None
    출력: dict payload
    """
    if not payload_json:
        return {}
    try:
        payload = json.loads(payload_json)
    except (TypeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _restored_wellness_contribution_score(
    *,
    raw_depression_score: float | None,
    audit_id: int | None,
    audit_payload_json: str | None,
    utterance_type: str | None,
    final_is_crisis: bool | None,
    hard_crisis: bool | None,
) -> float | None:
    """
    역할: 서버 재시작 시 오늘 발화의 실제 웰니스 반영 점수를 복원
    입력: 저장된 원점수, audit 메타데이터와 payload
    출력: EWMA 버퍼에 넣을 점수 또는 미반영 None
    """
    raw_score = float(raw_depression_score) if raw_depression_score is not None else None
    if audit_id is None:
        # audit 도입 전 과거 로그는 당시 동작 보존을 위해 원점수를 반영한다.
        return raw_score

    payload = _loads_audit_payload(audit_payload_json)
    if "wellness_contribution_score" in payload:
        if not payload.get("score_affects_wellness"):
            return None
        contribution_score = payload.get("wellness_contribution_score")
        if contribution_score is None:
            return None
        try:
            return float(contribution_score)
        except (TypeError, ValueError):
            return None

    contribution = compute_wellness_contribution(
        {
            "utterance_type": utterance_type,
            "depression_score": raw_score,
            "top_emotion": payload.get("top_emotion"),
            "emotion_guard": payload.get("emotion_guard"),
            "nli_guard": payload.get("nli_guard"),
        },
        is_crisis=bool(final_is_crisis or hard_crisis),
    )
    if not contribution["score_affects_wellness"]:
        return None
    return float(contribution["wellness_contribution_score"])


def get_today_user_depression_scores(
    db: Session,
    user_id: int,
    date_str: str,
) -> list[float]:
    """
    역할: 서버 재시작 복원을 위해 오늘 사용자 발화의 depression_score 조회
    입력: DB 세션, user_id, 날짜 문자열 "YYYY-MM-DD"
    출력: 오늘 사용자 발화 depression_score 리스트 (시간순)
    """
    session = (
        db.query(ChatSession)
        .filter(ChatSession.user_id == user_id, ChatSession.date == date_str)
        .first()
    )
    if session is None:
        return []

    # audit가 있는 신규 로그는 저장된 기여 점수 또는 현재 정책으로 복원한다.
    # audit가 없는 과거 로그는 이전 동작 보존을 위해 원점수를 포함한다.
    rows = (
        db.query(
            Utterance.depression_score,
            ModelAuditEvent.id,
            ModelAuditEvent.audit_payload_json,
            ModelAuditEvent.utterance_type,
            ModelAuditEvent.final_is_crisis,
            ModelAuditEvent.hard_crisis,
        )
        .join(ModelAuditEvent, ModelAuditEvent.utterance_id == Utterance.id, isouter=True)
        .filter(
            Utterance.session_id == session.id,
            Utterance.role == "user",
            Utterance.depression_score.isnot(None),
        )
        .order_by(Utterance.created_at.asc())
        .all()
    )
    restored_scores: list[float] = []
    for row in rows:
        contribution_score = _restored_wellness_contribution_score(
            raw_depression_score=row.depression_score,
            audit_id=row.id,
            audit_payload_json=row.audit_payload_json,
            utterance_type=row.utterance_type,
            final_is_crisis=row.final_is_crisis,
            hard_crisis=row.hard_crisis,
        )
        if contribution_score is not None:
            restored_scores.append(contribution_score)
    return restored_scores


def get_today_user_depression_tendency_scores(
    db: Session,
    user_id: int,
    date_str: str,
) -> list[float]:
    """
    역할: 서버 재시작 복원을 위해 오늘 사용자 발화의 depression_tendency_score 조회
    입력: DB 세션, user_id, 날짜 문자열 "YYYY-MM-DD"
    출력: 오늘 사용자 발화 depression_tendency_score 리스트 (시간순, None은 0.0으로 대체)
    """
    session = (
        db.query(ChatSession)
        .filter(ChatSession.user_id == user_id, ChatSession.date == date_str)
        .first()
    )
    if session is None:
        return []

    rows = (
        db.query(
            Utterance.depression_tendency_score,
            Utterance.depression_score,
            ModelAuditEvent.id,
            ModelAuditEvent.audit_payload_json,
            ModelAuditEvent.utterance_type,
            ModelAuditEvent.final_is_crisis,
            ModelAuditEvent.hard_crisis,
        )
        .join(ModelAuditEvent, ModelAuditEvent.utterance_id == Utterance.id, isouter=True)
        .filter(
            Utterance.session_id == session.id,
            Utterance.role == "user",
            Utterance.depression_tendency_score.isnot(None),
        )
        .order_by(Utterance.created_at.asc())
        .all()
    )
    restored_scores: list[float] = []
    for row in rows:
        contribution_score = _restored_wellness_contribution_score(
            raw_depression_score=row.depression_score,
            audit_id=row.id,
            audit_payload_json=row.audit_payload_json,
            utterance_type=row.utterance_type,
            final_is_crisis=row.final_is_crisis,
            hard_crisis=row.hard_crisis,
        )
        if contribution_score is not None:
            restored_scores.append(
                row.depression_tendency_score
                if row.depression_tendency_score is not None
                else 0.0
            )
    return restored_scores


# ── 위기 이벤트 ───────────────────────────────────────────────────────────────
def save_crisis_event(db: Session, user_id: int, data: dict) -> CrisisEvent:
    """
    역할: 위기 이벤트 저장
    입력: DB 세션, user_id, 이벤트 dict
          {text, source ("nli"|"qwen"), entailment_prob, utterance_id (optional)}
    출력: CrisisEvent ORM 객체
    """
    event = CrisisEvent(
        user_id         = user_id,
        utterance_id    = data.get("utterance_id"),
        text            = data["text"],
        source          = data["source"],
        entailment_prob = data.get("entailment_prob"),
    )
    db.add(event)
    db.commit()
    db.refresh(event)
    return event


def get_crisis_count_today(db: Session, user_id: int, date_str: str) -> int:
    """역할: 오늘 위기 이벤트 건수 조회"""
    return (
        db.query(CrisisEvent)
        .join(Utterance, CrisisEvent.utterance_id == Utterance.id, isouter=True)
        .filter(
            CrisisEvent.user_id == user_id,
            CrisisEvent.created_at >= datetime.strptime(date_str, "%Y-%m-%d"),
        )
        .count()
    )


def get_crisis_count_by_session(db: Session, session_id: int) -> int:
    """
    역할: 특정 대화 세션에 연결된 위기 이벤트 건수 조회
    입력: DB 세션, session_id
    출력: 위기 이벤트 개수
    """
    return (
        db.query(CrisisEvent)
        .join(Utterance, CrisisEvent.utterance_id == Utterance.id)
        .filter(Utterance.session_id == session_id)
        .count()
    )


# ── 모델 감사 이벤트 ─────────────────────────────────────────────────────────
def _json_dumps_or_none(value) -> str | None:
    """
    역할: 감사 로그용 중첩 데이터를 JSON 문자열로 안전하게 변환
    입력: 임의의 값
    출력: JSON 문자열 또는 None
    """
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, default=str)


def save_model_audit_event(db: Session, user_id: int, data: dict) -> ModelAuditEvent:
    """
    역할: 모델 판단 세부 근거를 운영 감사 로그로 저장
    입력: DB 세션, user_id, 감사 dict
          {utterance_id, hard_crisis, final_is_crisis, nli_candidate,
           qwen_called, qwen_crisis_tag, qwen_anchor_replaced,
           qwen_anchor_hits, qwen_anchor_similarities, qwen_self_check_verdict,
           qwen_self_check_category, cbt_top_category_source,
           cbt_class_confidence, cbt_head_non_distortion,
           utterance_type, utterance_type_confidence, audit_payload}
    출력: ModelAuditEvent ORM 객체
    """
    event = ModelAuditEvent(
        user_id                       = user_id,
        utterance_id                  = data.get("utterance_id"),
        hard_crisis                   = data.get("hard_crisis"),
        final_is_crisis               = data.get("final_is_crisis"),
        nli_candidate                 = data.get("nli_candidate"),
        qwen_called                   = data.get("qwen_called"),
        qwen_crisis_tag               = data.get("qwen_crisis_tag"),
        qwen_anchor_replaced          = data.get("qwen_anchor_replaced"),
        qwen_anchor_hits_json         = _json_dumps_or_none(data.get("qwen_anchor_hits")),
        qwen_anchor_similarities_json = _json_dumps_or_none(data.get("qwen_anchor_similarities")),
        qwen_self_check_verdict       = data.get("qwen_self_check_verdict"),
        qwen_self_check_category      = data.get("qwen_self_check_category"),
        cbt_top_category_source       = data.get("cbt_top_category_source"),
        cbt_class_confidence          = data.get("cbt_class_confidence"),
        cbt_head_non_distortion       = data.get("cbt_head_non_distortion"),
        utterance_type                = data.get("utterance_type"),
        utterance_type_confidence     = data.get("utterance_type_confidence"),
        audit_payload_json            = _json_dumps_or_none(data.get("audit_payload")),
    )
    db.add(event)
    db.commit()
    db.refresh(event)
    return event


# ── 사용자 피드백 ─────────────────────────────────────────────────────────────
def save_or_update_feedback(
    db: Session,
    user_id: int,
    utterance_id: int,
    feedback_kind: str,
    feedback_value: str,
    model_emotion: str | None = None,
) -> UtteranceFeedback:
    """
    역할: 사용자 피드백 저장 — 같은 (사용자, 발화, 종류) 조합이면 값을 갱신(upsert)
    입력: DB 세션, user_id, 사용자 발화 id, 피드백 종류, 피드백 값, 모델 감정 스냅샷
    출력: UtteranceFeedback ORM 객체
    """
    existing = (
        db.query(UtteranceFeedback)
        .filter(
            UtteranceFeedback.user_id == user_id,
            UtteranceFeedback.utterance_id == utterance_id,
            UtteranceFeedback.feedback_kind == feedback_kind,
        )
        .first()
    )
    if existing:
        existing.feedback_value = feedback_value
        if model_emotion is not None:
            existing.model_emotion = model_emotion
        existing.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(existing)
        return existing

    feedback = UtteranceFeedback(
        user_id        = user_id,
        utterance_id   = utterance_id,
        feedback_kind  = feedback_kind,
        feedback_value = feedback_value,
        model_emotion  = model_emotion,
    )
    db.add(feedback)
    db.commit()
    db.refresh(feedback)
    return feedback


def get_feedback_map_for_utterances(
    db: Session,
    user_id: int,
    utterance_ids: list[int],
) -> dict[int, dict[str, str]]:
    """
    역할: 발화 id 목록에 대한 피드백을 한 번에 조회해 화면 복원용 맵으로 반환
    입력: DB 세션, user_id, 발화 id 리스트
    출력: {utterance_id: {feedback_kind: feedback_value}} dict
    """
    if not utterance_ids:
        return {}
    rows = (
        db.query(UtteranceFeedback)
        .filter(
            UtteranceFeedback.user_id == user_id,
            UtteranceFeedback.utterance_id.in_(utterance_ids),
        )
        .all()
    )
    feedback_map: dict[int, dict[str, str]] = {}
    for row in rows:
        feedback_map.setdefault(row.utterance_id, {})[row.feedback_kind] = row.feedback_value
    return feedback_map


def get_feedback_rows_by_user(db: Session, user_id: int) -> list[UtteranceFeedback]:
    """
    역할: 데이터 내보내기용 — 사용자의 모든 피드백 행을 생성순으로 조회
    입력: DB 세션, user_id
    출력: UtteranceFeedback ORM 리스트
    """
    return (
        db.query(UtteranceFeedback)
        .filter(UtteranceFeedback.user_id == user_id)
        .order_by(UtteranceFeedback.id.asc())
        .all()
    )


# ── 사용자 날짜별 수동 감정 기록 ─────────────────────────────────────────────
def save_or_update_daily_emotion_note(
    db: Session,
    user_id: int,
    date_str: str,
    emotion_label: str,
    intensity: int,
    note: str | None = None,
) -> DailyEmotionNote:
    """
    역할: 날짜별 사용자 수동 감정 기록 저장 — 같은 날짜가 있으면 갱신(upsert)
    입력: DB 세션, user_id, 날짜 문자열, 감정 라벨, 강도(1~5), 메모
    출력: DailyEmotionNote ORM 객체
    """
    cleaned_note = note.strip() if isinstance(note, str) else None
    if cleaned_note == "":
        cleaned_note = None

    existing = (
        db.query(DailyEmotionNote)
        .filter(
            DailyEmotionNote.user_id == user_id,
            DailyEmotionNote.date == date_str,
        )
        .first()
    )
    if existing:
        existing.emotion_label = emotion_label
        existing.intensity = intensity
        existing.note = cleaned_note
        existing.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(existing)
        return existing

    row = DailyEmotionNote(
        user_id=user_id,
        date=date_str,
        emotion_label=emotion_label,
        intensity=intensity,
        note=cleaned_note,
    )
    db.add(row)
    try:
        db.commit()
        db.refresh(row)
        return row
    except IntegrityError:
        # 동시 저장 경쟁에서 먼저 생성된 행을 같은 값으로 갱신한다.
        db.rollback()
        existing = (
            db.query(DailyEmotionNote)
            .filter(
                DailyEmotionNote.user_id == user_id,
                DailyEmotionNote.date == date_str,
            )
            .one()
        )
        existing.emotion_label = emotion_label
        existing.intensity = intensity
        existing.note = cleaned_note
        existing.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(existing)
        return existing


def delete_daily_emotion_note(db: Session, user_id: int, date_str: str) -> bool:
    """
    역할: 특정 날짜의 사용자 수동 감정 기록 삭제
    입력: DB 세션, user_id, 날짜 문자열
    출력: 삭제 여부
    """
    deleted = (
        db.query(DailyEmotionNote)
        .filter(
            DailyEmotionNote.user_id == user_id,
            DailyEmotionNote.date == date_str,
        )
        .delete(synchronize_session=False)
    )
    db.commit()
    return deleted > 0


def get_daily_emotion_notes(
    db: Session,
    user_id: int,
    limit: int = 60,
) -> list[DailyEmotionNote]:
    """
    역할: 사용자의 최근 날짜별 수동 감정 기록 조회
    입력: DB 세션, user_id, 조회 개수
    출력: DailyEmotionNote ORM 리스트 (날짜 내림차순)
    """
    return (
        db.query(DailyEmotionNote)
        .filter(DailyEmotionNote.user_id == user_id)
        .order_by(DailyEmotionNote.date.desc(), DailyEmotionNote.id.desc())
        .limit(limit)
        .all()
    )


def get_daily_emotion_note_map(
    db: Session,
    user_id: int,
    dates: list[str],
) -> dict[str, DailyEmotionNote]:
    """
    역할: 날짜 목록에 해당하는 수동 감정 기록을 맵으로 조회
    입력: DB 세션, user_id, 날짜 문자열 리스트
    출력: {date: DailyEmotionNote} dict
    """
    if not dates:
        return {}
    rows = (
        db.query(DailyEmotionNote)
        .filter(
            DailyEmotionNote.user_id == user_id,
            DailyEmotionNote.date.in_(dates),
        )
        .all()
    )
    return {row.date: row for row in rows}


def get_daily_emotion_note_counts_by_date_between(
    db: Session,
    user_id: int,
    start_date: str,
    end_date: str,
) -> dict[str, dict[str, int]]:
    """
    역할: 리포트 요일 패턴용 — 날짜별 수동 감정 기록 분포 집계
    입력: DB 세션, user_id, 시작 날짜, 끝 날짜
    출력: {날짜: {수동 감정 라벨: 기록 수}} dict
    """
    from sqlalchemy import func

    rows = (
        db.query(DailyEmotionNote.date, DailyEmotionNote.emotion_label, func.count(DailyEmotionNote.id))
        .filter(
            DailyEmotionNote.user_id == user_id,
            DailyEmotionNote.date >= start_date,
            DailyEmotionNote.date <= end_date,
        )
        .group_by(DailyEmotionNote.date, DailyEmotionNote.emotion_label)
        .all()
    )

    counts_by_date: dict[str, dict[str, int]] = {}
    for day, emotion, count in rows:
        counts_by_date.setdefault(day, {})[emotion] = int(count)
    return counts_by_date


def get_daily_emotion_note_by_date(
    db: Session,
    user_id: int,
    date_str: str,
) -> DailyEmotionNote | None:
    """
    역할: 특정 날짜의 수동 감정 기록 단건 조회
    입력: DB 세션, user_id, 날짜 문자열
    출력: DailyEmotionNote ORM 객체 또는 None
    """
    return (
        db.query(DailyEmotionNote)
        .filter(
            DailyEmotionNote.user_id == user_id,
            DailyEmotionNote.date == date_str,
        )
        .first()
    )


# ── 데이터 내보내기 ───────────────────────────────────────────────────────────
def get_sessions_by_user(db: Session, user_id: int) -> list[ChatSession]:
    """
    역할: 데이터 내보내기용 — 사용자의 모든 대화 세션을 날짜순으로 조회
    입력: DB 세션, user_id
    출력: Session ORM 리스트 (날짜 오름차순)
    """
    return (
        db.query(ChatSession)
        .filter(ChatSession.user_id == user_id)
        .order_by(ChatSession.date.asc())
        .all()
    )


def get_all_daily_summaries_by_user(db: Session, user_id: int) -> list[DailySummary]:
    """
    역할: 데이터 내보내기용 — 사용자의 모든 일별 요약을 날짜순으로 조회
    입력: DB 세션, user_id
    출력: DailySummary ORM 리스트 (날짜 오름차순)
    """
    return (
        db.query(DailySummary)
        .filter(DailySummary.user_id == user_id)
        .order_by(DailySummary.date.asc())
        .all()
    )


def get_all_daily_emotion_notes_by_user(db: Session, user_id: int) -> list[DailyEmotionNote]:
    """
    역할: 데이터 내보내기용 — 사용자의 모든 수동 감정 기록을 날짜순으로 조회
    입력: DB 세션, user_id
    출력: DailyEmotionNote ORM 리스트 (날짜 오름차순)
    """
    return (
        db.query(DailyEmotionNote)
        .filter(DailyEmotionNote.user_id == user_id)
        .order_by(DailyEmotionNote.date.asc(), DailyEmotionNote.id.asc())
        .all()
    )


def get_crisis_events_by_user(db: Session, user_id: int) -> list[CrisisEvent]:
    """
    역할: 데이터 내보내기용 — 사용자의 모든 위기 이벤트를 생성순으로 조회
    입력: DB 세션, user_id
    출력: CrisisEvent ORM 리스트
    """
    return (
        db.query(CrisisEvent)
        .filter(CrisisEvent.user_id == user_id)
        .order_by(CrisisEvent.id.asc())
        .all()
    )
