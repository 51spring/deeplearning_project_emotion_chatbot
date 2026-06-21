"""
main.py
역할: FastAPI 메인 앱 — 채팅 API + 캘린더 API + DB 연동
실행: C:/Users/WD/anaconda3/envs/dl_study/python.exe -m uvicorn backend.main:app --reload
"""

from datetime import date, datetime, timedelta, timezone
from contextlib import asynccontextmanager

import os
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import FastAPI, Depends, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session, sessionmaker

from backend.db.models    import init_db, User
from backend.db           import crud
from backend.auth_utils   import (
    IS_PRODUCTION_ENV,
    TOKEN_TYPE,
    create_access_token,
    get_min_password_length,
    is_password_usable,
    verify_access_token,
    verify_password,
)
from backend.scheduler    import ModelScheduler
from backend.runtime_guards import KeyedLockPool, RateLimitRule, SlidingWindowRateLimiter
from backend.crisis_handler import handle_crisis, should_hard_interrupt
from backend.daily_summary  import save_day, get_calendar_data
from pipeline.score_pipeline import ScorePipeline
from pipeline.score_policy import compute_wellness_contribution
from pipeline.action_recommendation import build_chat_recommendations
from pipeline.wellness_score import (
    depression_to_display_label,
    depression_to_display_wellness,
    display_wellness_label,
)

# ── DB 초기화 ─────────────────────────────────────────────────────────────────
engine       = init_db()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def _seed_admin_accounts() -> None:
    """
    역할: 기본 관리자 계정들을 DB에 생성 또는 권한 보정
    입력: 없음
    출력: 없음
    """
    db = SessionLocal()
    try:
        users = crud.ensure_admin_users(db)
        prepared = ", ".join(f"{user.username}(id={user.id})" for user in users)
        print(f"[DB seed] 관리자 계정 준비 완료: {prepared}")
    finally:
        db.close()


_seed_admin_accounts()

# ── 모델 스케줄러 (앱 수명 동안 1개) ─────────────────────────────────────────
scheduler = ModelScheduler(use_cbt=True)

# ── 사용자별 ScorePipeline 인스턴스 (메모리 내 상태 유지) ───────────────────
_pipelines: dict[int, ScorePipeline] = {}

# ── 사용자별 운영 모니터링용 활성 날짜 ───────────────────────────────────────
# 실제 날짜를 기다리지 않고 하루씩 넘기는 UI 테스트 흐름에서만 사용한다.
_active_dates_by_user: dict[int, str] = {}

# ── 화면 채팅창별 Qwen 문맥 시작점 ───────────────────────────────────────────
# 화면을 새로고침하거나 채팅창이 새로 마운트되면 프론트가 새 client_session_id를 보낸다.
# DB의 날짜 단위 세션은 점수/기록 보존용으로 유지하되, Qwen history는 이 시작점 이후만 사용한다.
_client_context_starts: dict[tuple[int, str, str], int] = {}

# 사용자별 점수/날짜/문맥 메모리 상태는 같은 계정 요청끼리 직렬화한다.
_user_state_locks = KeyedLockPool()

# 가입/로그인/채팅 요청 횟수 제한은 프로세스 전체에서 공유한다.
_request_rate_limiter = SlidingWindowRateLimiter()
_login_attempt_locks = KeyedLockPool()

# Qwen 입력 context 제한: 최근 대화는 DB에서 직접 제한 조회하고, 오래된 당일 흐름은 요약만 전달한다.
RECENT_HISTORY_LIMIT = 20

# 리포트 요일 패턴은 한 주만 보면 요일당 표본이 1개뿐이라 최근 8주를 기본 창으로 둔다.
WEEKDAY_PATTERN_WINDOW_DAYS = 56
WEEKDAY_PATTERN_WEEKS = 8
WEEKDAY_LABELS_KO = ("월", "화", "수", "목", "금", "토", "일")
ROLLING_SUMMARY_OLDER_LIMIT = 40
ROLLING_SUMMARY_MAX_CHARS = 500
ROLLING_SUMMARY_SAMPLE_LIMIT = 4

# LLM 서사 요약: N 개의 새 사용자 발화가 누적될 때마다 Qwen 으로 갱신
NARRATIVE_REFRESH_EVERY = 8
NARRATIVE_MAX_INPUT_UTTS = 24  # 요약 입력으로 사용할 사용자 발화 최대 개수

# 자동 하루 마감: 한 번의 요청에서 따라잡을 수 있는 최대 과거 일수
AUTO_CLOSE_MAX_DAYS = 14


def _positive_env_int(name: str, default: int) -> int:
    """
    역할: 양의 정수 환경변수를 읽고 잘못된 값이면 기본값 사용
    입력: 환경변수 이름, 기본 정수
    출력: 1 이상의 정수
    """
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError:
        value = default
    return max(1, value)


REGISTER_RATE_RULE = RateLimitRule(
    _positive_env_int("EMOTION_CHATBOT_REGISTER_RATE_LIMIT", 20),
    _positive_env_int("EMOTION_CHATBOT_REGISTER_RATE_WINDOW_SECONDS", 3600),
)
LOGIN_RATE_RULE = RateLimitRule(
    _positive_env_int("EMOTION_CHATBOT_LOGIN_RATE_LIMIT", 30),
    _positive_env_int("EMOTION_CHATBOT_LOGIN_RATE_WINDOW_SECONDS", 300),
)
PASSWORD_RESET_RATE_RULE = RateLimitRule(
    _positive_env_int("EMOTION_CHATBOT_PASSWORD_RESET_RATE_LIMIT", 10),
    _positive_env_int("EMOTION_CHATBOT_PASSWORD_RESET_RATE_WINDOW_SECONDS", 3600),
)
CHAT_MINUTE_RATE_RULE = RateLimitRule(
    _positive_env_int("EMOTION_CHATBOT_CHAT_RATE_LIMIT", 12),
    _positive_env_int("EMOTION_CHATBOT_CHAT_RATE_WINDOW_SECONDS", 60),
)
CHAT_HOUR_RATE_RULE = RateLimitRule(
    _positive_env_int("EMOTION_CHATBOT_CHAT_HOURLY_LIMIT", 120),
    _positive_env_int("EMOTION_CHATBOT_CHAT_HOURLY_WINDOW_SECONDS", 3600),
)
LOGIN_MAX_FAILURES = _positive_env_int("EMOTION_CHATBOT_LOGIN_MAX_FAILURES", 5)
LOGIN_LOCK_SECONDS = _positive_env_int("EMOTION_CHATBOT_LOGIN_LOCK_SECONDS", 900)
TRUST_PROXY_HEADERS = (
    os.environ.get("EMOTION_CHATBOT_TRUST_PROXY_HEADERS", "")
    .strip()
    .lower()
    in {"1", "true", "yes", "on"}
)

# 운영 날짜 시간대: Docker/서버 OS가 UTC여도 사용자 날짜는 한국 시간 기준으로 계산한다.
APP_TIMEZONE_NAME = os.environ.get("EMOTION_CHATBOT_TIMEZONE", "Asia/Seoul").strip()
try:
    APP_TIMEZONE = ZoneInfo(APP_TIMEZONE_NAME)
except ZoneInfoNotFoundError:
    # 한국은 DST가 없어 tzdata가 없는 최소 이미지에서도 UTC+9 고정값으로 안전하게 동작한다.
    APP_TIMEZONE = timezone(timedelta(hours=9))
    print(
        f"[timezone] {APP_TIMEZONE_NAME} 정보를 찾지 못해 UTC+09:00 고정 시간대를 사용합니다."
    )

# 사용자 피드백 검증용 상수
FEEDBACK_KIND_RESPONSE = "response_rating"
FEEDBACK_KIND_EMOTION  = "emotion_correction"
FEEDBACK_RESPONSE_VALUES = {"good", "bad"}
# 7클래스 감정 한국어 라벨 — 감정 셀프 정정 허용 값
EMOTION_LABELS_KO = {"중립", "행복", "슬픔", "분노", "공포", "혐오", "놀람"}

# 이메일은 외부 패키지 없이 도메인 점과 TLD까지 보수적으로 검증한다.
# 실제 서비스는 여기에 메일 인증 토큰을 반드시 추가해야 한다.
EMAIL_LOCAL_ALLOWED_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.!#$%&'*+/=?^_`{|}~-")
EMAIL_DOMAIN_ALLOWED_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-")

SUMMARY_THEME_KEYWORDS = {
    "수면·피로": ["잠", "불면", "피곤", "지쳤", "무기력", "쉬고", "눕고"],
    "학업·일": ["공부", "과제", "시험", "회사", "일", "업무", "출근", "마감"],
    "관계": ["친구", "가족", "부모", "엄마", "아빠", "연인", "사람", "혼자"],
    "자기비난": ["내 탓", "못났", "실패", "한심", "싫어", "자책", "미안"],
    "불안·걱정": ["불안", "걱정", "두려", "무서", "긴장", "초조"],
    "상실·슬픔": ["슬프", "눈물", "외롭", "허전", "그리", "상실"],
}

# CBT 앵커 카테고리(JSON `name`) → rolling summary 테마 라벨 매핑
# 키워드 사전이 못 잡는 인지 왜곡 패턴을 임베딩 기반으로 보강한다.
# 자기비난·개인화는 기존 키워드 테마 "자기비난"과 dedup 되도록 같은 라벨로 매핑한다.
# v3 prototype-only(2026-04-26): KoACD 라벨 체계 기반 신규 5범주 추가 (총 10범주).
CBT_CATEGORY_TO_THEME = {
    "이분법적 사고":     "이분법적 사고",
    "과잉일반화":        "과잉일반화",
    "파국화":            "파국화",
    "자기비난·개인화":   "자기비난",
    "감정적 추론":       "감정적 추론",
    # v3 신규 5범주
    "부정적 편향":       "부정적 편향",
    "낙인찍기":          "낙인찍기",
    "긍정 축소화":       "긍정 축소화",
    "당위 진술":         "당위 진술",
    "성급한 판단":       "성급한 판단",
}
# 같은 카테고리가 N회 이상 누적되어야 테마로 채택 (단발 노이즈 차단)
CBT_THEME_MIN_COUNT = 2


def _serialize_daily_emotion_note(note) -> dict:
    """
    역할: DailyEmotionNote ORM 객체를 API 응답 dict로 변환
    입력: 수동 감정 기록 ORM 객체
    출력: 프론트 캘린더에서 사용하는 manual_emotion_* 필드 dict
    """
    return {
        "date": note.date,
        "manual_emotion_label": note.emotion_label,
        "manual_emotion_intensity": note.intensity,
        "manual_emotion_note": note.note,
        "manual_emotion_updated_at": (
            note.updated_at.isoformat() if note.updated_at else None
        ),
    }


def _get_app_today() -> date:
    """
    역할: 운영 시간대 기준 오늘 날짜 계산
    입력: 없음
    출력: 운영 시간대의 date 객체
    """
    return datetime.now(APP_TIMEZONE).date()


def get_pipeline(db: Session, user_id: int, date_str: str | None = None) -> ScorePipeline:
    """
    역할: 사용자별 ScorePipeline 인스턴스 반환 후 DB 히스토리로 상태 동기화
    입력: DB 세션, user_id, 날짜 문자열(기본값: 오늘)
    출력: ScorePipeline 인스턴스
    """
    if user_id not in _pipelines:
        # ScorePipeline은 모델을 직접 보유하지 않고 scheduler를 통해 추론
        _pipelines[user_id] = ScorePipeline(
            model=None, tokenizer=None, device=None,
            anchor_embs={}, use_cbt=False,  # scheduler가 대신 추론
        )
    pipeline = _pipelines[user_id]
    _sync_pipeline_history(pipeline, db, user_id, date_str or _get_app_today().isoformat())
    return pipeline


# ── FastAPI 앱 ────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[서버 시작] 모델 스케줄러 준비 완료")
    # 재배포 후에도 관리자 시뮬레이션 날짜를 복원해 채팅/캘린더 불일치를 막는다.
    _rehydrate_admin_active_dates()
    yield
    print("[서버 종료]")

app = FastAPI(title="감정 모니터링 상담 챗봇", lifespan=lifespan)

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_BUILD_DIR = BASE_DIR / "frontend" / "build"

DEFAULT_CORS_ORIGINS = "http://localhost:3000,http://127.0.0.1:3000"
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("EMOTION_CHATBOT_CORS_ORIGINS", DEFAULT_CORS_ORIGINS).split(",")
    if origin.strip()
]
STORE_QWEN_RAW_RESPONSE = (
    not IS_PRODUCTION_ENV
    or os.environ.get("EMOTION_CHATBOT_STORE_QWEN_RAW_RESPONSE", "").strip().lower()
    in {"1", "true", "yes", "on"}
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _mount_frontend_build(fastapi_app: FastAPI) -> None:
    """
    역할: React production build를 FastAPI 같은 origin에서 정적 파일로 제공
    입력: FastAPI 앱 인스턴스
    출력: 없음
    """
    index_file = FRONTEND_BUILD_DIR / "index.html"
    if not index_file.exists():
        print(
            "[frontend] frontend/build/index.html이 없어 정적 서빙을 건너뜁니다. "
            "배포 전 `npm --prefix frontend run build`를 실행하세요."
        )
        return

    # API 라우트 등록 뒤 맨 마지막에 mount해야 /chat, /auth/login 같은 엔드포인트가 우선 매칭된다.
    fastapi_app.mount(
        "/",
        StaticFiles(directory=str(FRONTEND_BUILD_DIR), html=True),
        name="frontend",
    )
    print(f"[frontend] React build 정적 서빙 활성화: {FRONTEND_BUILD_DIR}")


def get_db():
    """역할: FastAPI 의존성 — DB 세션 제공"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _get_active_date(user_id: int) -> str:
    """
    역할: 사용자별 현재 활성 날짜 조회
    입력: user_id
    출력: YYYY-MM-DD 날짜 문자열
    """
    return _active_dates_by_user.get(user_id, _get_app_today().isoformat())


def _advance_active_date(user_id: int) -> str:
    """
    역할: 사용자별 활성 날짜를 하루 증가
    입력: user_id
    출력: 증가 후 YYYY-MM-DD 날짜 문자열
    """
    try:
        current = date.fromisoformat(_get_active_date(user_id))
    except ValueError:
        current = _get_app_today()
    next_date = (current + timedelta(days=1)).isoformat()
    _active_dates_by_user[user_id] = next_date
    return next_date


def _rehydrate_admin_active_dates() -> None:
    """
    역할: 서버 재시작(재배포) 시 관리자(developer/root)의 "다음날로 넘기기"
          시뮬레이션 활성 날짜를 DB 기록으로부터 복원한다.
          `_active_dates_by_user`는 인메모리라 재시작 시 사라진다. 복원하지 않으면
          관리자가 시뮬레이션하던 날짜 대신 실제 today의 빈 대화창으로 되돌아가,
          캘린더(=DB)에는 기록이 남아 있는데 채팅 화면만 새 대화창으로 뜨는 불일치가 생긴다.
    입력: 없음 (전역 `_active_dates_by_user`를 갱신)
    출력: 없음
    """
    real_today = _get_app_today()
    db = SessionLocal()
    try:
        for username in crud.ADMIN_USERNAMES:
            user = crud.get_user_by_username(db, username)
            if user is None or user.id in _active_dates_by_user:
                continue

            # 시뮬레이션 활성 날짜 재구성 후보:
            #  - 마지막으로 발화가 있던 날 (아직 안 마감된 '열린' 날일 수 있음)
            #  - 마지막으로 마감된 날의 다음 날 (넘기기 직후의 빈 날)
            #  - 실제 today (하한)
            candidates = [real_today]
            sessions = crud.get_sessions_by_user(db, user.id)
            if sessions:
                try:
                    candidates.append(date.fromisoformat(sessions[-1].date))
                except ValueError:
                    pass
            summaries = crud.get_all_daily_summaries_by_user(db, user.id)
            if summaries:
                try:
                    candidates.append(
                        date.fromisoformat(summaries[-1].date) + timedelta(days=1)
                    )
                except ValueError:
                    pass

            resolved = max(candidates)
            # 시뮬레이션으로 실제 today를 넘어선 경우에만 복원한다.
            # (실제 today와 같으면 일반 경로의 자동 마감이 동작하도록 둔다.)
            if resolved > real_today:
                _active_dates_by_user[user.id] = resolved.isoformat()
                print(
                    f"[관리자 시뮬레이션 날짜 복원] {user.username} → {resolved.isoformat()}"
                )
    finally:
        db.close()


def _tendency_band_code(score: float | None) -> str | None:
    """
    역할: 우울 경향 점수(0~1)를 표시용 정성 밴드 코드로 변환
          (high ≥ 0.40 / mid ≥ 0.20 / low). depression_tendency_v15_spec 및
          프론트 classifyTendencyBand 임계값과 일치시킨다.
    입력: depression_tendency_score 또는 None
    출력: "high" | "mid" | "low" | None (점수가 없으면 None)
    """
    if score is None:
        return None
    value = float(score)
    if value >= 0.40:
        return "high"
    if value >= 0.20:
        return "mid"
    return "low"


def _gate_diagnostic_scores(payload: dict, is_developer: bool) -> dict:
    """
    역할: 비관리자 사용자에게 내려보내는 응답에서 모델 내부 raw 점수
          (종합 distress, 우울 경향 소수점)를 가린다. 대신 우울 경향은 정성 밴드
          (high/mid/low)로만 노출해 "우울 경향은 알 수 있게" 유지한다.
          관리자 계정(developer/root)은 진단·모니터링용 raw 점수를 그대로 받는다.
    입력: 응답 dict (daily_score/smoothed_score/depression_tendency_* 포함 가능),
          관리자 여부
    출력: 게이팅을 적용한 동일 dict (in-place 수정 후 반환)
    """
    # 밴드는 raw 제거 전에 먼저 계산한다. 평활값 우선, 없으면 일별값을 사용한다.
    tendency_for_band = payload.get("depression_tendency_smoothed")
    if tendency_for_band is None:
        tendency_for_band = payload.get("depression_tendency_daily")
    payload["depression_tendency_band"] = _tendency_band_code(tendency_for_band)

    # 관리자가 아니면 raw 내부 점수를 응답에서 비운다(브라우저로 누설 방지).
    # 프론트는 값이 None이면 해당 소수점 행/추세 차트를 자동으로 숨긴다.
    if not is_developer:
        for key in (
            "daily_score",
            "smoothed_score",
            "depression_tendency_daily",
            "depression_tendency_smoothed",
        ):
            if key in payload:
                payload[key] = None
    return payload


def _sorted_count_dict(counts: dict[str, int]) -> dict[str, int]:
    """
    역할: 감정 카운트 dict를 빈도 내림차순, 같은 빈도는 라벨 오름차순으로 정렬
    입력: {라벨: 개수}
    출력: 정렬된 dict
    """
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _weekday_focus_message(source_label: str, weekday_label: str, emotion: str) -> str:
    """
    역할: 특정 요일에 감정이 몰릴 때 리포트에 보여줄 관찰 문장 생성
    입력: 데이터 출처 라벨, 요일 라벨, 감정 라벨
    출력: 단정/진단을 피한 한국어 패턴 해석 문장
    """
    prefix = f"{source_label} 기준으로 {weekday_label}요일에 {emotion}이 반복적으로 나타났어요."
    if emotion in {"슬픔", "공포"}:
        if weekday_label in {"월", "일"}:
            return f"{prefix} 주간 전환이나 다음 일정 부담이 함께 있었는지 살펴볼 수 있어요."
        return f"{prefix} 그 요일 전후의 피로, 걱정, 회복 시간을 같이 점검해볼 수 있어요."
    if emotion in {"분노", "혐오"}:
        return f"{prefix} 반복되는 마찰 상황이나 환경 스트레스가 있었는지 돌아볼 수 있어요."
    if emotion == "행복":
        return f"{prefix} 회복감을 만든 활동이나 관계를 다음 주에도 남겨볼 만해요."
    if emotion == "놀람":
        return f"{prefix} 예상 밖 일정이나 변화가 잦았는지 확인해볼 수 있어요."
    return f"{prefix} 비교적 큰 감정 변화 없이 지나간 루틴이 있었는지 살펴볼 수 있어요."


def _build_weekday_patterns(rows: list[dict], source_label: str) -> list[dict]:
    """
    역할: 요일별 감정분포에서 특정 감정 쏠림을 찾아 해석 문장 생성
    입력: 요일별 row 리스트, 데이터 출처 라벨
    출력: [{weekday, emotion, message, tone}] 리스트
    """
    candidates: list[dict] = []
    total = sum(int(row.get("total") or 0) for row in rows)
    for row in rows:
        top_emotion = row.get("top_emotion")
        top_count = int(row.get("top_count") or 0)
        top_share = float(row.get("top_share") or 0)
        # 요일별 표본이 2개 이상이고 한 감정이 절반 이상일 때만 "몰림"으로 본다.
        if top_emotion and top_count >= 2 and top_share >= 0.5:
            candidates.append({
                "weekday": row["weekday"],
                "emotion": top_emotion,
                "count": top_count,
                "share": round(top_share, 2),
                "tone": "positive" if top_emotion == "행복" else "notice",
                "message": _weekday_focus_message(source_label, row["weekday"], top_emotion),
            })

    if candidates:
        return sorted(
            candidates,
            key=lambda item: (-item["count"], -item["share"], item["weekday"]),
        )[:3]

    if total < 4:
        return [{
            "weekday": None,
            "emotion": None,
            "count": 0,
            "share": None,
            "tone": "neutral",
            "message": f"{source_label} 기록이 조금 더 쌓이면 요일별 반복 패턴을 볼 수 있어요.",
        }]

    return [{
        "weekday": None,
        "emotion": None,
        "count": 0,
        "share": None,
        "tone": "neutral",
        "message": f"{source_label} 기준으로 특정 요일에 감정이 뚜렷하게 몰리지는 않았어요.",
    }]


def _build_weekday_emotion_source(
    counts_by_date: dict[str, dict[str, int]],
    source_key: str,
    source_label: str,
) -> dict:
    """
    역할: 날짜별 감정 카운트를 월~일 요일별 감정분포로 변환
    입력: {날짜: {감정: 개수}}, 출처 key/라벨
    출력: 출처별 요일 감정분포와 패턴 문장
    """
    buckets: list[dict[str, int]] = [{} for _ in range(7)]
    for day_text, emotion_counts in counts_by_date.items():
        try:
            weekday_index = date.fromisoformat(day_text).weekday()
        except ValueError:
            continue
        for emotion, count in emotion_counts.items():
            buckets[weekday_index][emotion] = buckets[weekday_index].get(emotion, 0) + int(count)

    rows: list[dict] = []
    for index, counts in enumerate(buckets):
        sorted_counts = _sorted_count_dict(counts)
        total = sum(sorted_counts.values())
        top_emotion = next(iter(sorted_counts), None)
        top_count = sorted_counts.get(top_emotion, 0) if top_emotion else 0
        rows.append({
            "weekday_index": index,
            "weekday": WEEKDAY_LABELS_KO[index],
            "total": total,
            "emotion_counts": sorted_counts,
            "top_emotion": top_emotion,
            "top_count": top_count,
            "top_share": round(top_count / total, 3) if total else None,
        })

    return {
        "source": source_key,
        "source_label": source_label,
        "total": sum(row["total"] for row in rows),
        "weekdays": rows,
        "patterns": _build_weekday_patterns(rows, source_label),
    }


def _build_weekly_summary_items(summary: dict, is_developer: bool) -> list[dict]:
    """
    역할: 주간 리포트 summary 값을 사용자가 읽기 쉬운 관찰 문장으로 변환
    입력: summary dict, 관리자 여부
    출력: [{title, body, tone}] 리스트
    """
    items: list[dict] = []
    top_emotion = summary.get("top_emotion")
    emotion_counts = summary.get("emotion_counts") or {}
    active_days = int(summary.get("active_days") or 0)
    total_utterances = int(summary.get("total_utterances") or 0)
    crisis_count = int(summary.get("crisis_count") or 0)

    if total_utterances == 0:
        items.append({
            "title": "기록 흐름",
            "body": "이번 주에는 대화 기록이 아직 적어요. 기록이 쌓이면 감정 흐름을 더 안정적으로 볼 수 있어요.",
            "tone": "neutral",
        })
    elif top_emotion:
        items.append({
            "title": "감정 흐름",
            "body": f"이번 주에는 {top_emotion} 감정이 가장 자주 감지됐어요({emotion_counts.get(top_emotion, 0)}회).",
            "tone": "neutral",
        })

    daily_avg = summary.get("avg_daily_wellness")
    cumulative_avg = summary.get("avg_cumulative_wellness")
    if cumulative_avg is None:
        cumulative_avg = summary.get("avg_wellness")
    cumulative_delta = summary.get("cumulative_wellness_delta")
    if cumulative_delta is None:
        cumulative_delta = summary.get("wellness_delta")
    if daily_avg is not None or cumulative_avg is not None:
        daily_label = display_wellness_label(daily_avg) if daily_avg is not None else None
        cumulative_label = (
            display_wellness_label(cumulative_avg)
            if cumulative_avg is not None
            else None
        )
        if is_developer:
            base = (
                f"마감된 날 기준 단일 평균은 {round(float(daily_avg), 1) if daily_avg is not None else '—'}점,"
                f" 누적 평균은 {round(float(cumulative_avg), 1) if cumulative_avg is not None else '—'}점이에요."
            )
            if cumulative_delta is not None:
                if cumulative_delta > 0:
                    base += f" 누적 흐름은 지난주보다 {abs(cumulative_delta)}점 높아요."
                elif cumulative_delta < 0:
                    base += f" 누적 흐름은 지난주보다 {abs(cumulative_delta)}점 낮아요."
                else:
                    base += " 누적 흐름은 지난주와 비슷해요."
        else:
            base = f"마감된 날 기준 단일 상태는 {daily_label or '—'}, 누적 상태는 {cumulative_label or '—'}로 보였어요."
            if cumulative_delta is not None:
                if cumulative_delta > 0:
                    base += " 지난주보다 누적 흐름이 조금 가벼워진 편이에요."
                elif cumulative_delta < 0:
                    base += " 지난주보다 누적 흐름이 조금 무거워진 편이에요."
                else:
                    base += " 지난주와 비교해 누적 흐름은 비슷해요."
        items.append({"title": "웰니스 흐름", "body": base, "tone": "neutral"})

    if active_days >= 5:
        items.append({
            "title": "기록 리듬",
            "body": f"이번 주에는 {active_days}일 동안 기록이 남아 있어 흐름을 비교하기 좋아요.",
            "tone": "positive",
        })
    elif total_utterances > 0:
        items.append({
            "title": "기록 리듬",
            "body": f"이번 주 기록은 {active_days}일에 모여 있어요. 기록하지 않은 날의 상태는 리포트에 적게 반영돼요.",
            "tone": "neutral",
        })

    items.append({
        "title": "안전 신호",
        "body": (
            f"이번 주 위기 신호가 {crisis_count}건 감지됐어요. 힘든 순간에는 안전 확인을 먼저 해주세요."
            if crisis_count > 0
            else "이번 주에는 감지된 위기 신호가 없었어요."
        ),
        "tone": "warning" if crisis_count > 0 else "positive",
    })
    return items


def _close_day_for_date(db: Session, user_id: int, date_str: str) -> dict:
    """
    역할: 지정 날짜를 하루 마감하고 DailySummary 저장
    입력: DB 세션, user_id, 날짜 문자열
    출력: 하루 요약 응답 dict
    """
    pipeline = get_pipeline(db, user_id, date_str)

    history_wellness = list(pipeline._daily_wellness)
    n_days = len(history_wellness) + 1

    # 파이프라인 close_day에 히스토리 주입
    pipeline._daily_wellness = history_wellness
    result = pipeline.close_day()
    result["n_days"] = n_days

    # DB count 쿼리로 해당 날짜 세션의 실제 사용자 발화와 위기 이벤트만 집계한다.
    session = crud.get_or_create_session(db, user_id, date_str)
    utterance_count = crud.count_user_utterances_by_session(db, session.id)
    crisis_count = crud.get_crisis_count_by_session(db, session.id)

    save_day(db, user_id, date_str, result, utterance_count, crisis_count)

    daily_wellness = result.get(
        "daily_wellness_score",
        depression_to_display_wellness(result["daily_score"]),
    )
    cumulative_wellness = result.get(
        "cumulative_wellness_score",
        result["wellness_score"],
    )

    return {
        "date":           date_str,
        "daily_score":    result["daily_score"],
        "smoothed_score": result["smoothed_score"],
        "daily_wellness_score": daily_wellness,
        "daily_wellness_label": result.get(
            "daily_wellness_label",
            display_wellness_label(daily_wellness),
        ),
        "cumulative_wellness_score": cumulative_wellness,
        "cumulative_wellness_label": result.get(
            "cumulative_wellness_label",
            result["label"],
        ),
        # 기존 호환용 필드: 하루 단독 점수가 아니라 누적/평활 웰니스 점수다.
        "wellness_score": result["wellness_score"],
        "label":          result["label"],
        "depression_tendency_daily":    result.get("depression_tendency_daily"),
        "depression_tendency_smoothed": result.get("depression_tendency_smoothed"),
        "utterance_count": utterance_count,
        "crisis_count":   crisis_count,
    }


def _auto_close_stale_days(db: Session, user_id: int) -> list[str]:
    """
    역할: 자동 하루 마감 — 실제 날짜가 지났는데 마감되지 않은 과거 날짜를
          사용자가 다시 접속한 시점에 오래된 순서로 자동 마감(lazy close)한다.
          서버가 자정에 꺼져 있어도 다음 접속 때 캘린더/리포트가 채워진다.
    입력: DB 세션, user_id
    출력: 이번 요청에서 자동 마감된 날짜 문자열 리스트
    """
    # 관리자 "다음날로 넘기기" 시뮬레이션 중에는 가상 날짜를 쓰므로
    # 실제 달력 기준 자동 마감을 적용하지 않는다 (/day/advance가 직접 마감).
    if user_id in _active_dates_by_user:
        return []

    today = _get_app_today().isoformat()
    stale_dates = crud.get_unclosed_session_dates(
        db, user_id, before_date=today, limit=AUTO_CLOSE_MAX_DAYS,
    )
    closed: list[str] = []
    for stale_date in stale_dates:
        try:
            # EWMA 히스토리가 날짜 순서대로 쌓이도록 오래된 날짜부터 마감한다.
            _close_day_for_date(db, user_id, stale_date)
            closed.append(stale_date)
        except Exception as exc:
            # 일부 날짜 마감 실패가 채팅/조회 요청 자체를 막지 않게 한다.
            print(f"[auto-close] {stale_date} 자동 마감 실패: {exc}")
            break
    if closed:
        print(f"[auto-close] 지난 날짜 자동 마감 완료: {', '.join(closed)}")
    return closed


# ── Pydantic 스키마 ───────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    username: str = Field(min_length=2, max_length=32)
    text: str = Field(min_length=1, max_length=2000)
    client_session_id: str | None = Field(default=None, max_length=128)


class ChatResponse(BaseModel):
    response: str
    is_crisis: bool
    crisis_message: str | None
    top_emotion: str
    roberta_score: float
    depression_score: float
    depression_tendency_score: float
    wellness_score: float
    label: str
    # 이번 교환의 사용자 발화 id — 프론트 피드백(응답 평가/감정 정정) 전송에 사용
    utterance_id: int | None = None
    # 오늘의 감정 기록 기반 자기관리 추천(행동 추천 v1, 실시간만 — DB 저장 없음)
    recommendations: list[dict] = Field(default_factory=list)


class DayCloseRequest(BaseModel):
    username: str
    date: str | None = None


class DayAdvanceRequest(BaseModel):
    username: str


class FeedbackRequest(BaseModel):
    """사용자 피드백 요청 스키마 — 응답 평가 또는 감정 셀프 정정"""
    username: str
    utterance_id: int
    kind: str    # "response_rating" | "emotion_correction"
    value: str   # "good"|"bad" 또는 7감정 한국어 라벨


class DailyEmotionNoteRequest(BaseModel):
    """캘린더 수동 감정 기록 요청 스키마"""
    username: str
    date: str
    emotion_label: str
    intensity: int = Field(default=3, ge=1, le=5)
    note: str | None = Field(default=None, max_length=300)


class ChangePasswordRequest(BaseModel):
    """비밀번호 변경 요청 스키마"""
    username: str
    current_password: str
    new_password: str


class AccountDeleteRequest(BaseModel):
    """계정 삭제 요청 스키마 — 비밀번호 재확인 + 확인 문구 필요"""
    username: str
    password: str
    confirm: str


class DbResetRequest(BaseModel):
    """관리자 DB 초기화 요청 스키마"""
    username: str
    confirm: str


class LoginRequest(BaseModel):
    """아이디/비밀번호 로그인 요청 스키마"""
    username: str = Field(min_length=2, max_length=32)
    password: str = Field(min_length=1, max_length=256)


class RegisterRequest(BaseModel):
    """닉네임/아이디/이메일/비밀번호 가입 요청 스키마"""
    nickname: str = Field(min_length=1, max_length=32)
    username: str = Field(min_length=2, max_length=32)
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=1, max_length=256)


class PasswordResetRequest(BaseModel):
    """이메일 확인 기반 비밀번호 재설정 요청 스키마"""
    username: str = Field(min_length=2, max_length=32)
    email: str = Field(min_length=3, max_length=254)
    new_password: str = Field(min_length=1, max_length=256)


class AuthResponse(BaseModel):
    """인증 성공 응답 스키마"""
    username: str
    nickname: str | None = None
    email: str | None = None
    is_developer: bool
    access_token: str
    token_type: str
    expires_at: int


# ── 엔드포인트 ────────────────────────────────────────────────────────────────
def _build_auth_response(user: User) -> AuthResponse:
    """
    역할: 인증 성공 사용자에게 프론트 저장용 토큰 응답 생성
    입력: User ORM 객체
    출력: AuthResponse
    """
    access_token, expires_at = create_access_token(user.id, user.username)
    return AuthResponse(
        username=user.username,
        nickname=user.nickname or user.username,
        email=user.email,
        is_developer=crud.is_developer_user(user),
        access_token=access_token,
        token_type=TOKEN_TYPE,
        expires_at=expires_at,
    )


def _validate_username(username: str) -> str:
    """
    역할: 로그인 식별자인 아이디를 공통 검증
    입력: 사용자 입력 아이디
    출력: 정리된 아이디
    """
    username = str(username).strip()
    if not (2 <= len(username) <= 32):
        raise HTTPException(
            status_code=400,
            detail="아이디는 2~32자여야 합니다.",
        )
    if any(ch.isspace() for ch in username):
        raise HTTPException(
            status_code=400,
            detail="아이디에는 공백을 사용할 수 없습니다.",
        )
    return username


def _validate_password(password: str, field_name: str = "비밀번호") -> str:
    """
    역할: 환경별 최소 길이에 맞춰 비밀번호 입력을 검증
    입력: 비밀번호 원문, 오류 메시지용 필드명
    출력: 검증된 비밀번호 원문
    """
    password = str(password)
    if not is_password_usable(password):
        raise HTTPException(
            status_code=400,
            detail=f"{field_name}는 {get_min_password_length()}자 이상이어야 합니다.",
        )
    return password


def _is_valid_email_format(email: str) -> bool:
    """
    역할: 이메일 문자열이 데모 계정용 기본 형식을 만족하는지 확인
    입력: 소문자 정규화된 이메일 문자열
    출력: 형식 유효 여부
    """
    if not (3 <= len(email) <= 254):
        return False
    if any(ch.isspace() or ord(ch) < 32 or ord(ch) == 127 for ch in email):
        return False
    if email.count("@") != 1:
        return False

    local_part, domain = email.split("@", 1)
    if not local_part or not domain:
        return False
    if len(local_part) > 64 or len(domain) > 253:
        return False
    if local_part.startswith(".") or local_part.endswith(".") or ".." in local_part:
        return False
    if any(ch not in EMAIL_LOCAL_ALLOWED_CHARS for ch in local_part):
        return False

    # `test@gmail`처럼 점이 없는 도메인은 브라우저가 허용할 수 있어 백엔드에서 차단한다.
    if "." not in domain:
        return False
    labels = domain.split(".")
    if any(not label or len(label) > 63 for label in labels):
        return False

    for label in labels:
        if label.startswith("-") or label.endswith("-"):
            return False
        if any(ch not in EMAIL_DOMAIN_ALLOWED_CHARS for ch in label):
            return False

    tld = labels[-1]
    if not (2 <= len(tld) <= 63) or not tld.isalpha():
        return False
    return True


def _validate_email(email: str) -> str:
    """
    역할: 계정 복구용 이메일 형식을 검증하고 정규화
    입력: 이메일 원문
    출력: 소문자 정규화 이메일
    """
    normalized = crud.normalize_email(email)
    if not _is_valid_email_format(normalized):
        raise HTTPException(
            status_code=400,
            detail="이메일은 example@gmail.com처럼 도메인 점(.)과 끝 주소를 포함해야 합니다.",
        )
    return normalized


def _validate_nickname(nickname: str) -> str:
    """
    역할: 화면 표시용 닉네임을 검증
    입력: 닉네임 원문
    출력: 정리된 닉네임
    """
    normalized = str(nickname).strip()
    if not (1 <= len(normalized) <= 32):
        raise HTTPException(
            status_code=400,
            detail="닉네임은 1~32자여야 합니다.",
        )
    return normalized


def _validate_login_request(req: LoginRequest) -> tuple[str, str]:
    """
    역할: 로그인 API 입력값을 공통 검증
    입력: LoginRequest
    출력: 정리된 사용자 아이디와 비밀번호
    """
    return _validate_username(req.username), _validate_password(req.password)


def _validate_register_request(req: RegisterRequest) -> tuple[str, str, str, str]:
    """
    역할: 가입 API의 닉네임/아이디/이메일/비밀번호 입력값을 검증
    입력: RegisterRequest
    출력: (아이디, 닉네임, 이메일, 비밀번호)
    """
    username = _validate_username(req.username)
    nickname = _validate_nickname(req.nickname)
    email = _validate_email(req.email)
    password = _validate_password(req.password)
    return username, nickname, email, password


def _validate_password_reset_request(req: PasswordResetRequest) -> tuple[str, str, str]:
    """
    역할: 비밀번호 재설정 입력값을 검증
    입력: PasswordResetRequest
    출력: (아이디, 이메일, 새 비밀번호)
    """
    username = _validate_username(req.username)
    email = _validate_email(req.email)
    new_password = _validate_password(req.new_password, "새 비밀번호")
    return username, email, new_password


def _extract_bearer_token(authorization: str | None) -> str:
    """
    역할: Authorization 헤더에서 Bearer 토큰 문자열 추출
    입력: Authorization 헤더 값
    출력: access token 문자열
    """
    if not authorization:
        raise HTTPException(status_code=401, detail="로그인이 필요합니다.")

    scheme, _, token = authorization.strip().partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(status_code=401, detail="인증 토큰 형식이 올바르지 않습니다.")
    return token


def get_authenticated_user(
    authorization: str | None = Header(default=None),
    db: Session = Depends(get_db),
) -> User:
    """
    역할: Bearer 토큰을 검증하고 요청 사용자를 DB에서 조회
    입력: Authorization 헤더, DB 세션
    출력: 인증된 User ORM 객체
    """
    token = _extract_bearer_token(authorization)
    payload = verify_access_token(token)
    if payload is None:
        raise HTTPException(status_code=401, detail="인증 토큰이 만료되었거나 올바르지 않습니다.")

    try:
        user_id = int(payload["sub"])
    except (TypeError, ValueError):
        raise HTTPException(status_code=401, detail="인증 토큰의 사용자 정보가 올바르지 않습니다.")

    user = db.query(User).filter(User.id == user_id).first()
    if user is None or user.username != payload.get("username"):
        raise HTTPException(status_code=401, detail="인증 사용자를 찾을 수 없습니다.")
    return user


def get_locked_authenticated_user(
    auth_user: User = Depends(get_authenticated_user),
):
    """
    역할: 인증 사용자의 메모리 상태를 요청 전체 동안 사용자별 잠금으로 보호
    입력: 인증된 User ORM 객체
    출력: 사용자 잠금을 보유한 User ORM 객체
    """
    lock = _user_state_locks.get(auth_user.id)
    lock.acquire()
    try:
        yield auth_user
    finally:
        lock.release()


def _get_client_ip(request: Request) -> str:
    """
    역할: rate limit에 사용할 클라이언트 IP 식별
    입력: FastAPI Request
    출력: 정규화된 IP 문자열
    """
    if TRUST_PROXY_HEADERS:
        cloudflare_ip = request.headers.get("cf-connecting-ip", "").strip()
        if cloudflare_ip:
            return cloudflare_ip
        forwarded = request.headers.get("x-forwarded-for", "").split(",", 1)[0].strip()
        if forwarded:
            return forwarded
    return request.client.host if request.client else "unknown"


def _enforce_rate_limit(scope: str, key: str, rule: RateLimitRule) -> None:
    """
    역할: 요청 횟수 제한을 적용하고 초과 시 Retry-After와 함께 429 반환
    입력: 제한 scope, 요청 식별 키, 제한 규칙
    출력: 없음
    """
    retry_after = _request_rate_limiter.consume(f"{scope}:{key}", rule)
    if retry_after:
        raise HTTPException(
            status_code=429,
            detail="요청이 너무 많습니다. 잠시 후 다시 시도해 주세요.",
            headers={"Retry-After": str(retry_after)},
        )


def get_rate_limited_chat_user(
    auth_user: User = Depends(get_authenticated_user),
):
    """
    역할: 채팅 rate limit을 먼저 검사한 뒤 사용자 상태 잠금을 요청 전체 동안 보유
    입력: 인증된 User ORM 객체
    출력: 채팅 제한과 사용자 잠금이 적용된 User ORM 객체
    """
    _enforce_rate_limit("chat-minute-user", str(auth_user.id), CHAT_MINUTE_RATE_RULE)
    _enforce_rate_limit("chat-hour-user", str(auth_user.id), CHAT_HOUR_RATE_RULE)
    lock = _user_state_locks.get(auth_user.id)
    lock.acquire()
    try:
        yield auth_user
    finally:
        lock.release()


def _require_same_user(auth_user: User, requested_username: str) -> User:
    """
    역할: 요청 body/path의 username이 토큰 사용자와 같은지 확인
    입력: 인증된 사용자, 요청 username
    출력: 인증된 사용자
    """
    if str(requested_username).strip() != auth_user.username:
        raise HTTPException(status_code=403, detail="다른 사용자의 데이터에는 접근할 수 없습니다.")
    return auth_user


def _require_admin_user(auth_user: User, requested_username: str) -> User:
    """
    역할: 요청 사용자가 본인 토큰의 관리자 계정인지 확인
    입력: 인증된 사용자, 요청 username
    출력: 관리자 User ORM 객체
    """
    user = _require_same_user(auth_user, requested_username)
    if not crud.is_developer_user(user):
        raise HTTPException(
            status_code=403,
            detail="관리자 계정만 사용할 수 있는 기능입니다.",
        )
    return user


def _clear_user_runtime_memory_state(user_id: int) -> None:
    """
    역할: DB 초기화 후 대상 계정의 서버 메모리 날짜/점수/문맥 캐시를 초기화
    입력: 초기화 대상 user_id
    출력: 없음
    """
    _pipelines.pop(user_id, None)
    _active_dates_by_user.pop(user_id, None)
    for key in [key for key in _client_context_starts if key[0] == user_id]:
        del _client_context_starts[key]


@app.post("/auth/register", response_model=AuthResponse)
def register(
    req: RegisterRequest,
    request: Request,
    db: Session = Depends(get_db),
):
    """
    역할: 닉네임/아이디/이메일/비밀번호 기반 사용자 가입
    입력: 닉네임, 아이디, 이메일, 비밀번호
    출력: 사용자 아이디, 닉네임, 이메일과 개발자 권한 여부
    """
    _enforce_rate_limit("register-ip", _get_client_ip(request), REGISTER_RATE_RULE)
    username, nickname, email, password = _validate_register_request(req)
    try:
        user = crud.create_user_with_password(
            db,
            username,
            password,
            nickname=nickname,
            email=email,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _build_auth_response(user)


@app.post("/auth/login", response_model=AuthResponse)
def login(
    req: LoginRequest,
    request: Request,
    db: Session = Depends(get_db),
):
    """
    역할: 아이디/비밀번호 기반 사용자 로그인
    입력: 아이디, 비밀번호
    출력: 사용자 아이디와 개발자 권한 여부
    """
    client_ip = _get_client_ip(request)
    _enforce_rate_limit("login-ip", client_ip, LOGIN_RATE_RULE)
    username, password = _validate_login_request(req)
    normalized_username = username.casefold()

    # 같은 계정의 동시 로그인 실패가 DB 카운터를 덮어쓰지 않도록 직렬화한다.
    with _login_attempt_locks.hold(normalized_username):
        user = crud.get_user_by_username(db, username)
        if user is not None:
            retry_after = crud.get_login_lock_remaining_seconds(user)
            if retry_after:
                raise HTTPException(
                    status_code=429,
                    detail="로그인 실패가 반복되어 계정이 잠겼습니다.",
                    headers={"Retry-After": str(retry_after)},
                )

        if user is None or not verify_password(password, user.password_hash):
            if user is not None:
                retry_after = crud.record_failed_login(
                    db,
                    user,
                    max_attempts=LOGIN_MAX_FAILURES,
                    lock_seconds=LOGIN_LOCK_SECONDS,
                )
                if retry_after:
                    raise HTTPException(
                        status_code=429,
                        detail="로그인 실패가 반복되어 계정이 잠겼습니다.",
                        headers={"Retry-After": str(retry_after)},
                    )
            raise HTTPException(
                status_code=401,
                detail="아이디 또는 비밀번호가 올바르지 않습니다.",
            )

        crud.reset_login_failures(db, user)
        return _build_auth_response(user)


@app.post("/auth/reset-password")
def reset_password(
    req: PasswordResetRequest,
    request: Request,
    db: Session = Depends(get_db),
):
    """
    역할: 비로그인 상태에서 아이디와 이메일 확인 후 비밀번호 재설정
    입력: 아이디, 가입 이메일, 새 비밀번호
    출력: {reset, username}
    """
    client_ip = _get_client_ip(request)
    _enforce_rate_limit("password-reset-ip", client_ip, PASSWORD_RESET_RATE_RULE)
    username, email, new_password = _validate_password_reset_request(req)
    normalized_username = username.casefold()

    # 로그인 실패 잠금과 같은 계정 단위 잠금을 재사용해 재설정 추측 시도를 제한한다.
    with _login_attempt_locks.hold(normalized_username):
        user = crud.get_user_by_username(db, username)
        if user is not None:
            retry_after = crud.get_login_lock_remaining_seconds(user)
            if retry_after:
                raise HTTPException(
                    status_code=429,
                    detail="계정 확인 실패가 반복되어 잠시 재설정할 수 없습니다.",
                    headers={"Retry-After": str(retry_after)},
                )

        if user is None or not crud.user_email_matches(user, email):
            if user is not None:
                retry_after = crud.record_failed_login(
                    db,
                    user,
                    max_attempts=LOGIN_MAX_FAILURES,
                    lock_seconds=LOGIN_LOCK_SECONDS,
                )
                if retry_after:
                    raise HTTPException(
                        status_code=429,
                        detail="계정 확인 실패가 반복되어 잠시 재설정할 수 없습니다.",
                        headers={"Retry-After": str(retry_after)},
                    )
            raise HTTPException(
                status_code=401,
                detail="아이디 또는 이메일이 올바르지 않습니다.",
            )

        crud.update_user_password(db, user, new_password)
        return {"reset": True, "username": user.username}


@app.post("/chat", response_model=ChatResponse)
def chat(
    req: ChatRequest,
    auth_user: User = Depends(get_rate_limited_chat_user),
    db: Session = Depends(get_db),
):
    """
    역할: 발화 처리 메인 엔드포인트
          1. RoBERTa 추론 → NLI 후보 및 하드 인터럽트 판별
          2. Qwen 응답 생성 → [CRISIS] 태그 확인 (소프트 인터럽트)
          3. 점수 DB 저장
    """
    user  = _require_same_user(auth_user, req.username)
    # 날짜가 바뀐 뒤 첫 발화라면 지난 날짜를 먼저 자동 마감해 기록 누락을 막는다.
    _auto_close_stale_days(db, user.id)
    today = _get_active_date(user.id)
    session = crud.get_or_create_session(db, user.id, today)
    pipeline = get_pipeline(db, user.id, today)

    # ── 1. RoBERTa 추론 ───────────────────────────────────────────────────────
    # 추론 동시 적재 정책 — RoBERTa 는 응답 anchor 검사·요약 임베딩용으로 유지한다.
    # (학습 모드 진입 시에만 외부에서 _unload_roberta 호출)
    roberta_out = scheduler.run_roberta(req.text)
    hard_crisis = should_hard_interrupt(
        req.text,
        roberta_out["is_crisis"],
        roberta_out["entailment_prob"],
    )

    # ── 2. 발화 DB 저장 ───────────────────────────────────────────────────────
    utt = crud.save_utterance(db, session.id, {
        "text":             req.text,
        "role":             "user",
        "roberta_score":    roberta_out["roberta_score"],
        "cbt_score":        roberta_out.get("cbt_score"),
        "cbt_top_category": roberta_out.get("cbt_top_category"),
        "depression_score": roberta_out["depression_score"],
        "depression_tendency_score": roberta_out.get("depression_tendency_score"),
        "top_emotion":      roberta_out["top_emotion"],
        "entailment_prob":  roberta_out["entailment_prob"],
        "is_crisis":        hard_crisis,
    })

    crisis_message = None

    # ── 3. NLI 하드 인터럽트 ─────────────────────────────────────────────────
    if hard_crisis:
        crisis_out = handle_crisis(
            db, user.id, req.text, source="nli",
            utterance_id=utt.id,
            entailment_prob=roberta_out["entailment_prob"],
        )
        crisis_message = crisis_out["message"]

        # 위기 시 Qwen 호출 생략 — 안내 메시지만 반환
        _append_score_if_affecting(pipeline, roberta_out, final_is_crisis=True)
        wellness_result = _get_current_wellness(pipeline)
        _save_model_audit(
            db=db,
            user_id=user.id,
            utterance_id=utt.id,
            roberta_out=roberta_out,
            hard_crisis=hard_crisis,
            final_is_crisis=True,
            qwen_out=None,
        )
        return ChatResponse(
            response=crisis_message,
            is_crisis=True,
            crisis_message=crisis_message,
            top_emotion=roberta_out["top_emotion"],
            roberta_score=roberta_out["roberta_score"],
            depression_score=roberta_out["depression_score"],
            depression_tendency_score=roberta_out.get("depression_tendency_score", 0.0),
            wellness_score=wellness_result["wellness_score"],
            label=wellness_result["label"],
            utterance_id=utt.id,
            recommendations=build_chat_recommendations(
                roberta_out, wellness_result, is_crisis=True
            ),
        )

    # ── 4. Qwen 응답 생성 ─────────────────────────────────────────────────────
    context_start_id = _get_client_context_start_id(
        user_id=user.id,
        date_str=today,
        client_session_id=req.client_session_id,
        current_utterance_id=utt.id,
    )
    # N발화마다 LLM 서사 요약 갱신 (이번 발화는 제외, 이전까지 누적분 대상)
    _maybe_refresh_narrative(
        db,
        session.id,
        min_utterance_id=context_start_id,
        current_utterance_id=utt.id,
    )
    rolling_summary = _build_rolling_summary(
        db,
        session.id,
        exclude_utterance_id=utt.id,
        min_utterance_id=context_start_id,
    )
    history = _build_history(
        db,
        session.id,
        rolling_summary=rolling_summary,
        exclude_utterance_id=utt.id,
        min_utterance_id=context_start_id,
    )
    qwen_out = scheduler.run_qwen(req.text, history, utterance_info=roberta_out)

    # Qwen [CRISIS] 소프트 인터럽트: 생성 원문 대신 안전 메시지를 반환/저장한다.
    assistant_response = qwen_out["response"]
    if qwen_out["has_crisis_tag"] and not hard_crisis:
        soft_out = handle_crisis(
            db, user.id, req.text, source="qwen",
            utterance_id=utt.id,
        )
        crisis_message = soft_out["message"]
        assistant_response = crisis_message

    _save_model_audit(
        db=db,
        user_id=user.id,
        utterance_id=utt.id,
        roberta_out=roberta_out,
        hard_crisis=hard_crisis,
        final_is_crisis=qwen_out["has_crisis_tag"],
        qwen_out=qwen_out,
    )

    # 봇 응답 저장
    crud.save_utterance(db, session.id, {
        "text": assistant_response,
        "role": "bot",
    })

    # ── 5. 파이프라인 발화 버퍼 업데이트 ─────────────────────────────────────
    _append_score_if_affecting(
        pipeline,
        roberta_out,
        final_is_crisis=qwen_out["has_crisis_tag"],
    )
    wellness_result = _get_current_wellness(pipeline)

    return ChatResponse(
        response=assistant_response,
        is_crisis=qwen_out["has_crisis_tag"],
        crisis_message=crisis_message,
        top_emotion=roberta_out["top_emotion"],
        roberta_score=roberta_out["roberta_score"],
        depression_score=roberta_out["depression_score"],
        depression_tendency_score=roberta_out.get("depression_tendency_score", 0.0),
        wellness_score=wellness_result["wellness_score"],
        label=wellness_result["label"],
        utterance_id=utt.id,
        recommendations=build_chat_recommendations(
            roberta_out, wellness_result, is_crisis=qwen_out["has_crisis_tag"]
        ),
    )


@app.post("/day/close")
def close_day(
    req: DayCloseRequest,
    auth_user: User = Depends(get_locked_authenticated_user),
    db: Session = Depends(get_db),
):
    """
    역할: 하루 종료 — EWMA 집계 + DailySummary DB 저장
          매일 자정 scheduler에서 자동 호출하거나 수동 호출 가능
          화면에 표시 중인 날짜를 명시하면 자정이 지난 뒤에도 그 날짜를 마감한다.
    입력: 사용자 이름, 선택적 마감 날짜
    출력: 마감 요약 + 현재 활성 날짜
    """
    user  = _require_same_user(auth_user, req.username)

    # 자정을 넘긴 채 화면을 열어둔 경우에도 사용자가 보고 있던 날짜를 정확히 마감한다.
    target_date = (req.date or _get_active_date(user.id)).strip()
    try:
        parsed_target = date.fromisoformat(target_date)
    except ValueError:
        raise HTTPException(status_code=400, detail="날짜 형식은 YYYY-MM-DD여야 합니다.")

    # 일반 사용자가 미래 날짜를 미리 마감하는 것은 허용하지 않는다.
    if user.id not in _active_dates_by_user and parsed_target > _get_app_today():
        raise HTTPException(status_code=400, detail="미래 날짜는 마감할 수 없습니다.")

    # 밀린 과거 날짜가 있으면 먼저 순서대로 마감해 EWMA 히스토리 순서를 보존한다.
    auto_closed = _auto_close_stale_days(db, user.id)
    summary = _close_day_for_date(db, user.id, target_date)
    summary["current_date"] = _get_active_date(user.id)
    summary["auto_closed_dates"] = auto_closed
    # 비관리자에게는 raw 종합 distress/우울 경향 소수점을 가리고 밴드만 노출한다.
    _gate_diagnostic_scores(summary, crud.is_developer_user(user))
    return summary


@app.get("/day/current/{username}")
def get_current_day(
    username: str,
    auth_user: User = Depends(get_locked_authenticated_user),
    db: Session = Depends(get_db),
):
    """
    역할: 현재 채팅에 적용 중인 활성 날짜 조회 (+ 지난 날짜 자동 마감 트리거)
    입력: 사용자 이름
    출력: {current_date, is_developer, auto_closed_dates}
    """
    user = _require_same_user(auth_user, username)
    # 채팅 화면 진입 시점이 날짜 경과를 가장 먼저 감지하는 entry point다.
    auto_closed = _auto_close_stale_days(db, user.id)
    return {
        "current_date": _get_active_date(user.id),
        "is_developer": crud.is_developer_user(user),
        "auto_closed_dates": auto_closed,
    }


@app.get("/day/utterances/{username}")
def get_day_utterances(
    username: str,
    date_str: str | None = Query(default=None, alias="date"),
    auth_user: User = Depends(get_locked_authenticated_user),
    db: Session = Depends(get_db),
):
    """
    역할: 날짜별 대화 기록 조회 — 새로고침 시 오늘 대화 복원 + 캘린더 과거 대화 보기
    입력: 사용자 이름, 조회 날짜(쿼리 파라미터 ?date=YYYY-MM-DD, 생략 시 활성 날짜)
    출력: {date, is_active_date, utterances:[{id, role, text, emotion, is_crisis,
           created_at, feedback}], (활성 날짜면) wellness_score, label}
    """
    user = _require_same_user(auth_user, username)
    _auto_close_stale_days(db, user.id)
    is_developer = crud.is_developer_user(user)

    active_date = _get_active_date(user.id)
    target_date = (date_str or active_date).strip()
    try:
        date.fromisoformat(target_date)
    except ValueError:
        raise HTTPException(status_code=400, detail="날짜 형식은 YYYY-MM-DD여야 합니다.")

    session = crud.get_session_by_user_date(db, user.id, target_date)
    utterances = crud.get_utterances_by_session(db, session.id) if session else []

    # 사용자 발화에 남긴 피드백을 함께 복원해 화면 새로고침 후에도 표시를 유지한다.
    user_utt_ids = [u.id for u in utterances if u.role == "user"]
    feedback_map = crud.get_feedback_map_for_utterances(db, user.id, user_utt_ids)

    items = []
    for utt in utterances:
        feedback = feedback_map.get(utt.id, {})
        items.append({
            "id":          utt.id,
            "role":        utt.role,
            "text":        utt.text,
            "emotion":     utt.top_emotion,
            "is_crisis":   bool(utt.is_crisis),
            # naive UTC 저장값 — 프론트에서 "Z"를 붙여 로컬 시간으로 변환한다.
            "created_at":  utt.created_at.isoformat() if utt.created_at else None,
            "feedback": {
                "response_rating":   feedback.get(FEEDBACK_KIND_RESPONSE),
                "corrected_emotion": feedback.get(FEEDBACK_KIND_EMOTION),
            },
        })

    payload: dict = {
        "date":           target_date,
        "is_active_date": target_date == active_date,
        "utterances":     items,
    }
    if target_date == active_date:
        # 오늘 대화 복원 시 실시간 웰니스 패널도 함께 복원할 수 있게 동봉한다.
        pipeline = get_pipeline(db, user.id, active_date)
        wellness_result = _get_current_wellness(pipeline)
        payload["wellness_score"] = wellness_result["wellness_score"]
        payload["label"] = wellness_result["label"]
    return payload


@app.post("/feedback")
def submit_feedback(
    req: FeedbackRequest,
    auth_user: User = Depends(get_locked_authenticated_user),
    db: Session = Depends(get_db),
):
    """
    역할: 사용자 피드백 저장 — 챗봇 응답 평가(good/bad) 또는 감정 셀프 정정
          수집된 라벨은 Qwen 품질 리뷰(genuine-bad/over-block 분리)와
          감정 분류 재학습용 실사용 gold label 후보로 사용한다.
    입력: 사용자 이름, 사용자 발화 id, 피드백 종류, 피드백 값
    출력: {saved, utterance_id, kind, value, model_emotion}
    """
    user = _require_same_user(auth_user, req.username)

    kind = str(req.kind).strip()
    value = str(req.value).strip()
    if kind == FEEDBACK_KIND_RESPONSE:
        if value not in FEEDBACK_RESPONSE_VALUES:
            raise HTTPException(
                status_code=400,
                detail="응답 평가 값은 good 또는 bad여야 합니다.",
            )
    elif kind == FEEDBACK_KIND_EMOTION:
        if value not in EMOTION_LABELS_KO:
            raise HTTPException(
                status_code=400,
                detail=f"감정 정정 값은 {sorted(EMOTION_LABELS_KO)} 중 하나여야 합니다.",
            )
    else:
        raise HTTPException(
            status_code=400,
            detail="kind는 response_rating 또는 emotion_correction이어야 합니다.",
        )

    # 본인 소유의 사용자 발화에만 피드백을 허용한다 (교환 단위 키 = 사용자 발화 id).
    utterance = crud.get_user_owned_utterance(db, user.id, req.utterance_id)
    if utterance is None or utterance.role != "user":
        raise HTTPException(
            status_code=404,
            detail="피드백 대상 발화를 찾을 수 없습니다.",
        )

    model_emotion = utterance.top_emotion if kind == FEEDBACK_KIND_EMOTION else None
    crud.save_or_update_feedback(
        db,
        user_id=user.id,
        utterance_id=utterance.id,
        feedback_kind=kind,
        feedback_value=value,
        model_emotion=model_emotion,
    )
    return {
        "saved": True,
        "utterance_id": utterance.id,
        "kind": kind,
        "value": value,
        "model_emotion": model_emotion,
    }


@app.post("/calendar/emotion-note")
def save_daily_emotion_note(
    req: DailyEmotionNoteRequest,
    auth_user: User = Depends(get_locked_authenticated_user),
    db: Session = Depends(get_db),
):
    """
    역할: 캘린더 날짜별 사용자 수동 감정 기록 저장 또는 갱신
          모델 점수에는 반영하지 않고, 사용자 체감 기록으로만 보존한다.
    입력: 사용자 이름, 날짜, 7감정 라벨, 강도(1~5), 선택 메모
    출력: {saved, date, manual_emotion_*}
    """
    user = _require_same_user(auth_user, req.username)
    target_date = str(req.date).strip()
    try:
        date.fromisoformat(target_date)
    except ValueError:
        raise HTTPException(status_code=400, detail="날짜 형식은 YYYY-MM-DD여야 합니다.")

    emotion_label = str(req.emotion_label).strip()
    if emotion_label not in EMOTION_LABELS_KO:
        raise HTTPException(
            status_code=400,
            detail=f"감정 값은 {sorted(EMOTION_LABELS_KO)} 중 하나여야 합니다.",
        )

    note = crud.save_or_update_daily_emotion_note(
        db,
        user_id=user.id,
        date_str=target_date,
        emotion_label=emotion_label,
        intensity=int(req.intensity),
        note=req.note,
    )
    return {"saved": True, **_serialize_daily_emotion_note(note)}


@app.delete("/calendar/emotion-note/{username}")
def delete_daily_emotion_note(
    username: str,
    date_str: str = Query(alias="date"),
    auth_user: User = Depends(get_locked_authenticated_user),
    db: Session = Depends(get_db),
):
    """
    역할: 캘린더 날짜별 사용자 수동 감정 기록 삭제
    입력: 사용자 이름(path), 날짜(query ?date=YYYY-MM-DD)
    출력: {deleted, date}
    """
    user = _require_same_user(auth_user, username)
    target_date = str(date_str).strip()
    try:
        date.fromisoformat(target_date)
    except ValueError:
        raise HTTPException(status_code=400, detail="날짜 형식은 YYYY-MM-DD여야 합니다.")

    deleted = crud.delete_daily_emotion_note(db, user.id, target_date)
    return {"deleted": deleted, "date": target_date}


@app.post("/day/advance")
def advance_day(
    req: DayAdvanceRequest,
    auth_user: User = Depends(get_locked_authenticated_user),
    db: Session = Depends(get_db),
):
    """
    역할: 현재 활성 날짜를 마감 저장한 뒤 다음 날짜로 전환
    입력: 사용자 이름
    출력: 이전 날짜, 새 날짜, 이전 날짜 요약
    """
    user = _require_admin_user(auth_user, req.username)

    previous_date = _get_active_date(user.id)
    closed_summary = _close_day_for_date(db, user.id, previous_date)
    current_date = _advance_active_date(user.id)

    # 다음 발화 전에도 새 날짜 히스토리가 즉시 복원되도록 파이프라인을 동기화한다.
    get_pipeline(db, user.id, current_date)

    return {
        "previous_date": previous_date,
        "current_date": current_date,
        "closed_summary": closed_summary,
    }


@app.post("/admin/reset-db")
def reset_db(
    req: DbResetRequest,
    auth_user: User = Depends(get_locked_authenticated_user),
    db: Session = Depends(get_db),
):
    """
    역할: 관리자용 해당 계정 DB 런타임 데이터 초기화
    입력: 사용자 이름, 확인 문구 RESET
    출력: 초기화 여부, 삭제 건수, 초기화 후 현재 날짜
    """
    user = _require_admin_user(auth_user, req.username)
    if req.confirm != "RESET":
        raise HTTPException(
            status_code=400,
            detail="DB 초기화를 실행하려면 confirm 값이 RESET이어야 합니다.",
        )

    deleted = crud.reset_runtime_data(db, user.id)
    _clear_user_runtime_memory_state(user.id)

    return {
        "reset": True,
        "username": user.username,
        "preserved_users": True,
        "deleted": deleted,
        "current_date": _get_active_date(user.id),
    }


@app.get("/calendar/{username}")
def get_calendar(
    username: str,
    limit: int = 60,
    auth_user: User = Depends(get_locked_authenticated_user),
    db: Session = Depends(get_db),
):
    """역할: 캘린더 화면용 날짜별 단일/누적 웰니스 상태 + 수동 감정 기록 반환"""
    user = _require_same_user(auth_user, username)
    # 캘린더를 열 때 밀린 날짜를 자동 마감해 빈 칸 없이 표시되게 한다.
    _auto_close_stale_days(db, user.id)
    data = get_calendar_data(db, user.id, limit=limit)
    # 비관리자에게는 날짜별 우울 경향 소수점을 가리고 밴드만 노출한다.
    is_developer = crud.is_developer_user(user)
    for row in data:
        _gate_diagnostic_scores(row, is_developer)
    return data


@app.get("/report/weekly/{username}")
def get_weekly_report(
    username: str,
    end_date: str | None = Query(default=None, alias="end_date"),
    auth_user: User = Depends(get_locked_authenticated_user),
    db: Session = Depends(get_db),
):
    """
    역할: 주간 감정 리포트 — 7일 구간의 단일/누적 웰니스 추이, 감정 분포,
          기록/위기 통계 집계
    입력: 사용자 이름, 주 마지막 날짜(쿼리 ?end_date=YYYY-MM-DD, 생략 시 활성 날짜)
    출력: {active_date, start_date, end_date, days:[7개], summary:{평균 웰니스, 전주 대비,
           기록한 날, 총 발화, 위기 수, 레이블 분포, 감정 분포, 최다 감정}}
    """
    user = _require_same_user(auth_user, username)
    _auto_close_stale_days(db, user.id)
    is_developer = crud.is_developer_user(user)

    active_date = _get_active_date(user.id)
    end_text = (end_date or active_date).strip()
    try:
        end_day = date.fromisoformat(end_text)
    except ValueError:
        raise HTTPException(status_code=400, detail="날짜 형식은 YYYY-MM-DD여야 합니다.")

    start_day = end_day - timedelta(days=6)
    start_text = start_day.isoformat()

    # 이번 주 집계 소스: 일별 요약(마감일) + 발화/위기 직접 집계(미마감일 포함)
    summaries = crud.get_daily_summaries_between(db, user.id, start_text, end_text)
    summary_map = {row.date: row for row in summaries}
    utterance_counts = crud.get_user_utterance_counts_by_date(db, user.id, start_text, end_text)
    crisis_counts = crud.get_crisis_counts_by_date(db, user.id, start_text, end_text)
    emotion_counts = crud.get_user_emotion_counts_between(db, user.id, start_text, end_text)

    days: list[dict] = []
    label_counts: dict[str, int] = {}
    wellness_values: list[float] = []
    daily_wellness_values: list[float] = []
    for offset in range(7):
        day_text = (start_day + timedelta(days=offset)).isoformat()
        row = summary_map.get(day_text)
        daily_wellness = (
            depression_to_display_wellness(row.daily_score) if row else None
        )
        cumulative_wellness = float(row.wellness_score) if row else None
        daily_label = depression_to_display_label(row.daily_score) if row else None
        cumulative_label = row.label if row else None
        if row is not None:
            wellness_values.append(cumulative_wellness)
            if daily_wellness is not None:
                daily_wellness_values.append(daily_wellness)
            label_counts[row.label] = label_counts.get(row.label, 0) + 1
        day_payload = {
            "date":            day_text,
            # 마감된 날만 점수/레이블이 있고, 미마감(보통 오늘)은 null로 둔다.
            "daily_wellness_score": daily_wellness,
            "daily_wellness_label": daily_label,
            "cumulative_wellness_score": cumulative_wellness,
            "cumulative_wellness_label": cumulative_label,
            # 기존 호환용 alias: 누적/평활 웰니스 점수다.
            "wellness_score":  cumulative_wellness,
            "label":           row.label if row else None,
            "depression_tendency_smoothed": (
                row.depression_tendency_smoothed if row else None
            ),
            "utterance_count": utterance_counts.get(day_text, 0),
            "crisis_count":    crisis_counts.get(day_text, 0),
            "is_closed":       row is not None,
        }
        _gate_diagnostic_scores(day_payload, is_developer)
        days.append(day_payload)

    # 전주 평균과 비교해 변화 방향을 보여준다.
    prev_end = start_day - timedelta(days=1)
    prev_start = prev_end - timedelta(days=6)
    prev_summaries = crud.get_daily_summaries_between(
        db, user.id, prev_start.isoformat(), prev_end.isoformat(),
    )
    prev_values = [float(row.wellness_score) for row in prev_summaries]
    prev_daily_values = [
        value
        for row in prev_summaries
        if (value := depression_to_display_wellness(row.daily_score)) is not None
    ]

    avg_wellness = round(sum(wellness_values) / len(wellness_values), 1) if wellness_values else None
    avg_daily_wellness = (
        round(sum(daily_wellness_values) / len(daily_wellness_values), 1)
        if daily_wellness_values
        else None
    )
    prev_avg_wellness = round(sum(prev_values) / len(prev_values), 1) if prev_values else None
    prev_avg_daily_wellness = (
        round(sum(prev_daily_values) / len(prev_daily_values), 1)
        if prev_daily_values
        else None
    )
    wellness_delta = (
        round(avg_wellness - prev_avg_wellness, 1)
        if avg_wellness is not None and prev_avg_wellness is not None
        else None
    )
    daily_wellness_delta = (
        round(avg_daily_wellness - prev_avg_daily_wellness, 1)
        if avg_daily_wellness is not None and prev_avg_daily_wellness is not None
        else None
    )
    top_emotion = next(iter(emotion_counts), None)
    summary_payload = {
        "avg_daily_wellness": avg_daily_wellness,
        "avg_cumulative_wellness": avg_wellness,
        "avg_wellness":      avg_wellness,
        "prev_avg_daily_wellness": prev_avg_daily_wellness,
        "prev_avg_cumulative_wellness": prev_avg_wellness,
        "prev_avg_wellness": prev_avg_wellness,
        "daily_wellness_delta": daily_wellness_delta,
        "cumulative_wellness_delta": wellness_delta,
        "wellness_delta":    wellness_delta,
        "active_days":       sum(1 for d in days if d["utterance_count"] > 0),
        "closed_days":       len(wellness_values),
        "total_utterances":  sum(d["utterance_count"] for d in days),
        "crisis_count":      sum(d["crisis_count"] for d in days),
        "label_counts":      label_counts,
        "emotion_counts":    emotion_counts,
        "top_emotion":       top_emotion,
    }

    pattern_start_day = end_day - timedelta(days=WEEKDAY_PATTERN_WINDOW_DAYS - 1)
    pattern_start_text = pattern_start_day.isoformat()
    model_counts_by_date = crud.get_user_emotion_counts_by_date_between(
        db, user.id, pattern_start_text, end_text,
    )
    manual_counts_by_date = crud.get_daily_emotion_note_counts_by_date_between(
        db, user.id, pattern_start_text, end_text,
    )
    weekday_emotion_patterns = {
        "window_start": pattern_start_text,
        "window_end": end_text,
        "weeks": WEEKDAY_PATTERN_WEEKS,
        "model": _build_weekday_emotion_source(
            model_counts_by_date,
            "model",
            "대화 기반",
        ),
        "manual": _build_weekday_emotion_source(
            manual_counts_by_date,
            "manual",
            "직접 기록",
        ),
    }

    return {
        "active_date": active_date,
        "start_date": start_text,
        "end_date":   end_text,
        "days":       days,
        "summary": summary_payload,
        "weekly_summary": {
            "title": "이번 주 정리",
            "items": _build_weekly_summary_items(summary_payload, is_developer),
        },
        "weekday_emotion_patterns": weekday_emotion_patterns,
    }


@app.post("/auth/change-password")
def change_password(
    req: ChangePasswordRequest,
    auth_user: User = Depends(get_locked_authenticated_user),
    db: Session = Depends(get_db),
):
    """
    역할: 로그인한 사용자의 비밀번호 변경 (현재 비밀번호 재확인 필수)
    입력: 사용자 이름, 현재 비밀번호, 새 비밀번호
    출력: {changed, username}
    """
    user = _require_same_user(auth_user, req.username)

    if not verify_password(req.current_password, user.password_hash):
        raise HTTPException(status_code=401, detail="현재 비밀번호가 올바르지 않습니다.")
    if not is_password_usable(req.new_password):
        raise HTTPException(
            status_code=400,
            detail=f"새 비밀번호는 {get_min_password_length()}자 이상이어야 합니다.",
        )
    if req.new_password == req.current_password:
        raise HTTPException(status_code=400, detail="새 비밀번호가 현재 비밀번호와 같습니다.")

    crud.update_user_password(db, user, req.new_password)
    return {"changed": True, "username": user.username}


@app.get("/export/{username}")
def export_user_data(
    username: str,
    auth_user: User = Depends(get_locked_authenticated_user),
    db: Session = Depends(get_db),
):
    """
    역할: 개인 데이터 내보내기 — 대화/점수/일별 요약/수동 감정/위기 이벤트/피드백을
          JSON 파일로 다운로드 (개인정보 이동권/열람 대응)
    입력: 사용자 이름
    출력: JSON 첨부 응답 (Content-Disposition attachment)
    """
    user = _require_same_user(auth_user, username)
    # 내보내기 직전에 밀린 날짜를 마감해 요약까지 포함되게 한다.
    _auto_close_stale_days(db, user.id)

    sessions = crud.get_sessions_by_user(db, user.id)
    summary_map = {
        row.date: row for row in crud.get_all_daily_summaries_by_user(db, user.id)
    }
    manual_note_rows = crud.get_all_daily_emotion_notes_by_user(db, user.id)
    manual_note_map = {row.date: row for row in manual_note_rows}

    days = []
    for session in sessions:
        utterances = crud.get_utterances_by_session(db, session.id)
        summary_row = summary_map.get(session.date)
        manual_note = manual_note_map.get(session.date)
        days.append({
            "date": session.date,
            "utterances": [
                {
                    "role":             utt.role,
                    "text":             utt.text,
                    "top_emotion":      utt.top_emotion,
                    "depression_score": utt.depression_score,
                    "depression_tendency_score": utt.depression_tendency_score,
                    "is_crisis":        bool(utt.is_crisis),
                    "created_at":       utt.created_at.isoformat() if utt.created_at else None,
                }
                for utt in utterances
            ],
            "daily_summary": (
                {
                    "daily_score":    summary_row.daily_score,
                    "smoothed_score": summary_row.smoothed_score,
                    "daily_wellness_score": depression_to_display_wellness(summary_row.daily_score),
                    "daily_wellness_label": depression_to_display_label(summary_row.daily_score),
                    "cumulative_wellness_score": summary_row.wellness_score,
                    "cumulative_wellness_label": summary_row.label,
                    "wellness_score": summary_row.wellness_score,
                    "label":          summary_row.label,
                    "depression_tendency_daily":    summary_row.depression_tendency_daily,
                    "depression_tendency_smoothed": summary_row.depression_tendency_smoothed,
                    "utterance_count":  summary_row.utterance_count,
                    "crisis_count_day": summary_row.crisis_count_day,
                }
                if summary_row
                else None
            ),
            "manual_emotion_note": (
                {
                    "emotion_label": manual_note.emotion_label,
                    "intensity": manual_note.intensity,
                    "note": manual_note.note,
                    "created_at": (
                        manual_note.created_at.isoformat()
                        if manual_note.created_at
                        else None
                    ),
                    "updated_at": (
                        manual_note.updated_at.isoformat()
                        if manual_note.updated_at
                        else None
                    ),
                }
                if manual_note
                else None
            ),
        })

    payload = {
        "export_version": 2,
        "generated_at":   datetime.utcnow().isoformat() + "Z",
        "username":       user.username,
        "nickname":       user.nickname or user.username,
        "email":          user.email,
        "account_created_at": user.created_at.isoformat() if user.created_at else None,
        "notice": (
            "이 파일은 본인 요청으로 내보낸 개인 데이터입니다. "
            "대화 원문이 포함되어 있으므로 공유에 주의하세요. "
            "점수는 의료 진단이 아닌 정서 모니터링 참고값입니다."
        ),
        "days": days,
        "crisis_events": [
            {
                "text":            event.text,
                "source":          event.source,
                "entailment_prob": event.entailment_prob,
                "created_at":      event.created_at.isoformat() if event.created_at else None,
            }
            for event in crud.get_crisis_events_by_user(db, user.id)
        ],
        "feedback": [
            {
                "utterance_id":   row.utterance_id,
                "feedback_kind":  row.feedback_kind,
                "feedback_value": row.feedback_value,
                "model_emotion":  row.model_emotion,
                "created_at":     row.created_at.isoformat() if row.created_at else None,
            }
            for row in crud.get_feedback_rows_by_user(db, user.id)
        ],
        "daily_emotion_notes": [
            {
                "date":          row.date,
                "emotion_label": row.emotion_label,
                "intensity":     row.intensity,
                "note":          row.note,
                "created_at":    row.created_at.isoformat() if row.created_at else None,
                "updated_at":    row.updated_at.isoformat() if row.updated_at else None,
            }
            for row in manual_note_rows
        ],
    }

    filename = f"emotion_chatbot_export_{user.username}_{_get_app_today().isoformat()}.json"
    return JSONResponse(
        content=payload,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/account/delete")
def delete_account(
    req: AccountDeleteRequest,
    auth_user: User = Depends(get_locked_authenticated_user),
    db: Session = Depends(get_db),
):
    """
    역할: 계정 삭제 — 본인 확인(비밀번호 + 확인 문구) 후 대화/요약/수동감정/위기/감사/피드백
          기록과 계정 row를 완전히 삭제 (개인정보 삭제권 대응)
    입력: 사용자 이름, 비밀번호, 확인 문구 "DELETE"
    출력: {deleted, username, removed}
    """
    user = _require_same_user(auth_user, req.username)

    # developer/root는 서버가 자동 재생성하는 시스템 계정이라 삭제 대상에서 제외한다.
    if crud.is_developer_user(user):
        raise HTTPException(
            status_code=403,
            detail="관리자 기본 계정은 삭제할 수 없습니다. DB 초기화 기능을 사용하세요.",
        )
    if req.confirm != "DELETE":
        raise HTTPException(
            status_code=400,
            detail="계정 삭제를 실행하려면 confirm 값이 DELETE여야 합니다.",
        )
    if not verify_password(req.password, user.password_hash):
        raise HTTPException(status_code=401, detail="비밀번호가 올바르지 않습니다.")

    # 삭제 커밋 후에는 ORM 객체 속성 접근이 불가하므로 식별자를 먼저 스냅샷한다.
    username_snapshot = user.username
    user_id_snapshot = user.id
    removed = crud.delete_user_account(db, user_id_snapshot)
    _clear_user_runtime_memory_state(user_id_snapshot)

    return {
        "deleted": True,
        "username": username_snapshot,
        "removed": removed,
    }


@app.get("/health")
def health():
    """역할: 서버 상태 확인"""
    return {"status": "ok"}


# ── 내부 헬퍼 ────────────────────────────────────────────────────────────────
def _get_current_wellness(pipeline: ScorePipeline) -> dict:
    """
    역할: 현재 발화 버퍼만 기준으로 실시간 오늘 wellness_score 추정
    입력: 사용자별 ScorePipeline 인스턴스
    출력: {wellness_score, label}
    """
    from pipeline.ewma          import utterance_to_daily
    from pipeline.wellness_score import compute_wellness

    if not pipeline._today_utterances:
        return {"wellness_score": 65.0, "label": "보통"}

    # 채팅 화면의 "오늘의 웰니스"는 과거 일별 EWMA가 아니라 오늘 발화만 반영한다.
    # 장기 추세 평활화는 /day/close와 캘린더 기록에서 유지한다.
    daily_score = utterance_to_daily(pipeline._today_utterances)
    return compute_wellness(daily_score, [], 1)


def _append_score_if_affecting(
    pipeline: ScorePipeline,
    roberta_out: dict,
    final_is_crisis: bool = False,
) -> bool:
    """
    역할: 정서 모니터링에 영향을 주는 발화를 정책별 기여 점수로 당일 버퍼에 추가
    입력: 사용자별 ScorePipeline, RoBERTa 결과, 최종 위기 여부
    출력: 점수 반영 여부
    """
    contribution = compute_wellness_contribution(
        roberta_out,
        is_crisis=bool(final_is_crisis),
    )
    affects_score = bool(contribution["score_affects_wellness"])
    if affects_score:
        pipeline._today_utterances.append(
            float(contribution["wellness_contribution_score"])
        )
        # 우울 경향 전용 버퍼도 동일 정책으로 함께 적재 (없으면 0.0 처리)
        tendency_score = roberta_out.get("depression_tendency_score")
        pipeline._today_tendency.append(
            float(tendency_score) if tendency_score is not None else 0.0
        )
    return affects_score


def _save_model_audit(
    db: Session,
    user_id: int,
    utterance_id: int,
    roberta_out: dict,
    hard_crisis: bool,
    final_is_crisis: bool,
    qwen_out: dict | None,
) -> None:
    """
    역할: 운영 모니터링용 모델 판단 근거를 감사 테이블에 저장
    입력: DB 세션, 사용자 id, 발화 id, RoBERTa 결과, 위기 여부, Qwen 결과
    출력: 없음
    """
    anchor_screen = (qwen_out or {}).get("anchor_screen") or {}
    self_check = anchor_screen.get("self_check") or {}
    entail_prob = roberta_out.get("entailment_prob")
    contribution = compute_wellness_contribution(
        roberta_out,
        is_crisis=bool(final_is_crisis or hard_crisis),
    )
    affects_score = bool(contribution["score_affects_wellness"])
    audit_payload = {
        "entailment_prob": entail_prob,
        "raw_entailment_prob": roberta_out.get("raw_entailment_prob"),
        "nli_guard": roberta_out.get("nli_guard"),
        "intensifier_delta_guard": roberta_out.get("intensifier_delta_guard"),
        "intensifier_attenuated_text": roberta_out.get("intensifier_attenuated_text"),
        "intensifier_attenuated_score": roberta_out.get("intensifier_attenuated_score"),
        "intensifier_allowed_delta": roberta_out.get("intensifier_allowed_delta"),
        "intensifier_original_score": roberta_out.get("intensifier_original_score"),
        "intensifier_cbt_delta_guard": roberta_out.get("intensifier_cbt_delta_guard"),
        "intensifier_cbt_attenuated_text": roberta_out.get("intensifier_cbt_attenuated_text"),
        "intensifier_cbt_attenuated_score": roberta_out.get("intensifier_cbt_attenuated_score"),
        "intensifier_cbt_allowed_delta": roberta_out.get("intensifier_cbt_allowed_delta"),
        "intensifier_cbt_original_score": roberta_out.get("intensifier_cbt_original_score"),
        "cbt_score": roberta_out.get("cbt_score"),
        "cbt_top_category": roberta_out.get("cbt_top_category"),
        "cbt_effect": roberta_out.get("cbt_effect"),
        "cbt_reliability_policy": roberta_out.get("cbt_reliability_policy"),
        "cbt_reliability_applied": roberta_out.get("cbt_reliability_applied"),
        "cbt_reliability_cap": roberta_out.get("cbt_reliability_cap"),
        "cbt_reliability_risk_points": roberta_out.get("cbt_reliability_risk_points"),
        "cbt_reliability_benign_points": roberta_out.get("cbt_reliability_benign_points"),
        "cbt_reliability_reasons": roberta_out.get("cbt_reliability_reasons"),
        "top_emotion": roberta_out.get("top_emotion"),
        "depression_score": roberta_out.get("depression_score"),
        "wellness_contribution_score": contribution["wellness_contribution_score"],
        "wellness_impact_type": contribution["wellness_impact_type"],
        "score_affects_wellness": affects_score,
        "score_policy": contribution["score_policy"],
        "qwen_dedupe_replaced": anchor_screen.get("dedupe_replaced"),
        "qwen_dedupe_avoid_count": anchor_screen.get("dedupe_avoid_count"),
    }
    if STORE_QWEN_RAW_RESPONSE:
        # raw Qwen 응답은 품질 리뷰에는 유용하지만 상담 원문성이 있어 production 기본값에서는 저장하지 않는다.
        audit_payload["qwen_raw_response"] = anchor_screen.get("raw_response")

    crud.save_model_audit_event(db, user_id, {
        "utterance_id": utterance_id,
        "hard_crisis": hard_crisis,
        "final_is_crisis": final_is_crisis,
        "nli_candidate": bool(roberta_out.get("is_crisis")),
        "qwen_called": qwen_out is not None,
        "qwen_crisis_tag": (qwen_out or {}).get("has_crisis_tag"),
        "qwen_anchor_replaced": anchor_screen.get("replaced"),
        "qwen_anchor_hits": anchor_screen.get("hits"),
        "qwen_anchor_similarities": anchor_screen.get("similarities"),
        "qwen_self_check_verdict": self_check.get("verdict"),
        "qwen_self_check_category": self_check.get("category"),
        "cbt_top_category_source": roberta_out.get("cbt_top_category_source"),
        "cbt_class_confidence": roberta_out.get("cbt_class_confidence"),
        "cbt_head_non_distortion": roberta_out.get("cbt_head_non_distortion"),
        "utterance_type": roberta_out.get("utterance_type"),
        "utterance_type_confidence": roberta_out.get("utterance_type_confidence"),
        "audit_payload": audit_payload,
    })


def _sync_pipeline_history(pipeline: ScorePipeline, db: Session, user_id: int, date_str: str):
    """
    역할: DB에 저장된 일별 히스토리로 메모리 파이프라인 상태를 복원
    입력: 파이프라인 인스턴스, DB 세션, user_id, 날짜 문자열
    출력: 없음
    """
    # 같은 날 재마감 시 오늘 요약이 과거 히스토리에 중복 집계되지 않도록
    # 기준 날짜 이전의 일별 히스토리만 메모리 상태로 복원한다.
    pipeline._daily_scores = crud.get_daily_score_history_before_date(db, user_id, date_str)
    pipeline._daily_wellness = crud.get_wellness_history_before_date(db, user_id, date_str)
    # 당일 사용자 발화 점수도 복원해 재시작 후 /day/close 값이 왜곡되지 않게 한다.
    pipeline._today_utterances = crud.get_today_user_depression_scores(db, user_id, date_str)
    # 우울 경향 전용 축도 동일하게 복원
    pipeline._daily_tendency = crud.get_daily_tendency_history_before_date(db, user_id, date_str)
    pipeline._today_tendency = crud.get_today_user_depression_tendency_scores(db, user_id, date_str)


def _get_client_context_start_id(
    user_id: int,
    date_str: str,
    client_session_id: str | None,
    current_utterance_id: int,
) -> int | None:
    """
    역할: 화면 채팅창 단위 Qwen 문맥 시작 발화 id를 반환한다.
    입력: user_id, 날짜 문자열, 프론트 채팅창 id, 현재 사용자 발화 id
    출력: 해당 화면 대화에서 포함할 최소 발화 id 또는 None
    """
    if not client_session_id:
        return None

    key = (user_id, date_str, client_session_id)
    if key not in _client_context_starts:
        _client_context_starts[key] = current_utterance_id
    return _client_context_starts[key]


def _truncate_for_summary(text: str, max_chars: int = 70) -> str:
    """
    역할: rolling summary에 넣을 발화 텍스트를 짧게 축약
    입력: 원문 텍스트, 최대 글자 수
    출력: 축약된 텍스트
    """
    cleaned = " ".join(str(text).split())
    if len(cleaned) <= max_chars:
        return cleaned
    return f"{cleaned[:max_chars].rstrip()}..."


def _infer_summary_themes(user_utts: list) -> list[str]:
    """
    역할: 사용자 발화의 반복 주제를 키워드 규칙 + CBT 앵커 카테고리 집계로 추정
    입력: 사용자 발화 ORM 리스트
    출력: 감지된 주제명 리스트(중복 제거, 키워드 테마 우선 배치)
    """
    joined_text = " ".join(str(utt.text) for utt in user_utts if utt.text).lower()
    themes: list[str] = []
    seen: set[str] = set()

    # 1) 키워드 기반 테마 (수면·피로 / 학업·일 / 관계 / 자기비난 / 불안·걱정 / 상실·슬픔)
    for theme, keywords in SUMMARY_THEME_KEYWORDS.items():
        if any(keyword.lower() in joined_text for keyword in keywords):
            if theme not in seen:
                themes.append(theme)
                seen.add(theme)

    # 2) CBT 앵커 카테고리 집계 (cbt_score >= CBT_THRESHOLD 일 때만 저장된 값)
    #    동일 카테고리가 CBT_THEME_MIN_COUNT 이상 반복될 때만 테마로 채택
    cbt_counts: dict[str, int] = {}
    for utt in user_utts:
        category = getattr(utt, "cbt_top_category", None)
        if not category:
            continue
        cbt_counts[category] = cbt_counts.get(category, 0) + 1

    # 빈도 내림차순 → 같은 빈도면 카테고리 사전 순서로 안정 정렬
    for category, count in sorted(cbt_counts.items(), key=lambda x: (-x[1], x[0])):
        if count < CBT_THEME_MIN_COUNT:
            continue
        theme = CBT_CATEGORY_TO_THEME.get(category)
        if not theme or theme in seen:
            continue
        themes.append(theme)
        seen.add(theme)

    return themes


def _build_response_guidance(
    top_emotion: str | None,
    themes: list[str],
    crisis_count: int,
    avg_score: float | None,
) -> str:
    """
    역할: rolling summary 기반으로 Qwen 응답 시 참고할 유의사항 생성
    입력: 주요 감정, 반복 주제, 위기 감지 수, 평균 종합 distress 점수
    출력: 짧은 응답 가이드 문장
    """
    guidance = ["이전 흐름을 알고 있다는 느낌을 주되, 사용자의 현재 발화를 우선해서 공감한다."]
    if top_emotion in {"슬픔", "공포"} or (avg_score is not None and avg_score >= 0.65):
        guidance.append("조언을 서두르기보다 감정 반영과 안전 확인을 먼저 한다.")
    if "자기비난" in themes:
        guidance.append("자기비난을 그대로 강화하지 말고 부담을 덜어주는 방향으로 말한다.")
    if "수면·피로" in themes:
        guidance.append("피로와 회복 욕구를 인정하고 작은 휴식 단위를 탐색한다.")
    if "파국화" in themes:
        guidance.append("최악 결말로 단정하지 않도록 사실과 추측을 가볍게 분리해준다.")
    if "이분법적 사고" in themes or "과잉일반화" in themes:
        guidance.append("'전부/항상' 같은 극단 표현을 부드럽게 풀어주고 중간 지점을 살린다.")
    if "감정적 추론" in themes:
        guidance.append("감정이 곧 사실은 아니라는 점을 강요 없이 슬쩍 비춘다.")
    # v3 신규 5범주 가이드
    if "부정적 편향" in themes:
        guidance.append("부정적인 부분에만 시선이 쏠려 있을 수 있다는 점을 환기하고 가려진 긍정 단서를 살핀다.")
    if "낙인찍기" in themes:
        guidance.append("자신을 단정적인 부정 라벨로 규정하지 않도록, 행동과 사람을 분리해 표현해준다.")
    if "긍정 축소화" in themes:
        guidance.append("잘한 일을 운/우연으로 깎아내리는 흐름을 가볍게 짚고 노력의 몫을 인정해준다.")
    if "당위 진술" in themes:
        guidance.append("'반드시/꼭' 같은 당위 표현이 부담을 키울 수 있음을 인정하고 작은 선택지를 제시한다.")
    if "성급한 판단" in themes:
        guidance.append("근거 없이 결론으로 비약하지 않도록, 확인 가능한 사실과 추측을 부드럽게 구분해준다.")
    if crisis_count:
        guidance.append("위기 신호 이력이 있으므로 직접적 위험 표현이 나오면 안전 안내를 우선한다.")
    return " ".join(guidance)


def _maybe_refresh_narrative(
    db: Session,
    session_id: int,
    min_utterance_id: int | None,
    current_utterance_id: int,
) -> None:
    """
    역할: 사용자 발화가 NARRATIVE_REFRESH_EVERY 개 새로 누적되면 Qwen 으로 서사 요약 갱신
          최초 호출 시 narrative_until_utterance_id == None 이면 임계 도달 즉시 1회 생성
    입력: DB 세션, session_id, 화면 컨텍스트 시작 utterance id, 현재 발화 id
    출력: 없음 (DB 갱신만 수행)
    """
    cur_summary, until_id = crud.get_session_narrative(db, session_id)

    # 같은 화면 컨텍스트 안의 사용자 발화만 대상으로
    all_utts = crud.get_utterances_by_session(db, session_id)
    user_utts = [
        u for u in all_utts
        if u.role == "user"
        and u.id != current_utterance_id
        and (min_utterance_id is None or u.id >= min_utterance_id)
    ]
    if not user_utts:
        return

    # 마지막 반영 이후로 새로 쌓인 발화 수
    new_since = sum(1 for u in user_utts if (until_id is None or u.id > until_id))
    if new_since < NARRATIVE_REFRESH_EVERY and cur_summary:
        return  # 기존 요약 유지

    # 입력 발화 텍스트 구성 (최근 NARRATIVE_MAX_INPUT_UTTS 만)
    texts = [u.text for u in user_utts[-NARRATIVE_MAX_INPUT_UTTS:] if u.text]
    if not texts:
        return

    try:
        # Qwen 요약도 일반 응답과 같은 단일 추론 큐에서 실행한다.
        new_summary = scheduler.generate_summary(texts)
    except Exception as exc:
        print(f"[narrative] Qwen 요약 실패 → 통계 요약만 사용: {exc}")
        return

    if not new_summary:
        return

    last_id = user_utts[-1].id
    crud.update_session_narrative(db, session_id, new_summary, last_id)
    print(f"[narrative] 갱신 완료 (session={session_id}, until_id={last_id})")


def _build_rolling_summary(
    db: Session,
    session_id: int,
    exclude_utterance_id: int | None = None,
    min_utterance_id: int | None = None,
) -> str | None:
    """
    역할: 최근 history 창 밖의 당일 대화 흐름을 Qwen 입력용 짧은 요약으로 구성
    입력: DB 세션, session_id, 제외할 현재 발화 id, 포함할 최소 발화 id
    출력: 요약 문자열 또는 None
    """
    older_utts = crud.get_older_utterances_for_summary(
        db,
        session_id,
        recent_limit=RECENT_HISTORY_LIMIT + (1 if exclude_utterance_id is not None else 0),
        summary_limit=ROLLING_SUMMARY_OLDER_LIMIT,
        min_utterance_id=min_utterance_id,
    )
    if exclude_utterance_id is not None:
        older_utts = [u for u in older_utts if u.id != exclude_utterance_id]

    user_utts = [u for u in older_utts if u.role == "user"]

    # older 가 비어도 narrative 가 있으면 narrative 만 반환 (history limit 미만에서도 흐름 노출)
    narrative_only_text, _ = crud.get_session_narrative(db, session_id)
    if not user_utts:
        if narrative_only_text:
            return _truncate_for_summary("[서사] " + narrative_only_text, ROLLING_SUMMARY_MAX_CHARS)
        return None

    emotion_counts: dict[str, int] = {}
    scores: list[float] = []
    crisis_count = 0
    for utt in user_utts:
        if utt.top_emotion:
            emotion_counts[utt.top_emotion] = emotion_counts.get(utt.top_emotion, 0) + 1
        if utt.depression_score is not None:
            scores.append(float(utt.depression_score))
        if utt.is_crisis:
            crisis_count += 1

    top_emotion = None
    if emotion_counts:
        top_emotion = max(emotion_counts, key=emotion_counts.get)

    themes = _infer_summary_themes(user_utts)
    avg_score = (sum(scores) / len(scores)) if scores else None
    sample_texts = [
        _truncate_for_summary(utt.text)
        for utt in user_utts[-ROLLING_SUMMARY_SAMPLE_LIMIT:]
        if utt.text
    ]

    # LLM 서사 요약 prepend (있으면) — 통계보다 정성적 흐름이 먼저 보이도록
    narrative_text, _ = crud.get_session_narrative(db, session_id)

    parts = []
    if narrative_text:
        parts.append("[서사] " + narrative_text)
    parts.append(f"[통계] 최근 context 밖의 오늘 사용자 발화 {len(user_utts)}개.")
    if top_emotion:
        parts.append(f"주요 감정: {top_emotion}.")
    if themes:
        parts.append("반복 주제: " + ", ".join(themes[:4]) + ".")
    if avg_score is not None:
        parts.append(f"정서 강도: 평균 종합 distress {avg_score:.2f}.")
    if crisis_count:
        parts.append(f"안전 이력: 위기 신호 {crisis_count}회.")
    if sample_texts:
        parts.append("대표 발화: " + " / ".join(sample_texts) + ".")
    parts.append("[유의] " + _build_response_guidance(top_emotion, themes, crisis_count, avg_score))

    summary = " ".join(parts)
    # 길이 cap 도달 시 통계/대표발화부터 잘리도록 우선순위 정렬은 추후 보강.
    # 현 구현은 단순 truncate — 서사가 앞단에 위치하므로 우선 보존된다.
    return _truncate_for_summary(summary, ROLLING_SUMMARY_MAX_CHARS)


def _build_history(
    db: Session,
    session_id: int,
    rolling_summary: str | None = None,
    exclude_utterance_id: int | None = None,
    min_utterance_id: int | None = None,
) -> list[dict]:
    """
    역할: Qwen 입력용 대화 히스토리 구성 (rolling summary + 최근 10턴)
    입력: DB 세션, session_id, 최근 창 밖의 당일 요약, 제외할 현재 발화 id, 포함할 최소 발화 id
    출력: Qwen chat template용 message 리스트
    """
    query_limit = RECENT_HISTORY_LIMIT + (1 if exclude_utterance_id is not None else 0)
    utts = crud.get_recent_utterances_by_session(
        db,
        session_id,
        limit=query_limit,
        min_utterance_id=min_utterance_id,
    )
    if exclude_utterance_id is not None:
        utts = [u for u in utts if u.id != exclude_utterance_id]
    utts = utts[-RECENT_HISTORY_LIMIT:]

    history = []
    if rolling_summary:
        history.append({
            "role": "assistant",
            "content": f"[오늘 이전 대화 요약]\n{rolling_summary}",
        })
    for u in utts:   # 최근 20개 발화 (10턴)
        role = "user" if u.role == "user" else "assistant"
        history.append({"role": role, "content": u.text})
    return history


_mount_frontend_build(app)
