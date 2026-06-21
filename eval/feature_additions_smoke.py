"""
feature_additions_smoke.py
역할: 2026-06-10 신규 사용자 기능 5종 엔드포인트를 임시 DB에서 스모크 검증
      1. GET  /day/utterances    — 오늘 대화 복원 + 과거 날짜 대화 조회
      2. POST /feedback          — 응답 평가(good/bad) + 감정 셀프 정정 upsert
      3. 자동 하루 마감           — 지난 날짜 lazy close (/day/current 진입 시)
      4. GET  /report/weekly     — 주간 리포트 집계
      5. POST /calendar/emotion-note — 캘린더 날짜별 수동 감정 기록 저장/삭제
      6. POST /auth/change-password, POST /auth/reset-password, GET /export, POST /account/delete
입력: 없음 (모델 추론 /chat은 호출하지 않고, 과거 발화는 crud로 직접 주입)
출력: 단계별 PASS/FAIL 콘솔 출력, 전체 통과 시 종료 코드 0
실행: C:/Users/WD/anaconda3/envs/dl_study/python.exe eval/feature_additions_smoke.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
TEST_DB_PATH = ROOT_DIR / "eval" / "_feature_additions_smoke.db"

# 백엔드 모듈 import 전에 테스트 DB 경로를 지정해야 init_db가 임시 DB를 사용한다.
os.environ["EMOTION_CHATBOT_DB_PATH"] = str(TEST_DB_PATH)

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if TEST_DB_PATH.exists():
    TEST_DB_PATH.unlink()

from fastapi.testclient import TestClient  # noqa: E402

from backend import main as backend_main  # noqa: E402
from backend.db import crud  # noqa: E402
from backend.db.models import DailyEmotionNote, DailySummary, User, UtteranceFeedback  # noqa: E402


PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    """
    역할: 단계별 검증 결과를 기록하고 콘솔에 출력
    입력: 검증 이름, 통과 여부, 실패 시 상세 메시지
    출력: 없음 (PASSED/FAILED 전역 리스트 갱신)
    """
    if condition:
        PASSED.append(name)
        print(f"[PASS] {name}")
    else:
        FAILED.append(name)
        print(f"[FAIL] {name} {detail}")


def register(client: TestClient, username: str, password: str) -> dict[str, str]:
    """
    역할: 이메일 포함 가입 후 Authorization 헤더 생성
    입력: TestClient, 사용자 이름, 비밀번호
    출력: Bearer 토큰 헤더 dict
    """
    response = client.post(
        "/auth/register",
        json={
            "username": username,
            "nickname": username,
            "email": f"{username}@example.local",
            "password": password,
        },
    )
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def login(client: TestClient, username: str, password: str) -> tuple[int, dict[str, str]]:
    """
    역할: 로그인 시도 후 상태 코드와 Authorization 헤더 반환
    입력: TestClient, 사용자 이름, 비밀번호
    출력: (상태 코드, 헤더 dict — 실패 시 빈 dict)
    """
    response = client.post(
        "/auth/login",
        json={"username": username, "password": password},
    )
    if response.status_code != 200:
        return response.status_code, {}
    return 200, {"Authorization": f"Bearer {response.json()['access_token']}"}


def seed_conversation(username: str, date_str: str, texts: list[tuple[str, str]]) -> list[int]:
    """
    역할: 모델 추론 없이 특정 날짜 세션에 발화를 직접 주입 (과거/오늘 대화 시뮬레이션)
    입력: 사용자 이름, 날짜 문자열, [(role, text), ...] 리스트
    출력: 주입된 사용자 발화 id 리스트
    """
    db = backend_main.SessionLocal()
    try:
        user = db.query(User).filter(User.username == username).first()
        session = crud.get_or_create_session(db, user.id, date_str)
        user_ids: list[int] = []
        for role, text in texts:
            data = {"text": text, "role": role}
            if role == "user":
                # 점수 필드는 자동 마감/리포트 계산에 쓰이는 최소 구성만 채운다.
                data.update({
                    "roberta_score": 0.55,
                    "depression_score": 0.55,
                    "depression_tendency_score": 0.1,
                    "top_emotion": "슬픔",
                    "entailment_prob": 0.05,
                    "is_crisis": False,
                })
            utt = crud.save_utterance(db, session.id, data)
            if role == "user":
                user_ids.append(utt.id)
        return user_ids
    finally:
        db.close()


def count_summaries(username: str, date_str: str) -> int:
    """
    역할: 특정 사용자/날짜의 daily_summaries row 수 조회
    입력: 사용자 이름, 날짜 문자열
    출력: row 수
    """
    db = backend_main.SessionLocal()
    try:
        user = db.query(User).filter(User.username == username).first()
        return (
            db.query(DailySummary)
            .filter(DailySummary.user_id == user.id, DailySummary.date == date_str)
            .count()
        )
    finally:
        db.close()


def count_feedback(username: str, utterance_id: int, feedback_kind: str) -> int:
    """
    역할: 특정 사용자 발화/종류 조합의 피드백 row 수 조회
    입력: 사용자 이름, 발화 id, 피드백 종류
    출력: row 수
    """
    db = backend_main.SessionLocal()
    try:
        user = db.query(User).filter(User.username == username).first()
        return (
            db.query(UtteranceFeedback)
            .filter(
                UtteranceFeedback.user_id == user.id,
                UtteranceFeedback.utterance_id == utterance_id,
                UtteranceFeedback.feedback_kind == feedback_kind,
            )
            .count()
        )
    finally:
        db.close()


def count_daily_emotion_notes(username: str, date_str: str | None = None) -> int:
    """
    역할: 특정 사용자/날짜의 수동 감정 기록 row 수 조회
    입력: 사용자 이름, 날짜 문자열(선택)
    출력: row 개수
    """
    db = backend_main.SessionLocal()
    try:
        user = db.query(User).filter(User.username == username).first()
        query = db.query(DailyEmotionNote).filter(DailyEmotionNote.user_id == user.id)
        if date_str is not None:
            query = query.filter(DailyEmotionNote.date == date_str)
        return query.count()
    finally:
        db.close()


def force_single_cumulative_label_mismatch(username: str, date_str: str) -> None:
    """
    역할: 단일 웰니스와 누적 웰니스 상태가 다른 회귀 케이스를 임시 DB에 구성
    입력: 사용자 이름, 날짜 문자열
    출력: 없음
    """
    db = backend_main.SessionLocal()
    try:
        user = db.query(User).filter(User.username == username).first()
        summary = (
            db.query(DailySummary)
            .filter(DailySummary.user_id == user.id, DailySummary.date == date_str)
            .one()
        )
        # 단일 웰니스는 85점(양호), 누적/평활 웰니스는 25점(위험)으로 일부러 분리한다.
        summary.daily_score = 0.15
        summary.smoothed_score = 0.75
        summary.wellness_score = 25.0
        summary.label = "위험"
        db.commit()
    finally:
        db.close()


def main() -> int:
    """
    역할: 신규 기능 스모크 테스트 전체 실행
    입력: 없음
    출력: 전체 통과 시 0, 실패 시 1
    """
    client = TestClient(backend_main.app)
    app_today = backend_main._get_app_today()
    today = app_today.isoformat()
    yesterday = (app_today - timedelta(days=1)).isoformat()
    two_days_ago = (app_today - timedelta(days=2)).isoformat()
    same_weekday_last_week = (app_today - timedelta(days=7)).isoformat()
    same_weekday_two_weeks_ago = (app_today - timedelta(days=14)).isoformat()

    response = client.post("/auth/register", json={
        "username": "bad_email_user",
        "nickname": "잘못된이메일",
        "email": "test@gmail",
        "password": "test1234",
    })
    check("auth-email: 점 없는 도메인 가입 400", response.status_code == 400, response.text)

    response = client.post("/auth/reset-password", json={
        "username": "bad_email_user",
        "email": "test@gmail",
        "new_password": "test1234",
    })
    check("auth-email: 점 없는 도메인 재설정 400", response.status_code == 400, response.text)

    username = "smoke_user"
    password = "test1234"
    headers = register(client, username, password)

    other_headers = register(client, "smoke_other", "test1234")

    # ── 한국 시간 날짜 전환: 서버 UTC 6/10 15:30은 앱 날짜 6/11이어야 한다 ────
    class FrozenUtcDateTime(datetime):
        """
        역할: 서버가 UTC 시각을 사용해도 앱 시간대 변환을 재현하는 고정 datetime
        입력: datetime.now()의 선택적 tz
        출력: 2026-06-10 15:30 UTC 기준 시각
        """

        @classmethod
        def now(cls, tz=None):
            """
            역할: 고정 UTC 시각을 요청 시간대로 변환
            입력: 선택적 tzinfo
            출력: 고정 datetime
            """
            fixed = cls(2026, 6, 10, 15, 30, tzinfo=timezone.utc)
            return fixed if tz is None else fixed.astimezone(tz)

    original_datetime = backend_main.datetime
    try:
        backend_main.datetime = FrozenUtcDateTime
        check(
            "timezone: UTC 6/10 15:30을 한국 날짜 6/11로 계산",
            backend_main._get_app_today().isoformat() == "2026-06-11",
            backend_main._get_app_today().isoformat(),
        )

        rollover_username = "smoke_rollover"
        rollover_headers = register(client, rollover_username, "test1234")
        seed_conversation(rollover_username, "2026-06-10", [
            ("user", "6월 10일 기록을 마감할게"),
            ("bot", "오늘 기록을 정리해 둘게요."),
        ])
        response = client.post(
            "/day/close",
            headers=rollover_headers,
            json={"username": rollover_username, "date": "2026-06-10"},
        )
        body = response.json() if response.status_code == 200 else {}
        check(
            "day-close: 6/10 마감 후 활성 날짜 6/11 전환",
            response.status_code == 200
            and body.get("date") == "2026-06-10"
            and body.get("current_date") == "2026-06-11"
            and count_summaries(rollover_username, "2026-06-10") == 1,
            response.text,
        )
    finally:
        backend_main.datetime = original_datetime

    # ── 배포 응답 스키마: /chat에 피드백 키 utterance_id가 포함돼야 한다 ───────
    schema_headers = register(client, "smoke_schema", "test1234")
    original_run_roberta = backend_main.scheduler.run_roberta
    original_run_qwen = backend_main.scheduler.run_qwen
    try:
        backend_main.scheduler.run_roberta = lambda text: {
            "roberta_score": 0.25,
            "cbt_score": 0.20,
            "depression_score": 0.25,
            "depression_tendency_score": 0.02,
            "top_emotion": "행복",
            "entailment_prob": 0.01,
            "is_crisis": False,
            "utterance_type": "positive_share",
        }
        backend_main.scheduler.run_qwen = (
            lambda text, history=None, utterance_info=None: {
                "response": "조금 나아진 마음이 느껴져요.",
                "has_crisis_tag": False,
            }
        )
        response = client.post(
            "/chat",
            headers=schema_headers,
            json={"username": "smoke_schema", "text": "오늘은 조금 괜찮아."},
        )
        body = response.json() if response.status_code == 200 else {}
        check(
            "chat-schema: utterance_id 포함",
            response.status_code == 200 and isinstance(body.get("utterance_id"), int),
            response.text,
        )
    finally:
        backend_main.scheduler.run_roberta = original_run_roberta
        backend_main.scheduler.run_qwen = original_run_qwen

    # ── 사전 데이터: 그제/어제(마감 안 됨) + 오늘 대화 주입 ────────────────────
    seed_conversation(username, two_days_ago, [
        ("user", "그제는 일이 너무 많아서 지쳤어"),
        ("bot", "많이 바쁘셨군요. 오늘은 좀 어떠세요?"),
    ])
    seed_conversation(username, yesterday, [
        ("user", "어제는 마음이 좀 가라앉았어"),
        ("bot", "그런 날도 있죠. 이야기해 주셔서 고마워요."),
    ])
    today_user_ids = seed_conversation(username, today, [
        ("user", "오늘은 조금 나아진 것 같아"),
        ("bot", "다행이에요. 어떤 점이 가장 나아졌나요?"),
    ])
    target_utt_id = today_user_ids[0]

    # ── 3. 자동 하루 마감: /day/current 진입 시 그제/어제 lazy close ──────────
    response = client.get(f"/day/current/{username}", headers=headers)
    check("auto-close: /day/current 200", response.status_code == 200, response.text)
    auto_closed = response.json().get("auto_closed_dates", [])
    check(
        "auto-close: 그제/어제 자동 마감",
        auto_closed == [two_days_ago, yesterday],
        f"auto_closed={auto_closed}",
    )
    check(
        "auto-close: daily_summaries 생성 확인",
        count_summaries(username, two_days_ago) == 1
        and count_summaries(username, yesterday) == 1
        and count_summaries(username, today) == 0,
    )
    force_single_cumulative_label_mismatch(username, yesterday)
    # 재호출 시 중복 마감이 없어야 한다
    response = client.get(f"/day/current/{username}", headers=headers)
    check(
        "auto-close: 재호출 시 중복 마감 없음",
        response.json().get("auto_closed_dates") == [],
        response.text,
    )

    # ── 1. 오늘 대화 복원 + 과거 날짜 조회 ─────────────────────────────────────
    response = client.get(f"/day/utterances/{username}", headers=headers)
    body = response.json() if response.status_code == 200 else {}
    check("day-utterances: 오늘 조회 200", response.status_code == 200, response.text)
    check(
        "day-utterances: 오늘 발화 2건 + 활성 날짜 + 웰니스 동봉",
        body.get("is_active_date") is True
        and len(body.get("utterances", [])) == 2
        and isinstance(body.get("wellness_score"), (int, float))
        and body.get("date") == today,
        json.dumps(body, ensure_ascii=False)[:300],
    )

    response = client.get(
        f"/day/utterances/{username}", params={"date": yesterday}, headers=headers,
    )
    body = response.json() if response.status_code == 200 else {}
    check(
        "day-utterances: 과거 날짜 조회",
        response.status_code == 200
        and body.get("is_active_date") is False
        and len(body.get("utterances", [])) == 2
        and body["utterances"][0]["emotion"] == "슬픔",
        response.text[:300],
    )

    response = client.get(
        f"/day/utterances/{username}", params={"date": "2026-13-99"}, headers=headers,
    )
    check("day-utterances: 잘못된 날짜 400", response.status_code == 400, response.text)

    response = client.get(f"/day/utterances/{username}")
    check("day-utterances: 비로그인 401", response.status_code == 401, response.text)

    response = client.get(f"/day/utterances/{username}", headers=other_headers)
    check("day-utterances: 타인 토큰 403", response.status_code == 403, response.text)

    # ── 2. 피드백: 응답 평가 + 감정 셀프 정정 ──────────────────────────────────
    response = client.post("/feedback", headers=headers, json={
        "username": username, "utterance_id": target_utt_id,
        "kind": "response_rating", "value": "good",
    })
    check("feedback: 응답 평가 good 저장", response.status_code == 200, response.text)

    response = client.post("/feedback", headers=headers, json={
        "username": username, "utterance_id": target_utt_id,
        "kind": "response_rating", "value": "bad",
    })
    check(
        "feedback: 같은 키 upsert(bad)",
        response.status_code == 200
        and count_feedback(username, target_utt_id, "response_rating") == 1,
        response.text,
    )

    response = client.post("/feedback", headers=headers, json={
        "username": username, "utterance_id": target_utt_id,
        "kind": "emotion_correction", "value": "행복",
    })
    body = response.json() if response.status_code == 200 else {}
    check(
        "feedback: 감정 정정 + 모델 감정 스냅샷",
        response.status_code == 200 and body.get("model_emotion") == "슬픔",
        response.text,
    )

    response = client.get(f"/day/utterances/{username}", headers=headers)
    feedback = response.json()["utterances"][0]["feedback"]
    check(
        "feedback: 복원 응답에 피드백 포함",
        feedback.get("response_rating") == "bad"
        and feedback.get("corrected_emotion") == "행복",
        json.dumps(feedback, ensure_ascii=False),
    )

    response = client.post("/feedback", headers=headers, json={
        "username": username, "utterance_id": target_utt_id,
        "kind": "unknown_kind", "value": "good",
    })
    check("feedback: 잘못된 kind 400", response.status_code == 400, response.text)

    response = client.post("/feedback", headers=headers, json={
        "username": username, "utterance_id": target_utt_id,
        "kind": "emotion_correction", "value": "기쁨아님",
    })
    check("feedback: 잘못된 감정 라벨 400", response.status_code == 400, response.text)

    response = client.post("/feedback", headers=other_headers, json={
        "username": "smoke_other", "utterance_id": target_utt_id,
        "kind": "response_rating", "value": "good",
    })
    check("feedback: 타인 발화 404", response.status_code == 404, response.text)

    # ── 4. 캘린더 수동 감정 기록 ───────────────────────────────────────────────
    response = client.post("/calendar/emotion-note", headers=headers, json={
        "username": username,
        "date": today,
        "emotion_label": "행복",
        "intensity": 4,
        "note": "오늘은 조금 가벼웠다",
    })
    body = response.json() if response.status_code == 200 else {}
    check(
        "manual-emotion: 오늘 수동 감정 저장",
        response.status_code == 200
        and body.get("manual_emotion_label") == "행복"
        and body.get("manual_emotion_intensity") == 4
        and count_daily_emotion_notes(username, today) == 1,
        response.text,
    )

    response = client.post("/calendar/emotion-note", headers=headers, json={
        "username": username,
        "date": today,
        "emotion_label": "슬픔",
        "intensity": 2,
        "note": "저녁에는 살짝 가라앉음",
    })
    body = response.json() if response.status_code == 200 else {}
    check(
        "manual-emotion: 같은 날짜 upsert",
        response.status_code == 200
        and body.get("manual_emotion_label") == "슬픔"
        and body.get("manual_emotion_intensity") == 2
        and count_daily_emotion_notes(username, today) == 1,
        response.text,
    )

    response = client.post("/calendar/emotion-note", headers=headers, json={
        "username": username,
        "date": two_days_ago,
        "emotion_label": "중립",
        "intensity": 3,
        "note": "",
    })
    check("manual-emotion: 삭제용 기록 저장", response.status_code == 200, response.text)
    response = client.delete(
        f"/calendar/emotion-note/{username}",
        params={"date": two_days_ago},
        headers=headers,
    )
    check(
        "manual-emotion: 수동 감정 삭제",
        response.status_code == 200
        and response.json().get("deleted") is True
        and count_daily_emotion_notes(username, two_days_ago) == 0,
        response.text,
    )

    response = client.post("/calendar/emotion-note", headers=headers, json={
        "username": username,
        "date": "2026-13-99",
        "emotion_label": "행복",
        "intensity": 3,
    })
    check("manual-emotion: 잘못된 날짜 400", response.status_code == 400, response.text)

    response = client.post("/calendar/emotion-note", headers=headers, json={
        "username": username,
        "date": today,
        "emotion_label": "기쁨아님",
        "intensity": 3,
    })
    check("manual-emotion: 잘못된 감정 400", response.status_code == 400, response.text)

    response = client.post("/calendar/emotion-note", headers=other_headers, json={
        "username": username,
        "date": today,
        "emotion_label": "행복",
        "intensity": 3,
    })
    check("manual-emotion: 타인 토큰 403", response.status_code == 403, response.text)

    # 리포트 요일별 패턴 검증용 — 현재 주 밖 같은 요일에 감정이 반복되도록 샘플을 보강한다.
    seed_conversation(username, same_weekday_last_week, [
        ("user", "지난주 같은 요일에도 마음이 가라앉았어"),
    ])
    seed_conversation(username, same_weekday_two_weeks_ago, [
        ("user", "그 전 같은 요일에도 조금 쓸쓸했어"),
    ])
    for note_date in [same_weekday_last_week, same_weekday_two_weeks_ago]:
        response = client.post("/calendar/emotion-note", headers=headers, json={
            "username": username,
            "date": note_date,
            "emotion_label": "행복",
            "intensity": 4,
            "note": "같은 요일 반복 확인",
        })
        check(
            f"manual-emotion: 요일 패턴용 수동 기록 {note_date}",
            response.status_code == 200,
            response.text,
        )

    # ── 5. 주간 리포트 ─────────────────────────────────────────────────────────
    response = client.get(f"/report/weekly/{username}", headers=headers)
    body = response.json() if response.status_code == 200 else {}
    summary = body.get("summary", {})
    weekly_summary = body.get("weekly_summary", {})
    weekday_patterns = body.get("weekday_emotion_patterns", {})
    days = body.get("days", [])
    day_map = {d["date"]: d for d in days}
    check("report: 주간 리포트 200", response.status_code == 200, response.text)
    check(
        "report: 7일 구성 + 마감/미마감 구분",
        len(days) == 7
        and day_map.get(yesterday, {}).get("wellness_score") is not None
        and day_map.get(yesterday, {}).get("daily_wellness_score") is not None
        and day_map.get(yesterday, {}).get("cumulative_wellness_score") is not None
        and day_map.get(today, {}).get("wellness_score") is None
        and day_map.get(today, {}).get("utterance_count") == 1,
        json.dumps(day_map.get(today, {}), ensure_ascii=False),
    )
    check(
        "report: 집계값 (기록 3일, 발화 3개, 최다 감정)",
        summary.get("active_days") == 3
        and summary.get("total_utterances") == 3
        and summary.get("top_emotion") in {"슬픔", "행복"}
        and summary.get("crisis_count") == 0,
        json.dumps(summary, ensure_ascii=False),
    )
    check(
        "report: 단일/누적 평균 웰니스 분리",
        summary.get("avg_daily_wellness") is not None
        and summary.get("avg_cumulative_wellness") is not None,
        json.dumps(summary, ensure_ascii=False),
    )
    check(
        "report: 주간 정리 문장 포함",
        weekly_summary.get("title") == "이번 주 정리"
        and len(weekly_summary.get("items", [])) >= 3,
        json.dumps(weekly_summary, ensure_ascii=False),
    )
    model_patterns = weekday_patterns.get("model", {})
    manual_patterns = weekday_patterns.get("manual", {})
    check(
        "report: 최근 8주 요일별 감정분포 출처 분리",
        weekday_patterns.get("weeks") == 8
        and len(model_patterns.get("weekdays", [])) == 7
        and len(manual_patterns.get("weekdays", [])) == 7
        and model_patterns.get("total", 0) >= 5
        and manual_patterns.get("total", 0) >= 3,
        json.dumps(weekday_patterns, ensure_ascii=False)[:600],
    )
    check(
        "report: 요일별 감정 몰림 해석",
        any(item.get("emotion") == "슬픔" for item in model_patterns.get("patterns", []))
        and any(item.get("emotion") == "행복" for item in manual_patterns.get("patterns", [])),
        json.dumps(weekday_patterns, ensure_ascii=False)[:600],
    )
    mismatch_report_day = day_map.get(yesterday, {})
    check(
        "report: 단일 상태는 단일 점수 기준",
        mismatch_report_day.get("daily_wellness_score") == 85.0
        and mismatch_report_day.get("daily_wellness_label") == "양호"
        and mismatch_report_day.get("cumulative_wellness_score") == 25.0
        and mismatch_report_day.get("cumulative_wellness_label") == "위험"
        and mismatch_report_day.get("label") == "위험",
        json.dumps(mismatch_report_day, ensure_ascii=False),
    )

    calendar_response = client.get(f"/calendar/{username}", headers=headers)
    calendar_rows = calendar_response.json() if calendar_response.status_code == 200 else []
    calendar_yesterday = next((row for row in calendar_rows if row.get("date") == yesterday), {})
    calendar_today = next((row for row in calendar_rows if row.get("date") == today), {})
    check(
        "calendar: 단일 상태는 단일 점수 기준",
        calendar_response.status_code == 200
        and calendar_yesterday.get("daily_wellness_score") == 85.0
        and calendar_yesterday.get("daily_wellness_label") == "양호"
        and calendar_yesterday.get("cumulative_wellness_score") == 25.0
        and calendar_yesterday.get("cumulative_wellness_label") == "위험"
        and calendar_yesterday.get("label") == "위험",
        json.dumps(calendar_yesterday, ensure_ascii=False),
    )
    check(
        "calendar: 수동 감정 기록 포함",
        calendar_response.status_code == 200
        and calendar_today.get("manual_emotion_label") == "슬픔"
        and calendar_today.get("manual_emotion_intensity") == 2
        and calendar_today.get("manual_emotion_note") == "저녁에는 살짝 가라앉음",
        json.dumps(calendar_today, ensure_ascii=False),
    )

    response = client.get(
        f"/report/weekly/{username}", params={"end_date": "bad-date"}, headers=headers,
    )
    check("report: 잘못된 날짜 400", response.status_code == 400, response.text)

    # ── 6a. 비밀번호 변경 ──────────────────────────────────────────────────────
    response = client.post("/auth/change-password", headers=headers, json={
        "username": username, "current_password": "wrong-pass", "new_password": "newpass99",
    })
    check("password: 현재 비밀번호 불일치 401", response.status_code == 401, response.text)

    response = client.post("/auth/change-password", headers=headers, json={
        "username": username, "current_password": password, "new_password": "newpass99",
    })
    check("password: 변경 성공", response.status_code == 200, response.text)

    status_old, _ = login(client, username, password)
    status_new, headers_new = login(client, username, "newpass99")
    check(
        "password: 이전 비밀번호 차단 + 새 비밀번호 로그인",
        status_old == 401 and status_new == 200,
        f"old={status_old}, new={status_new}",
    )
    headers = headers_new

    response = client.post("/auth/reset-password", json={
        "username": username,
        "email": "wrong@example.local",
        "new_password": "resetpass99",
    })
    check("password-reset: 이메일 불일치 401", response.status_code == 401, response.text)

    response = client.post("/auth/reset-password", json={
        "username": username,
        "email": f"{username}@example.local",
        "new_password": "resetpass99",
    })
    check("password-reset: 이메일 확인 후 재설정 성공", response.status_code == 200, response.text)

    status_changed_old, _ = login(client, username, "newpass99")
    status_reset, headers_reset = login(client, username, "resetpass99")
    check(
        "password-reset: 이전 새 비밀번호 차단 + 재설정 비밀번호 로그인",
        status_changed_old == 401 and status_reset == 200,
        f"old={status_changed_old}, reset={status_reset}",
    )
    headers = headers_reset

    # ── 6b. 데이터 내보내기 ────────────────────────────────────────────────────
    response = client.get(f"/export/{username}", headers=headers)
    body = response.json() if response.status_code == 200 else {}
    exported_dates = {day["date"] for day in body.get("days", [])}
    exported_today = next((day for day in body.get("days", []) if day.get("date") == today), {})
    check("export: 내보내기 200", response.status_code == 200, response.text[:200])
    check(
        "export: 대화/요약/피드백/수동감정 포함",
        {two_days_ago, yesterday, today} <= exported_dates
        and len(body.get("feedback", [])) == 2
        and len(body.get("daily_emotion_notes", [])) == 3
        and exported_today.get("manual_emotion_note", {}).get("emotion_label") == "슬픔"
        and body.get("username") == username
        and "attachment" in response.headers.get("content-disposition", ""),
        (
            f"dates={sorted(exported_dates)}, feedback={len(body.get('feedback', []))}, "
            f"notes={len(body.get('daily_emotion_notes', []))}"
        ),
    )

    # ── 6c. 계정 삭제 ──────────────────────────────────────────────────────────
    response = client.post("/account/delete", headers=headers, json={
        "username": username, "password": "resetpass99", "confirm": "WRONG",
    })
    check("delete: 확인 문구 불일치 400", response.status_code == 400, response.text)

    response = client.post("/account/delete", headers=headers, json={
        "username": username, "password": "wrong-pass", "confirm": "DELETE",
    })
    check("delete: 비밀번호 불일치 401", response.status_code == 401, response.text)

    response = client.post("/account/delete", headers=headers, json={
        "username": username, "password": "resetpass99", "confirm": "DELETE",
    })
    body = response.json() if response.status_code == 200 else {}
    check(
        "delete: 계정 삭제 성공 + users row 제거",
        response.status_code == 200
        and body.get("removed", {}).get("users") == 1
        and body.get("removed", {}).get("daily_emotion_notes") == 3,
        response.text,
    )

    status_after, _ = login(client, username, "resetpass99")
    check("delete: 삭제 후 로그인 차단", status_after == 401, f"status={status_after}")

    # 관리자 기본 계정은 삭제 차단
    status_admin, admin_headers = login(client, "developer", "developer")
    response = client.post("/account/delete", headers=admin_headers, json={
        "username": "developer", "password": "developer", "confirm": "DELETE",
    })
    check(
        "delete: 관리자 계정 삭제 차단 403",
        status_admin == 200 and response.status_code == 403,
        response.text,
    )

    # ── 점수 가시성 게이팅: 비관리자 raw 숨김 + 우울 경향 밴드, 관리자 raw 유지 ──────
    gate_user = "smoke_gate_user"
    gate_headers = register(client, gate_user, "test1234")
    seed_conversation(gate_user, "2026-06-09", [
        ("user", "요즘 계속 우울하고 아무것도 하기 싫어"),
        ("bot", "많이 힘드셨겠어요. 천천히 이야기 나눠요."),
    ])
    gate_close = client.post(
        "/day/close",
        headers=gate_headers,
        json={"username": gate_user, "date": "2026-06-09"},
    )
    gate_body = gate_close.json() if gate_close.status_code == 200 else {}
    check(
        "score-gate: 비관리자 /day/close raw 숨김 + 우울 경향 밴드 노출",
        gate_close.status_code == 200
        and gate_body.get("daily_score") is None
        and gate_body.get("smoothed_score") is None
        and gate_body.get("depression_tendency_daily") is None
        and gate_body.get("depression_tendency_smoothed") is None
        and gate_body.get("depression_tendency_band") in {"high", "mid", "low"}
        and gate_body.get("daily_wellness_score") is not None
        and gate_body.get("daily_wellness_label") in {"양호", "보통", "주의", "위험"}
        and gate_body.get("cumulative_wellness_score") is not None
        and gate_body.get("cumulative_wellness_label") in {"양호", "보통", "주의", "위험"}
        and gate_body.get("wellness_score") is not None,
        gate_close.text,
    )

    gate_cal = client.get(f"/calendar/{gate_user}", headers=gate_headers)
    cal_rows = gate_cal.json() if gate_cal.status_code == 200 else []
    gate_row = next((r for r in cal_rows if r.get("date") == "2026-06-09"), None)
    check(
        "score-gate: 비관리자 /calendar 소수점 숨김 + 밴드 노출",
        gate_cal.status_code == 200
        and gate_row is not None
        and gate_row.get("daily_wellness_score") is not None
        and gate_row.get("daily_wellness_label") in {"양호", "보통", "주의", "위험"}
        and gate_row.get("cumulative_wellness_score") is not None
        and gate_row.get("cumulative_wellness_label") in {"양호", "보통", "주의", "위험"}
        and gate_row.get("depression_tendency_daily") is None
        and gate_row.get("depression_tendency_smoothed") is None
        and gate_row.get("depression_tendency_band") in {"high", "mid", "low"},
        gate_cal.text,
    )

    # 관리자(developer)는 진단·모니터링용 raw 점수를 그대로 받는다.
    seed_conversation("developer", "2026-06-09", [
        ("user", "오늘은 비교적 괜찮게 보냈어"),
        ("bot", "다행이에요."),
    ])
    admin_close = client.post(
        "/day/close",
        headers=admin_headers,
        json={"username": "developer", "date": "2026-06-09"},
    )
    admin_body = admin_close.json() if admin_close.status_code == 200 else {}
    check(
        "score-gate: 관리자 /day/close raw 점수 유지",
        admin_close.status_code == 200
        and admin_body.get("daily_score") is not None
        and admin_body.get("smoothed_score") is not None
        and admin_body.get("daily_wellness_score") is not None
        and admin_body.get("daily_wellness_label") in {"양호", "보통", "주의", "위험"}
        and admin_body.get("cumulative_wellness_score") is not None
        and admin_body.get("cumulative_wellness_label") in {"양호", "보통", "주의", "위험"}
        and admin_body.get("depression_tendency_band") in {"high", "mid", "low"},
        admin_close.text,
    )

    # ── 결과 요약 ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"통과 {len(PASSED)} / 실패 {len(FAILED)}")
    if FAILED:
        print("실패 목록:")
        for name in FAILED:
            print(f"  - {name}")
        return 1
    print("신규 기능 스모크 전체 통과")
    return 0


if __name__ == "__main__":
    sys.exit(main())
