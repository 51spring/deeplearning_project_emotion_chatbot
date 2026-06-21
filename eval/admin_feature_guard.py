"""
admin_feature_guard.py
역할: 관리자 계정 seed, 다음날 전환, DB 초기화 보호 동작을 임시 DB에서 검증
입력: 없음
출력: 검증 성공 시 JSON 요약 출력
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
TEST_DB_PATH = ROOT_DIR / "eval" / "_admin_feature_guard.db"

# 백엔드 모듈 import 전에 테스트 DB 경로를 지정해야 init_db가 임시 DB를 사용한다.
os.environ["EMOTION_CHATBOT_DB_PATH"] = str(TEST_DB_PATH)

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if TEST_DB_PATH.exists():
    TEST_DB_PATH.unlink()

from fastapi.testclient import TestClient  # noqa: E402

from backend import main as backend_main  # noqa: E402
from backend.db.models import (  # noqa: E402
    CrisisEvent,
    DailySummary,
    ModelAuditEvent,
    Session as ChatSession,
    User,
    Utterance,
)


def _login(client: TestClient, username: str, password: str) -> dict[str, str]:
    """
    역할: 테스트 클라이언트로 로그인하고 Authorization 헤더를 생성
    입력: TestClient, 사용자 이름, 비밀번호
    출력: Bearer 토큰 헤더 dict
    """
    response = client.post(
        "/auth/login",
        json={"username": username, "password": password},
    )
    assert response.status_code == 200, response.text
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def _register(client: TestClient, username: str, password: str) -> dict[str, str]:
    """
    역할: 일반 사용자를 이메일 포함 가입 후 Authorization 헤더를 생성
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
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def _runtime_counts(username: str | None = None) -> dict[str, int]:
    """
    역할: 초기화 대상 런타임 테이블의 현재 row 수 조회
    입력: 사용자 이름 또는 None
    출력: 테이블별 row 수 dict
    """
    db = backend_main.SessionLocal()
    try:
        user = None
        if username is not None:
            user = db.query(User).filter(User.username == username).first()
            assert user is not None, username
            session_ids = [
                row.id
                for row in db.query(ChatSession.id).filter(ChatSession.user_id == user.id).all()
            ]
            utterance_count = (
                db.query(Utterance).filter(Utterance.session_id.in_(session_ids)).count()
                if session_ids
                else 0
            )
            return {
                "sessions": db.query(ChatSession).filter(ChatSession.user_id == user.id).count(),
                "utterances": utterance_count,
                "daily_summaries": db.query(DailySummary).filter(DailySummary.user_id == user.id).count(),
                "crisis_events": db.query(CrisisEvent).filter(CrisisEvent.user_id == user.id).count(),
                "model_audit_events": db.query(ModelAuditEvent).filter(ModelAuditEvent.user_id == user.id).count(),
            }
        return {
            "sessions": db.query(ChatSession).count(),
            "utterances": db.query(Utterance).count(),
            "daily_summaries": db.query(DailySummary).count(),
            "crisis_events": db.query(CrisisEvent).count(),
            "model_audit_events": db.query(ModelAuditEvent).count(),
        }
    finally:
        db.close()


def _admin_usernames() -> list[str]:
    """
    역할: 관리자 권한이 있는 사용자 이름 목록 조회
    입력: 없음
    출력: 관리자 사용자 이름 리스트
    """
    db = backend_main.SessionLocal()
    try:
        rows = db.query(User).filter(User.is_developer.is_(True)).order_by(User.username).all()
        return [row.username for row in rows]
    finally:
        db.close()


def main() -> None:
    """
    역할: 관리자 기능 회귀 검증을 실행
    입력: 없음
    출력: 표준 출력에 JSON 요약 출력
    """
    client = TestClient(backend_main.app)

    root_headers = _login(client, "root", "root")
    developer_headers = _login(client, "developer", "developer")
    normal_headers = _register(client, "normal_user", "pass1234")

    current_response = client.get("/day/current/root", headers=root_headers)
    assert current_response.status_code == 200, current_response.text
    assert current_response.json()["is_developer"] is True

    denied_advance = client.post(
        "/day/advance",
        json={"username": "normal_user"},
        headers=normal_headers,
    )
    assert denied_advance.status_code == 403, denied_advance.text

    denied_reset = client.post(
        "/admin/reset-db",
        json={"username": "normal_user", "confirm": "RESET"},
        headers=normal_headers,
    )
    assert denied_reset.status_code == 403, denied_reset.text

    advance_response = client.post(
        "/day/advance",
        json={"username": "root"},
        headers=root_headers,
    )
    assert advance_response.status_code == 200, advance_response.text

    developer_advance = client.post(
        "/day/advance",
        json={"username": "developer"},
        headers=developer_headers,
    )
    assert developer_advance.status_code == 200, developer_advance.text

    bad_confirm = client.post(
        "/admin/reset-db",
        json={"username": "root", "confirm": "NO"},
        headers=root_headers,
    )
    assert bad_confirm.status_code == 400, bad_confirm.text
    assert _runtime_counts("root")["sessions"] > 0
    assert _runtime_counts("developer")["sessions"] > 0

    reset_response = client.post(
        "/admin/reset-db",
        json={"username": "root", "confirm": "RESET"},
        headers=root_headers,
    )
    assert reset_response.status_code == 200, reset_response.text
    reset_payload = reset_response.json()
    assert reset_payload["reset"] is True
    assert reset_payload["preserved_users"] is True
    assert reset_payload["username"] == "root"

    root_counts_after = _runtime_counts("root")
    developer_counts_after = _runtime_counts("developer")
    assert all(count == 0 for count in root_counts_after.values()), root_counts_after
    assert developer_counts_after["sessions"] > 0, developer_counts_after
    assert developer_counts_after["daily_summaries"] > 0, developer_counts_after
    admins = _admin_usernames()
    assert "root" in admins and "developer" in admins, admins

    print(
        json.dumps(
            {
                "root_login": "ok",
                "admin_users": admins,
                "normal_user_denied": True,
                "advance_status": advance_response.status_code,
                "deleted": reset_payload["deleted"],
                "root_counts_after": root_counts_after,
                "developer_counts_after": developer_counts_after,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        backend_main.engine.dispose()
        if TEST_DB_PATH.exists():
            TEST_DB_PATH.unlink()
