"""
security_concurrency_guard.py
역할: 공개 배포 보안 제한과 추론/사용자 상태 직렬화 회귀 검증
입력: 없음
출력: 검증 성공 시 JSON 요약, 실패 시 assertion
실행: C:/Users/WD/anaconda3/envs/dl_study/python.exe eval/security_concurrency_guard.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
TEST_DB_PATH = ROOT_DIR / "eval" / "_security_concurrency_guard.db"
PYTHON_EXE = Path(r"C:\Users\WD\anaconda3\envs\dl_study\python.exe")

# 백엔드 import 전에 임시 DB와 짧은 테스트용 제한값을 설정한다.
os.environ["EMOTION_CHATBOT_DB_PATH"] = str(TEST_DB_PATH)
os.environ["EMOTION_CHATBOT_REGISTER_RATE_LIMIT"] = "100"
os.environ["EMOTION_CHATBOT_LOGIN_RATE_LIMIT"] = "100"
os.environ["EMOTION_CHATBOT_LOGIN_MAX_FAILURES"] = "3"
os.environ["EMOTION_CHATBOT_LOGIN_LOCK_SECONDS"] = "60"
os.environ["EMOTION_CHATBOT_CHAT_RATE_LIMIT"] = "2"
os.environ["EMOTION_CHATBOT_CHAT_HOURLY_LIMIT"] = "100"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if TEST_DB_PATH.exists():
    TEST_DB_PATH.unlink()

from fastapi.testclient import TestClient  # noqa: E402

from backend import main as backend_main  # noqa: E402
from backend.db import crud  # noqa: E402
from backend.db.models import User  # noqa: E402
from backend.runtime_guards import (  # noqa: E402
    RateLimitRule,
    SerializedInferenceQueue,
    SlidingWindowRateLimiter,
)


def _register(client: TestClient, username: str) -> dict[str, str]:
    """
    역할: 테스트 사용자를 이메일 포함 가입시키고 Bearer 헤더 생성
    입력: TestClient, 사용자 이름
    출력: Authorization 헤더 dict
    """
    response = client.post(
        "/auth/register",
        json={
            "username": username,
            "nickname": username,
            "email": f"{username}@example.local",
            "password": "test1234",
        },
    )
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def _mock_roberta_result() -> dict:
    """
    역할: 모델 로드 없이 /chat을 검증할 RoBERTa 결과 생성
    입력: 없음
    출력: /chat 호환 결과 dict
    """
    return {
        "roberta_score": 0.25,
        "cbt_score": 0.20,
        "depression_score": 0.25,
        "depression_tendency_score": 0.02,
        "top_emotion": "행복",
        "entailment_prob": 0.01,
        "is_crisis": False,
        "utterance_type": "positive_share",
    }


def _mock_qwen_result() -> dict:
    """
    역할: 모델 로드 없이 /chat을 검증할 Qwen 결과 생성
    입력: 없음
    출력: /chat 호환 결과 dict
    """
    return {
        "response": "오늘의 편안한 흐름을 이어가도 좋겠어요.",
        "has_crisis_tag": False,
    }


def _test_rate_limiter_unit() -> None:
    """
    역할: sliding-window 제한의 허용/차단/만료 동작 검증
    입력: 없음
    출력: 없음
    """
    limiter = SlidingWindowRateLimiter()
    rule = RateLimitRule(max_requests=2, window_seconds=10)
    assert limiter.consume("unit", rule, now=0.0) == 0
    assert limiter.consume("unit", rule, now=1.0) == 0
    assert limiter.consume("unit", rule, now=2.0) > 0
    assert limiter.consume("unit", rule, now=11.0) == 0


def _test_fifo_inference_queue() -> int:
    """
    역할: FIFO 추론 큐에서 동시에 하나의 작업만 활성화되는지 검증
    입력: 없음
    출력: 관측된 최대 동시 실행 수
    """
    queue = SerializedInferenceQueue()
    state_lock = threading.Lock()
    active = 0
    max_active = 0

    def worker() -> None:
        """
        역할: 추론 slot 안의 동시 실행 수 측정
        입력: 없음
        출력: 없음
        """
        nonlocal active, max_active
        with queue.slot():
            with state_lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.03)
            with state_lock:
                active -= 1

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(worker) for _ in range(4)]
        for future in futures:
            future.result()

    assert max_active == 1, max_active
    assert queue.queued_count() == 0
    return max_active


def _test_production_secret_rejection() -> None:
    """
    역할: production에서 알려진/짧은 token secret이 import 단계에서 거부되는지 검증
    입력: 없음
    출력: 없음
    """
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT_DIR)
    env["EMOTION_CHATBOT_ENV"] = "production"
    env["EMOTION_CHATBOT_AUTH_SECRET"] = (
        "class-demo-local-secret-20260521-change-before-public"
    )
    result = subprocess.run(
        [str(PYTHON_EXE), "-c", "import backend.auth_utils"],
        cwd=ROOT_DIR,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "32자 이상의 강한 임의 값" in (result.stderr + result.stdout)


def _test_account_lock(client: TestClient) -> int:
    """
    역할: 연속 로그인 실패 후 DB 영속 계정 잠금과 Retry-After 검증
    입력: TestClient
    출력: 잠금 응답의 Retry-After 초
    """
    _register(client, "lock_user")

    for _ in range(2):
        response = client.post(
            "/auth/login",
            json={"username": "lock_user", "password": "wrong-pass"},
        )
        assert response.status_code == 401, response.text

    response = client.post(
        "/auth/login",
        json={"username": "lock_user", "password": "wrong-pass"},
    )
    assert response.status_code == 429, response.text
    retry_after = int(response.headers["retry-after"])
    assert retry_after == 60

    response = client.post(
        "/auth/login",
        json={"username": "lock_user", "password": "test1234"},
    )
    assert response.status_code == 429, response.text

    db = backend_main.SessionLocal()
    try:
        user = crud.get_user_by_username(db, "lock_user")
        assert user is not None
        assert user.locked_until is not None
        assert crud.get_login_lock_remaining_seconds(user) > 0
    finally:
        db.close()
    return retry_after


def _test_chat_rate_limit(client: TestClient) -> None:
    """
    역할: 사용자별 /chat 분당 제한과 입력 길이 제한 검증
    입력: TestClient
    출력: 없음
    """
    headers = _register(client, "rate_user")
    for index in range(2):
        response = client.post(
            "/chat",
            headers=headers,
            json={"username": "rate_user", "text": f"테스트 발화 {index}"},
        )
        assert response.status_code == 200, response.text

    response = client.post(
        "/chat",
        headers=headers,
        json={"username": "rate_user", "text": "세 번째 요청"},
    )
    assert response.status_code == 429, response.text
    assert int(response.headers["retry-after"]) >= 1

    length_headers = _register(client, "length_user")
    response = client.post(
        "/chat",
        headers=length_headers,
        json={"username": "length_user", "text": "가" * 2001},
    )
    assert response.status_code == 422, response.text


def _test_same_user_request_lock(client: TestClient) -> int:
    """
    역할: 같은 사용자의 동시 /chat 요청이 사용자 상태 lock으로 직렬화되는지 검증
    입력: TestClient
    출력: 관측된 최대 동시 RoBERTa mock 실행 수
    """
    headers = _register(client, "concurrent_user")
    backend_main.CHAT_MINUTE_RATE_RULE = RateLimitRule(100, 60)
    backend_main._request_rate_limiter.reset()

    state_lock = threading.Lock()
    active = 0
    max_active = 0

    def slow_roberta(text: str) -> dict:
        """
        역할: 동시 실행 여부를 관측하는 지연 RoBERTa mock
        입력: 사용자 발화
        출력: 고정 RoBERTa 결과
        """
        del text
        nonlocal active, max_active
        with state_lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.08)
        with state_lock:
            active -= 1
        return _mock_roberta_result()

    backend_main.scheduler.run_roberta = slow_roberta

    def send(index: int) -> int:
        """
        역할: 동일 사용자 채팅 요청 1건 전송
        입력: 요청 순번
        출력: HTTP 상태 코드
        """
        response = client.post(
            "/chat",
            headers=headers,
            json={
                "username": "concurrent_user",
                "text": f"동시 요청 {index}",
                "client_session_id": "guard-session",
            },
        )
        return response.status_code

    with ThreadPoolExecutor(max_workers=2) as executor:
        statuses = list(executor.map(send, [1, 2]))

    assert statuses == [200, 200], statuses
    assert max_active == 1, max_active
    return max_active


def main() -> None:
    """
    역할: 보안/동시성 guard 전체 실행
    입력: 없음
    출력: 검증 결과 JSON
    """
    _test_rate_limiter_unit()
    inference_max_active = _test_fifo_inference_queue()
    _test_production_secret_rejection()

    original_run_roberta = backend_main.scheduler.run_roberta
    original_run_qwen = backend_main.scheduler.run_qwen
    original_generate_summary = backend_main.scheduler.generate_summary
    try:
        backend_main.scheduler.run_roberta = lambda text: _mock_roberta_result()
        backend_main.scheduler.run_qwen = (
            lambda text, history=None, utterance_info=None: _mock_qwen_result()
        )
        backend_main.scheduler.generate_summary = lambda texts: "테스트 요약"
        with TestClient(backend_main.app) as client:
            retry_after = _test_account_lock(client)
            _test_chat_rate_limit(client)
            same_user_max_active = _test_same_user_request_lock(client)
    finally:
        backend_main.scheduler.run_roberta = original_run_roberta
        backend_main.scheduler.run_qwen = original_run_qwen
        backend_main.scheduler.generate_summary = original_generate_summary

    run_deploy_text = (ROOT_DIR / "run_deploy.bat").read_text(encoding="utf-8")
    assert (
        'set "EMOTION_CHATBOT_AUTH_SECRET=class-demo-local-secret-20260521-change-before-public"'
        not in run_deploy_text
    )
    assert (
        'set "EMOTION_CHATBOT_DEVELOPER_PASSWORD=developer-demo-20260521"'
        not in run_deploy_text
    )
    assert (
        'set "EMOTION_CHATBOT_ROOT_PASSWORD=root-demo-20260521"'
        not in run_deploy_text
    )

    print(
        json.dumps(
            {
                "rate_limiter": "ok",
                "account_lock_retry_after": retry_after,
                "inference_max_active": inference_max_active,
                "same_user_max_active": same_user_max_active,
                "deploy_fixed_secrets_removed": True,
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
