"""
admin_active_date_rehydration_guard.py
역할: 관리자(developer/root)가 '다음날로 넘기기'로 날짜를 시뮬레이션하다가 서버를
      재시작(재배포)해도, 활성 날짜가 DB 기록에서 복원되는지 검증한다.
      재시작은 인메모리 `_active_dates_by_user`를 비우는 것으로 흉내내고,
      startup 훅이 호출하는 `_rehydrate_admin_active_dates()`를 직접 실행해 확인한다.
      (복원 전에는 실제 today로 fallback → 채팅이 빈 새 대화창으로 뜨던 버그)
입력: 없음 (임시 DB, 모델 추론은 mock)
출력: 단계별 PASS/FAIL, 전체 통과 시 종료 코드 0
실행: C:/Users/WD/anaconda3/envs/dl_study/python.exe eval/admin_active_date_rehydration_guard.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
TEST_DB_PATH = ROOT_DIR / "eval" / "_admin_rehydration_guard.db"

# 백엔드 import 전에 임시 DB 경로를 지정해야 init_db가 임시 DB를 사용한다.
os.environ["EMOTION_CHATBOT_DB_PATH"] = str(TEST_DB_PATH)
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if TEST_DB_PATH.exists():
    TEST_DB_PATH.unlink()

from fastapi.testclient import TestClient  # noqa: E402

from backend import main as backend_main  # noqa: E402
from backend.db import crud  # noqa: E402

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    """단일 케이스 통과/실패 기록."""
    if condition:
        PASSED.append(name)
        print(f"[PASS] {name}")
    else:
        FAILED.append(name)
        print(f"[FAIL] {name} :: {detail}")


def _mock_models() -> None:
    """RoBERTa/Qwen 추론을 가벼운 mock으로 대체(중립·비위기)."""
    backend_main.scheduler.run_roberta = lambda text: {
        "roberta_score": 0.25,
        "cbt_score": 0.20,
        "depression_score": 0.25,
        "depression_tendency_score": 0.02,
        "top_emotion": "중립",
        "entailment_prob": 0.01,
        "is_crisis": False,
        "utterance_type": "casual_neutral",
    }
    backend_main.scheduler.run_qwen = (
        lambda text, history=None, utterance_info=None: {
            "response": "그렇군요. 더 이야기해 주세요.",
            "has_crisis_tag": False,
        }
    )


def _user_id(username: str) -> int:
    db = backend_main.SessionLocal()
    try:
        user = crud.get_user_by_username(db, username)
        return user.id
    finally:
        db.close()


def main() -> int:
    """관리자 활성 날짜 복원 케이스를 검증한다."""
    _mock_models()
    client = TestClient(backend_main.app)

    # developer 관리자(_seed_admin_accounts로 자동 생성)로 로그인 — 로컬 기본 비밀번호
    login = client.post(
        "/auth/login", json={"username": "developer", "password": "developer"}
    )
    assert login.status_code == 200, login.text
    headers = {"Authorization": f"Bearer {login.json()['access_token']}"}

    dev_id = _user_id("developer")
    real_today = backend_main._get_app_today().isoformat()

    # 1) 오늘 대화 후 '다음날로 넘기기' 2회 → 시뮬레이션 미래 날짜로 이동
    client.post("/chat", headers=headers,
                json={"username": "developer", "text": "오늘 점검 메모"})
    client.post("/day/advance", headers=headers, json={"username": "developer"})
    client.post("/chat", headers=headers,
                json={"username": "developer", "text": "다음 날 점검 메모"})
    adv = client.post("/day/advance", headers=headers, json={"username": "developer"})
    simulated_date = adv.json().get("current_date")

    check(
        "시뮬레이션 날짜가 실제 today보다 앞섬",
        bool(simulated_date) and simulated_date > real_today,
        f"sim={simulated_date} today={real_today}",
    )
    check(
        "재시작 전 활성 날짜 = 시뮬레이션 날짜",
        backend_main._active_dates_by_user.get(dev_id) == simulated_date,
        f"mem={backend_main._active_dates_by_user.get(dev_id)}",
    )

    # 2) 재시작 흉내 — 인메모리 활성 날짜 전체 삭제
    backend_main._active_dates_by_user.clear()
    check(
        "재시작 직후(복원 전)에는 실제 today로 fallback (버그 재현)",
        backend_main._get_active_date(dev_id) == real_today,
        f"got={backend_main._get_active_date(dev_id)}",
    )

    # 3) startup 복원 실행
    backend_main._rehydrate_admin_active_dates()
    check(
        "복원 후 활성 날짜 = 시뮬레이션 날짜",
        backend_main._active_dates_by_user.get(dev_id) == simulated_date,
        f"restored={backend_main._active_dates_by_user.get(dev_id)} expected={simulated_date}",
    )

    # 4) /day/current도 시뮬레이션 날짜를 반환 → 채팅이 그 날로 복원됨
    cur = client.get("/day/current/developer", headers=headers)
    check(
        "/day/current = 시뮬레이션 날짜 (빈 새 대화창 아님)",
        cur.status_code == 200 and cur.json().get("current_date") == simulated_date,
        cur.text,
    )

    # 5) 그 날의 대화도 DB에서 조회되어 채팅 복원 가능
    utts = client.get(
        "/day/utterances/developer",
        params={"date": simulated_date}, headers=headers,
    )
    check(
        "시뮬레이션 날짜의 대화가 조회됨",
        utts.status_code == 200,
        utts.text,
    )

    # 6) 음성 케이스 — 일반(비관리자) 사용자는 복원 대상이 아님
    reg = client.post("/auth/register", json={
        "username": "norm_user", "nickname": "norm_user",
        "email": "norm_user@example.local", "password": "test1234",
    })
    nheaders = {"Authorization": f"Bearer {reg.json()['access_token']}"}
    client.post("/chat", headers=nheaders,
                json={"username": "norm_user", "text": "안녕하세요"})
    norm_id = _user_id("norm_user")
    backend_main._active_dates_by_user.clear()
    backend_main._rehydrate_admin_active_dates()
    check(
        "일반 사용자는 복원 대상이 아님(실제 today 사용)",
        norm_id not in backend_main._active_dates_by_user,
        f"mem_keys={list(backend_main._active_dates_by_user.keys())}",
    )

    print("\n" + "=" * 50)
    print(f"통과 {len(PASSED)} / 실패 {len(FAILED)}")
    if FAILED:
        print("실패 목록:")
        for name in FAILED:
            print(f"  - {name}")
        return 1
    print("관리자 활성 날짜 복원 가드 전체 통과")
    return 0


if __name__ == "__main__":
    sys.exit(main())
