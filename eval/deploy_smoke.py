"""
deploy_smoke.py
역할: 수업 데모 배포 서버의 핵심 API와 React 정적 서빙을 빠르게 점검한다.
입력: --base-url, --skip-chat 선택 인자
출력: 콘솔 요약과 eval/report/deploy_smoke_*.json 결과 파일
실행: C:/Users/WD/anaconda3/envs/dl_study/python.exe eval/deploy_smoke.py --base-url http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


BASE_DIR = Path(__file__).resolve().parent.parent
REPORT_DIR = BASE_DIR / "eval" / "report"


def parse_args() -> argparse.Namespace:
    """
    역할: 명령행 인자를 해석한다.
    입력: 없음
    출력: argparse.Namespace
    """
    parser = argparse.ArgumentParser(description="수업 데모 배포 서버 smoke 점검")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="점검할 배포 서버 URL",
    )
    parser.add_argument(
        "--skip-chat",
        action="store_true",
        help="모델 추론이 필요한 /chat 점검을 건너뜀",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=240,
        help="요청별 timeout 초",
    )
    return parser.parse_args()


def _read_response(req: Request, timeout: int) -> tuple[int, str]:
    """
    역할: HTTP 요청을 보내고 상태 코드와 본문을 반환한다.
    입력: urllib Request, timeout 초
    출력: (status_code, response_text)
    """
    try:
        with urlopen(req, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
            return response.status, body
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} {req.full_url}: {body}") from exc
    except URLError as exc:
        raise RuntimeError(f"요청 실패 {req.full_url}: {exc.reason}") from exc


def get_text(url: str, timeout: int) -> tuple[int, str]:
    """
    역할: GET 요청으로 텍스트 응답을 받는다.
    입력: URL, timeout 초
    출력: (status_code, response_text)
    """
    req = Request(url, method="GET")
    return _read_response(req, timeout)


def request_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: int = 240,
) -> dict[str, Any]:
    """
    역할: JSON API 요청을 보내고 JSON 응답을 dict로 반환한다.
    입력: HTTP method, URL, 요청 payload, headers, timeout 초
    출력: JSON dict
    """
    body = None
    request_headers = {"Accept": "application/json"}
    if headers:
        request_headers.update(headers)
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request_headers["Content-Type"] = "application/json; charset=utf-8"

    req = Request(url, data=body, headers=request_headers, method=method)
    status, text = _read_response(req, timeout)
    if not (200 <= status < 300):
        raise RuntimeError(f"HTTP {status} {url}: {text}")
    return json.loads(text) if text else {}


def make_smoke_username() -> str:
    """
    역할: 매 실행마다 충돌하지 않는 점검용 사용자 아이디를 만든다.
    입력: 없음
    출력: 사용자 아이디 문자열
    """
    return f"smoke{time.strftime('%Y%m%d%H%M%S')}"


def auth_header(token: str) -> dict[str, str]:
    """
    역할: 보호 API 호출용 Bearer 인증 헤더를 만든다.
    입력: access token
    출력: Authorization 헤더 dict
    """
    return {"Authorization": f"Bearer {token}"}


def run_smoke(base_url: str, skip_chat: bool, timeout: int) -> dict[str, Any]:
    """
    역할: 배포 서버의 핵심 시연 경로를 순서대로 점검한다.
    입력: base_url, /chat 생략 여부, timeout 초
    출력: 점검 결과 dict
    """
    base = base_url.rstrip("/")
    username = make_smoke_username()
    password = "SmokePass2026!"
    email = f"{username}@example.local"
    client_session_id = f"smoke-{username}"

    # 프론트 정적 파일과 백엔드 health를 먼저 확인해 배포 형태를 검증한다.
    health = request_json("GET", f"{base}/health", timeout=timeout)
    root_status, root_html = get_text(f"{base}/", timeout)
    root_ok = root_status == 200 and ("<div id=\"root\">" in root_html or "static/js" in root_html)

    register = request_json(
        "POST",
        f"{base}/auth/register",
        {
            "username": username,
            "nickname": username,
            "email": email,
            "password": password,
        },
        timeout=timeout,
    )
    login = request_json(
        "POST",
        f"{base}/auth/login",
        {"username": username, "password": password},
        timeout=timeout,
    )
    token = str(login["access_token"])
    headers = auth_header(token)

    chat_result: dict[str, Any] | None = None
    if not skip_chat:
        chat_result = request_json(
            "POST",
            f"{base}/chat",
            {
                "username": username,
                "text": "배포 점검용으로 오늘 기분이 괜찮아.",
                "client_session_id": client_session_id,
            },
            headers=headers,
            timeout=timeout,
        )

    close_day = request_json(
        "POST",
        f"{base}/day/close",
        {"username": username},
        headers=headers,
        timeout=timeout,
    )
    calendar = request_json(
        "GET",
        f"{base}/calendar/{quote(username)}?limit=7",
        headers=headers,
        timeout=timeout,
    )

    return {
        "ok": True,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "base_url": base,
        "username": username,
        "health": health,
        "root_ok": root_ok,
        "register_ok": bool(register.get("access_token")),
        "login_ok": bool(login.get("access_token")),
        "chat": chat_result,
        "close_day": close_day,
        "calendar_count": len(calendar) if isinstance(calendar, list) else None,
        "calendar_latest": calendar[0] if isinstance(calendar, list) and calendar else None,
    }


def write_report(result: dict[str, Any]) -> Path:
    """
    역할: smoke 결과를 timestamp 파일과 latest 파일로 저장한다.
    입력: 점검 결과 dict
    출력: timestamp report 경로
    """
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    report_path = REPORT_DIR / f"deploy_smoke_{stamp}.json"
    latest_path = REPORT_DIR / "deploy_smoke_latest.json"
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copyfile(report_path, latest_path)
    return report_path


def main() -> int:
    """
    역할: 배포 smoke 점검을 실행하고 성공/실패 exit code를 반환한다.
    입력: 없음
    출력: 프로세스 종료 코드
    """
    args = parse_args()
    try:
        result = run_smoke(args.base_url, args.skip_chat, args.timeout)
        report_path = write_report(result)
    except Exception as exc:  # noqa: BLE001 - 발표 전 점검 도구는 실패 메시지를 명확히 출력한다.
        print(f"[FAIL] {exc}")
        return 1

    chat = result.get("chat") or {}
    close_day = result.get("close_day") or {}
    print("[OK] 배포 smoke 점검 통과")
    print(f"- base_url: {result['base_url']}")
    print(f"- username: {result['username']}")
    print(f"- root_ok: {result['root_ok']}")
    if chat:
        print(
            "- chat: "
            f"emotion={chat.get('top_emotion')} "
            f"label={chat.get('label')} "
            f"wellness={chat.get('wellness_score')}"
        )
    print(
        "- close_day: "
        f"label={close_day.get('label')} "
        f"wellness={close_day.get('wellness_score')} "
        f"utterances={close_day.get('utterance_count')}"
    )
    print(f"- calendar_count: {result.get('calendar_count')}")
    print(f"- report: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
