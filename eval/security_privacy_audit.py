"""
security_privacy_audit.py
역할: DB/로그인/개인정보 보안 상태를 원문·수동 메모 노출 없이 점검
입력: 선택적 EMOTION_CHATBOT_DB_PATH, 현재 git 작업트리
출력: 콘솔 JSON 요약과 eval/report/security_privacy_audit_*.json 리포트
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT_DIR / "eval" / "report"
DEFAULT_DB_PATH = ROOT_DIR / "backend" / "db" / "emotion_chatbot.db"
ENV_KEYS = [
    "EMOTION_CHATBOT_ENV",
    "EMOTION_CHATBOT_AUTH_SECRET",
    "EMOTION_CHATBOT_DEVELOPER_PASSWORD",
    "EMOTION_CHATBOT_ROOT_PASSWORD",
    "EMOTION_CHATBOT_CORS_ORIGINS",
    "EMOTION_CHATBOT_DB_PATH",
    "EMOTION_CHATBOT_AUTH_EXPIRES_SECONDS",
    "EMOTION_CHATBOT_ALLOW_LEGACY_ACCOUNT_CLAIM",
    "EMOTION_CHATBOT_STORE_QWEN_RAW_RESPONSE",
    "EMOTION_CHATBOT_REGISTER_RATE_LIMIT",
    "EMOTION_CHATBOT_REGISTER_RATE_WINDOW_SECONDS",
    "EMOTION_CHATBOT_LOGIN_RATE_LIMIT",
    "EMOTION_CHATBOT_LOGIN_RATE_WINDOW_SECONDS",
    "EMOTION_CHATBOT_PASSWORD_RESET_RATE_LIMIT",
    "EMOTION_CHATBOT_PASSWORD_RESET_RATE_WINDOW_SECONDS",
    "EMOTION_CHATBOT_CHAT_RATE_LIMIT",
    "EMOTION_CHATBOT_CHAT_RATE_WINDOW_SECONDS",
    "EMOTION_CHATBOT_CHAT_HOURLY_LIMIT",
    "EMOTION_CHATBOT_CHAT_HOURLY_WINDOW_SECONDS",
    "EMOTION_CHATBOT_LOGIN_MAX_FAILURES",
    "EMOTION_CHATBOT_LOGIN_LOCK_SECONDS",
    "EMOTION_CHATBOT_TRUST_PROXY_HEADERS",
]
SENSITIVE_TRACKED_RE = re.compile(
    r"(^|/)(\.env|.*\.env|.*\.db|.*\.pt|.*\.pth|.*\.bin|.*\.safetensors|"
    r"frontend/build/|frontend/node_modules/|data/raw/|data/processed/|"
    r"models/.*/checkpoints/|github_submission_20260521\.zip$)",
    re.IGNORECASE,
)
SENSITIVE_SUBMISSION_RE = re.compile(
    r"(^|/)(\.env|.*\.env|.*\.db|.*\.pt|.*\.pth|.*\.bin|.*\.safetensors|"
    r"frontend/build/|frontend/node_modules/|data/raw/|data/processed/|"
    r"__pycache__/|.*\.pyc$|data/nli/nli_pairs\.backup_.*\.csv$)",
    re.IGNORECASE,
)


def _quote_identifier(name: str) -> str:
    """
    역할: SQLite 식별자를 안전하게 인용
    입력: 테이블 또는 컬럼 이름
    출력: 큰따옴표로 감싼 SQLite 식별자
    """
    return '"' + str(name).replace('"', '""') + '"'


def _env_flag(name: str) -> bool:
    """
    역할: 환경변수 플래그 값을 bool로 해석
    입력: 환경변수 이름
    출력: true 계열 문자열 여부
    """
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _env_positive_int(name: str, default: int) -> int:
    """
    역할: 양의 정수 환경변수의 현재 적용값 계산
    입력: 환경변수 이름, 기본값
    출력: 1 이상의 적용 정수
    """
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError:
        value = default
    return max(1, value)


def collect_environment_audit() -> dict[str, Any]:
    """
    역할: 보안 관련 환경변수 설정 여부를 값 노출 없이 수집
    입력: 없음
    출력: 환경변수별 설정 여부와 길이
    """
    app_env = os.environ.get("EMOTION_CHATBOT_ENV", "local").strip().lower()
    production = app_env in {"prod", "production"}
    env_status = {}
    for key in ENV_KEYS:
        value = os.environ.get(key)
        env_status[key] = {
            "is_set": value is not None,
            "length": len(value) if value is not None else 0,
        }

    warnings = []
    if production and not os.environ.get("EMOTION_CHATBOT_AUTH_SECRET"):
        warnings.append("production 모드에서 EMOTION_CHATBOT_AUTH_SECRET이 없습니다.")
    if production and not os.environ.get("EMOTION_CHATBOT_DEVELOPER_PASSWORD"):
        warnings.append("production 모드에서 관리자 developer 비밀번호가 없습니다.")
    if production and not os.environ.get("EMOTION_CHATBOT_ROOT_PASSWORD"):
        warnings.append("production 모드에서 관리자 root 비밀번호가 없습니다.")
    if production and len(os.environ.get("EMOTION_CHATBOT_AUTH_SECRET", "")) < 32:
        warnings.append("production token secret 길이가 32자 미만입니다.")
    developer_password = os.environ.get("EMOTION_CHATBOT_DEVELOPER_PASSWORD", "")
    root_password = os.environ.get("EMOTION_CHATBOT_ROOT_PASSWORD", "")
    if production and (
        len(developer_password) < 12 or len(root_password) < 12
    ):
        warnings.append("production 관리자 비밀번호 길이가 12자 미만입니다.")
    if production and developer_password and developer_password == root_password:
        warnings.append("production developer/root 관리자 비밀번호가 같습니다.")
    if not production and not os.environ.get("EMOTION_CHATBOT_AUTH_SECRET"):
        warnings.append("현재 셸은 로컬 개발용 기본 토큰 secret fallback 상태입니다.")

    return {
        "app_env": app_env,
        "is_production": production,
        "env_status": env_status,
        "legacy_claim_enabled": _env_flag("EMOTION_CHATBOT_ALLOW_LEGACY_ACCOUNT_CLAIM"),
        "store_qwen_raw_response_enabled": _env_flag("EMOTION_CHATBOT_STORE_QWEN_RAW_RESPONSE"),
        "trust_proxy_headers": _env_flag("EMOTION_CHATBOT_TRUST_PROXY_HEADERS"),
        "effective_limits": {
            "register": {
                "max_requests": _env_positive_int(
                    "EMOTION_CHATBOT_REGISTER_RATE_LIMIT", 20
                ),
                "window_seconds": _env_positive_int(
                    "EMOTION_CHATBOT_REGISTER_RATE_WINDOW_SECONDS", 3600
                ),
            },
            "login": {
                "max_requests": _env_positive_int(
                    "EMOTION_CHATBOT_LOGIN_RATE_LIMIT", 30
                ),
                "window_seconds": _env_positive_int(
                    "EMOTION_CHATBOT_LOGIN_RATE_WINDOW_SECONDS", 300
                ),
                "max_failures": _env_positive_int(
                    "EMOTION_CHATBOT_LOGIN_MAX_FAILURES", 5
                ),
                "lock_seconds": _env_positive_int(
                    "EMOTION_CHATBOT_LOGIN_LOCK_SECONDS", 900
                ),
            },
            "password_reset": {
                "max_requests": _env_positive_int(
                    "EMOTION_CHATBOT_PASSWORD_RESET_RATE_LIMIT", 10
                ),
                "window_seconds": _env_positive_int(
                    "EMOTION_CHATBOT_PASSWORD_RESET_RATE_WINDOW_SECONDS", 3600
                ),
            },
            "chat_minute": {
                "max_requests": _env_positive_int(
                    "EMOTION_CHATBOT_CHAT_RATE_LIMIT", 12
                ),
                "window_seconds": _env_positive_int(
                    "EMOTION_CHATBOT_CHAT_RATE_WINDOW_SECONDS", 60
                ),
            },
            "chat_hour": {
                "max_requests": _env_positive_int(
                    "EMOTION_CHATBOT_CHAT_HOURLY_LIMIT", 120
                ),
                "window_seconds": _env_positive_int(
                    "EMOTION_CHATBOT_CHAT_HOURLY_WINDOW_SECONDS", 3600
                ),
            },
        },
        "warnings": warnings,
    }


def _table_names(conn: sqlite3.Connection) -> list[str]:
    """
    역할: SQLite 사용자 테이블 목록 조회
    입력: SQLite 연결
    출력: 테이블명 리스트
    """
    rows = conn.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()
    return [row[0] for row in rows]


def _count_rows(conn: sqlite3.Connection, table: str) -> int:
    """
    역할: 지정 테이블의 row 수 조회
    입력: SQLite 연결, 테이블명
    출력: row 수
    """
    quoted = _quote_identifier(table)
    return int(conn.execute(f"SELECT COUNT(*) FROM {quoted}").fetchone()[0])


def collect_database_audit(db_path: Path) -> dict[str, Any]:
    """
    역할: 운영 SQLite DB 구조와 무결성을 원문 없이 점검
    입력: SQLite DB 경로
    출력: 테이블 row 수, 사용자 해시 상태, orphan 카운트
    """
    result: dict[str, Any] = {
        "db_path": str(db_path),
        "exists": db_path.exists(),
    }
    if not db_path.exists():
        return result

    result["size_bytes"] = db_path.stat().st_size
    conn = sqlite3.connect(db_path)
    try:
        # audit 연결 자체도 FK를 켜서 현재 무결성 검사와 운영 연결 정책을 같은 기준으로 본다.
        conn.execute("PRAGMA foreign_keys=ON")
        result["foreign_keys_pragma"] = int(conn.execute("PRAGMA foreign_keys").fetchone()[0])
        result["foreign_key_check_count"] = len(conn.execute("PRAGMA foreign_key_check").fetchall())
        tables = _table_names(conn)
        table_set = set(tables)
        result["tables"] = tables
        result["row_counts"] = {table: _count_rows(conn, table) for table in tables}

        if "users" in table_set:
            user_columns = {
                row[1] for row in conn.execute("PRAGMA table_info(users)").fetchall()
            }
            result["users"] = {
                "total": conn.execute("SELECT COUNT(*) FROM users").fetchone()[0],
                "admins": conn.execute(
                    "SELECT COUNT(*) FROM users WHERE is_developer = 1"
                ).fetchone()[0],
                "password_hash_null": conn.execute(
                    "SELECT COUNT(*) FROM users "
                    "WHERE password_hash IS NULL OR password_hash = ''"
                ).fetchone()[0],
                "hash_scheme_bad": conn.execute(
                    "SELECT COUNT(*) FROM users "
                    "WHERE password_hash IS NOT NULL "
                    "AND password_hash != '' "
                    "AND password_hash NOT LIKE 'pbkdf2_sha256$%'"
                ).fetchone()[0],
                "login_lock_columns_present": {
                    "failed_login_attempts": "failed_login_attempts" in user_columns,
                    "locked_until": "locked_until" in user_columns,
                },
                "profile_columns_present": {
                    "nickname": "nickname" in user_columns,
                    "email": "email" in user_columns,
                },
            }
            if "email" in user_columns:
                result["users"]["email_missing"] = conn.execute(
                    "SELECT COUNT(*) FROM users WHERE email IS NULL OR email = ''"
                ).fetchone()[0]
            if {"failed_login_attempts", "locked_until"} <= user_columns:
                result["users"]["currently_locked"] = conn.execute(
                    "SELECT COUNT(*) FROM users "
                    "WHERE locked_until IS NOT NULL AND locked_until > CURRENT_TIMESTAMP"
                ).fetchone()[0]
                result["users"]["pending_failed_attempts"] = conn.execute(
                    "SELECT COALESCE(SUM(failed_login_attempts), 0) FROM users"
                ).fetchone()[0]

        integrity = {}
        if {"sessions", "users"} <= table_set:
            integrity["sessions_without_user"] = conn.execute(
                "SELECT COUNT(*) FROM sessions s "
                "LEFT JOIN users u ON s.user_id = u.id WHERE u.id IS NULL"
            ).fetchone()[0]
        if {"utterances", "sessions"} <= table_set:
            integrity["utterances_without_session"] = conn.execute(
                "SELECT COUNT(*) FROM utterances utt "
                "LEFT JOIN sessions s ON utt.session_id = s.id WHERE s.id IS NULL"
            ).fetchone()[0]
        if {"daily_summaries", "users"} <= table_set:
            integrity["daily_summaries_without_user"] = conn.execute(
                "SELECT COUNT(*) FROM daily_summaries d "
                "LEFT JOIN users u ON d.user_id = u.id WHERE u.id IS NULL"
            ).fetchone()[0]
            integrity["duplicate_daily_summaries"] = conn.execute(
                "SELECT COUNT(*) FROM ("
                "SELECT user_id, date, COUNT(*) c FROM daily_summaries "
                "GROUP BY user_id, date HAVING c > 1)"
            ).fetchone()[0]
        if {"daily_emotion_notes", "users"} <= table_set:
            integrity["daily_emotion_notes_without_user"] = conn.execute(
                "SELECT COUNT(*) FROM daily_emotion_notes n "
                "LEFT JOIN users u ON n.user_id = u.id WHERE u.id IS NULL"
            ).fetchone()[0]
            integrity["duplicate_daily_emotion_notes"] = conn.execute(
                "SELECT COUNT(*) FROM ("
                "SELECT user_id, date, COUNT(*) c FROM daily_emotion_notes "
                "GROUP BY user_id, date HAVING c > 1)"
            ).fetchone()[0]
        if {"crisis_events", "users"} <= table_set:
            integrity["crisis_events_without_user"] = conn.execute(
                "SELECT COUNT(*) FROM crisis_events c "
                "LEFT JOIN users u ON c.user_id = u.id WHERE u.id IS NULL"
            ).fetchone()[0]
        if {"model_audit_events", "users"} <= table_set:
            integrity["model_audit_events_without_user"] = conn.execute(
                "SELECT COUNT(*) FROM model_audit_events m "
                "LEFT JOIN users u ON m.user_id = u.id WHERE u.id IS NULL"
            ).fetchone()[0]
        result["integrity_checks"] = integrity

        if "model_audit_events" in table_set:
            result["privacy_counters"] = {
                "audit_payload_rows": conn.execute(
                    "SELECT COUNT(*) FROM model_audit_events "
                    "WHERE audit_payload_json IS NOT NULL AND audit_payload_json != ''"
                ).fetchone()[0],
                "audit_payload_rows_with_qwen_raw_response_key": conn.execute(
                    "SELECT COUNT(*) FROM model_audit_events "
                    "WHERE audit_payload_json LIKE '%qwen_raw_response%'"
                ).fetchone()[0],
            }
    finally:
        conn.close()

    return result


def _git_ls_files() -> list[str]:
    """
    역할: git 추적 파일 목록 조회
    입력: 없음
    출력: 저장소 상대 경로 리스트
    """
    try:
        completed = subprocess.run(
            ["git", "ls-files"],
            cwd=ROOT_DIR,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def collect_sensitive_file_audit() -> dict[str, Any]:
    """
    역할: git/제출 폴더에 민감·대용량 파일이 포함됐는지 경로 기준 점검
    입력: 없음
    출력: 민감 파일 후보 경로 목록과 개수
    """
    tracked = _git_ls_files()
    tracked_sensitive = [path for path in tracked if SENSITIVE_TRACKED_RE.search(path)]

    submission_root = ROOT_DIR / "github_submission_20260521"
    submission_sensitive: list[str] = []
    if submission_root.exists():
        for path in submission_root.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(submission_root).as_posix()
            if SENSITIVE_SUBMISSION_RE.search(rel):
                submission_sensitive.append(rel)

    return {
        "tracked_sensitive_count": len(tracked_sensitive),
        "tracked_sensitive_files": tracked_sensitive,
        "submission_sensitive_count": len(submission_sensitive),
        "submission_sensitive_files": sorted(submission_sensitive),
    }


def build_audit() -> dict[str, Any]:
    """
    역할: 전체 보안/개인정보 점검 결과 구성
    입력: 없음
    출력: 점검 결과 dict
    """
    db_path = Path(os.environ.get("EMOTION_CHATBOT_DB_PATH") or DEFAULT_DB_PATH)
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "environment": collect_environment_audit(),
        "database": collect_database_audit(db_path),
        "files": collect_sensitive_file_audit(),
        "notes": [
            "이 리포트는 사용자 발화 원문, 비밀번호, 토큰 값을 출력하지 않는다.",
            "SQLite DB 자체는 평문 파일이므로 공개 배포/공유 대상에서 제외해야 한다.",
            "rate limit은 단일 Uvicorn worker의 프로세스 메모리 기준이며 계정 잠금은 DB에 저장한다.",
        ],
    }


def write_report(result: dict[str, Any]) -> Path:
    """
    역할: 보안 점검 리포트를 timestamp 파일과 latest 파일로 저장
    입력: 점검 결과 dict
    출력: timestamp report 경로
    """
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORT_DIR / f"security_privacy_audit_{stamp}.json"
    latest_path = REPORT_DIR / "security_privacy_audit_latest.json"
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copyfile(report_path, latest_path)
    return report_path


def main() -> int:
    """
    역할: 보안/개인정보 점검을 실행하고 결과를 출력
    입력: 없음
    출력: 프로세스 종료 코드
    """
    result = build_audit()
    report_path = write_report(result)
    result["report_path"] = str(report_path)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
