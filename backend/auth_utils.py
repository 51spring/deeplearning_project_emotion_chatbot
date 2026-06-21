"""
auth_utils.py
역할: 아이디/비밀번호 계정용 비밀번호 해시, 검증, 서명 토큰 유틸리티
입력: 평문 비밀번호, 저장된 해시 문자열, 사용자 식별 정보, 서명 토큰
출력: 저장 가능한 PBKDF2 해시 문자열, 검증 결과, Bearer 토큰 payload
"""

import base64
import binascii
import hashlib
import hmac
import json
import os
import secrets
import time


PASSWORD_SCHEME = "pbkdf2_sha256"
PBKDF2_ITERATIONS = 260_000
SALT_BYTES = 16
TOKEN_TYPE = "Bearer"
TOKEN_ALGORITHM = "HS256"
DEFAULT_TOKEN_SECRET = "emotion-chatbot-local-dev-secret"
KNOWN_INSECURE_TOKEN_SECRETS = {
    DEFAULT_TOKEN_SECRET,
    "class-demo-local-secret-20260521-change-before-public",
    "change-me-long-random",
}
MIN_PRODUCTION_SECRET_LENGTH = 32
APP_ENV = os.environ.get("EMOTION_CHATBOT_ENV", "local").strip().lower()
IS_PRODUCTION_ENV = APP_ENV in {"prod", "production"}
TOKEN_SECRET_FROM_ENV = os.environ.get("EMOTION_CHATBOT_AUTH_SECRET")
TOKEN_SECRET = TOKEN_SECRET_FROM_ENV or DEFAULT_TOKEN_SECRET
TOKEN_EXPIRES_SECONDS = int(
    os.environ.get("EMOTION_CHATBOT_AUTH_EXPIRES_SECONDS", str(60 * 60 * 24))
)
MIN_PASSWORD_LENGTH = 8 if IS_PRODUCTION_ENV else 4

if IS_PRODUCTION_ENV and (
    not TOKEN_SECRET_FROM_ENV
    or TOKEN_SECRET_FROM_ENV in KNOWN_INSECURE_TOKEN_SECRETS
    or len(TOKEN_SECRET_FROM_ENV) < MIN_PRODUCTION_SECRET_LENGTH
):
    raise RuntimeError(
        "production 모드에서는 EMOTION_CHATBOT_AUTH_SECRET을 "
        f"{MIN_PRODUCTION_SECRET_LENGTH}자 이상의 강한 임의 값으로 반드시 설정해야 합니다."
    )


def hash_password(password: str) -> str:
    """
    역할: 평문 비밀번호를 PBKDF2-SHA256 해시 문자열로 변환
    입력: 평문 비밀번호
    출력: scheme$iterations$salt_hex$digest_hex 형식의 저장 문자열
    """
    salt = secrets.token_bytes(SALT_BYTES)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PBKDF2_ITERATIONS,
    )
    return (
        f"{PASSWORD_SCHEME}${PBKDF2_ITERATIONS}"
        f"${salt.hex()}${digest.hex()}"
    )


def verify_password(password: str, password_hash: str | None) -> bool:
    """
    역할: 평문 비밀번호가 저장된 PBKDF2 해시와 일치하는지 확인
    입력: 평문 비밀번호, 저장된 해시 문자열
    출력: 일치 여부
    """
    if not password_hash:
        return False

    try:
        scheme, iterations_text, salt_hex, expected_hex = password_hash.split("$", 3)
        if scheme != PASSWORD_SCHEME:
            return False
        iterations = int(iterations_text)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(expected_hex)
    except (ValueError, TypeError):
        return False

    actual = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        iterations,
    )
    return hmac.compare_digest(actual, expected)


def is_password_usable(password: str) -> bool:
    """
    역할: 최소한의 로컬 계정 비밀번호 규칙을 확인
    입력: 평문 비밀번호
    출력: 사용 가능 여부
    """
    return len(str(password)) >= MIN_PASSWORD_LENGTH


def get_min_password_length() -> int:
    """
    역할: 현재 실행 모드에서 요구하는 최소 비밀번호 길이 반환
    입력: 없음
    출력: 최소 비밀번호 길이
    """
    return MIN_PASSWORD_LENGTH


def _b64url_encode(raw: bytes) -> str:
    """
    역할: 토큰 구성요소를 URL-safe base64 문자열로 인코딩
    입력: bytes 원문
    출력: padding 없는 base64url 문자열
    """
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64url_decode(text: str) -> bytes:
    """
    역할: padding 없는 base64url 문자열을 bytes로 디코딩
    입력: base64url 문자열
    출력: bytes 원문
    """
    padded = text + ("=" * (-len(text) % 4))
    return base64.urlsafe_b64decode(padded.encode("ascii"))


def _sign_token(unsigned_token: str) -> str:
    """
    역할: 토큰 header.payload 문자열에 HMAC-SHA256 서명 생성
    입력: header.payload 문자열
    출력: base64url 서명 문자열
    """
    digest = hmac.new(
        TOKEN_SECRET.encode("utf-8"),
        unsigned_token.encode("ascii"),
        hashlib.sha256,
    ).digest()
    return _b64url_encode(digest)


def create_access_token(user_id: int, username: str) -> tuple[str, int]:
    """
    역할: 인증된 사용자에게 발급할 HMAC 서명 access token 생성
    입력: 사용자 id, 사용자 아이디
    출력: (access token 문자열, 만료 epoch 초)
    """
    now = int(time.time())
    expires_at = now + TOKEN_EXPIRES_SECONDS
    header = {
        "typ": "JWT",
        "alg": TOKEN_ALGORITHM,
    }
    payload = {
        "sub": str(user_id),
        "username": str(username),
        "iat": now,
        "exp": expires_at,
    }
    header_text = _b64url_encode(
        json.dumps(header, separators=(",", ":")).encode("utf-8")
    )
    payload_text = _b64url_encode(
        json.dumps(payload, separators=(",", ":")).encode("utf-8")
    )
    unsigned = f"{header_text}.{payload_text}"
    signature = _sign_token(unsigned)
    return f"{unsigned}.{signature}", expires_at


def verify_access_token(token: str) -> dict | None:
    """
    역할: Bearer access token의 서명과 만료 시간을 검증
    입력: access token 문자열
    출력: 검증 성공 시 payload dict, 실패 시 None
    """
    try:
        header_text, payload_text, signature = str(token).split(".", 2)
        unsigned = f"{header_text}.{payload_text}"
        expected_signature = _sign_token(unsigned)
        if not hmac.compare_digest(signature, expected_signature):
            return None

        header = json.loads(_b64url_decode(header_text).decode("utf-8"))
        payload = json.loads(_b64url_decode(payload_text).decode("utf-8"))
    except (
        ValueError,
        TypeError,
        binascii.Error,
        json.JSONDecodeError,
        UnicodeDecodeError,
    ):
        return None

    if header.get("alg") != TOKEN_ALGORITHM:
        return None
    if int(payload.get("exp", 0)) < int(time.time()):
        return None
    if not payload.get("sub") or not payload.get("username"):
        return None
    return payload
