# 보안·개인정보 후속 과제 정리

작성일: 2026-05-28

## 1. 현재 완료된 보강


| 구분 | 현재 상태 |
|---|---|
| 토큰 secret | production/prod에서 `EMOTION_CHATBOT_AUTH_SECRET` 미설정 또는 기본값 사용 시 시작 실패 |
| 비밀번호 저장 | `users.password_hash`에 PBKDF2-SHA256 해시 저장 |
| 비밀번호 길이 | local/demo 4자, production/prod 8자 이상 |
| 관리자 계정 | production/prod에서 developer/root 관리자 비밀번호 환경변수 필수 |
| legacy 계정 claim | passwordless username-only 계정 claim 기본 차단 |
| DB 무결성 | SQLite 연결마다 `PRAGMA foreign_keys=ON` 적용 |
| audit 개인정보 | production/prod에서 Qwen raw response 기본 미저장 |
| 점검 도구 | `eval/security_privacy_audit.py`가 원문 없이 env/DB/민감 파일 상태 출력 |
| 제출 패키지 | DB, 체크포인트, `data/raw/`, `data/processed`, build/cache 제외 |

## 2. 공개 서비스 전 필수 후속 과제


| 우선순위 | 항목 | 이유 |
|---|---|---|
| P0 | HTTPS 고정 | 로그인 토큰과 발화가 네트워크에서 평문으로 오가지 않게 해야 한다. |
| P0 | production env 고정 | `EMOTION_CHATBOT_ENV=production`, 강한 secret, 관리자 비밀번호, CORS origin을 명시해야 한다. |
| P0 | 개인정보 고지/동의 UI | 사용자가 어떤 대화와 점수가 저장되는지 명확히 알아야 한다. |
| P0 | 계정 삭제/데이터 삭제 | 사용자가 자신의 대화·요약·위기·audit 기록 삭제를 요청할 수 있어야 한다. |
| P1 | httpOnly secure cookie | 현재 브라우저 token 저장 방식보다 XSS 노출 위험을 줄인다. |
| P1 | refresh token/revocation | 장기 세션과 강제 로그아웃, 탈취 토큰 무효화가 필요하다. |
| P1 | rate limit | 로그인 brute force와 `/chat` 비용 폭증을 막는다. |
| P1 | 비밀번호 변경/초기화 | 계정 운영에 필요한 기본 흐름이다. |
| P1 | 데이터 내보내기 | 개인정보 이동권/사용자 열람 요구에 대응한다. |
| P1 | 보존 기간 정책 | audit, 대화 원문, 요약, 위기 기록의 보존 기간을 분리해야 한다. |
| P2 | DB 암호화/백업 정책 | SQLite 파일과 백업본 유출 위험을 줄인다. |
| P2 | 중앙 로그/알림 | 장애, 위기 이벤트, 비정상 접근을 운영자가 확인할 수 있어야 한다. |
| P2 | 임상/윤리 검토 | 정신건강 서비스로 오인되지 않도록 범위와 책임을 검증해야 한다. |


## 3. 운영 전 점검 명령

```powershell
& 'C:\Users\WD\anaconda3\envs\dl_study\python.exe' eval\admin_feature_guard.py
& 'C:\Users\WD\anaconda3\envs\dl_study\python.exe' eval\security_privacy_audit.py
git diff --check
```

`security_privacy_audit.py` 출력은 사용자 발화 원문, 비밀번호, 토큰 값을 포함하지 않는다.
