# Legacy DB 정리 계획

작성일: 2026-05-28
기준 리포트: `eval/report/security_privacy_audit_20260528_000501.json`
중요: 이 문서는 정리 방침이며, 이번 작업에서 운영 DB 데이터를 삭제하거나 수정하지 않는다.

## 1. 현재 확인된 상태

`eval/security_privacy_audit.py` 최신 실행 기준 원문 없이 확인한 상태는 다음과 같다.

| 항목 | 값 |
|---|---:|
| users total | 14 |
| password_hash 없는 legacy 사용자 | 3 |
| password hash scheme 이상 | 0 |
| sessions | 13 |
| utterances | 63 |
| daily_summaries | 8 |
| model_audit_events | 32 |
| sessions orphan | 0 |
| utterances orphan | 0 |
| daily_summaries orphan | 0 |
| crisis_events orphan | 0 |
| model_audit_events orphan | 0 |
| audit payload rows | 32 |
| `qwen_raw_response` key 포함 audit payload | 32 |
| git 추적 민감 파일 | 0 |
| 제출 폴더 민감 파일 | 0 |

해석:

- legacy 사용자 3건은 기존 username-only 흐름에서 온 계정으로 보인다.
- 과거 audit 32건에는 raw Qwen 응답 key가 남아 있다.
- 현재 production/prod 저장 경로에서는 새 raw Qwen 응답을 기본 저장하지 않는다.
- 기존 DB 파일은 GitHub 제출/ZIP 대상이 아니며, 이번 작업에서는 데이터 원본을 수정하지 않는다.

## 2. 수업 데모 기준 방침

수업 시연에서는 기존 DB 정리보다 재현성과 안전한 제출 패키지 분리가 우선이다.

| 항목 | 방침 |
|---|---|
| legacy 사용자 3건 | 그대로 둔다. self-service claim은 코드에서 기본 차단되어 있다. |
| 과거 raw Qwen audit 32건 | 그대로 둔다. DB는 제출/공유 대상이 아니며 새 production 저장은 차단되어 있다. |
| 데모 계정 | password hash가 있는 새 계정 또는 관리자 계정만 사용한다. |
| 제출 패키지 | DB 파일, raw/processed 데이터, 체크포인트, cache를 계속 제외한다. |

## 3. 공개 운영 전 정리 절차

공개 운영 또는 장기 파일 보관 전에는 아래 순서로 별도 승인 후 처리한다.

1. DB 백업을 만든다.
   - 백업은 Git 밖 로컬 경로에 두고, ZIP/GitHub 제출에 포함하지 않는다.
   - 백업 파일명에는 날짜와 목적을 포함한다.
2. 원문 미노출 audit을 먼저 실행한다.
   - `C:\Users\WD\anaconda3\envs\dl_study\python.exe eval\security_privacy_audit.py`
3. legacy 사용자 처리 방식을 결정한다.
   - 사용하지 않는 데모 계정이면 삭제한다.
   - 보존이 필요한 계정이면 관리자가 직접 비밀번호를 설정한다.
   - `EMOTION_CHATBOT_ALLOW_LEGACY_ACCOUNT_CLAIM=1`은 로컬 마이그레이션 순간에만 켠다.
4. 과거 audit payload에서 `qwen_raw_response` key를 제거한다.
   - 모델 판단 메타데이터는 유지하고 raw 응답 key만 제거한다.
   - 처리 전후 row count와 key count만 기록한다.
5. 정리 후 audit을 다시 실행한다.
   - 목표는 `password_hash_null=0` 또는 보존 사유 문서화, `audit_payload_rows_with_qwen_raw_response_key=0`, orphan count 0 유지다.

## 4. 금지 사항

- `data/raw/` 수정 금지.
- 승인 없는 운영 DB 삭제/수정 금지.
- DB 백업, `.db`, `.env`, 체크포인트, cache를 GitHub 제출 폴더 또는 ZIP에 포함 금지.

## 5. 다음 적용 판단

현재는 수업 데모 제출 직전이므로 DB 정리 적용보다 문서화와 제출 패키지 분리가 적절하다. 실제 삭제/마이그레이션은 발표 후 별도 작업으로 진행한다.
