# GitHub 제출 패키지 최종 점검 (2026-06-12)

## 기준

- 기준 브랜치: `main`
- 동기화 시작 커밋: `9c1c7d22`
- 제출 폴더: `github_submission_20260521/`
- 제출 ZIP: `github_submission_20260521.zip`
- 공식 Python: `C:\Users\WD\anaconda3\envs\dl_study\python.exe`

## 최신화 범위

- 2026-06-10 사용자 기능: 대화 복원, 사용자 피드백, 자동 하루 마감, 주간 리포트, 계정 및 데이터 관리
- 2026-06-11 운영 보강: production secret 검증, rate limit, 계정 잠금, FIFO 추론 큐, 사용자별 lock, DB 복합 유일성, Qwen self-check fail-closed
- 정식 테스트 및 CI: pytest, Jest, React build, GitHub Actions
- 사용자 감정 정정 학습셋 readiness와 관계 후회 NLI 오탐 방어
- 2026-06-12 긍정 발화의 실시간 및 마감·캘린더 웰니스 `양호` 반영
- 최신 제출 보고서 Word/Markdown와 아키텍처 다이어그램

## 검증 결과

- 제출 폴더 Python 구문 검사: 154개 통과
- pytest: 7/7 통과
- 사용자 기능 smoke: 34/34 통과
- 보안·동시성 guard: 통과
- 긍정 발화 및 긍정 day 마감 guard: 통과
- Qwen 품질 guard: 통과
- 프론트 Jest: 2/2 통과
- React production build: 성공, main bundle gzip 136.88 kB
- 루트 대응 파일과 제출 폴더 내용 차이: 0건
- 금지 파일 패턴 유입: 0건

최신 배포 smoke는 긍정 발화의 실시간 웰니스, 하루 마감, 캘린더가 모두 84.62점 `양호`로 반영되는 것을 확인했다.

## 패키지 정책

포함:

- FastAPI 백엔드, 점수 파이프라인, 모델 학습·추론 코드
- React 프론트 소스
- 전처리·평가·회귀 테스트 코드
- GitHub Actions CI
- 실행·배포 문서와 선별된 평가 보고서

제외:

- `data/raw/`, `data/processed/`
- 모델 체크포인트와 가중치
- 실행 DB와 사용자 데이터
- `frontend/node_modules/`, `frontend/build/`
- 로컬 비밀값, 로그, 캐시
- 보호 문서 `상담챗봇_프로젝트_총정리_v14.md`

## 결론

제출 폴더와 ZIP은 2026-06-12 `main`의 최신 소스 및 제출 문서를 반영한다. 실제 시연에는 GitHub에서 제외된 RoBERTa/Qwen 체크포인트, processed runtime 파일, Hugging Face cache를 로컬에 별도로 준비해야 한다.
