# Deep Learning Emotion Chatbot

딥러닝실습 수업 기말 프로젝트로 개발한 한국어 감정 모니터링 및 상담형 챗봇입니다.  
일상 대화를 바탕으로 감정, 웰니스 상태, 우울 경향 신호를 참고용으로 보여주고, 위기 표현이 감지되면 안전 안내 흐름을 제공합니다.

> 본 프로젝트는 의료 진단 목적이 아닌 수업 프로젝트 및 정서 모니터링 보조 도구입니다.

## 주요 기능

- 한국어 대화 기반 7감정 분류
- 웰니스 상태 및 우울 경향 모니터링
- 위기 표현 감지와 안전 응답
- 로그인, 회원가입, 비밀번호 변경/재설정
- 날짜별 대화 복원, 캘린더 기록, 수동 감정 기록
- 주간 리포트와 요일별 감정 분포 확인
- FastAPI 백엔드와 React 프론트엔드 기반 웹 UI

## 기술 스택

- Python, PyTorch, Transformers
- KLUE-RoBERTa 기반 감정/NLI/CBT 분석
- Qwen2.5-3B-Instruct + LoRA 기반 상담 응답 생성
- FastAPI, SQLite
- React, Chart.js
- Docker 배포 설정

## 저장소에 포함하지 않은 항목

GitHub 제출용 저장소에는 다음 항목을 포함하지 않습니다.

- 원천 데이터셋 전체
- 대용량 모델 체크포인트 파일(`.pt`, `.safetensors` 등)
- 실제 운영 DB
- 실제 비밀번호, 토큰, `docker.env`
- 보고서와 발표자료 원본 파일

필요한 원천 데이터와 모델 파일은 각 제공처의 이용 조건에 따라 별도로 준비해야 합니다.

## 실행 준비

Python 환경은 수업 개발 환경 기준으로 `dl_study` conda 환경을 사용했습니다.

```bat
conda activate dl_study
pip install -r requirements.txt
```

프론트엔드 의존성은 별도로 설치합니다.

```bat
npm --prefix frontend install
npm --prefix frontend run build
```

운영 실행 전에는 `docker.env.example` 또는 아래 환경변수를 참고해 실제 값을 설정해야 합니다.

```text
EMOTION_CHATBOT_AUTH_SECRET
EMOTION_CHATBOT_DEVELOPER_PASSWORD
EMOTION_CHATBOT_ROOT_PASSWORD
EMOTION_CHATBOT_TIMEZONE
```

## 실행

백엔드 개발 실행:

```bat
run_backend.bat
```

React production build를 포함한 단일 서버 배포 실행:

```bat
run_deploy.bat
```

## 검증

대표 스모크 테스트:

```bat
C:\Users\WD\anaconda3\envs\dl_study\python.exe eval\feature_additions_smoke.py
```

배포 서버 실행 후 API/UI 흐름 점검:

```bat
C:\Users\WD\anaconda3\envs\dl_study\python.exe eval\deploy_smoke.py --base-url http://127.0.0.1:8000
```

## 모델 파일 위치

대용량 모델 파일은 GitHub에 포함하지 않았습니다. 전체 추론을 실행하려면 학습된 파일을 아래 위치에 배치해야 합니다.

```text
models/roberta/checkpoints/
models/qwen/checkpoints/qwen_lora_daily_style_v2/
```

## 라이선스

코드는 MIT License를 따릅니다.  
단, 데이터셋과 외부 사전학습 모델은 각 원 제공처의 라이선스와 이용 조건을 따릅니다.

