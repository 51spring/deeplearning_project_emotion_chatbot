# syntax=docker/dockerfile:1

# 수업용 Docker 배포 산출물
# - FastAPI 모델 서빙 API와 React production build를 한 컨테이너에서 제공한다.
# - 대용량 모델 체크포인트와 운영 DB는 이미지에 넣지 않고 volume mount로 주입한다.

FROM node:20-bookworm-slim AS frontend-build

WORKDIR /app/frontend

COPY frontend/package*.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build


FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Asia/Seoul \
    EMOTION_CHATBOT_ENV=production \
    EMOTION_CHATBOT_TIMEZONE=Asia/Seoul \
    EMOTION_CHATBOT_HOST=0.0.0.0 \
    EMOTION_CHATBOT_PORT=8000

WORKDIR /app

# Python 3.10 기준 런타임을 준비한다.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        python3.10 \
        python3-pip \
        python3.10-venv \
        tzdata \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python3.10 -m pip install --upgrade pip \
    && python3.10 -m pip install -r requirements.txt

# 소스 코드만 이미지에 포함한다. 체크포인트/DB/원본 데이터는 .dockerignore로 제외한다.
COPY backend/ backend/
COPY pipeline/ pipeline/
COPY models/ models/
COPY data/ data/
COPY eval/ eval/
COPY frontend/package*.json frontend/
COPY --from=frontend-build /app/frontend/build frontend/build

EXPOSE 8000

CMD ["sh", "-c", "python3.10 -m uvicorn backend.main:app --host ${EMOTION_CHATBOT_HOST:-0.0.0.0} --port ${EMOTION_CHATBOT_PORT:-8000}"]
