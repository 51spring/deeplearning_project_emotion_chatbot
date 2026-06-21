"""
run_posttrain_checks.py
역할: RoBERTa 재학습 완료 후 후속 검증을 한 번에 실행
      1) 감정 분류 평가(val/calib)
      2) NLI 위기 감지 평가
      3) roberta_score P95 측정
      4) FastAPI 스모크 시나리오 테스트(mock 스케줄러 사용)
      5) 실제 모델 기반 RoBERTa → Qwen 순차 통합 테스트
실행: C:/Users/WD/anaconda3/envs/dl_study/python.exe eval/run_posttrain_checks.py
"""

import json
import os
import sys
from typing import Any

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# backend.main import 시 init_db()가 실행되므로,
# 실제 앱 DB(backend/db/emotion_chatbot.db)를 건드리지 않도록
# 사전에 임시 SQLite 경로를 환경변수로 주입한다.
_SMOKE_DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "smoke_test_runtime.db",
)
os.environ.setdefault("EMOTION_CHATBOT_DB_PATH", _SMOKE_DB_PATH)

from backend.main import app
import backend.main as backend_main
from backend.db.models import Base
from eval.eval_crisis import evaluate_nli
from eval.eval_roberta import evaluate as evaluate_roberta
from models.roberta.train_roberta import MODEL_NAME, NUM_EMOTION_CLS, NUM_NLI_CLS, DEVICE
from pipeline.roberta_score import (
    CKPT_DIR,
    CRISIS_THRESHOLD,
    measure_score_p95,
    load_roberta_model,
    load_temperature,
    load_emotion_vector_T,
    load_emotion_logit_bias,
    load_runtime_config,
)


P95_RESULT_PATH = os.path.join(CKPT_DIR, "score_norm_result.json")


def run_p95_measurement(sample_size: int | None = None) -> dict[str, float]:
    """
    역할: 현재 체크포인트 기준 raw depression score의 95퍼센타일 측정
    입력: 샘플 수 (None이면 emotion_train.csv 전체 데이터 사용 — 양봉 분포에서 표본 변동을 제거)
    출력: {"p95": float, "sample_size": int, "T_emotion": float}
    """
    import pandas as pd
    from pipeline.roberta_score import BASE_DIR as _BASE_DIR

    model, tokenizer, device = load_roberta_model("roberta_final.pt")
    T_emotion, _ = load_temperature()
    vector_T_emotion = load_emotion_vector_T()
    emotion_logit_bias = load_emotion_logit_bias()
    p95 = measure_score_p95(
        model=model,
        tokenizer=tokenizer,
        device=device,
        T_emotion=T_emotion,
        sample_size=sample_size,
        vector_T_emotion=vector_T_emotion,
        emotion_logit_bias=emotion_logit_bias,
    )

    if sample_size is None:
        train_csv = os.path.join(_BASE_DIR, "data", "processed", "emotion_train.csv")
        actual_size = int(len(pd.read_csv(train_csv)))
    else:
        actual_size = int(sample_size)

    result = {
        "p95": round(float(p95), 6),
        "sample_size": actual_size,
        "T_emotion": round(float(T_emotion), 6),
        "vector_T_emotion": (
            [round(float(t), 6) for t in vector_T_emotion]
            if vector_T_emotion is not None else None
        ),
        "emotion_logit_bias": (
            [round(float(b), 6) for b in emotion_logit_bias]
            if emotion_logit_bias is not None else None
        ),
        "calibration_method": (
            "vector_scaling_plus_logit_bias"
            if vector_T_emotion is not None and emotion_logit_bias is not None
            else "vector_scaling" if vector_T_emotion is not None
            else "single_T_plus_logit_bias" if emotion_logit_bias is not None
            else "single_T"
        ),
    }
    with open(P95_RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result


def build_smoke_scenarios() -> list[dict[str, Any]]:
    """
    역할: 백엔드 API 스모크 테스트용 대표 시나리오 정의
    입력: 없음
    출력: 시나리오 dict 리스트
    """
    return [
        {
            "name": "일반 대화",
            "username": "smoke_general_user",
            "text": "오늘은 평소보다 좀 피곤했지만 그래도 버틸 만했어.",
            "roberta_out": {
                "roberta_score": 0.34,
                "cbt_score": 0.18,
                "depression_score": 0.34,
                "top_emotion": "중립",
                "entailment_prob": 0.08,
                "is_crisis": False,
            },
            "qwen_out": {
                "response": "오늘 피곤함이 있었군요. 그래도 버틸 만했다고 느낀 점이 중요해 보여요.",
                "has_crisis_tag": False,
            },
            "expect_crisis": False,
        },
        {
            "name": "NLI 후보 일반 대화",
            "username": "smoke_nli_candidate_general_user",
            "text": "오늘은 산책도 하고 저녁도 잘 먹어서 기분이 꽤 괜찮았어.",
            "roberta_out": {
                "roberta_score": 0.12,
                "cbt_score": 0.42,
                "depression_score": 0.42,
                "top_emotion": "행복",
                "entailment_prob": 0.40,
                "is_crisis": True,
            },
            "qwen_out": {
                "response": "일상 속에서 산책과 식사가 조금 힘이 되어준 하루였군요.",
                "has_crisis_tag": False,
            },
            "expect_crisis": False,
        },
        {
            "name": "소프트 위기",
            "username": "smoke_soft_user",
            "text": "요즘 너무 무기력해서 다 포기하고 싶다는 생각이 들어.",
            "roberta_out": {
                "roberta_score": 0.76,
                "cbt_score": 0.69,
                "depression_score": 0.76,
                "top_emotion": "슬픔",
                "entailment_prob": 0.41,
                "is_crisis": False,
            },
            "qwen_out": {
                "response": "[CRISIS] 많이 지쳐 계신 것 같아요. 지금 혼자 버티기 어렵다면 1393 도움을 받아보세요.",
                "has_crisis_tag": True,
            },
            "expect_crisis": True,
        },
        {
            "name": "직접 표현 하드 위기",
            "username": "smoke_direct_hard_user",
            "text": "죽고 싶어. 오늘 그 생각이 멈추질 않아.",
            "roberta_out": {
                "roberta_score": 0.86,
                "cbt_score": 0.77,
                "depression_score": 0.86,
                "top_emotion": "슬픔",
                "entailment_prob": 0.16,
                "is_crisis": False,
            },
            "qwen_out": {
                "response": "이 응답은 직접 표현 하드 인터럽트에서는 사용되지 않아야 합니다.",
                "has_crisis_tag": False,
            },
            "expect_crisis": True,
        },
        {
            "name": "하드 위기",
            "username": "smoke_hard_user",
            "text": "오늘 밤 끝내고 싶어. 더는 못 버티겠어.",
            "roberta_out": {
                "roberta_score": 0.98,
                "cbt_score": 0.82,
                "depression_score": 0.98,
                "top_emotion": "공포",
                "entailment_prob": 0.93,
                "is_crisis": True,
            },
            "qwen_out": {
                "response": "이 응답은 하드 인터럽트에서는 사용되지 않아야 합니다.",
                "has_crisis_tag": False,
            },
            "expect_crisis": True,
        },
    ]


def run_backend_smoke_test() -> list[dict[str, Any]]:
    """
    역할: mock 스케줄러로 `/chat`, `/day/close`, `/calendar` 흐름 스모크 테스트
    입력: 없음
    출력: 시나리오별 결과 리스트
    """
    original_run_roberta = backend_main.scheduler.run_roberta
    original_run_qwen = backend_main.scheduler.run_qwen
    original_unload = backend_main.scheduler._unload_roberta
    original_session_local = backend_main.SessionLocal
    original_overrides = dict(app.dependency_overrides)
    original_pipelines = backend_main._pipelines.copy()
    temp_engine = None
    # 모듈 상단에서 주입한 임시 DB 경로를 재사용해 위치 불일치를 방지한다.
    temp_db_path = os.environ.get("EMOTION_CHATBOT_DB_PATH", _SMOKE_DB_PATH)

    scenarios = build_smoke_scenarios()
    results: list[dict[str, Any]] = []

    # backend.main import 시점에 init_db()가 같은 파일로 엔진을 생성해 놓았을 수 있으므로
    # 파일 삭제 전에 기본 엔진 핸들을 먼저 반드시 dispose 한다. (Windows 파일 잠금 회피)
    default_engine = getattr(backend_main, "engine", None)
    if default_engine is not None:
        default_engine.dispose()

    try:
        if os.path.exists(temp_db_path):
            os.remove(temp_db_path)

        temp_engine = create_engine(
            f"sqlite:///{temp_db_path}",
            connect_args={"check_same_thread": False},
        )
        Base.metadata.create_all(temp_engine)
        temp_session_local = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=temp_engine,
        )

        backend_main.SessionLocal = temp_session_local
        backend_main._pipelines.clear()

        def override_get_db():
            """
            역할: 스모크 테스트에서 임시 SQLite 세션 제공
            입력: 없음
            출력: DB 세션 generator
            """
            db = temp_session_local()
            try:
                yield db
            finally:
                db.close()

        app.dependency_overrides[backend_main.get_db] = override_get_db

        client = TestClient(app)

        def auth_headers_for_user(username: str) -> dict[str, str]:
            """
            역할: 스모크 테스트 사용자 계정을 이메일 포함으로 만들고 Bearer 인증 헤더를 반환
            입력: 사용자 이름
            출력: Authorization 헤더 dict
            """
            password = "pass1234"
            response = client.post(
                "/auth/register",
                json={
                    "username": username,
                    "nickname": username,
                    "email": f"{username}@example.local",
                    "password": password,
                },
            )
            if response.status_code == 409:
                response = client.post(
                    "/auth/login",
                    json={"username": username, "password": password},
                )
            response.raise_for_status()
            return {"Authorization": f"Bearer {response.json()['access_token']}"}

        for scenario in scenarios:
            backend_main.scheduler.run_roberta = lambda text, payload=scenario["roberta_out"]: payload
            backend_main.scheduler.run_qwen = (
                lambda text, history=None, utterance_info=None, payload=scenario["qwen_out"]: payload
            )
            backend_main.scheduler._unload_roberta = lambda: None
            headers = auth_headers_for_user(scenario["username"])

            chat_resp = client.post(
                "/chat",
                json={"username": scenario["username"], "text": scenario["text"]},
                headers=headers,
            )
            chat_resp.raise_for_status()
            chat_data = chat_resp.json()

            close_resp = client.post(
                "/day/close",
                json={"username": scenario["username"]},
                headers=headers,
            )
            close_resp.raise_for_status()
            close_data = close_resp.json()

            calendar_resp = client.get(f"/calendar/{scenario['username']}", headers=headers)
            calendar_resp.raise_for_status()
            calendar_data = calendar_resp.json()

            results.append(
                {
                    "name": scenario["name"],
                    "chat_status": chat_resp.status_code,
                    "close_status": close_resp.status_code,
                    "calendar_status": calendar_resp.status_code,
                    "is_crisis": bool(chat_data["is_crisis"]),
                    "expected_crisis": bool(scenario["expect_crisis"]),
                    "calendar_items": len(calendar_data),
                    "label": close_data["label"],
                }
            )
    finally:
        backend_main.scheduler.run_roberta = original_run_roberta
        backend_main.scheduler.run_qwen = original_run_qwen
        backend_main.scheduler._unload_roberta = original_unload
        backend_main.SessionLocal = original_session_local
        backend_main._pipelines.clear()
        backend_main._pipelines.update(original_pipelines)
        app.dependency_overrides = original_overrides
        if temp_engine is not None:
            temp_engine.dispose()
        if default_engine is not None:
            default_engine.dispose()
        if os.path.exists(temp_db_path):
            try:
                os.remove(temp_db_path)
            except PermissionError:
                # Windows에서 연결이 늦게 해제되는 경우 다음 실행 때 제거되므로 경고만 남긴다.
                print(f"[경고] 임시 DB 삭제 실패(다음 실행 시 재시도): {temp_db_path}")

    return results


def run_real_model_integration_test() -> dict[str, Any]:
    """
    역할: 실제 체크포인트 기준 RoBERTa → Qwen → RoBERTa 재호출 흐름을 검증
    입력: 없음
    출력: 통합 테스트 결과 dict
    """
    from backend.scheduler import ModelScheduler

    scheduler = ModelScheduler(use_cbt=True)
    first_text = "요즘 잠도 잘 안 오고 계속 무기력해서 하루가 너무 길게 느껴져."
    second_text = "그래도 오늘은 조금이라도 버텨보려고 했어."

    try:
        roberta_result = scheduler.run_roberta(first_text)
        qwen_result = scheduler.run_qwen(first_text, history=[], utterance_info=roberta_result)
        followup_result = scheduler.run_roberta(second_text)

        return {
            "first_top_emotion": roberta_result["top_emotion"],
            "first_cbt_score": float(roberta_result["cbt_score"]),
            "first_depression_score": float(roberta_result["depression_score"]),
            "qwen_has_response": bool(qwen_result["response"].strip()),
            "qwen_has_crisis_tag": bool(qwen_result["has_crisis_tag"]),
            "second_top_emotion": followup_result["top_emotion"],
            "second_cbt_score": float(followup_result["cbt_score"]),
            "roundtrip_ok": True,
        }
    finally:
        scheduler._unload_roberta()
        scheduler._unload_qwen()


def main():
    """
    역할: 재학습 후 확인해야 할 핵심 검증 단계를 순차 실행
    입력: 없음
    출력: 콘솔 요약 + P95 결과 파일 저장
    """
    print("=" * 60)
    print("RoBERTa 후속 검증 시작")
    print("=" * 60)
    print(f"MODEL_NAME={MODEL_NAME}")
    print(f"DEVICE={DEVICE}")
    print(f"NLI/Emotion cls={NUM_NLI_CLS}/{NUM_EMOTION_CLS}")
    print(f"CRISIS_THRESHOLD={CRISIS_THRESHOLD:.4f}")
    print(f"runtime_config={load_runtime_config()}")

    val_result = evaluate_roberta("val")
    calib_result = evaluate_roberta("calib")
    crisis_result = evaluate_nli()
    p95_result = run_p95_measurement(sample_size=None)
    smoke_results = run_backend_smoke_test()
    real_model_result = run_real_model_integration_test()

    print("\n" + "=" * 60)
    print("후속 검증 요약")
    print("=" * 60)
    print(f"val Macro F1={val_result['macro_f1']:.4f}, ECE={val_result['ece']:.4f}")
    print(f"calib Macro F1={calib_result['macro_f1']:.4f}, ECE={calib_result['ece']:.4f}")
    print(
        f"NLI samples={crisis_result['n_samples']}, "
        f"false_pos={crisis_result['false_pos']}, false_neg={crisis_result['false_neg']}"
    )
    print(
        f"P95={p95_result['p95']:.6f} "
        f"(sample_size={p95_result['sample_size']}, T_emotion={p95_result['T_emotion']:.6f})"
    )
    print("스모크 테스트 결과:")
    for row in smoke_results:
        print(
            f"  - {row['name']}: chat={row['chat_status']}, close={row['close_status']}, "
            f"calendar={row['calendar_status']}, crisis={row['is_crisis']} "
            f"(expected={row['expected_crisis']}), calendar_items={row['calendar_items']}, "
            f"label={row['label']}"
        )
    print("실제 모델 통합 테스트:")
    print(
        f"  - roundtrip_ok={real_model_result['roundtrip_ok']}, "
        f"first_top_emotion={real_model_result['first_top_emotion']}, "
        f"first_cbt_score={real_model_result['first_cbt_score']:.4f}, "
        f"qwen_has_response={real_model_result['qwen_has_response']}, "
        f"qwen_has_crisis_tag={real_model_result['qwen_has_crisis_tag']}, "
        f"second_top_emotion={real_model_result['second_top_emotion']}, "
        f"second_cbt_score={real_model_result['second_cbt_score']:.4f}"
    )
    print(f"\nP95 저장 파일: {P95_RESULT_PATH}")


if __name__ == "__main__":
    main()
