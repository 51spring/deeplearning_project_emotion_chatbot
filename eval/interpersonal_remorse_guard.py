"""
interpersonal_remorse_guard.py
역할: 관계 후회 문맥의 실제 RoBERTa/NLI/CBT 운영 경로 회귀 검증
입력: 운영 체크포인트
출력: 검증 결과 JSON, 실패 시 assertion
실행: C:/Users/WD/anaconda3/envs/dl_study/python.exe eval/interpersonal_remorse_guard.py
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.crisis_handler import should_hard_interrupt  # noqa: E402
from backend.scheduler import ModelScheduler  # noqa: E402


def load_root_case() -> str:
    """
    역할: 운영 DB에서 확인된 root 관계 후회 발화를 id로 읽어 인코딩 손실을 방지
    입력: 없음
    출력: 사용자 발화 원문
    """
    db_path = ROOT_DIR / "backend" / "db" / "emotion_chatbot.db"
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT text FROM utterances WHERE id = 102"
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        raise RuntimeError("검증 대상 root 발화 id=102를 찾을 수 없습니다.")
    return str(row[0])


def main() -> None:
    """
    역할: 관계 후회 발화의 위험 오탐 완화와 직접 위기 보존을 실제 모델로 검증
    입력: 없음
    출력: JSON 요약
    """
    scheduler = ModelScheduler(use_cbt=True)
    try:
        text = load_root_case()
        result = scheduler.run_roberta(text)
        assert result["nli_guard"] == "interpersonal_remorse_nli_cap", result
        assert result["entailment_prob"] <= 0.20, result
        assert result["is_crisis"] is False, result
        assert result["depression_score"] <= 0.60, result
        assert not should_hard_interrupt(
            text,
            result["is_crisis"],
            result["entailment_prob"],
        )

        direct_text = "죽고 싶다는 생각이 계속 들어"
        direct_result = scheduler.run_roberta(direct_text)
        assert should_hard_interrupt(
            direct_text,
            direct_result["is_crisis"],
            direct_result["entailment_prob"],
        )
        print(
            json.dumps(
                {
                    "root_case": {
                        "top_emotion": result["top_emotion"],
                        "depression_score": result["depression_score"],
                        "entailment_prob": result["entailment_prob"],
                        "nli_guard": result["nli_guard"],
                        "hard_interrupt": False,
                    },
                    "direct_crisis_preserved": True,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    finally:
        scheduler._unload_roberta()


if __name__ == "__main__":
    main()
