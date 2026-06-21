"""
build_user_emotion_correction_trainset.py
역할: 운영 DB의 사용자 감정 정정을 semantic emotion 재학습용 gold 행으로 추출
입력: SQLite DB, 기존 semantic train CSV, 최소 정정 수
출력: gitignore된 정정/통합 CSV와 개인정보 없는 readiness JSON
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = ROOT_DIR / "backend" / "db" / "emotion_chatbot.db"
DEFAULT_BASE_TRAIN = ROOT_DIR / "data" / "processed" / "semantic_emotion_train.csv"
DEFAULT_CORRECTIONS_OUT = (
    ROOT_DIR / "data" / "processed" / "user_emotion_corrections.csv"
)
DEFAULT_AUGMENTED_OUT = (
    ROOT_DIR / "data" / "processed" / "semantic_emotion_train_user_corrections.csv"
)
DEFAULT_REPORT_OUT = (
    ROOT_DIR / "eval" / "report" / "user_emotion_correction_readiness.json"
)
EMOTIONS = ["중립", "행복", "슬픔", "분노", "공포", "혐오", "놀람"]
EMOTION_TO_LABEL = {emotion: index for index, emotion in enumerate(EMOTIONS)}
COMPACT_PATTERN = re.compile(r"[\W_]+", re.UNICODE)


def compact_text_key(value: Any) -> str:
    """
    역할: 사용자 정정과 기존 학습 행의 텍스트 중복 비교 키 생성
    입력: 임의 텍스트 값
    출력: 공백/문장부호를 제거한 소문자 키
    """
    return COMPACT_PATTERN.sub("", str(value or "").casefold())


def load_corrections(db_path: Path) -> list[dict[str, Any]]:
    """
    역할: 사용자 감정 정정과 대상 발화 텍스트를 DB에서 최신순으로 조회
    입력: SQLite DB 경로
    출력: 유효한 감정 정정 행 목록
    """
    if not db_path.exists():
        raise FileNotFoundError(f"운영 DB가 없습니다: {db_path}")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT f.id, f.feedback_value, f.model_emotion, u.text
            FROM utterance_feedback AS f
            JOIN utterances AS u ON u.id = f.utterance_id
            WHERE f.feedback_kind = 'emotion_correction'
            ORDER BY COALESCE(f.updated_at, f.created_at) DESC, f.id DESC
            """
        ).fetchall()
    finally:
        conn.close()

    latest_by_text: dict[str, dict[str, Any]] = {}
    for row in rows:
        emotion = str(row["feedback_value"] or "").strip()
        text_value = str(row["text"] or "").strip()
        key = compact_text_key(text_value)
        if not key or emotion not in EMOTION_TO_LABEL or key in latest_by_text:
            continue
        latest_by_text[key] = {
            "text": text_value,
            "emotion": emotion,
            "model_emotion": str(row["model_emotion"] or "").strip(),
        }
    return list(latest_by_text.values())


def build_correction_frame(corrections: list[dict[str, Any]]) -> pd.DataFrame:
    """
    역할: 감정 정정을 semantic emotion 학습 CSV 스키마로 변환
    입력: DB에서 읽은 정정 행 목록
    출력: semantic head gold DataFrame
    """
    rows = []
    for item in corrections:
        emotion = item["emotion"]
        rows.append(
            {
                "text": item["text"],
                "emotion": emotion,
                "label": EMOTION_TO_LABEL[emotion],
                "primary_emotion": emotion,
                # 사용자 정정은 감정 gold이며 distress 정답은 아니므로 loss에서 제외한다.
                "distress_level": 0,
                "situation_tag": "user_correction",
                "label_source": "user_emotion_correction",
                "weak_label_confidence": 1.0,
                "sample_weight": 1.0,
                "distress_sample_weight": 0.0,
                "is_weak_label": False,
                "source_dataset": "runtime_feedback",
                "source_detail": "emotion_correction",
                "source_split": "train",
            }
        )
    return pd.DataFrame(rows)


def merge_with_base_train(
    base_train_path: Path,
    correction_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    """
    역할: 정정 텍스트와 충돌하는 기존 행을 제거하고 사용자 gold 행을 병합
    입력: 기존 train CSV 경로, 정정 DataFrame
    출력: 통합 DataFrame과 제거한 기존 충돌 행 수
    """
    base = pd.read_csv(base_train_path, encoding="utf-8-sig")
    correction_keys = {
        compact_text_key(text_value)
        for text_value in correction_frame["text"].astype(str)
    }
    base_keys = base["text"].map(compact_text_key)
    keep_mask = ~base_keys.isin(correction_keys)
    removed_conflicts = int((~keep_mask).sum())
    base = base.loc[keep_mask].copy()
    if "distress_sample_weight" not in base.columns:
        base["distress_sample_weight"] = base["sample_weight"].astype(float)

    all_columns = list(base.columns)
    for column in correction_frame.columns:
        if column not in all_columns:
            all_columns.append(column)
    merged = pd.concat(
        [
            base.reindex(columns=all_columns),
            correction_frame.reindex(columns=all_columns),
        ],
        ignore_index=True,
    )
    return merged, removed_conflicts


def write_readiness_report(
    output_path: Path,
    *,
    corrections: list[dict[str, Any]],
    minimum_rows: int,
    ready: bool,
    augmented_rows: int | None,
    removed_conflicts: int,
) -> None:
    """
    역할: 발화 원문과 사용자 식별자를 제외한 재학습 준비 상태 저장
    입력: 출력 경로, 정정 통계, 최소 표본 수, 통합 결과 통계
    출력: 없음
    """
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "correction_rows": len(corrections),
        "minimum_rows": minimum_rows,
        "ready_for_retraining": ready,
        "emotion_distribution": dict(
            sorted(Counter(item["emotion"] for item in corrections).items())
        ),
        "model_mismatch_rows": sum(
            1
            for item in corrections
            if item["model_emotion"] and item["model_emotion"] != item["emotion"]
        ),
        "removed_base_conflicts": removed_conflicts,
        "augmented_train_rows": augmented_rows,
        "privacy": "발화 원문과 사용자 식별자는 readiness JSON에 저장하지 않음",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    """
    역할: 사용자 정정 학습셋 생성 CLI 인자 파싱
    입력: 명령행 인자
    출력: argparse Namespace
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--base-train", type=Path, default=DEFAULT_BASE_TRAIN)
    parser.add_argument("--corrections-out", type=Path, default=DEFAULT_CORRECTIONS_OUT)
    parser.add_argument("--augmented-out", type=Path, default=DEFAULT_AUGMENTED_OUT)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT_OUT)
    parser.add_argument(
        "--minimum-rows",
        type=int,
        default=30,
        help="통합 학습 CSV를 만들 최소 사용자 정정 수",
    )
    return parser.parse_args()


def main() -> None:
    """
    역할: 정정 gold CSV를 저장하고 최소 표본 충족 시 기존 train과 통합
    입력: CLI 인자
    출력: 파일과 readiness 요약
    """
    args = parse_args()
    corrections = load_corrections(args.db_path)
    correction_frame = build_correction_frame(corrections)
    args.corrections_out.parent.mkdir(parents=True, exist_ok=True)
    correction_frame.to_csv(args.corrections_out, index=False, encoding="utf-8-sig")

    ready = len(corrections) >= max(1, args.minimum_rows)
    augmented_rows = None
    removed_conflicts = 0
    if ready:
        merged, removed_conflicts = merge_with_base_train(
            args.base_train,
            correction_frame,
        )
        args.augmented_out.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(args.augmented_out, index=False, encoding="utf-8-sig")
        augmented_rows = len(merged)

    write_readiness_report(
        args.report_out,
        corrections=corrections,
        minimum_rows=max(1, args.minimum_rows),
        ready=ready,
        augmented_rows=augmented_rows,
        removed_conflicts=removed_conflicts,
    )
    print(
        json.dumps(
            {
                "correction_rows": len(corrections),
                "ready_for_retraining": ready,
                "corrections_out": str(args.corrections_out),
                "augmented_out": str(args.augmented_out) if ready else None,
                "report_out": str(args.report_out),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
