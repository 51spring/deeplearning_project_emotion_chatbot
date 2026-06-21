"""
reclassify_physical_fatigue_utterances.py
역할: 기존 DB에 저장된 근무·생활 신체 피로 발화를 새 physical_exertion_context 기준으로 보정한다.
입력: EMOTION_CHATBOT_DB_PATH 환경변수(없으면 기본 운영 DB), 선택 인자 --dry-run
출력: 보정 대상/적용 건수 콘솔 출력
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from typing import Any


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from pipeline.depression_tendency import compute_depression_tendency
from pipeline.ensemble import ensemble_scores
from pipeline.ewma import daily_to_smoothed, utterance_to_daily
from pipeline.score_policy import SCORE_AFFECTING_UTTERANCE_TYPES
from pipeline.utterance_type import is_physical_exertion_text
from pipeline.wellness_score import compute_wellness


ROBERTA_PHYSICAL_CAP = 0.35
CBT_PHYSICAL_CAP = 0.45


def get_db_path() -> str:
    """
    역할: 백필 대상 SQLite 경로를 결정한다.
    입력: EMOTION_CHATBOT_DB_PATH 환경변수
    출력: SQLite DB 파일 경로
    """
    default_path = os.path.join(BASE_DIR, "backend", "db", "emotion_chatbot.db")
    return os.environ.get("EMOTION_CHATBOT_DB_PATH", default_path)


def _safe_json_loads(raw: str | None) -> dict[str, Any]:
    """
    역할: audit_payload_json을 안전하게 dict로 읽는다.
    입력: JSON 문자열 또는 None
    출력: 파싱된 dict (실패 시 빈 dict)
    """
    if not raw:
        return {}
    try:
        loaded = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _cap_score(value: float | None, cap: float) -> float | None:
    """
    역할: 기존 점수를 상한 이하로 제한한다.
    입력: 기존 점수 또는 None, cap 값
    출력: 보정 점수 또는 None
    """
    if value is None:
        return None
    return min(float(value), cap)


def _build_reclassified_scores(row: sqlite3.Row) -> dict[str, Any] | None:
    """
    역할: 한 발화가 신체 피로 백필 대상이면 새 점수 dict를 만든다.
    입력: utterances/model_audit_events 조인 row
    출력: 보정 dict 또는 None
    """
    text = row["text"] or ""
    if row["is_crisis"] or not is_physical_exertion_text(text):
        return None

    roberta_score = _cap_score(row["roberta_score"], ROBERTA_PHYSICAL_CAP)
    if roberta_score is None:
        roberta_score = ROBERTA_PHYSICAL_CAP
    cbt_score = _cap_score(row["cbt_score"], CBT_PHYSICAL_CAP)
    depression_score = ensemble_scores(roberta_score, cbt_score)["depression_score"]
    tendency = compute_depression_tendency(
        text,
        top_emotion="중립",
        roberta_score=roberta_score,
        cbt_score=cbt_score,
        utterance_type="casual_neutral",
        type_reason="physical_exertion_backfill",
        is_crisis=False,
        entailment_prob=row["entailment_prob"],
    )
    score_changed = not (
        row["top_emotion"] == "중립"
        and abs(float(row["roberta_score"] or 0.0) - roberta_score) < 1e-9
        and (
            (row["cbt_score"] is None and cbt_score is None)
            or (
                row["cbt_score"] is not None
                and cbt_score is not None
                and abs(float(row["cbt_score"]) - cbt_score) < 1e-9
            )
        )
        and abs(float(row["depression_score"] or 0.0) - float(depression_score)) < 1e-9
        and abs(
            float(row["depression_tendency_score"] or 0.0)
            - float(tendency["depression_tendency_score"])
        ) < 1e-9
    )

    # 과거 로그에는 audit row가 없을 수 있다. 점수만 고쳐도 audit이 없으면
    # 재시작 복원/요약 재계산에서 이전 호환 로직(m.id IS NULL)에 의해 다시 반영된다.
    audit_missing = row["audit_id"] is None
    audit_needs_update = (
        row["audit_id"] is not None
        and row["audit_utterance_type"] != "casual_neutral"
    )
    if not score_changed and not audit_missing and not audit_needs_update:
        return None
    return {
        "roberta_score": roberta_score,
        "cbt_score": cbt_score,
        "depression_score": float(depression_score),
        "depression_tendency_score": float(tendency["depression_tendency_score"]),
        "top_emotion": "중립",
        "score_changed": score_changed,
        "audit_missing": audit_missing,
        "audit_needs_update": audit_needs_update,
    }


def _update_audit_payload(raw_payload: str | None, scores: dict[str, Any]) -> str:
    """
    역할: 기존 audit payload를 보존하면서 백필 후 점수/정책 정보를 갱신한다.
    입력: 기존 JSON 문자열, 새 점수 dict
    출력: 갱신된 JSON 문자열
    """
    payload = _safe_json_loads(raw_payload)
    payload.update(
        {
            "top_emotion": scores["top_emotion"],
            "roberta_score": scores["roberta_score"],
            "cbt_score": scores["cbt_score"],
            "depression_score": scores["depression_score"],
            "depression_tendency_score": scores["depression_tendency_score"],
            "score_affects_wellness": False,
            "score_policy": "non_affecting_chat_type",
            "reclassified_by": "physical_exertion_context_backfill_20260510",
        }
    )
    return json.dumps(payload, ensure_ascii=False)


def _insert_backfill_audit_event(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    scores: dict[str, Any],
) -> None:
    """
    역할: audit가 없던 과거 발화에 백필용 최소 감사 이벤트를 추가한다.
    입력: SQLite 연결, 원본 row, 보정 점수 dict
    출력: 없음
    """
    conn.execute(
        """
        INSERT INTO model_audit_events (
            user_id,
            utterance_id,
            hard_crisis,
            final_is_crisis,
            nli_candidate,
            qwen_called,
            qwen_crisis_tag,
            qwen_anchor_replaced,
            cbt_top_category_source,
            utterance_type,
            utterance_type_confidence,
            audit_payload_json
        )
        VALUES (?, ?, 0, 0, 0, NULL, NULL, NULL, ?, ?, ?, ?)
        """,
        (
            int(row["user_id"]),
            int(row["utterance_id"]),
            "physical_exertion_cbt_cap",
            "casual_neutral",
            0.80,
            _update_audit_payload(None, scores),
        ),
    )


def _score_filter_sql() -> str:
    """
    역할: 웰니스 반영 대상 audit type 조건 SQL을 만든다.
    입력: 없음
    출력: SQL IN 절에 넣을 문자열
    """
    quoted = [f"'{item}'" for item in sorted(SCORE_AFFECTING_UTTERANCE_TYPES)]
    return ", ".join(quoted)


def _fetch_daily_history(
    conn: sqlite3.Connection,
    user_id: int,
    date_str: str,
) -> tuple[list[float], list[float], list[float]]:
    """
    역할: 특정 날짜 이전의 daily/wellness/tendency 히스토리를 가져온다.
    입력: SQLite 연결, 사용자 id, 날짜 문자열
    출력: (daily_score 리스트, wellness_score 리스트, tendency_daily 리스트)
    """
    rows = conn.execute(
        """
        SELECT daily_score, wellness_score, depression_tendency_daily
        FROM daily_summaries
        WHERE user_id = ? AND date < ?
        ORDER BY date ASC
        """,
        (user_id, date_str),
    ).fetchall()
    daily_scores = [float(r["daily_score"]) for r in rows]
    wellness_scores = [float(r["wellness_score"]) for r in rows]
    tendency_scores = [
        float(r["depression_tendency_daily"] or 0.0)
        for r in rows
    ]
    return daily_scores, wellness_scores, tendency_scores


def _fetch_affecting_scores(
    conn: sqlite3.Connection,
    session_id: int,
    planned_non_affecting_ids: set[int] | None = None,
) -> tuple[list[float], list[float]]:
    """
    역할: 한 세션에서 웰니스 반영 대상 사용자 발화 점수만 가져온다.
    입력: SQLite 연결, session_id, 이번 백필로 제외될 발화 id 집합
    출력: (depression_score 리스트, depression_tendency_score 리스트)
    """
    score_types = _score_filter_sql()
    rows = conn.execute(
        f"""
        SELECT u.id, u.depression_score, u.depression_tendency_score
        FROM utterances u
        LEFT JOIN model_audit_events m ON m.utterance_id = u.id
        WHERE u.session_id = ?
          AND u.role = 'user'
          AND u.depression_score IS NOT NULL
          AND (
                m.id IS NULL
                OR m.utterance_type IN ({score_types})
                OR m.final_is_crisis = 1
                OR m.hard_crisis = 1
          )
        ORDER BY u.created_at ASC, u.id ASC
        """,
        (session_id,),
    ).fetchall()
    excluded_ids = planned_non_affecting_ids or set()
    rows = [r for r in rows if int(r["id"]) not in excluded_ids]
    distress_scores = [float(r["depression_score"]) for r in rows]
    tendency_scores = [
        float(r["depression_tendency_score"] or 0.0)
        for r in rows
    ]
    return distress_scores, tendency_scores


def _recompute_daily_summary(
    conn: sqlite3.Connection,
    user_id: int,
    session_id: int,
    date_str: str,
    dry_run: bool,
    planned_non_affecting_ids: set[int] | None = None,
) -> bool:
    """
    역할: 백필 영향을 받는 날짜의 daily_summaries 값을 다시 계산한다.
    입력: SQLite 연결, user_id, session_id, 날짜 문자열, dry-run 여부
    출력: 실제 갱신 필요 여부
    """
    summary = conn.execute(
        """
        SELECT id, daily_score, smoothed_score, wellness_score, label,
               depression_tendency_daily, depression_tendency_smoothed
        FROM daily_summaries
        WHERE user_id = ? AND date = ?
        """,
        (user_id, date_str),
    ).fetchone()
    if summary is None:
        return False

    history_daily, history_wellness, history_tendency = _fetch_daily_history(
        conn,
        user_id,
        date_str,
    )
    distress_scores, tendency_scores = _fetch_affecting_scores(
        conn,
        session_id,
        planned_non_affecting_ids=planned_non_affecting_ids,
    )

    daily_score = (
        utterance_to_daily(distress_scores)
        if distress_scores else
        (history_daily[-1] if history_daily else 0.0)
    )
    smoothed_score = daily_to_smoothed(history_daily + [daily_score])[-1]
    wellness = compute_wellness(
        depression_score=smoothed_score,
        history_wellness=history_wellness,
        n_days=len(history_daily) + 1,
    )
    tendency_daily = (
        utterance_to_daily(tendency_scores)
        if tendency_scores else
        (history_tendency[-1] if history_tendency else 0.0)
    )
    tendency_smoothed = daily_to_smoothed(history_tendency + [tendency_daily])[-1]

    changed = (
        abs(float(summary["daily_score"]) - float(daily_score)) > 1e-9
        or abs(float(summary["smoothed_score"]) - float(smoothed_score)) > 1e-9
        or abs(float(summary["wellness_score"]) - float(wellness["wellness_score"])) > 1e-9
        or summary["label"] != wellness["label"]
        or abs(float(summary["depression_tendency_daily"] or 0.0) - float(tendency_daily)) > 1e-9
        or abs(
            float(summary["depression_tendency_smoothed"] or 0.0)
            - float(tendency_smoothed)
        ) > 1e-9
    )
    if not changed:
        return False

    print(
        "[physical-backfill-summary]",
        f"user_id={user_id}",
        f"date={date_str}",
        f"wellness {summary['wellness_score']} -> {wellness['wellness_score']}",
        f"label {summary['label']} -> {wellness['label']}",
    )
    if dry_run:
        return True

    conn.execute(
        """
        UPDATE daily_summaries
        SET daily_score = ?,
            smoothed_score = ?,
            wellness_score = ?,
            label = ?,
            depression_tendency_daily = ?,
            depression_tendency_smoothed = ?
        WHERE id = ?
        """,
        (
            round(float(daily_score), 4),
            round(float(smoothed_score), 4),
            wellness["wellness_score"],
            wellness["label"],
            round(float(tendency_daily), 4),
            round(float(tendency_smoothed), 4),
            int(summary["id"]),
        ),
    )
    return True


def reclassify(db_path: str, dry_run: bool = False) -> int:
    """
    역할: DB의 기존 사용자 발화 중 physical fatigue 대상만 재분류한다.
    입력: DB 경로, dry-run 여부
    출력: 실제 보정된 utterance 수
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT
                u.id AS utterance_id,
                u.session_id AS session_id,
                s.user_id AS user_id,
                s.date AS session_date,
                u.text AS text,
                u.roberta_score AS roberta_score,
                u.cbt_score AS cbt_score,
                u.depression_score AS depression_score,
                u.depression_tendency_score AS depression_tendency_score,
                u.top_emotion AS top_emotion,
                u.entailment_prob AS entailment_prob,
                u.is_crisis AS is_crisis,
                m.id AS audit_id,
                m.utterance_type AS audit_utterance_type,
                m.audit_payload_json AS audit_payload_json
            FROM utterances u
            JOIN sessions s ON s.id = u.session_id
            LEFT JOIN model_audit_events m ON m.utterance_id = u.id
            WHERE u.role = 'user'
            ORDER BY u.id ASC
            """
        ).fetchall()

        changed_ids: set[int] = set()
        audit_repaired_ids: set[int] = set()
        affected_summaries: dict[tuple[int, int, str], set[int]] = {}
        for row in rows:
            scores = _build_reclassified_scores(row)
            if scores is None:
                continue

            utterance_id = int(row["utterance_id"])
            if scores["score_changed"]:
                changed_ids.add(utterance_id)
            if scores["audit_missing"] or scores["audit_needs_update"]:
                audit_repaired_ids.add(utterance_id)
            summary_key = (
                int(row["user_id"]),
                int(row["session_id"]),
                str(row["session_date"]),
            )
            affected_summaries.setdefault(summary_key, set()).add(utterance_id)
            print(
                "[physical-backfill]",
                f"id={utterance_id}",
                f"text={row['text']}",
                (
                    f"depression {row['depression_score']} -> {scores['depression_score']:.4f}"
                    if scores["score_changed"]
                    else "score already reclassified"
                ),
                (
                    "audit insert"
                    if scores["audit_missing"]
                    else ("audit update" if scores["audit_needs_update"] else "audit ok")
                ),
            )
            if dry_run:
                continue

            if scores["score_changed"]:
                conn.execute(
                    """
                    UPDATE utterances
                    SET roberta_score = ?,
                        cbt_score = ?,
                        cbt_top_category = NULL,
                        depression_score = ?,
                        depression_tendency_score = ?,
                        top_emotion = ?
                    WHERE id = ?
                    """,
                    (
                        scores["roberta_score"],
                        scores["cbt_score"],
                        scores["depression_score"],
                        scores["depression_tendency_score"],
                        scores["top_emotion"],
                        utterance_id,
                    ),
                )

            if row["audit_id"] is not None:
                conn.execute(
                    """
                    UPDATE model_audit_events
                    SET cbt_top_category_source = 'physical_exertion_cbt_cap',
                        utterance_type = 'casual_neutral',
                        utterance_type_confidence = 0.80,
                        audit_payload_json = ?
                    WHERE id = ?
                    """,
                    (
                        _update_audit_payload(row["audit_payload_json"], scores),
                        int(row["audit_id"]),
                    ),
                )
            else:
                _insert_backfill_audit_event(conn, row, scores)

        summary_updates = 0
        for (user_id, session_id, date_str), utterance_ids in sorted(
            affected_summaries.items()
        ):
            if _recompute_daily_summary(
                conn,
                user_id=user_id,
                session_id=session_id,
                date_str=date_str,
                dry_run=dry_run,
                planned_non_affecting_ids=utterance_ids,
            ):
                summary_updates += 1

        if not dry_run:
            conn.commit()
        if summary_updates:
            print(f"[physical-backfill] daily_summaries updates: {summary_updates}")
        if audit_repaired_ids:
            print(f"[physical-backfill] audit repairs: {len(audit_repaired_ids)}")
        return len(changed_ids | audit_repaired_ids)
    finally:
        conn.close()


def main() -> int:
    """
    역할: CLI 진입점으로 백필을 실행한다.
    입력: --dry-run 선택 인자
    출력: 프로세스 종료 코드
    """
    db_path = get_db_path()
    dry_run = "--dry-run" in sys.argv
    if not os.path.exists(db_path):
        print(f"[physical-backfill] DB 파일 없음: {db_path}")
        return 0
    changed = reclassify(db_path, dry_run=dry_run)
    mode = "dry-run" if dry_run else "applied"
    print(f"[physical-backfill] {mode}: {changed} utterance(s), db={db_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
