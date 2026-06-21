# -*- coding: utf-8 -*-
"""
positive_affect_routing_guard.py
역할: 명시 긍정/애정 발화("길고양이가 귀여웠어" 류)가 utterance-type head의
      routine_discomfort/emotional_distress 오분류에도 불구하고 positive_share 저영향으로
      라우팅되는지 룰 계층만으로 검증한다. (RoBERTa 체크포인트 불필요 — 순수 규칙 함수만 사용)
입력: 없음 (내부 고정 케이스)
출력: 콘솔 PASS/FAIL 리포트, 실패 시 종료코드 1
"""

import os
import sys

# 워크트리/메인 repo 루트를 import 경로에 추가 (eval/ 의 부모 = 프로젝트 루트)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from pipeline.utterance_type import is_positive_affect_text, UTTERANCE_TYPES
from pipeline.roberta_score import (
    apply_utterance_type_adjustment,
    POSITIVE_AFFECT_SCORE_CAP,
)
from pipeline.ensemble import ensemble_scores
from pipeline.score_policy import compute_wellness_contribution
from pipeline.wellness_score import depression_to_wellness

# head가 오분류로 내놓는 발화타입(softmax conf 포함)을 흉내 낸 입력
HEAD_ROUTINE = {
    "utterance_type": UTTERANCE_TYPES["ROUTINE_DISCOMFORT"],
    "type_confidence": 0.595,
    "type_reason": "roberta_utterance_intent_head",
}
HEAD_DISTRESS = {
    "utterance_type": UTTERANCE_TYPES["EMOTIONAL_DISTRESS"],
    "type_confidence": 0.61,
    "type_reason": "roberta_utterance_intent_head",
}

_failures: list[str] = []


def _check(name: str, cond: bool, detail: str = "") -> None:
    """역할: 단일 단언 검사 / 입력: 케이스명, 조건, 상세 / 출력: 없음(전역 실패목록 갱신)"""
    flag = "PASS" if cond else "FAIL"
    print(f"  [{flag}] {name}{(' — ' + detail) if detail else ''}")
    if not cond:
        _failures.append(name)


def _full_chain(text: str, head_info: dict, raw_roberta: float, cbt: float) -> dict:
    """
    역할: infer_single 의 라우팅 후반부를 룰만으로 재현해 최종 웰니스 기여까지 계산한다.
    입력: 발화, head 발화타입 정보, raw roberta 점수(정규화·cap 전), cbt 점수
    출력: {emotion, score, guard, utterance_type, dep, impact, contribution, wellness}
    """
    emo, score, guard, info = apply_utterance_type_adjustment(
        text, "중립", raw_roberta, dict(head_info)
    )
    ens = ensemble_scores(score, cbt, distress_severity=0.10)
    result = {
        "utterance_type": info["utterance_type"],
        "utterance_type_reason": info["type_reason"],
        "top_emotion": emo,
        "emotion_guard": guard,
        "depression_score": ens["depression_score"],
    }
    contrib = compute_wellness_contribution(result, is_crisis=False)
    return {
        "emotion": emo,
        "score": score,
        "guard": guard,
        "utterance_type": info["utterance_type"],
        "type_reason": info["type_reason"],
        "dep": ens["depression_score"],
        "impact": contrib["wellness_impact_type"],
        "contribution": contrib["wellness_contribution_score"],
        "wellness": depression_to_wellness(contrib["wellness_contribution_score"]),
    }


def test_target_cat_case() -> None:
    """역할: 리포트된 핵심 케이스(현재 root 6/11 발화) 수정 검증"""
    text = "오늘 길고양이 봤는데 조그마한게 너무 귀여웠어"
    print(f"\n[케이스] 긍정 애정 발화: {text!r}")
    print("  (head=routine_discomfort 0.595, raw_roberta=0.50, cbt=0.485 가정)")
    r = _full_chain(text, HEAD_ROUTINE, raw_roberta=0.50, cbt=0.485)
    print(f"   → emotion={r['emotion']} type={r['utterance_type']}({r['type_reason']}) "
          f"score={r['score']:.3f} dep={r['dep']:.3f} impact={r['impact']} "
          f"contribution={r['contribution']:.3f} wellness={r['wellness']:.1f}")
    _check("긍정 텍스트로 인식", is_positive_affect_text(text) is True)
    _check("감정 행복 보존", r["emotion"] == "행복", r["emotion"])
    _check("점수 긍정 cap 이하", r["score"] <= POSITIVE_AFFECT_SCORE_CAP + 1e-9, f"{r['score']:.3f}<= {POSITIVE_AFFECT_SCORE_CAP}")
    _check("타입 positive_share 재분류", r["utterance_type"] == UTTERANCE_TYPES["POSITIVE_SHARE"], r["utterance_type"])
    _check("웰니스 반영 low", r["impact"] == "low", r["impact"])
    _check("기여 점수 0.30 이하", r["contribution"] is not None and r["contribution"] <= 0.30 + 1e-9, f"{r['contribution']}")
    _check("웰니스 양호권(>=80, 수정 전 ~53.9)", r["wellness"] >= 80.0, f"{r['wellness']:.1f}")


def test_distress_head_misroute() -> None:
    """역할: head가 emotional_distress로 오분류해도 명시 긍정이면 positive_share로 보정되는지"""
    text = "강아지가 깜찍하고 사랑스러웠어"
    print(f"\n[케이스] 애정 발화 + head=emotional_distress: {text!r}")
    r = _full_chain(text, HEAD_DISTRESS, raw_roberta=0.55, cbt=0.50)
    print(f"   → emotion={r['emotion']} type={r['utterance_type']} impact={r['impact']} "
          f"wellness={r['wellness']:.1f}")
    _check("새 마커(깜찍/사랑스) 긍정 인식", is_positive_affect_text(text) is True)
    _check("타입 positive_share 재분류", r["utterance_type"] == UTTERANCE_TYPES["POSITIVE_SHARE"], r["utterance_type"])
    _check("웰니스 반영 low", r["impact"] == "low", r["impact"])


def test_positive_hyperbole() -> None:
    """역할: 긍정 과장 표현('재밌어서 미칠거같은')이 고강도 distress 마커에 안 걸리고 positive_share로 가는지(Task 2)"""
    text = "재밌어서 미칠거같은 하루야"
    print(f"\n[케이스] 긍정 과장 표현 + head=emotional_distress: {text!r}")
    r = _full_chain(text, HEAD_DISTRESS, raw_roberta=0.55, cbt=0.45)
    print(f"   → emotion={r['emotion']} type={r['utterance_type']} impact={r['impact']} wellness={r['wellness']:.1f}")
    _check("긍정으로 인식(미칠거같 오인 방지)", is_positive_affect_text(text) is True)
    _check("감정 행복 보존", r["emotion"] == "행복", r["emotion"])
    _check("타입 positive_share 재분류", r["utterance_type"] == UTTERANCE_TYPES["POSITIVE_SHARE"], r["utterance_type"])
    _check("웰니스 반영 low + 양호권", r["impact"] == "low" and r["wellness"] >= 80.0, f"{r['impact']}/{r['wellness']:.1f}")


def test_regression_negative_hyperbole() -> None:
    """역할: 부정 문맥 고강도 표현('무서워서 미칠거같아')은 긍정으로 빠지지 않고 distress full 유지(안전 회귀)"""
    text = "무서워서 미칠거같아"
    print(f"\n[회귀] 부정 과장 표현: {text!r}")
    r = _full_chain(text, HEAD_DISTRESS, raw_roberta=0.70, cbt=0.50)
    print(f"   → emotion={r['emotion']} type={r['utterance_type']} impact={r['impact']}")
    _check("긍정으로 오인하지 않음", is_positive_affect_text(text) is False)
    _check("positive_share로 빠지지 않음", r["utterance_type"] != UTTERANCE_TYPES["POSITIVE_SHARE"], r["utterance_type"])
    _check("웰니스 반영 full 유지", r["impact"] == "full", r["impact"])


def test_regression_routine_discomfort() -> None:
    """역할: 진짜 일상 불편 발화는 그대로 routine_discomfort full 유지(과수정 방지)"""
    text = "출근하기 너무 싫고 귀찮아"
    print(f"\n[회귀] 일상 불편 발화: {text!r}")
    r = _full_chain(text, HEAD_ROUTINE, raw_roberta=0.50, cbt=0.45)
    print(f"   → emotion={r['emotion']} type={r['utterance_type']} impact={r['impact']}")
    _check("긍정으로 오인하지 않음", is_positive_affect_text(text) is False)
    _check("타입 routine_discomfort 유지", r["utterance_type"] == UTTERANCE_TYPES["ROUTINE_DISCOMFORT"], r["utterance_type"])
    _check("웰니스 반영 full 유지", r["impact"] == "full", r["impact"])


def test_regression_anhedonia() -> None:
    """역할: 긍정 단어가 있어도 무쾌감/부정 문맥이면 긍정 라우팅 차단(POSITIVE_BLOCKING)"""
    text = "좋아하던 일도 손이 안 가고 마음이 움직이지 않아"
    print(f"\n[회귀] 무쾌감 발화(긍정 단어 포함): {text!r}")
    r = _full_chain(text, HEAD_DISTRESS, raw_roberta=0.60, cbt=0.55)
    print(f"   → emotion={r['emotion']} type={r['utterance_type']} impact={r['impact']}")
    _check("긍정으로 오인하지 않음", is_positive_affect_text(text) is False)
    _check("positive_share로 빠지지 않음", r["utterance_type"] != UTTERANCE_TYPES["POSITIVE_SHARE"], r["utterance_type"])


def test_close_day_positive_reaches_yangho() -> None:
    """역할: 긍정 day의 마감(close_day) 웰니스도 floor에 막히지 않고 양호권에 도달하는지(EWMA 경로, 모델 불필요)"""
    from pipeline.score_pipeline import ScorePipeline, _floor_no_signal_daily

    print("\n[케이스] close_day floor — 긍정 day 마감 웰니스")
    # floor 헬퍼 단위 검증
    _check("legacy 0.0 → baseline 0.30", abs(_floor_no_signal_daily(0.0) - 0.30) < 1e-9)
    _check("긍정 0.154 통과(floor 안 함)", abs(_floor_no_signal_daily(0.154) - 0.154) < 1e-9)
    _check("무신호 0.30 유지", abs(_floor_no_signal_daily(0.30) - 0.30) < 1e-9)
    # close_day는 모델 호출 없이 버퍼 EWMA만 계산하므로 model=None으로 구성 가능
    pipe = ScorePipeline(model=None, tokenizer=None, device="cpu", anchor_embs={})
    pipe._today_utterances = [0.154]  # 긍정 발화 기여 점수 1건
    out = pipe.close_day()
    print(f"   → 긍정 day close: daily={out['daily_score']} smoothed={out['smoothed_score']} "
          f"wellness={out['wellness_score']} [{out['label']}]")
    _check("마감 웰니스 양호권(>=80, 수정 전 70.0)", out["wellness_score"] >= 80.0, f"{out['wellness_score']}")
    _check("마감 라벨 양호", out["label"] == "양호", out["label"])
    # 회귀: 무신호 day는 여전히 baseline 70으로 막혀 과상승하지 않아야 함
    pipe2 = ScorePipeline(model=None, tokenizer=None, device="cpu", anchor_embs={})
    pipe2._today_utterances = []
    out2 = pipe2.close_day()
    print(f"   → 무신호 day close: wellness={out2['wellness_score']} [{out2['label']}]")
    _check("무신호 day 70 유지(과상승 방지)", abs(out2["wellness_score"] - 70.0) < 1e-6, f"{out2['wellness_score']}")


def main() -> int:
    """역할: 전체 가드 케이스 실행 / 출력: 종료코드(0 성공, 1 실패)"""
    print("=" * 70)
    print("positive_affect_routing_guard — 긍정/애정 발화 라우팅 회귀 가드")
    print("=" * 70)
    test_target_cat_case()
    test_distress_head_misroute()
    test_positive_hyperbole()
    test_regression_negative_hyperbole()
    test_regression_routine_discomfort()
    test_regression_anhedonia()
    test_close_day_positive_reaches_yangho()
    print("\n" + "-" * 70)
    if _failures:
        print(f"실패 {len(_failures)}건: {_failures}")
        return 1
    print("모든 가드 케이스 통과")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
