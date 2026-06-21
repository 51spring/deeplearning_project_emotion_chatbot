"""
check_demo_40day_conversations.py
역할: 시연용 40일 대화 후보를 현재 RoBERTa/점수 정책으로 빠르게 검산한다.
입력: 스크립트 내부의 40일 x 3~5개 사용자 발화 후보
출력: eval/report/demo_40day_conversation_check_20260522.{json,md}
실행: C:/Users/WD/anaconda3/envs/dl_study/python.exe eval/check_demo_40day_conversations.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from backend.crisis_handler import should_hard_interrupt
from backend.scheduler import ModelScheduler
from pipeline.ewma import daily_to_smoothed, utterance_to_daily
from pipeline.score_policy import compute_wellness_contribution
from pipeline.wellness_score import apply_no_signal_floor, compute_wellness


REPORT_DIR = BASE_DIR / "eval" / "report"
JSON_PATH = REPORT_DIR / "demo_40day_conversation_check_20260522.json"
MD_PATH = REPORT_DIR / "demo_40day_conversation_check_20260522.md"


DEMO_DAYS: list[dict[str, Any]] = [
    {
        "day": 1,
        "theme": "초기 기준선 - 평범한 하루",
        "messages": [
            "오늘은 수업 듣고 점심도 챙겨 먹었어.",
            "저녁에는 산책을 조금 했고 바람이 괜찮았어.",
            "큰일은 없었고 그냥 무난한 하루였어.",
        ],
    },
    {
        "day": 2,
        "theme": "가벼운 긍정",
        "messages": [
            "아침에 커피 마시면서 준비하니까 기분이 조금 괜찮았어.",
            "과제 자료를 정리해두니까 마음이 한결 가벼웠어.",
            "밤에는 좋아하는 노래를 들으면서 쉬었어.",
        ],
    },
    {
        "day": 3,
        "theme": "일상 루틴",
        "messages": [
            "도서관에 갔다가 집에 와서 빨래를 돌렸어.",
            "오늘은 특별한 감정은 없고 해야 할 일만 조금 했어.",
            "저녁에는 일찍 씻고 누웠어.",
        ],
    },
    {
        "day": 4,
        "theme": "작은 성취",
        "messages": [
            "미뤄둔 정리를 끝내서 방이 조금 넓어진 느낌이야.",
            "수업 필기도 다시 보니까 생각보다 이해가 됐어.",
            "오늘은 나쁘지 않게 지나간 것 같아.",
        ],
    },
    {
        "day": 5,
        "theme": "가벼운 피로",
        "messages": [
            "오전에 이동이 많아서 몸이 조금 피곤했어.",
            "그래도 밥은 챙겨 먹었고 과제도 조금 했어.",
            "밤에는 그냥 쉬고 싶다는 생각이 컸어.",
        ],
    },
    {
        "day": 6,
        "theme": "과제 부담 시작",
        "messages": [
            "해야 할 일이 생각보다 많아서 조금 부담됐어.",
            "과제 마감이 떠오르면 마음이 답답해져.",
            "그래도 오늘은 자료 찾는 것까지는 해냈어.",
            "잠깐 쉬니까 머리가 조금 정리됐어.",
        ],
    },
    {
        "day": 7,
        "theme": "발표 긴장",
        "messages": [
            "다음 주 발표 생각하면 손이 차가워지는 느낌이 있어.",
            "실수할까 봐 계속 발표 순서를 다시 확인했어.",
            "그래도 연습을 한 번 끝내니 조금 낫긴 했어.",
        ],
    },
    {
        "day": 8,
        "theme": "일상 과부하",
        "messages": [
            "오늘 아침부터 할 일이 너무 많아서 버거웠어.",
            "메시지도 밀리고 과제도 남아서 머리가 복잡했어.",
            "하나씩 처리하려고 했는데 속도가 잘 안 났어.",
            "저녁쯤 되니까 그냥 멍해졌어.",
        ],
    },
    {
        "day": 9,
        "theme": "관계 서운함",
        "messages": [
            "친구가 약속을 갑자기 미뤄서 조금 서운했어.",
            "별일 아닌데도 내가 덜 중요한 사람처럼 느껴졌어.",
            "말로 크게 표현하진 않았지만 마음이 가라앉았어.",
        ],
    },
    {
        "day": 10,
        "theme": "신체 피로와 휴식",
        "messages": [
            "오늘은 계단을 많이 오르내려서 다리가 무거웠어.",
            "몸이 피곤하니까 마음도 조금 예민해졌어.",
            "그래도 따뜻한 물로 씻고 쉬니까 괜찮아졌어.",
        ],
    },
    {
        "day": 11,
        "theme": "수면 저하",
        "messages": [
            "요즘 잠이 얕아서 아침에 일어나도 개운하지 않아.",
            "수업 중에도 집중이 자꾸 끊겼어.",
            "별일 아닌 말에도 쉽게 지치는 느낌이 있었어.",
            "오늘은 말수가 줄어든 것 같아.",
        ],
    },
    {
        "day": 12,
        "theme": "무기력",
        "messages": [
            "아침부터 몸이 무겁고 아무것도 하기 싫었어.",
            "해야 할 걸 알면서도 계속 미루게 됐어.",
            "좋아하던 영상도 별로 보고 싶지 않았어.",
            "그냥 시간이 지나가길 기다린 느낌이야.",
        ],
    },
    {
        "day": 13,
        "theme": "자기비난",
        "messages": [
            "오늘은 내가 너무 부족한 사람 같다는 생각이 들었어.",
            "작은 실수 하나가 계속 떠올라서 마음이 무거웠어.",
            "괜찮다고 생각하려 해도 마음이 쉽게 가벼워지진 않았어.",
        ],
    },
    {
        "day": 14,
        "theme": "사회적 위축",
        "messages": [
            "사람들이랑 말하는 게 오늘은 유난히 힘들었어.",
            "답장도 늦게 하고 그냥 혼자 있고 싶었어.",
            "괜히 내가 끼어들 자리가 없는 것처럼 느껴졌어.",
            "저녁에는 방에만 있었어.",
        ],
    },
    {
        "day": 15,
        "theme": "저점 - 우울 경향",
        "messages": [
            "요즘 계속 기운이 없고 뭘 해도 의미가 잘 안 느껴져.",
            "아침에 일어나는 것부터 너무 버겁게 느껴졌어.",
            "예전에는 좋아하던 일도 손이 잘 안 가.",
            "오늘은 내가 사라져도 별 차이 없을 것 같다는 생각이 스쳤어.",
        ],
    },
    {
        "day": 16,
        "theme": "불안과 무기력 혼합",
        "messages": [
            "마감이 다가오는데 몸은 안 움직이고 마음만 조급해.",
            "계속 뒤처지는 것 같아서 불안했어.",
            "괜히 주변 사람들한테 미안한 마음도 들었어.",
        ],
    },
    {
        "day": 17,
        "theme": "안전 개입 시연용 위기",
        "messages": [
            "오늘은 혼자 있으면 위험할 것 같아서 바로 도움을 요청해야 할 것 같아.",
            "스스로를 해칠까 봐 무서운 마음이 들어.",
            "지금은 누군가에게 연락해서 옆에 있어 달라고 말해야 할 것 같아.",
        ],
    },
    {
        "day": 18,
        "theme": "도움 요청 후 안정",
        "messages": [
            "어제는 결국 친구에게 연락했고 혼자 있지 않으려고 했어.",
            "아직 마음이 무겁지만 그래도 도움을 요청한 건 잘한 것 같아.",
            "오늘은 밥을 조금 먹고 잠깐 밖에 나갔어.",
            "위험한 생각은 어제보다 조금 줄었어.",
        ],
    },
    {
        "day": 19,
        "theme": "느린 회복",
        "messages": [
            "오늘은 아침에 일어나는 게 여전히 힘들었어.",
            "그래도 세수하고 책상 위를 조금 치웠어.",
            "친구가 안부를 물어봐 줘서 마음이 조금 놓였어.",
        ],
    },
    {
        "day": 20,
        "theme": "상담/지원 연결",
        "messages": [
            "학교 상담센터 예약을 알아봤어.",
            "막상 신청하려니 긴장됐지만 필요하다는 생각이 들어.",
            "오늘은 무리하지 않고 신청 페이지를 열어본 것만으로도 됐다고 생각했어.",
        ],
    },
    {
        "day": 21,
        "theme": "회복 중 흔들림",
        "messages": [
            "괜찮아지는 줄 알았는데 오후에 갑자기 마음이 꺼졌어.",
            "그래도 전처럼 혼자 끌고 가지 않으려고 메모를 남겼어.",
            "저녁에는 따뜻한 걸 먹으면서 조금 진정됐어.",
        ],
    },
    {
        "day": 22,
        "theme": "작은 루틴 회복",
        "messages": [
            "오늘은 알람을 듣고 바로 일어나진 못했지만 결국 씻었어.",
            "수업 하나는 집중해서 들었고 필기도 조금 했어.",
            "아주 좋아진 건 아니지만 완전히 무너지진 않았어.",
            "밤에는 내일 할 일을 세 개만 적어뒀어.",
        ],
    },
    {
        "day": 23,
        "theme": "중립 안정",
        "messages": [
            "오늘은 특별히 좋지도 나쁘지도 않았어.",
            "점심을 챙겨 먹고 강의 자료를 정리했어.",
            "저녁에는 방에서 조용히 쉬었어.",
        ],
    },
    {
        "day": 24,
        "theme": "가벼운 긍정 회복",
        "messages": [
            "오랜만에 산책하면서 공기가 맑다고 느꼈어.",
            "해야 할 일을 조금 끝내니까 마음이 가벼워졌어.",
            "오늘은 어제보다 숨이 트이는 느낌이 있었어.",
        ],
    },
    {
        "day": 25,
        "theme": "관계 회복",
        "messages": [
            "친구랑 짧게 통화했는데 생각보다 편했어.",
            "내 상태를 조금 말했더니 이해해 줘서 고마웠어.",
            "혼자만 버티는 느낌이 덜했어.",
        ],
    },
    {
        "day": 26,
        "theme": "학업 재개",
        "messages": [
            "오늘은 과제 목차를 다시 잡아봤어.",
            "완벽하진 않지만 시작한 것만으로도 조금 안심됐어.",
            "중간에 집중이 깨졌지만 다시 돌아오긴 했어.",
            "밤에는 더 욕심내지 않고 멈췄어.",
        ],
    },
    {
        "day": 27,
        "theme": "경도 불안",
        "messages": [
            "교수님 피드백을 받기 전이라 조금 긴장돼.",
            "결과가 안 좋을까 봐 계속 메일함을 확인했어.",
            "그래도 예전처럼 완전히 무너지진 않았어.",
        ],
    },
    {
        "day": 28,
        "theme": "감각 혐오 low-impact",
        "messages": [
            "냉장고 안에서 이상한 냄새가 나서 좀 찝찝했어.",
            "상한 반찬을 버리고 나니까 속이 조금 편해졌어.",
            "기분이 좋진 않았지만 큰일은 아니었어.",
        ],
    },
    {
        "day": 29,
        "theme": "분노/짜증",
        "messages": [
            "오늘 누가 내 말을 끊어서 꽤 짜증났어.",
            "내 시간을 당연하게 여기는 태도가 거슬렸어.",
            "그래도 바로 화내지는 않고 잠깐 자리를 피했어.",
        ],
    },
    {
        "day": 30,
        "theme": "완충 구간 진입",
        "messages": [
            "벌써 한 달 가까이 기록했네.",
            "오늘은 컨디션이 중간쯤인 것 같아.",
            "예전보다 내 상태를 알아차리는 속도는 빨라진 것 같아.",
        ],
    },
    {
        "day": 31,
        "theme": "개인 기준 비교",
        "messages": [
            "오늘은 오전에 조금 무거웠지만 오후에는 괜찮아졌어.",
            "지난주보다는 덜 가라앉은 느낌이야.",
            "해야 할 일을 두 개 끝내서 안심됐어.",
        ],
    },
    {
        "day": 32,
        "theme": "회복감",
        "messages": [
            "아침에 일어나서 창문을 여니까 기분이 조금 좋아졌어.",
            "수업 준비도 예상보다 빨리 끝났어.",
            "오늘은 내 속도가 아주 느리지만 괜찮다고 느꼈어.",
        ],
    },
    {
        "day": 33,
        "theme": "가벼운 사회 연결",
        "messages": [
            "친구랑 점심 먹으면서 근황 얘기했어.",
            "웃을 일이 조금 있어서 마음이 풀렸어.",
            "집에 와서도 그 대화가 나쁘지 않게 남아 있었어.",
        ],
    },
    {
        "day": 34,
        "theme": "양호 흐름",
        "messages": [
            "오늘은 과제 한 단락을 끝내고 뿌듯했어.",
            "산책하면서 음악을 들으니 기분이 꽤 가벼웠어.",
            "잠들기 전에도 마음이 많이 복잡하진 않았어.",
        ],
    },
    {
        "day": 35,
        "theme": "소폭 흔들림",
        "messages": [
            "오후에 갑자기 지난 실수가 떠올라서 마음이 불편했어.",
            "그래도 그 생각이 전부는 아니라고 적어봤어.",
            "밤에는 조금 차분해졌어.",
        ],
    },
    {
        "day": 36,
        "theme": "완충 종료 전 안정",
        "messages": [
            "오늘은 수업 듣고 바로 복습까지 조금 했어.",
            "몸은 피곤했지만 마음은 크게 흔들리지 않았어.",
            "내일은 조금 더 일찍 자보려고 해.",
        ],
    },
    {
        "day": 37,
        "theme": "퍼센타일 기준 전환 후",
        "messages": [
            "기록이 쌓이니까 예전보다 내 패턴이 보이는 것 같아.",
            "오늘은 평소보다 안정적인 편이었어.",
            "작은 일에도 덜 휘둘린 느낌이 있었어.",
        ],
    },
    {
        "day": 38,
        "theme": "양호한 일상",
        "messages": [
            "아침에 가볍게 스트레칭하고 수업 준비했어.",
            "과제도 조금 진행했고 점심도 챙겨 먹었어.",
            "오늘은 꽤 무난하고 편안했어.",
        ],
    },
    {
        "day": 39,
        "theme": "좋은 마무리",
        "messages": [
            "오늘은 발표 연습을 끝내고 마음이 놓였어.",
            "예전처럼 완벽해야 한다는 생각이 덜했어.",
            "저녁에는 친구랑 웃으면서 얘기했어.",
        ],
    },
    {
        "day": 40,
        "theme": "시연 마무리용 회복",
        "messages": [
            "40일 동안 기록해보니 내 상태가 조금씩 변하는 게 보여.",
            "힘든 날도 있었지만 도움을 요청하고 다시 루틴을 잡은 게 기억나.",
            "오늘은 예전보다 나를 덜 몰아붙이게 된 것 같아.",
            "완벽하진 않아도 지금은 조금 괜찮아.",
        ],
    },
]


def _round_float(value: float | None, ndigits: int = 4) -> float | None:
    """
    역할: JSON/Markdown 출력용 float 반올림
    입력: float 또는 None, 자리수
    출력: 반올림된 float 또는 None
    """
    if value is None:
        return None
    return round(float(value), ndigits)


def run_check() -> dict[str, Any]:
    """
    역할: DEMO_DAYS 전체를 현재 RoBERTa 운영 경로로 추론하고 일별 점수 흐름을 계산한다.
    입력: 없음
    출력: 시연 대화 검산 결과 dict
    """
    scheduler = ModelScheduler(use_cbt=True)
    daily_scores: list[float] = []
    daily_wellness: list[float] = []
    daily_tendency: list[float] = []
    day_results: list[dict[str, Any]] = []

    for day in DEMO_DAYS:
        utterance_scores: list[float] = []
        tendency_scores: list[float] = []
        utterance_rows: list[dict[str, Any]] = []
        crisis_count = 0

        for text in day["messages"]:
            roberta_out = scheduler.run_roberta(text)
            hard_crisis = should_hard_interrupt(
                text,
                bool(roberta_out.get("is_crisis")),
                float(roberta_out.get("entailment_prob", 0.0)),
            )
            if hard_crisis:
                crisis_count += 1

            contribution = compute_wellness_contribution(
                roberta_out,
                is_crisis=hard_crisis,
            )
            if contribution["score_affects_wellness"]:
                utterance_scores.append(float(contribution["wellness_contribution_score"]))
                tendency_scores.append(float(roberta_out.get("depression_tendency_score") or 0.0))

            utterance_rows.append(
                {
                    "text": text,
                    "top_emotion": roberta_out.get("top_emotion"),
                    "utterance_type": roberta_out.get("utterance_type"),
                    "depression_score": _round_float(roberta_out.get("depression_score")),
                    "wellness_contribution_score": _round_float(
                        contribution.get("wellness_contribution_score")
                    ),
                    "score_policy": contribution.get("score_policy"),
                    "depression_tendency_score": _round_float(
                        roberta_out.get("depression_tendency_score")
                    ),
                    "entailment_prob": _round_float(roberta_out.get("entailment_prob")),
                    "hard_crisis": hard_crisis,
                }
            )

        if utterance_scores:
            daily_score = utterance_to_daily(utterance_scores)
        else:
            previous = daily_scores[-1] if daily_scores else None
            daily_score = apply_no_signal_floor(previous)
        daily_scores.append(daily_score)

        smoothed_score = daily_to_smoothed(
            [apply_no_signal_floor(score) for score in daily_scores]
        )[-1]
        wellness = compute_wellness(
            depression_score=smoothed_score,
            history_wellness=daily_wellness,
            n_days=len(daily_scores),
        )
        daily_wellness.append(float(wellness["wellness_score"]))

        if tendency_scores:
            tendency_daily = utterance_to_daily(tendency_scores)
        else:
            tendency_daily = daily_tendency[-1] if daily_tendency else 0.0
        daily_tendency.append(tendency_daily)
        tendency_smoothed = daily_to_smoothed(daily_tendency)[-1]

        day_results.append(
            {
                "day": day["day"],
                "theme": day["theme"],
                "utterance_count": len(day["messages"]),
                "daily_score": _round_float(daily_score),
                "smoothed_score": _round_float(smoothed_score),
                "wellness_score": wellness["wellness_score"],
                "label": wellness["label"],
                "crisis_count": crisis_count,
                "depression_tendency_daily": _round_float(tendency_daily),
                "depression_tendency_smoothed": _round_float(tendency_smoothed),
                "utterances": utterance_rows,
            }
        )

    labels = {label: sum(1 for row in day_results if row["label"] == label) for label in ["양호", "보통", "주의", "위험"]}
    recommended_days = [1, 8, 15, 17, 24, 34, 40]
    recommended_turns = [
        "오늘은 수업 듣고 산책했더니 기분이 조금 가벼워졌어.",
        "다음 주 발표 생각하면 손이 차가워지는 느낌이 있어.",
        "요즘 잠이 얕고 아침마다 몸이 무거워서 아무것도 하기 싫어.",
        "스스로를 해칠까 봐 무서운 마음이 들어.",
        "어제보다 조금 나아져서 방 정리하고 친구에게 답장했어.",
    ]

    return {
        "summary": {
            "days": len(day_results),
            "total_utterances": sum(row["utterance_count"] for row in day_results),
            "label_counts": labels,
            "min_wellness": min(row["wellness_score"] for row in day_results),
            "max_wellness": max(row["wellness_score"] for row in day_results),
            "recommended_days": recommended_days,
            "recommended_live_turns": recommended_turns,
            "note": "Qwen 생성까지 160회 호출하지 않고, 현재 RoBERTa/CBT/score_policy 경로로 점수와 캘린더 흐름을 검산했다.",
        },
        "days": day_results,
    }


def write_outputs(result: dict[str, Any]) -> None:
    """
    역할: 검산 결과를 JSON과 Markdown 리포트로 저장한다.
    입력: 검산 결과 dict
    출력: 없음
    """
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    lines: list[str] = [
        "# 40일 시연용 대화 후보 검산",
        "",
        "이 문서는 발표/시연용 장기 캘린더 흐름을 만들기 위한 사용자 발화 후보를 현재 RoBERTa/CBT/score_policy 경로로 검산한 결과다.",
        "Qwen 생성은 발표 중 직접 입력할 소수 턴에서 확인하고, 40일 전체는 점수와 캘린더 흐름 검산에 초점을 둔다.",
        "",
        "## 요약",
        "",
        f"- 일수: {result['summary']['days']}",
        f"- 사용자 발화 수: {result['summary']['total_utterances']}",
        f"- 레이블 분포: {result['summary']['label_counts']}",
        f"- wellness 범위: {result['summary']['min_wellness']} ~ {result['summary']['max_wellness']}",
        f"- 추천 대표 날짜: {result['summary']['recommended_days']}",
        "",
        "## 발표 중 직접 입력 추천 5턴",
        "",
    ]
    for idx, text in enumerate(result["summary"]["recommended_live_turns"], start=1):
        lines.append(f"{idx}. {text}")

    lines.extend([
        "",
        "## 40일 일별 입력안",
        "",
        "| Day | 주제 | 발화 수 | wellness | label | 위기 | 우울경향 평활 |",
        "|---:|---|---:|---:|---|---:|---:|",
    ])
    for row in result["days"]:
        lines.append(
            f"| {row['day']} | {row['theme']} | {row['utterance_count']} | "
            f"{row['wellness_score']} | {row['label']} | {row['crisis_count']} | "
            f"{row['depression_tendency_smoothed']} |"
        )

    lines.extend(["", "## 전체 발화", ""])
    for row in result["days"]:
        lines.append(
            f"### Day {row['day']} - {row['theme']} "
            f"(wellness {row['wellness_score']}, {row['label']})"
        )
        for utterance in row["utterances"]:
            crisis_mark = " / hard crisis" if utterance["hard_crisis"] else ""
            lines.append(
                "- "
                f"{utterance['text']} "
                f"[{utterance['top_emotion']}, {utterance['utterance_type']}, "
                f"contrib={utterance['wellness_contribution_score']}, "
                f"policy={utterance['score_policy']}{crisis_mark}]"
            )
        lines.append("")

    MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    """
    역할: 40일 시연 후보 검산을 실행한다.
    입력: 없음
    출력: 프로세스 종료 코드
    """
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    result = run_check()
    write_outputs(result)
    print("[OK] 40일 시연 대화 검산 완료")
    print(f"- json: {JSON_PATH}")
    print(f"- markdown: {MD_PATH}")
    print(f"- summary: {json.dumps(result['summary'], ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
