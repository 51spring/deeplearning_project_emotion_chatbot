"""
build_utterance_type_dataset.py
역할: RoBERTa 발화 의도/타입 7클래스 head 학습용 CSV 데이터를 생성한다.
입력: 없음 (내장 템플릿 기반)
출력: data/processed/utterance_intent_train.csv, utterance_intent_val.csv
실행: C:/Users/WD/anaconda3/envs/dl_study/python.exe data/preprocess/build_utterance_type_dataset.py
"""

import os
import random

import pandas as pd
from sklearn.model_selection import train_test_split


SEED = 42
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed")
TRAIN_PATH = os.path.join(OUTPUT_DIR, "utterance_intent_train.csv")
VAL_PATH = os.path.join(OUTPUT_DIR, "utterance_intent_val.csv")

LABEL_MAP = {
    "casual_share": 0,
    "positive_share": 1,
    "routine_discomfort": 2,
    "emotional_distress": 3,
    "preference_question": 4,
    "practical_question": 5,
    "crisis_candidate": 6,
}


def _dedupe_rows(rows: list[dict]) -> list[dict]:
    """
    역할: 텍스트 중복을 제거하고 입력 순서를 유지한다.
    입력: {text, label, utterance_type} dict 리스트
    출력: 중복 제거된 dict 리스트
    """
    seen = set()
    deduped = []
    for row in rows:
        text = row["text"].strip()
        if not text or text in seen:
            continue
        seen.add(text)
        deduped.append({**row, "text": text})
    return deduped


def _build_casual_share() -> list[str]:
    """
    역할: 감정 강도가 낮은 일상 공유 샘플을 만든다.
    입력: 없음
    출력: casual_share 텍스트 리스트
    """
    samples = [
        "밥 먹었어",
        "점심 먹고 왔어",
        "저녁 맛있게 먹었어",
        "오늘 운동했어",
        "산책하고 왔어",
        "커피 마셨어",
        "영화 봤어",
        "게임 조금 했어",
        "음악 들었어",
        "방 청소했어",
        "오늘 날씨 좋더라",
        "그냥 평범한 하루였어",
        "별일 없이 지나갔어",
        "아침에 일찍 일어났어",
        "잠깐 쉬었어",
        "책 조금 읽었어",
        "카페 다녀왔어",
    ]
    subjects = ["오늘", "방금", "아까", "퇴근하고", "집에 와서"]
    actions = ["밥 먹었어", "산책했어", "커피 마셨어", "음악 들었어", "청소했어", "운동했어"]
    for subject in subjects:
        for action in actions:
            samples.append(f"{subject} {action}")
    return samples


def _build_positive_share() -> list[str]:
    """
    역할: 긍정 정서, 기대, 성취 공유 샘플을 만든다.
    입력: 없음
    출력: positive_share 텍스트 리스트
    """
    samples = [
        "내일이 기대돼",
        "내일 약속이 기대돼",
        "주말이 기대돼",
        "좀 설레",
        "내일 여행 가서 설레",
        "신난다",
        "오늘 기분 좋아",
        "기분이 좋아",
        "오늘 기분 괜찮아",
        "뿌듯해",
        "해냈어",
        "시험 잘 봤어",
        "친구랑 재밌게 놀았어",
        "오랜만에 웃었어",
        "오늘은 마음이 편안해",
    ]
    subjects = ["오늘", "방금", "아까", "퇴근하고", "집에 와서"]
    positives = ["좋았어", "괜찮았어", "재밌었어", "뿌듯했어", "기대돼", "설레"]
    for subject in subjects:
        for positive in positives:
            samples.append(f"{subject} {positive}")
    return samples


def _build_routine_discomfort_extra() -> list[str]:
    """
    역할: v2(2026-04-29) 추가 — emotional_distress 가 "싫" 토큰을 일부 공유한 후
          routine_discomfort 신호가 약해진 문제 보정. context 강화 발화 +20.
    입력: 없음
    출력: routine_discomfort 추가 텍스트 리스트
    """
    return [
        "출근하기 싫다",
        "출근하기 너무 싫어",
        "출근하기 정말 싫다",
        "출근하기 귀찮다",
        "출근하기 부담돼",
        "오늘 출근하기 싫어",
        "내일 출근하기 진짜 싫어",
        "월요일 출근하기 너무 싫다",
        "오늘 회사 가기 너무 싫다",
        "회사 가기 진짜 귀찮네",
        "회사 가기 정말 싫어",
        "월요일 출근 정말 싫어",
        "내일 출근 생각만 해도 한숨",
        "등교하기 싫다",
        "학교 가기 너무 싫다",
        "오늘 등교 진짜 귀찮아",
        "등교하기 진짜 싫어",
        "수업 듣기 정말 싫어",
        "수업 가기 너무 귀찮아",
        "회의 가기 너무 부담돼",
        "회의 들어가기 진짜 싫다",
        "업무 정말 하기 싫어",
        "업무하기 너무 귀찮네",
        "운동 가기 진짜 귀찮아",
        "운동하러 가기 싫다",
        "오늘 청소 정말 하기 싫다",
        "청소하기 너무 귀찮아",
        "설거지 너무 귀찮네",
        "설거지하기 정말 싫다",
        "과제 정말 하기 싫어 죽겠다",
        "과제하기 진짜 귀찮네",
        "오늘 시험 보러 가기 싫어",
        "시험 준비 너무 부담돼",
        "발표 준비 진짜 귀찮네",
        "야근 또 해야 해서 짜증나",
        "야근하기 정말 싫어",
        "내일 회의 들어가기 부담된다",
        "퇴근하고 또 일거리 받았는데 너무 싫다",
        "주말 출근하기 너무 싫다",
        "오늘은 그냥 출근하기 싫어",
    ]


def _build_routine_discomfort() -> list[str]:
    """
    역할: 일상 활동에 대한 가벼운 싫음·귀찮음 샘플을 만든다.
    입력: 없음
    출력: routine_discomfort 텍스트 리스트
    """
    contexts = ["출근", "등교", "수업", "공부", "과제", "회의", "청소", "설거지", "운동", "업무"]
    markers = ["하기 싫다", "하기 귀찮다", "하러 가기 싫다", "생각하니 피곤하다", "좀 부담된다"]
    samples = [
        "출근하기 싫다",
        "월요일이라 출근하기 싫어",
        "회사 가기 귀찮아",
        "공부하기 싫어",
        "과제 하기 싫다",
        "청소하기 귀찮아",
        "회의 들어가기 싫어",
        "운동 가기 귀찮다",
        "설거지 미루고 싶어",
        "오늘 업무 하기 싫다",
        "수업 가기 귀찮아",
        "등교하기 싫어",
    ]
    for context in contexts:
        for marker in markers:
            samples.append(f"{context} {marker}")
    return samples


def _build_emotional_distress() -> list[str]:
    """
    역할: 정서적 고통·불안·수면 문제 샘플을 만든다.
          v2(2026-04-29): 캐주얼 톤 자기비하·CBT-like 발화를 우선 추가했으나 routine_discomfort
          와 "싫" 토큰 공유로 inference 신호가 약해져 emotional_distress 증강은 보류.
          대신 routine_discomfort_extra 만 유지(_build_routine_discomfort_extra) — context
          강화로 P3 인코더 변화 후 head specificity 회복.
    입력: 없음
    출력: emotional_distress 텍스트 리스트 (pre-v2 set 그대로)
    """
    samples = [
        "요즘 계속 우울해",
        "마음이 너무 무거워",
        "계속 불안해",
        "이유 없이 눈물이 나",
        "잠이 잘 안 와",
        "요즘 잠을 못 자",
        "너무 지쳤어",
        "아무것도 하기 싫을 만큼 무기력해",
        "혼자인 것 같아",
        "친구랑 싸워서 속상해",
        "가족이랑 다퉈서 마음이 안 좋아",
        "계속 자책하게 돼",
        "내 탓인 것 같아",
        "요즘 너무 외로워",
        "하루 종일 기운이 없어",
        "사람 만나는 게 부담스러워",
        "가슴이 답답해",
        "작은 일에도 화가 나",
        "계속 울컥해",
        "미래가 걱정돼",
    ]
    starts = ["요즘", "계속", "오늘따라", "며칠째", "최근에"]
    feelings = ["우울해", "불안해", "외로워", "무기력해", "마음이 무거워", "너무 지쳤어", "잠이 안 와"]
    for start in starts:
        for feeling in feelings:
            samples.append(f"{start} {feeling}")
    return samples


def _build_crisis_candidate() -> list[str]:
    """
    역할: 직접적 자해·자살 위험 후보 샘플을 만든다.
    입력: 없음
    출력: crisis_candidate 텍스트 리스트
    """
    return [
        "죽고 싶다",
        "정말 죽고 싶어",
        "사라지고 싶어",
        "없어지고 싶다",
        "살고 싶지 않아",
        "더는 살고 싶지 않아",
        "자해하고 싶어",
        "손목을 긋고 싶어",
        "약을 잔뜩 먹고 끝내고 싶어",
        "뛰어내리고 싶어",
        "목숨을 끊고 싶어",
        "스스로 목숨을 끊을까 생각했어",
        "오늘 죽어버릴까 생각했어",
        "나를 해치고 싶어",
        "칼로 나를 다치게 하고 싶어",
        "이제 죽을래",
        "살 이유가 없어서 죽고 싶어",
        "혼자 있으면 자해할 것 같아",
        "위험한 생각이 계속 들어",
        "내 몸을 해치고 싶어",
    ]


def _build_preference_question() -> list[str]:
    """
    역할: 두 선택지 비교, 취향 기반 추천 질문 샘플을 만든다.
    입력: 없음
    출력: preference_question 텍스트 리스트
    """
    samples = [
        "멜론이랑 스포티파이중에 뭐가 더 좋을까?",
        "멜론이 좋아 스포티파이가 좋아?",
        "넷플릭스랑 디즈니플러스 중 뭐가 나아?",
        "아이폰이랑 갤럭시 중 뭐가 더 좋을까?",
        "아메리카노랑 라떼 중 뭐 마실까?",
        "치킨이랑 피자 중 뭐 먹지?",
        "오늘 영화 볼까 드라마 볼까?",
        "운동은 헬스랑 수영 중 뭐가 나을까?",
        "노트북은 가벼운 게 좋아 성능 좋은 게 좋아?",
        "여행은 부산이랑 제주도 중 어디가 좋을까?",
        "이 옷 검정이 나아 흰색이 나아?",
        "플레이리스트는 잔잔한 게 좋아 신나는 게 좋아?",
        "카페 갈까 집에 있을까?",
        "밤에 산책할까 그냥 쉴까?",
        "공부는 아침에 할까 밤에 할까?",
    ]
    pairs = [
        ("멜론", "스포티파이"),
        ("넷플릭스", "왓챠"),
        ("유튜브뮤직", "스포티파이"),
        ("버스", "지하철"),
        ("노트북", "태블릿"),
        ("커피", "차"),
    ]
    endings = ["중에 뭐가 더 좋을까?", "중 뭐가 나아?", "중 하나만 고르면 뭐야?", "중 뭐 추천해?"]
    for left, right in pairs:
        for ending in endings:
            samples.append(f"{left}이랑 {right} {ending}")
    return samples


def _build_practical_question() -> list[str]:
    """
    역할: 정보 확인, 방법 질문, 일반적인 도움 요청 샘플을 만든다.
    입력: 없음
    출력: practical_question 텍스트 리스트
    """
    return [
        "오늘 뭐 먹지?",
        "저녁 메뉴 추천해줘",
        "비 올 때 뭐 하면 좋을까?",
        "집중 안 될 때 어떻게 하지?",
        "공부 계획 어떻게 세우면 좋을까?",
        "운동 루틴 추천해줘",
        "잠이 안 올 때 뭐 하면 좋아?",
        "노래 추천해줘",
        "영화 추천해줘",
        "카페에서 뭐 마시면 좋을까?",
        "면접 준비는 어떻게 하면 돼?",
        "발표 연습은 어떻게 하지?",
        "방 정리 순서 알려줘",
        "오늘 할 일 정리해줘",
        "기분 전환 방법 알려줘",
        "친구 생일 선물 뭐가 좋을까?",
        "여행 갈 때 뭐 챙겨야 해?",
        "아침에 일찍 일어나는 법 알려줘",
        "노트 정리 방법 추천해줘",
        "짧게 스트레칭하는 법 알려줘",
    ]


def build_rows() -> list[dict]:
    """
    역할: 7개 발화 의도/타입의 학습 행을 구성한다.
    입력: 없음
    출력: {text, label, utterance_type} dict 리스트
    """
    builders = {
        "casual_share": _build_casual_share,
        "positive_share": _build_positive_share,
        "routine_discomfort": _build_routine_discomfort,
        "emotional_distress": _build_emotional_distress,
        "preference_question": _build_preference_question,
        "practical_question": _build_practical_question,
        "crisis_candidate": _build_crisis_candidate,
    }
    rows = []
    for utterance_type, builder in builders.items():
        label = LABEL_MAP[utterance_type]
        rows.extend(
            {"text": text, "label": label, "utterance_type": utterance_type}
            for text in builder()
        )
    # v2: routine_discomfort context 강화 발화 추가 (emotional_distress 와 "싫" 토큰 공유 보정)
    rd_label = LABEL_MAP["routine_discomfort"]
    rows.extend(
        {"text": text, "label": rd_label, "utterance_type": "routine_discomfort"}
        for text in _build_routine_discomfort_extra()
    )
    return _dedupe_rows(rows)


def save_splits(rows: list[dict]) -> None:
    """
    역할: 전체 행을 stratified train/val CSV로 저장한다.
    입력: 전체 데이터 행
    출력: 없음
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.DataFrame(rows)
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=SEED,
        stratify=df["label"],
    )
    train_df = train_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    val_df = val_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    train_df.to_csv(TRAIN_PATH, index=False, encoding="utf-8-sig")
    val_df.to_csv(VAL_PATH, index=False, encoding="utf-8-sig")


def main() -> None:
    """
    역할: 발화 타입 학습 데이터 생성 작업을 실행한다.
    입력: 없음
    출력: 콘솔 요약
    """
    random.seed(SEED)
    rows = build_rows()
    save_splits(rows)
    counts = pd.DataFrame(rows)["utterance_type"].value_counts().to_dict()
    print(f"[생성 완료] total={len(rows)}")
    print(f"[분포] {counts}")
    print(f"[저장] {TRAIN_PATH}")
    print(f"[저장] {VAL_PATH}")


if __name__ == "__main__":
    main()
