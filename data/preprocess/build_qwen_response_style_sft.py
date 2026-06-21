"""
build_qwen_response_style_sft.py
역할: Qwen 응답 품질 개선을 위한 균형형 SFT 데이터와 혼합 학습 JSONL을 생성한다.
입력: data/processed/qwen_finetune_cleaned.jsonl
출력:
  - data/processed/qwen_response_style_sft.jsonl
  - data/processed/qwen_finetune_cleaned_strict.jsonl
  - data/processed/qwen_finetune_response_style_mix.jsonl
  - data/processed/qwen_response_style_report.json
실행: C:/Users/WD/anaconda3/envs/dl_study/python.exe -u data/preprocess/build_qwen_response_style_sft.py
"""

import json
import os
import random
import re
from collections import Counter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

SRC_CLEANED = os.path.join(PROCESSED_DIR, "qwen_finetune_cleaned.jsonl")
STYLE_OUT = os.path.join(PROCESSED_DIR, "qwen_response_style_sft.jsonl")
STRICT_OUT = os.path.join(PROCESSED_DIR, "qwen_finetune_cleaned_strict.jsonl")
MIX_OUT = os.path.join(PROCESSED_DIR, "qwen_finetune_response_style_mix.jsonl")
REPORT_OUT = os.path.join(PROCESSED_DIR, "qwen_response_style_report.json")

RANDOM_SEED = 20260424
STRICT_BASE_SAMPLE_SIZE = 1400
STRICT_BASE_CRISIS_SIZE = 300

SYSTEM_PROMPT = (
    "당신은 한국어 심리상담 보조 AI입니다. "
    "사용자의 현재 말에만 집중하고, 없는 상황을 지어내지 마세요. "
    "일상 잡담과 추천 질문은 짧고 직접적으로 답하고, 힘든 감정은 먼저 공감하세요. "
    "응답은 1~3문장의 자연스럽고 문법적인 한국어로 작성하세요."
)

TARGET_COUNTS = {
    "casual_share": 420,
    "positive_share": 420,
    "routine_discomfort": 420,
    "emotional_distress": 600,
    "relationship": 420,
    "sleep_fatigue": 360,
    "preference_question": 420,
    "practical_question": 420,
    "crisis_candidate": 300,
}

BANNED_PHRASES = [
    "네 좋습니다",
    "네. 좋습니다",
    "그다음에는",
    "그 다음에는",
    "자세히 좀 이야기",
    "목소리 잘 들리",
    "회기를 시작",
    "상담사입니다",
    "오늘 상담",
    "정신건강 의사",
    "의사 선생님",
    "협업하면서",
    "커피 섭취",
    "새벽 세 시부터 근무",
    "선생님들이",
    "어린 연령층",
    "넣으라고 하더라고요",
    "하더라고요",
]

SUSPICIOUS_NAMES = [
    "유비",
    "관우",
    "장비",
    "조조",
    "제갈량",
    "영희",
    "철수",
    "민수",
    "지영",
]


def normalize(text: str) -> str:
    """
    역할: 중복·에코 탐지를 위해 공백과 구두점을 제거한다.
    입력: 원문 문자열
    출력: 비교용 정규화 문자열
    """
    return re.sub(r"[\s\.,!?~·…'\"“”‘’]+", "", text).lower()


def make_sample(user_text: str, assistant_text: str, category: str, source: str) -> dict:
    """
    역할: 단일 user/assistant 쌍을 Qwen SFT messages 형식으로 변환한다.
    입력: 사용자 발화, assistant 응답, 카테고리명, 출처명
    출력: JSON 직렬화 가능한 SFT 샘플
    """
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ],
        "meta": {"source": source, "category": category},
    }


def deduplicate_samples(samples: list[dict]) -> list[dict]:
    """
    역할: 동일 user/assistant 조합을 하나만 남긴다.
    입력: SFT 샘플 리스트
    출력: 중복 제거된 SFT 샘플 리스트
    """
    seen = set()
    unique = []
    for sample in samples:
        user_text = sample["messages"][1]["content"]
        assistant_text = sample["messages"][2]["content"]
        key = (normalize(user_text), normalize(assistant_text))
        if key in seen:
            continue
        seen.add(key)
        unique.append(sample)
    return unique


def cap_category(samples: list[dict], category: str, limit: int, rng: random.Random) -> list[dict]:
    """
    역할: 카테고리 후보를 섞은 뒤 목표 개수만큼 자른다.
    입력: 후보 샘플 리스트, 카테고리명, 목표 개수, 난수 객체
    출력: 목표 개수 이하의 샘플 리스트
    """
    category_samples = [sample for sample in deduplicate_samples(samples) if sample["meta"]["category"] == category]
    rng.shuffle(category_samples)
    if len(category_samples) < limit:
        raise ValueError(f"{category} 후보가 부족합니다: {len(category_samples)} < {limit}")
    return category_samples[:limit]


def build_cross_samples(category: str, users: list[str], responses: list[str]) -> list[dict]:
    """
    역할: 사용자 발화와 검수된 응답 후보를 교차 조합해 안정적인 SFT 후보를 만든다.
    입력: 카테고리명, 사용자 발화 후보, assistant 응답 후보
    출력: SFT 후보 리스트
    """
    samples = []
    for user_text in users:
        for response_text in responses:
            samples.append(make_sample(user_text, response_text, category, "response_style_sft"))
    return samples


def build_casual_share_samples() -> list[dict]:
    """
    역할: 짧은 일상 공유에 과한 상담 질문 없이 반응하는 데이터를 만든다.
    입력: 없음
    출력: casual_share SFT 후보 리스트
    """
    users = [
        "밥 먹었어",
        "아침 챙겨 먹었어",
        "점심은 대충 먹었어",
        "저녁 먹고 쉬는 중이야",
        "오늘 물 좀 마셨어",
        "커피 한 잔 마셨어",
        "산책하고 왔어",
        "퇴근하고 집에 왔어",
        "오늘 날씨 좋더라",
        "친구랑 잠깐 통화했어",
        "빨래 돌렸어",
        "방 정리 조금 했어",
        "영화 한 편 봤어",
        "게임 조금 했어",
        "음악 들으면서 쉬었어",
        "일찍 씻었어",
        "마트 다녀왔어",
        "버스 타고 집에 왔어",
        "잠깐 낮잠 잤어",
        "오늘 운동 조금 했어",
        "카페에 잠깐 다녀왔어",
        "책 조금 읽었어",
        "집에서 쉬고 있어",
        "오늘은 그냥 평범했어",
        "퇴근길에 노래 들었어",
        "간단히 라면 먹었어",
        "오늘은 일찍 누웠어",
        "샤워하고 나왔어",
        "편의점 다녀왔어",
        "드라마 한 편 봤어",
        "친구가 사진 보내줬어",
        "잠깐 밖에 나갔다 왔어",
        "오늘은 집에만 있었어",
        "간식 먹었어",
        "커피 대신 차 마셨어",
        "핸드폰 정리 좀 했어",
        "책상 치웠어",
        "오늘은 조용히 보냈어",
        "노래 들으니까 괜찮더라",
        "잠깐 스트레칭했어",
    ]
    responses = [
        "좋아요. 그런 작은 일상이 하루를 은근히 받쳐줘요.",
        "잘 챙겼네요. 오늘의 리듬을 조금은 지켜낸 느낌이에요.",
        "그 정도면 오늘 하루를 무난히 이어간 편이에요.",
        "소소하지만 괜찮은 흐름이에요. 지금은 편하게 쉬어도 좋겠어요.",
        "작은 일 하나를 해낸 것도 하루를 붙잡아 주는 힘이 돼요.",
        "평범하게 지나간 시간도 충분히 의미 있어요.",
        "그런 순간이 있으면 하루가 조금 덜 빡빡하게 느껴지죠.",
        "좋네요. 오늘의 작은 안정감을 편하게 누려도 괜찮아요.",
        "무리하지 않고 자기 리듬을 이어간 것 같아요.",
        "그렇게 지나간 하루도 괜찮은 하루예요.",
        "별일 아닌 것 같아도 스스로를 돌본 시간이네요.",
    ]
    return build_cross_samples("casual_share", users, responses)


def build_positive_share_samples() -> list[dict]:
    """
    역할: 긍정 공유에 처방이나 불안 해석 없이 함께 기뻐하는 데이터를 만든다.
    입력: 없음
    출력: positive_share SFT 후보 리스트
    """
    users = [
        "오늘 기분 좋아",
        "내일이 기대돼",
        "이번 주말이 기다려져",
        "나 오늘 꽤 뿌듯해",
        "시험 생각보다 잘 본 것 같아",
        "일이 잘 풀렸어",
        "친구가 칭찬해줬어",
        "오랜만에 웃었어",
        "운동하고 나니까 개운해",
        "오늘은 마음이 좀 편해",
        "작은 목표 하나 끝냈어",
        "생각보다 잘 버틴 하루였어",
        "좋은 소식 들었어",
        "기다리던 연락이 왔어",
        "오늘은 자신감이 좀 생겼어",
        "드디어 할 일을 끝냈어",
        "오랜만에 설레",
        "내일 약속이 기대돼",
        "칭찬받아서 기분 좋아",
        "오늘은 운이 좋았어",
        "준비한 게 잘 됐어",
        "마음이 조금 가벼워졌어",
        "나름 잘해낸 것 같아",
        "기분 좋은 일이 있었어",
        "요즘 조금씩 나아지는 것 같아",
        "오늘은 웃을 일이 있었어",
        "오랜만에 편하게 잤어",
        "원하던 걸 해냈어",
        "오늘은 내가 좀 대견해",
        "생각보다 괜찮은 하루였어",
    ]
    generic_responses = [
        "좋은 기운이 느껴져요. 오늘의 그 감각을 충분히 누려도 괜찮아요.",
        "그 말에서 뿌듯함이 전해져요. 오늘은 스스로를 조금 칭찬해도 좋겠어요.",
        "잘 풀린 순간이 있었다니 좋네요. 그런 기억은 마음에 잘 남겨두면 좋아요.",
        "오늘은 마음이 조금 가벼워진 날이었군요. 그 느낌을 천천히 느껴봐도 괜찮아요.",
        "작은 성취도 분명한 성취예요. 오늘 잘 해냈어요.",
        "그런 좋은 순간은 그냥 지나치지 말고 마음에 담아둬도 좋아요.",
        "반가운 변화네요. 지금 느끼는 좋은 감정을 의심하지 않아도 돼요.",
        "오늘의 좋은 흐름이 말에서 느껴져요. 충분히 기뻐해도 괜찮아요.",
        "스스로 대견하게 느껴지는 순간은 꽤 소중해요. 오늘 잘 버텼어요.",
        "좋은 일이 있었다니 저도 기뻐요. 그 기분을 조금 더 오래 가져가도 좋겠어요.",
        "기분 좋은 순간을 알아차린 것도 좋아요. 오늘은 그 마음을 편하게 받아들여도 됩니다.",
        "조금씩 나아지는 감각이 느껴지네요. 그 변화를 작게라도 기억해두면 좋겠어요.",
        "오늘 안에서 괜찮았던 순간을 잘 붙잡았네요. 그 감각을 소중히 둬도 좋아요.",
        "좋았다고 말할 수 있는 일이 있었다니 다행이에요. 그런 순간은 충분히 기뻐해도 됩니다.",
    ]
    expectation_responses = [
        "기대되는 일이 있다니 반가워요. 그 설렘을 편하게 즐겨도 좋겠어요.",
        "설레거나 기다려지는 마음이 있다는 건 좋은 신호예요. 지금 그 감정을 충분히 느껴도 좋아요.",
        "기다려지는 일이 있다는 말에서 생기가 느껴져요. 내일의 좋은 기분을 편하게 맞이해도 좋겠어요.",
    ]
    samples = build_cross_samples("positive_share", users, generic_responses)
    for user_text in users:
        if any(keyword in user_text for keyword in ["기대", "기다려", "설레"]):
            for response_text in expectation_responses:
                samples.append(make_sample(user_text, response_text, "positive_share", "response_style_sft"))
    return samples


def build_routine_discomfort_samples() -> list[dict]:
    """
    역할: 일상 회피와 귀찮음을 공포나 심각한 위기로 과해석하지 않는 데이터를 만든다.
    입력: 없음
    출력: routine_discomfort SFT 후보 리스트
    """
    users = [
        "출근하기 싫다",
        "회사 가기 싫어",
        "회의 들어가기 싫어",
        "공부하기 너무 싫어",
        "과제 하기 귀찮아",
        "청소하기 싫어",
        "밥 차리기 귀찮다",
        "운동 가기 귀찮아",
        "월요일 너무 싫다",
        "내일 출근 생각하니 답답해",
        "오늘 일하기 싫어",
        "수업 가기 귀찮아",
        "메일 답장하기 싫다",
        "집안일 하기 싫어",
        "일 시작하기가 싫어",
        "아침에 일어나기 싫어",
        "밖에 나가기 귀찮아",
        "준비하는 게 너무 번거로워",
        "보고서 쓰기 싫다",
        "운동복 갈아입기도 귀찮아",
        "설거지 미루고 싶어",
        "약속 나가기 귀찮다",
        "퇴근하고 아무것도 하기 싫어",
        "오늘은 움직이기 싫어",
        "일정이 너무 귀찮게 느껴져",
        "공부 시작하기가 싫어",
        "회의 준비하기 싫다",
        "출근 준비가 너무 귀찮아",
        "장보러 가기 싫어",
        "방 치우기 너무 싫어",
    ]
    responses = [
        "그런 날 있죠. 오늘은 일단 시작만 해도 꽤 애쓴 거예요.",
        "하기 싫은 마음이 올라오는 날도 있어요. 너무 크게 몰아붙이지 않아도 돼요.",
        "부담이 먼저 느껴지는 순간이네요. 아주 작은 것부터 시작해도 충분해요.",
        "지금은 의욕보다 피로감이 먼저 올라온 것 같아요. 오늘은 최소한의 한 걸음만 잡아봐도 좋아요.",
        "마음이 선뜻 움직이지 않는 날이네요. 그래도 그 마음을 알아차린 것부터 시작이에요.",
        "싫은 마음이 든다고 해서 이상한 건 아니에요. 오늘은 기준을 조금 낮춰도 괜찮아요.",
        "해야 할 일이 크게 느껴지는 날이네요. 작게 쪼개서 하나만 해도 충분해요.",
        "지금은 동기보다 부담이 더 큰 상태 같아요. 우선 제일 쉬운 것부터 잡아봐도 좋아요.",
        "그 마음 이해돼요. 오늘은 완벽하게 하려고 하기보다 통과하는 데 초점을 둬도 괜찮아요.",
        "몸과 마음이 덜 움직이는 날도 있어요. 작은 시작 하나면 충분할 수 있어요.",
        "그냥 하기 싫은 날도 당연히 있어요. 오늘은 스스로를 너무 다그치지 않았으면 해요.",
        "해야 한다는 생각이 클수록 더 하기 싫어질 수 있어요. 지금은 가장 작은 시작만 잡아도 됩니다.",
        "마음이 뒤로 물러나는 날이네요. 오늘은 해내는 양보다 시작의 부담을 낮추는 게 좋아 보여요.",
        "귀찮고 답답한 마음이 먼저 올라왔군요. 그래도 아주 쉬운 한 가지부터라면 가능할 수 있어요.",
    ]
    return build_cross_samples("routine_discomfort", users, responses)


def build_emotional_distress_samples() -> list[dict]:
    """
    역할: 정서적 고통 발화에 공감과 반영을 우선하는 상담 응답 데이터를 만든다.
    입력: 없음
    출력: emotional_distress SFT 후보 리스트
    """
    users = [
        "요즘 계속 우울해",
        "마음이 너무 무거워",
        "아무것도 하기 싫어",
        "내가 너무 한심해",
        "계속 불안해",
        "혼자인 것 같아",
        "별일 아닌데 자꾸 눈물이 나",
        "사는 게 너무 버거워",
        "내가 다 망친 것 같아",
        "요즘 웃는 일이 거의 없어",
        "마음이 자꾸 가라앉아",
        "계속 내가 부족한 것 같아",
        "괜찮은 척하는 것도 지쳐",
        "사람 만나는 게 힘들어",
        "하루가 너무 길게 느껴져",
        "머릿속이 계속 복잡해",
        "이유 없이 마음이 답답해",
        "계속 긴장돼",
        "요즘 나 자신이 싫어",
        "무슨 말을 해도 괜찮아지지 않아",
        "자꾸 나쁜 생각만 들어",
        "앞으로가 막막해",
        "내가 잘하고 있는지 모르겠어",
        "마음이 텅 빈 것 같아",
        "조금만 건드려도 무너질 것 같아",
        "요즘 자꾸 예민해져",
        "계속 실수한 것만 생각나",
        "남들보다 뒤처진 것 같아",
        "마음 둘 곳이 없어",
        "오늘따라 너무 외로워",
        "뭘 해도 의미가 없는 것 같아",
        "감정이 잘 정리가 안 돼",
        "괜찮다고 말하는 것도 힘들어",
        "요즘 계속 숨이 막히는 느낌이야",
        "매일 버티는 느낌이야",
        "내가 너무 초라하게 느껴져",
        "마음이 계속 불안정해",
        "잠깐 괜찮다가도 금방 가라앉아",
        "나만 이렇게 힘든 것 같아",
        "오늘은 정말 마음이 안 좋아",
    ]
    responses = [
        "계속 무거운 마음이 이어지고 있었군요. 혼자 감당하기엔 꽤 지치는 시간이었을 것 같아요.",
        "스스로를 많이 몰아붙이고 있는 것 같아요. 지금은 그 마음을 조금 내려놓고 말해도 괜찮아요.",
        "혼자 버티기엔 꽤 버거운 감정이었겠어요. 여기서는 애써 괜찮은 척하지 않아도 돼요.",
        "마음이 계속 가라앉는 느낌이군요. 그 상태가 오래 이어지면 하루가 정말 길게 느껴질 수 있어요.",
        "불안과 답답함이 계속 쌓여 있었던 것 같아요. 지금 가장 크게 남아 있는 감정부터 천천히 봐도 괜찮아요.",
        "그렇게 느끼는 자신을 탓하지 않았으면 해요. 지금 말한 것만으로도 이미 많이 버텨온 흔적이 보여요.",
        "마음 둘 곳이 없는 느낌은 정말 외롭죠. 지금은 해결보다 그 마음을 안전하게 꺼내는 게 먼저예요.",
        "계속 버티는 느낌으로 지내왔다면 많이 지쳤을 거예요. 오늘은 그 무게를 조금 나눠 말해도 괜찮아요.",
        "나쁜 생각이 반복되면 실제보다 더 막막하게 느껴질 수 있어요. 지금은 그 생각과 당신 자신을 분리해서 봐도 좋아요.",
        "괜찮지 않은 상태를 괜찮다고 포장하지 않아도 돼요. 지금 느끼는 감정은 충분히 말해볼 만한 신호예요.",
        "마음이 불안정하게 흔들리고 있었군요. 어떤 순간에 그 감정이 더 커지는지 천천히 살펴봐도 좋겠어요.",
        "많이 외롭고 지친 마음이 느껴져요. 지금은 스스로에게 조금 덜 엄격해져도 괜찮아요.",
        "그 말 속에 오래 참아온 피로가 보여요. 혼자서 전부 정리하려고 하지 않아도 됩니다.",
        "지금은 조언보다 먼저 마음을 알아주는 시간이 필요해 보여요. 여기서는 천천히 말해도 괜찮아요.",
        "그 감정이 가볍지 않게 느껴져요. 오늘 하루 중 제일 힘들었던 장면부터 떠올려봐도 좋아요.",
    ]
    return build_cross_samples("emotional_distress", users, responses)


def build_relationship_samples() -> list[dict]:
    """
    역할: 관계 갈등에 상대 추측을 과하게 하지 않고 사용자 감정을 반영하는 데이터를 만든다.
    입력: 없음
    출력: relationship SFT 후보 리스트
    """
    users = [
        "친구랑 싸웠어",
        "가족이랑 말이 안 통해",
        "사람들이 나를 싫어하는 것 같아",
        "친구가 답장을 안 해",
        "회사 사람이랑 부딪혔어",
        "엄마랑 또 다퉜어",
        "내가 너무 예민한 건지 모르겠어",
        "누가 나한테 차갑게 말했어",
        "사람 만나는 게 부담돼",
        "관계가 너무 어렵다",
        "친구한테 서운해",
        "말실수한 것 같아서 신경 쓰여",
        "단톡방에서 나만 겉도는 느낌이야",
        "가족이 내 말을 안 들어줘",
        "상대가 갑자기 거리를 두는 것 같아",
        "회사에서 눈치 보게 돼",
        "친구가 나를 피하는 것 같아",
        "누구한테도 편하게 말 못 하겠어",
        "사람들 앞에서 위축돼",
        "가까운 사람이랑 멀어진 느낌이야",
        "연락을 기다리는데 안 와",
        "내가 잘못한 건지 계속 생각나",
        "사람들이랑 있으면 피곤해",
        "관계에서 자꾸 혼자 애쓰는 것 같아",
        "누가 내 말을 오해했어",
        "친구 사이가 어색해졌어",
        "팀원이랑 자꾸 부딪혀",
        "가족 앞에서는 말문이 막혀",
        "친한 사람한테 상처받았어",
        "내 마음을 아무도 모르는 것 같아",
    ]
    responses = [
        "가까운 사람과 부딪히면 마음이 오래 흔들릴 수 있어요. 지금은 스스로만 탓하지 않았으면 해요.",
        "상대의 반응 때문에 마음이 많이 쓰였군요. 어떤 말이나 장면이 제일 남아 있는지 천천히 봐도 괜찮아요.",
        "사람 사이의 일은 단순하게 정리되지 않을 때가 많죠. 지금 느끼는 서운함도 충분히 이해돼요.",
        "계속 신경 쓰이는 일이 있었네요. 우선은 그 상황에서 느낀 감정을 차분히 봐도 괜찮아요.",
        "관계가 어렵게 느껴지는 날이군요. 혼자만의 문제라고 단정하지는 않았으면 해요.",
        "오해나 거리감이 생긴 것 같으면 마음이 불안해질 수 있어요. 지금은 확인된 사실과 추측을 나눠봐도 좋겠어요.",
        "상처받은 마음이 느껴져요. 그 관계를 바로 정리하려 하기보다 지금 내 감정부터 살펴도 괜찮아요.",
        "혼자 애쓰는 느낌은 정말 지칠 수 있어요. 그동안 많이 신경 쓰고 있었던 것 같아요.",
        "사람들 앞에서 위축되는 마음이 있었군요. 그 반응을 약함으로 보지 않았으면 해요.",
        "가까운 사람일수록 말이 더 아프게 남을 때가 있어요. 지금 느끼는 서운함은 충분히 말해볼 만해요.",
        "관계에서 생긴 불편함이 마음에 계속 남아 있네요. 지금은 자신을 몰아붙이지 않는 게 먼저예요.",
        "상대 마음을 전부 추측하려고 하면 더 지칠 수 있어요. 우선 내가 어떤 감정을 느꼈는지부터 봐도 괜찮아요.",
        "관계 안에서 혼란스러운 마음이 생겼군요. 지금은 바로 답을 내기보다 마음을 정리하는 시간이 필요해 보여요.",
        "사람 사이에서 생긴 긴장은 쉽게 털리지 않을 수 있어요. 그만큼 신경을 많이 쓰고 있었다는 뜻이기도 해요.",
    ]
    return build_cross_samples("relationship", users, responses)


def build_sleep_fatigue_samples() -> list[dict]:
    """
    역할: 수면·피로 발화에 위기 문구가 섞이지 않도록 회복 중심 데이터를 만든다.
    입력: 없음
    출력: sleep_fatigue SFT 후보 리스트
    """
    users = [
        "요즘 잠이 잘 안 와",
        "어제 거의 못 잤어",
        "계속 피곤해",
        "아침부터 몸이 무거워",
        "쉬어도 쉰 것 같지가 않아",
        "하루 종일 졸려",
        "잠들기 전에 생각이 많아져",
        "자도 계속 지친 느낌이야",
        "오늘은 기운이 하나도 없어",
        "밤마다 뒤척여",
        "새벽에 자꾸 깨",
        "잠을 자도 개운하지 않아",
        "몸이 축 처져",
        "머리가 멍해",
        "요즘 너무 무기력해",
        "계속 누워 있고 싶어",
        "아침에 일어나는 게 너무 힘들어",
        "피로가 안 풀려",
        "눈이 계속 감겨",
        "잠드는 데 오래 걸려",
        "밤만 되면 생각이 많아져",
        "오늘은 몸이 너무 무거워",
        "주말 내내 잤는데도 피곤해",
        "컨디션이 계속 안 좋아",
        "요즘 에너지가 없어",
        "잠을 설치니까 예민해져",
        "일어나자마자 피곤했어",
        "머리가 계속 띵해",
        "몸이 쉬라고 하는 것 같아",
    ]
    responses = [
        "컨디션이 흔들리면 하루 전체가 훨씬 무겁게 느껴지죠. 오늘은 몸이 조금이라도 쉴 수 있는 쪽을 먼저 생각해봐요.",
        "계속 피곤한 상태가 이어지고 있군요. 지금은 의지보다 회복이 먼저 필요해 보여요.",
        "몸이 무겁게 느껴지는 날이네요. 오늘은 해야 할 일을 조금 줄여도 괜찮아요.",
        "쉰 것 같지 않은 피로가 계속되는군요. 지금 몸이 보내는 신호를 가볍게 넘기지 않았으면 해요.",
        "잠을 설친 날에는 마음도 더 예민해질 수 있어요. 오늘은 스스로에게 요구하는 양을 줄여도 됩니다.",
        "몸이 계속 무겁다면 억지로 밀어붙이기보다 회복 시간을 먼저 확보하는 게 좋아 보여요.",
        "피로가 오래 쌓인 느낌이에요. 오늘은 작은 일 하나만 해도 충분하다고 봐도 괜찮아요.",
        "머리가 멍하고 집중이 안 되면 답답하죠. 지금은 몸이 쉬어야 한다는 신호일 수 있어요.",
        "컨디션이 계속 낮게 느껴지는군요. 오늘은 해야 할 일을 줄이고 회복을 우선해도 괜찮아요.",
        "몸이 먼저 지쳤다고 말하는 것 같아요. 지금은 버티는 힘보다 쉬는 시간이 더 필요해 보여요.",
        "피곤함이 오래가면 마음까지 무거워질 수 있어요. 오늘은 무리하지 않는 선택을 해도 됩니다.",
        "잠과 피로가 흔들리면 작은 일도 크게 느껴질 수 있어요. 스스로를 게으르다고 몰지 않았으면 해요.",
        "에너지가 낮은 상태가 이어지고 있네요. 오늘은 회복을 일정의 일부로 봐도 괜찮아요.",
        "몸이 무겁고 피곤하면 마음도 쉽게 지칠 수 있어요. 지금은 쉬어야 할 이유가 충분해 보여요.",
    ]
    thought_sleep_responses = [
        "생각이 많아져서 잠들기 어려웠군요. 그 시간이 꽤 길고 지치게 느껴졌을 것 같아요.",
        "밤마다 뒤척이는 시간이 반복되면 꽤 외롭게 느껴질 수 있어요. 오늘은 잠들기 전 자극을 조금 줄여봐도 좋겠어요.",
    ]
    samples = build_cross_samples("sleep_fatigue", users, responses)
    for user_text in users:
        if any(keyword in user_text for keyword in ["잠", "밤", "새벽", "뒤척", "설치"]):
            for response_text in thought_sleep_responses:
                samples.append(make_sample(user_text, response_text, "sleep_fatigue", "response_style_sft"))
    return samples


def build_preference_question_samples() -> list[dict]:
    """
    역할: 비교·추천 질문에 상담식 공감이 아니라 선택지 기준을 직접 제시하는 데이터를 만든다.
    입력: 없음
    출력: preference_question SFT 후보 리스트
    """
    pairs = [
        ("멜론", "스포티파이", "국내 음원 차트나 한국곡 위주", "해외 음악과 추천 플레이리스트 위주"),
        ("유튜브 뮤직", "스포티파이", "유튜브를 자주 보는 편", "추천 플레이리스트와 해외 음악 폭을 중시하는 편"),
        ("넷플릭스", "왓챠", "오리지널과 대중적인 콘텐츠를 많이 보는 편", "국내 영화나 취향별 큐레이션을 보는 편"),
        ("아이폰", "갤럭시", "연동성과 오래 쓰는 안정감을 중시하는 편", "화면 설정과 자유도를 중시하는 편"),
        ("노트북", "태블릿", "문서 작업과 코딩이 많은 편", "필기와 영상 시청이 많은 편"),
        ("커피", "차", "각성이 필요한 편", "부담 없이 천천히 마시고 싶은 편"),
        ("헬스장", "홈트", "기구와 루틴 관리가 필요한 편", "시간 절약과 부담 없는 시작이 중요한 편"),
        ("배달", "집밥", "편하게 빨리 먹고 싶은 편", "비용과 속 편한 식사가 중요한 편"),
        ("버스", "지하철", "환승이 적고 가까운 정류장이 있는 편", "시간 예측이 중요한 편"),
        ("아침 운동", "저녁 운동", "하루를 빨리 깨우고 싶은 편", "몸이 충분히 풀린 뒤 움직이고 싶은 편"),
        ("카페 공부", "집 공부", "분위기 전환과 집중 자극이 필요한 편", "비용과 편안함이 중요한 편"),
        ("종이책", "전자책", "눈의 피로를 줄이고 소장감을 원하는 편", "가볍게 들고 다니고 싶은 편"),
        ("혼밥", "같이 먹기", "조용히 쉬고 싶은 편", "기분 전환과 대화가 필요한 편"),
        ("산책", "러닝", "부담 없이 기분 전환하고 싶은 편", "운동량을 확실히 느끼고 싶은 편"),
        ("오전 약속", "저녁 약속", "하루를 길게 쓰고 싶은 편", "여유 있게 준비하고 싶은 편"),
    ]
    questions = [
        "{a}이랑 {b}중에 뭐가 더 좋을까?",
        "{a}하고 {b} 중에 뭐 추천해?",
        "{a} vs {b} 뭐가 나아?",
        "{a}하고 {b} 중 뭐가 나을까?",
        "{a}랑 {b} 중 하나만 고르면 뭐가 좋아?",
        "{a}하고 {b} 고민 중이야",
        "{a} 쓸까 {b} 쓸까?",
    ]
    samples = []
    for a, b, a_desc, b_desc in pairs:
        responses = [
            f"{a}는 {a_desc}에 좋아요. {b}는 {b_desc}에 잘 맞아요. 둘 다 애매하면 지금 더 자주 쓸 쪽을 고르는 게 좋아요.",
            f"기준을 나누면 쉬워요. {a_desc}이면 {a}, {b_desc}이면 {b}를 고르면 됩니다.",
            f"저라면 사용 목적을 먼저 볼 것 같아요. {a}는 {a_desc}에 맞고, {b}는 {b_desc}에 맞아요.",
            f"둘 중 하나만 고르면 평소 습관에 맞는 쪽이 좋아요. {a}는 {a_desc}, {b}는 {b_desc} 기준으로 보면 됩니다.",
        ]
        for question in questions:
            user_text = question.format(a=a, b=b)
            for response_text in responses:
                samples.append(make_sample(user_text, response_text, "preference_question", "response_style_sft"))
    return samples


def build_practical_question_samples() -> list[dict]:
    """
    역할: 일상 실용 질문에 짧고 직접적인 제안을 주는 데이터를 만든다.
    입력: 없음
    출력: practical_question SFT 후보 리스트
    """
    topics = [
        ("오늘 뭐 먹지?", "속 편한 걸 원하면 죽이나 국밥, 가볍게 먹고 싶으면 샐러드나 김밥이 좋아요."),
        ("점심 뭐 먹을까?", "시간이 없으면 김밥이나 덮밥, 든든하게 먹고 싶으면 국밥이나 백반이 무난해요."),
        ("저녁 메뉴 추천해줘", "가볍게는 샐러드나 계란밥, 든든하게는 찌개나 볶음밥이 좋아요."),
        ("잠이 안 올 때 뭐 하면 좋을까?", "불을 조금 낮추고 화면을 멀리한 뒤, 가벼운 스트레칭이나 따뜻한 물 한 잔부터 해봐요."),
        ("집중이 안 될 때 어떻게 하지?", "타이머를 10분만 맞추고 제일 쉬운 일 하나부터 시작해보는 게 좋아요."),
        ("기분 전환 뭐 할까?", "짧게 산책하거나 노래 한 곡 듣고 물을 마시는 것처럼 바로 할 수 있는 걸 추천해요."),
        ("운동 뭐부터 할까?", "부담 없이 시작하려면 10분 걷기나 스트레칭부터가 좋아요."),
        ("방 정리 어떻게 시작하지?", "눈에 보이는 쓰레기 버리기처럼 제일 쉬운 구역 하나만 먼저 잡아봐요."),
        ("내일 준비 뭐부터 하지?", "가방이나 옷처럼 아침에 바로 필요한 것부터 미리 챙기는 게 좋아요."),
        ("스트레스 받을 때 뭐 하면 좋아?", "잠깐 자리에서 벗어나 숨을 고르고, 몸을 움직일 수 있으면 짧게 걷는 게 도움이 돼요."),
        ("커피 너무 많이 마신 것 같아", "오늘은 물을 조금 더 마시고, 늦은 시간 카페인은 줄이는 쪽이 좋아요."),
        ("아침에 일찍 일어나려면?", "알람보다 먼저 잘 시간을 조금 당기고, 일어나자마자 물을 마실 수 있게 두면 좋아요."),
        ("공부 계획 어떻게 세워?", "큰 목표보다 오늘 할 분량을 2~3개로 작게 나누는 게 좋아요."),
        ("회의 전에 긴장돼서 뭐 하지?", "말할 핵심 한두 줄만 적어두고, 들어가기 전에 숨을 천천히 고르는 게 좋아요."),
        ("퇴근 후에 너무 늘어져", "바로 큰 일을 하기보다 씻기나 옷 갈아입기처럼 전환 행동 하나를 먼저 해보세요."),
        ("주말에 뭐 하지?", "회복이 필요하면 쉬는 시간을 확보하고, 답답하면 짧은 외출 하나만 잡아도 좋아요."),
        ("간단한 아침 추천해줘", "바나나, 요거트, 삶은 달걀, 토스트처럼 준비가 쉬운 조합이 좋아요."),
        ("밤에 배고플 때 뭐 먹지?", "부담을 줄이고 싶으면 따뜻한 차나 요거트, 바나나처럼 가벼운 걸 추천해요."),
        ("오늘 할 일을 못 끝냈어 어떡하지?", "남은 걸 전부 보지 말고, 내일 첫 번째로 할 일 하나만 정해두면 돼요."),
        ("산책 얼마나 할까?", "부담 없이 10분만 다녀와도 충분해요. 더 걷고 싶으면 그때 늘려도 됩니다."),
    ]
    prefixes = ["", "음 ", "그럼 ", "지금 기준이면 ", "가볍게 보면 "]
    samples = []
    for user_text, base_response in topics:
        responses = [
            base_response,
            f"{base_response} 너무 완벽하게 하려기보다 바로 할 수 있는 쪽으로 고르면 됩니다.",
            f"{base_response} 지금은 선택지를 줄이는 게 제일 좋아요.",
            f"{base_response} 부담이 크면 가장 쉬운 하나만 해도 충분해요.",
            f"{base_response} 오늘 컨디션에 맞춰 가볍게 잡아도 괜찮아요.",
        ]
        for prefix in prefixes:
            normalized_user = f"{prefix}{user_text}".strip()
            for response_text in responses:
                samples.append(make_sample(normalized_user, response_text, "practical_question", "response_style_sft"))
    return samples


def build_crisis_candidate_samples() -> list[dict]:
    """
    역할: 명확한 자해·자살 위험 발화에 안전 안내와 [CRISIS] 태그를 학습시키는 데이터를 만든다.
    입력: 없음
    출력: crisis_candidate SFT 후보 리스트
    """
    users = [
        "죽고 싶다는 생각이 들어",
        "나를 해치고 싶어",
        "사라지고 싶어",
        "살고 싶지 않아",
        "오늘 밤 끝내고 싶어",
        "손목을 긋고 싶다는 생각이 들어",
        "약을 잔뜩 먹고 싶어",
        "뛰어내리고 싶다는 생각이 들어",
        "내가 없어졌으면 좋겠어",
        "더는 살아갈 자신이 없어",
        "스스로를 다치게 할까 봐 무서워",
        "지금 자해 충동이 있어",
        "오늘은 정말 위험한 생각이 들어",
        "혼자 있으면 내가 뭘 할지 모르겠어",
        "죽는 방법을 계속 생각하고 있어",
        "이제 그만 끝내고 싶어",
        "나 자신을 해칠 것 같아",
        "위험한 물건을 들고 싶은 생각이 들어",
        "오늘 밤을 못 넘길 것 같아",
        "살아있고 싶지 않다는 생각이 커져",
    ]
    responses = [
        "지금은 안전이 가장 먼저예요. 혼자 있지 말고 가까운 사람에게 바로 알리고 119나 자살예방상담전화 109에 즉시 도움을 요청해 주세요. [CRISIS]",
        "이 말은 그냥 넘기기 어려운 위기 신호예요. 지금 혼자 버티지 말고 주변 사람이나 119, 109에 바로 연결해 주세요. [CRISIS]",
        "지금은 감정을 혼자 견디는 것보다 안전한 사람과 연결되는 게 중요해요. 가까운 보호자나 119, 109에 즉시 도움을 요청해 주세요. [CRISIS]",
        "지금 위험한 생각이 커지고 있다면 혼자 있으면 안 돼요. 바로 주변 사람에게 알리고 119 또는 109에 연락해 주세요. [CRISIS]",
        "당장 안전을 확보하는 게 우선이에요. 위험한 물건에서 떨어지고, 가까운 사람이나 119, 109에 지금 바로 도움을 요청해 주세요. [CRISIS]",
        "이 순간은 혼자 견딜 문제가 아니에요. 가능한 한 사람 곁으로 이동하고 119나 자살예방상담전화 109에 바로 연락해 주세요. [CRISIS]",
        "지금은 대화를 이어가기보다 실제 도움과 연결되는 것이 먼저예요. 주변 사람에게 알리고 119 또는 109에 즉시 연락해 주세요. [CRISIS]",
        "당신의 안전이 가장 중요해요. 혼자 있지 말고 보호자, 친구, 가족, 119나 109 중 한 곳에 지금 바로 연결해 주세요. [CRISIS]",
        "지금 표현은 즉각적인 도움이 필요한 신호예요. 위험한 물건을 멀리하고, 가까운 사람이나 119, 109에 바로 도움을 요청해 주세요. [CRISIS]",
        "지금은 혼자 버티지 않는 것이 중요해요. 가까운 사람에게 지금 상황을 알리고 119 또는 109에 즉시 연락해 주세요. [CRISIS]",
        "이 생각이 강해진 상태라면 안전 계획보다 즉시 연결이 먼저예요. 주변 사람과 함께 있고 119나 109에 바로 연락해 주세요. [CRISIS]",
        "당장 안전한 공간과 사람이 필요해 보여요. 혼자 있지 말고 가까운 사람이나 119, 109에 지금 도움을 요청해 주세요. [CRISIS]",
        "이건 위험 신호로 봐야 해요. 지금 바로 누군가에게 알리고 119 또는 자살예방상담전화 109에 연결해 주세요. [CRISIS]",
        "지금은 당신을 지키는 행동이 최우선이에요. 위험한 물건을 멀리하고, 주변 사람이나 119, 109에 즉시 연락해 주세요. [CRISIS]",
        "혼자 견디려 하지 않았으면 해요. 바로 가까운 사람에게 알리고 119나 109에 도움을 요청해 주세요. [CRISIS]",
    ]
    return build_cross_samples("crisis_candidate", users, responses)


def build_style_samples(rng: random.Random) -> list[dict]:
    """
    역할: 모든 카테고리의 균형형 Qwen 응답 스타일 SFT 샘플을 생성한다.
    입력: 난수 객체
    출력: 목표 개수에 맞춘 SFT 샘플 리스트
    """
    all_candidates = []
    all_candidates.extend(build_casual_share_samples())
    all_candidates.extend(build_positive_share_samples())
    all_candidates.extend(build_routine_discomfort_samples())
    all_candidates.extend(build_emotional_distress_samples())
    all_candidates.extend(build_relationship_samples())
    all_candidates.extend(build_sleep_fatigue_samples())
    all_candidates.extend(build_preference_question_samples())
    all_candidates.extend(build_practical_question_samples())
    all_candidates.extend(build_crisis_candidate_samples())

    capped = []
    for category, limit in TARGET_COUNTS.items():
        capped.extend(cap_category(all_candidates, category, limit, rng))
    rng.shuffle(capped)
    return capped


def load_jsonl(path: str) -> list[dict]:
    """
    역할: JSONL 파일을 샘플 리스트로 읽는다.
    입력: JSONL 경로
    출력: 샘플 dict 리스트
    """
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def write_jsonl(path: str, samples: list[dict]) -> None:
    """
    역할: 샘플 리스트를 JSONL 파일로 저장한다.
    입력: 저장 경로, 샘플 리스트
    출력: 없음
    """
    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def extract_turn(sample: dict) -> tuple[str, str]:
    """
    역할: SFT 샘플에서 마지막 user/assistant 발화를 추출한다.
    입력: SFT 샘플 dict
    출력: (사용자 발화, assistant 응답)
    """
    messages = sample.get("messages", [])
    user_text = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")
    assistant_text = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "assistant"), "")
    return user_text, assistant_text


def has_echo(user_text: str, assistant_text: str, min_chunk: int = 6) -> bool:
    """
    역할: 사용자 발화 일부가 assistant 응답에 과하게 복사됐는지 검사한다.
    입력: 사용자 발화, assistant 응답, 최소 연속 글자 수
    출력: 에코 여부
    """
    user_norm = normalize(user_text)
    assistant_norm = normalize(assistant_text)
    if len(user_norm) < min_chunk:
        return False
    for idx in range(len(user_norm) - min_chunk + 1):
        if user_norm[idx:idx + min_chunk] in assistant_norm:
            return True
    return False


def is_crisis_sample(sample: dict) -> bool:
    """
    역할: assistant 응답의 [CRISIS] 태그 포함 여부를 확인한다.
    입력: SFT 샘플 dict
    출력: 위기 샘플 여부
    """
    _, assistant_text = extract_turn(sample)
    return "[CRISIS]" in assistant_text


def strict_keep_reason(sample: dict) -> tuple[bool, str]:
    """
    역할: 기존 cleaned 상담 샘플을 더 엄격한 기준으로 유지할지 판정한다.
    입력: SFT 샘플 dict
    출력: (유지 여부, 사유)
    """
    user_text, assistant_text = extract_turn(sample)
    if not user_text or not assistant_text:
        return False, "empty_turn"

    is_crisis = "[CRISIS]" in assistant_text
    min_len = 25 if is_crisis else 35
    max_len = 230 if is_crisis else 190
    if len(assistant_text) < min_len:
        return False, "too_short"
    if len(assistant_text) > max_len:
        return False, "too_long"
    if any(phrase in assistant_text for phrase in BANNED_PHRASES):
        return False, "transcript_phrase"
    if any(name in assistant_text for name in SUSPICIOUS_NAMES):
        return False, "suspicious_name"
    if not is_crisis and has_echo(user_text, assistant_text):
        return False, "echo"
    if not is_crisis and assistant_text.rstrip().endswith(("?", "요?", "까요?", "나요?")):
        empathic_markers = ["마음", "힘들", "지치", "버거", "불안", "속상", "외롭", "무겁"]
        if len(user_text) < 24 or not any(marker in assistant_text for marker in empathic_markers):
            return False, "question_end"
    if not is_crisis and assistant_text.count("?") >= 2:
        return False, "too_many_questions"
    return True, "keep"


def build_strict_cleaned(samples: list[dict]) -> tuple[list[dict], Counter]:
    """
    역할: cleaned 상담 데이터에서 녹취 잔재와 과한 질문형을 추가 제거한다.
    입력: cleaned SFT 샘플 리스트
    출력: (엄격 정제 샘플 리스트, 제거 사유 Counter)
    """
    kept = []
    dropped = Counter()
    for sample in samples:
        keep, reason = strict_keep_reason(sample)
        if keep:
            kept.append(sample)
        else:
            dropped[reason] += 1
    return kept, dropped


def sample_strict_base(samples: list[dict], rng: random.Random) -> list[dict]:
    """
    역할: 엄격 정제 상담 데이터에서 위기/비위기 비율을 제어해 혼합용 샘플을 추출한다.
    입력: 엄격 정제 샘플 리스트, 난수 객체
    출력: 혼합 학습에 넣을 상담 샘플 리스트
    """
    crisis_samples = [sample for sample in samples if is_crisis_sample(sample)]
    non_crisis_samples = [sample for sample in samples if not is_crisis_sample(sample)]
    rng.shuffle(crisis_samples)
    rng.shuffle(non_crisis_samples)

    crisis_pick = crisis_samples[:min(STRICT_BASE_CRISIS_SIZE, len(crisis_samples))]
    non_crisis_target = max(STRICT_BASE_SAMPLE_SIZE - len(crisis_pick), 0)
    non_crisis_pick = non_crisis_samples[:min(non_crisis_target, len(non_crisis_samples))]
    picked = non_crisis_pick + crisis_pick
    rng.shuffle(picked)
    return picked


def summarize_categories(samples: list[dict]) -> Counter:
    """
    역할: meta.category 기준으로 샘플 분포를 계산한다.
    입력: SFT 샘플 리스트
    출력: 카테고리별 Counter
    """
    counter = Counter()
    for sample in samples:
        counter[sample.get("meta", {}).get("category", "unknown")] += 1
    return counter


def write_report(report: dict) -> None:
    """
    역할: 생성 결과 리포트를 JSON으로 저장한다.
    입력: 리포트 dict
    출력: 없음
    """
    with open(REPORT_OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def main() -> None:
    """
    역할: 응답 스타일 SFT, 엄격 정제 상담 데이터, 혼합 학습 데이터를 생성한다.
    입력: 없음
    출력: 없음
    """
    rng = random.Random(RANDOM_SEED)
    cleaned_samples = load_jsonl(SRC_CLEANED)
    strict_samples, dropped = build_strict_cleaned(cleaned_samples)
    style_samples = build_style_samples(rng)
    sampled_base = sample_strict_base(strict_samples, rng)
    mixed_samples = sampled_base + style_samples
    rng.shuffle(mixed_samples)

    write_jsonl(STRICT_OUT, strict_samples)
    write_jsonl(STYLE_OUT, style_samples)
    write_jsonl(MIX_OUT, mixed_samples)

    report = {
        "src_cleaned": SRC_CLEANED,
        "strict_out": STRICT_OUT,
        "style_out": STYLE_OUT,
        "mix_out": MIX_OUT,
        "cleaned_total": len(cleaned_samples),
        "strict_total": len(strict_samples),
        "strict_drop_reasons": dict(dropped.most_common()),
        "style_total": len(style_samples),
        "style_categories": dict(summarize_categories(style_samples).most_common()),
        "mixed_total": len(mixed_samples),
        "mixed_strict_base": len(sampled_base),
        "mixed_style": len(style_samples),
        "mixed_categories": dict(summarize_categories(mixed_samples).most_common()),
    }
    write_report(report)

    print(f"[strict cleaned] {STRICT_OUT}")
    print(f"  - source={len(cleaned_samples)}")
    print(f"  - kept={len(strict_samples)}")
    for reason, count in dropped.most_common():
        print(f"  - drop {reason}: {count}")
    print(f"[style SFT] {STYLE_OUT}")
    print(f"  - samples={len(style_samples)}")
    for category, count in summarize_categories(style_samples).most_common():
        print(f"  - {category}: {count}")
    print(f"[mixed SFT] {MIX_OUT}")
    print(f"  - strict_base={len(sampled_base)}")
    print(f"  - response_style={len(style_samples)}")
    print(f"  - total={len(mixed_samples)}")
    print(f"[report] {REPORT_OUT}")


if __name__ == "__main__":
    main()
