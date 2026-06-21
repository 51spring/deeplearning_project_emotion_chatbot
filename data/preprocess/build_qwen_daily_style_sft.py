"""
build_qwen_daily_style_sft.py
역할: Qwen 응답 문법 안정화를 위한 일상 말투 SFT JSONL과 혼합 학습 JSONL을 생성한다.
입력: data/processed/qwen_finetune_cleaned.jsonl
출력:
  - data/processed/qwen_daily_style_sft.jsonl
  - data/processed/qwen_finetune_daily_mix.jsonl
실행: C:/Users/WD/anaconda3/envs/dl_study/python.exe -u data/preprocess/build_qwen_daily_style_sft.py
"""

import json
import os
import random
from collections import Counter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

SRC_CLEANED = os.path.join(PROCESSED_DIR, "qwen_finetune_cleaned.jsonl")
DAILY_OUT = os.path.join(PROCESSED_DIR, "qwen_daily_style_sft.jsonl")
MIX_OUT = os.path.join(PROCESSED_DIR, "qwen_finetune_daily_mix.jsonl")

RANDOM_SEED = 42
BASE_SAMPLE_SIZE = 1600
BASE_CRISIS_SAMPLE_SIZE = 400

SYSTEM_PROMPT = (
    "당신은 한국어 심리상담 보조 AI입니다. "
    "사용자의 현재 말에만 집중하고, 없는 상황을 지어내지 마세요. "
    "일상 잡담은 짧게 반응하고, 힘든 감정은 먼저 공감하세요. "
    "응답은 1~3문장의 자연스럽고 문법적인 한국어로 작성하세요."
)

CATEGORIES = {
    "casual_neutral": {
        "users": [
            "밥 먹었어",
            "아침 챙겨 먹었어",
            "점심은 대충 먹었어",
            "오늘 물 좀 마셨어",
            "산책하고 왔어",
            "퇴근하고 집에 왔어",
            "오늘 날씨 좋더라",
            "커피 한 잔 마셨어",
            "친구랑 잠깐 통화했어",
            "오늘은 빨래 돌렸어",
            "방 정리 조금 했어",
            "영화 한 편 봤어",
            "게임 조금 했어",
            "음악 들으면서 쉬었어",
            "일찍 씻었어",
            "오늘은 그냥 평범했어",
            "마트 다녀왔어",
            "버스 타고 집에 왔어",
            "잠깐 낮잠 잤어",
            "오늘 운동 조금 했어",
        ],
        "responses": [
            "잘 챙겼네요. 그런 작은 일상이 하루를 은근히 받쳐줘요.",
            "좋아요. 오늘도 자기 리듬을 조금은 지켜낸 것 같아요.",
            "평범하게 지나간 하루도 충분히 의미 있어요.",
            "작은 일 하나를 해낸 것도 꽤 괜찮은 흐름이에요.",
            "그 정도면 오늘의 생활 리듬을 잘 이어간 편이에요.",
        ],
    },
    "positive_sharing": {
        "users": [
            "오늘 기분 좋아",
            "나 오늘 꽤 뿌듯해",
            "시험 생각보다 잘 본 것 같아",
            "일이 잘 풀렸어",
            "친구가 칭찬해줬어",
            "오랜만에 웃었어",
            "운동하고 나니까 개운해",
            "오늘은 마음이 좀 편해",
            "작은 목표 하나 끝냈어",
            "생각보다 잘 버틴 하루였어",
        ],
        "responses": [
            "좋은 기운이 느껴져요. 오늘의 그 감각을 충분히 누려도 괜찮아요.",
            "그 말에서 뿌듯함이 전해져요. 오늘은 스스로를 조금 칭찬해도 좋겠어요.",
            "잘 풀린 순간이 있었다니 반가워요. 그런 기억은 마음에 잘 남겨두면 좋아요.",
            "오늘은 마음이 조금 가벼워진 날이었군요. 그 느낌을 편하게 느껴봐도 좋아요.",
            "작은 성취도 분명한 성취예요. 오늘 잘 해냈어요.",
        ],
    },
    "routine_discomfort": {
        "users": [
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
        ],
        "responses": [
            "그런 날 있죠. 오늘은 일단 시작만 해도 꽤 애쓴 거예요.",
            "하기 싫은 마음이 올라오는 날도 있어요. 너무 크게 몰아붙이지 않아도 돼요.",
            "부담이 먼저 느껴지는 순간이네요. 아주 작은 것부터 시작해도 충분해요.",
            "지금은 의욕보다 피로감이 먼저 올라온 것 같아요. 오늘은 최소한의 한 걸음만 잡아봐도 좋아요.",
            "마음이 선뜻 움직이지 않는 날이네요. 그래도 그 마음을 알아차린 것부터 시작이에요.",
        ],
    },
    "sleep_fatigue": {
        "users": [
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
        ],
        "responses": [
            "잠이 흔들리면 하루 전체가 훨씬 무겁게 느껴지죠. 오늘은 몸이 조금이라도 쉴 수 있는 쪽을 먼저 생각해봐요.",
            "계속 피곤한 상태가 이어지고 있군요. 지금은 의지보다 회복이 먼저 필요해 보여요.",
            "몸이 무겁게 느껴지는 날이네요. 오늘은 해야 할 일을 조금 줄여도 괜찮아요.",
            "생각이 많아져서 잠들기 어려웠군요. 그 시간이 꽤 길고 지치게 느껴졌을 것 같아요.",
            "쉰 것 같지 않은 피로가 계속되는군요. 지금 몸이 보내는 신호를 가볍게 넘기지 않았으면 해요.",
        ],
    },
    "emotional_distress": {
        "users": [
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
        ],
        "responses": [
            "계속 무거운 마음이 이어지고 있었군요. 어떤 순간이 제일 버겁게 느껴졌어요?",
            "스스로를 많이 몰아붙이고 있는 것 같아요. 지금은 그 마음을 조금 내려놓고 말해도 괜찮아요.",
            "혼자 감당하기엔 꽤 지치는 마음이었겠어요. 요즘 가장 자주 떠오르는 생각은 뭐예요?",
            "괜찮은 척하느라 많이 지쳤을 것 같아요. 여기서는 애써 괜찮은 척하지 않아도 돼요.",
            "마음이 계속 가라앉는 느낌이군요. 그 감정이 언제부터 더 뚜렷해졌는지 같이 살펴볼까요?",
        ],
    },
    "relationship": {
        "users": [
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
        ],
        "responses": [
            "관계에서 마음이 흔들리면 생각이 오래 남을 수 있어요. 지금은 스스로를 너무 몰아붙이지 않았으면 해요.",
            "상대의 반응 때문에 마음이 많이 쓰였군요. 어떤 말이나 장면이 제일 남아 있어요?",
            "사람 사이의 일은 단순하게 정리되지 않을 때가 많죠. 지금 느끼는 서운함도 충분히 이해돼요.",
            "계속 신경 쓰이는 일이 있었네요. 우선은 그 상황에서 느낀 감정을 차분히 봐도 괜찮아요.",
            "관계가 어렵게 느껴지는 날이군요. 혼자만의 문제라고 단정하지는 않았으면 해요.",
        ],
    },
    "crisis_safety": {
        "users": [
            "죽고 싶다는 생각이 들어",
            "나를 해치고 싶어",
            "사라지고 싶어",
            "살고 싶지 않아",
            "오늘 밤 끝내고 싶어",
            "손목을 긋고 싶다는 생각이 들어",
            "약을 잔뜩 먹고 싶어",
            "뛰어내리고 싶다는 생각이 들어",
        ],
        "responses": [
            "지금은 안전이 가장 먼저예요. 혼자 있지 말고 가까운 사람에게 바로 알리고 119나 자살예방상담전화 109에 즉시 도움을 요청해 주세요. [CRISIS]",
            "이 말은 그냥 넘기기 어려운 위기 신호예요. 지금 혼자 버티지 말고 주변 사람이나 119, 109에 바로 연결해 주세요. [CRISIS]",
            "지금은 감정을 혼자 견디는 것보다 안전한 사람과 연결되는 게 중요해요. 가까운 보호자나 119, 109에 즉시 도움을 요청해 주세요. [CRISIS]",
        ],
    },
}


def make_sample(user_text: str, assistant_text: str, category: str) -> dict:
    """
    역할: 단일 user/assistant 쌍을 Qwen SFT messages 형식으로 변환한다.
    입력: 사용자 발화, assistant 응답, 카테고리명
    출력: JSON 직렬화 가능한 학습 샘플 dict
    """
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ],
        "meta": {"source": "daily_style_sft", "category": category},
    }


def build_daily_samples() -> list[dict]:
    """
    역할: 카테고리별 사용자 발화와 안정된 한국어 응답 전체를 조합해 일상 말투 SFT 샘플을 만든다.
    입력: 없음
    출력: SFT 샘플 리스트
    """
    samples = []
    for category, data in CATEGORIES.items():
        users = data["users"]
        responses = data["responses"]
        for user_text in users:
            # 문법적으로 검수한 응답 후보를 모두 학습시켜 같은 의도에도 다양한 안정 문장을 배우게 한다.
            for response_text in responses:
                samples.append(make_sample(user_text, response_text, category))
    return deduplicate_samples(samples)


def deduplicate_samples(samples: list[dict]) -> list[dict]:
    """
    역할: 동일 user/assistant 쌍 중복을 제거한다.
    입력: SFT 샘플 리스트
    출력: 중복 제거된 리스트
    """
    seen = set()
    unique = []
    for sample in samples:
        user = sample["messages"][1]["content"]
        assistant = sample["messages"][2]["content"]
        key = (user, assistant)
        if key in seen:
            continue
        seen.add(key)
        unique.append(sample)
    return unique


def load_jsonl(path: str) -> list[dict]:
    """
    역할: JSONL 파일을 읽어 샘플 리스트로 반환한다.
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


def is_crisis_sample(sample: dict) -> bool:
    """
    역할: assistant 응답에 [CRISIS] 태그가 포함된 학습 샘플인지 판별한다.
    입력: SFT 샘플 dict
    출력: 위기 샘플 여부
    """
    messages = sample.get("messages", [])
    if not messages:
        return False
    return "[CRISIS]" in str(messages[-1].get("content", ""))


def sample_base_data(samples: list[dict], size: int, crisis_size: int, rng: random.Random) -> list[dict]:
    """
    역할: 기존 cleaned 상담 데이터에서 비위기/위기 비율을 제어해 혼합 학습용 샘플을 추출한다.
    입력: 전체 cleaned 샘플, 전체 추출 수, 위기 추출 수, 난수 객체
    출력: 추출된 샘플 리스트
    """
    crisis_samples = [sample for sample in samples if is_crisis_sample(sample)]
    non_crisis_samples = [sample for sample in samples if not is_crisis_sample(sample)]
    rng.shuffle(crisis_samples)
    rng.shuffle(non_crisis_samples)

    crisis_pick = crisis_samples[:min(crisis_size, len(crisis_samples))]
    non_crisis_target = max(size - len(crisis_pick), 0)
    non_crisis_pick = non_crisis_samples[:min(non_crisis_target, len(non_crisis_samples))]

    picked = non_crisis_pick + crisis_pick
    rng.shuffle(picked)
    return picked


def summarize(samples: list[dict]) -> Counter:
    """
    역할: daily_style_sft 샘플의 카테고리 분포를 계산한다.
    입력: SFT 샘플 리스트
    출력: 카테고리별 Counter
    """
    counter = Counter()
    for sample in samples:
        counter[sample.get("meta", {}).get("category", "unknown")] += 1
    return counter


def main() -> None:
    """
    역할: 일상 말투 SFT와 기존 cleaned 데이터 혼합 JSONL을 생성한다.
    입력: 없음
    출력: 없음
    """
    rng = random.Random(RANDOM_SEED)
    daily_samples = build_daily_samples()
    rng.shuffle(daily_samples)
    write_jsonl(DAILY_OUT, daily_samples)

    base_samples = load_jsonl(SRC_CLEANED)
    sampled_base = sample_base_data(base_samples, BASE_SAMPLE_SIZE, BASE_CRISIS_SAMPLE_SIZE, rng)
    mixed_samples = sampled_base + daily_samples
    rng.shuffle(mixed_samples)
    write_jsonl(MIX_OUT, mixed_samples)

    print(f"[daily SFT] {DAILY_OUT}")
    print(f"  - samples={len(daily_samples)}")
    for category, count in summarize(daily_samples).most_common():
        print(f"  - {category}: {count}")
    print(f"[mixed SFT] {MIX_OUT}")
    print(f"  - base_cleaned={len(sampled_base)}")
    print(f"  - daily_style={len(daily_samples)}")
    print(f"  - total={len(mixed_samples)}")


if __name__ == "__main__":
    main()
