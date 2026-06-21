"""
inference_qwen.py
역할: Qwen2.5-3B-Instruct (+ LoRA 어댑터) 상담 응답 생성
      4bit 양자화 로드, DoT 시스템 프롬프트 자동 삽입
      ⚠️ RoBERTa 언로드 후 호출할 것 (VRAM 제한)
실행: scheduler.py의 ModelScheduler.run_qwen()에서 호출
"""

import os
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ── 경로 ────────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CKPT_DIR  = os.path.join(BASE_DIR, "models", "qwen", "checkpoints")
LEGACY_LORA_CKPT = os.path.join(CKPT_DIR, "qwen_lora_best")
DAILY_STYLE_LORA_CKPT = os.path.join(CKPT_DIR, "qwen_lora_daily_style_v2")
RESPONSE_STYLE_LORA_CKPT = os.path.join(CKPT_DIR, "qwen_lora_response_style_v3")
DEFAULT_LORA_CKPT = (
    # response_style_v3는 데이터/학습 산출물로 보존하되 실제 생성 검증에서 녹취 잔재가 남아 기본값에서 제외한다.
    DAILY_STYLE_LORA_CKPT
    if os.path.isdir(DAILY_STYLE_LORA_CKPT)
    else LEGACY_LORA_CKPT
)
LORA_CKPT = os.getenv("QWEN_LORA_CKPT", DEFAULT_LORA_CKPT)

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
MAX_NEW_TOKENS = int(os.getenv("QWEN_MAX_NEW_TOKENS", "96"))

# ── 생성 파라미터 (env로 오버라이드 가능, 응답 붕괴/반복 억제) ─────────────────
# do_sample: 작은 모델은 샘플링에서 상담 녹취 잔재가 흔들려 나올 수 있어 기본값은 결정형 생성으로 둔다.
# repetition_penalty: 동일 토큰 재출현 페널티 (>1 강화). 1.1→1.2로 상향해
#   "장문 반복" 패턴 억제 (Qwen 1 epoch 파일럿 응답에서 관찰됨).
# no_repeat_ngram_size: 동일 N-gram 차단. 3-gram 이상 반복 금지로
#   "네? 아니죠? 네?" 류 짧은 회로 붕괴 직접 차단.
GEN_DO_SAMPLE         = os.getenv("QWEN_GEN_DO_SAMPLE", "0").lower() in {"1", "true", "yes", "y"}
GEN_TEMPERATURE       = float(os.getenv("QWEN_GEN_TEMPERATURE", "0.65"))
GEN_TOP_P             = float(os.getenv("QWEN_GEN_TOP_P", "0.88"))
GEN_REPETITION_PENALTY = float(os.getenv("QWEN_GEN_REPETITION_PENALTY", "1.15"))
GEN_NO_REPEAT_NGRAM    = int(os.getenv("QWEN_GEN_NO_REPEAT_NGRAM", "4"))

BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

SYSTEM_PROMPT = (
    "당신은 따뜻하고 공감 능력이 뛰어난 심리상담 보조 AI입니다.\n"
    "사용자의 감정 상태에 맞게 자연스럽게 대화하세요.\n\n"
    "[핵심 원칙]\n"
    "- 사용자가 긍정적이면 함께 기뻐하고 격려하세요.\n"
    "- 사용자가 힘들어하면 판단 없이 먼저 충분히 공감하세요.\n"
    "- 의학적 진단·처방·즉각적 해결책 강요는 하지 마세요.\n"
    "- 감정을 축소하거나 '그래도 좋은 면이 있잖아요' 식의 전환을 강요하지 마세요.\n\n"
    "[대화 스타일 — 매우 중요]\n"
    "- 사용자의 말을 그대로 따라 쓰거나 의문문으로 복사하지 마세요. (X: '운동했어요?' ← '오늘 운동했어')\n"
    "- 일상적인 짧은 잡담(운동, 식사, 날씨, 일과 등)에는 공감·리액션 한 문장으로 짧게 반응하고 질문을 붙이지 마세요.\n"
    "- 감정이 담긴 발화에 한해 자연스러울 때만 탐색 질문을 한 가지만 덧붙이세요. 억지 질문 금지.\n"
    "- 사용자가 언급하지 않은 이름(유비, 관우, 장비, 영희, 철수 등)을 임의로 부르지 마세요.\n"
    "- 상담 녹취 말투('네 좋습니다', '그다음에는', '자세히 좀 이야기 좀 해봐 주세요')를 사용하지 마세요.\n\n"
    "[위기 감지]\n"
    "- 자해·자살 의도가 명확히 감지되면 응답 마지막에 반드시 [CRISIS] 태그를 추가하세요.\n"
    "- 단순한 스트레스·좌절 표현에는 [CRISIS]를 붙이지 마세요.\n\n"
    "응답은 1~3문장으로 간결하고 자연스럽게 작성하세요.\n\n"
    "[예시]\n"
    "사용자: 오늘 운동했어\n"
    "응답: 오 꾸준히 움직이셨네요, 기특하세요.\n\n"
    "사용자: 철봉 매달리기 했는데 버티기 힘들더라\n"
    "응답: 매달리기 은근히 팔 많이 쓰죠. 오늘 그만큼 해낸 것만으로도 충분해요.\n\n"
    "사용자: 요즘 계속 우울해\n"
    "응답: 계속 이어지는 우울감이 많이 무거우셨겠어요. 어떤 일이 가장 마음을 눌러요?"
)

DIRECT_CRISIS_PATTERNS = [
    # 직접적 자해·자살 표현만 포함 (일상 발화 오탐 방지를 위해 모호한 표현 제외)
    "자해",
    "죽고 싶",
    "죽고싶",
    "죽어버릴",
    "죽을래",
    "죽어버려",
    "사라지고 싶어",
    "없어지고 싶어",
    "살고 싶지 않",
    "살기 싫",
    "손목을 그",
    "칼로 그",
    "약을 잔뜩",
    "뛰어내리",
    "목매",
    "스스로 목숨",
    "내 목숨을 끊",
    "스스로를 다치",
    "나를 다치",
]

DIRECT_CRISIS_RESPONSE = (
    "지금 말씀을 보면 안전이 가장 먼저 필요한 위기 상황으로 보여요. "
    "혼자 버티지 말고 가까운 보호자나 가족, 친구에게 바로 알리고 119나 자살예방상담전화 109에 즉시 도움을 요청해 주세요. "
    "가능하면 혼자 있지 말고, 주변의 위험한 물건도 잠시 멀리해 주세요. [CRISIS]"
)

QUALITY_MAX_CHARS = 180
QUALITY_MAX_SENTENCES = 3
ECHO_MIN_CHUNK = 5
NON_KOREAN_LEAK_RE = re.compile(r"[\u0600-\u06ff\u3040-\u30ff\uff66-\uff9d\u4e00-\u9fff]")

TRANSCRIPT_PHRASES = [
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
    "저라면 저 스스로",
    "극도의 불안이나 두려움",
    "그런 거 들고",
    "너무 많은 정보를",
    "정보를 듣으면",
    "불편하실 수도",
    "약하게 쉬우실",
    "근육을 다",
    "몇 번 더 들고",
    "많으셔요",
    "많으셔서",
    "잘 앉고",
    "나쁘게 느껴",
    "그러면 또 이렇게",
    "네 네",
    "그럼 혹시 근데",
    "아니면 그냥 이런 느낌",
    "이제 네",
    "네 그러면",
    "그럼 이제",
    "집에서 혼자 살",
    "움켜잡으셔서 그런 거예요",
    "그런 거예요",
    "그런 시간 되니깐",
    "커피 섭취",
    "새벽 세 시부터 근무",
    "안전하게 위험",
    "스스로를 너무 곤경",
    "몰아넣으려고",
    "위험을 느끼고 있어요",
    "선생님들이",
    "어린 연령층",
    "넣으라고 하더라고요",
    "하더라고요",
    "그러니깐",
    "내가 왔다고",
    "시간을 정해두고 와서",
    "반이 차압",
    "외부에서 평온하게",
    "온전하게 느껴지지",
    "어떻게 돼있나",
    "커브드",
    "오토그래프",
    "셋째 컵",
    "차워",
    "넣으니까",
    "막아야 하나",
    "꾸중",
    "마음을 위협",
    "위독",
    "그런 저번에도",
    "어떻게 그렇게 계속",
    "그때 느낌은",
    "밤 10시",
    "잠자기 전 활동",
    "저녁까지 시간이",
    "저도 이제",
    "위안을 느끼고 있어요",
    "틀림없이",
    "그런 일 있어요",
    "몸이 어떻게 느껴",
    "위안이 되더라고요",
    "구분해보세요",
    "지금의 상태가 충분합니다",
    "기억이 나지 않는 순간",
    "몇 번 있었나요",
    "꽤 꺼내놓으",
    "조금씩 넣어서",
    "그렇게 크게 느껴진 게 아니",
    "더군요",
    "세밀하게 들여다보",
    "또 다른 느낌이 올 수도",
    "위안을 느꼈나 봐",
    "그 순간을 기억하는 게",
    "잘 지나갔다는 걸 기억",
    "어떻게 살아왔는지",
    "다른 걸 해야겠다는",
    "충분함을 찾을 수 있었다",
    # 2026-05-05 holdout BAD/SKIPPED 샘플에서 잡힌 환각·비문·부적절 강도 어휘.
    # 환각 metaphor 계열
    "마음이 흉흉",
    "흉흉해서 시작",
    "마음을 너무 많이 채워",
    "마음이 너무 쉽게 들켜",
    "다 밀고 나갈",
    "스스로에게 안겨주",
    # 부적절 강도/위협 어휘 (사용자 발화가 위기가 아닌데 들어오는 케이스)
    "위력을 갖고",
    "위협감이 느껴",
    "위험한 상태",
    "안전 우선이 돼야",
    "안전하게 피드백",
    "위험이 느껴진 순간",
    "스스로를 너무 곤두",
    # 부적절 평가/판단
    "혼자 너무 나아지",
    "게으른 것 같",
    # 의미 없는 양적 probe / echo question 연쇄
    "지금까지 몇 번째",
    "몇 번이나 됐나요",
    "몇 번이나 되었",
    "몇 점 정도 되실까",
    "그런 느낌이 들었어요? 지금도",
    # therapy 녹취체 잔재
    "혼자 견디느라 너무 피곤",
    "혼자 견디느라 얼마나",
    "잘 견신",
    # 한·영 혼용 오타
    "아ching",
    # 2026-05-11 style×emotion 모니터링 review (chatbot_self_state 정형 24건):
    # 봇이 사용자 부정 발화에 긍정 misread를 붙이거나, 자기 자신을 발화 주체로
    # 끌어와 1인칭 상태/경험을 진술하는 패턴들.
    "혼자 있는 시간이 너무 좋",          # 부정 발화에 긍정 misread
    "혼자 있는 시간이 마음을 위태롭",     # 단일 사건에 과한 위협 부여
    "혼자서 겪고 있는 것이 너무",        # 봇이 사용자 경험을 과장 특성화
    "혼자서 겪고 있는 몸부림",            # 동일 패턴 — 몸부림 변형
    "혼자서 겪고 있는 피로",              # 동일 패턴 — 피로 변형
    "혼자 감당해야 할 문제가 너무 크",   # 일상 발화에 위협 강도 부여
    "혼자서 감당해야 할 문제가 너무 크", # "혼자서" 변형 (holdout #51)
    "혼자 감당해야 할 몫이 너무",        # over-severity 동족 변형
    "혼자서 감당해야 할 일이 너무 많",   # 동족 변형 (style 케이스)
    "내가 얼마나 위험하게 느껴",         # 봇 1인칭 위협 자각 (holdout #127)
    "혼자 견디느라 몸",                   # 신체 상태 봇 주입
    "몸이 조금 피곤하게 느껴졌던",       # 봇 fictitious probe (몸 상태 환각)
    "생각하는 게 조금 어렵더라",         # 봇 1인칭 self state
    "생각하는 게 이렇게 어렵게",         # 변형
    "생각하는 게 너무 위험",              # over-severity self state
    "내가 너무 자기 자신을 비판",        # 봇 1인칭 자기 평가
    "제가 느끼는 게",                    # 봇 1인칭 감정 진술
    "스스로에게 칭찬할 수 있",           # 봇 1인칭 자기 긍정
    "저도 그런 경험 있",                  # 봇 personal disclosure
]

CASUAL_OVERTHERAPY_PHRASES = [
    "돌아보",
    "기억하는 게",
    "충분함을 찾",
    "다른 걸 해야",
    "세밀하게 들여다",
    "또 다른 느낌",
    "위안을 느꼈",
    "위안을 느끼",
]

NEUTRAL_OVERPOSITIVE_PHRASES = [
    "기분을 너무 빨리",
    "너무 빨리 흘려보내",
    "나아진 감각",
    "나아진 느낌",
    "편해진 순간",
    "밝아진 흐름",
    "좋은 쪽으로 마음",
]

PLAIN_NEUTRAL_CONTEXT_MARKERS = [
    "특별하진",
    "특별하진 않았",
    "별일 없이",
    "무난",
    "평소랑",
    "크게 다르지",
    "그냥 오늘",
    "그냥 하루",
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

CASUAL_KEYWORDS = [
    "운동",
    "산책",
    "밥",
    "식사",
    "먹었",
    "날씨",
    "게임",
    "영화",
    "음악",
    "카페",
    "커피",
    "청소",
    "씻",
    "출근",
    "퇴근",
    "멜론",
    "스포티파이",
    "넷플릭스",
    "왓챠",
    "디즈니",
    "유튜브뮤직",
]

POSITIVE_KEYWORDS = [
    "좋아",
    "괜찮",
    "기뻐",
    "행복",
    "해냈",
    "성공",
    "뿌듯",
    "편안",
    "웃",
    "잘했",
    "기대",
    "설레",
    "신나",
    "기다려",
]

DISTRESS_KEYWORDS = [
    "힘들",
    "우울",
    "불안",
    "긴장",
    "이상한 기분",
    "기분이 이상",
    "찜찜",
    "찝찝",
    "싱숭생숭",
    "뒤숭숭",
    "기분이 별로",
    "기분 별로",
    "컨디션 별로",
    "무서",
    "슬퍼",
    "외로",
    "지쳤",
    "피곤",
    "잠이 안",
    "화나",
    "화가",
    "짜증",
    "억울",
    "무례",
    "분이 안",
    "속상",
    "버거",
]

EMPATHIC_RESPONSE_MARKERS = [
    "힘드",
    "지치",
    "버거",
    "무겁",
    "마음",
    "긴장",
    "속상",
    "고생",
    "애썼",
    "괜찮",
    "피곤",
    "회복",
]

POSITIVE_RESPONSE_MARKERS = [
    "좋",
    "기대",
    "설레",
    "신나",
    "반가",
    "멋지",
    "기쁘",
    "즐거",
    "응원",
]

POSITIVE_MISMATCH_PHRASES = [
    "필요해요",
    "필요해",
    "쉬어",
    "쉬는",
    "쉬워질",
    "자고",
    "먹고",
    "회복",
    "무리하지",
]

STALE_FALLBACK_PHRASES = [
    "좋아요, 오늘의 작은 일상도 잘 지나가고 있네요.",
]

QUESTION_ENDINGS = ("?", "요?", "까?", "나요?", "세요?")

PREFERENCE_QUESTION_MARKERS = [
    "중에 뭐",
    "중 뭐",
    "뭐가 더",
    "뭐가 나아",
    "뭐가 좋아",
    "추천",
    "고르면",
    "골라",
]

UTTERANCE_STYLE_GUIDES = {
    "casual_neutral": (
        "현재 발화 타입: casual_neutral.\n"
        "응답 지침: 일상 공유나 가벼운 잡담으로 보고 1문장으로 짧게 반응하세요. "
        "상담처럼 깊게 파고들지 말고, 질문은 붙이지 마세요."
    ),
    "casual_share": (
        "현재 발화 타입: casual_share.\n"
        "응답 지침: 감정 상담으로 과하게 해석하지 말고, 일상 공유에 짧고 자연스럽게 반응하세요. "
        "질문은 붙이지 마세요."
    ),
    "positive_share": (
        "현재 발화 타입: positive_share.\n"
        "응답 지침: 사용자의 좋은 기분, 기대, 성취를 함께 반겨 주세요. "
        "휴식이나 문제 해결 조언으로 돌리지 마세요."
    ),
    "routine_discomfort": (
        "현재 발화 타입: routine_discomfort.\n"
        "응답 지침: 출근·공부·운동·집안일 같은 일상 불편으로 보고 가볍게 공감하세요. "
        "우울·위기처럼 과장하지 말고 1~2문장으로 답하세요."
    ),
    "emotional_distress": (
        "현재 발화 타입: emotional_distress.\n"
        "응답 지침: 실제 정서적 어려움으로 보고 감정을 먼저 반영하세요. "
        "조언을 서두르지 말고 필요할 때만 탐색 질문을 하나 덧붙이세요."
    ),
    "crisis_candidate": (
        "현재 발화 타입: crisis_candidate.\n"
        "응답 지침: 안전을 최우선으로 확인하세요. 자해·자살 의도가 명확하면 [CRISIS] 태그를 유지하세요."
    ),
    "preference_question": (
        "현재 발화 타입: preference_question.\n"
        "응답 지침: 감정 상담으로 돌리지 말고, 사용자가 제시한 선택지를 간단히 비교해 추천하세요. "
        "최신 가격·정책처럼 불확실한 정보는 단정하지 말고 일반 기준으로 말하세요."
    ),
    "practical_question": (
        "현재 발화 타입: practical_question.\n"
        "응답 지침: 실용적인 질문으로 보고 바로 도움이 되는 답을 짧게 제시하세요. "
        "상담 녹취 말투나 사용자가 말하지 않은 상황 추측은 피하세요."
    ),
}
DEFAULT_UTTERANCE_STYLE_GUIDE = (
    "현재 발화 타입: unknown.\n"
    "응답 지침: 현재 발화에만 집중하고, 사용자가 말하지 않은 상황을 지어내지 마세요."
)

# 모듈 수준 싱글턴 — scheduler가 명시적으로 unload 관리
_model     = None
_tokenizer = None


def load_qwen():
    """
    역할: Qwen 모델 + LoRA 어댑터 로드 (미로드 시에만 실행)
    출력: (model, tokenizer)
    """
    global _model, _tokenizer

    if _model is not None:
        return _model, _tokenizer

    # 학습 완료 후 저장된 tokenizer 파일이 있으면 우선 사용해
    # 오프라인 환경에서도 추가 네트워크 조회 없이 추론 가능하게 한다.
    tokenizer_source = LORA_CKPT if os.path.isdir(LORA_CKPT) else MODEL_NAME
    _tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        trust_remote_code=True,
        local_files_only=True,
    )
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    # LoRA 체크포인트가 있으면 어댑터 로드, 없으면 베이스 모델 사용
    if os.path.isdir(LORA_CKPT):
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=BNB_CONFIG,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
        )
        _model = PeftModel.from_pretrained(base, LORA_CKPT)
        print(f"[Qwen 로드] LoRA 어댑터: {LORA_CKPT}")
    else:
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=BNB_CONFIG,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
        )
        print(f"[Qwen 로드] 베이스 모델 (LoRA 없음): {MODEL_NAME}")

    _model.eval()
    return _model, _tokenizer


def unload_qwen():
    """역할: Qwen 모델 메모리 해제"""
    global _model, _tokenizer
    import gc
    if _model is not None:
        del _model
        del _tokenizer
        _model     = None
        _tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[Qwen 언로드] 완료")


def is_direct_crisis_text(user_text: str) -> bool:
    """
    역할: 직접적 자해·자살 위험 표현이 포함됐는지 규칙 기반으로 판별한다.
    입력: 사용자 발화 텍스트
    출력: 직접적 위기 여부 (bool)
    """
    normalized_text = user_text.replace(" ", "").lower()

    for pattern in DIRECT_CRISIS_PATTERNS:
        if pattern.replace(" ", "").lower() in normalized_text:
            return True

    return False


def _normalize_for_quality(text: str) -> str:
    """
    역할: 품질 검사용으로 공백과 구두점을 제거하고 소문자화한다.
    입력: 원문 텍스트
    출력: 정규화된 텍스트
    """
    return re.sub(r"[\s\.,!?~·…'\"“”‘’()\[\]{}]+", "", text).lower()


def _contains_echo(user_text: str, response_text: str) -> bool:
    """
    역할: 사용자 발화 일부가 응답에 그대로 반복됐는지 검사한다.
    입력: 사용자 발화, 모델 응답
    출력: 에코 패턴 여부
    """
    user_norm = _normalize_for_quality(user_text)
    response_norm = _normalize_for_quality(response_text)
    if len(user_norm) < ECHO_MIN_CHUNK:
        return False

    for idx in range(len(user_norm) - ECHO_MIN_CHUNK + 1):
        if user_norm[idx:idx + ECHO_MIN_CHUNK] in response_norm:
            return True
    return False


def _has_non_korean_leak(text: str) -> bool:
    """
    역할: 일본어·중국어·아랍 문자 등 한국어 상담 응답에 부적절한 문자 유출을 감지한다.
    입력: 모델 응답 텍스트
    출력: 다국어 문자 유출 여부
    """
    # Qwen이 한국어 문장 끝에 일본어 kana 종결어를 섞는 경우가 있어 정규식으로 즉시 차단한다.
    return bool(NON_KOREAN_LEAK_RE.search(text))


def _is_short_casual_user_text(user_text: str) -> bool:
    """
    역할: 짧은 일상 잡담 발화인지 휴리스틱으로 판별한다.
    입력: 사용자 발화 텍스트
    출력: 짧은 일상 발화 여부
    """
    stripped = user_text.strip()
    if len(stripped) > 24:
        return False
    return any(keyword in stripped for keyword in CASUAL_KEYWORDS)


def _has_distress_signal(user_text: str) -> bool:
    """
    역할: 발화에 정서적 어려움 신호가 있는지 판별한다.
    입력: 사용자 발화 텍스트
    출력: 어려움 신호 여부
    """
    return any(keyword in user_text for keyword in DISTRESS_KEYWORDS)


def _is_positive_user_text(user_text: str) -> bool:
    """
    역할: 발화가 긍정 정서나 성취 공유에 가까운지 판별한다.
    입력: 사용자 발화 텍스트
    출력: 긍정 발화 여부
    """
    return any(keyword in user_text for keyword in POSITIVE_KEYWORDS)


def _is_preference_question_text(user_text: str) -> bool:
    """
    역할: 선택지 비교나 취향 기반 추천 질문인지 휴리스틱으로 판별한다.
    입력: 사용자 발화 텍스트
    출력: 추천/비교 질문 여부
    """
    stripped = user_text.strip()
    if not stripped.endswith(QUESTION_ENDINGS):
        return False
    return any(marker in stripped for marker in PREFERENCE_QUESTION_MARKERS)


def _is_practical_question_text(user_text: str) -> bool:
    """
    역할: 메뉴·수면·집중처럼 바로 실행 가능한 조언을 묻는 실용 질문인지 판별한다.
    입력: 사용자 발화 텍스트
    출력: 실용 질문 여부
    """
    stripped = user_text.strip()
    if not stripped.endswith(QUESTION_ENDINGS):
        return False
    practical_markers = [
        "뭐 먹",
        "메뉴",
        "뭐 하지",
        "어떻게 하지",
        "뭐 하면",
        "방법",
        "추천해줘",
        "얼마나 할까",
        "어떻게 세워",
        "긴장 완화",
        "완화 방법",
        "불안감 해소",
        "불안 해소",
        "불안 완화",
        "불안 줄",
        "불안 낮",
        "불안 진정",
    ]
    return any(marker in stripped for marker in practical_markers)


def _is_anxiety_relief_question_text(user_text: str) -> bool:
    """
    역할: 불안 자체를 상담 호소로만 보지 않고 불안 완화 방법을 묻는 질문인지 판별한다.
    입력: 사용자 발화 텍스트
    출력: 불안 완화 질문 여부
    """
    stripped = user_text.strip()
    if not stripped.endswith(QUESTION_ENDINGS):
        return False
    compact = _normalize_for_quality(stripped)
    relief_markers = ["해소", "완화", "줄이", "낮추", "진정", "방법", "뭐하면", "어떻게"]
    return ("불안" in compact or "긴장" in compact) and any(
        marker in compact for marker in relief_markers
    )


def _is_mild_unease_text(user_text: str) -> bool:
    """
    역할: 막연한 이상함·찜찜함·뒤숭숭함을 말하는 발화인지 판별한다.
    입력: 사용자 발화 텍스트
    출력: 막연한 불편감 발화 여부
    """
    compact = _normalize_for_quality(user_text)
    unease_markers = [
        "이상한기분",
        "기분이이상",
        "묘한기분",
        "찜찜",
        "찝찝",
        "싱숭생숭",
        "뒤숭숭",
        "불편한기분",
    ]
    return any(marker in compact for marker in unease_markers)


def _is_mild_low_mood_text(user_text: str) -> bool:
    """
    역할: 가벼운 저조감·컨디션 저하를 말하는 발화인지 판별한다.
    입력: 사용자 발화 텍스트
    출력: 가벼운 저조감 발화 여부
    """
    compact = _normalize_for_quality(user_text)
    low_mood_markers = [
        "기분이별로",
        "기분별로",
        "기분이안좋",
        "기분안좋",
        "컨디션별로",
        "오늘별로",
        "좀별로",
        "별로야",
        "별로다",
    ]
    return any(marker in compact for marker in low_mood_markers)


def _is_anger_text(user_text: str) -> bool:
    """
    역할: 화남·억울함·짜증이 중심인 발화인지 판별한다.
    입력: 사용자 발화 텍스트
    출력: 분노 발화 여부
    """
    compact = _normalize_for_quality(user_text)
    anger_markers = [
        "화났",
        "화가났",
        "화가나",
        "짜증",
        "억울",
        "무례",
        "분이안풀",
        "무시",
        "책임을떠넘",
        "사과도안",
    ]
    return any(marker in compact for marker in anger_markers)


def _is_academic_anxiety_text(user_text: str) -> bool:
    """
    역할: 시험·면접·발표 등 평가 상황을 앞둔 긴장·불안 발화인지 판별한다.
    입력: 사용자 발화 텍스트
    출력: 평가 상황 예기불안 여부
    """
    compact = _normalize_for_quality(user_text)
    context_markers = [
        "시험",
        "수능",
        "중간고사",
        "기말고사",
        "면접",
        "발표",
        "평가",
        "과제",
        "마감",
    ]
    anxiety_markers = ["긴장", "불안", "걱정", "초조", "떨려", "부담", "무섭"]
    return any(marker in compact for marker in context_markers) and any(
        marker in compact for marker in anxiety_markers
    )


def _is_entertainment_request_text(user_text: str) -> bool:
    """
    역할: 심심함·재미·놀거리·볼거리 추천을 묻는 가벼운 일상 질문인지 판별한다.
    입력: 사용자 발화 텍스트
    출력: 엔터테인먼트 추천 요청 여부
    """
    stripped = user_text.strip()
    entertainment_markers = [
        "재밌는",
        "재미있는",
        "심심",
        "뭐 없",
        "볼만한",
        "볼 거",
        "볼거",
        "놀거리",
        "할만한",
        "킬링타임",
    ]
    return any(marker in stripped for marker in entertainment_markers)


def _shares_casual_keyword(user_text: str, response_text: str) -> bool:
    """
    역할: 짧은 일상 발화의 핵심 키워드를 응답이 그대로 반복하는지 검사한다.
    입력: 사용자 발화, 응답 텍스트
    출력: 일상 키워드 반복 여부
    """
    return any(keyword in user_text and keyword in response_text for keyword in CASUAL_KEYWORDS)


def _split_sentences(text: str) -> list[str]:
    """
    역할: 응답을 간단한 문장 단위로 분리한다.
    입력: 응답 텍스트
    출력: 문장 리스트
    """
    candidates = re.split(r"(?<=[.!?。！？])\s+|\n+", text.strip())
    return [candidate.strip() for candidate in candidates if candidate.strip()]


def _clean_generated_text(response_text: str) -> str:
    """
    역할: role 라벨, 과한 공백, 학습 포맷 잔여물을 제거한다.
    입력: 모델 원문 응답
    출력: 정리된 응답 텍스트
    """
    cleaned = response_text.strip()
    cleaned = re.sub(r"^(assistant|상담사|응답)\s*[:：]\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    sentences = _split_sentences(cleaned)
    if len(sentences) > QUALITY_MAX_SENTENCES:
        cleaned = " ".join(sentences[:QUALITY_MAX_SENTENCES])
    if len(cleaned) > QUALITY_MAX_CHARS:
        clipped = cleaned[:QUALITY_MAX_CHARS].rstrip()
        cleaned = re.sub(r"[,，:：;；-]\s*$", "", clipped) + "..."
    return cleaned


def _is_low_quality_response(user_text: str, response_text: str) -> bool:
    """
    역할: 생성 응답이 사용자에게 보여주기 어려운 품질인지 판별한다.
    입력: 사용자 발화, 정리된 응답
    출력: 저품질 여부
    """
    if not response_text or len(response_text) < 6:
        return True
    if _has_non_korean_leak(response_text):
        return True
    if any(name in response_text for name in SUSPICIOUS_NAMES):
        return True
    if any(phrase in response_text for phrase in STALE_FALLBACK_PHRASES):
        return True

    response_norm = _normalize_for_quality(response_text)
    if any(_normalize_for_quality(phrase) in response_norm for phrase in TRANSCRIPT_PHRASES):
        return True
    if response_text.count("네") >= 3 or response_text.count("이제") >= 3:
        return True
    if re.search(r"(.{2,8})\1{2,}", response_norm):
        return True

    # 짧은 일상 공유에 되묻기만 하는 패턴은 사용자가 보고한 대표 붕괴 케이스라 차단한다.
    if _is_short_casual_user_text(user_text) and _shares_casual_keyword(user_text, response_text):
        return True
    if _is_short_casual_user_text(user_text) and response_text.rstrip().endswith(QUESTION_ENDINGS):
        return True
    if (_is_short_casual_user_text(user_text) or _is_positive_user_text(user_text)) and len(response_text) > 90:
        return True
    if (_is_short_casual_user_text(user_text) or _is_positive_user_text(user_text)) and _has_distress_signal(response_text):
        return True
    if _is_positive_user_text(user_text) and "이제" in response_text:
        return True
    if _is_positive_user_text(user_text):
        has_positive_response = any(marker in response_text for marker in POSITIVE_RESPONSE_MARKERS)
        has_mismatch_phrase = any(phrase in response_text for phrase in POSITIVE_MISMATCH_PHRASES)
        if has_mismatch_phrase and not has_positive_response:
            return True
    if _is_preference_question_text(user_text):
        if not any(keyword in response_text for keyword in ["추천", "좋", "나아", "편", "멜론", "스포티파이"]):
            return True
    if _is_entertainment_request_text(user_text):
        entertainment_response_markers = ["영상", "예능", "영화", "게임", "산책", "음악", "가볍게", "재밌"]
        if not any(marker in response_text for marker in entertainment_response_markers):
            return True
    if _is_practical_question_text(user_text):
        if _is_anxiety_relief_question_text(user_text):
            anxiety_response_markers = ["불안", "숨", "호흡", "안정", "물", "산책", "몸"]
            if not any(marker in response_text for marker in anxiety_response_markers):
                return True
        if not any(keyword in response_text for keyword in ["추천", "좋", "먼저", "가볍", "먹", "메뉴", "해봐"]):
            return True
    if _has_distress_signal(user_text) and not any(marker in response_text for marker in EMPATHIC_RESPONSE_MARKERS):
        return True
    if _contains_echo(user_text, response_text) and not _has_distress_signal(user_text):
        return True

    return False


def _violates_typed_style(
    response_text: str,
    utterance_info: dict | None = None,
    user_text: str | None = None,
) -> bool:
    """
    역할: 발화 타입별 응답 스타일 제약을 어겼는지 판별한다.
    입력: 정리된 응답 텍스트, 발화 타입 정보, 사용자 발화
    출력: 타입 스타일 위반 여부
    """
    utterance_type = (utterance_info or {}).get("utterance_type")

    # 일상 공유·긍정 공유·가벼운 일상 불편은 상담 질문으로 이어가지 않고 짧게 반응한다.
    if utterance_type in {"casual_neutral", "casual_share", "positive_share", "routine_discomfort"}:
        if any(marker in response_text for marker in CASUAL_OVERTHERAPY_PHRASES):
            return True
        if utterance_type in {"casual_neutral", "casual_share"} and any(
            marker in response_text for marker in NEUTRAL_OVERPOSITIVE_PHRASES
        ):
            return True
        if user_text:
            if (
                any(marker in user_text for marker in PLAIN_NEUTRAL_CONTEXT_MARKERS)
                and any(marker in response_text for marker in NEUTRAL_OVERPOSITIVE_PHRASES)
            ):
                return True
            if "출근" in user_text and any(marker in response_text for marker in ["쉬길 바라는 마음", "집에서 쉬고 싶", "힘든 시간"]):
                return True
            if utterance_type == "positive_share" and any(marker in user_text for marker in ["기대", "설레", "기다려"]):
                return not any(marker in response_text for marker in ["기대", "설렘", "설레", "기다려", "좋은 기분"])
        return any(ending in response_text for ending in QUESTION_ENDINGS)

    if utterance_type == "emotional_distress" and user_text:
        if _is_academic_anxiety_text(user_text):
            return not any(marker in response_text for marker in ["시험", "긴장", "불안", "준비", "차분"])
        if _is_mild_unease_text(user_text):
            return not any(marker in response_text for marker in ["이상", "찜찜", "느낌", "기분", "숨", "몸", "긴장"])
        if _is_mild_low_mood_text(user_text):
            return not any(marker in response_text for marker in ["기분", "별로", "마음", "컨디션", "쉬", "괜찮"])
        if ("잠" in user_text or "불면" in user_text) and not any(
            marker in response_text for marker in ["잠", "수면", "피곤", "몸", "쉬", "회복"]
        ):
            return True

    # 비교·추천 질문에는 상담식 공감만 하지 말고 사용자가 제시한 선택지를 직접 다뤄야 한다.
    if utterance_type == "preference_question":
        if any(marker in response_text for marker in ["마음을 먼저", "힘드셨", "상담"]):
            return True
        if user_text and "멜론" in user_text and "스포티파이" in user_text:
            return "멜론" not in response_text and "스포티파이" not in response_text
        return not any(marker in response_text for marker in ["추천", "고르", "나아", "좋", "편"])

    if utterance_type == "practical_question":
        if any(marker in response_text for marker in ["저 지금", "제가", "먹지 않았"]):
            return True
        if user_text and _is_anxiety_relief_question_text(user_text):
            return not any(marker in response_text for marker in ["불안", "숨", "호흡", "안정", "물", "산책", "몸"])
        if user_text and "잠" in user_text:
            return not any(marker in response_text for marker in ["불", "화면", "스트레칭", "물", "잠"])
        if user_text and ("뭐 먹" in user_text or "메뉴" in user_text):
            return not any(marker in response_text for marker in ["김밥", "샐러드", "국밥", "덮밥", "메뉴", "가볍게", "든든하게"])

    return False


def _normalize_response_for_repeat(text: str) -> str:
    """
    역할: 반복 답변 비교를 위해 공백·문장부호를 줄인 비교용 문자열 생성
    입력: 응답 텍스트
    출력: 정규화된 비교 문자열
    """
    return re.sub(r"[\W_]+", "", (text or "").lower())


def _char_ngrams(text: str, n: int = 3) -> set[str]:
    """
    역할: 짧은 한국어 문장의 유사도 비교용 문자 n-gram 생성
    입력: 정규화된 텍스트, n-gram 길이
    출력: 문자 n-gram 집합
    """
    if len(text) < n:
        return {text} if text else set()
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def _is_similar_to_recent_response(candidate: str, avoid_texts: list[str] | None = None) -> bool:
    """
    역할: 새 응답이 최근 노출 응답과 거의 같은지 판정
    입력: 후보 응답, 회피할 최근 응답 목록
    출력: 반복으로 볼지 여부
    """
    cand_norm = _normalize_response_for_repeat(candidate)
    if not cand_norm or not avoid_texts:
        return False

    cand_grams = _char_ngrams(cand_norm)
    for old in avoid_texts:
        old_norm = _normalize_response_for_repeat(old)
        if not old_norm:
            continue
        if cand_norm == old_norm:
            return True
        if len(cand_norm) >= 18 and (cand_norm in old_norm or old_norm in cand_norm):
            return True
        old_grams = _char_ngrams(old_norm)
        union = cand_grams | old_grams
        if union and len(cand_grams & old_grams) / len(union) >= 0.78:
            return True
    return False


def _pick_fallback_variant(
    user_text: str,
    variants: list[str],
    avoid_texts: list[str] | None = None,
) -> str:
    """
    역할: 같은 fallback 문장이 계속 반복되지 않도록 발화 텍스트 기반으로 문장 선택
    입력: 사용자 발화, 후보 문장 리스트, 최근 노출 응답 목록
    출력: 선택된 fallback 문장
    """
    if not variants:
        return "말해줘서 고마워요. 지금 이야기의 흐름을 차분히 따라가 볼게요."
    idx = sum(ord(ch) for ch in user_text) % len(variants)
    for offset in range(len(variants)):
        candidate = variants[(idx + offset) % len(variants)]
        if not _is_similar_to_recent_response(candidate, avoid_texts):
            return candidate
    return "방금 흐름은 이어서 기록해둘게요. 지금 말한 내용은 이전 답변과 따로 보지 않고 함께 볼게요."


def _fallback_response(
    user_text: str,
    utterance_info: dict | None = None,
    avoid_texts: list[str] | None = None,
    *,
    distress_severity_scalar: float | None = None,
    distress_top_label: str | None = None,
) -> str:
    """
    역할: Qwen 응답이 품질 게이트를 통과하지 못했을 때 사용할 안전한 대체 응답 생성
    입력:
        user_text: 사용자 발화 텍스트
        utterance_info: 발화 타입 정보
        avoid_texts: 최근 노출 응답 목록
        distress_severity_scalar: Phase 5b distress head 출력 (0~1, 옵션)
        distress_top_label: distress 5클래스 top 라벨 (옵션, calm_or_positive/mild/moderate/high/crisis_candidate)
    출력: 짧고 자연스러운 대체 응답
    설명:
        Phase 5c — distress_top_label이 high/crisis 면 emotional_distress 분기에서
        가장 안전한 variant(index 0, 일반적으로 가장 보수적 톤)를 우선 선택.
        utterance_info 분기 자체는 변경하지 않고, _pick_fallback_variant 진입 직전의
        분기 안에서만 distress 정보를 참고한다.
    """
    utterance_type = (utterance_info or {}).get("utterance_type")
    # Phase 5c — distress severity 우선순위 힌트 (None이면 영향 없음)
    _is_high_distress = distress_top_label in {"high_distress", "crisis_candidate"} or (
        distress_severity_scalar is not None and distress_severity_scalar >= 0.65
    )
    _is_calm = distress_top_label == "calm_or_positive" or (
        distress_severity_scalar is not None and distress_severity_scalar < 0.20
    )
    if _is_academic_anxiety_text(user_text) and not _is_practical_question_text(user_text):
        return _pick_fallback_variant(user_text, [
            "시험을 앞두고 긴장되는 건 자연스러워요. 지금은 남은 범위를 전부 떠올리기보다, 바로 볼 수 있는 한 가지부터 차분히 잡아봐요.",
            "시험 전에는 생각이 크게 부풀 수 있어요. 오늘은 가장 가까운 범위 하나만 정해서 차분히 붙잡아봐요.",
            "긴장이 올라온 만큼 시험을 중요하게 느끼고 있다는 뜻일 수 있어요. 지금은 작은 단위로 준비를 나누는 게 좋아요.",
        ], avoid_texts=avoid_texts)
    if _is_entertainment_request_text(user_text):
        return _pick_fallback_variant(user_text, [
            "가볍게 기분 전환하고 싶으면 짧은 예능 클립이나 영화 한 편, 아니면 산책하면서 음악 듣는 걸 추천해요.",
            "머리를 잠깐 돌리고 싶으면 짧은 영상, 가벼운 게임 한 판, 좋아하는 음악 틀고 산책하기 중 하나가 무난해요.",
            "지금 바로 하기 쉬운 쪽으로는 짧은 코미디 영상이나 익숙한 음악이 좋아요. 에너지가 조금 있으면 산책도 괜찮고요.",
        ], avoid_texts=avoid_texts)
    if _is_anxiety_relief_question_text(user_text):
        return _pick_fallback_variant(user_text, [
            "불안감을 조금 낮추고 싶다면 먼저 숨을 천천히 내쉬고, 물 한 잔이나 짧은 산책처럼 몸을 안정시키는 행동부터 해봐요.",
            "불안이 올라올 때는 생각을 바로 해결하려 하기보다 호흡을 길게 내쉬고 몸의 긴장을 낮추는 것부터 해봐요.",
            "지금 할 수 있는 건 작게 시작하는 거예요. 숨을 고르고, 물을 마시고, 주변을 천천히 확인해봐요.",
        ], avoid_texts=avoid_texts)
    if _is_mild_unease_text(user_text):
        return _pick_fallback_variant(user_text, [
            "이상한 느낌이 들면 괜히 신경이 쓰일 수 있어요. 잠깐 숨을 고르고, 몸이 긴장했는지부터 천천히 살펴봐요.",
            "딱 설명하기 어려운 기분도 하루 안에 남을 수 있어요. 지금은 그 느낌을 크게 키우기보다 몸 상태부터 천천히 확인해봐요.",
            "묘하게 불편한 감각이 올라왔군요. 잠깐 멈춰서 숨을 고르고, 지금 가장 신경 쓰이는 부분만 작게 짚어봐요.",
        ], avoid_texts=avoid_texts)
    if _is_mild_low_mood_text(user_text):
        return _pick_fallback_variant(user_text, [
            "기분이 별로인 날도 있죠. 지금은 너무 잘해내려고 하기보다, 조금 쉬어 가도 괜찮아요.",
            "마음이 낮게 가라앉은 날에는 작은 일도 더 무겁게 느껴질 수 있어요. 오늘은 부담을 조금 낮춰봐요.",
            "컨디션이 마음까지 끌어내릴 때가 있어요. 지금은 해야 할 일을 작게 줄이고 쉬어 갈 틈을 만드는 게 좋아요.",
        ], avoid_texts=avoid_texts)
    if utterance_type == "preference_question" or _is_preference_question_text(user_text):
        if "멜론" in user_text and "스포티파이" in user_text:
            return "국내 음원 차트나 한국곡 위주로 들으면 멜론이 편하고, 해외 음악과 추천 플레이리스트를 많이 쓰면 스포티파이가 좋아요. 다양하게 듣는 편이면 스포티파이를 먼저 추천할게요."
        # 실제 비교 맥락이 있을 때만 "둘 중 고르라면" 폴백 사용
        # 없으면 아래 casual_neutral 분기로 넘긴다 (예: "오늘 뭐할까" 오탐 방어)
        _comparison_signals = ["중에", "아니면", "둘 중", "이냐", "vs", "VS"] + PREFERENCE_QUESTION_MARKERS
        if any(s in user_text for s in _comparison_signals):
            return _pick_fallback_variant(user_text, [
                "둘 중 고르라면, 자주 쓰는 기준을 먼저 보면 좋아요. 익숙함과 국내 콘텐츠가 중요하면 앞쪽 선택지, 추천 알고리즘이나 다양한 탐색이 중요하면 뒤쪽 선택지가 더 잘 맞을 수 있어요.",
                "선택 기준을 하나만 잡으면 더 쉬워요. 익숙함을 원하면 친숙한 쪽, 새롭게 탐색하고 싶으면 추천이 강한 쪽을 먼저 써보면 좋아요.",
                "둘 다 괜찮다면 지금 가장 자주 쓸 상황을 기준으로 고르면 돼요. 매일 쓰기 편한 쪽이 결국 만족도가 높아요.",
            ], avoid_texts=avoid_texts)
    if utterance_type == "practical_question" or _is_practical_question_text(user_text):
        if "잠" in user_text:
            return _pick_fallback_variant(user_text, [
                "불을 조금 낮추고 화면을 멀리한 뒤, 가벼운 스트레칭이나 따뜻한 물 한 잔부터 해봐요.",
                "잠이 안 올 때는 방을 조금 어둡게 하고, 몸에 힘을 빼는 짧은 루틴부터 해보는 게 좋아요.",
                "화면을 잠깐 내려놓고 따뜻한 물이나 가벼운 호흡으로 몸을 먼저 쉬는 쪽으로 돌려봐요.",
            ], avoid_texts=avoid_texts)
        if "뭐 먹" in user_text or "메뉴" in user_text:
            return _pick_fallback_variant(user_text, [
                "가볍게 먹고 싶으면 김밥이나 샐러드가 좋고, 든든하게 먹고 싶으면 국밥이나 덮밥이 무난해요.",
                "빨리 정하고 싶으면 김밥처럼 간단한 메뉴, 배가 많이 고프면 국밥이나 덮밥 쪽이 좋아요.",
                "오늘은 고르기 쉽게 가보면 좋아요. 가벼운 쪽은 샐러드나 김밥, 든든한 쪽은 국밥이나 덮밥이에요.",
            ], avoid_texts=avoid_texts)
        return _pick_fallback_variant(user_text, [
            "바로 도움이 되는 쪽으로 보면, 먼저 기준을 하나 정하고 선택지를 줄이는 게 좋아요. 지금 상황에서 가장 중요한 조건 하나를 기준으로 골라봐요.",
            "지금은 선택지를 많이 펼치기보다 기준 하나만 잡는 게 좋아요. 가장 중요한 조건부터 정해보면 금방 좁혀져요.",
            "실제로 움직일 수 있는 쪽부터 보면 돼요. 지금 가장 덜 부담스러운 선택 하나를 먼저 골라봐요.",
        ], avoid_texts=avoid_texts)
    if utterance_type in {"casual_neutral", "casual_share", "positive_share"}:
        if any(marker in user_text for marker in PLAIN_NEUTRAL_CONTEXT_MARKERS):
            return _pick_fallback_variant(user_text, [
                "그런 하루였군요. 큰 사건이 없어도 오늘의 흐름으로 기록해둘게요.",
                "무난하게 지나간 하루도 그대로 의미가 있어요. 오늘 흐름을 차분히 남겨둘게요.",
                "특별하지 않은 날도 하루의 일부죠. 말해준 그대로 가볍게 기록해둘게요.",
            ], avoid_texts=avoid_texts)
        if any(marker in user_text for marker in ["낫", "나아", "편해", "괜찮아졌", "괜찮아진"]):
            return _pick_fallback_variant(user_text, [
                "조금 나아진 느낌이 있다니 다행이에요. 지금 그 감각을 무리하지 말고 이어가 봐요.",
                "조금 편해진 순간이 생겼군요. 그 정도 변화도 오늘 흐름에서는 꽤 의미 있어요.",
                "나아진 감각이 있다면 지금은 그걸 작게 붙잡아도 좋아요. 무리해서 더 끌어올리려 하진 않아도 돼요.",
            ], avoid_texts=avoid_texts)
        if "기대" in user_text or "설레" in user_text or "신나" in user_text or "기다려" in user_text:
            return _pick_fallback_variant(user_text, [
                "기대되는 일이 있다니 좋네요. 내일의 좋은 기분을 편하게 누려도 좋겠어요.",
                "설레는 마음이 올라오는 건 반가운 일이에요. 그 기대감을 천천히 즐겨도 좋겠어요.",
                "기다려지는 일이 있다는 게 오늘의 기분을 조금 밝혀주는 것 같아요. 그 감각을 잘 남겨둘게요.",
            ], avoid_texts=avoid_texts)
        if "밥" in user_text or "먹" in user_text:
            return _pick_fallback_variant(user_text, [
                "잘 챙겨 먹었네요. 그런 기본적인 것들이 은근히 하루를 받쳐줘요.",
                "식사를 챙긴 것도 오늘의 리듬을 지키는 일이에요. 가볍게 기록해둘게요.",
                "먹는 일을 챙긴 하루였군요. 별일 아닌 듯해도 몸에는 꽤 중요한 흐름이에요.",
            ], avoid_texts=avoid_texts)
        if "운동" in user_text or "산책" in user_text:
            return _pick_fallback_variant(user_text, [
                "좋아요, 오늘도 몸을 조금 움직였네요.",
                "움직인 시간이 있었다는 게 좋네요. 오늘 몸의 리듬을 조금 챙긴 셈이에요.",
                "산책이나 운동처럼 몸을 움직인 건 작아 보여도 하루 흐름에 꽤 도움이 돼요.",
            ], avoid_texts=avoid_texts)
        if utterance_type == "positive_share":
            return _pick_fallback_variant(user_text, [
                "좋은 일이 있었군요. 오늘의 밝은 장면으로 기록해둘게요.",
                "홀가분하거나 반가운 느낌이 있었다면 충분히 누려도 좋아요. 오늘의 좋은 변화로 기록해둘게요.",
                "잘 지나간 일이 있었다는 게 반가워요. 지금 느낌을 가볍게 남겨둘게요.",
                "기분 좋은 순간이 있었군요. 오늘 안에 그런 장면이 있었다는 걸 잘 남겨둘게요.",
            ], avoid_texts=avoid_texts)
        return _pick_fallback_variant(user_text, [
            "알려줘서 좋아요. 오늘의 흐름은 가볍게 기록해둘게요.",
            "그런 순간도 하루의 일부로 남겨둘게요.",
            "좋아요. 오늘 있었던 일을 차분히 따라가고 있어요.",
            "그 일도 오늘 하루의 한 장면으로 남겨둘게요.",
            "큰 의미를 붙이지 않아도 괜찮아요. 말해준 흐름 그대로 기록해둘게요.",
            "오늘 지나간 일들을 하나씩 따라가고 있어요.",
        ], avoid_texts=avoid_texts)
    if utterance_type == "routine_discomfort":
        if "출근" in user_text or "회사" in user_text or "업무" in user_text:
            return _pick_fallback_variant(user_text, [
                "그런 날 있죠. 오늘은 일단 시작만 해도 꽤 애쓴 거예요.",
                "출근이나 업무가 유난히 버겁게 느껴지는 날도 있어요. 오늘은 첫 단계만 해도 충분히 의미 있어요.",
                "회사 생각만으로도 몸이 무거울 때가 있죠. 지금은 아주 작은 시작부터 잡아봐요.",
            ], avoid_texts=avoid_texts)
        if "공부" in user_text or "과제" in user_text or "시험" in user_text:
            return _pick_fallback_variant(user_text, [
                "하기 싫은 마음이 드는 날도 있어요. 아주 작은 단위로 시작해도 충분해요.",
                "공부나 과제가 막막하면 크게 잡지 말고 5분짜리 시작점만 만들어봐요.",
                "오늘은 의욕이 먼저 오지 않아도 괜찮아요. 작게 손대는 것부터 해도 충분해요.",
            ], avoid_texts=avoid_texts)
        return _pick_fallback_variant(user_text, [
            "귀찮고 싫은 마음이 올라오는 날도 있죠. 오늘은 너무 크게 몰아붙이지 않아도 돼요.",
            "하기 싫은 마음이 있다고 해서 이상한 건 아니에요. 오늘은 부담을 조금 낮춰서 시작해봐요.",
            "몸과 마음이 덜 따라오는 날도 있어요. 지금은 해야 할 일을 작게 쪼개는 쪽이 좋아요.",
        ], avoid_texts=avoid_texts)
    if utterance_type == "emotional_distress":
        if _is_anger_text(user_text):
            return _pick_fallback_variant(user_text, [
                "그 상황이면 화가 남는 게 자연스러워요. 지금은 그 억울함을 바로 누르기보다, 어떤 지점이 제일 컸는지 차분히 봐도 괜찮아요.",
                "무례하거나 부당하게 느껴진 일이 있었군요. 화가 올라온 이유를 여기서는 급히 정리하지 않아도 돼요.",
                "짜증과 억울함이 같이 남아 있는 것 같아요. 지금은 그 감정을 틀렸다고 밀어내지 않아도 괜찮아요.",
                "화가 쉽게 가라앉지 않는 순간이었겠어요. 우선 그만큼 불편했다는 사실부터 같이 잡아둘게요.",
            ], avoid_texts=avoid_texts)
        if any(marker in user_text for marker in ["잠", "불면", "못 잤", "피곤", "졸려", "쉰 것"]):
            return _pick_fallback_variant(user_text, [
                "잠이 잘 안 오면 하루 전체가 더 무겁게 느껴지죠. 오늘은 몸이 조금이라도 쉴 수 있는 쪽으로 같이 맞춰봐요.",
                "수면이 흔들리면 마음도 더 예민해질 수 있어요. 오늘은 회복을 조금이라도 돕는 쪽으로 가봐요.",
                "잠을 충분히 못 자면 작은 일도 크게 느껴질 수 있어요. 지금은 몸을 쉬게 하는 선택을 먼저 봐요.",
                "몸이 피곤하면 마음도 더 쉽게 가라앉을 수 있어요. 오늘은 회복에 방해되는 것부터 조금 줄여봐요.",
            ], avoid_texts=avoid_texts)
        if "친구" in user_text or "가족" in user_text or "싸웠" in user_text:
            return _pick_fallback_variant(user_text, [
                "가까운 사람과 부딪히면 마음이 오래 흔들릴 수 있어요. 지금은 스스로를 너무 몰아붙이지 않았으면 해요.",
                "관계에서 생긴 흔들림은 쉽게 가라앉지 않을 때가 있어요. 지금 마음을 천천히 풀어놔도 괜찮아요.",
                "친한 사람과의 일일수록 더 크게 남을 수 있죠. 여기서는 그 마음을 급히 정리하지 않아도 돼요.",
                "가까운 관계에서 생긴 말은 오래 남을 수 있어요. 지금은 그 장면을 천천히 꺼내도 괜찮아요.",
            ], avoid_texts=avoid_texts)
        # Phase 5c — high distress 신호일 때는 가장 보수적인 첫 variant를 우선 선택해
        # grounding/안전 톤을 강화. _pick_fallback_variant는 avoid_texts와의 매칭으로 회피하므로,
        # 동일 문장 반복 위험이 있으면 다음 variant로 자동 fallback.
        _emotional_distress_variants = [
            "그만큼 버거운 시간을 지나고 있었군요. 지금 느끼는 마음을 급히 정리하려 하기보다, 여기서는 그대로 말해도 괜찮아요.",
            "지금 마음이 꽤 버거웠던 것 같아요. 여기서는 그 느낌을 천천히 풀어놓아도 괜찮아요.",
            "쉽게 넘기기 어려운 감정이 있었군요. 지금은 판단보다, 어떤 부분이 제일 크게 남았는지 보는 게 좋아요.",
            "마음에 남은 무게가 꽤 있었던 것 같아요. 지금은 그걸 작게 나눠서 말해도 괜찮아요.",
            "그 감정이 쉽게 정리되지 않았겠어요. 여기서는 서두르지 않고 이어서 봐도 돼요.",
            "오늘 마음을 누른 부분이 있었군요. 가장 크게 남은 감각부터 천천히 따라가 볼게요.",
        ]
        if _is_high_distress:
            # 가장 따뜻하고 보수적인 톤(첫 2개)만 사용 — _pick_fallback_variant는 list 길이로
            # variant index를 결정하므로 후보 수를 제한해 grounding 톤 강제.
            _emotional_distress_variants = _emotional_distress_variants[:2]
        elif _is_calm:
            # calm 신호일 때는 더 가벼운 후반 variant(판단 회피, 흐름 보존) 우선
            _emotional_distress_variants = _emotional_distress_variants[3:] + _emotional_distress_variants[:3]
        return _pick_fallback_variant(user_text, _emotional_distress_variants, avoid_texts=avoid_texts)

    if _is_positive_user_text(user_text):
        if "기대" in user_text or "설레" in user_text or "신나" in user_text or "기다려" in user_text:
            return _pick_fallback_variant(user_text, [
                "기대되는 일이 있다니 좋네요. 내일의 좋은 기분을 편하게 누려도 좋겠어요.",
                "기다려지는 마음이 있다는 게 반가워요. 그 설렘을 오늘의 좋은 장면으로 남겨둘게요.",
                "좋은 기대감이 올라온 날이네요. 그 감각을 편하게 누려도 괜찮아요.",
            ], avoid_texts=avoid_texts)
        return _pick_fallback_variant(user_text, [
            "좋은 기운이 느껴져요. 오늘 그 감각을 조금 더 오래 붙잡아도 좋겠어요.",
            "기분이 밝아진 순간이 있었군요. 그런 장면은 작게라도 잘 남겨둘게요.",
            "오늘 안에 좋은 감각이 있었다는 게 반가워요. 그 흐름을 편하게 이어가 봐요.",
        ], avoid_texts=avoid_texts)
    if "잠" in user_text or "불면" in user_text:
        return "잠이 잘 안 오면 하루 전체가 더 무겁게 느껴지죠. 오늘은 몸이 조금이라도 쉴 수 있는 쪽으로 같이 맞춰봐요."
    if "운동" in user_text or "철봉" in user_text or "매달리기" in user_text:
        return "몸으로 버티는 운동은 생각보다 훨씬 힘이 많이 들어요. 오늘 해낸 만큼만으로도 충분히 의미 있어요."
    if "친구" in user_text or "가족" in user_text or "싸웠" in user_text:
        return "관계에서 마음이 흔들리면 생각이 계속 맴돌 수 있어요. 지금은 스스로를 너무 몰아붙이지 않았으면 해요."
    if _has_distress_signal(user_text):
        return _pick_fallback_variant(user_text, [
            "그만큼 버거운 시간을 지나고 있었군요. 지금 느끼는 마음을 급히 정리하려 하기보다, 여기서는 그대로 말해도 괜찮아요.",
            "마음에 걸리는 게 꽤 컸던 것 같아요. 지금은 그 느낌을 있는 그대로 말해도 괜찮아요.",
            "쉽게 넘기기 어려운 순간이었겠어요. 여기서는 천천히 이어서 말해도 괜찮아요.",
            "오늘 마음에 남은 게 분명 있었던 것 같아요. 지금은 그걸 급히 해결하지 않고 먼저 알아차려도 괜찮아요.",
        ], avoid_texts=avoid_texts)
    if _is_short_casual_user_text(user_text):
        return "좋아요, 오늘도 작은 흐름을 하나 만들어냈네요."
    return "말해줘서 고마워요. 지금 이야기의 흐름을 차분히 따라가 볼게요."


def diversify_repeated_response(
    user_text: str,
    response_text: str,
    utterance_info: dict | None = None,
    avoid_texts: list[str] | None = None,
) -> tuple[str, bool]:
    """
    역할: 최종 노출 직전 최근 답변과 같은 응답을 fallback 변형으로 치환
    입력: 사용자 발화, 후보 응답, 발화 타입 정보, 최근 노출 응답 목록
    출력: (최종 응답, 반복 치환 여부)
    """
    if not avoid_texts or "[CRISIS]" in (response_text or ""):
        return response_text, False
    if not _is_similar_to_recent_response(response_text, avoid_texts):
        return response_text, False

    replacement = _fallback_response(
        user_text,
        utterance_info=utterance_info,
        avoid_texts=avoid_texts,
    )
    return replacement, replacement != response_text


def postprocess_response(
    user_text: str,
    response_text: str,
    utterance_info: dict | None = None,
) -> str:
    """
    역할: Qwen 생성문을 사용자 노출 전 정리하고 저품질 응답은 안전 응답으로 대체한다.
    입력: 사용자 발화, 모델 원문 응답, 발화 타입 정보
    출력: 최종 응답 텍스트
    """
    # [CRISIS] 태그는 백엔드 안전 인터럽트가 감지해야 하므로 품질 대체로 제거하지 않는다.
    if "[CRISIS]" in response_text:
        return response_text.strip()
    if is_direct_crisis_text(user_text):
        return DIRECT_CRISIS_RESPONSE

    cleaned = _clean_generated_text(response_text)
    if _violates_typed_style(cleaned, utterance_info=utterance_info, user_text=user_text):
        return _fallback_response(user_text, utterance_info=utterance_info)
    if _is_low_quality_response(user_text, cleaned):
        return _fallback_response(user_text, utterance_info=utterance_info)
    return cleaned


def _distress_prompt_hint(
    utterance_type: str | None,
    distress_top_label: str | None,
    distress_severity_scalar: float | None,
) -> str | None:
    """
    역할: Phase 5d — distress severity 신호를 system prompt 톤 힌트로 변환한다.
    입력: 발화 타입, distress 5클래스 top 라벨, severity scalar(0~1)
    출력: 시스템 프롬프트에 덧붙일 힌트 문자열 또는 None(영향 없음)

    설계 메모:
        - distress 정보가 없으면 None을 반환해 backward compat을 유지한다.
        - emotional_distress 분기는 3단계(high/mid/calm)로 톤을 분리한다.
        - casual_share / routine_discomfort / positive_share 분기는 high/crisis 일 때만
          가볍게 넘기지 않도록 grounding 한 줄을 덧붙인다.
        - crisis_candidate 분기는 이미 안전 지침이 강하므로 추가하지 않는다.
    """
    # Phase 5b 이전 모델 또는 distress head 미로드 시 영향 없음
    is_high = distress_top_label in {"high_distress", "crisis_candidate"} or (
        distress_severity_scalar is not None and float(distress_severity_scalar) >= 0.65
    )
    is_calm = distress_top_label == "calm_or_positive" or (
        distress_severity_scalar is not None and float(distress_severity_scalar) < 0.20
    )
    has_signal = distress_top_label is not None or distress_severity_scalar is not None
    if not has_signal:
        return None

    if utterance_type == "emotional_distress":
        if is_high:
            return (
                "현재 distress 신호가 높음 — grounding과 안전감을 우선하고, "
                "조언이나 추가 탐색 질문은 자제하세요. 1~2문장으로 짧게 머물러 주세요."
            )
        if is_calm:
            return (
                "현재 distress 신호가 낮음 — 가벼운 공감 한 마디로 응답하고, "
                "감정을 더 깊게 파고들지 마세요."
            )
        # 중간 구간(0.20 ≤ severity < 0.65, mild/moderate): 기본 emotional_distress 가이드 유지
        return (
            "현재 distress 신호가 중간 — 감정을 먼저 짧게 반영한 뒤, "
            "필요하면 탐색 질문을 한 가지만 덧붙이세요."
        )

    if utterance_type in {"casual_share", "routine_discomfort", "positive_share"} and is_high:
        return (
            "단, 현재 distress 신호가 높음 — 가볍게 넘기지 말고 "
            "감정을 한 번 짚어준 뒤 본 분기 가이드를 따르세요."
        )

    return None


def build_utterance_style_instruction(utterance_info: dict | None = None) -> str:
    """
    역할: 발화 타입 분류 결과를 Qwen용 짧은 응답 스타일 지시문으로 변환한다.
    입력: {utterance_type, type_confidence/type_reason, distress_top_label,
          distress_severity_scalar ...} 형태의 dict 또는 None
    출력: 시스템 프롬프트에 추가할 지시문 문자열

    Phase 5d — utterance_info에 distress_top_label / distress_severity_scalar이 박제돼 있으면
    분기별 톤 힌트를 system prompt에 덧붙인다 (backward compat: 미박제 시 기존 동작).
    """
    if not utterance_info:
        return DEFAULT_UTTERANCE_STYLE_GUIDE

    utterance_type = utterance_info.get("utterance_type")
    guide = UTTERANCE_STYLE_GUIDES.get(utterance_type, DEFAULT_UTTERANCE_STYLE_GUIDE)
    reason = utterance_info.get("utterance_type_reason") or utterance_info.get("type_reason")
    confidence = utterance_info.get("utterance_type_confidence") or utterance_info.get("type_confidence")

    # Phase 5d — distress 신호가 있으면 분기별 톤 힌트 한 줄 추가
    distress_hint = _distress_prompt_hint(
        utterance_type,
        utterance_info.get("distress_top_label"),
        utterance_info.get("distress_severity_scalar"),
    )

    meta_parts = []
    if reason:
        meta_parts.append(f"판정 근거: {reason}")
    if confidence is not None:
        meta_parts.append(f"신뢰도: {float(confidence):.2f}")

    result = guide
    if distress_hint:
        result = result + "\n" + distress_hint
    if meta_parts:
        result = result + "\n" + " / ".join(meta_parts)
    return result


@torch.no_grad()
def generate_response(
    user_text: str,
    history: list[dict] | None = None,
    utterance_info: dict | None = None,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    """
    역할: 사용자 발화에 대한 상담 응답 생성
    입력: 사용자 발화, 대화 히스토리 [{"role": ..., "content": ...}],
          발화 타입 정보, 최대 생성 토큰 수
    출력: 생성된 응답 텍스트 (str)
    """
    # 직접적 위기 표현은 모델 출력 품질과 무관하게 안전 응답을 우선 반환한다.
    if is_direct_crisis_text(user_text):
        return DIRECT_CRISIS_RESPONSE

    model, tokenizer = load_qwen()

    # 시스템 프롬프트 + 히스토리 + 현재 발화 조합
    style_instruction = build_utterance_style_instruction(utterance_info)
    messages = [{"role": "system", "content": SYSTEM_PROMPT + "\n\n" + style_instruction}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_text})

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": GEN_DO_SAMPLE,
        "repetition_penalty": GEN_REPETITION_PENALTY,
        "no_repeat_ngram_size": GEN_NO_REPEAT_NGRAM,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if GEN_DO_SAMPLE:
        generation_kwargs["temperature"] = GEN_TEMPERATURE
        generation_kwargs["top_p"] = GEN_TOP_P

    output_ids = model.generate(**inputs, **generation_kwargs)

    # 입력 토큰 제거 후 디코딩
    generated = output_ids[0][inputs["input_ids"].shape[-1]:]
    raw_response = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return postprocess_response(user_text, raw_response, utterance_info=utterance_info)


SELF_CHECK_SYSTEM_PROMPT = (
    "너는 응답 검수기다. 응답에 문제 있으면 첫 토큰을 BAD, 없으면 OK 만 출력한다."
)

# few-shot 예시: 작은 모델이 "BAD" mode 에 collapse 하지 않도록
# 정상/문제 케이스 둘 다 보여준다.
_SELF_CHECK_FEWSHOT = [
    {"role": "user", "content":
        "[응답] 그런 날 있죠. 오늘은 일단 시작만 해도 꽤 애쓴 거예요."},
    {"role": "assistant", "content": "OK"},
    {"role": "user", "content":
        "[응답] 가볍게 먹고 싶으면 김밥이나 샐러드가 좋아요."},
    {"role": "assistant", "content": "OK"},
    {"role": "user", "content":
        "[응답] 선생님께서 그렇게 말씀하셨군요. 다음 회기에 다시 이야기해봐요."},
    {"role": "assistant", "content": "BAD"},
    {"role": "user", "content":
        "[응답] 저도 어제 그 영화를 봤는데 정말 좋았어요."},
    {"role": "assistant", "content": "BAD"},
    {"role": "user", "content":
        "[응답] 지금 당장 운동을 시작하세요. 회사를 그만두세요."},
    {"role": "assistant", "content": "BAD"},
    {"role": "user", "content":
        "[응답] 기분이 별로인 날도 있죠. 너무 잘해내려고 하기보다 조금 쉬어 가도 괜찮아요."},
    {"role": "assistant", "content": "OK"},
]


def self_check_response(
    user_text: str,
    response_text: str,
    max_new_tokens: int = 4,
) -> dict:
    """
    역할: Qwen 으로 자기 응답을 검수해 OK / BAD 판정
          작은 모델이 한쪽 답으로 collapse 하지 않도록 few-shot + LoRA 일시 비활성화 사용
    입력: 사용자 발화(미사용 — few-shot 단순화), 챗봇 응답, 최대 생성 토큰 수
    출력: {"verdict": "OK"|"BAD", "category": int|None, "raw": str}
          호출 실패 시 verdict="OK" (보수적: 자기검토 실패는 통과 처리)
    """
    response_text = (response_text or "").strip()
    if not response_text:
        return {"verdict": "OK", "category": None, "raw": ""}

    try:
        model, tokenizer = load_qwen()
    except Exception as exc:
        print(f"[self_check] Qwen 로드 실패 → OK 처리: {exc}")
        return {"verdict": "OK", "category": None, "raw": ""}

    messages = [{"role": "system", "content": SELF_CHECK_SYSTEM_PROMPT}]
    messages.extend(_SELF_CHECK_FEWSHOT)
    messages.append({"role": "user", "content": f"[응답] {response_text}"})

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # LoRA(상담/일상 응답 톤)를 일시 비활성화해 base Qwen 메타-task 능력으로 판정
    disable_ctx = None
    try:
        if hasattr(model, "disable_adapter"):
            disable_ctx = model.disable_adapter()
    except Exception:
        disable_ctx = None

    try:
        if disable_ctx is not None:
            with disable_ctx:
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    repetition_penalty=1.0,
                    pad_token_id=tokenizer.eos_token_id,
                )
        else:
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
    except Exception as exc:
        print(f"[self_check] generate 실패 → OK 처리: {exc}")
        return {"verdict": "OK", "category": None, "raw": ""}

    generated = output_ids[0][inputs["input_ids"].shape[-1]:]
    raw = tokenizer.decode(generated, skip_special_tokens=True).strip()
    head = raw.split()[0].upper() if raw else ""

    if head.startswith("BAD"):
        return {"verdict": "BAD", "category": None, "raw": raw}
    return {"verdict": "OK", "category": None, "raw": raw}


SUMMARY_SYSTEM_PROMPT = (
    "너는 한국어 상담 대화 요약기다. 아래 사용자 발화 묶음을 2문장 이내로 요약한다. "
    "규칙: (1) 사실만 적고 추측·진단은 금지. "
    "(2) 사용자가 직접 말하지 않은 감정/인물/사건을 만들어내지 마라. "
    "(3) 1인칭 '저는/제가' 사용 금지. (4) 위로·조언 금지, 객관 서술만."
)


def generate_summary(
    user_utterances: list[str],
    max_new_tokens: int = 80,
) -> str:
    """
    역할: 사용자 발화 묶음의 사실 요약(최대 2문장)을 Qwen 으로 생성
    입력: 사용자 발화 텍스트 리스트, 최대 생성 토큰 수
    출력: 요약 문자열 (실패 시 빈 문자열)
    """
    cleaned = [str(u).strip() for u in (user_utterances or []) if str(u).strip()]
    if not cleaned:
        return ""

    model, tokenizer = load_qwen()
    bullet_block = "\n".join(f"- {u}" for u in cleaned[-20:])
    user_msg = (
        "다음 사용자 발화들을 2문장 이내로 객관적으로 요약해줘. "
        "감정/사건을 새로 만들어내지 말고, 실제 언급된 내용만 정리해.\n\n"
        f"{bullet_block}"
    )

    messages = [
        {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=GEN_REPETITION_PENALTY,
        no_repeat_ngram_size=GEN_NO_REPEAT_NGRAM,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated = output_ids[0][inputs["input_ids"].shape[-1]:]
    raw = tokenizer.decode(generated, skip_special_tokens=True).strip()

    # 다중 줄 → 한 줄 압축, 길이 cap
    one_line = " ".join(raw.split())
    if len(one_line) > 220:
        one_line = one_line[:220].rstrip() + "..."
    return one_line
