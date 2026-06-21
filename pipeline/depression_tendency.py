"""
depression_tendency.py
역할: 우울 경향 전용 점수(depression_tendency_score) 규칙 기반 v1.5.2 계산기
      현재 운영 depression_score는 종합 정서 위험도(불안/분노/상황성 distress 포함)에 가깝다.
      본 함수는 "우울 경향 추적"이라는 프로젝트 목표에 맞게,
      명시 우울 / 무기력 / 흥미저하 / 무가치감 / 절망 / 수면식욕 / 사회적 위축 신호만 가산하고,
      시험불안 / 단일 사건성 분노·속상함 / 운동·근무 후 신체 피로 / 일상 루틴 / 긍정 회복은 cap으로 제한한다.
입력: 발화 텍스트 + 선택적 RoBERTa/CBT/utterance_type 출력
출력: {depression_tendency_score, hit_categories, persistence, caps_applied, debug}
"""

from __future__ import annotations

from typing import Optional

from pipeline.utterance_type import (
    compact_text,
    has_crisis_marker,
    is_academic_anxiety_text,
    is_daily_routine_neutral_text,
    is_limited_situational_distress_text,
    is_physical_exertion_text,
    is_positive_affect_text,
    is_sensory_disgust_text,
    is_situational_anger_text,
    is_situational_anxiety_surprise_text,
    is_situational_sadness_text,
    normalize_emotion_analysis_text,
)

# ---------------------------------------------------------------------------
# 우울 경향 신호 키워드 사전 (compact_text 정규화 후 substring 매칭)
# ---------------------------------------------------------------------------

DEPRESSION_EXPLICIT_KEYWORDS = (
    "우울",
    "마음이무거",
    "마음이무겁",
    "마음이좀무거",
    "마음이좀무겁",
    "마음이계속무거",
    "마음이계속무겁",
    "마음이많이무거",
    "마음이많이무겁",
    "마음이어둡",
    "가라앉",
    "처져",
    "처지네",
    "처지는",
    "마음무거",
    "마음무겁",
)

ANHEDONIA_KEYWORDS = (
    "재미가없",
    "재미가하나도없",
    "재미도없",
    "재미있는게없",
    "재미있는것도없",
    "재미있는게하나도없",
    "재밌는게없",
    "재밌는게하나도없",
    "흥미가없",
    "흥미없",
    "즐겁지않",
    "즐거움이없",
    "손이안가",
    "손이가지않",
    "손도안가",
    "좋아하던일도손이안",
    "좋아하던것도손이안",
    "마음이움직이지않",
    "마음이잘움직이지않",
    "아무일에도마음이움직이지않",
    "무감해",
    "감흥이없",
    "공허",
    "시들해",
    "시들",
    "부질없",
)

ENERGY_LOSS_KEYWORDS = (
    "의욕이없",
    "의욕없",
    "의욕도없",
    "의욕이안",
    "기운이없",
    "기운없",
    "기운도없",
    "기운이하나도없",
    "기운하나도없",
    "기력이없",
    "기력도없",
    "기력하나도없",
    "아무것도하기싫",
    "아무것도안하고싶",
    "몸이무거",
    "몸이천근",
    "무기력",
    "눈을떠도몸이무겁",
    "시작할힘이안나",
    "시작할힘이없",
    "힘이안나",
    "힘이없",
    "누워만",
    "누워있",
    "다귀찮",
    "전부귀찮",
)

WORTHLESSNESS_KEYWORDS = (
    "한심",
    "쓸모없",
    "가치가없",
    "부족한사람",
    "내가부족",
    "부족하다는생각",
    "머릿속을맴돌",
    "맴돌아",
    "내탓",
    "내잘못",
    "자책",
    "보잘것없",
    "초라",
)

HOPELESSNESS_KEYWORDS = (
    "희망이없",
    "희망없",
    "나아질것같지않",
    "나아지지않",
    "안나아질",
    "기대자체가안",
    "기대도안",
    "괜찮아질장면",
    "그려지지않",
    "잘그려지지않",
    "막막해",
    "끝이안보",
    "끝이없",
    "끝이없는",
    "변하지않을것같",
    "변하지않을",
    "변할것같지않",
    "달라지지않",
    "달라질것같지않",
    "의미가없",
)

SOMATIC_KEYWORDS = (
    "잠을못",
    "잠을잘못",
    "잠이안와",
    "잠이안",
    "잠도안",
    "잠도못",
    "계속자",
    "계속잠",
    "밥맛이없",
    "밥맛도줄",
    "밥맛줄",
    "밥맛이줄",
    "입맛이없",
    "입맛도없",
    "입맛이없어진",
    "먹고싶지않",
    "먹고싶지도않",
    "밤에도자꾸뒤척",
    "뒤척였",
    "뒤척",
    "불면",
)

WITHDRAWAL_KEYWORDS = (
    "만나기싫",
    "만나기도싫",       # 조사 "도" 삽입형 — 운영 시뮬 OS020에서 누락 확인
    "만나기가싫",       # 조사 "가" 삽입형
    "만나는게부담",
    "만나는게너무부담",
    "사람보기싫",
    "혼자있고싶",
    "혼자만있고싶",     # "혼자 만 있고 싶어" 조사 "만" 삽입형 — OS020
    "혼자만있고",       # "혼자 만 있고 싶 어"의 간단 변형
    "연락끊",
    "연락답장",
    "답장을미루",
    "답장미루",
    "사람들과더멀어",
    "멀어진것같",
    "방에만",
    "혼자방",
    "밖에나가기싫",
    "안나가",
    "다거절",
    "거절하고있",
    "다피하",
)

PERSISTENCE_MARKERS = (
    "요즘계속",
    "요즘자꾸",
    "요즘",
    "매일",
    "예전부터",
    "예전엔",
    "예전에는",
    "예전에",            # 단독형 — OS012 "예전에 좋아했던..."에서 누락 확인
    "전부터",
    "한참전부터",
    "한참째",
    "오랫동안",
    "오래야",
    "한달째",
    "며칠째",
    "며칠동안",
    "요며칠",
    "며칠",
    "몇주째",
    "몇달째",
    "한달동안",
    "자꾸",              # 단독형 — OS014 "자꾸 들어"에서 누락 확인. 카테고리 hit≥1일 때만 multiplier 적용되므로 일상 발화 오트리거 위험 없음.
    "계속",              # 카테고리 hit가 있을 때만 강화해 "계속 내가 부족" 같은 지속 신호를 살린다.
    "더심해",
    "더악화",
    "쭉그래",
    "계속그래",
    "계속우울",
    "계속힘들",
    "마음이계속",
)

# 단발성/금일 한정 신호 — 단일 카테고리만 잡혔을 때 mild cap에 사용
TRANSIENT_MARKERS = (
    "오늘은",
    "오늘좀",
    "오늘따라",
    "오늘은좀",
    "지금은",
    "이번한",
    "한번쯤",
    "그냥좀",
    "기분이드는날",
    "느낌이드는날",
    "비오는날",
)

# v1.5 보조: 모델이 중립/casual로 잘못 분류해도 잡아낼 수 있는 가벼운 슬픔 표현
MILD_SADNESS_PHRASES = (
    "마음이안좋",
    "맘이안좋",
    "기분이안좋",
    "마음이별로",
    "맘이별로",
    "기분이별로",
    "마음이가라앉",
    "기분이가라앉",
)

# 단일 사건성 distress — context + outcome 조합 시 cap 0.30
SINGLE_EVENT_DISTRESS_CONTEXTS = (
    "기대했던",
    "기대한",
    "발표",
    "시험",
    "면접",
    "과제",
    "프로젝트",
    "약속이취소",
    "약속취소",
    "약속이",
    "친구랑",
    "친구한테",
    "엄마한테",
    "아빠한테",
    "동료",
    "일이안",
    "일이잘안",
    "회사일이",
)

SINGLE_EVENT_DISTRESS_OUTCOMES = (
    "안됐",
    "안된",
    "망쳤",
    "안나",
    "안떠올",
    "막막",
    "어려웠",
    "안풀",
    "잘안풀",
    "다퉜",
    "서운한말",
    "잔소리",
    "혼났",
    "취소",
    "못해서",
    "부끄러웠",
)

# 슬픔이 명시적으로 포함된 outcome — 이쪽이 잡히면 cap을 0.30 mid 수준으로 유지한다.
SINGLE_EVENT_SADNESS_OUTCOMES = (
    "다퉜",
    "서운한말",
    "혼났",
    "헤어졌",
    "거절당",
    "외면당",
)

# ---------------------------------------------------------------------------
# 카테고리별 가중치
# ---------------------------------------------------------------------------

# (base, extra_per_additional_hit) — 같은 카테고리 안에서 추가 히트당 보너스, 최대 +0.15
CATEGORY_WEIGHTS = {
    "explicit":      (0.45, 0.15),
    "worthlessness": (0.45, 0.15),
    "hopelessness":  (0.45, 0.15),
    "energy_loss":   (0.40, 0.15),
    "anhedonia":     (0.40, 0.15),
    "somatic":       (0.30, 0.15),
    "withdrawal":    (0.30, 0.15),
}

CATEGORY_KEYWORDS = {
    "explicit":      DEPRESSION_EXPLICIT_KEYWORDS,
    "anhedonia":     ANHEDONIA_KEYWORDS,
    "energy_loss":   ENERGY_LOSS_KEYWORDS,
    "worthlessness": WORTHLESSNESS_KEYWORDS,
    "hopelessness":  HOPELESSNESS_KEYWORDS,
    "somatic":       SOMATIC_KEYWORDS,
    "withdrawal":    WITHDRAWAL_KEYWORDS,
}

# 지속성 multiplier (≥1 카테고리 히트일 때만 곱해진다)
PERSISTENCE_MULTIPLIER_SINGLE = 1.40
PERSISTENCE_MULTIPLIER_DOUBLE = 1.50  # 지속성 표현이 2개 이상이면 강화

# 최대값 — 점수가 1.0에 너무 쉽게 붙지 않게 0.95로 soft clip
SOFT_CLIP_MAX = 0.95

# ---------------------------------------------------------------------------
# cap 우선순위 — 작은 값일수록 강한 cap. 동시에 해당하면 가장 작은 cap이 적용된다.
# ---------------------------------------------------------------------------
CAP_DAILY_ROUTINE        = 0.10
CAP_POSITIVE_RECOVERY    = 0.10
CAP_LOW_RISK_DISGUST     = 0.10
CAP_SITUATIONAL_ANGER    = 0.15
CAP_SITUATIONAL_FEAR     = 0.15
CAP_PHYSICAL_EXERTION    = 0.15
CAP_ACADEMIC_ANXIETY     = 0.20
CAP_LIMITED_SITUATIONAL  = 0.30
CAP_SITUATIONAL_SADNESS  = 0.30
CAP_SINGLE_EVENT_DISTRESS = 0.30  # 사건 context + outcome 결합
CAP_TRANSIENT_SINGLE_CAT = 0.35  # 단발성 marker + 카테고리 1개 히트 시 부드러운 상한

# ---------------------------------------------------------------------------
# 보조: 카테고리 히트 카운팅
# ---------------------------------------------------------------------------

# v1.5 보조: 기존 utterance_type predicate가 못 잡는 운동·근무 후 신체 피로 표현 보강
PHYSICAL_V15_CONTEXTS = (
    "운동했",
    "운동하니",
    "운동하고",
    "일하",
    "근무",
    "알바",
    "가게",
    "매장",
    "마트",
    "편의점",
    "카페",
    "카운터",
    "퇴근",
    "진열",
    "상하차",
    "서빙",
    "주방",
    "서서",
    "서있",
    "이사",
    "짐나르",
    "짐옮",
    "박스",
    "상품",
    "물건",
    "창고",
    "물류",
    "배달",
    "택배",
    "청소",
    "오래걸",
    "오래뛰",
    "오래걸어",
    "오르내",
    "오르락내리락",
    "달리기",
    "조깅",
    "등산",
    "철봉",
    "스쿼트",
    "푸쉬업",
    "역기",
    "매달리기",
)

# 신체 활동 outcome 후보 — substring 검사용 짧은 단어들 (compact_text는 공백 제거이지만
# "몸이 좀 무거워" 같은 부사 삽입을 잡으려면 부분 키워드를 짧게 유지한다)
PHYSICAL_V15_OUTCOME_FRAGMENTS = (
    "무거",
    "힘들",
    "아파",
    "아프",
    "쑤셔",
    "쑤신",
    "쑤시",
    "뻐근",
    "온몸",
    "몸살",
    "허리",
    "다리",
    "어깨",
    "팔",
    "손목",
    "종아리",
    "허벅지",
    "무릎",
    "발바닥",
    "근육",
    "땀이",
    "지쳐",
    "피곤",
    "녹초",
    "땡겨",
    "당겨",
    "버티기",
    "쥐가",
)

PHYSICAL_V15_BLOCKERS = (
    "우울",
    "괴로",
    "외로",
    "공허",
    "무기력",
    "의욕없",
    "의욕이없",
    "살기힘",
    "마음",
    "한심",
    "자책",
    "절망",
    "죽고싶",
    "숨이안쉬",
    "호흡이안",
    "가슴통증",
    "쓰러질",
    "기절",
)

SITUATIONAL_FEAR_CONTEXTS = (
    "계약서",
    "조항",
    "지갑",
    "카드",
    "천둥",
    "골목",
    "인기척",
    "백업",
    "파일",
    "검사결과",
    "결과문자",
    "면담",
    "질문",
    "엘리베이터",
    "창문",
    "밤에",
    "발표파일",
    "새팀",
    "가족연락",
)

SITUATIONAL_FEAR_MARKERS = (
    "불안",
    "겁났",
    "무서",
    "걱정",
    "조마조마",
    "움츠러",
    "긴장",
    "손에땀",
    "땀이났",
    "확인",
    "깨질까",
    "갇힐까",
    "못따라갈까",
    "귀를기울",
    "별일아닌데도",
)


def _has_any_physical_v15(compact: str) -> bool:
    """
    역할: v1.5 전용 운동/근무/생활 신체 활동 보조 감지
    입력: compact_text 결과 문자열
    출력: 신체 활동 context + 피로/무거움 fragment 동시 매칭 여부
    """
    has_ctx = any(c in compact for c in PHYSICAL_V15_CONTEXTS)
    has_out = any(o in compact for o in PHYSICAL_V15_OUTCOME_FRAGMENTS)
    has_blocker = any(b in compact for b in PHYSICAL_V15_BLOCKERS)
    return has_ctx and has_out and not has_blocker


def _is_situational_fear_v15(compact: str) -> bool:
    """
    역할: 계약·분실·천둥·새 팀 적응처럼 우울 경향이 아닌 상황성 불안/공포를 감지한다.
    입력: compact_text 결과 문자열
    출력: 상황성 공포/불안 여부
    """
    has_context = any(c in compact for c in SITUATIONAL_FEAR_CONTEXTS)
    has_marker = any(m in compact for m in SITUATIONAL_FEAR_MARKERS)
    return has_context and has_marker


def _count_hits(compact: str, keywords: tuple[str, ...]) -> int:
    """
    역할: compact 텍스트에서 키워드 substring 매칭 횟수 계산
    입력: compact_text 결과 문자열, 키워드 튜플
    출력: 매칭된 키워드 수 (중복 키워드는 1개로 카운트)
    """
    return sum(1 for kw in keywords if kw in compact)


def _category_score(category: str, n_hits: int) -> float:
    """
    역할: 카테고리 히트 수 기반 점수 산출 (extra hit는 최대 1회 추가 보너스)
    입력: 카테고리명, 히트 수
    출력: 카테고리 점수 (0.0~base+extra)
    """
    if n_hits <= 0:
        return 0.0
    base, extra = CATEGORY_WEIGHTS[category]
    # 같은 카테고리에서 추가 hit은 최대 1번만 보너스 (더 누적되어도 의미 약함)
    bonus = extra if n_hits >= 2 else 0.0
    return base + bonus


def _probabilistic_or(scores: list[float]) -> float:
    """
    역할: 카테고리별 score를 확률 OR (1 - ∏(1-p))로 합산
    입력: 카테고리 점수 리스트
    출력: 합산 점수 (0.0~1.0)
    """
    if not scores:
        return 0.0
    if len(scores) == 1:
        return float(scores[0])
    product = 1.0
    for s in scores:
        product *= max(0.0, 1.0 - s)
    return float(1.0 - product)


# ---------------------------------------------------------------------------
# 메인 계산 함수
# ---------------------------------------------------------------------------

def compute_depression_tendency(
    text: str,
    *,
    top_emotion: Optional[str] = None,
    roberta_score: Optional[float] = None,
    cbt_score: Optional[float] = None,
    cbt_non_distortion: Optional[bool] = None,
    utterance_type: Optional[str] = None,
    type_reason: Optional[str] = None,
    is_crisis: bool = False,
    entailment_prob: Optional[float] = None,
) -> dict:
    """
    역할: 발화 단위 우울 경향 점수(depression_tendency_score) 규칙 기반 v1.5.2 계산
    입력:
      - text: 사용자 발화 원문
      - top_emotion: RoBERTa 7클래스 top 감정 (옵션)
      - roberta_score: 0~1, vector-scaled (옵션, 본 v1.5에서는 직접 가산 X, 보조 baseline용)
      - cbt_score, cbt_non_distortion: CBT 헤드 출력 (옵션, 보조)
      - utterance_type, type_reason: utterance_type.classify_utterance_type 결과 (옵션)
      - is_crisis: 위기 후보 여부 (점수 직접 가산 X, 메타데이터로만 보존)
      - entailment_prob: NLI 위기 entailment 확률 (옵션, 보조)
    출력: dict
      - depression_tendency_score: 0.0~1.0
      - hit_categories: 매칭된 우울 신호 카테고리 리스트
      - persistence_marker_hit: 지속성 표현 매칭 여부
      - caps_applied: 적용된 cap 사유 리스트
      - raw_score_before_cap: cap 적용 전 점수
      - components: 디버그용 카테고리별 점수 dict
    """
    analysis_text = normalize_emotion_analysis_text(text)
    compact = compact_text(analysis_text)
    has_crisis = bool(is_crisis) or has_crisis_marker(analysis_text)

    # 1. 카테고리 히트 카운팅
    hits: dict[str, int] = {}
    for cat, keywords in CATEGORY_KEYWORDS.items():
        n = _count_hits(compact, keywords)
        if n > 0:
            hits[cat] = n

    # 2. 카테고리별 점수 + 합산 (probabilistic OR)
    components = {cat: _category_score(cat, n) for cat, n in hits.items()}
    base_score = _probabilistic_or(list(components.values()))

    # 3. 지속성 multiplier (≥1 카테고리 히트 + persistence marker 동시일 때만)
    persistence_count = sum(1 for m in PERSISTENCE_MARKERS if m in compact)
    persistence_hit = persistence_count >= 1
    if persistence_hit and len(hits) >= 1:
        mult = PERSISTENCE_MULTIPLIER_DOUBLE if persistence_count >= 2 else PERSISTENCE_MULTIPLIER_SINGLE
        base_score = min(SOFT_CLIP_MAX, base_score * mult)

    # 4. cap 후보 수집 — sadness baseline 적용 전에 먼저 결정한다.
    #    (low-band cap이 잡히면 baseline을 적용하지 않아 false_high를 막는다)
    caps_applied: list[str] = []
    cap_candidates: list[tuple[float, str]] = []

    # 일상 루틴 / 긍정 회복 cap은 우울 카테고리 hit가 0이고
    # mild sadness phrase("마음이 안 좋아")도 없을 때만 적용한다.
    # "친구랑 다퉜는데 마음이 안 좋아"의 "안 좋아"는 POSITIVE_MARKERS의 "좋아"로 오트리거 되므로
    # mild sadness phrase가 있으면 cap을 보류한다.
    has_mild_sadness_phrase_for_cap = any(p in compact for p in MILD_SADNESS_PHRASES)
    if not hits and not has_mild_sadness_phrase_for_cap:
        if is_daily_routine_neutral_text(analysis_text):
            cap_candidates.append((CAP_DAILY_ROUTINE, "daily_routine_neutral_cap"))
        if is_positive_affect_text(analysis_text) and not has_crisis:
            cap_candidates.append((CAP_POSITIVE_RECOVERY, "positive_affect_safety_cap"))
        if is_sensory_disgust_text(analysis_text):
            cap_candidates.append((CAP_LOW_RISK_DISGUST, "low_risk_disgust_cap"))
        if is_situational_anxiety_surprise_text(analysis_text) or _is_situational_fear_v15(compact):
            cap_candidates.append((CAP_SITUATIONAL_FEAR, "situational_fear_cap"))
    if is_situational_anger_text(analysis_text):
        cap_candidates.append((CAP_SITUATIONAL_ANGER, "situational_anger_cap"))
    if is_physical_exertion_text(analysis_text):
        cap_candidates.append((CAP_PHYSICAL_EXERTION, "physical_exertion_context"))
    # v1.5 자체 physical 보조 감지 — 운동·근무 context + 신체 무거움/피로 표현
    elif _has_any_physical_v15(compact):
        cap_candidates.append((CAP_PHYSICAL_EXERTION, "physical_exertion_v15"))
    if is_academic_anxiety_text(analysis_text):
        cap_candidates.append((CAP_ACADEMIC_ANXIETY, "academic_anxiety_cbt_cap"))
    if is_limited_situational_distress_text(analysis_text) and not persistence_hit:
        cap_candidates.append((CAP_LIMITED_SITUATIONAL, "limited_situational_distress_cap"))
    if is_situational_sadness_text(analysis_text) and not persistence_hit and len(hits) <= 1:
        cap_candidates.append((CAP_SITUATIONAL_SADNESS, "situational_sadness_cap"))

    # 단일 사건성 distress: context 키워드 + outcome 키워드 동시 매칭 (지속성 없을 때만)
    has_single_event_ctx = any(c in compact for c in SINGLE_EVENT_DISTRESS_CONTEXTS)
    has_single_event_outcome = any(o in compact for o in SINGLE_EVENT_DISTRESS_OUTCOMES)
    single_event_distress = (
        has_single_event_ctx
        and has_single_event_outcome
        and not persistence_hit
        and len(hits) <= 1
    )
    if single_event_distress:
        # mild sadness phrase / sadness outcome / 명시 카테고리 hit 중 하나라도 있으면
        # 0.30 cap (단일 사건 + 슬픔 신호 → mid). 모두 없으면 0.15 cap (low).
        has_mild_sadness_for_cap = any(p in compact for p in MILD_SADNESS_PHRASES)
        has_sadness_outcome = any(o in compact for o in SINGLE_EVENT_SADNESS_OUTCOMES)
        is_sit_sadness = is_situational_sadness_text(analysis_text)
        if has_mild_sadness_for_cap or has_sadness_outcome or hits or is_sit_sadness:
            cap_candidates.append((CAP_SINGLE_EVENT_DISTRESS, "single_event_distress_cap"))
        else:
            cap_candidates.append((0.15, "single_event_distress_no_sadness_cap"))

    transient_hit = any(m in compact for m in TRANSIENT_MARKERS)
    if transient_hit and len(hits) <= 1 and not persistence_hit and not has_crisis:
        cap_candidates.append((CAP_TRANSIENT_SINGLE_CAT, "transient_single_category_cap"))

    # 가장 작은 cap이 우선
    cap_candidates.sort(key=lambda x: x[0])
    smallest_cap = cap_candidates[0][0] if cap_candidates else None

    # 5. 슬픔 emotional_distress baseline — cap이 0.20 미만이면 baseline 자체를 보류한다.
    #    "과제가 안 풀려서 답답해" / "운동·근무 뒤 몸이 무거워" 같은 단발성 distress는
    #    슬픔으로 분류되더라도 우울 경향에 가산하지 않는다.
    sadness_baseline_used = False
    has_mild_sadness_phrase = any(p in compact for p in MILD_SADNESS_PHRASES)
    sadness_eligible = (
        (top_emotion == "슬픔" and utterance_type == "emotional_distress")
        or has_mild_sadness_phrase
    )
    if (
        not hits
        and sadness_eligible
        and not has_crisis
        and (smallest_cap is None or smallest_cap >= 0.20)
    ):
        base_score = max(base_score, 0.20)
        sadness_baseline_used = True

    raw_before_cap = base_score

    if cap_candidates:
        cap_value, cap_reason = cap_candidates[0]
        if base_score > cap_value:
            base_score = cap_value
            caps_applied.append(cap_reason)

    # 6. soft clip
    final_score = max(0.0, min(SOFT_CLIP_MAX, base_score))

    return {
        "depression_tendency_score": round(final_score, 4),
        "hit_categories": sorted(hits.keys()),
        "category_hit_counts": dict(hits),
        "persistence_marker_hit": bool(persistence_hit),
        "transient_marker_hit": bool(transient_hit),
        "caps_applied": caps_applied,
        "raw_score_before_cap": round(raw_before_cap, 4),
        "components": {k: round(v, 4) for k, v in components.items()},
        "sadness_baseline_used": sadness_baseline_used,
        "is_crisis": bool(has_crisis),
        "analysis_text": analysis_text,
        "analysis_text_changed": analysis_text != str(text or "").strip(),
        "version": "v1.5.2",
    }
