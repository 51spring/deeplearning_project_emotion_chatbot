"""
action_recommendation.py
역할: 오늘 발화 분석 결과(roberta_out)와 실시간 웰니스 결과를 바탕으로
      "오늘의 감정 기록 기반 자기관리 추천"을 실시간으로 생성한다(행동 추천 v1).
      - DB 저장이나 새 테이블 없이 매 발화마다 실시간 추천 list만 반환한다.
      - 의료/진단 조언이 아니라 부드러운 자기관리 참고 문구로만 표현한다.
      - Qwen 상담 응답 생성 로직과는 분리되어 있으며, 응답 본문을 바꾸지 않는다.
입력: roberta_out(dict), wellness_result(dict), is_crisis(bool)
출력: list[dict] — 추천 카드 항목 리스트(최대 2개, 위기 시 안전 추천 1개)
"""

# 한 번에 보여줄 추천 최대 개수 (위기 안전 추천은 별도로 1개만 반환)
MAX_RECOMMENDATIONS = 2

# 우울 경향 점수 밴드 임계값 — depression_tendency_v15_spec과 동일
TENDENCY_HIGH = 0.40
TENDENCY_MID = 0.20


def _safety_recommendations() -> list[dict]:
    """
    역할: 위기 신호가 감지된 경우 일반 추천 대신 안전 확인 추천만 구성한다.
    입력: 없음
    출력: 안전 추천 1개를 담은 list[dict]
    """
    return [
        {
            "id": "safety_check",
            "title": "지금은 안전이 가장 먼저예요",
            "message": (
                "많이 힘드신 것 같아요. 혼자 견디지 마시고 가까운 사람이나 "
                "전문 상담에 연락해 보세요. 자살예방상담전화 1393은 24시간 열려 있어요."
            ),
            "reason": "위기 신호가 감지됨",
            "priority": "high",
            "category": "safety",
        }
    ]


def _tendency_high_recommendations() -> list[dict]:
    """우울 경향이 비교적 뚜렷할 때(>=0.40) 지원 추천을 구성한다."""
    return [
        {
            "id": "tendency_support",
            "title": "혼자 무겁게 두지 않기",
            "message": (
                "요즘 마음이 가라앉는 신호가 보여요. 믿을 만한 사람에게 가볍게 "
                "근황을 나눠보는 게 도움이 될 수 있어요."
            ),
            "reason": "우울 경향 신호가 비교적 뚜렷하게 감지됨",
            "priority": "high",
            "category": "tendency",
        },
        {
            "id": "tendency_small_step",
            "title": "아주 작은 한 가지만",
            "message": (
                "큰 계획 대신 물 한 잔 마시기나 잠깐 햇빛 쐬기처럼 아주 작은 것 "
                "하나만 가볍게 시도해보세요."
            ),
            "reason": "작은 행동 하나가 기분 회복에 도움이 될 수 있음",
            "priority": "medium",
            "category": "tendency",
        },
    ]


def _tendency_mid_recommendations() -> list[dict]:
    """약한 우울 경향(0.20~0.40)일 때 수면/식사/기운/흥미 점검 추천을 구성한다."""
    return [
        {
            "id": "selfcare_check",
            "title": "기본 컨디션 점검",
            "message": (
                "오늘 잠과 식사는 어땠는지, 평소 좋아하던 일에 마음이 가는지 "
                "가볍게 살펴보세요. 한 가지만 챙겨도 충분해요."
            ),
            "reason": "약한 우울 경향 신호가 감지됨",
            "priority": "medium",
            "category": "selfcare",
        },
        {
            "id": "selfcare_rest",
            "title": "기운 채우는 시간",
            "message": (
                "무리하지 말고 오늘은 조금 일찍 쉬거나 좋아하는 것으로 잠깐 "
                "기운을 채워보는 것도 좋아요."
            ),
            "reason": "기운·흥미 저하를 가볍게 챙기기 위함",
            "priority": "low",
            "category": "selfcare",
        },
    ]


def _routine_recommendations() -> list[dict]:
    """일상 과부하(routine_discomfort)일 때 부담을 줄이는 추천을 구성한다."""
    return [
        {
            "id": "routine_break",
            "title": "할 일 하나만 줄이기",
            "message": (
                "오늘은 부담 신호가 조금 보여요. 지금 해야 할 일 중 하나만 뒤로 "
                "미뤄보세요."
            ),
            "reason": "일상 과부하 표현이 감지됨",
            "priority": "medium",
            "category": "routine",
        },
        {
            "id": "routine_rest10",
            "title": "10분만 멈추기",
            "message": "잠깐 화면에서 벗어나 10분만 쉬어보세요. 짧은 휴식도 도움이 될 수 있어요.",
            "reason": "일상 과부하 표현이 감지됨",
            "priority": "low",
            "category": "routine",
        },
        {
            "id": "routine_priority",
            "title": "우선순위 하나만 정하기",
            "message": (
                "할 일이 많게 느껴진다면 그중 가장 중요한 한 가지만 골라보세요. "
                "나머지는 잠시 내려놔도 괜찮아요."
            ),
            "reason": "일상 과부하 표현이 감지됨",
            "priority": "low",
            "category": "routine",
        },
    ]


def _fear_recommendations() -> list[dict]:
    """공포/놀람일 때 사실 확인 + 호흡/grounding 추천을 구성한다."""
    return [
        {
            "id": "fear_fact",
            "title": "확인된 사실 하나 적기",
            "message": (
                "걱정이 앞설 때는 지금 분명히 알 수 있는 사실 한 가지를 적어보세요. "
                "막연함이 조금 줄어들 수 있어요."
            ),
            "reason": "불안·놀람 관련 감정이 감지됨",
            "priority": "medium",
            "category": "grounding",
        },
        {
            "id": "fear_breath",
            "title": "천천히 호흡하기",
            "message": (
                "숨을 4초 들이쉬고 6초 내쉬며 몇 번 반복해보세요. 발이 바닥에 닿는 "
                "감각에 집중하는 것도 도움이 될 수 있어요."
            ),
            "reason": "불안·놀람 관련 감정이 감지됨",
            "priority": "low",
            "category": "grounding",
        },
    ]


def _anger_recommendations() -> list[dict]:
    """분노일 때 즉시 반응을 미루고 감정을 정리하는 추천을 구성한다."""
    return [
        {
            "id": "anger_pause",
            "title": "바로 답하지 않기",
            "message": (
                "화가 날 때는 곧바로 답장하거나 결정하지 않는 게 도움이 될 수 있어요. "
                "잠깐 시간을 두어보세요."
            ),
            "reason": "분노 관련 감정이 감지됨",
            "priority": "medium",
            "category": "anger",
        },
        {
            "id": "anger_journal",
            "title": "감정 문장 따로 적기",
            "message": (
                "상대에게 보내기 전에 지금 드는 감정을 따로 한 문장으로 적어보세요. "
                "마음 정리에 도움이 될 수 있어요."
            ),
            "reason": "분노 관련 감정이 감지됨",
            "priority": "low",
            "category": "anger",
        },
    ]


def _sad_recommendations() -> list[dict]:
    """슬픔일 때 마음에 남은 장면을 가볍게 기록하는 추천을 구성한다."""
    return [
        {
            "id": "sad_note",
            "title": "마음에 남은 한 줄",
            "message": (
                "오늘 마음에 남은 장면이나 기분을 한 줄로 적어보세요. 꼭 정리하지 "
                "않아도 괜찮아요."
            ),
            "reason": "슬픔 관련 감정이 감지됨",
            "priority": "medium",
            "category": "reflect",
        },
        {
            "id": "sad_kind",
            "title": "스스로에게 다정하게",
            "message": (
                "힘든 마음이 들 땐 친한 친구에게 하듯 스스로에게도 다정한 말 한마디를 "
                "건네보세요."
            ),
            "reason": "슬픔 관련 감정이 감지됨",
            "priority": "low",
            "category": "reflect",
        },
    ]


def _positive_recommendations() -> list[dict]:
    """행복/긍정 공유일 때 좋은 순간을 저장하고 이어가는 추천을 구성한다."""
    return [
        {
            "id": "positive_save",
            "title": "좋았던 순간 저장하기",
            "message": (
                "오늘 기분 좋았던 순간을 한 줄로 남겨보세요. 나중에 다시 꺼내볼 수 있어요."
            ),
            "reason": "긍정적인 감정·공유가 감지됨",
            "priority": "low",
            "category": "positive",
        },
        {
            "id": "positive_repeat",
            "title": "내일 반복할 작은 것",
            "message": (
                "오늘 좋았던 것 중 내일도 가볍게 반복할 수 있는 작은 요소 하나를 "
                "골라보세요."
            ),
            "reason": "긍정적인 감정·공유가 감지됨",
            "priority": "low",
            "category": "positive",
        },
    ]


def _checkin_recommendations() -> list[dict]:
    """중립·저신호 또는 보통 상태일 때 짧은 체크인 유지 추천을 구성한다."""
    return [
        {
            "id": "checkin",
            "title": "짧은 체크인 유지",
            "message": (
                "특별한 신호는 크게 보이지 않아요. 오늘 기분을 한 줄로 가볍게 "
                "기록해두는 정도면 충분해요."
            ),
            "reason": "중립·저신호 또는 보통 상태",
            "priority": "low",
            "category": "checkin",
        }
    ]


def build_chat_recommendations(
    roberta_out: dict,
    wellness_result: dict,
    is_crisis: bool = False,
) -> list[dict]:
    """
    역할: 발화 분석 결과로 "오늘의 추천" 카드 항목을 실시간 생성한다(행동 추천 v1).
          우선순위는 위기 > 우울 경향(높음) > 우울 경향(약함) > 일상 과부하 >
          감정(공포·놀람/분노/슬픔/행복) > 중립·보통 순이며, 최대 2개만 반환한다.
    입력:
      - roberta_out: RoBERTa 추론 결과 dict (top_emotion, depression_tendency_score,
                     utterance_type 등을 사용; 키가 없어도 안전하게 동작)
      - wellness_result: 실시간 웰니스 결과 dict ({wellness_score, label})
      - is_crisis: 최종 위기 여부(하드/소프트 인터럽트 포함). True면 안전 추천만 반환.
    출력: list[dict] — {id, title, message, reason, priority, category} 형태(최대 2개)
    """
    roberta_out = roberta_out or {}
    wellness_result = wellness_result or {}

    # 1) 위기: 일반 추천 대신 안전 확인 추천만 반환한다.
    if is_crisis:
        return _safety_recommendations()

    # 입력값 방어적 추출 — 키가 없거나 None이어도 기본값으로 동작한다.
    try:
        tendency = float(roberta_out.get("depression_tendency_score") or 0.0)
    except (TypeError, ValueError):
        tendency = 0.0
    top_emotion = roberta_out.get("top_emotion")
    utterance_type = roberta_out.get("utterance_type")
    label = wellness_result.get("label")

    # 2) 우울 경향 우선 처리
    if tendency >= TENDENCY_HIGH:
        recs = _tendency_high_recommendations()
    elif tendency >= TENDENCY_MID:
        recs = _tendency_mid_recommendations()
    # 3) 일상 과부하
    elif utterance_type == "routine_discomfort":
        recs = _routine_recommendations()
    # 4) 감정 기반 추천
    elif top_emotion in ("공포", "놀람"):
        recs = _fear_recommendations()
    elif top_emotion == "분노":
        recs = _anger_recommendations()
    elif top_emotion == "슬픔":
        recs = _sad_recommendations()
    elif top_emotion == "행복" or utterance_type == "positive_share":
        recs = _positive_recommendations()
    # 5) 중립·저신호 또는 보통 상태
    elif utterance_type in ("casual_neutral", "casual_share") or label == "보통":
        recs = _checkin_recommendations()
    # 6) 그 외 매칭되지 않는 경우에도 빈 카드가 되지 않도록 체크인을 기본 제공한다.
    else:
        recs = _checkin_recommendations()

    return recs[:MAX_RECOMMENDATIONS]
