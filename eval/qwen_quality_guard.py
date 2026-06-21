"""
qwen_quality_guard.py
역할: Qwen 응답 후처리 품질 게이트를 모델 로딩 없이 단위 검증한다.
입력: 없음 (내장 케이스)
출력: 콘솔 검증 결과 및 실패 시 AssertionError
실행: C:/Users/WD/anaconda3/envs/dl_study/python.exe -u eval/qwen_quality_guard.py
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from models.qwen.inference_qwen import (
    build_utterance_style_instruction,
    diversify_repeated_response,
    postprocess_response,
)


BAD_CASES = [
    (
        "오늘 운동했어",
        "오늘 운동했어요?",
        "짧은 일상 발화 되묻기 차단",
    ),
    (
        "철봉 매달리기 했는데 버티기 힘들더라",
        "네 좋습니다. 그다음에는 자세히 좀 이야기 좀 해봐 주세요.",
        "상담 녹취 말투 차단",
    ),
    (
        "요즘 일이 너무 많아서 힘들어",
        "유비님도 새벽 세 시부터 근무하는데 커피 섭취하실까요?",
        "인물명·근무 페르소나 차단",
    ),
    (
        "안녕하세요",
        "最难坚持呢 nighttime insomnia",
        "다국어 유출 차단",
    ),
    (
        "브라우저 시연 확인용으로 오늘 기분이 괜찮아.",
        "좋은 시작이었군요. 오늘부터 다시 출발하자는 다짐이 있었던 순간이었군ですね.",
        "일본어 kana 잔여 토큰 차단",
    ),
    (
        "오늘 운동했어",
        "오늘 저녁에 운동하셨군요. 그럼 근육을 다 약하게 쉬우실 거 같아요.",
        "일상 키워드 반복·문법 붕괴 차단",
    ),
    (
        "요즘 일이 너무 많아서 힘들어",
        "많으셔요. 그럼 혹시 근데 최근에 그런 일이 있었나요?",
        "비문·되묻기 루프 차단",
    ),
    (
        "나 오늘 기분 너무 좋아",
        "전부 잘 됐으니까 이제 네 이제 이제 네 그러면 이렇게 네 그럼 이제 네 그래요.",
        "필러 반복 붕괴 차단",
    ),
    (
        "철봉 매달리기 했는데 버티기 힘들더라",
        "매달리기는 근데 손목이나 팔을 다 비틀고 움켜잡으셔서 그런 거예요.",
        "운동 원인 단정 차단",
    ),
    (
        "요즘 일이 너무 많아서 힘들어",
        "무엇보다 이제 집에서 혼자 살면서 그런 생각들이 들었으면 조금 안전하게 위안을 찾고 싶으실 거 같아요.",
        "없는 생활상 환각 차단",
    ),
    (
        "요즘 일이 너무 많아서 힘들어",
        "많으셔서 정말 잘 앉고 계시고 이런 것도 나쁘게 느껴요. 그러면 또 이렇게 걱정이 되면서.",
        "공감 없는 비문 응답 차단",
    ),
    (
        "잠이 잘 안 와",
        "지금은 스스로를 너무 곤경에 몰아넣으려고 하는 생각보다 더 안전하게 위험을 느끼고 있어요.",
        "수면 발화 위기 문구 오염 차단",
    ),
    (
        "나 오늘 기분 너무 좋아",
        "그래서 지금 이제 조금 더 평온하게 생각할 수 있는 그런 시간 되니깐 좋았던 거 같아요.",
        "긍정 발화 녹취 잔재 차단",
    ),
    (
        "내일이 기대돼",
        "잘 먹고 잘 자고 이렇게 쉬워질 수 있는 시간이 필요해요.",
        "긍정 기대 발화 처방형 오응답 차단",
    ),
    (
        "멜론이랑 스포티파이중에 뭐가 더 좋을까?",
        "그런데 선생님들이 조금 어린 연령층에서 넣으라고 하더라고요.",
        "추천 질문 상담 녹취 잔재 차단",
    ),
    (
        "밥 먹었어",
        "좋아요. 여기서는 조금 더 느긋할 수 있어요. 그러니깐 지금 이렇게 나와 있는 게 아니라 내가 왔다고 하는 거예요.",
        "v3 일상 발화 녹취 잔재 차단",
    ),
    (
        "내일이 기대돼",
        "좋아요. 그러면 저도 같이 생각을 나눠볼 수 있어요. 그리고 이렇게 시간을 정해두고 와서 얘기를 하는 거야.",
        "v3 긍정 발화 세션 말투 차단",
    ),
    (
        "요즘 계속 우울해",
        "그럼 지금 마음이 반이 차압되고 외부에서 평온하게 온전하게 느껴지지 않는 느낌이에요.",
        "v3 정서 발화 비문 차단",
    ),
    (
        "오늘 뭐 먹지?",
        "아까 편하게 커브드 아메리카 오토그래프 셋째 컵을 마셨어요. 네, 저녁엔 약간 조금씩 더 차워서 넣으니까 너무 막아야 하나 이렇게 걱정되는 거예요.",
        "v3 실용 질문 환각 응답 차단",
    ),
    (
        "밥 먹었어",
        "아니요? 그래도 오늘 아침부터 시작해서 꽤 꾸중했던 거예요.",
        "실제 일상 발화 질문형·비문 차단",
    ),
    (
        "요즘 계속 우울해",
        "계속해서 우울함이 있어요. 어떤 일이 지금 가장 마음을 위협하나요?",
        "실제 정서 발화 과한 표현 차단",
    ),
    (
        "친구랑 싸웠어",
        "그때 느낌은 어떤 것이었든 너무 마음이 위축되면서 시작됐던 거 같아요.",
        "실제 관계 발화 상황 환각 차단",
    ),
    (
        "잠이 잘 안 와",
        "그런 저번에도 또 그런 생각 들었다는 게 이렇게 계속 되더라고요. 그래서 지금까지 이게 어떻게 그렇게 계속 됐나라는 걸 기억하게 돼요.",
        "실제 수면 발화 녹취 잔재 차단",
    ),
    (
        "밥 먹었어",
        "좋아요. 오늘 아침부터 꽤 꾸중 안 하고 잘 수 있었던 것 같아요.",
        "실제 일상 발화 꾸중 환각 차단",
    ),
    (
        "요즘 계속 우울해",
        "말만 나와도 마음이 많이 위독해진 순간이 있었군요. 지금 몸이 얼마나 피곤해요?",
        "실제 정서 발화 위독 표현 차단",
    ),
    (
        "밥 먹었어",
        "좋아요. 저녁까지 시간이 조금 남았군요. 이제 밤 10시 이후이니 대부분 잠자기 전 활동이었습니다.",
        "실제 일상 발화 시간대 환각 차단",
    ),
    (
        "친구랑 싸웠어",
        "말씀을 너무 애써주셔서 저도 이제 조금씩 위안을 느끼고 있어요.",
        "실제 관계 발화 assistant 감정 주입 차단",
    ),
    (
        "재밌는거 뭐 없을까?",
        "둘 중 고르라면, 자주 쓰는 기준을 먼저 보면 좋아요. 익숙함과 국내 콘텐츠가 중요하면 앞쪽 선택지, 추천 알고리즘이나 다양한 탐색이 중요하면 뒤쪽 선택지가 더 잘 맞을 수 있어요.",
        "놀거리 추천 질문 비교 템플릿 오응답 차단",
    ),
    (
        "친구랑 싸웠어",
        "그런 일 있어요? 지금 몸이 어떻게 느껴져요? 어느 정도 위안이 되더라고요.",
        "결정형 생성 관계 발화 녹취 잔재 차단",
    ),
    (
        "잠이 잘 안 와",
        "마음이 조금 불안해서 그런 느낌이 들었나요? 아니면 그냥 지금 몸이 피곤해서 그래요. 이렇게 구분해보세요.",
        "결정형 생성 수면 발화 지시형 오응답 차단",
    ),
    (
        "오늘 아침에 커피 마시고 수업 준비했어.",
        "아침부터 꽤 꺼내놓으셨어요. 오늘은 그런 느낌으로 시작하셨네요.",
        "실제 일상 발화 어색한 비유 차단",
    ),
    (
        "그냥 오늘 있었던 일을 조금 더 말하면 저녁엔 별일 없이 쉬었어.",
        "말씀을 이렇게 조금씩 넣어서 해드릴 수 있다니깐 꽤 잘 하고 계시더라고요.",
        "실제 일상 후속 녹취 잔재 차단",
    ),
    (
        "그냥 오늘 있었던 일을 조금 더 말하면 저녁엔 별일 없이 쉬었어.",
        "아마 그런 느낌이 들 정도였어요. 지금 생각하기엔 그렇게 크게 느껴진 게 아니더군요.",
        "실제 일상 후속 부자연 종결 차단",
    ),
    (
        "도서관에 갔다가 집에 와서 쉬었어.",
        "좋아요. 그런 일상적인 걸 조금 더 세밀하게 들여다보면 또 다른 느낌이 올 수도 있어요.",
        "실제 일상 발화 과한 성찰 유도 차단",
    ),
    (
        "친구랑 별일 없이 근황 얘기했어.",
        "좋아요. 그런 일에서 조금 위안을 느꼈나 봐요.",
        "실제 평범 관계 발화 과잉 해석 차단",
    ),
    (
        "사실 그 서운함이 아직 조금 남아 있어.",
        "그런 생각이 있었어요? 지금 다시 돌아와서 그 순간을 기억하는 게 조금 어렵더군요.",
        "실제 슬픔 후속 회상형 오응답 차단",
    ),
    (
        "그 기쁜 느낌을 내일도 이어가고 싶어.",
        "말씀을 들으니 지금까지 잘 지나갔다는 걸 기억하는 게 조금 쉬운데, 그런 생각이 들 정도였어요.",
        "실제 긍정 후속 어색한 기억 표현 차단",
    ),
    (
        "특별하진 않았지만 그냥 무난하게 지나간 하루였어.",
        "오늘은 그런 느낌이었어요. 그래서 지금까지 내가 어떻게 살아왔는지를 돌아보면서 이제는 좀 다른 걸 해야겠다는 생각이 들기도 하고.",
        "실제 평범 발화 과도한 자기성찰 차단",
    ),
    (
        "특별하진 않았지만 그냥 무난하게 지나간 하루였어.",
        "그런 느낌이 있었어요. 그래서 지금까지 내가 느끼던 이런 느낌들을 다시 한번 돌아보면서 이렇게 충분함을 찾을 수 있었다는 걸 확인할 수 있어요.",
        "실제 평범 발화 장황한 충분함 서사 차단",
    ),
    # 2026-05-05 holdout 운영 모니터링 BAD/SKIPPED raw 샘플 (anchor/self-check가 실제로 잡은 문구)
    (
        "예전엔 즐겁던 일들이 요즘은 전부 시들하게 느껴져",
        "그런 생각이 들 정도면 지금까지 몇 번째 느낌이었어요?",
        "슬픔 발화 의미 없는 양적 probe 차단",
    ),
    (
        "기대했던 답장이 오지 않아서 조금 허전했어",
        "마음이 흉흉해서 시작이었군요. 그런 느낌이 들 정도였어요? 지금은 그런 느낌이 드는 게 있어요.",
        "허전 발화 환각 metaphor·echo question 차단",
    ),
    (
        "내가 없어도 아무도 크게 달라지지 않을 것 같다는 생각이 자꾸 들어",
        "생각하는 게 너무 어렵고 또 그걸 다 밀고 나갈 만큼 위력을 갖고 계시거든요.",
        "허무 발화 부적절 강도 어휘 차단",
    ),
    (
        "심사 결과를 기다리는데 혹시 떨어질까 봐 손이 차가워져",
        "말씀을 들으니 지금 혼자 감당하기엔 너무 위험한 상태 같아요. 안전 우선이 돼야 해요.",
        "경도 평가 불안 발화 비위기성 위기 응답 차단",
    ),
    (
        "약속한 자료를 안 보내줘서 준비가 꼬였고 화가 났어",
        "그런 생각이 들 정도면 지금 마음이 너무 피곤해서 게으른 것 같아요.",
        "분노 발화 부적절 평가 응답 차단",
    ),
    (
        "내 실수만 크게 지적하는 분위기가 불공평하게 느껴졌어",
        "말씀을 들으니 지금 혼자 너무 나아지고 있다는 느낌이 듭니다.",
        "분노/억울 발화 모순 환각 차단",
    ),
    (
        "마감 직전에 일을 떠넘겨서 화가 쉽게 가라앉지 않아",
        "말씀을 들으니 지금 혼자 견디느라 얼마나 피곤하셨을까요.",
        "분노 발화 녹취체 환각 차단",
    ),
    (
        "갑작스러운 알림음에 심장이 철렁했어",
        "마음이 너무 쉽게 들켜버렸어요. 그런 느낌이 들었으면 조금 더 안정감이 느껴졌을 거예요.",
        "놀람 발화 환각 metaphor 차단",
    ),
    (
        "오늘 들은 말이 계속 마음에 남아서 서운해",
        "마음이 아직 조금 아ching해요? 그런 느낌이랄까요.",
        "서운함 발화 한·영 혼용 오타 차단",
    ),
    (
        "한동안 계속 입맛도 없고 마음도 텅 빈 것 같아",
        "마음이 텅 비었을 만큼 지금 위험이 느껴진 순간들이 있었군요.",
        "지속 우울 발화 비위기성 위기 어휘 차단",
    ),
    # 2026-05-11 style×emotion review — chatbot_self_state 정형 패턴 박제
    (
        "사람들 만나기도 싫고 혼자만 있고 싶어",
        "말씀을 들으니 지금 혼자 있는 시간이 너무 좋아보여요. 그래도 스스로를 잘 돌봐줘야 돼요.",
        "고립 발화 긍정 misread 차단",
    ),
    (
        "내가 한 일을 다른 사람이 가져간 것처럼 말해서 분했어",
        "말씀을 들으니 지금 혼자서 겪고 있는 것이 너무 커보여요.",
        "분노 발화 봇 1인칭 과장 차단",
    ),
    (
        "내가 쓸모없는 사람 같다는 생각이 자꾸 들어",
        "말씀을 들으니 지금 혼자 감당해야 할 문제가 너무 크고 위협적으로 느껴질 때가 있어요.",
        "자기비하 발화 위협 강도 부여 차단",
    ),
    (
        "상한 냄새가 올라와서 속이 울렁거렸어",
        "생각하는 동안 몸이 조금 피곤하게 느껴졌던 순간이 있었나요?",
        "혐오 발화 봇 fictitious 신체 probe 차단",
    ),
    (
        "예전엔 즐겁던 일들이 요즘은 전부 시들하게 느껴져",
        "생각하는 게 조금 어렵더라고요. 그런 생각이 들었을 때가 언제부터 있었어요?",
        "무기력 발화 봇 1인칭 self state 차단",
    ),
    (
        "마감 직전에 일을 떠넘겨서 화가 쉽게 가라앉지 않아",
        "말씀을 들으니 지금 혼자 견디느라 몸 피곤해서 힘이 다 나갈 것 같애요.",
        "분노 발화 봇 신체 상태 주입 차단",
    ),
    (
        "내 실수만 크게 지적하는 분위기가 불공평하게 느껴졌어",
        "그런데 지금은 내가 너무 자기 자신을 비판해서 마음이 안 좋아져서 그런 거 같아요.",
        "관계 발화 봇 1인칭 자기 평가 차단",
    ),
    (
        "면접 질문에 막힐까 봐 계속 머릿속으로 연습하게 돼",
        "그런데 지금은 조금 더 그런 느낌이 들었어요. 제가 느끼는 게 좀 더 확실해졌어요.",
        "예기 불안 발화 봇 1인칭 감정 진술 차단",
    ),
    (
        "작게 정한 목표를 지켜서 나 자신이 조금 대견했어",
        "좋아요. 그런 작은 성취감이 있다니 이제 조금 더 스스로에게 칭찬할 수 있어요.",
        "긍정 발화 봇 1인칭 자기 긍정 차단",
    ),
    (
        "조용하던 방에서 큰 소리가 나서 깜짝 놀랐어",
        "아니요? 저도 그런 경험 있어요. 누군가가 와서 크게 문을 열었을 때는 정말 놀라워요.",
        "놀람 발화 봇 personal disclosure 차단",
    ),
    (
        "작게 정한 목표를 지켜서 나 자신이 조금 대견했어",
        "생각하는 게 이렇게 어렵게 느껴져요. 지금 생각하는 게 너무 위험하다고 느껴질 정도예요.",
        "긍정 발화 봇 over-severity self state 차단",
    ),
    (
        "내가 한 일을 다른 사람이 가져간 것처럼 말해서 분했어",
        "말씀을 들으니 지금 혼자서 감당해야 할 문제가 너무 크다고 느껴져요.",
        "분노 발화 '혼자서' 변형 over-severity 차단",
    ),
    (
        "거짓말을 아무렇지 않게 하는 태도가 역겨웠어",
        "맞아요. 지금은 내가 얼마나 위험하게 느껴졌는지를 생각하는 게 아니라 그냥 걱정이 너무 많아서.",
        "혐오 발화 봇 1인칭 위협 자각 차단",
    ),
]

GOOD_CASES = [
    (
        "요즘 일이 너무 많아서 힘들어",
        "일이 계속 밀려 있으면 몸도 마음도 쉽게 지칠 수 있어요. 오늘 가장 버거웠던 부분이 뭐였나요?",
    ),
    (
        "나 오늘 기분 너무 좋아",
        "그렇게 좋은 기분이 올라온 날이라니 반가워요. 오늘의 좋은 감각을 충분히 누려도 좋겠어요.",
    ),
]


def assert_bad_cases_are_replaced() -> None:
    """
    역할: 저품질 응답이 원문 그대로 통과하지 않는지 확인한다.
    입력: 없음
    출력: 없음
    """
    for user_text, raw_response, reason in BAD_CASES:
        final_response = postprocess_response(user_text, raw_response)
        assert final_response != raw_response, reason
        assert "[CRISIS]" not in final_response, reason
        print(f"[대체 확인] {reason}: {final_response}")


def assert_good_cases_pass() -> None:
    """
    역할: 정상 응답이 불필요하게 대체되지 않는지 확인한다.
    입력: 없음
    출력: 없음
    """
    for user_text, raw_response in GOOD_CASES:
        final_response = postprocess_response(user_text, raw_response)
        assert final_response == raw_response, final_response
        print(f"[통과 확인] {final_response}")


def assert_crisis_tag_is_preserved() -> None:
    """
    역할: [CRISIS] 태그가 후처리에서 제거되지 않는지 확인한다.
    입력: 없음
    출력: 없음
    """
    raw_response = "지금은 안전이 먼저예요. 109에 바로 도움을 요청해 주세요. [CRISIS]"
    final_response = postprocess_response("죽고 싶어", raw_response)
    assert "[CRISIS]" in final_response
    print("[태그 보존] [CRISIS] 유지")


def assert_utterance_style_instruction() -> None:
    """
    역할: 발화 타입별 Qwen 스타일 지시문이 생성되는지 확인한다.
    입력: 없음
    출력: 없음
    """
    cases = [
        ("casual_neutral", "질문은 붙이지 마세요"),
        ("routine_discomfort", "일상 불편"),
        ("emotional_distress", "정서적 어려움"),
        ("crisis_candidate", "[CRISIS]"),
    ]
    for utterance_type, expected_phrase in cases:
        instruction = build_utterance_style_instruction({
            "utterance_type": utterance_type,
            "utterance_type_confidence": 0.82,
            "utterance_type_reason": "test_reason",
        })
        assert expected_phrase in instruction, instruction
        assert "test_reason" in instruction, instruction
        print(f"[스타일 지시 확인] {utterance_type}")


def assert_typed_fallback_response() -> None:
    """
    역할: 저품질 응답 대체 문장이 발화 타입별로 달라지는지 확인한다.
    입력: 없음
    출력: 없음
    """
    final_response = postprocess_response(
        "출근하기 싫다",
        "네 네 이제 네 그러면 이제 네 그래요.",
        utterance_info={"utterance_type": "routine_discomfort"},
    )
    assert "시작" in final_response or "첫 단계" in final_response, final_response
    print(f"[타입별 fallback 확인] {final_response}")

    routine_question_response = postprocess_response(
        "출근하기 싫다",
        "아침부터 피로가 크게 느껴졌어요? 오늘은 조금 더 쉽게 시작되었으면 해요.",
        utterance_info={"utterance_type": "routine_discomfort"},
    )
    assert "시작" in routine_question_response or "첫 단계" in routine_question_response, routine_question_response
    print(f"[일상 불편 질문형 차단 확인] {routine_question_response}")

    routine_misread_response = postprocess_response(
        "출근하기 싫다",
        "너무 편하게 쉬길 바라는 마음이었군요. 그래도 몇 가지 준비하는 게 좋아보여요.",
        utterance_info={"utterance_type": "routine_discomfort"},
    )
    assert "시작" in routine_misread_response or "첫 단계" in routine_misread_response, routine_misread_response
    print(f"[일상 불편 오독 차단 확인] {routine_misread_response}")

    routine_home_response = postprocess_response(
        "출근하기 싫다",
        "저런! 이렇게 집에서 쉬고 싶었어요. 지금까지 너무 힘든 시간들이 있었네요.",
        utterance_info={"utterance_type": "routine_discomfort"},
    )
    assert "시작" in routine_home_response or "첫 단계" in routine_home_response, routine_home_response
    print(f"[일상 불편 집맥락 환각 차단 확인] {routine_home_response}")

    sleep_response = postprocess_response(
        "잠이 잘 안 와",
        "지금은 스스로를 너무 곤경에 몰아넣으려고 하는 생각보다 더 안전하게 위험을 느끼고 있어요.",
        utterance_info={"utterance_type": "emotional_distress"},
    )
    assert "잠" in sleep_response and ("쉬" in sleep_response or "회복" in sleep_response or "몸" in sleep_response), sleep_response
    print(f"[수면 fallback 확인] {sleep_response}")

    relationship_response = postprocess_response(
        "친구랑 싸웠어",
        "네 네 이제 네 그러면 이제 네 그래요.",
        utterance_info={"utterance_type": "emotional_distress"},
    )
    assert "관계" in relationship_response or "가까운 사람" in relationship_response or "친한 사람" in relationship_response, relationship_response
    print(f"[관계 fallback 확인] {relationship_response}")

    positive_expectation_response = postprocess_response(
        "내일이 기대돼",
        "잘 먹고 잘 자고 이렇게 쉬워질 수 있는 시간이 필요해요.",
        utterance_info={"utterance_type": "casual_neutral"},
    )
    assert any(marker in positive_expectation_response for marker in ["기대", "설레", "기다려", "좋은"]), positive_expectation_response
    print(f"[기대 발화 fallback 확인] {positive_expectation_response}")

    preference_response = postprocess_response(
        "멜론이랑 스포티파이중에 뭐가 더 좋을까?",
        "그런데 선생님들이 조금 어린 연령층에서 넣으라고 하더라고요.",
        utterance_info={"utterance_type": "preference_question"},
    )
    assert "멜론" in preference_response and "스포티파이" in preference_response, preference_response
    print(f"[추천 질문 fallback 확인] {preference_response}")

    preference_empathy_response = postprocess_response(
        "멜론이랑 스포티파이중에 뭐가 더 좋을까?",
        "맞아요. 그래도 나름의 이유가 있었던 거예요. 좋아요.",
        utterance_info={"utterance_type": "preference_question"},
    )
    assert "멜론" in preference_empathy_response and "스포티파이" in preference_empathy_response, preference_empathy_response
    print(f"[추천 질문 공감 회피 확인] {preference_empathy_response}")

    practical_response = postprocess_response(
        "오늘 뭐 먹지?",
        "아까 편하게 커브드 아메리카 오토그래프 셋째 컵을 마셨어요.",
        utterance_info={"utterance_type": "practical_question"},
    )
    assert "김밥" in practical_response or "국밥" in practical_response, practical_response
    print(f"[실용 질문 fallback 확인] {practical_response}")

    practical_first_person_response = postprocess_response(
        "오늘 뭐 먹지?",
        "아니요, 저 지금 먹지 않았어요. 제가 너무 편하게 먹으면 몇 시간 동안 쉬면서 먹거든요.",
        utterance_info={"utterance_type": "practical_question"},
    )
    assert "김밥" in practical_first_person_response or "국밥" in practical_first_person_response, practical_first_person_response
    print(f"[실용 질문 1인칭 환각 차단 확인] {practical_first_person_response}")

    entertainment_response = postprocess_response(
        "재밌는거 뭐 없을까?",
        "둘 중 고르라면, 자주 쓰는 기준을 먼저 보면 좋아요.",
        utterance_info={"utterance_type": "preference_question"},
    )
    assert "예능" in entertainment_response or "영화" in entertainment_response or "게임" in entertainment_response, entertainment_response
    print(f"[놀거리 추천 fallback 확인] {entertainment_response}")

    anxiety_relief_response = postprocess_response(
        "불안감 해소?",
        "좋아요, 오늘의 작은 일상도 잘 지나가고 있네요.",
        utterance_info={"utterance_type": "practical_question"},
    )
    assert "불안" in anxiety_relief_response
    assert "숨" in anxiety_relief_response or "안정" in anxiety_relief_response
    print(f"[불안 완화 질문 fallback 확인] {anxiety_relief_response}")

    tension_relief_response = postprocess_response(
        "시험 긴장 완화 방법?",
        "좋아요, 오늘의 작은 일상도 잘 지나가고 있네요.",
        utterance_info={"utterance_type": "practical_question"},
    )
    assert (
        "숨" in tension_relief_response
        or "안정" in tension_relief_response
        or "시험" in tension_relief_response
        or "긴장" in tension_relief_response
    )
    print(f"[시험 긴장 완화 질문 fallback 확인] {tension_relief_response}")

    academic_anxiety_response = postprocess_response(
        "내일 시험이 있어서 긴장돼",
        "좋아요, 오늘의 작은 일상도 잘 지나가고 있네요.",
        utterance_info={"utterance_type": "emotional_distress"},
    )
    assert "시험" in academic_anxiety_response
    assert "긴장" in academic_anxiety_response or "차분" in academic_anxiety_response
    print(f"[시험 전 긴장 fallback 확인] {academic_anxiety_response}")

    mild_unease_response = postprocess_response(
        "이상한 기분이 들어",
        "생각하는 게 너무 어렵고 또 피곤하게 느껴진다면 지금의 상태가 충분합니다.",
        utterance_info={"utterance_type": "emotional_distress"},
    )
    assert "이상한 느낌" in mild_unease_response
    assert "충분합니다" not in mild_unease_response
    print(f"[막연한 불편감 fallback 확인] {mild_unease_response}")

    mild_low_mood_response = postprocess_response(
        "기분이 별로야",
        "별 생각이 들지 않아서 그런 느낌이었나 봐요. 지금 기억이 나지 않는 순간들이 몇 번 있었나요?",
        utterance_info={"utterance_type": "emotional_distress"},
    )
    assert any(marker in mild_low_mood_response for marker in ["기분", "마음", "컨디션", "쉬"]), mild_low_mood_response
    assert "기억이 나지" not in mild_low_mood_response
    print(f"[가벼운 저조감 fallback 확인] {mild_low_mood_response}")

    positive_recovery_response = postprocess_response(
        "응 이제 조금 낫네",
        "좋아요, 오늘의 작은 일상도 잘 지나가고 있네요.",
        utterance_info={"utterance_type": "positive_share"},
    )
    assert any(marker in positive_recovery_response for marker in ["나아진", "편해진", "무리하지"]), positive_recovery_response
    assert "작은 일상" not in positive_recovery_response
    print(f"[회복 발화 fallback 확인] {positive_recovery_response}")

    anger_response = postprocess_response(
        "상대가 말을 너무 무례하게 해서 화가 났어.",
        "네 네 이제 네 그러면 이제 네 그래요.",
        utterance_info={"utterance_type": "emotional_distress"},
    )
    assert any(marker in anger_response for marker in ["화", "억울", "무례", "짜증"]), anger_response
    print(f"[화남 fallback 확인] {anger_response}")

    casual_default_response = postprocess_response(
        "그냥 창밖 봤어",
        "네 네 이제 네 그러면 이제 네 그래요.",
        utterance_info={"utterance_type": "casual_share"},
    )
    assert "작은 일상도 잘 지나가고" not in casual_default_response
    print(f"[일상 fallback 반복 문구 제거 확인] {casual_default_response}")

    neutral_overpositive_response = postprocess_response(
        "오늘 날씨가 괜찮아서 창문 열어뒀어.",
        "조금 나아진 느낌이 있다니 다행이에요. 지금 그 감각을 무리하지 말고 이어가 봐요.",
        utterance_info={"utterance_type": "casual_neutral"},
    )
    assert "나아진 느낌" not in neutral_overpositive_response
    assert "무리하지" not in neutral_overpositive_response
    print(f"[중립 발화 회복 문구 과적용 차단 확인] {neutral_overpositive_response}")

    neutral_positive_mismatch_response = postprocess_response(
        "특별하진 않았지만 그냥 무난하게 지나간 하루였어.",
        "잘 지나간 일이 있었다는 게 반가워요. 그 기분을 너무 빨리 흘려보내지 않아도 괜찮아요.",
        utterance_info={"utterance_type": "casual_neutral"},
    )
    assert "기분을 너무 빨리" not in neutral_positive_mismatch_response
    assert "너무 빨리 흘려보내" not in neutral_positive_mismatch_response
    print(f"[평범 발화 긍정 문구 과적용 차단 확인] {neutral_positive_mismatch_response}")

    neutral_quote_mismatch_response = postprocess_response(
        "응, '좋아요. 오늘 있었던 일을 차분히 따라가고 있어요.'라는 말처럼 특별하진 않았지만 그냥 무난하게 지나간 하루였어.",
        "좋은 쪽으로 마음이 움직인 순간이 있었군요. 그 감각을 오늘의 밝은 장면으로 남겨둘게요.",
        utterance_info={"utterance_type": "positive_share"},
    )
    assert "좋은 쪽으로 마음" not in neutral_quote_mismatch_response
    assert "밝은 장면" not in neutral_quote_mismatch_response
    print(f"[평범 인용 후속 긍정 과해석 차단 확인] {neutral_quote_mismatch_response}")

    positive_fallback_response = postprocess_response(
        "오랜만에 친구한테 좋은 소식을 들어서 기분이 밝아졌어.",
        "네 네 이제 네 그러면 이제 네 그래요.",
        utterance_info={"utterance_type": "positive_share"},
    )
    assert "밝아진 흐름" not in positive_fallback_response
    assert "좋은 쪽으로 마음" not in positive_fallback_response
    assert "너무 빨리 흘려보내" not in positive_fallback_response
    print(f"[긍정 fallback 잔여 반복 문구 제거 확인] {positive_fallback_response}")


def assert_repeated_responses_are_diversified() -> None:
    """
    역할: 최근 응답과 같은 최종 응답이 노출 전 다른 fallback 문장으로 치환되는지 확인한다.
    입력: 없음
    출력: 없음
    """
    distress_candidate = "그만큼 버거운 시간을 지나고 있었군요. 지금 느끼는 마음을 급히 정리하려 하기보다, 여기서는 그대로 말해도 괜찮아요."
    distress_response, distress_replaced = diversify_repeated_response(
        "요즘 일이 너무 많아서 힘들어",
        distress_candidate,
        utterance_info={"utterance_type": "emotional_distress"},
        avoid_texts=[distress_candidate],
    )
    assert distress_replaced
    assert distress_response != distress_candidate
    assert "버거" in distress_response or "감정" in distress_response or "마음" in distress_response
    print(f"[정서 fallback 반복 회피 확인] {distress_response}")

    positive_candidate = "좋아요. 그러면 지금은 조금 더 평온한 느낌이랄까."
    positive_response, positive_replaced = diversify_repeated_response(
        "오늘 발표 잘 끝나서 너무 홀가분해",
        positive_candidate,
        utterance_info={"utterance_type": "positive_share"},
        avoid_texts=[positive_candidate],
    )
    assert positive_replaced
    assert positive_response != positive_candidate
    assert "작은 일상" not in positive_response
    assert any(marker in positive_response for marker in ["좋은", "홀가분", "밝아", "반가"]), positive_response
    print(f"[긍정 응답 반복 회피 확인] {positive_response}")


def main() -> None:
    """
    역할: 품질 게이트 단위 검증을 실행한다.
    입력: 없음
    출력: 없음
    """
    assert_bad_cases_are_replaced()
    assert_good_cases_pass()
    assert_crisis_tag_is_preserved()
    assert_utterance_style_instruction()
    assert_typed_fallback_response()
    assert_repeated_responses_are_diversified()
    print("[완료] Qwen 품질 게이트 검증 통과")


if __name__ == "__main__":
    main()
