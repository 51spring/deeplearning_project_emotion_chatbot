"""
utterance_type.py
역할: 사용자 발화를 상담 점수화 전에 7가지 발화 타입으로 분류한다.
      casual_share / positive_share / routine_discomfort / emotional_distress /
      preference_question / practical_question / crisis_candidate
입력: 사용자 발화 텍스트
출력: 발화 타입, 신뢰도, 판정 사유
"""

import re

UTTERANCE_TYPES = {
    "CASUAL_NEUTRAL": "casual_neutral",
    "CASUAL_SHARE": "casual_share",
    "POSITIVE_SHARE": "positive_share",
    "ROUTINE_DISCOMFORT": "routine_discomfort",
    "EMOTIONAL_DISTRESS": "emotional_distress",
    "PREFERENCE_QUESTION": "preference_question",
    "PRACTICAL_QUESTION": "practical_question",
    "CRISIS_CANDIDATE": "crisis_candidate",
}

ROUTINE_CONTEXTS = [
    "출근",
    "퇴근",
    "등교",
    "학교",
    "회사",
    "일하",
    "업무",
    "할일",
    "할게",
    "해야할일",
    "해야할게",
    "해야될일",
    "해야될게",
    "해야하는일",
    "해야하는게",
    "공부",
    "과제",
    "시험",
    "회의",
    "수업",
    "운동",
    "청소",
    "밥차리",
    "설거지",
]

ROUTINE_DISCOMFORT_MARKERS = [
    "싫",
    "귀찮",
    "짜증",
    "하기싫",
    "가기싫",
    "나가기싫",
    "하기싫다",
    "못하겠",
    "힘들",
    "피곤",
    "부담",
]

PHYSICAL_EXERTION_CONTEXTS = [
    "철봉",
    "매달리",
    "운동",
    "헬스",
    "러닝",
    "달리기",
    "등산",
    "산책",
    "계단",
    "스쿼트",
    "팔굽혀",
    "푸쉬업",
    "근력",
    "유산소",
    "스트레칭",
    "가게",
    "매장",
    "마트",
    "편의점",
    "카페",
    "카운터",
    "알바",
    "근무",
    "퇴근",
    "학교",
    "바닥",
    "닦",
    "걸레",
    "걸레질",
    "대청소",
    "일하",
    "진열",
    "상하차",
    "서빙",
    "주방",
    "청소",
    "설거지",
    "빨래",
    "정리",
    "이사",
    "짐나르",
    "짐옮",
    "박스",
    "상품",
    "물건",
    "물류",
    "창고",
    "배달",
    "택배",
    "서서",
    "서있",
    "걸어",
    "걸었",
    "걸어서",
    "돌아다",
    "왔다갔다",
    "오래서",
    "오래걸",
    "오르내",
    "오르락내리락",
]

PHYSICAL_EXERTION_ACTIVITY_CONTEXTS = [
    "철봉",
    "매달리",
    "운동",
    "헬스",
    "러닝",
    "달리기",
    "뛰었",
    "등산",
    "산책",
    "계단",
    "스쿼트",
    "팔굽혀",
    "푸쉬업",
    "근력",
    "유산소",
    "스트레칭",
]

PHYSICAL_EXERTION_PAIN_MARKERS = [
    "숨차",
    "숨이차",
    "땀났",
    "땀이났",
    "근육",
    "알배",
    "뻐근",
    "쑤셔",
    "쑤신",
    "쑤시",
    "온몸",
    "몸살",
    "아파",
    "아프",
    "통증",
    "결려",
    "저려",
    "무거",
    "다리가아",
    "다리도아",
    "허리가아",
    "허리도아",
    "어깨가아",
    "어깨도아",
    "팔이아",
    "팔도아",
    "손목",
    "종아리",
    "허벅지",
    "목이아",
    "등이아",
    "땡겨",
    "당겨",
    "발바닥",
    "무릎",
]

PHYSICAL_EXERTION_FATIGUE_MARKERS = [
    "힘들",
    "힘들더라",
    "힘드",
    "피곤",
    "고되",
    "고됐",
    "고단",
    "지쳤",
    "지쳐",
    "녹초",
    "기운빠",
    "탈진",
    "뻗었",
]

PHYSICAL_EXERTION_EFFORT_MARKERS = [
    "하루종일",
    "온종일",
    "종일",
    "오래",
    "몇시간",
    "내내",
    "계속서",
    "서서",
    "서있",
    "걸어",
    "걸었",
    "걸어서",
    "돌아다",
    "왔다갔다",
    "일하다",
    "일하고",
    "일했",
    "일하느라",
    "근무하고",
    "근무했",
    "근무하느라",
    "알바하고",
    "알바했",
    "알바하느라",
    "퇴근하고",
    "퇴근했",
    "진열하",
    "상하차",
    "청소하다",
    "청소했",
    "청소하고",
    "청소하느라",
    "청소를했",
    "바닥닦",
    "닦았",
    "닦고",
    "닦느라",
    "닦다가",
    "걸레질",
    "설거지하",
    "빨래하",
    "정리하",
    "서빙하",
    "배달하",
    "택배하",
    "나르",
    "옮기",
    "들고",
    "들었",
    "들다가",
    "오르내",
    "오르락내리락",
    "버티기",
    "쉬지못",
    "쉴틈",
]

PHYSICAL_EXERTION_EMOTIONAL_BLOCKERS = [
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
    "내탓",
    "눈물",
    "절망",
    "희망이없",
    "요즘계속",
    "매일우울",
    "계속우울",
    "계속무기력",
    "계속의욕",
]

PHYSICAL_EXERTION_SAFETY_BLOCKERS = [
    "숨이안쉬",
    "호흡이안",
    "숨을못",
    "가슴통증",
    "가슴이아",
    "쓰러질",
    "쓰러질것",
    "기절",
    "응급",
    "피가",
    "피났",
    "다쳤",
    "부러",
    "병원가야",
    "119",
]

DAILY_ROUTINE_CONTEXTS = [
    "아침",
    "점심",
    "저녁",
    "밥",
    "김밥",
    "라면",
    "커피",
    "메뉴",
    "먹었",
    "먹고",
    "끓여",
    "쉬었",
    "쉬고",
    "씻고",
    "누웠",
    "집에와",
    "도서관",
    "강의",
    "자료",
    "창문",
    "빨래",
    "정리",
    "세탁",
    "건조대",
    "분리수거",
    "메모장",
    "달력",
    "가방",
]

ADMIN_TECH_NEUTRAL_CONTEXTS = [
    "번호",
    "조회",
    "기록",
    "버전",
    "변경내역",
    "표로",
    "문서",
    "제목",
    "형식",
    "파일",
    "폴더",
    "분류",
    "캘린더",
    "예약",
    "계좌",
    "이체",
    "영수증",
    "프린터",
    "테스트페이지",
    "회의록",
    "일정조정",
    "배송상태",
    "자료이름",
    "출력",
    "노트북",
    "충전",
    "케이블",
    "서류",
    "투명파일",
    "이어폰",
    "배터리",
]

CASUAL_MARKERS = [
    "밥",
    "먹었",
    "운동했",
    "산책",
    "날씨",
    "영화",
    "게임",
    "음악",
    "카페",
    "커피",
    "잤어",
    "봤어",
    "했어",
]

POSITIVE_MARKERS = [
    "좋아",
    "좋았",
    "기분좋",
    "행복",
    "감동",
    "뿌듯",
    "해냈",
    "성공",
    "괜찮",
    "편안",
    "여유",
    "대견",
    "재밌",
    "웃긴",
    "웃었",
    "많이웃",
    "웃음",
    "기대",
    "설레",
    "신나",
    "기다려",
    "맛있",
    "들뜨",
    "선물",
    "드디어",
    "목표",
    "다행",
    "안도",
    "빨리끝",
    "잘됐",
    "잘풀",
    "가벼워",
    "따뜻",
    "산뜻",
    "밝아",
    "고생했다고",
    "속이시원",
    "시원했",
    "예뻐",
    "귀여",
    "귀엽",
    "깜찍",
    "사랑스",
    "앙증",
    "포근",
    "흐뭇",
    "부드럽게풀",
    "풀렸",
    "새잎",
    "반가",
    "고맙",
    "홀가분",
    "머리가맑",
    "맑아졌",
    "마음에들",
    "응원",
    "다시해볼힘",
    "힘이생",
    "잘버틴",
    "잘버텼",
    "버틴느낌",
    "나름잘",
    "해결",
    "제출하고나니",
]

POSITIVE_BLOCKING_MARKERS = [
    "속상",
    "서운",
    "상처",
    "불안",
    "걱정",
    "우울",
    "슬퍼",
    "외로",
    "무기력",
    "힘들",
    "괴로",
    "지쳐",
    "막막",
    "망쳤",
    "안돼",
    "안되",
    "안와",
    "안오",
    "오지않",
    "잘안돼",
    "잘안되",
    "즐겁지않",
    "예전만큼즐겁지않",
    "좋아하던일이예전만큼",
    "기대했던연락이안",
    "기대했던일이안",
    "반응이없",
    "아쉬",
    "쓸쓸",
    "허전",
    "아무렇지않은척",
    "기대고싶",
    "괜찮지않",
    "웃기지않",
    "웃기지도않",
    "웃을수없",
    "재미가없",
    "재미도없",
    "아무재미",
    "흥미가없",
    "손이안가",
    "손이가지않",
    "손도안가",
    "마음이움직이지않",
    "마음이잘움직이지않",
    "좋아하던일도손이안",
    "좋아하던것도손이안",
    "좋은소식을들어도마음이잘움직이지않",
    "좋아하던영상",
    "어리둥절",
]

ANXIETY_RELIEF_MARKERS = [
    "불안감해소",
    "불안해소",
    "불안완화",
    "불안줄",
    "불안낮",
    "불안진정",
    "불안할때",
    "불안할때뭐",
    "불안하면뭐",
]

PRACTICAL_QUESTION_MARKERS = [
    "방법",
    "어떻게",
    "뭐하면",
    "뭐하지",
    "뭐먹",
    "메뉴",
    "추천",
    "추천해줘",
    "해소",
    "완화",
    "줄이",
    "낮추",
    "진정",
]

MILD_UNEASE_MARKERS = [
    "이상한기분",
    "기분이이상",
    "묘한기분",
    "붕떠",
    "붕뜬",
    "마음이붕",
    "생각이많",
    "생각이너무많",
    "머리가복잡",
    "찜찜",
    "찝찝",
    "싱숭생숭",
    "뒤숭숭",
    "개운하지않",
    "불편한기분",
]

SENSORY_DISGUST_DIRECT_MARKERS = [
    "상한냄새",
    "역한냄새",
    "하수구냄새",
    "상한우유",
    "젖은행주",
    "축축한촉감",
    "울렁",
    "끈적",
    "축축",
    "비위",
    "곰팡",
    "찝찝",
    "기름기",
    "기름묻",
    "이상한식감",
    "식감이느껴",
    "지저분",
    "구역질",
    "얼굴을찡그",
    "컵을내려놨",
    "컵을내려놓",
    "오래된반찬냄새",
    "시큼한냄새",
    "시큼",
    "쉰내",
    "찌꺼기",
    "남은찌꺼기",
    "컵바닥",
    "반찬통",
    "걸레냄새",
    "젖은걸레",
]

SENSORY_DISGUST_CONTEXT_MARKERS = [
    "하수구",
    "냄새",
    "음식",
    "우유",
    "식감",
    "행주",
    "촉감",
    "축축",
    "책상",
    "자국",
    "화장실",
    "손에",
    "손잡이",
    "기름기",
    "기름묻",
    "곰팡",
    "청결",
    "먼지",
    "벌레",
    "쓰레기",
    "쓰레기통",
    "국물",
    "냉장고",
    "반찬",
    "반찬통",
    "반죽",
    "찌꺼기",
    "컵바닥",
    "걸레",
    "수세미",
]

SENSORY_DISGUST_AFFECT_MARKERS = [
    "불쾌",
    "찝찝",
    "비위",
    "울렁",
    "싫",
    "찡그",
    "내려놨",
    "내려놓",
    "보기힘들",
    "올라왔",
    "올라와",
    "퍼졌",
    "닫았",
    "마실생각이사라",
]

SOCIAL_MORAL_DISGUST_DIRECT_MARKERS = [
    "정이떨어",
    "정떨어",
    "역하게",
    "역겨",
    "약점을재미",
    "남의약점",
    "따돌리는분위기",
    "은근히따돌",
    "깎아내리",
    "마음이확식",
    "태도가역",
    "거짓말을웃",
]

SOCIAL_MORAL_DISGUST_CONTEXT_MARKERS = [
    "약점",
    "거짓말",
    "따돌리",
    "깎아내리",
    "조롱",
    "태도",
]

SOCIAL_MORAL_DISGUST_AFFECT_MARKERS = [
    "정이떨어",
    "정떨어",
    "역하",
    "역겨",
    "불편",
    "불쾌",
    "마음이확식",
    "싫",
]

MILD_LOW_MOOD_MARKERS = [
    "기분이별로",
    "기분별로",
    "기분이안좋",
    "기분안좋",
    "기분이가라앉",
    "마음이가라앉",
    "가라앉아",
    "가라앉는",
    "컨디션별로",
    "오늘별로",
    "좀별로",
    "별로야",
    "별로다",
]

LIMITED_SITUATIONAL_DISTRESS_CONTEXTS = [
    "과제",
    "발표",
    "시험",
    "공부",
    "수업",
    "마감",
    "프로젝트",
    "일",
]

LIMITED_SITUATIONAL_DISTRESS_MARKERS = [
    "어려웠",
    "어렵",
    "막막",
    "생각이안나",
    "주제가생각이안나",
    "속상",
    "망쳤",
    "망친",
    "못봐",
    "못봤",
    "못봐서",
    "안돼서",
    "안되서",
    "잘안돼",
    "잘안되",
]

FEAR_MARKERS = [
    "무섭",
    "무서",
    "겁나",
    "겁났",
    "겁이",
    "두려",
    "공포",
    "떨려",
    "긴장",
    "위험",
    "불안",
    "패닉",
    "공황",
]

SITUATIONAL_ANXIETY_CONTEXTS = [
    "계약서",
    "조항",
    "가스레인지",
    "결제",
    "계단",
    "계좌",
    "지갑",
    "카드",
    "천둥",
    "골목",
    "인기척",
    "백업",
    "발표파일",
    "새팀",
    "심사",
    "결과",
    "발표",
    "면접",
    "질문",
    "검사",
    "병원",
    "메일",
    "연락",
    "전화",
    "도로",
    "난간",
    "새환경",
    "적응",
    "비행기",
    "엘리베이터",
    "시스템",
    "오류",
    "알림",
    "알림음",
    "밤길",
    "밤에",
    "새벽",
    "창문",
    "발소리",
    "마감",
    "현관",
    "낯선사람",
    "일어나지도않은일",
    "아직일어나지도않은일",
    "아직일어나지않은일",
]

SITUATIONAL_ANXIETY_MARKERS = [
    "혹시",
    "떨어질까",
    "막힐까",
    "안됐을까",
    "제대로안됐",
    "깨질까",
    "갇힐까",
    "못따라갈까",
    "잘못",
    "생긴건아닌지",
    "생길까",
    "걱정",
    "불안",
    "무서",
    "겁나",
    "겁났",
    "조마조마",
    "몸이굳",
    "굳었",
    "손잡이를꽉",
    "손끝이차",
    "손끝이차가",
    "차가워",
    "심장이빨",
    "심장이내려앉",
    "내려앉",
    "숨을죽",
    "입이바짝",
    "등골이서늘",
    "얼어붙",
    "다리가떨",
    "다시확인",
    "계속떠올라",
    "긴장이풀리지",
    "떨릴까",
    "손이차가",
    "손에땀",
    "땀이났",
    "움츠러",
    "귀를기울",
    "계속확인",
    "확인했",
    "확인을반복",
    "최악의상황",
    "상상하게돼",
]

SITUATIONAL_SURPRISE_MARKERS = [
    "갑자기",
    "갑작스러운",
    "깜짝",
    "놀랐",
    "당황",
    "순간",
    "멍해",
    "얼어붙",
    "어리둥절",
    "말이안나왔",
    "말이안나와",
    "말문이막",
    "철렁",
    "움찔",
    "하얘졌",
    "내려앉",
    "예상못",
    "예상보다",
    "생각지도못",
    "앞당겨",
    "큰소리",
    "세게닫",
    "꺼져",
    "사라진줄",
    "날아간줄",
]

HIGH_INTENSITY_FEAR_MARKERS = [
    "공황",
    "패닉",
    "숨이안쉬",
    "숨을못",
    "죽을것같",
    "죽을거같",
    "미칠것같",
    "미칠거같",
    "통제가안",
    "조절이안",
    "쓰러질것",
    "쓰러질거",
    "위협",
    "쫓아오",
    "따라오",
    "해칠것",
    "해칠거",
    "공격",
    "사고날",
    "다칠것",
    "다칠거",
]

ANGER_MARKERS = [
    "화났",
    "화가났",
    "화가나",
    "화가쉽게",
    "짜증",
    "분이안풀",
    "억울",
    "무례",
    "무시",
    "기분이나빴",
    "책임을떠넘",
    "사과도안",
    "신경안써",
    "약속을또안지",
    "기분이상했",
    "속이확끓",
    "듣고분했",
    "불공평",
    "납득이안",
    "몰아가",
    "선을넘",
    "계획이다꼬",
    "다꼬였",
    "답변을계속미뤄",
    "약속시간",
]

ANGER_HIGH_INTENSITY_MARKERS = [
    "때리고싶",
    "쳐버리고싶",
    "부숴버리고싶",
    "죽이고싶",
    "해치고싶",
    "가만안둘",
    "복수하고싶",
    "참을수없",
    "도저히참을수없",
    "통제가안",
    "조절이안",
    "폭발할것",
    "폭발할거",
    "미쳐버리",
]

ACADEMIC_ANXIETY_CONTEXTS = [
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

ACADEMIC_ANXIETY_MARKERS = [
    "긴장",
    "불안",
    "걱정",
    "초조",
    "떨려",
    "부담",
    "무섭",
]

DISTRESS_MARKERS = [
    "우울",
    "슬퍼",
    "외로",
    "공허",
    "무기력",
    "재미가없",
    "재미도없",
    "아무재미",
    "흥미가없",
    "마음이무거",
    "마음이무겁",
    "마음무거",
    "마음무겁",
    "몸이무겁",
    "의욕이없",
    "의욕없",
    "손이안가",
    "손이가지않",
    "손도안가",
    "좋아하던일도손이안",
    "좋아하던것도손이안",
    "마음이움직이지않",
    "마음이잘움직이지않",
    "버거",
    "지쳤",
    "지쳐",
    "잠이안",
    "불면",
    "밥맛",
    "입맛",
    "뒤척",
    "눈물",
    "서운",
    "쓸쓸",
    "허전",
    "아쉬",
    "상처",
    "마음상",
    "상했",
    "먹먹",
    "괴로",
    "한심",
    "자책",
    "내탓",
    "살기힘",
]

DISTRESS_INTENSIFIERS = [
    "요즘",
    "계속",
    "매일",
    "맨날",
    "하루종일",
    "아무것도",
    "도저히",
    "너무지쳐",
]

SITUATIONAL_SADNESS_MARKERS = [
    "쓸쓸",
    "허전",
    "외로",
    "아쉬",
    "서운",
    "눈물",
    "마음상",
    "상했",
    "상처받",
    "상처를받",
    "혼자있는시간",
    "유난히길게",
    "길게느껴",
    "즐겁지않",
    # 일상 전환·하루 단위 피로 표현("집에 오면 힘들어"류). situational context
    # (집에오/오늘은/하루가끝/사람들사이 등)와 함께일 때만 단일 사건성 슬픔으로 보고,
    # sustained(요즘/계속/매일·무기력·아무것도)나 crisis 단서가 있으면
    # is_situational_sadness_text 안전장치에서 제외되어 지속 우울·위기는 그대로 보존한다.
    "힘들",
]

SITUATIONAL_SADNESS_CONTEXTS = [
    "기대했던연락",
    "연락이안",
    "예전생각",
    "좋아하던일",
    "예전만큼",
    "집에오",
    "하루가끝",
    "약속이취소",
    "약속취소",
    "친구랑약속",
    "작은말",
    "말한마디",
    "괜히",
    "오늘은",
    "사람들사이",
]

SUSTAINED_DISTRESS_MARKERS = [
    "우울",
    "공허",
    "무기력",
    "마음이무거",
    "마음이무겁",
    "마음무거",
    "마음무겁",
    "몸이무겁",
    "의욕이없",
    "의욕없",
    "버거",
    "지쳤",
    "지쳐",
    "잠이안",
    "불면",
    "괴로",
    "한심",
    "자책",
    "내탓",
    "살기힘",
] + DISTRESS_INTENSIFIERS

CRISIS_MARKERS = [
    "자해",
    "죽고싶",
    "죽고싶다",
    "죽어버",
    "죽을래",
    "살고싶지않",
    "사라지고싶",
    "없어지고싶",
    "뛰어내리",
    "목매",
    "스스로목숨",
    "목숨을끊",
    "손목을그",
    "칼로그",
    "약을잔뜩",
]

ASSISTANT_QUOTE_META_MARKERS = [
    "네가",
    "방금",
    "아까",
    "말한",
    "말해",
    "해줘",
    "받아줘",
    "라는말",
    "라고",
]


def compact_text(text: str) -> str:
    """
    역할: 규칙 판정을 위해 공백과 구두점을 제거하고 소문자화한다.
    입력: 원문 발화 텍스트
    출력: 정규화된 텍스트
    """
    return re.sub(r"[\s\.,!?~·…'\"“”‘’()\[\]{}]+", "", str(text)).lower()


def is_laughter_only_text(text: str) -> bool:
    """
    역할: `ㅋㅋ`, `ㅎㅎ`처럼 웃음 토큰만 있는 짧은 저신호 발화인지 판별한다.
    입력: 사용자 발화 텍스트
    출력: 웃음 단독 발화 여부
    """
    compact = compact_text(text)
    return bool(compact) and bool(re.fullmatch(r"[ㅋㅎ]+", compact))


def normalize_emotion_analysis_text(text: str) -> str:
    """
    역할: 감정 판단 전에 기록형 wrapper, assistant 응답 인용구, 메타 발화를 약화한다.
    입력: 원문 사용자 발화 텍스트
    출력: RoBERTa/CBT 분석에 사용할 정리된 텍스트
    """
    original = str(text or "").strip()
    if not original:
        return original

    structured_core = extract_structured_emotion_core(original)
    if structured_core:
        original = structured_core

    # Qwen 답변을 그대로 인용한 구간은 사용자 감정 단서를 희석하므로 제거한다.
    quote_removed = re.sub(r"[\"'“”‘’][^\"'“”‘’]{4,160}[\"'“”‘’]", " ", original)
    compact_original = compact_text(original)
    has_meta_quote = any(marker in compact_original for marker in ASSISTANT_QUOTE_META_MARKERS)
    if quote_removed == original and not has_meta_quote:
        return original

    cleaned = quote_removed
    # 문장 첫머리의 "네가 ...라고 해줘서" 류 메타 절을 덜어내고 실제 감정 절을 남긴다.
    cleaned = re.sub(
        r"^\s*(?:응,?\s*)?(?:네가|방금|아까|방금\s*네가)?"
        r"[^,.!?。]{0,70}(?:라고|라는\s*말처럼|말한|말해주|받아줘|해줘서)"
        r"[^,.!?。]*(?:[,\.!?。]\s*)?",
        " ",
        cleaned,
    )
    cleaned = re.sub(r"^\s*응,?\s*(?:라는\s*말처럼|말처럼)\s*", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or original


def extract_structured_emotion_core(text: str) -> str | None:
    """
    역할: long_context/memo_log처럼 원문 앞뒤에 붙은 기록 프레임에서 실제 감정 절만 추출한다.
    입력: 원문 사용자 발화 텍스트
    출력: 감정 핵심 절 또는 추출할 수 없으면 None
    """
    original = str(text or "").strip()
    if not original:
        return None

    def _clean_core(core: str) -> str | None:
        """
        역할: 추출된 감정 핵심 절의 공백과 마침표만 가볍게 정리한다.
        입력: 후보 핵심 절
        출력: 사용할 수 있는 핵심 절 또는 None
        """
        cleaned = re.sub(r"\s+", " ", str(core or "")).strip(" .!?。")
        return cleaned if len(cleaned) >= 4 else None

    calendar_quote_match = re.search(
        r"(?:캘린더|일정)[^\"'“”‘’]{0,30}(?:여백|옆)[^\"'“”‘’]{0,30}"
        r"[\"'“”‘’](?P<core>[^\"'“”‘’]{4,180})[\"'“”‘’]",
        original,
        flags=re.DOTALL,
    )
    if calendar_quote_match:
        # 캘린더 메모형의 따옴표는 assistant 인용이 아니라 사용자의 핵심 감정절이다.
        core = _clean_core(calendar_quote_match.group("core"))
        if core:
            return core

    weather_context_match = re.search(
        r"^\s*창밖을\s*보(?:다가|니)?\s*갑자기\s*선명해졌는데\s*,\s*(?P<core>.+?)\s*$",
        original,
        flags=re.DOTALL,
    )
    if weather_context_match:
        # 창밖 wrapper의 "갑자기"가 모든 감정을 놀람으로 끌고 가지 않게 핵심 절만 남긴다.
        core = _clean_core(weather_context_match.group("core"))
        if core:
            return core

    memo_match = re.search(
        r"(?:^|\])\s*(?:[^;\n]{0,40};\s*)?내용\s*=\s*(?P<core>[^;\n]+)",
        original,
    )
    if memo_match:
        core = _clean_core(memo_match.group("core"))
        if core:
            return core

    long_context_match = re.search(
        r"그\s*흐름\s*속에서\s*(?P<core>.*?)(?:[\.!?。]\s*)?그래서\s*오늘\s*감정을\s*기록",
        original,
        flags=re.DOTALL,
    )
    if long_context_match:
        core = _clean_core(long_context_match.group("core"))
        if core:
            return core

    late_night_match = re.search(
        r"^\s*밤에\s*다시\s*(?:떠올려보니|생각해보니|정리해보니)\s*(?P<core>.+?)\s*$",
        original,
        flags=re.DOTALL,
    )
    if late_night_match:
        # 회상 wrapper의 "밤에 다시"가 중립 루틴을 슬픔/공포로 끌고 가지 않게 핵심 절만 남긴다.
        core = _clean_core(late_night_match.group("core"))
        if core:
            return core

    style_patterns = [
        r"^\s*오늘\s*상태를\s*말씀드리면\s*,?\s*(?P<core>.+?)\s*$",
        r"^(?P<core>.+?)[\.!?。]\s*그냥\s*짧게\s*남겨두면\s*이래",
        r"^(?P<core>.+?)[\.!?。]\s*이런\s*반응이\s*자연스러운\s*건지",
        r"결국\s*(?P<core>.+?)\s*[\.!?。]?\s*$",
        r"이어서\s*(?P<core>.+?)\s*[\.!?。]?\s*$",
        r"^(?P<core>.+?)[\.!?。]\s*마음속\s*작은\s*불빛",
        r"안쪽에서는\s*(?P<core>.+?)\s*[\.!?。]?\s*$",
        r"시간이\s*지나고\s*보니\s*(?P<core>.+?)\s*[\.!?。]?\s*$",
        r"^(?P<core>.+?)(?:\.\.\.|…)\s*(?:ㅎㅎ+|ㅠ+|메모)\s*$",
    ]
    for pattern in style_patterns:
        match = re.search(pattern, original, flags=re.DOTALL)
        if match:
            core = _clean_core(match.group("core"))
            if core:
                return core

    return None


def _has_any(compact: str, markers: list[str]) -> bool:
    """
    역할: 정규화 텍스트에 후보 키워드가 하나라도 있는지 확인한다.
    입력: 정규화 텍스트, 키워드 목록
    출력: 포함 여부
    """
    return any(marker in compact for marker in markers)


def has_crisis_marker(text: str) -> bool:
    """
    역할: 직접 위기 후보 표현이 있는지 확인한다.
    입력: 사용자 발화 텍스트
    출력: 위기 후보 표현 포함 여부
    """
    compact = compact_text(text)
    return _has_any(compact, CRISIS_MARKERS)


def has_fear_marker(text: str) -> bool:
    """
    역할: 실제 공포·불안 단서가 있는지 확인한다.
    입력: 사용자 발화 텍스트
    출력: 공포 단서 포함 여부
    """
    compact = compact_text(text)
    return _has_any(compact, FEAR_MARKERS)


def has_anger_marker(text: str) -> bool:
    """
    역할: 분노·억울함·짜증처럼 anger 라우팅이 필요한 단서가 있는지 확인한다.
    입력: 사용자 발화 텍스트
    출력: 분노 단서 포함 여부
    """
    compact = compact_text(text)
    return _has_any(compact, ANGER_MARKERS)


def has_high_intensity_anger_marker(text: str) -> bool:
    """
    역할: 폭력 충동·통제 어려움처럼 강한 분노 위험 단서가 있는지 확인한다.
    입력: 사용자 발화 텍스트
    출력: 강한 분노 위험 단서 포함 여부
    """
    compact = compact_text(text)
    return _has_any(compact, ANGER_HIGH_INTENSITY_MARKERS)


def is_situational_anger_text(text: str) -> bool:
    """
    역할: 일회성 무례·억울함·짜증처럼 분노는 맞지만 즉시 위험권으로 올리기 애매한 발화인지 판별한다.
    입력: 사용자 발화 텍스트
    출력: 상황성 분노 발화 여부
    """
    compact = compact_text(text)
    has_anger = _has_any(compact, ANGER_MARKERS)
    has_crisis = _has_any(compact, CRISIS_MARKERS)
    has_high_intensity = _has_any(compact, ANGER_HIGH_INTENSITY_MARKERS)
    return has_anger and not has_crisis and not has_high_intensity


def has_distress_marker(text: str) -> bool:
    """
    역할: 정서적 고통 단서가 있는지 확인한다.
    입력: 사용자 발화 텍스트
    출력: 정서 고통 단서 포함 여부
    """
    compact = compact_text(text)
    return _has_any(compact, DISTRESS_MARKERS)


def has_negative_safety_signal(text: str) -> bool:
    """
    역할: NLI 후보를 유지할 만한 부정 정서·불안·위기 신호가 있는지 판별한다.
    입력: 사용자 발화 텍스트
    출력: 안전 검토가 필요한 부정 신호 포함 여부
    """
    compact = compact_text(text)
    return _has_any(
        compact,
        CRISIS_MARKERS
        + FEAR_MARKERS
        + ANGER_MARKERS
        + DISTRESS_MARKERS
        + SITUATIONAL_ANXIETY_MARKERS
        + ROUTINE_DISCOMFORT_MARKERS
        + POSITIVE_BLOCKING_MARKERS,
    )


def is_positive_affect_text(text: str) -> bool:
    """
    역할: 기쁨·감동·안도처럼 정서 모니터링 점수를 낮춰야 하는 긍정/회복 발화인지 판별한다.
    입력: 사용자 발화 텍스트
    출력: 긍정/회복 발화 여부
    """
    compact = compact_text(text)
    has_positive = _has_any(compact, POSITIVE_MARKERS)
    return has_positive and not has_negative_safety_signal(text)


def is_sensory_disgust_text(text: str) -> bool:
    """
    역할: 냄새·식감·오염·끈적임 및 명시적 사회·도덕 혐오 표현인지 판별한다.
    입력: 사용자 발화 텍스트
    출력: 저위험 혐오 발화 여부
    """
    compact = compact_text(text)
    has_sensory_direct = _has_any(compact, SENSORY_DISGUST_DIRECT_MARKERS)
    has_sensory_contextual = _has_any(compact, SENSORY_DISGUST_CONTEXT_MARKERS) and _has_any(
        compact,
        SENSORY_DISGUST_AFFECT_MARKERS,
    )
    has_social_direct = _has_any(compact, SOCIAL_MORAL_DISGUST_DIRECT_MARKERS)
    has_social_contextual = _has_any(compact, SOCIAL_MORAL_DISGUST_CONTEXT_MARKERS) and _has_any(
        compact,
        SOCIAL_MORAL_DISGUST_AFFECT_MARKERS,
    )
    has_crisis = _has_any(compact, CRISIS_MARKERS)
    has_situational_anger = _has_any(compact, ANGER_MARKERS)
    has_sensory_disgust = has_sensory_direct or has_sensory_contextual
    # 사회·도덕 혐오는 분노 단서가 직접 있으면 분노 라우팅을 우선하고,
    # 명시 혐오 단서만 있을 때 혐오 top label을 보존한다.
    has_social_disgust = (has_social_direct or has_social_contextual) and not has_situational_anger
    return (has_sensory_disgust or has_social_disgust) and not has_crisis


def is_low_risk_sensory_disgust_text(text: str) -> bool:
    """
    역할: 사회·도덕 분노가 아닌 냄새·식감·오염 계열 저위험 감각 혐오만 판별한다.
    입력: 사용자 발화 텍스트
    출력: 저위험 감각 혐오 발화 여부
    """
    compact = compact_text(text)
    has_sensory_direct = _has_any(compact, SENSORY_DISGUST_DIRECT_MARKERS)
    has_sensory_contextual = _has_any(compact, SENSORY_DISGUST_CONTEXT_MARKERS) and _has_any(
        compact,
        SENSORY_DISGUST_AFFECT_MARKERS,
    )
    has_crisis = _has_any(compact, CRISIS_MARKERS)
    has_social_context = _has_any(compact, SOCIAL_MORAL_DISGUST_CONTEXT_MARKERS)
    has_anger = _has_any(compact, ANGER_MARKERS)
    return (has_sensory_direct or has_sensory_contextual) and not has_crisis and not has_social_context and not has_anger


def is_routine_discomfort_text(text: str) -> bool:
    """
    역할: 일상 활동에 대한 가벼운 싫음·귀찮음·피로 표현인지 판별한다.
    입력: 사용자 발화 텍스트
    출력: 일상 불편 발화 여부
    """
    compact = compact_text(text)
    has_context = _has_any(compact, ROUTINE_CONTEXTS)
    has_discomfort = _has_any(compact, ROUTINE_DISCOMFORT_MARKERS)
    has_fear = _has_any(compact, FEAR_MARKERS)
    has_strong_distress = _has_any(compact, DISTRESS_MARKERS + DISTRESS_INTENSIFIERS)
    return has_context and has_discomfort and not has_fear and not has_strong_distress


def is_physical_exertion_text(text: str) -> bool:
    """
    역할: 운동·근무·집안일·이동처럼 몸을 쓴 뒤 나온 통증/피로 표현인지 판별한다.
    입력: 사용자 발화 텍스트
    출력: 신체 활동 맥락의 저신호 발화 여부
    """
    compact = compact_text(text)
    has_context = _has_any(compact, PHYSICAL_EXERTION_CONTEXTS)
    has_activity_context = _has_any(compact, PHYSICAL_EXERTION_ACTIVITY_CONTEXTS)
    has_physical_pain = _has_any(compact, PHYSICAL_EXERTION_PAIN_MARKERS)
    has_fatigue = _has_any(compact, PHYSICAL_EXERTION_FATIGUE_MARKERS)
    has_effort_context = _has_any(compact, PHYSICAL_EXERTION_EFFORT_MARKERS)
    has_crisis = _has_any(compact, CRISIS_MARKERS)
    has_emotional_distress = _has_any(compact, PHYSICAL_EXERTION_EMOTIONAL_BLOCKERS)
    has_safety_blocker = _has_any(compact, PHYSICAL_EXERTION_SAFETY_BLOCKERS)
    # "일하기 힘들어"처럼 의미가 넓은 피로 표현은 실제로 몸을 쓴 단서가 있을 때만 cap한다.
    has_exertion = has_physical_pain or (
        has_fatigue and (has_effort_context or has_activity_context)
    )
    return (
        has_context
        and has_exertion
        and not has_crisis
        and not has_emotional_distress
        and not has_safety_blocker
    )


def is_daily_routine_neutral_text(text: str) -> bool:
    """
    역할: 음식·휴식·집안일처럼 정서 신호가 거의 없는 일상 루틴 발화인지 판별한다.
    입력: 사용자 발화 텍스트
    출력: 저신호 일상 루틴 발화 여부
    """
    compact = compact_text(text)
    has_context = _has_any(compact, DAILY_ROUTINE_CONTEXTS)
    has_crisis = _has_any(compact, CRISIS_MARKERS)
    has_negative = has_negative_safety_signal(text)
    has_distress_intensity = _has_any(compact, DISTRESS_INTENSIFIERS)
    has_positive = is_positive_affect_text(text)
    return (
        has_context
        and not has_crisis
        and not has_negative
        and not has_distress_intensity
        and not has_positive
    )


def is_administrative_technical_neutral_text(text: str) -> bool:
    """
    역할: 번호·버전·문서·예약처럼 정서 신호가 없는 행정/기술 처리 발화인지 판별한다.
    입력: 사용자 발화 텍스트
    출력: 행정/기술 저신호 발화 여부
    """
    compact = compact_text(text)
    has_context = _has_any(compact, ADMIN_TECH_NEUTRAL_CONTEXTS)
    has_crisis = _has_any(compact, CRISIS_MARKERS)
    has_emotional_marker = _has_any(
        compact,
        FEAR_MARKERS
        + ANGER_MARKERS
        + DISTRESS_MARKERS
        + SITUATIONAL_ANXIETY_MARKERS
        + SITUATIONAL_SURPRISE_MARKERS,
    )
    return has_context and not has_crisis and not has_emotional_marker


def has_high_intensity_fear_marker(text: str) -> bool:
    """
    역할: 경도 상황 불안 cap에서 제외할 강한 공포·안전 신호가 있는지 확인한다.
    입력: 사용자 발화 텍스트
    출력: 고강도 공포/안전 단서 포함 여부
    """
    compact = compact_text(text)
    return _has_any(compact, HIGH_INTENSITY_FEAR_MARKERS)


def has_situational_anxiety_marker(text: str) -> bool:
    """
    역할: 공포 단어가 직접 없어도 평가·확인 맥락의 예기불안 구조가 있는지 확인한다.
    입력: 사용자 발화 텍스트
    출력: 상황성 불안 단서 포함 여부
    """
    compact = compact_text(text)
    return _has_any(compact, SITUATIONAL_ANXIETY_CONTEXTS) and _has_any(
        compact,
        SITUATIONAL_ANXIETY_MARKERS + FEAR_MARKERS,
    )


def is_situational_anxiety_surprise_text(text: str) -> bool:
    """
    역할: 평가 대기·일정 변경·소리 놀람처럼 상황성 불안/놀람이지만 위기 신호는 아닌지 판별한다.
    입력: 사용자 발화 텍스트
    출력: 경도 상황 불안/놀람 발화 여부
    """
    compact = compact_text(text)
    has_anxiety = has_situational_anxiety_marker(text)
    has_surprise = _has_any(compact, SITUATIONAL_SURPRISE_MARKERS)
    has_crisis = _has_any(compact, CRISIS_MARKERS)
    has_high_intensity = _has_any(compact, HIGH_INTENSITY_FEAR_MARKERS)
    has_positive = is_positive_affect_text(text)
    return (
        (has_anxiety or has_surprise)
        and not has_crisis
        and not has_high_intensity
        and not has_positive
    )


def is_academic_anxiety_text(text: str) -> bool:
    """
    역할: 시험·면접·발표 등 평가 상황을 앞둔 예기불안 발화인지 판별한다.
    입력: 사용자 발화 텍스트
    출력: 평가 상황 예기불안 여부
    """
    compact = compact_text(text)
    has_context = _has_any(compact, ACADEMIC_ANXIETY_CONTEXTS)
    has_anxiety = _has_any(compact, ACADEMIC_ANXIETY_MARKERS)
    return has_context and has_anxiety and not _has_any(compact, CRISIS_MARKERS)


def is_practical_anxiety_relief_text(text: str) -> bool:
    """
    역할: 불안을 호소하기보다 불안 완화 방법을 짧게 묻는 실용 질문인지 판별한다.
    입력: 사용자 발화 텍스트
    출력: 불안 완화 실용 질문 여부
    """
    compact = compact_text(text)
    has_anxiety = "불안" in compact
    has_relief_marker = _has_any(compact, ANXIETY_RELIEF_MARKERS + PRACTICAL_QUESTION_MARKERS)
    asks_question = str(text).strip().endswith(("?", "까", "까?", "요?", "나요?"))
    return has_anxiety and has_relief_marker and asks_question


def is_practical_question_text(text: str) -> bool:
    """
    역할: 메뉴·행동 선택·가벼운 방법 요청처럼 상담보다 바로 답해야 하는 실용 질문인지 판별한다.
    입력: 사용자 발화 텍스트
    출력: 일반 실용 질문 여부
    """
    compact = compact_text(text)
    asks_question = str(text).strip().endswith(("?", "까", "까?", "요?", "나요?"))
    imperative_request = _has_any(compact, ["추천해줘", "골라줘", "알려줘"])
    return (asks_question or imperative_request) and _has_any(compact, PRACTICAL_QUESTION_MARKERS)


def is_mild_unease_text(text: str) -> bool:
    """
    역할: 명시적 공포·위기라기보다 막연한 불편감이나 찜찜함을 표현한 발화인지 판별한다.
    입력: 사용자 발화 텍스트
    출력: 막연한 불편감 발화 여부
    """
    compact = compact_text(text)
    return _has_any(compact, MILD_UNEASE_MARKERS)


def is_mild_low_mood_text(text: str) -> bool:
    """
    역할: 혐오·분노라기보다 가벼운 저조감이나 컨디션 저하를 표현한 발화인지 판별한다.
    입력: 사용자 발화 텍스트
    출력: 가벼운 저조감 발화 여부
    """
    compact = compact_text(text)
    return _has_any(compact, MILD_LOW_MOOD_MARKERS)


def is_limited_situational_distress_text(text: str) -> bool:
    """
    역할: 단일 과제·발표·기대 실패처럼 상황은 힘들지만 장기 우울/위기 신호는 아닌 발화인지 판별한다.
    입력: 사용자 발화 텍스트
    출력: 제한적 상황성 속상함 여부
    """
    compact = compact_text(text)
    has_context = _has_any(compact, LIMITED_SITUATIONAL_DISTRESS_CONTEXTS)
    has_marker = _has_any(compact, LIMITED_SITUATIONAL_DISTRESS_MARKERS)
    has_crisis = _has_any(compact, CRISIS_MARKERS)
    has_broad_distress = _has_any(compact, ["요즘", "계속", "매일", "하루종일", "아무것도", "도저히"])
    return has_context and has_marker and not has_crisis and not has_broad_distress


def is_interpersonal_remorse_text(text: str) -> bool:
    """
    역할: 말실수로 상대에게 정서적 상처를 줬을까 걱정하는 관계 후회 문맥 판별
    입력: 사용자 발화 텍스트
    출력: 관계 후회 문맥 여부
    """
    compact = compact_text(text)
    if _has_any(compact, CRISIS_MARKERS):
        return False

    has_interpersonal_target = _has_any(
        compact,
        [
            "친구",
            "동료",
            "가족",
            "상대",
            "그사람",
            "걔",
            "사람한테",
            "사람에게",
        ],
    )
    has_emotional_harm = _has_any(
        compact,
        [
            "상처를줬",
            "상처줬",
            "상처를준",
            "상처준",
            "마음을아프게",
            "기분을상하게",
            "기분상하게",
            "실망시켰",
            "실망시킨",
            "말실수",
        ],
    )
    has_remorse = _has_any(
        compact,
        [
            "것같",
            "거같",
            "미안",
            "후회",
            "걱정",
            "신경쓰",
            "잘못",
        ],
    )
    has_harm_intent = _has_any(
        compact,
        [
            "상처를주고싶",
            "상처주고싶",
            "해치고싶",
            "때리고싶",
            "죽이고싶",
        ],
    )
    return (
        has_interpersonal_target
        and has_emotional_harm
        and has_remorse
        and not has_harm_intent
    )


def is_situational_sadness_text(text: str) -> bool:
    """
    역할: 쓸쓸함·허전함처럼 단일 사건성 슬픔은 맞지만 지속 우울 신호는 아닌 발화인지 판별한다.
    입력: 사용자 발화 텍스트
    출력: 단일 사건성 슬픔 발화 여부
    """
    compact = compact_text(text)
    has_marker = _has_any(compact, SITUATIONAL_SADNESS_MARKERS)
    has_context = _has_any(compact, SITUATIONAL_SADNESS_CONTEXTS)
    has_crisis = _has_any(compact, CRISIS_MARKERS)
    has_anger = _has_any(compact, ANGER_MARKERS)
    has_sustained_distress = _has_any(compact, SUSTAINED_DISTRESS_MARKERS)
    return has_marker and has_context and not has_crisis and not has_anger and not has_sustained_distress


def classify_utterance_type(text: str) -> dict:
    """
    역할: 사용자 발화를 7가지 타입 중 하나로 분류한다.
    입력: 사용자 발화 텍스트
    출력: {utterance_type, type_confidence, type_reason}
    """
    compact = compact_text(text)

    if _has_any(compact, CRISIS_MARKERS):
        return {
            "utterance_type": UTTERANCE_TYPES["CRISIS_CANDIDATE"],
            "type_confidence": 0.95,
            "type_reason": "direct_crisis_marker",
        }

    if is_laughter_only_text(text):
        return {
            "utterance_type": UTTERANCE_TYPES["POSITIVE_SHARE"],
            "type_confidence": 0.86,
            "type_reason": "laughter_only_positive_low_signal_marker",
        }

    if is_practical_anxiety_relief_text(text):
        return {
            "utterance_type": UTTERANCE_TYPES["PRACTICAL_QUESTION"],
            "type_confidence": 0.80,
            "type_reason": "practical_anxiety_relief_question",
        }

    if is_practical_question_text(text):
        return {
            "utterance_type": UTTERANCE_TYPES["PRACTICAL_QUESTION"],
            "type_confidence": 0.78,
            "type_reason": "practical_question_marker",
        }

    if is_academic_anxiety_text(text):
        return {
            "utterance_type": UTTERANCE_TYPES["EMOTIONAL_DISTRESS"],
            "type_confidence": 0.84,
            "type_reason": "academic_anxiety_marker",
        }

    if has_high_intensity_fear_marker(text):
        return {
            "utterance_type": UTTERANCE_TYPES["EMOTIONAL_DISTRESS"],
            "type_confidence": 0.88,
            "type_reason": "high_intensity_fear_marker",
        }

    if is_low_risk_sensory_disgust_text(text):
        return {
            "utterance_type": UTTERANCE_TYPES["CASUAL_NEUTRAL"],
            "type_confidence": 0.80,
            "type_reason": "sensory_disgust_low_impact_marker",
        }

    if is_sensory_disgust_text(text):
        return {
            "utterance_type": UTTERANCE_TYPES["EMOTIONAL_DISTRESS"],
            "type_confidence": 0.80,
            "type_reason": "sensory_disgust_marker",
        }

    if is_physical_exertion_text(text):
        return {
            "utterance_type": UTTERANCE_TYPES["CASUAL_NEUTRAL"],
            "type_confidence": 0.80,
            "type_reason": "physical_exertion_context",
        }

    if is_limited_situational_distress_text(text):
        return {
            "utterance_type": UTTERANCE_TYPES["EMOTIONAL_DISTRESS"],
            "type_confidence": 0.76,
            "type_reason": "limited_situational_distress_marker",
        }

    if is_routine_discomfort_text(text):
        return {
            "utterance_type": UTTERANCE_TYPES["ROUTINE_DISCOMFORT"],
            "type_confidence": 0.82,
            "type_reason": "routine_context_with_mild_discomfort",
        }

    if is_daily_routine_neutral_text(text):
        return {
            "utterance_type": UTTERANCE_TYPES["CASUAL_NEUTRAL"],
            "type_confidence": 0.78,
            "type_reason": "daily_routine_neutral_context",
        }

    if is_administrative_technical_neutral_text(text):
        return {
            "utterance_type": UTTERANCE_TYPES["CASUAL_NEUTRAL"],
            "type_confidence": 0.78,
            "type_reason": "administrative_technical_neutral_context",
        }

    if is_situational_anxiety_surprise_text(text):
        return {
            "utterance_type": UTTERANCE_TYPES["EMOTIONAL_DISTRESS"],
            "type_confidence": 0.78,
            "type_reason": "situational_anxiety_surprise_marker",
        }

    if is_mild_unease_text(text):
        return {
            "utterance_type": UTTERANCE_TYPES["EMOTIONAL_DISTRESS"],
            "type_confidence": 0.74,
            "type_reason": "mild_unease_marker",
        }

    if is_mild_low_mood_text(text):
        return {
            "utterance_type": UTTERANCE_TYPES["EMOTIONAL_DISTRESS"],
            "type_confidence": 0.76,
            "type_reason": "mild_low_mood_marker",
        }

    if is_situational_sadness_text(text):
        return {
            "utterance_type": UTTERANCE_TYPES["EMOTIONAL_DISTRESS"],
            "type_confidence": 0.78,
            "type_reason": "situational_sadness_marker",
        }

    if _has_any(compact, ANGER_MARKERS):
        return {
            "utterance_type": UTTERANCE_TYPES["EMOTIONAL_DISTRESS"],
            "type_confidence": 0.84,
            "type_reason": "anger_marker",
        }

    if _has_any(compact, FEAR_MARKERS):
        return {
            "utterance_type": UTTERANCE_TYPES["EMOTIONAL_DISTRESS"],
            "type_confidence": 0.84,
            "type_reason": "fear_or_anxiety_marker",
        }

    if _has_any(compact, DISTRESS_MARKERS + DISTRESS_INTENSIFIERS):
        return {
            "utterance_type": UTTERANCE_TYPES["EMOTIONAL_DISTRESS"],
            "type_confidence": 0.78,
            "type_reason": "distress_marker",
        }

    if _has_any(compact, CASUAL_MARKERS + POSITIVE_MARKERS):
        return {
            "utterance_type": UTTERANCE_TYPES["CASUAL_NEUTRAL"],
            "type_confidence": 0.72,
            "type_reason": "casual_or_positive_marker",
        }

    return {
        "utterance_type": UTTERANCE_TYPES["CASUAL_NEUTRAL"],
        "type_confidence": 0.55,
        "type_reason": "default_low_signal",
    }
