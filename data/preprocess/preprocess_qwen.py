"""
preprocess_qwen.py
역할: Qwen2.5 QLoRA 파인튜닝용 대화 데이터 JSONL 생성
입력:
  - data/raw/웰니스_대화_스크립트_데이터셋.xlsx   (유효 쌍 1,034행)
  - data/raw/16.심리상담 데이터/.../*.txt        (상담사-내담자 대화)
출력: data/processed/qwen_finetune.jsonl
포맷: {"messages": [{"role":"system",...}, {"role":"user",...}, {"role":"assistant",...}]}
"""

import os
import re
import json
import glob
import math
import random
import pandas as pd

SEED = 42
random.seed(SEED)

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WELL_PATH = os.path.join(BASE_DIR, "data", "raw", "웰니스_대화_스크립트_데이터셋.xlsx")
AIHUB_DIR = os.path.join(BASE_DIR, "data", "raw", "16.심리상담 데이터", "1.데이터")
OUT_PATH  = os.path.join(BASE_DIR, "data", "processed", "qwen_finetune.jsonl")
CRISIS_JSON_PATH = os.path.join(BASE_DIR, "data", "nli", "crisis_utterances_aihub.json")
AUG_OUT_PATH = os.path.join(BASE_DIR, "data", "processed", "qwen_finetune_crisis_augmented.jsonl")
WEIGHTED_OUT_PATH = os.path.join(BASE_DIR, "data", "processed", "qwen_finetune_crisis_weighted.jsonl")
CRISIS_REPEAT_PER_UTT = int(os.getenv("QWEN_CRISIS_REPEAT_PER_UTT", "6"))
TARGET_CRISIS_RATIO = float(os.getenv("QWEN_TARGET_CRISIS_RATIO", "0.20"))

# ── DoT 시스템 프롬프트 ────────────────────────────────────────────────────────
SYSTEM_PROMPT = """당신은 공감 능력이 뛰어난 심리상담 보조 AI입니다.
사용자의 감정을 먼저 충분히 공감하고, 다음 단계를 따라 응답하세요.

[응답 지침]
1. 감정 인식: 사용자가 표현한 감정을 구체적으로 반영하세요.
2. 공감 표현: 판단 없이 따뜻하게 수용하세요.
3. 탐색 질문: 상황을 더 이해하기 위한 개방형 질문을 하나만 하세요.
4. 위기 감지: 자해·자살 의도가 감지되면 응답 마지막에 반드시 [CRISIS] 태그를 추가하세요.

[금지 사항]
- 의학적 진단이나 처방 금지
- 즉각적인 해결책 강요 금지
- 감정을 축소하거나 무시하는 표현 금지

응답은 2~4문장으로 간결하게 작성하세요."""

CRISIS_RESPONSE_TEMPLATES = [
    "지금 말씀에는 스스로를 해치고 싶을 만큼 벼랑 끝에 몰린 마음이 강하게 느껴져요. 혼자 버티지 않도록 곁의 사람이나 지역 정신건강복지센터, 109 같은 즉시 도움 자원에 바로 연결해볼 수 있을까요? 지금 가장 가까이에 있는 안전한 어른이나 보호자에게도 바로 알려주세요. [CRISIS]",
    "지금은 위기 신호가 분명해서, 감정보다 안전을 먼저 챙기는 것이 필요해 보여요. 혼자 있지 말고 가족이나 친구, 보호자에게 바로 연락하고 응급실이나 119, 자살예방 상담전화 109에 즉시 도움을 요청해 주세요. [CRISIS]",
    "말씀을 들으니 지금 혼자 감당하기엔 너무 위험한 상태처럼 느껴져요. 지금 이 대화를 잠시 멈추더라도 괜찮으니 주변 사람에게 바로 도움을 요청하고, 가까운 응급실이나 119, 109에 즉시 연결해 주세요. [CRISIS]",
]


def normalize_crisis_tag(bot_text: str) -> str:
    """
    역할: 응답 텍스트의 기존 [CRISIS] 태그만 보존하고 중복 태그는 정리
    입력: 챗봇 응답 텍스트
    출력: 정규화된 챗봇 응답 텍스트
    """
    assistant_text = bot_text.strip()
    has_existing_tag = "[CRISIS]" in assistant_text
    assistant_text = assistant_text.replace("[CRISIS]", "").strip()

    if has_existing_tag:
        assistant_text = f"{assistant_text} [CRISIS]".strip()

    return assistant_text


def make_message(user_text: str, bot_text: str) -> dict:
    """
    역할: (user, bot) 텍스트 쌍을 messages 포맷으로 변환
    입력: user 발화, bot 응답
    출력: {"messages": [...]} dict
    """
    # 검증되지 않은 키워드 휴리스틱으로 태그를 추가하지 않고,
    # 원본 데이터에 이미 존재하는 [CRISIS] 태그만 보존함
    assistant_text = normalize_crisis_tag(bot_text)

    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": user_text.strip()},
            {"role": "assistant", "content": assistant_text},
        ]
    }


def dedup_records(records: list) -> list:
    """
    역할: 완전히 동일한 (user, assistant) 대화쌍 중복 제거
    입력: messages 포맷 dict 리스트
    출력: 중복 제거된 dict 리스트
    """
    deduped_records = []
    seen_pairs = set()

    for record in records:
        user_text = record["messages"][1]["content"].strip()
        assistant_text = record["messages"][2]["content"].strip()
        pair_key = (user_text, assistant_text)

        if pair_key in seen_pairs:
            continue

        seen_pairs.add(pair_key)
        deduped_records.append(record)

    print(f"[중복 제거] {len(records)} → {len(deduped_records)}개")
    return deduped_records


def load_crisis_augmented_records(crisis_json_path: str, repeat_per_utt: int = 3) -> list:
    """
    역할: 위기 발화 목록을 상담 안전 응답 포맷의 학습 샘플로 확장한다.
    입력: 위기 발화 JSON 경로, 발화당 응답 반복 수
    출력: messages 포맷 dict 리스트
    """
    if not os.path.exists(crisis_json_path):
        print(f"[위기 보강] 파일 없음: {crisis_json_path}")
        return []

    with open(crisis_json_path, encoding="utf-8") as f:
        crisis_items = json.load(f)

    records = []
    for idx, item in enumerate(crisis_items):
        user_text = item if isinstance(item, str) else item.get("text", "")
        user_text = _clean_text(str(user_text))
        if len(user_text) < 5:
            continue

        # 같은 발화가 단조롭게 반복되지 않도록 템플릿을 순환 적용한다.
        for repeat_idx in range(repeat_per_utt):
            template_idx = (idx + repeat_idx) % len(CRISIS_RESPONSE_TEMPLATES)
            records.append(make_message(user_text, CRISIS_RESPONSE_TEMPLATES[template_idx]))

    print(f"[위기 보강] {len(records)}개 샘플 생성 (원본 발화 {len(crisis_items)}개)")
    return records


def build_weighted_crisis_records(base_records: list, crisis_records: list, target_ratio: float) -> list:
    """
    역할: 목표 위기 비율이 되도록 위기 샘플을 중복 허용 방식으로 추가한다.
    입력: 기본 레코드 리스트, 고유 위기 레코드 리스트, 목표 위기 비율
    출력: 위기 가중치가 반영된 레코드 리스트
    """
    if not crisis_records:
        print("[위기 가중치] 위기 샘플이 없어 기본 데이터셋을 그대로 사용")
        return list(base_records)

    if target_ratio <= 0.0 or target_ratio >= 1.0:
        raise ValueError("TARGET_CRISIS_RATIO는 0과 1 사이여야 합니다.")

    base_count = len(base_records)
    required_crisis_count = math.ceil((target_ratio * base_count) / (1.0 - target_ratio))
    weighted_records = list(base_records)

    # 실제 학습에서는 중복 샘플 자체가 위기 가중치 역할을 하므로 dedup하지 않는다.
    for idx in range(required_crisis_count):
        weighted_records.append(crisis_records[idx % len(crisis_records)])

    print(
        f"[위기 가중치] 목표 비율 {target_ratio:.2f} 기준 "
        f"위기 샘플 {required_crisis_count}개 추가"
    )
    return weighted_records


def write_jsonl(records: list, output_path: str) -> None:
    """
    역할: messages 포맷 레코드 리스트를 JSONL 파일로 저장한다.
    입력: 레코드 리스트, 저장 경로
    출력: 없음
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def print_record_stats(records: list, label: str) -> None:
    """
    역할: 전체 샘플 수와 [CRISIS] 태그 분포를 출력한다.
    입력: 레코드 리스트, 출력 라벨명
    출력: 없음
    """
    crisis_count = sum(
        1 for record in records
        if "[CRISIS]" in record["messages"][-1]["content"]
    )
    total_count = len(records)
    ratio = (crisis_count / total_count * 100.0) if total_count else 0.0

    print(f"\n[{label}] {total_count}개 샘플")
    print(f"  - [CRISIS] 태그 포함: {crisis_count}개 ({ratio:.1f}%)")
    print(f"  - 일반 응답: {total_count - crisis_count}개")


# ────────────────────────────────────────────────────────────────────────────────
# 소스 1: 웰니스01 xlsx
# ────────────────────────────────────────────────────────────────────────────────
def load_wellness(path: str) -> list:
    """
    역할: 웰니스 대화 스크립트 xlsx에서 (유저, 챗봇) 쌍 추출
    입력: xlsx 경로
    출력: messages 포맷 dict 리스트
    """
    df = pd.read_excel(path, skiprows=1, header=None, names=["category", "user", "bot"])
    # 챗봇 응답 있는 행만 유효
    df = df.dropna(subset=["bot"])
    df = df[df["user"].notna()]
    df = df[df["bot"].str.strip() != ""]

    records = []
    for _, row in df.iterrows():
        records.append(make_message(str(row["user"]), str(row["bot"])))

    print(f"[웰니스01] {len(records)}개 쌍 추출")
    return records


# ────────────────────────────────────────────────────────────────────────────────
# 소스 2: AI Hub 심리상담 txt (상담사 ↔ 내담자 파싱)
# ────────────────────────────────────────────────────────────────────────────────
def parse_session_txt(filepath: str) -> list:
    """
    역할: 상담 세션 txt 파일에서 (내담자→상담사) 인접 발화 쌍 추출
    입력: txt 파일 경로
    출력: (내담자발화, 상담사발화) 튜플 리스트
    """
    with open(filepath, encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    pairs = []
    prev_role, prev_text = None, None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith("상담사 :"):
            text = line[len("상담사 :"):].strip()
            # 이전이 내담자 발화였으면 쌍 완성
            if prev_role == "내담자" and prev_text and text:
                pairs.append((prev_text, text))
            prev_role, prev_text = "상담사", text

        elif line.startswith("내담자 :"):
            text = line[len("내담자 :"):].strip()
            prev_role, prev_text = "내담자", text

    return pairs


def _clean_text(text: str) -> str:
    """역할: @익명화 태그 제거 및 공백 정리"""
    text = re.sub(r"@\S+", "", text)   # @COUNSELOR, @TIME 등 익명화 태그 제거
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_aihub(aihub_dir: str, max_pairs_per_file: int = 20) -> list:
    """
    역할: AI Hub 심리상담 데이터에서 (내담자→상담사) 쌍 추출
    입력: AI Hub 루트 디렉터리, 파일당 최대 추출 쌍 수
    출력: messages 포맷 dict 리스트
    """
    txt_files = glob.glob(os.path.join(aihub_dir, "**", "*.txt"), recursive=True)
    print(f"[AI Hub] txt 파일 {len(txt_files)}개 발견")

    records = []
    skipped = 0

    for fpath in txt_files:
        pairs = parse_session_txt(fpath)
        # 파일당 max_pairs_per_file 개로 제한 (과대표 방지)
        selected = pairs[:max_pairs_per_file]

        for client_utt, counselor_utt in selected:
            client_utt   = _clean_text(client_utt)
            counselor_utt = _clean_text(counselor_utt)

            # 너무 짧은 발화 제거 (5자 미만)
            if len(client_utt) < 5 or len(counselor_utt) < 5:
                skipped += 1
                continue

            records.append(make_message(client_utt, counselor_utt))

    print(f"[AI Hub] {len(records)}개 쌍 추출 (건너뜀: {skipped})")
    return records


# ────────────────────────────────────────────────────────────────────────────────
# 메인
# ────────────────────────────────────────────────────────────────────────────────
def main():
    """
    역할: Qwen 기본 학습셋과 위기 보강 학습셋 JSONL을 생성한다.
    입력: 없음
    출력: 없음
    """
    print("=" * 60)
    print("Qwen 파인튜닝 데이터 전처리 시작")
    print("=" * 60)

    base_records = []

    # 소스 1: 웰니스01
    well_records = load_wellness(WELL_PATH)
    base_records.extend(well_records)

    # 소스 2: AI Hub
    aihub_records = load_aihub(AIHUB_DIR)
    base_records.extend(aihub_records)

    # 동일 대화쌍 중복 제거
    base_records = dedup_records(base_records)

    # 셔플
    random.shuffle(base_records)

    # 기본 데이터셋 저장
    print_record_stats(base_records, "기본 데이터셋")
    write_jsonl(base_records, OUT_PATH)
    print(f"\n[저장 완료] → {OUT_PATH}")

    # 위기 태그 보강 데이터셋 저장
    crisis_records = load_crisis_augmented_records(
        CRISIS_JSON_PATH,
        repeat_per_utt=CRISIS_REPEAT_PER_UTT,
    )
    unique_crisis_records = dedup_records(crisis_records)
    augmented_records = dedup_records(base_records + unique_crisis_records)
    random.shuffle(augmented_records)
    print_record_stats(augmented_records, "위기 보강 데이터셋")
    write_jsonl(augmented_records, AUG_OUT_PATH)
    print(f"[저장 완료] → {AUG_OUT_PATH}")

    # 위기 가중치 데이터셋 저장
    weighted_records = build_weighted_crisis_records(
        base_records=base_records,
        crisis_records=unique_crisis_records,
        target_ratio=TARGET_CRISIS_RATIO,
    )
    random.shuffle(weighted_records)
    print_record_stats(weighted_records, "위기 가중치 데이터셋")
    write_jsonl(weighted_records, WEIGHTED_OUT_PATH)
    print(f"[저장 완료] → {WEIGHTED_OUT_PATH}")

    print("=" * 60)


if __name__ == "__main__":
    main()
