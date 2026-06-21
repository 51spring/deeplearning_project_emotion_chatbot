# -*- coding: utf-8 -*-
"""제출 보고서용 시스템 아키텍처 플로우차트 이미지(PNG) 생성 스크립트.

역할:
    ASCII 텍스트 다이어그램 대신, 발화 입력 → FastAPI → RoBERTa 추론 →
    안전 게이트 → Qwen 응답 생성 → 품질/안전 검사 → 점수 저장 → React UI 흐름을
    한글 라벨이 깨지지 않는 matplotlib 플로우차트로 그린다.
입력:
    --out : 저장할 PNG 경로(기본 eval/report/architecture_diagram.png).
    --font: 한글 TTF 경로(기본 Windows Malgun Gothic).
출력:
    지정 경로에 PNG 파일을 저장한다(반환값 없음).
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")  # 화면 없이 파일로만 저장
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.pyplot as plt


# 색상 팔레트(은은한 파랑 계열 + 안전 게이트 강조용 주황)
C_INPUT = "#E8EEF7"   # 입출력 단계
C_PROC = "#D6E4F0"    # 일반 처리 단계
C_MODEL = "#CDE3D2"   # 모델 추론/생성 단계
C_SAFE = "#F6DfC9"    # 안전 게이트(강조)
C_DETAIL = "#F4F6F9"  # 세부 설명 박스
EDGE = "#5A6B86"      # 박스 테두리
EDGE_SAFE = "#C8843C" # 안전 게이트 테두리
ARROW = "#3B4A63"     # 화살표


def _fp(font_path, size, bold=False):
    """주어진 TTF로 FontProperties를 만든다(한글 렌더링용).

    입력: font_path(폰트 경로), size(pt), bold(굵게 여부).
    출력: matplotlib FontProperties.
    """
    weight = "bold" if bold else "normal"
    return font_manager.FontProperties(fname=font_path, size=size, weight=weight)


def _box(ax, x, y, w, h, text, fp, fill, edge, lw=1.3):
    """둥근 모서리 박스 1개를 그리고 가운데 텍스트를 넣는다.

    입력: ax, 중심좌표(x,y), 폭/높이(w,h), text, fp(FontProperties), fill/edge 색, lw 선두께.
    출력: 없음(축에 직접 그림).
    """
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=lw, edgecolor=edge, facecolor=fill, zorder=2,
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center", fontproperties=fp,
            color="#1B2433", zorder=3, linespacing=1.35)


def _arrow(ax, x, y1, y2):
    """두 단계 사이 수직 아래 방향 화살표를 그린다.

    입력: ax, x(공통 x좌표), y1(시작 y, 위), y2(끝 y, 아래).
    출력: 없음.
    """
    ax.add_patch(FancyArrowPatch(
        (x, y1), (x, y2), arrowstyle="-|>", mutation_scale=16,
        linewidth=1.6, color=ARROW, zorder=1,
    ))


def _connector(ax, x1, x2, y):
    """본문 박스와 오른쪽 세부 박스를 잇는 점선 연결선.

    입력: ax, x1(본문 우측 x), x2(세부 박스 좌측 x), y(공통 y).
    출력: 없음.
    """
    ax.plot([x1, x2], [y, y], linestyle=(0, (2, 2)), color="#9AA7BD",
            linewidth=1.1, zorder=1)


def build(out_path, font_path):
    """아키텍처 플로우차트를 그려 PNG로 저장한다.

    입력: out_path(저장 경로), font_path(한글 TTF 경로).
    출력: 없음(파일 저장).
    """
    fp_title = _fp(font_path, 15, bold=True)
    fp_main = _fp(font_path, 11.5, bold=True)
    fp_detail = _fp(font_path, 9.2)

    fig, ax = plt.subplots(figsize=(9.2, 11.4), dpi=200)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 24)
    ax.axis("off")

    # 제목
    ax.text(6, 23.3, "시스템 아키텍처 — /chat 처리 흐름",
            ha="center", va="center", fontproperties=fp_title, color="#1B2433")

    mx = 3.5          # 본문 박스 중심 x
    mw, mh = 4.6, 1.5  # 본문 박스 폭/높이
    dx = 9.1          # 세부 박스 중심 x
    dw = 4.6          # 세부 박스 폭

    # (라벨, 색, 테두리, 세부설명 또는 None)
    steps = [
        ("사용자 발화", C_INPUT, EDGE, None),
        ("FastAPI  /chat", C_PROC, EDGE, None),
        ("RoBERTa 추론", C_MODEL, EDGE,
         "감정 7클래스 · NLI 위기 후보\nCBT anchor score · CBT class head\n발화 타입(utterance type)"),
        ("안전 게이트 (위기 감지)", C_SAFE, EDGE_SAFE,
         "직접 위기 표현 hard interrupt\nNLI hard interrupt (≥0.80)\nsoft crisis 후보 (≥0.35)"),
        ("Qwen 상담 응답 생성", C_MODEL, EDGE,
         "오늘 rolling summary\n최근 대화 context + 현재 발화\nDoT 프롬프팅"),
        ("응답 품질·안전 검사", C_PROC, EDGE,
         "[CRISIS] 태그 · self-check\nresponse anchor · 다국어 잔여 검사\nfallback 응답"),
        ("점수 저장 (SQLite)", C_PROC, EDGE,
         "top_emotion · depression_score\ndepression_tendency_score\nwellness_score · crisis flag\nmodel_audit_events"),
        ("React UI", C_INPUT, EDGE,
         "채팅 · 오늘 웰니스\n하루 마감 · 캘린더"),
    ]

    n = len(steps)
    top, bottom = 21.6, 1.4
    ys = [top - (top - bottom) * i / (n - 1) for i in range(n)]

    for i, (label, fill, edge, detail) in enumerate(steps):
        y = ys[i]
        _box(ax, mx, y, mw, mh, label, fp_main, fill, edge,
             lw=1.8 if edge == EDGE_SAFE else 1.3)
        if i < n - 1:
            _arrow(ax, mx, y - mh / 2, ys[i + 1] + mh / 2)
        if detail:
            # 세부 박스 높이는 줄 수에 맞춰 조정
            lines = detail.count("\n") + 1
            dh = 0.55 * lines + 0.5
            _box(ax, dx, y, dw, dh, detail, fp_detail, C_DETAIL, "#C2CCDC", lw=1.0)
            _connector(ax, mx + mw / 2, dx - dw / 2, y)

    # 하단 캡션
    ax.text(6, 0.45,
            "FastAPI가 API 라우트 등록 뒤 frontend/build를 같은 origin으로 정적 서빙 → URL 하나로 로그인·채팅·캘린더 확인",
            ha="center", va="center", fontproperties=_fp(font_path, 8.6),
            color="#5A6B86")

    fig.tight_layout(pad=0.4)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[저장] {out_path}")


def main():
    """CLI 진입점."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="eval/report/architecture_diagram.png")
    parser.add_argument("--font", default=r"C:\Windows\Fonts\malgun.ttf")
    args = parser.parse_args()
    if not os.path.exists(args.font):
        raise SystemExit(f"한글 폰트를 찾을 수 없음: {args.font}")
    build(args.out, args.font)


if __name__ == "__main__":
    main()
