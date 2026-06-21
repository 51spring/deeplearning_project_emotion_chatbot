/**
 * RecommendationCard.js
 * 역할: "오늘의 추천" 카드 — 행동 추천 v1
 *       - /chat 응답의 recommendations(list)를 웰니스 패널 아래에 별도 카드로 표시
 *       - 항목별 title / message / reason / priority 표시
 *       - 위기(category="safety") 추천은 safety 스타일, 그 외는 calm 스타일
 *       - 의료/진단이 아니라 "오늘 감정 기록 기반 자기관리 참고"로 표현
 */
import React from "react";

// 우선순위 표시 라벨 — 단정적이지 않게 부드럽게 표기
const PRIORITY_LABEL = {
  high: "중요",
  medium: "추천",
  low: "참고",
};

// 추천이 비어 있을 때 보여줄 기본 체크인 추천 (백엔드 checkin과 동일 취지)
const DEFAULT_RECOMMENDATION = {
  id: "checkin",
  title: "짧은 체크인 유지",
  message: "오늘 기분을 한 줄로 가볍게 기록해두는 정도면 충분해요.",
  reason: "특별한 신호 없음",
  priority: "low",
  category: "checkin",
};

/**
 * RecommendationCard — 오늘의 추천 카드
 * @param {Array<object>} recommendations - 추천 항목 리스트(없으면 기본 체크인 표시)
 */
export default function RecommendationCard({ recommendations }) {
  // 추천이 없으면 기본 "짧은 체크인 유지" 추천을 보여준다.
  const items =
    recommendations && recommendations.length > 0
      ? recommendations
      : [DEFAULT_RECOMMENDATION];

  // 위기(safety) 추천이 포함되면 카드 전체를 safety 스타일로 표시한다.
  const isSafety = items.some((r) => r.category === "safety");

  return (
    <div className={`status-card reco-card ${isSafety ? "reco-safety" : "reco-calm"}`}>
      <h3 className="status-title">오늘의 추천</h3>
      <ul className="reco-list">
        {items.map((r) => (
          <li key={r.id} className="reco-item">
            <div className="reco-item-head">
              <span className="reco-item-title">{r.title}</span>
              {r.priority && (
                <span className={`reco-priority reco-priority-${r.priority}`}>
                  {PRIORITY_LABEL[r.priority] || r.priority}
                </span>
              )}
            </div>
            <p className="reco-message">{r.message}</p>
            {r.reason && <p className="reco-reason">{r.reason}</p>}
          </li>
        ))}
      </ul>
      <p className="reco-note">
        * 오늘 감정 기록을 바탕으로 한 자기관리 참고예요. 의료 진단이 아닙니다.
      </p>
    </div>
  );
}
