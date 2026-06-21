/**
 * DailySummary.js
 * 역할: 하루 마감 결과 모달
 *       - /day/close API 호출 → 단일/누적 wellness_score, label 표시
 *       - 발화 수 / 위기 이벤트 수 요약
 *       - 닫기 또는 캘린더 이동 액션
 */
import React, { useState, useEffect } from "react";
import { closeDay, getCurrentDay } from "../api";

// ── 레이블별 색상 ─────────────────────────────────────────────────────────────
const LABEL_COLOR = {
  양호: "#43a047",
  보통: "#29b6f6",
  주의: "#fb8c00",
  위험: "#e53935",
};

const LABEL_EMOJI = {
  양호: "😊",
  보통: "😐",
  주의: "😟",
  위험: "😨",
};

// ── 우울 경향 band 분류 (depression_tendency_v15_spec과 일치) ────────────────
// 0.40 이상 high / 0.20 이상 mid / 그 외 low
const TENDENCY_BAND_COLOR = {
  high: "#8e24aa",   // 보라 — 우울 경향 강함
  mid:  "#5e35b1",   // 짙은 보라 — 일부 신호
  low:  "#90a4ae",   // 회색 — 우울 경향 미감지
};

const TENDENCY_BAND_LABEL = {
  high: "강함",
  mid:  "일부",
  low:  "미감지",
};

/**
 * labelForWellnessScore — 표시 웰니스 점수만으로 상태 레이블 계산
 * @param {number|null} score - 웰니스 점수(0~100)
 * @returns {string|null} 점수 기준 레이블
 */
function labelForWellnessScore(score) {
  if (score == null || Number.isNaN(Number(score))) return null;
  const value = Number(score);
  if (value < 40) return "위험";
  if (value < 60) return "주의";
  if (value < 80) return "보통";
  return "양호";
}

function classifyTendencyBand(score) {
  if (score == null || isNaN(score)) return "low";
  if (score >= 0.40) return "high";
  if (score >= 0.20) return "mid";
  return "low";
}

/**
 * ScoreRing — 원형 점수 게이지 (SVG)
 */
function ScoreRing({ score, label, showScore = true }) {
  const color  = LABEL_COLOR[label] || "#43a047";
  const radius = 54;
  const circ   = 2 * Math.PI * radius;
  const dash   = circ * (score / 100);

  return (
    <div className="score-ring-wrap">
      <svg width="140" height="140" viewBox="0 0 140 140">
        {/* 배경 원 */}
        <circle
          cx="70" cy="70" r={radius}
          fill="none" stroke="#e0e0e0" strokeWidth="12"
        />
        {/* 점수 호 */}
        <circle
          cx="70" cy="70" r={radius}
          fill="none"
          stroke={color}
          strokeWidth="12"
          strokeDasharray={`${dash} ${circ - dash}`}
          strokeLinecap="round"
          transform="rotate(-90 70 70)"
          style={{ transition: "stroke-dasharray 0.8s ease" }}
        />
        {/* 점수 텍스트 — 비관리자는 숫자 대신 상태 이모지(색상 게이지만) */}
        {showScore ? (
          <>
            <text x="70" y="65" textAnchor="middle" fontSize="26" fontWeight="700" fill={color}>
              {Math.round(score)}
            </text>
            <text x="70" y="85" textAnchor="middle" fontSize="13" fill="#757575">
              / 100
            </text>
          </>
        ) : (
          <text x="70" y="82" textAnchor="middle" fontSize="36">
            {LABEL_EMOJI[label] || ""}
          </text>
        )}
      </svg>
      <p className="score-ring-label" style={{ color }}>
        {showScore ? `${LABEL_EMOJI[label] || ""} ${label}` : label}
      </p>
    </div>
  );
}

/**
 * DailySummary — 하루 마감 모달 컴포넌트
 * @param {string}   username     - 현재 사용자 이름
 * @param {string}   date         - 마감 대상 날짜
 * @param {Function} onClose      - 모달 닫기 콜백
 * @param {Function} onDayClosed  - 마감 완료 후 채팅 날짜 갱신 콜백
 * @param {Function} onGoCalendar - 캘린더 탭 이동 콜백
 */
export default function DailySummary({ username, date, onClose, onDayClosed, onGoCalendar }) {
  // ── 상태 ────────────────────────────────────────────────────────────────────
  const [loading,  setLoading]  = useState(true);
  const [error,    setError]    = useState(null);
  const [summary,  setSummary]  = useState(null);
  // 관리자 여부 — 비관리자에겐 웰니스 숫자를 숨기고 색상/상태(이모지·라벨)로만 보여준다
  const [isAdmin,  setIsAdmin]  = useState(false);

  // ── 관리자 여부 조회 ─────────────────────────────────────────────────────────
  useEffect(() => {
    let cancelled = false;
    if (!username) return undefined;
    getCurrentDay(username)
      .then((data) => {
        if (!cancelled) setIsAdmin(Boolean(data.is_developer));
      })
      .catch((err) => {
        console.error("getCurrentDay 오류:", err);
      });
    return () => { cancelled = true; };
  }, [username]);

  // ── 마운트 시 /day/close 호출 ────────────────────────────────────────────────
  useEffect(() => {
    let cancelled = false;

    async function fetchSummary() {
      try {
        const data = await closeDay(username, date);
        if (!cancelled) {
          setSummary(data);
          onDayClosed?.(data);
        }
      } catch (err) {
        if (!cancelled)
          setError("하루 마감 저장에 실패했습니다: " + (err.response?.data?.detail || err.message));
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    fetchSummary();
    return () => { cancelled = true; };
  }, [date, onDayClosed, username]);

  // ── 오버레이 클릭 닫기 ───────────────────────────────────────────────────────
  const handleOverlayClick = (e) => {
    if (e.target === e.currentTarget) onClose();
  };

  const dailyWellness =
    summary?.daily_wellness_score ?? summary?.wellness_score ?? null;
  const cumulativeWellness =
    summary?.cumulative_wellness_score ?? summary?.wellness_score ?? null;
  const dailyWellnessLabel =
    summary?.daily_wellness_label || labelForWellnessScore(dailyWellness);
  const cumulativeWellnessLabel =
    summary?.cumulative_wellness_label || summary?.label || labelForWellnessScore(cumulativeWellness);

  // ── 렌더링 ───────────────────────────────────────────────────────────────────
  return (
    <div className="modal-overlay" onClick={handleOverlayClick} role="dialog" aria-modal="true">
      <div className="modal-card">
        {/* 헤더 */}
        <div className="modal-header">
          <h2 className="modal-title">📋 하루 요약</h2>
          <button className="modal-close-btn" onClick={onClose} aria-label="닫기">✕</button>
        </div>

        {/* 본문 */}
        <div className="modal-body">
          {loading && (
            <div className="modal-loading">
              <div className="spinner" />
              <p>하루 기록을 저장하는 중...</p>
            </div>
          )}

          {error && (
            <div className="modal-error">
              <p>⚠️ {error}</p>
              <button className="btn-secondary" onClick={onClose}>닫기</button>
            </div>
          )}

          {!loading && !error && summary && (
            <>
              {/* 날짜 */}
              <p className="summary-date">{summary.date}</p>

              {/* 점수 링 — 비관리자는 숫자 없이 색상 게이지+상태만 */}
              <ScoreRing
                score={cumulativeWellness}
                label={summary.label}
                showScore={isAdmin}
              />

              {/* 세부 지표 */}
              <div className="summary-stats">
                {dailyWellness != null && (
                  <div className="stat-item">
                    <span className="stat-label">단일 웰니스</span>
                    <span
                      className="stat-value"
                      style={{ color: LABEL_COLOR[dailyWellnessLabel] || undefined }}
                    >
                      {isAdmin
                        ? `${Math.round(dailyWellness)}${dailyWellnessLabel ? ` (${dailyWellnessLabel})` : ""}`
                        : (dailyWellnessLabel || "—")}
                    </span>
                  </div>
                )}
                {cumulativeWellness != null && (
                  <div className="stat-item">
                    <span className="stat-label">누적 웰니스</span>
                    <span
                      className="stat-value"
                      style={{ color: LABEL_COLOR[cumulativeWellnessLabel] || undefined }}
                    >
                      {isAdmin
                        ? `${Math.round(cumulativeWellness)}${cumulativeWellnessLabel ? ` (${cumulativeWellnessLabel})` : ""}`
                        : (cumulativeWellnessLabel || "—")}
                    </span>
                  </div>
                )}
                {/* 종합 distress(모델 내부 raw)는 관리자 응답에만 포함되어 표시된다 */}
                {summary.daily_score != null && (
                  <div className="stat-item">
                    <span className="stat-label">일별 종합 distress</span>
                    <span className="stat-value">{summary.daily_score.toFixed(3)}</span>
                  </div>
                )}
                {summary.smoothed_score != null && (
                  <div className="stat-item">
                    <span className="stat-label">평활 종합 distress</span>
                    <span className="stat-value">{summary.smoothed_score.toFixed(3)}</span>
                  </div>
                )}
                {/* 우울 경향 밴드 — 비관리자는 소수점 대신 정성 밴드(강함/일부/미감지)만 본다 */}
                {summary.depression_tendency_band &&
                  summary.depression_tendency_daily == null &&
                  summary.depression_tendency_smoothed == null && (
                    <div className="stat-item">
                      <span className="stat-label">우울 경향</span>
                      <span
                        className="stat-value"
                        style={{
                          color: TENDENCY_BAND_COLOR[summary.depression_tendency_band],
                          fontWeight: 600,
                        }}
                      >
                        {TENDENCY_BAND_LABEL[summary.depression_tendency_band]}
                      </span>
                    </div>
                  )}
                {/* 우울 경향 소수점 두 축(일별/평활) — 종합 distress와 분리해서 관리자에게만 표시 */}
                {summary.depression_tendency_daily != null && (
                  <div className="stat-item">
                    <span className="stat-label">일별 우울 경향</span>
                    <span
                      className="stat-value"
                      style={{
                        color: TENDENCY_BAND_COLOR[
                          classifyTendencyBand(summary.depression_tendency_daily)
                        ],
                      }}
                    >
                      {summary.depression_tendency_daily.toFixed(3)}
                      <span style={{ marginLeft: 6, fontSize: "12px", fontWeight: 500 }}>
                        ({TENDENCY_BAND_LABEL[classifyTendencyBand(summary.depression_tendency_daily)]})
                      </span>
                    </span>
                  </div>
                )}
                {summary.depression_tendency_smoothed != null && (
                  <div className="stat-item">
                    <span className="stat-label">평활 우울 경향</span>
                    <span
                      className="stat-value"
                      style={{
                        color: TENDENCY_BAND_COLOR[
                          classifyTendencyBand(summary.depression_tendency_smoothed)
                        ],
                      }}
                    >
                      {summary.depression_tendency_smoothed.toFixed(3)}
                      <span style={{ marginLeft: 6, fontSize: "12px", fontWeight: 500 }}>
                        ({TENDENCY_BAND_LABEL[classifyTendencyBand(summary.depression_tendency_smoothed)]})
                      </span>
                    </span>
                  </div>
                )}
                <div className="stat-item">
                  <span className="stat-label">발화 수</span>
                  <span className="stat-value">{summary.utterance_count}회</span>
                </div>
                <div className="stat-item">
                  <span className="stat-label">위기 이벤트</span>
                  <span
                    className="stat-value"
                    style={{ color: summary.crisis_count > 0 ? "#e53935" : "#43a047" }}
                  >
                    {summary.crisis_count > 0 ? `⚠️ ${summary.crisis_count}건` : "없음"}
                  </span>
                </div>
              </div>
              <p className="summary-note">
                {summary.smoothed_score != null
                  ? "* 단일 웰니스는 해당 날짜 발화만, 누적 웰니스는 이전 날짜 흐름을 함께 반영합니다. 종합 distress와 우울 경향은 별도 축입니다."
                  : "* 단일 웰니스는 해당 날짜 발화만, 누적 웰니스는 이전 날짜 흐름을 함께 반영합니다. 의료 진단이 아닙니다."}
              </p>

              {/* 위기 경고 */}
              {summary.crisis_count > 0 && (
                <div className="crisis-notice">
                  <p>
                    오늘 위기 신호가 감지되었습니다. 힘드실 때는 언제든지{" "}
                    <strong>자살예방상담전화 1393</strong>에 연락하세요.
                  </p>
                </div>
              )}

              {/* 안내 메시지 */}
              <div className="summary-message">
                {summary.label === "양호" && <p>오늘 하루도 잘 지내셨네요! 내일도 좋은 하루 되세요. 😊</p>}
                {summary.label === "보통" && <p>오늘 하루 수고 많으셨어요. 내일은 더 좋은 하루가 될 거예요. 😐</p>}
                {summary.label === "주의" && <p>오늘 힘드셨을 것 같아요. 충분한 휴식을 취하고 내일을 맞이해 보세요. 😟</p>}
                {summary.label === "위험" && <p>많이 힘드신 것 같아요. 가까운 사람과 이야기 나눠 보시는 건 어떨까요? 필요하시면 전문 도움을 받으세요. 💙</p>}
                {summary.current_date && summary.current_date !== summary.date && (
                  <p>{summary.date} 기록을 마감하고 {summary.current_date}로 전환했어요.</p>
                )}
              </div>
            </>
          )}
        </div>

        {/* 하단 버튼 */}
        {!loading && !error && summary && (
          <div className="modal-footer">
            <button className="btn-secondary" onClick={onClose}>
              {summary.current_date && summary.current_date !== summary.date
                ? `${summary.current_date} 대화 시작`
                : "계속 대화하기"}
            </button>
            <button className="btn-primary" onClick={onGoCalendar}>
              📅 캘린더 보기
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
