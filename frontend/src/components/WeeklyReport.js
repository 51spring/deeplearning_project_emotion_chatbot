/**
 * WeeklyReport.js
 * 역할: 주간 감정 리포트 화면 컴포넌트
 *       - 주 단위 이동 (지난주/다음주)
 *       - 요약 카드: 평균 웰니스(전주 대비), 기록한 날, 총 대화 수, 위기 수
 *       - 7일 단일/누적 웰니스 막대 차트 (축별 별도 그래프)
 *       - 감정 분포 도넛 차트
 *       - 주간 정리와 최근 8주 요일별 감정분포
 *       - 레이블(양호/보통/주의/위험) 분포 칩
 */
import React, { useState, useEffect, useCallback } from "react";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";
import { Bar, Doughnut } from "react-chartjs-2";
import { getCurrentDay, getWeeklyReport } from "../api";

// Chart.js 필수 컴포넌트 등록 (막대 + 도넛)
ChartJS.register(
  CategoryScale, LinearScale,
  BarElement, ArcElement,
  Title, Tooltip, Legend
);

// ── 레이블/감정 색상 설정 ────────────────────────────────────────────────────
const LABEL_COLOR = {
  양호: "#43a047",
  보통: "#29b6f6",
  주의: "#fb8c00",
  위험: "#e53935",
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

/**
 * getScoreLabel — 단일/누적 막대 축에 맞는 레이블 선택
 * @param {object} day - 주간 리포트 일별 데이터
 * @param {string} scoreKey - 점수 필드명
 * @param {number|null} score - 현재 축 점수
 * @returns {string|null} 해당 축 전용 레이블
 */
function getScoreLabel(day, scoreKey, score) {
  if (scoreKey === "daily_wellness_score") {
    return day.daily_wellness_label || labelForWellnessScore(score);
  }
  return day.cumulative_wellness_label || day.label || labelForWellnessScore(score);
}

// 7감정 도넛 차트 색상 — 감정 간 시각 구분용 고정 팔레트
const EMOTION_COLOR = {
  중립: "#90a4ae",
  행복: "#00c853",
  슬픔: "#1e88e5",
  분노: "#d81b60",
  공포: "#8e24aa",
  혐오: "#6d4c41",
  놀람: "#ffb300",
};

const WEEKDAYS_KO = ["일", "월", "화", "수", "목", "금", "토"];

/**
 * formatDateLabel — "YYYY-MM-DD" → "M/D(요일)" 표시 문자열
 * @param {string} dateStr - 날짜 문자열
 * @returns {string} 차트/리스트용 짧은 날짜 라벨
 */
function formatDateLabel(dateStr) {
  const parsed = new Date(`${dateStr}T00:00:00`);
  if (isNaN(parsed.getTime())) return dateStr;
  return `${parsed.getMonth() + 1}/${parsed.getDate()}(${WEEKDAYS_KO[parsed.getDay()]})`;
}

/**
 * shiftDate — "YYYY-MM-DD" 날짜를 days만큼 이동
 * @param {string} dateStr - 기준 날짜
 * @param {number} days - 이동 일수 (음수 가능)
 * @returns {string} 이동 후 "YYYY-MM-DD"
 */
function shiftDate(dateStr, days) {
  const parsed = new Date(`${dateStr}T00:00:00`);
  parsed.setDate(parsed.getDate() + days);
  const y = parsed.getFullYear();
  const m = String(parsed.getMonth() + 1).padStart(2, "0");
  const d = String(parsed.getDate()).padStart(2, "0");
  return `${y}-${m}-${d}`;
}

/**
 * todayKey — 오늘 날짜 "YYYY-MM-DD"
 * @returns {string} 서버가 active_date를 주지 않는 오래된 응답을 위한 대체 기준일
 */
function todayKey() {
  const now = new Date();
  const y = now.getFullYear();
  const m = String(now.getMonth() + 1).padStart(2, "0");
  const d = String(now.getDate()).padStart(2, "0");
  return `${y}-${m}-${d}`;
}

/**
 * getMaxEndDate — 주간 리포트가 이동할 수 있는 최신 종료일 반환
 * @param {object|null} report - 주간 리포트 응답
 * @returns {string} 서버 활성 날짜 또는 브라우저 오늘 날짜
 */
function getMaxEndDate(report) {
  return report?.active_date || todayKey();
}

/**
 * SummaryCards — 주간 핵심 지표 카드 묶음
 * @param {object} summary - 주간 리포트 summary 블록
 */
function SummaryCards({ summary, isAdmin = true }) {
  const cumulativeDelta = summary.cumulative_wellness_delta ?? summary.wellness_delta;
  const dailyDelta = summary.daily_wellness_delta;
  const cumulativeDeltaText =
    cumulativeDelta == null
      ? null
      : cumulativeDelta > 0
        ? `▲ ${cumulativeDelta}`
        : cumulativeDelta < 0
          ? `▼ ${Math.abs(cumulativeDelta)}`
          : "변화 없음";
  const dailyDeltaText =
    dailyDelta == null
      ? null
      : dailyDelta > 0
        ? `▲ ${dailyDelta}`
        : dailyDelta < 0
          ? `▼ ${Math.abs(dailyDelta)}`
          : "변화 없음";
  const deltaColor =
    cumulativeDelta == null ? "#757575" : cumulativeDelta > 0 ? "#43a047" : cumulativeDelta < 0 ? "#e53935" : "#757575";
  const dailyAvg = summary.avg_daily_wellness;
  const cumulativeAvg = summary.avg_cumulative_wellness ?? summary.avg_wellness;

  return (
    <div className="report-cards">
      <div className="report-card">
        <span className="report-card-title">평균 웰니스</span>
        <span className="report-card-value">
          {isAdmin
            ? (dailyAvg != null ? Math.round(dailyAvg) : "—")
            : (labelForWellnessScore(dailyAvg) || "—")}
          <span className="report-card-inline-unit"> / </span>
          {isAdmin
            ? (cumulativeAvg != null ? Math.round(cumulativeAvg) : "—")
            : (labelForWellnessScore(cumulativeAvg) || "—")}
        </span>
        <span className="report-card-sub">단일 / 누적</span>
        {/* 비관리자는 숫자 변화량을 숨긴다(점수 비노출) */}
        {isAdmin && dailyDeltaText && (
          <span className="report-card-sub">
            단일 지난주 대비 {dailyDeltaText}
          </span>
        )}
        {isAdmin && cumulativeDeltaText && (
          <span className="report-card-sub" style={{ color: deltaColor }}>
            누적 지난주 대비 {cumulativeDeltaText}
          </span>
        )}
        {cumulativeAvg == null && (
          <span className="report-card-sub">마감된 날이 아직 없어요</span>
        )}
      </div>

      <div className="report-card">
        <span className="report-card-title">기록한 날</span>
        <span className="report-card-value">{summary.active_days}/7</span>
        <span className="report-card-sub">대화 {summary.total_utterances}개</span>
      </div>

      <div className="report-card">
        <span className="report-card-title">가장 많이 느낀 감정</span>
        <span className="report-card-value">{summary.top_emotion || "—"}</span>
        <span className="report-card-sub">
          {summary.top_emotion
            ? `${summary.emotion_counts[summary.top_emotion]}회 감지`
            : "기록이 쌓이면 표시돼요"}
        </span>
      </div>

      <div className="report-card">
        <span className="report-card-title">위기 신호</span>
        <span
          className="report-card-value"
          style={{ color: summary.crisis_count > 0 ? "#e53935" : "#43a047" }}
        >
          {summary.crisis_count}건
        </span>
        <span className="report-card-sub">
          {summary.crisis_count > 0 ? "안전을 먼저 챙겨 주세요" : "감지된 위기 없음"}
        </span>
      </div>
    </div>
  );
}

/**
 * WeeklyWellnessBar — 7일 웰니스 막대 차트 (한 점수 축만 표시)
 * @param {Array} days - 주간 리포트 days 배열
 */
function WeeklyWellnessBar({ days, scoreKey, title, color, showScore = true }) {
  const labels = days.map((d) => formatDateLabel(d.date));
  const values = days.map((d) => {
    const score = d[scoreKey] ?? (scoreKey === "cumulative_wellness_score" ? d.wellness_score : null);
    return score == null ? null : Math.round(score);
  });
  const colors = color === "label"
    ? days.map((d) => {
        const score = d[scoreKey] ?? (scoreKey === "cumulative_wellness_score" ? d.wellness_score : null);
        return LABEL_COLOR[getScoreLabel(d, scoreKey, score)] || "#b0bec5";
      })
    : days.map(() => color);

  const data = {
    labels,
    datasets: [
      {
        label: title,
        data: values,
        backgroundColor: colors,
        borderRadius: 6,
        maxBarThickness: 42,
      },
    ],
  };

  const options = {
    responsive: true,
    plugins: {
      legend: { display: false },
      tooltip: {
        callbacks: {
          label: (ctx) => {
            const day = days[ctx.dataIndex];
            if (ctx.parsed.y == null) return " 아직 마감되지 않은 날";
            const label = getScoreLabel(day, scoreKey, ctx.parsed.y);
            // 비관리자는 숫자 없이 상태(색상 라벨)만 보여준다
            if (!showScore) return ` ${title}: ${label || "—"}`;
            return ` ${title} ${ctx.parsed.y}점${label ? ` (${label})` : ""}`;
          },
        },
      },
    },
    scales: {
      y: {
        min: 0,
        max: 100,
        // 비관리자는 y축 점수 눈금 숫자를 숨긴다
        ticks: showScore ? { stepSize: 20 } : { display: false },
        grid: { color: "#eeeeee" },
      },
      x: { grid: { display: false }, ticks: { font: { size: 11 } } },
    },
  };

  return (
    <div className="wellness-chart-wrap">
      <h3 className="chart-title">이번 주 {title}</h3>
      <Bar data={data} options={options} />
      <p className="chart-note" style={{ fontSize: "12px", color: "#666", marginTop: 8 }}>
        * 막대가 비어 있는 날은 아직 마감되지 않았거나 기록이 없는 날입니다. 오늘 점수는 하루 마감 후 반영됩니다.
      </p>
    </div>
  );
}

/**
 * EmotionDoughnut — 주간 감정 분포 도넛 차트
 * @param {object} emotionCounts - {감정 라벨: 발화 수}
 */
function EmotionDoughnut({ emotionCounts }) {
  const entries = Object.entries(emotionCounts || {});
  if (entries.length === 0) {
    return (
      <div className="wellness-chart-wrap">
        <h3 className="chart-title">감정 분포</h3>
        <p className="chart-empty">이번 주에 감지된 감정 기록이 없습니다.</p>
      </div>
    );
  }

  const labels = entries.map(([label]) => label);
  const values = entries.map(([, count]) => count);
  const colors = labels.map((label) => EMOTION_COLOR[label] || "#b0bec5");

  const data = {
    labels,
    datasets: [
      {
        data: values,
        backgroundColor: colors,
        borderWidth: 2,
        borderColor: "#ffffff",
      },
    ],
  };

  const options = {
    responsive: true,
    cutout: "58%",
    plugins: {
      legend: { position: "right", labels: { boxWidth: 14, font: { size: 12 } } },
      tooltip: {
        callbacks: {
          label: (ctx) => ` ${ctx.label}: ${ctx.parsed}회`,
        },
      },
    },
  };

  return (
    <div className="wellness-chart-wrap report-doughnut-wrap">
      <h3 className="chart-title">감정 분포</h3>
      <div className="report-doughnut-box">
        <Doughnut data={data} options={options} />
      </div>
    </div>
  );
}

/**
 * WeeklyInsightPanel — 백엔드가 생성한 주간 관찰 문장 표시
 * @param {object|null} weeklySummary - {title, items:[{title, body, tone}]}
 */
function WeeklyInsightPanel({ weeklySummary }) {
  const items = weeklySummary?.items || [];
  if (items.length === 0) return null;

  return (
    <section className="report-insight-section">
      <h3 className="chart-title">{weeklySummary.title || "이번 주 정리"}</h3>
      <div className="report-insight-list">
        {items.map((item, index) => (
          <article
            key={`${item.title || "insight"}-${index}`}
            className={`report-insight-item report-insight-${item.tone || "neutral"}`}
          >
            <span className="report-insight-title">{item.title}</span>
            <p>{item.body}</p>
          </article>
        ))}
      </div>
    </section>
  );
}

/**
 * WeekdayEmotionSource — 한 출처의 최근 8주 요일별 감정분포 표시
 * @param {object} sourceData - 출처별 weekday_emotion_patterns 하위 데이터
 */
function WeekdayEmotionSource({ sourceData }) {
  const rows = sourceData?.weekdays || [];
  const patterns = sourceData?.patterns || [];
  const hasAnyRecord = Number(sourceData?.total || 0) > 0;

  return (
    <div className="weekday-emotion-source">
      <div className="weekday-emotion-source-head">
        <h4>{sourceData?.source_label || "감정 기록"}</h4>
        <span>{sourceData?.total || 0}회</span>
      </div>

      <div className="weekday-emotion-rows">
        {rows.map((row) => {
          const entries = Object.entries(row.emotion_counts || {});
          return (
            <div key={row.weekday} className="weekday-emotion-row">
              <span className="weekday-emotion-name">{row.weekday}</span>
              <div className="weekday-emotion-bar" aria-label={`${row.weekday}요일 감정분포`}>
                {entries.length === 0 && <span className="weekday-emotion-empty-bar" />}
                {entries.map(([emotion, count]) => (
                  <span
                    key={emotion}
                    className="weekday-emotion-segment"
                    style={{
                      backgroundColor: EMOTION_COLOR[emotion] || "#b0bec5",
                      flexGrow: count,
                    }}
                    title={`${emotion} ${count}회`}
                  />
                ))}
              </div>
              <span className="weekday-emotion-top">
                {row.top_emotion ? `${row.top_emotion} ${row.top_count}` : "—"}
              </span>
            </div>
          );
        })}
      </div>

      <div className="weekday-pattern-list">
        {patterns.map((pattern, index) => (
          <p
            key={`${pattern.weekday || "none"}-${pattern.emotion || "none"}-${index}`}
            className={`weekday-pattern-note weekday-pattern-${pattern.tone || "neutral"}`}
          >
            {pattern.message}
          </p>
        ))}
        {!hasAnyRecord && patterns.length === 0 && (
          <p className="weekday-pattern-note weekday-pattern-neutral">
            기록이 조금 더 쌓이면 요일별 반복 패턴을 볼 수 있어요.
          </p>
        )}
      </div>
    </div>
  );
}

/**
 * WeekdayEmotionPatterns — 최근 8주 요일별 감정분포 섹션
 * @param {object|null} patternData - weekday_emotion_patterns 응답 블록
 */
function WeekdayEmotionPatterns({ patternData }) {
  if (!patternData) return null;
  const rangeText = patternData.window_start && patternData.window_end
    ? `${formatDateLabel(patternData.window_start)} ~ ${formatDateLabel(patternData.window_end)}`
    : `최근 ${patternData.weeks || 8}주`;

  return (
    <section className="report-weekday-section">
      <div className="report-section-head">
        <h3 className="chart-title">요일별 감정분포</h3>
        <span>{rangeText}</span>
      </div>
      {/* 색상 → 감정 범례 (막대 hover 시 "감정 N회" tooltip도 표시됨) */}
      <div className="weekday-emotion-legend">
        {Object.entries(EMOTION_COLOR).map(([emotion, color]) => (
          <span key={emotion} className="weekday-emotion-legend-item">
            <span className="legend-dot" style={{ backgroundColor: color }} />
            {emotion}
          </span>
        ))}
      </div>
      <div className="weekday-emotion-grid">
        <WeekdayEmotionSource sourceData={patternData.model} />
        <WeekdayEmotionSource sourceData={patternData.manual} />
      </div>
    </section>
  );
}

/**
 * WeeklyReport — 주간 리포트 탭 메인 컴포넌트
 * @param {string} username - 현재 사용자 이름
 * @param {boolean} isActive - 현재 리포트 탭 활성 여부
 */
export default function WeeklyReport({ username, isActive = true }) {
  // endDate: 조회 중인 주의 마지막 날짜 (null이면 서버 활성 날짜 기준)
  const [endDate, setEndDate] = useState(null);
  const [report,  setReport]  = useState(null);
  const [loading, setLoading] = useState(true);
  const [error,   setError]   = useState(null);
  // 관리자 여부 — 비관리자에겐 평균/막대 점수 숫자를 숨기고 색상·상태로만 보여준다
  const [isAdmin, setIsAdmin] = useState(false);
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
    return () => {
      cancelled = true;
    };
  }, [username]);

  // ── 데이터 로드 ─────────────────────────────────────────────────────────────
  const loadReport = useCallback(async () => {
    if (!username) return;
    setLoading(true);
    setError(null);
    try {
      const data = await getWeeklyReport(username, endDate);
      setReport(data);
    } catch (err) {
      console.error("getWeeklyReport 오류:", err);
      setError("주간 리포트를 불러오지 못했습니다.");
    } finally {
      setLoading(false);
    }
  }, [username, endDate]);

  useEffect(() => {
    if (isActive) loadReport();
  }, [isActive, loadReport]);

  // ── 주 이동 ─────────────────────────────────────────────────────────────────
  const goPrevWeek = () => {
    if (!report || loading) return;
    setEndDate(shiftDate(report.end_date, -7));
  };

  const goNextWeek = () => {
    if (!report || loading) return;
    const maxEndDate = getMaxEndDate(report);
    if (report.end_date >= maxEndDate) return;
    const next = shiftDate(report.end_date, 7);
    // 최신 주는 null로 조회해 서버 활성 날짜 기준과 항상 맞춘다.
    setEndDate(next >= maxEndDate ? null : next);
  };

  const isLatestWeek = !report || report.end_date >= getMaxEndDate(report);

  return (
    <div className="report-page">
      {/* 주 네비게이션 */}
      <div className="cal-nav">
        <button className="cal-nav-btn" onClick={goPrevWeek} disabled={loading || !report} aria-label="지난주">‹</button>
        <h2 className="cal-month-title">
          {report ? `${formatDateLabel(report.start_date)} ~ ${formatDateLabel(report.end_date)}` : "주간 리포트"}
        </h2>
        <button
          className="cal-nav-btn"
          onClick={goNextWeek}
          disabled={loading || isLatestWeek}
          aria-label="다음주"
        >
          ›
        </button>
        <button className="cal-refresh-btn" onClick={loadReport} aria-label="새로고침">🔄</button>
      </div>

      {loading && <div className="cal-loading"><div className="spinner" /> 불러오는 중...</div>}
      {error   && <div className="cal-error">⚠️ {error}</div>}

      {!loading && !error && report && (
        <>
          {/* 요약 카드 */}
          <SummaryCards summary={report.summary} isAdmin={isAdmin} />

          {/* 주간 정리 */}
          <WeeklyInsightPanel weeklySummary={report.weekly_summary} />

          {/* 레이블 분포 칩 */}
          {Object.keys(report.summary.label_counts || {}).length > 0 && (
            <div className="report-label-chips">
              {Object.entries(LABEL_COLOR).map(([label, color]) => {
                const count = report.summary.label_counts[label];
                if (!count) return null;
                return (
                  <span key={label} className="report-label-chip" style={{ borderColor: color }}>
                    <span className="legend-dot" style={{ backgroundColor: color }} />
                    {label} {count}일
                  </span>
                );
              })}
            </div>
          )}

          {/* 차트 영역 */}
          <div className="chart-section">
            <WeeklyWellnessBar
              days={report.days}
              scoreKey="daily_wellness_score"
              title="단일 웰니스"
              color="label"
              showScore={isAdmin}
            />
            <WeeklyWellnessBar
              days={report.days}
              scoreKey="cumulative_wellness_score"
              title="누적 웰니스"
              color="label"
              showScore={isAdmin}
            />
            <EmotionDoughnut emotionCounts={report.summary.emotion_counts} />
          </div>

          {/* 최근 8주 요일별 감정 패턴 */}
          <WeekdayEmotionPatterns patternData={report.weekday_emotion_patterns} />

          <p className="report-disclaimer">
            이 리포트는 의료 진단이 아닌 개인 정서 모니터링 참고 자료입니다.
            힘든 시기가 이어진다면 전문가 상담(정신건강 위기상담 109)을 권해요.
          </p>
        </>
      )}
    </div>
  );
}
