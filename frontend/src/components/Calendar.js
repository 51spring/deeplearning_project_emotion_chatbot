/**
 * Calendar.js
 * 역할: 캘린더 화면 컴포넌트
 *       - 월별 날짜 그리드 — 웰니스 레이블에 따른 색상 도트 표시
 *       - 날짜별 사용자 수동 감정 기록 표시/저장
 *       - 위기 발생일 ⚠️ 뱃지 표시
 *       - 하단 단일/누적 웰니스와 우울 경향 추이 선형 차트 (Chart.js) — 축별 별도 그래프
 *       - 선택한 날짜의 상세 정보 팝오버 + 그날의 대화 다시 보기
 */
import React, { useState, useEffect, useCallback, useRef } from "react";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from "chart.js";
import { Line } from "react-chartjs-2";
import {
  deleteDailyEmotionNote,
  getCalendar,
  getCurrentDay,
  getDayUtterances,
  saveDailyEmotionNote,
} from "../api";

// Chart.js 필수 컴포넌트 등록
ChartJS.register(
  CategoryScale, LinearScale,
  PointElement, LineElement,
  Title, Tooltip, Legend, Filler
);

// ── 레이블 설정 ──────────────────────────────────────────────────────────────
const LABEL_COLOR = {
  양호: "#43a047",
  보통: "#29b6f6",
  주의: "#fb8c00",
  위험: "#e53935",
};

const LABEL_BG = {
  양호: "#e8f5e9",
  보통: "#e1f5fe",
  주의: "#fff3e0",
  위험: "#ffebee",
};

const EMOTION_OPTIONS = ["행복", "슬픔", "분노", "공포", "놀람", "혐오", "중립"];
const EMOTION_COLOR = {
  행복: "#00c853",
  슬픔: "#1e88e5",
  분노: "#d81b60",
  공포: "#8e24aa",
  놀람: "#ffb300",
  혐오: "#6d4c41",
  중립: "#90a4ae",
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
 * getScoreLabel — 단일/누적 점수 축에 맞는 레이블 선택
 * @param {object} item - 캘린더 일별 데이터
 * @param {string} scoreKey - 점수 필드명
 * @param {number|null} score - 현재 축 점수
 * @returns {string|null} 해당 축 전용 레이블
 */
function getScoreLabel(item, scoreKey, score) {
  if (scoreKey === "daily_wellness_score") {
    return item.daily_wellness_label || labelForWellnessScore(score);
  }
  return item.cumulative_wellness_label || item.label || labelForWellnessScore(score);
}

// 우울 경향 band — depression_tendency_v15_spec과 일치 (high≥0.40 / mid≥0.20 / low)
const TENDENCY_BAND_COLOR = {
  high: "#8e24aa",
  mid:  "#5e35b1",
  low:  "#90a4ae",
};
const TENDENCY_BAND_LABEL = {
  high: "강함",
  mid:  "일부",
  low:  "미감지",
};
function classifyTendencyBand(score) {
  if (score == null || isNaN(score)) return "low";
  if (score >= 0.40) return "high";
  if (score >= 0.20) return "mid";
  return "low";
}

// 색상 밴드 추이용 — 밴드 코드를 y 레벨로 매핑(비관리자: 소수점 없이 색상으로만 추이 표시)
const BAND_LEVEL = { low: 1, mid: 2, high: 3 };
const BAND_LEVEL_LABEL = { 1: "미감지", 2: "일부", 3: "강함" };

const WEEKDAYS = ["일", "월", "화", "수", "목", "금", "토"];

/**
 * buildCalendarGrid — 해당 월의 달력 그리드 생성
 * @param {number} year  - 연도
 * @param {number} month - 월 (0-indexed)
 * @returns {Array<Date|null>} 7×n 배열 (null은 빈 셀)
 */
function buildCalendarGrid(year, month) {
  const firstDay = new Date(year, month, 1).getDay(); // 0=일
  const lastDate = new Date(year, month + 1, 0).getDate();
  const grid = [];

  for (let i = 0; i < firstDay; i++) grid.push(null);
  for (let d = 1; d <= lastDate; d++) grid.push(new Date(year, month, d));
  // 마지막 주 패딩
  while (grid.length % 7 !== 0) grid.push(null);
  return grid;
}

/**
 * dateKey — Date → "YYYY-MM-DD" 문자열
 */
function dateKey(date) {
  if (!date) return null;
  const y = date.getFullYear();
  const m = String(date.getMonth() + 1).padStart(2, "0");
  const d = String(date.getDate()).padStart(2, "0");
  return `${y}-${m}-${d}`;
}

/**
 * hasModelDayData — 수동 감정 외의 모델/대화 기반 캘린더 데이터 존재 여부 확인
 * @param {object|null} info - 날짜별 캘린더 데이터
 * @returns {boolean} 모델 요약/점수/대화 카운트 존재 여부
 */
function hasModelDayData(info) {
  if (!info) return false;
  return Boolean(
    info.label
    || info.daily_wellness_label
    || info.cumulative_wellness_label
    || info.wellness_score != null
    || info.daily_wellness_score != null
    || info.cumulative_wellness_score != null
    || (info.utterance_count ?? 0) > 0
    || (info.crisis_count_day ?? 0) > 0
  );
}

/**
 * CalendarGrid — 월별 달력 그리드
 */
function CalendarGrid({ year, month, dataMap, selectedDate, onSelectDate }) {
  const grid = buildCalendarGrid(year, month);
  const today = dateKey(new Date());

  return (
    <div className="cal-grid">
      {/* 요일 헤더 */}
      {WEEKDAYS.map((w) => (
        <div key={w} className="cal-weekday">{w}</div>
      ))}

      {/* 날짜 셀 */}
      {grid.map((date, idx) => {
        if (!date) return <div key={`empty-${idx}`} className="cal-cell cal-empty" />;

        const key  = dateKey(date);
        const info = dataMap[key];
        const label = info?.label;
        const manualEmotion = info?.manual_emotion_label;
        const isCrisis = info?.crisis_count_day > 0;
        const isToday  = key === today;
        const isSelected = key === selectedDate;

        return (
          <div
            key={key}
            className={`cal-cell ${isToday ? "cal-today" : ""} ${isSelected ? "cal-selected" : ""}`}
            onClick={() => onSelectDate(key, info)}
            role="button"
            tabIndex={0}
            aria-label={`${date.getDate()}일${label ? " " + label : ""}${manualEmotion ? " 수동기록 " + manualEmotion : ""}`}
            onKeyDown={(e) => e.key === "Enter" && onSelectDate(key, info)}
          >
            <span className="cal-date-num">{date.getDate()}</span>
            {label && (
              <span
                className="cal-dot"
                style={{ backgroundColor: LABEL_COLOR[label] }}
                title={label}
              />
            )}
            {manualEmotion && (
              <span
                className="cal-manual-dot"
                style={{ backgroundColor: EMOTION_COLOR[manualEmotion] || "#78909c" }}
                title={`내 기록: ${manualEmotion}`}
              />
            )}
            {isCrisis && <span className="cal-crisis-badge" title="위기 감지">⚠</span>}
          </div>
        );
      })}
    </div>
  );
}

/**
 * DayConversation — 선택 날짜의 저장된 대화 다시 보기 목록
 * @param {Array|null} utterances - 해당 날짜 발화 리스트 (null이면 로딩 전)
 * @param {boolean}    loading    - 대화 불러오는 중 여부
 */
function DayConversation({ utterances, loading }) {
  if (loading) {
    return <p className="day-convo-status">대화를 불러오는 중...</p>;
  }
  if (!utterances || utterances.length === 0) {
    return <p className="day-convo-status">이날 저장된 대화가 없습니다.</p>;
  }
  return (
    <div className="day-convo-list">
      {utterances.map((utt) => (
        <div
          key={utt.id}
          className={`day-convo-item ${utt.role === "user" ? "day-convo-user" : "day-convo-bot"}`}
        >
          <span className="day-convo-role">
            {utt.role === "user" ? "나" : "챗봇"}
          </span>
          <span className="day-convo-text">{utt.text}</span>
          {utt.role === "user" && utt.emotion && (
            <span className="day-convo-emotion">
              {utt.feedback?.corrected_emotion || utt.emotion}
            </span>
          )}
        </div>
      ))}
    </div>
  );
}

/**
 * ManualEmotionEditor — 날짜별 사용자 수동 감정 기록 입력 폼
 * @param {string} date - 선택 날짜
 * @param {object} info - 선택 날짜 데이터
 * @param {boolean} saving - 저장/삭제 진행 여부
 * @param {string|null} error - 저장 오류 메시지
 * @param {Function} onSave - 저장 핸들러
 * @param {Function} onDelete - 삭제 핸들러
 */
function ManualEmotionEditor({ date, info, saving, error, onSave, onDelete }) {
  const [emotion, setEmotion] = useState(info?.manual_emotion_label || "중립");
  const [intensity, setIntensity] = useState(info?.manual_emotion_intensity || 3);
  const [note, setNote] = useState(info?.manual_emotion_note || "");
  const hasSavedNote = Boolean(info?.manual_emotion_label);
  const [isEditing, setIsEditing] = useState(!hasSavedNote);

  useEffect(() => {
    setEmotion(info?.manual_emotion_label || "중립");
    setIntensity(info?.manual_emotion_intensity || 3);
    setNote(info?.manual_emotion_note || "");
    setIsEditing(!info?.manual_emotion_label);
  }, [
    date,
    info?.manual_emotion_label,
    info?.manual_emotion_intensity,
    info?.manual_emotion_note,
  ]);

  /**
   * resetDraft — 저장된 수동 감정 기록으로 편집 값을 되돌림
   * @param {void} 없음
   * @returns {void}
   */
  const resetDraft = () => {
    setEmotion(info?.manual_emotion_label || "중립");
    setIntensity(info?.manual_emotion_intensity || 3);
    setNote(info?.manual_emotion_note || "");
  };

  /**
   * handleSubmit — 수동 감정 기록을 저장하고 성공 시 보기 모드로 전환
   * @param {Event} event - 폼 제출 이벤트
   * @returns {Promise<void>}
   */
  const handleSubmit = async (event) => {
    event.preventDefault();
    const saved = await onSave(date, { emotionLabel: emotion, intensity, note });
    if (saved) {
      setIsEditing(false);
    }
  };

  /**
   * handleCancel — 편집을 취소하고 저장된 보기 모드로 복귀
   * @param {void} 없음
   * @returns {void}
   */
  const handleCancel = () => {
    resetDraft();
    setIsEditing(false);
  };

  /**
   * handleDeleteClick — 저장된 수동 감정 기록을 삭제
   * @param {void} 없음
   * @returns {Promise<void>}
   */
  const handleDeleteClick = async () => {
    const deleted = await onDelete(date);
    if (deleted) {
      setIsEditing(true);
    }
  };

  if (hasSavedNote && !isEditing) {
    return (
      <div className="manual-emotion-form manual-emotion-view">
        <div className="manual-emotion-head">
          <span className="detail-key">내 감정 기록</span>
          <span className="manual-emotion-current">저장됨</span>
        </div>

        <div className="manual-emotion-summary">
          <div className="manual-emotion-summary-main">
            <span
              className="manual-emotion-swatch"
              style={{ backgroundColor: EMOTION_COLOR[info.manual_emotion_label] || "#78909c" }}
            />
            <strong>{info.manual_emotion_label}</strong>
            <span>{info.manual_emotion_intensity}/5</span>
          </div>
          {info.manual_emotion_note ? (
            <p className="manual-emotion-summary-note">{info.manual_emotion_note}</p>
          ) : (
            <p className="manual-emotion-summary-note manual-emotion-summary-empty">메모 없음</p>
          )}
        </div>

        {error && <p className="manual-emotion-error">{error}</p>}

        <div className="manual-emotion-actions">
          <button
            type="button"
            className="manual-emotion-edit"
            onClick={() => setIsEditing(true)}
            disabled={saving}
          >
            수정
          </button>
          <button
            type="button"
            className="manual-emotion-delete"
            onClick={handleDeleteClick}
            disabled={saving}
          >
            삭제
          </button>
        </div>
      </div>
    );
  }

  return (
    <form className="manual-emotion-form" onSubmit={handleSubmit}>
      <div className="manual-emotion-head">
        <span className="detail-key">내 감정 기록</span>
        {hasSavedNote && (
          <span
            className="manual-emotion-current"
            style={{ color: EMOTION_COLOR[info.manual_emotion_label] || "#78909c" }}
          >
            {info.manual_emotion_label} · {info.manual_emotion_intensity}/5
          </span>
        )}
      </div>

      <div className="manual-emotion-options" role="group" aria-label="감정 선택">
        {EMOTION_OPTIONS.map((label) => (
          <button
            key={label}
            type="button"
            className={`manual-emotion-chip ${emotion === label ? "manual-emotion-selected" : ""}`}
            onClick={() => setEmotion(label)}
            aria-pressed={emotion === label}
            disabled={saving}
          >
            <span
              className="manual-emotion-swatch"
              style={{ backgroundColor: EMOTION_COLOR[label] }}
            />
            {label}
          </button>
        ))}
      </div>

      <div className="manual-intensity-row" role="group" aria-label="감정 강도">
        <span className="manual-intensity-label">강도</span>
        {[1, 2, 3, 4, 5].map((value) => (
          <button
            key={value}
            type="button"
            className={`manual-intensity-btn ${intensity === value ? "manual-intensity-selected" : ""}`}
            onClick={() => setIntensity(value)}
            aria-pressed={intensity === value}
            disabled={saving}
          >
            {value}
          </button>
        ))}
      </div>

      <textarea
        className="manual-emotion-note"
        value={note}
        maxLength={300}
        onChange={(event) => setNote(event.target.value)}
        placeholder="짧은 메모"
        disabled={saving}
      />

      {error && <p className="manual-emotion-error">{error}</p>}

      <div className="manual-emotion-actions">
        <button type="submit" className="manual-emotion-save" disabled={saving}>
          {saving ? "저장 중" : "저장"}
        </button>
        {hasSavedNote && (
          <button
            type="button"
            className="manual-emotion-cancel"
            onClick={handleCancel}
            disabled={saving}
          >
            취소
          </button>
        )}
      </div>
    </form>
  );
}

/**
 * DetailPopover — 선택 날짜 상세 정보 팝오버 (+ 그날의 대화 다시 보기)
 */
function DetailPopover({
  date,
  info,
  utterances,
  utterancesLoading,
  isAdmin,
  manualSaving,
  manualError,
  onManualSave,
  onManualDelete,
  onClose,
}) {
  const dayInfo = info || { date };
  const dailyWellness = dayInfo.daily_wellness_score ?? dayInfo.wellness_score;
  const cumulativeWellness = dayInfo.cumulative_wellness_score ?? dayInfo.wellness_score;
  const dailyLabel = dayInfo.daily_wellness_label || labelForWellnessScore(dailyWellness);
  const cumulativeLabel = dayInfo.cumulative_wellness_label || dayInfo.label || labelForWellnessScore(cumulativeWellness);
  const color = LABEL_COLOR[cumulativeLabel] || "#757575";
  const bg    = LABEL_BG[cumulativeLabel]    || "#f5f5f5";

  return (
    <div className="detail-popover" style={{ borderLeft: `4px solid ${color}`, background: bg }}>
      <div className="detail-header">
        <strong>{date}</strong>
        <button className="detail-close" onClick={onClose}>✕</button>
      </div>
      <div className="detail-body">
        {cumulativeLabel && (
          <p><span className="detail-key">누적 상태</span>
             <span style={{ color, fontWeight: 600 }}>{cumulativeLabel}</span></p>
        )}
        {/* 비관리자: 숫자 없이 색상 상태만(단일 상태). 관리자: 단일/누적 웰니스 숫자 + 우울 경향 소수점 */}
        {isAdmin ? (
          <>
            {dailyWellness != null && (
              <p><span className="detail-key">단일 웰니스</span>
                 <span style={{ color: LABEL_COLOR[dailyLabel] || "#424242", fontWeight: 600 }}>
                   {Math.round(dailyWellness)}점{dailyLabel ? ` (${dailyLabel})` : ""}
                 </span></p>
            )}
            {cumulativeWellness != null && (
              <p><span className="detail-key">누적 웰니스</span>
                 <span style={{ color: LABEL_COLOR[cumulativeLabel] || "#424242", fontWeight: 600 }}>
                   {Math.round(cumulativeWellness)}점{cumulativeLabel ? ` (${cumulativeLabel})` : ""}
                 </span></p>
            )}
            {dayInfo.depression_tendency_smoothed != null && (
              <p>
                <span className="detail-key">우울 경향(평활)</span>
                <span
                  style={{
                    color: TENDENCY_BAND_COLOR[classifyTendencyBand(dayInfo.depression_tendency_smoothed)],
                    fontWeight: 600,
                  }}
                >
                  {dayInfo.depression_tendency_smoothed.toFixed(3)}{" "}
                  <span style={{ fontSize: "12px" }}>
                    ({TENDENCY_BAND_LABEL[classifyTendencyBand(dayInfo.depression_tendency_smoothed)]})
                  </span>
                </span>
              </p>
            )}
            {dayInfo.depression_tendency_daily != null && (
              <p>
                <span className="detail-key">우울 경향(일별)</span>
                <span
                  style={{
                    color: TENDENCY_BAND_COLOR[classifyTendencyBand(dayInfo.depression_tendency_daily)],
                  }}
                >
                  {dayInfo.depression_tendency_daily.toFixed(3)}
                </span>
              </p>
            )}
          </>
        ) : (
          dailyLabel && (
            <p><span className="detail-key">단일 상태</span>
               <span style={{ color: LABEL_COLOR[dailyLabel] || "#757575", fontWeight: 600 }}>
                 {dailyLabel}
               </span></p>
          )
        )}
        <p><span className="detail-key">발화 수</span>
           <span>{dayInfo.utterance_count ?? 0}회</span></p>
        <div className="detail-summary">
          <span className="detail-key">요약</span>
          <p>{dayInfo.summary_text || "아직 표시할 요약이 없습니다."}</p>
        </div>
        {dayInfo.crisis_count_day > 0 && (
          <p style={{ color: "#e53935" }}>
            ⚠️ 위기 이벤트 {dayInfo.crisis_count_day}건 감지됨
          </p>
        )}
        <div className="detail-summary">
          <ManualEmotionEditor
            date={date}
            info={dayInfo}
            saving={manualSaving}
            error={manualError}
            onSave={onManualSave}
            onDelete={onManualDelete}
          />
        </div>
        {/* 그날의 대화 다시 보기 */}
        <div className="detail-summary">
          <span className="detail-key">그날의 대화</span>
          <DayConversation utterances={utterances} loading={utterancesLoading} />
        </div>
      </div>
    </div>
  );
}

/**
 * WellnessChart — 웰니스 추이 선형 차트 (한 점수 축만 표시)
 */
function WellnessChart({ calData, scoreKey, title, lineColor, fillColor, pointColor, showScore = true }) {
  // 전체 기록을 좌우 스크롤로 보여주고, 갱신 시 최신(오른쪽)으로 스크롤한다.
  const scrollRef = useRef(null);
  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollLeft = el.scrollWidth;
  }, [calData]);

  const recent = calData;
  if (recent.length === 0) {
    return <p className="chart-empty">아직 기록이 없습니다.</p>;
  }

  const labels = recent.map((d) => d.date.slice(5)); // "MM-DD"
  const values = recent.map((d) => {
    const score = d[scoreKey] ?? (scoreKey === "cumulative_wellness_score" ? d.wellness_score : null);
    return score == null ? null : Math.round(score);
  });
  const pointColors = recent.map((d) =>
    pointColor === "label"
      ? (LABEL_COLOR[getScoreLabel(d, scoreKey, d[scoreKey])] || lineColor)
      : pointColor
  );

  const data = {
    labels,
    datasets: [
      {
        label: title,
        data: values,
        borderColor: lineColor,
        backgroundColor: fillColor,
        pointBackgroundColor: pointColors,
        pointRadius: 6,
        pointHoverRadius: 8,
        tension: 0.35,
        fill: true,
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: {
        callbacks: {
          label: (ctx) => {
            const item = recent[ctx.dataIndex];
            const score = ctx.parsed.y;
            const label = getScoreLabel(item, scoreKey, score);
            // 비관리자는 숫자 없이 상태(색상 라벨)만 보여준다
            if (!showScore) return ` ${title}: ${label || "—"}`;
            return ` ${title} ${score}점${label ? ` (${label})` : ""}`;
          },
        },
      },
    },
    scales: {
      y: {
        min: 0,
        max: 100,
        // 비관리자는 y축 점수 눈금 숫자를 숨기고 색상/추세만 보이게 한다
        ticks: showScore ? { stepSize: 20 } : { display: false },
        grid: { color: "#eeeeee" },
      },
      x: {
        ticks: {
          maxRotation: 45,
          font: { size: 11 },
        },
        grid: { display: false },
      },
    },
    // 위험/주의 레이블 영역 배경 표시
    annotation: undefined,
  };

  return (
    <div className="wellness-chart-wrap">
      <h3 className="chart-title">{title} 추이</h3>
      <div className="chart-scroll" ref={scrollRef}>
        <div className="chart-canvas" style={{ width: `${Math.max(100, (recent.length / 30) * 100)}%` }}>
          <Line data={data} options={options} />
        </div>
      </div>
      <div className="chart-legend">
        {Object.entries(LABEL_COLOR).map(([lbl, clr]) => (
          <span key={lbl} className="chart-legend-item">
            <span className="legend-dot" style={{ backgroundColor: clr }} />
            {lbl}
          </span>
        ))}
      </div>
    </div>
  );
}

/**
 * TendencyChart — 최근 30일 우울 경향(평활) 추이 — 0~1 스케일
 *   종합 wellness와 분리해서 보여주기 위한 별도 차트
 */
function TendencyChart({ calData }) {
  // 전체 기록을 좌우 스크롤로 보여주고, 갱신 시 최신(오른쪽)으로 스크롤한다.
  const scrollRef = useRef(null);
  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollLeft = el.scrollWidth;
  }, [calData]);

  const recent = calData;
  const hasAny = recent.some((d) => d.depression_tendency_smoothed != null);
  if (!hasAny) {
    return null;
  }

  const labels = recent.map((d) => d.date.slice(5));
  const values = recent.map((d) =>
    d.depression_tendency_smoothed == null ? null : Number(d.depression_tendency_smoothed.toFixed(3))
  );
  const pointColors = recent.map((d) =>
    TENDENCY_BAND_COLOR[classifyTendencyBand(d.depression_tendency_smoothed)]
  );

  const data = {
    labels,
    datasets: [
      {
        label: "우울 경향(평활)",
        data: values,
        borderColor: "#8e24aa",
        backgroundColor: "rgba(142,36,170,0.10)",
        pointBackgroundColor: pointColors,
        pointRadius: 6,
        pointHoverRadius: 8,
        tension: 0.35,
        fill: true,
        spanGaps: true,
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: {
        callbacks: {
          label: (ctx) => {
            const v = ctx.parsed.y;
            const band = classifyTendencyBand(v);
            return ` 우울 경향 ${v?.toFixed(3) ?? "—"} (${TENDENCY_BAND_LABEL[band]})`;
          },
        },
      },
    },
    scales: {
      y: {
        min: 0,
        max: 1,
        ticks: { stepSize: 0.2 },
        grid: { color: "#eeeeee" },
      },
      x: {
        ticks: { maxRotation: 45, font: { size: 11 } },
        grid: { display: false },
      },
    },
  };

  return (
    <div className="wellness-chart-wrap">
      <h3 className="chart-title">우울 경향 추이</h3>
      <div className="chart-scroll" ref={scrollRef}>
        <div className="chart-canvas" style={{ width: `${Math.max(100, (recent.length / 30) * 100)}%` }}>
          <Line data={data} options={options} />
        </div>
      </div>
      <div className="chart-legend">
        {Object.entries(TENDENCY_BAND_LABEL).map(([key, lbl]) => (
          <span key={key} className="chart-legend-item">
            <span className="legend-dot" style={{ backgroundColor: TENDENCY_BAND_COLOR[key] }} />
            {lbl}
          </span>
        ))}
      </div>
      <p
        className="chart-note"
        style={{ fontSize: "12px", color: "#666", marginTop: 8 }}
      >
        * 우울 경향 점수는 명시 우울/무기력/흥미저하/무가치감/절망/수면식욕/사회적 위축 신호만 추적합니다. 종합 distress와 별개 축입니다.
      </p>
    </div>
  );
}

/**
 * TendencyBandChart — 비관리자용 우울 경향 "색상 밴드" 추이
 *   0~1 소수점을 노출하지 않고 강함/일부/미감지 3단계 색상으로만 추이를 보여준다.
 *   서버가 비관리자에게도 내려보내는 day별 depression_tendency_band를 사용한다.
 */
function TendencyBandChart({ calData }) {
  const scrollRef = useRef(null);
  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollLeft = el.scrollWidth;
  }, [calData]);

  const hasAny = calData.some((d) => d.depression_tendency_band != null);
  if (!hasAny) {
    return null;
  }

  const labels = calData.map((d) => d.date.slice(5));
  const values = calData.map((d) =>
    d.depression_tendency_band ? BAND_LEVEL[d.depression_tendency_band] : null
  );
  const pointColors = calData.map((d) =>
    d.depression_tendency_band
      ? TENDENCY_BAND_COLOR[d.depression_tendency_band]
      : TENDENCY_BAND_COLOR.low
  );

  const data = {
    labels,
    datasets: [
      {
        label: "우울 경향",
        data: values,
        borderColor: "#8e24aa",
        backgroundColor: "rgba(142,36,170,0.10)",
        pointBackgroundColor: pointColors,
        pointRadius: 5,
        pointHoverRadius: 7,
        stepped: true,
        fill: false,
        spanGaps: true,
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: {
        callbacks: {
          label: (ctx) => ` 우울 경향: ${BAND_LEVEL_LABEL[ctx.parsed.y] ?? "—"}`,
        },
      },
    },
    scales: {
      y: {
        min: 0.5,
        max: 3.5,
        ticks: { stepSize: 1, callback: (v) => BAND_LEVEL_LABEL[v] || "" },
        grid: { color: "#eeeeee" },
      },
      x: {
        ticks: { maxRotation: 45, font: { size: 11 } },
        grid: { display: false },
      },
    },
  };

  return (
    <div className="wellness-chart-wrap">
      <h3 className="chart-title">우울 경향 추이</h3>
      <div className="chart-scroll" ref={scrollRef}>
        <div className="chart-canvas" style={{ width: `${Math.max(100, (calData.length / 30) * 100)}%` }}>
          <Line data={data} options={options} />
        </div>
      </div>
      <div className="chart-legend">
        {Object.entries(TENDENCY_BAND_LABEL).map(([key, lbl]) => (
          <span key={key} className="chart-legend-item">
            <span className="legend-dot" style={{ backgroundColor: TENDENCY_BAND_COLOR[key] }} />
            {lbl}
          </span>
        ))}
      </div>
      <p
        className="chart-note"
        style={{ fontSize: "12px", color: "#666", marginTop: 8 }}
      >
        * 우울 경향은 명시 우울/무기력/흥미저하 신호만 추적하는 참고 지표예요. 의료 진단이 아닙니다.
      </p>
    </div>
  );
}

/**
 * Calendar — 캘린더 탭 메인 컴포넌트
 * @param {string} username - 현재 사용자 이름
 * @param {boolean} isActive - 현재 캘린더 탭 활성 여부
 */
export default function Calendar({ username, isActive = true }) {
  const today = new Date();
  const [viewYear,  setViewYear]  = useState(today.getFullYear());
  const [viewMonth, setViewMonth] = useState(today.getMonth()); // 0-indexed
  const [calData,   setCalData]   = useState([]);
  const [dataMap,   setDataMap]   = useState({});
  const [loading,   setLoading]   = useState(true);
  const [error,     setError]     = useState(null);
  const [selectedDate, setSelectedDate] = useState(null);
  const [selectedInfo, setSelectedInfo] = useState(null);
  const [manualSaving, setManualSaving] = useState(false);
  const [manualError, setManualError] = useState(null);
  // 선택 날짜의 저장된 대화 (다시 보기)
  const [dayUtterances, setDayUtterances] = useState(null);
  const [dayUttsLoading, setDayUttsLoading] = useState(false);
  const dayRequestRef = useRef(0);
  // 관리자 여부 — 비관리자에겐 웰니스 숫자/그래프를 숨기고 색상 상태 + 우울경향 색상밴드만 노출한다
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
  const loadData = useCallback(async () => {
    if (!username) return;
    setLoading(true);
    setError(null);
    try {
      // 추이 차트 좌우 스크롤로 과거 값까지 보도록 충분한 기간을 가져온다.
      const data = await getCalendar(username, 365);
      setCalData(data);
      // date 키 → info 매핑
      const map = {};
      data.forEach((d) => { map[d.date] = d; });
      setDataMap(map);
    } catch (err) {
      setError("캘린더 데이터를 불러오지 못했습니다.");
    } finally {
      setLoading(false);
    }
  }, [username]);

  useEffect(() => {
    if (isActive) loadData();
  }, [isActive, loadData]);

  // ── 월 이동 ─────────────────────────────────────────────────────────────────
  const prevMonth = () => {
    setViewMonth((m) => {
      if (m === 0) { setViewYear((y) => y - 1); return 11; }
      return m - 1;
    });
    setSelectedDate(null);
    setSelectedInfo(null);
    setManualError(null);
  };

  const nextMonth = () => {
    setViewMonth((m) => {
      if (m === 11) { setViewYear((y) => y + 1); return 0; }
      return m + 1;
    });
    setSelectedDate(null);
    setSelectedInfo(null);
    setManualError(null);
  };

  /**
   * mergeCalendarRow — 저장/조회된 날짜 row를 캘린더 상태에 병합
   * @param {object} row - date 필드를 포함한 캘린더 row
   * @returns {void}
   */
  const mergeCalendarRow = useCallback((row) => {
    if (!row?.date) return;
    setDataMap((prev) => {
      const merged = { ...(prev[row.date] || { date: row.date }), ...row };
      return { ...prev, [row.date]: merged };
    });
    setCalData((prev) => {
      const next = [...prev];
      const index = next.findIndex((item) => item.date === row.date);
      if (index >= 0) {
        next[index] = { ...next[index], ...row };
      } else {
        next.push({ date: row.date, ...row });
      }
      next.sort((a, b) => a.date.localeCompare(b.date));
      return next;
    });
    setSelectedInfo((prev) => (
      selectedDate === row.date
        ? { ...(prev || { date: row.date }), ...row }
        : prev
    ));
  }, [selectedDate]);

  /**
   * clearManualEmotionFromRow — 수동 감정 필드를 제거하고 필요 시 캘린더 row 삭제
   * @param {string} dateStr - 대상 날짜
   * @returns {void}
   */
  const clearManualEmotionFromRow = useCallback((dateStr) => {
    const clearFields = {
      manual_emotion_label: null,
      manual_emotion_intensity: null,
      manual_emotion_note: null,
      manual_emotion_updated_at: null,
    };
    setDataMap((prev) => {
      const existing = prev[dateStr];
      if (!existing) return prev;
      const cleared = { ...existing, ...clearFields };
      if (!hasModelDayData(cleared)) {
        const next = { ...prev };
        delete next[dateStr];
        return next;
      }
      return { ...prev, [dateStr]: cleared };
    });
    setCalData((prev) => (
      prev
        .map((item) => (item.date === dateStr ? { ...item, ...clearFields } : item))
        .filter((item) => item.date !== dateStr || hasModelDayData(item))
    ));
    setSelectedInfo((prev) => {
      const cleared = { ...(prev || { date: dateStr }), ...clearFields };
      return selectedDate === dateStr
        ? (hasModelDayData(cleared) ? cleared : { date: dateStr })
        : prev;
    });
  }, [selectedDate]);

  // ── 날짜 선택 ───────────────────────────────────────────────────────────────
  const handleSelectDate = (key, info) => {
    if (selectedDate === key) {
      dayRequestRef.current += 1;
      setSelectedDate(null);
      setSelectedInfo(null);
      setDayUtterances(null);
      setManualError(null);
      return;
    }
    setSelectedDate(key);
    setSelectedInfo(info || { date: key });
    setManualError(null);

    // 빈 날짜에도 수동 감정 기록을 남길 수 있으므로 대화 조회는 항상 시도한다.
    setDayUtterances(null);
    const requestId = dayRequestRef.current + 1;
    dayRequestRef.current = requestId;
    setDayUttsLoading(true);
    getDayUtterances(username, key)
      .then((data) => {
        if (dayRequestRef.current === requestId) {
          setDayUtterances(data.utterances || []);
        }
      })
      .catch((err) => {
        console.error("getDayUtterances 오류:", err);
        if (dayRequestRef.current === requestId) {
          setDayUtterances([]);
        }
      })
      .finally(() => {
        if (dayRequestRef.current === requestId) {
          setDayUttsLoading(false);
        }
      });
  };

  /**
   * handleManualSave — 캘린더 수동 감정 기록 저장
   * @param {string} dateStr - 대상 날짜
   * @param {{emotionLabel:string, intensity:number, note:string}} values - 입력값
   * @returns {Promise<boolean>} 저장 성공 여부
   */
  const handleManualSave = async (dateStr, values) => {
    setManualSaving(true);
    setManualError(null);
    try {
      const saved = await saveDailyEmotionNote(
        username,
        dateStr,
        values.emotionLabel,
        values.intensity,
        values.note,
      );
      mergeCalendarRow(saved);
      return true;
    } catch (err) {
      setManualError("감정 기록을 저장하지 못했습니다.");
      return false;
    } finally {
      setManualSaving(false);
    }
  };

  /**
   * handleManualDelete — 캘린더 수동 감정 기록 삭제
   * @param {string} dateStr - 대상 날짜
   * @returns {Promise<boolean>} 삭제 성공 여부
   */
  const handleManualDelete = async (dateStr) => {
    setManualSaving(true);
    setManualError(null);
    try {
      await deleteDailyEmotionNote(username, dateStr);
      clearManualEmotionFromRow(dateStr);
      return true;
    } catch (err) {
      setManualError("감정 기록을 삭제하지 못했습니다.");
      return false;
    } finally {
      setManualSaving(false);
    }
  };

  // ── 렌더링 ───────────────────────────────────────────────────────────────────
  const monthLabel = `${viewYear}년 ${viewMonth + 1}월`;

  return (
    <div className="calendar-page">
      {/* 상단: 달력 */}
      <div className="cal-section">
        {/* 월 네비게이션 */}
        <div className="cal-nav">
          <button className="cal-nav-btn" onClick={prevMonth} aria-label="이전 달">‹</button>
          <h2 className="cal-month-title">{monthLabel}</h2>
          <button className="cal-nav-btn" onClick={nextMonth} aria-label="다음 달">›</button>
          <button className="cal-refresh-btn" onClick={loadData} aria-label="새로고침">🔄</button>
        </div>

        {loading && <div className="cal-loading"><div className="spinner" /> 불러오는 중...</div>}
        {error   && <div className="cal-error">⚠️ {error}</div>}

        {!loading && !error && (
          <CalendarGrid
            year={viewYear}
            month={viewMonth}
            dataMap={dataMap}
            selectedDate={selectedDate}
            onSelectDate={handleSelectDate}
          />
        )}

        {/* 선택 날짜 팝오버 */}
        {selectedDate && (
          <DetailPopover
            date={selectedDate}
            info={selectedInfo}
            utterances={dayUtterances}
            utterancesLoading={dayUttsLoading}
            isAdmin={isAdmin}
            manualSaving={manualSaving}
            manualError={manualError}
            onManualSave={handleManualSave}
            onManualDelete={handleManualDelete}
            onClose={() => {
              dayRequestRef.current += 1;
              setSelectedDate(null);
              setSelectedInfo(null);
              setDayUtterances(null);
              setManualError(null);
            }}
          />
        )}
      </div>

      {/* 레이블 범례 */}
      <div className="cal-legend">
        {Object.entries(LABEL_COLOR).map(([lbl, clr]) => (
          <span key={lbl} className="cal-legend-item">
            <span className="legend-dot" style={{ backgroundColor: clr }} />
            {lbl}
          </span>
        ))}
        <span className="cal-legend-item">
          <span style={{ fontSize: "13px" }}>⚠</span> 위기 감지
        </span>
        <span className="cal-legend-item">
          <span className="cal-manual-legend-dot" /> 내 감정 기록
        </span>
      </div>

      {/* 하단: 추이 차트 — 단일/누적 웰니스 그래프는 모두에게 표시하되 비관리자는 숫자 숨김(색상/추세만).
          우울 경향만 관리자=소수점 차트, 비관리자=색상 밴드 추이로 분리 */}
      {!loading && (
        <div className="chart-section">
          <WellnessChart
            calData={calData}
            scoreKey="daily_wellness_score"
            title="단일 웰니스"
            lineColor="#00acc1"
            fillColor="rgba(0,172,193,0.08)"
            pointColor="label"
            showScore={isAdmin}
          />
          <WellnessChart
            calData={calData}
            scoreKey="cumulative_wellness_score"
            title="누적 웰니스"
            lineColor="#1565c0"
            fillColor="rgba(21,101,192,0.08)"
            pointColor="label"
            showScore={isAdmin}
          />
          {isAdmin ? (
            <TendencyChart calData={calData} />
          ) : (
            <TendencyBandChart calData={calData} />
          )}
        </div>
      )}
    </div>
  );
}
