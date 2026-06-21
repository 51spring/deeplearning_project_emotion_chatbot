/**
 * ChatWindow.js
 * 역할: 채팅 UI 메인 컴포넌트
 *       - 메시지 버블 (사용자 / 봇 구분)
 *       - 새로고침 시 오늘 대화 자동 복원 (GET /day/utterances)
 *       - 챗봇 응답 평가(👍/👎) + 감정 셀프 정정 피드백 (POST /feedback)
 *       - 실시간 감정 점수 패널 (wellness_score + label)
 *       - 위기 감지 시 경고 배너 표시
 *       - 하루 마감 버튼 → DailySummary 모달 호출
 *       - 관리자 계정에서만 다음날 넘기기/DB 초기화 버튼 노출
 */
import React, { useState, useRef, useEffect, useCallback } from "react";
import {
  advanceDay,
  getCurrentDay,
  getDayUtterances,
  resetDatabase,
  sendFeedback,
  sendMessage,
} from "../api";
import RecommendationCard from "./RecommendationCard";

// ── 레이블별 스타일 설정 ─────────────────────────────────────────────────────
const LABEL_CONFIG = {
  양호: { color: "#43a047", bg: "#e8f5e9", emoji: "😊" },
  보통: { color: "#29b6f6", bg: "#e1f5fe", emoji: "😐" },
  주의: { color: "#fb8c00", bg: "#fff3e0", emoji: "😟" },
  위험: { color: "#e53935", bg: "#ffebee", emoji: "😨" },
};

// ── 감정 레이블 한국어 매핑 ──────────────────────────────────────────────────
const EMOTION_MAP = {
  neutral:   "중립",
  happiness: "행복",
  sadness:   "슬픔",
  anger:     "분노",
  fear:      "공포",
  disgust:   "혐오",
  surprise:  "놀람",
};

// 감정 셀프 정정 드롭다운 선택지 — 백엔드 EMOTION_LABELS_KO와 동일해야 함
const EMOTION_CHOICES = ["중립", "행복", "슬픔", "분노", "공포", "혐오", "놀람"];

// ── 아바타 이모티콘 선택 ──────────────────────────────────────────────────────
// 사용자/챗봇 말풍선 옆 아바타를 프리셋에서 고른다. ""(빈 값)은 "없음"(아바타 숨김).
const AVATAR_NONE = "";
const AVATAR_OPTIONS = [
  { value: AVATAR_NONE, label: "없음", title: "아바타 없음" },
  { value: "🙂", label: "🙂", title: "웃는 얼굴" },
  { value: "😀", label: "😀", title: "활짝 웃음" },
  { value: "😊", label: "😊", title: "미소" },
  { value: "😎", label: "😎", title: "선글라스" },
  { value: "🐱", label: "🐱", title: "고양이" },
  { value: "🐶", label: "🐶", title: "강아지" },
  { value: "🦊", label: "🦊", title: "여우" },
  { value: "🌟", label: "🌟", title: "별" },
  { value: "🌱", label: "🌱", title: "새싹" },
  { value: "🤖", label: "🤖", title: "로봇" },
  { value: "💬", label: "💬", title: "말풍선" },
];
const DEFAULT_USER_AVATAR = "🙂";
const DEFAULT_BOT_AVATAR  = "🤖";
const AVATAR_STORAGE_PREFIX = "chatAvatars:";

/**
 * loadAvatars — 계정별 아바타 설정을 localStorage에서 불러온다
 * @param {string} username - 현재 사용자 이름
 * @returns {{user: string, bot: string}} 저장값 또는 기본 아바타
 */
function loadAvatars(username) {
  try {
    const raw = localStorage.getItem(AVATAR_STORAGE_PREFIX + username);
    if (raw) {
      const parsed = JSON.parse(raw);
      return {
        user: typeof parsed.user === "string" ? parsed.user : DEFAULT_USER_AVATAR,
        bot:  typeof parsed.bot  === "string" ? parsed.bot  : DEFAULT_BOT_AVATAR,
      };
    }
  } catch (err) {
    // localStorage 비활성/파싱 실패 시 기본 아바타로 fallback
  }
  return { user: DEFAULT_USER_AVATAR, bot: DEFAULT_BOT_AVATAR };
}

/**
 * saveAvatars — 계정별 아바타 설정을 localStorage에 저장한다
 * @param {string} username - 현재 사용자 이름
 * @param {{user: string, bot: string}} avatars - 저장할 아바타 설정
 * @returns {void}
 */
function saveAvatars(username, avatars) {
  try {
    localStorage.setItem(AVATAR_STORAGE_PREFIX + username, JSON.stringify(avatars));
  } catch (err) {
    // 저장 실패는 무시 — 이번 세션 동안 화면 상태로만 유지된다
  }
}

/**
 * createClientSessionId — 화면 채팅창 단위 문맥 ID 생성
 * @returns {string} 새 화면 대화 문맥 ID
 */
function createClientSessionId() {
  return `${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

/**
 * formatUtcTime — 서버의 naive UTC 시각 문자열을 로컬 "HH:MM"으로 변환
 * @param {string|null} isoText - "YYYY-MM-DDTHH:MM:SS" 형식 (UTC 기준)
 * @returns {string} 로컬 시각 문자열 (변환 실패 시 빈 문자열)
 */
function formatUtcTime(isoText) {
  if (!isoText) return "";
  const parsed = new Date(isoText.endsWith("Z") ? isoText : `${isoText}Z`);
  if (isNaN(parsed.getTime())) return "";
  return parsed.toLocaleTimeString("ko-KR", { hour: "2-digit", minute: "2-digit" });
}

/**
 * WellnessBar — 현재 웰니스 점수 시각화 게이지
 */
function WellnessBar({ score, label, showScore = true }) {
  const cfg = LABEL_CONFIG[label] || LABEL_CONFIG["보통"];
  return (
    <div className="wellness-bar-container">
      <div className="wellness-bar-label">
        <span className="wellness-emoji">{cfg.emoji}</span>
        <span style={{ color: cfg.color, fontWeight: 600 }}>{label}</span>
        {/* 비관리자는 숫자 점수를 숨기고 색상 상태로만 표시한다 */}
        {showScore && <span className="wellness-score-num">{Math.round(score)}</span>}
      </div>
      <div className="wellness-bar-track">
        <div
          className="wellness-bar-fill"
          style={{
            // 비관리자는 점수 크기를 노출하지 않도록 라벨 색상으로 가득 채운 색 띠로 표시한다
            width: showScore ? `${Math.min(score, 100)}%` : "100%",
            backgroundColor: cfg.color,
            transition: "width 0.6s ease",
          }}
        />
      </div>
    </div>
  );
}

/**
 * AvatarPicker — 아바타 이모티콘 선택 위젯 (프리셋 + 없음)
 * @param {string}   label    - 대상 라벨 ("나" | "챗봇")
 * @param {string}   value    - 현재 선택값 (""=없음)
 * @param {Function} onChange - 선택 변경 콜백 (value) => void
 */
function AvatarPicker({ label, value, onChange }) {
  return (
    <div className="avatar-row">
      <span className="avatar-row-label">{label}</span>
      <div className="avatar-choices" role="group" aria-label={`${label} 아바타 선택`}>
        {AVATAR_OPTIONS.map((opt) => (
          <button
            key={opt.value || "none"}
            type="button"
            className={
              "avatar-choice" +
              (opt.value === AVATAR_NONE ? " avatar-choice-none" : "") +
              (value === opt.value ? " avatar-choice-selected" : "")
            }
            onClick={() => onChange(opt.value)}
            title={opt.title}
            aria-label={opt.title}
            aria-pressed={value === opt.value}
          >
            {opt.label}
          </button>
        ))}
      </div>
    </div>
  );
}

/**
 * FeedbackControls — 봇 응답 하단의 피드백 컨트롤
 *   - 👍/👎 응답 평가 (response_rating)
 *   - 감지 감정 칩 클릭 → 7감정 드롭다운으로 셀프 정정 (emotion_correction)
 * @param {object}   msg        - 대상 봇 메시지 (utteranceId, emotion, feedback 포함)
 * @param {Function} onFeedback - (messageId, utteranceId, kind, value) 콜백
 */
function FeedbackControls({ msg, onFeedback }) {
  const [emotionMenuOpen, setEmotionMenuOpen] = useState(false);
  const [pendingKind, setPendingKind] = useState(null);

  const rating = msg.feedback?.rating || null;
  const correctedEmotion = msg.feedback?.correctedEmotion || null;
  // 정정된 감정이 있으면 정정값을, 없으면 모델 감지 감정을 표시
  const displayedEmotion = correctedEmotion || msg.emotion;

  /**
   * handleRate — 👍/👎 클릭 처리 (같은 값 재클릭은 무시)
   * @param {string} value - "good" | "bad"
   * @returns {void}
   */
  const handleRate = async (value) => {
    if (rating === value || pendingKind) return;
    setPendingKind("response_rating");
    try {
      await onFeedback(msg.id, msg.utteranceId, "response_rating", value, {
        feedback: msg.feedback,
        emotion: msg.emotion,
      });
    } finally {
      setPendingKind(null);
    }
  };

  /**
   * handleCorrectEmotion — 감정 정정 선택 처리
   * @param {string} label - 7감정 한국어 라벨
   * @returns {void}
   */
  const handleCorrectEmotion = async (label) => {
    setEmotionMenuOpen(false);
    if (label === displayedEmotion || pendingKind) return;
    setPendingKind("emotion_correction");
    try {
      await onFeedback(msg.id, msg.utteranceId, "emotion_correction", label, {
        feedback: msg.feedback,
        emotion: msg.emotion,
      });
    } finally {
      setPendingKind(null);
    }
  };

  return (
    <div className="feedback-row">
      {/* 응답 평가 버튼 */}
      <button
        className={`feedback-thumb ${rating === "good" ? "feedback-selected" : ""}`}
        onClick={() => handleRate("good")}
        disabled={Boolean(pendingKind)}
        title="이 답변이 도움됐어요"
        aria-label="도움됐어요"
      >
        👍
      </button>
      <button
        className={`feedback-thumb ${rating === "bad" ? "feedback-selected" : ""}`}
        onClick={() => handleRate("bad")}
        disabled={Boolean(pendingKind)}
        title="이 답변은 별로예요"
        aria-label="별로예요"
      >
        👎
      </button>

      {/* 감정 칩 + 정정 드롭다운 */}
      {displayedEmotion && (
        <span className="feedback-emotion-wrap">
          <button
            className={`bubble-emotion bubble-emotion-btn ${correctedEmotion ? "emotion-corrected" : ""}`}
            onClick={() => setEmotionMenuOpen((open) => !open)}
            disabled={Boolean(pendingKind)}
            title="감지된 감정이 다르면 눌러서 정정할 수 있어요"
          >
            {EMOTION_MAP[displayedEmotion] || displayedEmotion}
            {correctedEmotion ? " ✓" : " ▾"}
          </button>
          {emotionMenuOpen && (
            <span className="emotion-menu" role="menu">
              <span className="emotion-menu-title">실제 감정 선택</span>
              {EMOTION_CHOICES.map((label) => (
                <button
                  key={label}
                  className={`emotion-menu-item ${label === displayedEmotion ? "emotion-menu-current" : ""}`}
                  onClick={() => handleCorrectEmotion(label)}
                  role="menuitem"
                >
                  {label}
                </button>
              ))}
            </span>
          )}
        </span>
      )}
    </div>
  );
}

/**
 * MessageBubble — 단일 메시지 버블
 * @param {object}   msg        - 메시지 객체
 * @param {Function} onFeedback - 피드백 전송 콜백 (봇 메시지 전용, 선택)
 * @param {string}   userAvatar - 사용자 아바타 이모티콘 (""=숨김)
 * @param {string}   botAvatar  - 챗봇 아바타 이모티콘 (""=숨김)
 */
function MessageBubble({ msg, onFeedback, userAvatar, botAvatar }) {
  const isUser = msg.role === "user";
  // utteranceId가 있는 봇 메시지에만 피드백 컨트롤을 노출한다
  const canFeedback = !isUser && msg.utteranceId != null && typeof onFeedback === "function";

  return (
    <div className={`bubble-row ${isUser ? "bubble-user" : "bubble-bot"}`}>
      {!isUser && botAvatar && (
        <div className="bot-avatar" aria-label="챗봇">{botAvatar}</div>
      )}
      <div className={`bubble ${isUser ? "bubble-user-body" : "bubble-bot-body"}`}>
        <p className="bubble-text">{msg.content}</p>
        {canFeedback ? (
          <FeedbackControls msg={msg} onFeedback={onFeedback} />
        ) : (
          msg.emotion && (
            <span className="bubble-emotion">
              {EMOTION_MAP[msg.emotion] || msg.emotion}
            </span>
          )
        )}
        <span className="bubble-time">{msg.time}</span>
      </div>
      {isUser && userAvatar && (
        <div className="user-avatar" aria-label="사용자">{userAvatar}</div>
      )}
    </div>
  );
}

/**
 * CrisisBanner — 위기 감지 시 경고 배너
 */
function CrisisBanner({ message, onDismiss }) {
  return (
    <div className="crisis-banner" role="alert">
      <span className="crisis-banner-icon">🚨</span>
      <p className="crisis-banner-text">{message}</p>
      <button className="crisis-banner-close" onClick={onDismiss} aria-label="닫기">
        ✕
      </button>
    </div>
  );
}

/**
 * ChatWindow — 채팅 메인 컴포넌트
 * @param {string}   username      - 현재 사용자 이름
 * @param {number}   dayRefreshKey - 하루 마감 뒤 서버 날짜 재조회 트리거
 * @param {Function} onCloseDay    - 하루 마감 버튼 클릭 콜백
 */
export default function ChatWindow({ username, dayRefreshKey = 0, onCloseDay }) {
  // ── 상태 ────────────────────────────────────────────────────────────────────
  const [messages, setMessages]         = useState([]);
  const [input, setInput]               = useState("");
  const [loading, setLoading]           = useState(false);
  const [wellnessScore, setWellnessScore] = useState(65);
  const [wellnessLabel, setWellnessLabel] = useState("보통");
  const [currentEmotion, setCurrentEmotion] = useState("");
  const [crisisMsg, setCrisisMsg]       = useState(null);
  const [errorMsg, setErrorMsg]         = useState(null);
  const [activeDate, setActiveDate]     = useState("");
  const [canAdvanceDay, setCanAdvanceDay] = useState(false);
  const [advanceLoading, setAdvanceLoading] = useState(false);
  const [resetLoading, setResetLoading] = useState(false);
  const [advanceNotice, setAdvanceNotice]   = useState(null);
  // 오늘의 추천(행동 추천 v1) — /chat 응답의 recommendations, 실시간만(DB 저장 없음)
  const [recommendations, setRecommendations] = useState([]);
  // 아바타 이모티콘 (계정별 localStorage 저장, ""=숨김)
  const [userAvatar, setUserAvatar] = useState(DEFAULT_USER_AVATAR);
  const [botAvatar, setBotAvatar]   = useState(DEFAULT_BOT_AVATAR);

  const messagesEndRef = useRef(null);
  const inputRef       = useRef(null);
  const clientSessionRef = useRef(createClientSessionId());

  // ── 메시지 목록 자동 스크롤 ─────────────────────────────────────────────────
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // ── 계정 전환 시 저장된 아바타 설정 복원 ────────────────────────────────────
  useEffect(() => {
    if (!username) return;
    const saved = loadAvatars(username);
    setUserAvatar(saved.user);
    setBotAvatar(saved.bot);
  }, [username]);

  // ── 첫 진입 시 안내 메시지 + 오늘 대화 복원 ────────────────────────────────
  useEffect(() => {
    if (username) {
      // 화면 채팅창이 새로 열릴 때마다 Qwen 문맥을 새로 시작한다.
      // (대화 "표시"는 복원하지만, Qwen 입력 문맥은 새 화면 기준으로 다시 쌓인다)
      clientSessionRef.current = createClientSessionId();
      setActiveDate("");
      setCanAdvanceDay(false);
      setResetLoading(false);
      setAdvanceNotice(null);
      setCrisisMsg(null);
      setErrorMsg(null);
      setRecommendations([]);

      const now = new Date().toLocaleTimeString("ko-KR", {
        hour: "2-digit", minute: "2-digit",
      });
      setMessages([
        {
          id: "intro",
          role: "bot",
          content: `안녕하세요, ${username}님! 오늘 어떻게 지내고 계신가요? 편하게 이야기해 주세요.`,
          time: now,
        },
      ]);

      let cancelled = false;

      /**
       * restoreToday — 활성 날짜 조회 후 오늘 대화/점수 패널을 DB 기준으로 복원
       * @returns {Promise<void>}
       */
      const restoreToday = async () => {
        // 1) 활성 날짜 + 관리자 여부 + 자동 마감 결과 조회
        try {
          const dayData = await getCurrentDay(username);
          if (cancelled) return;
          setActiveDate(dayData.current_date || "");
          setCanAdvanceDay(Boolean(dayData.is_developer));
          if (dayData.auto_closed_dates?.length) {
            setAdvanceNotice(
              `지난 ${dayData.auto_closed_dates.join(", ")} 기록을 자동으로 마감해 캘린더에 반영했어요.`
            );
          }
        } catch (err) {
          console.error("getCurrentDay 오류:", err);
        }

        // 2) 오늘 저장된 대화가 있으면 화면에 복원
        try {
          const data = await getDayUtterances(username);
          if (cancelled || !data.utterances?.length) return;

          const restored = [];
          let lastUserUtt = null;
          data.utterances.forEach((utt) => {
            const time = formatUtcTime(utt.created_at);
            if (utt.role === "user") {
              lastUserUtt = utt;
              restored.push({
                id: `db-${utt.id}`,
                role: "user",
                content: utt.text,
                time,
              });
            } else {
              // 봇 응답에는 직전 사용자 발화의 감정/피드백을 연결해
              // 실시간 대화와 같은 형태로 피드백 컨트롤을 복원한다.
              restored.push({
                id: `db-${utt.id}`,
                role: "bot",
                content: utt.text,
                emotion: lastUserUtt?.emotion || null,
                utteranceId: lastUserUtt?.id ?? null,
                feedback: {
                  rating: lastUserUtt?.feedback?.response_rating || null,
                  correctedEmotion: lastUserUtt?.feedback?.corrected_emotion || null,
                },
                time,
              });
            }
          });

          const restoredNow = new Date().toLocaleTimeString("ko-KR", {
            hour: "2-digit", minute: "2-digit",
          });
          setMessages([
            {
              id: "restore-notice",
              role: "bot",
              content: `오늘 나눈 대화 ${restored.filter((m) => m.role === "user").length}개를 불러왔어요. 이어서 편하게 이야기해 주세요.`,
              time: restoredNow,
            },
            ...restored,
          ]);

          // 실시간 점수 패널도 서버 계산값으로 복원
          if (typeof data.wellness_score === "number") {
            setWellnessScore(data.wellness_score);
            setWellnessLabel(data.label || "보통");
          }
          const lastEmotion = lastUserUtt?.feedback?.corrected_emotion || lastUserUtt?.emotion;
          if (lastEmotion) setCurrentEmotion(lastEmotion);
        } catch (err) {
          console.error("getDayUtterances 오류:", err);
        }
      };

      restoreToday();

      return () => {
        cancelled = true;
      };
    }
  }, [dayRefreshKey, username]);

  // ── 메시지 전송 처리 ─────────────────────────────────────────────────────────
  const handleSend = useCallback(async () => {
    const text = input.trim();
    if (!text || loading || advanceLoading || resetLoading) return;

    const now = new Date().toLocaleTimeString("ko-KR", {
      hour: "2-digit", minute: "2-digit",
    });

    // 사용자 메시지 즉시 추가
    const userMsg = {
      id: Date.now(),
      role: "user",
      content: text,
      time: now,
    };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setLoading(true);
    setErrorMsg(null);

    try {
      const res = await sendMessage(username, text, clientSessionRef.current);

      const botNow = new Date().toLocaleTimeString("ko-KR", {
        hour: "2-digit", minute: "2-digit",
      });

      // 봇 응답 메시지 추가 — utteranceId는 응답 평가/감정 정정 피드백 전송 키
      const botMsg = {
        id: Date.now() + 1,
        role: "bot",
        content: res.response,
        emotion: res.top_emotion,
        utteranceId: res.utterance_id ?? null,
        feedback: { rating: null, correctedEmotion: null },
        time: botNow,
      };
      setMessages((prev) => [...prev, botMsg]);

      // 점수 패널 업데이트
      setWellnessScore(res.wellness_score ?? 65);
      setWellnessLabel(res.label ?? "보통");
      setCurrentEmotion(res.top_emotion ?? "");
      // 오늘의 추천 갱신 (없으면 카드가 기본 체크인을 표시)
      setRecommendations(res.recommendations || []);

      // 위기 감지 처리
      if (res.is_crisis && res.crisis_message) {
        setCrisisMsg(res.crisis_message);
      }
    } catch (err) {
      console.error("sendMessage 오류:", err);
      setErrorMsg("서버 연결에 실패했습니다. 잠시 후 다시 시도해 주세요.");
    } finally {
      setLoading(false);
      inputRef.current?.focus();
    }
  }, [advanceLoading, input, loading, resetLoading, username]);

  // ── 다음날 전환 처리 ──────────────────────────────────────────────────────
  const handleAdvanceDay = useCallback(async () => {
    if (advanceLoading || loading || resetLoading) return;
    if (!canAdvanceDay) {
      setErrorMsg("관리자 계정에서만 다음 날짜 전환을 사용할 수 있습니다.");
      return;
    }

    setAdvanceLoading(true);
    setErrorMsg(null);
    setCrisisMsg(null);

    try {
      const data = await advanceDay(username);
      const nextDate = data.current_date || "";
      const prevDate = data.previous_date || activeDate || "현재 날짜";
      const now = new Date().toLocaleTimeString("ko-KR", {
        hour: "2-digit", minute: "2-digit",
      });

      // 날짜가 바뀌면 화면 문맥과 실시간 점수 패널을 새 하루 기준으로 초기화한다.
      clientSessionRef.current = createClientSessionId();
      setActiveDate(nextDate);
      setWellnessScore(65);
      setWellnessLabel("보통");
      setCurrentEmotion("");
      setRecommendations([]);
      setMessages([
        {
          id: `intro-${nextDate || Date.now()}`,
          role: "bot",
          content: `${nextDate} 새 날로 넘어왔어요. 오늘 기록도 편하게 이어서 남겨 주세요.`,
          time: now,
        },
      ]);
      setAdvanceNotice(`${prevDate} 기록을 저장하고 ${nextDate}로 넘어왔어요.`);
    } catch (err) {
      console.error("advanceDay 오류:", err);
      const detail = err.response?.data?.detail;
      setErrorMsg(detail || "다음 날짜로 전환하지 못했습니다. 서버 상태를 확인해 주세요.");
    } finally {
      setAdvanceLoading(false);
      inputRef.current?.focus();
    }
  }, [activeDate, advanceLoading, canAdvanceDay, loading, resetLoading, username]);

  // ── 관리자 DB 초기화 처리 ──────────────────────────────────────────────────
  const handleResetDb = useCallback(async () => {
    if (advanceLoading || loading || resetLoading) return;
    if (!canAdvanceDay) {
      setErrorMsg("관리자 계정에서만 DB 초기화를 사용할 수 있습니다.");
      return;
    }

    const confirmed = window.confirm(
      "현재 계정의 대화, 캘린더, 위기/감사 기록이 삭제됩니다. 다른 계정과 로그인 계정은 유지됩니다. 계속할까요?"
    );
    if (!confirmed) return;

    setResetLoading(true);
    setErrorMsg(null);
    setCrisisMsg(null);

    try {
      const data = await resetDatabase(username, "RESET");
      const currentDate = data.current_date || "";
      const deletedTotal = Object.values(data.deleted || {}).reduce(
        (sum, value) => sum + Number(value || 0),
        0
      );
      const now = new Date().toLocaleTimeString("ko-KR", {
        hour: "2-digit", minute: "2-digit",
      });

      // DB 초기화 뒤에는 화면 문맥과 실시간 점수 패널도 새 기록 기준으로 맞춘다.
      clientSessionRef.current = createClientSessionId();
      setActiveDate(currentDate);
      setWellnessScore(65);
      setWellnessLabel("보통");
      setCurrentEmotion("");
      setRecommendations([]);
      setMessages([
        {
          id: `reset-${Date.now()}`,
          role: "bot",
          content: "현재 계정의 DB 기록을 초기화했어요. 계정은 유지했고, 오늘부터 새 기록으로 다시 시작합니다.",
          time: now,
        },
      ]);
      setAdvanceNotice(`현재 계정 DB 초기화 완료: ${deletedTotal}개 기록을 삭제했습니다.`);
    } catch (err) {
      console.error("resetDatabase 오류:", err);
      const detail = err.response?.data?.detail;
      setErrorMsg(detail || "DB 초기화에 실패했습니다. 서버 상태를 확인해 주세요.");
    } finally {
      setResetLoading(false);
      inputRef.current?.focus();
    }
  }, [advanceLoading, canAdvanceDay, loading, resetLoading, username]);

  // ── 피드백 전송 처리 (응답 평가 / 감정 셀프 정정) ──────────────────────────
  const handleFeedback = useCallback(async (
    messageId,
    utteranceId,
    kind,
    value,
    previousState,
  ) => {
    if (utteranceId == null) return;

    // 낙관적 UI 갱신 — 실패 시 이전 피드백/감정 표시로 되돌려 재시도를 허용한다.
    setMessages((prev) =>
      prev.map((m) => {
        if (m.id !== messageId) return m;
        const feedback = { ...(m.feedback || {}) };
        if (kind === "response_rating") feedback.rating = value;
        else feedback.correctedEmotion = value;
        return { ...m, feedback };
      })
    );
    // 감정 정정은 우측 패널의 현재 감정 표시도 함께 맞춘다
    if (kind === "emotion_correction") {
      setCurrentEmotion(value);
    }

    try {
      await sendFeedback(username, utteranceId, kind, value);
    } catch (err) {
      console.error("sendFeedback 오류:", err);
      const previousFeedback = previousState?.feedback || {};
      setMessages((prev) =>
        prev.map((m) => {
          if (m.id !== messageId) return m;
          return { ...m, feedback: { ...previousFeedback } };
        })
      );
      if (kind === "emotion_correction") {
        setCurrentEmotion(
          previousFeedback.correctedEmotion || previousState?.emotion || ""
        );
      }
      setErrorMsg("피드백 저장에 실패했습니다. 잠시 후 다시 시도해 주세요.");
    }
  }, [username]);

  // ── Enter 키 전송 ─────────────────────────────────────────────────────────
  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // ── 아바타 선택 처리 (선택 즉시 계정별 localStorage 저장) ────────────────────
  const selectUserAvatar = (value) => {
    setUserAvatar(value);
    saveAvatars(username, { user: value, bot: botAvatar });
  };
  const selectBotAvatar = (value) => {
    setBotAvatar(value);
    saveAvatars(username, { user: userAvatar, bot: value });
  };

  // ── 렌더링 ───────────────────────────────────────────────────────────────────
  const emotionLabel = EMOTION_MAP[currentEmotion] || currentEmotion;

  return (
    <div className="chat-layout">
      {/* 왼쪽: 채팅 영역 */}
      <div className="chat-area">
        {/* 위기 배너 */}
        {crisisMsg && (
          <CrisisBanner
            message={crisisMsg}
            onDismiss={() => setCrisisMsg(null)}
          />
        )}

        {/* 오류 메시지 */}
        {errorMsg && (
          <div className="error-banner">
            ⚠️ {errorMsg}
            <button onClick={() => setErrorMsg(null)}>✕</button>
          </div>
        )}

        {/* 날짜 전환 완료 안내 */}
        {advanceNotice && (
          <div className="advance-banner">
            <span>{advanceNotice}</span>
            <button onClick={() => setAdvanceNotice(null)} aria-label="닫기">✕</button>
          </div>
        )}

        {/* 메시지 목록 */}
        <div className="messages-list">
          {messages.map((msg) => (
            <MessageBubble
              key={msg.id}
              msg={msg}
              onFeedback={handleFeedback}
              userAvatar={userAvatar}
              botAvatar={botAvatar}
            />
          ))}
          {loading && (
            <div className="bubble-row bubble-bot">
              {botAvatar && <div className="bot-avatar">{botAvatar}</div>}
              <div className="bubble bubble-bot-body typing-indicator">
                <span /><span /><span />
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* 입력창 */}
        <div className="input-area">
          <textarea
            ref={inputRef}
            className="chat-input"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="메시지를 입력하세요... (Enter로 전송)"
            rows={2}
            disabled={loading || advanceLoading || resetLoading}
          />
          <button
            className="send-btn"
            onClick={handleSend}
            disabled={loading || advanceLoading || resetLoading || !input.trim()}
            aria-label="전송"
          >
            {loading ? "⏳" : "▶"}
          </button>
        </div>
      </div>

      {/* 오른쪽: 상태 패널 */}
      <aside className="status-panel">
        <div className="status-card">
          <h3 className="status-title">오늘의 웰니스</h3>
          {/* 비관리자는 숫자 점수 없이 색상 상태(라벨·색 띠)만, 관리자는 점수까지 표시 */}
          <WellnessBar score={wellnessScore} label={wellnessLabel} showScore={canAdvanceDay} />
          <div className="status-detail">
            {activeDate && (
              <div className="status-row">
                <span className="status-key">날짜</span>
                <span className="status-val active-date-pill">{activeDate}</span>
              </div>
            )}
            {canAdvanceDay && (
              <div className="status-row">
                <span className="status-key">점수</span>
                <span className="status-val">{Math.round(wellnessScore)} / 100</span>
              </div>
            )}
            {emotionLabel && (
              <div className="status-row">
                <span className="status-key">감정</span>
                <span className="status-val">{emotionLabel}</span>
              </div>
            )}
          </div>
        </div>

        {/* 오늘의 추천 — 웰니스 패널 아래 별도 카드(행동 추천 v1) */}
        <RecommendationCard recommendations={recommendations} />

        {/* 아바타 이모티콘 선택 — 나/챗봇 각각, 계정별 localStorage 저장 */}
        <div className="status-card">
          <h3 className="status-title">아바타</h3>
          <AvatarPicker label="나" value={userAvatar} onChange={selectUserAvatar} />
          <AvatarPicker label="챗봇" value={botAvatar} onChange={selectBotAvatar} />
        </div>

        {canAdvanceDay && (
          <>
            {/* 안내 카드 */}
            <div className="status-card status-hint">
              <p>💡 관리자 점검은 <strong>다음날로 넘기기</strong>를 눌러 하루씩 저장하며 진행하세요.</p>
            </div>

            {/* 다음날 전환 버튼 */}
            <button
              className="advance-day-btn"
              onClick={handleAdvanceDay}
              disabled={loading || advanceLoading || resetLoading}
            >
              {advanceLoading ? "넘기는 중..." : "다음날로 넘기기"}
            </button>

            {/* DB 초기화 버튼 */}
            <button
              className="reset-db-btn"
              onClick={handleResetDb}
              disabled={loading || advanceLoading || resetLoading}
            >
              {resetLoading ? "초기화 중..." : "DB 초기화"}
            </button>
          </>
        )}

        {/* 하루 마감 버튼 */}
        <button
          className="close-day-btn"
          onClick={() => onCloseDay(activeDate)}
          disabled={messages.length <= 1 || advanceLoading || resetLoading}
        >
          📋 하루 마감
        </button>
      </aside>
    </div>
  );
}
