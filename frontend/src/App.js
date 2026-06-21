/**
 * App.js
 * 역할: 최상위 앱 컴포넌트
 *       - 닉네임/이메일 가입, 아이디/비밀번호 로그인, 비밀번호 재설정 (첫 진입)
 *       - 좌측 아이콘 사이드바 네비게이션 (채팅/기록/리포트/설정)
 *       - 하루 마감 모달 (DailySummary) 관리
 *       - 서버 연결 상태 표시
 */
import React, { useState, useEffect, useCallback } from "react";
import ChatWindow   from "./components/ChatWindow";
import DailySummary from "./components/DailySummary";
import Calendar     from "./components/Calendar";
import WeeklyReport from "./components/WeeklyReport";
import Settings     from "./components/Settings";
import {
  checkHealth,
  clearAuthToken,
  getStoredAuthToken,
  loginUser,
  registerUser,
  resetPassword,
  setAuthToken,
} from "./api";

const TAB_CHAT     = "chat";
const TAB_CALENDAR = "calendar";
const TAB_REPORT   = "report";
const TAB_SETTINGS = "settings";
const EMAIL_PATTERN = /^[^\s@]+@(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+[A-Za-z]{2,63}$/;
const EMAIL_INPUT_PATTERN = "^[^\\s@]+@(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\\.)+[A-Za-z]{2,63}$";

/**
 * isValidEmail — 가입/재설정 화면에서 점 있는 도메인 이메일인지 확인
 * @param {string} email - 사용자가 입력한 이메일
 * @returns {boolean} 이메일 형식 통과 여부
 */
function isValidEmail(email) {
  return EMAIL_PATTERN.test(String(email || "").trim());
}

/* ── SVG 아이콘 ──────────────────────────────────────────────────────────── */
const IconChat = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none"
    stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
  </svg>
);

const IconCalendar = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none"
    stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <rect x="3" y="4" width="18" height="18" rx="2" ry="2"/>
    <line x1="16" y1="2" x2="16" y2="6"/>
    <line x1="8"  y1="2" x2="8"  y2="6"/>
    <line x1="3"  y1="10" x2="21" y2="10"/>
  </svg>
);

const IconReport = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none"
    stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <line x1="4"  y1="20" x2="4"  y2="10"/>
    <line x1="10" y1="20" x2="10" y2="4"/>
    <line x1="16" y1="20" x2="16" y2="14"/>
    <line x1="22" y1="20" x2="2"  y2="20"/>
  </svg>
);

const IconSettings = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none"
    stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="12" r="3"/>
    <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09a1.65 1.65 0 0 0-1-1.51 1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09a1.65 1.65 0 0 0 1.51-1 1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33h.01a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51h.01a1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82v.01a1.65 1.65 0 0 0 1.51 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/>
  </svg>
);

const IconLogout = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
    stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/>
    <polyline points="16 17 21 12 16 7"/>
    <line x1="21" y1="12" x2="9" y2="12"/>
  </svg>
);

/**
 * LoginScreen — 로그인/가입/비밀번호 재설정 화면
 */
function LoginScreen({ onLogin }) {
  const [username, setUsername] = useState("");
  const [nickname, setNickname] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [mode, setMode] = useState("login");
  const [message, setMessage] = useState(null);
  const [loading, setLoading] = useState(false);
  const isLogin = mode === "login";
  const isRegister = mode === "register";
  const isReset = mode === "reset";

  /**
   * handleSubmit — 로그인, 가입, 비밀번호 재설정 요청을 백엔드에 전송
   * @param {Event} e - 폼 제출 이벤트
   * @returns {Promise<void>}
   */
  const handleSubmit = async (e) => {
    e.preventDefault();
    const trimmed = username.trim();
    const trimmedNickname = nickname.trim();
    const trimmedEmail = email.trim();
    if (!trimmed || !password || loading) return;
    if (isRegister && (!trimmedNickname || !trimmedEmail)) return;
    if (isReset && !trimmedEmail) return;
    if ((isRegister || isReset) && !isValidEmail(trimmedEmail)) {
      setMessage({
        type: "error",
        text: "이메일은 example@gmail.com처럼 도메인 점(.)과 끝 주소를 포함해야 합니다.",
      });
      return;
    }
    if (isReset && password !== confirmPassword) {
      setMessage({ type: "error", text: "새 비밀번호 확인이 일치하지 않습니다." });
      return;
    }

    setLoading(true);
    setMessage(null);
    try {
      if (isReset) {
        await resetPassword(trimmed, trimmedEmail, password);
        setMode("login");
        setPassword("");
        setConfirmPassword("");
        setMessage({ type: "ok", text: "비밀번호를 재설정했습니다. 새 비밀번호로 로그인하세요." });
        return;
      }

      const user = isLogin
        ? await loginUser(trimmed, password)
        : await registerUser(trimmed, trimmedNickname, trimmedEmail, password);
      onLogin(user);
    } catch (err) {
      const detail = err.response?.data?.detail;
      setMessage({ type: "error", text: detail || "계정 요청을 처리하지 못했습니다." });
    } finally {
      setLoading(false);
    }
  };

  /**
   * switchMode — 로그인/가입 화면 모드를 전환
   * @returns {void}
   */
  const switchMode = () => {
    setMode((prev) => (prev === "login" ? "register" : "login"));
    setMessage(null);
    setConfirmPassword("");
  };

  /**
   * switchResetMode — 비밀번호 재설정 화면으로 이동하거나 로그인 화면으로 복귀
   * @returns {void}
   */
  const switchResetMode = () => {
    setMode((prev) => (prev === "reset" ? "login" : "reset"));
    setMessage(null);
    setPassword("");
    setConfirmPassword("");
  };

  const primaryLabel = loading
    ? "처리 중..."
    : isLogin
      ? "로그인"
      : isRegister
        ? "가입하기"
        : "비밀번호 재설정";
  const submitDisabled = loading
    || !username.trim()
    || password.length < 4
    || (isRegister && (!nickname.trim() || !email.trim()))
    || (isReset && (!email.trim() || !confirmPassword));

  return (
    <div className="login-overlay">
      <div className="login-card">
        <h1 className="login-title">감정 상담 챗봇</h1>
        <p className="login-subtitle">
          매일 나누는 대화로 나의 감정을 기록하고<br />
          우울 경향을 조기에 발견해 보세요.
        </p>
        <form onSubmit={handleSubmit} className="login-form" noValidate>
          {isRegister && (
            <input
              className="login-input"
              type="text"
              value={nickname}
              onChange={(e) => setNickname(e.target.value)}
              placeholder="닉네임"
              maxLength={32}
              autoComplete="nickname"
            />
          )}
          <input
            className="login-input"
            type="text"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            placeholder="아이디"
            maxLength={32}
            autoComplete="username"
            autoFocus
          />
          {(isRegister || isReset) && (
            <input
              className="login-input"
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="이메일"
              maxLength={254}
              autoComplete="email"
              pattern={EMAIL_INPUT_PATTERN}
              title="example@gmail.com 형식으로 입력하세요."
            />
          )}
          <input
            className="login-input"
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder={isReset ? "새 비밀번호" : "비밀번호"}
            maxLength={72}
            autoComplete={isLogin ? "current-password" : "new-password"}
          />
          {isReset && (
            <input
              className="login-input"
              type="password"
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
              placeholder="새 비밀번호 확인"
              maxLength={72}
              autoComplete="new-password"
            />
          )}
          {message && (
            <p className={message.type === "ok" ? "login-success" : "login-error"}>
              {message.text}
            </p>
          )}
          <button
            className="login-btn"
            type="submit"
            disabled={submitDisabled}
          >
            {primaryLabel}
          </button>
          <div className="login-link-row">
            {!isReset && (
              <button
                className="login-link-btn"
                type="button"
                onClick={switchMode}
                disabled={loading}
              >
                {isLogin ? "새 계정 만들기" : "로그인으로 돌아가기"}
              </button>
            )}
            <button
              className="login-link-btn"
              type="button"
              onClick={switchResetMode}
              disabled={loading}
            >
              {isReset ? "로그인으로 돌아가기" : "비밀번호 재설정"}
            </button>
          </div>
        </form>
        <p className="login-disclaimer">
          이 서비스는 의료 진단이 아닌 개인 정서 모니터링 보조 도구입니다.
        </p>
      </div>
    </div>
  );
}

/**
 * Sidebar — 좌측 아이콘 네비게이션
 * @param {string}   activeTab      - 현재 활성 탭
 * @param {Function} onTabChange    - 탭 전환 콜백
 * @param {boolean|null} serverOnline - 서버 연결 상태
 * @param {string}   username       - 현재 사용자 아이디
 * @param {string}   displayName    - 화면 표시 닉네임
 * @param {Function} onLogout       - 로그아웃 콜백
 */
function Sidebar({ activeTab, onTabChange, serverOnline, username, displayName, onLogout }) {
  const shownName = displayName || username;
  const initial = shownName ? shownName.charAt(0).toUpperCase() : "?";

  return (
    <aside className="sidebar">
      {/* 탭 네비게이션 */}
      <nav className="sidebar-nav">
        <button
          className={`sidebar-btn ${activeTab === TAB_CHAT ? "sidebar-active" : ""}`}
          onClick={() => onTabChange(TAB_CHAT)}
          title="채팅"
          aria-label="채팅"
        >
          <IconChat />
          <span className="sidebar-label">채팅</span>
        </button>

        <button
          className={`sidebar-btn ${activeTab === TAB_CALENDAR ? "sidebar-active" : ""}`}
          onClick={() => onTabChange(TAB_CALENDAR)}
          title="기록"
          aria-label="기록"
        >
          <IconCalendar />
          <span className="sidebar-label">기록</span>
        </button>

        <button
          className={`sidebar-btn ${activeTab === TAB_REPORT ? "sidebar-active" : ""}`}
          onClick={() => onTabChange(TAB_REPORT)}
          title="리포트"
          aria-label="리포트"
        >
          <IconReport />
          <span className="sidebar-label">리포트</span>
        </button>

        <button
          className={`sidebar-btn ${activeTab === TAB_SETTINGS ? "sidebar-active" : ""}`}
          onClick={() => onTabChange(TAB_SETTINGS)}
          title="설정"
          aria-label="설정"
        >
          <IconSettings />
          <span className="sidebar-label">설정</span>
        </button>
      </nav>

      {/* 하단 영역: 서버 상태 + 사용자 */}
      <div className="sidebar-bottom">
        {/* 서버 상태 도트 */}
        <span
          className={`server-dot ${
            serverOnline === null ? "server-checking" :
            serverOnline ? "server-online" : "server-offline"
          }`}
          title={
            serverOnline === null ? "서버 연결 확인 중..." :
            serverOnline ? "서버 연결됨" : "서버 오프라인"
          }
        />

        {/* 사용자 아바타 + 로그아웃 */}
        <div className="sidebar-user-wrap">
          <div className="sidebar-avatar" title={`${shownName} (${username})`}>{initial}</div>
          <button
            className="sidebar-logout-btn"
            onClick={onLogout}
            title="로그아웃"
            aria-label="로그아웃"
          >
            <IconLogout />
          </button>
        </div>
      </div>
    </aside>
  );
}

/**
 * App — 루트 컴포넌트
 */
export default function App() {
  const [username,     setUsername]     = useState(null);
  const [nickname,     setNickname]     = useState(null);
  const [activeTab,    setActiveTab]    = useState(TAB_CHAT);
  const [summaryDate,  setSummaryDate]  = useState(null);
  const [dayRefreshKey, setDayRefreshKey] = useState(0);
  const [serverOnline, setServerOnline] = useState(null);

  /* 서버 헬스체크 */
  useEffect(() => {
    checkHealth()
      .then(() => setServerOnline(true))
      .catch(() => setServerOnline(false));
  }, []);

  /* 재방문 시 저장된 로그인 아이디, 닉네임, 토큰 복원 */
  useEffect(() => {
    const saved = localStorage.getItem("ec_auth_username");
    const savedNickname = localStorage.getItem("ec_auth_nickname");
    const token = getStoredAuthToken();
    if (saved && token) {
      setUsername(saved);
      setNickname(savedNickname || saved);
    } else {
      localStorage.removeItem("ec_auth_username");
      localStorage.removeItem("ec_auth_nickname");
      localStorage.removeItem("ec_username");
      clearAuthToken();
    }
  }, []);

  /**
   * handleLogin — 인증 성공 사용자를 앱 상태와 localStorage에 저장
   * @param {{username: string, nickname?: string, is_developer: boolean, access_token: string, expires_at: number}} user - 인증된 사용자
   * @returns {void}
   */
  const handleLogin = (user) => {
    const displayName = user.nickname || user.username;
    setUsername(user.username);
    setNickname(displayName);
    setAuthToken(user.access_token, user.expires_at);
    localStorage.setItem("ec_auth_username", user.username);
    localStorage.setItem("ec_auth_nickname", displayName);
    localStorage.removeItem("ec_username");
  };

  /**
   * handleLogout — 저장된 로그인 상태를 지우고 첫 화면으로 이동
   * @returns {void}
   */
  const handleLogout = () => {
    localStorage.removeItem("ec_auth_username");
    localStorage.removeItem("ec_auth_nickname");
    localStorage.removeItem("ec_username");
    clearAuthToken();
    setUsername(null);
    setNickname(null);
  };

  /**
   * handleGoCalendar — 하루 요약 모달을 닫고 기록 탭으로 이동
   * @returns {void}
   */
  const handleGoCalendar = () => {
    setSummaryDate(null);
    setActiveTab(TAB_CALENDAR);
  };

  /**
   * handleDayClosed — 하루 마감 완료 뒤 채팅 화면을 서버 활성 날짜 기준으로 갱신
   * @param {object} summary - /day/close 응답
   * @returns {void}
   */
  const handleDayClosed = useCallback((summary) => {
    if (summary?.current_date && summary.current_date !== summary.date) {
      setDayRefreshKey((value) => value + 1);
    }
  }, []);

  if (!username) return <LoginScreen onLogin={handleLogin} />;

  return (
    <div className="app-root">
      {/* 좌측 사이드바 */}
      <Sidebar
        activeTab={activeTab}
        onTabChange={setActiveTab}
        serverOnline={serverOnline}
        username={username}
        displayName={nickname || username}
        onLogout={handleLogout}
      />

      {/* 메인 컨텐츠 영역 */}
      <main className="app-main">
        <section
          className={`tab-panel ${activeTab === TAB_CHAT ? "tab-panel-active" : ""}`}
          aria-hidden={activeTab !== TAB_CHAT}
        >
          <ChatWindow
            username={username}
            dayRefreshKey={dayRefreshKey}
            onCloseDay={(date) => setSummaryDate(date)}
          />
        </section>
        <section
          className={`tab-panel ${activeTab === TAB_CALENDAR ? "tab-panel-active" : ""}`}
          aria-hidden={activeTab !== TAB_CALENDAR}
        >
          <Calendar
            username={username}
            isActive={activeTab === TAB_CALENDAR}
          />
        </section>
        <section
          className={`tab-panel ${activeTab === TAB_REPORT ? "tab-panel-active" : ""}`}
          aria-hidden={activeTab !== TAB_REPORT}
        >
          <WeeklyReport
            username={username}
            isActive={activeTab === TAB_REPORT}
          />
        </section>
        <section
          className={`tab-panel ${activeTab === TAB_SETTINGS ? "tab-panel-active" : ""}`}
          aria-hidden={activeTab !== TAB_SETTINGS}
        >
          <Settings
            username={username}
            displayName={nickname || username}
            onLogout={handleLogout}
          />
        </section>
      </main>

      {/* 하루 요약 모달 */}
      {summaryDate && (
        <DailySummary
          username={username}
          date={summaryDate}
          onClose={() => setSummaryDate(null)}
          onDayClosed={handleDayClosed}
          onGoCalendar={handleGoCalendar}
        />
      )}
    </div>
  );
}
