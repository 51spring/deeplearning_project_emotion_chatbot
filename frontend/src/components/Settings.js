/**
 * Settings.js
 * 역할: 설정 화면 컴포넌트 — 계정/데이터 권리 기능 모음
 *       - 비밀번호 변경 (현재 비밀번호 재확인)
 *       - 내 데이터 JSON 내보내기 (대화/점수/요약/위기/피드백)
 *       - 계정 삭제 (비밀번호 재확인 + 이중 확인, 관리자 계정은 차단)
 */
import React, { useState, useEffect } from "react";
import {
  changePassword,
  deleteAccount,
  exportUserData,
  getCurrentDay,
} from "../api";

/**
 * localDateKey — 브라우저 로컬 시간 기준 오늘 날짜 문자열 생성
 * @returns {string} YYYY-MM-DD
 */
function localDateKey() {
  const now = new Date();
  const year = now.getFullYear();
  const month = String(now.getMonth() + 1).padStart(2, "0");
  const day = String(now.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

/**
 * downloadJson — 객체를 JSON 파일로 브라우저 다운로드
 * @param {object} data - 다운로드할 데이터
 * @param {string} filename - 저장 파일명
 * @returns {void}
 */
function downloadJson(data, filename) {
  const blob = new Blob([JSON.stringify(data, null, 2)], {
    type: "application/json;charset=utf-8",
  });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  document.body.removeChild(anchor);
  URL.revokeObjectURL(url);
}

/**
 * PasswordChangeCard — 비밀번호 변경 폼 카드
 * @param {string} username - 현재 사용자 이름
 */
function PasswordChangeCard({ username }) {
  const [currentPw, setCurrentPw] = useState("");
  const [newPw, setNewPw]         = useState("");
  const [confirmPw, setConfirmPw] = useState("");
  const [message, setMessage]     = useState(null);   // {type: "ok"|"error", text}
  const [loading, setLoading]     = useState(false);

  /**
   * handleSubmit — 비밀번호 변경 요청 전송
   * @param {Event} e - 폼 제출 이벤트
   * @returns {Promise<void>}
   */
  const handleSubmit = async (e) => {
    e.preventDefault();
    if (loading) return;
    if (newPw !== confirmPw) {
      setMessage({ type: "error", text: "새 비밀번호 확인이 일치하지 않습니다." });
      return;
    }

    setLoading(true);
    setMessage(null);
    try {
      await changePassword(username, currentPw, newPw);
      setMessage({ type: "ok", text: "비밀번호를 변경했습니다. 다음 로그인부터 새 비밀번호를 사용하세요." });
      setCurrentPw("");
      setNewPw("");
      setConfirmPw("");
    } catch (err) {
      const detail = err.response?.data?.detail;
      setMessage({ type: "error", text: detail || "비밀번호 변경에 실패했습니다." });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="settings-card">
      <h3 className="settings-card-title">🔒 비밀번호 변경</h3>
      <p className="settings-card-desc">현재 비밀번호 확인 후 새 비밀번호로 교체합니다.</p>
      <form onSubmit={handleSubmit} className="settings-form">
        <input
          className="login-input"
          type="password"
          value={currentPw}
          onChange={(e) => setCurrentPw(e.target.value)}
          placeholder="현재 비밀번호"
          maxLength={72}
          autoComplete="current-password"
        />
        <input
          className="login-input"
          type="password"
          value={newPw}
          onChange={(e) => setNewPw(e.target.value)}
          placeholder="새 비밀번호"
          maxLength={72}
          autoComplete="new-password"
        />
        <input
          className="login-input"
          type="password"
          value={confirmPw}
          onChange={(e) => setConfirmPw(e.target.value)}
          placeholder="새 비밀번호 확인"
          maxLength={72}
          autoComplete="new-password"
        />
        {message && (
          <p className={message.type === "ok" ? "settings-msg-ok" : "settings-msg-error"}>
            {message.text}
          </p>
        )}
        <button
          className="settings-primary-btn"
          type="submit"
          disabled={loading || !currentPw || !newPw || !confirmPw}
        >
          {loading ? "변경 중..." : "비밀번호 변경"}
        </button>
      </form>
    </div>
  );
}

/**
 * DataExportCard — 내 데이터 JSON 내보내기 카드
 * @param {string} username - 현재 사용자 이름
 */
function DataExportCard({ username }) {
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState(null);

  /**
   * handleExport — 내보내기 요청 후 JSON 파일 다운로드
   * @returns {Promise<void>}
   */
  const handleExport = async () => {
    if (loading) return;
    setLoading(true);
    setMessage(null);
    try {
      const data = await exportUserData(username);
      const today = localDateKey();
      downloadJson(data, `emotion_chatbot_export_${username}_${today}.json`);
      setMessage({ type: "ok", text: "내보내기 파일을 다운로드했습니다. 대화 원문이 포함되니 보관에 주의하세요." });
    } catch (err) {
      const detail = err.response?.data?.detail;
      setMessage({ type: "error", text: detail || "데이터 내보내기에 실패했습니다." });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="settings-card">
      <h3 className="settings-card-title">📦 내 데이터 내보내기</h3>
      <p className="settings-card-desc">
        지금까지의 대화, 감정/웰니스 점수, 하루 요약, 위기 기록, 피드백을 JSON 파일 하나로 받습니다.
      </p>
      {message && (
        <p className={message.type === "ok" ? "settings-msg-ok" : "settings-msg-error"}>
          {message.text}
        </p>
      )}
      <button className="settings-primary-btn" onClick={handleExport} disabled={loading}>
        {loading ? "준비 중..." : "JSON으로 내보내기"}
      </button>
    </div>
  );
}

/**
 * AccountDeleteCard — 계정 삭제 카드 (위험 구역)
 * @param {string}   username - 현재 사용자 이름
 * @param {boolean}  isAdmin  - 관리자 계정 여부 (관리자는 삭제 불가)
 * @param {Function} onDeleted - 삭제 완료 후 로그아웃 콜백
 */
function AccountDeleteCard({ username, isAdmin, onDeleted }) {
  const [password, setPassword] = useState("");
  const [loading, setLoading]   = useState(false);
  const [message, setMessage]   = useState(null);

  /**
   * handleDelete — 이중 확인 후 계정 삭제 요청
   * @returns {Promise<void>}
   */
  const handleDelete = async () => {
    if (loading || !password) return;

    const confirmed = window.confirm(
      "계정과 모든 대화/점수/요약/위기 기록이 영구 삭제되며 복구할 수 없습니다. 정말 삭제할까요?"
    );
    if (!confirmed) return;

    setLoading(true);
    setMessage(null);
    try {
      await deleteAccount(username, password);
      window.alert("계정이 삭제되었습니다. 그동안 함께해 주셔서 감사합니다.");
      onDeleted();
    } catch (err) {
      const detail = err.response?.data?.detail;
      setMessage({ type: "error", text: detail || "계정 삭제에 실패했습니다." });
      setLoading(false);
    }
  };

  return (
    <div className="settings-card settings-danger-card">
      <h3 className="settings-card-title">🗑️ 계정 삭제</h3>
      {isAdmin ? (
        <p className="settings-card-desc">
          관리자 기본 계정(developer/root)은 서버가 자동 관리하므로 삭제할 수 없습니다.
          기록만 비우려면 채팅 화면의 <strong>DB 초기화</strong>를 사용하세요.
        </p>
      ) : (
        <>
          <p className="settings-card-desc">
            계정과 모든 기록(대화, 점수, 하루 요약, 위기/피드백 기록)을 영구 삭제합니다.
            삭제 전 필요하면 위에서 데이터를 먼저 내보내 두세요.
          </p>
          <input
            className="login-input"
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="비밀번호 확인"
            maxLength={72}
            autoComplete="current-password"
          />
          {message && <p className="settings-msg-error">{message.text}</p>}
          <button
            className="settings-danger-btn"
            onClick={handleDelete}
            disabled={loading || !password}
          >
            {loading ? "삭제 중..." : "계정 영구 삭제"}
          </button>
        </>
      )}
    </div>
  );
}

/**
 * Settings — 설정 탭 메인 컴포넌트
 * @param {string}   username - 현재 사용자 아이디
 * @param {string}   displayName - 화면 표시 닉네임
 * @param {Function} onLogout - 계정 삭제 후 로그아웃 콜백
 */
export default function Settings({ username, displayName, onLogout }) {
  const [isAdmin, setIsAdmin] = useState(false);

  // 관리자 여부 조회 — 관리자 계정은 삭제 카드 대신 안내를 보여준다
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

  return (
    <div className="settings-page">
      <h2 className="settings-title">설정</h2>
      <p className="settings-subtitle">
        <strong>{displayName || username}</strong> 계정의 보안과 데이터를 관리합니다.
      </p>

      <PasswordChangeCard username={username} />
      <DataExportCard username={username} />
      <AccountDeleteCard username={username} isAdmin={isAdmin} onDeleted={onLogout} />

      <p className="settings-footnote">
        이 서비스는 의료 진단이 아닌 개인 정서 모니터링 보조 도구입니다.
        데이터는 운영 서버의 로컬 DB에 저장됩니다.
      </p>
    </div>
  );
}
