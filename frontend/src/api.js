/**
 * api.js
 * 역할: FastAPI 백엔드 연동 유틸리티
 *       - POST /chat          : 발화 전송 → 챗봇 응답 + 점수
 *       - POST /day/close     : 하루 종료 → EWMA 집계 + 요약
 *       - POST /day/advance   : 하루 저장 후 다음 날짜로 전환
 *       - GET  /day/current/{username} : 현재 활성 날짜 조회
 *       - GET  /day/utterances/{username} : 날짜별 대화 기록 조회 (복원/과거 보기)
 *       - POST /feedback      : 응답 평가 + 감정 셀프 정정 저장
 *       - POST /calendar/emotion-note : 캘린더 날짜별 수동 감정 기록 저장
 *       - DELETE /calendar/emotion-note/{username} : 수동 감정 기록 삭제
 *       - GET  /report/weekly/{username} : 주간 감정 리포트 조회
 *       - POST /auth/change-password : 비밀번호 변경
 *       - GET  /export/{username} : 개인 데이터 JSON 내보내기
 *       - POST /account/delete : 계정 및 데이터 완전 삭제
 *       - POST /admin/reset-db: 계정 보존형 런타임 DB 초기화
 *       - GET  /calendar/{username} : 캘린더 데이터 조회
 *       - POST /auth/register : 닉네임/아이디/이메일/비밀번호 가입
 *       - POST /auth/login    : 아이디/비밀번호 로그인
 *       - POST /auth/reset-password : 아이디/이메일 기반 비밀번호 재설정
 *       - GET  /health        : 서버 상태 확인
 */
import axios from "axios";

// 백엔드 베이스 URL (package.json proxy 설정으로 /api 불필요)
const BASE_URL = "";
const AUTH_TOKEN_KEY = "ec_auth_token";
const AUTH_EXPIRES_KEY = "ec_auth_expires_at";

/**
 * setAuthToken — 백엔드 Bearer 토큰을 브라우저 저장소에 보관
 * @param {string} token - access token
 * @param {number|null} expiresAt - 만료 epoch 초
 * @returns {void}
 */
export function setAuthToken(token, expiresAt = null) {
  if (token) {
    localStorage.setItem(AUTH_TOKEN_KEY, token);
  }
  if (expiresAt) {
    localStorage.setItem(AUTH_EXPIRES_KEY, String(expiresAt));
  }
}

/**
 * clearAuthToken — 저장된 Bearer 토큰을 제거
 * @returns {void}
 */
export function clearAuthToken() {
  localStorage.removeItem(AUTH_TOKEN_KEY);
  localStorage.removeItem(AUTH_EXPIRES_KEY);
}

/**
 * getStoredAuthToken — 현재 저장된 유효 토큰 조회
 * @returns {string|null}
 */
export function getStoredAuthToken() {
  const token = localStorage.getItem(AUTH_TOKEN_KEY);
  const expiresAt = Number(localStorage.getItem(AUTH_EXPIRES_KEY) || 0);
  if (!token) return null;
  if (expiresAt && expiresAt <= Math.floor(Date.now() / 1000)) {
    clearAuthToken();
    return null;
  }
  return token;
}

/**
 * authConfig — 보호 API 요청에 붙일 axios 인증 설정 생성
 * @returns {{headers: {Authorization?: string}}}
 */
function authConfig() {
  const token = getStoredAuthToken();
  return token ? { headers: { Authorization: `Bearer ${token}` } } : { headers: {} };
}

/**
 * registerUser — 닉네임/아이디/이메일/비밀번호 신규 가입
 * @param {string} username - 사용자 아이디
 * @param {string} nickname - 화면 표시 닉네임
 * @param {string} email - 비밀번호 재설정용 이메일
 * @param {string} password - 비밀번호
 * @returns {Promise<{ username: string, nickname: string, email: string, is_developer: boolean, access_token: string, expires_at: number }>}
 */
export async function registerUser(username, nickname, email, password) {
  const res = await axios.post(`${BASE_URL}/auth/register`, {
    username,
    nickname,
    email,
    password,
  });
  return res.data;
}

/**
 * loginUser — 아이디/비밀번호 로그인
 * @param {string} username - 사용자 아이디
 * @param {string} password - 비밀번호
 * @returns {Promise<{ username: string, is_developer: boolean, access_token: string, expires_at: number }>}
 */
export async function loginUser(username, password) {
  const res = await axios.post(`${BASE_URL}/auth/login`, {
    username,
    password,
  });
  return res.data;
}

/**
 * sendMessage — 채팅 메시지 전송
 * @param {string} username - 사용자 이름
 * @param {string} text     - 발화 내용
 * @param {string} clientSessionId - 화면 채팅창 단위 문맥 ID
 * @returns {Promise<{
 *   response: string,
 *   is_crisis: boolean,
 *   crisis_message: string|null,
 *   top_emotion: string,
 *   roberta_score: number,
 *   depression_score: number,
 *   depression_tendency_score: number,
 *   wellness_score: number,
 *   label: string,
 *   utterance_id: number|null,
 *   recommendations: Array<{id:string, title:string, message:string, reason:string, priority:string, category:string}>
 * }>}
 */
export async function sendMessage(username, text, clientSessionId) {
  const res = await axios.post(`${BASE_URL}/chat`, {
    username,
    text,
    client_session_id: clientSessionId,
  }, authConfig());
  return res.data;
}

/**
 * closeDay — 하루 종료 집계 요청
 * @param {string} username
 * @param {string|null} date - 화면에 표시 중인 마감 대상 날짜
 * @returns {Promise<{
 *   date: string,
 *   current_date: string,
 *   daily_score: number,
 *   smoothed_score: number,
 *   daily_wellness_score: number,
 *   daily_wellness_label: string,
 *   cumulative_wellness_score: number,
 *   cumulative_wellness_label: string,
 *   wellness_score: number,
 *   label: string,
 *   depression_tendency_daily: number|null,
 *   depression_tendency_smoothed: number|null,
 *   utterance_count: number,
 *   crisis_count: number
 * }>}
 */
export async function closeDay(username, date = null) {
  const res = await axios.post(`${BASE_URL}/day/close`, { username, date }, authConfig());
  return res.data;
}

/**
 * resetPassword — 아이디와 이메일 확인 후 새 비밀번호 설정
 * @param {string} username - 사용자 아이디
 * @param {string} email - 가입 이메일
 * @param {string} newPassword - 새 비밀번호
 * @returns {Promise<{ reset: boolean, username: string }>}
 */
export async function resetPassword(username, email, newPassword) {
  const res = await axios.post(`${BASE_URL}/auth/reset-password`, {
    username,
    email,
    new_password: newPassword,
  });
  return res.data;
}

/**
 * advanceDay — 현재 날짜 저장 후 다음 날짜로 전환
 * @param {string} username
 * @returns {Promise<{
 *   previous_date: string,
 *   current_date: string,
 *   closed_summary: object
 * }>}
 */
export async function advanceDay(username) {
  const res = await axios.post(`${BASE_URL}/day/advance`, { username }, authConfig());
  return res.data;
}

/**
 * resetDatabase — 관리자용 런타임 DB 초기화
 * @param {string} username - 관리자 사용자 이름
 * @param {string} confirmText - 서버 확인 문구
 * @returns {Promise<{
 *   reset: boolean,
 *   username: string,
 *   preserved_users: boolean,
 *   deleted: object,
 *   current_date: string
 * }>}
 */
export async function resetDatabase(username, confirmText = "RESET") {
  const res = await axios.post(`${BASE_URL}/admin/reset-db`, {
    username,
    confirm: confirmText,
  }, authConfig());
  return res.data;
}

/**
 * getCurrentDay — 현재 활성 날짜 조회
 * @param {string} username
 * @returns {Promise<{ current_date: string, is_developer: boolean }>}
 */
export async function getCurrentDay(username) {
  const res = await axios.get(
    `${BASE_URL}/day/current/${encodeURIComponent(username)}`,
    authConfig()
  );
  return res.data;
}

/**
 * getCalendar — 캘린더 데이터 조회 (최근 60일)
 * @param {string} username
 * @returns {Promise<Array<{
 *   date: string,
 *   daily_wellness_score: number,
 *   daily_wellness_label: string,
 *   cumulative_wellness_score: number,
 *   cumulative_wellness_label: string,
 *   wellness_score: number,
 *   label: string,
 *   depression_tendency_daily: number|null,
 *   depression_tendency_smoothed: number|null,
 *   crisis_count_day: number,
 *   utterance_count: number,
 *   summary_text: string,
 *   manual_emotion_label: string|null,
 *   manual_emotion_intensity: number|null,
 *   manual_emotion_note: string|null
 * }>>}
 */
export async function getCalendar(username, limit = 60) {
  const res = await axios.get(`${BASE_URL}/calendar/${encodeURIComponent(username)}`, {
    params: { limit },
    ...authConfig(),
  });
  return res.data;
}

/**
 * saveDailyEmotionNote — 캘린더 날짜별 사용자 수동 감정 기록 저장
 * @param {string} username - 사용자 이름
 * @param {string} dateStr - 기록 날짜 "YYYY-MM-DD"
 * @param {string} emotionLabel - 7감정 한국어 라벨
 * @param {number} intensity - 감정 강도(1~5)
 * @param {string} note - 선택 메모
 * @returns {Promise<{
 *   saved: boolean,
 *   date: string,
 *   manual_emotion_label: string,
 *   manual_emotion_intensity: number,
 *   manual_emotion_note: string|null,
 *   manual_emotion_updated_at: string|null
 * }>}
 */
export async function saveDailyEmotionNote(username, dateStr, emotionLabel, intensity, note = "") {
  const res = await axios.post(`${BASE_URL}/calendar/emotion-note`, {
    username,
    date: dateStr,
    emotion_label: emotionLabel,
    intensity,
    note,
  }, authConfig());
  return res.data;
}

/**
 * deleteDailyEmotionNote — 캘린더 날짜별 사용자 수동 감정 기록 삭제
 * @param {string} username - 사용자 이름
 * @param {string} dateStr - 삭제할 날짜 "YYYY-MM-DD"
 * @returns {Promise<{ deleted: boolean, date: string }>}
 */
export async function deleteDailyEmotionNote(username, dateStr) {
  const res = await axios.delete(
    `${BASE_URL}/calendar/emotion-note/${encodeURIComponent(username)}`,
    {
      params: { date: dateStr },
      ...authConfig(),
    }
  );
  return res.data;
}

/**
 * getDayUtterances — 날짜별 대화 기록 조회 (새로고침 복원 + 캘린더 과거 대화 보기)
 * @param {string} username - 사용자 이름
 * @param {string|null} dateStr - 조회 날짜 "YYYY-MM-DD" (생략 시 활성 날짜)
 * @returns {Promise<{
 *   date: string,
 *   is_active_date: boolean,
 *   utterances: Array<{
 *     id: number, role: string, text: string, emotion: string|null,
 *     is_crisis: boolean, created_at: string|null,
 *     feedback: { response_rating: string|null, corrected_emotion: string|null }
 *   }>,
 *   wellness_score?: number,
 *   label?: string
 * }>}
 */
export async function getDayUtterances(username, dateStr = null) {
  const res = await axios.get(
    `${BASE_URL}/day/utterances/${encodeURIComponent(username)}`,
    {
      params: dateStr ? { date: dateStr } : {},
      ...authConfig(),
    }
  );
  return res.data;
}

/**
 * sendFeedback — 챗봇 응답 평가 또는 감정 셀프 정정 저장
 * @param {string} username - 사용자 이름
 * @param {number} utteranceId - 교환 단위 사용자 발화 id
 * @param {string} kind - "response_rating" | "emotion_correction"
 * @param {string} value - "good"|"bad" 또는 7감정 한국어 라벨
 * @returns {Promise<{ saved: boolean, utterance_id: number, kind: string, value: string }>}
 */
export async function sendFeedback(username, utteranceId, kind, value) {
  const res = await axios.post(`${BASE_URL}/feedback`, {
    username,
    utterance_id: utteranceId,
    kind,
    value,
  }, authConfig());
  return res.data;
}

/**
 * getWeeklyReport — 주간 감정 리포트 조회
 * @param {string} username - 사용자 이름
 * @param {string|null} endDate - 주 마지막 날짜 "YYYY-MM-DD" (생략 시 활성 날짜)
 * @returns {Promise<{
 *   active_date: string,
 *   start_date: string,
 *   end_date: string,
 *   days: Array<object>,
 *   summary: object,
 *   weekly_summary: object,
 *   weekday_emotion_patterns: object
 * }>}
 */
export async function getWeeklyReport(username, endDate = null) {
  const res = await axios.get(
    `${BASE_URL}/report/weekly/${encodeURIComponent(username)}`,
    {
      params: endDate ? { end_date: endDate } : {},
      ...authConfig(),
    }
  );
  return res.data;
}

/**
 * changePassword — 비밀번호 변경 (현재 비밀번호 재확인 필수)
 * @param {string} username - 사용자 이름
 * @param {string} currentPassword - 현재 비밀번호
 * @param {string} newPassword - 새 비밀번호
 * @returns {Promise<{ changed: boolean, username: string }>}
 */
export async function changePassword(username, currentPassword, newPassword) {
  const res = await axios.post(`${BASE_URL}/auth/change-password`, {
    username,
    current_password: currentPassword,
    new_password: newPassword,
  }, authConfig());
  return res.data;
}

/**
 * exportUserData — 개인 데이터 JSON 내보내기 (대화/점수/요약/수동감정/위기/피드백)
 * @param {string} username - 사용자 이름
 * @returns {Promise<object>} 내보내기 JSON payload
 */
export async function exportUserData(username) {
  const res = await axios.get(
    `${BASE_URL}/export/${encodeURIComponent(username)}`,
    authConfig()
  );
  return res.data;
}

/**
 * deleteAccount — 계정 및 모든 기록 완전 삭제
 * @param {string} username - 사용자 이름
 * @param {string} password - 본인 확인 비밀번호
 * @returns {Promise<{ deleted: boolean, username: string, removed: object }>}
 */
export async function deleteAccount(username, password) {
  const res = await axios.post(`${BASE_URL}/account/delete`, {
    username,
    password,
    confirm: "DELETE",
  }, authConfig());
  return res.data;
}

/**
 * checkHealth — 서버 상태 확인
 * @returns {Promise<{ status: string }>}
 */
export async function checkHealth() {
  const res = await axios.get(`${BASE_URL}/health`);
  return res.data;
}
