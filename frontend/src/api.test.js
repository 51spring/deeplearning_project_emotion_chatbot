/**
 * api.test.js
 * 역할: 인증 토큰 만료 처리와 보호 API Bearer 헤더 전달 회귀 테스트
 * 입력: Jest 실행
 * 출력: assertion 성공/실패
 */
import axios from "axios";
import {
  clearAuthToken,
  getStoredAuthToken,
  registerUser,
  resetPassword,
  sendMessage,
  setAuthToken,
} from "./api";

jest.mock("axios");

beforeEach(() => {
  localStorage.clear();
  jest.clearAllMocks();
});

test("만료된 인증 토큰은 조회 시 저장소에서 제거한다", () => {
  setAuthToken("expired-token", Math.floor(Date.now() / 1000) - 1);

  expect(getStoredAuthToken()).toBeNull();
  expect(localStorage.getItem("ec_auth_token")).toBeNull();
  expect(localStorage.getItem("ec_auth_expires_at")).toBeNull();
});

test("채팅 요청에 유효한 Bearer 토큰을 전달한다", async () => {
  setAuthToken("valid-token", Math.floor(Date.now() / 1000) + 60);
  axios.post.mockResolvedValue({ data: { response: "응답" } });

  await sendMessage("tester", "안녕하세요", "client-session");

  expect(axios.post).toHaveBeenCalledWith(
    "/chat",
    {
      username: "tester",
      text: "안녕하세요",
      client_session_id: "client-session",
    },
    {
      headers: {
        Authorization: "Bearer valid-token",
      },
    }
  );
});

test("가입 요청에 닉네임과 이메일을 함께 전달한다", async () => {
  axios.post.mockResolvedValue({ data: { username: "tester" } });

  await registerUser("tester", "테스터", "tester@example.com", "pass1234");

  expect(axios.post).toHaveBeenCalledWith(
    "/auth/register",
    {
      username: "tester",
      nickname: "테스터",
      email: "tester@example.com",
      password: "pass1234",
    }
  );
});

test("비밀번호 재설정 요청은 아이디와 이메일, 새 비밀번호를 전달한다", async () => {
  axios.post.mockResolvedValue({ data: { reset: true } });

  await resetPassword("tester", "tester@example.com", "newpass99");

  expect(axios.post).toHaveBeenCalledWith(
    "/auth/reset-password",
    {
      username: "tester",
      email: "tester@example.com",
      new_password: "newpass99",
    }
  );
});

afterEach(() => {
  clearAuthToken();
});
