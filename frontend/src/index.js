/**
 * index.js
 * 역할: React 앱 진입점 — root DOM에 App 컴포넌트 마운트
 */
import React from "react";
import ReactDOM from "react-dom/client";
import "./App.css";
import App from "./App";

const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
