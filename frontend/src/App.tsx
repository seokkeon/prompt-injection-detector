import React, { useState } from "react";
import UnifiedAnalyzer from "./components/UnifiedAnalyzer";
import ImageAnalyzer from "./components/ImageAnalyzer";
import EmailAnalyzer from "./components/EmailAnalyzer";
import ErrorBoundary from "./components/ErrorBoundary";
import "./App.css";

type Tab = "unified" | "image" | "email";

const TABS: { id: Tab; label: string }[] = [
  { id: "unified", label: "🔍 Text & Analysis" },
  { id: "image",   label: "🖼️ Image" },
  { id: "email",   label: "📧 Email" },
];

export default function App() {
  const [tab, setTab] = useState<Tab>("unified");

  return (
    <ErrorBoundary>
      <div className="app">
        <header className="header">
          <div className="header-content">
            <div className="logo">
              <span className="logo-icon">🛡️</span>
              <div>
                <h1>Prompt Injection Detector</h1>
                <p>Detect hidden prompt injections in text, images, emails, URLs, and conversations</p>
              </div>
            </div>
            <a href="http://localhost:8000/docs" target="_blank" rel="noreferrer" className="api-link">
              API Docs ↗
            </a>
          </div>
        </header>

        <main className="main">
          <div className="tabs">
            {TABS.map(t => (
              <button
                key={t.id}
                className={`tab ${tab === t.id ? "tab-active" : ""}`}
                onClick={() => setTab(t.id)}
              >
                {t.label}
              </button>
            ))}
          </div>

          <div className="panel">
            {tab === "unified" && <UnifiedAnalyzer />}
            {tab === "image"   && <ImageAnalyzer />}
            {tab === "email"   && <EmailAnalyzer />}
          </div>
        </main>
      </div>
    </ErrorBoundary>
  );
}