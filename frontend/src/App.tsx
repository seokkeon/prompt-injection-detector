import React, { useState } from "react";
import TextAnalyzer from "./components/TextAnalyzer";
import ImageAnalyzer from "./components/ImageAnalyzer";
import EmailAnalyzer from "./components/EmailAnalyzer";
import "./App.css";

type Tab = "text" | "image" | "email";

function App() {
  const [tab, setTab] = useState<Tab>("text");

  return (
    <div className="app">
      <header className="header">
        <div className="header-content">
          <div className="logo">
            <span className="logo-icon">🛡️</span>
            <div>
              <h1>Prompt Injection Detector</h1>
              <p>Detect hidden prompt injections in text, images, and emails</p>
            </div>
          </div>
          <a
            href="http://localhost:8000/docs"
            target="_blank"
            rel="noreferrer"
            className="api-link"
          >
            API Docs ↗
          </a>
        </div>
      </header>

      <main className="main">
        <div className="tabs">
          {(["text", "image", "email"] as Tab[]).map((t) => (
            <button
              key={t}
              className={`tab ${tab === t ? "tab-active" : ""}`}
              onClick={() => setTab(t)}
            >
              {t === "text" ? "📝 Text" : t === "image" ? "🖼️ Image" : "📧 Email"}
            </button>
          ))}
        </div>

        <div className="panel">
          {tab === "text"  && <TextAnalyzer />}
          {tab === "image" && <ImageAnalyzer />}
          {tab === "email" && <EmailAnalyzer />}
        </div>
      </main>
    </div>
  );
}

export default App;
