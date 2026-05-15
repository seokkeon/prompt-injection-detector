import React, { useState } from "react";
import TextAnalyzer from "./components/TextAnalyzer";
import ImageAnalyzer from "./components/ImageAnalyzer";
import EmailAnalyzer from "./components/EmailAnalyzer";
import IndirectAnalyzer from "./components/IndirectAnalyzer";
import ConversationAnalyzer from "./components/ConversationAnalyzer";
import "./App.css";

type Tab = "text" | "image" | "email" | "indirect" | "conversation";

function App() {
  const [tab, setTab] = useState<Tab>("text");

  const tabLabels: Record<Tab, string> = {
    text: "📝 Text",
    image: "🖼️ Image",
    email: "📧 Email",
    indirect: "🔗 Indirect",
    conversation: "💬 Conversation",
  };

  return (
    <div className="app">
      <header className="header">
        <div className="header-content">
          <div className="logo">
            <span className="logo-icon">🛡️</span>
            <div>
              <h1>Prompt Injection Detector</h1>
              <p>Detect hidden prompt injections in text, images, emails, and conversations</p>
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
          {(["text", "image", "email", "indirect", "conversation"] as Tab[]).map((t) => (
            <button
              key={t}
              className={`tab ${tab === t ? "tab-active" : ""}`}
              onClick={() => setTab(t)}
            >
              {tabLabels[t]}
            </button>
          ))}
        </div>

        <div className="panel">
          {tab === "text"  && <TextAnalyzer />}
          {tab === "image" && <ImageAnalyzer />}
          {tab === "email" && <EmailAnalyzer />}
          {tab === "indirect" && <IndirectAnalyzer />}
          {tab === "conversation" && <ConversationAnalyzer />}
        </div>
      </main>
    </div>
  );
}

export default App;
