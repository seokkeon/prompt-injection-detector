import React, { useState } from "react";
import "./App.css";
import TextAnalyzer from "./components/TextAnalyzer";
import EmailAnalyzer from "./components/EmailAnalyzer";
import ImageAnalyzer from "./components/ImageAnalyzer";
import ConversationAnalyzer from "./components/ConversationAnalyzer";
import IndirectAnalyzer from "./components/IndirectAnalyzer";

type TabType = "text" | "email" | "image" | "conversation" | "indirect";

function App() {
  const [activeTab, setActiveTab] = useState<TabType>("text");

  return (
    <>
      <header className="header">
        <div className="header-content">
          <div className="logo">
            <span className="logo-icon">🛡️</span>
            <div>
              <h1>Prompt Injection Detector</h1>
              <p>Detect potential prompt injection attacks</p>
            </div>
          </div>
          <a
            href="http://localhost:5000"
            className="api-link"
            target="_blank"
            rel="noopener noreferrer"
          >
            API Docs
          </a>
        </div>
      </header>

      <main className="main">
        <div className="tabs">
          <button
            className={`tab ${activeTab === "text" ? "tab-active" : ""}`}
            onClick={() => setActiveTab("text")}
          >
            Text
          </button>
          <button
            className={`tab ${activeTab === "email" ? "tab-active" : ""}`}
            onClick={() => setActiveTab("email")}
          >
            Email
          </button>
          <button
            className={`tab ${activeTab === "image" ? "tab-active" : ""}`}
            onClick={() => setActiveTab("image")}
          >
            Image
          </button>
          <button
            className={`tab ${activeTab === "conversation" ? "tab-active" : ""}`}
            onClick={() => setActiveTab("conversation")}
          >
            Conversation
          </button>
          <button
            className={`tab ${activeTab === "indirect" ? "tab-active" : ""}`}
            onClick={() => setActiveTab("indirect")}
          >
            Indirect Attacks
          </button>
        </div>

        {activeTab === "text" && <TextAnalyzer />}
        {activeTab === "email" && <EmailAnalyzer />}
        {activeTab === "image" && <ImageAnalyzer />}
        {activeTab === "conversation" && <ConversationAnalyzer />}
        {activeTab === "indirect" && <IndirectAnalyzer />}
      </main>
    </>
  );
}

export default App;
