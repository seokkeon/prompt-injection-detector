import React, { useState } from "react";
import ResultCard from "./ResultCard";

const API = "http://localhost:8000";

interface Message {
  role: "user" | "assistant";
  content: string;
}

export default function ConversationAnalyzer() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [newMessage, setNewMessage] = useState("");
  const [role, setRole] = useState<"user" | "assistant">("user");
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const addMessage = () => {
    if (!newMessage.trim()) return;
    setMessages([...messages, { role, content: newMessage }]);
    setNewMessage("");
  };

  const removeMessage = (index: number) => {
    setMessages(messages.filter((_, i) => i !== index));
  };

  const analyze = async () => {
    if (messages.length === 0) return;
    setLoading(true);
    setError("");
    setResult(null);

    try {
      const res = await fetch(`${API}/analyze/conversation`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages }),
      });
      if (!res.ok) throw new Error(`API error: ${res.status}`);
      setResult(await res.json());
    } catch (e: any) {
      setError(e.message || "Failed to reach API. Is the server running?");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <div className="section-title">Analyze Conversation</div>
      <div style={{ marginBottom: 12 }}>
        <p style={{ fontSize: 13, color: "#718096", marginBottom: 8 }}>
          Detect gradual jailbreaks and role drift across multi-turn chats
        </p>
      </div>

      {/* Message builder */}
      <div
        style={{
          border: "1px solid #e2e8f0",
          borderRadius: 8,
          padding: 12,
          marginBottom: 12,
          background: "#f7fafc",
        }}
      >
        <div style={{ marginBottom: 10, display: "flex", gap: 8 }}>
          <select
            value={role}
            onChange={(e) => setRole(e.target.value as "user" | "assistant")}
            style={{
              padding: "6px 8px",
              borderRadius: 4,
              border: "1px solid #cbd5e0",
              background: "white",
              cursor: "pointer",
              fontSize: 13,
            }}
          >
            <option value="user">👤 User</option>
            <option value="assistant">🤖 Assistant</option>
          </select>
        </div>

        <textarea
          value={newMessage}
          onChange={(e) => setNewMessage(e.target.value)}
          placeholder="Type a message..."
          style={{
            width: "100%",
            minHeight: 60,
            padding: 8,
            borderRadius: 4,
            border: "1px solid #cbd5e0",
            fontFamily: "monospace",
            fontSize: 12,
            marginBottom: 8,
            resize: "vertical",
          }}
          onKeyDown={(e) => {
            if (e.key === "Enter" && e.metaKey) addMessage();
          }}
        />

        <div style={{ display: "flex", gap: 8 }}>
          <button
            onClick={addMessage}
            disabled={!newMessage.trim()}
            style={{
              padding: "6px 12px",
              fontSize: 12,
              border: "1px solid #4299e1",
              borderRadius: 4,
              background: "#ebf8ff",
              color: "#2c5aa0",
              cursor: newMessage.trim() ? "pointer" : "not-allowed",
              fontWeight: 600,
            }}
          >
            Add Message
          </button>
          <span style={{ fontSize: 12, color: "#a0aec0" }}>⌘+Enter</span>
        </div>
      </div>

      {/* Messages list */}
      {messages.length > 0 && (
        <div style={{ marginBottom: 12 }}>
          <div style={{ fontSize: 12, fontWeight: 600, color: "#4a5568", marginBottom: 8 }}>
            Conversation ({messages.length} message{messages.length !== 1 ? "s" : ""})
          </div>
          <div
            style={{
              border: "1px solid #e2e8f0",
              borderRadius: 6,
              maxHeight: 300,
              overflowY: "auto",
            }}
          >
            {messages.map((msg, i) => (
              <div
                key={i}
                style={{
                  padding: 10,
                  borderBottom: i < messages.length - 1 ? "1px solid #edf2f7" : "none",
                  background: msg.role === "user" ? "white" : "#f7fafc",
                }}
              >
                <div
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "start",
                    gap: 8,
                  }}
                >
                  <div style={{ flex: 1 }}>
                    <div
                      style={{
                        fontSize: 11,
                        fontWeight: 600,
                        color: "#718096",
                        marginBottom: 4,
                      }}
                    >
                      {msg.role === "user" ? "👤 User" : "🤖 Assistant"}
                    </div>
                    <div style={{ fontSize: 12, color: "#2d3748", wordBreak: "break-word" }}>
                      {msg.content}
                    </div>
                  </div>
                  <button
                    onClick={() => removeMessage(i)}
                    style={{
                      background: "none",
                      border: "none",
                      color: "#cbd5e0",
                      cursor: "pointer",
                      fontSize: 16,
                      padding: 0,
                      marginTop: 2,
                    }}
                  >
                    ✕
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Analyze button */}
      <div style={{ display: "flex", gap: 12, marginBottom: 12 }}>
        <button
          className="btn"
          onClick={analyze}
          disabled={loading || messages.length === 0}
        >
          {loading ? "Analyzing..." : "Analyze Conversation"}
        </button>
        {messages.length > 0 && (
          <button
            onClick={() => {
              setMessages([]);
              setResult(null);
            }}
            style={{
              padding: "8px 16px",
              fontSize: 13,
              border: "1px solid #cbd5e0",
              borderRadius: 6,
              background: "white",
              cursor: "pointer",
              color: "#718096",
            }}
          >
            Clear All
          </button>
        )}
      </div>

      {loading && (
        <div className="loading">
          <div className="spinner" /> Analyzing...
        </div>
      )}
      {error && <div className="error">⚠️ {error}</div>}
      {result && <ResultCard result={result} />}
    </div>
  );
}
