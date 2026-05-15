import React, { useState } from "react";
import ResultCard from "./ResultCard";

const API = "http://localhost:8000";

type IndirectType = "url" | "html" | "text";

export default function IndirectAnalyzer() {
  const [type, setType] = useState<IndirectType>("url");
  const [input, setInput] = useState("");
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const analyze = async () => {
    if (!input.trim()) return;
    setLoading(true);
    setError("");
    setResult(null);

    try {
      const endpoint = `/analyze/indirect/${type}`;
      const body =
        type === "url"
          ? { url: input }
          : type === "html"
          ? { html: input }
          : { text: input };

      const res = await fetch(`${API}${endpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
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
      <div className="section-title">Analyze Indirect Injection</div>
      <div style={{ marginBottom: 12 }}>
        <p style={{ fontSize: 13, color: "#718096", marginBottom: 8 }}>
          Scan URLs, HTML, or documents for hidden injections
        </p>
      </div>

      {/* Type selector */}
      <div style={{ display: "flex", gap: 8, marginBottom: 12 }}>
        {(["url", "html", "text"] as IndirectType[]).map((t) => (
          <button
            key={t}
            onClick={() => {
              setType(t);
              setInput("");
              setResult(null);
            }}
            style={{
              padding: "6px 12px",
              fontSize: 13,
              border: type === t ? "2px solid #4299e1" : "1px solid #cbd5e0",
              borderRadius: 6,
              background: type === t ? "#ebf8ff" : "white",
              cursor: "pointer",
              color: type === t ? "#2c5aa0" : "#4a5568",
              fontWeight: type === t ? 600 : 400,
            }}
          >
            {t === "url" ? "🔗 URL" : t === "html" ? "📄 HTML" : "📋 Document"}
          </button>
        ))}
      </div>

      {/* Input */}
      <textarea
        value={input}
        onChange={(e) => setInput(e.target.value)}
        placeholder={
          type === "url"
            ? "https://example.com/document"
            : type === "html"
            ? "<html>...</html>"
            : "Paste document text or content..."
        }
        onKeyDown={(e) => {
          if (e.key === "Enter" && e.metaKey) analyze();
        }}
      />

      <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
        <button
          className="btn"
          onClick={analyze}
          disabled={loading || !input.trim()}
        >
          {loading ? "Analyzing..." : "Analyze"}
        </button>
        <span style={{ fontSize: 12, color: "#a0aec0" }}>or ⌘+Enter</span>
        {input && (
          <button
            onClick={() => {
              setInput("");
              setResult(null);
            }}
            style={{
              marginLeft: "auto",
              background: "none",
              border: "none",
              color: "#718096",
              cursor: "pointer",
              fontSize: 13,
            }}
          >
            Clear
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
