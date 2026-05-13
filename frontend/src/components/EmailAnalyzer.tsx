import React, { useState, useRef } from "react";
import ResultCard from "./ResultCard";

const API = "http://localhost:8000";

export default function EmailAnalyzer() {
  const [file, setFile]       = useState<File | null>(null);
  const [result, setResult]   = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState("");
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = (f: File) => {
    setFile(f); setResult(null); setError("");
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault(); setDragOver(false);
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  };

  const analyze = async () => {
    if (!file) return;
    setLoading(true); setError(""); setResult(null);
    const form = new FormData();
    form.append("file", file);
    try {
      const res = await fetch(`${API}/analyze/email`, { method: "POST", body: form });
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
      <div className="section-title">Analyze Email</div>

      {file ? (
        <div className="file-selected">
          <span>📧 {file.name} ({(file.size / 1024).toFixed(1)} KB)</span>
          <button className="file-clear" onClick={() => { setFile(null); setResult(null); }}>✕</button>
        </div>
      ) : (
        <div
          className={`dropzone ${dragOver ? "drag-over" : ""}`}
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={onDrop}
          onClick={() => inputRef.current?.click()}
        >
          <div className="dropzone-icon">📧</div>
          <strong>Drop a .eml file here or click to browse</strong>
          <p>Raw email files (.eml) supported</p>
          <p style={{ marginTop: 8, fontSize: 12 }}>
            In Gmail: open email → ⋮ menu → "Download message"
          </p>
        </div>
      )}

      <input
        ref={inputRef} type="file" accept=".eml,message/rfc822" style={{ display: "none" }}
        onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
      />

      {file && (
        <button className="btn" onClick={analyze} disabled={loading}>
          {loading ? "Analyzing..." : "Analyze Email"}
        </button>
      )}

      {loading && <div className="loading"><div className="spinner" /> Analyzing email...</div>}
      {error  && <div className="error">⚠️ {error}</div>}

      {result && (
        <div>
          {/* Email metadata */}
          {(result.subject || result.sender) && (
            <div style={{ marginTop: 16, padding: "12px 16px", background: "#f8fafc", borderRadius: 8, fontSize: 13, color: "#4a5568" }}>
              {result.subject && <div><strong>Subject:</strong> {result.subject}</div>}
              {result.sender  && <div style={{ marginTop: 4 }}><strong>From:</strong> {result.sender}</div>}
            </div>
          )}
          <ResultCard result={result} />
        </div>
      )}
    </div>
  );
}
