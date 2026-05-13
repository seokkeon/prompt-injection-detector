import React, { useState, useRef } from "react";
import ResultCard from "./ResultCard";

const API = "http://localhost:8000";

export default function ImageAnalyzer() {
  const [file, setFile]       = useState<File | null>(null);
  const [preview, setPreview] = useState<string>("");
  const [result, setResult]   = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState("");
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = (f: File) => {
    setFile(f);
    setResult(null);
    setError("");
    const reader = new FileReader();
    reader.onload = (e) => setPreview(e.target?.result as string);
    reader.readAsDataURL(f);
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault(); setDragOver(false);
    const f = e.dataTransfer.files[0];
    if (f && f.type.startsWith("image/")) handleFile(f);
    else setError("Please drop an image file (PNG, JPEG, GIF, WebP).");
  };

  const analyze = async () => {
    if (!file) return;
    setLoading(true); setError(""); setResult(null);
    const form = new FormData();
    form.append("file", file);
    try {
      const res = await fetch(`${API}/analyze/image`, { method: "POST", body: form });
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
      <div className="section-title">Analyze Image</div>

      {file ? (
        <div className="file-selected">
          <span>🖼️ {file.name} ({(file.size / 1024).toFixed(1)} KB)</span>
          <button className="file-clear" onClick={() => { setFile(null); setPreview(""); setResult(null); }}>✕</button>
        </div>
      ) : (
        <div
          className={`dropzone ${dragOver ? "drag-over" : ""}`}
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={onDrop}
          onClick={() => inputRef.current?.click()}
        >
          <div className="dropzone-icon">🖼️</div>
          <strong>Drop an image here or click to browse</strong>
          <p>PNG, JPEG, GIF, WebP supported</p>
        </div>
      )}

      <input
        ref={inputRef} type="file" accept="image/*" style={{ display: "none" }}
        onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
      />

      {preview && (
        <img src={preview} alt="preview" className="image-preview" style={{ display: "block", marginTop: 12 }} />
      )}

      {file && (
        <button className="btn" onClick={analyze} disabled={loading}>
          {loading ? "Analyzing..." : "Analyze Image"}
        </button>
      )}

      {loading && <div className="loading"><div className="spinner" /> Analyzing image...</div>}
      {error  && <div className="error">⚠️ {error}</div>}
      {result && <ResultCard result={result} />}
    </div>
  );
}
