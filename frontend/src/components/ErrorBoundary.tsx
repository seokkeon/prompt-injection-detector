import React, { Component, ErrorInfo, ReactNode } from "react";

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export default class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("ErrorBoundary caught:", error, info);
  }

  render() {
    if (this.state.hasError) {
      const msg = this.state.error?.message || "Unknown error";
      const isApi = msg.includes("fetch") || msg.includes("API") || msg.includes("network");
      return (
        <div style={{
          margin: "40px auto", maxWidth: 520, padding: "32px 28px",
          background: "white", borderRadius: 12, textAlign: "center",
          boxShadow: "0 2px 12px rgba(0,0,0,0.08)", border: "1px solid #fecaca",
        }}>
          <div style={{ fontSize: 40, marginBottom: 12 }}>
            {isApi ? "🔌" : "⚠️"}
          </div>
          <h2 style={{ color: "#991b1b", fontSize: 18, marginBottom: 8 }}>
            {isApi ? "API Unreachable" : "Something went wrong"}
          </h2>
          <p style={{ color: "#4a5568", fontSize: 14, marginBottom: 20, lineHeight: 1.6 }}>
            {isApi
              ? "Could not connect to the API server. Make sure it is running at http://localhost:8000"
              : msg}
          </p>
          {isApi && (
            <pre style={{
              background: "#f8fafc", border: "1px solid #e2e8f0", borderRadius: 6,
              padding: "10px 14px", fontSize: 12, textAlign: "left", marginBottom: 20,
              color: "#2d3748",
            }}>
              uvicorn api.main:app --reload
            </pre>
          )}
          <button
            onClick={() => this.setState({ hasError: false, error: null })}
            style={{
              padding: "9px 24px", background: "#1a1a2e", color: "white",
              border: "none", borderRadius: 8, cursor: "pointer", fontSize: 14,
            }}
          >
            Try again
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}
