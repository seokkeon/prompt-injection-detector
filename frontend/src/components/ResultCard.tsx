import React from "react";

interface Props {
  result: any;
}

export default function ResultCard({ result }: Props) {
  const level: string = result.risk_level || "low";
  const score: number = result.risk_score ?? 0;
  const pct = Math.round(score * 100);

  return (
    <div className="result">
      {/* Risk badge + score bar */}
      <div className="result-header">
        <span className={`badge badge-${level}`}>{level}</span>
        <div className="score-bar-wrap">
          <div className="score-label">Risk Score: {pct}%</div>
          <div className="score-bar">
            <div
              className={`score-fill fill-${level}`}
              style={{ width: `${pct}%` }}
            />
          </div>
        </div>
      </div>

      {/* Explanation */}
      {result.explanation && (
        <div className="explanation">{result.explanation}</div>
      )}

      {/* Rule score + ML score */}
      {(result.rule_score !== undefined || result.ml_score !== undefined) && (
        <div className="score-grid">
          {result.rule_score !== undefined && (
            <div className="score-card">
              <div className="score-card-label">Rule Score</div>
              <div className="score-card-value">
                {Math.round(result.rule_score * 100)}%
              </div>
            </div>
          )}
          {result.ml_score !== null && result.ml_score !== undefined && (
            <div className="score-card">
              <div className="score-card-label">ML Score</div>
              <div className="score-card-value">
                {Math.round(result.ml_score * 100)}%
              </div>
            </div>
          )}
        </div>
      )}

      {/* Categories detected */}
      {result.categories_detected?.length > 0 && (
        <div className="detail-section">
          <div className="detail-title">Categories</div>
          {result.categories_detected.map((c: string) => (
            <span key={c} className="category-chip">{c.replace(/_/g, " ")}</span>
          ))}
        </div>
      )}

      {/* Rule detail categories (from unified detector) */}
      {result.rule_detail?.categories_detected?.length > 0 && !result.categories_detected?.length && (
        <div className="detail-section">
          <div className="detail-title">Categories</div>
          {result.rule_detail.categories_detected.map((c: string) => (
            <span key={c} className="category-chip">{c.replace(/_/g, " ")}</span>
          ))}
        </div>
      )}

      {/* Flagged parts (email) */}
      {result.flagged_parts?.length > 0 && (
        <div className="detail-section">
          <div className="detail-title">Flagged Parts</div>
          {result.flagged_parts.map((p: string) => (
            <span key={p} className="pattern-chip">{p}</span>
          ))}
        </div>
      )}

      {/* Extracted text (image) — always shown */}
      {result.extracted_text !== undefined && (
        <div className="detail-section">
          <div className="detail-title">Extracted Text</div>
          {result.extracted_text ? (
            <div className="explanation" style={{ fontFamily: "monospace", fontSize: 12 }}>
              {result.extracted_text.slice(0, 500)}
              {result.extracted_text.length > 500 ? "..." : ""}
            </div>
          ) : (
            <div className="explanation" style={{ color: "#a0aec0", fontStyle: "italic" }}>
              No text detected in image.
            </div>
          )}
        </div>
      )}
    </div>
  );
}
