import { useRef, useState } from "react";

export default function UploadPanel({
  preview,
  loading,
  error,
  onFileSelect,
  onAnalyze,
  onReset,
  hasFile,
}) {
  const inputRef = useRef(null);
  const [dragging, setDragging] = useState(false);

  const handleDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files?.[0];
    if (f && (f.type === "image/jpeg" || f.type === "image/png")) {
      onFileSelect(f);
    }
  };

  const handleChange = (e) => {
    const f = e.target.files?.[0];
    if (f) onFileSelect(f);
  };

  return (
    <div className="card">
      <div className="card-header">
        <span className="card-icon">📤</span>
        <h2>Upload X-Ray Image</h2>
      </div>
      <div className="card-body">
        {/* Drop Zone */}
        <div
          className={`upload-zone ${dragging ? "drag-over" : ""}`}
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={handleDrop}
          onClick={() => !hasFile && inputRef.current?.click()}
        >
          <input
            ref={inputRef}
            type="file"
            accept="image/jpeg,image/png"
            onChange={handleChange}
            style={{ display: "none" }}
          />
          <span className="upload-icon">🩻</span>
          {hasFile ? (
            <p className="upload-text">Image selected — ready to analyze</p>
          ) : (
            <>
              <p className="upload-text">
                <strong>Click to browse</strong> or drag &amp; drop
              </p>
              <p className="upload-hint">JPEG or PNG, chest X-ray frontal view</p>
            </>
          )}
        </div>

        {/* Preview */}
        {preview && (
          <div className="preview-wrap">
            <span className="preview-label">Preview</span>
            <img src={preview} alt="X-ray preview" />
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="status-bar error">
            ⚠️ {error}
          </div>
        )}

        {/* Loading status */}
        {loading && (
          <div className="status-bar">
            <div className="spinner" /> Analyzing image…
          </div>
        )}

        {/* Actions */}
        <button
          id="analyze-btn"
          className="btn btn-primary"
          disabled={!hasFile || loading}
          onClick={onAnalyze}
        >
          {loading ? (
            <><div className="spinner" /> Analyzing…</>
          ) : (
            <>🔬 Analyze X-Ray</>
          )}
        </button>

        {hasFile && !loading && (
          <button
            id="reset-btn"
            className="btn btn-secondary"
            onClick={onReset}
          >
            ✕ Clear &amp; Reset
          </button>
        )}

        {/* Info note */}
        <p style={{ fontSize: "0.72rem", color: "var(--clr-subtle)", marginTop: "0.75rem", textAlign: "center" }}>
          Make sure the FastAPI backend is running on port 8000.
        </p>
      </div>
    </div>
  );
}
