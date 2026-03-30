import { useState, useRef, useCallback } from "react";
import UploadPanel from "./components/UploadPanel.jsx";
import ResultPanel from "./components/ResultPanel.jsx";

const API_URL = "http://localhost:8000";

export default function App() {
  const [file, setFile]       = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult]   = useState(null);
  const [error, setError]     = useState(null);

  const handleFileSelect = useCallback((selected) => {
    setFile(selected);
    setPreview(URL.createObjectURL(selected));
    setResult(null);
    setError(null);
  }, []);

  const handleReset = useCallback(() => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
  }, []);

  const handleAnalyze = useCallback(async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const form = new FormData();
      form.append("file", file);

      const res = await fetch(`${API_URL}/analyze`, {
        method: "POST",
        body: form,
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `Server error (${res.status})`);
      }

      const data = await res.json();
      setResult(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [file]);

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="container">
          <div className="header-inner">
            <div className="header-logo">🫁</div>
            <div>
              <div className="header-title">AutoMed</div>
              <div className="header-subtitle">Chest X-Ray Pneumonia Analyzer</div>
            </div>
          </div>
        </div>
      </header>

      {/* Main */}
      <main className="main">
        <div className="container">
          <div className="page-grid">
            {/* Left — Upload */}
            <UploadPanel
              preview={preview}
              loading={loading}
              error={error}
              onFileSelect={handleFileSelect}
              onAnalyze={handleAnalyze}
              onReset={handleReset}
              hasFile={!!file}
            />

            {/* Right — Results */}
            <ResultPanel result={result} loading={loading} />
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="footer">
        <div className="container">
          AutoMed · ResNet-50 + Grad-CAM + RAG · For educational use only — not a clinical diagnostic tool.
        </div>
      </footer>
    </div>
  );
}
