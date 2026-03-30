"""
main.py — Phase 5: FastAPI Backend
Exposes a single /analyze endpoint that runs the full pipeline:
  image upload → ResNet-50 → Grad-CAM → RAG → JSON response

Run:
    uvicorn backend.main:app --reload --port 8000

Test:
    curl -X POST http://localhost:8000/analyze \
         -F "file=@path/to/xray.jpg"
"""

import base64
import io
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

# Add src/ to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from classifier import build_model
from gradcam    import GradCAM, analyze_image
import report

# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AutoMed API",
    description="Chest X-ray pneumonia detection with Grad-CAM and RAG report generation",
    version="1.0.0",
)

# Allow React frontend on localhost:3000
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://localhost:5174"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global model state (loaded once at startup) ───────────────────────────────
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL   = None
GRADCAM = None
# ─────────────────────────────────────────────────────────────────────────────


@app.on_event("startup")
async def load_models():
    """Load classifier + Grad-CAM once when the server starts."""
    global MODEL, GRADCAM

    base_dir  = Path(__file__).resolve().parent.parent
    ckpt_path = base_dir / "models" / "classifier.pth"

    if not ckpt_path.exists():
        print(f"⚠️  No classifier checkpoint at {ckpt_path}. Train first.")
        return

    print(f"  Loading model on {DEVICE}...")
    MODEL = build_model(DEVICE)
    ckpt  = torch.load(ckpt_path, map_location=DEVICE)
    MODEL.load_state_dict(ckpt["model_state_dict"])
    MODEL.eval()

    GRADCAM = GradCAM(MODEL, DEVICE)
    print("✅  AutoMed model loaded and ready.")

    # FAISS index is optional — report.py handles it gracefully if missing
    index_path = base_dir / "data" / "faiss_index.pkl"
    if index_path.exists():
        print("✅  FAISS index found — RAG fallback available.")
    else:
        print("ℹ️  No FAISS index — using simple template reports (run src/rag.py --build_index to enable RAG).")


@app.get("/health")
async def health():
    """Quick health check."""
    index_path = Path(__file__).resolve().parent.parent / "data" / "faiss_index.pkl"
    return {
        "status": "ok",
        "device": str(DEVICE),
        "model":  "loaded" if MODEL else "not loaded",
        "faiss":  "available" if index_path.exists() else "not built",
    }


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """
    Accepts a chest X-ray image (JPEG/PNG) and returns:
        - label       : "NORMAL" or "PNEUMONIA"
        - probability : confidence percentage
        - location    : affected lung region (from Grad-CAM)
        - report      : AI-generated clinical summary
        - heatmap_b64 : base64-encoded PNG heatmap overlay
    """
    if MODEL is None or GRADCAM is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run: python src/train_classifier.py first."
        )

    # ── Validate file type ────────────────────────────────────────────────
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Upload JPEG or PNG."
        )

    # ── Save to temp file (analyze_image needs a path) ────────────────────
    contents = await file.read()
    suffix   = ".jpg" if "jpeg" in file.content_type else ".png"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        # ── Run Grad-CAM pipeline ─────────────────────────────────────────
        result = analyze_image(
            image_path=tmp_path,
            model=MODEL,
            gradcam=GRADCAM,
            device=DEVICE,
        )

        # ── Generate report (simple template first, RAG if available) ─────
        api_key     = os.environ.get("GROQ_API_KEY", "")
        report_text = report.generate(
            label       = result["label"],
            probability = result["probability"],
            location    = result["location"],
            severity    = result["severity"],
            query       = result["query"],
            api_key     = api_key,
        )

        # ── Encode heatmap overlay to base64 for JSON transport ───────────
        overlay_pil    = Image.fromarray(result["overlay"])
        buf            = io.BytesIO()
        overlay_pil.save(buf, format="PNG")
        heatmap_b64    = base64.b64encode(buf.getvalue()).decode("utf-8")

        return JSONResponse({
            "label":        result["label"],
            "probability":  result["probability"],
            "location":     result["location"],
            "severity":     result["severity"],
            "query":        result["query"],
            "report":       report_text,
            "heatmap_b64":  heatmap_b64,
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Clean up temp file
        Path(tmp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
