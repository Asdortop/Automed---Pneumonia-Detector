# AutoMed — Chest X-Ray Pneumonia Analyzer

End-to-end AI pipeline: **ResNet-50 classification** → **Grad-CAM heatmap** → **RAG clinical report**

---

## Project Structure

```
NNDL_CBP/
├── Dataset/chest_xray/       # Raw Kaggle dataset (train/val/test)
├── data/
│   ├── processed/            # Preprocessed 224×224 PNGs
│   ├── faiss_index.faiss     # FAISS vector index
│   └── faiss_index.pkl       # Chunk metadata
├── models/
│   └── classifier.pth        # Trained ResNet-50 checkpoint
├── outputs/
│   └── training_curves.png   # Loss & accuracy plots
├── src/
│   ├── preprocess.py         # Phase 1: resize + normalize images
│   ├── dataset.py            # Phase 1: PyTorch Dataset + DataLoaders
│   ├── classifier.py         # Phase 2: ResNet-50 model definition
│   ├── train_classifier.py   # Phase 2: training loop
│   ├── gradcam.py            # Phase 3: Grad-CAM heatmap
│   └── rag.py                # Phase 4: FAISS + Groq RAG pipeline
├── backend/
│   └── main.py               # Phase 5: FastAPI server
├── frontend/                 # Phase 6: React + Vite UI
└── requirements.txt
```

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Preprocess images *(already done)*
```bash
python src/preprocess.py
```

### 3. Train the classifier *(already done)*
```bash
python src/train_classifier.py
# Options: --epochs 15 --batch_size 32 --lr 3e-4 --patience 5 --workers 0
```

### 4. Build FAISS index *(already done)*
```bash
python src/rag.py --build_index
```

---

## Running the App

### Backend
```bash
# Optional: set Groq API key for LLM-generated reports
$env:GROQ_API_KEY = "gsk_..."

uvicorn backend.main:app --reload --port 8000
```
API docs available at: http://localhost:8000/docs

### Frontend
```bash
cd frontend
npm run dev
```
Open: http://localhost:5173

---

## Pipeline Overview

```
Upload X-Ray
     │
     ▼
ResNet-50 Classifier  →  NORMAL / PNEUMONIA + confidence
     │
     ▼
Grad-CAM              →  Heatmap overlay (affected lobe)
     │
     ▼
RAG Pipeline          →  FAISS retrieval + Groq LLM → Clinical report
     │
     ▼
FastAPI /analyze      →  JSON response → React UI
```

---

## API

### `POST /analyze`
Upload a chest X-ray (JPEG/PNG).

**Response:**
```json
{
  "label":       "PNEUMONIA",
  "probability": 94.3,
  "location":    "lower right lobe",
  "severity":    "high confidence, severe presentation",
  "report":      "Right lower lobe consolidation...",
  "heatmap_b64": "<base64 PNG>"
}
```

### `GET /health`
```json
{ "status": "ok", "device": "cpu", "model": "loaded", "faiss": "loaded" }
```

---

## Notes
- **For educational use only** — not a clinical diagnostic tool.
- Grad-CAM requires `enable_grad()` during inference; the backend handles this automatically.
- If no `GROQ_API_KEY` is set, a rule-based template report is generated instead.
