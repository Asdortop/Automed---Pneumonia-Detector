# AutoMed — Chest X-Ray Pneumonia Analyzer

End-to-end AI pipeline: **ResNet-50 classification** → **Grad-CAM heatmap** → **RAG clinical report**

---

## Project Structure

```
Automed---Pneumonia-Detector/
│
├── src/                             ← All core Python/AI code
│   ├── classifier.py                   ResNet-50 model architecture
│   ├── dataset.py                      PyTorch Dataset + DataLoaders
│   ├── preprocess.py                   Resize & normalize raw images
│   ├── train_classifier.py             Training loop
│   ├── gradcam.py                      Grad-CAM heatmap generator
│   ├── rag.py                          RAG pipeline (FAISS + Groq LLM)
│   └── report.py                       Report generator (template + RAG fallback)
│
├── backend/
│   └── main.py                         FastAPI server
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   ├── index.css
│   │   └── components/
│   │       ├── UploadPanel.jsx         X-ray upload UI
│   │       └── ResultPanel.jsx         Results + heatmap display
│   ├── package.json
│   └── vite.config.js
│
├── models/                          ← NOT in git — get from author
│   └── classifier.pth                  Trained ResNet-50 weights
│
├── data/                            ← NOT in git — get from author or rebuild
│   ├── faiss_index.faiss               FAISS vector index
│   └── faiss_index.pkl                 Chunk metadata
│
├── Dataset/                         ← NOT in git — download from Kaggle
│   └── chest_xray/train|val|test/      Raw chest X-ray images
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Quick Setup (Friend / Collaborator)

### 1. Clone the repo
```bash
git clone https://github.com/Asdortop/Automed---Pneumonia-Detector.git
cd Automed---Pneumonia-Detector
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```
> ⚠️ For GPU support, install PyTorch with CUDA from [pytorch.org](https://pytorch.org/get-started/locally/) before running the above.

### 3. Place model files (get these from the author via Google Drive)

| File | Place it at |
|---|---|
| `classifier.pth` | `models/classifier.pth` |
| `faiss_index.faiss` | `data/faiss_index.faiss` |
| `faiss_index.pkl` | `data/faiss_index.pkl` |

### 4. Run the app

**Terminal 1 — Backend**
```bash
# Optional: set Groq API key for LLM-enhanced reports
$env:GROQ_API_KEY = "gsk_..."

uvicorn backend.main:app --reload --port 8000
```

**Terminal 2 — Frontend**
```bash
cd frontend
npm install   # first time only
npm run dev
```

Open **http://localhost:5173** ✅

---

## Training from Scratch (Optional)

Only needed if you don't have the model files from the author:

```bash
# 1. Download dataset from Kaggle and place at Dataset/chest_xray/
#    https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

# 2. Preprocess images
python src/preprocess.py

# 3. Train ResNet-50 (~15 epochs, 30–60 min on CPU)
python src/train_classifier.py

# 4. Build FAISS index (optional, enables richer RAG reports)
python src/rag.py --build_index
```

---

## Pipeline Overview

```
Upload X-Ray
     │
     ▼
ResNet-50 Classifier  →  NORMAL / PNEUMONIA + confidence %
     │
     ▼
Grad-CAM              →  Heatmap overlay (highlights affected lobe)
     │
     ▼
report.py             →  Simple template (always works, no dependencies)
                      →  RAG via FAISS + Groq (if index available)
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
  "report":      "Dense consolidation detected in the lower right lobe...",
  "heatmap_b64": "<base64 PNG>"
}
```

### `GET /health`
```json
{ "status": "ok", "device": "cuda", "model": "loaded", "faiss": "available" }
```

API docs available at: **http://localhost:8000/docs**

---

## Notes
- **For educational use only** — not a clinical diagnostic tool.
- Reports use a rule-based template by default (works without any API key or FAISS index).
- To use RAG as primary report engine, set `USE_RAG_FIRST = True` in `src/report.py`.
