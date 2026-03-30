# AutoMed — Project Overview & Technical Reference

> **Chest X-Ray Pneumonia Detection using Deep Learning**  
> Course: Neural Networks & Deep Learning (NNDL) | Course-Based Project

---

## 1. What Is This Project?

AutoMed is an end-to-end AI-powered medical imaging tool that:
1. **Accepts** a chest X-ray image (JPEG/PNG)
2. **Classifies** it as **NORMAL** or **PNEUMONIA** using a fine-tuned ResNet-50 deep learning model
3. **Visualizes** *why* the model made its decision using **Grad-CAM** heatmaps (highlights the affected lung region)
4. **Generates** a plain-English clinical summary using a **RAG pipeline** (Retrieval-Augmented Generation with an LLM)
5. **Serves** everything through a **FastAPI** backend and a **React** web frontend

This is a full-stack ML project — it covers deep learning, computer vision explainability, NLP/RAG, and web development.

---

## 2. Dataset

- **Source:** [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) — Kaggle
- **Classes:** `NORMAL` (healthy) and `PNEUMONIA` (bacterial or viral)
- **Splits:** Train / Validation / Test
- **Size:** ~5,216 images total
- **Class imbalance:** ~3:1 PNEUMONIA:NORMAL (handled via `WeightedRandomSampler` and `pos_weight` in loss)

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         React Frontend                              │
│   Upload X-Ray → [Analyze Button] → Show heatmap + report         │
└────────────────────────────┬────────────────────────────────────────┘
                             │ HTTP POST /analyze
┌────────────────────────────▼────────────────────────────────────────┐
│                       FastAPI Backend                               │
│    Receives image → runs pipeline → returns JSON response           │
└──────┬───────────────────────┬──────────────────────────────────────┘
       │                       │
┌──────▼──────┐        ┌───────▼──────────────────────────────────────┐
│  ResNet-50  │        │              RAG Pipeline                    │
│ Classifier  │        │  FAISS vector search → top-3 radiology chunks│
│ (Phase 2)   │        │  → Groq LLM (Llama 3) → clinical report     │
└──────┬──────┘        └──────────────────────────────────────────────┘
       │
┌──────▼──────┐
│  Grad-CAM   │
│  (Phase 3)  │
│  Heatmap    │
└─────────────┘
```

---

## 4. Phases Breakdown

### Phase 1 — Preprocessing (`src/preprocess.py`, `src/dataset.py`)
- All X-rays resized to **224×224 px** (required by ResNet-50)
- Converted to RGB (handles grayscale X-rays)
- Normalized to `[0, 1]` float32 range, saved as lossless PNG
- PyTorch `Dataset` class with ImageNet normalization transforms
- Training split uses `WeightedRandomSampler` to handle class imbalance

### Phase 2 — ResNet-50 Classifier (`src/classifier.py`, `src/train_classifier.py`)
- **Base model:** ResNet-50 pretrained on ImageNet (`torchvision`)
- **Frozen layers:** `conv1`, `bn1`, `layer1`, `layer2`, `layer3` (feature extraction)
- **Trainable:** `layer4` + custom head
- **Custom head:** `Linear(2048→512) → BatchNorm → ReLU → Dropout(0.4) → Linear(512→1)`
- **Loss:** `BCEWithLogitsLoss` with `pos_weight` for class imbalance
- **Optimizer:** AdamW, LR = 3e-4, Cosine Annealing schedule
- **Early stopping:** patience = 5 epochs on val loss
- **Result:** ~87% weighted F1 on test set, 100% val accuracy

### Phase 3 — Grad-CAM Explainability (`src/gradcam.py`)
- Hooks into `layer4` (last convolutional block of ResNet-50)
- Computes gradient-weighted class activation map
- Overlays a colored heatmap on the original X-ray
- **Quadrant analysis:** divides the 224×224 map into 4 regions to report *which lobe* is affected (upper/lower left/right)

### Phase 4 — RAG Report Generation (`src/rag.py`)
- **Knowledge base:** 40 hand-curated radiology report excerpts covering different pneumonia types, locations, severities, and treatment guidelines
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (384-dim)
- **Vector store:** FAISS with cosine similarity (`IndexFlatIP` on normalized vectors)
- **LLM:** Groq API (Llama 3 8B) — free tier; falls back to template if no API key
- **Flow:** classifier result → query string → FAISS top-3 retrieval → LLM prompt → report

### Phase 5 — FastAPI Backend (`backend/main.py`)
- Single endpoint: `POST /analyze` — accepts image, returns JSON
- `GET /health` — returns model load status
- Models loaded **once at startup** (not per-request) for performance
- CORS enabled for localhost dev
- Heatmap returned as **base64-encoded PNG** in JSON

### Phase 6 — React Frontend (`frontend/`)
- Built with **Vite + React**
- Drag-and-drop image upload with preview
- Diagnosis badge (NORMAL ✅ / PNEUMONIA ⚠️) with confidence bar
- Grad-CAM heatmap display with explanation caption
- Typewriter-style animated clinical report display

---

## 5. File Structure

```
NNDL_CBP/
├── Dataset/
│   └── chest_xray/          # Raw Kaggle dataset (not committed to git)
│       ├── train/
│       ├── val/
│       └── test/
├── data/
│   ├── processed/           # Preprocessed 224x224 PNGs
│   ├── faiss_index.faiss    # FAISS vector index (binary)
│   └── faiss_index.pkl      # Chunk metadata
├── models/
│   └── classifier.pth       # Trained ResNet-50 checkpoint
├── outputs/
│   └── training_curves.png  # Loss & accuracy plots
├── src/
│   ├── preprocess.py        # Phase 1
│   ├── dataset.py           # Phase 1
│   ├── classifier.py        # Phase 2
│   ├── train_classifier.py  # Phase 2
│   ├── gradcam.py           # Phase 3
│   └── rag.py               # Phase 4
├── backend/
│   └── main.py              # Phase 5 — FastAPI server
├── frontend/                # Phase 6 — React + Vite
│   ├── src/
│   │   ├── App.jsx
│   │   ├── index.css
│   │   └── components/
│   │       ├── UploadPanel.jsx
│   │       └── ResultPanel.jsx
│   └── package.json
├── requirements.txt
└── README.md
```

---

## 6. Key Design Decisions

| Decision | Reason |
|---|---|
| ResNet-50 over training from scratch | Transfer learning from ImageNet gives strong low-level feature extraction; much less data needed |
| Freeze layers 1-3, fine-tune layer4 | Layers 1-3 detect generic edges/textures; layer4 detects task-specific patterns (opacities, consolidations) |
| BCEWithLogitsLoss + pos_weight | More numerically stable than BCE + Sigmoid; pos_weight corrects class imbalance |
| Grad-CAM on layer4 | Last conv layer has best spatial resolution vs. semantic richness tradeoff |
| FAISS + sentence-transformers | Lightweight, runs locally, no API needed for retrieval |
| Groq API for LLM | Free tier, fast inference (~1s), no local GPU required for text generation |
| FastAPI over Flask | Async, automatic OpenAPI docs, better for production-grade ML serving |

---

## 7. API Reference

### `POST /analyze`
Upload a chest X-ray image.

**Request:** `multipart/form-data` with field `file` (JPEG or PNG)

**Response:**
```json
{
  "label":       "PNEUMONIA",
  "probability": 94.3,
  "location":    "lower right lobe",
  "severity":    "high confidence, severe presentation",
  "query":       "PNEUMONIA detected, lower right lobe, high confidence, severe presentation",
  "report":      "Right lower lobe consolidation with air bronchograms...",
  "heatmap_b64": "<base64-encoded PNG string>"
}
```

### `GET /health`
```json
{ "status": "ok", "device": "cuda", "model": "loaded", "faiss": "loaded" }
```

---

## 8. How to Run

### Prerequisites
- Python 3.10+ with pip
- Node.js 18+ with npm
- (Recommended) NVIDIA GPU with CUDA for faster training

---

### Step 1 — Install Python dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Preprocess images *(skip if `data/processed/` already exists)*
```bash
python src/preprocess.py
```

### Step 3 — Train the classifier *(skip if `models/classifier.pth` already exists)*
```bash
python src/train_classifier.py
# ~5 min on GPU, ~30 min on CPU
# Flags: --epochs 15  --batch_size 32  --lr 3e-4  --patience 5  --workers 0
```

### Step 4 — Build the FAISS index *(skip if `data/faiss_index.pkl` already exists)*
```bash
python src/rag.py --build_index
```

### Step 5 — Start the backend
```bash
# PowerShell (Windows)
$env:GROQ_API_KEY = "gsk_..."    # optional — for LLM reports; get free key at console.groq.com
uvicorn backend.main:app --reload --port 8000
```
```bash
# bash (Linux/Mac)
export GROQ_API_KEY="gsk_..."
uvicorn backend.main:app --reload --port 8000
```
> Backend runs at: http://localhost:8000  
> API docs at: http://localhost:8000/docs

### Step 6 — Start the frontend
```bash
cd frontend
npm install       # first time only
npm run dev
```
> Frontend runs at: http://localhost:5173 (or 5174 if 5173 is busy)

### Step 7 — Test it
Upload any `.jpeg` from `Dataset/chest_xray/test/PNEUMONIA/` or `Dataset/chest_xray/test/NORMAL/` in the UI.

---

## 9. Limitations & Disclaimer

- **Not a clinical tool.** This system is built for educational purposes as part of an NNDL course project. It must not be used for real medical diagnosis.
- The model was trained on a single public Kaggle dataset and may not generalize to X-rays from different machines or patient populations.
- Grad-CAM highlights are approximate and should be interpreted with caution.
- The RAG-generated reports are not written by a licensed radiologist.
