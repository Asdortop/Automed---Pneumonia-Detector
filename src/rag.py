"""
rag.py — Phase 4
RAG-based clinical report generator.

Pipeline:
  1. Build a FAISS index from curated pneumonia / radiology text chunks
  2. At inference: convert classifier output → query string
  3. Search FAISS → retrieve top-k text chunks
  4. Send retrieved context + query to Groq LLM → clinical report

Setup:
    Set your Groq API key (free at console.groq.com):
    export GROQ_API_KEY="gsk_..."    (Linux/Mac)
    $env:GROQ_API_KEY="gsk_..."     (PowerShell)

Usage:
    python src/rag.py --build_index          # one-time: build + save FAISS index
    python src/rag.py --query "Pneumonia, lower right lobe, high confidence"
"""

import os
import argparse
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
DATA_DIR    = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
INDEX_PATH  = DATA_DIR / "faiss_index.pkl"   # saved FAISS index + chunks
DATA_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)
# ─────────────────────────────────────────────────────────────────────────────


# ── Curated Medical Knowledge Base ───────────────────────────────────────────
# 40 carefully written radiology report excerpts covering:
#   - bacterial pneumonia, viral pneumonia, normal findings
#   - different lobes, severity levels, clinical recommendations
KNOWLEDGE_BASE = [
    # --- PNEUMONIA: Lower Right Lobe ---
    "Right lower lobe consolidation with increased opacity. Air bronchograms visible. "
    "Findings consistent with bacterial pneumonia. Right costophrenic angle blunting "
    "suggests associated pleural effusion. Recommend antibiotic therapy.",

    "Homogeneous opacity in the right lower lobe extending to the costophrenic angle. "
    "Clinical presentation and imaging findings are consistent with community-acquired "
    "pneumonia. Follow-up imaging in 4-6 weeks post-treatment is advised.",

    "Dense consolidation in the right lower lobe with air bronchogram sign. "
    "No pneumothorax. Heart size normal. Consistent with lobar pneumonia."
    " Sputum culture recommended to identify causative organism.",

    # --- PNEUMONIA: Lower Left Lobe ---
    "Left lower lobe consolidation with obscured hemidiaphragm. Patchy opacification "
    "extending from the hilum. Findings suggest bacterial pneumonia. "
    "No pleural effusion on this side. Systemic antibiotics indicated.",

    "Retrocardiac opacity in the left lower lobe consistent with pneumonia. "
    "Left hemidiaphragm partially obscured. SpO2 monitoring recommended. "
    "Clinical correlation with fever, cough, and elevated WBC advised.",

    # --- PNEUMONIA: Upper Right Lobe ---
    "Right upper lobe opacity with peripheral consolidation. No cavitation identified. "
    "Pattern is consistent with viral or atypical pneumonia. Consider Mycoplasma or "
    "Legionella serology. Macrolide antibiotics may be appropriate.",

    "Focal consolidation in the right upper lobe without volume loss. "
    "Findings are concerning for pneumonia versus early malignancy. "
    "CT chest with contrast recommended for further characterization.",

    # --- PNEUMONIA: Upper Left Lobe ---
    "Left upper lobe patchy consolidation with ground-glass opacity. "
    "Distribution suggests viral etiology. No hilar lymphadenopathy. "
    "Supportive care and antiviral therapy may be appropriate.",

    "Segmental consolidation involving the left upper lobe. "
    "Mild peribronchial thickening noted. Findings are consistent with "
    "community-acquired pneumonia. Outpatient antibiotic therapy appropriate if stable.",

    # --- BILATERAL PNEUMONIA ---
    "Bilateral lower lobe consolidations, right greater than left. "
    "Perihilar distribution suggests atypical or aspiration pneumonia. "
    "Heart size mildly enlarged. Pulmonary edema cannot be excluded.",

    "Bilateral patchy opacities with lower lobe predominance. "
    "No pleural effusion. Pattern is consistent with viral pneumonitis or "
    "COVID-19 related pneumonia. High-resolution CT recommended.",

    "Bilateral perihilar infiltrates with air bronchograms. "
    "Consistent with severe community-acquired pneumonia. "
    "Intensive care unit monitoring and broad-spectrum antibiotics indicated.",

    # --- HIGH CONFIDENCE SEVERE PNEUMONIA ---
    "Dense lobar consolidation consistent with severe bacterial pneumonia. "
    "Significant parapneumonic effusion present. Thoracentesis may be required "
    "if effusion is large or patient is febrile. IV antibiotics recommended.",

    "Extensive right-sided consolidation consistent with severe lobar pneumonia. "
    "Significant respiratory compromise expected. Hospitalization and IV piperacillin-"
    "tazobactam or ceftriaxone therapy recommended. Monitor for empyema.",

    "Widespread consolidation with air bronchograms and early cavitation. "
    "Findings suggest necrotizing pneumonia. Blood cultures and bronchoscopy "
    "with BAL recommended. Aggressive antibiotic coverage required.",

    # --- MODERATE CONFIDENCE ---
    "Subtle opacity in the right lower lobe. Cannot exclude early consolidation. "
    "Findings are borderline for pneumonia versus atelectasis. "
    "Clinical correlation with fever and inflammatory markers recommended.",

    "Mild perihilar haziness with early consolidative changes. "
    "May represent early pneumonia or pulmonary edema. "
    "Follow-up chest X-ray in 24-48 hours if symptoms persist.",

    "Patchy subsegmental opacity in the left lower lobe. In a febrile patient, "
    "this may represent early pneumonia. In an afebrile patient, consider "
    "atelectasis. Clinical correlation required.",

    # --- LOW CONFIDENCE / BORDERLINE ---
    "Minimal opacity at the right base, possibly representing early consolidation "
    "versus overlying soft tissue. Inspiratory effort is suboptimal. "
    "Repeat PA view in upright position recommended.",

    "Questionable increased density in the left lower lobe. This may be "
    "artifactual or represent mild atelectasis. No definitive consolidation identified. "
    "Clinical features should guide management.",

    # --- PLEURAL EFFUSION + PNEUMONIA ---
    "Right-sided pleural effusion with associated compressive atelectasis. "
    "Underlying consolidation cannot be excluded. Decubitus views or ultrasound "
    "recommended to characterize the effusion.",

    "Small bilateral pleural effusions with basal opacities consistent with "
    "pneumonia with reactive pleural disease. Antibiotic therapy and monitoring "
    "of effusion size recommended.",

    # --- BACTERIAL vs VIRAL PATTERNS ---
    "Segmental lobar consolidation consistent with bacterial pneumonia. "
    "Air bronchograms are a classic sign of bacterial consolidation. "
    "Sputum Gram stain and culture recommended.",

    "Bilateral interstitial infiltrates in a reticular pattern. "
    "Distribution suggests viral pneumonia or Pneumocystis jirovecii in "
    "immunocompromised patients. Serology and LDH levels recommended.",

    "Ground-glass opacity in bilateral lower lobes. "
    "Pattern consistent with viral pneumonitis or early organizing pneumonia. "
    "HRCT chest for better characterization. Antiviral therapy consideration.",

    # --- NORMAL FINDINGS ---
    "Lungs are clear bilaterally. No consolidation, effusion, or pneumothorax. "
    "Cardiac silhouette within normal limits. Bony thorax intact. "
    "No acute cardiopulmonary abnormality.",

    "Clear bilateral lung fields with no evidence of pneumonia, edema, or "
    "pleural effusion. Diaphragm and costophrenic angles are well-defined. "
    "Normal chest radiograph.",

    "No focal consolidation or airspace disease identified. "
    "Trachea is midline. Pulmonary vascularity is normal. "
    "No acute abnormality detected.",

    "Bilateral lung fields demonstrate normal aeration. "
    "No perihilar infiltrates, effusion, or pneumothorax. "
    "Cardiac size is within normal limits. Normal study.",

    # --- TREATMENT RECOMMENDATIONS ---
    "For community-acquired bacterial pneumonia: oral amoxicillin-clavulanate "
    "or azithromycin for 5-7 days in outpatient setting. Hospitalization if "
    "PSI score Class IV-V or CURB-65 score ≥ 2.",

    "Severe pneumonia requiring hospitalization: IV ceftriaxone plus macrolide, "
    "or respiratory fluoroquinolone monotherapy. Duration 5-7 days minimum.",

    "Atypical pneumonia (Mycoplasma, Chlamydia): macrolide antibiotics preferred. "
    "Doxycycline as alternative. Clinical improvement expected within 48-72 hours.",

    # --- FOLLOW-UP ---
    "Radiographic resolution of pneumonia lags clinical improvement by 4-6 weeks. "
    "Follow-up chest X-ray recommended at 6-8 weeks to confirm resolution "
    "and exclude underlying malignancy.",

    "In elderly patients or smokers with pneumonia, follow-up imaging is mandatory "
    "to exclude an obstructing lesion or malignancy as precipitating cause.",

    # --- COMPLICATIONS ---
    "Parapneumonic effusion identified. If exudative or loculated, "
    "drainage may be required. Pleural fluid analysis recommended.",

    "Early cavitation within consolidation raises concern for lung abscess "
    "or necrotizing pneumonia. Prolonged antibiotic course (4-6 weeks) required.",

    "Post-pneumonia bronchiectasis may develop in recurrent infections. "
    "HRCT and pulmonary function testing recommended after resolution.",

    # --- PEDIATRIC ---
    "Round pneumonia in pediatric patient — spherical opacity in posterior segment "
    "of upper or lower lobe. More common in children. "
    "Oral amoxicillin is first-line therapy.",

    "Viral pneumonia in pediatric patient: bilateral perihilar infiltrates, "
    "peribronchial thickening, and hyperinflation. Supportive care is mainstay. "
    "Antibiotics only if secondary bacterial infection suspected.",

    # --- ADDITIONAL CONTEXT ---
    "Hospital-acquired pneumonia (HAP): consider resistant organisms (MRSA, "
    "Pseudomonas). Broad-spectrum piperacillin-tazobactam or meropenem plus "
    "vancomycin until culture results available.",
]
# ─────────────────────────────────────────────────────────────────────────────


def build_faiss_index(chunks: list[str]) -> tuple:
    """
    Encodes knowledge base chunks using sentence-transformers,
    builds and returns a FAISS index.
    """
    try:
        from sentence_transformers import SentenceTransformer
        import faiss
    except ImportError:
        raise ImportError("Run: pip install faiss-cpu sentence-transformers")

    print("  Loading sentence-transformer model (all-MiniLM-L6-v2)...")
    encoder      = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings   = encoder.encode(chunks, show_progress_bar=True,
                                  convert_to_numpy=True, normalize_embeddings=True)

    dim   = embeddings.shape[1]                   # 384 for MiniLM
    index = faiss.IndexFlatIP(dim)                # Inner product = cosine similarity on normalized vecs
    index.add(embeddings.astype(np.float32))

    print(f"  ✅  FAISS index built: {index.ntotal} vectors, dim={dim}")
    return index, encoder, chunks


def save_index(index, encoder, chunks: list[str]):
    """Pickle the FAISS index, encoder, and chunks to disk."""
    import faiss
    payload = {"chunks": chunks, "encoder_name": "all-MiniLM-L6-v2"}
    faiss.write_index(index, str(INDEX_PATH.with_suffix(".faiss")))
    with open(INDEX_PATH, "wb") as f:
        pickle.dump(payload, f)
    print(f"  ✅  Index saved → {INDEX_PATH}")


def load_index() -> tuple:
    """Load the FAISS index, chunk list, and re-initialize encoder."""
    try:
        import faiss
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError("Run: pip install faiss-cpu sentence-transformers")

    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {INDEX_PATH}.\n"
            "Run: python src/rag.py --build_index"
        )
    with open(INDEX_PATH, "rb") as f:
        payload = pickle.load(f)
    index   = faiss.read_index(str(INDEX_PATH.with_suffix(".faiss")))
    encoder = SentenceTransformer(payload["encoder_name"])
    return index, encoder, payload["chunks"]


def retrieve(query: str, index, encoder, chunks: list[str], top_k: int = 3) -> list[str]:
    """Embed query and retrieve top-k most similar text chunks from FAISS."""
    import numpy as np
    q_vec  = encoder.encode([query], normalize_embeddings=True).astype(np.float32)
    scores, indices = index.search(q_vec, top_k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]


def generate_report(
    query: str,
    retrieved_chunks: list[str],
    api_key: Optional[str] = None,
) -> str:
    """
    Calls the Groq API (free tier) with retrieved context + query
    to generate a clinical radiological summary.

    Falls back to a rule-based template if no API key is available.
    """
    api_key = api_key or os.environ.get("GROQ_API_KEY", "")

    context = "\n\n".join(
        [f"[Reference {i+1}]: {chunk}" for i, chunk in enumerate(retrieved_chunks)]
    )

    system_prompt = (
        "You are an expert radiologist writing concise, grounded clinical summaries. "
        "Use only information from the provided reference reports. "
        "Write 2-3 sentences. Be specific about location and clinical recommendations. "
        "Do not fabricate details not present in the references."
    )
    user_prompt = (
        f"Based on the following reference reports, write a clinical summary for this finding:\n"
        f"Finding: {query}\n\n"
        f"Reference Reports:\n{context}\n\n"
        f"Clinical Summary:"
    )

    if not api_key:
        # Fallback: template-based report when no API key
        print("  ⚠️  No GROQ_API_KEY found — using template fallback.")
        return _template_report(query, retrieved_chunks)

    try:
        from groq import Groq
        client   = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="llama3-8b-8192",     # free Groq model
            messages=[
                {"role": "system",  "content": system_prompt},
                {"role": "user",    "content": user_prompt},
            ],
            max_tokens=200,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"  ⚠️  Groq API error: {e}. Using template fallback.")
        return _template_report(query, retrieved_chunks)


def _template_report(query: str, chunks: list[str]) -> str:
    """Rule-based fallback report using the top retrieved chunk."""
    top_chunk = chunks[0] if chunks else "No reference found."
    return (
        f"Radiological Analysis: {query}. "
        f"Based on imaging characteristics: {top_chunk[:200]}... "
        f"Clinical correlation and follow-up imaging recommended."
    )


def run_full_pipeline(gradcam_result: dict, api_key: Optional[str] = None) -> str:
    """
    Entry point called by FastAPI backend.
    Takes the dict returned by gradcam.analyze_image() and returns a report string.
    """
    index, encoder, chunks = load_index()
    query      = gradcam_result["query"]
    retrieved  = retrieve(query, index, encoder, chunks, top_k=3)
    report     = generate_report(query, retrieved, api_key)
    return report


def main(args):
    if args.build_index:
        print("Building FAISS knowledge base index...")
        index, encoder, chunks = build_faiss_index(KNOWLEDGE_BASE)
        save_index(index, encoder, chunks)
        return

    if args.query:
        print(f"Query: '{args.query}'")
        index, encoder, chunks = load_index()
        retrieved = retrieve(args.query, index, encoder, chunks, top_k=3)

        print("\n  📚  Retrieved chunks:")
        for i, chunk in enumerate(retrieved):
            print(f"  [{i+1}] {chunk[:120]}...")

        report = generate_report(args.query, retrieved, args.api_key)
        print(f"\n  📋  Generated Report:\n  {report}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoMed RAG Pipeline")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--build_index", action="store_true",
                       help="Build and save FAISS index from knowledge base")
    group.add_argument("--query", type=str,
                       help="Query string to test retrieval + generation")
    parser.add_argument("--api_key", type=str, default=None,
                        help="Groq API key (or set GROQ_API_KEY env var)")
    args = parser.parse_args()
    main(args)
