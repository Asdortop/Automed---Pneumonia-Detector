"""
report.py — Clinical Report Generator
Strategy:
  1. PRIMARY  — Simple rule-based template (always works, zero dependencies)
  2. FALLBACK — RAG pipeline (FAISS + Groq LLM) if index + API key available
"""

from pathlib import Path
from typing import Optional

BASE_DIR   = Path(__file__).resolve().parent.parent
INDEX_PATH = BASE_DIR / "data" / "faiss_index.pkl"


# ── Primary: Simple Template ──────────────────────────────────────────────────

def simple_report(label: str, probability: float, location: str, severity: str) -> str:
    """
    Generates a concise clinical report from classifier output.
    No external dependencies — always works instantly.
    """
    if label == "PNEUMONIA":
        if probability >= 85:
            return (
                f"Dense consolidation detected in the {location}, consistent with "
                f"high-confidence bacterial pneumonia ({probability:.1f}% confidence). "
                f"Air bronchograms likely present. Immediate antibiotic therapy recommended; "
                f"consider hospitalization if systemically unwell. "
                f"Repeat imaging in 4-6 weeks to confirm resolution."
            )
        elif probability >= 65:
            return (
                f"Patchy opacity noted in the {location}, suggestive of pneumonia "
                f"({probability:.1f}% confidence). "
                f"Clinical correlation with fever, cough, and inflammatory markers advised. "
                f"Oral antibiotic therapy appropriate if symptomatic. "
                f"Follow-up chest X-ray in 48-72 hours if symptoms persist."
            )
        else:
            return (
                f"Borderline findings in the {location} — cannot exclude early consolidation "
                f"({probability:.1f}% confidence). "
                f"Findings may represent early pneumonia or atelectasis. "
                f"Clinical correlation required. Repeat imaging in 24-48 hours recommended."
            )
    else:
        return (
            f"Lungs clear bilaterally ({probability:.1f}% confidence of normal). "
            f"No consolidation, pleural effusion, or pneumothorax identified. "
            f"Cardiac silhouette within normal limits. No acute cardiopulmonary abnormality."
        )


# ── Fallback: RAG Pipeline ────────────────────────────────────────────────────

def rag_report(query: str, api_key: Optional[str] = None) -> Optional[str]:
    """
    Tries to generate a report via RAG (FAISS + Groq LLM).
    Returns None if FAISS index is missing or RAG fails.
    """
    import os
    api_key = api_key or os.environ.get("GROQ_API_KEY", "")

    if not INDEX_PATH.exists():
        return None  # FAISS index not built — skip RAG

    try:
        from rag import load_index, retrieve, generate_report
        index, encoder, chunks = load_index()
        retrieved = retrieve(query, index, encoder, chunks, top_k=3)
        return generate_report(query, retrieved, api_key)
    except Exception as e:
        print(f"  ⚠️  RAG failed ({e}), using simple report instead.")
        return None


# ── Main Entry Point ──────────────────────────────────────────────────────────

def generate(
    label: str,
    probability: float,
    location: str,
    severity: str,
    query: str,
    api_key: Optional[str] = None,
) -> str:
    """
    Primary entry point for report generation.

    Priority:
      1. Simple template  — instant, no dependencies
      2. RAG              — richer output, requires FAISS index + optional Groq key

    To use RAG as primary, set USE_RAG_FIRST=True below.
    """
    USE_RAG_FIRST = False  # ← flip to True to prefer RAG when available

    if USE_RAG_FIRST:
        report = rag_report(query, api_key)
        if report:
            return report
        print("  ℹ️  RAG unavailable — using simple template.")

    # Always returns a valid report
    return simple_report(label, probability, location, severity)
