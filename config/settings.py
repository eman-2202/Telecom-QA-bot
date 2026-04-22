# ====================== config/settings.py ======================
# Centralized configuration for Telecom RAG Bot

# ── Paths ─────────────────────────────────────────────────
DATA_PATH   = "data"
CHROMA_PATH = "chroma"

# ── Embedding ──────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "bge-m3"

# ── LLM ────────────────────────────────────────────────────
LLM_MODEL   = "mistral"
TEMPERATURE = 0

# ── Retrieval ──────────────────────────────────────────────
TOP_K = 5

# ── Chunking ───────────────────────────────────────────────
PDF_CHUNK_SIZE        = 1500
PDF_CHUNK_OVERLAP     = 200
RUNBOOK_CHUNK_SIZE    = 800
RUNBOOK_CHUNK_OVERLAP = 100
DEFAULT_CHUNK_SIZE    = 1200
DEFAULT_CHUNK_OVERLAP = 200

# ── UI Filter Options ──────────────────────────────────────
FILTER_OPTIONS = {
    "All Sources"              : "all",
    "PDF Only (3GPP Specs)"    : "pdf",
    "KPI Table Only (Excel)"   : "excel",
    "Runbook Only (Procedures)": "runbook",
}
# ── UI Example Questions ──────────────────────────────────────
EXAMPLES=[
                    ["What is the definition of PRB Utilization and what is its warning threshold?"],
                    ["What does 3GPP TS 28.554 define as the measurement period for handover KPIs?"],
                    ["What are the steps to follow when PRB utilization exceeds the critical threshold?"],
                    ["Which network function is responsible for session management in 5G core?"],
                    ["What KPIs have a warning threshold below 90%?"]
                ]
