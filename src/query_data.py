# ====================== src/query_data.py ======================
import ollama
from typing import List, Dict, Optional
from langchain_chroma import Chroma
from src.embedding import get_embedding_function
from config.settings import CHROMA_PATH, LLM_MODEL, TOP_K ,TEMPERATURE


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — PROMPT TEMPLATE
# The LLM receives only the retrieved context — it is NOT allowed to invent
# answers. Edit the template text here to change the assistant's persona or
# answer style without touching any pipeline logic.
# ══════════════════════════════════════════════════════════════════════════════

PROMPT_TEMPLATE = """You are a professional telecom engineer assistant.
Answer the question using ONLY the context provided below.
If the answer is not in the context, say: "I could not find this information in the available documents."
Do not make up information. Be concise and technical.

─────────────────────────────────────────
CONTEXT:
{context}
─────────────────────────────────────────

QUESTION:
{question}

ANSWER:"""


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — SOURCE LOCATION HELPER
# Each doc_type stores a different "where in the file" field:
#   • pdf     → page number  (set by PyPDFDirectoryLoader)
#   • excel   → row number   (1-based, set in load_excel_as_documents)
#   • runbook → procedure number (extracted by extract_procedure_number)
# ══════════════════════════════════════════════════════════════════════════════

def get_location_label(chunk_metadata: dict) -> str:
    """
    Returns a human-readable location string for a retrieved chunk.

    Parameters
    ----------
    chunk_metadata : dict
        The metadata dict attached to the chunk — comes straight from ChromaDB.

    Returns
    -------
    str
        A short label shown in the UI and printed to the console, e.g.:
          • "Page: 54"          ← for PDF
          • "Row: 7"            ← for Excel  (never shows Row: 0 or Row: N/A)
          • "Proc: 3 " ← for runbook (shows N/A if not found in text)
          • "N/A"               ← fallback for unknown types
    """
    doc_type = chunk_metadata.get("doc_type", "unknown")

    if doc_type == "pdf":
        # PyPDFDirectoryLoader stores 0-based page index; add 1 for human display
        raw_page = chunk_metadata.get("page", "N/A")
        page_display = (int(raw_page) + 1) if isinstance(raw_page, int) else raw_page
        return f"Page: {page_display}"

    elif doc_type == "excel":
        # Row is already 1-based (set in load_excel_as_documents with +2 offset).
        # Guard against missing or zero values just in case.
        row = chunk_metadata.get("row", "N/A")
        if row == "N/A" or row == 0:
            return "Row: N/A"
        return f"Row: {row}"

    elif doc_type == "runbook":
        # procedure_no is set in split_documents() by regex scanning.
        # It is "N/A" when the chunk text doesn't contain a recognisable procedure header.
        proc = chunk_metadata.get("procedure_no", "N/A")
        return f"Proc: {proc}"

    # Unknown doc_type — nothing useful to show
    return "N/A"


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — RETRIEVAL  (Step 1 of the RAG pipeline)
# Converts the user's question to an embedding vector and finds the TOP_K
# most similar chunks in ChromaDB.
# ══════════════════════════════════════════════════════════════════════════════

def retrieve_chunks(question: str, filter_doc_type: Optional[str] = None) -> List[Dict]:
    """
    Queries ChromaDB for the top-K chunks most similar to `question`.

    Parameters
    ----------
    question        : The user's natural-language question.
    filter_doc_type : Optional filter — only return chunks of this doc_type.
                      Pass None or "all" to search across all types.

    Returns
    -------
    list of dicts, each containing:
      • text          → raw chunk text (fed into the prompt)
      • source        → file path
      • doc_type      → "pdf" | "excel" | "runbook"
      • location      → human-readable location label (page / row / proc)
      • score         → cosine similarity 0-1 (higher = more relevant)
    """
    embedding_function = get_embedding_function()
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_function,
    )

    # Run similarity search — with or without a doc_type filter
    if not filter_doc_type or filter_doc_type == "all":
        results = db.similarity_search_with_relevance_scores(question, k=TOP_K)
    else:
        results = db.similarity_search_with_relevance_scores(
            question, k=TOP_K,
            filter={"doc_type": filter_doc_type},
        )

    chunks = []
    for doc, score in results:
        # Build the location label based on doc_type using our helper above
        location = get_location_label(doc.metadata)

        chunks.append({
            "text"    : doc.page_content,
            "source"  : doc.metadata.get("source",   "unknown"),
            "doc_type": doc.metadata.get("doc_type", "unknown"),
            "location": location,                        
            "score"   : score,              #  higher = more similar
        })

    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — PROMPT BUILDER  (Step 2 of the RAG pipeline)
# Assembles the final prompt by injecting the retrieved chunks as context.
# ══════════════════════════════════════════════════════════════════════════════

def build_prompt(question: str, chunks: List[Dict]) -> str:
    """
    Builds the full prompt string that is sent to the LLM.

    Each chunk is formatted as:
      [N] Source: <path> | Type: <doc_type> | <location label>
      <chunk text>

    If no chunks were retrieved, the context section says so explicitly
    so the LLM correctly responds that it could not find information.
    """
    if not chunks:
        context = "No relevant documents were found."
    else:
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            source   = chunk.get("source",   "unknown")
            doc_type = chunk.get("doc_type", "unknown")
            location = chunk.get("location", "N/A")   # page / row / proc
            text     = chunk.get("text",     "").strip()
            context_parts.append(
                f"[{i}] Source: {source} | Type: {doc_type} | {location}\n{text}"
            )
        context = "\n\n".join(context_parts)

    return PROMPT_TEMPLATE.format(context=context, question=question)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — LLM CALL  (Step 3 of the RAG pipeline)
# Sends the assembled prompt to Mistral (via Ollama) and returns the answer.
# Temperature is kept very low (0.1) to reduce hallucination.
# ══════════════════════════════════════════════════════════════════════════════

def call_mistral(prompt: str) -> str:
    """
    Calls the Mistral model through Ollama with the given prompt.

    Returns the model's response as a plain string.
    If Ollama is unreachable or throws an error, returns an error message
    prefixed with "[LLM ERROR]" so the caller can detect it.
    """
    try:
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a professional telecom engineer assistant. "
                        "Answer only from the provided context. "
                        "Be concise, accurate, and technical."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            options={"temperature": TEMPERATURE , "num_predict": 512}
        )
        return response["message"]["content"].strip()

    except ollama.ResponseError as e:
        return f"[LLM ERROR] Ollama response error: {e}"
    except Exception as e:
        return f"[LLM ERROR] Unexpected error: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — MAIN QUERY FUNCTION
# This is the single public function called by the UI / CLI.
# It orchestrates all three RAG steps: retrieve → build prompt → call LLM.
# ══════════════════════════════════════════════════════════════════════════════

def query_rag(query_text: str, filter_by_doc_type: str = "all"):
    """
    Full RAG pipeline: retrieve relevant chunks → build prompt → ask Mistral.

    Parameters
    ----------
    query_text         : Natural-language question from the user.
    filter_by_doc_type : "all" | "pdf" | "excel" | "runbook"
                         Filters retrieved chunks to only the chosen doc_type.

    Returns
    -------
    answer  : str  — Mistral's answer based strictly on retrieved context.
    sources : list of dicts — one entry per retrieved chunk, containing:
                • source   → file path
                • doc_type → "pdf" | "excel" | "runbook"
                • location → "Page: N" | "Row: N" | "Proc: N "
                • score    → cosine similarity score (0-1)
    """
    print(f"\n🔎 Question: {query_text}\n")
    print(f"📂 Filter applied: {filter_by_doc_type.upper()}\n")

    # Convert "all" to None so retrieve_chunks knows not to filter
    filter_type = None if filter_by_doc_type == "all" else filter_by_doc_type

    # ── Step 1: Retrieve relevant chunks from ChromaDB ────────────────────────
    chunks = retrieve_chunks(question=query_text, filter_doc_type=filter_type)

    # Build the sources list that gets returned to the UI
    # Each entry mirrors what retrieve_chunks stored, keeping only display fields
    sources = [
        {
            "source"  : c["source"],
            "doc_type": c["doc_type"],
            "location": c["location"],   # ← replaces separate page/row fields
            "score"   : c["score"],
        }
        for c in chunks
    ]

    if not chunks:
        print("❌ No relevant documents found.")
        return "I could not find relevant information in the available documents.", []

    # ── Step 2: Build the LLM prompt from retrieved chunks ────────────────────
    prompt = build_prompt(question=query_text, chunks=chunks)

    # ── Step 3: Call the LLM and get the answer ───────────────────────────────
    print("[Pipeline] Calling Mistral...")
    answer = call_mistral(prompt)
    print(f"[Pipeline] Answer: {answer[:100]}...")

    # ── Print results to console ──────────────────────────────────────────────
    print("📌 Answer:")
    print(answer)
    print("\n📚 Sources:")
    for s in sources:
        # location now shows "Page: N", "Row: N", or "Proc: N "
        print(f"   • {s['doc_type'].upper()} | {s['source']} | {s['location']} | Score: {s['score']:.4f}")

    return answer, sources


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — CLI ENTRY POINT
# Allows running the pipeline directly from the terminal for quick testing:
#   python -m src.query_data "your question here" pdf
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        question    = sys.argv[1]
        filter_type = sys.argv[2] if len(sys.argv) > 2 else "all"
    else:
        question = input("\nEnter your telecom question: ")
        print("\nAvailable filters: all | pdf | excel | runbook")
        filter_type = input("Choose filter (press Enter for 'all'): ").strip().lower() or "all"

    answer, sources = query_rag(question, filter_type)