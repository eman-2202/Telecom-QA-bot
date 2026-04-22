# ====================== src/populate_data.py ======================
import re
from langchain_community.document_loaders import PyPDFDirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from src.embedding import get_embedding_function
import pandas as pd
from pathlib import Path

from config.settings import (
    DATA_PATH,
    CHROMA_PATH,
    PDF_CHUNK_SIZE,
    PDF_CHUNK_OVERLAP,
    RUNBOOK_CHUNK_SIZE,
    RUNBOOK_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — LOADERS
# Each loader reads a different file type and tags docs with "doc_type"
# so the rest of the pipeline knows how to split and display them.
# ══════════════════════════════════════════════════════════════════════════════

def load_documents():
    """
    Master loader — reads all file types from DATA_PATH and returns
    a flat list of LangChain Document objects.

    doc_type tags set here:
      • "pdf"     → PDF pages (page number comes from PyPDFDirectoryLoader)
      • "excel"   → Excel rows  (row number set in load_excel_as_documents)
      • "runbook" → TXT files   (procedure numbers extracted at split time)
    """
    all_docs = []

    # ── PDFs ──────────────────────────────────────────────────────────────────
    print("📄 Loading PDFs...")
    pdf_loader = PyPDFDirectoryLoader(DATA_PATH)
    pdf_docs = pdf_loader.load()
    for doc in pdf_docs:
        # Tag every PDF page so downstream code knows its origin
        doc.metadata["doc_type"] = "pdf"
    all_docs.extend(pdf_docs)
    print(f"   → Loaded {len(pdf_docs)} PDF pages")

    # ── Excel ─────────────────────────────────────────────────────────────────
    print("📊 Loading Excel files...")
    for excel_file in Path(DATA_PATH).rglob("*.xlsx"):
        excel_docs = load_excel_as_documents(str(excel_file))
        all_docs.extend(excel_docs)

    # ── TXT runbooks ──────────────────────────────────────────────────────────
    print("📝 Loading TXT files...")
    for txt_file in Path(DATA_PATH).rglob("*.txt"):
        try:
            txt_loader = TextLoader(str(txt_file), encoding="utf-8", autodetect_encoding=True)
            txt_docs = txt_loader.load()
            for doc in txt_docs:
                # Tag as runbook 
                doc.metadata["doc_type"] = "runbook"
            all_docs.extend(txt_docs)
            print(f"   → Loaded {txt_file.name}")
        except Exception as e:
            print(f"   ⚠️  Skipped {txt_file.name} → {e}")

    print(f"✅ Total documents loaded: {len(all_docs)}")
    return all_docs


def load_excel_as_documents(excel_path: str):
    """
    Converts each Excel row into a LangChain Document.

    Metadata stored per document:
      • source   → file path
      • row      → 1-based row number (header = row 1, data starts at row 2)
                   We add 2 so the number matches what you see in Excel:
                     - +1 because pandas index is 0-based
                     - +1 to skip the header row
      • doc_type → "excel"

    The page_content is every column joined as "ColName: Value | ColName: Value ..."
    so the LLM can read it naturally.
    """
    df = pd.read_excel(excel_path)
    documents = []
    for index, row in df.iterrows():
        content = " | ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
        doc = Document(
            page_content=content,
            metadata={
                "source"  : excel_path,
                # index is 0-based; +2 converts to Excel row number (1 header + 1 offset)
                "row"     : int(index) + 2,
                "doc_type": "excel",
            }
        )
        documents.append(doc)
    print(f"   → Converted {len(documents)} rows from {Path(excel_path).name}")
    return documents


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — RUNBOOK HELPERS
# These functions parse the raw text of a TXT runbook chunk to find
# procedure numbers so they can be stored as metadata.
# ══════════════════════════════════════════════════════════════════════════════

def extract_procedure_number(text: str) -> str:
    """
    Scans chunk text for a procedure header and returns its number.

    Patterns recognised (case-insensitive):
      • "Procedure 3"       → "3"
      • "Procedure 3.1"     → "3.1"
      • "PROCEDURE 7:"      → "7"
      • "proc. 2"           → "2"

    Returns "N/A" when no procedure number is found in the chunk.
    """
    # Regex breakdown:
    #   proc(?:edure)?  → matches "proc" or "procedure" (the "edure" part is optional)
    #   \.?             → optional dot (for "proc.")
    #   \s*             → optional whitespace
    #   (\d+(?:\.\d+)?) → capture group: one or more digits, optionally followed by .digits
    match = re.search(r'proc(?:edure)?\.?\s*(\d+(?:\.\d+)?)', text, re.IGNORECASE)
    return match.group(1) if match else "N/A"



# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — CHUNKING
# Documents are split into smaller pieces so they fit the embedding model.
# Each doc_type uses different chunk sizes tuned for its content density.
# ══════════════════════════════════════════════════════════════════════════════

def get_splitter(doc_type: str):
    """
    Returns the correct text splitter for each doc_type:
      • "pdf"     → larger chunks (technical specs are dense)
      • "excel"   → None (each row is already one chunk — no further splitting)
      • "runbook" → medium chunks (step-by-step procedures)
      • default   → fallback splitter for anything unrecognised
    """
    if doc_type == "pdf":
        return RecursiveCharacterTextSplitter(
            chunk_size=PDF_CHUNK_SIZE,
            chunk_overlap=PDF_CHUNK_OVERLAP,
            length_function=len,
        )
    elif doc_type == "excel":
        # Excel rows are already atomic — each row is one self-contained Document
        return None
    elif doc_type == "runbook":
        return RecursiveCharacterTextSplitter(
            chunk_size=RUNBOOK_CHUNK_SIZE,
            chunk_overlap=RUNBOOK_CHUNK_OVERLAP,
            length_function=len,
        )
    # Unknown doc_type — use safe defaults
    return RecursiveCharacterTextSplitter(
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
    )


def split_documents(documents):
    """
    Splits every document into chunks using the splitter for its doc_type.

    After splitting, runbook chunks are scanned for procedure numbers
    and those are saved as metadata fields:
      • "procedure_no" → e.g. "3" or "3.1", or "N/A"

    Excel docs are not split — they pass through as-is.
    """
    print("\n🔪 Splitting documents with type-specific strategies...")
    chunks = []

    for doc in documents:
        doc_type = doc.metadata.get("doc_type")
        if not doc_type:
            print(f"⚠️ Warning: No doc_type found for {doc.metadata.get('source')}")
            doc_type = "unknown"

        splitter = get_splitter(doc_type)

        if splitter is None:
            # ── Excel: no splitting needed — assign ID directly ────────────
            doc.metadata["id"] = f"{doc.metadata['source']}:{doc.metadata.get('row', 0)}:0"
            chunks.append(doc)

        else:
            # ── PDF / Runbook: split into smaller chunks ───────────────────
            split_docs = splitter.split_documents([doc])
            for i, chunk in enumerate(split_docs):
                chunk.metadata["doc_type"] = doc_type
                chunk.metadata["id"] = (
                    f"{chunk.metadata['source']}:{chunk.metadata.get('page', 0)}:{i}"
                )

                # ── Runbook only: extract procedure number ─────────────────
                if doc_type == "runbook":
                    chunk.metadata["procedure_no"] = extract_procedure_number(chunk.page_content)

                chunks.append(chunk)

    print(f"✅ Created {len(chunks)} chunks")
    return chunks


def calculate_chunk_ids(chunks):
    """
    Assigns a unique, stable ID to every chunk in the format:
        <source_path>:<page_or_row>:<chunk_index>

    The chunk_index resets to 0 whenever the page/row changes so IDs
    remain stable across re-ingestion runs (no duplicate inserts).
    """
    print("🔑 Assigning unique chunk IDs...")
    last_page_id        = None
    current_chunk_index = 0

    for chunk in chunks:
        source      = chunk.metadata.get("source")
        page_or_row = chunk.metadata.get("row") or chunk.metadata.get("page", 0)
        current_page_id = f"{source}:{page_or_row}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk.metadata["id"] = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — INGESTION PIPELINE
# Orchestrates loading → splitting → deduplication → storing in ChromaDB.
# ══════════════════════════════════════════════════════════════════════════════

def run_ingestion():
    print("=== Telecom RAG Data Ingestion Started ===\n")

    # Step 1: Load all files from DATA_PATH
    documents = load_documents()

    # Step 2: Split into chunks with type-specific strategies
    chunks = split_documents(documents)

    # Step 3: Assign stable unique IDs for deduplication
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Step 4: Open (or create) the ChromaDB vector store
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function()
    )

    # Step 5: Only add chunks that are not already in the DB (incremental update)
    existing_items = db.get(include=[])
    existing_ids   = set(existing_items["ids"]) if existing_items.get("ids") else set()

    new_chunks = [
        chunk for chunk in chunks_with_ids
        if chunk.metadata["id"] not in existing_ids
    ]

    if new_chunks:
        db.add_documents(new_chunks, ids=[c.metadata["id"] for c in new_chunks])
        print(f"✅ Added {len(new_chunks)} NEW chunks to ChromaDB")
    else:
        print("✅ Database is already up-to-date — no new chunks added")

    print("\n🎉 Pipeline finished successfully!")
    print(f"Total chunks now in vector database: {len(db.get()['ids'])}")
    print(f"Sample chunk ID: {chunks_with_ids[0].metadata['id'] if chunks_with_ids else 'None'}")


if __name__ == "__main__":
    run_ingestion()