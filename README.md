

<h1>📡 Telecom Q&A RAG ChatBot</h1>
<div align="center">
<p><em>A production-grade Multi-Source Retrieval-Augmented Generation system built for telecom engineers.<br/>
Query 3GPP technical specifications, KPI threshold tables, and NOC runbooks using natural language — with full source attribution.</em></p>

<br/>

<!-- Tech Stack Badges -->
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-Orchestration-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white)](https://www.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-FF6B35?style=for-the-badge&logo=databricks&logoColor=white)](https://www.trychroma.com/)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-000000?style=for-the-badge&logo=ollama&logoColor=white)](https://ollama.com/)
[![Mistral](https://img.shields.io/badge/Mistral-LLM-FF7000?style=for-the-badge&logo=openai&logoColor=white)](https://mistral.ai/)
[![bge-m3](https://img.shields.io/badge/bge--m3-Embeddings-6236FF?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/BAAI/bge-m3)
[![Gradio](https://img.shields.io/badge/Gradio-UI-FF7C00?style=for-the-badge&logo=gradio&logoColor=white)](https://www.gradio.app/)
[![3GPP](https://img.shields.io/badge/Domain-3GPP%20%7C%20NOC%20%7C%20KPI-00897B?style=for-the-badge)](https://www.3gpp.org/)

<br/><br/>

<!-- Retrieval Evaluation Results -->
<table>
<thead>
<tr>
<th>Metric</th>
<th>Hit@1</th>
<th>Hit@3</th>
<th>Hit@5</th>
<th>Sources</th>
<th>Similarity</th>
</tr>
</thead>
<tbody>
<tr>
<td><b>Score</b></td>
<td><code>80%</code> </td>
<td><code>80%</code> </td>
<td><code>100%</code> </td>
<td><code>PDF · Excel · TXT</code></td>
<td><code>Cosine</code></td>
</tr>
</tbody>
</table>

</div>

---



## 🗂️ Project Structure

```
telecom-rag-bot/
├── app.py                    # Gradio web UI entry point
├── config/
│   ├── settings.py           # Centralized configuration (paths, models, chunking)
│   └── real_id.py            # Utility: list all chunk IDs in ChromaDB
├── src/
│   ├── embedding.py          # Embedding function (bge-m3 via Ollama)
│   ├── populate_data.py      # Data ingestion pipeline (load → chunk → store)
│   └── query_data.py         # RAG query pipeline (retrieve → prompt → LLM)
├── styles/
│   └── ui.py                 # CSS and HTML constants for the Gradio interface
├── tests/
│   └── test_RAG.py           # Retrieval evaluation: Hit@k metrics + generation tests
├── data/                     # Place your source documents here
│   ├── 3GPP_Technical_Specifications.pdf
│   ├── KPI_Thresholds.xlsx
│   └── NOC_Runbook.txt
└── chroma/                   # Auto-created: ChromaDB vector store
```

---

## 🚀 Features

- **Multi-source RAG** — queries across three document types simultaneously
- **PDF support** — 3GPP technical specification documents (e.g., TS 28.554)
- **Excel support** — KPI threshold tables with row-level retrieval
- **Runbook support** — NOC procedure documents with procedure-number tagging
- **Metadata filtering** — filter retrieval by document type at query time
- **Source attribution** — every answer shows the exact source file, page/row/procedure, and similarity score
- **Gradio UI** — clean dark-themed chat interface with example questions
- **Hit@k evaluation** — built-in retrieval benchmarking with bar chart output

---

## ⚙️ Architecture

```
User Question
     │
     ▼
[Embedding: bge-m3]
     │
     ▼
[ChromaDB Similarity Search] ←── optional doc_type filter
     │
     ▼
[Top-K Chunks Retrieved]
     │
     ▼
[Prompt Builder] — injects chunks as context
     │
     ▼
[Mistral LLM via Ollama]
     │
     ▼
Answer + Source Citations
```

---

## 📋 Setup Instructions

### 1. Prerequisites

- Python 3.9+
- [Ollama](https://ollama.com/) installed and running locally
- Required Ollama models pulled:

```bash
ollama pull mistral
ollama pull bge-m3
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare Your Data

Place your source documents in the `data/` directory:

```
data/
├── *.pdf         → 3GPP specs or any PDF documents
├── *.xlsx        → KPI tables or structured Excel data
└── *.txt         → NOC runbooks or procedure guides
```

### 4. Ingest Documents

```bash
python -m src.populate_data
```

This will load all documents, split them into chunks, embed them using `bge-m3`, and store them in ChromaDB under `chroma/`.

### 5. Launch the UI

```bash
python app.py
```

The Gradio interface will open at `http://localhost:7860`.

---

## 🖥️ Usage

### Web Interface

Open the app and type any telecom question in the input box. Use the **Document Type** dropdown to filter retrieval:

| Filter | Searches |
|--------|----------|
| All Sources | All document types |
| PDF Only (3GPP Specs) | PDF pages only |
| KPI Table Only (Excel) | Excel rows only |
| Runbook Only (Procedures) | TXT runbook chunks only |

### CLI (Quick Testing)

```bash
python -m src.query_data "What is PRB Utilization?" pdf
python -m src.query_data "What are the steps when SMF overloads?" runbook
python -m src.query_data "Which KPIs have a warning threshold below 90%?" excel
```

### Inspect Stored Chunks

```bash
python -m config.real_id
```

---

## 🧪 Evaluation

Run the built-in retrieval evaluation suite:

```bash
python -m tests.test_RAG
```

This runs **Hit@k** evaluation (k = 1, 3, 5) across 5 test questions and generates a bar chart (`hit_at_k.png`).

**Current results:**

| Metric | Score |
|--------|-------|
| Hit@1  | 80%   |
| Hit@3  | 80%   |
| Hit@5  | 100%  |

---

## 🔧 Configuration

All parameters are centralized in `config/settings.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DATA_PATH` | `"data"` | Directory containing source documents |
| `CHROMA_PATH` | `"chroma"` | ChromaDB persistence directory |
| `EMBEDDING_MODEL_NAME` | `"bge-m3"` | Ollama embedding model |
| `LLM_MODEL` | `"mistral"` | Ollama LLM model |
| `TEMPERATURE` | `0` | LLM temperature (0 = deterministic) |
| `TOP_K` | `5` | Number of chunks retrieved per query |
| `PDF_CHUNK_SIZE` | `1500` | Chunk size for PDFs (characters) |
| `PDF_CHUNK_OVERLAP` | `200` | Overlap for PDF chunks |
| `RUNBOOK_CHUNK_SIZE` | `800` | Chunk size for runbooks |
| `RUNBOOK_CHUNK_OVERLAP` | `100` | Overlap for runbook chunks |

---

## 📚 Supported Document Types

### PDFs (3GPP Specs)
- Loaded via `PyPDFDirectoryLoader`
- Chunked at 1500 characters with 200-character overlap
- Source location shown as `Page: N`

### Excel (KPI Tables)
- Each row becomes one document chunk
- Source location shown as `Row: N` (matches Excel row number)
- Columns joined as `ColName: Value | ColName: Value`

### TXT (NOC Runbooks)
- Chunked at 800 characters with 100-character overlap
- Procedure numbers extracted by regex (`Procedure N` pattern)
- Source location shown as `Proc: N`

---

## 🔍 Self-Learning Extension — Metadata Filtering

ChromaDB supports filtering results by metadata at query time. This means you can tell the retriever: only search chunks where `doc_type == 'excel'`. This is useful when you know the answer lives in a specific document type.

A `filter_by_doc_type` parameter is implemented in the query pipeline. When the parameter is set, it passes a `where` clause to ChromaDB's `similarity_search` call. In the Gradio UI, a dropdown menu provides the following options:

| Option | Description |
|--------|-------------|
| All Sources | Searches all document types simultaneously |
| PDF Only | Restricts retrieval to 3GPP spec PDF pages |
| KPI Table Only | Restricts retrieval to Excel KPI rows |
| Runbook Only | Restricts retrieval to TXT runbook chunks |

### When to Use Metadata Filtering

**✅ Use filtering when:**
- You know the answer lives in a specific document type (e.g., thresholds → Excel)
- Questions are very domain-specific (e.g., procedures → Runbook)
- You want to reduce noise from irrelevant sources

**⛔ Skip filtering (use "All Sources") when:**
- Questions span multiple documents (e.g., *"What is PRB Utilization and what are the steps when it's exceeded?"*)
- You're unsure which document type contains the answer
- Questions are exploratory or high-level

---

## 📝 Test Queries the System Must Answer

Each query below was run **twice**: once without filtering (all sources) and once with the appropriate filter applied. Results were compared to observe any difference in precision and relevance.

| # | Query | Recommended Filter |
|---|-------|--------------------|
| 1 | What is the definition of PRB Utilization and what is its warning threshold? | KPI Table Only |
| 2 | What does 3GPP TS 28.554 define as the measurement period for handover KPIs? | PDF Only |
| 3 | What are the steps to follow when PRB utilization exceeds the critical threshold? | Runbook Only |
| 4 | Which network function is responsible for session management in 5G core? | No Filter |
| 5 | What KPIs have a warning threshold below 90%? | KPI Table Only |

> **Observation:** Applying the correct metadata filter consistently improved answer precision by eliminating irrelevant chunks from other document types, especially for domain-specific procedural and threshold queries.

-----

## 🛠️ Stack

| Component | Technology |
|-----------|-----------|
| LLM | Mistral (via Ollama) |
| Embeddings | bge-m3 (via Ollama) |
| Vector Store | ChromaDB (cosine similarity) |
| Orchestration | LangChain |
| UI | Gradio |
| PDF Loading | LangChain PyPDFDirectoryLoader |
| Excel Loading | pandas |

-----

## 👥 Team Members

This project was developed collaboratively, with each member owning a core component of the RAG pipeline.

- ### Eman Tarek
- ### Tasneem Amin
- ### Mohamed Salah
