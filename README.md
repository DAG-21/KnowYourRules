# KnowYourRules

A RAG-backed chatbot for looking up Indian traffic rules and regulations, grounded strictly
in official government documents. It never answers from general knowledge — if a rule isn't
in the indexed corpus, it says so instead of guessing.

## Approach

```
Gov. websites → Downloaded PDFs → Text extraction → Chunking → Embeddings → Chroma
                                                                                 │
Chat UI ← Groq LLM (strict prompt) ← Retrieved chunks ─────────────────────────┘
```

The bot asks for your state if it doesn't know it (rules vary by state), retrieves the most
relevant chunks — your state's rules plus the nationwide Central rules — and answers using
*only* that text, citing the source. Follow-up questions are resolved using recent
conversation history before retrieval runs.

## The process

1. **Gather sources** — India's traffic regulations are split across Central, state, union
   territory, city/district, and judicial bodies, with no single index. A research pass
   produced **126 verified official government document sources** across this hierarchy.
   → [`research/india_traffic_document_links.xlsx`](research/india_traffic_document_links.xlsx)

2. **Extract documents** — [`scripts/extract_documents.py`](scripts/extract_documents.py)
   downloads every reachable document from those sources, tracking full provenance (URL,
   SHA-256, timestamp) in a manifest, and resumes safely if interrupted.
   → 1,400+ PDFs in `data/raw/full_extraction/`

3. **Triage for OCR cost** — OCR on scanned PDFs measured at ~9–40 sec/page; the full corpus
   would take 50+ hours. Rather than eat that cost blindly, documents were filtered to real
   Acts/Rules (not routine notices) and split by **text-native** (instant to extract) vs.
   **scanned** (needs OCR). This produced **84 text-native files** usable immediately.

4. **Build the RAG pipeline** — [`notebooks/rag_pipeline.ipynb`](notebooks/rag_pipeline.ipynb),
   built and validated phase by phase: document loading, chunking
   (`RecursiveCharacterTextSplitter`), embedding (Ollama), vector storage (Chroma), and
   strict-grounded retrieval + generation (Groq). A multi-language OCR + translation path
   was also built and validated for scanned documents (any Indian language → English) but
   isn't in the production index yet, given time constraints.

5. **Narrow to a production scope** — embedding all 84 files would take ~70 minutes; given a
   hard deadline, scope was narrowed to **24 curated files**: Central Motor Vehicles Act/Rules
   plus Gujarat, Delhi, Mizoram, Himachal Pradesh, Andaman & Nicobar, and Haryana. This step
   also caught a real bug — two files had been downloaded via other states' pages and were
   mistagged with that state's name, even though their content is the nationwide Central Act.
   Every chunk now carries a verified `state` metadata field used to filter retrieval.

6. **Backend** — [`backend/server.py`](backend/server.py) (FastAPI) wraps the RAG chain with
   what production needs: asks for and remembers your location, filters retrieval by state
   (falling back to Central rules + an honest "not covered" note if your state isn't
   indexed), and keeps short conversation history so follow-ups resolve correctly.

7. **Frontend** — [`frontend/index.html`](frontend/index.html): a single-page chat UI served
   directly by the backend, with markdown-rendered answers and a session that resets cleanly
   on reload.

## Tech stack

| Layer | Choice |
|---|---|
| LLM | Groq (`openai/gpt-oss-120b`) |
| Embeddings | Ollama `nomic-embed-text` (local) |
| Framework | LangChain (loaders, splitters, prompts, LCEL chains) |
| Vector store | Chroma (persisted locally) |
| PDF/OCR | `unstructured` (`fast` for text-native, `hi_res` + Tesseract for scanned) |
| Backend | FastAPI |
| Frontend | Vanilla HTML/CSS/JS + `marked.js` |

## Project structure

```
├── research/       # Government source inventory (spreadsheet)
├── scripts/        # Document downloader
├── data/           # Downloaded PDFs + manifest (gitignored)
├── notebooks/      # Pipeline build & validation notebook
├── backend/        # Indexing script + FastAPI server
├── frontend/       # Chat UI
└── requirements.txt
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
ollama pull nomic-embed-text
```

Copy `.env.example` to `.env` and add your `GROQ_API_KEY`.

## Running it

```bash
python backend/build_index.py     # build the vector store
uvicorn backend.server:app --reload --port 8000
```

Open http://localhost:8000.

## Current coverage

Indexed: **Central Motor Vehicles Act/Rules, Gujarat, Delhi.** Extracted but not yet embedded:
Mizoram, Himachal Pradesh, Andaman & Nicobar, Haryana — re-run `build_index.py` to add them.
Any state not indexed gets an honest "not covered" answer, never another state's rules.

## Acknowledgments

The LangChain/Groq/Ollama code patterns used here — embedding/chat model setup, prompt
templates, LCEL chain composition — follow the syntax and conventions taught in our
coursework on building RAG systems.
