# KnowYourRules

A RAG-backed chatbot for looking up Indian traffic rules and regulations, sourced from
official government documents: the Central Motor Vehicles Act/Rules, state Acts/Rules,
union territories, and city/local traffic police.

## Status

Working end-to-end: document extraction, chunking, embedding, retrieval, and a
location-aware chat UI backed by a real FastAPI server.

- **Research** (`research/`) — 126 verified/discovered official government document
  sources across the full regulatory hierarchy.
- **Extraction** (`scripts/extract_documents.py`) — downloads verified sources into
  `data/raw/full_extraction/`, with full provenance in `manifest.csv`.
- **Indexing** (`backend/build_index.py`) — extracts, chunks, and embeds (Ollama
  `nomic-embed-text`) a curated text-native subset into a persisted Chroma vector store,
  tagged with correct per-document state metadata for state-aware retrieval.
- **Backend** (`backend/server.py`) — FastAPI app wrapping the RAG chain (Groq LLM,
  strict grounding prompt). Tracks per-session location and a short conversation history
  so follow-up questions ("what's the fine for that?") resolve correctly, and asks for
  the user's state/city/PIN code when it's needed and not yet known.
- **Frontend** (`frontend/index.html`) — single-page chat UI served by the backend.
- **Reference** (`reference/`) — the course notebooks this project's LangChain/Groq/Ollama
  patterns are built from.
- **Build notebook** (`notebooks/rag_pipeline.ipynb`) — phase-by-phase development and
  validation of the pipeline (OCR experiments, translation, chunking, embedding,
  retrieval, generation).

Currently indexed: Central Motor Vehicles Act/Rules, Gujarat, Delhi. Mizoram, Himachal
Pradesh, Andaman & Nicobar, and Haryana are extracted but not yet embedded - see
`backend/build_index.py`'s scope list to top up the index. The system explicitly tells
the user when their state isn't covered rather than presenting another state's rules as
if they applied.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in `GROQ_API_KEY`. Requires
[Ollama](https://ollama.com) running locally with `nomic-embed-text` pulled
(`ollama pull nomic-embed-text`).

## Running it

```bash
uvicorn backend.server:app --reload --port 8000
```

Then open http://localhost:8000.

## Project structure

```
├── research/       # Government source inventory (spreadsheets)
├── scripts/        # Document extraction (scraping) script
├── backend/        # FastAPI server + vector store indexing script
├── frontend/       # Chat UI (served by the backend)
├── notebooks/      # Phase-by-phase pipeline build/validation notebook
├── reference/      # Reference course notebooks (LangChain/Groq/Ollama patterns)
├── data/           # Downloaded documents + Chroma vector store (gitignored)
└── requirements.txt
```
