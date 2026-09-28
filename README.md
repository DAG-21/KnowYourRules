# India Traffic Rules RAG Chatbot

A RAG-backed chatbot for looking up Indian traffic rules and regulations, sourced from
official government documents across the full regulatory hierarchy: central (MoRTH,
statutory/technical bodies), state transport departments and road-safety authorities,
union territories, city/local traffic police, and district-level orders.

This is an independent project, separate from any other repository in this workspace.

## Status

Research phase complete. See [`research/`](research/) for the compiled inventory of
official government sources:

- `india_traffic_document_links.xlsx` — 125 verified/discovered document-source links
  across the hierarchy, with a verification status per row (live-confirmed, needs
  special TLS handling, unconfirmed, or excluded-transactional). This is the working
  target list for the document-extraction step.
- `india_traffic_gov_websites.xlsx` — earlier, superseded site-level inventory (kept
  for reference).

Not yet started: document downloading/extraction, chunking, embeddings, vector store,
retrieval, generation, and the chatbot frontend. Architecture choices for those stages
(vector DB, embedding model, LLM provider, backend/frontend framework) haven't been
made yet.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

## Project structure

```
GenAI/
├── research/        # Government source inventories (spreadsheets)
├── data/
│   ├── raw/          # Downloaded source documents (gitignored - regenerable)
│   └── processed/    # Cleaned/chunked output (gitignored - regenerable)
├── requirements.txt
└── README.md
```

## Next step

Build the document-extraction script that walks the "Verified - Live" rows in
`research/india_traffic_document_links.xlsx` and downloads the source PDFs/pages into
`data/raw/`.
