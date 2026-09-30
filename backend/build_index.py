"""
Rebuilds the Chroma vector store with CORRECT state metadata on every chunk.

Fixes a real bug: two files were downloaded via other states' pages and kept those
states' source-name prefixes even though their actual content is the Central Motor
Vehicles Act/Rules (applies nationwide, not state-specific). Without a correct
"state" tag, retrieval had no way to filter by state and the LLM was citing the
Central Act as if it were Madhya Pradesh state law.

Run with:
    .venv/Scripts/python.exe backend/build_index.py
(from the project root, GenAI/)
"""

import time
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env", override=True)

CORPUS = BASE_DIR / "data" / "raw" / "full_extraction"

# (relative path, correct state) - the two "Central" entries are the files that were
# discovered via Madhya Pradesh's and Andaman & Nicobar's pages but are actually the
# nationwide Motor Vehicles Act/Rules, not state-specific content.
SCOPE_FILES_WITH_STATE = [
    ("state/madhya-pradesh-mv-rules-1994-direct-pdf__a1988-59-b740e529.pdf", "Central"),
    ("union-territory/andaman-nicobar-islands-transport-acts-r__the-central-motor-vehicles-rules-1989-14ddf947.pdf", "Central"),
    ("state/gujarat-commissionerate-of-transport-act__gmv-rules-1989-a99d3b52.pdf", "Gujarat"),
    ("state/gujarat-commissionerate-of-transport-act__bmv-taact-1958-ba51970a.pdf", "Gujarat"),
    ("state/gujarat-commissionerate-of-transport-act__bmv-tax-rule-1959-cb64db21.pdf", "Gujarat"),
    ("union-territory/delhi-nct-transport-acts-and-rules-notif__delhi-motor-vehicles-rules-1993-as-amended-secreta-52c5b4c7.pdf", "Delhi"),
    ("union-territory/delhi-nct-transport-acts-and-rules-notif__the-delhi-motor-vehicle-taxation-act-r-2af515c1.pdf", "Delhi"),
    ("union-territory/delhi-nct-transport-acts-and-rules-notif__mact-6a89af46.pdf", "Delhi"),
    ("state/mizoram-act-rules__the-mizoram-motor-vehicle-taxation-act1996-4d1a51ab.pdf", "Mizoram"),
    ("state/mizoram-act-rules__online-special-road-permit-dil-dan-kimchang-1-8ac37036.pdf", "Mizoram"),
    ("state/mizoram-act-rules__sop-for-vlteas-1893dd12.pdf", "Mizoram"),
    ("state-road-safety-authority-cell/himachal-pradesh-road-safety-cell__action-plan-2024-25-beaf0f7b.pdf", "Himachal Pradesh"),
    ("union-territory/andaman-nicobar-islands-transport-acts-r__an-motor-aggregator-scheme-dca46278.pdf", "Andaman & Nicobar"),
    ("union-territory/andaman-nicobar-islands-transport-acts-r__finedet-e27a23fe.pdf", "Andaman & Nicobar"),
    ("central-statutory-technical-body/cpcb-vehicular-exhaust-emission-standard__it-technical-activity-2108d906.pdf", "Central"),
] + [
    (f"state/{p.name}", "Haryana")
    for p in (CORPUS / "state").glob("haryana-hartrans-gov-in-acts-rules__*.pdf")
]

print(f"{len(SCOPE_FILES_WITH_STATE)} files to index", flush=True)

all_docs = []
for rel_path, state in SCOPE_FILES_WITH_STATE:
    f = CORPUS / rel_path
    try:
        loader = UnstructuredPDFLoader(str(f), strategy="fast", mode="single")
        docs = loader.load()
        for d in docs:
            d.metadata["source_file"] = f.name
            d.metadata["state"] = state
        all_docs.extend(docs)
        print(f"  [{state}] {f.name} ({len(docs[0].page_content)} chars)", flush=True)
    except Exception as e:
        print(f"  [{state}] {f.name} FAILED: {e}", flush=True)
    except BaseException as e:
        print(f"  [{state}] {f.name} FATAL: {e}", flush=True)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
all_splits = text_splitter.split_documents(all_docs)
print(f"\nSplit {len(all_docs)} documents into {len(all_splits)} chunks.", flush=True)

embeddings_model = OllamaEmbeddings(model="nomic-embed-text")

vector_store = Chroma(
    collection_name="india-traffic-rules",
    embedding_function=embeddings_model,
    persist_directory=str(BASE_DIR / "chroma_langchain_db"),
)

# Clear out the old (mislabeled) collection contents before rebuilding
existing = vector_store.get(include=[])
if existing["ids"]:
    print(f"Clearing {len(existing['ids'])} existing chunks...", flush=True)
    vector_store.delete(ids=existing["ids"])

BATCH_SIZE = 50
uuids = [str(uuid4()) for _ in range(len(all_splits))]
for i in range(0, len(all_splits), BATCH_SIZE):
    batch_docs = all_splits[i:i + BATCH_SIZE]
    batch_ids = uuids[i:i + BATCH_SIZE]
    for attempt in range(3):
        try:
            vector_store.add_documents(documents=batch_docs, ids=batch_ids)
            break
        except Exception as e:
            print(f"  batch {i}-{i+len(batch_docs)} attempt {attempt+1} failed: {e}", flush=True)
            time.sleep(2)
    else:
        print(f"  batch {i}-{i+len(batch_docs)} FAILED after 3 attempts - skipped", flush=True)
    if i % 500 == 0:
        print(f"embedded {i}/{len(all_splits)}", flush=True)

print(f"\nDone. Indexed {len(all_splits)} chunks with correct state metadata.", flush=True)
