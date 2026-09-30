"""
FastAPI backend wrapping the RAG chain built in notebooks/rag_pipeline.ipynb.
Serves the frontend at / and answers questions at POST /chat.

Run with:
    uvicorn backend.server:app --reload --port 8000
(from the project root, GenAI/)
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env", override=True)

if not os.environ.get("GROQ_API_KEY") or "placeholder" in os.environ.get("GROQ_API_KEY", ""):
    raise RuntimeError("GROQ_API_KEY is missing or still a placeholder in .env")

embeddings_model = OllamaEmbeddings(model="nomic-embed-text")

vector_store = Chroma(
    collection_name="india-traffic-rules",
    embedding_function=embeddings_model,
    persist_directory=str(BASE_DIR / "chroma_langchain_db"),
)

llm = init_chat_model("openai/gpt-oss-120b", model_provider="groq")

# States (well, states + UTs + "Central") actually indexed in the vector store - keep in
# sync with backend/build_index.py's SCOPE_FILES_WITH_STATE.
INDEXED_STATES = {"Central", "Gujarat", "Delhi"}  # top-up to the full set is pending (see backend/build_index.py)

strict_template = """You are a traffic-rules assistant for India. Answer the question using ONLY
the context below, which comes from official government Acts, Rules and regulations. Cite the
source document and its state (from the "Source:" / "State:" tags on each context chunk) in your
answer.

{coverage_note}

If the context does not contain enough information to answer, say clearly that this specific
knowledge base does not cover it - do NOT use outside knowledge or guess. Never present a rule
from one state's documents as if it applies to a different state.

Context:
{context}

Question: {question}
"""
strict_prompt = ChatPromptTemplate.from_template(strict_template)


def format_docs(docs):
    return "\n\n".join(
        f"Source: {d.metadata.get('source_file', 'unknown')}\nState: {d.metadata.get('state', 'unknown')}\n{d.page_content}"
        for d in docs
    )


def retrieve_for_state(question: str, state: str):
    """State-filtered retrieval with a Central-law fallback.

    Returns (docs, state_is_indexed). Central documents (the nationwide MV Act/Rules)
    are always included since they apply regardless of state; state-specific documents
    are added on top when that state is actually indexed.
    """
    central_docs = vector_store.as_retriever(
        search_kwargs={"k": 4, "filter": {"state": "Central"}}
    ).invoke(question)

    state_is_indexed = state in INDEXED_STATES and state != "Central"
    state_docs = []
    if state_is_indexed:
        state_docs = vector_store.as_retriever(
            search_kwargs={"k": 4, "filter": {"state": state}}
        ).invoke(question)

    seen = set()
    combined = []
    for d in state_docs + central_docs:
        key = (d.metadata.get("source_file"), d.page_content[:80])
        if key not in seen:
            seen.add(key)
            combined.append(d)

    return combined, (state_is_indexed or state == "Central")


contextualize_template = """Given the recent conversation history and a new message, rewrite the
new message as a fully self-contained question that includes whatever context from the history
is needed to understand it on its own (e.g. what violation, rule, or topic is being discussed).
Resolve references like "that", "it", or "this" using the history. If the new message is already
self-contained, return it unchanged. Do NOT answer the question - only rewrite it.

Conversation history (most recent last):
{history}

New message: "{message}"

Respond with ONLY the rewritten, self-contained question - no commentary.
"""
contextualize_prompt = ChatPromptTemplate.from_template(contextualize_template)
contextualize_chain = contextualize_prompt | llm | StrOutputParser()


def format_history(history: list[dict]) -> str:
    if not history:
        return "(none yet)"
    lines = []
    for turn in history:
        lines.append(f"User: {turn['question']}")
        lines.append(f"Assistant: {turn['answer']}")
    return "\n".join(lines)


def contextualize_question(question: str, history: list[dict]) -> str:
    if not history:
        return question
    try:
        return contextualize_chain.invoke({"history": format_history(history), "message": question}).strip()
    except Exception:
        return question


def answer_question(question: str, state: str, history: list[dict]) -> tuple[str, str]:
    """Returns (answer, contextualized_question) - the latter is what actually got retrieved/answered."""
    standalone_question = contextualize_question(question, history)

    docs, state_covered = retrieve_for_state(standalone_question, state)
    if state_covered:
        coverage_note = f"The user is in {state}. State-specific documents for {state} are included below where relevant, alongside the nationwide Central Motor Vehicles Act/Rules."
    else:
        coverage_note = (
            f"IMPORTANT: No documents for {state} specifically are indexed in this knowledge base. "
            f"The context below is ONLY the nationwide Central Motor Vehicles Act/Rules, plus possibly "
            f"other states' documents that happened to match semantically - those other states' rules "
            f"do NOT apply to {state} and must not be presented as if they do. Tell the user plainly "
            f"that {state}-specific rules aren't in this knowledge base yet, and only answer using the "
            f"Central (nationwide) provisions if any are relevant."
        )
    context = format_docs(docs)
    chain = strict_prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": standalone_question, "coverage_note": coverage_note})
    return answer, standalone_question

# Traffic rules vary by Indian state, so we need to know where the user is before
# answering. Session state is in-memory only (fine for a single local dev server;
# would need a real store - Redis, a DB row - for multiple server processes).
sessions: dict[str, dict] = {}

location_extraction_template = """Look at this message from someone asking about Indian traffic rules.

1. Does it mention (directly or indirectly) an Indian state, city, or 6-digit PIN code? If so,
   name the STATE it corresponds to (e.g. a mention of "Ahmedabad" or "380001" means "Gujarat").
   If no location is mentioned, answer NONE.
2. Is the user trying to find out something - a rule, a fine, a requirement, a consequence?
   This includes describing a situation or a violation even with NO question mark and NO
   interrogative phrasing (e.g. "I jumped a red light", "I wasn't wearing a seatbelt", "my bike
   has no insurance") - all of those imply "what happens / what's the rule for this" and count as
   YES. Only answer NO if the message is JUST a location and nothing else.

Examples:
- "navi mumbai" -> HAS_QUESTION: NO (just a location, nothing else)
- "I jumped a red light" -> HAS_QUESTION: YES (implies: what's the penalty for this)
- "my helmet is not ISI marked" -> HAS_QUESTION: YES (implies: is this legal / what's the fine)
- "what is the seatbelt fine" -> HAS_QUESTION: YES

Message: "{message}"

Respond in exactly this format, nothing else:
LOCATION: <state name or NONE>
HAS_QUESTION: <YES or NO>
"""
location_extraction_prompt = ChatPromptTemplate.from_template(location_extraction_template)
location_extraction_chain = location_extraction_prompt | llm | StrOutputParser()


def extract_location_and_intent(message: str) -> tuple[str | None, bool]:
    try:
        raw = location_extraction_chain.invoke({"message": message})
        location = None
        has_question = True
        for line in raw.splitlines():
            line = line.strip()
            if line.upper().startswith("LOCATION:"):
                value = line.split(":", 1)[1].strip()
                location = None if value.upper() == "NONE" else value
            elif line.upper().startswith("HAS_QUESTION:"):
                has_question = line.split(":", 1)[1].strip().upper().startswith("Y")
        return location, has_question
    except Exception:
        # if extraction fails for any reason, fall back to "treat as a real question,
        # no location found" rather than blocking the user entirely
        return None, True


app = FastAPI(title="India Traffic Rules RAG API")


class ChatRequest(BaseModel):
    message: str
    session_id: str


class ChatResponse(BaseModel):
    response: str


MAX_HISTORY_TURNS = 4


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    session = sessions.setdefault(req.session_id, {"location": None, "pending_question": None, "history": []})

    location, has_question = extract_location_and_intent(req.message)
    if location:
        session["location"] = location

    if not session["location"]:
        # Still don't know where they are. If this message was a real question, remember it so
        # we can answer it automatically once the location comes in, instead of making them
        # repeat themselves.
        if has_question:
            session["pending_question"] = req.message
        return ChatResponse(
            response="Which state, city, or PIN code are you asking about? Traffic rules vary "
                     "by state in India, so let me know your location and I'll tailor the answer."
        )

    question_to_answer = None
    if has_question:
        session["pending_question"] = None
        question_to_answer = req.message
    elif session["pending_question"]:
        question_to_answer = session["pending_question"]
        session["pending_question"] = None

    if question_to_answer is None:
        return ChatResponse(
            response=f"Got it — I'll use {session['location']} for your questions. "
                     f"What would you like to know about traffic rules there?"
        )

    answer, standalone_question = answer_question(question_to_answer, session["location"], session["history"])
    session["history"].append({"question": standalone_question, "answer": answer})
    session["history"] = session["history"][-MAX_HISTORY_TURNS:]
    return ChatResponse(response=answer)


@app.get("/health")
def health():
    return {"status": "ok"}


FRONTEND_DIR = BASE_DIR / "frontend"


@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
