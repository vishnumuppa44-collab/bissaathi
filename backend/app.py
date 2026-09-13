import json
import os
import re
from pathlib import Path
from typing import Any

import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from pypdf import PdfReader


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DATA_DIR = BASE_DIR / "data"
STORE_PATH = BASE_DIR / "vector_store.json"
FRONTEND_DIR = PROJECT_DIR / "frontend"
load_dotenv(PROJECT_DIR / ".env")


app = FastAPI(title="BIS Saathi API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins (fine for demo; restrict later if needed)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    language: str = Field(default="en", pattern="^(en|hi|te)$")
    history: list[dict[str, str]] = Field(default_factory=list, max_length=12)


class ChatResponse(BaseModel):
    answer: str
    citations: list[dict[str, Any]]
    language: str


def extract_file(path: Path) -> str:
    try:
        if path.suffix.lower() == ".pdf":
            reader = PdfReader(str(path))
            return "\n".join((page.extract_text() or "") for page in reader.pages)
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def chunk_text(text: str, size: int = 950, overlap: int = 180) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks, start = [], 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            cut = max(text.rfind(". ", start, end), text.rfind(" ", start, end))
            if cut > start + size // 2:
                end = cut + 1
        part = text[start:end].strip()
        if part:
            chunks.append(part)
        start = max(end - overlap, start + 1)
    return chunks


def build_index() -> dict[str, Any]:
    DATA_DIR.mkdir(exist_ok=True)
    records = []
    supported = {".pdf", ".txt", ".md"}
    for file in sorted(DATA_DIR.rglob("*")):
        if file.is_file() and file.suffix.lower() in supported:
            text = extract_file(file)
            for i, chunk in enumerate(chunk_text(text)):
                records.append({"id": f"{file.name}-{i+1}", "source": file.name, "chunk": i + 1, "text": chunk})
    data = {"records": records}
    STORE_PATH.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data


def load_index() -> dict[str, Any]:
    if not STORE_PATH.exists():
        return build_index()
    try:
        return json.loads(STORE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return build_index()


def token_set(value: str) -> set[str]:
    return set(re.findall(r"[\w\u0900-\u097F\u0C00-\u0C7F]+", value.lower()))


def retrieve(query: str, limit: int = 5) -> list[dict[str, Any]]:
    q_tokens = token_set(query)
    scored = []
    for record in load_index().get("records", []):
        text = record["text"].lower()
        score = sum(1 for token in q_tokens if len(token) > 2 and token in text)
        if score:
            scored.append((score, record))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scored[:limit]]


def model_or_error():
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(status_code=503, detail="AI is not configured. Add GEMINI_API_KEY to .env, then restart the server.")
    genai.configure(api_key=api_key)
    # Use explicit model path
    return genai.GenerativeModel("models/gemini-1.5-flash")


LANG = {"en": "English", "hi": "Hindi", "te": "Telugu"}


@app.get("/")
def home():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/health")
def health():
    index = load_index()
    return {
        "status": "ok",
        "documents_indexed": len({r["source"] for r in index.get("records", [])}),
        "chunks_indexed": len(index.get("records", [])),
        "ai_configured": bool(os.getenv("GEMINI_API_KEY", "").strip())
    }


@app.post("/api/admin/reindex")
def reindex(x_admin_token: str | None = Header(default=None)):
    expected = os.getenv("ADMIN_TOKEN", "").strip()
    if expected and x_admin_token != expected:
        raise HTTPException(status_code=401, detail="Invalid admin token")
    data = build_index()
    return {"ok": True, "chunks_indexed": len(data["records"]), "documents_indexed": len({r["source"] for r in data["records"]})}


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    docs = retrieve(request.message)
    if not docs:
        return ChatResponse(answer={
            "en": "I could not find a verified answer in the currently indexed BIS documents. Please add authorised documents or check the official BIS website for the latest information.",
            "hi": "मुझे वर्तमान में अनुक्रमित BIS दस्तावेज़ों में सत्यापित उत्तर नहीं मिला। कृपया अधिकृत दस्तावेज़ जोड़ें या नवीनतम जानकारी के लिए आधिकारिक BIS वेबसाइट देखें।",
            "te": "ప్రస్తుతం ఇండెక్స్ చేసిన BIS పత్రాలలో ధృవీకరించిన సమాధానం దొరకలేదు. దయచేసి అధీకృత పత్రాలను జోడించండి లేదా తాజా సమాచారం కోసం అధికారిక BIS వెబ్‌సైట్‌ను చూడండి."
        }[request.language], citations=[], language=request.language)

    context = "\n\n".join(f"[S{i+1}] Source: {d['source']} | Chunk: {d['chunk']}\n{d['text']}" for i, d in enumerate(docs))
    prior = "\n".join(f"{m.get('role','user')}: {m.get('content','')}" for m in request.history[-8:])

    system = f"""You are BIS Saathi, a careful assistant for Bureau of Indian Standards information. Answer in {LANG[request.language]}. Use ONLY the supplied source excerpts for factual claims. If the excerpts do not answer a point, clearly say you cannot verify it from the available documents. Do not invent Indian Standard numbers, clauses, deadlines, fees, certification eligibility, URLs, or official procedures. Be concise, practical, and polite. Mention source labels such as [S1] only when support exists. State that users should verify current requirements with official BIS sources when giving process guidance."""

    user = f"""User question: {request.message}\n\nConversation context:\n{prior or '(none)'}\n\nVerified retrieved excerpts:\n{context}"""

    try:
        model = model_or_error()
        response = model.generate_content(
            contents=[system, user],
            generation_config=genai.types.GenerationConfig(
                temperature=0.15
            )
        )
        answer = response.text or "I could not produce an answer."
    except Exception as e:
        # Avoid 500 without CORS; return a normal error response
        raise HTTPException(status_code=502, detail=f"AI service error: {e}")

    citations = [{"label": f"S{i+1}", "source": d["source"], "chunk": d["chunk"], "excerpt": d["text"][:380] + ("…" if len(d["text"]) > 380 else "")} for i, d in enumerate(docs)]
    return ChatResponse(answer=answer, citations=citations, language=request.language)