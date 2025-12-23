import io
import os
import unicodedata
import asyncio
import time
import csv
from datetime import datetime, timedelta
from typing import Any, List, Optional, Dict, Literal
import re

from fastapi import (
    FastAPI,
    HTTPException,
    Depends,
    UploadFile,
    File,
    Header,
    Form,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from jose import jwt, JWTError
from passlib.context import CryptContext

from motor.motor_asyncio import AsyncIOMotorClient
import httpx
import pinecone
from pypdf import PdfReader

from config import settings

# ======================================================
# OPTIONAL PDF TOOLS (tables / OCR / figures)
# ======================================================
try:
    from pdf2image import convert_from_bytes
    import pdfplumber
    from PIL import Image
    import pytesseract
except Exception:
    convert_from_bytes = None
    pdfplumber = None
    Image = None
    pytesseract = None


# ======================================================
# FASTAPI APP
# ======================================================
app = FastAPI(title="CA RAG Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://ca-frontend-sooty.vercel.app",
        "http://localhost:3000",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ======================================================
# DATABASES & CLIENTS
# ======================================================
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

mongo_client = AsyncIOMotorClient(
    settings.MONGO_URI,
    tls=True,
    tlsAllowInvalidCertificates=True,
    serverSelectionTimeoutMS=5000,
)
db = mongo_client[settings.MONGO_DB]
users_collection = db["users"]
docs_collection = db["documents"]

pinecone_client = pinecone.Pinecone(api_key=settings.PINECONE_API_KEY)
index = pinecone_client.Index(settings.PINECONE_INDEX)

# ======================================================
# GLOBAL RAG CONFIG (MATCHES COLAB)
# ======================================================
RAG_NAMESPACE = "v1"
CURRENT_YEAR = 2025

JWT_EXP_MINUTES = 60 * 24

UPLOAD_ROOT = getattr(settings, "UPLOAD_ROOT", "./uploads")
os.makedirs(UPLOAD_ROOT, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_ROOT), name="uploads")


# ======================================================
# MODELS
# ======================================================
class UserCreate(BaseModel):
    email: str
    password: str
    role: str = "student"


class UserLogin(BaseModel):
    email: str
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserOut(BaseModel):
    email: str
    role: str


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: str
    history: Optional[List[ChatMessage]] = None
    mode: Optional[str] = "qa"


class ChatResponse(BaseModel):
    answer: str
    sources: List[dict]


class UploadResult(BaseModel):
    chunks: int
    filename: str


# ======================================================
# AUTH HELPERS
# ======================================================
def create_access_token(data: dict):
    expire = datetime.utcnow() + timedelta(minutes=JWT_EXP_MINUTES)
    data.update({"exp": expire})
    return jwt.encode(data, settings.JWT_SECRET, algorithm=settings.JWT_ALGO)


async def get_user_by_email(email: str):
    return await users_collection.find_one({"email": email})


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


async def get_current_user(authorization: str = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    token = authorization.replace("Bearer ", "").strip()
    try:
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGO])
        email = payload.get("sub")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

    user = await get_user_by_email(email)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


async def get_current_admin(user=Depends(get_current_user)):
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


# ======================================================
# NORMALIZATION HELPERS (CRITICAL)
# ======================================================
def sanitize_id(s: str, max_len: int = 200) -> str:
    nk = unicodedata.normalize("NFKD", s)
    ascii_str = nk.encode("ascii", "ignore").decode("ascii")
    ascii_str = re.sub(r"[^0-9A-Za-z]+", "_", ascii_str)
    ascii_str = re.sub(r"_+", "_", ascii_str).strip("_")
    return ascii_str[:max_len] or "id"


def normalize_level(course: str) -> str:
    course = (course or "").upper()
    if "FINAL" in course:
        return "final"
    if "INTER" in course:
        return "intermediate"
    if "FOUNDATION" in course:
        return "foundation"
    return "unknown"


def normalize_doc_type(raw: str) -> str:
    raw = (raw or "").lower()
    if raw in ["dynamic", "amendment", "circular", "notification"]:
        return "dynamic"
    return "static"


# ======================================================
# EMBEDDINGS & LLM
# ======================================================
async def embed_texts(texts: List[str]) -> List[List[float]]:
    async with httpx.AsyncClient(timeout=90) as client:
        r = await client.post(
            "https://api.openai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {settings.OPENAI_API_KEY}"},
            json={"model": "text-embedding-3-small", "input": texts},
        )
        r.raise_for_status()
        return [d["embedding"] for d in r.json()["data"]]


async def embed_single(text: str) -> List[float]:
    return (await embed_texts([text]))[0]


async def call_llm(messages: List[dict]) -> str:
    async with httpx.AsyncClient(timeout=90) as client:
        r = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {settings.OPENAI_API_KEY}"},
            json={
                "model": "gpt-4o-mini",
                "messages": messages,
                "temperature": 0.3,
            },
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


# ======================================================
# RE-RANKING LOGIC (EXAM SAFE)
# ======================================================
def rerank_matches(matches: list[dict]) -> list[dict]:
    reranked = []

    for m in matches:
        meta = m["metadata"]
        semantic = m["score"]

        level_boost = {
            "final": 1.0,
            "intermediate": 0.6,
            "foundation": 0.3,
        }.get(meta.get("level"), 0.1)

        doc_boost = 1.0 if meta.get("doc_type") == "dynamic" else 0.5

        year = int(meta.get("year", 0) or 0)
        year_boost = 1.0 if year >= CURRENT_YEAR else 0.6

        final_score = (
            semantic * 0.65
            + level_boost * 0.10
            + doc_boost * 0.15
            + year_boost * 0.10
        )

        reranked.append({**m, "final_score": final_score})

    return sorted(reranked, key=lambda x: x["final_score"], reverse=True)


# ======================================================
# CHAT ENDPOINT (RAG)
# ======================================================
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, user=Depends(get_current_user)):
    query_embedding = await embed_single(req.message)

    res = index.query(
        vector=query_embedding,
        top_k=12,
        include_metadata=True,
        namespace=RAG_NAMESPACE,
    )

    matches = res.get("matches", [])
    if not matches:
        return ChatResponse(
            answer="No relevant syllabus content found.",
            sources=[]
        )

    reranked = rerank_matches(matches)

    context_blocks = []
    sources = []
    total = 0
    MAX_CONTEXT_CHARS = 6000

    for m in reranked[:6]:
        meta = m["metadata"]
        block = f"[Source: {meta['source']} | Page {meta['page']}]\n{meta['text']}"
        if total + len(block) > MAX_CONTEXT_CHARS:
            break
        context_blocks.append(block)
        total += len(block)

        sources.append({
            "source": meta["source"],
            "page": meta["page"],
            "level": meta["level"],
            "subject": meta["subject"],
            "doc_type": meta["doc_type"],
        })

    system_prompt = (
        "You are an expert Indian CA tutor. "
        "Answer ONLY using the context below. "
        "Explain clearly in exam-oriented language. "
        "If the answer is not found, say so.\n\n"
        + "\n\n".join(context_blocks)
    )

    answer = await call_llm(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": req.message},
        ]
    )

    return ChatResponse(answer=answer, sources=sources)


# ======================================================
# ADMIN: PDF UPLOAD (FULLY ALIGNED WITH COLAB)
# ======================================================
@app.post("/admin/upload_pdf", response_model=UploadResult)
async def upload_pdf(
    file: UploadFile = File(...),
    metadata: Optional[str] = Form(None),
    admin=Depends(get_current_admin),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF allowed")

    import json
    doc_meta = json.loads(metadata) if metadata else {}

    course = doc_meta.get("course", "CA_FINAL")
    subject = doc_meta.get("subject", "general")
    year = doc_meta.get("year", CURRENT_YEAR)
    doc_type = normalize_doc_type(doc_meta.get("doc_type"))

    file_bytes = await file.read()
    reader = PdfReader(io.BytesIO(file_bytes))

    vectors = []
    chunk_count = 0

    for page_no, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue

        chunks = [text[i:i+1800] for i in range(0, len(text), 1600)]
        embeddings = await embed_texts(chunks)

        for chunk, emb in zip(chunks, embeddings):
            meta = {
                "text": chunk[:2000],
                "source": file.filename,
                "page": page_no,
                "level": normalize_level(course),
                "subject": subject,
                "doc_type": doc_type,
                "year": year,
                "authority": "ICAI",
                "ingested_at": datetime.utcnow().isoformat(),
            }

            vectors.append({
                "id": sanitize_id(f"{file.filename}_p{page_no}_{chunk_count}"),
                "values": emb,
                "metadata": meta,
            })
            chunk_count += 1

    for i in range(0, len(vectors), 100):
        index.upsert(vectors=vectors[i:i+100], namespace=RAG_NAMESPACE)

    await docs_collection.insert_one({
        "filename": file.filename,
        "level": normalize_level(course),
        "subject": subject,
        "doc_type": doc_type,
        "year": year,
        "uploaded_at": datetime.utcnow(),
        "chunks": chunk_count,
    })

    return UploadResult(chunks=chunk_count, filename=file.filename)


# ======================================================
# HEALTH
# ======================================================
@app.get("/health")
async def health():
    return {"status": "ok"}
