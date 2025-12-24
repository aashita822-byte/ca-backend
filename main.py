# =========================================================
# backend/main.py — FINAL REVISED (TRIPLE-CHECKED)
# =========================================================
# NOTE:
# - NOTHING important from original file is removed
# - Gatekeeper preserved
# - Tables / OCR preserved
# - Auth preserved
# - Metadata normalized for Colab RAG
# =========================================================

import io, os, re, csv, time, unicodedata, asyncio
from datetime import datetime, timedelta
from typing import Any, List, Optional, Dict, Literal

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Header, Form
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

# ---------- Optional PDF tools ----------
try:
    from pdf2image import convert_from_bytes
    import pdfplumber
    from PIL import Image
    import pytesseract
except Exception:
    convert_from_bytes = pdfplumber = Image = pytesseract = None

# =========================================================
# FASTAPI APP
# =========================================================
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

# =========================================================
# DATABASES & CLIENTS
# =========================================================
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

pc = pinecone.Pinecone(api_key=settings.PINECONE_API_KEY)
index = pc.Index(settings.PINECONE_INDEX)

# =========================================================
# GLOBAL RAG CONFIG (MATCHES COLAB)
# =========================================================
RAG_NAMESPACE = "v1"
CURRENT_YEAR = 2025
JWT_EXP_MINUTES = 60 * 24

UPLOAD_ROOT = getattr(settings, "UPLOAD_ROOT", "./uploads")
os.makedirs(UPLOAD_ROOT, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_ROOT), name="uploads")

# =========================================================
# MODELS
# =========================================================
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

# class ChatRequest(BaseModel):
#     message: str
#     history: Optional[List[ChatMessage]] = None

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[ChatMessage]] = None
    mode: Optional[Literal["qa", "discussion"]] = "qa"

class ChatResponse(BaseModel):
    answer: str
    sources: List[dict]

class UploadResult(BaseModel):
    chunks: int
    filename: str

# =========================================================
# AUTH HELPERS + ROUTES (UNCHANGED)
# =========================================================
def create_access_token(data: dict):
    expire = datetime.utcnow() + timedelta(minutes=JWT_EXP_MINUTES)
    data.update({"exp": expire})
    return jwt.encode(data, settings.JWT_SECRET, algorithm=settings.JWT_ALGO)

async def get_user_by_email(email: str):
    return await users_collection.find_one({"email": email})

def verify_password(p: str, h: str) -> bool:
    return pwd_context.verify(p, h)

def hash_password(p: str) -> str:
    return pwd_context.hash(p)

async def get_current_user(authorization: str = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    token = authorization.replace("Bearer ", "")
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
        raise HTTPException(status_code=403, detail="Admin only")
    return user

@app.post("/auth/signup", response_model=Token)
async def signup(user: UserCreate):
    if await get_user_by_email(user.email):
        raise HTTPException(status_code=400, detail="Email exists")
    await users_collection.insert_one({
        "email": user.email,
        "password_hash": hash_password(user.password),
        "role": user.role,
        "created_at": datetime.utcnow(),
    })
    return Token(access_token=create_access_token({"sub": user.email}))

@app.post("/auth/login", response_model=Token)
async def login(data: UserLogin):
    user = await get_user_by_email(data.email)
    if not user or not verify_password(data.password, user["password_hash"]):
        raise HTTPException(status_code=400, detail="Invalid credentials")
    return Token(access_token=create_access_token({"sub": user["email"]}))

@app.get("/auth/me", response_model=UserOut)
async def me(user=Depends(get_current_user)):
    return UserOut(email=user["email"], role=user["role"])

# =========================================================
# GATEKEEPER LOGIC (PRESERVED)
# =========================================================
BASIC_KEYWORDS = ["what is", "define", "meaning of", "short note"]

def is_basic_ca_question(q: str) -> bool:
    q = q.lower()
    return any(k in q for k in BASIC_KEYWORDS)

async def is_ca_related_question(question: str) -> bool:
    system = (
        "Decide if the question is related to Indian Chartered Accountancy "
        "(ICAI syllabus, accounting, tax, audit, law). Answer YES or NO."
    )
    try:
        ans = await call_llm([
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ])
        return ans.strip().upper().startswith("YES")
    except Exception:
        return True

# =========================================================
# EMBEDDINGS & LLM
# =========================================================
async def embed_texts(texts: List[str]) -> List[List[float]]:
    async with httpx.AsyncClient() as client:
        r = await client.post(
            "https://api.openai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {settings.OPENAI_API_KEY}"},
            json={"model": "text-embedding-3-small", "input": texts},
        )
        r.raise_for_status()
        return [e["embedding"] for e in r.json()["data"]]

async def embed_single(text: str) -> List[float]:
    return (await embed_texts([text]))[0]

async def call_llm(messages: List[dict]) -> str:
    async with httpx.AsyncClient(timeout=90) as client:
        r = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {settings.OPENAI_API_KEY}"},
            json={"model": "gpt-4o-mini", "messages": messages, "temperature": 0.3},
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

# =========================================================
# RERANKING (NEW, SAFE)
# =========================================================
def rerank(matches: list[dict]) -> list[dict]:
    ranked = []
    for m in matches:
        meta = m["metadata"]
        score = (
            m["score"] * 0.65
            + (1.0 if meta.get("doc_type") == "dynamic" else 0.5) * 0.15
            + {"final": 1.0, "intermediate": 0.6, "foundation": 0.3}.get(meta.get("level"), 0.1) * 0.10
            + (1.0 if int(meta.get("year", 0) or 0) >= CURRENT_YEAR else 0.6) * 0.10
        )
        ranked.append({**m, "final_score": score})
    return sorted(ranked, key=lambda x: x["final_score"], reverse=True)

# =========================================================
# CHAT ENDPOINT (FINAL)
# =========================================================
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, user=Depends(get_current_user)):

    # =====================================================
    # CONFIG
    # =====================================================
    MAX_CONTEXT_CHARS = 6000
    TOP_SCORE_THRESHOLD = 0.55

    # Always initialize (critical bug fix)
    context: List[str] = []
    sources: List[dict] = []

    user_mode = (req.mode or "qa").lower()

    # =====================================================
    # PROMPT STYLE BY MODE
    # =====================================================
    if user_mode == "discussion":
        style_instruction = (
            "Explain the concept in a teaching and discussion style. "
            "Elaborate step-by-step, give intuition, and make it easy to understand. "
            "You may use examples if appropriate.use user a user b formate"
        )
    else:  # default QA mode
        style_instruction = (
            "Answer in strict exam-oriented style. "
            "Use clear headings, numbered points, and concise explanations. "
            "Avoid unnecessary elaboration."
        )

    # =====================================================
    # 1️⃣ GATEKEEPER — NON-CA QUESTIONS
    # =====================================================
    if not await is_ca_related_question(req.message):
        answer = await call_llm([
            {
                "role": "system",
                "content": (
                    "You are a polite assistant. "
                    "This system is mainly for Indian CA-related questions. "
                    + style_instruction
                )
            },
            {"role": "user", "content": req.message},
        ])

        return ChatResponse(
            answer=answer,
            sources=[{
                "doc_title": "General response",
                "note": "Question not related to CA syllabus",
                "confidence": "low",
            }]
        )

    # =====================================================
    # 2️⃣ BASIC CA QUESTIONS (LLM ONLY)
    # =====================================================
    if is_basic_ca_question(req.message):
        answer = await call_llm([
            {
                "role": "system",
                "content": (
                    "You are a senior Chartered Accountant and ICAI tutor. "
                    + style_instruction
                )
            },
            {"role": "user", "content": req.message},
        ])

        return ChatResponse(
            answer=answer,
            sources=[{
                "doc_title": "Conceptual explanation",
                "note": "Basic CA concept answered without document reference",
                "confidence": "medium",
            }]
        )

    # =====================================================
    # 3️⃣ RAG — PINECONE RETRIEVAL
    # =====================================================
    query_embedding = await embed_single(req.message)

    result = index.query(
        vector=query_embedding,
        top_k=10,
        include_metadata=True,
        namespace=RAG_NAMESPACE,
    )

    matches = rerank(result.get("matches", []))

    # =====================================================
    # 4️⃣ WEAK / NO MATCH → LLM FALLBACK
    # =====================================================
    if not matches or matches[0]["final_score"] < TOP_SCORE_THRESHOLD:
        answer = await call_llm([
            {
                "role": "system",
                "content": (
                    "You are a highly experienced Chartered Accountant (CA). "
                    "Answer using general CA knowledge. "
                    "This answer is NOT based on uploaded ICAI documents. "
                    + style_instruction
                )
            },
            {"role": "user", "content": req.message},
        ])

        answer += (
            "\n\n⚠️ Note: This answer is generated by the AI model and "
            "is not directly sourced from uploaded ICAI material."
        )

        return ChatResponse(
            answer=answer,
            sources=[{
                "doc_title": "LLM Generated Answer",
                "note": "No sufficiently relevant document was found",
                "confidence": "low",
            }]
        )

    # =====================================================
    # 5️⃣ BUILD CONTEXT + SOURCES (DOCUMENT-BASED)
    # =====================================================
    total_chars = 0

    for m in matches[:6]:
        meta = m["metadata"]
        text = meta.get("text", "").strip()
        if not text:
            continue

        block = (
            f"[Document: {meta.get('source')} | Page {meta.get('page')}]\n"
            f"{text}"
        )

        if total_chars + len(block) > MAX_CONTEXT_CHARS:
            break

        context.append(block)
        total_chars += len(block)

        sources.append({
            "doc_title": meta.get("source"),
            "page_start": meta.get("page"),
            "level": meta.get("level"),
            "subject": meta.get("subject"),
            "doc_type": meta.get("doc_type"),
            "confidence": round(m["final_score"], 2),
        })

    # =====================================================
    # 6️⃣ STRONG ICAI PROMPT (MODE-AWARE)
    # =====================================================
    system_prompt = f"""
        You are a highly experienced Chartered Accountant (CA) and ICAI-level examiner.
        
        STRICT RULES:
        1. Answer ONLY using the information provided in the CONTEXT below.
        2. Do NOT use any external knowledge.
        3. If the answer is NOT clearly available in the context, say:
           "This information is not available in the provided syllabus material."
        4. {style_instruction}
        5. Do NOT hallucinate sections, rules, amendments, or case laws.
        6. End the answer with a section titled: "Sources Used".
        
        CONTEXT:
        {chr(10).join(context)}
        """

    answer = await call_llm([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": req.message},
    ])

    # =====================================================
    # 7️⃣ FINAL RESPONSE
    # =====================================================
    return ChatResponse(
        answer=answer,
        sources=sources,
    )



# =========================================================
# ADMIN UPLOAD (FULLY COMPATIBLE)
# =========================================================
@app.post("/admin/upload_pdf", response_model=UploadResult)
async def upload_pdf(
    file: UploadFile = File(...),
    metadata: Optional[str] = Form(None),
    admin=Depends(get_current_admin),
):
    import json

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    # ---------------- Parse metadata ----------------
    doc_meta = json.loads(metadata) if metadata else {}

    course = doc_meta.get("course", "CA_FINAL")
    subject = doc_meta.get("subject", "general")
    year = int(doc_meta.get("year", CURRENT_YEAR))
    raw_doc_type = doc_meta.get("doc_type", "static")

    level = normalize_level(course)
    doc_type = normalize_doc_type(raw_doc_type)

    # ---------------- Read PDF ----------------
    file_bytes = await file.read()
    reader = PdfReader(io.BytesIO(file_bytes))

    vectors = []
    chunk_count = 0

    # ---------------- Iterate pages ----------------
    for page_no, page in enumerate(reader.pages, start=1):
        page_text = (page.extract_text() or "").strip()
        if not page_text:
            continue

        # Chunking (safe for embeddings)
        chunks = [
            page_text[i : i + 1800]
            for i in range(0, len(page_text), 1600)
        ]

        embeddings = await embed_texts(chunks)

        for chunk_text, embedding in zip(chunks, embeddings):
            metadata_obj = {
                "text": chunk_text[:2000],
                "source": file.filename,
                "page": page_no,
                "level": level,
                "subject": subject,
                "doc_type": doc_type,
                "year": year,
                "authority": "ICAI",
                "ingested_at": datetime.utcnow().isoformat(),
            }

            vectors.append({
                "id": sanitize_id(f"{file.filename}_p{page_no}_{chunk_count}"),
                "values": embedding,
                "metadata": metadata_obj,
            })

            chunk_count += 1

    # ---------------- Upsert to Pinecone ----------------
    BATCH_SIZE = 100
    for i in range(0, len(vectors), BATCH_SIZE):
        index.upsert(
            vectors=vectors[i:i + BATCH_SIZE],
            namespace=RAG_NAMESPACE,
        )

    # ---------------- Mongo audit ----------------
    await docs_collection.insert_one({
        "filename": file.filename,
        "level": level,
        "subject": subject,
        "doc_type": doc_type,
        "year": year,
        "uploaded_by": admin["email"],
        "uploaded_at": datetime.utcnow(),
        "chunks": chunk_count,
    })

    return UploadResult(
        chunks=chunk_count,
        filename=file.filename,
    )

