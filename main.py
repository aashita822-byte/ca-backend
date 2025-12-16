# backend/main.py
import io
import os
import unicodedata
import asyncio
import time
import csv
from datetime import datetime, timedelta
from typing import Any, List, Optional, Dict
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
from typing import Literal

# Optional libs for table/chart extraction + OCR
# Make sure to pip install: pdf2image, pdfplumber, pillow, pytesseract
try:
    from pdf2image import convert_from_bytes
    import pdfplumber
    from PIL import Image
    import pytesseract
except Exception:
    # If packages aren't installed, we still allow server to start; upload will raise helpful errors
    convert_from_bytes = None
    pdfplumber = None
    Image = None
    pytesseract = None

# ---------- FastAPI app & CORS ----------
app = FastAPI(title="CA RAG Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://ca-frontend-sooty.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


# ---------- DB & external clients ----------
pwd_context = CryptContext(
    schemes=["pbkdf2_sha256"],
    default="pbkdf2_sha256",
    deprecated="auto",
)

mongo_client = AsyncIOMotorClient(settings.MONGO_URI)
db = mongo_client[settings.MONGO_DB]
users_collection = db["users"]
docs_collection = db["documents"]

pinecone_client = pinecone.Pinecone(api_key=settings.PINECONE_API_KEY)
index = pinecone_client.Index(settings.PINECONE_INDEX)

JWT_EXP_MINUTES = 60 * 24

# Embedding configuration (tweak in config if desired)
EMBED_BATCH_SIZE = getattr(settings, "EMBED_BATCH_SIZE", 12)
EMBED_TIMEOUT_SECS = getattr(settings, "EMBED_TIMEOUT_SECS", 120)
EMBED_MAX_RETRIES = getattr(settings, "EMBED_MAX_RETRIES", 3)
EMBED_BACKOFF_BASE = getattr(settings, "EMBED_BACKOFF_BASE", 1.8)
MAX_TEXT_LENGTH_FOR_EMBED = getattr(settings, "MAX_TEXT_LENGTH_FOR_EMBED", 8000)

# Storage paths (local). Change to S3/presigned URLs if needed.
UPLOAD_ROOT = getattr(settings, "UPLOAD_ROOT", "./uploads")
os.makedirs(UPLOAD_ROOT, exist_ok=True)

app.mount("/uploads", StaticFiles(directory=UPLOAD_ROOT), name="uploads")

# ---------- Models ----------
class UserCreate(BaseModel):
    email: str
    password: str
    role: str = "student"  # student or admin


class UserLogin(BaseModel):
    email: str
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserOut(BaseModel):
    email: str
    role: str


class ChatResponse(BaseModel):
    answer: str
    sources: List[dict]


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: str
    history: Optional[List[ChatMessage]] = None
    mode: Optional[str] = "qa"  # "qa" | "discussion"


class UploadResult(BaseModel):
    chunks: int
    filename: str


# ---------- Utility functions ----------
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=JWT_EXP_MINUTES))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(
        to_encode, settings.JWT_SECRET, algorithm=settings.JWT_ALGO
    )
    return encoded_jwt


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
        payload = jwt.decode(
            token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGO]
        )
        email: str = payload.get("sub")
        if email is None:
            raise HTTPException(status_code=401, detail="Invalid token payload")
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


# ---------- ID sanitization helper ----------
def sanitize_id(s: str, max_len: int = 200) -> str:
    """
    Convert a string to an ASCII-safe id suitable for Pinecone vector IDs.
    - Normalize unicode to NFKD and drop non-ascii.
    - Replace non-alnum characters with underscores.
    - Collapse repeated underscores and trim.
    - Truncate to max_len.
    """
    if not s:
        return "id"
    nk = unicodedata.normalize("NFKD", s)
    ascii_bytes = nk.encode("ascii", "ignore")
    ascii_str = ascii_bytes.decode("ascii")
    replaced = re.sub(r"[^0-9A-Za-z]+", "_", ascii_str)
    collapsed = re.sub(r"_+", "_", replaced).strip("_")
    if not collapsed:
        collapsed = "id"
    if len(collapsed) > max_len:
        collapsed = collapsed[:max_len].rstrip("_")
    return collapsed


# ---------- Embedding & LLM (robust) ----------
async def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Robust embedding call:
      - Splits into smaller batches (EMBED_BATCH_SIZE)
      - Retries transient failures with exponential backoff
      - Uses EMBED_TIMEOUT_SECS per request
    Returns embeddings in same order as `texts`.
    """
    if not texts:
        return []

    url = "https://openrouter.ai/api/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    # split into batches preserving order
    batches = [texts[i : i + EMBED_BATCH_SIZE] for i in range(0, len(texts), EMBED_BATCH_SIZE)]
    results: List[List[float]] = []

    async with httpx.AsyncClient(timeout=EMBED_TIMEOUT_SECS) as client:
        for batch_idx, batch_texts in enumerate(batches):
            payload = {"model": settings.EMBEDDING_MODEL, "input": batch_texts}
            attempt = 0
            while True:
                attempt += 1
                try:
                    resp = await client.post(url, headers=headers, json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                    emb_batch = [d["embedding"] for d in data["data"]]
                    if len(emb_batch) != len(batch_texts):
                        raise HTTPException(status_code=502, detail="Embedding response length mismatch")
                    results.extend(emb_batch)
                    break
                except (httpx.ReadTimeout, httpx.WriteTimeout, httpx.ConnectError, httpx.RemoteProtocolError) as exc:
                    # network/transient errors -> retry
                    if attempt >= EMBED_MAX_RETRIES:
                        raise HTTPException(
                            status_code=504,
                            detail=f"Embedding request timed out after {EMBED_MAX_RETRIES} attempts (batch {batch_idx}). Last error: {str(exc)}",
                        )
                    backoff = EMBED_BACKOFF_BASE ** (attempt - 1)
                    jitter = (0.1 * backoff) * (0.5 - (time.time() % 1))
                    await asyncio.sleep(backoff + jitter)
                    continue
                except httpx.HTTPStatusError as exc:
                    status = exc.response.status_code
                    text = ""
                    try:
                        text = exc.response.text
                    except Exception:
                        pass
                    # Retry on 5xx
                    if 500 <= status < 600 and attempt < EMBED_MAX_RETRIES:
                        await asyncio.sleep(EMBED_BACKOFF_BASE ** (attempt - 1))
                        continue
                    raise HTTPException(
                        status_code=502,
                        detail=f"Embedding service returned status {status}: {text[:200]}",
                    )
                except Exception as exc:
                    raise HTTPException(status_code=500, detail=f"Embedding failed: {str(exc)}")

    return results


async def embed_single(text: str) -> List[float]:
    embs = await embed_texts([text])
    return embs[0]


async def call_llm(messages: List[dict]) -> str:
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": settings.LLM_MODEL,
        "messages": messages,
    }
    async with httpx.AsyncClient(timeout=90) as client:
        resp = await client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


# ---------- Helpers ----------
def chunk_text(text: str, chunk_size: int = 800, overlap: int = 200) -> List[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks


def extract_pdf_text(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    full_text = ""
    for page in reader.pages:
        page_text = page.extract_text() or ""
        full_text += page_text + "\n"
    return full_text


def extract_pdf_pages(file_bytes: bytes) -> List[str]:
    """Return a list of raw text per page."""
    reader = PdfReader(io.BytesIO(file_bytes))
    pages: List[str] = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        pages.append(page_text)
    return pages


def normalize_page_text(text: str) -> str:
    """Clean up page text a bit."""
    # remove extra spaces, keep line breaks
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]  # drop empty
    return "\n".join(lines)


def detect_headings_for_page(page_text: str) -> Dict[str, Optional[str]]:
    lines = [ln.strip() for ln in page_text.splitlines() if ln.strip()]
    chapter = None
    topic = None

    for ln in lines:
        lower = ln.lower()
        if chapter is None and (
            lower.startswith("chapter ")
            or lower.startswith("chap. ")
            or lower.startswith("paper ")
            or "paper -" in lower
        ):
            chapter = ln.strip()
            continue

        if topic is None:
            if len(ln) <= 80 and ln.upper() == ln and any(c.isalpha() for c in ln):
                topic = ln.title().strip()

        if chapter and topic:
            break

    return {"chapter": chapter, "topic": topic}


def chunk_text_words(text: str, chunk_size: int = 180, overlap: int = 40) -> List[str]:
    words = text.split()
    chunks: List[str] = []
    start = 0
    n = len(words)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        if end == n:
            break
        start = end - overlap
        if start < 0:
            start = 0
    return chunks


def is_basic_ca_question(question: str) -> bool:
    q = question.lower().strip()

    BASIC_QUESTIONS = [
        "what is interim",
        "what is interim rules",
        "what is interim reporting",
        "define interim",
        "meaning of interim",
        "what is ind as 34",
        "interim financial reporting",
        "what is ca",
        "what is ca final",
    ]

    BASIC_KEYWORDS = [
        "accounting",
        "audit",
        "tax",
        "gst",
        "ind as",
        "financial reporting",
        "law",
    ]

    return any(b in q for b in BASIC_QUESTIONS) or any(k in q for k in BASIC_KEYWORDS)



def detect_query_focus(question: str) -> str:
    q = (question or "").lower()
    has_table = any(w in q for w in ["table", "tabular", "sheet", "schedule"])
    has_figure = any(w in q for w in ["chart", "diagram", "figure", "flowchart", "graph", "plot", "flow chart", "flow"])
    if has_table and has_figure:
        return "mixed"
    if has_table:
        return "table"
    if has_figure:
        return "figure"
    return "text"


async def is_ca_related_question(question: str) -> bool:
    if is_basic_ca_question(question):
        return True

    system = (
        "You are a classifier. Decide if the user question is even loosely "
        "related to Chartered Accountancy in India (ICAI syllabus), accounting, taxation, finance, law, or auditing. "
        "Answer with YES or NO only."
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]

    try:
        result = await call_llm(messages)
        return result.strip().upper().startswith("YES")
    except Exception:
        return True


# ---------- Charts / Tables extraction utilities ----------
def ensure_doc_upload_folder(filename: str):
    safe = sanitize_id(filename, max_len=80)
    folder = os.path.join(UPLOAD_ROOT, safe)
    os.makedirs(folder, exist_ok=True)
    return folder


def save_image_thumbnail(img: Image.Image, out_path: str, max_width: int = 800):
    """Save thumbnail for the PIL Image object, maintain aspect ratio."""
    if img is None:
        return
    w, h = img.size
    if w > max_width:
        ratio = max_width / float(w)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    img.save(out_path, format="PNG")


def ocr_text_from_image(img: Image.Image) -> str:
    if pytesseract is None:
        return ""
    try:
        txt = pytesseract.image_to_string(img, lang="eng")
        return txt.strip()
    except Exception:
        return ""


def save_table_csv(table_rows: List[List[Any]], out_path: str):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for row in table_rows:
            # convert None to "" and ensure str values
            writer.writerow([("" if c is None else str(c)) for c in row])


def extract_tables_pdfplumber_from_bytes(file_bytes: bytes, out_dir: str) -> List[Dict[str, Any]]:
    """
    Use pdfplumber to extract tables. Returns list of metas:
      {"page": int, "csv_path": str, "excerpt": str}
    """
    if pdfplumber is None:
        raise HTTPException(status_code=500, detail="pdfplumber not installed on server")

    metas: List[Dict[str, Any]] = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for p_index, page in enumerate(pdf.pages, start=1):
            try:
                tables = page.extract_tables()
            except Exception:
                tables = []
            for t_idx, table in enumerate(tables):
                if not table or not any(any(cell for cell in row) for row in table):
                    continue
                csv_name = f"table_p{p_index}_{t_idx}.csv"
                csv_path = os.path.join(out_dir, csv_name)
                save_table_csv(table, csv_path)
                # build a short excerpt (header + first row or first 2 rows)
                headers = table[0] if table and len(table) > 0 else []
                sample_rows = []
                if len(table) > 1:
                    sample_rows = table[1:3]
                excerpt_parts = []
                if headers:
                    excerpt_parts.append(" | ".join([str(h) for h in headers[:6]]))
                for r in sample_rows:
                    excerpt_parts.append(" ; ".join([str(c) for c in (r[:6] if r else [])]))
                excerpt = " | ".join([p for p in excerpt_parts if p])
                metas.append({"page": p_index, "csv_path": csv_path, "excerpt": excerpt, "table_index": t_idx})
    return metas


def save_page_images_and_thumbs(file_bytes: bytes, out_dir: str, dpi: int = 200) -> List[Dict[str, Any]]:
    """
    Save page images and thumbnails. Returns list of page metas:
     {"page": n, "image_path": path, "thumb_path": thumb_path}
    """
    if convert_from_bytes is None or Image is None:
        raise HTTPException(status_code=500, detail="pdf2image / pillow not installed on server")

    pages = convert_from_bytes(file_bytes, dpi=dpi)
    metas: List[Dict[str, Any]] = []
    for idx, pil_img in enumerate(pages, start=1):
        img_name = f"page_{idx}.png"
        img_path = os.path.join(out_dir, img_name)
        pil_img.save(img_path, format="PNG")
        thumb_name = f"thumb_p{idx}.png"
        thumb_path = os.path.join(out_dir, thumb_name)
        save_image_thumbnail(pil_img, thumb_path, max_width=900)
        metas.append({"page": idx, "image_path": img_path, "thumb_path": thumb_path})
    return metas


# ---------- Auth routes ----------
@app.post("/auth/signup", response_model=Token)
async def signup(user: UserCreate):
    existing = await get_user_by_email(user.email)
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    await users_collection.insert_one(
        {
            "email": user.email,
            "password_hash": hash_password(user.password),
            "role": user.role,
            "created_at": datetime.utcnow(),
        }
    )
    token = create_access_token({"sub": user.email})
    return Token(access_token=token)


@app.post("/auth/login", response_model=Token)
async def login(data: UserLogin):
    user = await get_user_by_email(data.email)
    if not user or not verify_password(data.password, user["password_hash"]):
        raise HTTPException(status_code=400, detail="Invalid credentials")
    token = create_access_token({"sub": user["email"]})
    return Token(access_token=token)


@app.get("/auth/me", response_model=UserOut)
async def me(user=Depends(get_current_user)):
    return UserOut(email=user["email"], role=user["role"])


# ---------- Chat (RAG) ----------
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, user=Depends(get_current_user)):
    try:
        # --------------------------------------------------
        # 1. CA gatekeeper
        # --------------------------------------------------
        is_ca = await is_ca_related_question(req.message)
        if not is_ca:
            return ChatResponse(
                answer=(
                    "This assistant is designed for Indian CA students. "
                    "Please ask a question related to accounting, tax, audit, law, "
                    "or CA exams (Foundation / Inter / Final)."
                ),
                sources=[],
            )

        # --------------------------------------------------
        # 2. BASIC QUESTION → LLM ONLY
        # --------------------------------------------------
        if is_basic_ca_question(req.message):
            system_prompt = (
                "You are a friendly Indian CA tutor. "
                "Explain clearly in simple language for CA students. "
                "Keep it concise and exam-oriented."
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": req.message},
            ]

            answer = await call_llm(messages)

            return ChatResponse(
                answer=answer,
                sources=[
                    {
                        "doc_title": "Conceptual explanation",
                        "note": "General CA concept explained by the assistant",
                    }
                ],
            )

        # --------------------------------------------------
        # 3. RAG FLOW (Pinecone)
        # --------------------------------------------------
        query_embedding = await embed_single(req.message)
        namespace = "CA_FINAL"

        res = index.query(
            vector=query_embedding,
            top_k=8,
            include_metadata=True,
            namespace=namespace,
        )

        matches = res.get("matches") or []

        if not matches:
            system_prompt = (
                "You are an Indian CA tutor. "
                "No document source is available. "
                "Answer using your CA knowledge and clearly say no source was matched."
            )

            answer = await call_llm(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": req.message},
                ]
            )

            return ChatResponse(
                answer=answer,
                sources=[
                    {
                        "doc_title": "LLM only",
                        "note": "No uploaded document matched",
                    }
                ],
            )

        # --------------------------------------------------
        # 4. BUILD CONTEXT (SAFE SIZE)
        # --------------------------------------------------
        context_blocks = []
        sources = []

        matches = sorted(matches, key=lambda m: m.get("score", 0), reverse=True)

        for m in matches:
            meta = m.get("metadata", {})
            text = meta.get("text", "")
            if not text:
                continue

            header = []
            if meta.get("doc_title"):
                header.append(f"Document: {meta['doc_title']}")
            if meta.get("chapter"):
                header.append(f"Chapter: {meta['chapter']}")
            if meta.get("topic"):
                header.append(f"Topic: {meta['topic']}")
            if meta.get("page_start"):
                header.append(f"Page: {meta['page_start']}")

            block = f"{' | '.join(header)}\n{text}"
            context_blocks.append(block)

            sources.append(
                {
                    "doc_title": meta.get("doc_title"),
                    "page_start": meta.get("page_start"),
                    "chapter": meta.get("chapter"),
                    "topic": meta.get("topic"),
                    "type": meta.get("type", "text"),
                }
            )

        # ---- HARD CONTEXT LIMIT (IMPORTANT)
        MAX_CONTEXT_CHARS = 6000
        trimmed = []
        total = 0
        for b in context_blocks:
            if total + len(b) > MAX_CONTEXT_CHARS:
                break
            trimmed.append(b)
            total += len(b)

        context_str = "\n\n---\n\n".join(trimmed)

        # --------------------------------------------------
        # 5. FINAL ANSWER
        # --------------------------------------------------
        system_prompt = (
            "You are an expert Indian CA tutor. "
            "Answer strictly using the context below. "
            "If tables or figures are present, refer to them explicitly. "
            "End with a short 'Sources Used' section.\n\n"
            f"Context:\n{context_str}"
        )

        answer = await call_llm(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": req.message},
            ]
        )

        return ChatResponse(answer=answer, sources=sources)

    # --------------------------------------------------
    # 6. SAFE FALLBACK (NO BLANK ANSWERS)
    # --------------------------------------------------
    except Exception as e:
        print("CHAT ERROR:", str(e))

        answer = await call_llm(
            [
                {
                    "role": "system",
                    "content": "You are a helpful Indian CA tutor.",
                },
                {"role": "user", "content": req.message},
            ]
        )

        return ChatResponse(
            answer=answer,
            sources=[
                {
                    "doc_title": "LLM fallback",
                    "note": "Answered without document sources due to system issue",
                }
            ],
        )




# ---------- Admin: upload PDF (enhanced: tables + figures extraction) ----------
@app.post("/admin/upload_pdf", response_model=UploadResult)
async def upload_pdf(
    file: UploadFile = File(...),
    # optional extra metadata from admin panel (JSON string)
    metadata: Optional[str] = Form(None),
    admin=Depends(get_current_admin),
):
    """
    Upload pipeline:
     - extract page text chunks (existing)
     - extract tables via pdfplumber -> save CSVs + index short summary vectors
     - render page images (pdf2image) -> save thumbnails + run OCR -> index figure-summary vectors
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF allowed")

    # Parse metadata JSON if provided
    doc_meta: Dict[str, Optional[str]] = {}
    if metadata:
        import json

        try:
            doc_meta = json.loads(metadata)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid metadata JSON")

    # Set some defaults for CA Final if not provided
    course = doc_meta.get("course") or "CA_FINAL"
    subject = doc_meta.get("subject") or "Unknown Subject"
    doc_type = doc_meta.get("doc_type") or "study_notes"
    title = doc_meta.get("title") or file.filename
    year = doc_meta.get("year")
    version = doc_meta.get("version") or "v1"
    author = doc_meta.get("author") or "Unknown"

    file_bytes = await file.read()

    # create per-document upload folder
    upload_folder = ensure_doc_upload_folder(file.filename)

    # --- Extract per-page text and detect headings ---
    raw_pages = extract_pdf_pages(file_bytes)
    pages = [normalize_page_text(p) for p in raw_pages]

    chunks_for_index = []
    chunk_global_index = 0

    for page_num, page_text in enumerate(pages, start=1):
        if not page_text.strip():
            continue

        heading_info = detect_headings_for_page(page_text)
        chapter_guess = heading_info.get("chapter")
        topic_guess = heading_info.get("topic")

        # chunk this page's text
        page_chunks = chunk_text_words(page_text, chunk_size=180, overlap=40)

        for local_idx, chunk_text in enumerate(page_chunks):
            raw_chunk_id = f"{file.filename}_p{page_num}_c{local_idx}"
            chunk_id = sanitize_id(raw_chunk_id, max_len=200)
            chunks_for_index.append(
                {
                    "id": chunk_id,
                    "text": chunk_text,
                    "page_start": page_num,
                    "page_end": page_num,
                    "chapter": chapter_guess,
                    "topic": topic_guess,
                }
            )
            chunk_global_index += 1

    if not chunks_for_index:
        raise HTTPException(status_code=400, detail="No text extracted from PDF")

    # --- Extract tables (pdfplumber) and page images + OCR ---
    table_metas = []
    image_metas = []
    try:
        # tables
        try:
            table_metas = extract_tables_pdfplumber_from_bytes(file_bytes, upload_folder)
        except HTTPException:
            table_metas = []
        # page images + thumbs
        try:
            image_metas = save_page_images_and_thumbs(file_bytes, upload_folder, dpi=200)
        except HTTPException:
            image_metas = []
    except Exception as e:
        # non-fatal - proceed but log
        print("Warning: table/figure extraction failed:", str(e))
        table_metas = table_metas or []
        image_metas = image_metas or []

    # --- Embed in batches & build vectors (text chunks + table summaries + figure ocr) ---
    batch_size = EMBED_BATCH_SIZE
    vectors = []

    # 1) Text chunks (existing)
    for i in range(0, len(chunks_for_index), batch_size):
        batch = chunks_for_index[i : i + batch_size]
        texts = [
            (c["text"] if len(c["text"]) <= MAX_TEXT_LENGTH_FOR_EMBED else c["text"][:MAX_TEXT_LENGTH_FOR_EMBED])
            for c in batch
        ]
        embeddings = await embed_texts(texts)
        for c, emb in zip(batch, embeddings):
            raw_meta = {
                "text": c["text"][:2000],  # limit stored text length (keep under Pinecone metadata limits)
                "source": file.filename,
                "doc_title": title,
                "course": course,
                "subject": subject,
                "doc_type": doc_type,
                "year": year,
                "version": version,
                "page_start": c["page_start"],
                "page_end": c["page_end"],
                "chapter": c["chapter"],
                "topic": c["topic"],
                "chunk_id": c["id"],
                "uploaded_by": admin["email"],
                "uploaded_at": datetime.utcnow().isoformat(),
                "author": author,
                "type": "text",
            }

            # Sanitize metadata: remove None values and ensure allowed types
            sanitized_meta: Dict[str, object] = {}
            for k, v in raw_meta.items():
                if v is None:
                    continue
                if isinstance(v, (str, int, float, bool)):
                    sanitized_meta[k] = v
                elif isinstance(v, list) and all(isinstance(i, str) for i in v):
                    sanitized_meta[k] = v
                else:
                    try:
                        sanitized_meta[k] = str(v)
                    except Exception:
                        continue

            vectors.append(
                {
                    "id": c["id"],
                    "values": emb,
                    "metadata": sanitized_meta,
                }
            )

    # 2) Tables: index a short summary for each extracted table and attach CSV path
    for tm in table_metas:
        page = tm["page"]
        csv_path = tm["csv_path"]
        excerpt = tm.get("excerpt") or ""
        summary = f"Table (page {page}) excerpt: {excerpt}"
        # create small embedding
        try:
            emb = await embed_single(summary)
        except Exception as e:
            print("Table embed failed:", e)
            continue

        table_id = sanitize_id(f"{file.filename}_table_p{page}_{tm.get('table_index',0)}", max_len=200)
        # create accessible URL path for UI (here local filesystem path; change to presigned S3 if needed)
        csv_url = f"/uploads/{sanitize_id(file.filename, max_len=80)}/{os.path.basename(csv_path)}"
        meta = {
            "text": summary,
            "source": file.filename,
            "doc_title": title,
            "page_start": page,
            "page_end": page,
            "table_csv_url": csv_url,
            "type": "table",
            "chunk_id": table_id,
            "uploaded_by": admin["email"],
            "uploaded_at": datetime.utcnow().isoformat(),
        }
        # sanitize meta values
        sanitized_meta: Dict[str, object] = {}
        for k, v in meta.items():
            if v is None:
                continue
            if isinstance(v, (str, int, float, bool)):
                sanitized_meta[k] = v
            else:
                try:
                    sanitized_meta[k] = str(v)
                except Exception:
                    continue

        vectors.append({"id": table_id, "values": emb, "metadata": sanitized_meta})

    # 3) Figures/OCR: index OCR text from thumbnails (if available)
    for im in image_metas:
        page = im["page"]
        thumb = im.get("thumb_path")
        img_path = im.get("image_path")
        ocr_text = ""
        if thumb and Image is not None and pytesseract is not None:
            try:
                pil = Image.open(thumb)
                ocr_text = ocr_text_from_image(pil)
            except Exception:
                ocr_text = ""
        if ocr_text:
            summary = f"Figure OCR (page {page}): {ocr_text[:800]}"
            try:
                emb = await embed_single(summary)
            except Exception as e:
                print("Figure embed failed:", e)
                continue

            fig_id = sanitize_id(f"{file.filename}_fig_p{page}", max_len=200)
            thumb_url = f"/uploads/{sanitize_id(file.filename, max_len=80)}/{os.path.basename(thumb)}" if thumb else None
            meta = {
                "text": summary,
                "source": file.filename,
                "doc_title": title,
                "page_start": page,
                "page_end": page,
                "thumb_url": thumb_url,
                "type": "figure",
                "chunk_id": fig_id,
                "uploaded_by": admin["email"],
                "uploaded_at": datetime.utcnow().isoformat(),
            }
            sanitized_meta: Dict[str, object] = {}
            for k, v in meta.items():
                if v is None:
                    continue
                if isinstance(v, (str, int, float, bool)):
                    sanitized_meta[k] = v
                else:
                    try:
                        sanitized_meta[k] = str(v)
                    except Exception:
                        continue

            vectors.append({"id": fig_id, "values": emb, "metadata": sanitized_meta})

    # --- Upsert to Pinecone ---
    namespace = course  # e.g. "CA_FINAL"
    try:
        # Upsert in batches (Pinecone expects reasonable sized upsert calls)
        CHUNK = 100
        for j in range(0, len(vectors), CHUNK):
            slice_v = vectors[j : j + CHUNK]
            index.upsert(vectors=slice_v, namespace=namespace)
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Pinecone upsert failed: {str(e)}")

    # --- Store doc-level metadata in Mongo ---
    await docs_collection.insert_one(
        {
            "filename": file.filename,
            "title": title,
            "course": course,
            "subject": subject,
            "doc_type": doc_type,
            "year": year,
            "author": author,
            "version": version,
            "uploaded_by": admin["email"],
            "uploaded_at": datetime.utcnow(),
            "chunks": len(chunks_for_index),
            "tables": len(table_metas),
            "figures": len(image_metas),
            "namespace": namespace,
        }
    )

    return UploadResult(chunks=len(chunks_for_index), filename=file.filename)


@app.get("/admin/documents")
async def list_documents(admin=Depends(get_current_admin)):
    docs_cursor = docs_collection.find().sort("uploaded_at", -1)
    docs = []
    async for d in docs_cursor:
        d["_id"] = str(d["_id"])
        docs.append(d)
    return docs


@app.get("/health")
async def health():
    return {"status": "ok"}
