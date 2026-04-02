# backend/main.py
import io
import os
import unicodedata
import asyncio
import time
from datetime import datetime, timedelta
from typing import Any, List, Optional, Dict
import re
import tempfile
import secrets
import traceback

from fastapi import (
    FastAPI,
    HTTPException,
    Depends,
    UploadFile,
    File,
    Header,
    Form,
    APIRouter,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from jose import jwt, JWTError
from passlib.context import CryptContext
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
import httpx
import pinecone
from pypdf import PdfReader
from typing import Literal

from config import settings
from ca_text_normalizer import expand_ca_abbreviations
from ingestion.enhanced_upload_service import process_pdf, process_pdf_enhanced
from email_service import send_admin_signup_notification, send_password_reset_otp
from payment_router import router as payment_router
from s3_service import upload_pdf_to_s3, delete_pdf_from_s3, is_s3_configured

# ============================================================
# APP + CORS
# ============================================================

app = FastAPI(title="CA Chatbot")

_raw_origin    = settings.FRONTEND_ORIGIN.strip()
_allow_origins = ["*"] if _raw_origin == "*" else [
    o.strip() for o in _raw_origin.split(",") if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allow_origins,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

app.include_router(payment_router, prefix="/payments", tags=["payments"])

CHAT_URL = "https://api.openai.com/v1/chat/completions"
EMBED_URL = "https://api.openai.com/v1/embeddings"


# ============================================================
# DB + EXTERNAL CLIENTS
# ============================================================

pwd_context = CryptContext(schemes=["pbkdf2_sha256"], default="pbkdf2_sha256", deprecated="auto")

mongo_client         = AsyncIOMotorClient(settings.MONGO_URI)
db                   = mongo_client[settings.MONGO_DB]
users_collection     = db["users"]
docs_collection      = db["documents"]
dashboard_collection = db["ca_dashboard"]

pinecone_client = pinecone.Pinecone(api_key=settings.PINECONE_API_KEY)
index           = pinecone_client.Index(settings.PINECONE_INDEX)

JWT_EXP_MINUTES           = 60 * 24
EMBED_BATCH_SIZE          = getattr(settings, "EMBED_BATCH_SIZE",          12)
EMBED_TIMEOUT_SECS        = getattr(settings, "EMBED_TIMEOUT_SECS",        120)
EMBED_MAX_RETRIES         = getattr(settings, "EMBED_MAX_RETRIES",         3)
EMBED_BACKOFF_BASE        = getattr(settings, "EMBED_BACKOFF_BASE",        1.8)
MAX_TEXT_LENGTH_FOR_EMBED = getattr(settings, "MAX_TEXT_LENGTH_FOR_EMBED", 8000)

# Local uploads folder — used as fallback when S3 is not configured
UPLOAD_ROOT = getattr(settings, "UPLOAD_ROOT", "./uploads")
os.makedirs(UPLOAD_ROOT, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_ROOT), name="uploads")

# Add near the top of main.py, after settings are loaded

@app.on_event("startup")
async def validate_config():
    if not settings.OPENAI_API_KEY or not settings.OPENAI_API_KEY.startswith("sk-"):
        raise RuntimeError("OPENAI_API_KEY is missing or invalid in .env")
    if not getattr(settings, "LLM_MODEL", ""):
        raise RuntimeError("LLM_MODEL is not set in .env")
    print(f"[startup] LLM_MODEL = {settings.LLM_MODEL}")
    print(f"[startup] EMBEDDING_MODEL = {settings.EMBEDDING_MODEL}")
# ============================================================
# PYDANTIC MODELS
# ============================================================

class UserCreate(BaseModel):
    email: str
    password: str
    name: str
    phone: str
    ca_level: str
    ca_attempt: int
    role: str = "student"
    plan: Optional[str] = "free"
    payment_id: Optional[str] = None

class UserLogin(BaseModel):
    email: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class UserOut(BaseModel):
    email: str
    role: str
    plan: Optional[str] = "free"
    subscription_status: Optional[str] = "free"

class ChatResponse(BaseModel):
    answer: str
    sources: List[dict]

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[ChatMessage]] = None
    mode: Optional[str] = "qa"

class UploadResponse(BaseModel):
    success: bool
    message: str
    filename: str
    statistics: dict
    metadata: dict

class ForgotPasswordRequest(BaseModel):
    email: str

class VerifyOTPRequest(BaseModel):
    email: str
    otp: str

class ResetPasswordRequest(BaseModel):
    email: str
    otp: str
    new_password: str


# ============================================================
# AUTH HELPERS
# ============================================================

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire    = datetime.utcnow() + (expires_delta or timedelta(minutes=JWT_EXP_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.JWT_SECRET, algorithm=settings.JWT_ALGO)

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
        email: str = payload.get("sub")
        if not email:
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


# ============================================================
# ADMIN MATERIALS ROUTER
# ============================================================

router = APIRouter(prefix="/admin/materials")


# ----------------------------------------------------------
# UPLOAD  →  S3 (or local)  +  Pinecone  +  MongoDB
# ----------------------------------------------------------

@router.post("/upload_enhanced", response_model=UploadResponse)
async def upload_pdf_enhanced(
    file:    UploadFile        = File(...),
    course:  str               = Form(...),
    subject: Optional[str]     = Form(None),
    module:  Optional[str]     = Form(None),
    chapter: Optional[str]     = Form(None),
    unit:    Optional[str]     = Form(None),
    section: Optional[str]     = Form(None),
    custom_heading: Optional[str] = Form(None),
    enable_image_descriptions: bool = Form(True),
    admin = Depends(get_current_admin),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    temp_file_path: Optional[str] = None

    try:
        content = await file.read()

        level            = course.strip()
        resolved_subject = (subject or chapter or "General").strip()
        resolved_module  = (module  or section or "General").strip()
        resolved_chapter = (chapter or "").strip()
        resolved_unit    = (unit    or "").strip()
        safe_filename    = file.filename.replace(" ", "_")

        s3_ready = is_s3_configured()
        print(f"[upload] S3 configured={s3_ready}  file={safe_filename}")

        if s3_ready:
            try:
                pdf_url         = upload_pdf_to_s3(
                    file_bytes = content,
                    filename   = safe_filename,
                    level      = level,
                    subject    = resolved_subject,
                )
                storage_backend = "s3"
                print(f"[upload] ✅ S3 upload OK → {pdf_url}")
            except RuntimeError as s3_err:
                print(f"[upload] ⚠️  S3 failed, falling back to local: {s3_err}")
                pdf_url         = _save_local(content, safe_filename)
                storage_backend = f"local_fallback (s3_error: {s3_err})"
        else:
            pdf_url         = _save_local(content, safe_filename)
            storage_backend = "local"
            print(f"[upload] S3 not configured — stored locally: {pdf_url}")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(content)
            temp_file_path = tmp.name

        now = datetime.utcnow()

        extra_meta = {
            "title":          file.filename,
            "course":         course,
            "level":          level,
            "subject":        resolved_subject,
            "chapter":        resolved_chapter,
            "section":        section        or "",
            "unit":           resolved_unit,
            "module":         resolved_module,
            "custom_heading": custom_heading or "",
            "uploaded_by":    admin["email"],
            "uploaded_at":    now.isoformat(),
            "pdf_url":        pdf_url        or "",
        }

        result = await process_pdf_enhanced(
            file_path                 = temp_file_path,
            file_name                 = file.filename,
            extra_meta                = extra_meta,
            enable_image_descriptions = enable_image_descriptions,
            openai_api_key            = settings.OPENAI_API_KEY,
        )

        doc_record = {
            "filename":        file.filename,
            "safe_filename":   safe_filename,
            "course":          course,
            "level":           level,
            "subject":         resolved_subject,
            "chapter":         resolved_chapter,
            "section":         section        or "",
            "unit":            resolved_unit,
            "module":          resolved_module,
            "custom_heading":  custom_heading or "",
            "pdf_url":         pdf_url        or "",
            "storage_backend": storage_backend,
            "uploaded_by":     admin["email"],
            "uploaded_at":     now,
            "total_vectors":   result["total_vectors"],
        }
        inserted   = await docs_collection.insert_one(doc_record)
        doc_id_str = str(inserted.inserted_id)

        await dashboard_collection.insert_one({
            "level":       level            if level             else "Others",
            "subject":     resolved_subject,
            "module":      resolved_module   if resolved_module   else resolved_subject,
            "chapter":     resolved_chapter  if resolved_chapter  else file.filename.replace(".pdf", ""),
            "unit":        resolved_unit,
            "title":       file.filename.replace(".pdf", "").replace("_", " "),
            "pdf_url":     pdf_url           or "",
            "video_url":   "",
            "source_doc":  doc_id_str,
            "uploaded_by": admin["email"],
            "created_at":  now,
        })

        _safe_unlink(temp_file_path)

        return UploadResponse(
            success    = True,
            message    = f"Successfully processed '{file.filename}' (storage: {storage_backend})",
            filename   = file.filename,
            statistics = {
                "total_vectors":      result["total_vectors"],
                "text_chunks":        result["text_chunks"],
                "table_chunks":       result["table_chunks"],
                "image_chunks":       result["image_chunks"],
                "total_images_found": result["total_images"],
                "total_tables_found": result["total_tables"],
                "storage_backend":    storage_backend,
            },
            metadata = extra_meta,
        )

    except Exception as e:
        _safe_unlink(temp_file_path)
        print(f"[upload_pdf_enhanced] Error — {file.filename}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")


# ----------------------------------------------------------
# DELETE  →  Pinecone  +  S3/local  +  Mongo docs  +  Mongo dashboard
# ----------------------------------------------------------

@router.delete("/{doc_id}", tags=["Admin Materials"])
async def delete_document(doc_id: str, admin=Depends(get_current_admin)):
    try:
        obj_id = ObjectId(doc_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid document ID format")

    doc = await docs_collection.find_one({"_id": obj_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    filename        = doc.get("filename",        "")
    pdf_url         = doc.get("pdf_url",         "")
    safe_filename   = doc.get("safe_filename",   filename.replace(" ", "_"))
    storage_backend = doc.get("storage_backend", "local")

    report: Dict[str, Any] = {
        "doc_id":           doc_id,
        "filename":         filename,
        "pinecone_deleted": 0,
        "s3_deleted":       False,
        "local_deleted":    False,
        "mongo_docs":       False,
        "mongo_dashboard":  0,
        "errors":           [],
    }

    try:
        report["pinecone_deleted"] = await _delete_pinecone_by_source(filename)
    except Exception as e:
        report["errors"].append(f"Pinecone: {e}")

    if storage_backend == "s3" and pdf_url:
        try:
            ok = delete_pdf_from_s3(pdf_url)
            report["s3_deleted"] = ok
            if not ok:
                report["errors"].append("S3 delete returned False — file may not exist in bucket")
        except Exception as e:
            report["errors"].append(f"S3: {e}")
    else:
        local_path = os.path.join(UPLOAD_ROOT, safe_filename)
        try:
            if os.path.exists(local_path):
                os.remove(local_path)
                report["local_deleted"] = True
        except Exception as e:
            report["errors"].append(f"Local file: {e}")

    del_result           = await docs_collection.delete_one({"_id": obj_id})
    report["mongo_docs"] = del_result.deleted_count > 0

    dash_result               = await dashboard_collection.delete_many({"source_doc": doc_id})
    report["mongo_dashboard"] = dash_result.deleted_count

    return {"message": f"'{filename}' deleted successfully", "report": report}


async def _delete_pinecone_by_source(source_filename: str) -> int:
    try:
        index_info = index.describe_index_stats()
        dim = index_info.get("dimension", 3072)
    except Exception:
        dim = 3072

    zero_vec      = [0.0] * dim
    total_deleted = 0

    while True:
        try:
            res = index.query(
                vector=zero_vec, top_k=1000,
                include_metadata=False,
                filter={"source": {"$eq": source_filename}},
            )
        except Exception:
            break

        matches = res.get("matches") or []
        if not matches:
            break

        index.delete(ids=[m["id"] for m in matches])
        total_deleted += len(matches)

        if len(matches) < 1000:
            break

    return total_deleted


@router.get("/upload_health")
async def upload_service_health():
    from s3_service import debug_s3_config, create_bucket_if_not_exists

    s3_cfg = debug_s3_config()

    health: Dict[str, Any] = {
        "service": "upload_enhanced",
        "status":  "operational",
        "storage": "s3" if is_s3_configured() else "local",
        "s3_config": s3_cfg,
        "features": {
            "aws_s3":            is_s3_configured(),
            "docling_parser":    True,
            "pdfplumber_tables": True,
            "enhanced_chunking": True,
            "pinecone_delete":   True,
        },
    }

    try:
        import docling, pdfplumber, fitz
        health["dependencies"] = "all_available"
    except ImportError as e:
        health["status"]       = "degraded"
        health["dependencies"] = f"missing: {e}"

    if is_s3_configured():
        try:
            bucket_ready = create_bucket_if_not_exists()
            health["s3_connectivity"] = (
                f"✅ bucket '{s3_cfg['AWS_S3_BUCKET']}' ready"
                if bucket_ready else
                "❌ bucket creation failed — check IAM permissions"
            )
        except Exception as e:
            health["s3_connectivity"] = f"❌ {e}"

    return health


app.include_router(router)


# ============================================================
# HELPERS  (used by upload route and chat route)
# ============================================================

def _save_local(content: bytes, safe_filename: str) -> str:
    """Write bytes to UPLOAD_ROOT and return the URL path."""
    with open(os.path.join(UPLOAD_ROOT, safe_filename), "wb") as f:
        f.write(content)
    base_url = getattr(settings, "BASE_URL", "")
    return f"{base_url}/uploads/{safe_filename}" if base_url else f"/uploads/{safe_filename}"


def _safe_unlink(path: Optional[str]) -> None:
    if path:
        try:
            if os.path.exists(path):
                os.unlink(path)
        except Exception:
            pass


# ============================================================
# EMBEDDING + LLM
# ============================================================

async def embed_texts(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    headers = {
        "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
        "Content-Type":  "application/json",
    }
    batches: List[List[str]] = [
        texts[i : i + EMBED_BATCH_SIZE] for i in range(0, len(texts), EMBED_BATCH_SIZE)
    ]
    results: List[List[float]] = []

    async with httpx.AsyncClient(timeout=EMBED_TIMEOUT_SECS) as client:
        for batch_idx, batch_texts in enumerate(batches):
            payload = {"model": settings.EMBEDDING_MODEL, "input": batch_texts}
            attempt = 0
            while True:
                attempt += 1
                try:
                    resp = await client.post(EMBED_URL, headers=headers, json=payload)
                    resp.raise_for_status()
                    emb_batch = [d["embedding"] for d in resp.json()["data"]]
                    if len(emb_batch) != len(batch_texts):
                        raise HTTPException(status_code=502, detail="Embedding length mismatch")
                    results.extend(emb_batch)
                    break
                except (httpx.ReadTimeout, httpx.WriteTimeout, httpx.ConnectError, httpx.RemoteProtocolError) as exc:
                    if attempt >= EMBED_MAX_RETRIES:
                        raise HTTPException(status_code=504, detail=f"Embedding timeout (batch {batch_idx}): {exc}")
                    await asyncio.sleep(EMBED_BACKOFF_BASE ** (attempt - 1))
                except httpx.HTTPStatusError as exc:
                    sc = exc.response.status_code
                    et = ""
                    try: et = exc.response.text
                    except Exception: pass
                    if 500 <= sc < 600 and attempt < EMBED_MAX_RETRIES:
                        await asyncio.sleep(EMBED_BACKOFF_BASE ** (attempt - 1))
                        continue
                    raise HTTPException(status_code=502, detail=f"Embedding error {sc}: {et[:200]}")
                except Exception as exc:
                    raise HTTPException(status_code=500, detail=f"Embedding failed: {exc}")

    return results


async def embed_single(text: str) -> List[float]:
    return (await embed_texts([text]))[0]


async def call_llm(messages: List[dict]) -> str:
    headers = {
        "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
        "Content-Type":  "application/json",
    }
    async with httpx.AsyncClient(timeout=90) as client:
        resp = await client.post(CHAT_URL, headers=headers, json={
            "model": settings.LLM_MODEL, "messages": messages, "temperature": 0.2
        })
        if resp.status_code == 404:
            raise HTTPException(
                status_code=500,
                detail=f"OpenAI model '{settings.LLM_MODEL}' not found. Check LLM_MODEL in your .env — valid options: gpt-4o, gpt-4o-mini, gpt-4.1, gpt-4.1-mini"
            )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


async def call_llm_with_chain(*, user_question: str, context: str, final_system_prompt: str) -> str:
    chain_prompt = (
        "You are reasoning internally as an expert Indian CA tutor.\n\n"
        "INTERNAL STEPS (do NOT reveal):\n"
        "1. Understand the exact exam intent. Identify the overall concept.\n"
        "2. Identify relevant context blocks. Merge info from multiple blocks.\n"
        "3. Decide depth per ICAI expectations.\n\n"
        "Then produce ONLY the final answer as instructed below.\n\n"
        f"{final_system_prompt}"
    )
    return await call_llm([
        {"role": "system", "content": chain_prompt},
        {"role": "system", "content": f"CONTEXT:\n{context}"},
        {"role": "user",   "content": user_question},
    ])


# ============================================================
# QUERY HELPERS  (used only by /chat)
# ============================================================

def enrich_query_for_rag(question: str) -> str:
    q, hints = question.lower(), []
    if any(w in q for w in ["ind as", "financial", "asset", "liability", "consolidation"]):
        hints.append("financial reporting accounting")
    if any(w in q for w in ["audit", "sa ", "assurance"]):
        hints.append("auditing")
    if any(w in q for w in ["gst", "input tax", "itc"]):
        hints.append("indirect tax gst")
    if any(w in q for w in ["tds", "income tax", "section 80"]):
        hints.append("direct tax")
    if any(w in q for w in ["company act", "directors", "board"]):
        hints.append("law")
    return question + (" " + " ".join(hints) if hints else "")


def detect_subject(question: str) -> Optional[str]:
    q = question.lower()
    if any(w in q for w in ["ind as", "as ", "financial statement", "consolidat", "revenue recognition"]):
        return "Accounting"
    if any(w in q for w in ["audit", "sa ", "assurance", "internal audit"]):
        return "Auditing"
    if any(w in q for w in ["gst", "indirect tax", "customs", "excise"]):
        return "Indirect Tax"
    if any(w in q for w in ["income tax", "direct tax", "tds", "section 80", "capital gain"]):
        return "Direct Tax"
    if any(w in q for w in ["company", "director", "board", "sebi", "securities"]):
        return "Law"
    if any(w in q for w in ["cost", "marginal", "budget", "variance"]):
        return "Costing"
    return None


async def is_ca_related_question(question: str) -> bool:
    system = (
        "You are a domain classifier for an Indian Chartered Accountancy (CA) assistant.\n\n"
        "Answer YES if the question relates to: ICAI syllabus, Accounting, Auditing, "
        "Direct Tax/GST, Corporate Law, Financial management, Costing, or basic "
        "commerce concepts commonly studied by CA students.\n\n"
        "Answer NO only if it is clearly unrelated (science, coding, sports, entertainment).\n\n"
        "Respond with YES or NO only."
    )
    try:
        result = await call_llm([
            {"role": "system", "content": system},
            {"role": "user",   "content": question},
        ])
        return result.strip().upper().startswith("YES")
    except Exception:
        return True


# ============================================================
# AUTH ROUTES
# ============================================================

@app.post("/auth/signup")
async def signup(user: UserCreate):
    if await get_user_by_email(user.email):
        raise HTTPException(status_code=400, detail="Email already registered")

    plan = (user.plan or "free").lower()
    if plan not in ("free", "paid"): plan = "free"
    if plan == "paid" and not user.payment_id:
        raise HTTPException(status_code=400, detail="payment_id is required for paid plan")

    now = datetime.utcnow()
    await users_collection.insert_one({
        "email":               user.email,
        "password_hash":       hash_password(user.password),
        "name":                user.name,
        "phone":               user.phone,
        "ca_level":            user.ca_level,
        "ca_attempt":          user.ca_attempt,
        "role":                "student",
        "status":              "approved",
        "plan":                plan,
        "subscription_status": "active" if plan == "paid" else "free",
        "payment_id":          user.payment_id,
        "plan_activated_at":   now if plan == "paid" else None,
        "plan_expires_at": (
            datetime(now.year, now.month + 1 if now.month < 12 else 1, now.day)
            if plan == "paid" else None
        ),
        "created_at": now,
    })
    try:
        send_admin_signup_notification({**user.dict(), "plan": plan})
    except Exception as e:
        print("Signup email to admin failed:", e)
    return {"message": "Signup successful."}


@app.post("/auth/login", response_model=Token)
async def login(data: UserLogin):
    user = await get_user_by_email(data.email)
    if not user or user.get("status") != "approved":
        raise HTTPException(status_code=403, detail="Your account is pending admin approval.")
    if not verify_password(data.password, user["password_hash"]):
        raise HTTPException(status_code=400, detail="Invalid credentials")
    return Token(access_token=create_access_token({"sub": user["email"]}))


@app.get("/auth/me", response_model=UserOut)
async def me(user=Depends(get_current_user)):
    return UserOut(
        email=user["email"], role=user["role"],
        plan=user.get("plan", "free"),
        subscription_status=user.get("subscription_status", "free"),
    )


@app.post("/auth/forgot-password")
async def forgot_password(body: ForgotPasswordRequest):
    user = await get_user_by_email(body.email)
    if not user:
        return {"message": "If this email is registered, an OTP has been sent."}
    otp        = str(secrets.randbelow(900000) + 100000)
    expires_at = datetime.utcnow() + timedelta(minutes=10)
    await users_collection.update_one(
        {"email": body.email},
        {"$set": {"reset_otp": otp, "reset_otp_expires": expires_at}},
    )
    try:
        send_password_reset_otp(email=body.email, otp=otp, name=user.get("name", "Student"))
    except Exception as e:
        print("OTP email failed:", e)
        raise HTTPException(status_code=500, detail="Failed to send OTP email.")
    return {"message": "If this email is registered, an OTP has been sent."}


@app.post("/auth/verify-otp")
async def verify_otp(body: VerifyOTPRequest):
    user = await get_user_by_email(body.email)
    if not user:
        raise HTTPException(status_code=400, detail="Invalid OTP or email.")
    stored_otp, stored_expires = user.get("reset_otp"), user.get("reset_otp_expires")
    if not stored_otp or not stored_expires:
        raise HTTPException(status_code=400, detail="No OTP requested. Please request a new one.")
    if datetime.utcnow() > stored_expires:
        raise HTTPException(status_code=400, detail="OTP has expired.")
    if stored_otp != body.otp.strip():
        raise HTTPException(status_code=400, detail="Incorrect OTP.")
    return {"message": "OTP verified."}


@app.post("/auth/reset-password")
async def reset_password(body: ResetPasswordRequest):
    user = await get_user_by_email(body.email)
    if not user:
        raise HTTPException(status_code=400, detail="Invalid request.")
    stored_otp, stored_expires = user.get("reset_otp"), user.get("reset_otp_expires")
    if not stored_otp or not stored_expires:
        raise HTTPException(status_code=400, detail="No OTP requested.")
    if datetime.utcnow() > stored_expires:
        raise HTTPException(status_code=400, detail="OTP has expired.")
    if stored_otp != body.otp.strip():
        raise HTTPException(status_code=400, detail="Incorrect OTP.")
    if len(body.new_password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters.")
    await users_collection.update_one(
        {"email": body.email},
        {"$set":   {"password_hash": hash_password(body.new_password)},
         "$unset": {"reset_otp": "", "reset_otp_expires": ""}},
    )
    return {"message": "Password reset successfully. You can now log in."}


# ============================================================
# ADMIN — STUDENT MANAGEMENT
# ============================================================

@app.get("/admin/students")
async def get_all_students(admin=Depends(get_current_admin)):
    students = []
    async for user in users_collection.find({"role": "student"}):
        user["_id"] = str(user["_id"])
        students.append(user)
    return students


@app.post("/admin/approve/{user_id}")
async def approve_student(user_id: str, admin=Depends(get_current_admin)):
    result = await users_collection.update_one(
        {"_id": ObjectId(user_id)}, {"$set": {"status": "approved"}}
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Student not found")
    return {"message": "Student approved"}


@app.post("/admin/reject/{user_id}")
async def reject_student(user_id: str, admin=Depends(get_current_admin)):
    await users_collection.delete_one({"_id": ObjectId(user_id)})
    return {"message": "Student rejected"}


# ============================================================
# ADMIN — DOCUMENTS LIST  (AdminUpload panel)
# ============================================================

@app.get("/admin/documents/grouped", tags=["Admin Materials"])
async def get_grouped_documents(admin=Depends(get_current_admin)):
    grouped: Dict[str, List[Any]] = {}
    async for doc in docs_collection.find().sort("uploaded_at", -1):
        doc["_id"] = str(doc["_id"])
        if isinstance(doc.get("uploaded_at"), datetime):
            doc["uploaded_at"] = doc["uploaded_at"].isoformat()
        course = doc.get("course") or doc.get("level") or "Other"
        grouped.setdefault(course, [])
        grouped[course].append(doc)
    return grouped


# ============================================================
# CA DASHBOARD ROUTES
# ============================================================

@app.get("/dashboard/tree")
async def get_dashboard_tree(user=Depends(get_current_user)):
    tree: Dict[str, Any] = {}
    async for doc in dashboard_collection.find().sort("created_at", 1):
        doc["_id"] = str(doc["_id"])
        level   = (doc.get("level")   or "Others").strip()
        subject = (doc.get("subject") or "General").strip()
        module  = (doc.get("module")  or subject).strip()
        chapter = (doc.get("chapter") or doc.get("title", "General")).strip()

        tree.setdefault(level, {})
        tree[level].setdefault(subject, {})
        tree[level][subject].setdefault(module, {})
        tree[level][subject][module].setdefault(chapter, [])

        tree[level][subject][module][chapter].append({
            "_id":       doc["_id"],
            "title":     doc.get("title",     ""),
            "pdf_url":   doc.get("pdf_url",   ""),
            "video_url": doc.get("video_url", ""),
            "chapter":   chapter,
            "unit":      doc.get("unit",      ""),
        })
    return tree


@app.post("/dashboard/add")
async def add_dashboard_resource(
    level: str = Form(...), subject: str = Form(...),
    module: str = Form(...), chapter: str = Form(...),
    unit: str = Form(...), title: str = Form(...),
    pdf_url: str = Form(...), video_url: str = Form(""),
    admin=Depends(get_current_admin),
):
    await dashboard_collection.insert_one({
        "level": level, "subject": subject, "module": module,
        "chapter": chapter, "unit": unit, "title": title,
        "pdf_url": pdf_url, "video_url": video_url,
        "created_at": datetime.utcnow(),
    })
    return {"message": "Added successfully"}


# ============================================================
# PERSONALISATION
# ============================================================

def build_personalized_layer(user: dict) -> str:
    name  = user.get("name", "Student").split()[0]
    level = user.get("ca_level", "Foundation")
    guidance = {
        "Foundation":   "Explain in very simple language with basic concepts and easy examples.",
        "Intermediate": "Explain with clarity and practical understanding. Include examples.",
        "Final":        "Provide detailed, professional-level explanation with ICAI exam perspective.",
    }
    return (
        f"PERSONALIZATION CONTEXT:\n- Student Name: {name}\n- Level: {level}\n\n"
        f"INSTRUCTIONS:\n"
        f"- Occasionally address the student as {name}\n"
        f"- Teaching style: {guidance.get(level, guidance['Foundation'])}\n"
        f"- Keep tone supportive and engaging\n"
        f"- Provide extra clarity, motivation, and simplify difficult parts."
    )


# ============================================================
# CHAT (RAG)
# ============================================================

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, user=Depends(get_current_user)):
    try:
        # 🔥 Expand CA abbreviations BEFORE everything
        req.message = expand_ca_abbreviations(req.message)

        # --------------------------------------------------
        # 1. CA gatekeeper
        # --------------------------------------------------
        if not await is_ca_related_question(req.message):
            return ChatResponse(
                answer=(
                    "This assistant is designed for Indian CA students. "
                    "Please ask a question related to Indian CA topics such as accounting, tax, audit, law, "
                    "or CA exams (Foundation / Inter / Final)."
                ),
                sources=[],
            )

        # --------------------------------------------------
        # 2. RAG FLOW (Pinecone)
        # --------------------------------------------------
        query_embedding = await embed_single(enrich_query_for_rag(req.message))
        detected_subj   = detect_subject(req.message)

        q_kwargs: Dict[str, Any] = {"vector": query_embedding, "top_k": 20, "include_metadata": True}
        if detected_subj:
            q_kwargs["filter"] = {"subject": {"$eq": detected_subj}}

        res     = index.query(**q_kwargs)
        matches = res.get("matches") or []

        # Fallback: retry without subject filter if too few results
        if len(matches) < 4 and detected_subj:
            res     = index.query(vector=query_embedding, top_k=20, include_metadata=True)
            matches = res.get("matches") or []

        # Sort by score, apply dynamic threshold
        matches = sorted(matches, key=lambda m: m.get("score", 0), reverse=True)
        q_len   = len(req.message.split())
        thr     = 0.52 if q_len <= 6 else 0.60 if q_len <= 15 else 0.65
        matches = [m for m in matches if m.get("score", 0) >= thr] or matches[:5]

        personal_context = build_personalized_layer(user)

        # --------------------------------------------------
        # 3. NO MATCHES → LLM-only fallback
        # --------------------------------------------------
        if not matches:
            system_prompt = (
                personal_context + "\n\n"
                "You are a senior Indian Chartered Accountant (CA) faculty with experience "
                "in teaching and evaluating ICAI exams (Foundation, Inter, Final).\n\n"

                "Language rule (MANDATORY):\n"
                "- Reply strictly in the SAME language as the user's question "
                "(English, Hindi, or Hinglish).\n\n"

                "Knowledge & safety rules:\n"
                "- Answer using your standard CA knowledge and well-established ICAI principles.\n"
                "- Do NOT guess exact section numbers, limits, percentages, or year-specific amendments.\n"
                "- If precise data is uncertain, explain the concept without giving risky figures.\n\n"

                "Answer structure (EXAM-ORIENTED):\n"
                "1. Begin with a clear definition or core concept.\n"
                "2. Explain in logical steps using proper CA terminology.\n"
                "3. Where relevant, mention accounting treatment / legal position / tax implication.\n"
                "4. Include ONE short exam-oriented or practical illustration if helpful.\n\n"

                "Exam guidance:\n"
                "- Add ONE short CA exam tip or common mistake to avoid.\n"
                "- Keep the answer concise, structured, and revision-friendly.\n\n"

                "Tone & presentation:\n"
                "- Maintain a professional, faculty-level tone.\n"
                "- Avoid casual language, storytelling, or over-explanation."
            )

            answer = await call_llm([
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": req.message},
            ])
            return ChatResponse(
                answer=answer,
                sources=[{"doc_title": "General CA Knowledge (LLM based)", "note": "No match in uploaded docs"}],
            )

        # --------------------------------------------------
        # 4. BUILD CONTEXT (SAFE SIZE)
        # --------------------------------------------------
        context_blocks: List[str] = []
        sources: List[dict]       = []

        for m in sorted(matches, key=lambda m: m.get("score", 0), reverse=True):
            meta = m.get("metadata", {})
            text = meta.get("text", "")
            if not text:
                continue

            header = []
            if meta.get("doc_title"):  header.append(f"Document: {meta['doc_title']}")
            if meta.get("chapter"):    header.append(f"Chapter: {meta['chapter']}")
            if meta.get("topic"):      header.append(f"Topic: {meta['topic']}")
            if meta.get("page_start"): header.append(f"Page: {meta['page_start']}")
            context_blocks.append(f"{' | '.join(header)}\n{text}")

            sources.append({
                "doc_title":  meta.get("doc_title"),
                "source":     meta.get("source"),
                "page_start": meta.get("page_start"),
                "chapter":    meta.get("chapter"),
                "topic":      meta.get("topic"),
                "type":       meta.get("type", "text"),
            })

        trimmed, total = [], 0
        for b in context_blocks:
            if total + len(b) > 12000:
                break
            trimmed.append(b)
            total += len(b)
        context_str = "\n\n---\n\n".join(trimmed)

        # --------------------------------------------------
        # 5. FINAL ANSWER (QA vs DISCUSSION)
        # --------------------------------------------------
        if req.mode == "discussion":
            sys_p = (
                personal_context + "\n\n"
                "You are an expert Indian CA tutor simulating a healthy academic discussion "
                "between two CA students preparing for exams.\n\n"

                "Language rules:\n"
                "- Reply strictly in the SAME language as the user's question (English, Hindi, or Hinglish).\n\n"

                "Discussion format rules:\n"
                "- Write the answer as a discussion between 'User A:' and 'User B:'.\n"
                "- Alternate clearly between User A and User B.\n"
                "- Provide at least 4 to 6 exchanges.\n\n"

                "Content rules (VERY IMPORTANT):\n"
                "- Explain concepts step-by-step in a teaching style.\n"
                "- Keep explanations exam-oriented as per ICAI expectations.\n"
                "- Use simple intuition first, then technical clarity.\n"
                "- Include 1 very short practical or exam-oriented example if relevant.\n"
                "- Add a quick very short CA exam tip, memory aid, or common mistake to avoid.\n"
                "- Avoid unnecessary storytelling or casual chat.\n\n"

                "Source rules:\n"
                "- Answer using the context provided below.\n"
                "- Only use well-trusted facts based on the given context.\n\n"

                f"Context:\n{context_str}"
            )
        else:
            sys_p = (
                personal_context + "\n\n"
                "You are an expert Indian Chartered Accountant (CA) tutor preparing students "
                "for ICAI exams (Foundation, Inter, Final).\n\n"

                "Language rule:\n"
                "- Reply strictly in the SAME language as the user's question "
                "(English, Hindi, or Hinglish).\n\n"

                "Answering style rules:\n"
                "- Answer using the context provided below.\n"
                "- Keep the explanation clear, concise, and exam-oriented, in detail.\n"
                "- Start with a direct definition or core concept in elaborative style.\n"
                "- Then briefly explain or elaborate as required for marks.\n"
                "- If applicable, include a short practical or exam-oriented example.\n"
                "- If tables or figures are present in the context, refer to them explicitly.\n\n"

                "Exam guidance:\n"
                "- Add one very short important CA exam tip or a common mistake to avoid.\n"
                "- Avoid unnecessary storytelling or over-explanation.\n\n"

                f"Context:\n{context_str}"
            )

        answer = await call_llm_with_chain(
            user_question=req.message,
            context=context_str,
            final_system_prompt=sys_p,
        )

        seen: set             = set()
        clean_sources: List[dict] = []
        for s in sources:
            key = (s.get("doc_title"), s.get("page_start"))
            if key not in seen:
                seen.add(key)
                clean_sources.append(s)

        return ChatResponse(answer=answer, sources=clean_sources[:5])

    # --------------------------------------------------
    # 6. SAFE FALLBACK — never return a blank answer
    # --------------------------------------------------
    except Exception as e:
        traceback.print_exc()
        try:
            answer = await call_llm([
                {"role": "system", "content": "You are a helpful Indian CA tutor."},
                {"role": "user",   "content": req.message},
            ])
            return ChatResponse(
                answer=answer,
                sources=[{
                    "doc_title": "LLM fallback",
                    "note": "Answered without document sources due to a system issue",
                }],
            )
        except Exception as inner_e:
            traceback.print_exc()
            raise HTTPException(
                status_code=500,
                detail=f"Chat error: {str(e)} | Fallback error: {str(inner_e)}"
            )


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
async def health():
    return {"status": "ok"}
