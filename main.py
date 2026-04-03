# # # backend/main.py
# # import io
# # import os
# # import unicodedata
# # import asyncio
# # import time
# # import csv
# # from datetime import datetime, timedelta
# # from typing import Any, List, Optional, Dict
# # import re
# # from typing import Optional

# # from fastapi import (
# #     FastAPI,
# #     HTTPException,
# #     Depends,
# #     UploadFile,
# #     File,
# #     Header,
# #     Form,
# # )
# # from fastapi.middleware.cors import CORSMiddleware
# # from fastapi.staticfiles import StaticFiles
# # from pydantic import BaseModel
# # from jose import jwt, JWTError
# # from passlib.context import CryptContext
# # from ca_text_normalizer import expand_ca_abbreviations
# # # from admin_materials import router as admin_materials_router
# # from ingestion.enhanced_upload_service import process_pdf
# # from motor.motor_asyncio import AsyncIOMotorClient
# # import httpx
# # import pinecone
# # from pypdf import PdfReader
# # from fastapi import APIRouter

# # from config import settings
# # from typing import Literal
# # from email_service import send_admin_signup_notification, send_password_reset_otp
# # import secrets
# # from payment_router import router as payment_router

# # # Optional libs for table/chart extraction + OCR
# # try:
# #     import pdfplumber
# # except Exception:
# #     # If packages aren't installed, we still allow server to start; upload will raise helpful errors
# #     convert_from_bytes = None
# #     pdfplumber = None
# #     Image = None

# # # ---------- FastAPI app & CORS ----------
# # app = FastAPI(title="CA Chatbot")

# # # Build allowed origins — supports comma-separated list in .env
# # # e.g. FRONTEND_ORIGIN=https://myapp.vercel.app,https://www.myapp.com
# # _raw_origin = settings.FRONTEND_ORIGIN.strip()
# # if _raw_origin == "*":
# #     _allow_origins = ["*"]
# # else:
# #     _allow_origins = [o.strip() for o in _raw_origin.split(",") if o.strip()]

# # app.add_middleware(
# #     CORSMiddleware,
# #     allow_origins=_allow_origins,
# #     allow_origin_regex=r"https://.*\.vercel\.app",  # covers all Vercel preview URLs
# #     allow_credentials=True,
# #     allow_methods=["*"],
# #     allow_headers=["*"],
# #     expose_headers=["*"],
# # )

# # app.include_router(payment_router, prefix="/payments", tags=["payments"])

# # CHAT_URL = "https://api.openai.com/v1/chat/completions"
# # EMBED_URL = "https://api.openai.com/v1/embeddings"

# # # ---------- DB & external clients ----------
# # pwd_context = CryptContext(
# #     schemes=["pbkdf2_sha256"],
# #     default="pbkdf2_sha256",
# #     deprecated="auto",
# # )

# # mongo_client = AsyncIOMotorClient(settings.MONGO_URI)
# # db = mongo_client[settings.MONGO_DB]
# # users_collection = db["users"]
# # docs_collection = db["documents"]
# # dashboard_collection = db["ca_dashboard"]

# # pinecone_client = pinecone.Pinecone(api_key=settings.PINECONE_API_KEY)
# # index = pinecone_client.Index(settings.PINECONE_INDEX)

# # JWT_EXP_MINUTES = 60 * 24

# # # Embedding configuration (tweak in config if desired)
# # EMBED_BATCH_SIZE = getattr(settings, "EMBED_BATCH_SIZE", 12)
# # EMBED_TIMEOUT_SECS = getattr(settings, "EMBED_TIMEOUT_SECS", 120)
# # EMBED_MAX_RETRIES = getattr(settings, "EMBED_MAX_RETRIES", 3)
# # EMBED_BACKOFF_BASE = getattr(settings, "EMBED_BACKOFF_BASE", 1.8)
# # MAX_TEXT_LENGTH_FOR_EMBED = getattr(settings, "MAX_TEXT_LENGTH_FOR_EMBED", 8000)

# # # Storage paths (local). Change to S3/presigned URLs if needed.
# # UPLOAD_ROOT = getattr(settings, "UPLOAD_ROOT", "./uploads")
# # os.makedirs(UPLOAD_ROOT, exist_ok=True)

# # app.mount("/uploads", StaticFiles(directory=UPLOAD_ROOT), name="uploads")
# # router = APIRouter(prefix="/admin/materials")

# # # ---------- Models ----------
# # class UserCreate(BaseModel):
# #     email: str
# #     password: str
# #     name: str
# #     phone: str
# #     ca_level: str
# #     ca_attempt: int
# #     role: str = "student"
# #     plan: Optional[str] = "free"          # "free" | "paid"
# #     payment_id: Optional[str] = None      # Razorpay payment ID (paid only)


# # class UserLogin(BaseModel):
# #     email: str
# #     password: str


# # class Token(BaseModel):
# #     access_token: str
# #     token_type: str = "bearer"


# # class UserOut(BaseModel):
# #     email: str
# #     role: str
# #     plan: Optional[str] = "free"
# #     subscription_status: Optional[str] = "free"


# # class ChatResponse(BaseModel):
# #     answer: str
# #     sources: List[dict]


# # class ChatMessage(BaseModel):
# #     role: Literal["user", "assistant"]
# #     content: str


# # class ChatRequest(BaseModel):
# #     message: str
# #     history: Optional[List[ChatMessage]] = None
# #     mode: Optional[str] = "qa"  # "qa" | "discussion"


# # class UploadResult(BaseModel):
# #     chunks: int
# #     filename: str

# # class DashboardItem(BaseModel):
# #     level: str
# #     subject: str
# #     module: str
# #     chapter: str
# #     unit: str
# #     title: str
# #     pdf_url: str
# #     video_url: Optional[str] = ""

# # # ---------- Utility functions ----------
# # def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
# #     to_encode = data.copy()
# #     expire = datetime.utcnow() + (expires_delta or timedelta(minutes=JWT_EXP_MINUTES))
# #     to_encode.update({"exp": expire})
# #     encoded_jwt = jwt.encode(
# #         to_encode, settings.JWT_SECRET, algorithm=settings.JWT_ALGO
# #     )
# #     return encoded_jwt


# # async def get_user_by_email(email: str):
# #     return await users_collection.find_one({"email": email})


# # def verify_password(plain: str, hashed: str) -> bool:
# #     return pwd_context.verify(plain, hashed)


# # def hash_password(password: str) -> str:
# #     return pwd_context.hash(password)


# # async def get_current_user(authorization: str = Header(None)):
# #     if not authorization:
# #         raise HTTPException(status_code=401, detail="Missing Authorization header")
# #     token = authorization.replace("Bearer ", "").strip()
# #     try:
# #         payload = jwt.decode(
# #             token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGO]
# #         )
# #         email: str = payload.get("sub")
# #         if email is None:
# #             raise HTTPException(status_code=401, detail="Invalid token payload")
# #     except JWTError:
# #         raise HTTPException(status_code=401, detail="Invalid token")

# #     user = await get_user_by_email(email)
# #     if not user:
# #         raise HTTPException(status_code=401, detail="User not found")
# #     return user


# # async def get_current_admin(user=Depends(get_current_user)):
# #     if user.get("role") != "admin":
# #         raise HTTPException(status_code=403, detail="Admin access required")
# #     return user



# # @app.post("/upload_new")
# # async def upload(
# #     file: UploadFile = File(...),

# #     course: str = Form(...),
# #     chapter: str = Form(None),
# #     section: str = Form(None),
# #     unit: str = Form(None),
# #     custom_heading: str = Form(None),

# #     admin=Depends(get_current_admin)
# # ):
# #     file_path = f"/tmp/{file.filename}"

# #     with open(file_path, "wb") as f:
# #         f.write(await file.read())

# #     # 🔥 Build full metadata object
# #     extra_meta = {
# #         "title": file.filename,
# #         "course": course,
# #         "level": course,  # optional mapping
# #         "subject": chapter,  # you can refine later
# #         "chapter": chapter,
# #         "section": section,
# #         "unit": unit,
# #         "custom_heading": custom_heading,
# #         "uploaded_by": admin["email"],
# #         "uploaded_at": datetime.utcnow().isoformat(),
# #     }

# #     count = await process_pdf(file_path, file.filename, extra_meta)

# #     return {"chunks": count}



# # """
# # Enhanced Upload API Endpoint

# # This replaces the /upload_new endpoint with advanced features:
# # - Progress tracking
# # - Detailed statistics
# # - Image and table extraction
# # - Better error handling
# # """

# # from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
# # from typing import Optional
# # from datetime import datetime
# # import os
# # import tempfile
# # from pydantic import BaseModel

# # # Import enhanced upload service
# # from ingestion.enhanced_upload_service import process_pdf_enhanced


# # # router = APIRouter()


# # class UploadResponse(BaseModel):
# #     """Enhanced response model with detailed statistics"""
# #     success: bool
# #     message: str
# #     filename: str
# #     statistics: dict
# #     metadata: dict


# # @router.post("/upload_enhanced", response_model=UploadResponse)
# # async def upload_pdf_enhanced(
# #     file: UploadFile = File(...),
    
# #     # Required metadata
# #     course: str = Form(...),  # Foundation / Intermediate / Final
    
# #     # Optional metadata for better organization
# #     subject: Optional[str] = Form(None),
# #     chapter: Optional[str] = Form(None),
# #     section: Optional[str] = Form(None),
# #     unit: Optional[str] = Form(None),
# #     module: Optional[str] = Form(None),
# #     custom_heading: Optional[str] = Form(None),
    
# #     # Processing options
# #     enable_image_descriptions: bool = Form(True),
    
# #     # Authentication (from your existing auth system)
# #     admin=Depends(get_current_admin)
# # ):
# #     """
# #     Enhanced PDF upload endpoint with advanced processing
    
# #     Features:
# #     - Extracts text, tables, and images
# #     - Generates AI descriptions for images
# #     - Creates optimized chunks for CA study materials
# #     - Returns detailed processing statistics
# #     """
    
# #     # Validate file type
# #     if not file.filename.lower().endswith(".pdf"):
# #         raise HTTPException(
# #             status_code=400,
# #             detail="Only PDF files are supported"
# #         )
    
# #     # Create temporary file
# #     temp_file = None
    
# #     try:
# #         # Save uploaded file to temporary location
# #         with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
# #             content = await file.read()
# #             temp_file.write(content)
# #             temp_file_path = temp_file.name
        
# #         # Build metadata
# #         extra_meta = {
# #             "title": file.filename,
# #             "course": course,
# #             "level": course,
# #             "subject": subject or chapter,  # Fallback to chapter if subject not provided
# #             "chapter": chapter,
# #             "section": section,
# #             "unit": unit,
# #             "module": module,
# #             "custom_heading": custom_heading,
# #             "uploaded_at": datetime.utcnow().isoformat(),
# #             # "uploaded_by": admin["email"],  # Uncomment when integrating auth
# #         }
        
# #         # Get API key from settings
# #         from config import settings
        
# #         # Process PDF with enhanced pipeline
# #         result = await process_pdf_enhanced(
# #             file_path=temp_file_path,
# #             file_name=file.filename,
# #             extra_meta=extra_meta,
# #             enable_image_descriptions=enable_image_descriptions,
# #             openai_api_key=settings.OPENAI_API_KEY
# #         )
        
# #         # Clean up temporary file
# #         try:
# #             os.unlink(temp_file_path)
# #         except:
# #             pass
        
# #         # Return success response with statistics
# #         return UploadResponse(
# #             success=True,
# #             message=f"Successfully processed {file.filename}",
# #             filename=file.filename,
# #             statistics={
# #                 "total_vectors": result["total_vectors"],
# #                 "text_chunks": result["text_chunks"],
# #                 "table_chunks": result["table_chunks"],
# #                 "image_chunks": result["image_chunks"],
# #                 "total_images_found": result["total_images"],
# #                 "total_tables_found": result["total_tables"],
# #             },
# #             metadata=extra_meta
# #         )
    
# #     except Exception as e:
# #         # Clean up temporary file on error
# #         if temp_file and os.path.exists(temp_file.name):
# #             try:
# #                 os.unlink(temp_file.name)
# #             except:
# #                 pass
        
# #         # Log error
# #         print(f"Error processing PDF {file.filename}: {str(e)}")
# #         import traceback
# #         traceback.print_exc()
        
# #         raise HTTPException(
# #             status_code=500,
# #             detail=f"Failed to process PDF: {str(e)}"
# #         )


# # # @router.post("/upload_batch")
# # # async def upload_multiple_pdfs(
# # #     # This endpoint can handle multiple files at once
# # #     # Implementation depends on your frontend requirements
# # #     pass
# # # ):
# # #     """
# # #     Batch upload endpoint for processing multiple PDFs
    
# # #     This can be implemented if you need to handle bulk uploads
# # #     """
# # #     # TODO: Implement batch processing
# # #     pass


# # # Health check for the upload service
# # @router.get("/upload_health")
# # async def upload_service_health():
# #     """
# #     Check if all required services are available
# #     """
# #     health_status = {
# #         "service": "upload_enhanced",
# #         "status": "operational",
# #         "features": {
# #             "docling_parser": True,
# #             "pdfplumber_tables": True,
# #             "pymupdf_images": True,
# #             "image_descriptions": True,
# #             "enhanced_chunking": True,
# #         }
# #     }
    
# #     # Check if required packages are available
# #     try:
# #         import docling
# #         import pdfplumber
# #         import fitz
        
# #         health_status["dependencies"] = "all_available"
    
# #     except ImportError as e:
# #         health_status["status"] = "degraded"
# #         health_status["dependencies"] = f"missing: {str(e)}"
    
# #     return health_status

# # app.include_router(router)

# # # ---------- ID sanitization helper ----------
# # def sanitize_id(s: str, max_len: int = 200) -> str:
# #     """
# #     Convert a string to an ASCII-safe id suitable for Pinecone vector IDs.
# #     - Normalize unicode to NFKD and drop non-ascii.
# #     - Replace non-alnum characters with underscores.
# #     - Collapse repeated underscores and trim.
# #     - Truncate to max_len.
# #     """
# #     if not s:
# #         return "id"
# #     nk = unicodedata.normalize("NFKD", s)
# #     ascii_bytes = nk.encode("ascii", "ignore")
# #     ascii_str = ascii_bytes.decode("ascii")
# #     replaced = re.sub(r"[^0-9A-Za-z]+", "_", ascii_str)
# #     collapsed = re.sub(r"_+", "_", replaced).strip("_")
# #     if not collapsed:
# #         collapsed = "id"
# #     if len(collapsed) > max_len:
# #         collapsed = collapsed[:max_len].rstrip("_")
# #     return collapsed


# # # ---------- Embedding & LLM (robust) ----------
# # async def embed_texts(texts: List[str]) -> List[List[float]]:
# #     """
# #     Robust embedding call:
# #       - Splits into smaller batches (EMBED_BATCH_SIZE)
# #       - Retries transient failures with exponential backoff
# #       - Uses EMBED_TIMEOUT_SECS per request
# #     Returns embeddings in same order as `texts`.
# #     """
# #     if not texts:
# #         return []

# #     url = EMBED_URL
# #     headers = {
# #         "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
# #         "Content-Type": "application/json",
# #     }

# #     # split into batches preserving order
# #     batches = [texts[i : i + EMBED_BATCH_SIZE] for i in range(0, len(texts), EMBED_BATCH_SIZE)]
# #     results: List[List[float]] = []

# #     async with httpx.AsyncClient(timeout=EMBED_TIMEOUT_SECS) as client:
# #         for batch_idx, batch_texts in enumerate(batches):
# #             payload = {"model": settings.EMBEDDING_MODEL, "input": batch_texts}
# #             attempt = 0
# #             while True:
# #                 attempt += 1
# #                 try:
# #                     resp = await client.post(url, headers=headers, json=payload)
# #                     resp.raise_for_status()
# #                     data = resp.json()
# #                     emb_batch = [d["embedding"] for d in data["data"]]
# #                     if len(emb_batch) != len(batch_texts):
# #                         raise HTTPException(status_code=502, detail="Embedding response length mismatch")
# #                     results.extend(emb_batch)
# #                     break
# #                 except (httpx.ReadTimeout, httpx.WriteTimeout, httpx.ConnectError, httpx.RemoteProtocolError) as exc:
# #                     # network/transient errors -> retry
# #                     if attempt >= EMBED_MAX_RETRIES:
# #                         raise HTTPException(
# #                             status_code=504,
# #                             detail=f"Embedding request timed out after {EMBED_MAX_RETRIES} attempts (batch {batch_idx}). Last error: {str(exc)}",
# #                         )
# #                     backoff = EMBED_BACKOFF_BASE ** (attempt - 1)
# #                     jitter = (0.1 * backoff) * (0.5 - (time.time() % 1))
# #                     await asyncio.sleep(backoff + jitter)
# #                     continue
# #                 except httpx.HTTPStatusError as exc:
# #                     status = exc.response.status_code
# #                     text = ""
# #                     try:
# #                         text = exc.response.text
# #                     except Exception:
# #                         pass
# #                     # Retry on 5xx
# #                     if 500 <= status < 600 and attempt < EMBED_MAX_RETRIES:
# #                         await asyncio.sleep(EMBED_BACKOFF_BASE ** (attempt - 1))
# #                         continue
# #                     raise HTTPException(
# #                         status_code=502,
# #                         detail=f"Embedding service returned status {status}: {text[:200]}",
# #                     )
# #                 except Exception as exc:
# #                     raise HTTPException(status_code=500, detail=f"Embedding failed: {str(exc)}")

# #     return results


# # async def embed_single(text: str) -> List[float]:
# #     embs = await embed_texts([text])
# #     return embs[0]


# # async def call_llm(messages: List[dict]) -> str:
# #     url = CHAT_URL
# #     headers = {
# #         "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
# #         "Content-Type": "application/json",
# #     }
# #     payload = {
# #         "model": settings.LLM_MODEL,
# #         "messages": messages,
# #         "temperature": 0.2
# #     }
# #     async with httpx.AsyncClient(timeout=90) as client:
# #         resp = await client.post(url, headers=headers, json=payload)
# #         resp.raise_for_status()
# #         data = resp.json()
# #         return data["choices"][0]["message"]["content"]

# # async def call_llm_with_chain(
# #     *,
# #     user_question: str,
# #     context: str,
# #     final_system_prompt: str,
# # ) -> str:
# #     """
# #     Internal 3-step reasoning chain for CA answers.
# #     Reasoning is hidden; only final answer is returned.
# #     """

# #     chain_prompt = (
# #         "You are reasoning internally as an expert Indian CA tutor.\n\n"
# #         "INTERNAL STEPS (do NOT reveal):\n"
# #         "1. Understand the exact exam intent of the question.First identify the overall concept being asked.\n"
# #         "2. Identify which parts of the context are relevant.Merge information from multiple context blocks\n"
# #         "3. Decide the depth needed as per ICAI expectations.Do NOT focus on only one standard unless the question specifically asks\n\n"

# #         "Then produce ONLY the final answer as instructed below.\n\n"
# #         f"{final_system_prompt}"
# #     )

# #     messages = [
# #         {"role": "system", "content": chain_prompt},
# #         {"role": "system", "content": f"CONTEXT:\n{context}"},
# #         {"role": "user", "content": user_question},
# #     ]

# #     return await call_llm(messages)

# # # ---------- Helpers ----------
# # def chunk_text(text: str, chunk_size: int = 800, overlap: int = 200) -> List[str]:
# #     words = text.split()
# #     chunks = []
# #     start = 0
# #     while start < len(words):
# #         end = min(start + chunk_size, len(words))
# #         chunk = " ".join(words[start:end])
# #         chunks.append(chunk)
# #         if end == len(words):
# #             break
# #         start += chunk_size - overlap
# #     return chunks


# # def extract_pdf_text(file_bytes: bytes) -> str:
# #     reader = PdfReader(io.BytesIO(file_bytes))
# #     full_text = ""
# #     for page in reader.pages:
# #         page_text = page.extract_text() or ""
# #         full_text += page_text + "\n"
# #     return full_text


# # def extract_pdf_pages(file_bytes: bytes) -> List[str]:
# #     """Return a list of raw text per page."""
# #     reader = PdfReader(io.BytesIO(file_bytes))
# #     pages: List[str] = []
# #     for page in reader.pages:
# #         page_text = page.extract_text() or ""
# #         pages.append(page_text)
# #     return pages


# # def normalize_page_text(text: str) -> str:
# #     """Clean up page text a bit."""
# #     # remove extra spaces, keep line breaks
# #     lines = [re.sub(r"\s+", " ", ln).strip() for ln in text.splitlines()]
# #     lines = [ln for ln in lines if ln]  # drop empty
# #     return "\n".join(lines)

# # def enrich_query_for_rag(question: str) -> str:
# #     q = question.lower()

# #     subject_hints = []

# #     if any(w in q for w in ["ind as", "financial", "asset", "liability", "consolidation"]):
# #         subject_hints.append("financial reporting accounting")

# #     if any(w in q for w in ["audit", "sa ", "assurance"]):
# #         subject_hints.append("auditing")

# #     if any(w in q for w in ["gst", "input tax", "itc"]):
# #         subject_hints.append("indirect tax gst")

# #     if any(w in q for w in ["tds", "income tax", "section 80"]):
# #         subject_hints.append("direct tax")

# #     if any(w in q for w in ["company act", "directors", "board"]):
# #         subject_hints.append("law")

# #     if subject_hints:
# #         return question + " " + " ".join(subject_hints)

# #     return question


# # def detect_headings_for_page(page_text: str) -> Dict[str, Optional[str]]:
# #     lines = [ln.strip() for ln in page_text.splitlines() if ln.strip()]
# #     chapter = None
# #     topic = None

# #     for ln in lines:
# #         lower = ln.lower()
# #         if chapter is None and (
# #             lower.startswith("chapter ")
# #             or lower.startswith("chap. ")
# #             or lower.startswith("paper ")
# #             or "paper -" in lower
# #         ):
# #             chapter = ln.strip()
# #             continue

# #         if topic is None:
# #             if len(ln) <= 80 and ln.upper() == ln and any(c.isalpha() for c in ln):
# #                 topic = ln.title().strip()

# #         if chapter and topic:
# #             break

# #     return {"chapter": chapter, "topic": topic}


# # def chunk_text_words(text: str, chunk_size: int = 180, overlap: int = 40) -> List[str]:
# #     words = text.split()
# #     chunks: List[str] = []
# #     start = 0
# #     n = len(words)
# #     while start < n:
# #         end = min(start + chunk_size, n)
# #         chunk = " ".join(words[start:end])
# #         if chunk.strip():
# #             chunks.append(chunk)
# #         if end == n:
# #             break
# #         start = end - overlap
# #         if start < 0:
# #             start = 0
# #     return chunks


# # def is_basic_ca_question(question: str) -> bool:
# #     q = question.lower().strip()

# #     BASIC_QUESTIONS = [
# #         "what is interim",
# #         "what is interim rules",
# #         "what is interim reporting",
# #         "define interim",
# #         "meaning of interim",
# #         "what is ind as 34",
# #         "interim financial reporting",
# #         "what is ca",
# #         "what is ca final",
# #     ]

# #     BASIC_KEYWORDS = [
# #         "accounting",
# #         "audit",
# #         "tax",
# #         "gst",
# #         "ind as",
# #         "financial reporting",
# #         "law",
# #     ]

# #     return any(b in q for b in BASIC_QUESTIONS) or any(k in q for k in BASIC_KEYWORDS)



# # def detect_query_focus(question: str) -> str:
# #     q = (question or "").lower()
# #     has_table = any(w in q for w in ["table", "tabular", "sheet", "schedule"])
# #     has_figure = any(w in q for w in ["chart", "diagram", "figure", "flowchart", "graph", "plot", "flow chart", "flow"])
# #     if has_table and has_figure:
# #         return "mixed"
# #     if has_table:
# #         return "table"
# #     if has_figure:
# #         return "figure"
# #     return "text"


# # async def is_ca_related_question(question: str) -> bool:
# #     # if is_basic_ca_question(question):
# #     #     return True

# #     system = (
# #         "You are a domain classifier for an Indian Chartered Accountancy (CA) assistant.\n\n"
    
# #         "Task:\n"
# #         "- Decide whether the user's question is related to CA studies or core commerce subjects in India.\n\n"
    
# #         "Answer YES if the question relates to:\n"
# #         "- ICAI syllabus (CA Foundation, Inter, Final)\n"
# #         "- Accounting (basic or advanced concepts)\n"
# #         "- Auditing and assurance\n"
# #         "- Direct Tax or GST\n"
# #         "- Corporate or Business Law\n"
# #         "- Financial management or costing\n"
# #         "- Economics or business studies taught in CA Foundation\n"
# #         "- Basic commerce concepts commonly studied by CA students\n\n"
    
# #         "IMPORTANT RULE:\n"
# #         "- If the concept is commonly taught in commerce or CA Foundation "
# #         "(such as mutual fund, contract, partnership, shares, debentures, capital, etc.), answer YES.\n"
# #         "- If the question is reasonably related to accounting, finance, taxation, or law, answer YES.\n\n"
    
# #         "Answer NO only if the question is clearly unrelated to commerce or CA, "
# #         "such as science, medical topics, coding, sports, entertainment, or general trivia.\n\n"
    
# #         "Respond strictly with YES or NO only."
# #     )

# #     messages = [
# #         {"role": "system", "content": system},
# #         {"role": "user", "content": question},
# #     ]

# #     try:
# #         result = await call_llm(messages)
# #         return result.strip().upper().startswith("YES")
# #     except Exception:
# #         return True


# # # # ---------- Charts / Tables extraction utilities ----------
# # # def ensure_doc_upload_folder(filename: str):
# # #     safe = sanitize_id(filename, max_len=80)
# # #     folder = os.path.join(UPLOAD_ROOT, safe)
# # #     os.makedirs(folder, exist_ok=True)
# # #     return folder


# # # def save_image_thumbnail(img: Image.Image, out_path: str, max_width: int = 800):
# # #     """Save thumbnail for the PIL Image object, maintain aspect ratio."""
# # #     if img is None:
# # #         return
# # #     w, h = img.size
# # #     if w > max_width:
# # #         ratio = max_width / float(w)
# # #         img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
# # #     img.save(out_path, format="PNG")


# # # def ocr_text_from_image(img: Image.Image) -> str:
# # #         return ""
# # #     try:
# # #         return txt.strip()
# # #     except Exception:
# # #         return ""


# # # def save_table_csv(table_rows: List[List[Any]], out_path: str):
# # #     with open(out_path, "w", newline="", encoding="utf-8") as f:
# # #         writer = csv.writer(f)
# # #         for row in table_rows:
# # #             # convert None to "" and ensure str values
# # #             writer.writerow([("" if c is None else str(c)) for c in row])


# # # def extract_tables_pdfplumber_from_bytes(file_bytes: bytes, out_dir: str) -> List[Dict[str, Any]]:
# # #     """
# # #     Use pdfplumber to extract tables. Returns list of metas:
# # #       {"page": int, "csv_path": str, "excerpt": str}
# # #     """
# # #     if pdfplumber is None:
# # #         raise HTTPException(status_code=500, detail="pdfplumber not installed on server")

# # #     metas: List[Dict[str, Any]] = []
# # #     with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
# # #         for p_index, page in enumerate(pdf.pages, start=1):
# # #             try:
# # #                 tables = page.extract_tables()
# # #             except Exception:
# # #                 tables = []
# # #             for t_idx, table in enumerate(tables):
# # #                 if not table or not any(any(cell for cell in row) for row in table):
# # #                     continue
# # #                 csv_name = f"table_p{p_index}_{t_idx}.csv"
# # #                 csv_path = os.path.join(out_dir, csv_name)
# # #                 save_table_csv(table, csv_path)
# # #                 # build a short excerpt (header + first row or first 2 rows)
# # #                 headers = table[0] if table and len(table) > 0 else []
# # #                 sample_rows = []
# # #                 if len(table) > 1:
# # #                     sample_rows = table[1:3]
# # #                 excerpt_parts = []
# # #                 if headers:
# # #                     excerpt_parts.append(" | ".join([str(h) for h in headers[:6]]))
# # #                 for r in sample_rows:
# # #                     excerpt_parts.append(" ; ".join([str(c) for c in (r[:6] if r else [])]))
# # #                 excerpt = " | ".join([p for p in excerpt_parts if p])
# # #                 metas.append({"page": p_index, "csv_path": csv_path, "excerpt": excerpt, "table_index": t_idx})
# # #     return metas


# # # def save_page_images_and_thumbs(file_bytes: bytes, out_dir: str, dpi: int = 200) -> List[Dict[str, Any]]:
# # #     """
# # #     Save page images and thumbnails. Returns list of page metas:
# # #      {"page": n, "image_path": path, "thumb_path": thumb_path}
# # #     """
# # #     if convert_from_bytes is None or Image is None:

# # #     pages = convert_from_bytes(file_bytes, dpi=dpi)
# # #     metas: List[Dict[str, Any]] = []
# # #     for idx, pil_img in enumerate(pages, start=1):
# # #         img_name = f"page_{idx}.png"
# # #         img_path = os.path.join(out_dir, img_name)
# # #         pil_img.save(img_path, format="PNG")
# # #         thumb_name = f"thumb_p{idx}.png"
# # #         thumb_path = os.path.join(out_dir, thumb_name)
# # #         save_image_thumbnail(pil_img, thumb_path, max_width=900)
# # #         metas.append({"page": idx, "image_path": img_path, "thumb_path": thumb_path})
# # #     return metas


# # # ---------- Auth routes ----------
# # # @app.post("/auth/signup", response_model=Token)
# # # async def signup(user: UserCreate):
# # #     existing = await get_user_by_email(user.email)
# # #     if existing:
# # #         raise HTTPException(status_code=400, detail="Email already registered")

# # #     await users_collection.insert_one(
# # #         {
# # #             "email": user.email,
# # #             "password_hash": hash_password(user.password),
# # #             "name": user.name,
# # #             "phone": user.phone,
# # #             "ca_level": user.ca_level,
# # #             "ca_attempt": user.ca_attempt,
# # #             "role": user.role,
# # #             "status": "pending",  # 🔥 important
# # #             "created_at": datetime.utcnow(),
# # #         }
# # #     )
# # #     token = create_access_token({"sub": user.email})
# # #     return Token(access_token=token)

# # @app.post("/auth/signup")
# # async def signup(user: UserCreate):
# #     existing = await get_user_by_email(user.email)
# #     if existing:
# #         raise HTTPException(status_code=400, detail="Email already registered")

# #     plan = (user.plan or "free").lower()
# #     if plan not in ("free", "paid"):
# #         plan = "free"

# #     # paid plan requires a payment_id
# #     if plan == "paid" and not user.payment_id:
# #         raise HTTPException(status_code=400, detail="payment_id is required for paid plan")

# #     now = datetime.utcnow()

# #     await users_collection.insert_one(
# #         {
# #             "email": user.email,
# #             "password_hash": hash_password(user.password),
# #             "name": user.name,
# #             "phone": user.phone,
# #             "ca_level": user.ca_level,
# #             "ca_attempt": user.ca_attempt,
# #             "role": "student",            # always force student on self-signup
# #             "status": "approved",         # auto-approved; admin can review later

# #             # ── Subscription fields ──────────────────────────────
# #             "plan": plan,                 # "free" | "paid"
# #             "subscription_status": "active" if plan == "paid" else "free",
# #             "payment_id": user.payment_id,
# #             "plan_activated_at": now if plan == "paid" else None,
# #             # next billing date = 30 days from now for paid users
# #             "plan_expires_at": (
# #                 datetime(now.year, now.month + 1 if now.month < 12 else 1,
# #                          now.day,
# #                          tzinfo=None)
# #                 if plan == "paid" else None
# #             ),
# #             # ─────────────────────────────────────────────────────
# #             "created_at": now,
# #         }
# #     )

# #     try:
# #         send_admin_signup_notification({**user.dict(), "plan": plan})
# #     except Exception as e:
# #         print("Email failed:", e)

# #     return {
# #         "message": "Signup successful. Please wait for admin approval before logging in."
# #     }

# # @app.post("/auth/login", response_model=Token)
# # async def login(data: UserLogin):
# #     user = await get_user_by_email(data.email)
# #     if not user or user.get("status") != "approved":
# #         raise HTTPException(
# #             status_code=403,
# #             detail="Your account is pending admin approval."
# #         )

# #     if not verify_password(data.password, user["password_hash"]):
# #         raise HTTPException(status_code=400, detail="Invalid credentials")
# #     token = create_access_token({"sub": user["email"]})
# #     return Token(access_token=token)


# # @app.get("/auth/me", response_model=UserOut)
# # async def me(user=Depends(get_current_user)):
# #     return UserOut(
# #         email=user["email"],
# #         role=user["role"],
# #         plan=user.get("plan", "free"),
# #         subscription_status=user.get("subscription_status", "free"),
# #     )


# # # ── Password reset models ────────────────────────────────────────────────────

# # class ForgotPasswordRequest(BaseModel):
# #     email: str

# # class VerifyOTPRequest(BaseModel):
# #     email: str
# #     otp: str

# # class ResetPasswordRequest(BaseModel):
# #     email: str
# #     otp: str
# #     new_password: str


# # # ── POST /auth/forgot-password ───────────────────────────────────────────────
# # @app.post("/auth/forgot-password")
# # async def forgot_password(body: ForgotPasswordRequest):
# #     user = await get_user_by_email(body.email)
# #     # Always return success to avoid email enumeration
# #     if not user:
# #         return {"message": "If this email is registered, an OTP has been sent."}

# #     otp = str(secrets.randbelow(900000) + 100000)   # 6-digit OTP
# #     expires_at = datetime.utcnow() + timedelta(minutes=10)

# #     await users_collection.update_one(
# #         {"email": body.email},
# #         {"$set": {
# #             "reset_otp":        otp,
# #             "reset_otp_expires": expires_at,
# #         }}
# #     )

# #     try:
# #         send_password_reset_otp(
# #             email=body.email,
# #             otp=otp,
# #             name=user.get("name", "Student"),
# #         )
# #     except Exception as e:
# #         print("OTP email failed:", e)
# #         raise HTTPException(status_code=500, detail="Failed to send OTP email. Please try again.")

# #     return {"message": "If this email is registered, an OTP has been sent."}


# # # ── POST /auth/verify-otp ────────────────────────────────────────────────────
# # @app.post("/auth/verify-otp")
# # async def verify_otp(body: VerifyOTPRequest):
# #     user = await get_user_by_email(body.email)
# #     if not user:
# #         raise HTTPException(status_code=400, detail="Invalid OTP or email.")

# #     stored_otp     = user.get("reset_otp")
# #     stored_expires = user.get("reset_otp_expires")

# #     if not stored_otp or not stored_expires:
# #         raise HTTPException(status_code=400, detail="No OTP requested. Please request a new one.")

# #     if datetime.utcnow() > stored_expires:
# #         raise HTTPException(status_code=400, detail="OTP has expired. Please request a new one.")

# #     if stored_otp != body.otp.strip():
# #         raise HTTPException(status_code=400, detail="Incorrect OTP. Please try again.")

# #     return {"message": "OTP verified."}


# # # ── POST /auth/reset-password ────────────────────────────────────────────────
# # @app.post("/auth/reset-password")
# # async def reset_password(body: ResetPasswordRequest):
# #     user = await get_user_by_email(body.email)
# #     if not user:
# #         raise HTTPException(status_code=400, detail="Invalid request.")

# #     stored_otp     = user.get("reset_otp")
# #     stored_expires = user.get("reset_otp_expires")

# #     if not stored_otp or not stored_expires:
# #         raise HTTPException(status_code=400, detail="No OTP requested. Please start over.")

# #     if datetime.utcnow() > stored_expires:
# #         raise HTTPException(status_code=400, detail="OTP has expired. Please request a new one.")

# #     if stored_otp != body.otp.strip():
# #         raise HTTPException(status_code=400, detail="Incorrect OTP.")

# #     if len(body.new_password) < 6:
# #         raise HTTPException(status_code=400, detail="Password must be at least 6 characters.")

# #     await users_collection.update_one(
# #         {"email": body.email},
# #         {"$set":   {"password_hash": hash_password(body.new_password)},
# #          "$unset": {"reset_otp": "", "reset_otp_expires": ""}}
# #     )

# #     return {"message": "Password reset successfully. You can now log in."}


# # @app.get("/admin/students")
# # async def get_all_students(admin=Depends(get_current_admin)):
# #     cursor = users_collection.find({"role": "student"})
# #     students = []
# #     async for user in cursor:
# #         user["_id"] = str(user["_id"])
# #         students.append(user)
# #     return students

# # from bson import ObjectId

# # @app.post("/admin/approve/{user_id}")
# # async def approve_student(user_id: str, admin=Depends(get_current_admin)):
# #     result = await users_collection.update_one(
# #         {"_id": ObjectId(user_id)},
# #         {"$set": {"status": "approved"}}
# #     )

# #     if result.matched_count == 0:
# #         raise HTTPException(status_code=404, detail="Student not found")

# #     return {"message": "Student approved"}

# # @app.post("/admin/reject/{user_id}")
# # async def reject_student(user_id: str, admin=Depends(get_current_admin)):
# #     await users_collection.delete_one({"_id": ObjectId(user_id)})
# #     return {"message": "Student rejected"}


# # # ============================================================
# # # CA DASHBOARD ROUTES
# # # ============================================================

# # @app.get("/dashboard/tree")
# # async def get_dashboard_tree(user=Depends(get_current_user)):
# #     cursor = dashboard_collection.find().sort("order", 1)

# #     tree = {}

# #     async for doc in cursor:
# #         doc["_id"] = str(doc["_id"])

# #         level = doc.get("level", "Others")
# #         subject = doc.get("subject", "General")
# #         module = doc.get("module", "General")
# #         chapter = doc.get("chapter", "General")

# #         tree.setdefault(level, {})
# #         tree[level].setdefault(subject, {})
# #         tree[level][subject].setdefault(module, {})
# #         tree[level][subject][module].setdefault(chapter, [])

# #         tree[level][subject][module][chapter].append(doc)

# #     return tree


# # @app.post("/dashboard/add")
# # async def add_dashboard_resource(
# #     level: str = Form(...),
# #     subject: str = Form(...),
# #     module: str = Form(...),
# #     chapter: str = Form(...),
# #     unit: str = Form(...),
# #     title: str = Form(...),
# #     pdf_url: str = Form(...),
# #     video_url: str = Form(""),
# #     admin=Depends(get_current_admin),
# # ):
# #     await dashboard_collection.insert_one({
# #         "level": level,
# #         "subject": subject,
# #         "module": module,
# #         "chapter": chapter,
# #         "unit": unit,
# #         "title": title,
# #         "pdf_url": pdf_url,
# #         "video_url": video_url,
# #         "created_at": datetime.utcnow()
# #     })

# #     return {"message": "Added successfully"}



# # def build_personalized_layer(user: dict) -> str:
# #     name = user.get("name", "Student").split()[0]
# #     level = user.get("ca_level", "Foundation")
# #     # attempt = user.get("ca_attempt", 1)

# #     # 🎯 Level-based teaching style
# #     level_guidance = {
# #         "Foundation": (
# #             "Explain in very simple language. Focus on basic concepts, definitions, "
# #             "and easy examples. Avoid heavy technical jargon."
# #         ),
# #         "Intermediate": (
# #             "Explain with clarity and practical understanding. Include examples "
# #             "and connect concepts logically."
# #         ),
# #         "Final": (
# #             "Provide detailed, professional-level explanation. Include case-based "
# #             "understanding, depth, and ICAI exam perspective."
# #         ),
# #     }

# #     encouragement = (
# #         f"provide extra clarity in topics, subjective detailed answers, motivation, "
# #         "and simplify difficult parts."
# #     )

# #     return f"""
# # PERSONALIZATION CONTEXT:
# # - Student Name: {name}
# # - Level: {level}

# # INSTRUCTIONS:
# # - Occasionally address the student as {name}
# # - Teaching style: {level_guidance.get(level, level_guidance["Foundation"])}
# # - Keep tone supportive and engaging
# # - {encouragement}
# # """

# # # ---------- Chat (RAG) ----------
# # @app.post("/chat", response_model=ChatResponse)
# # async def chat(req: ChatRequest, user=Depends(get_current_user)):
# #     try:
# #         # 🔥 NEW: Expand CA abbreviations BEFORE everything
# #         original_question = req.message
# #         expanded_question = expand_ca_abbreviations(original_question)

# #         # Use expanded question everywhere internally
# #         req.message = expanded_question
# #         # --------------------------------------------------
# #         # 1. CA gatekeeper
# #         # --------------------------------------------------
# #         is_ca = await is_ca_related_question(req.message)
# #         if not is_ca:
# #             return ChatResponse(
# #                 answer=(
# #                     "This assistant is designed for Indian CA students. "
# #                     "Please ask a question related to CA topics such as accounting, tax, audit, law, "
# #                     "or CA exams (Foundation / Inter / Final)."
# #                 ),
# #                 sources=[],
# #             )

# #         # --------------------------------------------------
# #         # 2. BASIC QUESTION → LLM ONLY
# #         # --------------------------------------------------
# #         # if is_basic_ca_question(req.message):
# #         #     system_prompt = (
# #         #         "You are a friendly Indian CA tutor. "
# #         #         "Explain clearly in simple language for CA students. "
# #         #         "Keep it concise and exam-oriented."
# #         #     )

# #         #     messages = [
# #         #         {"role": "system", "content": system_prompt},
# #         #         {"role": "user", "content": req.message},
# #         #     ]

# #         #     answer = await call_llm(messages)

# #         #     return ChatResponse(
# #         #         answer=answer,
# #         #         sources=[
# #         #             {
# #         #                 "doc_title": "Conceptual explanation",
# #         #                 "note": "General CA concept explained by the assistant",
# #         #             }
# #         #         ],
# #         #     )

# #         # --------------------------------------------------
# #         # 3. RAG FLOW (Pinecone)
# #         # --------------------------------------------------
# #         # 1. Query enrichment
# #         rag_query = enrich_query_for_rag(req.message)
# #         query_embedding = await embed_single(rag_query)

# #         # 2. Subject detection
# #         detected_subject = detect_subject(req.message)

# #         query_kwargs = {
# #             "vector": query_embedding,
# #             "top_k": 20,
# #             "include_metadata": True,
# #         }

# #         if detected_subject:
# #             query_kwargs["filter"] = {
# #                 "subject": {"$eq": detected_subject}
# #             }

# #         res = index.query(**query_kwargs)
# #         matches = res.get("matches") or []

# #         # fallback search without filter
# #         if len(matches) < 4 and detected_subject:
# #             res = index.query(
# #                 vector=query_embedding,
# #                 top_k=20,
# #                 include_metadata=True,
# #             )
# #             matches = res.get("matches") or []

# #         # sort by score
# #         matches = sorted(matches, key=lambda m: m.get("score", 0), reverse=True)

# #         # dynamic threshold
# #         q_len = len(req.message.split())
# #         threshold = 0.52 if q_len <= 6 else 0.60 if q_len <= 15 else 0.65

# #         filtered_matches = [m for m in matches if m.get("score", 0) >= threshold]

# #         if not filtered_matches:
# #             filtered_matches = matches[:5]

# #         matches = filtered_matches

# #         personal_context = build_personalized_layer(user)

# #         if not matches:
# #             system_prompt = (
# #                 personal_context + "\n\n" +
# #                 "You are a senior Indian Chartered Accountant (CA) faculty with experience "
# #                 "in teaching and evaluating ICAI exams (Foundation, Inter, Final).\n\n"

# #                 "Language rule (MANDATORY):\n"
# #                 "- Reply strictly in the SAME language as the user's question "
# #                 "(English, Hindi, or Hinglish).\n\n"

# #                 "Knowledge & safety rules:\n"
# #                 "- Answer using your standard CA knowledge and well-established ICAI principles.\n"
# #                 "- Do NOT guess exact section numbers, limits, percentages, or year-specific amendments.\n"
# #                 "- If precise data is uncertain, explain the concept without giving risky figures.\n\n"

# #                 "Answer structure (EXAM-ORIENTED):\n"
# #                 "1. Begin with a clear definition or core concept.\n"
# #                 "2. Explain in logical steps using proper CA terminology.\n"
# #                 "3. Where relevant, mention accounting treatment / legal position / tax implication.\n"
# #                 "4. Include ONE short exam-oriented or practical illustration if helpful.\n\n"

# #                 "Exam guidance:\n"
# #                 "- Add ONE short CA exam tip or common mistake to avoid.\n"
# #                 "- Keep the answer concise, structured, and revision-friendly.\n\n"

# #                 "Tone & presentation:\n"
# #                 "- Maintain a professional, faculty-level tone.\n"
# #                 "- Avoid casual language, storytelling, or over-explanation."
# #             )


# #             answer = await call_llm(
# #                 [
# #                     {"role": "system", "content": system_prompt},
# #                     {"role": "user", "content": req.message},
# #                 ]
# #             )

# #             return ChatResponse(
# #                 answer=answer,
# #                 sources=[
# #                     {
# #                         "doc_title": "General CA Knowledge (LLM based)",
# #                         "note": "No match found in uploaded documents",
# #                     }
# #                 ],
# #             )

# #         # --------------------------------------------------
# #         # 4. BUILD CONTEXT (SAFE SIZE)
# #         # --------------------------------------------------s
# #         context_blocks = []
# #         sources = []

# #         matches = sorted(matches, key=lambda m: m.get("score", 0), reverse=True)

# #         for m in matches:
# #             meta = m.get("metadata", {})
# #             text = meta.get("text", "")
# #             if not text:
# #                 continue

# #             header = []
# #             if meta.get("doc_title"):
# #                 header.append(f"Document: {meta['doc_title']}")
# #             if meta.get("chapter"):
# #                 header.append(f"Chapter: {meta['chapter']}")
# #             if meta.get("topic"):
# #                 header.append(f"Topic: {meta['topic']}")
# #             if meta.get("page_start"):
# #                 header.append(f"Page: {meta['page_start']}")

# #             block = f"{' | '.join(header)}\n{text}"
# #             context_blocks.append(block)

# #             doc_title = meta.get("doc_title")

# #             # ✅ skip sources without a title
# #             # if not doc_title:
# #             #     continue

# #             sources.append(
# #                 {
# #                     "doc_title": doc_title,
# #                     "source": meta.get("source"),  
# #                     "page_start": meta.get("page_start"),
# #                     "chapter": meta.get("chapter"),
# #                     "topic": meta.get("topic"),
# #                     "type": meta.get("type", "text"),
# #                 }
# #             )


# #         # ---- HARD CONTEXT LIMIT (IMPORTANT)
# #         MAX_CONTEXT_CHARS = 6000
# #         trimmed = []
# #         total = 0
# #         for b in context_blocks:
# #             if total + len(b) > MAX_CONTEXT_CHARS:
# #                 break
# #             trimmed.append(b)
# #             total += len(b)

# #         context_str = "\n\n---\n\n".join(trimmed)

# #         # --------------------------------------------------
# #         # 5. FINAL ANSWER (QA vs DISCUSSION)
# #         # --------------------------------------------------

# #         if req.mode == "discussion":
# #             system_prompt = (
# #                 personal_context + "\n\n" +
# #                 "You are an expert Indian CA tutor simulating a healthy academic discussion "
# #                 "between two CA students preparing for exams.\n\n"

# #                 "Language rules:\n"
# #                 "- Reply strictly in the SAME language as the user's question (English, Hindi, or Hinglish).\n\n"

# #                 "Discussion format rules:\n"
# #                 "- Write the answer as a discussion between 'User A:' and 'User B:'.\n"
# #                 "- Alternate clearly between User A and User B.\n"
# #                 # "- Do NOT use bullet points, markdown symbols, or asterisks.\n"
# #                 "- Provide at least 4 to 6 exchanges.\n\n"

# #                 "Content rules (VERY IMPORTANT):\n"
# #                 "- Explain concepts step-by-step in a teaching style.\n"
# #                 "- Keep explanations exam-oriented as per ICAI expectations.\n"
# #                 "- Use simple intuition first, then technical clarity.\n"
# #                 "- Include 1 very short practical or exam-oriented example if relevant.\n"
# #                 "- add a quick very short CA exam tip, memory aid, or common mistake to avoid.\n"
# #                 "- Avoid unnecessary storytelling or casual chat.\n\n"

# #                 "Source rules:\n"
# #                 "- Answer using the context provided below.\n"
# #                 "- invent facts but only well trusted and well tested ones based on the given context.\n\n"
# #                 f"Context:\n{context_str}"
# #             )

# #         else:
# #             system_prompt = (
# #                 personal_context + "\n\n" +
# #                 "You are an expert Indian Chartered Accountant (CA) tutor preparing students "
# #                 "for ICAI exams (Foundation, Inter, Final).\n\n"

# #                 "Language rule:\n"
# #                 "- Reply strictly in the SAME language as the user's question "
# #                 "(English, Hindi, or Hinglish).\n\n"

# #                 "Answering style rules:\n"
# #                 "- Answer using the context provided below.\n"
# #                 "- Keep the explanation clear, concise, and exam-oriented, in detail.\n"
# #                 "- Start with a direct definition or core concept in elaborative style.\n"
# #                 "- Then briefly explain or elaborate as required for marks.\n"
# #                 "- If applicable, include a short practical or exam-oriented example.\n"
# #                 "- If tables or figures are present in the context, refer to them explicitly.\n\n"

# #                 "Exam guidance:\n"
# #                 "- add one very short important CA exam tip or a common mistake to avoid in very short.\n"
# #                 "- Avoid unnecessary storytelling or over-explanation.\n"
# #                 # "- Do not use markdown symbols, bullet points, or asterisks.\n\n"

# #                 f"Context:\n{context_str}"
# #             )



# #         # answer = await call_llm(
# #         #     [
# #         #         {"role": "system", "content": system_prompt},
# #         #         {"role": "user", "content": req.message},
# #         #     ]
# #         # )
# #         answer = await call_llm_with_chain(
# #             user_question=req.message,
# #             context=context_str,
# #             final_system_prompt=system_prompt,
# #         )

# #         # --- CLEAN SOURCES (REMOVE UNKNOWN + DUPLICATES) ---
# #         clean_sources = []
# #         seen = set()

# #         for s in sources:
# #             title = s.get("doc_title") or s.get("source")
# #             # if not title:
# #             #     continue  # ❌ remove Unknown source

# #             key = (title, s.get("page_start"))
# #             if key in seen:
# #                 continue  # ❌ remove duplicates

# #             seen.add(key)
# #             clean_sources.append(s)


# #         return ChatResponse(answer=answer, sources=clean_sources)

# #     # --------------------------------------------------
# #     # 6. SAFE FALLBACK (NO BLANK ANSWERS)
# #     # --------------------------------------------------
# #     except Exception as e:
# #         print("CHAT ERROR:", str(e))

# #         answer = await call_llm(
# #             [
# #                 {
# #                     "role": "system",
# #                     "content": "You are a helpful Indian CA tutor.",
# #                 },
# #                 {"role": "user", "content": req.message},
# #             ]
# #         )

# #         return ChatResponse(
# #             answer=answer,
# #             sources=[
# #                 {
# #                     "doc_title": "LLM fallback",
# #                     "note": "Answered without document sources due to system issue",
# #                 }
# #             ],
# #         )


# # def detect_subject(question: str) -> Optional[str]:

# #     q = question.lower()

# #     if any(w in q for w in ["ind as", "asset", "liability", "financial statement", "consolidation"]):
# #         return "financial reporting"

# #     if any(w in q for w in ["audit", "sa ", "assurance", "audit report"]):
# #         return "audit"

# #     if any(w in q for w in ["gst", "itc", "input tax", "cgst", "sgst"]):
# #         return "indirect_tax"

# #     if any(w in q for w in ["income tax", "tds", "section 80", "capital gains"]):
# #         return "direct_tax"

# #     if any(w in q for w in ["company act", "director", "board", "mca"]):
# #         return "law"

# #     return None


# # # # ---------- Admin: upload PDF (enhanced: tables + figures extraction) ----------
# # # @app.post("/admin/upload_pdf", response_model=UploadResult)
# # # async def upload_pdf(
# # #     file: UploadFile = File(...),
# # #     # optional extra metadata from admin panel (JSON string)
# # #     metadata: Optional[str] = Form(None),
# # #     admin=Depends(get_current_admin),
# # # ):
# # #     """
# # #     Upload pipeline:
# # #      - extract page text chunks (existing)
# # #      - extract tables via pdfplumber -> save CSVs + index short summary vectors
# # #     """
# # #     if not file.filename.lower().endswith(".pdf"):
# # #         raise HTTPException(status_code=400, detail="Only PDF allowed")

# # #     # Parse metadata JSON if provided
# # #     doc_meta: Dict[str, Optional[str]] = {}
# # #     if metadata:
# # #         import json

# # #         try:
# # #             doc_meta = json.loads(metadata)
# # #         except Exception:
# # #             raise HTTPException(status_code=400, detail="Invalid metadata JSON")

# # #     # Set some defaults for CA Final if not provided
# # #     course = doc_meta.get("course") or "CA_FINAL"
# # #     subject = doc_meta.get("subject") or "Unknown Subject"
# # #     doc_type = doc_meta.get("doc_type") or "study_notes"
# # #     title = doc_meta.get("title") or file.filename
# # #     year = doc_meta.get("year")
# # #     version = doc_meta.get("version") or "v1"
# # #     author = doc_meta.get("author") or "Unknown"

# # #     file_bytes = await file.read()

# # #     # create per-document upload folder
# # #     upload_folder = ensure_doc_upload_folder(file.filename)

# # #     # --- Extract per-page text and detect headings ---
# # #     raw_pages = extract_pdf_pages(file_bytes)
# # #     pages = [normalize_page_text(p) for p in raw_pages]

# # #     chunks_for_index = []
# # #     chunk_global_index = 0

# # #     for page_num, page_text in enumerate(pages, start=1):
# # #         if not page_text.strip():
# # #             continue

# # #         heading_info = detect_headings_for_page(page_text)
# # #         chapter_guess = heading_info.get("chapter")
# # #         topic_guess = heading_info.get("topic")

# # #         # chunk this page's text
# # #         page_chunks = chunk_text_words(page_text, chunk_size=180, overlap=40)

# # #         for local_idx, chunk_text in enumerate(page_chunks):
# # #             raw_chunk_id = f"{file.filename}_p{page_num}_c{local_idx}"
# # #             chunk_id = sanitize_id(raw_chunk_id, max_len=200)
# # #             chunks_for_index.append(
# # #                 {
# # #                     "id": chunk_id,
# # #                     "text": chunk_text,
# # #                     "page_start": page_num,
# # #                     "page_end": page_num,
# # #                     "chapter": chapter_guess,
# # #                     "topic": topic_guess,
# # #                 }
# # #             )
# # #             chunk_global_index += 1

# # #     if not chunks_for_index:
# # #         raise HTTPException(status_code=400, detail="No text extracted from PDF")

# # #     # --- Extract tables (pdfplumber) and page images + OCR ---
# # #     table_metas = []
# # #     image_metas = []
# # #     try:
# # #         # tables
# # #         try:
# # #             table_metas = extract_tables_pdfplumber_from_bytes(file_bytes, upload_folder)
# # #         except HTTPException:
# # #             table_metas = []
# # #         # page images + thumbs
# # #         try:
# # #             image_metas = save_page_images_and_thumbs(file_bytes, upload_folder, dpi=200)
# # #         except HTTPException:
# # #             image_metas = []
# # #     except Exception as e:
# # #         # non-fatal - proceed but log
# # #         print("Warning: table/figure extraction failed:", str(e))
# # #         table_metas = table_metas or []
# # #         image_metas = image_metas or []

# # #     # --- Embed in batches & build vectors (text chunks + table summaries + figure ocr) ---
# # #     batch_size = EMBED_BATCH_SIZE
# # #     vectors = []

# # #     # 1) Text chunks (existing)
# # #     for i in range(0, len(chunks_for_index), batch_size):
# # #         batch = chunks_for_index[i : i + batch_size]
# # #         texts = [
# # #             (c["text"] if len(c["text"]) <= MAX_TEXT_LENGTH_FOR_EMBED else c["text"][:MAX_TEXT_LENGTH_FOR_EMBED])
# # #             for c in batch
# # #         ]
# # #         embeddings = await embed_texts(texts)
# # #         for c, emb in zip(batch, embeddings):
# # #             raw_meta = {
# # #                 "text": c["text"][:2000],  # limit stored text length (keep under Pinecone metadata limits)
# # #                 "source": file.filename,
# # #                 "doc_title": title,
# # #                 "course": course,
# # #                 "subject": subject,
# # #                 "doc_type": doc_type,
# # #                 "year": year,
# # #                 "version": version,
# # #                 "page_start": c["page_start"],
# # #                 "page_end": c["page_end"],
# # #                 "chapter": c["chapter"],
# # #                 "topic": c["topic"],
# # #                 "chunk_id": c["id"],
# # #                 "uploaded_by": admin["email"],
# # #                 "uploaded_at": datetime.utcnow().isoformat(),
# # #                 "author": author,
# # #                 "type": "text",
# # #             }

# # #             # Sanitize metadata: remove None values and ensure allowed types
# # #             sanitized_meta: Dict[str, object] = {}
# # #             for k, v in raw_meta.items():
# # #                 if v is None:
# # #                     continue
# # #                 if isinstance(v, (str, int, float, bool)):
# # #                     sanitized_meta[k] = v
# # #                 elif isinstance(v, list) and all(isinstance(i, str) for i in v):
# # #                     sanitized_meta[k] = v
# # #                 else:
# # #                     try:
# # #                         sanitized_meta[k] = str(v)
# # #                     except Exception:
# # #                         continue

# # #             vectors.append(
# # #                 {
# # #                     "id": c["id"],
# # #                     "values": emb,
# # #                     "metadata": sanitized_meta,
# # #                 }
# # #             )

# # #     # 2) Tables: index a short summary for each extracted table and attach CSV path
# # #     for tm in table_metas:
# # #         page = tm["page"]
# # #         csv_path = tm["csv_path"]
# # #         excerpt = tm.get("excerpt") or ""
# # #         summary = f"Table (page {page}) excerpt: {excerpt}"
# # #         # create small embedding
# # #         try:
# # #             emb = await embed_single(summary)
# # #         except Exception as e:
# # #             print("Table embed failed:", e)
# # #             continue

# # #         table_id = sanitize_id(f"{file.filename}_table_p{page}_{tm.get('table_index',0)}", max_len=200)
# # #         # create accessible URL path for UI (here local filesystem path; change to presigned S3 if needed)
# # #         csv_url = f"/uploads/{sanitize_id(file.filename, max_len=80)}/{os.path.basename(csv_path)}"
# # #         meta = {
# # #             "text": summary,
# # #             "source": file.filename,
# # #             "doc_title": title,
# # #             "page_start": page,
# # #             "page_end": page,
# # #             "table_csv_url": csv_url,
# # #             "type": "table",
# # #             "chunk_id": table_id,
# # #             "uploaded_by": admin["email"],
# # #             "uploaded_at": datetime.utcnow().isoformat(),
# # #         }
# # #         # sanitize meta values
# # #         sanitized_meta: Dict[str, object] = {}
# # #         for k, v in meta.items():
# # #             if v is None:
# # #                 continue
# # #             if isinstance(v, (str, int, float, bool)):
# # #                 sanitized_meta[k] = v
# # #             else:
# # #                 try:
# # #                     sanitized_meta[k] = str(v)
# # #                 except Exception:
# # #                     continue

# # #         vectors.append({"id": table_id, "values": emb, "metadata": sanitized_meta})

# # #     # 3) Figures/OCR: index OCR text from thumbnails (if available)
# # #     for im in image_metas:
# # #         page = im["page"]
# # #         thumb = im.get("thumb_path")
# # #         img_path = im.get("image_path")
# # #         ocr_text = ""
# # #             try:
# # #                 pil = Image.open(thumb)
# # #                 ocr_text = ocr_text_from_image(pil)
# # #             except Exception:
# # #                 ocr_text = ""
# # #         if ocr_text:
# # #             summary = f"Figure OCR (page {page}): {ocr_text[:800]}"
# # #             try:
# # #                 emb = await embed_single(summary)
# # #             except Exception as e:
# # #                 print("Figure embed failed:", e)
# # #                 continue

# # #             fig_id = sanitize_id(f"{file.filename}_fig_p{page}", max_len=200)
# # #             thumb_url = f"/uploads/{sanitize_id(file.filename, max_len=80)}/{os.path.basename(thumb)}" if thumb else None
# # #             meta = {
# # #                 "text": summary,
# # #                 "source": file.filename,
# # #                 "doc_title": title,
# # #                 "page_start": page,
# # #                 "page_end": page,
# # #                 "thumb_url": thumb_url,
# # #                 "type": "figure",
# # #                 "chunk_id": fig_id,
# # #                 "uploaded_by": admin["email"],
# # #                 "uploaded_at": datetime.utcnow().isoformat(),
# # #             }
# # #             sanitized_meta: Dict[str, object] = {}
# # #             for k, v in meta.items():
# # #                 if v is None:
# # #                     continue
# # #                 if isinstance(v, (str, int, float, bool)):
# # #                     sanitized_meta[k] = v
# # #                 else:
# # #                     try:
# # #                         sanitized_meta[k] = str(v)
# # #                     except Exception:
# # #                         continue

# # #             vectors.append({"id": fig_id, "values": emb, "metadata": sanitized_meta})

# # #     # --- Upsert to Pinecone ---
# # #     # namespace = course  # e.g. "CA_FINAL"
# # #     try:
# # #         # Upsert in batches (Pinecone expects reasonable sized upsert calls)
# # #         CHUNK = 100
# # #         for j in range(0, len(vectors), CHUNK):
# # #             slice_v = vectors[j : j + CHUNK]
# # #             index.upsert(vectors=slice_v)
# # #     except Exception as e:
# # #         import traceback

# # #         traceback.print_exc()
# # #         raise HTTPException(status_code=500, detail=f"Pinecone upsert failed: {str(e)}")

# # #     # --- Store doc-level metadata in Mongo ---
# # #     await docs_collection.insert_one(
# # #         {
# # #             "filename": file.filename,
# # #             "title": title,
# # #             "course": course,
# # #             "subject": subject,
# # #             "doc_type": doc_type,
# # #             "year": year,
# # #             "author": author,
# # #             "version": version,
# # #             "uploaded_by": admin["email"],
# # #             "uploaded_at": datetime.utcnow(),
# # #             "chunks": len(chunks_for_index),
# # #             "tables": len(table_metas),
# # #             "figures": len(image_metas),
# # #             # "namespace": namespace,
# # #         }
# # #     )

# # #     return UploadResult(chunks=len(chunks_for_index), filename=file.filename)


# # @app.get("/admin/documents")
# # async def list_documents(admin=Depends(get_current_admin)):
# #     docs_cursor = docs_collection.find().sort("uploaded_at", -1)
# #     docs = []
# #     async for d in docs_cursor:
# #         d["_id"] = str(d["_id"])
# #         docs.append(d)
# #     return docs


# # @app.get("/health")
# # async def health():
# #     return {"status": "ok"}


# # # ============================================================
# # # ADMIN MATERIALS ROUTES (Merged from admin_materials.py)
# # # ============================================================

# # # from fastapi import UploadFile, File, Form
# # # from typing import Optional, Dict, Any
# # # from bson import ObjectId
# # # import io
# # # from pypdf import PdfReader


# # # -----------------------------
# # # Utility: Extract text pages
# # # -----------------------------
# # # def extract_pdf_pages(file_bytes: bytes):
# # #     reader = PdfReader(io.BytesIO(file_bytes))
# # #     pages = []
# # #     for page in reader.pages:
# # #         text = page.extract_text() or ""
# # #         pages.append(text)
# # #     return pages


# # # ------------------------------------------------------------
# # # Upload PDF with structured metadata
# # # ------------------------------------------------------------
# # # @app.post("/admin/materials/upload", tags=["Admin Materials"])
# # # async def upload_material(
# # #     file: UploadFile = File(...),
# # #     course: str = Form(...),
# # #     chapter: Optional[str] = Form(None),
# # #     section: Optional[str] = Form(None),
# # #     unit: Optional[str] = Form(None),
# # #     custom_heading: Optional[str] = Form(None),
# # #     admin=Depends(get_current_admin),
# # # ):
# # #     if not file.filename.lower().endswith(".pdf"):
# # #         raise HTTPException(status_code=400, detail="Only PDF files allowed")

# # #     file_bytes = await file.read()
# # #     pages = extract_pdf_pages(file_bytes)

# # #     chunks_for_index = []

# # #     for page_num, page_text in enumerate(pages, start=1):
# # #         if not page_text.strip():
# # #             continue

# # #         page_chunks = chunk_text_words(page_text, chunk_size=200, overlap=50)

# # #         for idx, chunk in enumerate(page_chunks):
# # #             chunk_id = sanitize_id(f"{file.filename}_p{page_num}_c{idx}")

# # #             chunks_for_index.append(
# # #                 {
# # #                     "id": chunk_id,
# # #                     "text": chunk,
# # #                     "page": page_num,
# # #                 }
# # #             )

# # #     if not chunks_for_index:
# # #         raise HTTPException(status_code=400, detail="No text extracted from PDF")

# # #     # -----------------------------
# # #     # Embed & Push to Pinecone
# # #     # -----------------------------
# # #     texts = [c["text"] for c in chunks_for_index]
# # #     embeddings = await embed_texts(texts)

# # #     vectors = []

# # #     for chunk, emb in zip(chunks_for_index, embeddings):
# # #         metadata: Dict[str, Any] = {
# # #             "text": chunk["text"][:2000],
# # #             "source": file.filename,
# # #             "course": course,
# # #             "chapter": chapter,
# # #             "section": section,
# # #             "unit": unit,
# # #             "custom_heading": custom_heading,
# # #             "page": chunk["page"],
# # #             "uploaded_by": admin["email"],
# # #             "uploaded_at": datetime.utcnow().isoformat(),
# # #             "type": "text",
# # #         }

# # #         metadata = {k: v for k, v in metadata.items() if v is not None}

# # #         vectors.append(
# # #             {
# # #                 "id": chunk["id"],
# # #                 "values": emb,
# # #                 "metadata": metadata,
# # #             }
# # #         )

# # #     # Upsert in batches
# # #     BATCH_SIZE = 100
# # #     for i in range(0, len(vectors), BATCH_SIZE):
# # #         index.upsert(vectors=vectors[i : i + BATCH_SIZE])

# # #     # -----------------------------
# # #     # Store document-level record
# # #     # -----------------------------
# # #     await docs_collection.insert_one(
# # #         {
# # #             "filename": file.filename,
# # #             "course": course,
# # #             "chapter": chapter,
# # #             "section": section,
# # #             "unit": unit,
# # #             "custom_heading": custom_heading,
# # #             "uploaded_by": admin["email"],
# # #             "uploaded_at": datetime.utcnow(),
# # #             "chunks": len(chunks_for_index),
# # #         }
# # #     )

# # #     return {
# # #         "message": "Material uploaded successfully",
# # #         "chunks_indexed": len(chunks_for_index),
# # #     }


# # # ------------------------------------------------------------
# # # Get documents grouped by course
# # # ------------------------------------------------------------
# # @app.get("/admin/documents/grouped", tags=["Admin Materials"])
# # async def get_grouped_documents(admin=Depends(get_current_admin)):
# #     cursor = docs_collection.find()
# #     grouped = {}

# #     async for doc in cursor:
# #         doc["_id"] = str(doc["_id"])
# #         course = doc.get("course", "Other")

# #         if course not in grouped:
# #             grouped[course] = []

# #         grouped[course].append(doc)

# #     return grouped


# # # ------------------------------------------------------------
# # # Delete Document
# # # ------------------------------------------------------------
# # @app.delete("/admin/materials/{doc_id}", tags=["Admin Materials"])
# # async def delete_document(doc_id: str, admin=Depends(get_current_admin)):
# #     result = await docs_collection.delete_one({"_id": ObjectId(doc_id)})

# #     if result.deleted_count == 0:
# #         raise HTTPException(status_code=404, detail="Document not found")

# #     return {"message": "Document deleted successfully"}

# # # ============================================================
# # # ADVANCED ADMIN PDF UPLOAD (COLAB FORMAT COMPATIBLE)
# # # ============================================================

# # # import uuid
# # # from pypdf import PdfReader

# # # # -----------------------------
# # # # Clean Text
# # # # -----------------------------
# # # def clean_text(text: str) -> str:
# # #     text = re.sub(r"\s+", " ", text)
# # #     return text.strip()


# # # -----------------------------
# # # Extract PDF Pages
# # # -----------------------------
# # # def extract_pdf_pages_advanced(file_bytes: bytes):
# # #     reader = PdfReader(io.BytesIO(file_bytes))
# # #     pages = []

# # #     for i, page in enumerate(reader.pages):
# # #         text = page.extract_text()
# # #         if text and len(text.strip()) > 50:
# # #             pages.append({
# # #                 "page": i + 1,
# # #                 "text": clean_text(text)
# # #             })

# # #     return pages


# # # -----------------------------
# # # # Token-based Chunking (Production Ready)
# # # # -----------------------------
# # # def chunk_text_advanced(text: str, chunk_size: int = 1000, overlap: int = 100):
# # #     words = text.split()
# # #     chunks = []

# # #     start = 0
# # #     while start < len(words):
# # #         end = start + chunk_size
# # #         chunk = " ".join(words[start:end])
# # #         chunks.append(chunk)
# # #         start += chunk_size - overlap

# # #     return chunks


# # # ============================================================
# # # SINGLE ADVANCED UPLOAD ROUTE
# # # ============================================================

# # # @app.post("/admin/materials/upload", tags=["Admin Materials"])
# # # async def upload_advanced_pdf(
# # #     file: UploadFile = File(...),
# # #     level: str = Form(...),           # Foundation / Intermediate / Final
# # #     subject: str = Form(...),
# # #     chapter: str = Form(...),
# # #     doc_type: str = Form("static"),
# # #     year: str = Form("2026"),
# # #     authority: str = Form("ICAI"),
# # #     admin=Depends(get_current_admin),
# # # ):

# # #     if not file.filename.lower().endswith(".pdf"):
# # #         raise HTTPException(status_code=400, detail="Only PDF files allowed")

# # #     file_bytes = await file.read()

# # #     pages = extract_pdf_pages_advanced(file_bytes)

# # #     if not pages:
# # #         raise HTTPException(status_code=400, detail="No readable content found")

# # #     vectors = []

# # #     for page in pages:
# # #         chunks = chunk_text_advanced(page["text"])

# # #         embeddings = await embed_texts(chunks)

# # #         for chunk, emb in zip(chunks, embeddings):

# # #             metadata = {
# # #                 "level": level,
# # #                 "subject": subject,
# # #                 "chapter": chapter,
# # #                 "doc_type": doc_type,
# # #                 "year": year,
# # #                 "authority": authority,
# # #                 "source": file.filename,
# # #                 "page": page["page"],
# # #                 "text": chunk,
# # #                 "uploaded_by": admin["email"],
# # #                 "uploaded_at": datetime.utcnow().isoformat(),
# # #             }

# # #             vectors.append({
# # #                 "id": str(uuid.uuid4()),   # UUID like Colab
# # #                 "values": emb,
# # #                 "metadata": metadata
# # #             })

# # #     # Batch Upsert (No Namespace → Default)
# # #     BATCH_SIZE = 100

# # #     for i in range(0, len(vectors), BATCH_SIZE):
# # #         index.upsert(
# # #             vectors=vectors[i:i + BATCH_SIZE]
# # #         )

# # #     # Store Document Record in Mongo
# # #     await docs_collection.insert_one({
# # #         "filename": file.filename,
# # #         "level": level,
# # #         "subject": subject,
# # #         "chapter": chapter,
# # #         "doc_type": doc_type,
# # #         "year": year,
# # #         "authority": authority,
# # #         "uploaded_by": admin["email"],
# # #         "uploaded_at": datetime.utcnow(),
# # #         "total_chunks": len(vectors),
# # #     })

# # #     return {
# # #         "message": "Upload successful",
# # #         "total_chunks_indexed": len(vectors)
# # #     }


# # # dashboard_collection = db["ca_dashboard"]

# # # @app.get("/dashboard/tree")
# # # async def get_dashboard_tree(user=Depends(get_current_user)):
# # #     cursor = dashboard_collection.find().sort("order", 1)

# # #     tree = {}

# # #     async for doc in cursor:
# # #         doc["_id"] = str(doc["_id"])

# # #         level = doc.get("level", "Other")
# # #         subject = doc.get("subject", "General")
# # #         module = doc.get("module", "General")
# # #         chapter = doc.get("chapter", "General")

# # #         tree.setdefault(level, {})
# # #         tree[level].setdefault(subject, {})
# # #         tree[level][subject].setdefault(module, {})
# # #         tree[level][subject][module].setdefault(chapter, [])

# # #         tree[level][subject][module][chapter].append(doc)

# # #     return tree




# # # @app.post("/dashboard/add")
# # # async def add_dashboard_resource(
# # #     level: str = Form(...),
# # #     subject: str = Form(...),
# # #     module: str = Form(...),
# # #     chapter: str = Form(...),
# # #     unit: str = Form(...),
# # #     title: str = Form(...),
# # #     pdf_url: str = Form(...),
# # #     video_url: str = Form(""),
# # #     admin=Depends(get_current_admin),
# # # ):
# # #     await dashboard_collection.insert_one({
# # #         "level": level,
# # #         "subject": subject,
# # #         "module": module,
# # #         "chapter": chapter,
# # #         "unit": unit,
# # #         "title": title,
# # #         "pdf_url": pdf_url,
# # #         "video_url": video_url,
# # #         "created_at": datetime.utcnow()
# # #     })

# # #     return {"message": "Added"}


# # backend/main.py
# import io
# import os
# import unicodedata
# import asyncio
# import time
# import csv
# from datetime import datetime, timedelta
# from typing import Any, List, Optional, Dict
# import re
# import tempfile
# import secrets
# import traceback

# from fastapi import (
#     FastAPI,
#     HTTPException,
#     Depends,
#     UploadFile,
#     File,
#     Header,
#     Form,
#     APIRouter,
# )
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.staticfiles import StaticFiles
# from pydantic import BaseModel
# from jose import jwt, JWTError
# from passlib.context import CryptContext
# from motor.motor_asyncio import AsyncIOMotorClient
# from bson import ObjectId
# import httpx
# import pinecone
# from pypdf import PdfReader
# from typing import Literal

# from config import settings
# from ca_text_normalizer import expand_ca_abbreviations
# from ingestion.enhanced_upload_service import process_pdf, process_pdf_enhanced
# from email_service import send_admin_signup_notification, send_password_reset_otp
# from payment_router import router as payment_router
# from s3_service import upload_pdf_to_s3, delete_pdf_from_s3, is_s3_configured

# # ============================================================
# # APP + CORS
# # ============================================================

# app = FastAPI(title="CA Chatbot")

# _raw_origin    = settings.FRONTEND_ORIGIN.strip()
# _allow_origins = ["*"] if _raw_origin == "*" else [
#     o.strip() for o in _raw_origin.split(",") if o.strip()
# ]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=_allow_origins,
#     allow_origin_regex=r"https://.*\.vercel\.app",
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
#     expose_headers=["*"],
# )

# app.include_router(payment_router, prefix="/payments", tags=["payments"])

# CHAT_URL = "https://api.openai.com/v1/chat/completions"
# EMBED_URL = "https://api.openai.com/v1/embeddings"


# # ============================================================
# # DB + EXTERNAL CLIENTS
# # ============================================================

# pwd_context = CryptContext(schemes=["pbkdf2_sha256"], default="pbkdf2_sha256", deprecated="auto")

# mongo_client         = AsyncIOMotorClient(settings.MONGO_URI)
# db                   = mongo_client[settings.MONGO_DB]
# users_collection     = db["users"]
# docs_collection      = db["documents"]
# dashboard_collection = db["ca_dashboard"]

# pinecone_client = pinecone.Pinecone(api_key=settings.PINECONE_API_KEY)
# index           = pinecone_client.Index(settings.PINECONE_INDEX)

# JWT_EXP_MINUTES           = 60 * 24
# EMBED_BATCH_SIZE          = getattr(settings, "EMBED_BATCH_SIZE",          12)
# EMBED_TIMEOUT_SECS        = getattr(settings, "EMBED_TIMEOUT_SECS",        120)
# EMBED_MAX_RETRIES         = getattr(settings, "EMBED_MAX_RETRIES",         3)
# EMBED_BACKOFF_BASE        = getattr(settings, "EMBED_BACKOFF_BASE",        1.8)
# MAX_TEXT_LENGTH_FOR_EMBED = getattr(settings, "MAX_TEXT_LENGTH_FOR_EMBED", 8000)

# # Local uploads folder — used as fallback when S3 is not configured
# UPLOAD_ROOT = getattr(settings, "UPLOAD_ROOT", "./uploads")
# os.makedirs(UPLOAD_ROOT, exist_ok=True)
# app.mount("/uploads", StaticFiles(directory=UPLOAD_ROOT), name="uploads")


# # ============================================================
# # PYDANTIC MODELS
# # ============================================================

# class UserCreate(BaseModel):
#     email: str
#     password: str
#     name: str
#     phone: str
#     ca_level: str
#     ca_attempt: int
#     role: str = "student"
#     plan: Optional[str] = "free"
#     payment_id: Optional[str] = None

# class UserLogin(BaseModel):
#     email: str
#     password: str

# class Token(BaseModel):
#     access_token: str
#     token_type: str = "bearer"

# class UserOut(BaseModel):
#     email: str
#     role: str
#     plan: Optional[str] = "free"
#     subscription_status: Optional[str] = "free"

# class ChatResponse(BaseModel):
#     answer: str
#     sources: List[dict]

# class ChatMessage(BaseModel):
#     role: Literal["user", "assistant"]
#     content: str

# class ChatRequest(BaseModel):
#     message: str
#     history: Optional[List[ChatMessage]] = None
#     mode: Optional[str] = "qa"

# class UploadResult(BaseModel):
#     chunks: int
#     filename: str

# class DashboardItem(BaseModel):
#     level: str
#     subject: str
#     module: str
#     chapter: str
#     unit: str
#     title: str
#     pdf_url: str
#     video_url: Optional[str] = ""

# class UploadResponse(BaseModel):
#     success: bool
#     message: str
#     filename: str
#     statistics: dict
#     metadata: dict

# class ForgotPasswordRequest(BaseModel):
#     email: str

# class VerifyOTPRequest(BaseModel):
#     email: str
#     otp: str

# class ResetPasswordRequest(BaseModel):
#     email: str
#     otp: str
#     new_password: str


# # ============================================================
# # AUTH HELPERS
# # ============================================================

# def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
#     to_encode = data.copy()
#     expire    = datetime.utcnow() + (expires_delta or timedelta(minutes=JWT_EXP_MINUTES))
#     to_encode.update({"exp": expire})
#     return jwt.encode(to_encode, settings.JWT_SECRET, algorithm=settings.JWT_ALGO)

# async def get_user_by_email(email: str):
#     return await users_collection.find_one({"email": email})

# def verify_password(plain: str, hashed: str) -> bool:
#     return pwd_context.verify(plain, hashed)

# def hash_password(password: str) -> str:
#     return pwd_context.hash(password)

# async def get_current_user(authorization: str = Header(None)):
#     if not authorization:
#         raise HTTPException(status_code=401, detail="Missing Authorization header")
#     token = authorization.replace("Bearer ", "").strip()
#     try:
#         payload = jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGO])
#         email: str = payload.get("sub")
#         if not email:
#             raise HTTPException(status_code=401, detail="Invalid token payload")
#     except JWTError:
#         raise HTTPException(status_code=401, detail="Invalid token")
#     user = await get_user_by_email(email)
#     if not user:
#         raise HTTPException(status_code=401, detail="User not found")
#     return user

# async def get_current_admin(user=Depends(get_current_user)):
#     if user.get("role") != "admin":
#         raise HTTPException(status_code=403, detail="Admin access required")
#     return user


# # ============================================================
# # ADMIN MATERIALS ROUTER
# # ============================================================

# router = APIRouter(prefix="/admin/materials")


# # ----------------------------------------------------------
# # UPLOAD  →  S3 (or local)  +  Pinecone  +  MongoDB
# # ----------------------------------------------------------

# @router.post("/upload_enhanced", response_model=UploadResponse)
# async def upload_pdf_enhanced(
#     file:    UploadFile        = File(...),
#     course:  str               = Form(...),          # Foundation / Intermediate / Final / Other
#     subject: Optional[str]     = Form(None),         # e.g. Accounting, Tax, Audit
#     module:  Optional[str]     = Form(None),         # e.g. Module 1, Paper 2
#     chapter: Optional[str]     = Form(None),         # e.g. Chapter 5: Depreciation
#     unit:    Optional[str]     = Form(None),         # e.g. Unit 1 – Overview
#     section: Optional[str]     = Form(None),
#     custom_heading: Optional[str] = Form(None),
#     enable_image_descriptions: bool = Form(True),
#     admin = Depends(get_current_admin),
# ):
#     """
#     Complete PDF ingestion pipeline:

#     1. Read file bytes once (reused for S3 and Pinecone processing)
#     2. Upload PDF to AWS S3  →  get public/presigned URL
#        └─ Falls back to local /uploads folder if S3 is not configured
#     3. Write to a temp file, run Pinecone embedding pipeline
#        └─ pdf_url stored in every Pinecone vector metadata for RAG linking
#     4. Insert record in MongoDB `documents`      (AdminUpload list)
#     5. Insert record in MongoDB `ca_dashboard`   (CADashboard tree)
#     """

#     if not file.filename.lower().endswith(".pdf"):
#         raise HTTPException(status_code=400, detail="Only PDF files are supported")

#     temp_file_path: Optional[str] = None

#     try:
#         # ── 1. Read bytes once ───────────────────────────────────────────────
#         content = await file.read()

#         # ── 2. Resolve metadata fields ───────────────────────────────────────
#         level            = course.strip()
#         resolved_subject = (subject or chapter or "General").strip()
#         resolved_module  = (module  or section or "General").strip()
#         resolved_chapter = (chapter or "").strip()
#         resolved_unit    = (unit    or "").strip()
#         safe_filename    = file.filename.replace(" ", "_")

#         # ── 3. Store PDF ─────────────────────────────────────────────────────
#         # Log the storage decision so it's visible in server logs
#         s3_ready = is_s3_configured()
#         print(f"[upload] S3 configured={s3_ready}  file={safe_filename}")

#         if s3_ready:
#             # upload_pdf_to_s3 now RAISES on failure so we know the exact error
#             try:
#                 pdf_url         = upload_pdf_to_s3(
#                     file_bytes = content,
#                     filename   = safe_filename,
#                     level      = level,
#                     subject    = resolved_subject,
#                 )
#                 storage_backend = "s3"
#                 print(f"[upload] ✅ S3 upload OK → {pdf_url}")
#             except RuntimeError as s3_err:
#                 # S3 is configured but upload failed — fall back to local
#                 # and record the warning so it appears in the upload stats
#                 print(f"[upload] ⚠️  S3 failed, falling back to local: {s3_err}")
#                 pdf_url         = _save_local(content, safe_filename)
#                 storage_backend = f"local_fallback (s3_error: {s3_err})"
#         else:
#             pdf_url         = _save_local(content, safe_filename)
#             storage_backend = "local"
#             print(f"[upload] S3 not configured — stored locally: {pdf_url}")

#         # ── 4. Pinecone processing ────────────────────────────────────────────
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
#             tmp.write(content)
#             temp_file_path = tmp.name

#         now = datetime.utcnow()

#         extra_meta = {
#             "title":          file.filename,
#             "course":         course,
#             "level":          level,
#             "subject":        resolved_subject,
#             "chapter":        resolved_chapter,
#             "section":        section        or "",
#             "unit":           resolved_unit,
#             "module":         resolved_module,
#             "custom_heading": custom_heading or "",
#             "uploaded_by":    admin["email"],
#             "uploaded_at":    now.isoformat(),
#             # PDF URL stored in vector metadata so RAG can surface source links
#             "pdf_url":        pdf_url        or "",
#         }

#         result = await process_pdf_enhanced(
#             file_path                 = temp_file_path,
#             file_name                 = file.filename,
#             extra_meta                = extra_meta,
#             enable_image_descriptions = enable_image_descriptions,
#             openai_api_key            = settings.OPENAI_API_KEY,
#         )

#         # ── 5. Insert into MongoDB `documents` ───────────────────────────────
#         doc_record = {
#             "filename":        file.filename,
#             "safe_filename":   safe_filename,
#             "course":          course,
#             "level":           level,
#             "subject":         resolved_subject,
#             "chapter":         resolved_chapter,
#             "section":         section        or "",
#             "unit":            resolved_unit,
#             "module":          resolved_module,
#             "custom_heading":  custom_heading or "",
#             "pdf_url":         pdf_url        or "",
#             "storage_backend": storage_backend,
#             "uploaded_by":     admin["email"],
#             "uploaded_at":     now,
#             "total_vectors":   result["total_vectors"],
#         }
#         inserted      = await docs_collection.insert_one(doc_record)
#         doc_id_str    = str(inserted.inserted_id)

#         # ── 6. Insert into MongoDB `ca_dashboard` ────────────────────────────
#         await dashboard_collection.insert_one({
#             "level":       level            if level             else "Others",
#             "subject":     resolved_subject,
#             "module":      resolved_module   if resolved_module   else resolved_subject,
#             "chapter":     resolved_chapter  if resolved_chapter  else file.filename.replace(".pdf", ""),
#             "unit":        resolved_unit,
#             "title":       file.filename.replace(".pdf", "").replace("_", " "),
#             "pdf_url":     pdf_url           or "",
#             "video_url":   "",
#             "source_doc":  doc_id_str,
#             "uploaded_by": admin["email"],
#             "created_at":  now,
#         })

#         _safe_unlink(temp_file_path)

#         return UploadResponse(
#             success    = True,
#             message    = f"Successfully processed '{file.filename}' (storage: {storage_backend})",
#             filename   = file.filename,
#             statistics = {
#                 "total_vectors":      result["total_vectors"],
#                 "text_chunks":        result["text_chunks"],
#                 "table_chunks":       result["table_chunks"],
#                 "image_chunks":       result["image_chunks"],
#                 "total_images_found": result["total_images"],
#                 "total_tables_found": result["total_tables"],
#                 "storage_backend":    storage_backend,
#             },
#             metadata = extra_meta,
#         )

#     except Exception as e:
#         _safe_unlink(temp_file_path)
#         print(f"[upload_pdf_enhanced] Error — {file.filename}: {e}")
#         traceback.print_exc()
#         raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")


# # ----------------------------------------------------------
# # DELETE  →  Pinecone  +  S3/local  +  Mongo docs  +  Mongo dashboard
# # ----------------------------------------------------------

# @router.delete("/{doc_id}", tags=["Admin Materials"])
# async def delete_document(doc_id: str, admin=Depends(get_current_admin)):
#     """
#     Full cascading delete:

#     1. Fetch document record from MongoDB
#     2. Delete all Pinecone vectors whose metadata.source == filename
#     3. Delete the PDF from S3 (or local storage)
#     4. Delete the record from MongoDB `documents`
#     5. Delete all linked records from MongoDB `ca_dashboard`

#     Returns a detailed deletion report so the admin can see exactly
#     what was removed from each system.
#     """

#     # ── 1. Fetch record ──────────────────────────────────────────────────────
#     try:
#         obj_id = ObjectId(doc_id)
#     except Exception:
#         raise HTTPException(status_code=400, detail="Invalid document ID format")

#     doc = await docs_collection.find_one({"_id": obj_id})
#     if not doc:
#         raise HTTPException(status_code=404, detail="Document not found")

#     filename        = doc.get("filename",        "")
#     pdf_url         = doc.get("pdf_url",         "")
#     safe_filename   = doc.get("safe_filename",   filename.replace(" ", "_"))
#     storage_backend = doc.get("storage_backend", "local")

#     report: Dict[str, Any] = {
#         "doc_id":            doc_id,
#         "filename":          filename,
#         "pinecone_deleted":  0,
#         "s3_deleted":        False,
#         "local_deleted":     False,
#         "mongo_docs":        False,
#         "mongo_dashboard":   0,
#         "errors":            [],
#     }

#     # ── 2. Delete Pinecone vectors ───────────────────────────────────────────
#     try:
#         report["pinecone_deleted"] = await _delete_pinecone_by_source(filename)
#     except Exception as e:
#         report["errors"].append(f"Pinecone: {e}")

#     # ── 3. Delete file from S3 or local ─────────────────────────────────────
#     if storage_backend == "s3" and pdf_url:
#         try:
#             ok = delete_pdf_from_s3(pdf_url)
#             report["s3_deleted"] = ok
#             if not ok:
#                 report["errors"].append("S3 delete returned False — file may not exist in bucket")
#         except Exception as e:
#             report["errors"].append(f"S3: {e}")
#     else:
#         # Always also try local in case of local_fallback
#         local_path = os.path.join(UPLOAD_ROOT, safe_filename)
#         try:
#             if os.path.exists(local_path):
#                 os.remove(local_path)
#                 report["local_deleted"] = True
#         except Exception as e:
#             report["errors"].append(f"Local file: {e}")

#     # ── 4. Delete from MongoDB `documents` ───────────────────────────────────
#     del_result             = await docs_collection.delete_one({"_id": obj_id})
#     report["mongo_docs"]   = del_result.deleted_count > 0

#     # ── 5. Delete from MongoDB `ca_dashboard` ────────────────────────────────
#     dash_result              = await dashboard_collection.delete_many({"source_doc": doc_id})
#     report["mongo_dashboard"] = dash_result.deleted_count

#     return {
#         "message": f"'{filename}' deleted successfully",
#         "report":  report,
#     }


# async def _delete_pinecone_by_source(source_filename: str) -> int:
#     """
#     Delete all Pinecone vectors whose metadata.source == source_filename.

#     Uses a dummy zero-vector query with a metadata filter to page through
#     all matching IDs, then batch-deletes them.  Handles indexes that don't
#     support metadata filtering gracefully (returns 0 without raising).
#     """
#     # Dimension must match your index text-embedding-3-large Dimensions 3072
#     try:
#         index_info = index.describe_index_stats()
#         dim = index_info.get("dimension", 3072)  # Default to 3072 if not found
#     except Exception:
#         dim = 3072

#     zero_vec   = [0.0] * dim
#     total_deleted = 0

#     while True:
#         try:
#             res = index.query(
#                 vector           = zero_vec,
#                 top_k            = 1000,
#                 include_metadata = False,
#                 filter           = {"source": {"$eq": source_filename}},
#             )
#         except Exception:
#             # Metadata filtering may not be supported on all plan tiers
#             break

#         matches = res.get("matches") or []
#         if not matches:
#             break

#         ids = [m["id"] for m in matches]
#         index.delete(ids=ids)
#         total_deleted += len(ids)

#         if len(matches) < 1000:
#             break           # No more pages

#     return total_deleted


# @router.get("/upload_health")
# async def upload_service_health():
#     """
#     Diagnostic endpoint — call this first when debugging S3 issues.
#     Also auto-creates the S3 bucket if it doesn't exist yet.
#     """
#     from s3_service import debug_s3_config, create_bucket_if_not_exists

#     s3_cfg = debug_s3_config()
#     bucket_ready = False

#     health: Dict[str, Any] = {
#         "service":  "upload_enhanced",
#         "status":   "operational",
#         "storage":  "s3" if is_s3_configured() else "local",
#         "s3_config": s3_cfg,
#         "features": {
#             "aws_s3":            is_s3_configured(),
#             "docling_parser":    True,
#             "pdfplumber_tables": True,
#             "enhanced_chunking": True,
#             "pinecone_delete":   True,
#         },
#     }

#     try:
#         import docling, pdfplumber, fitz
#         health["dependencies"] = "all_available"
#     except ImportError as e:
#         health["status"]       = "degraded"
#         health["dependencies"] = f"missing: {e}"

#     # S3 connectivity + auto bucket creation
#     if is_s3_configured():
#         try:
#             bucket_ready = create_bucket_if_not_exists()
#             if bucket_ready:
#                 health["s3_connectivity"] = f"✅ bucket '{s3_cfg['AWS_S3_BUCKET']}' ready"
#             else:
#                 health["s3_connectivity"] = f"❌ bucket creation failed — check IAM permissions"
#         except Exception as e:
#             health["s3_connectivity"] = f"❌ {e}"

#     return health


# app.include_router(router)


# # ============================================================
# # HELPERS
# # ============================================================

# def _save_local(content: bytes, safe_filename: str) -> str:
#     """Write bytes to UPLOAD_ROOT and return the URL path."""
#     with open(os.path.join(UPLOAD_ROOT, safe_filename), "wb") as f:
#         f.write(content)
#     base_url = getattr(settings, "BASE_URL", "")
#     return f"{base_url}/uploads/{safe_filename}" if base_url else f"/uploads/{safe_filename}"


# def _safe_unlink(path: Optional[str]) -> None:
#     if path:
#         try:
#             if os.path.exists(path):
#                 os.unlink(path)
#         except Exception:
#             pass


# def sanitize_id(s: str, max_len: int = 200) -> str:
#     if not s: return "id"
#     nk        = unicodedata.normalize("NFKD", s)
#     ascii_str = nk.encode("ascii", "ignore").decode("ascii")
#     collapsed = re.sub(r"_+", "_", re.sub(r"[^0-9A-Za-z]+", "_", ascii_str)).strip("_") or "id"
#     return collapsed[:max_len].rstrip("_")


# # ============================================================
# # EMBEDDING + LLM
# # ============================================================

# async def embed_texts(texts: List[str]) -> List[List[float]]:
#     if not texts:
#         return []
#     headers = {
#         "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
#         "Content-Type":  "application/json",
#     }
#     batches: List[List[str]] = [
#         texts[i : i + EMBED_BATCH_SIZE] for i in range(0, len(texts), EMBED_BATCH_SIZE)
#     ]
#     results: List[List[float]] = []

#     async with httpx.AsyncClient(timeout=EMBED_TIMEOUT_SECS) as client:
#         for batch_idx, batch_texts in enumerate(batches):
#             payload = {"model": settings.EMBEDDING_MODEL, "input": batch_texts}
#             attempt = 0
#             while True:
#                 attempt += 1
#                 try:
#                     resp = await client.post(EMBED_URL, headers=headers, json=payload)
#                     resp.raise_for_status()
#                     emb_batch = [d["embedding"] for d in resp.json()["data"]]
#                     if len(emb_batch) != len(batch_texts):
#                         raise HTTPException(status_code=502, detail="Embedding length mismatch")
#                     results.extend(emb_batch)
#                     break
#                 except (httpx.ReadTimeout, httpx.WriteTimeout, httpx.ConnectError, httpx.RemoteProtocolError) as exc:
#                     if attempt >= EMBED_MAX_RETRIES:
#                         raise HTTPException(status_code=504, detail=f"Embedding timeout (batch {batch_idx}): {exc}")
#                     await asyncio.sleep(EMBED_BACKOFF_BASE ** (attempt - 1))
#                 except httpx.HTTPStatusError as exc:
#                     sc = exc.response.status_code
#                     et = ""
#                     try: et = exc.response.text
#                     except Exception: pass
#                     if 500 <= sc < 600 and attempt < EMBED_MAX_RETRIES:
#                         await asyncio.sleep(EMBED_BACKOFF_BASE ** (attempt - 1))
#                         continue
#                     raise HTTPException(status_code=502, detail=f"Embedding error {sc}: {et[:200]}")
#                 except Exception as exc:
#                     raise HTTPException(status_code=500, detail=f"Embedding failed: {exc}")

#     return results


# async def embed_single(text: str) -> List[float]:
#     return (await embed_texts([text]))[0]


# async def call_llm(messages: List[dict]) -> str:
#     headers = {
#         "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
#         "Content-Type":  "application/json",
#     }
#     async with httpx.AsyncClient(timeout=90) as client:
#         resp = await client.post(CHAT_URL, headers=headers, json={
#             "model": settings.LLM_MODEL, "messages": messages, "temperature": 0.2
#         })
#         resp.raise_for_status()
#         return resp.json()["choices"][0]["message"]["content"]


# async def call_llm_with_chain(*, user_question: str, context: str, final_system_prompt: str) -> str:
#     chain_prompt = (
#         "You are reasoning internally as an expert Indian CA tutor.\n\n"
#         "INTERNAL STEPS (do NOT reveal):\n"
#         "1. Understand the exact exam intent. Identify the overall concept.\n"
#         "2. Identify relevant context blocks. Merge info from multiple blocks.\n"
#         "3. Decide depth per ICAI expectations.\n\n"
#         "Then produce ONLY the final answer as instructed below.\n\n"
#         f"{final_system_prompt}"
#     )
#     return await call_llm([
#         {"role": "system", "content": chain_prompt},
#         {"role": "system", "content": f"CONTEXT:\n{context}"},
#         {"role": "user",   "content": user_question},
#     ])


# # ============================================================
# # TEXT / QUERY HELPERS
# # ============================================================

# def chunk_text(text: str, chunk_size: int = 800, overlap: int = 200) -> List[str]:
#     words  = text.split()
#     chunks = []
#     start  = 0
#     while start < len(words):
#         end = min(start + chunk_size, len(words))
#         chunks.append(" ".join(words[start:end]))
#         if end == len(words): break
#         start += chunk_size - overlap
#     return chunks


# def extract_pdf_text(file_bytes: bytes) -> str:
#     reader = PdfReader(io.BytesIO(file_bytes))
#     return "".join((page.extract_text() or "") + "\n" for page in reader.pages)


# def extract_pdf_pages(file_bytes: bytes) -> List[str]:
#     return [page.extract_text() or "" for page in PdfReader(io.BytesIO(file_bytes)).pages]


# def normalize_page_text(text: str) -> str:
#     return "\n".join(
#         re.sub(r"\s+", " ", ln).strip()
#         for ln in text.splitlines()
#         if ln.strip()
#     )


# def enrich_query_for_rag(question: str) -> str:
#     q, hints = question.lower(), []
#     if any(w in q for w in ["ind as", "financial", "asset", "liability", "consolidation"]):
#         hints.append("financial reporting accounting")
#     if any(w in q for w in ["audit", "sa ", "assurance"]):
#         hints.append("auditing")
#     if any(w in q for w in ["gst", "input tax", "itc"]):
#         hints.append("indirect tax gst")
#     if any(w in q for w in ["tds", "income tax", "section 80"]):
#         hints.append("direct tax")
#     if any(w in q for w in ["company act", "directors", "board"]):
#         hints.append("law")
#     return question + (" " + " ".join(hints) if hints else "")


# def detect_subject(question: str) -> Optional[str]:
#     q = question.lower()
#     if any(w in q for w in ["ind as", "as ", "financial statement", "consolidat", "revenue recognition"]):
#         return "Accounting"
#     if any(w in q for w in ["audit", "sa ", "assurance", "internal audit"]):
#         return "Auditing"
#     if any(w in q for w in ["gst", "indirect tax", "customs", "excise"]):
#         return "Indirect Tax"
#     if any(w in q for w in ["income tax", "direct tax", "tds", "section 80", "capital gain"]):
#         return "Direct Tax"
#     if any(w in q for w in ["company", "director", "board", "sebi", "securities"]):
#         return "Law"
#     if any(w in q for w in ["cost", "marginal", "budget", "variance"]):
#         return "Costing"
#     return None


# def chunk_text_words(text: str, chunk_size: int = 180, overlap: int = 40) -> List[str]:
#     words  = text.split()
#     chunks: List[str] = []
#     start, n = 0, len(words)
#     while start < n:
#         end   = min(start + chunk_size, n)
#         chunk = " ".join(words[start:end])
#         if chunk.strip(): chunks.append(chunk)
#         if end == n: break
#         start = max(0, end - overlap)
#     return chunks


# async def is_ca_related_question(question: str) -> bool:
#     system = (
#         "You are a domain classifier for an Indian Chartered Accountancy (CA) assistant.\n\n"
#         "Answer YES if the question relates to: ICAI syllabus, Accounting, Auditing, "
#         "Direct Tax/GST, Corporate Law, Financial management, Costing, or basic "
#         "commerce concepts commonly studied by CA students.\n\n"
#         "Answer NO only if it is clearly unrelated (science, coding, sports, entertainment).\n\n"
#         "Respond with YES or NO only."
#     )
#     try:
#         result = await call_llm([
#             {"role": "system", "content": system},
#             {"role": "user",   "content": question},
#         ])
#         return result.strip().upper().startswith("YES")
#     except Exception:
#         return True


# # ============================================================
# # AUTH ROUTES
# # ============================================================

# @app.post("/auth/signup")
# async def signup(user: UserCreate):
#     if await get_user_by_email(user.email):
#         raise HTTPException(status_code=400, detail="Email already registered")

#     plan = (user.plan or "free").lower()
#     if plan not in ("free", "paid"): plan = "free"
#     if plan == "paid" and not user.payment_id:
#         raise HTTPException(status_code=400, detail="payment_id is required for paid plan")

#     now = datetime.utcnow()
#     await users_collection.insert_one({
#         "email":               user.email,
#         "password_hash":       hash_password(user.password),
#         "name":                user.name,
#         "phone":               user.phone,
#         "ca_level":            user.ca_level,
#         "ca_attempt":          user.ca_attempt,
#         "role":                "student",
#         "status":              "approved",
#         "plan":                plan,
#         "subscription_status": "active" if plan == "paid" else "free",
#         "payment_id":          user.payment_id,
#         "plan_activated_at":   now if plan == "paid" else None,
#         "plan_expires_at": (
#             datetime(now.year, now.month + 1 if now.month < 12 else 1, now.day)
#             if plan == "paid" else None
#         ),
#         "created_at": now,
#     })
#     try:
#         send_admin_signup_notification({**user.dict(), "plan": plan})
#     except Exception as e:
#         print("Signup Email to admin failed:", e)
#     return {"message": "Signup successful."}


# @app.post("/auth/login", response_model=Token)
# async def login(data: UserLogin):
#     user = await get_user_by_email(data.email)
#     if not user or user.get("status") != "approved":
#         raise HTTPException(status_code=403, detail="Your account is pending admin approval.")
#     if not verify_password(data.password, user["password_hash"]):
#         raise HTTPException(status_code=400, detail="Invalid credentials")
#     return Token(access_token=create_access_token({"sub": user["email"]}))


# @app.get("/auth/me", response_model=UserOut)
# async def me(user=Depends(get_current_user)):
#     return UserOut(
#         email=user["email"], role=user["role"],
#         plan=user.get("plan", "free"),
#         subscription_status=user.get("subscription_status", "free"),
#     )


# @app.post("/auth/forgot-password")
# async def forgot_password(body: ForgotPasswordRequest):
#     user = await get_user_by_email(body.email)
#     if not user:
#         return {"message": "If this email is registered, an OTP has been sent."}
#     otp        = str(secrets.randbelow(900000) + 100000)
#     expires_at = datetime.utcnow() + timedelta(minutes=10)
#     await users_collection.update_one(
#         {"email": body.email},
#         {"$set": {"reset_otp": otp, "reset_otp_expires": expires_at}},
#     )
#     try:
#         send_password_reset_otp(email=body.email, otp=otp, name=user.get("name", "Student"))
#     except Exception as e:
#         print("OTP email failed:", e)
#         raise HTTPException(status_code=500, detail="Failed to send OTP email.")
#     return {"message": "If this email is registered, an OTP has been sent."}


# @app.post("/auth/verify-otp")
# async def verify_otp(body: VerifyOTPRequest):
#     user = await get_user_by_email(body.email)
#     if not user:
#         raise HTTPException(status_code=400, detail="Invalid OTP or email.")
#     stored_otp, stored_expires = user.get("reset_otp"), user.get("reset_otp_expires")
#     if not stored_otp or not stored_expires:
#         raise HTTPException(status_code=400, detail="No OTP requested. Please request a new one.")
#     if datetime.utcnow() > stored_expires:
#         raise HTTPException(status_code=400, detail="OTP has expired.")
#     if stored_otp != body.otp.strip():
#         raise HTTPException(status_code=400, detail="Incorrect OTP.")
#     return {"message": "OTP verified."}


# @app.post("/auth/reset-password")
# async def reset_password(body: ResetPasswordRequest):
#     user = await get_user_by_email(body.email)
#     if not user:
#         raise HTTPException(status_code=400, detail="Invalid request.")
#     stored_otp, stored_expires = user.get("reset_otp"), user.get("reset_otp_expires")
#     if not stored_otp or not stored_expires:
#         raise HTTPException(status_code=400, detail="No OTP requested.")
#     if datetime.utcnow() > stored_expires:
#         raise HTTPException(status_code=400, detail="OTP has expired.")
#     if stored_otp != body.otp.strip():
#         raise HTTPException(status_code=400, detail="Incorrect OTP.")
#     if len(body.new_password) < 6:
#         raise HTTPException(status_code=400, detail="Password must be at least 6 characters.")
#     await users_collection.update_one(
#         {"email": body.email},
#         {"$set":   {"password_hash": hash_password(body.new_password)},
#          "$unset": {"reset_otp": "", "reset_otp_expires": ""}},
#     )
#     return {"message": "Password reset successfully. You can now log in."}


# # ============================================================
# # ADMIN — STUDENT MANAGEMENT
# # ============================================================

# @app.get("/admin/students")
# async def get_all_students(admin=Depends(get_current_admin)):
#     students = []
#     async for user in users_collection.find({"role": "student"}):
#         user["_id"] = str(user["_id"])
#         students.append(user)
#     return students


# @app.post("/admin/approve/{user_id}")
# async def approve_student(user_id: str, admin=Depends(get_current_admin)):
#     result = await users_collection.update_one(
#         {"_id": ObjectId(user_id)}, {"$set": {"status": "approved"}}
#     )
#     if result.matched_count == 0:
#         raise HTTPException(status_code=404, detail="Student not found")
#     return {"message": "Student approved"}


# @app.post("/admin/reject/{user_id}")
# async def reject_student(user_id: str, admin=Depends(get_current_admin)):
#     await users_collection.delete_one({"_id": ObjectId(user_id)})
#     return {"message": "Student rejected"}


# # ============================================================
# # ADMIN — DOCUMENTS LIST  (AdminUpload panel)
# # ============================================================

# @app.get("/admin/documents/grouped", tags=["Admin Materials"])
# async def get_grouped_documents(admin=Depends(get_current_admin)):
#     """All uploaded documents grouped by course/level, newest first."""
#     grouped: Dict[str, List[Any]] = {}
#     async for doc in docs_collection.find().sort("uploaded_at", -1):
#         doc["_id"] = str(doc["_id"])
#         if isinstance(doc.get("uploaded_at"), datetime):
#             doc["uploaded_at"] = doc["uploaded_at"].isoformat()
#         course = doc.get("course") or doc.get("level") or "Other"
#         grouped.setdefault(course, [])
#         grouped[course].append(doc)
#     return grouped


# # ============================================================
# # CA DASHBOARD ROUTES
# # ============================================================

# @app.get("/dashboard/tree")
# async def get_dashboard_tree(user=Depends(get_current_user)):
#     """4-level tree: level → subject → module → chapter → [items]"""
#     tree: Dict[str, Any] = {}
#     async for doc in dashboard_collection.find().sort("created_at", 1):
#         doc["_id"] = str(doc["_id"])
#         level   = (doc.get("level")   or "Others").strip()
#         subject = (doc.get("subject") or "General").strip()
#         module  = (doc.get("module")  or subject).strip()
#         chapter = (doc.get("chapter") or doc.get("title", "General")).strip()

#         tree.setdefault(level, {})
#         tree[level].setdefault(subject, {})
#         tree[level][subject].setdefault(module, {})
#         tree[level][subject][module].setdefault(chapter, [])

#         tree[level][subject][module][chapter].append({
#             "_id":       doc["_id"],
#             "title":     doc.get("title",     ""),
#             "pdf_url":   doc.get("pdf_url",   ""),
#             "video_url": doc.get("video_url", ""),
#             "chapter":   chapter,
#             "unit":      doc.get("unit",      ""),
#         })
#     return tree


# @app.post("/dashboard/add")
# async def add_dashboard_resource(
#     level: str = Form(...), subject: str = Form(...),
#     module: str = Form(...), chapter: str = Form(...),
#     unit: str = Form(...), title: str = Form(...),
#     pdf_url: str = Form(...), video_url: str = Form(""),
#     admin=Depends(get_current_admin),
# ):
#     await dashboard_collection.insert_one({
#         "level": level, "subject": subject, "module": module,
#         "chapter": chapter, "unit": unit, "title": title,
#         "pdf_url": pdf_url, "video_url": video_url,
#         "created_at": datetime.utcnow(),
#     })
#     return {"message": "Added successfully"}


# # ============================================================
# # PERSONALISATION
# # ============================================================

# def build_personalized_layer(user: dict) -> str:
#     name  = user.get("name", "Student").split()[0]
#     level = user.get("ca_level", "Foundation")
#     guidance = {
#         "Foundation":   "Explain in very simple language with basic concepts and easy examples.",
#         "Intermediate": "Explain with clarity and practical understanding. Include examples.",
#         "Final":        "Provide detailed, professional-level explanation with ICAI exam perspective.",
#     }
#     return (
#         f"PERSONALIZATION CONTEXT:\n- Student Name: {name}\n- Level: {level}\n\n"
#         f"INSTRUCTIONS:\n"
#         f"- Occasionally address the student as {name}\n"
#         f"- Teaching style: {guidance.get(level, guidance['Foundation'])}\n"
#         f"- Keep tone supportive and engaging\n"
#         f"- provide extra clarity, motivation, and simplify difficult parts."
#     )


# # ============================================================
# # CHAT (RAG)
# # ============================================================

# @app.post("/chat", response_model=ChatResponse)
# async def chat(req: ChatRequest, user=Depends(get_current_user)):
#     try:
#         # 🔥 Expand CA abbreviations BEFORE everything
#         req.message = expand_ca_abbreviations(req.message)

#         # --------------------------------------------------
#         # 1. CA gatekeeper
#         # --------------------------------------------------
#         if not await is_ca_related_question(req.message):
#             return ChatResponse(
#                 answer=(
#                     "This assistant is designed for Indian CA students. "
#                     "Please ask a question related to Indian CA topics such as accounting, tax, audit, law, "
#                     "or CA exams (Foundation / Inter / Final)."
#                 ),
#                 sources=[],
#             )

#         # --------------------------------------------------
#         # 2. RAG FLOW (Pinecone)
#         # --------------------------------------------------
#         query_embedding = await embed_single(enrich_query_for_rag(req.message))
#         detected_subj   = detect_subject(req.message)

#         q_kwargs: Dict[str, Any] = {"vector": query_embedding, "top_k": 20, "include_metadata": True}
#         if detected_subj:
#             q_kwargs["filter"] = {"subject": {"$eq": detected_subj}}

#         res     = index.query(**q_kwargs)
#         matches = res.get("matches") or []

#         # Fallback: retry without subject filter if too few results
#         if len(matches) < 4 and detected_subj:
#             res     = index.query(vector=query_embedding, top_k=20, include_metadata=True)
#             matches = res.get("matches") or []

#         # Sort by score, apply dynamic threshold
#         matches = sorted(matches, key=lambda m: m.get("score", 0), reverse=True)
#         q_len   = len(req.message.split())
#         thr     = 0.52 if q_len <= 6 else 0.60 if q_len <= 15 else 0.65
#         matches = [m for m in matches if m.get("score", 0) >= thr] or matches[:5]

#         personal_context = build_personalized_layer(user)

#         # --------------------------------------------------
#         # 3. NO MATCHES → LLM-only fallback
#         # --------------------------------------------------
#         if not matches:
#             system_prompt = (
#                 personal_context + "\n\n"
#                 "You are a senior Indian Chartered Accountant (CA) faculty with experience "
#                 "in teaching and evaluating ICAI exams (Foundation, Inter, Final).\n\n"

#                 "Language rule (MANDATORY):\n"
#                 "- Reply strictly in the SAME language as the user's question "
#                 "(English, Hindi, or Hinglish).\n\n"

#                 "Knowledge & safety rules:\n"
#                 "- Answer using your standard CA knowledge and well-established ICAI principles.\n"
#                 "- Do NOT guess exact section numbers, limits, percentages, or year-specific amendments.\n"
#                 "- If precise data is uncertain, explain the concept without giving risky figures.\n\n"

#                 "Answer structure (EXAM-ORIENTED):\n"
#                 "1. Begin with a clear definition or core concept.\n"
#                 "2. Explain in logical steps using proper CA terminology.\n"
#                 "3. Where relevant, mention accounting treatment / legal position / tax implication.\n"
#                 "4. Include ONE short exam-oriented or practical illustration if helpful.\n\n"

#                 "Exam guidance:\n"
#                 "- Add ONE short CA exam tip or common mistake to avoid.\n"
#                 "- Keep the answer concise, structured, and revision-friendly.\n\n"

#                 "Tone & presentation:\n"
#                 "- Maintain a professional, faculty-level tone.\n"
#                 "- Avoid casual language, storytelling, or over-explanation."
#             )

#             answer = await call_llm([
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user",   "content": req.message},
#             ])
#             return ChatResponse(
#                 answer=answer,
#                 sources=[{"doc_title": "General CA Knowledge (LLM based)", "note": "No match in uploaded docs"}],
#             )

#         # --------------------------------------------------
#         # 4. BUILD CONTEXT (SAFE SIZE)
#         # --------------------------------------------------
#         context_blocks: List[str] = []
#         sources: List[dict]       = []

#         for m in sorted(matches, key=lambda m: m.get("score", 0), reverse=True):
#             meta = m.get("metadata", {})
#             text = meta.get("text", "")
#             if not text:
#                 continue

#             header = []
#             if meta.get("doc_title"):  header.append(f"Document: {meta['doc_title']}")
#             if meta.get("chapter"):    header.append(f"Chapter: {meta['chapter']}")
#             if meta.get("topic"):      header.append(f"Topic: {meta['topic']}")
#             if meta.get("page_start"): header.append(f"Page: {meta['page_start']}")
#             context_blocks.append(f"{' | '.join(header)}\n{text}")

#             sources.append({
#                 "doc_title":  meta.get("doc_title"),
#                 "source":     meta.get("source"),
#                 "page_start": meta.get("page_start"),
#                 "chapter":    meta.get("chapter"),
#                 "topic":      meta.get("topic"),
#                 "type":       meta.get("type", "text"),
#             })

#         # Hard context character limit
#         trimmed, total = [], 0
#         for b in context_blocks:
#             if total + len(b) > 6000:
#                 break
#             trimmed.append(b)
#             total += len(b)
#         context_str = "\n\n---\n\n".join(trimmed)

#         # --------------------------------------------------
#         # 5. FINAL ANSWER (QA vs DISCUSSION)
#         # --------------------------------------------------
#         if req.mode == "discussion":
#             sys_p = (
#                 personal_context + "\n\n"
#                 "You are an expert Indian CA tutor simulating a healthy academic discussion "
#                 "between two CA students preparing for exams.\n\n"

#                 "Language rules:\n"
#                 "- Reply strictly in the SAME language as the user's question (English, Hindi, or Hinglish).\n\n"

#                 "Discussion format rules:\n"
#                 "- Write the answer as a discussion between 'User A:' and 'User B:'.\n"
#                 "- Alternate clearly between User A and User B.\n"
#                 "- Provide at least 4 to 6 exchanges.\n\n"

#                 "Content rules (VERY IMPORTANT):\n"
#                 "- Explain concepts step-by-step in a teaching style.\n"
#                 "- Keep explanations exam-oriented as per ICAI expectations.\n"
#                 "- Use simple intuition first, then technical clarity.\n"
#                 "- Include 1 very short practical or exam-oriented example if relevant.\n"
#                 "- Add a quick very short CA exam tip, memory aid, or common mistake to avoid.\n"
#                 "- Avoid unnecessary storytelling or casual chat.\n\n"

#                 "Source rules:\n"
#                 "- Answer using the context provided below.\n"
#                 "- Only use well-trusted facts based on the given context.\n\n"

#                 f"Context:\n{context_str}"
#             )
#         else:
#             sys_p = (
#                 personal_context + "\n\n"
#                 "You are an expert Indian Chartered Accountant (CA) tutor preparing students "
#                 "for ICAI exams (Foundation, Inter, Final).\n\n"

#                 "Language rule:\n"
#                 "- Reply strictly in the SAME language as the user's question "
#                 "(English, Hindi, or Hinglish).\n\n"

#                 "Answering style rules:\n"
#                 "- Answer using the context provided below.\n"
#                 "- Keep the explanation clear, concise, and exam-oriented, in detail.\n"
#                 "- Start with a direct definition or core concept in elaborative style.\n"
#                 "- Then briefly explain or elaborate as required for marks.\n"
#                 "- If applicable, include a short practical or exam-oriented example.\n"
#                 "- If tables or figures are present in the context, refer to them explicitly.\n\n"

#                 "Exam guidance:\n"
#                 "- Add one very short important CA exam tip or a common mistake to avoid.\n"
#                 "- Avoid unnecessary storytelling or over-explanation.\n\n"

#                 f"Context:\n{context_str}"
#             )

#         answer = await call_llm_with_chain(
#             user_question=req.message,
#             context=context_str,
#             final_system_prompt=sys_p,
#         )

#         # --- CLEAN SOURCES: remove duplicates ---
#         seen: set             = set()
#         clean_sources: List[dict] = []
#         for s in sources:
#             key = (s.get("doc_title"), s.get("page_start"))
#             if key not in seen:
#                 seen.add(key)
#                 clean_sources.append(s)

#         return ChatResponse(answer=answer, sources=clean_sources[:5])

#     # --------------------------------------------------
#     # 6. SAFE FALLBACK — never return a blank answer
#     # --------------------------------------------------
#     except Exception as e:
#         traceback.print_exc()
#         try:
#             answer = await call_llm([
#                 {"role": "system", "content": "You are a helpful Indian CA tutor."},
#                 {"role": "user",   "content": req.message},
#             ])
#             return ChatResponse(
#                 answer=answer,
#                 sources=[{
#                     "doc_title": "LLM fallback",
#                     "note": "Answered without document sources due to a system issue",
#                 }],
#             )
#         except Exception as inner_e:
#             traceback.print_exc()
#             raise HTTPException(status_code=500, detail=f"Chat error: {str(e)} | Fallback error: {str(inner_e)}")



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
