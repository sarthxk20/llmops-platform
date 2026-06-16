"""
FastAPI serving layer for the fine-tuned LLM.

Endpoints:
  POST /generate   — text generation (with optional RAG)
  POST /embed      — produce embeddings for a list of texts
  GET  /health     — liveness probe
  GET  /ready      — readiness probe (model loaded?)
  GET  /metrics    — Prometheus metrics (auto-mounted by instrumentator)

Run locally:
    uvicorn serving.app:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field

from serving.model_loader import ModelLoader
from serving.rag_pipeline import RAGPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger(__name__)

# ── Globals ────────────────────────────────────────────────────────────────────
MODEL_LOADER: Optional[ModelLoader] = None
RAG:          Optional[RAGPipeline] = None
_READY = False


# ── Lifespan: load model once at startup ──────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL_LOADER, RAG, _READY
    log.info("Loading model and RAG pipeline...")
    try:
        adapter_path = os.getenv("ADAPTER_PATH", "")
        # Only use adapter path if it exists locally (skip during CI/testing)
        resolved_adapter = adapter_path if adapter_path and os.path.exists(adapter_path) else None

        MODEL_LOADER = ModelLoader(
            base_model=os.getenv("BASE_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
            adapter_path=resolved_adapter,
        )
        MODEL_LOADER.load()

        index_path = os.getenv("FAISS_INDEX_PATH", "data/faiss_index")
        if os.path.exists(index_path):
            RAG = RAGPipeline(index_path=index_path)
            RAG.load()
            log.info("RAG pipeline loaded.")
        else:
            log.warning("No FAISS index found — RAG disabled. Run serving/build_index.py first.")

        _READY = True
        log.info("Model ready.")
    except Exception as e:
        log.error(f"Startup failed: {e}")
        _READY = False
    yield
    log.info("Shutting down — releasing model.")
    if MODEL_LOADER:
        MODEL_LOADER.unload()


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="LLMOps Serving Platform",
    description="Fine-tuned LLM with RAG, served via FastAPI.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Prometheus metrics at /metrics
Instrumentator().instrument(app).expose(app)


# ── Schemas ────────────────────────────────────────────────────────────────────
class GenerateRequest(BaseModel):
    instruction: str = Field(..., min_length=1, max_length=2000,
                             json_schema_extra={"example": "How do I cancel my subscription?"})
    input_text:  str = Field("", max_length=2000)
    use_rag:     bool = Field(True,  description="Prepend retrieved context")
    max_new_tokens: int = Field(256, ge=1, le=1024)
    temperature:    float = Field(0.7, ge=0.0, le=2.0)
    top_p:          float = Field(0.9, ge=0.0, le=1.0)
    top_k:          int   = Field(50,  ge=0)
    do_sample:      bool  = Field(True)


class GenerateResponse(BaseModel):
    response:        str
    retrieved_docs:  List[str]
    latency_ms:      float
    tokens_generated: int
    model_name:      str


class EmbedRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=64)


class EmbedResponse(BaseModel):
    embeddings:  List[List[float]]
    model_name:  str
    latency_ms:  float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    rag_enabled:  bool
    device:       str


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health():
    return HealthResponse(
        status="ok",
        model_loaded=MODEL_LOADER is not None and MODEL_LOADER.is_loaded,
        rag_enabled=RAG is not None,
        device=str(MODEL_LOADER.device if MODEL_LOADER else "unknown"),
    )


@app.get("/ready", tags=["ops"])
async def ready():
    if not _READY:
        raise HTTPException(status_code=503, detail="Model not ready")
    return {"ready": True}


@app.post("/generate", response_model=GenerateResponse, tags=["inference"])
async def generate(req: GenerateRequest, request: Request):
    if not _READY or MODEL_LOADER is None:
        raise HTTPException(status_code=503, detail="Model not ready")

    t0 = time.perf_counter()

    # Build prompt
    retrieved_docs = []
    context        = ""
    if req.use_rag and RAG is not None:
        retrieved_docs = RAG.retrieve(req.instruction, k=3)
        context = "\n\n".join(retrieved_docs)

    prompt = _build_prompt(req.instruction, req.input_text, context)

    # Generate
    try:
        output, n_tokens = MODEL_LOADER.generate(
            prompt=prompt,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            do_sample=req.do_sample,
        )
    except Exception as e:
        log.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

    latency_ms = (time.perf_counter() - t0) * 1000
    log.info(f"Generated {n_tokens} tokens in {latency_ms:.1f}ms | RAG={req.use_rag}")

    return GenerateResponse(
        response=output,
        retrieved_docs=retrieved_docs,
        latency_ms=round(latency_ms, 2),
        tokens_generated=n_tokens,
        model_name=MODEL_LOADER.model_name,
    )


@app.post("/embed", response_model=EmbedResponse, tags=["inference"])
async def embed(req: EmbedRequest):
    if not _READY or MODEL_LOADER is None:
        raise HTTPException(status_code=503, detail="Model not ready")

    t0 = time.perf_counter()
    try:
        embeddings = MODEL_LOADER.embed(req.texts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return EmbedResponse(
        embeddings=embeddings,
        model_name=MODEL_LOADER.model_name,
        latency_ms=round((time.perf_counter() - t0) * 1000, 2),
    )


# ── Helpers ────────────────────────────────────────────────────────────────────

def _build_prompt(instruction: str, input_text: str, context: str) -> str:
    parts = ["### Instruction:", instruction]
    if context:
        parts += ["\n### Context (retrieved):", context]
    if input_text:
        parts += ["\n### Input:", input_text]
    parts.append("\n### Response:")
    return "\n".join(parts)


# ── Error handlers ─────────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def generic_handler(request: Request, exc: Exception):
    log.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
