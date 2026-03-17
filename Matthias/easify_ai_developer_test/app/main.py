"""FastAPI application: REST API + minimal web interface for the RAG system."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import get_settings
from app.schemas import QuestionRequest, AnswerResponse, IngestResponse
from app.rag.ingestion import load_documents, chunk_documents
from app.rag.retriever import ingest_chunks, clear_collection, get_vectorstore, build_bm25_index
from app.rag.chain import answer_question

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """On startup, check if the vector store has documents; if not, auto-ingest."""
    settings = get_settings()
    try:
        store = get_vectorstore()
        count = store._collection.count()
        if count == 0:
            logger.info("Vector store is empty — running auto-ingestion...")
            _run_ingestion(settings.document_directories)
        else:
            logger.info(f"Vector store has {count} chunks. Rebuilding BM25 index...")
            documents = load_documents(settings.document_directories)
            chunks = chunk_documents(
                documents,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            )
            build_bm25_index(chunks)
            logger.info("BM25 index rebuilt.")
    except Exception as e:
        logger.warning(f"Could not check vector store on startup: {e}")
    yield


app = FastAPI(
    title="Easify RAG Assistant",
    description="Retrieval-Augmented Generation system for Easify platform documentation",
    version="1.0.0",
    lifespan=lifespan,
)

# Mount static files if directory exists
static_dir = BASE_DIR / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


def _run_ingestion(doc_dirs: list[str]) -> IngestResponse:
    """Run the full ingestion pipeline."""
    settings = get_settings()
    clear_collection()

    documents = load_documents(doc_dirs)
    chunks = chunk_documents(
        documents,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    stored = ingest_chunks(chunks)

    return IngestResponse(
        status="success",
        documents_processed=len(documents),
        chunks_created=stored,
        collection=settings.chroma_collection,
    )


# ─── API Endpoints ────────────────────────────────────────────────────────────


@app.post("/api/ask", response_model=AnswerResponse)
async def ask_question(req: QuestionRequest):
    """Ask a question and get a grounded answer from the documentation."""
    return answer_question(
        question=req.question,
        top_k=req.top_k,
        dataset=req.dataset,
    )


@app.post("/api/ingest", response_model=IngestResponse)
async def ingest_documents():
    """Re-ingest all documents (clears existing collection first)."""
    settings = get_settings()
    return _run_ingestion(settings.document_directories)


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    try:
        store = get_vectorstore()
        count = store._collection.count()
        return {"status": "healthy", "chunks_in_store": count}
    except Exception as e:
        return {"status": "degraded", "error": str(e)}


# ─── Web Interface ─────────────────────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
async def web_ui(request: Request):
    """Serve the minimal web interface."""
    return templates.TemplateResponse("index.html", {"request": request})