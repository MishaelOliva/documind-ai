"""DocuMind FastAPI Application Entry Point."""
import logging
import re
import uuid
from pathlib import Path
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from app.config import settings
from app.models import (
    QueryRequest,
    QueryResponse,
    IngestResponse,
    DocumentInfo,
    HealthResponse
)
from app.rag.pipeline import RAGPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Numeric literals rather than the `status.*` constants: the named HTTP_413 and
# HTTP_422 aliases were renamed in recent Starlette releases, and pinning the
# numbers keeps this working across the whole supported FastAPI range without
# emitting a deprecation warning.
HTTP_CONTENT_TOO_LARGE = 413
HTTP_UNPROCESSABLE_CONTENT = 422

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=(
        "Document QA and semantic search with dense vector retrieval and "
        "source-grounded synthesis. No LLM framework: chunking, embedding, cosine "
        "search, and citation formatting are implemented directly."
    )
)

# CORS is opt-in. The SPA is served from the same origin, so no cross-origin
# access is needed; a wildcard origin combined with credentials is rejected by
# browsers anyway and would be a needless exposure on a public deployment.
if settings.CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type", "Authorization"],
    )
else:
    logger.info("CORS_ORIGINS not set; serving same-origin only.")

# Initialize RAG Pipeline singleton
pipeline = RAGPipeline()

# Mount Static Files
if settings.STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(settings.STATIC_DIR)), name="static")

# Characters permitted in a stored filename. Uploaded names are attacker-controlled
# and are echoed back to every client, so they are normalized here at the boundary
# rather than relying on each consumer to escape them.
_SAFE_FILENAME = re.compile(r"[^A-Za-z0-9._ -]")
_MAX_FILENAME_LEN = 120


def _sanitize_filename(raw: str | None) -> str:
    """
    Reduces an uploaded filename to a safe basename.

    Strips any directory component, removes characters that could break out of
    an HTML attribute or a filesystem path, and caps the length. The extension is
    validated separately against `settings.ALLOWED_EXTENSIONS`.
    """
    name = Path(raw or "upload.txt").name
    name = _SAFE_FILENAME.sub("_", name).strip(" .") or "upload.txt"
    if len(name) > _MAX_FILENAME_LEN:
        suffix = Path(name).suffix
        name = name[: _MAX_FILENAME_LEN - len(suffix)] + suffix
    return name


@app.get("/", include_in_schema=False)
async def serve_ui():
    """Serves the single-page application frontend."""
    index_file = settings.STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": f"Welcome to {settings.PROJECT_NAME}. Open /docs for Swagger documentation."}


# The handlers below are deliberately plain `def` rather than `async def`.
# They run synchronous, CPU-bound work (ONNX inference, disk I/O, JSON encoding),
# and FastAPI executes sync endpoints in a thread pool. Declaring them `async def`
# would put that work on the event loop and stall every other in-flight request.
@app.get(f"{settings.API_PREFIX}/health", response_model=HealthResponse)
def health_check():
    """Returns application health, indexed stats, and current provider configuration."""
    stats = pipeline.get_stats()
    return HealthResponse(
        status="healthy",
        version=settings.VERSION,
        total_documents=stats.get("total_documents", 0),
        total_chunks=stats.get("total_chunks", 0),
        active_provider=settings.LLM_PROVIDER,
        vector_store_type="In-Memory / Persistent JSON Index",
        embedding_backend=stats.get("embedding_backend", "unknown"),
        min_similarity_score=settings.MIN_SIMILARITY_SCORE,
    )


@app.post(
    f"{settings.API_PREFIX}/upload",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_document(file: UploadFile = File(...)):
    """
    Uploads and indexes a document (.pdf, .txt, .md, .csv).

    Extracts text, generates chunks, computes embeddings, and stores the result
    in the vector index.
    """
    safe_filename = _sanitize_filename(file.filename)
    file_ext = Path(safe_filename).suffix.lower()
    if file_ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported format '{file_ext or 'none'}'. "
                f"Allowed: {', '.join(settings.ALLOWED_EXTENSIONS)}"
            )
        )

    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    temp_path = settings.UPLOAD_DIR / f"temp_{uuid.uuid4().hex}{file_ext}"
    size = 0
    try:
        with open(temp_path, "wb") as buffer:
            while chunk := await file.read(1024 * 1024):  # 1MB chunks
                size += len(chunk)
                if size > settings.MAX_UPLOAD_SIZE_BYTES:
                    raise HTTPException(
                        status_code=HTTP_CONTENT_TOO_LARGE,
                        detail=(
                            "File exceeds maximum allowed upload size "
                            f"({settings.MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)} MB)."
                        )
                    )
                buffer.write(chunk)

        # Chunking, embedding, and index persistence are synchronous and CPU-bound.
        return await run_in_threadpool(pipeline.ingest_document, temp_path, safe_filename)
    except HTTPException:
        raise
    except ValueError as e:
        # Empty or unreadable content is a client problem, not a server fault.
        raise HTTPException(status_code=HTTP_UNPROCESSABLE_CONTENT, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=HTTP_UNPROCESSABLE_CONTENT, detail=str(e)) from e
    except Exception as e:
        logger.exception("Ingestion failed for %s", safe_filename)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {e}",
        ) from e
    finally:
        # Always remove the temp file, including on the oversize path.
        temp_path.unlink(missing_ok=True)


@app.post(f"{settings.API_PREFIX}/query", response_model=QueryResponse)
def query_documents(req: QueryRequest):
    """
    Performs semantic retrieval against indexed chunks, formats a prompt with
    strict source grounding, and returns the answer with exact citations.

    If the best chunk scores below the configured similarity floor the response is
    an explicit refusal with `is_grounded=false` and no citations.
    """
    if not pipeline.list_documents():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No documents have been indexed yet. Please upload a document first."
        )

    try:
        return pipeline.query(
            question=req.question,
            top_k=req.top_k or settings.TOP_K_RESULTS,
            override_provider=req.provider
        )
    except Exception as e:
        logger.exception("Query failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {e}",
        ) from e


@app.get(f"{settings.API_PREFIX}/documents", response_model=List[DocumentInfo])
def list_documents():
    """Lists all indexed documents and their chunk counts."""
    return pipeline.list_documents()


@app.delete(f"{settings.API_PREFIX}/documents/{{doc_id}}")
def delete_document(doc_id: str):
    """Deletes a document and its embeddings from the vector index."""
    success = pipeline.delete_document(doc_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{doc_id}' not found.",
        )
    return {"status": "success", "message": f"Document '{doc_id}' deleted successfully."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.APP_HOST, port=settings.APP_PORT, reload=True)
