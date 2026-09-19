"""FastAPI Application Entry Point for Applied AI Document QA & RAG System."""
import os
import shutil
from pathlib import Path
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.models import (
    QueryRequest,
    QueryResponse,
    IngestResponse,
    DocumentInfo,
    HealthResponse
)
from app.rag.pipeline import RAGPipeline

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Full-stack, production-ready Retrieval-Augmented Generation (RAG) system with hybrid LLM provider support."
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG Pipeline singleton
pipeline = RAGPipeline()

# Mount Static Files
if settings.STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(settings.STATIC_DIR)), name="static")

@app.get("/", include_in_schema=False)
async def serve_ui():
    """Serves the single-page application frontend."""
    index_file = settings.STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "Welcome to AI Document QA & RAG API. Open /docs for Swagger documentation."}

@app.get(f"{settings.API_PREFIX}/health", response_model=HealthResponse)
async def health_check():
    """Returns application health, indexed stats, and current LLM provider configuration."""
    stats = pipeline.get_stats()
    return HealthResponse(
        status="healthy",
        version=settings.VERSION,
        total_documents=stats.get("total_documents", 0),
        total_chunks=stats.get("total_chunks", 0),
        active_provider=settings.LLM_PROVIDER,
        vector_store_type="In-Memory / Persistent JSON Index"
    )

@app.post(f"{settings.API_PREFIX}/upload", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(file: UploadFile = File(...)):
    """
    Uploads and indexes a document (.pdf, .txt, .md, .csv).
    Extracts text, generates chunks, computes embeddings, and stores in vector database.
    """
    valid_exts = [".pdf", ".txt", ".md", ".csv"]
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in valid_exts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported format '{file_ext}'. Allowed: {', '.join(valid_exts)}"
        )

    # Save temporary upload
    temp_path = settings.UPLOAD_DIR / f"temp_{file.filename}"
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Ingest through RAG pipeline
        response = pipeline.ingest_document(temp_path, file.filename)
        return response
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    finally:
        if temp_path.exists():
            temp_path.unlink()

@app.post(f"{settings.API_PREFIX}/query", response_model=QueryResponse)
async def query_documents(req: QueryRequest):
    """
    Performs semantic retrieval against indexed chunks, formats prompt with strict source grounding,
    and returns answer along with exact document citations.
    """
    if not pipeline.list_documents():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No documents have been indexed yet. Please upload a document first."
        )

    try:
        response = pipeline.query(
            question=req.question,
            top_k=req.top_k or settings.TOP_K_RESULTS,
            override_provider=req.provider
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@app.get(f"{settings.API_PREFIX}/documents", response_model=List[DocumentInfo])
async def list_documents():
    """Lists all indexed documents and their chunk counts."""
    return pipeline.list_documents()

@app.delete(f"{settings.API_PREFIX}/documents/{{doc_id}}")
async def delete_document(doc_id: str):
    """Deletes a document and its embeddings from the vector index."""
    success = pipeline.delete_document(doc_id)
    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Document {doc_id} not found.")
    return {"status": "success", "message": f"Document {doc_id} deleted successfully."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.APP_HOST, port=settings.APP_PORT, reload=True)
