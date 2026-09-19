"""Data models and schemas for the RAG application."""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class DocumentChunk(BaseModel):
    chunk_id: str
    doc_id: str
    source_name: str
    page_number: int
    chunk_index: int
    content: str
    char_count: int
    metadata: Dict[str, Any] = Field(default_factory=dict)

class Citation(BaseModel):
    chunk_id: str
    source_name: str
    page_number: int
    chunk_index: int
    similarity_score: float
    snippet: str

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=2, description="User question to answer from indexed docs")
    top_k: Optional[int] = Field(default=3, ge=1, le=10, description="Number of context chunks to retrieve")
    provider: Optional[str] = Field(default=None, description="Override LLM provider ('gemini', 'ollama', 'mock')")

class QueryResponse(BaseModel):
    question: str
    answer: str
    provider_used: str
    model_name: str
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float
    citations: List[Citation]
    retrieved_chunk_count: int

class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    file_type: str
    char_count: int
    chunk_count: int
    upload_timestamp: str

class IngestResponse(BaseModel):
    status: str
    message: str
    document: DocumentInfo

class HealthResponse(BaseModel):
    status: str
    version: str
    total_documents: int
    total_chunks: int
    active_provider: str
    vector_store_type: str
