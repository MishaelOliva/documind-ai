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
    provider_used: str = Field(
        ...,
        description="Provider that actually produced the answer, after any fallback."
    )
    model_name: str
    embedding_latency_ms: float = Field(
        ..., description="Query embedding (ONNX transformer) inference time."
    )
    search_latency_ms: float = Field(..., description="Vector similarity search time only.")
    retrieval_latency_ms: float = Field(..., description="embedding_latency_ms + search_latency_ms.")
    generation_latency_ms: float
    total_latency_ms: float
    citations: List[Citation]
    retrieved_chunk_count: int
    top_similarity: float = Field(..., description="Cosine similarity of the best chunk.")
    is_grounded: bool = Field(
        ...,
        description="False when top_similarity fell below MIN_SIMILARITY_SCORE and the "
        "system refused to answer.",
    )

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
    embedding_backend: str = Field(..., description="Which embedder is actually loaded.")
    min_similarity_score: float
