"""Unit and Integration Tests for RAG Pipeline."""
import pytest
from pathlib import Path
from app.rag.chunker import RecursiveChunker
from app.rag.embeddings import EmbeddingEngine
from app.rag.vector_store import VectorStore
from app.rag.pipeline import RAGPipeline
from app.models import DocumentInfo, DocumentChunk

def test_chunker_basic():
    chunker = RecursiveChunker(chunk_size=100, chunk_overlap=20)
    sample_text = (
        "This is paragraph one about modern software engineering and systems.\n\n"
        "This is paragraph two discussing Retrieval Augmented Generation pipelines, "
        "dense vector embeddings, and cosine similarity search."
    )
    chunks = chunker.split_text(sample_text)
    assert len(chunks) >= 2
    for c in chunks:
        assert len(c) <= 120  # bounded near chunk_size

def test_embedding_engine():
    engine = EmbeddingEngine(provider="mock")
    text1 = "FastAPI backend with vector database"
    text2 = "FastAPI application with vector embeddings"
    text3 = "Baking chocolate chip cookies in an oven"

    vec1 = engine.embed_text(text1)
    vec2 = engine.embed_text(text2)
    vec3 = engine.embed_text(text3)

    assert len(vec1) == 384
    assert len(vec2) == 384
    assert len(vec3) == 384

    # Cosine similarity between 1 & 2 should be noticeably higher than 1 & 3
    import numpy as np
    sim_1_2 = float(np.dot(vec1, vec2))
    sim_1_3 = float(np.dot(vec1, vec3))
    assert sim_1_2 > sim_1_3

def test_vector_store(tmp_path):
    store = VectorStore(persist_path=tmp_path / "test_store.json")
    doc_info = DocumentInfo(
        doc_id="test_doc",
        filename="test.txt",
        file_type=".txt",
        char_count=200,
        chunk_count=2,
        upload_timestamp="2026-09-19T00:00:00Z"
    )
    chunk1 = DocumentChunk(
        chunk_id="test_doc_p1_c0",
        doc_id="test_doc",
        source_name="test.txt",
        page_number=1,
        chunk_index=0,
        content="Enterprise AI policy requires TLS 1.3 encryption.",
        char_count=48
    )
    chunk2 = DocumentChunk(
        chunk_id="test_doc_p1_c1",
        doc_id="test_doc",
        source_name="test.txt",
        page_number=1,
        chunk_index=1,
        content="Local LLM weights must reside on BitLocker drives.",
        char_count=50
    )
    engine = EmbeddingEngine(provider="mock")
    embs = engine.embed_batch([chunk1.content, chunk2.content])

    store.add_document(doc_info, [chunk1, chunk2], embs)

    # Search for TLS
    q_vec = engine.embed_text("What encryption standard is required for API requests?")
    results = store.search(q_vec, top_k=1)
    assert len(results) == 1
    assert "TLS 1.3" in results[0][0].content

    # Cleanup
    store.delete_document("test_doc")
    assert len(store.search(q_vec, top_k=1)) == 0

def test_pipeline_end_to_end(tmp_path):
    # Create temp sample file
    test_file = tmp_path / "policy.txt"
    test_file.write_text(
        "Standard operating procedure mandates 90 days retention for audit logs.\n"
        "Security incident reporting must occur within two hours of discovery.",
        encoding="utf-8"
    )

    pipeline = RAGPipeline()
    # Isolate vector store for testing
    pipeline.vector_store = VectorStore(persist_path=tmp_path / "pipe_store.json")

    ingest_res = pipeline.ingest_document(test_file, "policy.txt")
    assert ingest_res.status == "success"
    assert ingest_res.document.chunk_count >= 1

    # Query
    query_res = pipeline.query(
        question="How long must audit logs be retained?",
        top_k=2,
        override_provider="mock"
    )
    assert query_res.question == "How long must audit logs be retained?"
    assert len(query_res.citations) >= 1
    assert "90 days" in query_res.citations[0].snippet or "audit logs" in query_res.citations[0].snippet
