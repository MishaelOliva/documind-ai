"""Unit and integration tests for the DocuMind RAG pipeline."""
import json
from pathlib import Path

import pytest

from app.config import settings
from app.models import DocumentChunk, DocumentInfo
from app.rag.chunker import RecursiveChunker
from app.rag.embeddings import EmbeddingEngine
from app.rag.generator import LLMGenerator, REFUSAL_ANSWER
from app.rag.loader import DocumentLoader
from app.rag.pipeline import RAGPipeline
from app.rag.vector_store import VectorStore

SAMPLE_DOC = Path(__file__).resolve().parent.parent / "sample_docs" / "company_ai_policy.txt"


# --------------------------------------------------------------------- chunker

def test_chunker_respects_chunk_size():
    chunker = RecursiveChunker(chunk_size=200, chunk_overlap=40)
    text = " ".join(f"Sentence number {i} carries some filler content." for i in range(60))
    chunks = chunker.split_text(text)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)


def test_chunker_never_splits_mid_word():
    """Regression test: the hard-split path used to cut at exact char offsets,
    producing chunks that began mid-word ("ust never be exposed")."""
    chunker = RecursiveChunker(chunk_size=120, chunk_overlap=20)
    text = (
        "The quick brown fox jumps over the lazy dog while the enterprise policy "
        "requires TLS 1.3 encryption for every cloud request that leaves the network."
    )
    for chunk in chunker.split_text(text):
        first = chunk.split()[0]
        # A real word from the source, not a fragment of one.
        assert first in text.split(), f"chunk starts mid-word: {first!r}"


def test_chunker_rejects_overlap_ge_size():
    with pytest.raises(ValueError):
        RecursiveChunker(chunk_size=100, chunk_overlap=100)


def test_chunk_document_assigns_unique_ids_and_metadata():
    chunker = RecursiveChunker(chunk_size=200, chunk_overlap=40)
    pages = [{"page_number": 1, "text": "alpha beta gamma. " * 30}]
    chunks = chunker.chunk_document("doc1", "sample.txt", pages)
    assert chunks
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert all(c.doc_id == "doc1" for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


# ------------------------------------------------------------------ embeddings

def test_embedding_engine_dimensions_and_similarity_ordering():
    engine = EmbeddingEngine(provider="mock")
    related = engine.embed_text("FastAPI backend with a vector database")
    also_related = engine.embed_text("FastAPI application with vector embeddings")
    unrelated = engine.embed_text("Baking chocolate chip cookies in an oven")

    assert len(related) == engine.DIMENSION
    assert len(also_related) == engine.DIMENSION
    assert len(unrelated) == engine.DIMENSION

    import numpy as np
    assert float(np.dot(related, also_related)) > float(np.dot(related, unrelated))


def test_embedding_vectors_are_unit_norm():
    import numpy as np
    engine = EmbeddingEngine(provider="mock")
    vec = engine.embed_text("unit norm check for the persistence layer")
    assert pytest.approx(1.0, abs=1e-4) == float(np.linalg.norm(vec))


def test_embedding_batch_empty_returns_empty():
    assert EmbeddingEngine(provider="mock").embed_batch([]) == []


# ---------------------------------------------------------------- vector store

def _make_chunk(chunk_id: str, doc_id: str, content: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id, doc_id=doc_id, source_name=f"{doc_id}.txt",
        page_number=1, chunk_index=0, content=content, char_count=len(content),
    )


def test_vector_store_roundtrip_and_search(tmp_path):
    store = VectorStore(persist_path=tmp_path / "s.json")
    info = DocumentInfo(
        doc_id="d1", filename="d1.txt", file_type=".txt", char_count=40,
        chunk_count=2, upload_timestamp="2026-09-19T00:00:00Z",
    )
    chunks = [
        _make_chunk("d1_c0", "d1", "Enterprise AI policy requires TLS 1.3 encryption."),
        _make_chunk("d1_c1", "d1", "Local LLM weights must reside on BitLocker drives."),
    ]
    engine = EmbeddingEngine(provider="mock")
    store.add_document(info, chunks, engine.embed_batch([c.content for c in chunks]))

    results = store.search(engine.embed_text("Which encryption standard is required?"), top_k=1)
    assert len(results) == 1
    assert "TLS 1.3" in results[0][0].content
    assert -1.0 <= results[0][1] <= 1.0

    assert store.delete_document("d1") is True
    assert store.delete_document("d1") is False
    assert store.search(engine.embed_text("anything"), top_k=1) == []


def test_vector_store_persists_across_instances(tmp_path):
    path = tmp_path / "p.json"
    store = VectorStore(persist_path=path)
    info = DocumentInfo(
        doc_id="d2", filename="d2.txt", file_type=".txt", char_count=10,
        chunk_count=1, upload_timestamp="2026-09-19T00:00:00Z",
    )
    chunk = _make_chunk("d2_c0", "d2", "persisted chunk content")
    engine = EmbeddingEngine(provider="mock")
    store.add_document(info, [chunk], [engine.embed_text(chunk.content)])

    reloaded = VectorStore(persist_path=path)
    assert reloaded.get_stats()["total_chunks"] == 1
    assert "d2_c0" in reloaded.chunks


def test_vector_store_write_is_atomic(tmp_path):
    """A crash mid-write must not destroy the index: no .tmp file may survive
    and the real file must always be complete JSON."""
    path = tmp_path / "a.json"
    store = VectorStore(persist_path=path)
    info = DocumentInfo(
        doc_id="d3", filename="d3.txt", file_type=".txt", char_count=5,
        chunk_count=1, upload_timestamp="2026-09-19T00:00:00Z",
    )
    chunk = _make_chunk("d3_c0", "d3", "atomic write check")
    engine = EmbeddingEngine(provider="mock")
    store.add_document(info, [chunk], [engine.embed_text(chunk.content)])

    assert not list(tmp_path.glob("*.tmp"))
    assert json.loads(path.read_text(encoding="utf-8"))["chunks"]


def test_vector_store_quarantines_corrupt_index(tmp_path):
    """A corrupt index is moved aside rather than silently discarded, so the
    failure is visible instead of looking like an empty database."""
    path = tmp_path / "c.json"
    path.write_text("{ this is not valid json", encoding="utf-8")

    store = VectorStore(persist_path=path)
    assert store.get_stats()["total_chunks"] == 0
    quarantined = list(tmp_path.glob("c.json.corrupt-*"))
    assert quarantined, "corrupt index should be preserved for inspection"


def test_vector_store_rejects_count_mismatch(tmp_path):
    store = VectorStore(persist_path=tmp_path / "m.json")
    info = DocumentInfo(
        doc_id="d4", filename="d4.txt", file_type=".txt", char_count=5,
        chunk_count=1, upload_timestamp="2026-09-19T00:00:00Z",
    )
    with pytest.raises(ValueError):
        store.add_document(info, [_make_chunk("d4_c0", "d4", "x")], [])


# --------------------------------------------------------------------- loader

def test_loader_rejects_unsupported_extension(tmp_path):
    bad = tmp_path / "file.exe"
    bad.write_text("nope", encoding="utf-8")
    with pytest.raises(ValueError):
        DocumentLoader.load_file(bad)


def test_loader_rejects_empty_text_file(tmp_path):
    empty = tmp_path / "empty.txt"
    empty.write_text("   \n\n  ", encoding="utf-8")
    with pytest.raises(ValueError):
        DocumentLoader.load_file(empty)


def test_loader_collapses_blank_lines():
    text = DocumentLoader._clean_text("a\n\n\n\nb\n   \n\nc")
    assert text == "a\n\nb\n\nc"


def test_loader_accepts_every_extension_the_api_allows(tmp_path):
    """The API allow-list and the loader must agree, or a file can be accepted
    by the upload endpoint and then rejected during ingestion."""
    for ext in settings.ALLOWED_EXTENSIONS:
        f = tmp_path / f"sample{ext}"
        if ext == ".pdf":
            # A real PDF is required; pypdf rejects arbitrary bytes in a .pdf.
            from pypdf import PdfWriter
            writer = PdfWriter()
            writer.add_blank_page(width=200, height=200)
            with open(f, "wb") as fh:
                writer.write(fh)
            # A blank page has no text layer, so ingestion is expected to fail
            # with the "no readable text" error rather than a format error.
            with pytest.raises(ValueError, match="no readable text"):
                DocumentLoader.load_file(f)
        else:
            f.write_text("Some ingestible content for the loader contract test.", encoding="utf-8")
            assert DocumentLoader.load_file(f)


# ------------------------------------------------------------------ generator

def test_generator_reports_the_provider_that_actually_ran(monkeypatch):
    """A failing remote provider must be reported as the fallback that produced
    the answer, not as the provider that was requested."""
    gen = LLMGenerator(provider="gemini")
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(
        gen, "_call_gemini", lambda p: (_ for _ in ()).throw(RuntimeError("network down"))
    )
    chunk = _make_chunk("c0", "d", "Chunk text about policy requirements.")
    answer, provider_used, model_name, _ = gen.generate_answer("policy?", [(chunk, 0.9)])
    assert provider_used == "mock"
    assert "Deterministic" in model_name
    assert answer


def test_generator_uses_gemini_when_available(monkeypatch):
    gen = LLMGenerator(provider="gemini")
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(gen, "_call_gemini", lambda p: ("grounded reply", "Gemini (m)"))
    chunk = _make_chunk("c0", "d", "Chunk text about policy requirements.")
    _, provider_used, _, _ = gen.generate_answer("policy?", [(chunk, 0.9)])
    assert provider_used == "gemini"


def test_generator_refuses_with_no_chunks():
    answer, model_name = LLMGenerator(provider="mock")._call_mock_extractor("q", [])
    assert answer == REFUSAL_ANSWER


# ------------------------------------------------------------------- pipeline

@pytest.fixture
def pipeline(tmp_path):
    p = RAGPipeline()
    p.vector_store = VectorStore(persist_path=tmp_path / "pipe.json")
    return p


def test_pipeline_ingest_and_query(pipeline):
    ingest = pipeline.ingest_document(SAMPLE_DOC, SAMPLE_DOC.name)
    assert ingest.status == "success"
    assert ingest.document.chunk_count > 0
    assert ingest.document.upload_timestamp.endswith("Z")

    res = pipeline.query(
        question="What is the maximum token chunking limit for internal RAG vector databases?",
        top_k=3, override_provider="mock",
    )
    assert res.is_grounded is True
    assert res.citations
    assert res.retrieved_chunk_count == len(res.citations)
    # retrieval must be reported as the sum of its two parts
    assert res.retrieval_latency_ms == pytest.approx(
        res.embedding_latency_ms + res.search_latency_ms, abs=0.05
    )


def test_pipeline_retrieves_the_correct_section(pipeline):
    """Passage-level check: the top chunk must overlap the subsection the
    question is actually about. Stricter than keyword matching."""
    pipeline.ingest_document(SAMPLE_DOC, SAMPLE_DOC.name)
    cases = [
        ("Which local LLM architectures are authorized for air-gapped development?", "Qwen 2.5"),
        ("What telemetry must be recorded during each LLM query?", "90 days"),
        ("Can proprietary source code be uploaded to consumer web chatbots?", "strictly prohibited"),
    ]
    for question, expected in cases:
        res = pipeline.query(question=question, top_k=1, override_provider="mock")
        top = pipeline.vector_store.chunks[res.citations[0].chunk_id].content
        assert expected in top, f"{expected!r} missing for {question!r}"


def test_pipeline_refuses_when_above_similarity_floor(pipeline, monkeypatch):
    """With the floor raised above any achievable score, the pipeline must
    refuse rather than hand a weakly-related chunk to the generator."""
    pipeline.ingest_document(SAMPLE_DOC, SAMPLE_DOC.name)
    monkeypatch.setattr(settings, "MIN_SIMILARITY_SCORE", 1.01)
    res = pipeline.query(question="What is the audit retention period?", override_provider="mock")
    assert res.is_grounded is False
    assert res.answer == REFUSAL_ANSWER
    assert res.citations == []
    assert res.provider_used == "none"


def test_pipeline_rejects_document_with_no_text(pipeline, tmp_path):
    blank = tmp_path / "blank.md"
    blank.write_text("   \n\n", encoding="utf-8")
    with pytest.raises(ValueError):
        pipeline.ingest_document(blank, "blank.md")


# ----------------------------------------------------------------------- API

@pytest.fixture
def client(tmp_path, monkeypatch):
    """
    TestClient with an isolated vector index.

    `app.main` holds a module-level pipeline singleton backed by the on-disk index
    in `data/`. Without swapping in a temporary store, API tests inherit whatever
    happens to be indexed on the developer's machine, which makes them order- and
    environment-dependent.
    """
    from fastapi.testclient import TestClient
    from app.main import app, pipeline

    monkeypatch.setattr(
        pipeline, "vector_store", VectorStore(persist_path=tmp_path / "api_store.json")
    )
    return TestClient(app)


def test_health_reports_embedding_backend_and_floor(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "healthy"
    assert body["embedding_backend"]
    assert body["min_similarity_score"] == settings.MIN_SIMILARITY_SCORE


def test_upload_rejects_unsupported_format(client):
    res = client.post(
        "/api/upload",
        files={"file": ("payload.exe", b"binary", "application/octet-stream")},
    )
    assert res.status_code == 400
    assert "Unsupported format" in res.json()["detail"]


def test_upload_sanitizes_hostile_filename(client):
    """Uploaded names are attacker-controlled and are echoed back to every
    client, so they are reduced to a safe basename at the boundary."""
    hostile = 'x"><img src=x onerror=alert(1)>.txt'
    res = client.post("/api/upload", files={"file": (hostile, b"ingestible body", "text/plain")})
    assert res.status_code == 201
    stored = res.json()["document"]["filename"]
    for char in "<>&\"'":
        assert char not in stored, f"dangerous character {char!r} survived sanitisation"
    assert stored.endswith(".txt")


def test_upload_oversize_returns_413(client, monkeypatch):
    monkeypatch.setattr(settings, "MAX_UPLOAD_SIZE_BYTES", 512)
    res = client.post("/api/upload", files={"file": ("big.txt", b"x" * 4096, "text/plain")})
    assert res.status_code == 413


def test_upload_empty_document_returns_422_not_500(client):
    res = client.post("/api/upload", files={"file": ("empty.md", b"   \n\n", "text/markdown")})
    assert res.status_code == 422


def test_query_without_documents_returns_400(client):
    res = client.post("/api/query", json={"question": "anything?"})
    assert res.status_code == 400
    assert "No documents" in res.json()["detail"]


def test_delete_unknown_document_returns_404(client):
    assert client.delete("/api/documents/does-not-exist").status_code == 404


def test_query_rejects_out_of_range_top_k(client):
    res = client.post("/api/query", json={"question": "valid question", "top_k": 99})
    assert res.status_code == 422


def test_query_response_exposes_split_latency(client):
    client.post(
        "/api/upload",
        files={"file": ("api_test.txt", b"Section 9.1 Audit retention is 90 days.", "text/plain")},
    )
    res = client.post("/api/query", json={"question": "How long are logs retained?", "top_k": 1})
    assert res.status_code == 200
    body = res.json()
    for field in ("embedding_latency_ms", "search_latency_ms", "retrieval_latency_ms",
                  "top_similarity", "is_grounded", "provider_used"):
        assert field in body, f"missing telemetry field: {field}"
    assert body["retrieval_latency_ms"] == pytest.approx(
        body["embedding_latency_ms"] + body["search_latency_ms"], abs=0.05
    )


def test_static_index_is_served(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "DocuMind" in res.text
