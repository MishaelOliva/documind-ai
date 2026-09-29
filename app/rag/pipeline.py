"""RAG Pipeline: Orchestrates Document Ingestion, Semantic Retrieval, and LLM Generation."""
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from app.models import (
    Citation,
    QueryResponse,
    DocumentInfo,
    IngestResponse,
)
from app.rag.loader import DocumentLoader
from app.rag.chunker import RecursiveChunker
from app.rag.embeddings import EmbeddingEngine
from app.rag.vector_store import VectorStore
from app.rag.generator import LLMGenerator, REFUSAL_ANSWER
from app.config import settings

logger = logging.getLogger(__name__)

class RAGPipeline:
    """Complete End-to-End Retrieval-Augmented Generation Pipeline."""

    def __init__(self):
        self.loader = DocumentLoader()
        self.chunker = RecursiveChunker(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )
        self.embedder = EmbeddingEngine()
        self.vector_store = VectorStore()
        self.generator = LLMGenerator()

    def ingest_document(self, file_path: Path, original_filename: str) -> IngestResponse:
        """
        Loads document, chunks text, generates embeddings, and saves to vector index.
        """
        doc_id = str(uuid.uuid4())[:8]
        pages = self.loader.load_file(file_path)

        # Chunk pages
        chunks = self.chunker.chunk_document(
            doc_id=doc_id,
            source_name=original_filename,
            pages=pages
        )

        if not chunks:
            raise ValueError(f"No textual content could be extracted from {original_filename}")

        # Compute embeddings
        chunk_texts = [c.content for c in chunks]
        embeddings = self.embedder.embed_batch(chunk_texts)

        total_chars = sum(len(p.get("text", "")) for p in pages)
        doc_info = DocumentInfo(
            doc_id=doc_id,
            filename=original_filename,
            file_type=file_path.suffix.lower(),
            char_count=total_chars,
            chunk_count=len(chunks),
            upload_timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        )

        self.vector_store.add_document(doc_info, chunks, embeddings)

        return IngestResponse(
            status="success",
            message=f"Indexed {len(chunks)} chunks across {len(pages)} page(s).",
            document=doc_info
        )

    def query(
        self,
        question: str,
        top_k: int = 3,
        override_provider: Optional[str] = None
    ) -> QueryResponse:
        """
        Executes semantic retrieval for the question and generates a grounded response.

        Embedding time and search time are measured separately: they have very
        different cost profiles (transformer inference vs. a matrix-vector product)
        and reporting only their sum makes a retrieval benchmark meaningless.

        If the best chunk falls below `settings.MIN_SIMILARITY_SCORE`, the query is
        answered with an explicit refusal and no citations, rather than being
        passed to a generator that would be primed to invent something.
        """
        total_start = time.time()

        # Step 1: Embed the query and retrieve. Timed separately.
        embedding_start = time.time()
        q_embedding = self.embedder.embed_text(question)
        embedding_latency = (time.time() - embedding_start) * 1000.0

        search_start = time.time()
        retrieved = self.vector_store.search(q_embedding, top_k=top_k)
        search_latency = (time.time() - search_start) * 1000.0

        top_similarity = retrieved[0][1] if retrieved else 0.0
        is_grounded = bool(retrieved) and top_similarity >= settings.MIN_SIMILARITY_SCORE

        if not is_grounded:
            logger.info(
                "Refusing to answer %r: top similarity %.4f < floor %.2f",
                question, top_similarity, settings.MIN_SIMILARITY_SCORE,
            )
            return QueryResponse(
                question=question,
                answer=REFUSAL_ANSWER,
                provider_used="none",
                model_name="Grounding Guard",
                embedding_latency_ms=round(embedding_latency, 2),
                search_latency_ms=round(search_latency, 2),
                retrieval_latency_ms=round(embedding_latency + search_latency, 2),
                generation_latency_ms=0.0,
                total_latency_ms=round((time.time() - total_start) * 1000.0, 2),
                citations=[],
                retrieved_chunk_count=0,
                top_similarity=round(top_similarity, 4),
                is_grounded=False,
            )

        # Step 2: Build citations
        citations = []
        for chunk, score in retrieved:
            snippet = chunk.content[:180].strip() + ("..." if len(chunk.content) > 180 else "")
            citations.append(
                Citation(
                    chunk_id=chunk.chunk_id,
                    source_name=chunk.source_name,
                    page_number=chunk.page_number,
                    chunk_index=chunk.chunk_index,
                    similarity_score=round(score, 4),
                    snippet=snippet
                )
            )

        # Step 3: Generate Grounded Answer
        answer_text, provider_used, model_name, gen_latency = self.generator.generate_answer(
            question=question,
            retrieved_chunks=retrieved,
            override_provider=override_provider
        )

        return QueryResponse(
            question=question,
            answer=answer_text,
            provider_used=provider_used,
            model_name=model_name,
            embedding_latency_ms=round(embedding_latency, 2),
            search_latency_ms=round(search_latency, 2),
            retrieval_latency_ms=round(embedding_latency + search_latency, 2),
            generation_latency_ms=round(gen_latency, 2),
            total_latency_ms=round((time.time() - total_start) * 1000.0, 2),
            citations=citations,
            retrieved_chunk_count=len(retrieved),
            top_similarity=round(top_similarity, 4),
            is_grounded=True,
        )

    def delete_document(self, doc_id: str) -> bool:
        """Deletes document and its embeddings from index."""
        return self.vector_store.delete_document(doc_id)

    def list_documents(self) -> List[DocumentInfo]:
        """Lists all indexed documents."""
        return self.vector_store.list_documents()

    def get_stats(self):
        """Returns pipeline and store statistics."""
        stats = self.vector_store.get_stats()
        stats["active_provider"] = settings.LLM_PROVIDER
        stats["embedding_backend"] = self.embedder.backend_name
        stats["gemini_configured"] = bool(settings.GEMINI_API_KEY)
        stats["min_similarity_score"] = settings.MIN_SIMILARITY_SCORE
        return stats
