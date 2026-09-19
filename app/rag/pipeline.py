"""RAG Pipeline: Orchestrates Document Ingestion, Semantic Retrieval, and LLM Generation."""
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from app.models import (
    DocumentChunk,
    Citation,
    QueryResponse,
    DocumentInfo,
    IngestResponse,
)
from app.rag.loader import DocumentLoader
from app.rag.chunker import RecursiveChunker
from app.rag.embeddings import EmbeddingEngine
from app.rag.vector_store import VectorStore
from app.rag.generator import LLMGenerator
from app.config import settings

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
            upload_timestamp=datetime.utcnow().isoformat() + "Z"
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
        """
        total_start = time.time()

        # Step 1: Embed Query & Retrieve
        retrieval_start = time.time()
        q_embedding = self.embedder.embed_text(question)
        retrieved = self.vector_store.search(q_embedding, top_k=top_k)
        retrieval_latency = (time.time() - retrieval_start) * 1000.0

        # Step 2: Build Citations
        citations = []
        for chunk, score in retrieved:
            # First 180 chars as snippet
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
        answer_text, model_name, gen_latency = self.generator.generate_answer(
            question=question,
            retrieved_chunks=retrieved,
            override_provider=override_provider
        )

        total_latency = (time.time() - total_start) * 1000.0

        return QueryResponse(
            question=question,
            answer=answer_text,
            provider_used=override_provider or settings.LLM_PROVIDER,
            model_name=model_name,
            retrieval_latency_ms=round(retrieval_latency, 2),
            generation_latency_ms=round(gen_latency, 2),
            total_latency_ms=round(total_latency, 2),
            citations=citations,
            retrieved_chunk_count=len(retrieved)
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
        stats["gemini_configured"] = bool(settings.GEMINI_API_KEY)
        return stats
