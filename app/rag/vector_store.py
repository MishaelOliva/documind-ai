"""Vector Store: In-memory & persisted vector database for chunk embeddings and semantic similarity search."""
import json
import os
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from app.models import DocumentChunk, Citation, DocumentInfo
from app.config import settings

class VectorStore:
    """
    Manages vector index, chunk metadata, and cosine similarity retrieval.
    Stores index state on disk in JSON format for zero-dependency portability.
    """

    def __init__(self, persist_path: Optional[Path] = None):
        self.persist_path = persist_path or (settings.VECTOR_DB_DIR / "vector_index.json")
        self.chunks: Dict[str, DocumentChunk] = {}
        self.documents: Dict[str, DocumentInfo] = {}
        self.embeddings: Dict[str, List[float]] = {}
        self._load_from_disk()

    def add_document(self, doc_info: DocumentInfo, chunks: List[DocumentChunk], embeddings: List[List[float]]) -> None:
        """Stores a document, its chunks, and corresponding embedding vectors."""
        if len(chunks) != len(embeddings):
            raise ValueError(f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings")

        self.documents[doc_info.doc_id] = doc_info
        for chunk, emb in zip(chunks, embeddings):
            self.chunks[chunk.chunk_id] = chunk
            self.embeddings[chunk.chunk_id] = emb

        self._save_to_disk()

    def delete_document(self, doc_id: str) -> bool:
        """Removes a document and all its associated chunks and embeddings."""
        if doc_id not in self.documents:
            return False

        del self.documents[doc_id]
        chunk_ids_to_del = [cid for cid, chunk in self.chunks.items() if chunk.doc_id == doc_id]
        for cid in chunk_ids_to_del:
            self.chunks.pop(cid, None)
            self.embeddings.pop(cid, None)

        self._save_to_disk()
        return True

    def search(self, query_embedding: List[float], top_k: int = 3) -> List[Tuple[DocumentChunk, float]]:
        """
        Calculates cosine similarity between query embedding and all indexed chunk vectors.
        Returns top_k (chunk, similarity_score) pairs ranked in descending order.
        """
        if not self.embeddings:
            return []

        chunk_ids = list(self.embeddings.keys())
        matrix = np.array([self.embeddings[cid] for cid in chunk_ids], dtype=np.float32)
        q_vec = np.array(query_embedding, dtype=np.float32)

        # Dot product for L2-normalized vectors == Cosine Similarity
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm

        scores = np.dot(matrix, q_vec)

        # Rank indices
        top_k = min(top_k, len(chunk_ids))
        top_indices = np.argsort(scores)[::-1][:top_k]

        results: List[Tuple[DocumentChunk, float]] = []
        for idx in top_indices:
            cid = chunk_ids[idx]
            score = float(scores[idx])
            results.append((self.chunks[cid], score))

        return results

    def list_documents(self) -> List[DocumentInfo]:
        """Returns metadata for all indexed documents."""
        return list(self.documents.values())

    def get_stats(self) -> Dict[str, Any]:
        """Returns statistics on total documents, chunks, and storage."""
        return {
            "total_documents": len(self.documents),
            "total_chunks": len(self.chunks),
            "vector_dimension": len(next(iter(self.embeddings.values()))) if self.embeddings else 384,
            "persist_file": str(self.persist_path)
        }

    def _save_to_disk(self) -> None:
        """Serializes index state to disk."""
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "documents": {k: v.model_dump() for k, v in self.documents.items()},
            "chunks": {k: v.model_dump() for k, v in self.chunks.items()},
            "embeddings": self.embeddings
        }
        with open(self.persist_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    def _load_from_disk(self) -> None:
        """Deserializes index state from disk if present."""
        if not self.persist_path.exists():
            return
        try:
            with open(self.persist_path, "r", encoding="utf-8") as f:
                payload = json.load(f)

            self.documents = {
                k: DocumentInfo(**v) for k, v in payload.get("documents", {}).items()
            }
            self.chunks = {
                k: DocumentChunk(**v) for k, v in payload.get("chunks", {}).items()
            }
            self.embeddings = payload.get("embeddings", {})
        except Exception as e:
            print(f"[Warning] Failed to load vector store from disk: {e}. Initializing empty store.")
            self.documents = {}
            self.chunks = {}
            self.embeddings = {}
