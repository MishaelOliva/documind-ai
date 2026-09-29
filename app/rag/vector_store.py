"""Vector Store: In-memory & persisted vector database for chunk embeddings and semantic similarity search."""
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from app.models import DocumentChunk, DocumentInfo
from app.config import settings

logger = logging.getLogger(__name__)

class VectorStore:
    """
    Manages vector index, chunk metadata, and cosine similarity retrieval.
    Stores index state on disk in JSON format for zero-dependency portability.

    Mutations are serialised with a re-entrant lock so concurrent uploads cannot
    interleave a read-modify-write of the on-disk snapshot.
    """

    def __init__(self, persist_path: Optional[Path] = None):
        self.persist_path = persist_path or (settings.VECTOR_DB_DIR / "vector_index.json")
        self.chunks: Dict[str, DocumentChunk] = {}
        self.documents: Dict[str, DocumentInfo] = {}
        self.embeddings: Dict[str, List[float]] = {}
        self._lock = threading.RLock()
        self._matrix: Optional[np.ndarray] = None
        self._matrix_ids: List[str] = []
        self._load_from_disk()

    def add_document(self, doc_info: DocumentInfo, chunks: List[DocumentChunk], embeddings: List[List[float]]) -> None:
        """Stores a document, its chunks, and corresponding embedding vectors."""
        if len(chunks) != len(embeddings):
            raise ValueError(f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings")

        with self._lock:
            self.documents[doc_info.doc_id] = doc_info
            for chunk, emb in zip(chunks, embeddings):
                self.chunks[chunk.chunk_id] = chunk
                self.embeddings[chunk.chunk_id] = emb
            self._invalidate_matrix()
            self._save_to_disk()

    def delete_document(self, doc_id: str) -> bool:
        """Removes a document and all its associated chunks and embeddings."""
        with self._lock:
            if doc_id not in self.documents:
                return False

            del self.documents[doc_id]
            chunk_ids_to_del = [cid for cid, chunk in self.chunks.items() if chunk.doc_id == doc_id]
            for cid in chunk_ids_to_del:
                self.chunks.pop(cid, None)
                self.embeddings.pop(cid, None)

            self._invalidate_matrix()
            self._save_to_disk()
            return True

    def _invalidate_matrix(self) -> None:
        """Drops the cached embedding matrix after a mutation."""
        self._matrix = None
        self._matrix_ids = []

    def _get_matrix(self) -> tuple[Optional[np.ndarray], List[str]]:
        """
        Returns the stacked embedding matrix, building and caching it on first use.

        Stacking every vector on each query dominated the search cost for larger
        corpora; caching it means a query is a single matrix-vector product.
        """
        with self._lock:
            if self._matrix is None or self._matrix_ids != list(self.embeddings.keys()):
                if not self.embeddings:
                    self._matrix, self._matrix_ids = None, []
                else:
                    self._matrix_ids = list(self.embeddings.keys())
                    self._matrix = np.array(
                        [self.embeddings[cid] for cid in self._matrix_ids], dtype=np.float32
                    )
            return self._matrix, self._matrix_ids

    def search(self, query_embedding: List[float], top_k: int = 3) -> List[Tuple[DocumentChunk, float]]:
        """
        Calculates cosine similarity between query embedding and all indexed chunk vectors.
        Returns top_k (chunk, similarity_score) pairs ranked in descending order.
        """
        matrix, chunk_ids = self._get_matrix()
        if matrix is None or not chunk_ids:
            return []

        q_vec = np.array(query_embedding, dtype=np.float32)

        # Dot product for L2-normalized vectors == Cosine Similarity
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm

        # argpartition is O(N) versus the O(N log N) of a full argsort; only the
        # top-k slice is ever ordered.
        k = min(max(top_k, 1), len(chunk_ids))
        scores = matrix @ q_vec
        top_indices = np.argpartition(-scores, k - 1)[:k]
        top_indices = top_indices[np.argsort(-scores[top_indices])]

        results: List[Tuple[DocumentChunk, float]] = []
        for idx in top_indices:
            cid = chunk_ids[idx]
            results.append((self.chunks[cid], float(scores[idx])))

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
        """
        Serializes index state to disk atomically.

        Writes to a temporary file in the same directory and then renames it over
        the target, so an interrupted write can never leave a half-written index
        behind. Combined with `_load_from_disk` failing loudly, this keeps a crash
        from silently turning into total data loss.
        """
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "documents": {k: v.model_dump() for k, v in self.documents.items()},
            "chunks": {k: v.model_dump() for k, v in self.chunks.items()},
            "embeddings": self.embeddings
        }
        tmp_path = self.persist_path.with_suffix(self.persist_path.suffix + ".tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f)
            os.replace(tmp_path, self.persist_path)  # atomic on POSIX and Windows
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def _load_from_disk(self) -> None:
        """
        Deserializes index state from disk if present.

        A corrupt or unreadable index is moved aside rather than silently
        discarded, so the failure is visible to the operator instead of looking
        like an empty database.
        """
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
            quarantine = self.persist_path.with_suffix(
                self.persist_path.suffix + f".corrupt-{int(time.time())}"
            )
            self.persist_path.replace(quarantine)
            logger.error(
                "Vector index at %s is unreadable (%s); moved to %s and starting empty. "
                "Re-upload your documents to rebuild it.",
                self.persist_path, e, quarantine,
            )
            self.documents = {}
            self.chunks = {}
            self.embeddings = {}
