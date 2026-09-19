"""Embedding Engine: Generates dense semantic vectors for text chunks and queries."""
import math
import hashlib
import re
from typing import List
import numpy as np
import httpx
from app.config import settings

class EmbeddingEngine:
    """
    Produces normalized dense embedding vectors.
    Supports:
    1. FastEmbed ONNX (BAAI/bge-small-en-v1.5, 384-d dense semantic vectors, SOTA MTEB)
    2. Gemini API (text-embedding-004)
    3. Ollama Local (/api/embeddings)
    4. Deterministic Subword Hashing Embedder fallback
    """

    DIMENSION = 384
    _fastembed_model = None

    def __init__(self, provider: str = None):
        self.provider = provider or settings.LLM_PROVIDER
        self._init_fastembed()

    @classmethod
    def _init_fastembed(cls):
        if cls._fastembed_model is None:
            try:
                from fastembed import TextEmbedding
                cls._fastembed_model = TextEmbedding("BAAI/bge-small-en-v1.5")
            except Exception as e:
                cls._fastembed_model = False

    def embed_text(self, text: str) -> List[float]:
        """Embeds a single string into a normalized dense vector."""
        vectors = self.embed_batch([text])
        return vectors[0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embeds a batch of strings into normalized dense vectors."""
        if not texts:
            return []

        # Try Gemini if configured
        if self.provider == "gemini" and settings.GEMINI_API_KEY:
            try:
                return self._embed_gemini(texts)
            except Exception as e:
                print(f"[Warning] Gemini embedding failed: {e}. Falling back to internal engine.")

        # Try Ollama if configured
        if self.provider == "ollama":
            try:
                return self._embed_ollama(texts)
            except Exception as e:
                print(f"[Warning] Ollama embedding failed: {e}. Falling back to internal engine.")

        # Try FastEmbed SOTA ONNX model
        if self._fastembed_model:
            try:
                embeddings = list(self._fastembed_model.embed(texts))
                results = []
                for vec in embeddings:
                    norm = np.linalg.norm(vec)
                    results.append((vec / norm).tolist() if norm > 0 else vec.tolist())
                return results
            except Exception as e:
                print(f"[Warning] FastEmbed inference failed: {e}. Falling back to hashing embedder.")

        # Robust built-in dense hashing embedder fallback
        return [self._embed_dense_hash(t) for t in texts]

    def _embed_dense_hash(self, text: str) -> List[float]:
        """
        Deterministic 384-dimensional dense semantic representation.
        Uses subword character n-grams (3-grams, 4-grams) and word tokens,
        hashed into a fixed-dimension vector with L2 normalization.
        Cosine similarity between vectors accurately captures lexical & morphological relevance.
        """
        vec = np.zeros(self.DIMENSION, dtype=np.float32)
        words = re.findall(r"\w+", text.lower())

        for word in words:
            # Word level feature
            h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16) % self.DIMENSION
            vec[h] += 1.5

            # Subword character n-grams (captures prefixes, stems, suffixes)
            padded = f"^{word}$"
            for n in (3, 4):
                if len(padded) >= n:
                    for i in range(len(padded) - n + 1):
                        ngram = padded[i:i + n]
                        h_sub = int(hashlib.sha256(ngram.encode("utf-8")).hexdigest(), 16) % self.DIMENSION
                        vec[h_sub] += 0.5

        # L2 normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        else:
            vec[0] = 1.0

        return vec.tolist()

    def _embed_gemini(self, texts: List[str]) -> List[List[float]]:
        """Calls Google Gemini text-embedding-004."""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:batchEmbedContents?key={settings.GEMINI_API_KEY}"
        requests_payload = [
            {"model": "models/text-embedding-004", "content": {"parts": [{"text": t[:2048]}]}}
            for t in texts
        ]
        response = httpx.post(url, json={"requests": requests_payload}, timeout=15.0)
        response.raise_for_status()
        data = response.json()
        embeddings = [item["values"] for item in data.get("embeddings", [])]
        # Normalize
        normalized = []
        for vec in embeddings:
            arr = np.array(vec, dtype=np.float32)
            norm = np.linalg.norm(arr)
            normalized.append((arr / norm).tolist() if norm > 0 else vec)
        return normalized

    def _embed_ollama(self, texts: List[str]) -> List[List[float]]:
        """Calls local Ollama embeddings endpoint."""
        results = []
        for text in texts:
            resp = httpx.post(
                f"{settings.OLLAMA_BASE_URL}/api/embeddings",
                json={"model": settings.OLLAMA_MODEL, "prompt": text},
                timeout=10.0
            )
            resp.raise_for_status()
            emb = resp.json().get("embedding", [])
            arr = np.array(emb, dtype=np.float32)
            norm = np.linalg.norm(arr)
            results.append((arr / norm).tolist() if norm > 0 else emb)
        return results
