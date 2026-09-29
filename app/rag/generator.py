"""LLM Generator: Prompts LLM providers (Gemini, Ollama, Mock) with retrieved context."""
import logging
import re
import time
from typing import List, Tuple
import httpx
from app.models import DocumentChunk
from app.config import settings

logger = logging.getLogger(__name__)

STRICT_RAG_SYSTEM_PROMPT = """You are an accurate, source-grounded technical AI assistant.
Answer the user's question ONLY based on the provided context excerpts below.
Follow these strict rules:
1. Ground every claim directly in the context. Do NOT use prior external knowledge or fabricate details.
2. If the context does not contain enough information to answer the question, state: "The provided documents do not contain sufficient information to answer this question."
3. Cite the exact sources in your answer using bracket notation like [Source: filename, Page: X].
4. Keep the response clear, professional, and concise.
"""

REFUSAL_ANSWER = (
    "The provided documents do not contain sufficient information to answer this question."
)

class LLMGenerator:
    """Dispatches prompt requests to the configured LLM provider."""

    def __init__(self, provider: str = None):
        self.provider = provider or settings.LLM_PROVIDER

    def generate_answer(
        self,
        question: str,
        retrieved_chunks: List[Tuple[DocumentChunk, float]],
        override_provider: str = None
    ) -> Tuple[str, str, str, float]:
        """
        Builds the grounded prompt and queries the LLM.

        Returns: (answer_text, provider_actually_used, model_name, latency_ms)

        The second element reports the provider that *actually* produced the
        answer, which differs from the requested one whenever a remote call fails
        and the deterministic extractor takes over.
        """
        active_provider = override_provider or self.provider
        start_time = time.time()

        context_blocks = []
        for idx, (chunk, score) in enumerate(retrieved_chunks):
            context_blocks.append(
                f"--- EXCERPT {idx + 1} [Source: {chunk.source_name}, Page: {chunk.page_number}] ---\n"
                f"{chunk.content}\n"
            )
        context_str = "\n".join(context_blocks)

        user_prompt = (
            f"CONTEXT EXCERPTS:\n{context_str}\n\n"
            f"USER QUESTION: {question}\n\n"
            f"ANSWER (Ground strictly on the excerpts above with citations):"
        )

        if active_provider == "gemini" and settings.GEMINI_API_KEY:
            try:
                answer, model_tag = self._call_gemini(user_prompt)
                provider_used = "gemini"
            except Exception as e:
                logger.warning("Gemini API failed (%s); falling back to deterministic extractor.", e)
                answer, model_tag = self._call_mock_extractor(question, retrieved_chunks)
                provider_used = "mock"
        elif active_provider == "ollama":
            try:
                answer, model_tag = self._call_ollama(user_prompt)
                provider_used = "ollama"
            except Exception as e:
                logger.warning("Ollama failed (%s); falling back to deterministic extractor.", e)
                answer, model_tag = self._call_mock_extractor(question, retrieved_chunks)
                provider_used = "mock"
        else:
            answer, model_tag = self._call_mock_extractor(question, retrieved_chunks)
            provider_used = "mock"

        latency_ms = (time.time() - start_time) * 1000.0
        return answer, provider_used, model_tag, latency_ms

    def _call_gemini(self, prompt: str) -> Tuple[str, str]:
        """Calls Google Gemini Generative Language API."""
        model = settings.GEMINI_MODEL
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={settings.GEMINI_API_KEY}"
        payload = {
            "contents": [
                {"role": "user", "parts": [{"text": f"{STRICT_RAG_SYSTEM_PROMPT}\n\n{prompt}"}]}
            ],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 800
            }
        }
        resp = httpx.post(url, json=payload, timeout=20.0)
        resp.raise_for_status()
        data = resp.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        return text.strip(), f"Gemini ({model})"

    def _call_ollama(self, prompt: str) -> Tuple[str, str]:
        """Calls local Ollama instance."""
        model = settings.OLLAMA_MODEL
        url = f"{settings.OLLAMA_BASE_URL}/api/generate"
        payload = {
            "model": model,
            "system": STRICT_RAG_SYSTEM_PROMPT,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2}
        }
        resp = httpx.post(url, json=payload, timeout=30.0)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "").strip(), f"Ollama ({model})"

    def _call_mock_extractor(
        self,
        question: str,
        retrieved_chunks: List[Tuple[DocumentChunk, float]]
    ) -> Tuple[str, str]:
        """
        Deterministic, offline extraction engine.
        Extracts relevant sentences from the top retrieved chunks without requiring external LLM keys.
        """
        if not retrieved_chunks:
            return REFUSAL_ANSWER, "Deterministic Extractor (Offline)"

        top_chunk, score = retrieved_chunks[0]
        keywords = [w.lower() for w in re.findall(r"\w+", question) if len(w) > 3]

        sentences = re.split(r"(?<=[.!?])\s+", top_chunk.content)
        matched_sentences = []
        for s in sentences:
            s_clean = s.strip()
            if any(kw in s_clean.lower() for kw in keywords):
                matched_sentences.append(s_clean)

        if not matched_sentences:
            matched_sentences = sentences[:2]

        extracted_text = " ".join(matched_sentences)
        answer = (
            f"Based on the indexed document, {extracted_text} "
            f"[Source: {top_chunk.source_name}, Page: {top_chunk.page_number}]"
        )
        return answer, "Deterministic Extractor (Zero-Key Demo)"
