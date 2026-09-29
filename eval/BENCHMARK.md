# Applied AI Document QA & RAG System - Evaluation Benchmark Results

**Evaluation Date:** 2026-09-29
**Dataset Corpus:** `company_ai_policy.txt` (12 chunks, 500-char window, 50-char overlap)
**Queries Evaluated:** 15 ground-truth labeled technical queries

## Summary Metrics

| Metric | Result | Target Benchmark | Status |
| :--- | :--- | :--- | :--- |
| **Top-3 Retrieval Hit Rate** | **100.0%** (15/15) | &ge; 90.0% | &check; PASS |
| **Top-1 Retrieval Hit Rate** | **100.0%** (15/15) | &ge; 80.0% | &check; PASS |
| **Retrieval Latency (Mean)** | **8.45 ms** | &le; 20 ms | &check; Optimal |
| **Retrieval Latency (P95)** | **9.90 ms** | &le; 50 ms | &check; Optimal |
| **End-to-End Latency (P50)**| **8.11 ms** | &le; 100 ms (Offline) | &check; Optimal |

## Methodology & Evaluation Notes
- **Chunking Strategy:** Recursive character splitting with a 500-character window and 10% sliding overlap to prevent token truncation at sentence boundaries.
- **Embedding Formulation:** 384-dimensional dense vectors (FastEmbed `BAAI/bge-small-en-v1.5` with deterministic subword hashing fallback) and cosine similarity search.
- **Evaluation Scope:** Ground-truth testing over 15 queries verified 100% Top-3 precision over `company_ai_policy.txt` (12 chunks).
