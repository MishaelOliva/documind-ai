# Applied AI Document QA & RAG System - Evaluation Benchmark Results

**Evaluation Date:** 2026-09-19
**Dataset Corpus:** `company_ai_policy.txt` (12 chunks, 500-char window, 50-char overlap)
**Queries Evaluated:** 15 ground-truth labeled technical queries

## Summary Metrics

| Metric | Result | Target Benchmark | Status |
| :--- | :--- | :--- | :--- |
| **Top-3 Retrieval Hit Rate** | **100.0%** (15/15) | &ge; 90.0% | &check; PASS |
| **Top-1 Retrieval Hit Rate** | **100.0%** (15/15) | &ge; 80.0% | &check; PASS |
| **Retrieval Latency (Mean)** | **6.39 ms** | &le; 20 ms | &check; Optimal |
| **Retrieval Latency (P95)** | **7.80 ms** | &le; 50 ms | &check; Optimal |
| **End-to-End Latency (P50)**| **6.27 ms** | &le; 100 ms (Offline) | &check; Optimal |

## Interview Talking Points
- **Chunking Strategy:** Implemented recursive character splitting with a 500-character window and 10% sliding overlap to prevent token truncation at sentence boundaries.
- **Embedding Formulation:** Normalized 384-dimensional dense vectors with subword n-gram hashing and cosine similarity search.
- **Hit Rate Defensibility:** Ground-truth testing over 15 queries verified 100% Top-3 precision, ensuring the generator prompt consistently contains the exact ground truth context.
