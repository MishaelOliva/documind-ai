"""
Retrieval & Grounding Evaluation Harness for DocuMind.

Reports four things the previous version did not, each of which changes how the
headline numbers should be read:

1. A random-retrieval baseline. A hit rate is uninterpretable without a control;
   with only 12 chunks, chance already scores well at top-3.
2. Passage-level accuracy from the `section` label in the ground truth, rather
   than substring matching against a bag of keywords. Keyword matching is so
   loose that dropping one gold keyword still leaves every query "matchable".
3. Embedding latency and search latency separately. Their sum is dominated by
   transformer inference, so reporting the sum as "vector retrieval latency"
   overstates the cost of the search by more than an order of magnitude.
4. Off-topic refusal behaviour against the similarity floor, including the
   measured weakness of the policy's 0.35 threshold.

Usage:
    python eval/benchmark.py            # print report
    python eval/benchmark.py --markdown # also write eval/BENCHMARK.md
"""
import argparse
import json
import random
import re
import statistics
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.rag.pipeline import RAGPipeline
from app.rag.vector_store import VectorStore
from app.config import settings

# Questions deliberately outside the corpus domain, used to measure whether the
# similarity floor actually refuses to answer.
OFF_TOPIC_QUESTIONS = [
    "What is the airspeed velocity of an unladen swallow in Patagonia?",
    "How do I bake sourdough bread at home?",
    "Who won the 1998 FIFA World Cup final?",
    "What is the boiling point of water at sea level?",
    "Recommend a good book for learning quantum mechanics.",
]

_SUBSECTION = re.compile(r"^\s*(\d+\.\d+)\s+\S", re.M)


def build_section_spans(raw_text: str) -> list[tuple[str, int, int]]:
    """Maps '4.2' to its (start, end) character span in the source document."""
    marks = [(m.start(), m.group(1)) for m in _SUBSECTION.finditer(raw_text)]
    spans = []
    for i, (pos, num) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(raw_text)
        spans.append((num, pos, end))
    return spans


def _collapse_with_index(raw_text: str) -> tuple[str, list[int]]:
    """
    Collapses whitespace runs while keeping an exact collapsed->raw index map.

    Chunk text has been whitespace-normalised by the loader and re-joined by the
    chunker, so chunk offsets do not line up with the source document. Searching a
    whitespace-collapsed copy fixes that, but only if the offset can be converted
    back precisely. A ratio-based estimate is wrong because whitespace is not
    distributed evenly through the document, so the map is built explicitly.
    """
    chars: list[str] = []
    index_map: list[int] = []
    prev_was_space = False
    for i, ch in enumerate(raw_text):
        if ch.isspace():
            if prev_was_space:
                continue
            chars.append(" ")
            index_map.append(i)
            prev_was_space = True
        else:
            chars.append(ch)
            index_map.append(i)
            prev_was_space = False
    return "".join(chars), index_map


def map_chunks_to_sections(chunks: dict, raw_text: str, spans: list) -> dict[str, set[str]]:
    """
    Assigns each chunk every subsection its text overlaps.

    Chunks are ~450 characters and subsections are ~300-400, so a chunk routinely
    straddles a boundary. Scoring only the section containing the chunk's first
    character would penalise a chunk for starting a few words early even though it
    carries the answer, so a chunk is credited with any section it overlaps at all.
    """
    collapsed_doc, index_map = _collapse_with_index(raw_text)
    mapping: dict[str, set[str]] = {}
    for chunk_id, chunk in chunks.items():
        # Must be collapsed the same way as the document: runs of whitespace
        # become a single space, not nothing.
        flat = re.sub(r"\s+", " ", chunk.content).strip()
        head = flat[:60]
        tail = flat[-60:] if len(flat) > 60 else flat
        start = collapsed_doc.find(head) if head else -1
        if start == -1:
            mapping[chunk_id] = set()
            continue
        tail_pos = collapsed_doc.find(tail, start)
        end_idx = tail_pos + len(tail) if tail_pos != -1 else start + len(head)
        if end_idx >= len(index_map):
            end_idx = len(index_map) - 1
        raw_start, raw_end = index_map[start], index_map[end_idx]
        mapping[chunk_id] = {num for num, s, e in spans if s < raw_end and raw_start < e}
    return mapping


def percentile(values: list[float], pct: float) -> float:
    """Linear-interpolation percentile, matching numpy's default."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    k = (len(ordered) - 1) * (pct / 100.0)
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def run_benchmark(seed: int = 0) -> dict:
    sample_doc = BASE_DIR / "sample_docs" / "company_ai_policy.txt"
    ground_truth_path = BASE_DIR / "eval" / "ground_truth.json"
    raw_text = sample_doc.read_text(encoding="utf-8")
    spans = build_section_spans(raw_text)

    print("=" * 78)
    print("  DOCUMIND - RETRIEVAL & GROUNDING EVALUATION")
    print("=" * 78)
    print(f"Corpus        : {sample_doc.name}")
    print("Embedder      : ", end="")

    pipeline = RAGPipeline()
    # Keep the evaluation hermetic: never read or write the developer's index.
    pipeline.vector_store = VectorStore(
        persist_path=BASE_DIR / "data" / "vector_db" / "benchmark_scratch.json"
    )
    print(pipeline.embedder.backend_name)
    print(f"Similarity floor: {settings.MIN_SIMILARITY_SCORE}")

    ingest = pipeline.ingest_document(sample_doc, sample_doc.name)
    chunks = pipeline.vector_store.chunks
    chunk_ids = list(chunks.keys())
    print(f"Indexed       : {ingest.document.chunk_count} chunks, "
          f"{ingest.document.char_count} chars")
    print(f"Ground truth  : {ground_truth_path.name}\n")

    chunk_sections = map_chunks_to_sections(chunks, raw_text, spans)
    ground_truth = json.loads(ground_truth_path.read_text(encoding="utf-8"))
    n = len(ground_truth)

    # ---------------------------------------------------------------- in-domain
    print("-" * 78)
    print(f"[1/4] In-domain queries ({n} labeled questions)")
    print("-" * 78)
    header = (f"{'ID':<5} {'Ground':<7} {'Sect@1':<7} {'Sect@3':<7} "
              f"{'Embed':<8} {'Search':<8} {'TopSim':<8}")
    print(header)
    print("-" * 78)

    kw_top1 = passage_top1 = passage_top3 = 0
    embed_ms, search_ms, total_ms, sims = [], [], [], []

    for item in ground_truth:
        res = pipeline.query(question=item["question"], top_k=3, override_provider="mock")
        embed_ms.append(res.embedding_latency_ms)
        search_ms.append(res.search_latency_ms)
        total_ms.append(res.total_latency_ms)
        sims.append(res.top_similarity)

        # (a) loose keyword-any criterion, as previously reported
        top1_text = chunks[res.citations[0].chunk_id].content.lower()
        if any(kw.lower() in top1_text for kw in item["expected_keywords"]):
            kw_top1 += 1

        # (b) strict passage-level criterion using the ignored `section` label
        want = item["section"].replace("Section ", "")
        got1 = chunk_sections.get(res.citations[0].chunk_id, set())
        got3 = {s for c in res.citations for s in chunk_sections.get(c.chunk_id, set())}
        hit1, hit3 = want in got1, want in got3
        passage_top1 += hit1
        passage_top3 += hit3

        print(f"{item['id']:<5} {str(res.is_grounded):<7} {('HIT' if hit1 else 'miss'):<7} "
              f"{('HIT' if hit3 else 'miss'):<7} {res.embedding_latency_ms:<8.2f} "
              f"{res.search_latency_ms:<8.3f} {res.top_similarity:<8.4f}")

    # ------------------------------------------------------------- baseline
    print("\n" + "-" * 78)
    print("[2/4] Random-retrieval baseline (the control a hit rate needs)")
    print("-" * 78)
    rng = random.Random(seed)
    base_top1 = base_top3 = 0
    for item in ground_truth:
        for k, acc in ((1, "top1"), (3, "top3")):
            picked = rng.sample(chunk_ids, k)
            text = " ".join(chunks[c].content.lower() for c in picked)
            hit = any(kw.lower() in text for kw in item["expected_keywords"])
            if k == 1 and hit:
                base_top1 += 1
            elif k == 3 and hit:
                base_top3 += 1
    print(f"  random top-1 (keyword-any) : {base_top1/n*100:6.1f}%  ({base_top1}/{n})")
    print(f"  random top-3 (keyword-any) : {base_top3/n*100:6.1f}%  ({base_top3}/{n})")
    print(f"  model  top-1 (keyword-any) : {kw_top1/n*100:6.1f}%  ({kw_top1}/{n})")
    print("  -> top-1 is the discriminating metric; top-3 is near-saturated by chance")

    # -------------------------------------------------------- off-topic refusal
    print("\n" + "-" * 78)
    print(f"[3/4] Off-topic refusal (floor = {settings.MIN_SIMILARITY_SCORE})")
    print("-" * 78)
    refused = 0
    off_sims = []
    for q in OFF_TOPIC_QUESTIONS:
        res = pipeline.query(question=q, top_k=3, override_provider="mock")
        off_sims.append(res.top_similarity)
        refused += not res.is_grounded
        print(f"  {'REFUSED' if not res.is_grounded else 'answered'} "
              f"(sim {res.top_similarity:.4f})  {q[:46]}")
    print(f"\n  off-topic refusal rate     : {refused/len(OFF_TOPIC_QUESTIONS)*100:6.1f}%")
    print(f"  in-domain mean top-1 sim  : {statistics.mean(sims):6.4f}")
    print(f"  off-topic  mean top-1 sim : {statistics.mean(off_sims):6.4f}")
    print("  -> a single absolute floor cannot separate these two distributions;")
    print("     see the Limitations section of the README.")

    # ------------------------------------------------------------------ summary
    print("\n" + "=" * 78)
    print("[4/4] Summary")
    print("=" * 78)
    print(f"  Passage-level top-1 (section label) : {passage_top1/n*100:6.1f}%  ({passage_top1}/{n})")
    print(f"  Passage-level top-3 (section label) : {passage_top3/n*100:6.1f}%  ({passage_top3}/{n})")
    print(f"  Keyword-any    top-1 (legacy)       : {kw_top1/n*100:6.1f}%  ({kw_top1}/{n})")
    print(f"  Embedding latency  mean {statistics.mean(embed_ms):6.2f} ms "
          f"| P95 {percentile(embed_ms, 95):6.2f} ms")
    print(f"  Search latency     mean {statistics.mean(search_ms):6.3f} ms "
          f"| P95 {percentile(search_ms, 95):6.3f} ms")
    print(f"  End-to-end (offline, mock generator) mean {statistics.mean(total_ms):6.2f} ms")
    print("  Note: generation is the deterministic offline extractor, not an LLM.")

    return {
        "date": time.strftime("%Y-%m-%d"),
        "chunk_count": ingest.document.chunk_count,
        "char_count": ingest.document.char_count,
        "n_queries": n,
        "passage_top1": passage_top1,
        "passage_top3": passage_top3,
        "kw_top1": kw_top1,
        "random_top1": base_top1,
        "random_top3": base_top3,
        "embed_mean": statistics.mean(embed_ms),
        "embed_p95": percentile(embed_ms, 95),
        "search_mean": statistics.mean(search_ms),
        "search_p95": percentile(search_ms, 95),
        "e2e_mean": statistics.mean(total_ms),
        "offtopic_refusal_pct": refused / len(OFF_TOPIC_QUESTIONS) * 100,
        "indomain_mean_sim": statistics.mean(sims),
        "offtopic_mean_sim": statistics.mean(off_sims),
        "embedder": pipeline.embedder.backend_name,
    }


def write_markdown(m: dict) -> None:
    n = m["n_queries"]
    out = f"""# DocuMind - Evaluation Results

Generated by `python eval/benchmark.py`. This file is a build artifact and is
not tracked; the canonical numbers live in the README.

**Date:** {m['date']}

**Corpus:** `company_ai_policy.txt` ({m['chunk_count']} chunks, {m['char_count']} chars)

**Embedder:** `{m['embedder']}`

**Queries:** {n} labeled questions

## Retrieval accuracy

| Metric | Result | Random baseline | Notes |
| :--- | :--- | :--- | :--- |
| Passage-level top-1 | {m['passage_top1']/n*100:.1f}% ({m['passage_top1']}/{n}) | {m['random_top1']/n*100:.1f}% | Scored on the `section` label |
| Passage-level top-3 | {m['passage_top3']/n*100:.1f}% ({m['passage_top3']}/{n}) | {m['random_top3']/n*100:.1f}% | Scored on the `section` label |
| Keyword-any top-1 | {m['kw_top1']/n*100:.1f}% ({m['kw_top1']}/{n}) | {m['random_top1']/n*100:.1f}% | Legacy, deliberately loose criterion |

Top-1 is the metric that discriminates. At top-3 the 12-chunk corpus is close to
saturated by chance, so a high top-3 score on its own says very little.

## Latency

| Stage | Mean | P95 |
| :--- | :--- | :--- |
| Query embedding (ONNX transformer) | {m['embed_mean']:.2f} ms | {m['embed_p95']:.2f} ms |
| Vector search (matrix-vector product) | {m['search_mean']:.3f} ms | {m['search_p95']:.3f} ms |
| End-to-end, offline mock generator | {m['e2e_mean']:.2f} ms | - |

Embedding dominates by roughly two orders of magnitude. Reporting only their sum
as "retrieval latency" would overstate search cost, so the two are reported
separately. End-to-end figures use the deterministic offline extractor; a
hosted LLM adds 0.5-5 s of network and generation time.

## Grounding

| Metric | Result |
| :--- | :--- |
| Off-topic refusal rate @ floor | {m['offtopic_refusal_pct']:.1f}% |
| In-domain mean top-1 similarity | {m['indomain_mean_sim']:.4f} |
| Off-topic mean top-1 similarity | {m['offtopic_mean_sim']:.4f} |

The in-domain and off-topic similarity distributions overlap, so the policy's
0.35 floor alone does not reject out-of-domain questions. See README
Limitations for the analysis and the recommended remedies.
"""
    target = BASE_DIR / "eval" / "BENCHMARK.md"
    target.write_text(out, encoding="utf-8")
    print(f"\nWrote {target.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markdown", action="store_true",
                        help="also write eval/BENCHMARK.md")
    parser.add_argument("--seed", type=int, default=0, help="baseline sampling seed")
    args = parser.parse_args()
    results = run_benchmark(seed=args.seed)
    if args.markdown:
        write_markdown(results)
