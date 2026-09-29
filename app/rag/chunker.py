"""Recursive Character Text Splitter with Sliding Window Overlap."""
from typing import List, Dict, Any
from app.models import DocumentChunk

class RecursiveChunker:
    """
    Recursively splits text on natural boundaries (paragraphs -> sentences -> words)
    to generate chunks of roughly `chunk_size` characters with `chunk_overlap`.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = ["\n\n", "\n", ". ", "; ", ", ", " "]
        self.WORD_SNAP_WINDOW = 24

    def split_text(self, text: str) -> List[str]:
        """Splits a single body of text into overlapping chunks."""
        return self._split_recursive(text, self.separators)

    def _snap_to_word_boundary(self, text: str, index: int, forward: bool) -> int:
        """
        Moves `index` to the nearest word boundary within a small window.

        Without this the hard-split path slices at an exact character offset and
        routinely produces chunks that begin or end mid-word ("ust never be
        exposed" from "must never be exposed"). Truncated leading tokens are
        noise in the embedding space, so pulling the boundary onto whitespace
        measurably improves retrieval for the same chunk budget.
        """
        if index <= 0 or index >= len(text):
            return index
        if not text[index].isalnum():
            return index  # already on a boundary

        window = min(self.WORD_SNAP_WINDOW, index, len(text) - index)
        for offset in range(1, window + 1):
            candidate = index + offset if forward else index - offset
            if not text[candidate].isalnum():
                return candidate
        return index

    def _split_recursive(self, text: str, separators: List[str]) -> List[str]:
        """Recursively breaks text using the highest-priority separator available."""
        if len(text) <= self.chunk_size:
            return [text.strip()] if text.strip() else []

        separator = ""
        next_separators = []
        for i, sep in enumerate(separators):
            if sep in text:
                separator = sep
                next_separators = separators[i + 1:]
                break

        if not separator:
            # Hard split on character limit if no separators exist, snapping each
            # boundary onto whitespace so words stay intact.
            chunks = []
            start = 0
            stride = self.chunk_size - self.chunk_overlap
            while start < len(text):
                end = min(start + self.chunk_size, len(text))
                if end < len(text):
                    end = self._snap_to_word_boundary(text, end, forward=False)
                chunk = text[start:end].strip()
                if chunk:
                    chunks.append(chunk)
                if end >= len(text):
                    break
                nxt = self._snap_to_word_boundary(text, end + self.chunk_overlap, forward=True)
                start = nxt if nxt > start else start + stride
            return chunks

        # Split on chosen separator
        splits = text.split(separator)
        chunks: List[str] = []
        current_chunk = ""

        for part in splits:
            candidate = f"{current_chunk}{separator}{part}" if current_chunk else part
            if len(candidate) <= self.chunk_size:
                current_chunk = candidate
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    # Preserve sliding overlap from the end of current_chunk,
                    # starting the carried-over text on a word boundary.
                    overlap_start = max(0, len(current_chunk) - self.chunk_overlap)
                    overlap_start = self._snap_to_word_boundary(
                        current_chunk, overlap_start, forward=True
                    )
                    current_chunk = current_chunk[overlap_start:].strip()
                    if current_chunk:
                        candidate = f"{current_chunk}{separator}{part}"
                    else:
                        candidate = part

                # If single part is still larger than chunk_size, split it further
                if len(candidate) > self.chunk_size:
                    if next_separators:
                        sub_chunks = self._split_recursive(candidate, next_separators)
                        chunks.extend(sub_chunks)
                        current_chunk = ""
                    else:
                        # Character slice
                        chunks.append(candidate[:self.chunk_size].strip())
                        current_chunk = candidate[self.chunk_size - self.chunk_overlap:].strip()
                else:
                    current_chunk = candidate

        if current_chunk and current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def chunk_document(
        self,
        doc_id: str,
        source_name: str,
        pages: List[Dict[str, Any]]
    ) -> List[DocumentChunk]:
        """
        Chunks an entire loaded document page-by-page and returns typed DocumentChunk objects.
        """
        all_chunks: List[DocumentChunk] = []
        global_chunk_idx = 0

        for page_data in pages:
            page_num = page_data.get("page_number", 1)
            page_text = page_data.get("text", "")
            raw_chunks = self.split_text(page_text)

            for raw_chunk in raw_chunks:
                if not raw_chunk:
                    continue
                chunk_id = f"{doc_id}_p{page_num}_c{global_chunk_idx}"
                chunk = DocumentChunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    source_name=source_name,
                    page_number=page_num,
                    chunk_index=global_chunk_idx,
                    content=raw_chunk,
                    char_count=len(raw_chunk),
                    metadata={
                        "source": source_name,
                        "page": page_num,
                        "index": global_chunk_idx
                    }
                )
                all_chunks.append(chunk)
                global_chunk_idx += 1

        return all_chunks
