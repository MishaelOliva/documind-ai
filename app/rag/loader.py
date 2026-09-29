"""Document Loader: Extracts clean text and metadata from PDF and text files."""
import logging
from pathlib import Path
from typing import List, Dict, Any

from app.config import settings

logger = logging.getLogger(__name__)

class DocumentLoader:
    """
    Loads and extracts text page-by-page from PDFs or plain text documents.

    The accepted format list comes from `settings.ALLOWED_EXTENSIONS`, which the
    API layer also uses, so the HTTP contract and the loader cannot drift apart.
    """

    @staticmethod
    def load_file(file_path: Path) -> List[Dict[str, Any]]:
        """
        Loads a document and returns a list of page dicts:
        [{ 'page_number': 1, 'text': '...' }, ...]
        """
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            return DocumentLoader._load_pdf(file_path)
        if suffix in settings.ALLOWED_EXTENSIONS:
            return DocumentLoader._load_text(file_path)
        raise ValueError(
            f"Unsupported file format: '{suffix}'. "
            f"Supported: {', '.join(settings.ALLOWED_EXTENSIONS)}"
        )

    @staticmethod
    def _load_pdf(file_path: Path) -> List[Dict[str, Any]]:
        """Extracts text page-by-page from a PDF using pypdf."""
        pages = []
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(file_path))
            for idx, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                clean_text = DocumentLoader._clean_text(text)
                if clean_text:
                    pages.append({
                        "page_number": idx + 1,
                        "text": clean_text
                    })
        except Exception as e:
            raise RuntimeError(f"Error parsing PDF '{file_path.name}': {str(e)}") from e

        if not pages:
            raise ValueError(
                f"PDF '{file_path.name}' contains no readable text or is image-only. "
                "Scanned PDFs need an OCR step before ingestion."
            )
        return pages

    @staticmethod
    def _load_text(file_path: Path) -> List[Dict[str, Any]]:
        """Loads a plain text file as a single page."""
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            raise RuntimeError(f"Error reading text file '{file_path.name}': {str(e)}") from e

        clean_text = DocumentLoader._clean_text(content)
        if not clean_text:
            raise ValueError(f"File '{file_path.name}' is empty.")
        return [{
            "page_number": 1,
            "text": clean_text
        }]

    @staticmethod
    def _clean_text(text: str) -> str:
        """Normalizes newlines, tabs, and multiple blank lines."""
        if not text:
            return ""
        lines = [line.rstrip() for line in text.replace("\r\n", "\n").split("\n")]
        cleaned_lines: List[str] = []
        prev_blank = False
        for line in lines:
            if line.strip():
                cleaned_lines.append(line)
                prev_blank = False
            elif not prev_blank:
                cleaned_lines.append("")
                prev_blank = True
        return "\n".join(cleaned_lines).strip()
