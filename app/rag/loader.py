"""Document Loader: Extracts clean text and metadata from PDF and text files."""
import os
from pathlib import Path
from typing import List, Dict, Any

class DocumentLoader:
    """Loads and extracts text page-by-page from PDFs or plain text documents."""

    @staticmethod
    def load_file(file_path: Path) -> List[Dict[str, Any]]:
        """
        Loads a document and returns a list of page dicts:
        [{ 'page_number': 1, 'text': '...' }, ...]
        """
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            return DocumentLoader._load_pdf(file_path)
        elif suffix in [".txt", ".md", ".json", ".csv", ".log"]:
            return DocumentLoader._load_text(file_path)
        else:
            raise ValueError(f"Unsupported file format: '{suffix}'. Supported: .pdf, .txt, .md, .csv")

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
            raise RuntimeError(f"Error parsing PDF '{file_path.name}': {str(e)}")

        if not pages:
            raise ValueError(f"PDF '{file_path.name}' contains no readable text or is image-only.")
        return pages

    @staticmethod
    def _load_text(file_path: Path) -> List[Dict[str, Any]]:
        """Loads a plain text file as a single page."""
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            clean_text = DocumentLoader._clean_text(content)
            if not clean_text:
                raise ValueError(f"File '{file_path.name}' is empty.")
            return [{
                "page_number": 1,
                "text": clean_text
            }]
        except Exception as e:
            raise RuntimeError(f"Error reading text file '{file_path.name}': {str(e)}")

    @staticmethod
    def _clean_text(text: str) -> str:
        """Normalizes whitespaces, tabs, and multiple blank lines."""
        if not text:
            return ""
        # Normalize newlines
        lines = [line.strip() for line in text.splitlines()]
        # Filter excessive blank lines
        cleaned_lines = []
        prev_blank = False
        for line in lines:
            if line:
                cleaned_lines.append(line)
                prev_blank = False
            elif not prev_blank:
                cleaned_lines.append("")
                prev_blank = True
        return "\n".join(cleaned_lines).strip()
