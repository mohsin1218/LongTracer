"""
PDF Parser — Load PDF documents from a directory using PyMuPDF (fitz).

Produces LangChain Document objects with page-level metadata.
"""

import os
from pathlib import Path
from typing import List

from langchain_core.documents import Document

try:
    import fitz  # PyMuPDF
except ImportError:
    raise ImportError(
        "PyMuPDF is required for PDF parsing. "
        "Install it with: pip install pymupdf"
    )


def load_pdfs_from_directory(directory: str) -> List[Document]:
    """
    Load all PDFs from a directory, returning one Document per page.

    Each Document has:
        - page_content: extracted text from the page
        - metadata: {"source": <filepath>, "page": <0-indexed page number>}

    Args:
        directory: Path to directory containing PDF files.

    Returns:
        List of LangChain Document objects (one per page).
    """
    directory = Path(directory).resolve()
    if not directory.exists():
        raise FileNotFoundError(f"PDF directory not found: {directory}")

    documents: List[Document] = []

    pdf_files = sorted(
        f for f in directory.iterdir()
        if f.suffix.lower() == ".pdf" and f.is_file()
    )

    if not pdf_files:
        print(f"  ⚠️  No PDF files found in: {directory}")
        return documents

    for pdf_path in pdf_files:
        try:
            doc = fitz.open(str(pdf_path))
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text("text")
                if text.strip():
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": str(pdf_path),
                                "page": page_num,
                                "filename": pdf_path.name,
                                "total_pages": len(doc),
                            },
                        )
                    )
            doc.close()
            print(f"    ✓ {pdf_path.name}: {len(doc)} pages")
        except Exception as e:
            print(f"    ✗ Failed to load {pdf_path.name}: {e}")

    return documents
