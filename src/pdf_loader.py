"""
PDF text extraction and intelligent chunking for Agentic AI eBook.
Supports chapter detection, metadata enrichment, and recursive chunking.
"""

import re
from typing import List, Dict, Any
from pathlib import Path
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import settings

CHAPTER_TITLES = [
    "Introduction to Agentic AI",
    "Anatomy of an Agentic AI System",
    "Multi-Agent Systems",
    "Orchestrating Agentic AI Systems",
    "Your Readiness for Agentic AI",
    "Practical Applications of Agentic AI",
    "The Terminology Maze",
    "Understanding the Shift from Reactive to Proactive",
    "How Agentic AI Stands Apart",
    "Capabilities of Agentic AI",
    "The Core Pillars: From Perception to Execution",
    "Understanding the Building Blocks",
    "Defining Characteristics of an Agent",
    "Categories and Types of Agentic Systems",
    "How Agentic AI Shapes Industries",
    "MAS Scenario: A Supply Chain in Crisis",
    "Challenges and Mitigation Strategies",
    "Key Aspects of Orchestrating Multi-Agent Systems",
    "Planning and Task Breakdown",
    "Organizational Maturity & Readiness"
]

def clean_text(text: str) -> str:
    """Clean redundant spaces and normalize unicode quotes and hyphens."""
    if not text:
        return ""
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u2013', '-').replace('\u2014', '--')
    text = text.replace('\ufeff', '')
    text = re.sub(r'[\u0562\uf0b7\u2022]', '•', text)
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def detect_chapter(page_text: str, current_chapter: str) -> str:
    """Detect if page starts or contains a major section heading."""
    first_lines = [l.strip() for l in page_text.splitlines() if l.strip()][:5]
    for line in first_lines:
        for title in CHAPTER_TITLES:
            if title.lower() in line.lower():
                return title
        m = re.match(r'^(\d+\.\d+\s+[A-Za-z0-9\s:,\-]+)', line)
        if m:
            return m.group(1).strip()
    return current_chapter

class PDFDocumentLoader:
    def __init__(self, pdf_path: str = None, chunk_size: int = None, chunk_overlap: int = None):
        self.pdf_path = pdf_path or settings.PDF_PATH
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", "; ", " ", ""]
        )

    def load_and_chunk(self) -> List[Dict[str, Any]]:
        """
        Reads the PDF, extracts text per page, associates chapter metadata,
        and produces chunked documents with rich metadata.
        """
        pdf_file = Path(self.pdf_path)
        if not pdf_file.exists():
            raise FileNotFoundError(f"PDF file not found at: {self.pdf_path}")

        doc = fitz.open(str(pdf_file))
        raw_pages = []
        current_chapter = "Agentic AI Overview"

        for page_idx in range(len(doc)):
            page_num = page_idx + 1
            page = doc[page_idx]
            raw_text = page.get_text()
            cleaned = clean_text(raw_text)
            
            if not cleaned or len(cleaned) < 20:
                # Skip blank or cover-only graphic pages
                continue

            current_chapter = detect_chapter(cleaned, current_chapter)
            raw_pages.append({
                "page_num": page_num,
                "chapter": current_chapter,
                "text": cleaned
            })

        doc.close()

        # Chunk pages while preserving metadata
        chunks = []
        chunk_counter = 0

        for page_info in raw_pages:
            page_num = page_info["page_num"]
            chapter = page_info["chapter"]
            page_text = page_info["text"]

            page_chunks = self.splitter.split_text(page_text)
            for idx, chunk_text in enumerate(page_chunks):
                chunk_counter += 1
                chunk_id = f"chunk_p{page_num}_{idx+1}"
                chunks.append({
                    "id": chunk_id,
                    "index": chunk_counter,
                    "text": chunk_text,
                    "page": page_num,
                    "chapter": chapter,
                    "source": "Ebook-Agentic-AI.pdf",
                    "char_count": len(chunk_text)
                })

        return chunks
