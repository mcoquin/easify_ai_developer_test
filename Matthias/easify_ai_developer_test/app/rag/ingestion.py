"""Document ingestion: load, classify, chunk, and store in ChromaDB."""

import os
import re
import logging
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Patterns that indicate a document may be outdated or unreliable
UNRELIABLE_MARKERS = [
    r"(?i)archived?\s+document",
    r"(?i)do\s+not\s+rely\s+on\s+this",
    r"(?i)experimental\s+(architecture|structure|feature)",
    r"(?i)early\s+(versions?|prototypes?)",
    r"(?i)these\s+messages\s+may\s+not\s+represent\s+final",
    r"(?i)do\s+not\s+treat\s+this\s+.*(official|specification)",
    r"(?i)random\s+notes\s+copied\s+from",
    r"(?i)internal\s+notes\s+copied\s+from",
]

# Keywords that signal the doc contains current/authoritative info
AUTHORITATIVE_MARKERS = [
    r"(?i)current\s+(rule|system|model)",
    r"(?i)invoices\s+are\s+now",
    r"(?i)the\s+current\s+system",
]


def classify_document(text: str, filename: str) -> dict:
    """Classify a document's reliability based on content and filename patterns.

    Returns metadata dict with 'doc_type' and 'reliability' fields.
    """
    metadata = {"doc_type": "standard", "reliability": "current", "contains_current_info": False}

    # Check filename patterns
    fname_lower = filename.lower()
    if "archived" in fname_lower or "historical" in fname_lower:
        metadata["doc_type"] = "archived"
        metadata["reliability"] = "outdated"
    elif "slack_dump" in fname_lower:
        metadata["doc_type"] = "informal"
        metadata["reliability"] = "informal"
    elif "engeneering_notes" in fname_lower or "engineering_notes" in fname_lower:
        metadata["doc_type"] = "informal"
        metadata["reliability"] = "informal"

    # Check content patterns
    first_500 = text[:500]
    for pattern in UNRELIABLE_MARKERS:
        if re.search(pattern, first_500):
            metadata["reliability"] = "outdated"
            break

    # Even outdated docs may contain authoritative corrections
    for pattern in AUTHORITATIVE_MARKERS:
        if re.search(pattern, text):
            metadata["contains_current_info"] = True
            break

    return metadata


def load_documents(doc_dirs: list[str]) -> list[Document]:
    """Load all text documents from the given directories.

    Supports .md, .txt, and extensionless files.
    """
    documents = []
    supported_extensions = {".md", ".txt", ""}

    for dir_path in doc_dirs:
        dir_path = Path(dir_path)
        if not dir_path.exists():
            logger.warning(f"Directory not found: {dir_path}")
            continue

        dataset_name = dir_path.name  # 'docs' or 'docs_brutal'

        for file_path in sorted(dir_path.iterdir()):
            if file_path.is_dir():
                continue

            ext = file_path.suffix
            if ext not in supported_extensions:
                continue

            try:
                text = file_path.read_text(encoding="utf-8")
            except Exception as e:
                logger.error(f"Failed to read {file_path}: {e}")
                continue

            # Classify the document
            doc_metadata = classify_document(text, file_path.name)
            doc_metadata.update(
                {
                    "source": file_path.name,
                    "source_path": str(file_path),
                    "dataset": dataset_name,
                    "file_type": ext or "none",
                }
            )

            documents.append(Document(page_content=text, metadata=doc_metadata))
            logger.info(
                f"Loaded: {file_path.name} (dataset={dataset_name}, "
                f"reliability={doc_metadata['reliability']})"
            )

    logger.info(f"Total documents loaded: {len(documents)}")
    return documents


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> list[Document]:
    """Split documents into chunks while preserving metadata.

    Uses RecursiveCharacterTextSplitter which respects paragraph/sentence boundaries.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
        add_start_index=True,
    )

    chunks = []
    for doc in documents:
        splits = splitter.split_documents([doc])
        for i, chunk in enumerate(splits):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["total_chunks"] = len(splits)
            chunks.append(chunk)

    logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
    return chunks
