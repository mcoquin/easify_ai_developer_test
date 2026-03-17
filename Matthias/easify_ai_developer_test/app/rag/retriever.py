"""Retriever: ChromaDB vector store + BM25 hybrid search with RRF."""

import logging
import re
from pathlib import Path

import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from app.config import get_settings

logger = logging.getLogger(__name__)

_vectorstore: Chroma | None = None
_embeddings: HuggingFaceEmbeddings | None = None
_bm25_index: BM25Okapi | None = None
_bm25_chunks: list[Document] = []


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer for BM25."""
    return re.findall(r"\w+", text.lower())


def get_embeddings() -> HuggingFaceEmbeddings:
    """Get or create the embedding model (singleton)."""
    global _embeddings
    if _embeddings is None:
        settings = get_settings()
        logger.info(f"Loading embedding model: {settings.embedding_model}")
        _embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def get_vectorstore(collection_name: str | None = None) -> Chroma:
    """Get or create the ChromaDB vector store (singleton)."""
    global _vectorstore
    settings = get_settings()
    name = collection_name or settings.chroma_collection

    if _vectorstore is None or (collection_name and collection_name != settings.chroma_collection):
        persist_dir = settings.chroma_persist_dir
        Path(persist_dir).mkdir(parents=True, exist_ok=True)

        client = chromadb.PersistentClient(path=persist_dir)
        embeddings = get_embeddings()

        _vectorstore = Chroma(
            client=client,
            collection_name=name,
            embedding_function=embeddings,
        )
        logger.info(f"ChromaDB initialized: collection='{name}', persist='{persist_dir}'")

    return _vectorstore


def build_bm25_index(chunks: list[Document]) -> None:
    """Build a BM25 index from document chunks for keyword search."""
    global _bm25_index, _bm25_chunks
    _bm25_chunks = chunks
    tokenized = [_tokenize(c.page_content) for c in chunks]
    _bm25_index = BM25Okapi(tokenized)
    logger.info(f"BM25 index built with {len(chunks)} chunks")


def ingest_chunks(chunks: list[Document], collection_name: str | None = None) -> int:
    """Add document chunks to the vector store and build BM25 index.

    Returns the number of chunks stored.
    """
    if not chunks:
        return 0

    store = get_vectorstore(collection_name)

    # ChromaDB doesn't accept non-string metadata values well, so sanitize
    sanitized_chunks = []
    for chunk in chunks:
        clean_meta = {}
        for k, v in chunk.metadata.items():
            if isinstance(v, bool):
                clean_meta[k] = str(v)
            elif isinstance(v, (str, int, float)):
                clean_meta[k] = v
            else:
                clean_meta[k] = str(v)
        sanitized_chunks.append(Document(page_content=chunk.page_content, metadata=clean_meta))

    store.add_documents(sanitized_chunks)

    # Build BM25 index for hybrid search
    build_bm25_index(sanitized_chunks)

    logger.info(f"Ingested {len(sanitized_chunks)} chunks into '{store._collection.name}'")
    return len(sanitized_chunks)


def _bm25_search(query: str, top_k: int, dataset_filter: str | None = None) -> list[tuple[Document, float]]:
    """BM25 keyword search over ingested chunks."""
    global _bm25_index, _bm25_chunks

    if _bm25_index is None or not _bm25_chunks:
        logger.warning("BM25 index not built — falling back to empty results")
        return []

    tokens = _tokenize(query)
    scores = _bm25_index.get_scores(tokens)

    # Pair chunks with scores and filter by dataset
    scored = []
    for i, (chunk, score) in enumerate(zip(_bm25_chunks, scores)):
        if dataset_filter and dataset_filter != "all":
            if chunk.metadata.get("dataset") != dataset_filter:
                continue
        scored.append((chunk, float(score)))

    # Sort by score descending, take top_k
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def _reciprocal_rank_fusion(
    vector_results: list[tuple[Document, float]],
    bm25_results: list[tuple[Document, float]],
    vector_weight: float = 0.7,
    bm25_weight: float = 0.3,
    k: int = 60,
) -> list[Document]:
    """Combine vector and BM25 results using Reciprocal Rank Fusion.

    RRF score = weight * (1 / (k + rank))
    """
    doc_scores: dict[str, tuple[Document, float]] = {}

    def _doc_key(doc: Document) -> str:
        return f"{doc.metadata.get('source', '')}:{doc.page_content[:100]}"

    for rank, (doc, _score) in enumerate(vector_results):
        key = _doc_key(doc)
        rrf = vector_weight * (1.0 / (k + rank + 1))
        if key in doc_scores:
            doc_scores[key] = (doc, doc_scores[key][1] + rrf)
        else:
            doc_scores[key] = (doc, rrf)

    for rank, (doc, _score) in enumerate(bm25_results):
        key = _doc_key(doc)
        rrf = bm25_weight * (1.0 / (k + rank + 1))
        if key in doc_scores:
            doc_scores[key] = (doc, doc_scores[key][1] + rrf)
        else:
            doc_scores[key] = (doc, rrf)

    # Sort by fused score descending
    fused = sorted(doc_scores.values(), key=lambda x: x[1], reverse=True)

    # Attach RRF score to metadata
    docs = []
    for doc, score in fused:
        doc.metadata["relevance_score"] = round(score, 6)
        docs.append(doc)

    return docs


def retrieve(
    query: str,
    top_k: int | None = None,
    dataset_filter: str | None = None,
) -> list[Document]:
    """Retrieve relevant chunks using hybrid search (vector + BM25) or vector-only.

    Args:
        query: The user question.
        top_k: Number of final results to return.
        dataset_filter: Optional filter by dataset name ('docs' or 'docs_brutal').

    Returns:
        List of Documents sorted by relevance.
    """
    settings = get_settings()
    k = top_k or settings.top_k
    store = get_vectorstore()

    # Build optional where filter for vector search
    where_filter = None
    if dataset_filter and dataset_filter != "all":
        where_filter = {"dataset": dataset_filter}

    if settings.use_hybrid_search and _bm25_index is not None:
        # Fetch more candidates for fusion
        fetch_k = settings.retrieval_top_k

        vector_results = store.similarity_search_with_relevance_scores(
            query, k=fetch_k, filter=where_filter,
        )
        bm25_results = _bm25_search(query, top_k=fetch_k, dataset_filter=dataset_filter)

        docs = _reciprocal_rank_fusion(
            vector_results=vector_results,
            bm25_results=bm25_results,
            vector_weight=settings.vector_weight,
            bm25_weight=settings.bm25_weight,
        )

        logger.info(
            f"Hybrid search: {len(vector_results)} vector + {len(bm25_results)} BM25 "
            f"→ {len(docs)} fused results for: '{query[:60]}'"
        )
        return docs[:k]

    else:
        # Vector-only fallback
        results = store.similarity_search_with_relevance_scores(
            query, k=k, filter=where_filter,
        )
        docs = []
        for doc, score in results:
            doc.metadata["relevance_score"] = round(score, 4)
            docs.append(doc)

        logger.info(f"Vector search: {len(docs)} chunks for: '{query[:60]}'")
        return docs


def clear_collection(collection_name: str | None = None) -> None:
    """Delete all documents from the collection and clear BM25 index."""
    global _vectorstore, _bm25_index, _bm25_chunks
    settings = get_settings()
    name = collection_name or settings.chroma_collection
    persist_dir = settings.chroma_persist_dir

    client = chromadb.PersistentClient(path=persist_dir)
    try:
        client.delete_collection(name)
        logger.info(f"Deleted collection: {name}")
    except Exception:
        logger.info(f"Collection '{name}' did not exist")

    _vectorstore = None
    _bm25_index = None
    _bm25_chunks = []
