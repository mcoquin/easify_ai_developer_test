"""Re-ranker: cross-encoder model to re-score retrieved chunks against the query."""

import logging
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

from app.config import get_settings

logger = logging.getLogger(__name__)

_reranker: CrossEncoder | None = None


def get_reranker() -> CrossEncoder:
    """Get or create the cross-encoder re-ranker (singleton)."""
    global _reranker
    if _reranker is None:
        settings = get_settings()
        logger.info(f"Loading re-ranker model: {settings.reranker_model}")
        _reranker = CrossEncoder(settings.reranker_model, max_length=512)
    return _reranker


def rerank(query: str, documents: list[Document], top_k: int | None = None) -> list[Document]:
    """Re-rank documents using a cross-encoder model.

    The cross-encoder scores each (query, chunk) pair directly, which is
    more accurate than cosine similarity on independent embeddings.

    Args:
        query: The user question.
        documents: Retrieved candidate chunks.
        top_k: Number of results to return after re-ranking.

    Returns:
        Re-ranked list of Documents with updated relevance scores.
    """
    if not documents:
        return []

    settings = get_settings()
    k = top_k or settings.reranker_top_k
    model = get_reranker()

    # Build query-document pairs for the cross-encoder
    pairs = [(query, doc.page_content) for doc in documents]
    scores = model.predict(pairs)

    # Pair documents with their cross-encoder scores
    scored_docs = list(zip(documents, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    # Attach re-ranker score and return top-k
    results = []
    for doc, score in scored_docs[:k]:
        doc.metadata["reranker_score"] = round(float(score), 4)
        results.append(doc)

    logger.info(
        f"Re-ranked {len(documents)} → {len(results)} chunks. "
        f"Top score: {results[0].metadata['reranker_score'] if results else 'n/a'}"
    )
    return results
