"""Application configuration loaded from environment variables."""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """RAG system settings."""

    anthropic_api_key: str = ""

    # Model
    claude_model: str = "claude-sonnet-4-20250514"
    embedding_model: str = "all-MiniLM-L6-v2"

    # ChromaDB
    chroma_persist_dir: str = "./chroma_db"
    chroma_collection: str = "easify_docs"

    # RAG parameters
    chunk_size: int = 500
    chunk_overlap: int = 100
    top_k: int = 5

    # Hybrid search & re-ranking
    use_hybrid_search: bool = True
    bm25_weight: float = 0.3
    vector_weight: float = 0.7
    use_reranker: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_top_k: int = 5
    retrieval_top_k: int = 15  # fetch more candidates before re-ranking

    # Document directories (comma-separated string)
    doc_dirs: str = "./docs,./docs_brutal"

    model_config = {
        "extra": "ignore",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }

    @property
    def document_directories(self) -> list[str]:
        return [d.strip() for d in self.doc_dirs.split(",")]


@lru_cache
def get_settings() -> Settings:
    return Settings()
