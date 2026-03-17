"""Pydantic models for API request and response schemas."""

from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    """Incoming question from user."""

    question: str = Field(..., min_length=1, max_length=1000, description="The question to answer")
    top_k: int | None = Field(None, ge=1, le=20, description="Number of chunks to retrieve")
    dataset: str = Field("all", description="Which doc set to search: 'docs', 'docs_brutal', or 'all'")


class SourceChunk(BaseModel):
    """A retrieved document chunk used as context."""

    source: str
    content: str
    score: float | None = None
    metadata: dict = {}


class AnswerResponse(BaseModel):
    """Generated answer with source references."""

    question: str
    answer: str
    sources: list[SourceChunk]
    num_chunks_retrieved: int


class IngestResponse(BaseModel):
    """Result of document ingestion."""

    status: str
    documents_processed: int
    chunks_created: int
    collection: str
