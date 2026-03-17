"""RAG chain: retrieve context and generate grounded answers with Claude."""

import logging
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from pydantic import SecretStr

from app.config import get_settings
from app.rag.retriever import retrieve
from app.rag.reranker import rerank
from app.schemas import AnswerResponse, SourceChunk

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are an AI assistant for the Easify construction software platform.
Your job is to answer questions **exclusively** using the provided documentation context.

Rules:
1. ONLY use information from the context below to answer.
2. If the context does not contain enough information, say:
   "The provided documentation does not contain information about this topic."
3. When documents disagree, prefer documents marked as "current" reliability over
   "outdated" or "informal" ones. Mention if older docs say something different.
4. Reference which source document(s) support your answer.
5. Be precise and concise. Do not invent information.
6. If a concept was deprecated or removed, explain that clearly.

Context from internal documentation:
---
{context}
---
"""

USER_PROMPT = "Question: {question}"


def format_context(docs: list[Document]) -> str:
    """Format retrieved documents into a context string for the LLM."""
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        reliability = doc.metadata.get("reliability", "unknown")
        dataset = doc.metadata.get("dataset", "unknown")
        score = doc.metadata.get("relevance_score", "n/a")

        header = (
            f"[Source {i}: {source} | dataset: {dataset} | "
            f"reliability: {reliability} | relevance: {score}]"
        )
        parts.append(f"{header}\n{doc.page_content}")

    return "\n\n---\n\n".join(parts)


def get_llm() -> ChatAnthropic:
    """Create the Claude LLM instance."""
    settings = get_settings()
    return ChatAnthropic(
        model_name=settings.claude_model,
        api_key=SecretStr(settings.anthropic_api_key),
        temperature=0.0,
        max_tokens_to_sample=1024,
        timeout=None,
        stop=None,
    )


def answer_question(
    question: str,
    top_k: int | None = None,
    dataset: str = "all",
) -> AnswerResponse:
    """Full RAG pipeline: retrieve context → re-rank → generate answer.

    Args:
        question: User's question.
        top_k: Number of chunks to retrieve.
        dataset: Filter by dataset ('docs', 'docs_brutal', or 'all').

    Returns:
        AnswerResponse with the answer and source references.
    """
    # 1. Retrieve relevant chunks (hybrid search fetches extra candidates)
    settings = get_settings()
    fetch_k = settings.retrieval_top_k if settings.use_reranker else top_k
    docs = retrieve(query=question, top_k=fetch_k, dataset_filter=dataset)

    if not docs:
        return AnswerResponse(
            question=question,
            answer="The provided documentation does not contain information about this topic.",
            sources=[],
            num_chunks_retrieved=0,
        )

    # 1.5. Re-rank if enabled
    if settings.use_reranker:
        docs = rerank(query=question, documents=docs, top_k=top_k or settings.reranker_top_k)

    # 2. Format context
    context = format_context(docs)

    # 3. Build prompt and generate answer
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", USER_PROMPT),
        ]
    )

    llm = get_llm()
    chain = prompt | llm | StrOutputParser()

    logger.info(f"Generating answer for: '{question[:80]}'")
    answer = chain.invoke({"context": context, "question": question})

    # 4. Build source references — deduplicate by source + dataset combo
    sources = []
    seen = set()
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        dataset_label = doc.metadata.get("dataset", "")
        dedup_key = f"{dataset_label}:{source}:{doc.page_content[:150]}"
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        sources.append(
            SourceChunk(
                source=f"{source} ({dataset_label})",
                content=doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else ""),
                score=doc.metadata.get("relevance_score"),
                metadata={
                    "dataset": dataset_label,
                    "reliability": doc.metadata.get("reliability", ""),
                    "doc_type": doc.metadata.get("doc_type", ""),
                },
            )
        )

    return AnswerResponse(
        question=question,
        answer=answer,
        sources=sources,
        num_chunks_retrieved=len(docs),
    )