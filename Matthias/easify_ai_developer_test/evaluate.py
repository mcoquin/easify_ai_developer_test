#!/usr/bin/env python3
"""Evaluation harness for the Easify RAG Assistant.

Tests retrieval quality and answer accuracy against known Q&A pairs.

Usage:
    python evaluate.py              # Run all tests
    python evaluate.py --verbose    # Show full answers
    python evaluate.py --retrieval  # Test retrieval only (no LLM calls)
"""

import argparse
import logging
import time
import sys

from app.config import get_settings
from app.rag.ingestion import load_documents, chunk_documents
from app.rag.retriever import ingest_chunks, clear_collection, get_vectorstore, retrieve
from app.rag.reranker import rerank
from app.rag.chain import answer_question

logging.basicConfig(level=logging.WARNING)

# ─── Test Cases ────────────────────────────────────────────────────────────────
# Each test has:
#   - question: the input query
#   - expected_keywords: words/phrases that SHOULD appear in a correct answer
#   - negative_keywords: words/phrases that should NOT appear (hallucination check)
#   - expected_sources: filenames that should be retrieved
#   - category: grouping for reporting

TEST_CASES = [
    {
        "question": "What is the difference between a block and a unit?",
        "expected_keywords": ["block", "unit", "sellable", "structural", "financial"],
        "negative_keywords": [],
        "expected_sources": ["project_model.md", "project_overview.md"],
        "category": "project_structure",
    },
    {
        "question": "Can invoices exist on blocks?",
        "expected_keywords": ["no", "unit", "cannot", "block"],
        "negative_keywords": [],
        "expected_sources": ["finance_notes.md", "finance_rules.txt", "project_model.md"],
        "category": "invoicing",
    },
    {
        "question": "Who can approve supplier proposals?",
        "expected_keywords": ["project manager"],
        "negative_keywords": ["customer can approve"],
        "expected_sources": ["permissions_internal.txt", "permission_matrix.txt"],
        "category": "permissions",
    },
    {
        "question": "What happens when costs are approved after invoicing is completed?",
        "expected_keywords": ["warning", "separate invoice", "new"],
        "negative_keywords": [],
        "expected_sources": ["invoicing_edge_cases.md"],
        "category": "invoicing",
    },
    {
        "question": "What are addendums used for?",
        "expected_keywords": ["additional work", "cost"],
        "negative_keywords": [],
        "expected_sources": ["supplier_contracts.md", "supplier_workflow"],
        "category": "supplier",
    },
    {
        "question": "Which roles can modify financial configuration?",
        "expected_keywords": ["administrator"],
        "negative_keywords": [],
        "expected_sources": ["permissions_internal.txt", "permission_matrix.txt"],
        "category": "permissions",
    },
    {
        "question": "How does the platform integrate with Stripe payments?",
        "expected_keywords": ["not", "documentation does not"],
        "negative_keywords": ["stripe api", "webhook", "payment intent"],
        "category": "negative_test",
        "expected_sources": [],
    },
    {
        "question": "What is the margin formula?",
        "expected_keywords": ["revenue", "supplier_cost"],
        "negative_keywords": [],
        "expected_sources": ["cost_tracking.md"],
        "category": "finance",
    },
    {
        "question": "What are the invoice states?",
        "expected_keywords": ["draft", "pending", "approved", "sent", "paid"],
        "negative_keywords": [],
        "expected_sources": ["finance_notes.md", "finance_rules.txt"],
        "category": "invoicing",
    },
    {
        "question": "What can customers do in the platform?",
        "expected_keywords": ["view", "approve design", "sign contract"],
        "negative_keywords": ["customers can approve invoices", "customers can modify financial"],
        "expected_sources": ["permissions_internal.txt", "permission_matrix.txt"],
        "category": "permissions",
    },
    {
        "question": "Was block-level invoicing ever supported?",
        "expected_keywords": ["experimental", "abandoned", "deprecated", "removed"],
        "negative_keywords": [],
        "expected_sources": ["archived_specs_2019.md", "historical_changes.md"],
        "category": "historical",
    },
    {
        "question": "What information does a supplier proposal contain?",
        "expected_keywords": ["cost", "timeline", "work item"],
        "negative_keywords": [],
        "expected_sources": ["supplier_contracts.md", "supplier_workflow"],
        "category": "supplier",
    },
]


# ─── Scoring ───────────────────────────────────────────────────────────────────


def check_keywords(text: str, keywords: list[str]) -> tuple[int, int, list[str]]:
    """Check how many expected keywords appear in the text.

    Returns (found, total, missing_list).
    """
    text_lower = text.lower()
    found = 0
    missing = []
    for kw in keywords:
        if kw.lower() in text_lower:
            found += 1
        else:
            missing.append(kw)
    return found, len(keywords), missing


def check_negative_keywords(text: str, keywords: list[str]) -> list[str]:
    """Check that none of the negative keywords appear (hallucination check)."""
    text_lower = text.lower()
    violations = []
    for kw in keywords:
        if kw.lower() in text_lower:
            violations.append(kw)
    return violations


def check_sources(retrieved_sources: list[str], expected: list[str]) -> tuple[int, int]:
    """Check how many expected source files were retrieved."""
    found = 0
    for exp in expected:
        # Partial match: "supplier_workflow" matches "supplier_workflow.txt"
        if any(exp in s for s in retrieved_sources):
            found += 1
    return found, len(expected)


# ─── Runners ───────────────────────────────────────────────────────────────────


def ensure_ingested():
    """Make sure documents are ingested."""
    store = get_vectorstore()
    if store._collection.count() == 0:
        print("Vector store empty — ingesting documents...")
        settings = get_settings()
        docs = load_documents(settings.document_directories)
        chunks = chunk_documents(docs, settings.chunk_size, settings.chunk_overlap)
        ingest_chunks(chunks)
        print(f"Ingested {len(chunks)} chunks.\n")


def run_retrieval_eval(verbose: bool = False):
    """Test retrieval quality only — no LLM calls."""
    ensure_ingested()
    settings = get_settings()

    print("=" * 70)
    print("  RETRIEVAL EVALUATION")
    print("=" * 70)

    total_source_hits = 0
    total_source_expected = 0

    for i, tc in enumerate(TEST_CASES, 1):
        question = tc["question"]
        expected_sources = tc.get("expected_sources", [])

        docs = retrieve(query=question, top_k=settings.retrieval_top_k)

        # Re-rank if enabled
        if settings.use_reranker:
            docs = rerank(query=question, documents=docs, top_k=settings.reranker_top_k)

        retrieved_sources = [d.metadata.get("source", "") for d in docs]
        src_found, src_total = check_sources(retrieved_sources, expected_sources)
        total_source_hits += src_found
        total_source_expected += src_total

        status = "PASS" if src_total == 0 or src_found == src_total else "PARTIAL" if src_found > 0 else "FAIL"
        print(f"\n[{i:02d}] [{status}] {question}")
        print(f"     Sources: {src_found}/{src_total} expected found")
        if verbose:
            print(f"     Retrieved: {retrieved_sources}")
            print(f"     Expected:  {expected_sources}")

    pct = (total_source_hits / total_source_expected * 100) if total_source_expected else 100
    print(f"\n{'─' * 70}")
    print(f"Source Recall: {total_source_hits}/{total_source_expected} ({pct:.0f}%)")
    print(f"{'─' * 70}\n")


def run_full_eval(verbose: bool = False):
    """Full evaluation: retrieval + answer generation."""
    ensure_ingested()

    print("=" * 70)
    print("  FULL RAG EVALUATION (retrieval + answer generation)")
    print("=" * 70)

    results = {
        "keyword_hits": 0,
        "keyword_total": 0,
        "negative_violations": 0,
        "source_hits": 0,
        "source_total": 0,
        "pass": 0,
        "partial": 0,
        "fail": 0,
    }

    times = []

    for i, tc in enumerate(TEST_CASES, 1):
        question = tc["question"]
        expected_kw = tc["expected_keywords"]
        negative_kw = tc.get("negative_keywords", [])
        expected_sources = tc.get("expected_sources", [])

        start = time.time()
        response = answer_question(question=question)
        elapsed = time.time() - start
        times.append(elapsed)

        answer = response.answer
        retrieved_sources = [s.source for s in response.sources]

        # Score keywords
        kw_found, kw_total, kw_missing = check_keywords(answer, expected_kw)
        neg_violations = check_negative_keywords(answer, negative_kw)
        src_found, src_total = check_sources(retrieved_sources, expected_sources)

        results["keyword_hits"] += kw_found
        results["keyword_total"] += kw_total
        results["negative_violations"] += len(neg_violations)
        results["source_hits"] += src_found
        results["source_total"] += src_total

        # Determine status
        all_kw_ok = kw_found == kw_total
        no_neg = len(neg_violations) == 0
        src_ok = src_total == 0 or src_found == src_total

        if all_kw_ok and no_neg and src_ok:
            status = "PASS"
            results["pass"] += 1
        elif kw_found > 0 and no_neg:
            status = "PARTIAL"
            results["partial"] += 1
        else:
            status = "FAIL"
            results["fail"] += 1

        print(f"\n[{i:02d}] [{status}] {question}  ({elapsed:.1f}s)")
        print(f"     Keywords: {kw_found}/{kw_total}  |  Sources: {src_found}/{src_total}  |  Hallucination: {'NONE' if no_neg else neg_violations}")

        if verbose:
            print(f"     Answer: {answer[:200]}{'...' if len(answer) > 200 else ''}")
            if kw_missing:
                print(f"     Missing keywords: {kw_missing}")

    # Summary
    total = len(TEST_CASES)
    kw_pct = (results["keyword_hits"] / results["keyword_total"] * 100) if results["keyword_total"] else 100
    src_pct = (results["source_hits"] / results["source_total"] * 100) if results["source_total"] else 100
    avg_time = sum(times) / len(times) if times else 0

    print(f"\n{'=' * 70}")
    print(f"  RESULTS SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Tests:           {total}")
    print(f"  PASS:            {results['pass']}")
    print(f"  PARTIAL:         {results['partial']}")
    print(f"  FAIL:            {results['fail']}")
    print(f"  Keyword Recall:  {results['keyword_hits']}/{results['keyword_total']} ({kw_pct:.0f}%)")
    print(f"  Source Recall:   {results['source_hits']}/{results['source_total']} ({src_pct:.0f}%)")
    print(f"  Hallucinations:  {results['negative_violations']}")
    print(f"  Avg Time:        {avg_time:.1f}s per question")
    print(f"{'=' * 70}\n")

    # Exit code for CI
    if results["fail"] > 0:
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Evaluate the RAG system")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show full answers and details")
    parser.add_argument("--retrieval", "-r", action="store_true", help="Test retrieval only (no LLM)")
    args = parser.parse_args()

    if args.retrieval:
        run_retrieval_eval(verbose=args.verbose)
    else:
        run_full_eval(verbose=args.verbose)


if __name__ == "__main__":
    main()
