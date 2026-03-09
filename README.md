# AI Developer Technical Exercise

## Retrieval-Augmented Generation (RAG) System

### Overview

In this exercise you will build a **Retrieval-Augmented Generation
(RAG)** system that can answer questions based on a set of internal
documentation files.

The provided repository contains a collection of documents that simulate
**internal company knowledge**. The documents describe how a fictional
construction software platform works.

Your goal is to build a system that can:

1.  Ingest the documentation
2.  Retrieve relevant information
3.  Generate answers grounded in the documentation

The system should **only answer questions using the provided
documents**.

If the answer cannot be found in the documentation, the system should
explicitly state that the information is not available.

------------------------------------------------------------------------

# Dataset

The repository contains two sets of documents:

### `/docs`

A structured internal knowledge base describing:

-   project hierarchy
-   financial rules
-   supplier workflows
-   permissions
-   invoicing logic

### `/docs_brutal`

A deliberately **messy dataset** designed to simulate real-world company
documentation.

This dataset includes:

-   duplicated information
-   historical documents
-   outdated specifications
-   engineering notes
-   partial contradictions

Your system should still retrieve correct information from this dataset.

------------------------------------------------------------------------

# Assignment

Build a RAG-based assistant that can answer questions about the system
described in the documentation.

Your implementation should:

1.  Ingest the documents
2.  Split the documents into chunks
3.  Generate embeddings
4.  Store embeddings in a vector database
5.  Retrieve relevant chunks based on a question
6.  Use an LLM to generate an answer based on retrieved context

The answer should include **references to the source documents used**.

------------------------------------------------------------------------

# Example Questions

Your system should be able to answer questions such as:

-   What is the difference between a block and a unit?
-   Can invoices exist on blocks?
-   Who can approve supplier proposals?
-   What happens when costs are approved after invoicing is completed?
-   What are addendums used for?
-   Which roles can modify financial configuration?

------------------------------------------------------------------------

# Important Requirement

The system must **not hallucinate information**.

If the documentation does not contain the answer, the system should
respond with something similar to:

> The provided documentation does not contain information about this
> topic.

Example:

Question:

    How does the platform integrate with Stripe payments?

Correct answer:

    The documentation does not mention any Stripe integration.

------------------------------------------------------------------------

# Technical Requirements

You are free to choose the tools and libraries you prefer.

Possible options include:

-   Python or Node.js
-   OpenAI, local LLMs, or other LLM providers
-   Any vector database (e.g. Chroma, Pinecone, pgvector, Weaviate,
    etc.)

Your solution must include:

-   document ingestion
-   embedding generation
-   vector search
-   answer generation using retrieved context

------------------------------------------------------------------------

# Deliverables

Your submission should include:

1.  Source code
2.  A short README explaining:
    -   how to run the system
    -   architecture decisions
    -   any improvements you would make with more time
3.  A simple interface to ask questions, for example:
    -   CLI
    -   small API
    -   minimal web interface

------------------------------------------------------------------------

# Evaluation Criteria

Submissions will be evaluated based on:

  Area                  Description
  --------------------- ----------------------------------------------
  Retrieval Quality     Ability to retrieve relevant document chunks
  Grounded Answers      Answers must be based on retrieved content
  Architecture          Overall design of the RAG pipeline
  Code Quality          Clarity and maintainability
  Handling Edge Cases   Correct handling of missing information

------------------------------------------------------------------------

# Time Expectation

This exercise is designed to take approximately **4--6 hours**.

We are not looking for a production-ready system.\
Focus on demonstrating your **approach and reasoning**.

------------------------------------------------------------------------

# Bonus (Optional)

If you want to go further, consider improvements such as:

-   improving retrieval quality
-   re-ranking retrieved documents
-   hybrid search (vector + keyword)
-   evaluation of answer accuracy
-   document metadata filtering

------------------------------------------------------------------------

# Final Notes

The documentation intentionally contains:

-   duplicated information
-   outdated notes
-   archived specifications

This reflects real-world engineering environments where documentation
evolves over time.

Your system should still be able to retrieve the **correct and relevant
information**.
