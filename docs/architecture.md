# Architecture

## Overview

**emRAG** is a multi-agent Retrieval-Augmented Generation (RAG) system that combines:

- dense retrieval using SentenceTransformers embeddings
- lexical retrieval using TF-IDF
- agent-based corpus partitioning
- a three-tier conversation memory architecture

The system is designed to route a user query to the most relevant document collection, retrieve supporting evidence, and generate an answer grounded only in the retrieved context.

## High-Level Pipeline

1. Discover agents from the dataset directory
2. Build or load indexes for each agent
3. Route the query using memory-aware orchestration
4. Retrieve relevant chunks using dense, lexical, or hybrid search
5. Assemble context from top-ranked chunks
6. Generate an answer using Ollama
7. Verify and evaluate the answer before returning it

## Agent Structure

Each subfolder inside the dataset root becomes a separate agent.

Example:

```text
data/
  animals/
    Dogs.txt
    Cats.txt
  tech/
    GPUs.txt
    CPUs.txt
  food/
    Ramen.txt
```

This allows the system to partition knowledge by domain and route queries to the most relevant agent rather than searching a single monolithic index.

For the `single_agent` evaluation mode, the system instead builds one unified agent over the full corpus.

## Document Processing and Chunking

Each `.txt` file is read and converted into `DocumentChunk` objects. A chunk stores:

- `chunk_id`
- `source_path`
- `text`

Chunking is paragraph-based, with sentence-aware fallback for long paragraphs. The system:

- splits text into paragraphs
- keeps paragraphs intact if short enough
- splits long paragraphs into sentences
- repacks them into chunks up to a configurable size

This design keeps chunks semantically coherent while limiting context length during retrieval and answer generation.

## Retrieval Layer

### Dense Retrieval

Dense retrieval is implemented through the `EmbeddingIndex` class. It uses a SentenceTransformer model, normalizes embeddings, and performs similarity search using a dot product on normalized vectors, equivalent to cosine similarity.

### Lexical Retrieval

Lexical retrieval is implemented with `TfidfVectorizer`. It captures exact lexical overlap and is useful when important keywords or named entities appear explicitly in the documents.

### Hybrid Retrieval

In hybrid mode, dense and lexical scores are combined as:

```text
score = alpha * dense + (1 - alpha) * lexical
```

where `hybrid_alpha` controls the balance between semantic and lexical matching.

### Retrieval Modes

The system supports four operating modes:

- `dense_only`
- `lexical_only`
- `single_agent`
- `full`

These are configured through `SystemConfig.mode`.

## Caching and Index Persistence

To avoid rebuilding document embeddings every time the system starts, emRAG uses a persistent on-disk cache.

For each agent, the system stores:

- `meta.json`
- `chunks.jsonl`
- `embeddings.npy`

under a `.rag_cache/<model_name>/` directory.

Cache validity is determined using a directory fingerprint based on relative paths, modification times, and file sizes. If the fingerprint matches, the cached index is reused. Otherwise, the index is rebuilt.

## Three-Tier Memory Architecture

### Tier 1: Recent Window Memory

`RecentWindowMemory` stores the most recent turns of the conversation. It is used when the current query refers to something said very recently, for example with pronouns such as “it” or “that”.

### Tier 2: Summary State Memory

`SummaryStateMemory` maintains a compact running summary of the conversation, updated through the LLM after each turn.

### Tier 3: Long-Term Conversation Store

`LongTermConversationStore` stores all turns in a persistent memory store and supports hybrid retrieval over past conversation turns using embeddings and TF-IDF.

## Memory Routing

The `MemoryRouter` decides which memory tier is sufficient for interpreting a given user query. The routing logic distinguishes between:

- recent-reference queries
- self-contained factual queries
- older context-dependent queries

The router also contains safeguards to prevent overuse of recent-window memory for standalone factual questions.

## Query Rewriting

Before retrieval, the system may rewrite the query into a more explicit standalone form. However, the rewrite process is conservative:

- if the query already looks self-contained, it is left unchanged
- rewriting is only attempted when unresolved references are likely
- the rewritten query is sanitized to avoid topic drift and unrelated entities

## Answer Generation

Once relevant chunks are selected, the system builds a structured context block and prompts the LLM to answer using only the retrieved evidence.

The answer prompt enforces several constraints:

- use only the provided context for factual claims
- abstain if the answer is not supported
- provide verbatim evidence quotes
- avoid unsupported generalization

This keeps the system grounded in retrieved documents rather than background model knowledge.

## Verification and Evaluation

After generation, emRAG applies an additional entailment-style verification step. The generated answer is checked against the retrieved context, and unsupported answers can be rejected before final output.

The `AnswerEvaluator` then decides whether the answer is acceptable based on factors such as:

- retrieval confidence
- presence of sources
- missing content terms
- abstention behavior

## Orchestration

The `Orchestrator` is the central coordination module. It is responsible for:

- reading memory state
- calling the memory router
- rewriting the query if needed
- ranking agents by retrieval compatibility
- running one or more retrieval attempts
- verifying generated answers
- returning the final response and debug metadata

This makes the system modular while still allowing end-to-end control over retrieval, generation, and validation.

## CLI Behavior

The main interactive loop supports:

- `add <path>` to auto-create an agent from a folder or `.txt` file
- `describe <agent-name>` to show an agent description
- `clear` to clear recent and summary memory
- `reset` to clear recent and summary memory and delete long-term memory on disk
- `exit` or `quit` to leave the program

This allows both experimentation and incremental dataset extension without changing the code.

## Summary

emRAG is designed as an experimental RAG framework that combines:

- modular multi-agent organization
- hybrid dense + lexical retrieval
- memory-aware query interpretation
- grounded answer generation
- answer verification and configurable evaluation modes

Its design makes it suitable for comparing retrieval strategies, studying routing behavior, and evaluating the effect of memory and corpus partitioning in a controlled prototype setting.
