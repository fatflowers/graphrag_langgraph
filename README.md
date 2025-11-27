# GraphRAG LangGraph

A lightweight GraphRAG implementation inspired by the paper _From Local to Global: A Graph RAG Approach to Query-Focused Summarization_. The project uses [LangGraph](https://github.com/langchain-ai/langgraph) to orchestrate indexing and query flows and keeps everything offline-friendly with deterministic stubs that can be swapped for real LLMs.

## Features

- Indexing pipeline that splits documents into text units, extracts entities/claims, builds a knowledge graph, detects communities, summarizes them, and creates embeddings.
- Query pipeline with automatic routing between **global**, **local**, and **basic** search modes.
- Minimal in-memory vector store and hash-based embedding fallback to keep tests fast and offline.
- High-level `GraphRAGEngine` API plus an example script.

## Installation

```bash
pip install -e .[dev]
```

## Running the demo

```bash
python examples/index_and_query_demo.py
```

## Running tests

```bash
pytest
```

## Configuration

Edit `examples/index_and_query_demo.py` or provide a YAML file consumed by `GraphRAGEngine.from_config_file` to tune chunk sizes, model names, and retrieval parameters.
