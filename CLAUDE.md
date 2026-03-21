# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A RAG (Retrieval Augmented Generation) playground using Weaviate as the vector database, Ollama for local model serving, and the Weaviate Python client. The stack targets local/self-hosted inference — no external LLM API calls in the current setup.

## Setup & Running

Dependencies are managed with `uv`:

```bash
uv sync              # install dependencies
uv run main.py       # run the main script
```

Docker services must be running before executing Python code:

```bash
docker compose up -d                                          # start Weaviate + Ollama
docker compose exec ollama ollama pull nomic-embed-text       # embedding model
docker compose exec ollama ollama pull llama3.2               # generative model (or llama4:scout)
docker compose down                                           # stop services
```

## Architecture

**Two-service Docker setup** ([docker-compose.yml](docker-compose.yml)):

- **Weaviate** (port 8080/50051): vector database with `text2vec-ollama` and `generative-ollama` modules enabled
- **Ollama** (port 11434): local model server; Weaviate communicates with it at `http://ollama:11434` inside Docker

**Two distinct model roles** — these are not interchangeable:

- **Embedding model** (`nomic-embed-text`): converts text to vectors for semantic search — configured via `Configure.Vectors.text2vec_ollama()`
- **Generative model** (`llama3.2` / `llama4:scout`): generates answers from retrieved context — configured via `Configure.Generative.ollama()`

**Collection configuration** happens at creation time in [main.py](main.py) and binds both models to the collection schema. The generative module is enabled in Docker but not yet wired up in code.

## Key Dependencies

- `weaviate-client[agents]` — Weaviate Python client (v4 API)
- `langchain`, `langgraph`, `langsmith` — intended for orchestration/agent pipelines (not yet used in main.py)
- Python ≥ 3.13 required
