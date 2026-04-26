# ContextMindLM server

Backend for a production RAG system built with FastAPI. It handles ingestion, retrieval, query routing, validation, and streaming responses.

## Architecture

![Architecture](https://github.com/user-attachments/assets/0a24673c-7d5d-45a4-aa53-8d17054a2f38)

## Ingestion Pipeline

Handles:
- Multi-format document ingestion
- Web content ingestion
- Async processing with Celery and Redis
- Large file uploads through pre-signed S3 URLs

## Retrieval Pipeline

Supports:
- Semantic search
- Hybrid search
- Multi-query semantic search
- Multi-query hybrid search
- RRF and reranking

## Query Orchestration

Uses LangChain and LangGraph to route requests between:
- Document retrieval
- Web search
- Tool-based reasoning

External search tools:
- DuckDuckGo
- Tavily

## License

This project is proprietary and not open source.

© 2026 Aditya Khichi. All rights reserved.

Unauthorized copying, use, or distribution is strictly prohibited.
