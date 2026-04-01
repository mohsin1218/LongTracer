# LongTracer Demo Application

This directory contains a complete RAG pipeline demo using LongTracer with ChromaDB + Ollama. It is NOT part of the published `longtracer` PyPI package.

## Prerequisites

- Python 3.10+
- Ollama running locally with `llama3.1` model
- PDF documents to ingest

## Setup

```bash
# From the repo root
pip install -e ".[all]"
pip install pymupdf langchain-ollama langchain-text-splitters langchain-chroma langchain-huggingface

# Copy and edit the env file
cp examples/.env.example examples/.env

# Pull the Ollama model
ollama pull llama3.1
```

## Usage

```bash
cd examples

# Ingest PDFs
python demo_pipeline.py --ingest './your-pdfs/'

# Query with verification
python demo_pipeline.py --query 'What is the main contribution?'
```

## What this demo shows

1. PDF ingestion → ChromaDB vector store
2. RAG retrieval + Ollama LLM generation
3. LongTracer parallel verification pipeline
4. Full trace report with claim-level results
