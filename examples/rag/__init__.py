"""
RAG — Simple RAG pipeline for LongTracer.

Modules:
    pdf_parser: Load PDFs from a directory using PyMuPDF
    chunker: Split documents into chunks using RecursiveCharacterTextSplitter
    embedder: HuggingFace BGE embeddings
    store: Chroma vector store wrapper
    retriever: RAG retriever with Ollama LLM
"""

from rag.pdf_parser import load_pdfs_from_directory
from rag.chunker import chunk_documents
from rag.embedder import get_embeddings
from rag.store import VectorStore
from rag.retriever import RAGRetriever

__all__ = [
    "load_pdfs_from_directory",
    "chunk_documents",
    "get_embeddings",
    "VectorStore",
    "RAGRetriever",
]
