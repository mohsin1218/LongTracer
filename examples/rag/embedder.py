"""
Embedder — HuggingFace BGE embedding model for vector store and retrieval.

Uses langchain-huggingface's HuggingFaceEmbeddings (latest API).
Same model as the context relevance scorer for consistent embedding space.
"""

import os

from langchain_huggingface import HuggingFaceEmbeddings


DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


def get_embeddings(
    model_name: str | None = None,
) -> HuggingFaceEmbeddings:
    """
    Create a HuggingFace embedding function for use with vector stores.

    Args:
        model_name: HuggingFace model ID. Defaults to BGE-small-en-v1.5.

    Returns:
        HuggingFaceEmbeddings instance compatible with LangChain vector stores.
    """
    model_name = model_name or os.environ.get("EMBEDDING_MODEL", DEFAULT_MODEL)

    print(f"  ⏳ Loading embedding model: {model_name}")
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    print(f"  ✓ Embedding model ready")
    return embeddings
