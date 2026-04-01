"""
Vector Store — Chroma wrapper for document storage and retrieval.

Uses langchain-chroma (latest API) with local persistence.
"""

import os
from pathlib import Path
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


DEFAULT_PERSIST_DIR = str(Path(__file__).parent.parent / "chroma_db")


class VectorStore:
    """
    Chroma-backed vector store for RAG document storage and retrieval.
    """

    def __init__(
        self,
        embedding_function: HuggingFaceEmbeddings,
        persist_directory: Optional[str] = None,
        collection_name: Optional[str] = None,
    ):
        self.persist_directory = persist_directory or os.environ.get(
            "CHROMA_PERSIST_DIR", DEFAULT_PERSIST_DIR
        )
        # CHROMA_COLLECTION env var allows override; default keeps backward compat
        # with existing chroma_db that was built with "citation_guard" collection name
        self.collection_name = collection_name or os.environ.get(
            "CHROMA_COLLECTION", "citation_guard"
        )

        self._store = Chroma(
            collection_name=self.collection_name,
            embedding_function=embedding_function,
            persist_directory=self.persist_directory,
        )

        try:
            count = self._store._collection.count()
            if count > 0:
                print(f"  📚 Loaded existing collection: {count} documents")
        except Exception:
            pass

    def add_documents(self, documents: List[Document]) -> List[str]:
        if not documents:
            return []
        ids = self._store.add_documents(documents)
        print(f"  💾 Stored {len(ids)} chunks in Chroma ({self.persist_directory})")
        return ids

    def similarity_search(self, query: str, k: int = 10) -> List[Document]:
        return self._store.similarity_search(query, k=k)

    def as_retriever(self, **kwargs):
        return self._store.as_retriever(**kwargs)

    @property
    def vectorstore(self) -> Chroma:
        return self._store
