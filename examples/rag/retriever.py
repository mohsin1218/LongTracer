"""
RAG Retriever — Retrieve context and generate answers using Ollama LLM.

Uses langchain-ollama's ChatOllama (latest API) for local LLM inference.
"""

import os
import time
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

from rag.store import VectorStore


DEFAULT_PROMPT = PromptTemplate.from_template(
    """You are a helpful research assistant. Answer the question based ONLY on the
provided context. If the context doesn't contain enough information, say so honestly.

<context>
{context}
</context>

Question: {question}

Answer:"""
)


class RAGRetriever:
    """
    RAG retriever that combines vector search with LLM generation.
    """

    def __init__(
        self,
        store: VectorStore,
        model: Optional[str] = None,
        temperature: float = 0.0,
        prompt: Optional[PromptTemplate] = None,
    ):
        self.store = store
        self.prompt = prompt or DEFAULT_PROMPT

        model_name = model or os.environ.get("OLLAMA_MODEL", "llama3.1")
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

        print(f"  🤖 Loading LLM: {model_name} @ {base_url}")
        self.llm = ChatOllama(
            model=model_name,
            temperature=temperature,
            base_url=base_url,
        )
        print(f"  ✓ LLM ready")

    def retrieve(self, query: str, k: int = 10) -> List[Document]:
        start = time.time()
        docs = self.store.similarity_search(query, k=k)
        elapsed_ms = (time.time() - start) * 1000
        print(f"  🔍 Retrieved {len(docs)} chunks ({elapsed_ms:.0f}ms)")
        return docs

    def generate(self, query: str, docs: List[Document]) -> str:
        if not docs:
            return "No relevant documents found to answer the question."

        context = "\n\n".join(
            f"[Source: {doc.metadata.get('source', 'Unknown')}]\n{doc.page_content}"
            for doc in docs
        )

        formatted_prompt = self.prompt.format(context=context, question=query)

        start = time.time()
        response = self.llm.invoke(formatted_prompt)
        elapsed_ms = (time.time() - start) * 1000

        answer = response.content if hasattr(response, "content") else str(response)
        print(f"  💬 LLM responded ({elapsed_ms:.0f}ms)")
        return answer
