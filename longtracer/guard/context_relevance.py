"""
Context Relevance Scorer - Score A using Bi-Encoder Embedding Similarity.

Measures: "Did we fetch the right stuff?"
Method: Bi-encoder embedding similarity between query and retrieved chunks.
"""

import time
from typing import List, Dict, Optional
import numpy as np
from sentence_transformers import SentenceTransformer


class ContextRelevanceScorer:
    """
    Score A: Measures how relevant retrieved chunks are to the query.
    Uses bi-encoder (sentence-transformers) for fast cosine similarity.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        relevance_threshold: float = 0.7,
        verbose: bool = True
    ):
        self.relevance_threshold = relevance_threshold
        self.verbose = verbose

        if verbose:
            print("  ⏳ Loading bi-encoder for context relevance...")
        start = time.time()
        self.model = SentenceTransformer(model_name)
        if verbose:
            print(f"     ✓ Bi-encoder loaded in {(time.time()-start)*1000:.0f}ms")

        self.last_latency_ms = 0.0

    def score(
        self,
        query: str,
        chunks: List[str],
        chunk_ids: Optional[List[str]] = None
    ) -> Dict:
        """Compute cosine similarity between query and each chunk."""
        if not chunks:
            return {
                "average_relevance": 0.0, "top_relevance": 0.0,
                "per_chunk_scores": [], "chunk_rankings": [],
                "threshold_pass": False, "latency_ms": 0.0
            }

        start = time.time()

        query_with_prefix = f"Represent this sentence for searching relevant passages: {query}"
        query_embedding = self.model.encode(query_with_prefix, normalize_embeddings=True)
        chunk_embeddings = self.model.encode(chunks, normalize_embeddings=True)

        scores = np.dot(chunk_embeddings, query_embedding)
        scores = scores.tolist()

        latency_ms = (time.time() - start) * 1000
        self.last_latency_ms = latency_ms

        if chunk_ids is None:
            chunk_ids = [f"chunk_{i}" for i in range(len(chunks))]

        chunk_rankings = [
            {
                "chunk_id": chunk_ids[i],
                "score": scores[i],
                "preview": chunks[i][:100] + "..." if len(chunks[i]) > 100 else chunks[i]
            }
            for i in range(len(chunks))
        ]
        chunk_rankings.sort(key=lambda x: x["score"], reverse=True)

        avg_score = sum(scores) / len(scores)
        top_score = max(scores)

        return {
            "average_relevance": avg_score,
            "top_relevance": top_score,
            "per_chunk_scores": scores,
            "chunk_rankings": chunk_rankings,
            "threshold_pass": avg_score >= self.relevance_threshold,
            "latency_ms": latency_ms
        }

    def score_with_metadata(
        self,
        query: str,
        chunks: List[str],
        metadata: List[Dict]
    ) -> Dict:
        """Score chunks and include source metadata for backtracking."""
        chunk_ids = []
        for i, meta in enumerate(metadata):
            source = meta.get("source", "unknown")
            page = meta.get("page", "?")
            chunk_ids.append(f"{source}:p{page}")

        result = self.score(query, chunks, chunk_ids)

        # Build a lookup from chunk_id → original index for O(1) access
        chunk_id_to_idx = {cid: i for i, cid in enumerate(chunk_ids)}

        for ranking in result["chunk_rankings"]:
            original_idx = chunk_id_to_idx.get(ranking["chunk_id"])
            ranking["metadata"] = metadata[original_idx] if original_idx is not None and original_idx < len(metadata) else {}

        return result


def create_scorer(model_name: str = "BAAI/bge-small-en-v1.5") -> ContextRelevanceScorer:
    """Factory function to create a scorer with default settings."""
    return ContextRelevanceScorer(model_name=model_name)
