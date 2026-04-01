"""
LongTracer - RAG verification guardrails.

Components:
- verifier: CitationVerifier for claim verification
- tracer: Pluggable tracer with cache backends
- cache: Database backends (memory, sqlite, mongo, redis, postgres)
- context_relevance: Bi-encoder relevance scorer
- nli_model: STS + NLI hybrid verification
- parallel_pipeline: Optimized parallel verification
"""

from longtracer.guard.verifier import CitationVerifier, VerificationResult
from longtracer.guard.tracer import Tracer
from longtracer.guard.context_relevance import ContextRelevanceScorer
from longtracer.guard.nli_model import HybridVerificationModel
from longtracer.guard.claim_splitter import split_into_claims, analyze_claim
from longtracer.guard.cache import TraceCacheBackend, create_backend

# Backward compatibility alias
MongoTracer = Tracer

__all__ = [
    "CitationVerifier",
    "VerificationResult",
    "Tracer",
    "MongoTracer",
    "TraceCacheBackend",
    "create_backend",
    "ContextRelevanceScorer",
    "HybridVerificationModel",
    "split_into_claims",
    "analyze_claim"
]
