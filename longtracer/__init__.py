"""
LongTracer SDK — One-liner RAG verification.

Usage:
    from longtracer import LongTracer, CitationVerifier
    LongTracer.init()

    # Or with framework adapters:
    from longtracer import instrument_langchain, instrument_llamaindex
"""

from longtracer.core import LongTracer
from longtracer.guard.verifier import CitationVerifier, VerificationResult


def instrument_langchain(chain, verbose=None):
    """Lazy-loaded LangChain adapter."""
    from longtracer.adapters.langchain_handler import instrument_langchain as _impl
    return _impl(chain, verbose=verbose)


def instrument_llamaindex(query_engine, verbose=None):
    """Lazy-loaded LlamaIndex adapter."""
    from longtracer.adapters.llamaindex_handler import instrument_llamaindex as _impl
    return _impl(query_engine, verbose=verbose)


# Backward compatibility
CitationGuard = LongTracer

__all__ = [
    "LongTracer",
    "CitationGuard",  # backward compat
    "CitationVerifier",
    "VerificationResult",
    "instrument_langchain",
    "instrument_llamaindex",
]
