"""
LongTracer Adapters.

Framework-specific instrumentation functions.
Imports are lazy — importing this module does NOT require LangChain or LlamaIndex.
"""


def __getattr__(name: str):
    """Lazy attribute access for adapter classes."""
    if name in ("CitationGuardCallbackHandler", "instrument_langchain"):
        from longtracer.adapters.langchain_handler import (
            CitationGuardCallbackHandler,
            instrument_langchain,
        )
        return CitationGuardCallbackHandler if name == "CitationGuardCallbackHandler" else instrument_langchain

    if name in ("CitationGuardLlamaIndexHandler", "instrument_llamaindex"):
        from longtracer.adapters.llamaindex_handler import (
            CitationGuardLlamaIndexHandler,
            instrument_llamaindex,
        )
        return CitationGuardLlamaIndexHandler if name == "CitationGuardLlamaIndexHandler" else instrument_llamaindex

    raise AttributeError(f"module 'longtracer.adapters' has no attribute {name!r}")


__all__ = [
    "CitationGuardCallbackHandler",
    "instrument_langchain",
    "CitationGuardLlamaIndexHandler",
    "instrument_llamaindex",
]
