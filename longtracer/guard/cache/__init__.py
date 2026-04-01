"""
Cache module - Pluggable trace storage backends.

Usage:
    from longtracer.guard.cache import create_backend, TraceCacheBackend

    backend = create_backend()
    backend = create_backend("memory")
    backend = create_backend("sqlite", path="./traces.db")
    backend = create_backend("mongo", uri="mongodb://localhost:27017")
    backend = create_backend("redis", host="localhost", port=6379)
    backend = create_backend("postgres", host="localhost", database="traces")
"""

from .backend import TraceCacheBackend
from .factory import create_backend, get_default_backend

from .kv_backend import CacheBackend, CacheStats, cache_key
from .kv_factory import get_cache

__all__ = [
    "TraceCacheBackend",
    "create_backend",
    "get_default_backend",
    "CacheBackend",
    "CacheStats",
    "cache_key",
    "get_cache",
]
