"""
Factory for the pluggable key-value cache.

Singleton pattern — returns the same backend instance per process.
Logs which backend is active exactly once on first call.

Priority:
    1. CACHE_BACKEND=sqlite  →  SQLite directly
    2. MONGODB_URI set       →  try MongoDB; on any failure → SQLite fallback
    3. No config             →  SQLite fallback
"""

import os
import logging
import threading
from datetime import datetime
from typing import Callable, Optional

from .kv_backend import CacheBackend

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_instance: Optional[CacheBackend] = None


def get_cache(
    now_fn: Callable[[], datetime] = datetime.utcnow,
) -> CacheBackend:
    """
    Return the global cache backend (singleton, thread-safe).

    Namespace is always a method argument — this function returns
    the backend, not a namespaced wrapper.

    The backend is selected once on first call and reused.
    """
    global _instance

    if _instance is not None:
        return _instance

    with _lock:
        # Double-check after acquiring lock
        if _instance is not None:
            return _instance

        _instance = _create_backend(now_fn)
        return _instance


def _create_backend(now_fn: Callable[[], datetime]) -> CacheBackend:
    """Resolve and instantiate the appropriate backend."""

    explicit = os.environ.get("CACHE_BACKEND", "").lower().strip()

    # ── Forced SQLite ───────────────────────────────────────────
    if explicit == "sqlite":
        return _make_sqlite(now_fn)

    # ── Try MongoDB if URI is configured ────────────────────────
    mongo_uri = os.environ.get("MONGODB_URI")
    if mongo_uri:
        try:
            from .kv_mongo import MongoCacheBackend
            backend = MongoCacheBackend(now_fn=now_fn)
            if backend.is_connected():
                return backend
            # Connected=False means _connect already warned
        except ImportError:
            logger.warning("pymongo not installed — falling back to SQLite cache")
        except Exception as exc:
            exc_name = type(exc).__name__
            db_name = os.environ.get("MONGODB_DATABASE", "longtracer")
            print(
                f"  ⚠️  MongoDB unavailable [DB: {db_name}] ({exc_name}) "
                f"— falling back to SQLite"
            )

    # ── Fallback to SQLite ──────────────────────────────────────
    return _make_sqlite(now_fn)


def _make_sqlite(now_fn: Callable[[], datetime]) -> CacheBackend:
    from .kv_sqlite import SQLiteCacheBackend
    return SQLiteCacheBackend(now_fn=now_fn)


def reset_cache() -> None:
    """
    Reset the singleton (used in tests).
    Closes the existing backend if it has a ``close()`` method.
    """
    global _instance
    with _lock:
        if _instance is not None:
            if hasattr(_instance, "close"):
                _instance.close()  # type: ignore[attr-defined]
            _instance = None
